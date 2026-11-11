"""Faithful, deliberately unoptimised EAGLE-3 inference port.

The algorithm is ported from vendor/EAGLE/eagle/model/ea_model.py:eagenerate.
The target and draft implementations live beside this module so the candidate is
self-contained behind runtime.api.
"""
from __future__ import annotations

import time
import copy
from typing import Any

import torch

from runtime.api import GenerationResult, truncate_generation, validate_prompts

from .config import EConfig
from .draft import LlamaRotaryEmbedding, Model as DraftModel
from .target_kv import Qwen3ForCausalLM, initialize_past_key_values
from .tree import (
    evaluate_posterior,
    initialize_tree,
    reset_tree_mode,
    tree_decoding,
    update_inference_inputs,
)


# The intentionally prominent first optimisation knobs (upstream Qwen3 defaults).
TOTAL_TOKEN = 60
DEPTH = 7
TOP_K = 10
THRESHOLD = 1.0


def _shared_parameter_evidence(source, rebuilt) -> dict[str, Any]:
    source_params = dict(source.named_parameters())
    rebuilt_params = dict(rebuilt.named_parameters())
    if source_params.keys() != rebuilt_params.keys():
        missing = sorted(source_params.keys() ^ rebuilt_params.keys())
        raise ValueError(f"ported target state_dict mismatch: {missing}")
    mismatched = [
        name for name in source_params
        if source_params[name].untyped_storage().data_ptr() != rebuilt_params[name].untyped_storage().data_ptr()
    ]
    if mismatched:
        raise RuntimeError(f"ported target copied weight storage: {mismatched[:3]}")
    unique_storages = {
        (p.untyped_storage().data_ptr(), p.untyped_storage().nbytes())
        for p in list(source_params.values()) + list(rebuilt_params.values())
    }
    target_storages = {
        (p.untyped_storage().data_ptr(), p.untyped_storage().nbytes())
        for p in source_params.values()
    }
    return {
        "all_parameter_storages_shared": True,
        "parameter_count": len(source_params),
        "target_storage_bytes": sum(size for _, size in target_storages),
        "combined_unique_storage_bytes": sum(size for _, size in unique_storages),
    }


def build(target, draft, tokenizer, contract):
    before = torch.cuda.memory_allocated(contract.device) if torch.cuda.is_available() else 0
    # Upstream's custom from_pretrained auto-selects "sdpa" in the config even
    # though Qwen3Attention explicitly executes eager_attention_forward.  The
    # masking helpers inspect this flag, so reproduce it independently of the
    # runtime reference model's policy (which is explicitly eager).
    tree_config = copy.deepcopy(target.config)
    tree_config._attn_implementation = "sdpa"
    tree_config._attn_implementation_autoset = True
    with torch.device("meta"):
        tree_target = Qwen3ForCausalLM(tree_config)
    incompatible = tree_target.load_state_dict(target.state_dict(), strict=True, assign=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError(f"ported target load mismatch: {incompatible}")
    tree_target.eval().requires_grad_(False)
    # RoPE state is non-persistent and therefore absent from state_dict. Reuse the
    # handed-over target's tiny buffer too; leaving the meta buffer would be invalid.
    tree_target.model.rotary_emb.inv_freq = target.model.rotary_emb.inv_freq
    tree_target.model.rotary_emb.original_inv_freq = target.model.rotary_emb.original_inv_freq
    sharing = _shared_parameter_evidence(target, tree_target)

    clean_config = {k: v for k, v in draft.config.items() if not k.startswith("_runtime_")}
    draft_config = EConfig(**clean_config)
    with torch.device("meta"):
        draft_model = DraftModel(
            draft_config,
            load_emb=False,
            bias=bool(clean_config.get("bias", True)),
            total_tokens=TOTAL_TOKEN,
            depth=DEPTH,
            top_k=TOP_K,
            threshold=THRESHOLD,
        )
    incompatible = draft_model.load_state_dict(dict(draft.state_dict), strict=True, assign=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError(f"draft head load mismatch: {incompatible}")
    # The legacy drafter's RoPE cache is non-persistent and is not a checkpoint
    # tensor. Construct only this small derived cache on the requested device.
    draft_model.midlayer.self_attn.rotary_emb = LlamaRotaryEmbedding(
        draft_model.midlayer.self_attn.head_dim,
        max_position_embeddings=draft_config.max_position_embeddings,
        base=getattr(draft_config, "rope_theta", 10000),
        device="cpu",
    ).to(dtype=contract.dtype).to(contract.device)
    draft_model.eval().requires_grad_(False)
    draft_model.diff_device = False
    draft_model.init_tree()
    after = torch.cuda.memory_allocated(contract.device) if torch.cuda.is_available() else 0
    sharing["build_allocated_delta_bytes"] = after - before
    sharing["draft_strict_load_missing_keys"] = []
    sharing["draft_strict_load_unexpected_keys"] = []
    return Engine(tree_target, draft_model, tokenizer, contract, sharing)


class Engine:
    def __init__(self, base_model, ea_layer, tokenizer, contract, memory_evidence):
        self.base_model = base_model
        self.ea_layer = ea_layer
        self.tokenizer = tokenizer
        self.contract = contract
        self.config = base_model.config
        self.use_eagle3 = True
        self.memory_evidence = memory_evidence
        self.past_key_values = None
        self.past_key_values_data = None
        self.current_length_data = None

    def __call__(self, input_ids=None, attention_mask=None, past_key_values=None,
                 output_orig=False, position_ids=None):
        outputs = self.base_model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )
        orig = self.base_model.lm_head(outputs[0]) if output_orig else None
        hidden_states = outputs[0]
        if output_orig:
            return outputs, orig, hidden_states
        return outputs, hidden_states

    def _ensure_cache(self):
        if self.past_key_values is None:
            # Requests stop at max_request_tokens, but the engine's KV/position
            # storage must cover the full contract bound so verification-tree
            # nodes fit in the reserved 128-position tail.
            self.past_key_values, self.past_key_values_data, self.current_length_data = initialize_past_key_values(
                self.base_model, max_length=self.contract.max_sequence_length
            )
        else:
            self.current_length_data.zero_()

    def _generate_one(self, prompt: torch.Tensor, spec):
        input_ids = prompt.unsqueeze(0).clone()
        input_len = prompt.numel()
        padding = torch.full((1, 1), -1, dtype=torch.long, device=prompt.device)
        self.ea_layer.reset_kv()
        self._ensure_cache()
        reset_tree_mode(self)

        prefill_start = time.perf_counter()
        draft_tokens, retrieve_indices, tree_mask, tree_position_ids, _, _, sample_token = initialize_tree(
            input_ids, self, self.past_key_values, None
        )
        prefill_ms = (time.perf_counter() - prefill_start) * 1000.0
        new_token = 0
        cycles = 0
        upstream_length_bound = self.contract.max_sequence_length - self.ea_layer.total_tokens - 10

        for _ in range(upstream_length_bound):
            cycles += 1
            self.base_model.model.tree_mask = tree_mask
            draft_tokens = draft_tokens.to(input_ids.device)
            logits, hidden_state_new, _ = tree_decoding(
                self, draft_tokens, self.past_key_values, tree_position_ids, input_ids, retrieve_indices
            )
            candidates = torch.cat((draft_tokens, padding), dim=1)[0, retrieve_indices]
            best_candidate, accept_length, sample_p = evaluate_posterior(
                logits, candidates, None, retrieve_indices
            )
            selected = candidates[best_candidate]
            accepted_mask = torch.arange(selected.shape[0], device=selected.device) <= accept_length
            stop_now = torch.zeros((), dtype=torch.bool, device=selected.device)
            for stop_id in self.contract.eos_token_ids:
                stop_now |= ((selected == stop_id) & accepted_mask).any()
            # The only per-cycle host synchronisation: transfer acceptance length
            # together with the stop decision for the newly accepted chunk.
            cycle_state = torch.stack((accept_length, stop_now.to(torch.long))).to("cpu")
            accept_length_host = int(cycle_state[0])
            stop_now_host = bool(cycle_state[1])
            (
                input_ids, draft_tokens, retrieve_indices, tree_mask, tree_position_ids,
                new_token, _, sample_token,
            ) = update_inference_inputs(
                input_ids, candidates, best_candidate, accept_length_host, retrieve_indices,
                None, new_token, self.past_key_values_data, self.current_length_data,
                self, hidden_state_new, sample_p,
            )
            if stop_now_host:
                break
            if new_token > spec.max_new_tokens:
                break
            if input_ids.shape[1] > upstream_length_bound:
                break

        output = truncate_generation(
            input_ids[0], input_len, spec.max_new_tokens, self.contract.eos_token_ids
        )
        accepted = output.numel() - input_len
        return output, {
            "cycles": cycles,
            "accepted_tokens": accepted,
            "mean_accept_length": accepted / cycles if cycles else 0.0,
            "prefill_ms": prefill_ms,
            "upstream_new_token_counter": new_token,
        }

    @torch.inference_mode()
    def generate(self, prompt_ids, spec):
        spec.validate(self.contract)
        validate_prompts(prompt_ids, self.contract, spec)
        outputs = []
        traces = []
        for prompt in prompt_ids:
            output, trace = self._generate_one(prompt, spec)
            outputs.append(output)
            traces.append(trace)
        return GenerationResult(token_ids=outputs, trace={"prompts": traces, "memory": self.memory_evidence})
