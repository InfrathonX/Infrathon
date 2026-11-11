"""Participant-facing contract for the EAGLE-3 speculative-decoding engine task.

Provisional v0 (owner-authored 2026-09-04). This file is the ABI: the seed engine,
the public evaluator and the Judge all import it. Anything not defined here is not
part of the contract. See DESIGN_AND_RUN.md §3, §6.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch

API_VERSION = "eagle3-spec-decode-engine/v0"


@dataclass(frozen=True)
class EngineContract:
    """Frozen facts about the pinned target/head pair and the evaluation bounds."""

    target_model_id: str
    target_revision: str
    draft_model_id: str
    draft_revision: str
    device: str = "cuda:0"
    dtype: torch.dtype = torch.bfloat16
    max_sequence_length: int = 4096      # KV/position bound an engine must support (prompt + generation + draft headroom)
    max_request_tokens: int = 3968       # runtime never issues prompt_len + max_new_tokens above this (128-token draft reserve)
    max_batch_size: int = 8              # generate() receives 1..max_batch_size prompts
    eos_token_ids: Tuple[int, ...] = (151645, 151643)  # <|im_end|>, <|endoftext|>
    vocab_size: int = 151936
    draft_vocab_size: int = 32000
    hidden_size: int = 4096
    num_hidden_layers: int = 36
    aux_layer_ids: Tuple[int, int, int] = (2, 18, 33)  # hidden states *entering* these target layers are fused
    build_timeout_s: int = 600


@dataclass(frozen=True)
class DraftHead:
    """The frozen EAGLE-3 head.

    `state_dict` uses upstream `eagle.model.cnets.Model` key names (fc.weight,
    midlayer.*, norm.weight, lm_head.weight, d2t, t2d, embed_tokens.weight). Weights are
    BF16 on `EngineContract.device`; `d2t` is int64 [draft_vocab_size]; `t2d` is bool
    [vocab_size]; `embed_tokens.weight` is a copy of the target's input embedding.
    `config` is the head's config.json as a dict. Tensors are read-only inputs: an engine
    may repack or re-quantise copies but must not mutate them in place.
    """

    config: Mapping[str, Any]
    state_dict: Mapping[str, torch.Tensor]


@dataclass(frozen=True)
class GenerationSpec:
    """Greedy decoding is fixed; the only request parameter is the token budget."""

    max_new_tokens: int

    def validate(self, contract: EngineContract) -> None:
        if not isinstance(self.max_new_tokens, int) or self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be a positive int")
        if self.max_new_tokens > contract.max_request_tokens:
            raise ValueError("max_new_tokens exceeds max_request_tokens")


@dataclass
class GenerationResult:
    """One int64 tensor per prompt: prompt ⧺ generated (stop token included if produced)."""

    token_ids: List[torch.Tensor]
    trace: Dict[str, Any] = field(default_factory=dict)


def validate_prompts(prompt_ids: Sequence[torch.Tensor], contract: EngineContract, spec: GenerationSpec) -> None:
    """Checks the runtime applies before calling the engine (so engines may assume them)."""
    if not 1 <= len(prompt_ids) <= contract.max_batch_size:
        raise ValueError(f"batch of {len(prompt_ids)} prompts outside 1..{contract.max_batch_size}")
    for p in prompt_ids:
        if not isinstance(p, torch.Tensor) or p.dtype != torch.int64 or p.dim() != 1 or p.numel() < 1:
            raise ValueError("each prompt must be a non-empty 1-D int64 tensor")
        if str(p.device) != contract.device:
            raise ValueError(f"prompt must live on {contract.device}")
        if p.numel() + spec.max_new_tokens > contract.max_request_tokens:
            raise ValueError("prompt + max_new_tokens exceeds max_request_tokens")
        if int(p.min()) < 0 or int(p.max()) >= contract.vocab_size:
            raise ValueError("prompt token id out of range")


def truncate_generation(tokens: torch.Tensor, prompt_len: int, max_new_tokens: int,
                        eos_token_ids: Sequence[int]) -> torch.Tensor:
    """Canonical post-processing (the O3 semantics): cut at the first stop token
    (inclusive) and at `max_new_tokens`. Engines that, like upstream, overshoot by a
    partial accepted chunk call this before returning."""
    if tokens.dim() != 1:
        raise ValueError("tokens must be 1-D")
    gen = tokens[prompt_len:prompt_len + max_new_tokens]
    if gen.numel():
        stop = torch.zeros_like(gen, dtype=torch.bool)
        for e in eos_token_ids:
            stop |= gen == e
        hits = torch.nonzero(stop, as_tuple=False)
        if hits.numel():
            gen = gen[: int(hits[0, 0]) + 1]
    return torch.cat([tokens[:prompt_len], gen])


def validate_result(prompt_ids: Sequence[torch.Tensor], result: GenerationResult,
                    spec: GenerationSpec, contract: EngineContract) -> List[int]:
    """Shape/termination checks (O3, O4). Returns the generated length per prompt.
    Raises ValueError with a precise reason on the first violation."""
    if not isinstance(result, GenerationResult):
        raise ValueError("generate() must return runtime.api.GenerationResult")
    if len(result.token_ids) != len(prompt_ids):
        raise ValueError("one output per prompt is required")
    lengths: List[int] = []
    eos = torch.tensor(list(contract.eos_token_ids), dtype=torch.int64)
    for i, (p, y) in enumerate(zip(prompt_ids, result.token_ids)):
        if not isinstance(y, torch.Tensor) or y.dtype != torch.int64 or y.dim() != 1:
            raise ValueError(f"output {i}: must be a 1-D int64 tensor")
        if str(y.device) != contract.device:
            raise ValueError(f"output {i}: must live on {contract.device}")
        L = p.numel()
        if y.numel() <= L:
            raise ValueError(f"output {i}: no generated token")
        if not torch.equal(y[:L], p):
            raise ValueError(f"output {i}: prompt prefix not preserved")
        g = y[L:].cpu()
        n = g.numel()
        if n > spec.max_new_tokens:
            raise ValueError(f"output {i}: {n} generated tokens exceed max_new_tokens={spec.max_new_tokens}")
        if int(g.min()) < 0 or int(g.max()) >= contract.vocab_size:
            raise ValueError(f"output {i}: token id out of range")
        is_stop = torch.isin(g, eos)
        if is_stop[:-1].any():
            raise ValueError(f"output {i}: stop token before the last position")
        if not bool(is_stop[-1]) and n != spec.max_new_tokens:
            raise ValueError(f"output {i}: stopped without a stop token after {n} < {spec.max_new_tokens} tokens")
        lengths.append(n)
    return lengths
