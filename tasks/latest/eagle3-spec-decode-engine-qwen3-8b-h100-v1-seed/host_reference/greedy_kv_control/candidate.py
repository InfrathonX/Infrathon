"""Disclosed lossless control: standard HF greedy decoding with a static KV cache."""
from __future__ import annotations

import time

import torch
from transformers import StaticCache

from runtime.api import GenerationResult, truncate_generation, validate_prompts


def build(target, draft, tokenizer, contract):
    return Engine(target, contract)


class Engine:
    def __init__(self, target, contract):
        self.target = target
        self.contract = contract
        self.cache = StaticCache(
            target.config,
            max_batch_size=1,
            max_cache_len=contract.max_sequence_length,
            device=contract.device,
            dtype=contract.dtype,
        )

    def _generate_one(self, prompt, spec):
        self.cache.reset()
        ids = prompt.unsqueeze(0).clone()
        start = time.perf_counter()
        positions = torch.arange(ids.shape[1], device=ids.device)
        output = self.target(
            input_ids=ids,
            past_key_values=self.cache,
            cache_position=positions,
            use_cache=True,
        )
        prefill_ms = (time.perf_counter() - start) * 1000.0
        generated = []
        next_token = output.logits[:, -1:].argmax(dim=-1)
        for index in range(spec.max_new_tokens):
            generated.append(next_token)
            if int(next_token.item()) in self.contract.eos_token_ids or index + 1 == spec.max_new_tokens:
                break
            position = torch.tensor([ids.shape[1] + index], dtype=torch.long, device=ids.device)
            output = self.target(
                input_ids=next_token,
                past_key_values=self.cache,
                cache_position=position,
                use_cache=True,
            )
            next_token = output.logits[:, -1:].argmax(dim=-1)
        full = torch.cat([ids] + generated, dim=1)[0]
        full = truncate_generation(full, prompt.numel(), spec.max_new_tokens, self.contract.eos_token_ids)
        n = full.numel() - prompt.numel()
        return full, {"cycles": n, "accepted_tokens": n, "mean_accept_length": 1.0, "prefill_ms": prefill_ms}

    @torch.inference_mode()
    def generate(self, prompt_ids, spec):
        spec.validate(self.contract)
        validate_prompts(prompt_ids, self.contract, spec)
        token_ids, traces = [], []
        for prompt in prompt_ids:
            output, trace = self._generate_one(prompt, spec)
            token_ids.append(output)
            traces.append(trace)
        return GenerationResult(token_ids=token_ids, trace={"prompts": traces})

