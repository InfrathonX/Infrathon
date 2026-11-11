"""Deliberately FP8-perturbed negative-control engine for tier qualification."""

from __future__ import annotations

from typing import Any

import torch
from transformers import StaticCache

from runtime.api import GenerationResult, truncate_generation, validate_prompts


def build(
    target: Any,
    draft: Any,
    tokenizer: Any,
    contract: Any,
) -> "Engine":
    """Round target linear weights through FP8 before greedy generation."""
    del draft, tokenizer
    for module in target.modules():
        if isinstance(module, torch.nn.Linear):
            rounded = module.weight.data.to(torch.float8_e4m3fn).to(torch.bfloat16)
            module.weight.data.copy_(rounded)
    return Engine(target, contract)


class Engine:
    """Greedily decode using an intentionally perturbed target."""

    def __init__(self, target: Any, contract: Any) -> None:
        self.target = target
        self.contract = contract
        self.cache = StaticCache(
            target.config,
            max_batch_size=1,
            max_cache_len=contract.max_sequence_length,
            device=contract.device,
            dtype=contract.dtype,
        )

    @torch.inference_mode()
    def generate(
        self,
        prompt_ids: list[torch.Tensor],
        spec: Any,
    ) -> GenerationResult:
        """Generate streams whose small margin errors should fail A2 tiers."""
        spec.validate(self.contract)
        validate_prompts(prompt_ids, self.contract, spec)
        outputs = []
        for prompt in prompt_ids:
            self.cache.reset()
            ids = prompt.unsqueeze(0)
            positions = torch.arange(ids.shape[1], device=ids.device)
            result = self.target(
                input_ids=ids,
                past_key_values=self.cache,
                cache_position=positions,
                use_cache=True,
            )
            generated = []
            token = result.logits[:, -1:].argmax(-1)
            for index in range(spec.max_new_tokens):
                generated.append(token)
                if (
                    int(token.item()) in self.contract.eos_token_ids
                    or index + 1 == spec.max_new_tokens
                ):
                    break
                position = torch.tensor([ids.shape[1] + index], device=ids.device)
                result = self.target(
                    input_ids=token,
                    past_key_values=self.cache,
                    cache_position=position,
                    use_cache=True,
                )
                token = result.logits[:, -1:].argmax(-1)
            combined = torch.cat([ids, *generated], dim=1)[0]
            outputs.append(
                truncate_generation(
                    combined,
                    prompt.numel(),
                    spec.max_new_tokens,
                    self.contract.eos_token_ids,
                )
            )
        return GenerationResult(outputs, {"negative_control": "fp8-target"})
