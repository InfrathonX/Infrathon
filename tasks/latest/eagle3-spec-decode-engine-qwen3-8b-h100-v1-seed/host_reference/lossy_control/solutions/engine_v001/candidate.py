"""Deliberately lossy negative-control engine for oracle qualification."""

from __future__ import annotations

from typing import Any

import torch

from runtime.api import GenerationResult, validate_prompts


def build(
    target: Any,
    draft: Any,
    tokenizer: Any,
    contract: Any,
) -> "Engine":
    """Build the intentionally invalid all-zero generator."""
    del target, draft, tokenizer
    return Engine(contract)


class Engine:
    """Emit zero tokens without consulting the target model."""

    def __init__(self, contract: Any) -> None:
        self.contract = contract

    def generate(
        self,
        prompt_ids: list[torch.Tensor],
        spec: Any,
    ) -> GenerationResult:
        """Return structurally valid but algorithmically incorrect streams."""
        spec.validate(self.contract)
        validate_prompts(prompt_ids, self.contract, spec)
        outputs = [
            torch.cat(
                (
                    prompt,
                    torch.zeros(
                        spec.max_new_tokens,
                        dtype=torch.int64,
                        device=prompt.device,
                    ),
                )
            )
            for prompt in prompt_ids
        ]
        return GenerationResult(
            outputs,
            {"negative_control": "unverified-zero"},
        )
