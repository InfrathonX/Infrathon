"""Stable generation ABI used by the supplied runtime and visible tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


MASK_TOKEN_ID = 126336
EOS_TOKEN_ID = 126081
VOCAB_SIZE = 126464
MAX_SEQUENCE_LENGTH = 4096


@dataclass(frozen=True, slots=True)
class GenerationSpec:
    gen_length: int
    block_length: int = 32
    temperature: float = 0.0
    mask_id: int = MASK_TOKEN_ID

    def validate(self, prompt_length: int) -> None:
        if self.gen_length <= 0:
            raise ValueError("gen_length must be positive")
        if self.block_length <= 0 or self.gen_length % self.block_length:
            raise ValueError("block_length must divide gen_length")
        if self.temperature != 0.0:
            raise ValueError("deterministic generation requires temperature 0")
        if prompt_length + self.gen_length > MAX_SEQUENCE_LENGTH:
            raise ValueError("prompt plus generation exceeds the model limit")


@dataclass(slots=True)
class GenerationResult:
    token_ids: torch.Tensor
    trace: dict[str, Any] = field(default_factory=dict)


def validate_result(
    result: GenerationResult,
    prompt_ids: torch.Tensor,
    spec: GenerationSpec,
) -> None:
    if not isinstance(result, GenerationResult):
        raise TypeError("generate() must return runtime.api.GenerationResult")
    tokens = result.token_ids
    expected_shape = (1, prompt_ids.shape[1] + spec.gen_length)
    if tuple(tokens.shape) != expected_shape:
        raise ValueError(f"token_ids shape {tuple(tokens.shape)} != {expected_shape}")
    if tokens.dtype != torch.long:
        raise TypeError(f"token_ids must be torch.int64, got {tokens.dtype}")
    if tokens.device != prompt_ids.device:
        raise ValueError("token_ids must stay on the prompt CUDA device")
    if not torch.equal(tokens[:, : prompt_ids.shape[1]], prompt_ids):
        raise ValueError("candidate modified prompt tokens")
    generated = tokens[:, prompt_ids.shape[1] :]
    if bool((generated == spec.mask_id).any().item()):
        raise ValueError("generation contains residual diffusion MASK tokens")
    if bool(((generated < 0) | (generated >= VOCAB_SIZE)).any().item()):
        raise ValueError("generation contains token ids outside the vocabulary")
