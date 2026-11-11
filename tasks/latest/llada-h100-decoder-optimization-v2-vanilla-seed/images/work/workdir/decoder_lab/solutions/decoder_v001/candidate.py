"""Correct but deliberately unaccelerated LLaDA diffusion decoder.

This seed performs one full-sequence model evaluation for every committed
token. It is a safe deterministic starting point, not an optimized engine.
"""

from __future__ import annotations

import torch

from runtime.api import GenerationResult, GenerationSpec


class VanillaDecoder:
    def __init__(self, model, tokenizer) -> None:
        self.model = model
        self.tokenizer = tokenizer

    @torch.inference_mode()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        spec: GenerationSpec,
    ) -> GenerationResult:
        spec.validate(int(prompt_ids.shape[1]))
        if prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1:
            raise ValueError("prompt_ids must be shaped [1, P]")
        allow_cpu = bool(getattr(self.model, "mlsysforge_allow_cpu", False))
        if prompt_ids.dtype != torch.long or (
            prompt_ids.device.type != "cuda" and not allow_cpu
        ):
            raise TypeError("prompt_ids must be CUDA int64")

        prompt_length = int(prompt_ids.shape[1])
        tokens = torch.full(
            (1, prompt_length + spec.gen_length),
            spec.mask_id,
            dtype=torch.long,
            device=prompt_ids.device,
        )
        tokens[:, :prompt_length] = prompt_ids

        nfe = 0
        for block_offset in range(0, spec.gen_length, spec.block_length):
            start = prompt_length + block_offset
            end = start + spec.block_length
            while bool((tokens[:, start:end] == spec.mask_id).any().item()):
                logits = self.model(tokens).logits
                nfe += 1

                block_mask = tokens[:, start:end] == spec.mask_id
                block_logits = logits[:, start:end]
                predictions = torch.argmax(block_logits, dim=-1)
                probabilities = torch.softmax(block_logits.float(), dim=-1)
                confidence = probabilities.gather(
                    -1, predictions.unsqueeze(-1)
                ).squeeze(-1)
                confidence = confidence.masked_fill(~block_mask, -torch.inf)

                flat_index = int(torch.argmax(confidence, dim=-1).item())
                tokens[0, start + flat_index] = predictions[0, flat_index]

        return GenerationResult(token_ids=tokens, trace={"candidate_nfe": nfe})


def build(model, tokenizer) -> VanillaDecoder:
    return VanillaDecoder(model, tokenizer)
