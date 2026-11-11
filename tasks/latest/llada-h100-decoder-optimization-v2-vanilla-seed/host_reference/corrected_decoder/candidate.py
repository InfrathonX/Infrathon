"""Host-only corrected confidence-threshold + Prefix KV reference."""

from __future__ import annotations

import torch

from runtime.api import GenerationResult, GenerationSpec


THRESHOLD = 0.9


def _crop_prefix(past_key_values, length: int):
    return tuple(
        tuple(value[:, :, :length, :].contiguous() for value in layer)
        for layer in past_key_values
    )


def _commit(tokens, start: int, end: int, logits: torch.Tensor, mask_id: int) -> int:
    block = tokens[:, start:end]
    masked = block == mask_id
    block_logits = logits[:, : end - start]
    predictions = block_logits.argmax(dim=-1)
    fp32 = block_logits.float()
    chosen_logits = fp32.gather(-1, predictions.unsqueeze(-1)).squeeze(-1)
    confidence = torch.exp(chosen_logits - torch.logsumexp(fp32, dim=-1))
    confidence = confidence.masked_fill(~masked, -torch.inf)
    transfer = masked & (confidence >= THRESHOLD)
    if not bool(transfer.any().item()):
        top = confidence.argmax(dim=-1)
        transfer.zero_()
        transfer.scatter_(1, top.unsqueeze(1), True)
    block[transfer] = predictions[transfer]
    return int(transfer.sum().item())


class PrefixThresholdDecoder:
    def __init__(self, model, tokenizer) -> None:
        self.model = model
        self.tokenizer = tokenizer

    @torch.inference_mode()
    def generate(self, prompt_ids: torch.Tensor, spec: GenerationSpec) -> GenerationResult:
        spec.validate(int(prompt_ids.shape[1]))
        prompt_length = int(prompt_ids.shape[1])
        tokens = torch.full(
            (1, prompt_length + spec.gen_length),
            spec.mask_id,
            dtype=torch.long,
            device=prompt_ids.device,
        )
        tokens[:, :prompt_length] = prompt_ids
        nfe = 0
        committed = 0
        for offset in range(0, spec.gen_length, spec.block_length):
            start = prompt_length + offset
            end = start + spec.block_length
            full = self.model(tokens, use_cache=True)
            nfe += 1
            prefix = _crop_prefix(full.past_key_values, start)
            committed += _commit(
                tokens,
                start,
                end,
                full.logits[:, start:end],
                spec.mask_id,
            )
            while bool((tokens[:, start:end] == spec.mask_id).any().item()):
                suffix = self.model(
                    tokens[:, start:],
                    past_key_values=prefix,
                    use_cache=False,
                )
                nfe += 1
                committed += _commit(
                    tokens,
                    start,
                    end,
                    suffix.logits[:, : spec.block_length],
                    spec.mask_id,
                )
        return GenerationResult(
            token_ids=tokens,
            trace={"candidate_nfe": nfe, "committed": committed},
        )


def build(model, tokenizer) -> PrefixThresholdDecoder:
    return PrefixThresholdDecoder(model, tokenizer)
