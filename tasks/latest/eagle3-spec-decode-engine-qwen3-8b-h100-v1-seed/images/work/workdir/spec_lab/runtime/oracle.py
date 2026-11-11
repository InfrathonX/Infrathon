"""Public teacher-forced greedy-consistency oracle (DESIGN §6)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from .api import EngineContract, GenerationResult, GenerationSpec, validate_result


@dataclass
class OracleRecord:
    emitted: torch.Tensor
    argmax: torch.Tensor
    margin: torch.Tensor
    top2_gap: torch.Tensor
    prompt_ids: Sequence[torch.Tensor] | None = None
    result: GenerationResult | None = None
    spec: GenerationSpec | None = None
    contract: EngineContract | None = None


@torch.inference_mode()
def teacher_forced_check(reference_model, y: torch.Tensor, prompt_len: int, eps_hard: float) -> OracleRecord:
    if y.dim() != 1 or y.dtype != torch.int64:
        raise ValueError("y must be a 1-D int64 tensor")
    if not 1 <= prompt_len < y.numel():
        raise ValueError("prompt_len must leave at least one emitted token")
    outputs = reference_model(input_ids=y.unsqueeze(0), use_cache=False)
    logits = outputs.logits[0, prompt_len - 1 : y.numel() - 1].float()
    emitted = y[prompt_len:]
    maxima, argmax = logits.max(dim=-1)
    emitted_logits = logits.gather(-1, emitted[:, None]).squeeze(-1)
    top2 = torch.topk(logits, 2, dim=-1).values
    return OracleRecord(
        emitted=emitted.detach(),
        argmax=argmax.detach(),
        margin=(maxima - emitted_logits).detach(),
        top2_gap=(top2[:, 0] - top2[:, 1]).detach(),
    )


def judge_outputs(records: Sequence[OracleRecord], eps_hard: float, p_soft: float,
                  tiers: Sequence[Mapping[str, Any]] | None = None) -> dict:
    if not records:
        raise ValueError("at least one oracle record is required")
    if eps_hard < 0 or not 0 <= p_soft <= 1:
        raise ValueError("invalid oracle thresholds")
    positions = sum(int(r.emitted.numel()) for r in records)
    soft = sum(int((r.emitted != r.argmax).sum().item()) for r in records)
    max_margin = max(float(r.margin.max().item()) if r.margin.numel() else 0.0 for r in records)
    o1 = max_margin <= eps_hard
    soft_rate = soft / positions if positions else 0.0
    o2 = soft_rate <= p_soft
    tier_results = []
    tier_ok = True
    for index, tier in enumerate(tiers or []):
        if set(tier) != {"margin", "max_rate"}:
            raise ValueError(f"oracle tier {index} fields differ")
        threshold = float(tier["margin"])
        max_rate = float(tier["max_rate"])
        if threshold < 0 or not 0 <= max_rate <= 1:
            raise ValueError(f"oracle tier {index} is invalid")
        count = sum(int((record.margin > threshold).sum().item()) for record in records)
        rate = count / positions if positions else 0.0
        passed = rate <= max_rate
        tier_ok = tier_ok and passed
        tier_results.append({"margin": threshold, "max_rate": max_rate,
                             "count": count, "rate": rate, "passed": passed})
    structure_error = None
    for record in records:
        supplied = (record.prompt_ids, record.result, record.spec, record.contract)
        if any(x is not None for x in supplied):
            if any(x is None for x in supplied):
                structure_error = "incomplete structural oracle context"
                break
            try:
                validate_result(record.prompt_ids, record.result, record.spec, record.contract)
            except ValueError as exc:
                structure_error = str(exc)
                break
    o34 = structure_error is None
    return {
        "passed": bool(o1 and o2 and tier_ok and o34),
        "o1_passed": bool(o1),
        "o2_passed": bool(o2),
        "tiers_passed": bool(tier_ok),
        "tier_results": tier_results,
        "o3_o4_passed": bool(o34),
        "positions": positions,
        "soft_divergences": soft,
        "soft_rate": soft_rate,
        "max_margin": max_margin,
        "eps_hard": eps_hard,
        "p_soft": p_soft,
        "structure_error": structure_error,
    }
