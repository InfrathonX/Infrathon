"""Thin ABI adapter around the pinned, unmodified Liger fused ORPO loss."""

from __future__ import annotations

import torch

from liger_kernel.chunked_loss import LigerFusedLinearORPOLoss
from runtime.api import ORPOResult, ObjectiveContract


class LigerORPOObjective(torch.nn.Module):
    def __init__(self, contract: ObjectiveContract):
        super().__init__()
        self.implementation = LigerFusedLinearORPOLoss(
            ignore_index=contract.ignore_index,
            beta=contract.beta,
            compute_nll_loss=True,
            compiled=True,
            chunk_size=1,
        )

    def forward(
        self,
        weight: torch.Tensor,
        hidden_states: torch.Tensor,
        labels: torch.Tensor,
        bias: torch.Tensor | None = None,
        nll_target: torch.Tensor | None = None,
    ) -> ORPOResult:
        loss, aux = self.implementation(
            weight,
            hidden_states,
            labels,
            bias,
            nll_target=nll_target,
        )
        return ORPOResult(loss, tuple(aux))


def build_objective(contract: ObjectiveContract) -> torch.nn.Module:
    return LigerORPOObjective(contract)
