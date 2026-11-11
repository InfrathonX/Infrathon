"""Pinned full-Liger Qwen3 hook plus the stock fused ORPO objective."""

from __future__ import annotations

import torch

from liger_kernel.chunked_loss import LigerFusedLinearORPOLoss
from liger_kernel.transformers.rms_norm import LigerRMSNorm
from liger_kernel.transformers.rope import liger_rotary_pos_emb
from liger_kernel.transformers.swiglu import LigerSwiGLUMLP
from runtime.api import ORPOResult, ObjectiveContract, Qwen3ModelPatch


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


def build_model_patch(
    contract: ObjectiveContract,
) -> Qwen3ModelPatch:
    """Declare pinned kernels; task-owned runtime performs every attribute write."""

    del contract
    return Qwen3ModelPatch(
        rope=liger_rotary_pos_emb,
        rms_norm_forward=LigerRMSNorm.forward,
        swiglu_forward=LigerSwiGLUMLP.forward,
    )
