"""Pinned full-Liger Qwen3 hook plus the stock fused ORPO objective (v002).

Identical to v001 except for one glue fix found on the formal Judge: the
official TRL 0.12.1 ORPO path returns ``chosen_logits_mean`` and
``rejected_logits_mean`` in the model dtype (BF16), while the stock
``LigerFusedLinearORPOLoss`` returns them in FP32; the Judge checks the
auxiliary dtypes, so the adapter casts those two (aux[2], aux[3]) to the
hidden-state dtype.  Loss, gradients and all other auxiliaries are untouched.
"""

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
        aux = list(aux)
        for index in (2, 3):
            aux[index] = aux[index].to(hidden_states.dtype)
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
