"""Stable, Agent-visible objective ABI.

The task-owned driver aligns language-model labels before invoking the
candidate.  Each batch row is one independent prompt-plus-completion sequence;
labels are already next-token shifted and completion-masked.  Hugging Face
TRL's GKD contract fixes the ignored-label sentinel at ``-100``; it is
deliberately not a tunable part of this benchmark.

The candidate receives pre-projection hidden states and the tied lm-head
weights of both models.  Neither projection is performed by the harness: the
student and teacher vocabulary projections and the generalized JSD reduction
are the candidate's optimization surface.  Teacher tensors never require
gradients.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, NamedTuple, Protocol

import torch


OFFICIAL_TRL_IGNORE_INDEX = -100


AUX_NAMES = (
    "student_completion_logps",
    "teacher_completion_logps",
    "student_logits_mean",
    "teacher_logits_mean",
    "jsd_per_sequence",
    "kl_teacher_mixture_mean",
    "kl_student_mixture_mean",
    "student_entropy_mean",
    "teacher_entropy_mean",
)


@dataclass(frozen=True)
class ObjectiveContract:
    """Immutable semantics supplied by the trusted runtime."""

    beta: float = 0.5
    temperature: float = 0.9
    ignore_index: int = OFFICIAL_TRL_IGNORE_INDEX

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.beta) <= 1.0:
            raise ValueError("beta must be in [0, 1]")
        if not float(self.temperature) > 0.0:
            raise ValueError("temperature must be positive")
        if isinstance(self.ignore_index, bool) or not isinstance(self.ignore_index, int):
            raise TypeError("ignore_index must be an integer")
        if self.ignore_index != OFFICIAL_TRL_IGNORE_INDEX:
            raise ValueError("official TRL GKD requires ignore_index=-100")


class ModelOptimizer(Protocol):
    """Legacy host-calibration compatibility hook; not a participant ABI.

    Participant admission reserves and rejects ``optimize_model``. Released
    candidates can customize model execution only through
    :class:`Qwen3ModelPatch`; this protocol remains solely so historical host
    calibration archives can still be inspected.
    """

    def __call__(
        self,
        model: torch.nn.Module,
        contract: ObjectiveContract,
    ) -> torch.nn.Module | None: ...


@dataclass(frozen=True)
class Qwen3ModelPatch:
    """Declarative, source-policy-compatible Qwen3 executable patch.

    Candidate code supplies callables only.  The trusted runtime owns every
    global or nested attribute write needed to install them after loading the
    pinned models and before FSDP.  The same patch shape applies to the
    trainable student (``build_model_patch``) and the frozen teacher
    (``build_teacher_patch``); the runtime installs each independently.
    """

    rope: Callable[..., object] | None = None
    rms_norm_forward: Callable[..., object] | None = None
    swiglu_forward: Callable[..., object] | None = None

    def __post_init__(self) -> None:
        values = (self.rope, self.rms_norm_forward, self.swiglu_forward)
        if not any(value is not None for value in values):
            raise ValueError("Qwen3ModelPatch must provide at least one callable")
        if any(value is not None and not callable(value) for value in values):
            raise TypeError("every Qwen3ModelPatch field must be callable or None")


class GKDResult(NamedTuple):
    """Canonical candidate result: scalar loss plus nine auxiliary tensors."""

    loss: torch.Tensor
    aux: tuple[torch.Tensor, ...]


def build_official_trl_objective(contract: ObjectiveContract) -> torch.nn.Module:
    """Build the provenance-checked official TRL baseline.

    This public factory keeps the starter candidate free of task-authored GKD
    loss math.  The local import avoids an ``api``/``hf_reference`` import
    cycle while keeping candidate imports inside the admitted ``runtime.api``
    surface.
    """

    from .hf_reference import OfficialTRLGKDObjective

    return OfficialTRLGKDObjective(contract)


def normalize_result(value: object) -> GKDResult:
    """Normalize the narrow tuple-compatible return surface."""

    if isinstance(value, GKDResult):
        result = value
    elif isinstance(value, tuple) and len(value) == 2:
        loss, aux = value
        if not isinstance(aux, (tuple, list)):
            raise TypeError("candidate auxiliary output must be a tuple or list")
        result = GKDResult(loss, tuple(aux))
    else:
        raise TypeError("candidate must return GKDResult or (loss, aux)")
    if not isinstance(result.loss, torch.Tensor) or result.loss.ndim != 0:
        raise TypeError("candidate loss must be a scalar torch.Tensor")
    if len(result.aux) != len(AUX_NAMES):
        raise ValueError(f"candidate must return {len(AUX_NAMES)} auxiliary tensors")
    if not all(isinstance(item, torch.Tensor) for item in result.aux):
        raise TypeError("every auxiliary output must be a torch.Tensor")
    return result


def validate_inputs(
    student_weight: torch.Tensor,
    student_hidden: torch.Tensor,
    teacher_weight: torch.Tensor,
    teacher_hidden: torch.Tensor,
    labels: torch.Tensor,
    student_bias: torch.Tensor | None,
    teacher_bias: torch.Tensor | None,
    contract: ObjectiveContract,
) -> None:
    """Shared shape checks; formal inputs are generated by the trusted driver."""

    if student_hidden.ndim != 3 or teacher_hidden.ndim != 3:
        raise ValueError("expected student/teacher hidden of shape [B,T,H]")
    if student_weight.ndim != 2 or teacher_weight.ndim != 2 or labels.ndim != 2:
        raise ValueError("expected weight [V,H] and labels [B,T]")
    if student_hidden.shape[0] == 0:
        raise ValueError("GKD requires a non-empty batch")
    if student_hidden.shape[:2] != labels.shape:
        raise ValueError("student_hidden and labels batch/sequence dimensions differ")
    if teacher_hidden.shape[:2] != labels.shape:
        raise ValueError("teacher_hidden and labels batch/sequence dimensions differ")
    if student_hidden.shape[2] != student_weight.shape[1]:
        raise ValueError("student hidden size differs from student lm-head weight")
    if teacher_hidden.shape[2] != teacher_weight.shape[1]:
        raise ValueError("teacher hidden size differs from teacher lm-head weight")
    if student_weight.shape[0] != teacher_weight.shape[0]:
        raise ValueError("student and teacher must share one vocabulary size")
    if student_bias is not None and tuple(student_bias.shape) != (student_weight.shape[0],):
        raise ValueError("student_bias must have shape [V]")
    if teacher_bias is not None and tuple(teacher_bias.shape) != (teacher_weight.shape[0],):
        raise ValueError("teacher_bias must have shape [V]")
    if teacher_hidden.requires_grad or teacher_weight.requires_grad:
        raise ValueError("teacher tensors must not require gradients")
    if teacher_bias is not None and teacher_bias.requires_grad:
        raise ValueError("teacher_bias must not require gradients")
    valid_per_sequence = labels.ne(contract.ignore_index).sum(dim=-1)
    if bool((valid_per_sequence == 0).any()):
        raise ValueError("each sequence needs at least one scored token")
