"""Official Hugging Face TRL 0.12.1 GKD objective adapter.

The generalized Jensen-Shannon divergence executed here is the
``GKDTrainer.generalized_jsd_loss`` static method installed from the pinned
``trl`` wheel.  This module only adapts the task's already next-token-aligned
tensor ABI — performing the student and teacher vocabulary projections and
packaging task-owned auxiliary witnesses as a ``GKDResult`` — it does not
reimplement the GKD equations.
"""

from __future__ import annotations

import hashlib
import inspect
from dataclasses import dataclass
from functools import lru_cache
from importlib import metadata
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from .api import GKDResult, ObjectiveContract, validate_inputs


OFFICIAL_TRL_VERSION = "0.12.1"
OFFICIAL_TRL_TAG = "v0.12.1"
OFFICIAL_TRL_COMMIT = "c1e9ea8ecfdaf1486e2fc84bc55379b2338d5320"
OFFICIAL_TRL_SOURCE_SHA256 = (
    "09e6e8932297ca3793aa5c429852bb46e3f395f9f3aaa48479d3b7965a028f1a"
)
OFFICIAL_TRL_MODULE = "trl.trainer.gkd_trainer"


@dataclass(frozen=True)
class _OfficialTRLBinding:
    trainer_type: type
    source_path: Path
    source_sha256: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=1)
def _load_official_trl_binding() -> _OfficialTRLBinding:
    """Import and fail closed on anything except the pinned official source."""

    installed_version = metadata.version("trl")
    if installed_version != OFFICIAL_TRL_VERSION:
        raise RuntimeError(
            f"official GKD baseline requires trl=={OFFICIAL_TRL_VERSION}; "
            f"found {installed_version}"
        )

    import trl
    from trl.trainer.gkd_trainer import GKDTrainer

    package_root = Path(trl.__file__).resolve(strict=True).parent
    expected_source = package_root / "trainer" / "gkd_trainer.py"
    source_path = Path(inspect.getfile(GKDTrainer)).resolve(strict=True)
    if source_path != expected_source or GKDTrainer.__module__ != OFFICIAL_TRL_MODULE:
        raise RuntimeError("GKDTrainer was not imported from the installed official TRL module")

    source_sha256 = _sha256_file(source_path)
    if source_sha256 != OFFICIAL_TRL_SOURCE_SHA256:
        raise RuntimeError(
            "official TRL GKD source hash mismatch: "
            f"{source_sha256} != {OFFICIAL_TRL_SOURCE_SHA256}"
        )
    method = inspect.getattr_static(GKDTrainer, "generalized_jsd_loss")
    function = method.__func__ if isinstance(method, staticmethod) else method
    if getattr(function, "__module__", None) != OFFICIAL_TRL_MODULE:
        raise RuntimeError("GKDTrainer.generalized_jsd_loss is not official TRL source")
    if Path(inspect.getsourcefile(function) or "").resolve() != source_path:
        raise RuntimeError("GKDTrainer.generalized_jsd_loss source path mismatch")
    return _OfficialTRLBinding(GKDTrainer, source_path, source_sha256)


def official_trl_provenance() -> dict[str, str]:
    """Return the runtime-verified upstream identity for tests and telemetry."""

    binding = _load_official_trl_binding()
    return {
        "distribution": "trl",
        "version": OFFICIAL_TRL_VERSION,
        "tag": OFFICIAL_TRL_TAG,
        "commit": OFFICIAL_TRL_COMMIT,
        "module": OFFICIAL_TRL_MODULE,
        "source_path": str(binding.source_path),
        "source_sha256": binding.source_sha256,
    }


def _masked_sequence_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-sequence mean of ``values`` over scored positions; mask is [B, T]."""

    counts = mask.sum(dim=-1).clamp_min(1)
    return (values * mask).sum(dim=-1) / counts


class OfficialTRLGKDObjective(nn.Module):
    """Thin tensor-ABI adapter around the official TRL ``GKDTrainer`` method."""

    def __init__(self, contract: ObjectiveContract):
        super().__init__()
        if contract.ignore_index != -100:  # defensive even though the contract checks it
            raise ValueError("official TRL GKD fixes the ignore index at -100")
        self.contract = contract
        self._official_binding = _load_official_trl_binding()

    def forward(
        self,
        student_weight: torch.Tensor,
        student_hidden: torch.Tensor,
        teacher_weight: torch.Tensor,
        teacher_hidden: torch.Tensor,
        labels: torch.Tensor,
        student_bias: torch.Tensor | None = None,
        teacher_bias: torch.Tensor | None = None,
    ) -> GKDResult:
        validate_inputs(
            student_weight,
            student_hidden,
            teacher_weight,
            teacher_hidden,
            labels,
            student_bias,
            teacher_bias,
            self.contract,
        )
        student_logits = student_hidden @ student_weight.t()
        if student_bias is not None:
            student_logits = student_logits + student_bias
        with torch.no_grad():
            teacher_logits = teacher_hidden @ teacher_weight.t()
            if teacher_bias is not None:
                teacher_logits = teacher_logits + teacher_bias

        # The MLSys Forge ABI supplies already-shifted hidden states and
        # completion-masked labels, exactly the tensors GKDTrainer.compute_loss
        # passes to its static loss after slicing; the official method applies
        # temperature scaling, the beta mixture, and the batchmean reduction.
        trainer_type = self._official_binding.trainer_type
        loss = trainer_type.generalized_jsd_loss(
            student_logits,
            teacher_logits,
            labels=labels,
            beta=float(self.contract.beta),
            temperature=float(self.contract.temperature),
            reduction="batchmean",
        )

        aux = self._auxiliary_witnesses(student_logits, teacher_logits, labels)
        return GKDResult(loss, aux)

    def _auxiliary_witnesses(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Task-owned correctness witnesses over the temperature-scaled
        distributions the official loss consumes.  Computed without autograd:
        the loss tensor is the only differentiable output of the objective.
        """

        beta = float(self.contract.beta)
        temperature = float(self.contract.temperature)
        with torch.no_grad():
            mask = labels.ne(self.contract.ignore_index)
            mask_f32 = mask.to(torch.float32)
            gather_index = labels.clamp_min(0).unsqueeze(-1)

            student_logits_mean = student_logits[mask].mean()
            teacher_logits_mean = teacher_logits[mask].mean()

            student_log_probs = F.log_softmax(
                student_logits.to(torch.float32) / temperature, dim=-1
            )
            teacher_log_probs = F.log_softmax(
                teacher_logits.to(torch.float32) / temperature, dim=-1
            )

            student_token_logps = student_log_probs.gather(-1, gather_index).squeeze(-1)
            teacher_token_logps = teacher_log_probs.gather(-1, gather_index).squeeze(-1)
            student_completion_logps = _masked_sequence_mean(student_token_logps, mask_f32)
            teacher_completion_logps = _masked_sequence_mean(teacher_token_logps, mask_f32)

            student_entropy = -(student_log_probs.exp() * student_log_probs).sum(dim=-1)
            teacher_entropy = -(teacher_log_probs.exp() * teacher_log_probs).sum(dim=-1)
            scored = mask_f32.sum().clamp_min(1)
            student_entropy_mean = (student_entropy * mask_f32).sum() / scored
            teacher_entropy_mean = (teacher_entropy * mask_f32).sum() / scored

            # The same log-space mixture the official method constructs:
            # M = beta * P_student + (1 - beta) * P_teacher.
            beta_tensor = torch.tensor(beta, dtype=student_log_probs.dtype)
            mixture_log_probs = torch.logsumexp(
                torch.stack(
                    [
                        student_log_probs + torch.log(beta_tensor),
                        teacher_log_probs + torch.log(1 - beta_tensor),
                    ]
                ),
                dim=0,
            )
            kl_teacher = (
                teacher_log_probs.exp() * (teacher_log_probs - mixture_log_probs)
            ).sum(dim=-1)
            kl_student = (
                student_log_probs.exp() * (student_log_probs - mixture_log_probs)
            ).sum(dim=-1)
            kl_teacher_mixture_mean = (kl_teacher * mask_f32).sum() / scored
            kl_student_mixture_mean = (kl_student * mask_f32).sum() / scored

            jsd_token = beta * kl_teacher + (1.0 - beta) * kl_student
            jsd_per_sequence = _masked_sequence_mean(jsd_token, mask_f32)

        return (
            student_completion_logps,
            teacher_completion_logps,
            student_logits_mean,
            teacher_logits_mean,
            jsd_per_sequence,
            kl_teacher_mixture_mean,
            kl_student_mixture_mean,
            student_entropy_mean,
            teacher_entropy_mean,
        )


def reference_gkd(
    student_weight: torch.Tensor,
    student_hidden: torch.Tensor,
    teacher_weight: torch.Tensor,
    teacher_hidden: torch.Tensor,
    labels: torch.Tensor,
    student_bias: torch.Tensor | None = None,
    teacher_bias: torch.Tensor | None = None,
    *,
    contract: ObjectiveContract,
) -> GKDResult:
    """Compatibility function backed by the provenance-checked TRL module."""

    return OfficialTRLGKDObjective(contract)(
        student_weight,
        student_hidden,
        teacher_weight,
        teacher_hidden,
        labels,
        student_bias,
        teacher_bias,
    )


def _clear_official_trl_binding_cache_for_tests() -> None:
    """Permit isolated provenance-negative tests without exposing a public knob."""

    _load_official_trl_binding.cache_clear()
