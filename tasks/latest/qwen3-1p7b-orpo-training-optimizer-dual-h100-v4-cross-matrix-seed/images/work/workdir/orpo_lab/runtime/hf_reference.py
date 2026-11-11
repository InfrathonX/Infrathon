"""Official Hugging Face TRL 0.12.1 ORPO objective adapter.

The ORPO log-probability and odds-ratio implementations executed here are the
methods installed from the pinned ``trl`` wheel.  This module only adapts the
task's already next-token-aligned tensor ABI and packages TRL's outputs as an
``ORPOResult``; it does not reimplement the ORPO equations.
"""

from __future__ import annotations

import hashlib
import inspect
from dataclasses import dataclass
from functools import lru_cache
from importlib import metadata
from pathlib import Path

import torch
from torch import nn

from .api import ORPOResult, ObjectiveContract, validate_inputs


OFFICIAL_TRL_VERSION = "0.12.1"
OFFICIAL_TRL_TAG = "v0.12.1"
OFFICIAL_TRL_COMMIT = "c1e9ea8ecfdaf1486e2fc84bc55379b2338d5320"
OFFICIAL_TRL_SOURCE_SHA256 = (
    "ac04e1e4b530af2f8fc64e5d47202b4a123a74bf08f1ccc0b60ba250a670c0fe"
)
OFFICIAL_TRL_MODULE = "trl.trainer.orpo_trainer"


@dataclass(frozen=True)
class _OfficialTRLBinding:
    trainer_type: type
    source_path: Path
    source_sha256: str


@dataclass(frozen=True)
class _AcceleratorDevice:
    device: torch.device


@dataclass(frozen=True)
class _OddsRatioContext:
    """Minimal receiver used by the official unbound instance method."""

    beta: float
    accelerator: _AcceleratorDevice


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
            f"official ORPO baseline requires trl=={OFFICIAL_TRL_VERSION}; "
            f"found {installed_version}"
        )

    import trl
    from trl.trainer.orpo_trainer import ORPOTrainer

    package_root = Path(trl.__file__).resolve(strict=True).parent
    expected_source = package_root / "trainer" / "orpo_trainer.py"
    source_path = Path(inspect.getfile(ORPOTrainer)).resolve(strict=True)
    if source_path != expected_source or ORPOTrainer.__module__ != OFFICIAL_TRL_MODULE:
        raise RuntimeError("ORPOTrainer was not imported from the installed official TRL module")

    source_sha256 = _sha256_file(source_path)
    if source_sha256 != OFFICIAL_TRL_SOURCE_SHA256:
        raise RuntimeError(
            "official TRL ORPO source hash mismatch: "
            f"{source_sha256} != {OFFICIAL_TRL_SOURCE_SHA256}"
        )
    for method_name in ("get_batch_logps", "odds_ratio_loss"):
        method = inspect.getattr_static(ORPOTrainer, method_name)
        function = method.__func__ if isinstance(method, staticmethod) else method
        if getattr(function, "__module__", None) != OFFICIAL_TRL_MODULE:
            raise RuntimeError(f"ORPOTrainer.{method_name} is not official TRL source")
        if Path(inspect.getsourcefile(function) or "").resolve() != source_path:
            raise RuntimeError(f"ORPOTrainer.{method_name} source path mismatch")
    return _OfficialTRLBinding(ORPOTrainer, source_path, source_sha256)


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


class OfficialTRLORPOObjective(nn.Module):
    """Thin tensor-ABI adapter around official TRL ``ORPOTrainer`` methods."""

    def __init__(self, contract: ObjectiveContract):
        super().__init__()
        if contract.ignore_index != -100:  # defensive even though the contract checks it
            raise ValueError("official TRL ORPO fixes label_pad_token_id at -100")
        self.contract = contract
        self._official_binding = _load_official_trl_binding()

    def forward(
        self,
        weight: torch.Tensor,
        hidden_states: torch.Tensor,
        labels: torch.Tensor,
        bias: torch.Tensor | None = None,
        nll_target: torch.Tensor | None = None,
    ) -> ORPOResult:
        validate_inputs(weight, hidden_states, labels, bias, nll_target, self.contract)
        logits = hidden_states @ weight.t()
        if bias is not None:
            logits = logits + bias
        pair_batch = hidden_states.shape[0] // 2

        # TRL's public helper normally performs the decoder-only next-token
        # shift itself.  The MLSys Forge ABI is already shifted by the trusted
        # TrainStepModule, so is_encoder_decoder=True selects the official
        # helper's aligned-tensor branch without shifting a second time.
        trainer_type = self._official_binding.trainer_type
        sequence_logps = trainer_type.get_batch_logps(
            logits,
            labels,
            average_log_prob=True,
            label_pad_token_id=-100,
            is_encoder_decoder=True,
        )
        chosen_logps = sequence_logps[:pair_batch]
        rejected_logps = sequence_logps[pair_batch:]

        # ORPOTrainer.concatenated_forward uses nn.CrossEntropyLoss on chosen
        # decoder-only next-token targets.  Our ABI supplies those already
        # shifted, with nll_target independently retaining prompt targets.
        nll_labels = labels if nll_target is None else nll_target
        chosen_nll = nn.CrossEntropyLoss()(
            logits[:pair_batch].contiguous().view(-1, logits.shape[-1]),
            nll_labels[:pair_batch].contiguous().view(-1),
        )

        odds_context = _OddsRatioContext(
            beta=float(self.contract.beta),
            accelerator=_AcceleratorDevice(logits.device),
        )
        losses, chosen_rewards, rejected_rewards, ratio_mean, log_odds_mean = (
            trainer_type.odds_ratio_loss(
                odds_context,
                chosen_logps,
                rejected_logps,
            )
        )
        loss = chosen_nll - losses.mean()
        aux = (
            chosen_logps,
            rejected_logps,
            logits[:pair_batch].detach().mean(),
            logits[pair_batch:].detach().mean(),
            chosen_nll,
            chosen_rewards,
            rejected_rewards,
            ratio_mean,
            log_odds_mean,
        )
        return ORPOResult(loss, aux)


def reference_orpo(
    weight: torch.Tensor,
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    bias: torch.Tensor | None = None,
    nll_target: torch.Tensor | None = None,
    *,
    contract: ObjectiveContract,
) -> ORPOResult:
    """Compatibility function backed by the provenance-checked TRL module."""

    return OfficialTRLORPOObjective(contract)(
        weight,
        hidden_states,
        labels,
        bias,
        nll_target,
    )


def _clear_official_trl_binding_cache_for_tests() -> None:
    """Permit isolated provenance-negative tests without exposing a public knob."""

    _load_official_trl_binding.cache_clear()
