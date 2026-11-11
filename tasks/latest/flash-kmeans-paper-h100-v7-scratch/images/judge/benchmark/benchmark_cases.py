"""Private immutable case schema for the K-means kernel Judge.

Concrete performance workloads and seeds live in the root-only
``hidden_cases.py`` module.  Keeping this module to the schema and binary-size
constants avoids shipping any dead or ambiguous "public benchmark" manifest.

Unit convention
---------------
The benchmark workloads use compact ``K`` and ``M`` labels. This executable
manifest makes their integer convention explicit and canonicalizes:

* ``1K`` to ``K = 2**10 = 1_024``;
* ``1M`` to ``M = 2**20 = 1_048_576``.

Every displayed multiplier is applied literally.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final


K: Final[int] = 1 << 10
M: Final[int] = 1 << 20
ASSIGN_WEIGHT: Final[float] = 0.10
UPDATE_WEIGHT: Final[float] = 0.10
FULL_WEIGHT: Final[float] = 0.20
CONTROL_DEPLOY_WEIGHT: Final[float] = 0.10
LARGE_K_FAMILY_WEIGHT: Final[float] = 0.25
LARGE_N_HIGH_D_FAMILY_WEIGHT: Final[float] = 0.25
DEPLOYMENT_WEIGHT: Final[float] = (
    CONTROL_DEPLOY_WEIGHT
    + LARGE_K_FAMILY_WEIGHT
    + LARGE_N_HIGH_D_FAMILY_WEIGHT
)
RUNTIME_WEIGHT: Final[float] = ASSIGN_WEIGHT + UPDATE_WEIGHT + FULL_WEIGHT
SCORE_PROFILE: Final[str] = "v7_family_tail60"
SCORE_GROUP_WEIGHTS: Final[dict[str, float]] = {
    "assign": ASSIGN_WEIGHT,
    "update": UPDATE_WEIGHT,
    "full": FULL_WEIGHT,
    "deploy_control": CONTROL_DEPLOY_WEIGHT,
    "deploy_large_k": LARGE_K_FAMILY_WEIGHT,
    "deploy_large_n_high_d": LARGE_N_HIGH_D_FAMILY_WEIGHT,
}

_ALLOWED_VISIBILITIES = frozenset({"hidden"})
_ALLOWED_ROLES = frozenset({"performance", "correctness"})
_ALLOWED_STAGES = frozenset({"assign", "update", "full"})
_ALLOWED_COMPONENTS = frozenset(
    {"assignment", "centroid_update", "combined_iteration", "deployment_bundle"}
)
_ALLOWED_MEASUREMENT_MODES = frozenset({"steady_state", "fresh_process"})
_ALLOWED_SCORE_GROUPS = frozenset(SCORE_GROUP_WEIGHTS)
_COMPONENT_STAGES = {
    "assignment": "assign",
    "centroid_update": "update",
    "combined_iteration": "full",
    "deployment_bundle": "full",
}


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    """One immutable private benchmark or correctness specification.

    ``weight`` is an absolute score weight. Performance weights are normalized
    by the concrete private manifest. Correctness cases are hard gates and
    always have zero weight.
    """

    name: str
    suite: str
    visibility: str
    role: str
    workload_label: str
    stage: str
    component: str
    measurement_mode: str
    b: int
    n: int
    k: int
    d: int
    warmup: int
    iters: int
    repeats: int
    weight: float
    score_group: str
    data_family: str = "gaussian_high_margin"
    verification: str = "sampled_exact"
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.name or not self.suite:
            raise ValueError("case name and suite must be non-empty")
        if self.visibility not in _ALLOWED_VISIBILITIES:
            raise ValueError(f"invalid visibility: {self.visibility}")
        if self.role not in _ALLOWED_ROLES:
            raise ValueError(f"invalid role: {self.role}")
        if self.stage not in _ALLOWED_STAGES:
            raise ValueError(f"invalid stage: {self.stage}")
        if not self.workload_label:
            raise ValueError("workload label must be non-empty")
        if self.component not in _ALLOWED_COMPONENTS:
            raise ValueError(f"invalid component: {self.component}")
        if self.measurement_mode not in _ALLOWED_MEASUREMENT_MODES:
            raise ValueError(
                f"invalid measurement mode: {self.measurement_mode}"
            )
        expected_stage = _COMPONENT_STAGES[self.component]
        if self.stage != expected_stage:
            raise ValueError(
                f"component {self.component!r} requires stage "
                f"{expected_stage!r}, got {self.stage!r}"
            )
        if (
            self.measurement_mode == "fresh_process"
        ) != (
            self.component == "deployment_bundle"
        ):
            raise ValueError(
                "fresh_process measurement mode is reserved for the "
                "deployment_bundle component"
            )
        if min(self.b, self.n, self.k, self.d) <= 0:
            raise ValueError("B, N, K, and D must be positive")
        if self.warmup < 0 or self.iters <= 0 or self.repeats <= 0:
            raise ValueError("invalid warmup/iters/repeats")
        if not math.isfinite(self.weight) or self.weight < 0.0:
            raise ValueError("weight must be finite and non-negative")
        if self.role == "correctness" and self.weight != 0.0:
            raise ValueError("correctness-only cases must have weight zero")
        if self.role == "performance":
            if self.score_group not in _ALLOWED_SCORE_GROUPS:
                raise ValueError(f"invalid score group: {self.score_group}")
        elif self.score_group:
            raise ValueError("correctness-only cases must not have a score group")

    @property
    def shape(self) -> tuple[int, int, int, int]:
        """Return the canonical ``(B, N, K, D)`` tuple."""

        return (self.b, self.n, self.k, self.d)

    @property
    def timed_calls(self) -> int:
        """Number of measured calls requested by this case."""

        return self.iters * self.repeats


__all__ = [
    "ASSIGN_WEIGHT",
    "CONTROL_DEPLOY_WEIGHT",
    "DEPLOYMENT_WEIGHT",
    "FULL_WEIGHT",
    "K",
    "LARGE_K_FAMILY_WEIGHT",
    "LARGE_N_HIGH_D_FAMILY_WEIGHT",
    "M",
    "RUNTIME_WEIGHT",
    "SCORE_PROFILE",
    "SCORE_GROUP_WEIGHTS",
    "UPDATE_WEIGHT",
    "BenchmarkCase",
]
