"""v001: thin factory for the pinned official Hugging Face TRL baseline."""

from __future__ import annotations

from runtime.api import ObjectiveContract, build_official_trl_objective


def build_objective(contract: ObjectiveContract):
    """Return the provenance-checked official TRL 0.12.1 objective adapter."""

    return build_official_trl_objective(contract)
