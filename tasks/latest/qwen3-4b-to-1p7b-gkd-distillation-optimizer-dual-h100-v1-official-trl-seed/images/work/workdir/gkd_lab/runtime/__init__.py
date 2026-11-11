"""Frozen runtime contract for the Qwen3 GKD distillation optimization task."""

from .api import AUX_NAMES, GKDResult, ObjectiveContract, normalize_result

__all__ = ["AUX_NAMES", "GKDResult", "ObjectiveContract", "normalize_result"]
