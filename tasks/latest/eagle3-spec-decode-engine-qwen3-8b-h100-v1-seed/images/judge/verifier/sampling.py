"""Deterministic HMAC ranking for interactive anchor selection."""

from __future__ import annotations

import hashlib
import hmac
from typing import Any


def hmac_sample(
    rows: list[dict[str, Any]],
    count: int,
    secret: bytes,
    domain: str,
) -> list[dict[str, Any]]:
    """Select the first ``count`` rows under secret domain-separated ranking."""
    if not secret or not 0 <= count <= len(rows):
        raise ValueError("invalid HMAC sample request")
    ranked = sorted(
        rows,
        key=lambda row: hmac.new(
            secret,
            (domain + ":" + row["prompt_id"]).encode(),
            hashlib.sha256,
        ).digest(),
    )
    return ranked[:count]
