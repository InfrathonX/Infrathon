"""Policy-valid but mathematically wrong candidate for correctness-gate smoke."""

from __future__ import annotations

import torch


@torch.no_grad()
def assign(
    x: torch.Tensor,
    centroids: torch.Tensor,
    x_sq: torch.Tensor,
) -> torch.Tensor:
    # A constant assignment is wrong for the deterministic smoke cases.
    return torch.zeros(x.shape[:2], dtype=torch.int32, device=x.device)


@torch.no_grad()
def update(
    x: torch.Tensor,
    labels: torch.Tensor,
    old_centroids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Shape/dtype/device are plausible, but counts and non-empty means are wrong.
    counts = torch.zeros(
        old_centroids.shape[:2],
        dtype=torch.int32,
        device=x.device,
    )
    return counts, old_centroids.clone()

