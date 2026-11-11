"""Deliberately policy-invalid candidate used only by host smoke tests."""

import subprocess  # source admission must reject process creation imports

import torch


def assign(
    x: torch.Tensor,
    centroids: torch.Tensor,
    x_sq: torch.Tensor,
) -> torch.Tensor:
    return torch.zeros(x.shape[:2], dtype=torch.int32, device=x.device)


def update(
    x: torch.Tensor,
    labels: torch.Tensor,
    old_centroids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    counts = torch.zeros(
        old_centroids.shape[:2],
        dtype=torch.int32,
        device=x.device,
    )
    return counts, old_centroids.clone()

