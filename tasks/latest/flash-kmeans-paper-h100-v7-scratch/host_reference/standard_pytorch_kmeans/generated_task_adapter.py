"""Faithful PyTorch implementation of paper Algorithm 1.

This is intentionally the conventional, materializing implementation:

1. build the complete FP32 ``[B, N, K]`` squared-distance matrix;
2. run a separate argmin reduction;
3. aggregate FP32 sums and integer counts with ``index_add_``;
4. normalize the sums.

It deliberately does not chunk or stream the distance matrix.  That behavior
is the defining property of the paper's standard baseline.
"""

from __future__ import annotations

import torch


@torch.no_grad()
def assign(
    x: torch.Tensor,
    centroids: torch.Tensor,
    x_sq: torch.Tensor,
) -> torch.Tensor:
    """Materialize all distances, then return the smallest-id argmin."""
    x_f32 = x.float()
    centroids_f32 = centroids.float()
    centroid_sq = centroids_f32.square().sum(dim=-1)

    distances = torch.bmm(
        x_f32,
        centroids_f32.transpose(1, 2),
    )
    distances.mul_(-2.0)
    distances.add_(x_sq.unsqueeze(-1))
    distances.add_(centroid_sq.unsqueeze(-2))
    return distances.argmin(dim=-1)


@torch.no_grad()
def update(
    x: torch.Tensor,
    labels: torch.Tensor,
    old_centroids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Atomic-style scatter aggregation followed by normalization."""
    batch, points, dimensions = x.shape
    clusters = old_centroids.shape[1]

    cluster_sums = torch.zeros(
        (batch, clusters, dimensions),
        device=x.device,
        dtype=torch.float32,
    )
    cluster_counts = torch.zeros(
        (batch, clusters),
        device=x.device,
        dtype=torch.int32,
    )

    for batch_id in range(batch):
        batch_labels = labels[batch_id].to(torch.int64)
        cluster_sums[batch_id].index_add_(
            0,
            batch_labels,
            x[batch_id].float(),
        )
        cluster_counts[batch_id].index_add_(
            0,
            batch_labels,
            torch.ones(
                (points,),
                device=x.device,
                dtype=torch.int32,
            ),
        )

    normalized = (
        cluster_sums
        / cluster_counts.clamp_min(1).float().unsqueeze(-1)
    ).to(x.dtype)
    new_centroids = torch.where(
        (cluster_counts > 0).unsqueeze(-1),
        normalized,
        old_centroids,
    )
    return cluster_counts, new_centroids


@torch.no_grad()
def standard_kmeans_iteration(
    x: torch.Tensor,
    centroids: torch.Tensor,
    x_sq: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run exactly one assignment/update iteration."""
    labels = assign(x, centroids, x_sq)
    counts, new_centroids = update(x, labels, centroids)
    return labels, counts, new_centroids
