"""Host-only AnswerDotAI/fastkmeans Triton task-ABI adapter.

The assignment kernel and H100-default data chunk size come from upstream
fastkmeans.  The update preserves upstream FP32 index_add accumulation.  ABI
bridges add a batch loop, GPU-resident labels/counts, exact integer counts,
and retain-old empty-cluster behavior.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.heuristics(
    {
        "BLOCK_M": lambda x: 128 if x["D"] <= 384 else 64,
        "BLOCK_N": lambda x: 128 if x["D"] <= 384 else 64,
        "BLOCK_K": lambda x: 16 if x["D"] <= 32 or x["D"] > 384 else 32,
        "GROUP_SIZE_M": lambda x: 8 if x["D"] <= 32 else 16,
        "num_warps": lambda x: 4,
    }
)
@triton.jit
def _kmeans_kernel(
    x_ptr,
    x_norm_ptr,
    c_ptr,
    c_norm_ptr,
    best_dist_ptr,
    best_idx_ptr,
    B,
    C,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(B, BLOCK_M)
    num_pid_n = tl.cdiv(C, BLOCK_N)
    num_pid_in_grp = GROUP_SIZE_M * num_pid_n
    first_pid_m = (pid // num_pid_in_grp) * GROUP_SIZE_M
    group_rows = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_grp) % group_rows)
    pid_n = (pid % num_pid_in_grp) // group_rows

    row_start = pid_m * BLOCK_M
    col_start = pid_n * BLOCK_N
    rows = row_start + tl.arange(0, BLOCK_M)
    cols = col_start + tl.arange(0, BLOCK_N)
    row_mask = rows < B
    col_mask = cols < C

    x_n = tl.load(x_norm_ptr + rows, mask=row_mask, other=0.0)
    c_n = tl.load(c_norm_ptr + cols, mask=col_mask, other=0.0)
    dot_acc = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)
    for k0 in range(0, D, BLOCK_K):
        k_range = k0 + tl.arange(0, BLOCK_K)
        x_ptrs = x_ptr + rows[:, None] * D + k_range[None, :]
        xk = tl.load(
            x_ptrs,
            mask=row_mask[:, None],
        ).to(tl.float16)
        c_ptrs = c_ptr + cols[:, None] * D + k_range[None, :]
        ck = tl.load(
            c_ptrs,
            mask=col_mask[:, None],
        ).to(tl.float16)
        dot_acc += tl.dot(
            xk,
            tl.trans(ck),
            out_dtype=tl.float32,
        )

    dist = tl.fma(
        dot_acc,
        -2.0,
        x_n[:, None] + c_n[None, :],
    )
    dist = tl.where(col_mask[None, :], dist, 1e38)
    tile_min, tile_idx = tl.min(
        dist,
        axis=1,
        return_indices=True,
    )
    prev = tl.atomic_min(
        best_dist_ptr + rows,
        tile_min,
        mask=row_mask,
    )
    improved = tile_min < prev
    tl.store(
        best_idx_ptr + rows,
        tl.where(
            improved,
            col_start + tile_idx,
            tl.load(best_idx_ptr + rows),
        ),
        mask=row_mask,
    )


def _triton_kmeans(
    data_chunk,
    data_chunk_norms,
    centroids,
    centroids_sqnorm,
    best_ids,
):
    points, dimensions = data_chunk.shape
    clusters = centroids.shape[0]
    best_dist = torch.full(
        (points,),
        1e38,
        device=data_chunk.device,
        dtype=torch.float32,
    )

    def grid(meta):
        return (
            triton.cdiv(points, meta["BLOCK_M"])
            * triton.cdiv(clusters, meta["BLOCK_N"]),
        )

    _kmeans_kernel[grid](
        data_chunk,
        data_chunk_norms,
        centroids,
        centroids_sqnorm,
        best_dist,
        best_ids,
        points,
        clusters,
        dimensions,
    )


@torch.no_grad()
def assign(
    x: torch.Tensor,
    centroids: torch.Tensor,
    x_sq: torch.Tensor,
) -> torch.Tensor:
    batch, points, _ = x.shape
    labels = torch.empty(
        (batch, points),
        device=x.device,
        dtype=torch.int64,
    )
    chunk_size_data = 51_200
    for batch_id in range(batch):
        centroid_norms = (
            centroids[batch_id] ** 2
        ).sum(dim=1)
        start_idx = 0
        while start_idx < points:
            end_idx = min(start_idx + chunk_size_data, points)
            best_ids = torch.zeros(
                (end_idx - start_idx,),
                device=x.device,
                dtype=torch.int64,
            )
            _triton_kmeans(
                x[batch_id, start_idx:end_idx],
                x_sq[batch_id, start_idx:end_idx].to(x.dtype),
                centroids[batch_id],
                centroid_norms,
                best_ids,
            )
            labels[batch_id, start_idx:end_idx] = best_ids
            start_idx = end_idx
    return labels


@torch.no_grad()
def update(
    x: torch.Tensor,
    labels: torch.Tensor,
    old_centroids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, points, dimensions = x.shape
    clusters = old_centroids.shape[1]
    counts = torch.zeros(
        (batch, clusters),
        device=x.device,
        dtype=torch.int32,
    )
    new_centroids = torch.empty_like(old_centroids)
    chunk_size_data = 51_200

    for batch_id in range(batch):
        cluster_sums = torch.zeros(
            (clusters, dimensions),
            device=x.device,
            dtype=torch.float32,
        )
        cluster_counts = torch.zeros(
            (clusters,),
            device=x.device,
            dtype=torch.int32,
        )
        start_idx = 0
        while start_idx < points:
            end_idx = min(start_idx + chunk_size_data, points)
            best_ids = labels[
                batch_id,
                start_idx:end_idx,
            ].to(torch.int64)
            cluster_sums.index_add_(
                0,
                best_ids,
                x[batch_id, start_idx:end_idx].float(),
            )
            cluster_counts += torch.bincount(
                best_ids,
                minlength=clusters,
            ).to(torch.int32)
            start_idx = end_idx

        counts[batch_id] = cluster_counts
        non_empty = cluster_counts > 0
        means = (
            cluster_sums
            / cluster_counts.clamp(min=1).unsqueeze(1)
        ).to(dtype=x.dtype)
        new_centroids[batch_id] = torch.where(
            non_empty.unsqueeze(-1),
            means,
            old_centroids[batch_id],
        )

    return counts, new_centroids
