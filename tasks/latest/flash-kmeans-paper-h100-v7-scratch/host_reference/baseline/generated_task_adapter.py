"""Task adapter mechanically extracted from upstream flash_kmeans/torch_fallback.py."""

import torch
import torch.nn.functional as F

def euclid_assign_torch_native_chunked(x, centroids, x_sq, chunk_size_N=32768, chunk_size_K=1024):
    """
    Torch naive implementation for assignment with chunking to avoid OOM.
    
    Args:
        x: (B, N, D) input points
        centroids: (B, K, D) cluster centers
        x_sq: (B, N) pre-computed ||x||^2
        chunk_size_N: chunk size along N dimension
        chunk_size_K: chunk size along K dimension
        
    Returns:
        cluster_ids: (B, N) int32 cluster assignment per point
    """
    B, N, D = x.shape
    K = centroids.shape[1]
    assert centroids.shape[2] == x.shape[2], "Dimension mismatch between x and centroids"
    assert x.shape[0] == centroids.shape[0], "Batch size mismatch between x and centroids"

    cent_sq = (centroids ** 2).sum(dim=-1)  # (B, K)

    cluster_ids = torch.empty((B, N), dtype=torch.int32, device=x.device)

    # Process in chunks to avoid OOM
    for n_start in range(0, N, chunk_size_N):
        n_end = min(n_start + chunk_size_N, N)
        x_chunk = x[:, n_start:n_end, :]  # (B, n_chunk, D)
        x_sq_chunk = x_sq[:, n_start:n_end]  # (B, n_chunk)

        dists_chunk = torch.empty((B, n_end - n_start, K), device=x.device)

        for k_start in range(0, K, chunk_size_K):
            k_end = min(k_start + chunk_size_K, K)
            cent_chunk = centroids[:, k_start:k_end, :]  # (B, k_chunk, D)
            cent_sq_chunk = cent_sq[:, k_start:k_end]  # (B, k_chunk)

            # Compute squared distances
            dists_partial = (
                x_sq_chunk.unsqueeze(-1)  # (B, n_chunk, 1)
                - 2 * torch.bmm(x_chunk, cent_chunk.transpose(1, 2))  # (B, n_chunk, k_chunk)
                + cent_sq_chunk.unsqueeze(-2)  # (B, 1, k_chunk)
            )  # (B, n_chunk, k_chunk)

            dists_chunk[:, :, k_start:k_end] = dists_partial

        # Assign cluster ids
        cluster_ids[:, n_start:n_end] = torch.argmin(dists_chunk, dim=-1)
    
    return cluster_ids

def _centroid_update_torch_native(x, cluster_ids, old_centroids, mode = 'euclid'):
    """
    Torch naive implementation for centroid update.
    
    Args:
        x: (B, N, D) input points
        cluster_ids: (B, N) cluster assignment per point
        old_centroids: (B, K, D) previous centroids
        mode: 'euclid' or 'cosine'
        
    Returns:
        centroids_new: (B, K, D) updated centroids
    """
    B, N, D = x.shape
    K = old_centroids.shape[1]
    assert mode in ['euclid', 'cosine'], "Mode must be 'euclid' or 'cosine'"
    

    cluster_sums = torch.zeros((B, K, D), device=x.device, dtype=torch.float32)
    cluster_counts = torch.zeros((B, K), device=x.device, dtype=torch.float32)
    
    for b in range(B):
        # Accumulate sums and counts by torch
        cluster_sums[b].index_add_(0, cluster_ids[b], x[b].to(torch.float32))
        ones = torch.ones((N,), device=x.device, dtype=torch.float32)
        cluster_counts[b].index_add_(0, cluster_ids[b], ones)



    cluster_counts.unsqueeze_(-1)  # Avoid division by zero
    empty_mask = (cluster_counts == 0)
    cluster_counts.clamp_min_(1.0)
    centroids_new = cluster_sums / cluster_counts
    centroids = torch.where(empty_mask, old_centroids, centroids_new)
    if mode == 'cosine':
        centroids = F.normalize(centroids, p=2, dim=-1)
    return centroids.to(x.dtype)


# Task-ABI adapter appended by host calibration tooling.
@torch.no_grad()
def assign(
    x: torch.Tensor,
    centroids: torch.Tensor,
    x_sq: torch.Tensor,
) -> torch.Tensor:
    return euclid_assign_torch_native_chunked(x, centroids, x_sq)


@torch.no_grad()
def update(
    x: torch.Tensor,
    labels: torch.Tensor,
    old_centroids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, _, _ = x.shape
    clusters = old_centroids.shape[1]
    counts = torch.zeros(
        (batch, clusters),
        device=x.device,
        dtype=torch.int32,
    )
    for batch_id in range(batch):
        counts[batch_id] = torch.bincount(
            labels[batch_id].to(torch.int64),
            minlength=clusters,
        ).to(torch.int32)
    centroids = _centroid_update_torch_native(
        x,
        labels,
        old_centroids,
        mode="euclid",
    )
    return counts, centroids
