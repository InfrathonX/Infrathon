"""Host-only fast_pytorch_kmeans task-ABI adapter.

Derived from DeMoriarty/fast_pytorch_kmeans KMeans.euc_sim, KMeans.max_sim,
and the non-minibatch update in KMeans.fit_predict.  Batch elements are
processed sequentially, matching the Flash-KMeans repository benchmark
wrapper.  The task-only bridge returns exact counts and retains old centroids
for empty clusters.
"""

import math
import torch


def _will_it_fit(size, device):
    try:
        torch.empty(size, device=device, dtype=torch.uint8)
    except Exception:
        return False
    return True


def _find_optimal_splits(n, get_required_memory, device):
    splits = 1
    break_next = False
    while True:
        if break_next:
            break
        if splits > n:
            splits = n
            break_next = True
        sub_n = math.ceil(n / splits)
        required_memory = get_required_memory(sub_n)
        if _will_it_fit(required_memory, device):
            break
        splits *= 2
    return splits


def _euc_sim(a, b):
    return (
        2 * a @ b.transpose(-2, -1)
        - (a ** 2).sum(dim=1)[..., :, None]
        - (b ** 2).sum(dim=1)[..., None, :]
    )


def _max_sim(a, b):
    device = a.device
    n_samples = a.shape[0]
    max_sim_v = torch.empty(n_samples, device=device, dtype=a.dtype)
    max_sim_i = torch.empty(n_samples, device=device, dtype=torch.int64)

    def get_required_memory(chunk_size):
        return (
            chunk_size
            * a.shape[1]
            * b.shape[0]
            * a.element_size()
            + n_samples * 2 * 4
        )

    splits = _find_optimal_splits(
        n_samples,
        get_required_memory,
        device=device,
    )
    chunk_size = math.ceil(n_samples / splits)
    for split_id in range(splits):
        start = split_id * chunk_size
        if start >= n_samples:
            continue
        end = min((split_id + 1) * chunk_size, n_samples)
        sub_sim = _euc_sim(a[start:end], b)
        sub_max_sim_v, sub_max_sim_i = sub_sim.max(dim=-1)
        max_sim_v[start:end] = sub_max_sim_v
        max_sim_i[start:end] = sub_max_sim_i
    return max_sim_v, max_sim_i


@torch.no_grad()
def assign(
    x: torch.Tensor,
    centroids: torch.Tensor,
    x_sq: torch.Tensor,
) -> torch.Tensor:
    batch = x.shape[0]
    labels = torch.empty(
        x.shape[:2],
        device=x.device,
        dtype=torch.int64,
    )
    for batch_id in range(batch):
        labels[batch_id] = _max_sim(
            x[batch_id],
            centroids[batch_id],
        )[1]
    return labels


@torch.no_grad()
def update(
    x: torch.Tensor,
    labels: torch.Tensor,
    old_centroids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, _, _ = x.shape
    clusters = old_centroids.shape[1]
    counts = torch.empty(
        (batch, clusters),
        device=x.device,
        dtype=torch.int32,
    )
    new_centroids = torch.empty_like(old_centroids)
    arranged_mask = torch.arange(
        clusters,
        device=x.device,
    )[:, None]

    for batch_id in range(batch):
        batch_labels = labels[batch_id].to(torch.int64)
        counts[batch_id] = torch.bincount(
            batch_labels,
            minlength=clusters,
        ).to(torch.int32)

        expanded_closest = batch_labels[None].expand(clusters, -1)
        mask = (expanded_closest == arranged_mask).to(x.dtype)
        c_grad = (
            mask @ x[batch_id]
            / mask.sum(-1)[..., :, None]
        )
        torch.nan_to_num_(c_grad)
        new_centroids[batch_id] = torch.where(
            (counts[batch_id] == 0).unsqueeze(-1),
            old_centroids[batch_id],
            c_grad,
        )

    return counts, new_centroids
