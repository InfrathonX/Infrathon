#!/usr/bin/env python3
"""Build a host-only MLSys Forge task adapter for fast_pytorch_kmeans.

The upstream checkout remains intact under host_reference/.  The generated
candidate preserves the upstream Euclidean similarity, adaptive assignment
chunking, and dense membership-matrix centroid update.  The wrapper adds only
the task's batched ABI, exact integer counts, and retain-old empty-cluster
semantics.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import re
import io
import json
from pathlib import Path
import tarfile


ADAPTER = r'''"""Host-only fast_pytorch_kmeans task-ABI adapter.

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
'''


def add_bytes(
    archive: tarfile.TarFile,
    name: str,
    data: bytes,
    mode: int = 0o644,
) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mode = mode
    info.mtime = 0
    archive.addfile(info, io.BytesIO(data))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-output", type=Path, required=True)
    args = parser.parse_args()

    upstream_root = args.upstream_root.resolve()
    upstream_files = [
        upstream_root / "fast_pytorch_kmeans" / "kmeans.py",
        upstream_root / "fast_pytorch_kmeans" / "util.py",
    ]
    upstream_hashes = {
        path.relative_to(upstream_root).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in upstream_files
    }
    # The distribution ships the upstream snapshot without its .git metadata;
    # the pinned commit travels as a marker file (as for the official oracle).
    marker = upstream_root / "UPSTREAM_COMMIT"
    if marker.is_file() and not marker.is_symlink():
        commit = marker.read_text(encoding="utf-8").strip()
    else:
        commit = (
            upstream_root / ".git" / "refs" / "heads" / "master"
        ).read_text(encoding="utf-8").strip()
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise SystemExit(f"invalid fast_pytorch_kmeans upstream commit marker: {commit!r}")

    source_bytes = ADAPTER.encode("utf-8")
    adapter_sha = hashlib.sha256(source_bytes).hexdigest()
    provenance = {
        "kind": "fast_pytorch_kmeans_task_abi_adapter",
        "upstream_url": "https://github.com/DeMoriarty/fast_pytorch_kmeans",
        "upstream_commit": commit,
        "upstream_sha256": upstream_hashes,
        "adapter_sha256": adapter_sha,
        "semantic_bridges": [
            "sequential loop over task batch dimension",
            "return exact int32 cluster counts",
            "retain old centroid for empty clusters",
        ],
        "preserved_algorithm_properties": [
            "upstream Euclidean similarity expression",
            "upstream adaptive assignment split memory estimate",
            "upstream dense K-by-N membership-mask centroid update",
        ],
    }

    args.source_output.parent.mkdir(parents=True, exist_ok=True)
    args.source_output.write_bytes(source_bytes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as raw_output:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=raw_output,
            mtime=0,
        ) as compressed:
            with tarfile.open(
                fileobj=compressed,
                mode="w",
                format=tarfile.PAX_FORMAT,
            ) as archive:
                add_bytes(
                    archive,
                    "kernels/flash_kmeans_v001.py",
                    source_bytes,
                )
                add_bytes(archive, "FINAL_VERSION", b"1\n")
                add_bytes(
                    archive,
                    "notes.md",
                    (
                        "# Host-only fast_pytorch_kmeans baseline\n\n"
                        "```json\n"
                        + json.dumps(provenance, indent=2, sort_keys=True)
                        + "\n```\n"
                    ).encode("utf-8"),
                )

    print(args.output)
    print(f"upstream_commit={commit}")
    print(f"adapter_sha256={adapter_sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
