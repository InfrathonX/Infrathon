#!/usr/bin/env python3
"""Build a host-only task-ABI archive from upstream torch_fallback.py."""

from __future__ import annotations

import argparse
import ast
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile


WRAPPERS = r'''

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
'''


def extract_function(source: str, name: str) -> str:
    """Mechanically extract one complete top-level upstream function."""

    tree = ast.parse(source)
    matches = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one upstream function {name!r}, got "
            f"{len(matches)}"
        )
    segment = ast.get_source_segment(source, matches[0])
    if not segment:
        raise ValueError(f"could not extract upstream function {name!r}")
    return segment.rstrip()


def build_adapter_source(upstream: str) -> str:
    """Keep only the exact upstream algorithms needed by the task ABI.

    The repository file also contains a full K-means driver and an executable
    self-test. Those are unrelated to the two-function task and intentionally
    violate the candidate clean-room policy (`print`, random seeding, and a
    `__main__` entry point). The host copy remains byte-exact; this archive
    mechanically selects the two relevant function definitions without
    changing their bodies.
    """

    selected = (
        extract_function(upstream, "euclid_assign_torch_native_chunked"),
        extract_function(upstream, "_centroid_update_torch_native"),
    )
    return (
        '"""Task adapter mechanically extracted from upstream '
        'flash_kmeans/torch_fallback.py."""\n\n'
        "import torch\n"
        "import torch.nn.functional as F\n\n"
        + "\n\n".join(selected)
        + "\n"
        + WRAPPERS
    )


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
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-output", type=Path, required=True)
    args = parser.parse_args()

    upstream = args.source.resolve().read_text(encoding="utf-8")
    source = build_adapter_source(upstream)
    source_bytes = source.encode("utf-8")
    upstream_sha = hashlib.sha256(upstream.encode("utf-8")).hexdigest()
    adapter_sha = hashlib.sha256(source_bytes).hexdigest()

    args.source_output.parent.mkdir(parents=True, exist_ok=True)
    args.source_output.write_bytes(source_bytes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Keep both tar metadata and the outer gzip header deterministic.  Without
    # an explicit gzip mtime, rebuilding identical source changes the archive
    # digest even though every extracted member is byte-identical.
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
                provenance = {
                    "kind": "upstream_torch_fallback_task_abi_adapter",
                    "upstream_path": "flash_kmeans/torch_fallback.py",
                    "upstream_sha256": upstream_sha,
                    "adapter_sha256": adapter_sha,
                    "mechanically_extracted_functions": [
                        "euclid_assign_torch_native_chunked",
                        "_centroid_update_torch_native",
                    ],
                }
                add_bytes(
                    archive,
                    "notes.md",
                    (
                        "# Host-only upstream torch fallback baseline\n\n"
                        "```json\n"
                        + json.dumps(provenance, indent=2, sort_keys=True)
                        + "\n```\n"
                    ).encode("utf-8"),
                )
    print(args.output)
    print(f"upstream_sha256={upstream_sha}")
    print(f"adapter_sha256={adapter_sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
