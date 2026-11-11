#!/usr/bin/env python3
"""Build a deterministic v7-scratch candidate archive for the standard baseline."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile


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
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(__file__).with_name("standard_kmeans.py"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-output", type=Path)
    args = parser.parse_args()

    source = args.source.resolve().read_bytes()
    source_sha = hashlib.sha256(source).hexdigest()
    if args.source_output is not None:
        args.source_output.parent.mkdir(parents=True, exist_ok=True)
        args.source_output.write_bytes(source)

    provenance = {
        "kind": "paper_algorithm_1_standard_pytorch_kmeans",
        "source_sha256": source_sha,
        "algorithm_properties": [
            "complete FP32 B-by-N-by-K distance materialization",
            "separate argmin reduction",
            "FP32 index_add aggregation",
            "integer index_add counts",
            "separate normalization",
        ],
        "task_abi_bridge": [
            "batched tensors",
            "return exact integer counts",
            "retain old centroid for an empty cluster",
        ],
    }

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
                    source,
                )
                add_bytes(archive, "FINAL_VERSION", b"1\n")
                add_bytes(
                    archive,
                    "notes.md",
                    (
                        "# Paper Algorithm 1: standard PyTorch baseline\n\n"
                        "```json\n"
                        + json.dumps(provenance, indent=2, sort_keys=True)
                        + "\n```\n"
                    ).encode("utf-8"),
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
