#!/usr/bin/env python3
"""Fail-closed hardware/model preflight; this script never stops services."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import subprocess
from pathlib import Path


# Capability envelope; identical to the trusted runtime and the Judge controller.
MIN_COMPUTE_CAPABILITY = (8, 0)
MIN_DEVICE_MEMORY_BYTES = 60 * 1024**3
MODEL_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
# No host GPU is baked in here: the pair is named by ORPO_GPU_UUIDS (or
# --expected-gpu-uuids) on whatever host the run happens on.
EXPECTED_GPU_UUIDS_ENV = "ORPO_GPU_UUIDS"
PUBLIC_MANIFEST_SHA256 = "ba9816131647aefe0d813fc7c7d3706894988943bc58e92c472667f722528c20"
PUBLIC_TENSORS_SHA256 = "e1218b35260ea9e7afd048962b53c25257d7f298e2a787628d8f5b640cfc7e79"
PRIVATE_CROSS_ARTIFACTS = {
    "private_cross_512": (
        "554bf1261f72a848de3ac93d21e5c24a6581a52e0e802adea2454fac6d5d1369",
        "2968c2b87f03b9fb2ee57b69d5c8a702ed994cdb16889bb4bbfe40e9d104315b",
    ),
    "private_cross_1024": (
        "b731a9642512720ac116b13ff88ccc8fa8a4751f94c7a7204d792e7e65fb6672",
        "1d347aa8f5047de37ca7f257439585a7caaff44179bc44c521190328f8883368",
    ),
    "private_cross_2048": (
        "fea2455400c3461f9c5797d4f8288728b1cdfec16d2258a431215fe47b5f429d",
        "05cc2316e4db9af9bec8f13695a07760ab30a77b4062b99b6baf020c0277a0e8",
    ),
}


def command(*args: str) -> str:
    return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def topology_link(topology: str, source_index: int, target_index: int) -> str:
    row_prefix = f"GPU{source_index}"
    for line in topology.splitlines():
        fields = line.split()
        if fields and fields[0] == row_prefix:
            offset = 1 + target_index
            if len(fields) <= offset:
                break
            return fields[offset]
    raise SystemExit("cannot resolve selected GPU topology link")


def validate_partition(path: Path, manifest_hash: str, tensors_hash: str) -> dict[str, str]:
    manifest = path / "manifest.json"
    tensors = path / "tensors.safetensors"
    if not path.is_dir() or not manifest.is_file() or not tensors.is_file():
        raise SystemExit(f"frozen dataset partition is incomplete: {path}")
    observed_manifest = sha256(manifest)
    observed_tensors = sha256(tensors)
    if observed_manifest != manifest_hash or observed_tensors != tensors_hash:
        raise SystemExit(f"frozen dataset digest mismatch: {path}")
    return {
        "path": str(path.resolve()),
        "manifest_sha256": observed_manifest,
        "tensors_sha256": observed_tensors,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu-indices",
        required=True,
        help=(
            "the two physical GPU indices of the NVLink pair this run owns, "
            "comma separated; there is no default "
            "(nvidia-smi --query-gpu=index,uuid --format=csv,noheader)"
        ),
    )
    parser.add_argument(
        "--expected-gpu-uuids",
        default=os.environ.get(EXPECTED_GPU_UUIDS_ENV, ""),
    )
    parser.add_argument("--max-used-mib", type=int, default=1024)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--private-data-root", type=Path)
    parser.add_argument(
        "--allow-non-nvlink",
        action="store_true",
        help="diagnostic escape hatch; formal NVLink-pair runs must not use it",
    )
    args = parser.parse_args()
    indices = [int(value) for value in args.gpu_indices.split(",")]
    if len(indices) != 2 or len(set(indices)) != 2:
        raise SystemExit("exactly two distinct physical GPU indices are required")
    expected_uuids = tuple(value.strip() for value in args.expected_gpu_uuids.split(","))
    if len(expected_uuids) != 2 or any(not value.startswith("GPU-") for value in expected_uuids):
        raise SystemExit("exactly two explicit expected GPU UUIDs are required")
    raw = command(
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.total,memory.used,compute_cap",
        "--format=csv,noheader,nounits",
    )
    rows = {}
    for row in csv.reader(io.StringIO(raw), skipinitialspace=True):
        rows[int(row[0])] = {
            "index": int(row[0]),
            "uuid": row[1],
            "name": row[2],
            "memory_total_mib": int(row[3]),
            "memory_used_mib": int(row[4]),
            "compute_capability": row[5],
        }
    selected = [rows[index] for index in indices]
    if tuple(gpu["uuid"] for gpu in selected) != expected_uuids:
        raise SystemExit("selected physical GPU UUIDs differ from the frozen pair")
    for gpu in selected:
        # The GPU model is provenance, not a gate: the trusted runtime and the
        # Judge controller enforce the capability envelope (compute capability
        # >= 8.0, per-rank total memory floor, native BF16, NCCL, two ranks).
        parts = gpu["compute_capability"].split(".")
        if len(parts) != 2 or not all(part.isascii() and part.isdigit() for part in parts):
            raise SystemExit(
                f"GPU {gpu['index']} did not report a compute capability: "
                f"{gpu['compute_capability']!r}"
            )
        capability = (int(parts[0]), int(parts[1]))
        if capability < MIN_COMPUTE_CAPABILITY:
            raise SystemExit(
                f"GPU {gpu['index']} compute capability {gpu['compute_capability']} "
                "is below the task minimum"
            )
        if gpu["memory_total_mib"] * 1024**2 < MIN_DEVICE_MEMORY_BYTES:
            raise SystemExit(
                f"GPU {gpu['index']} has {gpu['memory_total_mib']} MiB, below the task floor"
            )
        if gpu["memory_used_mib"] > args.max_used_mib:
            raise SystemExit(
                f"GPU {gpu['index']} is busy ({gpu['memory_used_mib']} MiB); stop its owning service first"
            )
    applications = command(
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits",
    )
    selected_uuid_set = set(expected_uuids)
    owners = [
        row
        for row in csv.reader(io.StringIO(applications), skipinitialspace=True)
        if row and row[0] in selected_uuid_set
    ]
    if owners:
        raise SystemExit(f"selected GPU pair has active compute owners: {owners}")
    model_config = json.loads((args.model_path / "config.json").read_text(encoding="utf-8"))
    expected = {
        "model_type": "qwen3",
        "hidden_size": 2048,
        "num_hidden_layers": 28,
        "vocab_size": 151936,
        "tie_word_embeddings": True,
    }
    if any(model_config.get(key) != value for key, value in expected.items()):
        raise SystemExit("model config differs from the frozen Qwen3-1.7B contract")
    topo = command("nvidia-smi", "topo", "-m")
    forward_link = topology_link(topo, indices[0], indices[1])
    reverse_link = topology_link(topo, indices[1], indices[0])
    if not args.allow_non_nvlink and not (
        forward_link.startswith("NV") and reverse_link.startswith("NV")
    ):
        raise SystemExit(
            f"selected GPUs are not an NVLink pair: {forward_link}/{reverse_link}"
        )
    data = None
    if args.data_root is not None:
        data_root = args.data_root.resolve(strict=True)
        data = {
            "public": validate_partition(
                data_root / "public", PUBLIC_MANIFEST_SHA256, PUBLIC_TENSORS_SHA256
            ),
        }
    if args.private_data_root is not None:
        private_root = args.private_data_root.resolve(strict=True)
        private = {
            name: validate_partition(private_root / name, *digests)
            for name, digests in PRIVATE_CROSS_ARTIFACTS.items()
        }
        data = dict(data or {}) | {"private_cross_matrix": private}
    print(
        json.dumps(
            {
                "gpus": selected,
                "model_path": str(args.model_path.resolve()),
                "data": data,
                "topology_link": [forward_link, reverse_link],
                "topology": topo,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
