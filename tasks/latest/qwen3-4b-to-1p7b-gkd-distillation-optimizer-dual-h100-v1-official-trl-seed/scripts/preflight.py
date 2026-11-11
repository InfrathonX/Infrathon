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
MIN_DEVICE_MEMORY_BYTES = 79 * 1024**3
STUDENT_MODEL_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
TEACHER_MODEL_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
PUBLIC_MANIFEST_SHA256 = "e653d0a59a07d7341197263a28424f85940ef633cce1e3a93891095cf2e04956"
PUBLIC_TENSORS_SHA256 = "9a92460dad2938181bfc51f9cebefc81b6f9b59ac21c2144f4cba66739a953b6"
PRIVATE_CROSS_ARTIFACTS = {
    "private_gkd_512": (
        "5fdb86a9ed8d1ce90db3b5354f2fdfd9a11dad0078d679555512ab8ea01cb715",
        "2acb96c7a24151b1e38d000654568910b68eb1e6071697bf14f034e7c5461b26",
    ),
    "private_gkd_1024": (
        "896d4a1424483d090c9dc5679ce5ad562c8058201cf7e783454d1cf16a887d8f",
        "8f2003d6ab8129687207b5cdf86242576abfb42470eb661032bad843c5d31d67",
    ),
    "private_gkd_2048": (
        "a119c7062e3280d58eea03c44f553cbf917465b2ddae303e5ab1eb05bcd8ee5c",
        "abf918fbc2e6d3263a0bfed70decc863f1e17f7671f937b79cf37d9a9afdc56a",
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


def require_deprivileged_reachable(path: Path, label: str) -> None:
    """Fail closed unless a de-privileged process could read this path.

    Candidate code runs inside the Judge as an unprivileged user while the
    reference runs as root, so a mode that only root can traverse produces a
    one-sided failure deep inside a formal run instead of here. Directories
    need the traversal bit and files need the read bit for "other"; a data
    root may stay unlistable (0711), which still lets an exact path resolve.
    """

    needed = 0o001 if path.is_dir() else 0o004
    mode = path.stat().st_mode
    if not mode & needed:
        raise SystemExit(
            f"{label} is not reachable by a de-privileged process: "
            f"{path} has mode {mode & 0o777:04o}"
        )


def validate_partition(path: Path, manifest_hash: str, tensors_hash: str) -> dict[str, str]:
    manifest = path / "manifest.json"
    tensors = path / "tensors.safetensors"
    if not path.is_dir() or not manifest.is_file() or not tensors.is_file():
        raise SystemExit(f"frozen dataset partition is incomplete: {path}")
    require_deprivileged_reachable(path, "frozen dataset partition")
    require_deprivileged_reachable(manifest, "frozen dataset manifest")
    require_deprivileged_reachable(tensors, "frozen dataset tensors")
    observed_manifest = sha256(manifest)
    observed_tensors = sha256(tensors)
    if observed_manifest != manifest_hash or observed_tensors != tensors_hash:
        raise SystemExit(f"frozen dataset digest mismatch: {path}")
    return {
        "path": str(path.resolve()),
        "manifest_sha256": observed_manifest,
        "tensors_sha256": observed_tensors,
        "mode": f"{path.stat().st_mode & 0o777:04o}",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu-indices",
        default=os.environ.get("GKD_GPU_INDICES", ""),
        help=(
            "two distinct physical GPU indices; defaults to GKD_GPU_INDICES. "
            "This task pins no pair of its own"
        ),
    )
    parser.add_argument(
        "--expected-gpu-uuids",
        default=os.environ.get("GKD_GPU_UUIDS", ""),
        help=(
            "optional UUID pin for the selected indices; defaults to "
            "GKD_GPU_UUIDS, and is resolved from the indices when unset"
        ),
    )
    parser.add_argument("--max-used-mib", type=int, default=1024)
    parser.add_argument("--model-path", "--student-model-path", dest="model_path", type=Path, required=True)
    parser.add_argument("--teacher-model-path", dest="teacher_model_path", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--private-data-root", type=Path)
    parser.add_argument(
        "--allow-non-nvlink",
        action="store_true",
        help="diagnostic escape hatch; a formal run must not use it",
    )
    args = parser.parse_args()
    if not args.gpu_indices.strip():
        raise SystemExit(
            "name the GPU pair: pass --gpu-indices or set GKD_GPU_INDICES"
        )
    indices = [int(value) for value in args.gpu_indices.split(",")]
    if len(indices) != 2 or len(set(indices)) != 2:
        raise SystemExit("exactly two distinct physical GPU indices are required")
    expected_uuids: tuple[str, ...] | None = None
    if args.expected_gpu_uuids.strip():
        expected_uuids = tuple(
            value.strip() for value in args.expected_gpu_uuids.split(",")
        )
        if len(expected_uuids) != 2 or any(
            not value.startswith("GPU-") for value in expected_uuids
        ):
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
    missing = [index for index in indices if index not in rows]
    if missing:
        raise SystemExit(f"this host does not expose GPU indices: {missing}")
    selected = [rows[index] for index in indices]
    observed_uuids = tuple(gpu["uuid"] for gpu in selected)
    if expected_uuids is not None and observed_uuids != expected_uuids:
        raise SystemExit("selected physical GPU UUIDs differ from the pinned pair")
    expected_uuids = observed_uuids
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
        raise SystemExit("student config differs from the frozen Qwen3-1.7B contract")
    teacher_config = json.loads(
        (args.teacher_model_path / "config.json").read_text(encoding="utf-8")
    )
    expected_teacher = {
        "model_type": "qwen3",
        "hidden_size": 2560,
        "num_hidden_layers": 36,
        "vocab_size": 151936,
        "tie_word_embeddings": True,
    }
    if any(teacher_config.get(key) != value for key, value in expected_teacher.items()):
        raise SystemExit("teacher config differs from the frozen Qwen3-4B contract")
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
        require_deprivileged_reachable(data_root, "frozen dataset root")
        data = {
            "public": validate_partition(
                data_root / "public", PUBLIC_MANIFEST_SHA256, PUBLIC_TENSORS_SHA256
            ),
        }
    if args.private_data_root is not None:
        private_root = args.private_data_root.resolve(strict=True)
        require_deprivileged_reachable(private_root, "frozen private dataset root")
        private = {
            name: validate_partition(private_root / name, *digests)
            for name, digests in PRIVATE_CROSS_ARTIFACTS.items()
        }
        data = dict(data or {}) | {"private_cross_matrix": private}
    print(
        json.dumps(
            {
                "gpus": selected,
                "student_model_path": str(args.model_path.resolve()),
                "teacher_model_path": str(args.teacher_model_path.resolve()),
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
