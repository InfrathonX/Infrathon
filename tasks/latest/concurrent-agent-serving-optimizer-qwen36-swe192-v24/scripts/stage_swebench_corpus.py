#!/usr/bin/env python3
"""Stage the frozen 192-instance SWE-bench Lite corpus from public sources.

The corpus is a P1 *generated* input.  Its two definitions are in-tree:

* ``corpus/swebench-lite-300-source.json`` — the 300 SWE-bench Lite test rows
  (``princeton-nlp/SWE-Bench_Lite`` @ ``6ec7bb89``) with the digest of the
  official ``swebench/sweb.eval.x86_64.<instance>`` image for each one;
* ``corpus/swebench-heterogeneous-192-selection.json`` — the 192 instances the
  task uses, chosen as the lowest ``sha256("<salt>:<instance_id>")`` and locked
  by digest.

This script re-derives the selection from the source, pulls the 192 images
digest-locked from Docker Hub (about 230 GB on disk; a registry mirror may be given),
and writes ``manifest.json`` into the corpus root (``KVBENCH_CORPUS_ROOT``,
default ``~/.cache/harbor-kv-scheduler-v2/swebench-lite-heterogeneous-192-gpu45``).
The manifest is a pure function of the two in-tree files, so its SHA-256 is
the same on every host: ``9dbc85ca7b58b3596e473cfa0dabe5b09f4b2213b023b747cfd21b34b7cb3362``.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import tempfile
import threading
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any


TASK_ROOT = Path(__file__).resolve().parents[1]
SELECTION = TASK_ROOT / "corpus" / "swebench-heterogeneous-192-selection.json"
DEFAULT_SOURCE = TASK_ROOT / "corpus" / "swebench-lite-300-source.json"
DEFAULT_OUTPUT = Path(
    os.environ.get("KVBENCH_CORPUS_ROOT")
    or Path.home() / ".cache/harbor-kv-scheduler-v2/swebench-lite-heterogeneous-192-gpu45"
).expanduser()
EXPECTED_MANIFEST_SHA256 = (
    "9dbc85ca7b58b3596e473cfa0dabe5b09f4b2213b023b747cfd21b34b7cb3362"
)
DATASET = "princeton-nlp/SWE-Bench_Lite"
REVISION = "6ec7bb89b9342f664a54a6e0a6ea6501d3437cc2"
PARQUET_SHA256 = "7a21f37b8bc179c7db5beeb14e88ac538ba283455c776e6b2535bbfb6e3551b4"
EXPECTED_COUNT = 192
PRINT_LOCK = threading.Lock()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def command(*argv: str, timeout: int = 3600) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )


def inspect_image(reference: str) -> dict[str, Any] | None:
    result = command("docker", "image", "inspect", reference, timeout=60)
    if result.returncode != 0:
        return None
    payload = json.loads(result.stdout)
    if not isinstance(payload, list) or len(payload) != 1:
        raise RuntimeError(f"Docker returned malformed image metadata for {reference}")
    return payload[0]


def image_matches(entry: dict[str, Any]) -> bool:
    image = inspect_image(str(entry["image_name"]))
    if image is None:
        return False
    # The classic Docker image store reports the config digest as the image ID, the containerd image store
    # the manifest digest; the sandbox worker accepts either form of the recorded identity.
    if str(image.get("Id") or "").lower() not in {
        str(entry["image_digest"]).lower(),
        str(entry["official_manifest_digest"]).lower(),
    }:
        return False
    suffix = f"@{str(entry['official_manifest_digest']).lower()}"
    return any(
        str(value).lower().endswith(suffix)
        for value in image.get("RepoDigests") or []
    )


def pull_one(
    ordinal: int,
    total: int,
    entry: dict[str, Any],
    *,
    mirror: str | None,
    no_pull: bool,
) -> tuple[str, str]:
    instance_id = str(entry["instance_id"])
    reference = str(entry["image_name"])
    if image_matches(entry):
        status = "REUSED"
    elif no_pull:
        raise RuntimeError(f"required image is not local: {instance_id} ({reference})")
    else:
        repository = reference.removeprefix("docker.io/").rsplit(":", 1)[0]
        digest = str(entry["official_manifest_digest"])
        candidates = []
        if mirror:
            candidates.append(f"{mirror.rstrip('/')}/{repository}@{digest}")
        candidates.append(f"docker.io/{repository}@{digest}")
        outputs: list[str] = []
        for locked_reference in candidates:
            pulled = command("docker", "pull", locked_reference)
            outputs.append(f"{locked_reference}: rc={pulled.returncode}\n{pulled.stdout[-2000:]}")
            if pulled.returncode != 0:
                continue
            tagged = command("docker", "tag", locked_reference, reference, timeout=60)
            if tagged.returncode != 0:
                outputs.append(f"tag rc={tagged.returncode}\n{tagged.stdout[-2000:]}")
                continue
            if image_matches(entry):
                status = "PULLED"
                break
        else:
            raise RuntimeError(
                f"digest-locked pull failed for {instance_id}:\n" + "\n".join(outputs)
            )
    with PRINT_LOCK:
        print(f"[{ordinal:03d}/{total:03d}] {status} {instance_id}", flush=True)
    return instance_id, status


def load_contract(source_path: Path) -> tuple[dict[str, Any], dict[str, Any], bytes, bytes]:
    source_bytes = source_path.read_bytes()
    selection_bytes = SELECTION.read_bytes()
    source = json.loads(source_bytes)
    selection = json.loads(selection_bytes)
    if (
        source.get("dataset") != DATASET
        or source.get("dataset_revision") != REVISION
        or source.get("dataset_parquet_sha256") != PARQUET_SHA256
        or not isinstance(source.get("instances"), list)
        or len(source["instances"]) != 300
    ):
        raise RuntimeError("source manifest is not the pinned 300-row SWE-bench Lite corpus")
    if (
        selection.get("schema_version") != 1
        or selection.get("dataset") != DATASET
        or selection.get("dataset_revision") != REVISION
        or selection.get("source_manifest_sha256") != sha256_bytes(source_bytes)
        or selection.get("selected_instance_count") != EXPECTED_COUNT
        or not isinstance(selection.get("instance_ids"), list)
        or len(selection["instance_ids"]) != EXPECTED_COUNT
        or len(set(selection["instance_ids"])) != EXPECTED_COUNT
    ):
        raise RuntimeError("192-instance selection lock does not match the pinned source manifest")
    expected_ids = [
        str(entry["instance_id"])
        for entry in sorted(
            source["instances"],
            key=lambda entry: hashlib.sha256(
                (
                    f"{selection['selection_salt']}:"
                    f"{entry['instance_id']}"
                ).encode()
            ).hexdigest(),
        )[:EXPECTED_COUNT]
    ]
    if (
        selection.get("selection_policy") != "lowest-sha256(salt:instance_id)"
        or selection.get("selection_salt") != "kvbench-qwen36-swe192-heterogeneous-v1"
        or selection["instance_ids"] != expected_ids
    ):
        raise RuntimeError(
            "192-instance selection is not the deterministic lowest-SHA256 set"
        )
    return source, selection, source_bytes, selection_bytes


def write_manifest(
    output_root: Path,
    source: dict[str, Any],
    selection: dict[str, Any],
    source_bytes: bytes,
    selection_bytes: bytes,
) -> Path:
    by_id = {str(entry["instance_id"]): entry for entry in source["instances"]}
    missing = sorted(set(selection["instance_ids"]) - set(by_id))
    if missing:
        raise RuntimeError(f"selection contains unknown SWE-bench IDs: {missing}")
    entries = []
    for instance_id in selection["instance_ids"]:
        entry = deepcopy(by_id[instance_id])
        entry["partition"] = "public"
        entries.append(entry)
    manifest = {
        "schema_version": 1,
        "corpus_mode": "heterogeneous-192-distinct",
        "dataset": DATASET,
        "dataset_revision": REVISION,
        "dataset_split": "test",
        "dataset_parquet_path": source.get("dataset_parquet_path"),
        "dataset_parquet_sha256": PARQUET_SHA256,
        "source_instance_count": 300,
        "selected_instance_count": EXPECTED_COUNT,
        "source_manifest_sha256": sha256_bytes(source_bytes),
        "selection_lock_sha256": sha256_bytes(selection_bytes),
        "selection_policy": selection["selection_policy"],
        "selection_salt": selection["selection_salt"],
        "partition_policy": {
            "algorithm": "single-shared-exploration-partition",
            "public_instances": EXPECTED_COUNT,
            "hidden_instances": 0,
            "rollout_reuse": False,
            "fresh_sibling_container_per_rollout": True,
        },
        "repository_counts": dict(
            sorted(Counter(str(entry["repo"]) for entry in entries).items())
        ),
        "instances": entries,
    }
    encoded = (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=".manifest.", suffix=".json", dir=output_root, delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.chmod(0o444)
    temporary.replace(output_root / "manifest.json")
    return output_root / "manifest.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--registry-mirror", default=None,
                        help="optional registry mirror tried before docker.io")
    parser.add_argument("--no-pull", action="store_true",
                        help="only verify that every image is already local")
    parser.add_argument("--manifest-only", action="store_true",
                        help="write manifest.json without touching Docker (for verification)")
    args = parser.parse_args()
    if not 1 <= args.workers <= 12:
        raise ValueError("--workers must be in 1..12")
    source_path = args.source_manifest.expanduser().resolve(strict=True)
    output_root = args.output_root.expanduser().resolve()
    source, selection, source_bytes, selection_bytes = load_contract(source_path)
    by_id = {str(entry["instance_id"]): entry for entry in source["instances"]}
    entries = [by_id[instance_id] for instance_id in selection["instance_ids"]]

    results: list[tuple[str, str]] = []
    failures: list[str] = []
    if args.manifest_only:
        results = [(str(entry["instance_id"]), "SKIPPED") for entry in entries]
        entries = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                pull_one,
                ordinal,
                len(entries),
                entry,
                mirror=args.registry_mirror or None,
                no_pull=args.no_pull,
            ): str(entry["instance_id"])
            for ordinal, entry in enumerate(entries, start=1)
        }
        for future in concurrent.futures.as_completed(futures):
            try:
                results.append(future.result())
            except Exception as exc:
                failures.append(f"{futures[future]}: {type(exc).__name__}: {exc}")
    if failures:
        raise RuntimeError(
            f"{len(failures)} of {len(entries)} images failed staging:\n"
            + "\n".join(failures)
        )
    manifest_path = write_manifest(
        output_root, source, selection, source_bytes, selection_bytes
    )
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if manifest_sha256 != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError(
            f"staged manifest sha256 {manifest_sha256} != {EXPECTED_MANIFEST_SHA256}"
        )
    print(
        json.dumps(
            {
                "status": "ready",
                "manifest": str(manifest_path),
                "manifest_sha256": manifest_sha256,
                "instances": len(results),
                "distinct_instances": len({item[0] for item in results}),
                "pulled": sum(item[1] == "PULLED" for item in results),
                "reused": sum(item[1] == "REUSED" for item in results),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
