#!/usr/bin/env python3
"""Refresh the portable manifest digests and the task JSON that binds them.

The portable contract freezes the TaskCore byte-for-byte: ``portable/*.json``
record the SHA-256 of every scientific and delivery resource, and the task
JSON records the SHA-256 of each manifest document.  Any legitimate edit to a
hashed file (a prompt sentence, a launcher line covered by the build manifest,
a manifest field) therefore has to be followed by a digest refresh.  Doing that
by hand is friction with no security value, so this tool recomputes every
digest from the actual bytes and rewrites only the digest strings in place,
leaving the hand-formatted documents otherwise untouched.

    scripts/refresh_portable_manifests.py            # rewrite in place
    scripts/refresh_portable_manifests.py --check    # report drift, change nothing

It never adds or removes a resource entry: the resource path sets are part of
the contract and are reviewed (and re-derived by tests/test_portable_contract.py),
not generated.  The task JSON is the digest-binding root and is deliberately
not a member of any manifest; each manifest also excludes itself.

The release-derived parts of the hidden suite manifests and the oracle
protocol are the one exception to "digests only": the 17-case table, the
per-model gates and budgets, the controller bound, the release digest and
status are copied from ``configs/h100_scratch_v4_release.json`` (and the
standalone final timeout from the task JSON's ``eval_timeout``), so a
calibration that fills in goldens, byte counts and gates needs one refresh,
not 17 hand edits.  Their prose fields stay hand-written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

TASK_ROOT = Path(__file__).resolve().parents[1]
PORTABLE = TASK_ROOT / "portable"
RELEASE_RELATIVE = "configs/h100_scratch_v4_release.json"
RESOURCE_LINE = re.compile(
    r'\{"path": "(?P<path>[^"]+)", "sha256": "sha256:(?P<sha>[0-9a-f]{64})"\}'
)
# Manifest document -> where the task JSON records its digest.
TASK_CORE_MANIFEST = "resource-manifest.json"
DELIVERY_MANIFEST = "build-resource-manifest.json"
SUITE_ROLE_MANIFESTS = {
    "public": "public-suite-manifest.json",
    "interactive": "interactive-suite-manifest.json",
    "final": "final-suite-manifest.json",
}
SUITE_PROTOCOLS = {
    "oracle": "oracle-protocol.json",
    "selection": "selection-protocol.json",
}
PROFILE_PROTOCOLS = {
    "software_abi": "software-abi.json",
    "calibration_protocol": "calibration-protocol.json",
}
HIDDEN_SUITE_MANIFESTS = ("interactive-suite-manifest.json", "final-suite-manifest.json")
ORACLE_PROTOCOL = "oracle-protocol.json"
MEASUREMENT_MODE = "summed_persistent_kv_bytes_with_mean_and_case_floor_gates"


def release_path(task_root: Path) -> Path:
    """The Judge-only release config: the in-repository copy when present, else the
    asset-root copy ``<INFRATHON_ASSETS_ROOT>/<task directory>/judge/h100_scratch_v4_release.json``
    written by ``assets/fetch.sh`` (default root ``~/.cache/infrathon/assets``)."""
    local = task_root / "configs/h100_scratch_v4_release.json"
    if local.is_file():
        return local
    assets = os.environ.get("INFRATHON_ASSETS_ROOT") or str(
        Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "infrathon" / "assets")
    return Path(assets).expanduser() / task_root.name / "judge/h100_scratch_v4_release.json"


RELEASE = release_path(TASK_ROOT)


def resource_path(relative: str) -> Path:
    """A hashed resource; the Judge-only release config may live in the asset root."""
    return RELEASE if relative == RELEASE_RELATIVE else TASK_ROOT / relative


def sha256_file(path: Path) -> str:
    if not path.is_file() or path.is_symlink():
        raise SystemExit(f"hashed resource is missing or not a regular file: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def task_json_path() -> Path:
    candidates = sorted(
        path
        for path in (TASK_ROOT / "tasks").glob("*.json")
        if not path.name.startswith("BENCHMARK")
    )
    if len(candidates) != 1:
        raise SystemExit(f"expected exactly one task JSON under tasks/, found {candidates}")
    return candidates[0]


def controller_bound_seconds(release: dict, gpu_count: int) -> int:
    """Wall bound of the sharded controller on ``gpu_count`` devices.

    Every case at its full budget (stage + max(fidelity, VBench) + case
    overhead), placed heaviest first on the device that becomes free earliest
    (device i starts at i x start_stagger_seconds), plus the controller
    overhead -- judge/judge_shards.plan_shards and plan_makespan.
    """

    judge = release["judge"]
    free = [index * judge["start_stagger_seconds"] for index in range(gpu_count)]
    units = []
    for case in release["cases"]:
        model = release["models"][case["model"]]
        units.append(model["stage_seconds"] + max(model["fidelity_metric_seconds"], model["vbench_metric_seconds"])
                     + judge["case_overhead_seconds"])
    for unit in sorted(units, reverse=True):
        target = min(range(gpu_count), key=lambda index: (free[index], index))
        free[target] += unit
    return max(free) + judge["controller_overhead_seconds"]


def _gate_text(gates: dict) -> dict:
    """Gate values as exact decimal strings, as the earlier manifests carried them."""

    return {
        key: ({dim: json.dumps(v) for dim, v in value.items()} if isinstance(value, dict) else json.dumps(value))
        for key, value in gates.items()
    }


def hidden_cases(release: dict) -> list[dict]:
    per_workload = {name: sum(case["model"] == name for case in release["cases"]) for name in release["models"]}
    cases = []
    for case in release["cases"]:
        entry = {"case_id": case["case_id"], "workload": case["model"], "role": "performance", "seed": case["seed"]}
        if case["model"] == "longcat":
            entry |= {"prompt_index": case["prompt_index"], "prompt_source": release["prompt_file"],
                      "segments": case["segments"], "init_video_sha256": case["init_video_sha256"]}
        else:
            entry |= {"image_source": case["image_source"], "image_sha256": case["image_sha256"],
                      "video_length": case["video_length"], "pose": case["pose"]}
        entry |= {
            "frame_count": case["frame_count"],
            "fidelity_frames": case["fidelity_frames"],
            "goldens": {"bf16_frames_sha256": case["bf16_frames_sha256"], "bf16_kv_bytes": case["bf16_kv_bytes"],
                        "bf16_vbench": case["bf16_vbench"]},
            "measurement_mode": MEASUREMENT_MODE,
            "weight": {"numerator": 1, "denominator": per_workload[case["model"]]},
        }
        cases.append(entry)
    return cases


def suite_timeouts(release: dict) -> dict:
    judge = release["judge"]
    return {
        "workloads": {
            name: {key: model[key] for key in ("stage_seconds", "fidelity_metric_seconds", "vbench_metric_seconds")}
            for name, model in release["models"].items()
        },
        "case_overhead_seconds": judge["case_overhead_seconds"],
        "start_stagger_seconds": judge["start_stagger_seconds"],
        "controller_overhead_seconds": judge["controller_overhead_seconds"],
        "max_gpu_count": judge["max_gpu_count"],
        "min_gpu_count": judge["min_gpu_count"],
        "controller_seconds": controller_bound_seconds(release, judge["max_gpu_count"]),
    }


def release_derived(name: str, release: dict, release_sha256: str, eval_timeout: int) -> dict:
    """The fields of one portable document that are copied from the release."""

    if name == ORACLE_PROTOCOL:
        judge = release["judge"]
        return {"policy": {
            "release_status": release["status"],
            "paper_protocol_sha256": release["paper_protocol_sha256"],
            "metric_implementation_sha256": release["metric_implementation_sha256"],
            "prompt_file_sha256": release["prompt_file_sha256"],
            "gates": {name: {kind: _gate_text(value) for kind, value in model["gates"].items()}
                      for name, model in release["models"].items()},
            "execution_budget": suite_timeouts(release),
            "judge": {key: judge[key] for key in ("min_device_memory_bytes", "slot_uids", "threads_per_slot",
                                                  "master_port_base")},
        }}
    cases = hidden_cases(release)
    derived = {
        "case_counts": {"scored_performance": len(cases), "total": len(cases), "zero_weight_hard_gate": 0},
        "cases": cases,
        "counts_per_workload": {name: sum(case["workload"] == name for case in cases) for name in release["models"]},
        "release_sha256": release_sha256,
        "release_status": release["status"],
        "suite_gate_timeouts": suite_timeouts(release),
        "workload_gates": {
            name: {"mean": _gate_text(model["gates"]["mean"]), "case_floor": _gate_text(model["gates"]["case_floor"]),
                   "wall_seconds_max": model["stage_seconds"]}
            for name, model in release["models"].items()
        },
    }
    if name == "final-suite-manifest.json":
        derived["orchestration_timeouts"] = {"standalone_final_evaluation_seconds": eval_timeout}
    return derived


def _render_value(value, depth: int) -> str:
    if isinstance(value, dict) and value:
        pad = "  " * (depth + 1)
        items = [f"{pad}{json.dumps(key)}: {_render_value(item, depth + 1)}" for key, item in sorted(value.items())]
        return "{\n" + ",\n".join(items) + "\n" + "  " * depth + "}"
    return json.dumps(value, ensure_ascii=False)


def render_document(document: dict) -> str:
    """Sorted keys, nested objects expanded, lists inline, one line per case."""

    items = []
    for key, value in sorted(document.items()):
        if key == "cases":
            body = ",\n".join("    " + json.dumps(case, ensure_ascii=False) for case in value)
            items.append(f'  "cases": [\n{body}\n  ]')
        else:
            items.append(f"  {json.dumps(key)}: {_render_value(value, 1)}")
    return "{\n" + ",\n".join(items) + "\n}\n"


def sync_release_fields(drift: list[str], eval_timeout: int) -> dict[Path, str]:
    """Rewrite the release-derived fields; return the documents that changed."""

    release_bytes = RELEASE.read_bytes()
    release = json.loads(release_bytes)
    release_sha256 = hashlib.sha256(release_bytes).hexdigest()
    writes: dict[Path, str] = {}
    for name in (*HIDDEN_SUITE_MANIFESTS, ORACLE_PROTOCOL):
        path = PORTABLE / name
        text = path.read_text(encoding="utf-8")
        document = json.loads(text)
        for key, value in release_derived(name, release, release_sha256, eval_timeout).items():
            if document.get(key) != value:
                drift.append(f"{name}: {key} differs from {RELEASE.name}")
            document[key] = value
        rendered = render_document(document)
        if rendered != text:
            if not any(entry.startswith(f"{name}: ") for entry in drift):
                drift.append(f"{name}: not in canonical form")
            writes[path] = rendered
    return writes


def refresh_manifest(path: Path, drift: list[str]) -> str:
    """Rewrite every resource digest in one manifest; return the new text."""

    text = path.read_text(encoding="utf-8")
    manifest = json.loads(text)
    resources = manifest.get("resources")
    if not isinstance(resources, list):
        return text
    seen: set[str] = set()

    def replace(match: re.Match[str]) -> str:
        relative = match.group("path")
        seen.add(relative)
        actual = sha256_file(resource_path(relative))
        if actual != match.group("sha"):
            drift.append(f"{path.name}: {relative}: {match.group('sha')[:12]} -> {actual[:12]}")
        return f'{{"path": "{relative}", "sha256": "sha256:{actual}"}}'

    updated = RESOURCE_LINE.sub(replace, text)
    declared = {entry["path"] for entry in resources}
    if seen != declared:
        raise SystemExit(
            f"{path.name}: could not locate every resource line textually; "
            f"missing {sorted(declared - seen)}"
        )
    return updated


def bind_digest(text: str, *, current: str, actual: str, where: str, drift: list[str]) -> str:
    if current == actual:
        return text
    if text.count(f'"{current}"') != 1:
        raise SystemExit(
            f"task JSON digest for {where} is not textually unique; refresh it by hand"
        )
    drift.append(f"task json: {where}: {current[7:19]} -> {actual[7:19]}")
    return text.replace(f'"{current}"', f'"{actual}"', 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit non-zero on drift instead of rewriting files",
    )
    args = parser.parse_args()
    drift: list[str] = []
    task_path = task_json_path()
    task_text = task_path.read_text(encoding="utf-8")
    task = json.loads(task_text)
    writes: dict[Path, str] = sync_release_fields(drift, task["judge"]["eval_timeout"])

    for manifest in sorted(PORTABLE.glob("*.json")):
        original = manifest.read_text(encoding="utf-8")
        updated = refresh_manifest(manifest, drift)
        if updated != original:
            writes[manifest] = updated

    def digest_of(name: str) -> str:
        path = PORTABLE / name
        payload = writes.get(path)
        if payload is None:
            return "sha256:" + sha256_file(path)
        return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()

    portable = task["portable"]
    bindings: list[tuple[str, str, str]] = [
        (
            "portable.task_core.resource_manifest_sha256",
            portable["task_core"]["resource_manifest_sha256"],
            digest_of(TASK_CORE_MANIFEST),
        ),
        (
            "delivery.build_resource_manifest_sha256",
            task["delivery"]["build_resource_manifest_sha256"],
            digest_of(DELIVERY_MANIFEST),
        ),
    ]
    for suite_id, suite in portable["evaluation_suites"].items():
        for role, name in SUITE_ROLE_MANIFESTS.items():
            bindings.append(
                (
                    f"portable.evaluation_suites.{suite_id}.roles.{role}.manifest_sha256",
                    suite["roles"][role]["manifest_sha256"],
                    digest_of(name),
                )
            )
            cases = json.loads(writes.get(PORTABLE / name) or (PORTABLE / name).read_text(encoding="utf-8"))["cases"]
            if suite["roles"][role]["case_count"] != len(cases):
                drift.append(
                    f"task json: {suite_id}.roles.{role}.case_count is "
                    f"{suite['roles'][role]['case_count']} but {name} has {len(cases)} "
                    "cases (fix by hand: case counts are contract, not generated)"
                )
        for key, name in SUITE_PROTOCOLS.items():
            bindings.append(
                (
                    f"portable.evaluation_suites.{suite_id}.{key}.spec_sha256",
                    suite[key]["spec_sha256"],
                    digest_of(name),
                )
            )
    for profile_id, profile in portable["execution_profiles"].items():
        for key, name in PROFILE_PROTOCOLS.items():
            bindings.append(
                (
                    f"portable.execution_profiles.{profile_id}.{key}.spec_sha256",
                    profile[key]["spec_sha256"],
                    digest_of(name),
                )
            )
    updated_task = task_text
    for where, current, actual in bindings:
        updated_task = bind_digest(
            updated_task, current=current, actual=actual, where=where, drift=drift
        )
    if updated_task != task_text:
        writes[task_path] = updated_task

    if args.check:
        for entry in drift:
            print(entry)
        return 1 if drift else 0
    if not drift:
        print("portable digests already current")
        return 0
    for path, payload in writes.items():
        path.write_text(payload, encoding="utf-8")
    for entry in drift:
        print(entry)
    print(f"rewrote {len(writes)} file(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
