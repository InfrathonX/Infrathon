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
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

TASK_ROOT = Path(__file__).resolve().parents[1]
PORTABLE = TASK_ROOT / "portable"
RESOURCE_LINE = re.compile(
    r'\{"path": "(?P<path>[^"]+)", "sha256": "sha256:(?P<sha>[0-9a-f]{64})"\}'
)
# The edge fixture is recipe-bound: external_fixtures entries carry the SHA-256
# of fixtures/recipe/RECIPE.json rather than of any one host's captured bytes.
RECIPE_BINDING = re.compile(
    r'"recipe": "(?P<recipe>[^"]+)", "recipe_sha256": "sha256:(?P<sha>[0-9a-f]{64})"'
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


def refresh_recipe_bindings(path: Path, text: str, drift: list[str]) -> str:
    """Rewrite every external_fixtures recipe digest in one manifest."""

    def replace(match: re.Match[str]) -> str:
        relative = match.group("recipe")
        actual = sha256_file(TASK_ROOT / relative)
        if actual != match.group("sha"):
            drift.append(
                f"{path.name}: {relative}: {match.group('sha')[:12]} -> {actual[:12]}"
            )
        return f'"recipe": "{relative}", "recipe_sha256": "sha256:{actual}"'

    return RECIPE_BINDING.sub(replace, text)


def refresh_manifest(path: Path, drift: list[str]) -> str:
    """Rewrite every resource digest in one manifest; return the new text."""

    text = refresh_recipe_bindings(path, path.read_text(encoding="utf-8"), drift)
    manifest = json.loads(text)
    resources = manifest.get("resources")
    if not isinstance(resources, list):
        return text
    seen: set[str] = set()

    def replace(match: re.Match[str]) -> str:
        relative = match.group("path")
        seen.add(relative)
        actual = sha256_file(TASK_ROOT / relative)
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
    writes: dict[Path, str] = {}

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

    task_path = task_json_path()
    task_text = task_path.read_text(encoding="utf-8")
    task = json.loads(task_text)
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
            cases = json.loads((PORTABLE / name).read_text(encoding="utf-8"))["cases"]
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
