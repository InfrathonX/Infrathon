#!/usr/bin/env python3
"""Deep CPU-only validation of the EAGLE-3 task release layout."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_protected_hashes import private_data_dir  # noqa: E402

TASK_ID = "eagle3_spec_decode_engine_qwen3_8b_h100_v1"
DOCS = (
    "README.md",
    "docs/DESIGN.md",
    "docs/SCORING_CONTRACT.md",
    "docs/SECURITY.md",
    "docs/SOURCE_PROVENANCE.md",
    "docs/MODEL_MANIFEST.md",
    "docs/HEAD_MANIFEST.md",
    "docs/DATASET_MANIFEST.md",
    "docs/UPSTREAM_PIN.md",
    "docs/CALIBRATION.md",
    "docs/RUN_RESULTS.md",
    "docs/PORT_LEDGER.md",
    "docs/SINGLE_GPU_MIGRATION.md",
    "images/work/AGENT_TASK.md",
    "images/work/workdir/spec_lab/README_AGENT.md",
    "portable/REBUILD_PROMPT.md",
)
# Maintainer documents under docs/ are not part of the public release: checked
# only when present.
MAINTAINER_DOCS = tuple(name for name in DOCS if name.startswith("docs/"))
PORTABLE_FILES = (
    "portable/resource-manifest.json",
    "portable/build-resource-manifest.json",
    "portable/public-suite-manifest.json",
    "portable/interactive-suite-manifest.json",
    "portable/final-suite-manifest.json",
    "portable/oracle-protocol.json",
    "portable/selection-protocol.json",
    "portable/software-abi.json",
    "portable/calibration-protocol.json",
    "images/work/public_tests/prepare_submit.sh",
)
SCRIPTS = (
    "scripts/lib.sh",
    "scripts/run_fixed_budget.sh",
    "scripts/run_single_gpu_fixed_budget.sh",
    "scripts/single_gpu_runtime_preflight.py",
    "scripts/refresh_portable_manifests.py",
    "scripts/eval_archive.sh",
    "scripts/replay_candidates.sh",
    "scripts/preflight.py",
    "scripts/smoke_test.sh",
    "scripts/status.sh",
    "scripts/validate_layout.py",
    "scripts/validate_formal_artifacts.py",
    "scripts/validate_eval_artifacts.py",
    "scripts/generate_hidden_manifest.py",
    "scripts/generate_protected_hashes.py",
    "scripts/calibrate_oracle.py",
    "scripts/restricted_proxy.py",
)
JUDGE_FILES = (
    "images/judge/Dockerfile",
    "images/judge/run_judge.py",
    "images/judge/verify_eagle3.py",
    "images/judge/verifier/controller.py",
    "images/judge/verifier/candidate_worker.py",
    "images/judge/verifier/source_admission.py",
    "images/judge/verifier/oracle_reference.py",
    "images/judge/verifier/sampling.py",
    "images/judge/verifier/scoring.py",
    "images/judge/protected_hashes.txt",
    "images/judge/seed_reference/solutions/__init__.py",
)
# Judge-only holdout data: a dataset asset (images/judge/private_data in the
# development tree, assets/fetch.sh otherwise), resolved by private_data_dir().
PRIVATE_FILES = (
    "policy.json",
    "anchor_manifest.json",
    "final_manifest.json",
    "warmup_manifest.json",
    "commitments.json",
    "reference_streams.json",
)


def load_json(path: Path, errors: list[str]) -> dict[str, Any]:
    """Load one JSON object, accumulating a readable error."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        errors.append(f"invalid JSON {path}: {exc}")
        return {}
    if not isinstance(value, dict):
        errors.append(f"JSON root must be an object: {path}")
        return {}
    return value


def validate_required(root: Path, errors: list[str], private: Path | None) -> None:
    """Require every documented release entry."""
    for relative in (*DOCS, *SCRIPTS, *JUDGE_FILES, *PORTABLE_FILES):
        path = root / relative
        if relative in MAINTAINER_DOCS and not path.exists() and not path.is_symlink():
            continue
        if not path.is_file() or path.is_symlink():
            errors.append(f"missing or unsafe required file: {relative}")
    if private is not None:
        for name in PRIVATE_FILES:
            path = private / name
            if not path.is_file() or path.is_symlink():
                errors.append(f"missing or unsafe required file: private_data/{name}")


def validate_python(root: Path, errors: list[str]) -> None:
    """Parse Python sources and require module documentation."""
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, UnicodeError, SyntaxError) as exc:
            errors.append(f"invalid Python {path.relative_to(root)}: {exc}")
            continue
        if (
            path.parent.name in {"scripts", "judge", "verifier", "tests"}
            and not ast.get_docstring(tree)
        ):
            errors.append(f"Python module lacks a docstring: {path.relative_to(root)}")


def validate_shell(root: Path, errors: list[str]) -> None:
    """Parse all shell lifecycle entries and require strict mode."""
    for path in sorted((root / "scripts").glob("*.sh")):
        completed = subprocess.run(
            ["bash", "-n", str(path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if completed.returncode:
            errors.append(f"invalid shell {path.name}: {completed.stderr.strip()}")
        text = path.read_text(encoding="utf-8")
        if "set -Eeuo pipefail" not in text:
            errors.append(f"shell lacks strict mode: {path.name}")


def validate_metadata(root: Path, errors: list[str]) -> None:
    """Bind task metadata to version enumeration and Judge entry points."""
    spec_path = root / "tasks/eagle3_spec_decode_engine_qwen3_8b_h100_v1.json"
    spec = load_json(spec_path, errors)
    if spec.get("task_id") != TASK_ID:
        errors.append("task metadata ID differs")
    if spec.get("submit_paths") != ["solutions", "FINAL_VERSION", "notes.md"]:
        errors.append("submit_paths differ from the disclosed archive contract")
    enumeration = spec.get("version_enumeration", {})
    if enumeration.get("root") != "solutions":
        errors.append("version enumeration root differs")
    pattern = enumeration.get("directory_pattern")
    try:
        compiled = re.compile(pattern)
    except (TypeError, re.error):
        errors.append("version directory pattern is invalid")
    else:
        if compiled.groups != 1 or not compiled.fullmatch("engine_v001"):
            errors.append("version directory pattern does not select engine_vNNN")
    judge_command = spec.get("judge", {}).get("eval_cmd", "")
    if "run_judge.py" not in judge_command:
        errors.append("Judge command does not use the redacting adapter")


def validate_policy(root: Path, private: Path, errors: list[str]) -> None:
    """Require A2's frozen tiers and a valid qualification state."""
    policy = load_json(private / "policy.json", errors)
    if policy.get("status") not in {"PENDING_QUALIFICATION", "QUALIFIED"}:
        errors.append("private policy status differs")
    if policy.get("eps_hard") != 24.0 or policy.get("scope") != "global_run":
        errors.append("private policy differs from A2 calibration")
    tiers = policy.get("tiers")
    if not isinstance(tiers, list) or len(tiers) != 8:
        errors.append("private policy must contain eight tiers")
    public = load_json(
        root / "images/work/workdir/spec_lab/runtime/policy_public.json",
        errors,
    )
    for key in ("attn_implementation", "eps_hard", "p_soft", "scope", "tiers"):
        if public.get(key) != policy.get(key):
            errors.append(f"public/private frozen policy differs at {key}")


def manifest_rows(private: Path, name: str, errors: list[str]) -> dict[str, set[str]]:
    """Return prompt IDs per cell after commitment-aware manifest loading."""
    manifest = load_json(private / f"{name}_manifest.json", errors)
    by_cell: dict[str, set[str]] = {}
    for entry in manifest.get("cells", []):
        cell = entry.get("cell")
        path = private / name / f"{cell}.jsonl"
        identifiers = set()
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    identifiers.add(json.loads(line)["prompt_id"])
        except (OSError, KeyError, json.JSONDecodeError) as exc:
            errors.append(f"invalid {name}/{cell} prompt file: {exc}")
        by_cell[str(cell)] = identifiers
    return by_cell


def validate_splits(root: Path, private: Path, errors: list[str]) -> None:
    """Prove anchor/final prompt IDs are disjoint from each other and public."""
    anchor = manifest_rows(private, "anchor", errors)
    final = manifest_rows(private, "final", errors)
    public_manifest = load_json(
        root / "images/work/workdir/spec_lab/public_prompts/manifest.json",
        errors,
    )
    public: dict[str, set[str]] = {}
    for entry in public_manifest.get("cells", []):
        cell = str(entry.get("cell"))
        path = root / f"images/work/workdir/spec_lab/public_prompts/{cell}.jsonl"
        identifiers = set()
        try:
            identifiers = {
                json.loads(line)["prompt_id"]
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            }
        except (OSError, KeyError, json.JSONDecodeError) as exc:
            errors.append(f"invalid public/{cell} prompt file: {exc}")
        public[cell] = identifiers
    # Public cells are named differently from the hidden cells, so compare the
    # public prompt IDs against every hidden cell rather than cell by cell.
    every_public = set().union(*public.values())
    if every_public & set().union(*anchor.values()):
        errors.append("public/anchor prompt overlap")
    if every_public & set().union(*final.values()):
        errors.append("public/final prompt overlap")
    for cell in set(anchor) | set(final):
        if anchor.get(cell, set()) & final.get(cell, set()):
            errors.append(f"anchor/final prompt overlap in {cell}")


# sha256 of the authoritative candidate ABI. The lab tree it was calibrated in is
# an internal artifact and is not part of this repository, so the ABI identity is
# pinned here. Changing runtime/api.py is a contract change: update this constant
# and tests/test_contract_layout.py together, deliberately.
ABI_SHA256 = "477fd276807239a04180003cd318d8c9289321fe96ae4043a6206274869b38ed"


def validate_abi(root: Path, errors: list[str]) -> None:
    """Keep the shipping ABI byte-identical to the recorded authoritative one."""
    shipping = root / "images/work/workdir/spec_lab/runtime/api.py"
    if not shipping.is_file():
        errors.append("shipping runtime/api.py is missing")
    elif hashlib.sha256(shipping.read_bytes()).hexdigest() != ABI_SHA256:
        errors.append("shipping runtime/api.py differs from the recorded ABI")


HOOK = "/app/workdir/spec_lab/.sforge/prepare_submit.sh"


def validate_single_gpu_handoff(root: Path, errors: list[str]) -> None:
    """Deployment-only single-allocation contract: one UUID lock, NVML idle
    attestation before the first Docker probe, the root-owned cooperative hook
    inside the task cwd, the adapter forwarding the coordinator allocation, and
    the explicit standalone final handoff on the still-alive coordinator."""
    run_text = (root / "scripts/run_fixed_budget.sh").read_text(encoding="utf-8")
    for fragment in (
        'SINGLE_GPU_HANDOFF="${SFORGE_SINGLE_GPU_HANDOFF:-0}"',
        "acquire_single_gpu_lock",
        'phase="before_work_create"',
        "assert_portable_gpu_idle",
        "assert_container_accelerator",
        f"SFORGE_PRE_SUBMIT_HOOK={HOOK}",
        "--standalone-final-handoff",
        '--handoff-source-run-id "${RUN_ID}"',
        '--judge-url "http://127.0.0.1:${JUDGE_PORT}"',
        'serve --host "${bridge_gateway}" "${judge_serve_extra_hosts[@]}" --port "${JUDGE_PORT}"',
        'scheduling_mode="single_gpu_lease_manual_judge_then_standalone_final_handoff"',
        'scheduling_mode="dual_h100_manual_judge_then_independent_holdout_final"',
        '--evaluation-mode "${scheduling_mode}"',
        "deployment_attestation_json",
        "measurement_profile_sha256",
        "raw_score_comparison_identity_sha256",
        "--disable-auto-eval",
    ):
        if fragment not in run_text:
            errors.append(f"scripts/run_fixed_budget.sh: missing {fragment!r}")
    lock = run_text.find("acquire_single_gpu_lock \"")
    attest = run_text.find("NvmlAcceleratorObserver().attest_idle(")
    probe = run_text.find("assert_container_accelerator")
    if not 0 <= lock < attest < probe:
        errors.append(
            "scripts/run_fixed_budget.sh: single-allocation branch must lock, "
            "attest idle, then probe the container DeviceRequest, in that order"
        )
    wrapper = (root / "scripts/run_single_gpu_fixed_budget.sh").read_text(encoding="utf-8")
    for fragment in (
        "SFORGE_SINGLE_GPU_HANDOFF=1",
        "SFORGE_ACCELERATOR_OBSERVER=nvml",
        "canonical_allocation_key",
        "single_gpu_runtime_preflight.py",
        'exec "${SCRIPT_DIR}/run_fixed_budget.sh"',
    ):
        if fragment not in wrapper:
            errors.append(f"scripts/run_single_gpu_fixed_budget.sh: missing {fragment!r}")
    work = (root / "images/work/Dockerfile").read_text(encoding="utf-8")
    if f"COPY public_tests/prepare_submit.sh {HOOK}" not in work:
        errors.append("Work Dockerfile must install the cooperative pre-submit hook")
    if "chown -R root:root /app/workdir/spec_lab/.sforge" not in work:
        errors.append("Work Dockerfile must leave the pre-submit hook root-owned")
    if "portable" in work.lower() or "final-suite-manifest" in work.lower():
        errors.append("Work Dockerfile must not reference the host-private portable manifests")
    adapter = (root / "scripts/sforge_h100_entry.py").read_text(encoding="utf-8")
    for fragment in (
        "accelerator_allocation: AcceleratorAllocation | None = None",
        "_validated_handoff_binding_labels",
        "DockerBackend.create_container = role_bound_create_container",
    ):
        if fragment not in adapter:
            errors.append(f"scripts/sforge_h100_entry.py: missing {fragment!r}")


def main(argv: list[str] | None = None) -> int:
    """Run every layout check and print one JSON result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, nargs="?", default=Path(__file__).parents[1])
    args = parser.parse_args(argv)
    root = args.root.resolve()
    errors: list[str] = []
    private: Path | None = private_data_dir(root)
    if not private.is_dir():
        private = None
    validate_required(root, errors, private)
    validate_python(root, errors)
    validate_shell(root, errors)
    validate_metadata(root, errors)
    if private is not None:
        validate_policy(root, private, errors)
        validate_splits(root, private, errors)
    validate_abi(root, errors)
    validate_single_gpu_handoff(root, errors)
    result = {
        "schema_version": 1,
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
    }
    if private is None:
        result["skipped"] = [
            "Judge-only private_data is not present (run assets/fetch.sh): "
            "policy and prompt-split checks skipped"
        ]
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
