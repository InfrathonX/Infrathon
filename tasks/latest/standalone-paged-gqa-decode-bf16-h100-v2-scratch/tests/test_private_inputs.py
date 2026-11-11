"""The final holdout and the private edge cases are frozen, and only the Judge sees them.

Both Judge-only instances are committed under ``judge/private/`` and are the
default build inputs: ``final_cases.json`` (``final-holdout-v2``, one case per
model cell and query family, a final context-length slot and a 63-bit seed
each) and ``edge_cases.json`` (``edge-holdout-v2``, 160 edge cases).
``portable/final-suite-manifest.json`` records both digests.
``scripts/generate_final_manifest.py`` draws from a seeded ``random.Random``
whose default seed is derived from the suite identity and needs ``--redraw``
to mint a new suite identity.  These tests assert the instances exist and are
structurally what the build and the Judge expect, that the manifest records
their digests, that the generators are deterministic, and that nothing a Work
build context admits is byte-identical to either instance.
"""

from __future__ import annotations

import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import TASK_ROOT, judge_only_file, load_module

ROOT = TASK_ROOT
FINAL = "judge/private/final_cases.json"
EDGE = "judge/private/edge_cases.json"


def _layout():
    return load_module("private_inputs_layout", "scripts/validate_layout.py")


def _generator():
    return load_module("private_inputs_generator", "scripts/generate_final_manifest.py")


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _final_manifest() -> dict:
    return json.loads((ROOT / "portable/final-suite-manifest.json").read_text(encoding="utf-8"))


def test_frozen_instances_are_committed_on_the_judge_only_path() -> None:
    layout = _layout()
    assert layout.SENSITIVE_JUDGE_INPUTS == (FINAL, EDGE)
    for relative, suite_id, count in ((FINAL, "final-holdout-v4", 1114), (EDGE, "edge-holdout-v4", 168)):
        path = judge_only_file(relative)
        assert path.is_file() and not path.is_symlink() and path.stat().st_size > 0
        instance = json.loads(path.read_text(encoding="utf-8"))
        assert instance["suite_id"] == suite_id
        assert instance["schema_version"] == 4
        assert len(instance["cases"]) == count
    source = (ROOT / "scripts/validate_layout.py").read_text(encoding="utf-8")
    assert "for relative in SENSITIVE_JUDGE_INPUTS:" in source
    generator = _generator()
    frozen = json.loads(judge_only_file(FINAL).read_text(encoding="utf-8"))
    assert generator.SUITE_ID == "final-holdout-v4" and generator.FINAL_CASE_COUNT == 1114
    for case in frozen["cases"]:
        keys = generator.STREAM_KEYS if case["family"] == "stream" else generator.CALL_KEYS
        assert set(case) == set(keys)
        assert generator.FINAL_SEED_MINIMUM <= case["seed"] < (1 << 63)
    verifier = load_module("private_inputs_verifier", "images/judge/verify_paged_gqa.py")
    assert len(verifier.validate_final_manifest(frozen)) == 1114
    # The whole suite plus the edge cases reaches the worker on stdin in one piece.
    worker = load_module("private_inputs_worker", "images/judge/candidate_worker.py")
    edge = json.loads(judge_only_file(EDGE).read_text(encoding="utf-8"))
    payload = json.dumps({"cases": frozen["cases"], "edge_cases": edge["cases"]}, sort_keys=True).encode()
    assert len(payload) * 2 < worker.MAX_STDIN_BYTES
    assert len(verifier.validate_edge_manifest(json.loads(judge_only_file(EDGE).read_text(encoding="utf-8")))) == 168


def test_the_frozen_plan_in_the_source_asset_reproduces_both_suites() -> None:
    """judge/source/ and the repository stay in sync: converting the frozen plan
    gives the hidden holdout and the shipped public cases byte for byte."""
    plan_path = judge_only_file("judge/private/source/paged-gqa-plan-v2.json")
    generator = _generator()
    judge, public = generator.convert(json.loads(plan_path.read_text(encoding="utf-8")))
    hidden = generator.encode({"schema_version": generator.SCHEMA_VERSION, "suite_id": generator.SUITE_ID,
                               "cases": judge})
    shipped = generator.encode({"schema_version": generator.SCHEMA_VERSION, "suite_id": generator.PUBLIC_SUITE_ID,
                                "cases": public})
    assert hidden == judge_only_file(FINAL).read_bytes()
    assert shipped == generator.PUBLIC_PATH.read_bytes()


def test_final_suite_manifest_records_both_frozen_instances() -> None:
    manifest = _final_manifest()
    assert manifest["frozen_instance"]["path"] == FINAL
    assert manifest["frozen_instance"]["sha256"] == _digest(judge_only_file(FINAL))
    assert manifest["frozen_instance"]["provenance"].strip()
    assert manifest["edge_suite"]["frozen_instance"]["path"] == EDGE
    assert manifest["edge_suite"]["frozen_instance"]["sha256"] == _digest(judge_only_file(EDGE))
    policy = manifest["draw_policy"]
    assert "frozen-instance-by-default" in policy["default"]
    assert "NEW suite identity" in policy["redraw"]
    assert policy["seed_range"] == [1 << 32, (1 << 63) - 1]


def test_build_defaults_to_the_frozen_instances_and_still_honours_overrides() -> None:
    lib = (ROOT / "scripts/lib.sh").read_text(encoding="utf-8")
    assert f'DEFAULT_FINAL_CASES_PATH="${{MIGRATION_ROOT}}/{FINAL}"' in lib
    assert f'DEFAULT_EDGE_CASES_PATH="${{MIGRATION_ROOT}}/{EDGE}"' in lib
    build = (ROOT / "scripts/build_images.sh").read_text(encoding="utf-8")
    assert 'stage_private_manifest "${FINAL_CASES_PATH:-${DEFAULT_FINAL_CASES_PATH}}"' in build
    assert 'stage_private_manifest "${EDGE_CASES_PATH:-${DEFAULT_EDGE_CASES_PATH}}"' in build
    assert 'install -m 0600 "${source}" "${staged}"' in build
    assert "[[ \"$(stat -c '%a' \"${staged}\")\" == \"600\" ]]" in build
    assert "[[ \"$(stat -c '%u' \"${staged}\")\" == \"$(id -u)\" ]]" in build
    # No per-build draw remains: build_images.sh never executes a generator.
    executed = [
        line for line in build.splitlines()
        if "generate_" in line and not line.lstrip().startswith("#")
    ]
    assert executed == []
    assert '--output "${final_cases_source}"' not in build


def _synthetic_plan() -> dict:
    """A frozen plan in the v4 schema: the scoring tests' hidden cases plus a
    public split with its own head configurations and seeds below 2**32."""
    from test_scoring import build_final_manifest

    judge = build_final_manifest()["cases"]
    public = []
    for index, case in enumerate(judge[: _generator().PUBLIC_CASE_COUNT]):
        moved = json.loads(json.dumps(case))
        moved["h_q"], moved["h_kv"] = (40, 10) if index % 2 else (20, 5)
        moved["seed"] = 1000 + index
        public.append(moved)
    stream = next(case for case in judge if case["family"] == "stream")
    public[-1] = dict(json.loads(json.dumps(stream)), h_q=40, h_kv=10, seed=999)
    return {"judge": judge, "public": public}


def test_the_converter_is_deterministic_and_checks_the_plan(tmp_path: Path) -> None:
    generator = _generator()
    plan = _synthetic_plan()
    judge, public = generator.convert(plan)
    again, _ = generator.convert(json.loads(json.dumps(plan)))
    assert generator.encode({"schema_version": 4, "suite_id": generator.SUITE_ID, "cases": judge}) == \
        generator.encode({"schema_version": 4, "suite_id": generator.SUITE_ID, "cases": again})
    assert [case["case_id"] for case in judge][:2] == ["final-0000-decode", "final-0001-spec"]
    assert {case["family"] for case in public} == set(generator.FAMILIES)
    overlap = json.loads(json.dumps(plan))
    overlap["public"][0]["h_q"], overlap["public"][0]["h_kv"] = 32, 8
    with pytest.raises(SystemExit, match="overlap"):
        generator.convert(overlap)
    low_seed = json.loads(json.dumps(plan))
    low_seed["judge"][0]["seed"] = 77
    with pytest.raises(SystemExit, match="hidden seeds"):
        generator.convert(low_seed)


def test_the_converter_never_writes_the_hidden_suite_into_the_task_tree(tmp_path: Path) -> None:
    script = ROOT / "scripts/generate_final_manifest.py"
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(_synthetic_plan()), encoding="utf-8")
    refused = subprocess.run(
        [sys.executable, "-B", str(script), "--plan", str(plan),
         "--private-output", str(ROOT / "judge/private/final_cases.json")],
        capture_output=True, text=True,
    )
    assert refused.returncode != 0 and "inside the task tree" in refused.stderr
    assert not (ROOT / "judge/private/final_cases.json").exists()
    out = tmp_path / "hidden" / "final_cases.json"
    subprocess.run([sys.executable, "-B", str(script), "--plan", str(plan), "--private-output", str(out)],
                   capture_output=True, text=True, check=True)
    assert out.stat().st_mode & 0o777 == 0o600
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["suite_id"] == "final-holdout-v4" and len(payload["cases"]) == 1114
    again = subprocess.run([sys.executable, "-B", str(script), "--plan", str(plan), "--private-output", str(out)],
                           capture_output=True, text=True)
    assert again.returncode != 0 and "refusing to replace" in again.stderr


def test_no_work_build_context_admits_a_frozen_instance() -> None:
    layout = _layout()
    hidden = {_digest(judge_only_file(FINAL)), _digest(judge_only_file(EDGE))}
    seen = 0
    for context in layout.WORK_BUILD_CONTEXTS:
        base = ROOT / context
        assert base.is_dir()
        for relative in (FINAL, EDGE):
            assert not (ROOT / relative).resolve().is_relative_to(base.resolve())
        for path in base.rglob("*"):
            if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            seen += 1
            assert _digest(path) not in hidden, path
    assert seen > 0
    work = (ROOT / "images/work/Dockerfile").read_text(encoding="utf-8")
    assert "judge/private" not in work and "final_cases" not in work and "edge_cases.json" not in work
    assert layout.validate_private_inputs(ROOT) == []
    assert layout.validate_frozen_instances(ROOT) == []


def test_planted_copy_in_the_work_context_fails_the_layout_guard(tmp_path: Path) -> None:
    layout = _layout()
    root = tmp_path / "task"
    (root / "judge/private").mkdir(parents=True)
    for relative in (FINAL, EDGE):
        (root / relative).write_bytes(judge_only_file(relative).read_bytes())
    for context in layout.WORK_BUILD_CONTEXTS:
        (root / context).mkdir(parents=True, exist_ok=True)
    (root / "images/work/Dockerfile").write_text("FROM scratch\n", encoding="utf-8")
    (root / "portable").mkdir(parents=True, exist_ok=True)
    manifest = {
        "frozen_instance": {"path": FINAL, "sha256": _digest(judge_only_file(FINAL)), "provenance": "test"},
        "edge_suite": {"frozen_instance": {"path": EDGE, "sha256": _digest(judge_only_file(EDGE))}},
    }
    (root / "portable/final-suite-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert layout.validate_private_inputs(root) == []

    for leaked in (FINAL, EDGE):
        leak = root / layout.WORK_BUILD_CONTEXTS[0] / "leaked_cases.json"
        leak.parent.mkdir(parents=True, exist_ok=True)
        leak.write_bytes(judge_only_file(leaked).read_bytes())
        errors = layout.validate_private_inputs(root)
        assert any("byte-identical to the Judge-only" in error for error in errors), leaked
        leak.unlink()

    manifest["edge_suite"]["frozen_instance"]["sha256"] = "0" * 64
    (root / "portable/final-suite-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    errors = layout.validate_private_inputs(root)
    assert any("does not match the committed instance" in error for error in errors)


def test_frozen_instance_identity_guard_rejects_drift(tmp_path: Path) -> None:
    layout = _layout()
    root = tmp_path / "task"
    for relative in (FINAL, EDGE, layout.PUBLIC_EDGE_CASES, "portable/public-suite-manifest.json"):
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        (root / relative).write_bytes(judge_only_file(relative).read_bytes())
    (root / "scripts").mkdir()
    (root / "scripts/generate_final_manifest.py").write_bytes(
        (ROOT / "scripts/generate_final_manifest.py").read_bytes()
    )
    assert layout.validate_frozen_instances(root) == []

    final = json.loads((root / FINAL).read_text(encoding="utf-8"))
    final["cases"][0]["cell"] = "M1"
    final["cases"][1]["seed"] = 12345
    (root / FINAL).write_text(json.dumps(final), encoding="utf-8")
    public = json.loads((root / layout.PUBLIC_EDGE_CASES).read_text(encoding="utf-8"))
    public["cases"][0]["native"] = False
    (root / layout.PUBLIC_EDGE_CASES).write_text(json.dumps(public), encoding="utf-8")
    errors = layout.validate_frozen_instances(root)
    assert any("final holdout case is malformed" in error for error in errors)
    assert any("seed outside" in error for error in errors)
    assert any("must run natively" in error for error in errors)
    assert any("edge_suite must record" in error for error in errors)
