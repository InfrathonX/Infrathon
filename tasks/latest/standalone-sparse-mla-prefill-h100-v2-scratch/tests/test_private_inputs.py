"""The final holdout and the private edge cases are frozen, and only the Judge sees them.

Both Judge-only instances live on the Judge-only path (a development checkout's
``judge/private/``, else the fetched dataset asset) and are the default build
inputs: ``final_cases.json`` (``final-holdout-v4``, schema 5, 997 cases in
eight groups with a 63-bit seed each) and ``edge_cases.json``
(``edge-holdout-v3``, 470 edge cases).
``portable/final-suite-manifest.json`` records both digests.
``scripts/generate_final_manifest.py`` draws seeds, sinks and the checked half
over a Judge-only ``--plan`` from a seeded ``random.Random`` whose default seed
is derived from the suite identity, and needs ``--redraw`` to mint a new suite
identity; the tests here drive it with a synthetic plan.  These tests assert the instances exist and are
structurally what the build and the Judge expect, that the manifest records
their digests, that the generators are deterministic, and that nothing a Work
build context admits is byte-identical to either instance.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import sys
from pathlib import Path

from conftest import TASK_ROOT, judge_only_file, load_module
from suite_fixtures import fake_plan

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


def _plan_of(instance: dict) -> list[dict]:
    """The plan rows a manifest was drawn from (what the generator does not draw)."""
    rows = []
    for case in instance["cases"]:
        row = {key: case[key] for key in ("group", "family", "h_q", "s_q", "s_kv")}
        if case["group"] == "length":
            row["length"] = f"topk/{case['length_divisor']}"
            row["ragged"] = case["ragged"]
        elif case["group"] == "pattern":
            row["index_pattern"] = case["index_pattern"]
        elif case["group"] == "multi":
            row["request_lengths"] = case["request_lengths"]
        elif case["group"] in ("trace", "replay"):
            row["call_list"] = case["calls"]
        rows.append(row)
    return rows


def test_frozen_instances_are_committed_on_the_judge_only_path() -> None:
    layout = _layout()
    assert layout.SENSITIVE_JUDGE_INPUTS == (FINAL, EDGE)
    for relative, suite_id, schema, count in (
        (FINAL, "final-holdout-v4", 5, 997), (EDGE, "edge-holdout-v3", 3, 470),
    ):
        path = judge_only_file(relative)
        assert path.is_file() and not path.is_symlink() and path.stat().st_size > 0
        instance = json.loads(path.read_text(encoding="utf-8"))
        assert instance["suite_id"] == suite_id
        assert instance["schema_version"] == schema
        assert len(instance["cases"]) == count
    source = (ROOT / "scripts/validate_layout.py").read_text(encoding="utf-8")
    assert "for relative in SENSITIVE_JUDGE_INPUTS:" in source
    generator = _generator()
    frozen = json.loads(judge_only_file(FINAL).read_text(encoding="utf-8"))
    assert generator.SUITE_ID == "final-holdout-v4" and generator.EXPECTED_CASE_COUNT == 997
    assert generator.SCHEMA_VERSION == 5
    for case in frozen["cases"]:
        assert set(generator.BASE_KEYS) <= set(case)
        assert case["group"] in generator.GROUPS and case["family"] in generator.FAMILIES
        assert generator.SEED_RANGE[0] <= case["seed"] < generator.SEED_RANGE[1]
    # The frozen instance is the default-seed draw over its own (Judge-only) plan.
    assert frozen == generator.generate_manifest(_plan_of(frozen))
    assert frozen != generator.generate_manifest(_plan_of(frozen), random.Random(1))
    verifier = load_module("private_inputs_verifier", "images/judge/verify_sparse_mla.py")
    assert len(verifier.validate_final_manifest(frozen)) == 997
    edges = json.loads(judge_only_file(EDGE).read_text(encoding="utf-8"))
    assert len(verifier.validate_edge_manifest(edges)) == 470
    edge_generator = load_module("private_inputs_edge_generator", "scripts/generate_edge_manifest.py")
    assert edge_generator.encode(edge_generator.generate(edge_generator.PRIVATE_SUITE_ID, public=False)) == (
        judge_only_file(EDGE).read_bytes()
    )


def test_the_judge_loaders_accept_both_frozen_instances(tmp_path: Path) -> None:
    """The root-only loaders (size caps included) read the frozen instances as staged."""
    verifier = load_module("private_inputs_loader_verifier", "images/judge/verify_sparse_mla.py")
    for relative, loader, count in (
        (FINAL, verifier.load_final_manifest, 997), (EDGE, verifier.load_edge_manifest, 470),
    ):
        staged = tmp_path / Path(relative).name
        staged.write_bytes(judge_only_file(relative).read_bytes())
        staged.chmod(0o600)
        cases, digest = loader(staged, expected_uid=os.getuid())
        assert len(cases) == count
        assert digest == _digest(judge_only_file(relative))


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


def test_default_seed_is_derived_from_the_suite_identity_and_is_deterministic() -> None:
    generator = _generator()
    expected = int.from_bytes(
        hashlib.sha256(f"{generator.SUITE_ID}|{generator.TASK_ID}".encode("utf-8")).digest()[:8],
        "big",
    )
    assert generator.DEFAULT_SEED == generator.default_seed() == expected
    source = (ROOT / "scripts/generate_final_manifest.py").read_text(encoding="utf-8")
    assert "secrets" not in source and "SystemRandom()" not in source
    # The plan is a Judge-only input; a synthetic plan exercises the draw.
    plan = fake_plan()
    assert generator.generate_manifest(plan) == generator.generate_manifest(
        plan, random.Random(generator.DEFAULT_SEED)
    )
    assert generator.generate_manifest(plan) == generator.generate_manifest(plan)


def test_redraw_with_an_explicit_seed_differs_and_is_reproducible(tmp_path: Path) -> None:
    generator = _generator()
    plan = fake_plan()
    redrawn = generator.generate_manifest(plan, random.Random(20261006))
    assert redrawn != generator.generate_manifest(plan)
    assert redrawn == generator.generate_manifest(plan, random.Random(20261006))
    # The draw changes only seeds, sinks and the checked half, never the plan.
    assert _plan_of(redrawn) == _plan_of(generator.generate_manifest(plan))
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps({"judge": plan}), encoding="utf-8")
    script = ROOT / "scripts/generate_final_manifest.py"
    refused = subprocess.run(
        [sys.executable, "-B", str(script), "--plan", str(plan_path), "--output", str(tmp_path / "refused.json")],
        capture_output=True,
        text=True,
    )
    assert refused.returncode != 0 and "--redraw" in refused.stderr
    assert not (tmp_path / "refused.json").exists()
    no_plan = subprocess.run(
        [sys.executable, "-B", str(script), "--output", str(tmp_path / "no-plan.json"), "--redraw"],
        capture_output=True,
        text=True,
    )
    assert no_plan.returncode != 0 and "--plan" in no_plan.stderr
    assert not (tmp_path / "no-plan.json").exists()
    out = tmp_path / "redrawn.json"
    done = subprocess.run(
        [sys.executable, "-B", str(script), "--plan", str(plan_path), "--output", str(out), "--redraw",
         "--seed", "20261006"],
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload == redrawn
    assert json.loads(done.stdout)["sha256"] == _digest(out)
    assert out.stat().st_mode & 0o777 == 0o600
    # An existing manifest is never replaced.
    again = subprocess.run(
        [sys.executable, "-B", str(script), "--plan", str(plan_path), "--output", str(out), "--redraw"],
        capture_output=True,
        text=True,
    )
    assert again.returncode != 0 and json.loads(out.read_text(encoding="utf-8")) == redrawn


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
    final["cases"][0]["group"] = "bonus"  # not a scored group
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
