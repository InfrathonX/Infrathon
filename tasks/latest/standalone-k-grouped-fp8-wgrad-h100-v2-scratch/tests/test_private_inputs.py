"""The scored cases and the private edge cases are frozen, and only the Judge sees them.

Both Judge-only instances are Infrathon assets (or a development checkout's
``judge/private/`` copies) and are the default build inputs;
``portable/final-suite-manifest.json`` and ``assets/manifest.json`` record both
digests.  ``scripts/generate_final_manifest.py`` deterministically converts a
frozen, Judge-only suite plan.  These tests assert the instances exist and are
structurally what the build expects, that the manifests record their digests,
that the converter reproduces them, and that nothing a Work build context
admits is byte-identical to either instance.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

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
    for relative, suite_id, count in ((FINAL, "final-holdout-v3", 1082), (EDGE, "edge-holdout-v3", 166)):
        path = judge_only_file(relative)
        assert path.is_file() and not path.is_symlink() and path.stat().st_size > 0
        instance = json.loads(path.read_text(encoding="utf-8"))
        assert instance["suite_id"] == suite_id
        assert instance["schema_version"] == 4
        assert len(instance["cases"]) == count
    source = (ROOT / "scripts/validate_layout.py").read_text(encoding="utf-8")
    assert "for relative in SENSITIVE_JUDGE_INPUTS:" in source
    runtime = load_module("private_inputs_runtime", "runtime/k_grouped_wgrad_runtime.py")
    frozen = json.loads(judge_only_file(FINAL).read_text(encoding="utf-8"))
    assert all(set(case) == set(runtime.SCORED_KEYS) for case in frozen["cases"])


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
    assets = json.loads((ROOT / "assets/manifest.json").read_text(encoding="utf-8"))
    recorded = {asset["path"]: asset for asset in assets["assets"]}
    for relative, asset_path in ((FINAL, "judge/final_cases.json"), (EDGE, "judge/edge_cases.json")):
        path = judge_only_file(relative)
        assert recorded[asset_path]["sha256"] == _digest(path)
        assert recorded[asset_path]["size"] == path.stat().st_size


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


def _as_plan(case: dict, *, judge: bool) -> dict:
    """Map a converted case back to the frozen-plan vocabulary."""
    if case["dist"] == "uniform":
        return {"family": "legacy", "g": case["g"], "m": case["m"], "n": case["n"],
                "k_per_group": case["total_k"] // case["g"]}
    plan = {"family": case["family"], "model": case["model"], "g": case["g"], "total": case["total_k"],
            "alpha": case["alpha"], "hint_scale": case["hint_scale"]}
    if case["family"] == "sequence":
        plan["l1"], plan["l2"] = [case["m"], case["n"]], [case["m2"], case["n2"]]
    else:
        plan["m"], plan["n"] = case["m"], case["n"]
    plan["seed"] = case["seed"] - (1 << 32) if judge else 0
    return plan


def test_the_converter_reproduces_the_frozen_instance_and_public_cases() -> None:
    generator = _generator()
    frozen = json.loads(judge_only_file(FINAL).read_text(encoding="utf-8"))
    public = json.loads((ROOT / "images/work/benchmark/public_cases.json").read_text(encoding="utf-8"))
    plan = {
        "judge": [_as_plan(case, judge=True) for case in frozen["cases"]],
        "public": [_as_plan(case, judge=False) for case in public["cases"]],
    }
    judge, converted_public = generator.convert_plan(plan)
    assert generator.final_manifest(judge) == frozen
    assert generator.public_manifest(converted_public) == public
    source = (ROOT / "scripts/generate_final_manifest.py").read_text(encoding="utf-8")
    assert "secrets" not in source and "SystemRandom()" not in source and "import random" not in source


def test_the_converter_refuses_to_overwrite_and_writes_root_only(tmp_path: Path) -> None:
    generator = _generator()
    frozen = json.loads(judge_only_file(FINAL).read_text(encoding="utf-8"))
    out = tmp_path / "final_cases.json"
    generator._atomic_write_private(out, frozen)
    assert out.stat().st_mode & 0o777 == 0o600
    assert json.loads(out.read_text(encoding="utf-8")) == frozen
    try:
        generator._atomic_write_private(out, frozen)
    except FileExistsError:
        pass
    else:  # pragma: no cover - the converter must refuse
        raise AssertionError("an existing private manifest was replaced")


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
    assert "judge/private" not in work and "final_cases" not in work
    assert layout.validate_private_inputs(ROOT) == []


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

    leak = root / layout.WORK_BUILD_CONTEXTS[0] / "leaked_cases.json"
    leak.parent.mkdir(parents=True, exist_ok=True)
    leak.write_bytes(judge_only_file(EDGE).read_bytes())
    errors = layout.validate_private_inputs(root)
    assert any("byte-identical to the Judge-only" in error for error in errors)
    leak.unlink()

    manifest["edge_suite"]["frozen_instance"]["sha256"] = "0" * 64
    (root / "portable/final-suite-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    errors = layout.validate_private_inputs(root)
    assert any("does not match the committed instance" in error for error in errors)
