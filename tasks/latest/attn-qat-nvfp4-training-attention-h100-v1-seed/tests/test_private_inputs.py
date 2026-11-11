"""The final holdout is frozen, not drawn per build, and only the Judge sees it.

Since 2026-09-13 the qualification draw - the instance every historical score in
this task's spec doc was measured on - is committed inside the task on the
Judge-only path ``judge/private/final_cases.json`` and is what
``scripts/build_images.sh`` reads by default, so two hosts building from a plain
checkout score the same suite.  ``portable/final-suite-manifest.json`` records
its SHA-256 under ``frozen_instance``.  ``scripts/generate_final_manifest.py`` no
longer uses a system random source: it draws from a seeded ``random.Random``
whose default seed is derived from the suite identity, and ``--redraw`` is the
acknowledgement that a fresh draw is a new suite identity.

These tests assert the instance exists and is structurally what the build
expects, that the manifest records its digest, that the seeded generator is
deterministic and that an explicit ``--seed`` redraw differs, and that nothing a
Work build context admits is byte-identical to the instance.
"""

from __future__ import annotations

import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path

import conftest
from conftest import TASK_ROOT, judge_only_file

# Task conftests expose the path loader under one of two names.
load_module = getattr(conftest, "load_module", None) or conftest._load

ROOT = TASK_ROOT
FROZEN = "judge/private/final_cases.json"
ENTRY_KEY = "cases"
EXPECTED_ENTRIES = 103


def _layout():
    return load_module("private_inputs_layout", "scripts/validate_layout.py")


def _generator():
    return load_module("private_inputs_generator", "scripts/generate_final_manifest.py")


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _final_manifest() -> dict:
    return json.loads(
        (ROOT / "portable/final-suite-manifest.json").read_text(encoding="utf-8")
    )


def test_frozen_instance_is_committed_on_the_judge_only_path() -> None:
    layout = _layout()
    assert layout.SENSITIVE_JUDGE_INPUTS == (FROZEN,)
    path = judge_only_file(FROZEN)
    assert path.is_file() and not path.is_symlink() and path.stat().st_size > 0
    source = (ROOT / "scripts/validate_layout.py").read_text(encoding="utf-8")
    assert "for relative in SENSITIVE_JUDGE_INPUTS" in source
    instance = json.loads(path.read_text(encoding="utf-8"))
    generator = _generator()
    assert instance["suite_id"] == generator.SUITE_ID
    assert instance["schema_version"] == 1
    assert len(instance[ENTRY_KEY]) == EXPECTED_ENTRIES
    # the build refuses any instance with a different identity or size
    build = (ROOT / "scripts/build_images.sh").read_text(encoding="utf-8")
    assert '"${final_cases_count}" -eq %d' % EXPECTED_ENTRIES in build
    assert '== "%s"' % generator.SUITE_ID in build
    # a fresh draw has the same shape as the frozen instance
    drawn = generator.generate_manifest(random.Random(1))
    assert set(drawn) == set(instance)
    assert len(drawn[ENTRY_KEY]) == EXPECTED_ENTRIES
    for left, right in zip(drawn[ENTRY_KEY], instance[ENTRY_KEY]):
        assert set(left) == set(right)


def test_final_suite_manifest_records_the_frozen_instance() -> None:
    manifest = _final_manifest()
    frozen = manifest["frozen_instance"]
    assert frozen["path"] == FROZEN
    assert frozen["sha256"] == _digest(judge_only_file(FROZEN))
    assert frozen["provenance"].strip()
    policy = manifest["draw_policy"]
    assert "frozen-instance-by-default" in policy["default"]
    assert "NEW suite identity" in policy["redraw"]
    assert "per image build" not in policy.get("privacy", "")


def test_build_defaults_to_the_frozen_instance_and_still_honours_an_override() -> None:
    lib = (ROOT / "scripts/lib.sh").read_text(encoding="utf-8")
    assert 'DEFAULT_FINAL_CASES_PATH="${MIGRATION_ROOT}/' + FROZEN + '"' in lib
    build = (ROOT / "scripts/build_images.sh").read_text(encoding="utf-8")
    assert 'final_cases_source="${FINAL_CASES_PATH:-${DEFAULT_FINAL_CASES_PATH}}"' in build
    # No per-build draw remains: build_images.sh never invokes the generator.
    assert 'generator="${MIGRATION_ROOT}/scripts/generate_final_manifest.py"' not in build
    assert '--output "${final_cases_source}"' not in build
    # The fail-closed mode/owner checks now guard the staged 0600 copy.
    assert (
        'install -m 0600 "${final_cases_source}" "${final_cases_context}/final_cases.json"'
        in build
    )
    assert "\"$(stat -c '%a' \"${final_cases_context}/final_cases.json\")\" == \"600\"" in build
    assert (
        "\"$(stat -c '%u' \"${final_cases_context}/final_cases.json\")\" == \"$(id -u)\""
        in build
    )


def test_default_seed_is_derived_from_the_suite_identity_and_is_deterministic() -> None:
    generator = _generator()
    expected = int.from_bytes(
        hashlib.sha256(
            f"{generator.SUITE_ID}|{generator.TASK_ID}".encode("utf-8")
        ).digest()[:8],
        "big",
    )
    assert generator.DEFAULT_SEED == generator.default_seed() == expected
    source = (ROOT / "scripts/generate_final_manifest.py").read_text(encoding="utf-8")
    assert "secrets" not in source
    assert "SystemRandom()" not in source
    first = generator.generate_manifest()
    second = generator.generate_manifest()
    assert first == second == generator.generate_manifest(
        random.Random(generator.DEFAULT_SEED)
    )


def test_redraw_with_an_explicit_seed_differs_and_is_reproducible(tmp_path: Path) -> None:
    generator = _generator()
    redrawn = generator.generate_manifest(random.Random(20260913))
    assert redrawn != generator.generate_manifest()
    assert redrawn == generator.generate_manifest(random.Random(20260913))

    script = ROOT / "scripts/generate_final_manifest.py"
    refused = subprocess.run(
        [sys.executable, "-B", str(script), "--output", str(tmp_path / "refused.json")],
        capture_output=True,
        text=True,
    )
    assert refused.returncode != 0 and "--redraw" in refused.stderr
    assert not (tmp_path / "refused.json").exists()

    out = tmp_path / "redrawn.json"
    done = subprocess.run(
        [
            sys.executable,
            "-B",
            str(script),
            "--output",
            str(out),
            "--redraw",
            "--seed",
            "20260913",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload[ENTRY_KEY] == redrawn[ENTRY_KEY]
    assert json.loads(done.stdout)["sha256"] == _digest(out)
    assert payload != json.loads(judge_only_file(FROZEN).read_text(encoding="utf-8"))
    assert out.stat().st_mode & 0o777 == 0o600


def test_no_work_build_context_admits_the_frozen_instance() -> None:
    layout = _layout()
    hidden = _digest(judge_only_file(FROZEN))
    seen = 0
    for context in layout.WORK_BUILD_CONTEXTS:
        base = ROOT / context
        assert base.is_dir()
        assert not (ROOT / FROZEN).resolve().is_relative_to(base.resolve())
        for path in base.rglob("*"):
            if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            seen += 1
            assert _digest(path) != hidden, path
    assert seen > 0
    work = (ROOT / "images/work/Dockerfile").read_text(encoding="utf-8")
    assert "judge/private" not in work and "final_cases" not in work
    assert layout.validate_private_inputs(ROOT) == []


def test_planted_copy_in_the_work_context_fails_the_layout_guard(tmp_path: Path) -> None:
    layout = _layout()
    root = tmp_path / "task"
    (root / "judge/private").mkdir(parents=True)
    instance = judge_only_file(FROZEN).read_bytes()
    (root / FROZEN).write_bytes(instance)
    for context in layout.WORK_BUILD_CONTEXTS:
        (root / context).mkdir(parents=True, exist_ok=True)
    (root / "images/work").mkdir(parents=True, exist_ok=True)
    (root / "images/work/Dockerfile").write_text("FROM scratch\n", encoding="utf-8")
    (root / "portable").mkdir(parents=True, exist_ok=True)
    (root / "portable/final-suite-manifest.json").write_text(
        json.dumps(
            {
                "frozen_instance": {
                    "path": FROZEN,
                    "sha256": hashlib.sha256(instance).hexdigest(),
                    "provenance": "test",
                }
            }
        ),
        encoding="utf-8",
    )
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    (root / "scripts/generate_final_manifest.py").write_text(
        "def default_seed():\n    return 0\nDEFAULT_SEED = 0\n# --seed --redraw\n",
        encoding="utf-8",
    )
    assert layout.validate_private_inputs(root) == []

    leak = root / layout.WORK_BUILD_CONTEXTS[0] / "leaked_cases.json"
    leak.parent.mkdir(parents=True, exist_ok=True)
    leak.write_bytes(instance)
    errors = layout.validate_private_inputs(root)
    assert any("byte-identical to the Judge-only" in error for error in errors)
    leak.unlink()

    (root / "portable/final-suite-manifest.json").write_text(
        json.dumps(
            {"frozen_instance": {"path": FROZEN, "sha256": "0" * 64, "provenance": "x"}}
        ),
        encoding="utf-8",
    )
    errors = layout.validate_private_inputs(root)
    assert any("does not match the committed instance" in error for error in errors)
