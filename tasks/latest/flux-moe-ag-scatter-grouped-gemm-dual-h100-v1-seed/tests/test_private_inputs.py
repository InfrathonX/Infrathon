"""The final holdout is frozen, not drawn per build, and only the Judge sees it.

Since 2026-09-13 the qualification draw - the instance every historical score in
the spec doc was measured on - is committed inside the task on the Judge-only
path ``judge/private/final_cases.json`` and is what ``scripts/build_images.sh``
reads by default, so two hosts building from a plain checkout score the same
five cells.  Flux has no ``portable/`` manifests and the task JSON's ``portable``
block is an exact-keyed capability envelope that rejects extra fields, so the
task's final-suite record - the role ``portable/final-suite-manifest.json`` plays
for the nine portable tasks - is ``judge/private/MANIFEST.json``, which records
the instance's SHA-256 under ``frozen_instance``.
``scripts/generate_final_manifest.py`` no longer uses a system random source: it
draws from a seeded ``random.Random`` whose default seed is derived from the
suite identity, and ``--redraw`` is the acknowledgement that a fresh draw is a
new suite identity.

These tests assert the instance exists and is structurally what the Judge
expects, that the task JSON records its digest, that the seeded generator is
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

from conftest import TASK_ROOT, judge_only_file, load_module

ROOT = TASK_ROOT
FROZEN = "judge/private/final_cases.json"
RECORD = "judge/private/MANIFEST.json"
# The generator's real tier reads the Judge routing fixtures (repository copy or fetched asset).
ROUTING_PROBE = "judge/private/routing/gpt-oss-20b.json"


def _layout():
    return load_module("private_inputs_layout", "scripts/validate_layout.py")


def _generator():
    return load_module("private_inputs_generator", "scripts/generate_final_manifest.py")


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _final_suite() -> dict:
    return json.loads((ROOT / RECORD).read_text(encoding="utf-8"))


def test_frozen_instance_is_committed_on_the_judge_only_path() -> None:
    layout = _layout()
    assert layout.SENSITIVE_JUDGE_INPUTS == (FROZEN,) + tuple(
        f"judge/private/routing/{name}.npy" for name in layout.JUDGE_ROUTING_FIXTURES
    )
    path = judge_only_file(FROZEN)
    assert path.is_file() and not path.is_symlink() and path.stat().st_size > 0
    source = (ROOT / "scripts/validate_layout.py").read_text(encoding="utf-8")
    assert "for relative in SENSITIVE_JUDGE_INPUTS" in source
    instance = json.loads(path.read_text(encoding="utf-8"))
    generator = _generator()
    assert instance["suite_id"] == generator.SUITE_ID == "final-holdout-v2"
    assert instance["schema_version"] == 2
    assert len(instance["cases"]) == generator.EXPECTED_CASE_COUNT == 986
    runtime = load_module("private_inputs_runtime", "runtime/moe_ag_runtime.py")
    for case in instance["cases"]:
        runtime.validate_case(case)
    # The Judge's own verifier accepts it unchanged.
    verifier = load_module("private_inputs_verifier", "images/judge/verify_moe_ag.py")
    verifier.RUNTIME_ROOT = ROOT / "runtime"
    assert len(verifier.validate_final_manifest(instance, runtime)) == 986


def test_final_suite_record_records_the_frozen_instance() -> None:
    layout = _layout()
    assert layout.FINAL_SUITE_RECORD == RECORD
    suite = _final_suite()
    assert suite["suite_id"] == "final-holdout-v2"
    assert suite["case_count"] == 967
    assert suite["generator"] == "scripts/generate_final_manifest.py"
    # The record carries the digest, never the drawn cases.
    assert "cases" not in suite
    frozen = suite["frozen_instance"]
    assert frozen["path"] == FROZEN
    assert frozen["sha256"] == _digest(judge_only_file(FROZEN))
    assert frozen["provenance"].strip()
    policy = suite["draw_policy"]
    assert "frozen-instance-by-default" in policy["default"]
    assert "NEW suite identity" in policy["redraw"]


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
        "\"$(stat -c '%u' \"${final_cases_context}/final_cases.json\")\" == \"$(id -u)\"" in build
    )


def test_default_seed_is_derived_from_the_suite_identity_and_is_deterministic() -> None:
    judge_only_file(ROUTING_PROBE)
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
    assert "SystemRandom" not in source
    first = generator.generate_manifest()
    second = generator.generate_manifest()
    assert first == second == generator.generate_manifest(random.Random(generator.DEFAULT_SEED))


def test_redraw_with_an_explicit_seed_differs_and_is_reproducible(tmp_path: Path) -> None:
    judge_only_file(ROUTING_PROBE)
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
    assert payload["cases"] == redrawn["cases"]
    assert json.loads(done.stdout)["sha256"] == _digest(out)
    # A redraw is a new suite identity: it is not the frozen instance.
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
    (root / "images/work/Dockerfile").write_text("FROM scratch\n", encoding="utf-8")

    def write_record(sha: str) -> None:
        (root / RECORD).write_text(
            json.dumps(
                {
                    "suite_id": "final-holdout-v2",
                    "case_count": 967,
                    "frozen_instance": {
                        "path": FROZEN,
                        "sha256": sha,
                        "provenance": "test",
                    },
                    "draw_policy": {
                        "default": "frozen-instance-by-default",
                        "redraw": "a redraw is a NEW suite identity",
                    },
                    "routing_fixtures": {name: "0" * 64 for name in layout.JUDGE_ROUTING_FIXTURES},
                }
            ),
            encoding="utf-8",
        )

    write_record(hashlib.sha256(instance).hexdigest())
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    (root / "scripts/generate_final_manifest.py").write_text(
        "def default_seed():\n    return 0\nDEFAULT_SEED = 0\n# --seed --redraw\n",
        encoding="utf-8",
    )
    assert layout.validate_private_inputs(root) == []

    leak = root / "images/work/leaked_cases.json"
    leak.parent.mkdir(parents=True, exist_ok=True)
    leak.write_bytes(instance)
    errors = layout.validate_private_inputs(root)
    assert any("byte-identical to the Judge-only" in error for error in errors)
    leak.unlink()

    write_record("0" * 64)
    errors = layout.validate_private_inputs(root)
    assert any("does not match the committed instance" in error for error in errors)
