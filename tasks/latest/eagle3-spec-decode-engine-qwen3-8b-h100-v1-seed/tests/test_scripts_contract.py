"""Static and executable contracts for lifecycle scripts."""

from __future__ import annotations

import ast
import hashlib
import io
import json
import re
import subprocess
import tarfile
from types import SimpleNamespace

import pytest

from conftest import ROOT, load_file


SHELL_SCRIPTS = tuple(sorted((ROOT / "scripts").glob("*.sh")))
eval_validator = load_file(
    "validate_eval_artifacts_test",
    ROOT / "scripts/validate_eval_artifacts.py",
)
formal_validator = load_file(
    "validate_formal_artifacts_test",
    ROOT / "scripts/validate_formal_artifacts.py",
)


@pytest.mark.parametrize("path", SHELL_SCRIPTS, ids=lambda path: path.name)
def test_shell_scripts_are_bash_syntax_clean(path) -> None:
    """Every lifecycle shell entry must parse under Bash."""
    subprocess.run(["bash", "-n", str(path)], check=True)


@pytest.mark.parametrize("path", SHELL_SCRIPTS, ids=lambda path: path.name)
def test_shell_scripts_use_strict_mode_and_named_functions(path) -> None:
    """Prevent lifecycle entries from collapsing back into golfed stubs."""
    text = path.read_text(encoding="utf-8")
    assert "set -Eeuo pipefail" in text
    assert re.search(r"(?m)^[a-zA-Z_][a-zA-Z0-9_]*\(\) \{", text)


def test_python_scripts_are_pep8_shaped_modules() -> None:
    """Trusted Python entries have module and callable documentation."""
    paths = [
        *(ROOT / "scripts").glob("*.py"),
        *(ROOT / "images/judge").rglob("*.py"),
        *(ROOT / "images/work/public_tests").glob("*.py"),
    ]
    for path in sorted(paths):
        if path.name == "restricted_proxy.py":
            # Byte-identical copy of flash-kmeans' restricted_connect_proxy.py (D5); its shape is not re-styled here.
            continue
        module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        assert ast.get_docstring(module), path.name
        undocumented = [
            node.name
            for node in ast.walk(module)
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            and ast.get_docstring(node) is None
        ]
        assert undocumented == [], f"{path.name}: {undocumented}"


def test_formal_validator_supports_each_lifecycle_phase() -> None:
    """Pre, post, and standalone-eval attestations share one validator."""
    text = (ROOT / "scripts/validate_formal_artifacts.py").read_text(encoding="utf-8")
    for token in ('"pre"', '"post"', '"eval"', "expected-archive-sha256"):
        assert token in text


def test_eval_validator_exists_and_checks_consistency() -> None:
    """Evaluation result and archive records are independently cross-checked."""
    path = ROOT / "scripts/validate_eval_artifacts.py"
    assert path.is_file()
    text = path.read_text(encoding="utf-8")
    for token in ("reward.json", "report.json", "submission.tar.gz", "sha256"):
        assert token in text


def test_eval_validator_accepts_native_sforge_report_schema() -> None:
    """SForge persists EvalReport fields at top level, without `status`."""
    report = {
        "task_id": "eagle3_spec_decode_engine_qwen3_8b_h100_v1",
        "submission_id": "c2-final",
        "valid": True,
        "timed_out": False,
        "score": 123.0,
        "pass_rate": 1.0,
        "total_tests": 1,
        "passed": 1,
        "failed": 0,
        "errors": 0,
        "metrics": {
            "source": "eagle3_spec_decode_v1_verifier_adapter",
            "adapter_schema_version": 1,
            "candidate_output_suppressed": True,
            "verifier_completed": True,
            "verifier_returncode": 0,
            "result_protocol_records": 1,
            "primary_score_name": "score_tokens_per_s",
            "primary_score_unit": "tokens/s",
            "primary_score_direction": "maximize",
            "score_tokens_per_s": 123.0,
            **{name: 1.0 for name in eval_validator.REQUIRED_GATES},
        },
    }
    result = eval_validator.validate_report(report)
    assert result["status"] == "PASS"
    assert result["score"] == 123.0


def test_eval_validator_rejects_inconsistent_native_counts() -> None:
    """A nominally valid report cannot hide failed tests in its totals."""
    report = {
        "task_id": eval_validator.TASK_ID,
        "submission_id": "c2-final",
        "valid": True,
        "timed_out": False,
        "score": 123.0,
        "pass_rate": 1.0,
        "total_tests": 2,
        "passed": 1,
        "failed": 0,
        "errors": 0,
        "metrics": {
            "source": eval_validator.ADAPTER_SOURCE,
            "adapter_schema_version": 1,
            "candidate_output_suppressed": True,
            "verifier_completed": True,
            "verifier_returncode": 0,
            "result_protocol_records": 1,
            "primary_score_name": "score_tokens_per_s",
            "primary_score_unit": "tokens/s",
            "primary_score_direction": "maximize",
            "score_tokens_per_s": 123.0,
            **{name: 1.0 for name in eval_validator.REQUIRED_GATES},
        },
    }
    with pytest.raises(ValueError, match="counts"):
        eval_validator.validate_report(report)


def test_eval_validator_rejects_duplicate_archive_member(tmp_path) -> None:
    """Tar member replacement must not create an ambiguous final selector."""
    archive = tmp_path / "submission.tar.gz"
    with tarfile.open(archive, mode="w:gz") as bundle:
        for name, payload in (
            ("FINAL_VERSION", b"1\n"),
            ("FINAL_VERSION", b"2\n"),
            ("notes.md", b"notes\n"),
            ("solutions/engine_v001/candidate.py", b"# candidate\n"),
        ):
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            bundle.addfile(member, io.BytesIO(payload))
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="duplicate"):
        eval_validator.validate_archive(archive, digest)


def test_private_artifact_binds_score_gates_and_cell_geomean(tmp_path) -> None:
    """Formal validation requires the unredacted root-only evidence archive."""
    detail = {
        "status": "PASS",
        "qualification_state": "QUALIFIED",
        "score_tokens_per_s": 123.0,
        "cell_tokens_per_s": {"only": 123.0},
        "gates": {name: True for name in eval_validator.REQUIRED_GATES},
    }
    payload = json.dumps(detail).encode("utf-8")
    artifact = tmp_path / eval_validator.PRIVATE_ARTIFACT
    with tarfile.open(artifact, mode="w:gz") as archive:
        member = tarfile.TarInfo(eval_validator.PRIVATE_MEMBER)
        member.size = len(payload)
        member.mode = 0o600
        archive.addfile(member, io.BytesIO(payload))
    artifact.chmod(0o600)
    record = eval_validator.validate_private_artifact(
        artifact,
        expected_score=123.0,
    )
    assert record["member"] == eval_validator.PRIVATE_MEMBER
    assert record["cell_count"] == 1


def test_pre_final_binds_history_submission_counts(tmp_path) -> None:
    """The freeze barrier must reject a history that contradicts final_result."""
    archive = tmp_path / "final_archive.tar.gz"
    with tarfile.open(archive, mode="w:gz") as bundle:
        for name, payload in (
            ("FINAL_VERSION", b"1\n"),
            ("notes.md", b"notes\n"),
            ("solutions/engine_v001/candidate.py", b"# candidate\n"),
        ):
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            bundle.addfile(member, io.BytesIO(payload))
    (tmp_path / "final_result.json").write_text(
        json.dumps(
            {
                "agent": "codex",
                "task": formal_validator.TASK_ID,
                "run_id": "c2-test-run",
                "model": "example-model",
                "timed_out": True,
                "runtime_seconds": 120,
                "archive_size_bytes": archive.stat().st_size,
                "agent_submissions": 0,
                "auto_submissions": 0,
                "total_rounds": 0,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "final_snapshot.json").write_text(
        json.dumps({"snapshot_elapsed_seconds": 120}),
        encoding="utf-8",
    )
    (tmp_path / "run_history.json").write_text(
        json.dumps(
            {
                "run_id": "c2-test-run",
                "agent_submissions": 1,
                "auto_submissions": 0,
                "public_submissions": 0,
                "entries": [],
            }
        ),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        task_root=tmp_path,
        task_id=formal_validator.TASK_ID,
        run_id="c2-test-run",
        model="example-model",
        budget_seconds=120,
        interval_seconds=60,
    )
    with pytest.raises(ValueError, match="history.agent_submissions"):
        formal_validator.validate_pre_final(args)


def test_smoke_has_docker_and_no_docker_modes() -> None:
    """D gets an image smoke while C2 can use the lab environment directly."""
    text = (ROOT / "scripts/smoke_test.sh").read_text(encoding="utf-8")
    assert "--docker" in text
    assert "--no-docker" in text
    assert 'results_root="${OUTPUT_ROOT}/smokes"' in text
    assert '--env EAGLE3_ALL_ANCHOR=1' in text
    assert '--env EAGLE3_QUALIFICATION_MODE=1' in text
    assert 'find "${workspace}" -type d -exec chmod 0555 {} +' in text


def test_controller_emits_cap_calibration_measurements() -> None:
    """Qualification evidence must expose the bounded inputs used to freeze caps."""
    text = (
        ROOT / "images" / "judge" / "verifier" / "controller.py"
    ).read_text(encoding="utf-8")
    assert '"measurements": [' in text
    assert '"wall_s": float(row["wall_s"])' in text
    assert '"peak_memory_bytes": int(row["peak_memory_bytes"])' in text


def test_prompt_module_has_no_host_depth_assumption() -> None:
    """The protected runtime must import at Docker's shallow /app path."""
    text = (
        ROOT / "images" / "work" / "workdir" / "spec_lab" / "runtime" / "prompts.py"
    ).read_text(encoding="utf-8")
    assert "parents[5]" not in text
    assert "for parent in Path(__file__).resolve().parents" in text


@pytest.mark.parametrize(
    "name",
    (
        "eval_archive.sh",
        "replay_candidates.sh",
        "smoke_test.sh",
        "smoke_submission_plane.sh",
    ),
)
def test_post_parse_docker_reexec_preserves_original_arguments(name: str) -> None:
    """A docker-group re-exec must not silently lose already parsed flags."""
    text = (ROOT / "scripts" / name).read_text(encoding="utf-8")
    assert 'ORIGINAL_ARGS=("$@")' in text
    assert 'ensure_docker_group "${ORIGINAL_ARGS[@]}"' in text


def test_archive_eval_supplies_both_judge_deadlines() -> None:
    """The strict H100 adapter requires positive interactive and final caps."""
    text = (ROOT / "scripts" / "eval_archive.sh").read_text(encoding="utf-8")
    assert "EAGLE3_INTERACTIVE_JUDGE_TIMEOUT=" in text
    assert "EAGLE3_FINAL_JUDGE_TIMEOUT=" in text


INTERACTIVE_JUDGE_ENV_KEYS = (
    "NVIDIA_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES",
    "EAGLE3_GPU_ROLE",
    "EAGLE3_COHORT_SECRET",
    "EAGLE3_INTERACTIVE_JUDGE_TIMEOUT",
    "EAGLE3_FINAL_JUDGE_TIMEOUT",
)


def _run_lib_snippet(snippet: str) -> subprocess.CompletedProcess[str]:
    """Source the task library in a fresh Bash and run one snippet."""
    script = f'source "{ROOT / "scripts/lib.sh"}"\n{snippet}\n'
    return subprocess.run(
        ["bash", "-c", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )


def test_image_source_digest_ignores_the_publish_record(tmp_path) -> None:
    """Publishing writes images/published.json after the build; it must not change the digest."""
    for directory in ("images/work", "tasks"):
        (tmp_path / directory).mkdir(parents=True)
    (tmp_path / "images/work/Dockerfile").write_text("FROM scratch\n", encoding="utf-8")
    (tmp_path / "tasks/task.json").write_text("{}\n", encoding="utf-8")

    def digest() -> str:
        result = _run_lib_snippet(f'MIGRATION_ROOT="{tmp_path}"; image_source_digest')
        assert result.returncode == 0, result.stderr
        return result.stdout.strip()

    first = digest()
    (tmp_path / "images/published.json").write_text("{}\n", encoding="utf-8")
    assert digest() == first
    (tmp_path / "images/work/Dockerfile").write_text("FROM scratch\nRUN true\n", encoding="utf-8")
    assert digest() != first

def test_launcher_and_smoke_share_the_interactive_judge_environment_guard() -> None:
    """Formal runs and the submission-plane smoke cannot drift on Judge variables."""
    library = (ROOT / "scripts/lib.sh").read_text(encoding="utf-8")
    run = (ROOT / "scripts/run_fixed_budget.sh").read_text(encoding="utf-8")
    smoke = (ROOT / "scripts/smoke_submission_plane.sh").read_text(encoding="utf-8")
    assert "judge_extra_env_spec() {" in library
    assert "assert_interactive_judge_env_spec() {" in library
    for key in INTERACTIVE_JUDGE_ENV_KEYS:
        assert f"  {key}\n" in library
    run_guard = 'assert_interactive_judge_env_spec "${SFORGE_JUDGE_EXTRA_ENV}" "${judge_uuid}"'
    smoke_guard = 'assert_interactive_judge_env_spec "${judge_extra}" "${judge_uuid}"'
    assert run.index(run_guard) < run.index('serve --host "${bridge_gateway}"')
    assert 'judge_extra_env_spec \\\n    "${judge_uuid}"' in smoke
    assert smoke.index(smoke_guard) < smoke.index('serve --host "${bridge_gateway}"')
    assert 'SFORGE_JUDGE_EXTRA_ENV="${judge_extra}"' in smoke


def test_launcher_judge_environment_literal_matches_shared_builder() -> None:
    """The launcher's literal export must be byte-identical to the builder output."""
    run = (ROOT / "scripts/run_fixed_budget.sh").read_text(encoding="utf-8")
    literal = re.search(r'export SFORGE_JUDGE_EXTRA_ENV="([^"\n]+)"', run)
    assert literal is not None
    secret = "ab" * 32
    expanded = (
        literal.group(1)
        .replace("${judge_uuid}", "GPU-test-uuid")
        .replace("${cohort_secret}", secret)
        .replace("${FINAL_EVAL_TIMEOUT}", "5400")
    )
    assert "${" not in expanded
    built = _run_lib_snippet(
        f'judge_extra_env_spec GPU-test-uuid {secret} 5400 5400'
    )
    assert built.returncode == 0, built.stderr
    assert built.stdout.strip() == expanded
    guarded = _run_lib_snippet(
        f'assert_interactive_judge_env_spec "{expanded}" GPU-test-uuid'
    )
    assert guarded.returncode == 0, guarded.stderr


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda spec: spec.split(",", 1)[1], "omits NVIDIA_VISIBLE_DEVICES"),
        (lambda spec: spec.replace("GPU-test-uuid", "GPU-other"), "is not GPU-test-uuid"),
        (lambda spec: spec + ",EAGLE3_QUALIFICATION_MODE=1", "unexpected keys"),
        (lambda spec: spec.replace("EAGLE3_FINAL_JUDGE_TIMEOUT=5400", "EAGLE3_FINAL_JUDGE_TIMEOUT=0"), "positive integer"),
        (lambda spec: ",".join(reversed(spec.split(","))), "differs from the shared specification"),
    ),
    ids=("missing-nvidia", "wrong-uuid", "extra-key", "zero-timeout", "reordered"),
)
def test_interactive_judge_environment_guard_fails_closed(mutation, message) -> None:
    """The exact defect class of the first formal run must fail before serve."""
    secret = "cd" * 32
    spec = (
        "NVIDIA_VISIBLE_DEVICES=GPU-test-uuid,CUDA_VISIBLE_DEVICES=0,"
        "EAGLE3_GPU_ROLE=judge,"
        f"EAGLE3_COHORT_SECRET={secret},"
        "EAGLE3_INTERACTIVE_JUDGE_TIMEOUT=3600,EAGLE3_FINAL_JUDGE_TIMEOUT=5400"
    )
    guarded = _run_lib_snippet(
        f'assert_interactive_judge_env_spec "{mutation(spec)}" GPU-test-uuid'
    )
    assert guarded.returncode != 0
    assert message in guarded.stderr


def test_submission_plane_smoke_asserts_a_graded_interactive_round() -> None:
    """The smoke must inspect the live Judge container and the persisted record."""
    smoke = (ROOT / "scripts/smoke_submission_plane.sh").read_text(encoding="utf-8")
    for token in (
        "observe_judge_container() {",
        "assert_judge_container_environment() {",
        "assert_report_graded() {",
        "assert_round_graded() {",
        "assert_persisted_submission_record() {",
        'judge_env_snapshot="${smoke_log_root}/judge-container-env.json"',
        '"NVIDIA_VISIBLE_DEVICES=" + $uuid',
        '.device_requests[0].DeviceIDs == [$uuid]',
        '!= "infrastructure_error"',
        'eagle3_spec_decode_v1_verifier_adapter',
        '[[ "${round_id}" == "agent-1" ]]',
        "submissions/${round_id}",
        'grep -Fq "Judge container started"',
        "private_judge_artifacts.tar.gz",
        "-H 'Content-Type: application/gzip'",
        "-H 'X-SForge-Submission-Kind: agent'",
        "PENDING_QUALIFICATION",
        "QUALIFIED)",
    ):
        assert token in smoke, token


def test_pending_image_qualification_archive_route_is_explicit() -> None:
    """Qualification may expose a private score without weakening formal mode."""
    eval_script = (ROOT / "scripts/eval_archive.sh").read_text(encoding="utf-8")
    verifier = (ROOT / "images/judge/verify_eagle3.py").read_text(encoding="utf-8")
    validator = (ROOT / "scripts/validate_formal_artifacts.py").read_text(encoding="utf-8")
    assert "--qualification" in eval_script
    assert "EAGLE3_QUALIFICATION_MODE=1" in eval_script
    assert 'os.environ.get("EAGLE3_QUALIFICATION_MODE") == "1"' in verifier
    assert '("post", "eval", "qualify")' in validator


def test_invalid_final_leaves_a_durable_postcheck_and_completion_record() -> None:
    """A rejected final fails closed but persists its reason."""
    import subprocess
    import sys

    validator = ROOT / "scripts/validate_formal_artifacts.py"
    rejected = subprocess.run(
        [sys.executable, "-B", str(validator), "post", "--task-root", "/nonexistent",
         "--final-task-root", "/nonexistent", "--final-eval-id", "manual-1",
         "--task-id", "eagle3_spec_decode_engine_qwen3_8b_h100_v1",
         "--expected-archive-sha256", "a" * 64],
        capture_output=True, text=True, check=False,
    )
    assert rejected.returncode != 0
    record = json.loads(rejected.stdout)
    assert record["status"] == "invalid_final" and record["phase"] == "post"
    assert record["reason"] and record["reason"] in rejected.stderr
    for forbidden in ("evaluated_archive", "final_report", "private_judge_artifact"):
        assert forbidden not in record
    run = (ROOT / "scripts/run_fixed_budget.sh").read_text(encoding="utf-8")
    assert "record_invalid_final() {" in run
    assert 'status:"failed",failure_stage:"formal_postcheck"' in run
    assert '.status == "invalid_final"' in run and '.status == "PASS"' in run
    assert run.index("record_invalid_final ") < run.index('mv -- "${formal_postcheck}.tmp" "${formal_postcheck}"')
