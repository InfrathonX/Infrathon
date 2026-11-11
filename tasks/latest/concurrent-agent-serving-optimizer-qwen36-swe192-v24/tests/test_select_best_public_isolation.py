from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


TASK_ROOT = Path(__file__).resolve().parents[1]
SELECTOR = TASK_ROOT / "scripts/select_best.py"
TASK_ID = "concurrent_agent_serving_optimizer_qwen36_swe192_v24"
IDENTITY = {
    "file_sha256": "1" * 64,
    "valid_steps_per_min": 88.5,
    "backend_serving_contract_sha256": "2" * 64,
    "workload_nonce_sha256": "3" * 64,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_preflight(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "passed",
                "hidden_baseline_identity": IDENTITY,
                "baseline_file_sha256": IDENTITY["file_sha256"],
                "baseline_valid_steps_per_min": IDENTITY[
                    "valid_steps_per_min"
                ],
                "backend_serving_contract_sha256": IDENTITY[
                    "backend_serving_contract_sha256"
                ],
                "nonce_commitment_sha256": IDENTITY[
                    "workload_nonce_sha256"
                ],
            }
        ),
        encoding="utf-8",
    )


def _write_fake_validator(path: Path) -> None:
    path.write_text(
        textwrap.dedent(
            """
            import argparse
            import hashlib
            import json
            from pathlib import Path

            def digest(path):
                return hashlib.sha256(path.read_bytes()).hexdigest()

            parser = argparse.ArgumentParser()
            parser.add_argument("--report", required=True, type=Path)
            parser.add_argument("--private-artifact", required=True, type=Path)
            parser.add_argument("--expected-submission-id", required=True)
            parser.add_argument("--eval-rc", required=True, type=int)
            parser.add_argument("--evaluated-archive", required=True, type=Path)
            parser.add_argument("--expected-archive", required=True, type=Path)
            parser.add_argument("--expected-preflight", required=True, type=Path)
            parser.add_argument("--output", required=True, type=Path)
            args = parser.parse_args()

            report = json.loads(args.report.read_text())
            private = json.loads(args.private_artifact.read_text())
            preflight = json.loads(args.expected_preflight.read_text())
            identity = dict(preflight["hidden_baseline_identity"])
            mode = private["mode"]
            if mode == "bad-anchor":
                identity["file_sha256"] = "f" * 64
            authoritative = mode != "bad-private"
            valid = True if authoritative else None
            classification = "candidate_valid" if authoritative else "malformed"
            payload = {
                "schema_version": 1,
                "classification": classification,
                "authoritative": authoritative,
                "valid": valid,
                "task_id": report["task_id"],
                "submission_id": args.expected_submission_id,
                "score": report["score"] if authoritative else None,
                "hidden_baseline_identity": identity,
                "report": {"path": str(args.report), "sha256": digest(args.report)},
                "private_artifact": {
                    "path": str(args.private_artifact),
                    "sha256": digest(args.private_artifact),
                },
                "evaluated_archive": {
                    "path": str(args.evaluated_archive),
                    "sha256": digest(args.evaluated_archive),
                },
                "expected_archive": {
                    "path": str(args.expected_archive),
                    "sha256": digest(args.expected_archive),
                },
                "errors": [] if authoritative else ["bad private evidence"],
            }
            args.output.write_text(json.dumps(payload))
            raise SystemExit(0 if authoritative else 20)
            """
        ),
        encoding="utf-8",
    )


def _write_round(
    root: Path,
    name: str,
    score: float,
    *,
    submitted_at: float,
    mode: str = "valid",
) -> None:
    directory = root / "submissions" / name
    directory.mkdir(parents=True)
    (directory / "submission.tar.gz").write_bytes(name.encode())
    (directory / "private_judge_artifacts.tar.gz").write_text(
        json.dumps({"mode": mode}), encoding="utf-8"
    )
    (directory / "report.json").write_text(
        json.dumps(
            {
                "task_id": TASK_ID,
                "submission_id": name,
                "valid": True,
                "score": score,
                "pass_rate": 1.0,
                "submitted_at": submitted_at,
            }
        ),
        encoding="utf-8",
    )


def _run_selector(
    tmp_path: Path, source: Path, output: Path
) -> subprocess.CompletedProcess[str]:
    validator = tmp_path / "validator.py"
    preflight = tmp_path / "preflight.json"
    _write_fake_validator(validator)
    _write_preflight(preflight)
    return subprocess.run(
        [
            sys.executable,
            str(SELECTOR),
            "--task-id",
            TASK_ID,
            "--source-root",
            str(source),
            "--output-dir",
            str(output),
            "--validator",
            str(validator),
            "--expected-preflight",
            str(preflight),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def _selection(output: Path) -> dict:
    return json.loads((output / "best_selection.json").read_text(encoding="utf-8"))


def test_public_throughput_never_enters_hidden_selection(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "selection"
    _write_round(source, "agent-1", 101.0, submitted_at=10.0)
    _write_round(source, "public-1", 999.0, submitted_at=1.0)

    completed = _run_selector(tmp_path, source, output)

    assert completed.returncode == 0, completed.stderr
    selection = _selection(output)
    assert selection["schema_version"] == 2
    assert selection["discovered_candidate_count"] == 1
    assert selection["eligible_candidate_count"] == 1
    assert selection["rejections"] == []
    assert selection["best"]["round"] == "agent-1"
    assert selection["best"]["score"] == 101.0
    assert selection["hidden_baseline_identity"] == IDENTITY


@pytest.mark.parametrize(
    ("mode", "expected_reason"),
    [
        ("bad-private", "validator_exit_nonzero"),
        ("bad-anchor", "hidden_baseline_identity_mismatch"),
    ],
)
def test_bad_high_score_falls_back_to_authoritative_lower_score(
    tmp_path: Path, mode: str, expected_reason: str
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "selection"
    _write_round(source, "agent-bad", 200.0, submitted_at=1.0, mode=mode)
    _write_round(source, "agent-good", 110.0, submitted_at=2.0)

    completed = _run_selector(tmp_path, source, output)

    assert completed.returncode == 0, completed.stderr
    selection = _selection(output)
    assert selection["discovered_candidate_count"] == 2
    assert selection["eligible_candidate_count"] == 1
    assert selection["best"]["round"] == "agent-good"
    assert selection["rejections"][0]["round"] == "agent-bad"
    assert selection["rejections"][0]["reason"] == expected_reason
    assert set(selection["best"]) == {
        "round",
        "submission_id",
        "score",
        "pass_rate",
        "submitted_at",
        "archive_sha256",
        "report_sha256",
        "private_artifact_sha256",
        "qualification_validation_sha256",
    }
    validation = Path(selection["preserved_qualification_validation"])
    preserved = {
        "archive": Path(selection["preserved_archive"]),
        "report": Path(selection["preserved_report"]),
        "private": Path(selection["preserved_private_artifact"]),
        "qualification": validation,
    }
    assert {path.parent for path in preserved.values()} == {output / "best"}
    assert {name: path.name for name, path in preserved.items()} == {
        "archive": "submission.tar.gz",
        "report": "report.json",
        "private": "private_judge_artifacts.tar.gz",
        "qualification": "qualification-validation.json",
    }
    assert (output / "best").stat().st_mode & 0o777 == 0o700
    assert selection["best_validation_sha256"] == _sha256(validation)
    assert selection["best"]["qualification_validation_sha256"] == _sha256(
        validation
    )
    assert selection["best"]["archive_sha256"] == _sha256(
        Path(selection["preserved_archive"])
    )
    assert selection["best"]["report_sha256"] == _sha256(
        Path(selection["preserved_report"])
    )
    assert selection["best"]["private_artifact_sha256"] == _sha256(
        Path(selection["preserved_private_artifact"])
    )


def test_no_authoritatively_eligible_candidate_exits_nonzero(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "selection"
    _write_round(
        source, "agent-bad", 200.0, submitted_at=1.0, mode="bad-private"
    )

    completed = _run_selector(tmp_path, source, output)

    assert completed.returncode != 0
    assert "no authoritatively eligible hidden submission" in completed.stderr
    assert not (output / "best_selection.json").exists()


def test_equal_raw_score_uses_earliest_submitted_at(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "selection"
    _write_round(source, "agent-a-later", 123.0, submitted_at=20.0)
    _write_round(source, "agent-z-earlier", 123.0, submitted_at=10.0)

    completed = _run_selector(tmp_path, source, output)

    assert completed.returncode == 0, completed.stderr
    selection = _selection(output)
    assert selection["tie_break"] == "highest_raw_score_then_earliest_submitted_at"
    assert selection["best"]["round"] == "agent-z-earlier"
    assert selection["eligible_candidate_count"] == 2
