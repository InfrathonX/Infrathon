from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import math
import shutil
import tarfile
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VALIDATOR_PATH = PROJECT_ROOT / "scripts" / "validate_formal_artifacts.py"


def _load_validator():
    spec = importlib.util.spec_from_file_location(
        "validate_formal_artifacts", VALIDATOR_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


validator = _load_validator()


class FormalArtifactFixture:
    run_id = "formal-test-run"
    final_eval_id = f"{run_id}-final-120m"
    task_id = "flash_kmeans_paper_h100_kernel_lab_v7_scratch"
    model = "example-model"
    score_group_weights = {
        "assign": 0.10,
        "update": 0.10,
        "full": 0.20,
        "deploy_control": 0.10,
        "deploy_large_k": 0.25,
        "deploy_large_n_high_d": 0.25,
    }
    score_group_latencies = {
        "assign": 40.0,
        "update": 50.0,
        "full": 60.0,
        "deploy_control": 80.0,
        "deploy_large_k": 100.0,
        "deploy_large_n_high_d": 120.0,
    }

    @classmethod
    def expected_score(cls) -> float:
        return math.exp(
            sum(
                cls.score_group_weights[group]
                * math.log(cls.score_group_latencies[group])
                for group in cls.score_group_weights
            )
        )

    def __init__(self, root: Path):
        self.task_root = root / "main" / self.task_id
        self.final_task_root = root / self.final_eval_id / self.task_id
        self.task_root.mkdir(parents=True)
        self.final_submission_root = (
            self.final_task_root / "submissions" / "manual-1"
        )
        self.final_submission_root.mkdir(parents=True)

        self._write_final_archive(
            [
                ("kernels", b"", tarfile.DIRTYPE, ""),
                ("kernels/flash_kmeans_v002.py", b"# Python\n", tarfile.REGTYPE, ""),
                ("FINAL_VERSION", b"001\n", tarfile.REGTYPE, ""),
                ("notes.md", b"fixture\n", tarfile.REGTYPE, ""),
            ]
        )

        entries = []
        for sequence in (1, 2, 3):
            round_id = f"auto-{sequence}"
            submission_id = f"submission-{sequence}"
            round_root = self.task_root / "submissions" / round_id
            round_root.mkdir(parents=True)
            (round_root / "submission.tar.gz").write_bytes(
                f"auto archive {sequence}".encode()
            )
            self._write_json(
                round_root / "report.json",
                {
                    "task_id": self.task_id,
                    "submission_id": submission_id,
                    "valid": True,
                    "pass_rate": 1.0,
                    "score": 100.0 + sequence,
                    "score_0_100": 20.0 + sequence,
                },
            )
            entries.append(
                {
                    "type": "submission",
                    "status": "completed",
                    "submission_id": submission_id,
                    "task_id": self.task_id,
                    "round": round_id,
                }
            )

        self._write_json(
            self.task_root / "run_history.json",
            {
                "run_id": self.run_id,
                "agent_submissions": 0,
                "auto_submissions": 3,
                "entries": entries,
            },
        )
        self._write_json(
            self.task_root / "final_snapshot.json",
            {
                "captured_at": 1_800_007_200.5,
                "snapshot_elapsed_seconds": 7_200.5,
            },
        )
        self._write_result()

        final_archive = self.task_root / "final_archive.tar.gz"
        (self.final_submission_root / "submission.tar.gz").write_bytes(
            final_archive.read_bytes()
        )
        self._write_json(
            self.final_submission_root / "report.json",
            {
                "task_id": self.task_id,
                "submission_id": self.final_eval_id,
                "total_tests": 4,
                "passed": 4,
                "failed": 0,
                "errors": 0,
                "pass_rate": 1.0,
                "score": self.expected_score(),
                "score_0_100": 54.636,
                "timed_out": False,
                "runtime_seconds": 20.0,
                "valid": True,
                "metrics": {
                    "source": "flash_kmeans_h100_verifier_adapter",
                    "adapter_schema_version": 8,
                    "feedback_schema_version": 7,
                    "candidate_output_suppressed": True,
                    "result_protocol_records": 1,
                    "verifier_completed": True,
                    "verifier_returncode": 0,
                    "deployment_family_feedback": {
                        "large-K": {
                            "status": "passed",
                            "warmed_latency_band_ms": {
                                "lower_ms": 100.0,
                                "upper_ms": 125.0,
                            },
                        },
                        "large-N/high-D": {
                            "status": "passed",
                            "warmed_latency_band_ms": {
                                "lower_ms": 120.0,
                                "upper_ms": 150.0,
                            },
                        },
                    },
                },
            },
        )
        self._write_private_judge_artifact()

    @staticmethod
    def _write_json(path: Path, payload: object) -> None:
        path.write_text(json.dumps(payload), encoding="utf-8")

    def _write_result(self, **overrides: object) -> None:
        payload = {
            "agent": "codex",
            "task": self.task_id,
            "run_id": self.run_id,
            "model": self.model,
            "timed_out": True,
            "runtime_seconds": 7_200.25,
            "archive_size_bytes": (
                self.task_root / "final_archive.tar.gz"
            ).stat().st_size,
            "agent_submissions": 0,
            "auto_submissions": 3,
            "total_rounds": 3,
        }
        payload.update(overrides)
        self._write_json(self.task_root / "final_result.json", payload)

    def _write_final_archive(
        self,
        members: list[tuple[str, bytes, bytes, str]],
    ) -> None:
        path = self.task_root / "final_archive.tar.gz"
        with tarfile.open(path, mode="w:gz") as bundle:
            for name, content, member_type, link_name in members:
                info = tarfile.TarInfo(name)
                info.type = member_type
                info.linkname = link_name
                if member_type == tarfile.REGTYPE:
                    info.size = len(content)
                    bundle.addfile(info, io.BytesIO(content))
                else:
                    bundle.addfile(info)
        if (self.task_root / "final_result.json").exists():
            result = json.loads(
                (self.task_root / "final_result.json").read_text(encoding="utf-8")
            )
            result["archive_size_bytes"] = path.stat().st_size
            self._write_json(self.task_root / "final_result.json", result)

    def _write_private_judge_artifact(self) -> None:
        cases: list[dict[str, object]] = []
        for index in range(14):
            deployment = index < 6
            runtime_index = index - 6
            runtime_stage = (
                "assign"
                if runtime_index < 2
                else ("update" if runtime_index < 4 else "full")
            )
            if deployment:
                score_group = (
                    "deploy_control"
                    if index == 0
                    else (
                        "deploy_large_k"
                        if index in {1, 3, 5}
                        else "deploy_large_n_high_d"
                    )
                )
                weight = {
                    "deploy_control": 0.10,
                    "deploy_large_k": 0.25 / 3.0,
                    "deploy_large_n_high_d": 0.25 / 2.0,
                }[score_group]
            else:
                score_group = runtime_stage
                weight = 0.05
            candidate_ms = self.score_group_latencies[score_group]
            row: dict[str, object] = {
                "name": (
                    f"hidden_deployment_fixture_{index}"
                    if deployment
                    else f"hidden_runtime_fixture_{index}"
                ),
                "stage": "full" if deployment else runtime_stage,
                "shape": [1, 1024 + index, 32, 128],
                "weight": weight,
                "score_group": score_group,
                "candidate_ms": candidate_ms,
                "measurement_mode": (
                    "fresh_process" if deployment else "steady_state"
                ),
                "all_gates_ok": True,
            }
            if deployment:
                row["candidate_first_run_ms"] = candidate_ms + 50.0
                row["candidate_warmed_iteration_ms"] = candidate_ms
            cases.append(row)
        for index in range(8):
            cases.append(
                {
                    "name": f"hidden_gate_fixture_{index}",
                    "stage": "update",
                    "shape": [1, 64 + index, 8, 16],
                    "weight": 0.0,
                    "score_group": "",
                    "candidate_ms": 0.0,
                    "measurement_mode": "steady_state",
                    "all_gates_ok": True,
                }
            )
        payload = json.dumps(
            {
                "schema_version": 6,
                "task": self.task_id,
                "score_profile": "v7_family_tail60",
                "score_group_weights": self.score_group_weights,
                "coarse_deployment_family_statuses": {
                    "large_k": "passed",
                    "large_n_high_d": "passed",
                },
                "final_version": 1.0,
                "score": self.expected_score(),
                "cases": cases,
            },
            sort_keys=True,
        ).encode("utf-8")
        path = (
            self.final_submission_root
            / validator.PRIVATE_JUDGE_ARTIFACT_NAME
        )
        with tarfile.open(path, mode="w:gz") as archive:
            info = tarfile.TarInfo(validator.PRIVATE_JUDGE_MEMBER_NAME)
            info.size = len(payload)
            info.mode = 0o600
            archive.addfile(info, io.BytesIO(payload))

    def pre(self):
        return validator.validate_pre_final(
            self.task_root,
            run_id=self.run_id,
            task_id=self.task_id,
            model=self.model,
            budget_seconds=7_200,
            interval_seconds=1_800,
        )

    def post(self, expected_digest: str):
        return validator.validate_final_eval(
            self.task_root,
            self.final_task_root,
            final_eval_id=self.final_eval_id,
            task_id=self.task_id,
            expected_archive_sha256=expected_digest,
        )

    def load_json(self, relative_path: str) -> dict:
        return json.loads(
            (self.task_root / relative_path).read_text(encoding="utf-8")
        )

    def write_json(self, relative_path: str, payload: object) -> None:
        self._write_json(self.task_root / relative_path, payload)

    def add_agent_submission(self) -> None:
        history = self.load_json("run_history.json")
        sequence = history["agent_submissions"] + 1
        round_id = f"agent-{sequence}"
        submission_id = f"agent-submission-{sequence}"
        round_root = self.task_root / "submissions" / round_id
        round_root.mkdir(parents=True)
        (round_root / "submission.tar.gz").write_bytes(
            f"agent archive {sequence}".encode()
        )
        self._write_json(
            round_root / "report.json",
            {
                "task_id": self.task_id,
                "submission_id": submission_id,
                "valid": True,
                "pass_rate": 1.0,
                "score": 123.0 + sequence,
                "score_0_100": 25.0 + sequence,
            },
        )
        history["agent_submissions"] = sequence
        history["entries"].insert(
            min(2 * sequence - 1, len(history["entries"])),
            {
                "type": "submission",
                "status": "completed",
                "submission_id": submission_id,
                "task_id": self.task_id,
                "round": round_id,
            },
        )
        self.write_json("run_history.json", history)
        self._write_result(
            agent_submissions=sequence,
            total_rounds=3 + sequence,
        )


class FormalArtifactValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.fixture = FormalArtifactFixture(Path(self.temp_dir.name))

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def assert_validation_error(self, pattern: str, callback) -> None:
        with self.assertRaisesRegex(validator.ArtifactValidationError, pattern):
            callback()

    def test_accepts_complete_pre_and_post_artifact_set(self) -> None:
        pre = self.fixture.pre()
        digest = pre["final_archive"]["sha256"]
        post = self.fixture.post(digest)

        self.assertEqual(len(pre["auto_rounds"]), 3)
        self.assertEqual(post["final_archive"]["sha256"], digest)
        self.assertEqual(post["evaluated_archive"]["sha256"], digest)
        self.assertTrue(post["final_report"]["valid"])
        self.assertEqual(
            post["private_judge_artifact"]["performance_case_count"],
            14,
        )
        self.assertEqual(
            post["private_judge_artifact"]["gate_case_count"],
            8,
        )
        self.assertAlmostEqual(
            post["private_judge_artifact"]["weighted_latency_ms"],
            self.fixture.expected_score(),
        )
        self.assertEqual(
            post["validated_report"]["submission_id"], self.fixture.final_eval_id
        )

    def test_accepts_subsecond_snapshot_runtime_reporting_skew(self) -> None:
        self.fixture._write_result(runtime_seconds=7_200.75)
        pre = self.fixture.pre()
        self.assertEqual(pre["snapshot_elapsed_seconds"], 7_200.5)
        self.assertEqual(pre["runtime_seconds"], 7_200.75)

    def test_rejects_snapshot_runtime_reporting_skew_above_tolerance(self) -> None:
        self.fixture._write_result(runtime_seconds=7_202.0)
        self.assert_validation_error(
            "final snapshot precedes the measured agent runtime",
            self.fixture.pre,
        )

    def test_accepts_one_interleaved_agent_submission(self) -> None:
        self.fixture.add_agent_submission()
        pre = self.fixture.pre()
        self.assertEqual(
            [round_data["round_id"] for round_data in pre["auto_rounds"]],
            ["auto-1", "auto-2", "auto-3"],
        )
        self.assertEqual(
            [round_data["round_id"] for round_data in pre["agent_rounds"]],
            ["agent-1"],
        )

    def test_accepts_multiple_interleaved_agent_submissions(self) -> None:
        self.fixture.add_agent_submission()
        self.fixture.add_agent_submission()
        pre = self.fixture.pre()
        self.assertEqual(
            [round_data["round_id"] for round_data in pre["agent_rounds"]],
            ["agent-1", "agent-2"],
        )

    def test_accepts_three_hour_five_snapshot_schedule(self) -> None:
        history = self.fixture.load_json("run_history.json")
        for sequence in (4, 5):
            round_id = f"auto-{sequence}"
            submission_id = f"submission-{sequence}"
            round_root = self.fixture.task_root / "submissions" / round_id
            round_root.mkdir(parents=True)
            (round_root / "submission.tar.gz").write_bytes(
                f"auto archive {sequence}".encode()
            )
            self.fixture._write_json(
                round_root / "report.json",
                {
                    "task_id": self.fixture.task_id,
                    "submission_id": submission_id,
                    "valid": True,
                    "pass_rate": 1.0,
                    "score": 100.0 + sequence,
                    "score_0_100": 20.0 + sequence,
                },
            )
            history["entries"].append(
                {
                    "type": "submission",
                    "status": "completed",
                    "submission_id": submission_id,
                    "task_id": self.fixture.task_id,
                    "round": round_id,
                }
            )
        history["auto_submissions"] = 5
        self.fixture.write_json("run_history.json", history)
        self.fixture.write_json(
            "final_snapshot.json",
            {
                "captured_at": 1_800_010_800.5,
                "snapshot_elapsed_seconds": 10_800.5,
            },
        )
        self.fixture._write_result(
            runtime_seconds=10_800.25,
            auto_submissions=5,
            total_rounds=5,
        )

        pre = validator.validate_pre_final(
            self.fixture.task_root,
            run_id=self.fixture.run_id,
            task_id=self.fixture.task_id,
            model=self.fixture.model,
            budget_seconds=10_800,
            interval_seconds=1_800,
        )
        self.assertEqual(
            [round_data["round_id"] for round_data in pre["auto_rounds"]],
            ["auto-1", "auto-2", "auto-3", "auto-4", "auto-5"],
        )

    def test_accepts_one_hour_one_snapshot_schedule(self) -> None:
        for sequence in (2, 3):
            shutil.rmtree(
                self.fixture.task_root / "submissions" / f"auto-{sequence}"
            )
        history = self.fixture.load_json("run_history.json")
        history["entries"] = [
            entry
            for entry in history["entries"]
            if entry.get("round") == "auto-1"
        ]
        history["auto_submissions"] = 1
        self.fixture.write_json("run_history.json", history)
        self.fixture.write_json(
            "final_snapshot.json",
            {
                "captured_at": 1_800_003_600.5,
                "snapshot_elapsed_seconds": 3_600.5,
            },
        )
        self.fixture._write_result(
            runtime_seconds=3_600.25,
            auto_submissions=1,
            total_rounds=1,
        )

        pre = validator.validate_pre_final(
            self.fixture.task_root,
            run_id=self.fixture.run_id,
            task_id=self.fixture.task_id,
            model=self.fixture.model,
            budget_seconds=3_600,
            interval_seconds=1_800,
        )
        self.assertEqual(
            [round_data["round_id"] for round_data in pre["auto_rounds"]],
            ["auto-1"],
        )

    def test_rejects_empty_required_artifact(self) -> None:
        (self.fixture.task_root / "final_snapshot.json").write_bytes(b"")
        self.assert_validation_error("final_snapshot.json is empty", self.fixture.pre)

    def test_rejects_symlink_required_artifact(self) -> None:
        snapshot = self.fixture.task_root / "final_snapshot.json"
        target = self.fixture.task_root / "real-final-snapshot.json"
        snapshot.rename(target)
        snapshot.symlink_to(target)
        self.assert_validation_error(
            "symlink path component", self.fixture.pre
        )

    def test_rejects_symlink_parent_component(self) -> None:
        round_root = self.fixture.task_root / "submissions" / "auto-2"
        target = self.fixture.task_root / "submissions" / "real-auto-2"
        round_root.rename(target)
        round_root.symlink_to(target, target_is_directory=True)
        self.assert_validation_error(
            "symlink path component", self.fixture.pre
        )

    def test_rejects_missing_or_extra_auto_round(self) -> None:
        history = self.fixture.load_json("run_history.json")
        history["entries"].pop()
        self.fixture.write_json("run_history.json", history)
        self.assert_validation_error(
            "exactly 3 auto submissions", self.fixture.pre
        )

    def test_rejects_noncompleted_auto_round(self) -> None:
        history = self.fixture.load_json("run_history.json")
        history["entries"][1]["status"] = "running"
        self.fixture.write_json("run_history.json", history)
        self.assert_validation_error("status must be 'completed'", self.fixture.pre)

    def test_rejects_history_report_submission_id_mismatch(self) -> None:
        report = self.fixture.load_json("submissions/auto-2/report.json")
        report["submission_id"] = "different"
        self.fixture.write_json("submissions/auto-2/report.json", report)
        self.assert_validation_error(
            "differs between history and report", self.fixture.pre
        )

    def test_rejects_archive_outside_exact_whitelist(self) -> None:
        self.fixture._write_final_archive(
            [
                ("kernels/flash_kmeans_v002.py", b"# Python\n", tarfile.REGTYPE, ""),
                ("FINAL_VERSION", b"001\n", tarfile.REGTYPE, ""),
                ("notes.md", b"fixture\n", tarfile.REGTYPE, ""),
                ("tests/hidden.py", b"no\n", tarfile.REGTYPE, ""),
            ]
        )
        self.assert_validation_error("outside the whitelist", self.fixture.pre)

    def test_rejects_unsafe_archive_path(self) -> None:
        self.fixture._write_final_archive(
            [
                ("kernels/flash_kmeans_v002.py", b"# Python\n", tarfile.REGTYPE, ""),
                ("kernels/../notes.md", b"escape\n", tarfile.REGTYPE, ""),
                ("FINAL_VERSION", b"001\n", tarfile.REGTYPE, ""),
                ("notes.md", b"fixture\n", tarfile.REGTYPE, ""),
            ]
        )
        self.assert_validation_error("unsafe member", self.fixture.pre)

    def test_rejects_missing_required_archive_member(self) -> None:
        self.fixture._write_final_archive(
            [
                ("kernels/flash_kmeans_v002.py", b"# Python\n", tarfile.REGTYPE, ""),
                ("FINAL_VERSION", b"001\n", tarfile.REGTYPE, ""),
            ]
        )
        self.assert_validation_error(
            "lacks required regular files", self.fixture.pre
        )

    def test_rejects_duplicate_archive_member(self) -> None:
        self.fixture._write_final_archive(
            [
                ("kernels/flash_kmeans_v002.py", b"one\n", tarfile.REGTYPE, ""),
                ("kernels/flash_kmeans_v002.py", b"two\n", tarfile.REGTYPE, ""),
                ("FINAL_VERSION", b"001\n", tarfile.REGTYPE, ""),
                ("notes.md", b"fixture\n", tarfile.REGTYPE, ""),
            ]
        )
        self.assert_validation_error("duplicate member", self.fixture.pre)

    def test_rejects_archive_symlink_and_hardlink(self) -> None:
        for member_type, description in (
            (tarfile.SYMTYPE, "symlink"),
            (tarfile.LNKTYPE, "hardlink"),
        ):
            with self.subTest(member_type=description):
                self.fixture._write_final_archive(
                    [
                        (
                            "kernels/flash_kmeans_v002.py",
                            b"# Python\n",
                            tarfile.REGTYPE,
                            "",
                        ),
                        (
                            "kernels/alias.cu",
                            b"",
                            member_type,
                            "kernels/flash_kmeans_v002.py",
                        ),
                        ("FINAL_VERSION", b"001\n", tarfile.REGTYPE, ""),
                        ("notes.md", b"fixture\n", tarfile.REGTYPE, ""),
                    ]
                )
                self.assert_validation_error(
                    "permits only regular files and directories", self.fixture.pre
                )

    def test_rejects_archive_regular_file_ancestor_conflict(self) -> None:
        self.fixture._write_final_archive(
            [
                ("kernels/tree", b"not a directory\n", tarfile.REGTYPE, ""),
                (
                    "kernels/tree/flash_kmeans_v002.py",
                    b"# Python\n",
                    tarfile.REGTYPE,
                    "",
                ),
                ("FINAL_VERSION", b"001\n", tarfile.REGTYPE, ""),
                ("notes.md", b"fixture\n", tarfile.REGTYPE, ""),
            ]
        )
        self.assert_validation_error(
            "regular-file ancestor conflict", self.fixture.pre
        )

    def test_rejects_archive_resource_limits(self) -> None:
        original_members = validator.MAX_ARCHIVE_MEMBERS
        original_bytes = validator.MAX_ARCHIVE_SIZE_BYTES
        try:
            validator.MAX_ARCHIVE_MEMBERS = 3
            self.assert_validation_error(
                "member-count limit", self.fixture.pre
            )
            validator.MAX_ARCHIVE_MEMBERS = original_members
            validator.MAX_ARCHIVE_SIZE_BYTES = 1
            self.assert_validation_error(
                "byte limit", self.fixture.pre
            )
        finally:
            validator.MAX_ARCHIVE_MEMBERS = original_members
            validator.MAX_ARCHIVE_SIZE_BYTES = original_bytes

    def test_rejects_archive_size_mismatch(self) -> None:
        result = self.fixture.load_json("final_result.json")
        result["archive_size_bytes"] += 1
        self.fixture.write_json("final_result.json", result)
        self.assert_validation_error(
            "size does not match final_result.json", self.fixture.pre
        )

    def test_rejects_invalid_or_timed_out_final_report(self) -> None:
        digest = self.fixture.pre()["final_archive"]["sha256"]
        report_path = self.fixture.final_submission_root / "report.json"
        for field, value, pattern in (
            ("valid", False, "valid=true"),
            ("timed_out", True, "timed_out=false"),
        ):
            with self.subTest(field=field):
                report = json.loads(report_path.read_text(encoding="utf-8"))
                report[field] = value
                FormalArtifactFixture._write_json(report_path, report)
                self.assert_validation_error(
                    pattern, lambda: self.fixture.post(digest)
                )
                report[field] = not value
                FormalArtifactFixture._write_json(report_path, report)

    def test_rejects_missing_or_corrupt_private_judge_artifact(self) -> None:
        digest = self.fixture.pre()["final_archive"]["sha256"]
        artifact = (
            self.fixture.final_submission_root
            / validator.PRIVATE_JUDGE_ARTIFACT_NAME
        )
        original = artifact.read_bytes()
        artifact.unlink()
        self.assert_validation_error(
            "private Judge artifact is missing",
            lambda: self.fixture.post(digest),
        )
        artifact.write_bytes(b"not a tarball")
        self.assert_validation_error(
            "not a readable tar.gz",
            lambda: self.fixture.post(digest),
        )
        artifact.write_bytes(original)

    def test_rejects_private_judge_score_mismatch(self) -> None:
        digest = self.fixture.pre()["final_archive"]["sha256"]
        artifact = (
            self.fixture.final_submission_root
            / validator.PRIVATE_JUDGE_ARTIFACT_NAME
        )
        with tarfile.open(artifact, mode="r:gz") as archive:
            detail = json.load(
                archive.extractfile(validator.PRIVATE_JUDGE_MEMBER_NAME)
            )
        detail["cases"][0]["candidate_ms"] = 600.0
        detail["cases"][0]["candidate_warmed_iteration_ms"] = 600.0
        payload = json.dumps(detail).encode("utf-8")
        with tarfile.open(artifact, mode="w:gz") as archive:
            info = tarfile.TarInfo(validator.PRIVATE_JUDGE_MEMBER_NAME)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
        self.assert_validation_error(
            "recomputed score differs",
            lambda: self.fixture.post(digest),
        )

    def test_rejects_wrong_final_submission_identity(self) -> None:
        digest = self.fixture.pre()["final_archive"]["sha256"]
        report_path = self.fixture.final_submission_root / "report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["submission_id"] = "wrong-final-run"
        FormalArtifactFixture._write_json(report_path, report)
        self.assert_validation_error(
            "submission_id must be", lambda: self.fixture.post(digest)
        )

    def test_accepts_null_display_score_without_rescale(self) -> None:
        digest = self.fixture.pre()["final_archive"]["sha256"]
        report_path = self.fixture.final_submission_root / "report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["score_0_100"] = None
        FormalArtifactFixture._write_json(report_path, report)
        result = self.fixture.post(digest)
        self.assertIsNone(result["final_report"]["score_0_100"])

    def test_rejects_missing_or_forged_adapter_provenance(self) -> None:
        digest = self.fixture.pre()["final_archive"]["sha256"]
        report_path = self.fixture.final_submission_root / "report.json"
        original = json.loads(report_path.read_text(encoding="utf-8"))
        mutations = (
            ("metrics", None, ".metrics must be an object"),
            ("source", "candidate_spoof", ".metrics.source must be"),
            ("adapter_schema_version", True, "adapter_schema_version must be integer 8"),
            ("feedback_schema_version", 6, "feedback_schema_version must be integer 7"),
            (
                "candidate_output_suppressed",
                1,
                "candidate_output_suppressed must be true",
            ),
            ("result_protocol_records", 2, "result_protocol_records must be integer 1"),
            ("verifier_completed", 1, "verifier_completed must be true"),
            ("verifier_returncode", False, "verifier_returncode must be integer 0"),
        )
        for field, value, pattern in mutations:
            with self.subTest(field=field):
                report = json.loads(json.dumps(original))
                if field == "metrics":
                    report["metrics"] = value
                else:
                    report["metrics"][field] = value
                FormalArtifactFixture._write_json(report_path, report)
                self.assert_validation_error(
                    pattern,
                    lambda: self.fixture.post(digest),
                )
        FormalArtifactFixture._write_json(report_path, original)

    def test_rejects_nonfinite_final_score(self) -> None:
        digest = self.fixture.pre()["final_archive"]["sha256"]
        report_path = self.fixture.final_submission_root / "report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["score"] = float("nan")
        FormalArtifactFixture._write_json(report_path, report)
        self.assert_validation_error(
            "non-finite JSON constant", lambda: self.fixture.post(digest)
        )

    def test_rejects_nonpositive_final_score(self) -> None:
        digest = self.fixture.pre()["final_archive"]["sha256"]
        report_path = self.fixture.final_submission_root / "report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["score"] = 0.0
        FormalArtifactFixture._write_json(report_path, report)
        self.assert_validation_error(
            "score must be positive", lambda: self.fixture.post(digest)
        )

    def test_rejects_final_submission_archive_hash_mismatch(self) -> None:
        digest = self.fixture.pre()["final_archive"]["sha256"]
        (self.fixture.final_submission_root / "submission.tar.gz").write_bytes(
            b"different archive"
        )
        self.assert_validation_error(
            "did not use the pre-validated archive",
            lambda: self.fixture.post(digest),
        )

    def test_rejects_final_archive_changed_after_precheck(self) -> None:
        digest = self.fixture.pre()["final_archive"]["sha256"]
        (self.fixture.task_root / "final_archive.tar.gz").write_bytes(
            b"changed after precheck"
        )
        self.assert_validation_error(
            "changed after pre-validation",
            lambda: self.fixture.post(digest),
        )

    def test_postcheck_binds_single_read_report_payload_and_digest(self) -> None:
        digest = self.fixture.pre()["final_archive"]["sha256"]
        report_path = self.fixture.final_submission_root / "report.json"
        original_bytes = report_path.read_bytes()
        post = self.fixture.post(digest)

        report_path.write_text('{"valid": false}', encoding="utf-8")
        self.assertTrue(post["validated_report"]["valid"])
        self.assertEqual(
            post["final_report"]["sha256"],
            hashlib.sha256(original_bytes).hexdigest(),
        )


if __name__ == "__main__":
    unittest.main()
