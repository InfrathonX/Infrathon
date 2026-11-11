from __future__ import annotations

import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = PROJECT_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


collect_curve = _load_script("collect_curve")
validate_layout = _load_script("validate_layout")


class CurveFixture:
    start = 1_800_000_000.0
    task_id = "flash_kmeans_paper_h100_kernel_lab_v7_scratch"

    def __init__(self, root: Path):
        self.run_id_root = root / "sample-run"
        self.task_root = self.run_id_root / self.task_id
        (self.task_root / "submissions").mkdir(parents=True)
        (self.task_root / "started_at").write_text(
            f"2027-01-15T08:00:00\n{self.start}\n", encoding="utf-8"
        )

    def report(
        self,
        round_id: str,
        elapsed: float,
        *,
        score: float | None,
        valid: bool = True,
        pass_rate: float = 1.0,
        score_0_100: float | None = None,
    ) -> Path:
        path = self.task_root / "submissions" / round_id / "report.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "task_id": self.task_id,
            "submission_id": f"id-{round_id}",
            "score": score,
            "score_0_100": score_0_100,
            "valid": valid,
            "pass_rate": pass_rate,
            "submitted_at": self.start + elapsed,
        }
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def all_auto_reports(self) -> None:
        self.report("auto-1", 1_803.25, score=101.0, score_0_100=18.36)
        self.report(
            "auto-2",
            3_607.5,
            score=None,
            valid=False,
            pass_rate=0.25,
            score_0_100=None,
        )
        self.report("auto-3", 5_405.75, score=303.0, score_0_100=55.09)

    def final_report(self, *, name: str = "final-120m-report.json") -> Path:
        path = self.task_root / name
        payload = {
            "task_id": self.task_id,
            "submission_id": "id-final",
            "score": 404.0,
            "score_0_100": 73.45,
            "valid": True,
            "pass_rate": 1.0,
            "submitted_at": self.start + 7_222.125,
        }
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path


class CollectCurveTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.fixture = CurveFixture(self.root)
        self.fixture.all_auto_reports()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_collects_raw_fixed_snapshots_and_real_elapsed_times(self) -> None:
        final = self.fixture.final_report()
        curve = collect_curve.collect_curve(self.fixture.task_root, final)

        self.assertEqual(curve["task_id"], self.fixture.task_id)
        self.assertIn("not cumulative", curve["point_semantics"])
        self.assertEqual(
            [point["label"] for point in curve["points"]],
            ["30m", "60m", "90m", "120m"],
        )
        self.assertEqual(
            [point["kind"] for point in curve["points"]],
            [
                "auto_snapshot",
                "auto_snapshot",
                "auto_snapshot",
                "independent_final_eval",
            ],
        )
        self.assertEqual(
            [point["scheduled_elapsed_seconds"] for point in curve["points"]],
            [1800, 3600, 5400, 7200],
        )
        self.assertAlmostEqual(curve["points"][0]["actual_elapsed_seconds"], 1803.25)
        self.assertAlmostEqual(curve["points"][3]["actual_elapsed_seconds"], 7222.125)
        self.assertEqual(curve["points"][0]["raw_score"], 101.0)
        self.assertEqual(curve["points"][0]["score_0_100"], 18.36)
        self.assertIsNone(curve["points"][1]["raw_score"])
        self.assertFalse(curve["points"][1]["valid"])
        self.assertEqual(curve["points"][1]["pass_rate"], 0.25)
        self.assertEqual(
            curve["points"][3]["timestamp_source"], "report.submitted_at"
        )

        markdown = collect_curve.render_markdown(curve)
        self.assertIn("raw fixed-time snapshots", markdown)
        self.assertIn("separate judge evaluation", markdown)
        self.assertIn("| 120m | 120.37m |", markdown)

    def test_collects_dynamic_three_hour_trajectory(self) -> None:
        self.fixture.report("auto-4", 7_205.0, score=350.0)
        self.fixture.report("auto-5", 9_005.0, score=375.0)
        final = self.fixture.task_root / "final-180m-report.json"
        final.write_text(
            json.dumps(
                {
                    "task_id": self.fixture.task_id,
                    "submission_id": "id-final-180m",
                    "score": 410.0,
                    "score_0_100": 74.55,
                    "valid": True,
                    "pass_rate": 1.0,
                    "submitted_at": self.fixture.start + 10_822.0,
                }
            ),
            encoding="utf-8",
        )
        snapshots = tuple(
            (sequence, minutes)
            for sequence, minutes in enumerate(
                range(30, 180, 30), start=1
            )
        )
        curve = collect_curve.collect_curve(
            self.fixture.task_root,
            final,
            auto_snapshots=snapshots,
            final_minutes=180,
        )

        self.assertEqual(
            [point["label"] for point in curve["points"]],
            ["30m", "60m", "90m", "120m", "150m", "180m"],
        )
        self.assertEqual(curve["points"][-1]["round_id"], "final-180m")
        self.assertIn(
            "180m row is a separate judge evaluation",
            collect_curve.render_markdown(curve),
        )

    def test_accepts_run_id_parent_and_discovers_default_final_report(self) -> None:
        self.fixture.final_report()
        curve = collect_curve.collect_curve(self.fixture.run_id_root)
        self.assertEqual(Path(curve["task_run_dir"]), self.fixture.task_root.resolve())
        self.assertEqual(
            curve["points"][-1]["source_report"], "flash_kmeans_paper_h100_kernel_lab_v7_scratch/final-120m-report.json"
        )

    def test_extracts_report_from_sforge_eval_stdout_log(self) -> None:
        report = {
            "task_id": self.fixture.task_id,
            "submission_id": "log-final",
            "score": 499.5,
            "score_0_100": 90.8,
            "valid": True,
            "pass_rate": 1.0,
            "submitted_at": self.fixture.start + 7_240,
            "metrics": {"inner": 1},
        }
        log = self.fixture.task_root / "final-120m-eval.log"
        log.write_text(
            "Evaluating archive for task: flash_kmeans_paper_h100_kernel_lab_v7_scratch\n"
            "Results:\n"
            + json.dumps(report, indent=2)
            + "\n",
            encoding="utf-8",
        )

        curve = collect_curve.collect_curve(self.fixture.task_root)
        self.assertEqual(curve["points"][-1]["raw_score"], 499.5)
        self.assertEqual(curve["points"][-1]["submission_id"], "log-final")
        self.assertEqual(
            curve["points"][-1]["source_report"], "final-120m-eval.log"
        )

    def test_wrapper_can_record_explicit_final_elapsed(self) -> None:
        final = self.fixture.task_root / "final-120m-report.json"
        final.write_text(
            json.dumps(
                {
                    "snapshot_elapsed_seconds": 7_211.5,
                    "report": {
                        "task_id": self.fixture.task_id,
                        "submission_id": "wrapped-final",
                        "score": 450,
                        "valid": True,
                        "pass_rate": 1.0,
                    },
                }
            ),
            encoding="utf-8",
        )
        curve = collect_curve.collect_curve(self.fixture.task_root, final)
        point = curve["points"][-1]
        self.assertEqual(point["actual_elapsed_seconds"], 7_211.5)
        self.assertEqual(point["elapsed_source"], "wrapper.snapshot_elapsed_seconds")
        self.assertEqual(
            point["timestamp_epoch"], self.fixture.start + 7_211.5
        )
        self.assertEqual(
            point["timestamp_source"],
            "started_at + wrapper.snapshot_elapsed_seconds",
        )

    def test_single_gpu_final_is_scheduled_against_its_seal(self) -> None:
        """Seen live (fk-single-1h-0912): a Judge queued just before the
        deadline was drained after the seal, so the frozen snapshot landed
        +105 s from the target while the agent's work boundary was on time."""
        final = self.fixture.task_root / "final-120m-report.json"
        payload = {
            "snapshot_elapsed_seconds": 7_305.1,
            "seal_requested_elapsed_seconds": 7_200.05,
            "report": {
                "task_id": self.fixture.task_id,
                "submission_id": "sealed-final",
                "score": 452,
                "valid": True,
                "pass_rate": 1.0,
            },
        }
        final.write_text(json.dumps(payload), encoding="utf-8")
        curve = collect_curve.collect_curve(self.fixture.task_root, final)
        point = curve["points"][-1]
        self.assertEqual(point["actual_elapsed_seconds"], 7_305.1)
        self.assertEqual(point["seal_elapsed_seconds"], 7_200.05)
        self.assertEqual(point["schedule_anchor"], "seal")
        self.assertAlmostEqual(point["schedule_delta_seconds"], 0.05)
        # Without the seal the same snapshot is still rejected.
        payload.pop("seal_requested_elapsed_seconds")
        final.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(collect_curve.CurveError, r"snapshot is \+105\.1s"):
            collect_curve.collect_curve(self.fixture.task_root, final)
        # The seal itself must be on schedule, and the freeze may not precede it.
        payload["seal_requested_elapsed_seconds"] = 7_100.0
        final.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(collect_curve.CurveError, r"seal is -100\.0s"):
            collect_curve.collect_curve(self.fixture.task_root, final)
        payload["seal_requested_elapsed_seconds"] = 7_400.0
        final.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(collect_curve.CurveError, r"precedes its seal"):
            collect_curve.collect_curve(self.fixture.task_root, final)

    def test_wrapper_snapshot_time_wins_over_late_eval_timestamp(self) -> None:
        final = self.fixture.task_root / "final-120m-report.json"
        final.write_text(
            json.dumps(
                {
                    "snapshot_elapsed_seconds": 7_205.0,
                    "snapshot_captured_at": self.fixture.start + 7_205.0,
                    "report": {
                        "task_id": self.fixture.task_id,
                        "submission_id": "late-eval",
                        "score": 451,
                        "valid": True,
                        "pass_rate": 1.0,
                        "submitted_at": self.fixture.start + 8_000,
                    },
                }
            ),
            encoding="utf-8",
        )
        curve = collect_curve.collect_curve(self.fixture.task_root, final)
        point = curve["points"][-1]
        self.assertEqual(point["timestamp_epoch"], self.fixture.start + 7_205.0)
        self.assertEqual(
            point["timestamp_source"], "wrapper.snapshot_captured_at"
        )

    def test_rejects_final_result_even_if_it_looks_like_a_report(self) -> None:
        final = self.fixture.final_report(name="final_result.json")
        with self.assertRaisesRegex(
            collect_curve.CurveError, "best-so-far run summary"
        ):
            collect_curve.collect_curve(self.fixture.task_root, final)

    def test_rejects_an_auto_report_as_the_final_point(self) -> None:
        final = (
            self.fixture.task_root
            / "submissions"
            / "auto-4"
            / "report.json"
        )
        final.parent.mkdir()
        final.write_text(
            json.dumps(
                {
                    "task_id": self.fixture.task_id,
                    "score": 500,
                    "valid": True,
                    "pass_rate": 1,
                    "submitted_at": self.fixture.start + 7200,
                }
            ),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(collect_curve.CurveError, "auto-eval report"):
            collect_curve.collect_curve(self.fixture.task_root, final)

    def test_missing_required_auto_snapshot_is_an_error(self) -> None:
        shutil.rmtree(self.fixture.task_root / "submissions" / "auto-2")
        final = self.fixture.final_report()
        with self.assertRaisesRegex(collect_curve.CurveError, "missing required 60m"):
            collect_curve.collect_curve(self.fixture.task_root, final)

    def test_rejects_snapshot_far_from_scheduled_boundary(self) -> None:
        self.fixture.report("auto-2", 3_300, score=202.0)
        final = self.fixture.final_report()
        with self.assertRaisesRegex(collect_curve.CurveError, "60m target"):
            collect_curve.collect_curve(self.fixture.task_root, final)

    def test_rejects_early_final_even_with_a_valid_report(self) -> None:
        final = self.fixture.task_root / "final-120m-report.json"
        final.write_text(
            json.dumps(
                {
                    "snapshot_elapsed_seconds": 7_000,
                    "report": {
                        "task_id": self.fixture.task_id,
                        "submission_id": "too-early",
                        "score": 450,
                        "valid": True,
                        "pass_rate": 1.0,
                    },
                }
            ),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(collect_curve.CurveError, "120m target"):
            collect_curve.collect_curve(self.fixture.task_root, final)

    def test_auto_tick_timestamp_wins_over_delayed_judge_start(self) -> None:
        (self.fixture.task_root / "auto_eval_ticks.log").write_text(
            "[2027-01-15T08:30:00] submitted 10 bytes -> abc round=auto-1\n",
            encoding="utf-8",
        )
        final = self.fixture.final_report()
        curve = collect_curve.collect_curve(self.fixture.task_root, final)
        point = curve["points"][0]
        self.assertEqual(point["timestamp_source"], "auto_eval_ticks.log")
        self.assertAlmostEqual(point["actual_elapsed_seconds"], 1800.0)

    def test_write_curve_outputs_json_and_markdown(self) -> None:
        final = self.fixture.final_report()
        curve = collect_curve.collect_curve(self.fixture.task_root, final)
        output = self.root / "curve"
        json_path, markdown_path = collect_curve.write_curve(curve, output)
        saved = json.loads(json_path.read_text(encoding="utf-8"))
        self.assertEqual(saved["points"][3]["round_id"], "final-120m")
        self.assertTrue(markdown_path.read_text(encoding="utf-8").endswith("\n"))


class ValidateLayoutTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name) / "migration"
        self.root.mkdir()
        for directory in (
            "images",
            "judge",
            "host_reference",
            "scripts",
            "tasks",
        ):
            source = PROJECT_ROOT / directory
            if source.is_dir():
                shutil.copytree(source, self.root / directory)
        # Maintainer run configurations; the public tree does not ship them.
        for path in PROJECT_ROOT.glob("experiment-*.yaml"):
            shutil.copy2(path, self.root / path.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _two_hour_experiment(self) -> Path:
        """A maintainer 2-hour run profile with its run wrapper, if present."""
        wrappers = validate_layout.run_wrappers(self.root)
        for path in sorted(self.root.glob("experiment-*-2h.yaml")):
            runner = "run_" + path.stem[len("experiment-"):].replace("-", "_") + ".sh"
            if runner in wrappers and "timeout: 7200" in path.read_text(encoding="utf-8"):
                return path
        self.skipTest("maintainer experiment configuration is not shipped")

    def _detached_launcher(self) -> Path:
        """A maintainer tmux launcher of a run wrapper, if present."""
        wrappers = validate_layout.run_wrappers(self.root)
        for path in sorted((self.root / "scripts").glob("launch_*_tmux.sh")):
            if "run_" + path.name[len("launch_"):-len("_tmux.sh")] + ".sh" in wrappers:
                return path
        self.skipTest("maintainer tmux launchers are not shipped")

    def test_current_layout_passes(self) -> None:
        self.assertEqual(validate_layout.validate_layout(self.root), [])

    def test_task_submit_paths_are_an_exact_whitelist(self) -> None:
        path = self.root / "tasks/flash_kmeans_paper_h100_kernel_lab_v7_scratch.json"
        task = json.loads(path.read_text(encoding="utf-8"))
        task["submit_paths"] = ["."]
        path.write_text(json.dumps(task), encoding="utf-8")
        errors = validate_layout.validate_layout(self.root)
        self.assertTrue(any("exact whitelist" in error for error in errors), errors)
        self.assertTrue(any("unsafe submit path" in error for error in errors), errors)

    def test_candidate_latency_contract_rejects_rescale_anchors(self) -> None:
        path = self.root / "tasks/flash_kmeans_paper_h100_kernel_lab_v7_scratch.json"
        task = json.loads(path.read_text(encoding="utf-8"))
        task["judge"]["rescale"] = {
            "kind": "linear",
            "lower": 0.0,
            "upper": 150.0,
        }
        task["judge"]["rescale_calibration"] = "done"
        path.write_text(json.dumps(task), encoding="utf-8")
        errors = validate_layout.validate_layout(self.root)
        self.assertTrue(
            any("must not use a baseline rescale" in error for error in errors)
        )

    def test_modelscope_base_image_cannot_drift_from_work_dockerfile(self) -> None:
        path = self.root / "tasks/BENCHMARK.yaml"
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                "pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel",
                "wrong-image-tag",
            ),
            encoding="utf-8",
        )
        errors = validate_layout.validate_layout(self.root)
        self.assertTrue(any("official_image must be" in error for error in errors), errors)

    def test_experiment_task_id_and_timing_must_match(self) -> None:
        path = self._two_hour_experiment()
        text = path.read_text(encoding="utf-8")
        text = text.replace("timeout: 7200", "timeout: 7100")
        text = text.replace(
            "flash_kmeans_paper_h100_kernel_lab_v7_scratch: {}", "flash-kmeans-h100-kernel-lab: {}"
        )
        path.write_text(text, encoding="utf-8")
        errors = validate_layout.validate_layout(self.root)
        self.assertTrue(any("tasks must contain exactly" in error for error in errors), errors)
        self.assertTrue(any("effective timeout must be 7200" in error for error in errors), errors)

    def test_experiment_codex_cli_version_must_match(self) -> None:
        path = self._two_hour_experiment()
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                'SFORGE_CODEX_CLI_VERSION: "0.148.0"',
                'SFORGE_CODEX_CLI_VERSION: "0.999.0"',
            ),
            encoding="utf-8",
        )
        errors = validate_layout.validate_layout(self.root)
        self.assertTrue(
            any("SFORGE_CODEX_CLI_VERSION must be" in error for error in errors),
            errors,
        )

    def test_renamed_private_file_in_work_context_is_detected_by_hash(self) -> None:
        shutil.copyfile(
            self.root / "images/judge/verify_flash_kmeans.py",
            self.root / "images/work/public_tests/innocent_helper.py",
        )
        errors = validate_layout.validate_layout(self.root)
        self.assertTrue(any("duplicates private asset" in error for error in errors), errors)

    def test_protected_manifest_rejects_hash_drift(self) -> None:
        protected = self.root / "images/judge/benchmark/kernel_bench.py"
        protected.write_text(
            protected.read_text(encoding="utf-8") + "\n# drift\n",
            encoding="utf-8",
        )
        errors = validate_layout.validate_layout(self.root)
        self.assertTrue(any("hash drift" in error for error in errors), errors)

    def test_protected_manifest_requires_every_critical_runtime_path(self) -> None:
        manifest = self.root / "images/judge/protected_hashes.txt"
        entries = [
            line
            for line in manifest.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ][:4]
        manifest.write_text("\n".join(entries) + "\n", encoding="utf-8")
        errors = validate_layout.validate_layout(self.root)
        self.assertTrue(
            any("is missing required paths" in error for error in errors),
            errors,
        )

    def test_judge_adapter_mirror_must_match_canonical_source(self) -> None:
        mirror = self.root / "images/judge/run_judge.py"
        mirror.write_text(
            mirror.read_text(encoding="utf-8") + "\n# stale mirror\n",
            encoding="utf-8",
        )
        errors = validate_layout.validate_layout(self.root)
        self.assertTrue(
            any("judge adapter mirror differs" in error for error in errors),
            errors,
        )

    def test_auth_material_in_work_context_is_rejected(self) -> None:
        auth = self.root / "images/work/public_tests/session.json"
        auth.write_text(
            json.dumps({"access_token": "not-a-real-token"}), encoding="utf-8"
        )
        errors = validate_layout.validate_layout(self.root)
        self.assertTrue(
            any("embedded authentication material" in error for error in errors),
            errors,
        )

    def test_image_name_drift_is_rejected(self) -> None:
        path = self.root / "scripts/lib.sh"
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                'WORK_IMAGE="flash_kmeans.work.${TASK_ID}:local"',
                'WORK_IMAGE="wrong.work.${TASK_ID}:local"',
            ),
            encoding="utf-8",
        )
        errors = validate_layout.validate_layout(self.root)
        self.assertTrue(any("WORK_IMAGE must be" in error for error in errors), errors)

    def test_docker_context_allowlist_is_required(self) -> None:
        (self.root / "images/judge/.dockerignore").unlink()
        errors = validate_layout.validate_layout(self.root)
        self.assertTrue(
            any("build-context allowlist" in error for error in errors),
            errors,
        )

    def test_docker_context_cannot_reinclude_generated_bytecode(self) -> None:
        path = self.root / "images/judge/.dockerignore"
        text = path.read_text(encoding="utf-8")
        text = text.replace(
            "**/__pycache__/\n**/*.pyc\n",
            "",
        )
        text = text.replace(
            "!benchmark/**\n",
            "**/__pycache__/\n**/*.pyc\n!benchmark/**\n",
        )
        path.write_text(text, encoding="utf-8")
        errors = validate_layout.validate_layout(self.root)
        self.assertTrue(
            any(
                "must follow every re-include" in error
                for error in errors
            ),
            errors,
        )

    def test_layout_rejects_tmux_pipeline_without_pipefail(self) -> None:
        path = self._detached_launcher()
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                "bash -o pipefail -c", "bash -c"
            ),
            encoding="utf-8",
        )
        errors = validate_layout.validate_layout(self.root)
        self.assertTrue(
            any(
                "detached-run status fragment" in error
                and "pipefail" in error
                for error in errors
            ),
            errors,
        )

    def test_layout_requires_run_id_validation_before_detached_paths(self) -> None:
        path = self._detached_launcher()
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                '[[ "${run_id}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]',
                '[[ -n "${run_id}" ]]',
            ),
            encoding="utf-8",
        )
        errors = validate_layout.validate_layout(self.root)
        self.assertTrue(
            any(
                "detached-run status fragment" in error
                and "run_id" in error
                for error in errors
            ),
            errors,
        )

    def test_layout_rejects_tmux_launcher_without_exit_artifact(self) -> None:
        path = self._detached_launcher()
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                "launcher_exit=", "discarded_exit="
            ),
            encoding="utf-8",
        )
        errors = validate_layout.validate_layout(self.root)
        self.assertTrue(
            any(
                "detached-run status fragment" in error
                and "launcher_exit" in error
                for error in errors
            ),
            errors,
        )

    def test_layout_requires_formal_artifact_validator_in_launcher(self) -> None:
        path = self.root / "scripts/run_fixed_budget.sh"
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                'validate_formal_artifacts.py" pre',
                'missing_formal_validator.py" pre',
            ),
            encoding="utf-8",
        )
        errors = validate_layout.validate_layout(self.root)
        self.assertTrue(
            any(
                "missing required fixed-run fragment" in error
                and "validate_formal_artifacts.py" in error
                for error in errors
            ),
            errors,
        )

    def test_layout_requires_formal_artifact_validator_module(self) -> None:
        (self.root / "scripts/validate_formal_artifacts.py").unlink()
        errors = validate_layout.validate_layout(self.root)
        self.assertTrue(
            any(
                "validate_formal_artifacts.py" in error
                and "cannot read" in error
                for error in errors
            ),
            errors,
        )


if __name__ == "__main__":
    unittest.main()
