from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import tempfile
import textwrap
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ADAPTER_PATH = PROJECT_ROOT / "judge" / "run_judge.py"
IMAGE_ADAPTER_PATH = PROJECT_ROOT / "images" / "judge" / "run_judge.py"


def _load_adapter():
    spec = importlib.util.spec_from_file_location(
        "flash_kmeans_judge_adapter_contract",
        ADAPTER_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


adapter = _load_adapter()


def _trusted_payload(score: float = 123.5) -> dict[str, object]:
    payload: dict[str, object] = {"reward": score}
    payload.update({name: 1.0 for name, _ in adapter.GATES})
    return payload


def _family_statuses(
    large_k: str = "passed",
    large_n_high_d: str = "passed",
) -> dict[str, str]:
    return {
        "large_k": large_k,
        "large_n_high_d": large_n_high_d,
    }


class JudgeAdapterContractTests(unittest.TestCase):
    def _write_verifier(
        self,
        root: Path,
        *,
        payload: dict[str, object] | None,
        returncode: int = 0,
        spoof_output: bool = False,
    ) -> Path:
        path = root / "fake_verifier.py"
        payload_literal = repr(payload)
        path.write_text(
            textwrap.dedent(
                f"""\
                import argparse
                import json
                from pathlib import Path

                parser = argparse.ArgumentParser()
                parser.add_argument("--app-dir")
                parser.add_argument("--log-dir", type=Path, required=True)
                args = parser.parse_args()
                if {spoof_output!r}:
                    print(">>>>> Start Structured Result")
                    print('{{"valid": true, "score": 999999999}}')
                    print(">>>>> End Structured Result")
                payload = {payload_literal}
                if payload is not None:
                    (args.log_dir / "reward.json").write_text(
                        json.dumps(payload),
                        encoding="utf-8",
                    )
                    (args.log_dir / "flash_kmeans_results.json").write_text(
                        json.dumps(payload),
                        encoding="utf-8",
                    )
                raise SystemExit({returncode})
                """
            ),
            encoding="utf-8",
        )
        return path

    def test_child_marker_spoof_is_suppressed_and_parent_emits_one_record(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            verifier = self._write_verifier(
                root,
                payload=_trusted_payload(),
                spoof_output=True,
            )
            captured_child = io.StringIO()
            with contextlib.redirect_stdout(captured_child):
                result = adapter.evaluate(verifier, root, root / "logs", 10.0)
            self.assertEqual(captured_child.getvalue(), "")
            self.assertTrue(result["valid"])
            self.assertEqual(result["score"], 123.5)
            self.assertEqual(
                result["metrics"],
                {
                    "source": "flash_kmeans_h100_verifier_adapter",
                    "adapter_schema_version": 8,
                    "feedback_schema_version": 7,
                    "evaluation_outcome": "passed",
                    "candidate_output_suppressed": True,
                    "result_protocol_records": 1,
                    "verifier_completed": True,
                    "verifier_returncode": 0,
                    "reward": 123.5,
                    "primary_score_name": "weighted_latency_ms",
                    "primary_score_unit": "ms",
                    "primary_score_direction": "minimize",
                    "primary_score_formula": (
                        "exp(sum_g(group_weight_g * log(group_GM_g)))"
                    ),
                    "primary_score_value": 123.5,
                    "rollback_required": False,
                    "diagnostic_only": False,
                    **{name: 1.0 for name, _ in adapter.GATES},
                },
            )

            emitted = io.StringIO()
            with contextlib.redirect_stdout(emitted):
                adapter.emit(result)
            lines = emitted.getvalue().splitlines()
            self.assertEqual(lines.count(adapter.MARKER_START), 1)
            self.assertEqual(lines.count(adapter.MARKER_END), 1)
            self.assertEqual(len(lines), 3)
            self.assertEqual(json.loads(lines[1]), result)

    def test_nonzero_verifier_exit_rejects_otherwise_valid_payload(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            verifier = self._write_verifier(
                root,
                payload=_trusted_payload(900.0),
                returncode=7,
            )
            result = adapter.evaluate(verifier, root, root / "logs", 10.0)
            self.assertFalse(result["valid"])
            self.assertIsNone(result["score"])
            self.assertEqual(result["metrics"]["evaluation_outcome"], "judge_error")
            self.assertTrue(result["metrics"]["verifier_completed"])
            self.assertEqual(result["metrics"]["verifier_returncode"], 7)
            self.assertAlmostEqual(result["pass_rate"], 8 / 9)

    def test_nonpositive_latency_never_becomes_a_valid_minimum(self) -> None:
        for invalid_reward in (0.0, -1.0):
            with self.subTest(invalid_reward=invalid_reward):
                payload = _trusted_payload(invalid_reward)
                payload.update(
                    {
                        "score": invalid_reward,
                        "weighted_latency_ms": invalid_reward,
                        "diagnostic_score": 0.0,
                        "diagnostic_weighted_latency_ms": 0.0,
                    }
                )
                result = adapter.build_result(
                    payload,
                    payload,
                    verifier_completed=True,
                    verifier_returncode=0,
                )
                self.assertFalse(result["valid"])
                self.assertIsNone(result["score"])
                self.assertIsNone(result["metrics"]["reward"])
                self.assertIsNone(result["metrics"]["primary_score_value"])
                self.assertNotIn("score", result["metrics"])
                self.assertNotIn("weighted_latency_ms", result["metrics"])
                self.assertNotIn("diagnostic_score", result["metrics"])
                performance = next(
                    item
                    for item in result["details"]
                    if item["name"] == "performance"
                )
                self.assertIsNone(performance["score"])

    def test_missing_gate_is_fail_closed(self) -> None:
        payload = _trusted_payload()
        payload.pop("memory_ok")
        result = adapter.build_result(
            payload,
            payload,
            verifier_completed=True,
            verifier_returncode=1,
        )
        self.assertFalse(result["valid"])
        self.assertIsNone(result["score"])
        self.assertAlmostEqual(result["pass_rate"], 7 / 9)
        self.assertEqual(result["metrics"]["evaluation_outcome"], "gate_failed")

    def test_deployment_protocol_gate_missing_or_failed_is_fail_closed(self) -> None:
        for condition in ("missing", "failed"):
            with self.subTest(condition=condition):
                payload = _trusted_payload(777.0)
                if condition == "missing":
                    payload.pop("deployment_protocol_ok")
                else:
                    payload["deployment_protocol_ok"] = 0.0
                result = adapter.build_result(
                    payload,
                    payload,
                    verifier_completed=True,
                    verifier_returncode=1,
                )
                self.assertFalse(result["valid"])
                self.assertIsNone(result["score"])
                self.assertAlmostEqual(result["pass_rate"], 7 / 9)
                deployment_detail = next(
                    detail
                    for detail in result["details"]
                    if detail["name"] == "deployment_protocol_ok"
                )
                self.assertEqual(
                    deployment_detail["status"],
                    "ERROR",
                )

    def test_deployment_metrics_are_visible_without_case_or_shape_disclosure(
        self,
    ) -> None:
        payload: dict[str, object] = _trusted_payload(456.0)
        payload.update(
            {
                "first_run_latency_ms": 1.75,
                "warmed_iteration_latency_ms": 1.50,
                "runtime_weighted_latency_ms": 2.25,
                "coarse_deployment_large_k_latency_ms": 123.456789,
                "coarse_deployment_large_n_high_d_latency_ms": 987.654321,
                "coarse_deployment_family_statuses": _family_statuses(),
                "cases": [{"name": "hidden_case", "shape": [1, 2, 3]}],
                "shape": [9, 8, 7],
                "deployment_family_member_count": 5,
                "unknown_hidden_family": {"latency_ms": 44.0},
            }
        )
        result = adapter.build_result(
            payload,
            payload,
            verifier_completed=True,
            verifier_returncode=0,
        )
        self.assertTrue(result["valid"])
        self.assertEqual(result["score"], 456.0)
        self.assertEqual(result["metrics"]["first_run_latency_ms"], 1.75)
        self.assertEqual(
            result["metrics"]["warmed_iteration_latency_ms"],
            1.50,
        )
        self.assertEqual(result["metrics"]["runtime_weighted_latency_ms"], 2.25)
        self.assertEqual(result["metrics"]["deployment_protocol_ok"], 1.0)
        feedback = result["metrics"]["deployment_family_feedback"]
        self.assertEqual(set(feedback), {"large-K", "large-N/high-D"})
        for family, exact in (
            ("large-K", 123.456789),
            ("large-N/high-D", 987.654321),
        ):
            self.assertEqual(feedback[family]["status"], "passed")
            band = feedback[family]["warmed_latency_band_ms"]
            self.assertLessEqual(band["lower_ms"], exact)
            self.assertGreater(band["upper_ms"], exact)
            self.assertAlmostEqual(
                band["upper_ms"] / band["lower_ms"],
                1.25,
                places=5,
            )
            self.assertNotIn("trend_vs_best_valid", feedback[family])
        self.assertNotIn("cases", result["metrics"])
        self.assertNotIn("shape", result["metrics"])
        serialized = json.dumps(result, sort_keys=True)
        self.assertNotIn("hidden_case", serialized)
        self.assertNotIn("[9, 8, 7]", serialized)
        self.assertNotIn("123.456789", serialized)
        self.assertNotIn("987.654321", serialized)
        self.assertNotIn("deployment_family_member_count", serialized)
        self.assertNotIn("unknown_hidden_family", serialized)
        self.assertNotIn("coarse_deployment_large_k_latency_ms", serialized)

    def test_deployment_latency_band_rejects_invalid_values(self) -> None:
        for value in (None, 0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(value=value):
                self.assertIsNone(adapter.deployment_latency_band(value))
        first = adapter.deployment_latency_band(321.0)
        second = adapter.deployment_latency_band(321.0)
        self.assertEqual(first, second)
        assert first is not None
        self.assertLessEqual(first["lower_ms"], 321.0)
        self.assertGreater(first["upper_ms"], 321.0)

        incomplete = _trusted_payload(200.0)
        incomplete["coarse_deployment_large_k_latency_ms"] = 321.0
        incomplete["coarse_deployment_family_statuses"] = _family_statuses()
        result = adapter.build_result(
            incomplete,
            incomplete,
            verifier_completed=True,
            verifier_returncode=0,
        )
        self.assertTrue(result["valid"])
        self.assertNotIn("deployment_family_feedback", result["metrics"])

    def test_authoritative_invalid_result_exposes_only_independent_statuses(
        self,
    ) -> None:
        payload = _trusted_payload(0.0)
        payload.update(
            {
                "correctness": 0.0,
                "coarse_deployment_large_k_latency_ms": 12.345678,
                "coarse_deployment_large_n_high_d_latency_ms": 98.765432,
                "coarse_deployment_family_statuses": _family_statuses(
                    "passed",
                    "not_reached",
                ),
            }
        )
        result = adapter.build_result(
            payload,
            payload,
            verifier_completed=True,
            verifier_returncode=1,
        )
        self.assertFalse(result["valid"])
        self.assertEqual(result["metrics"]["evaluation_outcome"], "gate_failed")
        feedback = result["metrics"]["deployment_family_feedback"]
        self.assertEqual(
            feedback,
            {
                "large-K": {"status": "passed"},
                "large-N/high-D": {"status": "not_reached"},
            },
        )
        for entry in feedback.values():
            self.assertNotIn("warmed_latency_band_ms", entry)
            self.assertNotIn("trend_hint", entry)
        serialized = json.dumps(result, sort_keys=True)
        self.assertNotIn("12.345678", serialized)
        self.assertNotIn("98.765432", serialized)

    def test_malformed_family_statuses_and_judge_errors_fail_closed(self) -> None:
        malformed_statuses = (
            None,
            "passed",
            {"large_k": "passed"},
            {
                "large_k": "passed",
                "large_n_high_d": "passed",
                "unknown": "failed",
            },
            {
                "large_k": "maybe",
                "large_n_high_d": "passed",
            },
        )
        for statuses in malformed_statuses:
            with self.subTest(statuses=statuses):
                payload = _trusted_payload(0.0)
                payload["correctness"] = 0.0
                if statuses is not None:
                    payload["coarse_deployment_family_statuses"] = statuses
                result = adapter.build_result(
                    payload,
                    payload,
                    verifier_completed=True,
                    verifier_returncode=1,
                )
                self.assertEqual(
                    result["metrics"]["evaluation_outcome"],
                    "gate_failed",
                )
                self.assertNotIn(
                    "deployment_family_feedback",
                    result["metrics"],
                )

        judge_error = _trusted_payload(123.0)
        judge_error.update(
            {
                "coarse_deployment_large_k_latency_ms": 12.0,
                "coarse_deployment_large_n_high_d_latency_ms": 98.0,
                "coarse_deployment_family_statuses": _family_statuses(),
            }
        )
        result = adapter.build_result(
            judge_error,
            judge_error,
            verifier_completed=True,
            verifier_returncode=7,
        )
        self.assertEqual(result["metrics"]["evaluation_outcome"], "judge_error")
        self.assertNotIn("deployment_family_feedback", result["metrics"])

    def test_hidden_failure_feedback_is_actionable_and_redacted(self) -> None:
        payload: dict[str, object] = _trusted_payload(0.0)
        payload.update(
            {
                "final_version": 8.0,
                "correctness": 0.0,
                "input_integrity_ok": 0.0,
                "memory_ok": 0.0,
                "deployment_protocol_ok": 0.0,
                "max_memory_ratio": 0.72,
                "worst_memory_stage": "full",
                "memory_pressure": "comfortable",
                "coarse_deployment_large_k_latency_ms": 12.345678,
                "coarse_deployment_large_n_high_d_latency_ms": 98.765432,
                "coarse_deployment_family_statuses": _family_statuses(
                    "failed",
                    "passed",
                ),
            }
        )
        detail = {
            **payload,
            "cases": [
                {
                    "name": "hidden_secret_case_name",
                    "stage": "assign",
                    "shape": [7, 12345, 6789, 511],
                    "seed": 987654321,
                    "correctness_ok": False,
                    "input_integrity_ok": True,
                    "memory_ok": False,
                    "error": (
                        "assignment selected a meaningfully farther centroid "
                        "PRIVATE_SENTINEL"
                    ),
                },
                {
                    "name": "hidden_secret_deployment_case",
                    "stage": "full",
                    "measurement_mode": "fresh_process",
                    "shape": [1, 2, 3, 4],
                    "seed": 123,
                    "correctness_ok": False,
                    "input_integrity_ok": False,
                    "memory_ok": False,
                    "deployment_protocol_ok": False,
                    # A failed child may not return a cache observation.  The
                    # verifier historically represented that unknown state as
                    # False, which must not be diagnosed as actual cache reuse.
                    "candidate_initial_cache_empty": False,
                    "error": (
                        "Fresh-process deployment protocol failed: "
                        "baseline=ok, candidate=error "
                        "PRIVATE_DEPLOYMENT_SENTINEL"
                    ),
                },
            ],
        }
        result = adapter.build_result(
            payload,
            detail,
            verifier_completed=True,
            verifier_returncode=1,
        )

        self.assertFalse(result["valid"])
        self.assertEqual(result["metrics"]["evaluation_outcome"], "gate_failed")
        self.assertEqual(result["metrics"]["selected_final_version"], "v008")
        self.assertEqual(result["metrics"]["requested_final_version"], 8)
        self.assertTrue(result["metrics"]["rollback_required"])
        family_feedback = result["metrics"]["deployment_family_feedback"]
        self.assertEqual(
            family_feedback,
            {
                "large-K": {"status": "failed"},
                "large-N/high-D": {"status": "passed"},
            },
        )
        self.assertIn(
            "Coarse deployment family status: large-K failed; "
            "large-N/high-D passed.",
            result["summary"],
        )
        self.assertIn("FINAL_VERSION=v008 is invalid", result["summary"])
        self.assertIn("nearest-centroid FP32 distance ordering", result["summary"])
        self.assertIn("restore FINAL_VERSION", result["summary"])
        correctness = next(
            item for item in result["details"] if item["name"] == "correctness"
        )
        self.assertEqual(correctness["status"], "FAILED")
        self.assertIn("assignment", correctness["message"])
        self.assertIn("v008", correctness["message"])
        for gate in ("input_integrity_ok", "memory_ok"):
            blocked = next(
                item for item in result["details"] if item["name"] == gate
            )
            self.assertEqual(blocked["status"], "ERROR")
            self.assertIn("not independently established", blocked["message"])
        deployment_detail = next(
            item
            for item in result["details"]
            if item["name"] == "deployment_protocol_ok"
        )
        self.assertEqual(deployment_detail["status"], "FAILED")
        self.assertIn(
            "fresh-process combined iteration",
            deployment_detail["message"],
        )
        self.assertIn(
            "did not complete a valid isolated "
            "first-run/warmed-iteration measurement",
            deployment_detail["message"],
        )
        self.assertNotIn(
            "cache was not empty",
            deployment_detail["message"],
        )
        for partial_memory_metric in (
            "memory_ratio_bucket_lower",
            "memory_ratio_bucket_upper",
            "memory_overage_bucket_upper",
            "memory_pressure",
            "worst_memory_stage",
        ):
            self.assertNotIn(partial_memory_metric, result["metrics"])
        passed = sum(
            item["status"] == "PASSED" for item in result["details"]
        )
        self.assertAlmostEqual(result["pass_rate"], passed / len(result["details"]))

        serialized = json.dumps(result, sort_keys=True)
        for private_value in (
            "hidden_secret_case_name",
            "hidden_secret_deployment_case",
            "987654321",
            "PRIVATE_SENTINEL",
            "PRIVATE_DEPLOYMENT_SENTINEL",
            "[7, 12345, 6789, 511]",
            "12.345678",
            "98.765432",
        ):
            self.assertNotIn(private_value, serialized)

    def test_memory_failure_reports_ratio_and_diagnostic_score_without_shape(
        self,
    ) -> None:
        payload: dict[str, object] = _trusted_payload(0.0)
        payload.update(
            {
                "final_version": 17.0,
                "memory_ok": 0.0,
                "diagnostic_score": 757.64,
                "diagnostic_weighted_latency_ms": 757.64,
                "max_memory_ratio": 1.03125,
                "memory_headroom_ratio": -0.03125,
                "memory_overage_ratio": 0.03125,
                "worst_memory_stage": "full",
                "memory_pressure": "over_budget",
            }
        )
        detail = {
            **payload,
            "cases": [
                {
                    "name": "PRIVATE_MEMORY_CASE",
                    "stage": "full",
                    "shape": [31, 1234567, 8191, 511],
                    "seed": 999999,
                    "memory_ok": False,
                    "error": (
                        "candidate exceeded the streaming-memory budget by "
                        "PRIVATE_BYTES"
                    ),
                }
            ],
        }
        result = adapter.build_result(
            payload,
            detail,
            verifier_completed=True,
            verifier_returncode=1,
        )
        self.assertFalse(result["valid"])
        self.assertIsNone(result["score"])
        self.assertEqual(result["metrics"]["diagnostic_score"], 757.64)
        self.assertNotIn("max_memory_ratio", result["metrics"])
        self.assertEqual(
            result["metrics"]["memory_ratio_bucket_lower"],
            1.0,
        )
        self.assertEqual(
            result["metrics"]["memory_ratio_bucket_upper"],
            1.05,
        )
        self.assertEqual(
            result["metrics"]["memory_overage_bucket_upper"],
            0.05,
        )
        self.assertEqual(
            result["metrics"]["worst_memory_stage"],
            "combined iteration",
        )
        self.assertEqual(result["metrics"]["memory_pressure"], "over_budget")
        self.assertTrue(result["metrics"]["diagnostic_only"])
        self.assertIn("[1.00x, 1.05x)", result["summary"])
        self.assertIn("less than 5% over", result["summary"])
        self.assertIn("757.640000", result["summary"])
        performance = next(
            item for item in result["details"] if item["name"] == "performance"
        )
        self.assertEqual(performance["status"], "FAILED")
        self.assertIn(
            "Diagnostic weighted latency 757.640000 ms",
            performance["message"],
        )

        serialized = json.dumps(result, sort_keys=True)
        for private_value in (
            "PRIVATE_MEMORY_CASE",
            "PRIVATE_BYTES",
            "1234567",
            "8191",
            "999999",
            "1.03125",
        ):
            self.assertNotIn(private_value, serialized)

    def test_source_rejection_still_reports_requested_final_version_safely(self) -> None:
        payload: dict[str, object] = _trusted_payload(0.0)
        payload.update(
            {
                "final_version": 999.0,
                "requested_final_version": 999.0,
                "source_ok": 0.0,
            }
        )
        detail = {
            **payload,
            "source_problems": [
                {
                    "path": "kernels/private_secret.py",
                    "message": "PRIVATE_SOURCE_SENTINEL",
                }
            ],
        }
        result = adapter.build_result(
            payload,
            detail,
            verifier_completed=True,
            verifier_returncode=1,
        )
        self.assertFalse(result["valid"])
        self.assertIn("FINAL_VERSION=v999 is invalid", result["summary"])
        self.assertEqual(result["metrics"]["selected_final_version"], "v999")
        serialized = json.dumps(result, sort_keys=True)
        self.assertNotIn("private_secret.py", serialized)
        self.assertNotIn("PRIVATE_SOURCE_SENTINEL", serialized)

    def test_packaged_adapter_is_byte_identical(self) -> None:
        self.assertEqual(ADAPTER_PATH.read_bytes(), IMAGE_ADAPTER_PATH.read_bytes())

    def test_prepare_log_dir_removes_stale_trusted_results(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            log_dir = root / "logs"
            log_dir.mkdir()
            (log_dir / "reward.json").write_text(
                json.dumps(_trusted_payload()),
                encoding="utf-8",
            )
            verifier = self._write_verifier(root, payload=None)
            result = adapter.evaluate(verifier, root, log_dir, 10.0)
            self.assertFalse(result["valid"])
            self.assertIsNone(result["score"])
            self.assertFalse((log_dir / "reward.json").exists())


if __name__ == "__main__":
    unittest.main()
