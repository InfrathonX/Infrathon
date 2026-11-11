from __future__ import annotations

import math
import sys
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = ROOT / "images" / "judge" / "benchmark"
JUDGE_DIR = ROOT / "images" / "judge"
sys.path.insert(0, str(BENCHMARK_DIR))
sys.path.insert(0, str(JUDGE_DIR))

import benchmark_cases as schema  # noqa: E402
from _judge_assets import load_hidden_cases  # noqa: E402

hidden_manifest = load_hidden_cases(required=True)


def _schema_fixture(**overrides):
    values = {
        "name": "schema_fixture",
        "suite": "unit_contract",
        "visibility": "hidden",
        "role": "performance",
        "workload_label": "B=1, N=17, K=5, D=3",
        "stage": "assign",
        "component": "assignment",
        "measurement_mode": "steady_state",
        "b": 1,
        "n": 17,
        "k": 5,
        "d": 3,
        "warmup": 2,
        "iters": 3,
        "repeats": 4,
        "weight": 0.25,
        "score_group": "assign",
    }
    values.update(overrides)
    return schema.BenchmarkCase(**values)


class BenchmarkCaseManifestTests(unittest.TestCase):
    def test_binary_unit_convention_is_explicit(self) -> None:
        self.assertEqual(schema.K, 1_024)
        self.assertEqual(schema.M, 1_048_576)
        massive = next(
            case
            for case in hidden_manifest.HIDDEN_RUNTIME_CASES
            if case.name == "hidden_massive_update"
        )
        self.assertEqual(massive.n, 33 * schema.M)

    def test_schema_is_immutable_and_derives_shape_and_timed_calls(self) -> None:
        case = _schema_fixture()
        self.assertEqual(case.shape, (1, 17, 5, 3))
        self.assertEqual(case.timed_calls, 12)
        with self.assertRaises(FrozenInstanceError):
            case.weight = 1.0  # type: ignore[misc]

    def test_schema_rejects_invalid_component_mode_and_correctness_weight(
        self,
    ) -> None:
        with self.assertRaisesRegex(ValueError, "requires stage"):
            _schema_fixture(stage="update")
        with self.assertRaisesRegex(ValueError, "fresh_process"):
            _schema_fixture(measurement_mode="fresh_process")
        with self.assertRaisesRegex(ValueError, "weight zero"):
            _schema_fixture(role="correctness", weight=0.25)

        deployment = _schema_fixture(
            name="deployment_fixture",
            stage="full",
            component="deployment_bundle",
            measurement_mode="fresh_process",
            score_group="deploy_control",
        )
        self.assertEqual(deployment.stage, "full")
        self.assertEqual(deployment.measurement_mode, "fresh_process")
        with self.assertRaisesRegex(ValueError, "invalid score group"):
            _schema_fixture(score_group="unknown")
        with self.assertRaisesRegex(ValueError, "must not have a score group"):
            _schema_fixture(
                role="correctness",
                weight=0.0,
                score_group="assign",
            )

    def test_hidden_runtime_points_match_declared_execution_contract(self) -> None:
        expected = {
            "hidden_small_n_small_k_full": (
                "full",
                (1, 16 * schema.K, 8 * schema.K, 512),
                0.05,
            ),
            "hidden_large_n_small_k_full": (
                "full",
                (1, 8 * schema.M, 1 * schema.K, 128),
                0.05,
            ),
            "hidden_large_n_large_k_full": (
                "full",
                (1, 1 * schema.M, 8 * schema.K, 128),
                0.05,
            ),
            "hidden_batched_full": (
                "full",
                (32, 256 * schema.K, 1 * schema.K, 128),
                0.05,
            ),
            "hidden_single_assign": (
                "assign",
                (1, 512 * schema.K, 8 * schema.K, 128),
                0.05,
            ),
            "hidden_batched_assign": (
                "assign",
                (32, 64 * schema.K, 4 * schema.K, 128),
                0.05,
            ),
            "hidden_massive_update": (
                "update",
                (1, 33 * schema.M, 4 * schema.K, 128),
                0.05,
            ),
            "hidden_batched_update": (
                "update",
                (32, 1 * schema.M, 4 * schema.K, 128),
                0.05,
            ),
        }
        self.assertEqual(len(hidden_manifest.HIDDEN_RUNTIME_CASES), len(expected))
        for case in hidden_manifest.HIDDEN_RUNTIME_CASES:
            stage, shape, weight = expected[case.name]
            self.assertEqual((case.stage, case.shape, case.weight), (stage, shape, weight))
            self.assertEqual(case.measurement_mode, "steady_state")
            self.assertEqual(
                case.component,
                {
                    "assign": "assignment",
                    "update": "centroid_update",
                    "full": "combined_iteration",
                }[stage],
            )
            self.assertEqual(case.score_group, stage)
            self.assertGreaterEqual(case.warmup, 1)
            self.assertGreater(case.timed_calls, 0)

    def test_hidden_runtime_stage_weights_are_deliberate(self) -> None:
        totals = {
            stage: sum(
                case.weight
                for case in hidden_manifest.HIDDEN_RUNTIME_CASES
                if case.stage == stage
            )
            for stage in ("full", "assign", "update")
        }
        self.assertEqual(totals, {"full": 0.20, "assign": 0.10, "update": 0.10})
        self.assertEqual(schema.SCORE_PROFILE, "v7_family_tail60")
        self.assertEqual(
            (
                schema.ASSIGN_WEIGHT,
                schema.UPDATE_WEIGHT,
                schema.FULL_WEIGHT,
                schema.DEPLOYMENT_WEIGHT,
            ),
            (0.10, 0.10, 0.20, 0.60),
        )
        self.assertEqual(
            schema.SCORE_GROUP_WEIGHTS,
            {
                "assign": 0.10,
                "update": 0.10,
                "full": 0.20,
                "deploy_control": 0.10,
                "deploy_large_k": 0.25,
                "deploy_large_n_high_d": 0.25,
            },
        )
        self.assertTrue(
            math.isclose(
                sum(schema.SCORE_GROUP_WEIGHTS.values()),
                1.0,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        )

    def test_hidden_deployment_has_all_six_fresh_process_points(self) -> None:
        expected_shapes = {
            (1, 1 * schema.M, 1 * schema.K, 128),
            (1, 1 * schema.M, 64 * schema.K, 128),
            (1, 8 * schema.M, 8 * schema.K, 512),
            (1, 4 * schema.M, 64 * schema.K, 128),
            (1, 16 * schema.M, 8 * schema.K, 512),
            (1, 8 * schema.M, 64 * schema.K, 128),
        }
        self.assertEqual(len(hidden_manifest.HIDDEN_DEPLOYMENT_CASES), 6)
        self.assertEqual(
            {case.shape for case in hidden_manifest.HIDDEN_DEPLOYMENT_CASES},
            expected_shapes,
        )
        expected_groups = {
            "hidden_deployment_1m_1k_d128": (
                "deploy_control",
                0.10,
            ),
            "hidden_deployment_1m_64k_d128": (
                "deploy_large_k",
                0.25 / 3.0,
            ),
            "hidden_deployment_8m_8k_d512": (
                "deploy_large_n_high_d",
                0.25 / 2.0,
            ),
            "hidden_deployment_4m_64k_d128": (
                "deploy_large_k",
                0.25 / 3.0,
            ),
            "hidden_deployment_16m_8k_d512": (
                "deploy_large_n_high_d",
                0.25 / 2.0,
            ),
            "hidden_deployment_8m_64k_d128": (
                "deploy_large_k",
                0.25 / 3.0,
            ),
        }
        for case in hidden_manifest.HIDDEN_DEPLOYMENT_CASES:
            self.assertEqual(case.stage, "full")
            self.assertEqual(case.component, "deployment_bundle")
            self.assertEqual(case.measurement_mode, "fresh_process")
            self.assertEqual((case.warmup, case.iters, case.repeats), (0, 1, 1))
            expected_group, expected_weight = expected_groups[case.name]
            self.assertEqual(case.score_group, expected_group)
            self.assertTrue(
                math.isclose(
                    case.weight,
                    expected_weight,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            )
            self.assertIn("warmed-iteration", case.notes)
        self.assertTrue(
            math.isclose(
                sum(
                    case.weight
                    for case in hidden_manifest.HIDDEN_DEPLOYMENT_CASES
                ),
                0.60,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        )

    def test_hidden_manifest_is_complete_unique_and_weighted_to_one(self) -> None:
        hidden_manifest.validate_hidden_manifest()
        cases = hidden_manifest.HIDDEN_BENCHMARK_CASES
        names = [case.name for case in cases]
        self.assertEqual(len(names), len(set(names)))
        self.assertTrue(
            math.isclose(
                sum(case.weight for case in cases),
                1.0,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        )
        self.assertTrue(
            all(
                case.visibility == "hidden"
                and case.role == "performance"
                and case.workload_label
                and case.timed_calls > 0
                for case in cases
            )
        )
        observed_group_weights = {
            group: sum(
                case.weight for case in cases if case.score_group == group
            )
            for group in schema.SCORE_GROUP_WEIGHTS
        }
        for group, expected in schema.SCORE_GROUP_WEIGHTS.items():
            self.assertTrue(
                math.isclose(
                    observed_group_weights[group],
                    expected,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            )

    def test_schema_module_contains_no_obsolete_formal_case_registry(self) -> None:
        source = (BENCHMARK_DIR / "benchmark_cases.py").read_text(
            encoding="utf-8"
        )
        for obsolete in (
            "PUBLIC_BENCHMARK_CASES",
            "CORRECTNESS_CASES",
            "ALL_CASES",
            "MANIFEST",
            "cases_for_suite",
        ):
            with self.subTest(obsolete=obsolete):
                self.assertNotIn(obsolete, source)


if __name__ == "__main__":
    unittest.main()
