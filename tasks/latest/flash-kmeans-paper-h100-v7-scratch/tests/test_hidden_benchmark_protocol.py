from __future__ import annotations

import importlib.util
import math
import sys
import types
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = ROOT / "images" / "judge" / "benchmark"
JUDGE_DIR = ROOT / "images" / "judge"
VERIFY_PATH = ROOT / "images" / "judge" / "verify_flash_kmeans.py"
sys.path.insert(0, str(BENCHMARK_DIR))
sys.path.insert(0, str(JUDGE_DIR))

from _judge_assets import load_hidden_cases  # noqa: E402

hidden_cases = load_hidden_cases(required=True)


@dataclass(frozen=True)
class _CaseSpec:
    name: str
    stage: str
    category: str
    b: int
    n: int
    k: int
    d: int
    seed: int
    distribution: str = "uniform"
    tie: bool = False
    weight: float = 0.0
    warmup: int = 1
    iters: int = 1
    repeats: int = 1
    role: str = "performance"
    component: str = "combined_iteration"
    measurement_mode: str = "steady_state"
    score_group: str = ""
    verification: str = "full_exact"
    visibility: str = "unspecified"


def _benchmark_case_to_spec(case, seed):
    return _CaseSpec(
        name=case.name,
        stage=case.stage,
        category="online",
        b=case.b,
        n=case.n,
        k=case.k,
        d=case.d,
        seed=seed,
        distribution=(
            "balanced_shuffled"
            if case.data_family == "balanced_shuffled"
            else "uniform"
        ),
        tie=False,
        weight=case.weight,
        warmup=case.warmup,
        iters=case.iters,
        repeats=case.repeats,
        role=case.role,
        component=case.component,
        measurement_mode=case.measurement_mode,
        score_group=case.score_group,
        verification=case.verification,
        visibility=case.visibility,
    )


def _load_verifier():
    fake_torch = types.ModuleType("torch")
    fake_torch.Tensor = object
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: True)

    fake_kernel = types.ModuleType("kernel_bench")
    fake_kernel.CaseSpec = _CaseSpec
    for name in (
        "DEVICE_SYNCHRONIZE",
        "EMPTY_CACHE",
        "MAX_MEMORY_ALLOCATED",
        "MEMORY_ALLOCATED",
        "RESET_PEAK_MEMORY",
        "benchmark_stable_latency_ms",
        "check_output",
        "clone_guards",
        "inputs_unchanged",
        "load_candidate",
        "make_inputs",
        "memory_limit_bytes",
        "peak_output",
        "run_stage",
    ):
        setattr(fake_kernel, name, lambda *args, **kwargs: None)
    fake_kernel.benchmark_stable_latency_ms = lambda *args, **kwargs: {
        "latency_ms": 1.0,
        "trial_medians_ms": [1.0],
        "trial_count": 1,
        "spread_ratio": 1.0,
        "adaptive_retry": False,
        "stable": True,
        "inlier_count": 1,
        "peak_bytes": 0,
        "policy": "manifest_median",
    }
    fake_kernel.benchmark_case_to_spec = _benchmark_case_to_spec
    fake_kernel.memory_limit_bytes = lambda spec: 1024
    fake_kernel.memory_diagnostics = lambda peak, limit: {
        "memory_ratio": peak / limit,
        "memory_headroom_bytes": limit - peak,
        "memory_headroom_ratio": 1.0 - peak / limit,
        "memory_overage_bytes": max(0, peak - limit),
        "memory_overage_ratio": max(0.0, peak / limit - 1.0),
        "memory_pressure": (
            "over_budget"
            if peak > limit
            else ("tight" if peak / limit > 0.85 else "comfortable")
        ),
        "memory_comfort_ratio": 0.85,
    }

    fake_deployment = types.ModuleType("deployment_bench")
    fake_deployment.run_deployment_case = lambda *args, **kwargs: {}
    fake_admission = types.ModuleType("source_admission")
    fake_admission.scan_submission = lambda root: {}
    module_name = "_test_hidden_benchmark_protocol"
    spec = importlib.util.spec_from_file_location(module_name, VERIFY_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with patch.dict(
        sys.modules,
        {
            "torch": fake_torch,
            "kernel_bench": fake_kernel,
            "deployment_bench": fake_deployment,
            "source_admission": fake_admission,
            module_name: module,
        },
    ):
        spec.loader.exec_module(module)
    return module


class HiddenBenchmarkProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.verifier = _load_verifier()

    def test_runtime_is_eight_benchmark_points_plus_private_zero_weight_gates(
        self,
    ) -> None:
        specs = self.verifier.hidden_runtime_specs("full")
        scored = [spec for spec in specs if spec.weight > 0.0]
        gates = [spec for spec in specs if spec.weight == 0.0]
        self.assertEqual(len(scored), 8)
        self.assertEqual(
            [(s.b, s.n, s.k, s.d) for s in scored],
            [case.shape for case in hidden_cases.HIDDEN_RUNTIME_CASES],
        )
        self.assertTrue(all(math.isclose(spec.weight, 0.05) for spec in scored))
        self.assertTrue(math.isclose(sum(spec.weight for spec in scored), 0.40))
        totals = {
            stage: sum(spec.weight for spec in scored if spec.stage == stage)
            for stage in ("assign", "update", "full")
        }
        self.assertEqual(totals, {"assign": 0.10, "update": 0.10, "full": 0.20})
        self.assertTrue(
            all(spec.score_group == spec.stage for spec in scored)
        )
        self.assertEqual(len(gates), 8)
        tie_gate = next(
            spec for spec in gates if spec.name == "hidden_exact_tie_tile_boundaries"
        )
        self.assertTrue(tie_gate.tie)
        self.assertEqual((tie_gate.b, tie_gate.n, tie_gate.k, tie_gate.d), (2, 4099, 257, 33))
        self.assertTrue(all(spec.role == "correctness" for spec in gates))
        self.assertTrue(all(spec.visibility == "hidden" for spec in gates))
        scored_manifest_names = {
            case.name for case in hidden_cases.HIDDEN_BENCHMARK_CASES
        }
        self.assertTrue(
            scored_manifest_names.isdisjoint(spec.name for spec in gates)
        )
        streaming = next(
            spec
            for spec in gates
            if spec.name == "hidden_streaming_memory_assign"
        )
        self.assertEqual(
            streaming.verification,
            "sampled_exact_and_streaming_memory",
        )
        self.assertTrue(
            all(
                "sampled" not in spec.verification
                for spec in gates
                if spec is not streaming
            )
        )

    def test_deployment_track_contains_six_fresh_process_workloads(self) -> None:
        specs = self.verifier.hidden_deployment_specs("full")
        self.assertEqual(len(specs), 6)
        self.assertEqual(
            [(s.b, s.n, s.k, s.d) for s in specs],
            [case.shape for case in hidden_cases.HIDDEN_DEPLOYMENT_CASES],
        )
        self.assertTrue(math.isclose(sum(spec.weight for spec in specs), 0.60))
        group_weights = {
            group: sum(spec.weight for spec in specs if spec.score_group == group)
            for group in {
                "deploy_control",
                "deploy_large_k",
                "deploy_large_n_high_d",
            }
        }
        self.assertEqual(
            group_weights,
            {
                "deploy_control": 0.10,
                "deploy_large_k": 0.25,
                "deploy_large_n_high_d": 0.25,
            },
        )
        self.assertTrue(all(spec.component == "deployment_bundle" for spec in specs))
        self.assertTrue(
            all(spec.measurement_mode == "fresh_process" for spec in specs)
        )
        self.assertTrue(all(spec.stage == "full" for spec in specs))

    def test_full_suite_has_fourteen_positive_and_eight_gate_rows(self) -> None:
        runtime = self.verifier.hidden_runtime_specs("full")
        deployment_specs = self.verifier.hidden_deployment_specs("full")
        successful = {
            "status": "ok",
            "first_run_ms": 10.0,
            "warmed_iteration_ms": 1.0,
            "correctness_ok": True,
            "input_integrity_ok": True,
            "output_contract_ok": True,
            "cache_was_empty": True,
            "peak_extra_bytes": 8,
        }
        candidates = {
            spec.name: dict(successful)
            for spec in deployment_specs
        }
        deployment_rows = self.verifier.compose_deployment_rows(
            deployment_specs,
            candidates,
        )
        runtime_positive = [spec for spec in runtime if spec.weight > 0.0]
        gates = [spec for spec in runtime if spec.weight == 0.0]
        positive_weights = [
            *(spec.weight for spec in runtime_positive),
            *(row["weight"] for row in deployment_rows if row["weight"] > 0.0),
        ]

        self.assertEqual(
            (len(runtime_positive), len(deployment_rows), len(gates)),
            (8, 6, 8),
        )
        self.assertEqual(len(positive_weights), 14)
        self.assertEqual(len(runtime) + len(deployment_rows), 22)
        self.assertTrue(
            math.isclose(
                sum(positive_weights),
                1.0,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        )
        self.assertEqual(sum(row["weight"] == 0.0 for row in deployment_rows), 0)
        self.assertEqual(sum(row["weight"] > 0.0 for row in deployment_rows), 6)

    def test_coarse_deployment_families_are_private_nonoverlapping_groups(
        self,
    ) -> None:
        families = hidden_cases.COARSE_DEPLOYMENT_FAMILIES
        self.assertEqual(set(families), {"large_k", "large_n_high_d"})
        self.assertTrue(all(len(members) >= 2 for members in families.values()))
        flattened = [name for members in families.values() for name in members]
        self.assertEqual(len(flattened), len(set(flattened)))
        deployment_names = {
            case.name for case in hidden_cases.HIDDEN_DEPLOYMENT_CASES
        }
        self.assertTrue(set(flattened) < deployment_names)

    def test_coarse_deployment_latency_is_a_family_geometric_mean(self) -> None:
        latencies = {"large_k": 4.0, "large_n_high_d": 9.0}
        rows = []
        for family, members in hidden_cases.COARSE_DEPLOYMENT_FAMILIES.items():
            rows.extend(
                {
                    "name": name,
                    "candidate_ms": latencies[family],
                    "all_gates_ok": True,
                }
                for name in members
            )
        control = next(
            case.name
            for case in hidden_cases.HIDDEN_DEPLOYMENT_CASES
            if case.name not in {row["name"] for row in rows}
        )
        rows.append(
            {"name": control, "candidate_ms": 10_000.0, "all_gates_ok": True}
        )
        result = self.verifier.trusted_coarse_deployment_latencies(rows)
        for family, expected in latencies.items():
            self.assertAlmostEqual(result[family], expected)

        missing = self.verifier.trusted_coarse_deployment_latencies(rows[:-2])
        self.assertEqual(missing, {"large_k": 4.0, "large_n_high_d": 0.0})
        failed = [dict(row) for row in rows]
        failed[0]["all_gates_ok"] = False
        failed_result = self.verifier.trusted_coarse_deployment_latencies(
            failed
        )
        self.assertEqual(failed_result["large_k"], 0.0)
        self.assertAlmostEqual(failed_result["large_n_high_d"], 9.0)

    def test_coarse_deployment_statuses_are_independent_and_fail_closed(
        self,
    ) -> None:
        rows = [
            {"name": name, "all_gates_ok": True}
            for members in hidden_cases.COARSE_DEPLOYMENT_FAMILIES.values()
            for name in members
        ]
        self.assertEqual(
            self.verifier.trusted_coarse_deployment_statuses(rows),
            {"large_k": "passed", "large_n_high_d": "passed"},
        )

        failed = [dict(row) for row in rows]
        failed[0]["all_gates_ok"] = False
        self.assertEqual(
            self.verifier.trusted_coarse_deployment_statuses(failed),
            {"large_k": "failed", "large_n_high_d": "passed"},
        )

        large_n_member = hidden_cases.COARSE_DEPLOYMENT_FAMILIES[
            "large_n_high_d"
        ][0]
        missing = [row for row in rows if row["name"] != large_n_member]
        self.assertEqual(
            self.verifier.trusted_coarse_deployment_statuses(missing),
            {"large_k": "passed", "large_n_high_d": "not_reached"},
        )

        duplicate = rows + [dict(rows[0])]
        self.assertEqual(
            self.verifier.trusted_coarse_deployment_statuses(duplicate),
            {"large_k": "not_reached", "large_n_high_d": "passed"},
        )
        self.assertEqual(
            self.verifier.trusted_coarse_deployment_statuses([]),
            {"large_k": "not_reached", "large_n_high_d": "not_reached"},
        )

    def test_six_deployment_bundles_mean_six_candidate_child_invocations(self) -> None:
        specs = self.verifier.hidden_deployment_specs("full")
        calls = []

        def run_child(source_file, **kwargs):
            calls.append((source_file, kwargs))
            return {}

        with patch.object(
            self.verifier,
            "run_deployment_case",
            side_effect=run_child,
        ) as runner:
            candidate = self.verifier.deployment_pass(
                specs,
                Path("/candidate/flash_kmeans_v001.py"),
                timeout_sec=180.0,
                candidate_root=Path("/candidate"),
            )

        expected_names = [spec.name for spec in specs]
        self.assertEqual(list(candidate), expected_names)
        self.assertEqual(runner.call_count, 6)
        self.assertEqual(
            [kwargs["case_name"] for _source, kwargs in calls],
            expected_names,
        )
        self.assertTrue(all("trusted_source" not in kwargs for _source, kwargs in calls))
        self.assertTrue(
            all(kwargs["verification"] == "sampled" for _source, kwargs in calls)
        )
        self.assertTrue(
            all(kwargs["warmed_repeats"] == 3 for _source, kwargs in calls)
        )

    def test_candidate_latency_score_uses_six_group_gms_and_fails_closed(
        self,
    ) -> None:
        runtime = self.verifier.hidden_runtime_specs("full")
        deployment = self.verifier.hidden_deployment_specs("full")
        specs = runtime + deployment
        large_k_latencies = dict(
            zip(
                hidden_cases.COARSE_DEPLOYMENT_FAMILIES["large_k"],
                (4.0, 16.0, 64.0),
            )
        )
        large_n_latencies = dict(
            zip(
                hidden_cases.COARSE_DEPLOYMENT_FAMILIES["large_n_high_d"],
                (9.0, 81.0),
            )
        )
        expected_groups = {
            "assign": 2.0,
            "update": 3.0,
            "full": 5.0,
            "deploy_control": 7.0,
            "deploy_large_k": 16.0,
            "deploy_large_n_high_d": 27.0,
        }

        def latency(spec: _CaseSpec) -> float:
            if spec.name in large_k_latencies:
                return large_k_latencies[spec.name]
            if spec.name in large_n_latencies:
                return large_n_latencies[spec.name]
            return expected_groups.get(spec.score_group, 1.0)

        rows = [
            {
                "name": spec.name,
                "stage": spec.stage,
                "component": spec.component,
                "measurement_mode": spec.measurement_mode,
                "score_group": spec.score_group,
                "weight": spec.weight,
                "candidate_ms": latency(spec),
                "all_gates_ok": True,
                "non_memory_gates_ok": True,
            }
            for spec in specs
        ]
        score, groups = self.verifier.trusted_latency_score(rows, specs)
        expected = math.exp(
            0.10 * math.log(2.0)
            + 0.10 * math.log(3.0)
            + 0.20 * math.log(5.0)
            + 0.10 * math.log(7.0)
            + 0.25 * math.log(16.0)
            + 0.25 * math.log(27.0)
        )
        self.assertAlmostEqual(score, expected)
        self.assertEqual(set(groups), set(expected_groups))
        for group, group_latency in expected_groups.items():
            self.assertAlmostEqual(groups[group], group_latency)

        first_positive = next(
            index
            for index, row in enumerate(rows)
            if float(row["weight"]) > 0.0
        )
        missing = rows[:first_positive] + rows[first_positive + 1 :]
        duplicated = rows + [dict(rows[first_positive])]
        self.assertEqual(
            self.verifier.trusted_latency_score(missing, specs)[0],
            0.0,
        )
        self.assertEqual(
            self.verifier.trusted_latency_score(duplicated, specs)[0],
            0.0,
        )
        for field, value in (
            ("weight", rows[first_positive]["weight"] + 0.001),
            ("score_group", "update"),
            ("stage", "update"),
            ("candidate_ms", 0.0),
            ("candidate_ms", float("nan")),
        ):
            with self.subTest(field=field, value=value):
                tampered = [dict(row) for row in rows]
                tampered[first_positive][field] = value
                self.assertEqual(
                    self.verifier.trusted_latency_score(tampered, specs)[0],
                    0.0,
                )
        unexpected = [dict(row) for row in rows]
        unexpected.append(
            {
                **rows[first_positive],
                "name": "unexpected_positive_case",
            }
        )
        self.assertEqual(
            self.verifier.trusted_latency_score(unexpected, specs)[0],
            0.0,
        )

        memory_failed = [
            {
                **row,
                "all_gates_ok": False
                if float(row["weight"]) > 0.0
                else row["all_gates_ok"],
                "non_memory_gates_ok": True,
            }
            for row in rows
        ]
        accepted, _ = self.verifier.trusted_latency_score(memory_failed, specs)
        diagnostic, diagnostic_components = (
            self.verifier.trusted_diagnostic_score(memory_failed, specs)
        )
        self.assertEqual(accepted, 0.0)
        self.assertAlmostEqual(diagnostic, expected)
        for group, group_latency in expected_groups.items():
            self.assertAlmostEqual(
                diagnostic_components[group],
                group_latency,
            )

    def test_deployment_bundle_reports_first_run_and_scores_warmed_iteration(
        self,
    ) -> None:
        spec = self.verifier.hidden_deployment_specs("smoke")[0]
        self.assertEqual(
            (spec.b, spec.n, spec.k, spec.d),
            (1, 1 << 20, 1 << 10, 128),
        )
        candidate = {
            "status": "ok",
            "first_run_ms": 5.0,
            "warmed_iteration_ms": 1.1,
            "correctness_ok": True,
            "input_integrity_ok": True,
            "output_contract_ok": True,
            "cache_was_empty": True,
            "peak_extra_bytes": 8,
        }
        rows = self.verifier.compose_deployment_rows(
            [spec],
            {spec.name: candidate},
        )
        self.assertEqual(
            {row["component"] for row in rows},
            {"deployment_bundle"},
        )
        self.assertEqual(
            {row["measurement_phase"] for row in rows},
            {"warmed_iteration"},
        )
        self.assertEqual(rows[0]["score_group"], "deploy_control")
        self.assertEqual(sum(row["weight"] for row in rows), spec.weight)
        self.assertEqual(rows[0]["candidate_first_run_ms"], 5.0)
        self.assertEqual(rows[0]["candidate_ms"], 1.1)
        self.assertAlmostEqual(rows[0]["compile_overhead_ratio"], 5.0 / 1.1)
        self.assertTrue(all(row["all_gates_ok"] for row in rows))

        missing_first_run = {**candidate, "first_run_ms": 0.0}
        failed = self.verifier.compose_deployment_rows(
            [spec],
            {spec.name: missing_first_run},
        )
        self.assertTrue(all(not row["all_gates_ok"] for row in failed))


if __name__ == "__main__":
    unittest.main()
