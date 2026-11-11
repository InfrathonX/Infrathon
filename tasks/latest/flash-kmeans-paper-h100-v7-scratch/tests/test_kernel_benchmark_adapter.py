from __future__ import annotations

import importlib.util
import inspect
import random
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = ROOT / "images" / "judge" / "benchmark"
JUDGE_DIR = ROOT / "images" / "judge"
sys.path.insert(0, str(BENCHMARK_DIR))
sys.path.insert(0, str(JUDGE_DIR))

from _judge_assets import load_hidden_cases, requires_hidden_cases  # noqa: E402

hidden_cases = load_hidden_cases()
import benchmark_cases  # noqa: E402


def _schema_case(**overrides):
    values = {
        "name": "adapter_fixture",
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
        "warmup": 1,
        "iters": 2,
        "repeats": 3,
        "weight": 0.0,
        "score_group": "assign",
    }
    values.update(overrides)
    return benchmark_cases.BenchmarkCase(**values)


class _FakeTensor:
    def __init__(self, data):
        self.data = data

    def remainder(self, divisor: int):
        return _FakeTensor([value % divisor for value in self.data])

    def to(self, _dtype):
        return _FakeTensor(
            [list(row) if isinstance(row, list) else row for row in self.data]
        )

    def __getitem__(self, key):
        if isinstance(key, _FakeTensor):
            return _FakeTensor([self.data[index] for index in key.data])
        value = self.data[key]
        return _FakeTensor(value) if isinstance(value, list) else value

    def __setitem__(self, key, value):
        self.data[key] = list(value.data if isinstance(value, _FakeTensor) else value)

    def tolist(self):
        return self.data


class _FakeGenerator:
    def __init__(self, device=None):
        self.device = device
        self.random = random.Random()

    def manual_seed(self, seed: int):
        self.random.seed(seed)
        return self


def _fake_torch_module() -> types.ModuleType:
    module = types.ModuleType("torch")

    class _Cuda:
        synchronize = staticmethod(lambda: None)
        memory_allocated = staticmethod(lambda: 0)
        reset_peak_memory_stats = staticmethod(lambda: None)
        max_memory_allocated = staticmethod(lambda: 0)
        empty_cache = staticmethod(lambda: None)

    def no_grad():
        return lambda function: function

    def arange(stop, *, generator=None, device=None, dtype=None):
        del generator, device, dtype
        return _FakeTensor(list(range(stop)))

    def empty(shape, *, device=None, dtype=None):
        del device, dtype
        rows, columns = shape
        return _FakeTensor([[0 for _ in range(columns)] for _ in range(rows)])

    def randperm(stop, *, generator, device=None):
        del device
        values = list(range(stop))
        generator.random.shuffle(values)
        return _FakeTensor(values)

    module.Tensor = _FakeTensor
    module.Generator = _FakeGenerator
    module.cuda = _Cuda()
    module.no_grad = no_grad
    module.arange = arange
    module.empty = empty
    module.randperm = randperm
    for dtype in (
        "float16",
        "float32",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
    ):
        setattr(module, dtype, dtype)
    return module


def _load_kernel_module():
    fake_torch = _fake_torch_module()
    fake_baseline = types.ModuleType("basic_baseline")
    fake_baseline.assign = lambda *args: None
    fake_baseline.update = lambda *args: None
    module_name = "_test_kernel_benchmark_adapter"
    spec = importlib.util.spec_from_file_location(
        module_name,
        BENCHMARK_DIR / "kernel_bench.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load kernel_bench.py")
    module = importlib.util.module_from_spec(spec)
    with patch.dict(
        sys.modules,
        {
            "torch": fake_torch,
            "basic_baseline": fake_baseline,
            module_name: module,
        },
    ):
        spec.loader.exec_module(module)
    return module, fake_torch


class KernelBenchmarkAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.kernel, cls.fake_torch = _load_kernel_module()

    def test_old_positional_case_spec_construction_remains_compatible(self) -> None:
        spec = self.kernel.CaseSpec(
            "legacy",
            "assign",
            "online",
            1,
            17,
            5,
            33,
            91,
            "zipf",
            True,
            0.125,
        )
        self.assertEqual(
            (
                spec.name,
                spec.stage,
                spec.category,
                spec.b,
                spec.n,
                spec.k,
                spec.d,
                spec.seed,
                spec.distribution,
                spec.tie,
                spec.weight,
            ),
            (
                "legacy",
                "assign",
                "online",
                1,
                17,
                5,
                33,
                91,
                "zipf",
                True,
                0.125,
            ),
        )
        self.assertEqual((spec.warmup, spec.iters, spec.repeats), (1, 1, 1))
        self.assertEqual(spec.role, "performance")
        self.assertIn(
            spec.component,
            {
                "assignment",
                "centroid_update",
                "combined_iteration",
                "deployment_bundle",
            },
        )
        self.assertEqual(spec.measurement_mode, "steady_state")
        self.assertEqual(spec.verification, "full_exact")
        self.assertEqual(spec.visibility, "unspecified")
        self.assertEqual(spec.score_group, "")

    def test_memory_diagnostics_keep_a_hard_cap_but_expose_signed_margin(
        self,
    ) -> None:
        comfortable = self.kernel.memory_diagnostics(80, 100)
        self.assertEqual(comfortable["memory_pressure"], "comfortable")
        self.assertAlmostEqual(comfortable["memory_ratio"], 0.8)
        self.assertEqual(comfortable["memory_headroom_bytes"], 20)
        self.assertAlmostEqual(comfortable["memory_headroom_ratio"], 0.2)
        self.assertEqual(comfortable["memory_overage_bytes"], 0)

        tight = self.kernel.memory_diagnostics(90, 100)
        self.assertEqual(tight["memory_pressure"], "tight")
        self.assertAlmostEqual(tight["memory_ratio"], 0.9)

        over = self.kernel.memory_diagnostics(103, 100)
        self.assertEqual(over["memory_pressure"], "over_budget")
        self.assertAlmostEqual(over["memory_ratio"], 1.03)
        self.assertEqual(over["memory_headroom_bytes"], -3)
        self.assertAlmostEqual(over["memory_headroom_ratio"], -0.03)
        self.assertEqual(over["memory_overage_bytes"], 3)
        self.assertAlmostEqual(over["memory_overage_ratio"], 0.03)

    def test_exact_tie_pairs_span_tile_boundaries_and_tail(self) -> None:
        self.assertEqual(
            self.kernel.exact_tie_pairs(129),
            ((0, 1), (63, 64), (127, 128)),
        )
        self.assertEqual(
            self.kernel.exact_tie_pairs(257),
            ((0, 1), (127, 128), (255, 256)),
        )
        self.assertEqual(self.kernel.exact_tie_pairs(1), ())

    def test_every_scored_runtime_stage_retries_outlier_and_uses_median(
        self,
    ) -> None:
        for stage, weight in (("assign", 0.10), ("update", 0.10), ("full", 0.20)):
            with self.subTest(stage=stage):
                spec = self.kernel.CaseSpec(
                    stage,
                    stage,
                    "online",
                    1,
                    1024,
                    32,
                    32,
                    91,
                    weight=weight,
                )
                calls = []

                def benchmark(*args, **kwargs):
                    calls.append(kwargs)
                    return (64.0, 4) if len(calls) == 1 else (12.0, 8)

                with patch.object(
                    self.kernel,
                    "benchmark_fresh_latency_ms",
                    side_effect=benchmark,
                ):
                    measured = self.kernel.benchmark_stable_latency_ms(
                        spec,
                        object(),
                        object(),
                        warmup=1,
                        iters=3,
                        repeats=3,
                        track_memory=True,
                    )

                self.assertEqual(
                    measured["trial_medians_ms"],
                    [64.0, 12.0, 12.0],
                )
                self.assertEqual(measured["latency_ms"], 12.0)
                self.assertEqual(measured["peak_bytes"], 8)
                self.assertTrue(measured["adaptive_retry"])
                self.assertEqual(
                    [call["seed_offset"] for call in calls],
                    [10_000_000, 11_000_000, 12_000_000],
                )

    def test_scored_runtime_stops_after_two_consistent_trials(self) -> None:
        spec = self.kernel.CaseSpec(
            "full",
            "full",
            "online",
            1,
            1024,
            32,
            32,
            92,
            weight=0.20,
        )
        with patch.object(
            self.kernel,
            "benchmark_fresh_latency_ms",
            side_effect=[10.0, 10.1],
        ) as benchmark:
            measured = self.kernel.benchmark_stable_latency_ms(
                spec,
                object(),
                object(),
                warmup=1,
                iters=3,
                repeats=3,
            )
        self.assertEqual(benchmark.call_count, 2)
        self.assertEqual(measured["trial_count"], 2)
        self.assertFalse(measured["adaptive_retry"])
        self.assertTrue(measured["stable"])
        self.assertEqual(measured["policy"], "adaptive_median_2_to_3")

    def test_runtime_spread_boundary_is_inclusive(self) -> None:
        spec = self.kernel.CaseSpec(
            "assign",
            "assign",
            "online",
            1,
            1024,
            32,
            32,
            93,
            weight=0.10,
        )
        with patch.object(
            self.kernel,
            "benchmark_fresh_latency_ms",
            side_effect=[10.0, 12.5],
        ) as benchmark:
            measured = self.kernel.benchmark_stable_latency_ms(
                spec,
                object(),
                object(),
                warmup=1,
                iters=3,
                repeats=3,
            )
        self.assertEqual(benchmark.call_count, 2)
        self.assertEqual(measured["spread_ratio"], 1.25)
        self.assertTrue(measured["stable"])

    def test_runtime_three_way_instability_fails_closed(self) -> None:
        spec = self.kernel.CaseSpec(
            "update",
            "update",
            "online",
            1,
            1024,
            32,
            32,
            94,
            weight=0.10,
        )
        with patch.object(
            self.kernel,
            "benchmark_fresh_latency_ms",
            side_effect=[10.0, 20.0, 40.0],
        ):
            measured = self.kernel.benchmark_stable_latency_ms(
                spec,
                object(),
                object(),
                warmup=1,
                iters=3,
                repeats=3,
            )
        self.assertEqual(measured["trial_count"], 3)
        self.assertEqual(measured["inlier_count"], 1)
        self.assertFalse(measured["stable"])

    def test_zero_weight_and_nonruntime_cases_keep_one_manifest_trial(
        self,
    ) -> None:
        specs = (
            self.kernel.CaseSpec(
                "gate",
                "assign",
                "online",
                1,
                1024,
                32,
                32,
                95,
                weight=0.0,
            ),
            self.kernel.CaseSpec(
                "deployment",
                "full",
                "online",
                1,
                1024,
                32,
                32,
                96,
                weight=0.60,
                component="deployment_bundle",
                measurement_mode="fresh_process",
            ),
        )
        for spec in specs:
            with self.subTest(name=spec.name), patch.object(
                self.kernel,
                "benchmark_fresh_latency_ms",
                return_value=10.0,
            ) as benchmark:
                measured = self.kernel.benchmark_stable_latency_ms(
                    spec,
                    object(),
                    object(),
                    warmup=1,
                    iters=3,
                    repeats=3,
                )
            self.assertEqual(benchmark.call_count, 1)
            self.assertEqual(measured["trial_count"], 1)
            self.assertEqual(measured["policy"], "manifest_median")
            self.assertTrue(measured["stable"])

    @requires_hidden_cases
    def test_every_manifest_family_maps_to_generator_semantics(self) -> None:
        expected = {
            "gaussian_high_margin": ("online", "uniform", False),
            "balanced_shuffled": ("online", "balanced_shuffled", False),
            "multiway_tie_shuffled": ("online_tie", "uniform", True),
            "empty_clusters_shuffled": ("online", "hot99_empty", False),
            "close_centroids_secret_permutation": (
                "close_centroid",
                "uniform",
                False,
            ),
            "hot99_secret_permutation": ("online", "hot99_empty", False),
            "zipf_secret_permutation": ("online", "zipf", False),
        }
        explicit_families = tuple(
            _schema_case(
                name=f"family_{family}",
                data_family=family,
            )
            for family in expected
        )
        all_cases = hidden_cases.HIDDEN_BENCHMARK_CASES + explicit_families
        manifest_families = {case.data_family for case in all_cases}
        self.assertTrue(manifest_families.issubset(set(expected)))
        for index, case in enumerate(all_cases):
            spec = self.kernel.benchmark_case_to_spec(case, seed=70_000 + index)
            self.assertEqual(
                (spec.category, spec.distribution, spec.tie),
                expected[case.data_family],
            )

    @requires_hidden_cases
    def test_adapter_preserves_weight_timing_and_metadata(self) -> None:
        all_cases = hidden_cases.HIDDEN_BENCHMARK_CASES
        for index, case in enumerate(all_cases):
            seed = 80_000 + index
            spec = self.kernel.benchmark_case_to_spec(case, seed)
            self.assertEqual(spec.name, case.name)
            self.assertEqual(spec.stage, case.stage)
            self.assertEqual(spec.category in {"online", "online_tie", "close_centroid"}, True)
            self.assertEqual((spec.b, spec.n, spec.k, spec.d), case.shape)
            self.assertEqual(spec.seed, seed)
            self.assertEqual(spec.weight, case.weight)
            self.assertEqual(spec.score_group, case.score_group)
            self.assertEqual(
                (spec.warmup, spec.iters, spec.repeats),
                (case.warmup, case.iters, case.repeats),
            )
            self.assertEqual(spec.role, case.role)
            self.assertEqual(spec.component, case.component)
            self.assertEqual(spec.measurement_mode, case.measurement_mode)
            self.assertEqual(spec.verification, case.verification)
            self.assertEqual(spec.visibility, case.visibility)

    @requires_hidden_cases
    def test_deployment_case_uses_full_stage_and_fresh_process_mode(self) -> None:
        case = hidden_cases.HIDDEN_DEPLOYMENT_CASES[0]
        self.assertEqual(case.stage, "full")
        self.assertEqual(case.measurement_mode, "fresh_process")
        spec = self.kernel.benchmark_case_to_spec(case, seed=123)
        self.assertEqual(spec.stage, "full")
        self.assertEqual(spec.component, "deployment_bundle")
        self.assertEqual(spec.measurement_mode, "fresh_process")
        self.assertEqual(spec.role, "performance")
        self.assertEqual(spec.score_group, "deploy_control")
        self.assertEqual((spec.warmup, spec.iters, spec.repeats), (0, 1, 1))

    def test_warm_runtime_sampling_is_capped_at_130_points_and_18_clusters(
        self,
    ) -> None:
        assignment_source = inspect.getsource(
            self.kernel._assignment_sample_indices
        )
        update_source = inspect.getsource(self.kernel._check_update_sampled)
        self.assertIn("min(128,", assignment_source)
        self.assertIn("min(16, spec.k)", update_source)
        self.assertIn("torch.bincount", update_source)

        point_ids = self.kernel._deterministic_sample_ids(
            1 << 20,
            128,
            123,
        )
        cluster_ids = self.kernel._deterministic_sample_ids(
            1 << 16,
            16,
            123,
        )
        self.assertEqual(len(point_ids), 130)
        self.assertEqual((point_ids[0], point_ids[-1]), (0, (1 << 20) - 1))
        self.assertEqual(len(cluster_ids), 18)
        self.assertEqual(
            (cluster_ids[0], cluster_ids[-1]),
            (0, (1 << 16) - 1),
        )

    @requires_hidden_cases
    def test_memory_envelope_covers_repo_sort_scratch_but_rejects_bnk(self) -> None:
        calibrated = (
            (
                next(
                    case
                    for case in hidden_cases.HIDDEN_RUNTIME_CASES
                    if case.name == "hidden_batched_full"
                ),
                439_012_864,
            ),
            (
                next(
                    case
                    for case in hidden_cases.HIDDEN_RUNTIME_CASES
                    if case.name == "hidden_massive_update"
                ),
                1_152_393_728,
            ),
            (
                next(
                    case
                    for case in hidden_cases.HIDDEN_DEPLOYMENT_CASES
                    if case.name == "hidden_deployment_16m_8k_d512"
                ),
                643_825_664,
            ),
        )
        for case, measured_repo_peak in calibrated:
            spec = self.kernel.benchmark_case_to_spec(case, seed=1)
            self.assertGreaterEqual(
                self.kernel.memory_limit_bytes(spec),
                measured_repo_peak,
            )
            fp16_distance_bytes = 2 * spec.b * spec.n * spec.k
            self.assertLess(
                self.kernel.memory_limit_bytes(spec),
                fp16_distance_bytes,
            )

    def test_unknown_family_fails_closed(self) -> None:
        unknown = _schema_case(
            name="unknown_family",
            data_family="not_supported",
        )
        with self.assertRaisesRegex(
            ValueError,
            "unsupported benchmark data family",
        ):
            self.kernel.benchmark_case_to_spec(unknown, seed=1)

    def test_balanced_shuffled_is_exact_seeded_and_permuted(self) -> None:
        spec = self.kernel.CaseSpec(
            "balanced",
            "update",
            "online",
            3,
            17,
            5,
            8,
            123,
            "balanced_shuffled",
        )

        def generate(seed: int):
            generator = self.fake_torch.Generator(device="cuda")
            generator.manual_seed(seed)
            return self.kernel.make_labels(spec, generator, "cuda").tolist()

        first = generate(123)
        again = generate(123)
        different = generate(124)
        self.assertEqual(first, again)
        self.assertNotEqual(first, different)
        self.assertNotEqual(first[0], [index % spec.k for index in range(spec.n)])
        self.assertEqual(len({tuple(row) for row in first}), spec.b)
        for row in first:
            counts = [row.count(cluster) for cluster in range(spec.k)]
            self.assertLessEqual(max(counts) - min(counts), 1)
            self.assertEqual(sum(counts), spec.n)

    def test_normalize_uses_replace_and_preserves_metadata(self) -> None:
        specs = [
            self.kernel.CaseSpec(
                f"case_{stage}",
                stage,
                "online",
                1,
                32,
                8,
                16,
                index,
                warmup=2,
                iters=3,
                repeats=4,
                component=self.kernel.COMPONENT_BY_STAGE[stage],
                visibility="hidden",
            )
            for index, stage in enumerate(("assign", "update", "full"))
        ]
        normalized = self.kernel.normalize_stage_weights(specs)
        self.assertEqual(
            [spec.weight for spec in normalized],
            [
                self.kernel.ASSIGN_WEIGHT,
                self.kernel.UPDATE_WEIGHT,
                self.kernel.FULL_WEIGHT,
            ],
        )
        for before, after in zip(specs, normalized):
            self.assertIsNot(before, after)
            self.assertEqual((after.warmup, after.iters, after.repeats), (2, 3, 4))
            self.assertEqual(after.component, before.component)
            self.assertEqual(after.visibility, "hidden")


if __name__ == "__main__":
    unittest.main()
