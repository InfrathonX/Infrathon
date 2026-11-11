from __future__ import annotations

import ast
import math
import re
import sys
import unittest
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_DIR = ROOT / "images" / "work" / "public_tests"
PUBLIC_SOURCE = PUBLIC_DIR / "public_test.py"
PUBLIC_WORKER = PUBLIC_DIR / "public_fresh_worker.py"
PUBLIC_RUNNER = PUBLIC_DIR / "run_public_tests.sh"
sys.path.insert(0, str(ROOT / "images" / "judge" / "benchmark"))
sys.path.insert(0, str(ROOT / "images" / "judge"))

from _judge_assets import load_hidden_cases, requires_hidden_cases  # noqa: E402

hidden_cases = load_hidden_cases()


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    matches = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    if len(matches) != 1:
        raise AssertionError(f"expected exactly one {name} function")
    return matches[0]


def _call_name(call: ast.Call) -> str:
    parts: list[str] = []
    node: ast.expr = call.func
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _calls(node: ast.AST) -> set[str]:
    return {
        _call_name(child)
        for child in ast.walk(node)
        if isinstance(child, ast.Call)
    }


def _dict_string_keys(node: ast.Dict) -> dict[str, ast.expr]:
    result: dict[str, ast.expr] = {}
    for key, value in zip(node.keys, node.values):
        if isinstance(key, ast.Constant) and isinstance(key.value, str):
            result[key.value] = value
    return result


def _literal_assignment(tree: ast.Module, name: str):
    for statement in tree.body:
        target: ast.expr | None = None
        value: ast.expr | None = None
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target = statement.targets[0]
            value = statement.value
        elif isinstance(statement, ast.AnnAssign):
            target = statement.target
            value = statement.value
        if isinstance(target, ast.Name) and target.id == name and value is not None:
            return ast.literal_eval(value)
    raise AssertionError(f"missing literal assignment {name}")


def _load_function(path: Path, name: str) -> Callable[..., Any]:
    source = path.read_text(encoding="utf-8")
    segment = ast.get_source_segment(source, _function(_tree(path), name))
    assert segment is not None
    namespace: dict[str, Any] = {"math": math, "Any": Any}
    exec(
        compile(
            "from __future__ import annotations\n" + segment,
            f"{path}:{name}",
            "exec",
        ),
        namespace,
    )
    function = namespace[name]
    assert callable(function)
    return function


class VisiblePublicAssetTests(unittest.TestCase):
    def test_visible_public_assets_are_exactly_the_declared_self_contained_files(
        self,
    ) -> None:
        self.assertEqual(
            {path.name for path in PUBLIC_DIR.iterdir() if path.is_file()},
            {
                "public_fresh_worker.py",
                "public_test.py",
                "prepare_submit.sh",
                "run_public_tests.sh",
            },
        )

        runner = PUBLIC_RUNNER.read_text(encoding="utf-8")
        self.assertIn("timeout 1500s", runner)
        self.assertIn("/workspace/public_tests/public_test.py", runner)
        self.assertIn("--root /app/workdir/flash_kmeans_lab", runner)
        for private_path in (
            "/opt/judge",
            "/opt/flash_kmeans_benchmark",
            "/workspace/benchmark",
        ):
            with self.subTest(private_path=private_path):
                self.assertNotIn(private_path, runner)

    def test_public_python_imports_no_private_evaluator_module(self) -> None:
        forbidden = {
            "benchmark_cases",
            "deployment_bench",
            "hidden_cases",
            "kernel_bench",
            "paired_timing_reference",
            "source_admission",
            "verify_flash_kmeans",
        }
        for path in (PUBLIC_SOURCE, PUBLIC_WORKER):
            imported: set[str] = set()
            for node in ast.walk(_tree(path)):
                if isinstance(node, ast.Import):
                    imported.update(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported.add(node.module)
            with self.subTest(path=path.name):
                self.assertTrue(imported.isdisjoint(forbidden), imported & forbidden)

    def test_public_harness_has_sampled_verifiers_but_no_reusable_candidate(
        self,
    ) -> None:
        combined = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (PUBLIC_SOURCE, PUBLIC_WORKER)
        )
        self.assertNotIn("import triton", combined)
        self.assertNotIn("@triton", combined)
        self.assertNotIn("host_reference", combined)
        self.assertNotIn("official_oracle", combined)
        self.assertNotIn("torch_fallback", combined)

        definitions: dict[str, ast.FunctionDef] = {}
        for path in (PUBLIC_SOURCE, PUBLIC_WORKER):
            for node in _tree(path).body:
                if isinstance(node, ast.FunctionDef):
                    definitions[node.name] = node
        self.assertNotIn("assign", definitions)
        self.assertNotIn("update", definitions)
        for forbidden in (
            "reference_assign",
            "reference_update",
            "standard_blockwise_assign",
            "standard_scatter_update",
            "baseline_assign",
            "baseline_update",
        ):
            self.assertNotIn(forbidden, definitions)

        assignment_check = definitions["sampled_assignment_reference"]
        assignment_args = {arg.arg for arg in assignment_check.args.args}
        self.assertTrue({"labels", "indices"}.issubset(assignment_args))
        self.assertIn("index_select", {
            name.rsplit(".", 1)[-1] for name in _calls(assignment_check)
        })

        update_check = definitions["sampled_update_reference"]
        update_args = {arg.arg for arg in update_check.args.args}
        self.assertTrue(
            {"labels", "counts", "new_centroids"}.issubset(update_args)
        )
        # This is deliberately a validator over selected clusters, not an ABI
        # implementation that returns a complete candidate result.
        self.assertIn("guard_indices", _calls(update_check))
        self.assertIn("torch.bincount", _calls(update_check))
        returns = [
            node
            for node in ast.walk(update_check)
            if isinstance(node, ast.Return)
        ]
        self.assertTrue(returns)
        self.assertTrue(
            all(isinstance(node.value, ast.Tuple) for node in returns)
        )


class LiteralAndSteadyStateTests(unittest.TestCase):
    def test_tiny_correctness_oracles_remain_literal_golden_tensors(self) -> None:
        tree = _tree(PUBLIC_SOURCE)
        required_literals = {
            "golden_assignment": {"x", "centroids", "expected"},
            "golden_update": {
                "x",
                "labels",
                "old",
                "expected_counts",
                "expected_centroids",
            },
            "golden_combined": {
                "x",
                "old",
                "expected_counts",
                "expected_centroids",
            },
        }
        candidate_calls = {
            "golden_assignment": "assign",
            "golden_update": "update",
            "golden_combined": "assign",
        }
        for function_name, expected_names in required_literals.items():
            function = _function(tree, function_name)
            literal_assignments: dict[str, int] = {}
            candidate_call_line: int | None = None
            for node in ast.walk(function):
                if isinstance(node, ast.Assign) and len(node.targets) == 1:
                    target = node.targets[0]
                    if (
                        isinstance(target, ast.Name)
                        and isinstance(node.value, ast.Call)
                        and _call_name(node.value) == "torch.tensor"
                        and node.value.args
                        and isinstance(node.value.args[0], (ast.List, ast.Tuple))
                    ):
                        literal_assignments[target.id] = node.lineno
                if (
                    isinstance(node, ast.Call)
                    and _call_name(node) == candidate_calls[function_name]
                ):
                    candidate_call_line = min(
                        node.lineno,
                        candidate_call_line or node.lineno,
                    )

            with self.subTest(function=function_name):
                self.assertTrue(
                    expected_names.issubset(literal_assignments),
                    expected_names - literal_assignments.keys(),
                )
                self.assertIsNotNone(candidate_call_line)
                for expected_name in (
                    name for name in expected_names if name.startswith("expected")
                ):
                    self.assertLess(
                        literal_assignments[expected_name],
                        int(candidate_call_line),
                    )

    def test_steady_state_performance_path_executes_only_candidate_entrypoints(
        self,
    ) -> None:
        tree = _tree(PUBLIC_SOURCE)
        call_stage = _function(tree, "call_stage")
        self.assertEqual(_calls(call_stage), {"assign", "update"})

        probe = _function(tree, "performance_probe")
        names = {
            node.id.lower()
            for node in ast.walk(probe)
            if isinstance(node, ast.Name)
        }
        for forbidden in (
            "baseline",
            "denominator",
            "oracle",
            "reference",
            "speedup",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, names)
        self.assertTrue(
            _calls(probe).isdisjoint(
                {"sampled_assignment_reference", "sampled_update_reference"}
            )
        )


class FreshProcessStructureTests(unittest.TestCase):
    def test_each_proxy_uses_a_new_process_and_independent_empty_caches(self) -> None:
        tree = _tree(PUBLIC_SOURCE)
        source = PUBLIC_SOURCE.read_text(encoding="utf-8")
        probe = _function(tree, "fresh_process_probe")
        calls = _calls(probe)
        self.assertIn("tempfile.mkdtemp", calls)
        self.assertIn("subprocess.Popen", calls)
        self.assertIn("fresh_cache_environment", calls)
        self.assertIn("proc.wait", calls)
        self.assertIn("os.killpg", calls)
        self.assertIn("shutil.rmtree", calls)
        self.assertIn("start_new_session=True", source)
        self.assertIn("timeout=FRESH_CASE_TIMEOUT_SECONDS", source)
        self.assertIn("FRESH_CASE_TIMEOUT_SECONDS = 180", source)
        self.assertIn("signal.SIGKILL", source)

        cache_source = ast.get_source_segment(
            source, _function(tree, "fresh_cache_environment")
        )
        assert cache_source is not None
        for variable in (
            "TRITON_CACHE_DIR",
            "CUDA_CACHE_PATH",
            "TORCHINDUCTOR_CACHE_DIR",
            "XDG_CACHE_HOME",
            "TMPDIR",
            "TORCH_EXTENSIONS_DIR",
        ):
            self.assertIn(variable, cache_source)

        worker_source = PUBLIC_WORKER.read_text(encoding="utf-8")
        self.assertIn("arm_parent_death_signal()", worker_source)
        self.assertIn("validate_empty_cache_root(args.cache_root)", worker_source)
        self.assertIn("any(actual.iterdir())", worker_source)

    def test_full_and_quick_case_lists_have_the_intended_scope(self) -> None:
        tree = _tree(PUBLIC_SOURCE)
        self.assertEqual(
            _literal_assignment(tree, "FRESH_CASES_FULL"),
            (
                "control",
                "large_k_small",
                "large_k_large",
                "large_n_high_d_small",
                "large_n_high_d_large",
            ),
        )
        self.assertEqual(
            _literal_assignment(tree, "FRESH_CASES_QUICK"),
            ("control", "large_k_small", "large_n_high_d_small"),
        )

        main = _function(tree, "main")
        main_calls = _calls(main)
        self.assertIn("fresh_process_probe", main_calls)
        self.assertIn("scale_ladders", main_calls)


class PublicShapeAndLadderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.worker_tree = _tree(PUBLIC_WORKER)
        cls.cases = _literal_assignment(cls.worker_tree, "PUBLIC_CASES")

    def test_full_control_and_two_family_ladders_are_exact(self) -> None:
        expected = {
            "control": {
                "family": "control",
                "rung": "control",
                "shape": (1, 1_048_576, 8_192, 512),
            },
            "large_k_small": {
                "family": "large-K",
                "rung": "small",
                "shape": (1, 393_216, 49_152, 160),
            },
            "large_k_large": {
                "family": "large-K",
                "rung": "large",
                "shape": (1, 1_048_576, 49_152, 160),
            },
            "large_n_high_d_small": {
                "family": "large-N/high-D",
                "rung": "small",
                "shape": (1, 3_145_728, 6_144, 384),
            },
            "large_n_high_d_large": {
                "family": "large-N/high-D",
                "rung": "large",
                "shape": (1, 6_291_456, 6_144, 384),
            },
        }
        self.assertEqual(set(self.cases), set(expected))
        for case_id, contract in expected.items():
            with self.subTest(case_id=case_id):
                self.assertEqual(self.cases[case_id]["family"], contract["family"])
                self.assertEqual(self.cases[case_id]["rung"], contract["rung"])
                self.assertEqual(self.cases[case_id]["full"], contract["shape"])

        for small_id, large_id in (
            ("large_k_small", "large_k_large"),
            ("large_n_high_d_small", "large_n_high_d_large"),
        ):
            small = self.cases[small_id]["full"]
            large = self.cases[large_id]["full"]
            self.assertGreater(math.prod(large), math.prod(small))
            self.assertGreater(large[1], small[1])

    @requires_hidden_cases
    def test_all_public_quick_and_full_shapes_are_disjoint_from_hidden(self) -> None:
        public_shapes = {
            tuple(case[mode])
            for case in self.cases.values()
            for mode in ("quick", "full")
        }
        hidden_shapes = {
            case.shape for case in hidden_cases.HIDDEN_BENCHMARK_CASES
        }
        self.assertTrue(public_shapes.isdisjoint(hidden_shapes))


class SampledCorrectnessContractTests(unittest.TestCase):
    def test_large_verifier_checks_sampled_assignment_exact_counts_and_centroids(
        self,
    ) -> None:
        tree = _tree(PUBLIC_WORKER)
        source = PUBLIC_WORKER.read_text(encoding="utf-8")
        guard = _function(tree, "guard_indices")
        defaults = [ast.literal_eval(value) for value in guard.args.defaults]
        self.assertIn(257, defaults)

        assignment = _function(tree, "sampled_assignment_reference")
        self.assertIn("torch.equal", _calls(assignment))
        self.assertIn("index_select", {
            name.rsplit(".", 1)[-1] for name in _calls(assignment)
        })
        self.assertIn("indices", {arg.arg for arg in assignment.args.args})

        update = _function(tree, "sampled_update_reference")
        update_calls = _calls(update)
        self.assertIn("torch.bincount", update_calls)
        self.assertIn("torch.equal", update_calls)
        self.assertIn("guard_indices", update_calls)
        self.assertIn("sums.index_add_", update_calls)
        self.assertIn("new_centroids", {arg.arg for arg in update.args.args})

        verify = _function(tree, "verify_iteration")
        verify_calls = _calls(verify)
        self.assertIn("sampled_assignment_reference", verify_calls)
        self.assertIn("sampled_update_reference", verify_calls)
        self.assertIn("inputs_unchanged", verify_calls)

        for field in (
            '"assignment_points_checked"',
            '"counts_checked": "all"',
            '"centroid_clusters_checked"',
            '"first_and_every_warmed_checked": True',
        ):
            self.assertIn(field, source)

    def test_first_run_and_every_warmed_iteration_are_verified(self) -> None:
        tree = _tree(PUBLIC_WORKER)
        source = PUBLIC_WORKER.read_text(encoding="utf-8")
        execute = _function(tree, "execute")
        verify_calls = [
            node
            for node in ast.walk(execute)
            if isinstance(node, ast.Call)
            and _call_name(node) == "verify_iteration"
        ]
        self.assertEqual(len(verify_calls), 2)
        self.assertIn("for repeat in range(warmed_repeats):", source)
        self.assertIn("seed = int(case[\"seed\"]) + 1_000_000 + repeat", source)
        self.assertIn("data = make_inputs(shape, seed)", source)
        self.assertIn("first-run sampled correctness", source)
        self.assertIn("warmed sampled correctness", source)


class DiagnosticMetricTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.effective_assignment_tflops = staticmethod(
            _load_function(PUBLIC_WORKER, "effective_assignment_tflops")
        )
        cls.scale_ladders = staticmethod(
            _load_function(PUBLIC_SOURCE, "scale_ladders")
        )

    def test_effective_tflops_is_raw_and_has_no_gpu_specific_band(self) -> None:
        shape = (1, 1_000_000, 1_000_000, 1)
        for latency_ms in (10.0, 20.0, 100.0, 1000.0):
            with self.subTest(latency_ms=latency_ms):
                tflops = self.effective_assignment_tflops(shape, latency_ms)
                self.assertTrue(math.isfinite(tflops) and tflops > 0.0)
                self.assertAlmostEqual(tflops, 2000.0 / latency_ms)

        execute_source = ast.get_source_segment(
            PUBLIC_WORKER.read_text(encoding="utf-8"),
            _function(_tree(PUBLIC_WORKER), "execute"),
        )
        assert execute_source is not None
        self.assertIn('"status": "passed" if memory_ok else "failed"', execute_source)
        self.assertIn('"effective_assignment_tflops": tflops', execute_source)
        self.assertNotIn("performance_band", PUBLIC_WORKER.read_text(encoding="utf-8"))
        # The host smoke reads the same rows; it must not demand the removed bucket.
        smoke = (PUBLIC_WORKER.parents[3] / "scripts" / "smoke_test.sh").read_text(encoding="utf-8")
        self.assertNotIn("performance_band", smoke)

    def test_scale_ladder_reports_throughput_retention_in_four_bands(self) -> None:
        examples = {
            2.0: "healthy",       # 2x work / 2x latency = 1.0
            3.0: "moderate",      # 2/3 retention
            6.0: "poor",          # 1/3 retention
            10.0: "pathological", # 1/5 retention
        }
        for large_ms, expected in examples.items():
            rows = [
                {
                    "family": "large-K",
                    "rung": "small",
                    "status": "passed",
                    "shape": [1, 10, 10, 1],
                    "warmed_iteration_ms": 1.0,
                },
                {
                    "family": "large-K",
                    "rung": "large",
                    "status": "passed",
                    "shape": [1, 20, 10, 1],
                    "warmed_iteration_ms": large_ms,
                },
            ]
            result = next(
                row
                for row in self.scale_ladders(rows)
                if row["family"] == "large-K"
            )
            with self.subTest(large_ms=large_ms):
                self.assertEqual(result["status"], "available")
                self.assertEqual(result["work_multiplier"], 2.0)
                self.assertAlmostEqual(
                    result["throughput_retention"], 2.0 / large_ms
                )
                self.assertEqual(result["scaling_band"], expected)


class PublicResultBoundaryTests(unittest.TestCase):
    def test_public_result_is_non_authoritative_and_unscored(self) -> None:
        tree = _tree(PUBLIC_SOURCE)
        main = _function(tree, "main")
        initial_result: ast.Dict | None = None
        for statement in main.body:
            if (
                isinstance(statement, ast.AnnAssign)
                and isinstance(statement.target, ast.Name)
                and statement.target.id == "result"
                and isinstance(statement.value, ast.Dict)
            ):
                initial_result = statement.value
                break
        self.assertIsNotNone(initial_result)
        fields = _dict_string_keys(initial_result)  # type: ignore[arg-type]
        self.assertNotIn("score", fields)
        self.assertIsInstance(fields.get("authoritative"), ast.Constant)
        self.assertIs(fields["authoritative"].value, False)  # type: ignore[union-attr]
        self.assertEqual(ast.literal_eval(fields["schema_version"]), 3)
        self.assertEqual(
            ast.literal_eval(fields["deployment_protocol"]),
            "public_fresh_process_v1",
        )

        forbidden_keys: set[str] = set()
        for path in (PUBLIC_SOURCE, PUBLIC_WORKER):
            for node in ast.walk(_tree(path)):
                if not isinstance(node, ast.Dict):
                    continue
                keys = _dict_string_keys(node)
                forbidden_keys.update(
                    keys.keys()
                    & {
                        "baseline_ms",
                        "denominator",
                        "denominator_ms",
                        "reference_ms",
                        "score",
                        "speedup",
                        "weighted_score",
                    }
                )
        self.assertEqual(forbidden_keys, set())

    def test_public_candidate_loader_is_scoped_to_versioned_kernel(self) -> None:
        text = PUBLIC_SOURCE.read_text(encoding="utf-8")
        self.assertIn('root / "kernels"', text)
        self.assertIn('root / "FINAL_VERSION"', text)
        self.assertIn("flash_kmeans_v{version:03d}.py", text)
        self.assertIn("candidate must export callable assign and update", text)
        for forbidden in ("host_reference", "official_oracle", "torch_fallback"):
            self.assertNotIn(forbidden, text)


if __name__ == "__main__":
    unittest.main()
