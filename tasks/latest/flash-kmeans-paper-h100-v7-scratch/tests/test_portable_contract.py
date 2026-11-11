from __future__ import annotations

import ast
import importlib.util
import json
import re
import sys
import unittest
from fractions import Fraction
from pathlib import Path

from sforge.harness.portable_spec import parse_portable_spec, verify_sha256_file

import _judge_assets


ROOT = Path(__file__).resolve().parents[1]
TASK_PATH = ROOT / "tasks/flash_kmeans_paper_h100_kernel_lab_v7_scratch.json"
PORTABLE = ROOT / "portable"


def _literal_assignment(path: Path, name: str):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == name
        ) or (
            isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"missing literal assignment {name} in {path}")


def _full_probe_shapes() -> dict[str, tuple[int, int, int, int]]:
    path = ROOT / "images/work/public_tests/public_test.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name != "make_probe":
            continue
        candidates = []
        for child in ast.walk(node):
            if isinstance(child, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "shapes"
                for target in child.targets
            ):
                candidates.append(ast.literal_eval(child.value))
        return next(
            value
            for value in candidates
            if value["assign"] == (1, 327_680, 3_072, 128)
        )
    raise AssertionError("make_probe full shapes not found")


def _tensor_shape(value: object) -> tuple[int, ...]:
    shape = []
    while isinstance(value, list):
        shape.append(len(value))
        value = value[0]
    return tuple(shape)


def _golden_shapes() -> dict[str, tuple[int, int, int, int]]:
    path = ROOT / "images/work/public_tests/public_test.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    result = {}
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or node.name not in {
            "golden_assignment",
            "golden_update",
            "golden_combined",
        }:
            continue
        tensors = {}
        for child in node.body:
            if (
                isinstance(child, ast.Assign)
                and len(child.targets) == 1
                and isinstance(child.targets[0], ast.Name)
                and isinstance(child.value, ast.Call)
                and child.value.args
            ):
                try:
                    tensors[child.targets[0].id] = _tensor_shape(
                        ast.literal_eval(child.value.args[0])
                    )
                except (ValueError, TypeError):
                    pass
        x_shape = tensors["x"]
        centroid_shape = tensors.get("centroids", tensors.get("old"))
        assert centroid_shape is not None
        result[node.name] = (
            x_shape[0],
            x_shape[1],
            centroid_shape[1],
            x_shape[2],
        )
    return result


def _hidden_cases():
    benchmark_dir = ROOT / "images/judge/benchmark"
    hidden_path = _judge_assets.resolve("images/judge/hidden_cases.py")
    if hidden_path is None:
        raise unittest.SkipTest(_judge_assets.SKIP_REASON)
    prior_benchmark = sys.modules.pop("benchmark_cases", None)
    prior_no_bytecode = sys.dont_write_bytecode
    sys.path.insert(0, str(benchmark_dir))
    sys.dont_write_bytecode = True
    try:
        spec = importlib.util.spec_from_file_location(
            "flash_kmeans_portable_hidden_cases", hidden_path
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.HIDDEN_BENCHMARK_CASES
    finally:
        sys.dont_write_bytecode = prior_no_bytecode
        sys.path.remove(str(benchmark_dir))
        sys.modules.pop("benchmark_cases", None)
        if prior_benchmark is not None:
            sys.modules["benchmark_cases"] = prior_benchmark


def _full_hidden_specs():
    loader_path = ROOT / "tests/test_hidden_benchmark_protocol.py"
    module_name = "flash_kmeans_portable_verifier_loader"
    prior_module = sys.modules.get(module_name)
    prior_path = list(sys.path)
    try:
        spec = importlib.util.spec_from_file_location(
            module_name, loader_path
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        verifier = module._load_verifier()
        return (
            verifier.hidden_runtime_specs("full")
            + verifier.hidden_deployment_specs("full")
        )
    finally:
        sys.path[:] = prior_path
        if prior_module is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = prior_module


def _deployment_warmed_repeats() -> int:
    verifier_path = ROOT / "images/judge/verify_flash_kmeans.py"
    tree = ast.parse(
        verifier_path.read_text(encoding="utf-8"), filename=str(verifier_path)
    )
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "deployment_pass"
    )
    call = next(
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "run_deployment_case"
    )
    keyword = next(
        item for item in call.keywords if item.arg == "warmed_repeats"
    )
    value = ast.literal_eval(keyword.value)
    if not isinstance(value, int):
        raise AssertionError("deployment warmed_repeats must be a literal integer")
    return value


class PortableContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.raw_task = json.loads(TASK_PATH.read_text(encoding="utf-8"))
        cls.portable = parse_portable_spec(
            cls.raw_task["portable"], task_id=cls.raw_task["task_id"]
        )

    def test_every_declared_digest_is_the_actual_file_digest(self) -> None:
        suite = self.portable.evaluation_suites["v7-family-tail60"]
        profile = self.portable.execution_profiles["single-cuda-sm80-80g"]
        declarations = {
            "resource-manifest.json": self.portable.task_core.resource_manifest_sha256,
            "public-suite-manifest.json": suite.roles["public"].manifest_sha256,
            "final-suite-manifest.json": suite.roles["final"].manifest_sha256,
            "oracle-protocol.json": suite.oracle.spec_sha256,
            "selection-protocol.json": suite.selection.spec_sha256,
            "software-abi.json": profile.software_abi.spec_sha256,
            "calibration-protocol.json": profile.calibration_protocol.spec_sha256,
        }
        for filename, declared in declarations.items():
            with self.subTest(filename=filename):
                self.assertEqual(
                    verify_sha256_file(PORTABLE / filename, declared), declared
                )

    def test_resource_manifest_hashes_every_real_resource(self) -> None:
        manifest = json.loads(
            (PORTABLE / "resource-manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["schema_version"], 1)
        expected_paths = {
            "images/work/AGENT_TASK.md",
            "images/work/workdir/flash_kmeans_lab/README_AGENT.md",
            "images/work/public_tests/run_public_tests.sh",
            "images/work/public_tests/public_test.py",
            "images/work/public_tests/public_fresh_worker.py",
            "images/judge/protected_hashes.txt",
            "images/judge/hidden_cases.py",
            "images/judge/verify_flash_kmeans.py",
            "images/judge/run_judge.py",
            "judge/run_judge.py",
            "images/judge/benchmark/benchmark_cases.py",
            "images/judge/benchmark/kernel_bench.py",
            "images/judge/benchmark/deployment_bench.py",
            "images/judge/benchmark/source_admission.py",
        }
        self.assertEqual(
            {resource["path"] for resource in manifest["resources"]},
            expected_paths,
        )
        for resource in manifest["resources"]:
            with self.subTest(path=resource["path"]):
                verify_sha256_file(
                    _judge_assets.require(self, resource["path"]),
                    resource["sha256"],
                )

    def test_build_manifest_covers_exact_portable_delivery_inputs(self) -> None:
        path = PORTABLE / "build-resource-manifest.json"
        declared = self.raw_task["delivery"][
            "build_resource_manifest_sha256"
        ]
        self.assertEqual(verify_sha256_file(path, declared), declared)
        manifest = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(
            manifest["boundary"]["kind"], "portable_image_delivery_inputs"
        )
        expected_paths = {
            "tasks/BENCHMARK.yaml",
            "images/work/.dockerignore",
            "images/work/Dockerfile",
            "images/work/AGENT_TASK.md",
            "images/work/codex-config.toml",
            "images/work/public_tests/prepare_submit.sh",
            "images/work/public_tests/public_fresh_worker.py",
            "images/work/public_tests/public_test.py",
            "images/work/public_tests/run_public_tests.sh",
            "images/work/workdir/flash_kmeans_lab/README_AGENT.md",
            "images/work/workdir/flash_kmeans_lab/notes.md",
            "images/judge/.dockerignore",
            "images/judge/Dockerfile",
            "images/judge/Dockerfile.private",
            "images/judge/verify_flash_kmeans.py",
            "images/judge/run_judge.py",
            "images/judge/hidden_cases.py",
            "images/judge/protected_hashes.txt",
            "images/judge/benchmark/benchmark_cases.py",
            "images/judge/benchmark/deployment_bench.py",
            "images/judge/benchmark/kernel_bench.py",
            "images/judge/benchmark/source_admission.py",
            "judge/run_judge.py",
            "scripts/build_images.sh",
            "scripts/lib.sh",
            "scripts/validate_layout.py",
            "portable/REBUILD_PROMPT.md",
        }
        declared_paths = {
            resource["path"] for resource in manifest["resources"]
        }
        self.assertEqual(declared_paths, expected_paths)
        for context_name in ("work", "judge"):
            context = ROOT / f"images/{context_name}"
            actual_context_files = {
                file.relative_to(ROOT).as_posix()
                for file in context.rglob("*")
                if file.is_file()
                and "__pycache__" not in file.parts
                and file.suffix != ".pyc"
            }
            # Dataset assets are staged into the build context from the asset root
            # when the repository does not carry them.
            actual_context_files |= {
                resource
                for resource in declared_paths
                if resource.startswith(f"images/{context_name}/")
                and _judge_assets.is_dataset_copy(resource)
            }
            self.assertEqual(
                {
                    resource
                    for resource in declared_paths
                    if resource.startswith(f"images/{context_name}/")
                },
                actual_context_files,
            )
        for resource in manifest["resources"]:
            with self.subTest(delivery_path=resource["path"]):
                verify_sha256_file(
                    _judge_assets.require(self, resource["path"]),
                    resource["sha256"],
                )

    def test_host_private_final_manifest_cannot_enter_the_work_image(self) -> None:
        final_manifest = PORTABLE / "final-suite-manifest.json"
        work_context = ROOT / "images/work"
        self.assertFalse(final_manifest.is_relative_to(work_context))
        dockerfile = (work_context / "Dockerfile").read_text(encoding="utf-8")
        self.assertNotIn("portable", dockerfile.casefold())
        self.assertNotIn("final-suite-manifest", dockerfile.casefold())

    def test_public_manifest_is_the_actual_eleven_case_suite(self) -> None:
        manifest = json.loads(
            (PORTABLE / "public-suite-manifest.json").read_text(encoding="utf-8")
        )
        cases = manifest["cases"]
        suite_role = self.portable.evaluation_suites[
            "v7-family-tail60"
        ].roles["public"]
        self.assertEqual(len(cases), suite_role.case_count)
        self.assertEqual(len(cases), 11)

        indexed = {case["case_id"]: case for case in cases}
        self.assertEqual(
            {name: tuple(indexed[name]["shape"]) for name in _golden_shapes()},
            _golden_shapes(),
        )
        full_probes = _full_probe_shapes()
        for stage, shape in full_probes.items():
            self.assertEqual(
                tuple(indexed[f"{stage}_candidate_only_probe"]["shape"]),
                shape,
            )

        fresh_path = ROOT / "images/work/public_tests/public_fresh_worker.py"
        actual_fresh = _literal_assignment(fresh_path, "PUBLIC_CASES")
        for case_id, value in actual_fresh.items():
            with self.subTest(case_id=case_id):
                declared = indexed[case_id]
                self.assertEqual(tuple(declared["shape"]), value["full"])
                self.assertEqual(declared["family"], value["family"])
                self.assertEqual(declared["seed"], value["seed"])

    def test_suite_and_orchestration_timeout_budgets_are_source_derived(self) -> None:
        public_manifest = json.loads(
            (PORTABLE / "public-suite-manifest.json").read_text(
                encoding="utf-8"
            )
        )
        public_source = ROOT / "images/work/public_tests/public_test.py"
        self.assertEqual(
            public_manifest["qualification_timeouts"]["fresh_case_seconds"],
            _literal_assignment(public_source, "FRESH_CASE_TIMEOUT_SECONDS"),
        )
        public_runner = (
            ROOT / "images/work/public_tests/run_public_tests.sh"
        ).read_text(encoding="utf-8")
        wrapper_timeout = re.search(r"\btimeout\s+([0-9]+)s\b", public_runner)
        self.assertIsNotNone(wrapper_timeout)
        self.assertEqual(
            public_manifest["qualification_timeouts"][
                "public_wrapper_seconds"
            ],
            int(wrapper_timeout.group(1)),
        )

        final_manifest = json.loads(
            (PORTABLE / "final-suite-manifest.json").read_text(
                encoding="utf-8"
            )
        )
        judge_dockerfile = (ROOT / "images/judge/Dockerfile").read_text(
            encoding="utf-8"
        )
        docker_env = {
            name: int(value)
            for name, value in re.findall(
                r"(?m)^ENV\s+(FLASH_KMEANS_[A-Z_]+)=([0-9]+)\s*$",
                judge_dockerfile,
            )
        }
        self.assertEqual(
            final_manifest["suite_gate_timeouts"][
                "deployment_case_seconds"
            ],
            docker_env["FLASH_KMEANS_DEPLOYMENT_CASE_TIMEOUT"],
        )
        self.assertEqual(
            final_manifest["suite_gate_timeouts"]["verifier_seconds"],
            docker_env["FLASH_KMEANS_VERIFY_TIMEOUT"],
        )
        final_timeout = self.raw_task["judge"]["eval_timeout"]
        self.assertEqual(
            final_manifest["orchestration_timeouts"][
                "standalone_final_evaluation_seconds"
            ],
            final_timeout,
        )
        fixed_runner = (ROOT / "scripts/run_fixed_budget.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn(f"--timeout {final_timeout}", fixed_runner)

    def test_final_manifest_matches_all_twenty_two_hidden_cases(self) -> None:
        manifest = json.loads(
            (PORTABLE / "final-suite-manifest.json").read_text(encoding="utf-8")
        )
        cases = manifest["cases"]
        suite_role = self.portable.evaluation_suites[
            "v7-family-tail60"
        ].roles["final"]
        hidden = _full_hidden_specs()
        self.assertEqual(len(cases), suite_role.case_count)
        self.assertEqual(len(cases), len(hidden))
        self.assertEqual(len(cases), 22)
        performance = [case for case in cases if case["role"] == "performance"]
        gates = [case for case in cases if case["role"] == "correctness"]
        self.assertEqual((len(performance), len(gates)), (14, 8))
        self.assertEqual(
            manifest["case_counts"],
            {
                "total": 22,
                "scored_performance": 14,
                "zero_weight_hard_gate": 8,
            },
        )
        warmed_repeats = _deployment_warmed_repeats()
        self.assertEqual(warmed_repeats, 3)
        for declared, actual in zip(cases, hidden):
            with self.subTest(case_id=actual.name):
                self.assertEqual(declared["case_id"], actual.name)
                self.assertEqual(
                    tuple(declared["shape"]),
                    (actual.b, actual.n, actual.k, actual.d),
                )
                for field in (
                    "stage",
                    "component",
                    "category",
                    "distribution",
                    "tie",
                    "seed",
                    "measurement_mode",
                    "role",
                    "score_group",
                    "verification",
                ):
                    self.assertEqual(declared[field], getattr(actual, field))
                weight = Fraction(
                    declared["weight"]["numerator"],
                    declared["weight"]["denominator"],
                )
                self.assertEqual(weight, Fraction(actual.weight).limit_denominator())
                if actual.role == "correctness":
                    self.assertEqual(declared["timing"], "not_measured")
                    self.assertEqual(weight, 0)
                elif actual.measurement_mode == "fresh_process":
                    self.assertEqual(declared["warmup"], actual.warmup)
                    self.assertEqual(declared["iters"], actual.iters)
                    self.assertEqual(declared["first_run_calls"], 1)
                    self.assertEqual(declared["repeats"], warmed_repeats)
                    self.assertEqual(
                        declared["scored_statistic"],
                        "median_of_warmed_iterations",
                    )
                else:
                    for field in ("warmup", "iters", "repeats"):
                        self.assertEqual(declared[field], getattr(actual, field))

        selection = json.loads(
            (PORTABLE / "selection-protocol.json").read_text(encoding="utf-8")
        )["final"]
        self.assertEqual(selection["case_count"], 22)
        self.assertEqual(selection["scored_case_count"], 14)
        self.assertEqual(selection["hard_gate_case_count"], 8)
        self.assertEqual(selection["hard_gate_policy"], "all_must_pass")

    def test_portable_layers_preserve_metric_and_use_capabilities(self) -> None:
        metric = self.portable.metrics["family-first-latency"]
        self.assertEqual(metric.direction, "minimize")
        self.assertEqual(metric.unit, "milliseconds")
        self.assertEqual(metric.comparability, "same_measurement_profile")
        self.assertEqual(metric.baseline_policy, "none")
        self.assertEqual(
            metric.aggregation.spec["group_weight_basis_points"],
            {
                "assign": 1000,
                "update": 1000,
                "full": 2000,
                "deploy_control": 1000,
                "deploy_large_k": 2500,
                "deploy_large_n_high_d": 2500,
            },
        )
        profile = self.portable.execution_profiles["single-cuda-sm80-80g"]
        self.assertEqual(profile.vendor, "nvidia")
        self.assertEqual(profile.count, 1)
        self.assertEqual(profile.min_memory_bytes, 79 * 1024**3)
        self.assertEqual(profile.min_compute_capability, "8.0")
        portable_text = json.dumps(self.raw_task["portable"], sort_keys=True)
        self.assertNotIn("h100", portable_text.casefold())
        self.assertNotIn("gpu-", portable_text.casefold())
        self.assertNotIn("h100", self.raw_task["name"].casefold())


if __name__ == "__main__":
    unittest.main()
