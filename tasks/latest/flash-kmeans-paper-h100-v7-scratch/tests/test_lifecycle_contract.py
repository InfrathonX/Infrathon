from __future__ import annotations

import json
import importlib.util
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TASK_ID = "flash_kmeans_paper_h100_kernel_lab_v7_scratch"


def _load_source_admission():
    path = (
        PROJECT_ROOT
        / "images"
        / "judge"
        / "benchmark"
        / "source_admission.py"
    )
    spec = importlib.util.spec_from_file_location(
        "flash_kmeans_source_admission_contract", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


class TaskMetadataContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.task_path = PROJECT_ROOT / "tasks" / f"{TASK_ID}.json"
        self.task = json.loads(self.task_path.read_text(encoding="utf-8"))

    def test_submission_and_judge_selection_are_fail_closed(self) -> None:
        self.assertEqual(
            self.task["submit_paths"], ["kernels", "FINAL_VERSION", "notes.md"]
        )
        self.assertEqual(self.task["cwd"], "/app/workdir/flash_kmeans_lab")
        judge = self.task["judge"]
        self.assertEqual(judge["parser"], "structured_json")
        self.assertEqual(judge["selection"], "valid_then_score")
        self.assertEqual(judge["score_direction"], "minimize")
        self.assertNotIn("rescale", judge)
        self.assertNotIn("rescale_calibration", judge)

    def test_image_names_derive_to_expected_local_tags(self) -> None:
        benchmark = (
            PROJECT_ROOT / "tasks" / "BENCHMARK.yaml"
        ).read_text(encoding="utf-8")
        match = re.search(r"(?m)^name:\s*(\S+)\s*$", benchmark)
        self.assertIsNotNone(match)
        name = match.group(1)
        self.assertEqual(
            f"{name}.work.{TASK_ID}:{self.task['work']['image_tag']}",
            "flash_kmeans.work."
            "flash_kmeans_paper_h100_kernel_lab_v7_scratch:local",
        )
        self.assertEqual(
            f"{name}.judge.{TASK_ID}:{self.task['judge']['image_tag']}",
            "flash_kmeans.judge."
            "flash_kmeans_paper_h100_kernel_lab_v7_scratch:local",
        )

    def test_agent_query_requests_actionable_manual_feedback_and_rollback(self) -> None:
        query = self.task["work"]["agent_query"]
        lower_query = query.lower()
        self.assertIn("sforge-submit --details", query)
        self.assertIn("300-second cooldown", query)
        self.assertIn("submit again", query)
        self.assertIn("without sending feedback to you", query)
        self.assertIn("restore FINAL_VERSION", query)
        self.assertIn("intentionally contains no candidate", query)
        self.assertIn(
            "create a complete kernels/flash_kmeans_v001.py", lower_query
        )
        self.assertIn("if none exists, select the simplest locally correct", query)


class HostScriptContractTests(unittest.TestCase):
    def test_v7_host_namespace_does_not_reuse_older_ports_or_run_prefixes(self) -> None:
        scripts = {
            path.name: path.read_text(encoding="utf-8")
            for path in (PROJECT_ROOT / "scripts").glob("*")
            if path.is_file()
        }
        combined = "\n".join(scripts.values())
        for old_port in (
            "18084",
            "18085",
            "18086",
            "18184",
            "18185",
            "18186",
        ):
            self.assertNotIn(old_port, combined)
        self.assertIn('JUDGE_PORT="${JUDGE_PORT:-18284}"', scripts["run_fixed_budget.sh"])
        self.assertIn('JUDGE_PORT="${JUDGE_PORT:-18285}"', scripts["run_codex_smoke.sh"])
        self.assertIn("flash-kmeans-v7-scratch-h100-", combined)
        self.assertNotIn("flash-kmeans-h100-${", combined)

    def test_all_shell_scripts_parse(self) -> None:
        scripts = sorted((PROJECT_ROOT / "scripts").glob("*.sh"))
        self.assertTrue(scripts)
        for script in scripts:
            with self.subTest(script=script.name):
                result = subprocess.run(
                    ["bash", "-n", str(script)],
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_gpu_environment_is_flash_kmeans_specific(self) -> None:
        scripts = "\n".join(
            path.read_text(encoding="utf-8")
            for path in sorted((PROJECT_ROOT / "scripts").glob("*.sh"))
        )
        self.assertIn("FLASH_KMEANS_TARGET=H100", scripts)
        self.assertNotIn("GEMM_CUDA_ARCH", scripts)
        common = (PROJECT_ROOT / "scripts" / "run_fixed_budget.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn('WORK_GPU="${WORK_GPU:-1}"', common)
        self.assertIn('JUDGE_GPU="${JUDGE_GPU:-2}"', common)

    def test_host_judge_traffic_bypasses_inherited_proxies(self) -> None:
        for name in ("run_fixed_budget.sh", "run_codex_smoke.sh"):
            with self.subTest(name=name):
                text = (PROJECT_ROOT / "scripts" / name).read_text(
                    encoding="utf-8"
                )
                self.assertIn(
                    "unset HTTP_PROXY HTTPS_PROXY ALL_PROXY "
                    "http_proxy https_proxy all_proxy",
                    text,
                )
                self.assertIn(
                    'export NO_PROXY="${restricted_proxy_ip},127.0.0.1,localhost"',
                    text,
                )
                self.assertIn('export no_proxy="${NO_PROXY}"', text)

    def test_live_web_search_follows_the_task_json(self) -> None:
        codex_config = (
            PROJECT_ROOT / "images/work/codex-config.toml"
        ).read_text(encoding="utf-8")
        isolated = (
            PROJECT_ROOT / "scripts/sforge_isolated_entry.py"
        ).read_text(encoding="utf-8")
        common = (
            PROJECT_ROOT / "scripts/run_fixed_budget.sh"
        ).read_text(encoding="utf-8")
        self.assertIn('web_search = "live"', codex_config)
        self.assertIn(
            'kwargs["internet"] = _task_internet(kwargs.get("internet", True))',
            isolated,
        )
        self.assertIn('export MLSYSFORGE_WORK_INTERNET="${task_internet}"', common)
        self.assertIn('codex_web_search="live"', common)
        self.assertIn('--arg codex_web_search "${codex_web_search}"', common)
        self.assertIn('codex_web_search: $codex_web_search', common)

    def test_fixed_runner_has_no_default_model(self) -> None:
        common = (
            PROJECT_ROOT / "scripts/run_fixed_budget.sh"
        ).read_text(encoding="utf-8")
        self.assertIn('MODEL="${MODEL:-}"', common)
        self.assertIn(
            '[[ "${MODEL}" =~ ^[A-Za-z0-9][A-Za-z0-9_./:-]*$ ]]', common
        )

    def test_docker_group_membership_handles_first_only_and_exact_member(self) -> None:
        lib = PROJECT_ROOT / "scripts" / "lib.sh"
        probe = r'''
source "$1"
group_line="$2"
getent() {
  [[ "$1" == "group" && "$2" == "docker" ]] || return 2
  printf '%s\n' "${group_line}"
}
user_in_docker_group "$3"
'''
        cases = (
            ("docker:x:993:alice", "alice", True),
            ("docker:x:993:alice,bob", "alice", True),
            ("docker:x:993:bob,alice", "alice", True),
            ("docker:x:993:malice,bob", "alice", False),
            ("docker:x:993:", "alice", False),
        )
        for group_line, user, expected in cases:
            with self.subTest(group_line=group_line, user=user):
                result = subprocess.run(
                    ["bash", "-c", probe, "group-probe", str(lib), group_line, user],
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertEqual(result.returncode == 0, expected, result.stderr)

    def test_smoke_uses_reduced_hidden_suite(self) -> None:
        for name in ("smoke_test.sh", "run_codex_smoke.sh"):
            text = (PROJECT_ROOT / "scripts" / name).read_text(encoding="utf-8")
            with self.subTest(script=name):
                self.assertIn("FLASH_KMEANS_VERIFY_SUITE=smoke", text)
                self.assertIn("FLASH_KMEANS_VERIFY_WARMUP=1", text)
                self.assertIn("FLASH_KMEANS_VERIFY_ITERS=2", text)
                self.assertIn("FLASH_KMEANS_VERIFY_REPEATS=1", text)

    def test_codex_smoke_requires_a_valid_positive_final_score(self) -> None:
        text = (
            PROJECT_ROOT / "scripts" / "run_codex_smoke.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("--disable-auto-eval", text)
        self.assertIn("expected_rounds = []", text)
        self.assertIn("compatibility interval", text)
        self.assertNotIn("at least two periodic snapshots", text)
        self.assertIn('report.get("valid") is not True', text)
        self.assertIn('report.get("pass_rate") != 1.0', text)
        self.assertIn("float(score) <= 0.0", text)
        self.assertIn("flash_kmeans_h100_verifier_adapter", text)
        self.assertIn("candidate_output_suppressed", text)
        self.assertIn("result_protocol_records", text)
        self.assertIn("verifier_returncode", text)

    def test_compatibility_archive_uses_host_only_upstream_baseline(self) -> None:
        text = (
            PROJECT_ROOT / "scripts" / "make_repo_port_archive.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("make_host_calibration_archives.sh", text)
        self.assertIn("torch-fallback-baseline.tar.gz", text)
        self.assertNotIn("images/work/workdir", text)

    def test_smoke_exercises_policy_and_correctness_failures(self) -> None:
        text = (
            PROJECT_ROOT / "scripts" / "smoke_test.sh"
        ).read_text(encoding="utf-8")
        self.assertIn('assert_invalid_fixture "invalid-source"', text)
        self.assertIn('assert_invalid_fixture "incorrect-output"', text)
        self.assertIn('report.get("valid") is not False', text)
        self.assertIn("invalid score must be null", text)
        for fixture in ("invalid-source", "incorrect-output"):
            root = PROJECT_ROOT / "tests" / "fixtures" / fixture
            self.assertTrue((root / "FINAL_VERSION").is_file())
            self.assertTrue((root / "notes.md").is_file())
            self.assertEqual(len(list((root / "kernels").glob("*.py"))), 1)

    def test_smoke_runs_visible_public_tests_without_a_public_score(self) -> None:
        text = (
            PROJECT_ROOT / "scripts" / "smoke_test.sh"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "/workspace/public_tests/run_public_tests.sh", text
        )
        self.assertIn('result.get("all_public_gates_ok") is not True', text)
        self.assertIn('result.get("authoritative") is not False', text)
        self.assertIn('if "score" in result', text)
        self.assertNotIn("/workspace/benchmark/run_public_tests.sh", text)

    def test_smoke_runs_pinned_numerical_envelope_regression(self) -> None:
        text = (
            PROJECT_ROOT / "scripts" / "smoke_test.sh"
        ).read_text(encoding="utf-8")
        regression = PROJECT_ROOT / "tests" / "gpu_numerical_regression.py"
        self.assertTrue(regression.is_file())
        self.assertIn("gpu_numerical_regression.py", text)
        regression_text = regression.read_text(encoding="utf-8")
        self.assertIn("centroids[0, 1, 0] += 0.03125", regression_text)
        self.assertIn("(reference + 1) % spec.k", regression_text)
        self.assertIn("public_close", regression_text)
        self.assertIn("hidden_close", regression_text)


class NegativeFixtureContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.admission = _load_source_admission()

    def test_invalid_source_is_rejected_for_subprocess_import(self) -> None:
        root = PROJECT_ROOT / "tests" / "fixtures" / "invalid-source"
        result = self.admission.scan_submission(root)
        self.assertEqual(result["status"], "rejected")
        self.assertTrue(
            any(
                "prohibited import: subprocess" in problem["message"]
                for problem in result["problems"]
            ),
            result,
        )

    def test_incorrect_output_is_policy_clean_before_correctness_gate(self) -> None:
        root = PROJECT_ROOT / "tests" / "fixtures" / "incorrect-output"
        result = self.admission.scan_submission(root)
        self.assertEqual(result["status"], "clean", result)
        self.assertEqual(result["selected_version"], 998)

    def _scan_source(self, source: str):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        kernels = root / "kernels"
        kernels.mkdir()
        (kernels / "flash_kmeans_v001.py").write_text(
            source,
            encoding="utf-8",
        )
        (root / "FINAL_VERSION").write_text("1\n", encoding="utf-8")
        return root, self.admission.scan_submission(root)

    def test_dynamic_builtins_lookup_is_rejected(self) -> None:
        _, result = self._scan_source(
            "key = 'eval'\n"
            "escaped = __builtins__[key]\n"
            "def assign(x, centroids, x_sq):\n"
            "    return escaped('x')\n"
            "def update(x, labels, old_centroids):\n"
            "    return labels, old_centroids\n"
        )
        self.assertEqual(result["status"], "rejected", result)
        self.assertTrue(
            any(
                "__builtins__" in problem["message"]
                for problem in result["problems"]
            ),
            result,
        )

    def test_torch_bridge_symbol_aliases_are_rejected(self) -> None:
        _, result = self._scan_source(
            "from torch import builtins as b, sys as s\n"
            "def assign(x, centroids, x_sq):\n"
            "    return x\n"
            "def update(x, labels, old_centroids):\n"
            "    return labels, old_centroids\n"
        )
        self.assertEqual(result["status"], "rejected", result)
        messages = {problem["message"] for problem in result["problems"]}
        self.assertIn("prohibited imported symbol: builtins", messages)
        self.assertIn("prohibited imported symbol: sys", messages)

    def test_cuda_peak_memory_reset_is_rejected(self) -> None:
        _, result = self._scan_source(
            "import torch\n"
            "def assign(x, centroids, x_sq):\n"
            "    torch.cuda.reset_peak_memory_stats()\n"
            "    return x\n"
            "def update(x, labels, old_centroids):\n"
            "    return labels, old_centroids\n"
        )
        self.assertEqual(result["status"], "rejected", result)
        self.assertTrue(
            any(
                "reset_peak_memory_stats" in problem["message"]
                for problem in result["problems"]
            ),
            result,
        )

    def test_cuda_peak_memory_reset_alias_is_rejected(self) -> None:
        _, result = self._scan_source(
            "import torch\n"
            "reset_peak = torch.cuda.reset_peak_memory_stats\n"
            "def assign(x, centroids, x_sq):\n"
            "    reset_peak()\n"
            "    return x\n"
            "def update(x, labels, old_centroids):\n"
            "    return labels, old_centroids\n"
        )
        self.assertEqual(result["status"], "rejected", result)
        self.assertTrue(
            any(
                "prohibited attribute/module-bridge reference"
                in problem["message"]
                for problem in result["problems"]
            ),
            result,
        )

    def test_low_level_torch_c_cuda_reset_is_rejected(self) -> None:
        _, result = self._scan_source(
            "import torch\n"
            "def assign(x, centroids, x_sq):\n"
            "    torch._C._cuda_resetPeakMemoryStats(0)\n"
            "    return x\n"
            "def update(x, labels, old_centroids):\n"
            "    return labels, old_centroids\n"
        )
        self.assertEqual(result["status"], "rejected", result)
        self.assertTrue(
            any(
                "module-bridge" in problem["message"]
                or "prohibited call" in problem["message"]
                for problem in result["problems"]
            ),
            result,
        )

    def test_symlink_directory_under_kernels_is_rejected(self) -> None:
        root, clean = self._scan_source(
            "def assign(x, centroids, x_sq):\n"
            "    return x\n"
            "def update(x, labels, old_centroids):\n"
            "    return labels, old_centroids\n"
        )
        self.assertEqual(clean["status"], "clean", clean)
        outside = root / "outside"
        outside.mkdir()
        (root / "kernels" / "linked").symlink_to(
            outside,
            target_is_directory=True,
        )
        result = self.admission.scan_submission(root)
        self.assertEqual(result["status"], "rejected", result)
        self.assertTrue(
            any(
                problem["message"] == "symlinks are not admitted"
                for problem in result["problems"]
            ),
            result,
        )


if __name__ == "__main__":
    unittest.main()
