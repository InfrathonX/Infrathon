from __future__ import annotations

import re
import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AGENT_VISIBLE_TEXT = (
    ROOT / "images/work/AGENT_TASK.md",
    ROOT / "images/work/workdir/flash_kmeans_lab/README_AGENT.md",
    ROOT / "images/work/workdir/flash_kmeans_lab/notes.md",
)


class AgentVisibleContractTests(unittest.TestCase):
    def test_agent_visible_text_does_not_leak_research_solution_hints(self) -> None:
        forbidden = re.compile(
            r"(?:paper|figure\s*[345]|fig[345]|flashassign|sort[-_ ]inverse|"
            r"exhaustive|official\s+solution)",
            re.IGNORECASE,
        )
        for path in AGENT_VISIBLE_TEXT:
            with self.subTest(path=path.relative_to(ROOT)):
                text = path.read_text(encoding="utf-8")
                self.assertIsNone(forbidden.search(text))

    def test_instruction_states_the_generic_optimization_contract(self) -> None:
        text = AGENT_VISIBLE_TEXT[0].read_text(encoding="utf-8")
        normalized = " ".join(text.split())
        for required in (
            "assign(x, centroids, x_sq) -> labels",
            "update(x, labels, old_centroids) -> (counts, new_centroids)",
            "warmed fresh-process",
            "first-run latency is diagnostic",
            "FINAL_VERSION",
            "/workspace/public_tests/run_public_tests.sh",
            "--candidate-version N",
            "Do not materialize the complete `[B,N,K]` distance tensor",
        ):
            with self.subTest(required=required):
                self.assertIn(required, text)
        self.assertIn(
            "Public correctness uses literal golden inputs/outputs", normalized
        )
        self.assertIn(
            "timing rows execute only your candidate", normalized
        )
        self.assertIn(
            "Public timings are absolute candidate-only diagnostics", normalized
        )
        self.assertIn(
            "do not produce a score", normalized
        )

    def test_work_context_has_no_starting_candidate_or_benchmark(self) -> None:
        lab = ROOT / "images/work/workdir/flash_kmeans_lab"
        self.assertFalse((lab / "FINAL_VERSION").exists())
        self.assertEqual(list((lab / "kernels").glob("*.py")), [])
        benchmark_files = [
            path
            for path in (ROOT / "images/work/benchmark").rglob("*")
            if path.is_file()
        ]
        self.assertEqual(benchmark_files, [])
        dockerfile = (ROOT / "images/work/Dockerfile").read_text(
            encoding="utf-8"
        )
        self.assertNotRegex(dockerfile, r"(?m)^\s*COPY\s+benchmark(?:/|\s)")
        text = AGENT_VISIBLE_TEXT[0].read_text(encoding="utf-8")
        self.assertIn("intentionally starts with no implementation", text)
        self.assertIn("kernels/flash_kmeans_v001.py", text)

    def test_work_context_includes_only_implementation_free_public_tests(
        self,
    ) -> None:
        public = ROOT / "images/work/public_tests"
        source = public / "public_test.py"
        runner = public / "run_public_tests.sh"
        self.assertTrue(source.is_file())
        self.assertTrue(runner.is_file())
        tree = ast.parse(source.read_text(encoding="utf-8"))
        definitions = {
            node.name.lower()
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        }
        for forbidden in (
            "baseline",
            "oracle",
            "reference_assign",
            "reference_update",
            "standard_blockwise_assign",
            "standard_scatter_update",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, definitions)

        dockerfile = (ROOT / "images/work/Dockerfile").read_text(
            encoding="utf-8"
        )
        self.assertIn("COPY public_tests/ /workspace/public_tests/", dockerfile)
        dockerignore = (ROOT / "images/work/.dockerignore").read_text(
            encoding="utf-8"
        )
        self.assertIn("!public_tests/**", dockerignore)

    def test_task_query_matches_visible_public_and_private_judge_boundary(
        self,
    ) -> None:
        import json

        task = json.loads(
            (
                ROOT
                / "tasks/flash_kmeans_paper_h100_kernel_lab_v7_scratch.json"
            ).read_text(encoding="utf-8")
        )
        query = task["work"]["agent_query"]
        for required in (
            "no candidate",
            "no candidate, K-Means reference implementation, baseline, oracle",
            "no candidate, K-Means reference implementation, baseline, oracle, formal performance cases, or timing denominator",
            "literal golden fixtures",
            "candidate-only diagnostic timings",
            "large-K, and large-N/high-D deployment proxies",
            "coarse large-K and large-N/high-D deployment latency bands",
            "per-case timings, weights, seeds, case names, and family membership remain hidden",
            "Judge as authoritative",
            "sforge-submit --details",
            "disables in-run automatic hidden evaluation",
            "replayed offline without sending feedback to you",
        ):
            with self.subTest(required=required):
                self.assertIn(required, query)

    def test_host_calibration_sources_are_not_inside_work_context(self) -> None:
        work = ROOT / "images/work"
        work_hashes = {
            path.read_bytes()
            for path in work.rglob("*")
            if path.is_file()
        }
        for source in (
            ROOT / "host_reference/baseline/torch_fallback.py",
            ROOT / "host_reference/official_oracle/assign_euclid_triton.py",
            ROOT / "host_reference/official_oracle/centroid_update_triton.py",
        ):
            with self.subTest(source=source.relative_to(ROOT)):
                self.assertNotIn(source.read_bytes(), work_hashes)
        self.assertFalse(
            (ROOT / "images/judge/paired_timing_reference.py").exists()
        )

    @unittest.skipUnless(
        (ROOT / "docs/design.md").is_file(), "maintainer design notes are not shipped"
    )
    def test_host_readme_separates_execution_from_provenance(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        design = (ROOT / "docs/design.md").read_text(encoding="utf-8")
        self.assertIn("三个角色不能混在一起", readme)
        self.assertIn("root-only", readme)
        self.assertIn("Host-only 外部比较实现", readme)
        self.assertIn("Public tests 与 benchmark 不是同一个概念", design)
        self.assertIn("论文图号不进入", design)


if __name__ == "__main__":
    unittest.main()
