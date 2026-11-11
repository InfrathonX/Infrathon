from __future__ import annotations

import importlib.util
import inspect
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEPLOYMENT_PATH = (
    PROJECT_ROOT
    / "images"
    / "judge"
    / "benchmark"
    / "deployment_bench.py"
)


def _load_deployment_module():
    name = "kmeans_deployment_bench_contract"
    spec = importlib.util.spec_from_file_location(name, DEPLOYMENT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


deployment = _load_deployment_module()


def _request() -> dict[str, object]:
    return {
        "case_name": "caller_supplied_case",
        "stage": "full",
        "b": 1,
        "n": 32,
        "k": 8,
        "d": 16,
        "seed": 123,
        "category": "deployment",
        "distribution": "uniform",
        "tie": False,
        "weight": 1.0,
        "verification": "sampled",
        "warmed_repeats": 2,
    }


def _success_payload(
    *,
    nonce: str,
    request: dict[str, object],
    source_kind: str = "candidate",
) -> dict[str, object]:
    return {
        "schema_version": deployment.SCHEMA_VERSION,
        "protocol": deployment.PROTOCOL,
        "status": "ok",
        "_nonce": nonce,
        "case_name": request["case_name"],
        "stage": request["stage"],
        "shape": [
            request["b"],
            request["n"],
            request["k"],
            request["d"],
        ],
        "source_kind": source_kind,
        "first_run_ms": 12.5,
        "warmed_iteration_ms": 1.25,
        "warmed_iteration_samples_ms": [1.2, 1.3],
        "correctness_ok": True,
        "input_integrity_ok": True,
        "output_contract_ok": True,
        "memory_observed": True,
        "peak_extra_bytes": 4096,
        "candidate_output_suppressed": True,
        "cache_was_empty": True,
    }


class _SuccessfulFakePopen:
    calls: list[dict[str, object]] = []

    def __init__(self, command, **kwargs):
        self.command = command
        self.kwargs = kwargs
        self.returncode = 0
        self.pid = 999999
        result_fd = kwargs["pass_fds"][0]

        def argument(name: str) -> str:
            return command[command.index(name) + 1]

        request = {
            "case_name": argument("--case-name"),
            "stage": argument("--stage"),
            "b": int(argument("--b")),
            "n": int(argument("--n")),
            "k": int(argument("--k")),
            "d": int(argument("--d")),
        }
        payload = _success_payload(
            nonce=argument("--_nonce"),
            request=request,
            source_kind="candidate",
        )
        os.write(
            result_fd,
            json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        )
        self.__class__.calls.append(
            {
                "command": list(command),
                "kwargs": kwargs,
                "triton_cache": kwargs["env"]["TRITON_CACHE_DIR"],
            }
        )

    def communicate(self, timeout=None):
        # The parent must ignore even marker-looking data returned here.
        return (
            b">>>>> Start Structured Result\n"
            b'{"first_run_ms":0,"correctness_ok":true}\n'
            b">>>>> End Structured Result\n",
            b"",
        )

    def kill(self):
        self.returncode = -9


class DeploymentApiContractTests(unittest.TestCase):
    def test_api_requires_all_shape_fields_and_exposes_policy_controls(self) -> None:
        signature = inspect.signature(deployment.run_deployment_case)
        expected = {
            "candidate_file",
            "case_name",
            "stage",
            "b",
            "n",
            "k",
            "d",
            "seed",
            "timeout_sec",
            "warmed_repeats",
            "candidate_root",
            "category",
            "distribution",
            "tie",
            "weight",
            "verification",
        }
        self.assertEqual(set(signature.parameters), expected)
        self.assertEqual(
            signature.parameters["verification"].default, "sampled"
        )
        self.assertNotIn("trusted_source", signature.parameters)

    def test_helper_contains_no_embedded_shape_manifest(self) -> None:
        source = DEPLOYMENT_PATH.read_text(encoding="utf-8")
        self.assertNotIn("cold", source.lower())
        self.assertNotIn("N=1M", source)
        self.assertNotIn("N=8M", source)
        self.assertNotIn("N=16M", source)
        self.assertIn("--b", source)
        self.assertIn("--n", source)
        self.assertIn("--k", source)
        self.assertIn("--d", source)

    def test_every_warmed_iteration_is_correctness_and_peak_memory_checked(
        self,
    ) -> None:
        source = DEPLOYMENT_PATH.read_text(encoding="utf-8")
        self.assertIn(
            "warmed_correct, warmed_contract = _verify_output(",
            source,
        )
        self.assertIn(
            "warmed-iteration correctness verification failed",
            source,
        )
        self.assertIn(
            "int(MAX_MEMORY_ALLOCATED())\n"
            "                        - warmed_allocated_before",
            source,
        )
        self.assertIn("resident_before_input", source)
        self.assertIn(
            "int(MEMORY_ALLOCATED())\n"
            "                        - process_initial_allocated",
            source,
        )
        self.assertIn(
            "peak_extra_bytes = max(\n"
            "                peak_extra_bytes,\n"
            "                warmed_peak_extra_bytes,",
            source,
        )

    def test_sampled_mode_is_257_points_plus_full_update(self) -> None:
        guard_source = inspect.getsource(deployment._guard_indices)
        assignment_source = inspect.getsource(
            deployment._sampled_assignment_check
        )
        verification_source = inspect.getsource(deployment._verify_output)
        update_source = inspect.getsource(deployment._chunked_update_check)

        self.assertIn("count = min(length, 257)", guard_source)
        self.assertIn("_guard_indices(", assignment_source)
        self.assertIn("_chunked_update_check(", verification_source)
        self.assertIn("counts_ref", update_source)
        self.assertIn("sums", update_source)
        self.assertIn("for start in range(0, spec.n, chunk)", update_source)

    def test_child_arms_linux_parent_death_signal(self) -> None:
        source = DEPLOYMENT_PATH.read_text(encoding="utf-8")
        self.assertIn("PR_SET_PDEATHSIG", source)
        self.assertIn("_arm_parent_death_signal()", source)
        self.assertIn("if os.getppid() == 1:", source)

    def test_invalid_request_fails_before_any_subprocess(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "candidate.py"
            source.write_text(
                "def assign(*args): pass\ndef update(*args): pass\n",
                encoding="utf-8",
            )
            with mock.patch.object(
                deployment.subprocess, "Popen"
            ) as popen:
                result = deployment.run_deployment_case(
                    source,
                    case_name="bad name with spaces",
                    stage="full",
                    b=1,
                    n=32,
                    k=8,
                    d=16,
                    seed=1,
                    timeout_sec=2.0,
                )
            popen.assert_not_called()
            self.assertEqual(result["status"], "error")
            self.assertEqual(result["error_type"], "ValueError")

    def test_untrusted_forbidden_source_is_rejected_before_spawn(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            kernels = root / "kernels"
            kernels.mkdir()
            (root / "FINAL_VERSION").write_text("1\n", encoding="utf-8")
            source = kernels / "flash_kmeans_v001.py"
            source.write_text(
                textwrap.dedent(
                    """\
                    import os
                    def assign(x, centroids, x_sq):
                        return x
                    def update(x, labels, old_centroids):
                        return labels, old_centroids
                    """
                ),
                encoding="utf-8",
            )
            with mock.patch.object(
                deployment.subprocess, "Popen"
            ) as popen:
                result = deployment.run_deployment_case(
                    source,
                    candidate_root=root,
                    case_name="policy_case",
                    stage="full",
                    b=1,
                    n=32,
                    k=8,
                    d=16,
                    seed=1,
                    timeout_sec=2.0,
                )
            popen.assert_not_called()
            self.assertEqual(result["status"], "error")
            self.assertEqual(result["source_kind"], "candidate")

    def test_runtime_copy_must_match_admitted_selected_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = base / "submission"
            kernels = root / "kernels"
            kernels.mkdir(parents=True)
            (root / "FINAL_VERSION").write_text("1\n", encoding="utf-8")
            selected = kernels / "flash_kmeans_v001.py"
            selected.write_text(
                "def assign(x,c,xs): return x\n"
                "def update(x,l,c): return l,c\n",
                encoding="utf-8",
            )
            runtime = base / "flash_kmeans_v001.py"
            runtime.write_text(
                selected.read_text(encoding="utf-8") + "# changed\n",
                encoding="utf-8",
            )
            result = deployment.run_deployment_case(
                runtime,
                candidate_root=root,
                case_name="hash_case",
                stage="assign",
                b=1,
                n=32,
                k=8,
                d=16,
                seed=1,
                timeout_sec=2.0,
            )
            self.assertEqual(result["status"], "error")
            self.assertEqual(result["error_type"], "ValueError")


class TrustedResultChannelTests(unittest.TestCase):
    def setUp(self) -> None:
        _SuccessfulFakePopen.calls.clear()

    def test_each_call_uses_fresh_process_and_empty_private_cache(self) -> None:
        request = _request()
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "candidate.py"
            source.write_text(
                "def assign(*args): pass\ndef update(*args): pass\n",
                encoding="utf-8",
            )
            with mock.patch.object(
                deployment.subprocess, "Popen", _SuccessfulFakePopen
            ):
                first = deployment._spawn_child(
                    source,
                    request=request,
                    timeout_sec=2.0,
                )
                second = deployment._spawn_child(
                    source,
                    request=request,
                    timeout_sec=2.0,
                )
        self.assertEqual(first["status"], "ok")
        self.assertEqual(first["first_run_ms"], 12.5)
        self.assertEqual(first["warmed_iteration_ms"], 1.25)
        self.assertTrue(first["correctness_ok"])
        self.assertEqual(len(_SuccessfulFakePopen.calls), 2)
        caches = [
            str(call["triton_cache"])
            for call in _SuccessfulFakePopen.calls
        ]
        self.assertNotEqual(caches[0], caches[1])
        for call in _SuccessfulFakePopen.calls:
            kwargs = call["kwargs"]
            self.assertIs(kwargs["stdout"], subprocess.DEVNULL)
            self.assertIs(kwargs["stderr"], subprocess.DEVNULL)
            self.assertTrue(kwargs["start_new_session"])
            self.assertEqual(len(kwargs["pass_fds"]), 1)

    def test_nonce_mismatch_and_nonfinite_timing_fail_closed(self) -> None:
        request = _request()
        payload = _success_payload(nonce="wrong", request=request)
        with self.assertRaisesRegex(ValueError, "nonce"):
            deployment._validate_child_payload(
                payload,
                nonce="expected",
                case_name=str(request["case_name"]),
                stage=str(request["stage"]),
                shape=[1, 32, 8, 16],
                source_kind="candidate",
            )
        payload["_nonce"] = "expected"
        payload["first_run_ms"] = float("nan")
        with self.assertRaisesRegex(ValueError, "first_run_ms"):
            deployment._validate_child_payload(
                payload,
                nonce="expected",
                case_name=str(request["case_name"]),
                stage=str(request["stage"]),
                shape=[1, 32, 8, 16],
                source_kind="candidate",
            )

    def test_timeout_is_structured_and_kills_the_child_group(self) -> None:
        request = _request()

        class TimeoutPopen:
            def __init__(self, command, **kwargs):
                self.pid = 999999
                self.returncode = None

            def communicate(self, timeout=None):
                raise subprocess.TimeoutExpired(["child"], timeout)

        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "candidate.py"
            source.write_text("# fixture\n", encoding="utf-8")
            with (
                mock.patch.object(
                    deployment.subprocess, "Popen", TimeoutPopen
                ),
                mock.patch.object(
                    deployment, "_kill_process_group"
                ) as kill,
            ):
                result = deployment._spawn_child(
                    source,
                    request=request,
                    timeout_sec=0.5,
                )
        self.assertEqual(result["status"], "timeout")
        self.assertEqual(result["error_type"], "TimeoutExpired")
        kill.assert_called_once()


if __name__ == "__main__":
    unittest.main()
