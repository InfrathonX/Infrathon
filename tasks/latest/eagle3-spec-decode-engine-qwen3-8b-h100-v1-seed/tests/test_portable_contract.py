"""The portable contract freezes the TaskCore byte-for-byte.

Every digest, path set and cell count declared by ``portable/*.json`` and the
task JSON is recomputed here from the shipped sources (the public and hidden
prompt manifests and their commitments, the qualified policy, the Judge
adapter gate list, the verifier routing, Dockerfiles, runner), never trusted
from the declaration.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from fractions import Fraction

import pytest

from conftest import FETCH_HINT, PRIVATE_DATA, ROOT, dataset_path, load_file, private_data
from sforge.harness.portable_spec import parse_portable_spec, verify_sha256_file

PORTABLE = ROOT / "portable"
TASK_PATH = ROOT / "tasks/eagle3_spec_decode_engine_qwen3_8b_h100_v1.json"
SUITE_ID = "eagle3-spec-decode-v1"
PROFILE_ID = "single-cuda-sm80-34g"
METRIC_ID = "primary"
CELLS = ("chat_bs1", "math_bs1", "code_bs1", "longctx_bs1", "think_bs1", "mixed_bs8")
# The public smoke suite deliberately uses different cells and budgets from the Judge.
PUBLIC_CELLS = ("chat_bs2", "math_bs4", "code_bs3", "longctx_bs2", "mixed_bs6", "think_bs2")


@pytest.fixture(scope="module")
def raw_task() -> dict:
    """The task JSON."""
    return json.loads(TASK_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def portable(raw_task):
    """The strict portable contract parsed by the harness."""
    return parse_portable_spec(raw_task["portable"], task_id=raw_task["task_id"])


@pytest.fixture(scope="module")
def judge():
    """The Judge adapter module (gate list)."""
    return load_file("portable_test_run_judge", ROOT / "images/judge/run_judge.py")


@pytest.fixture(scope="module")
def policy() -> dict:
    """The qualified private policy."""
    return json.loads((private_data() / "policy.json").read_text(encoding="utf-8"))


def _manifest(name: str) -> dict:
    """Load one portable manifest."""
    return json.loads((PORTABLE / name).read_text(encoding="utf-8"))


def _judge_env() -> dict[str, int]:
    """Judge timeouts baked into the Judge Dockerfile."""
    dockerfile = (ROOT / "images/judge/Dockerfile").read_text(encoding="utf-8")
    return {name: int(value) for name, value in re.findall(r"(EAGLE3_[A-Z_]+_JUDGE_TIMEOUT)=([0-9]+)", dockerfile)}


def _verify_resources(manifest: dict) -> None:
    """Verify every listed digest; Judge-only data through the fetched asset."""
    missing = []
    for resource in manifest["resources"]:
        path = ROOT / resource["path"]
        if resource["path"].startswith(PRIVATE_DATA + "/"):
            path = dataset_path(resource["path"])
            if path is None:
                missing.append(resource["path"])
                continue
        verify_sha256_file(path, resource["sha256"])
    if missing:
        pytest.skip(FETCH_HINT)


def _prompt_count(path) -> int:
    """Number of prompt rows in one JSONL slice."""
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def test_every_declared_digest_is_the_actual_file_digest(raw_task, portable) -> None:
    """Task JSON digests equal the manifest bytes."""
    suite = portable.evaluation_suites[SUITE_ID]
    profile = portable.execution_profiles[PROFILE_ID]
    declarations = {
        "resource-manifest.json": portable.task_core.resource_manifest_sha256,
        "build-resource-manifest.json": raw_task["delivery"]["build_resource_manifest_sha256"],
        "public-suite-manifest.json": suite.roles["public"].manifest_sha256,
        "interactive-suite-manifest.json": suite.roles["interactive"].manifest_sha256,
        "final-suite-manifest.json": suite.roles["final"].manifest_sha256,
        "oracle-protocol.json": suite.oracle.spec_sha256,
        "selection-protocol.json": suite.selection.spec_sha256,
        "software-abi.json": profile.software_abi.spec_sha256,
        "calibration-protocol.json": profile.calibration_protocol.spec_sha256,
    }
    assert set(declarations) == {path.name for path in PORTABLE.glob("*.json")}
    for filename, declared in declarations.items():
        assert verify_sha256_file(PORTABLE / filename, declared) == declared


def test_resource_manifest_hashes_every_real_scientific_resource() -> None:
    """The scientific boundary is exactly the shipped science."""
    manifest = _manifest("resource-manifest.json")
    lab = "images/work/workdir/spec_lab"
    expected = {
        "images/work/AGENT_TASK.md", f"{lab}/README_AGENT.md",
        "images/work/public_tests/public_eval.py", "images/work/public_tests/run_public_tests.sh",
        "images/work/public_tests/source_admission.py", "images/work/public_tests/public_cases.json",
        f"{lab}/public_prompts/manifest.json", f"{lab}/public_prompts/reference_outputs.json",
        *(f"{lab}/public_prompts/{cell}.jsonl" for cell in PUBLIC_CELLS),
        *(f"{lab}/runtime/{name}" for name in ("__init__.py", "api.py", "evaluate.py", "loader.py", "oracle.py", "policy_public.json", "prompts.py")),
        *(f"{lab}/solutions/engine_v001/{name}" for name in ("__init__.py", "candidate.py", "config.py", "draft.py", "target_kv.py", "tree.py")),
        "images/judge/run_judge.py", "images/judge/verify_eagle3.py", "images/judge/protected_hashes.txt",
        *(f"images/judge/verifier/{name}" for name in ("candidate_worker.py", "controller.py", "oracle_reference.py", "sampling.py", "scoring.py", "source_admission.py")),
        *(f"images/judge/private_data/{name}" for name in ("anchor_manifest.json", "final_manifest.json", "warmup_manifest.json", "commitments.json", "policy.json", "reference_streams.json")),
        *(f"images/judge/private_data/{kind}/{cell}.jsonl" for kind in ("anchor", "final", "warmup") for cell in CELLS),
        "images/judge/seed_reference/solutions/__init__.py", "scripts/generate_hidden_manifest.py",
    }
    assert {resource["path"] for resource in manifest["resources"]} == expected
    protected = (ROOT / "images/judge/protected_hashes.txt").read_text(encoding="utf-8")
    for label in ("judge:verifier/controller.py", "judge:private_data/commitments.json", "judge:private_data/policy.json", "seed:solutions/engine_v001/candidate.py", "worker:runtime/api.py"):
        assert label in protected
    assert (ROOT / "images/judge/verifier/source_admission.py").read_bytes() == (ROOT / "images/work/public_tests/source_admission.py").read_bytes()
    _verify_resources(manifest)


def test_build_manifest_covers_exact_portable_delivery_inputs(raw_task) -> None:
    """The delivery digest covers both build contexts and nothing host-only."""
    path = PORTABLE / "build-resource-manifest.json"
    declared = raw_task["delivery"]["build_resource_manifest_sha256"]
    assert verify_sha256_file(path, declared) == declared
    manifest = json.loads(path.read_text(encoding="utf-8"))
    declared_paths = {resource["path"] for resource in manifest["resources"]}
    for context_name in ("work", "judge"):
        context = ROOT / f"images/{context_name}"
        actual = {f.relative_to(ROOT).as_posix() for f in context.rglob("*") if f.is_file() and "__pycache__" not in f.parts and f.suffix != ".pyc" and "/results/" not in f.as_posix()}
        if context_name == "judge" and not (ROOT / PRIVATE_DATA).is_dir():
            # Public checkout: the Judge-only data is a dataset asset.
            fetched = dataset_path(PRIVATE_DATA)
            if fetched is None:
                actual |= {path for path in declared_paths if path.startswith(PRIVATE_DATA + "/")}
            else:
                actual |= {f"{PRIVATE_DATA}/{f.relative_to(fetched).as_posix()}" for f in fetched.rglob("*") if f.is_file()}
        assert {r for r in declared_paths if r.startswith(f"images/{context_name}/")} == actual
    extras = {"tasks/BENCHMARK.yaml", "scripts/build_images.sh", "scripts/generate_protected_hashes.py", "scripts/generate_hidden_manifest.py",
              "scripts/sforge_h100_entry.py", "scripts/restricted_proxy.py", "scripts/lib.sh", "scripts/validate_layout.py", "portable/REBUILD_PROMPT.md"}
    assert {p for p in declared_paths if not p.startswith("images/")} == extras
    assert "images/work/public_tests/prepare_submit.sh" in declared_paths
    assert not any(p.startswith("host_reference/") for p in declared_paths)
    _verify_resources(manifest)


def test_host_private_final_manifest_cannot_enter_the_work_image() -> None:
    """The Work image carries no private data or portable documents."""
    work = (ROOT / "images/work/Dockerfile").read_text(encoding="utf-8").casefold()
    assert "portable" not in work and "final-suite-manifest" not in work and "private_data" not in work and "host_reference" not in work
    judge = (ROOT / "images/judge/Dockerfile").read_text(encoding="utf-8")
    # private_data/ enters only the host-built holdout layer FROM the public Judge.
    assert "COPY private_data/" not in judge
    holdout_layer = (ROOT / "images/judge/Dockerfile.private").read_text(encoding="utf-8")
    assert "FROM ${JUDGE_PUBLIC_IMAGE}" in holdout_layer
    assert "COPY --from=private_data . /opt/eagle3_judge/private_data/" in holdout_layer
    assert judge.startswith("ARG WORK_IMAGE=")


def _assert_cells(rows: list[dict], manifest: dict, kind: str, prompt_dir, commitment_prefix: str | None, cells=CELLS) -> None:
    """Cell rows equal the shipped manifest entries and their prompt files."""
    entries = {entry["cell"]: entry for entry in manifest["cells"]}
    assert [row["cell"] for row in rows] == [entry["cell"] for entry in manifest["cells"]]
    assert set(entries) == set(cells)
    commitments = json.loads((private_data() / "commitments.json").read_text(encoding="utf-8")) if commitment_prefix else {}
    for row in rows:
        entry = entries[row["cell"]]
        path = prompt_dir / f"{row['cell']}.jsonl"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert row["case_id"] == f"{kind}-{row['cell']}"
        assert row["prompt_file_sha256"] == entry["sha256"] == digest
        if commitment_prefix:
            assert commitments[f"{commitment_prefix}:{row['cell']}"] == digest
        # Manifest "count" is the number of batches; the slice holds count x batch_size prompts.
        assert row["prompt_count"] == entry["count"] * entry["batch_size"] == _prompt_count(path)
        assert row["batch_size"] == entry["batch_size"]
        assert row["batch_count"] == entry["count"]
        assert row["max_new_tokens"] == entry["max_new_tokens"]
        assert row["enable_thinking"] == entry["enable_thinking"]
        assert row["role"] == "performance"
        assert Fraction(row["weight"]["numerator"], row["weight"]["denominator"]) == Fraction(1, 6)


def test_public_manifest_is_the_actual_public_pools(portable, judge) -> None:
    """Public cells mirror the disclosed prompt pools."""
    manifest = _manifest("public-suite-manifest.json")
    assert len(manifest["cases"]) == portable.evaluation_suites[SUITE_ID].roles["public"].case_count == 6
    _assert_cells(manifest["cases"], json.loads((ROOT / "images/work/workdir/spec_lab/public_prompts/manifest.json").read_text()), "public", ROOT / "images/work/workdir/spec_lab/public_prompts", None, PUBLIC_CELLS)
    assert manifest["prompt_counts"] == {"chat_bs2": 4, "math_bs4": 8, "code_bs3": 6, "longctx_bs2": 4, "think_bs2": 4, "mixed_bs6": 12}
    assert manifest["run_gates"] == list(judge.GATES)
    assert manifest["case_counts"] == {"total": 6, "scored_performance": 6, "zero_weight_hard_gate": 0}
    for cell in PUBLIC_CELLS:
        assert (ROOT / f"images/work/workdir/spec_lab/public_prompts/warmup/{cell}.jsonl").is_file()


def test_interactive_manifest_is_the_complete_frozen_final(portable, judge) -> None:
    interactive = _manifest("interactive-suite-manifest.json")
    final = _manifest("final-suite-manifest.json")
    assert interactive["role"] == "interactive"
    assert interactive["cases"] == final["cases"]
    assert interactive["suite_id"] == final["suite_id"]
    assert interactive["run_gates"] == final["run_gates"]
    assert interactive["suite_gate_timeouts"] == final["suite_gate_timeouts"]
    assert len(interactive["cases"]) == portable.evaluation_suites[SUITE_ID].roles["interactive"].case_count
    selection = _manifest("selection-protocol.json")
    assert selection["same_suite_for_interactive_and_final"] is True
    assert selection["interactive"]["suite_id"] == selection["final"]["suite_id"]
    assert selection["evaluation_policy"] == "full-final-shared-v2"


def test_final_manifest_is_the_committed_final_slices(raw_task, portable, judge, policy) -> None:
    """Final cells mirror the committed final slices, warm-ups and the seed reference."""
    PRIVATE = private_data()
    manifest = _manifest("final-suite-manifest.json")
    assert len(manifest["cases"]) == portable.evaluation_suites[SUITE_ID].roles["final"].case_count == 6
    _assert_cells(manifest["cases"], json.loads((PRIVATE / "final_manifest.json").read_text()), "final", PRIVATE / "final", "final")
    assert manifest["prompt_counts"] == {"chat_bs1": 20, "math_bs1": 20, "code_bs1": 20, "longctx_bs1": 20, "think_bs1": 20, "mixed_bs8": 40}
    assert manifest["total_prompts"] == 140
    warm = json.loads((PRIVATE / "warmup_manifest.json").read_text())
    for entry in warm["cells"]:
        assert manifest["warmup"][entry["cell"]] == {"prompt_count": entry["count"] * entry["batch_size"], "batch_count": entry["count"], "batch_size": entry["batch_size"], "sha256": entry["sha256"]}
        assert hashlib.sha256((PRIVATE / "warmup" / f"{entry['cell']}.jsonl").read_bytes()).hexdigest() == entry["sha256"]
        assert _prompt_count(PRIVATE / "warmup" / f"{entry['cell']}.jsonl") == entry["count"] * entry["batch_size"]
    judge_dockerfile = (ROOT / "images/judge/Dockerfile").read_text(encoding="utf-8")
    assert "cp -a /app/workdir/spec_lab/solutions/engine_v001" in judge_dockerfile
    assert "/opt/eagle3_judge/seed_reference/solutions/engine_v001" in judge_dockerfile
    env = _judge_env()
    assert manifest["suite_gate_timeouts"]["final_judge_seconds"] == env["EAGLE3_FINAL_JUDGE_TIMEOUT"]
    final_timeout = raw_task["judge"]["eval_timeout"]
    assert manifest["orchestration_timeouts"]["standalone_final_evaluation_seconds"] == final_timeout == 5400
    assert final_timeout >= env["EAGLE3_FINAL_JUDGE_TIMEOUT"] > policy["worker_timeout_s"] > policy["build_cap_s"]
    runner = (ROOT / "scripts/run_fixed_budget.sh").read_text(encoding="utf-8")
    assert f'FINAL_EVAL_TIMEOUT="${{FINAL_EVAL_TIMEOUT:-{final_timeout}}}"' in runner
    assert 'JUDGE_PORT="${JUDGE_PORT:-18386}"' in runner
    selection = _manifest("selection-protocol.json")
    assert selection["cells"] == list(CELLS)
    assert selection["final"]["scored_case_count"] == 6 and selection["final"]["hard_gate_case_count"] == 0
    assert selection["interactive"]["prompt_counts"] == manifest["prompt_counts"]
    assert manifest["run_gates"] == list(judge.GATES) and len(manifest["run_gates"]) == 11


def test_oracle_and_calibration_match_the_qualified_policy(judge, policy) -> None:
    """The oracle document is the qualified policy and the gate list."""
    oracle = _manifest("oracle-protocol.json")
    assert oracle["gates"]["names"] == list(judge.GATES)
    assert oracle["gates"]["labels"] == dict(judge.GATE_LABELS)
    declared = oracle["policy"]
    assert float(declared["eps_hard"]) == policy["eps_hard"] == 24.0
    assert float(declared["p_soft"]) == policy["p_soft"]
    assert declared["scope"] == policy["scope"] == "global_run"
    assert declared["attn_implementation"] == policy["attn_implementation"] == "eager"
    assert [(float(t["margin"]), float(t["max_rate"])) for t in declared["tiers"]] == [(t["margin"], t["max_rate"]) for t in policy["tiers"]]
    assert len(declared["tiers"]) == 8
    assert declared["resource_cap_mode"] == policy["resource_cap_mode"] == "seed_relative_same_run"
    assert float(declared["wall_cap_seed_multiple"]) == policy["wall_cap_seed_multiple"]
    assert float(declared["memory_cap_seed_multiple"]) == policy["memory_cap_seed_multiple"]
    assert declared["memory_cap_headroom_bytes"] == policy["memory_cap_headroom_bytes"]
    assert float(declared["oracle_rate_seed_multiple"]) == policy["oracle_rate_seed_multiple"]
    assert policy["status"] == "QUALIFIED"
    controller = (ROOT / "images/judge/verifier/controller.py").read_text(encoding="utf-8")
    assert "def derive_resource_caps(" in controller and "def seed_relative_oracle_limits(" in controller
    calibration = _manifest("calibration-protocol.json")
    assert calibration["comparability"] == "same_measurement_profile"
    assert "same device in the same evaluation" in calibration["seed_reference"]


def test_software_abi_matches_the_pinned_images_and_adapter() -> None:
    """The ABI document equals the Dockerfile pins and the adapter mounts."""
    abi = _manifest("software-abi.json")
    work = (ROOT / "images/work/Dockerfile").read_text(encoding="utf-8")
    assert work.startswith("FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel")
    assert "transformers==4.53.1 tokenizers==0.21.4" in work and "safetensors==0.8.0 accelerate==1.14.0 numpy==2.4.6 huggingface-hub==0.33.5" in work
    assert 'ARG TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;10.0"' in work
    assert abi["torch_cuda_arch_list"] == "8.0;8.6;8.9;9.0;10.0"
    assert abi["source_runtime"] == ["pytorch-2.8.0", "triton-3.4.0", "transformers-4.53.1", "tokenizers-0.21.4", "safetensors-0.8.0", "accelerate-1.14.0", "numpy-2.4.6", "huggingface-hub-0.33.5"]
    adapter = (ROOT / "scripts/sforge_h100_entry.py").read_text(encoding="utf-8")
    assert 'MODEL_MOUNT = "/opt/models/qwen3-8b-hf"' in adapter and 'HEAD_MOUNT = "/opt/models/eagle3-head"' in adapter
    assert abi["model_binding"]["adapter"] == "scripts/sforge_h100_entry.py"


def test_portable_layers_preserve_metric_and_use_capabilities(raw_task, portable) -> None:
    """The metric, profile and TaskCore definition match the task."""
    preflight = load_file("portable_test_preflight", ROOT / "scripts/preflight.py")
    metric = portable.metrics[METRIC_ID]
    assert metric.direction == "maximize" == raw_task["judge"]["score_direction"] == raw_task["scoring"]["direction"]
    assert metric.unit == "tokens_per_s"
    assert metric.comparability == "same_measurement_profile"
    assert metric.baseline_policy == "none"
    assert list(metric.aggregation.spec["cells"]) == list(CELLS)
    scoring = (ROOT / "images/judge/verifier/scoring.py").read_text(encoding="utf-8")
    assert "math.exp(math.fsum(math.log(value) for value in values) / len(values))" in scoring
    profile = portable.execution_profiles[PROFILE_ID]
    assert profile.vendor == "nvidia" and profile.count == 1
    assert profile.min_memory_bytes == preflight.MIN_DEVICE_MEMORY_MIB * 1024 * 1024 == 34 * 1024**3
    assert tuple(int(p) for p in profile.min_compute_capability.split(".")) == preflight.MIN_COMPUTE_CAPABILITY == (8, 0)
    assert "eager-attention" in profile.features and "nvcc" in profile.features
    definition = portable.task_core.definition
    assert list(definition["cells"]) == list(CELLS)
    assert dict(definition["max_new_tokens"]) == {"chat_bs1": 512, "math_bs1": 512, "code_bs1": 512, "longctx_bs1": 256, "think_bs1": 1024, "mixed_bs8": 512}
    assert dict(definition["batch_sizes"]) == {"chat_bs1": 1, "math_bs1": 1, "code_bs1": 1, "longctx_bs1": 1, "think_bs1": 1, "mixed_bs8": 8}
    assert portable.defaults.evaluation_suite_id == SUITE_ID
    assert portable.defaults.execution_profile_id == PROFILE_ID
    assert portable.defaults.metric_id == METRIC_ID
    portable_text = json.dumps(raw_task["portable"], sort_keys=True)
    assert "h100" not in portable_text.casefold() and "gpu-" not in portable_text.casefold()
    assert "h100" not in raw_task["name"].casefold()
    assert raw_task["cwd"] == "/app/workdir/spec_lab"


def test_refresh_tool_reports_no_drift() -> None:
    """The refresh tool finds every digest current."""
    environment = {**os.environ, "EAGLE3_PRIVATE_DATA": str(private_data())}
    result = subprocess.run([sys.executable, "-B", str(ROOT / "scripts/refresh_portable_manifests.py"), "--check"], capture_output=True, text=True, check=False, env=environment)
    assert result.returncode == 0, result.stdout + result.stderr
