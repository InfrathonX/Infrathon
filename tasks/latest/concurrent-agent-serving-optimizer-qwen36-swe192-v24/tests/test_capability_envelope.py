"""The backend requirement is a capability envelope, not a device model.

Nothing in this task's oracle, metric, hard gates or admission policy depends
on which NVIDIA part serves the model.  The only genuine hardware facts are
two devices, compute capability >= 8.0, and about 92.8 GiB of device memory
per card (25.9 GiB of BF16 weights per rank at TP=2 plus the frozen 30 GiB FP8
KV cache), which the contract states as a 95000 MiB floor.  These tests keep
the device-model whitelist from coming back and keep the revision registry
consistent across every trusted component.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest

from conftest import load_module


ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENT = (
    ROOT
    / "runtime/frozen/harbor_tasks/"
    "concurrent-agent-serving-optimizer-qwen36-27b-swe192-v2.1-gpu45/environment"
)
CURRENT_CONTRACT = ENVIRONMENT / "backend-serving-contract.json"
LEGACY_CONTRACT = ENVIRONMENT / "backend-serving-contract-v1.json"
PREVIOUS_CONTRACT = ENVIRONMENT / "backend-serving-contract-v2.5.json"
BACKEND_DOCKERFILE = ROOT / "images/backend/Dockerfile"
MODEL_PROFILE = ENVIRONMENT / "vllm-runtime/model-profile-qwen3.6-27b.json"
RUNTIME_CONTRACT = ROOT / "images/work/RUNTIME_CONTRACT.json"
TASK_ID = "concurrent_agent_serving_optimizer_qwen36_swe192_v24"
MIN_COMPUTE_CAPABILITY = "8.0"
MIN_DEVICE_MEMORY_MIB = 95000
DEVICE_COUNT = 2
FOREIGN_DEVICE_NAMES = (
    "NVIDIA H200",
    "NVIDIA H20",
    "NVIDIA A100-SXM4-80GB",
    "NVIDIA RTX A6000",
    "NVIDIA B200",
)


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).hexdigest()


@pytest.fixture(scope="module")
def preflight():
    return load_module("kvbench_capability_preflight", ROOT / "scripts/preflight.py")


def test_shipped_contract_states_a_capability_predicate() -> None:
    capability = _json(CURRENT_CONTRACT)["accelerator_capability"]

    assert "allowed_names" not in capability
    assert "compute_capability" not in capability
    assert capability["count"] == DEVICE_COUNT
    assert capability["min_compute_capability"] == MIN_COMPUTE_CAPABILITY
    assert capability["minimum_memory_mib"] == MIN_DEVICE_MEMORY_MIB


def test_all_revisions_describe_one_serving_system(preflight) -> None:
    current = _json(CURRENT_CONTRACT)
    previous = _json(PREVIOUS_CONTRACT)
    legacy = _json(LEGACY_CONTRACT)

    # The retained revisions are the ones historical Direct anchors were
    # measured under, so they must still hash to their digests.
    assert _canonical_sha256(legacy) == (
        preflight.LEGACY_BACKEND_SERVING_CONTRACT_SHA256
    )
    assert _canonical_sha256(previous) == (
        preflight.PREVIOUS_BACKEND_SERVING_CONTRACT_SHA256
    )
    assert _canonical_sha256(current) == (
        preflight.EXPECTED_BACKEND_SERVING_CONTRACT_SHA256
    )
    assert len({_canonical_sha256(c) for c in (legacy, previous, current)}) == 3

    # v1 -> v2.5 changed only the accelerator declaration; v2.5 -> v3 changed
    # only the runtime-image binding.  Everything else that determines what is
    # measured is identical across the three.
    keys = preflight.SERVING_SYSTEM_IDENTITY_KEYS
    assert set(keys) == {"behavior", "model", "runtime", "schema_version"}
    binding = preflight.RUNTIME_IMAGE_BINDING_KEYS
    assert binding == {"runtime_image_id", "runtime_image_ref", "runtime_image_recipe"}

    def measured(contract: dict) -> dict:
        projected = {key: contract[key] for key in keys}
        projected["runtime"] = {
            k: v for k, v in contract["runtime"].items() if k not in binding
        }
        return projected

    assert measured(legacy) == measured(previous) == measured(current)
    assert {key: legacy[key] for key in keys} == {key: previous[key] for key in keys}
    assert legacy["accelerator_capability"]["allowed_names"] == ["NVIDIA H100 NVL"]
    assert previous["accelerator_capability"] == current["accelerator_capability"]


def test_v3_binds_the_runtime_image_to_its_recipe(preflight) -> None:
    current = _json(CURRENT_CONTRACT)
    previous = _json(PREVIOUS_CONTRACT)
    runtime = current["runtime"]
    recipe = runtime["runtime_image_recipe"]

    assert current["contract_revision"] == "v3-recipe-bound-runtime"
    # v1 / v2.5 pinned the Docker image ID of the qualification host's build.
    assert previous["runtime"]["runtime_image_id"].startswith("sha256:1d8369b9")
    assert "runtime_image_recipe" not in previous["runtime"]
    # v3's identity is the sha256 of the recipe, and the recipe is the
    # digest-pinned official vLLM release image built by the in-tree Dockerfile.
    assert runtime["runtime_image_id"] == "sha256:" + _canonical_sha256(recipe)
    assert recipe["dockerfile"] == "images/backend/Dockerfile"
    assert recipe["dockerfile_sha256"] == hashlib.sha256(
        BACKEND_DOCKERFILE.read_bytes()
    ).hexdigest()
    assert recipe["base_image"] == "vllm/vllm-openai:v0.24.0"
    assert recipe["vllm_package_version"] == runtime["vllm_package_version"]
    assert recipe["vllm_engine_version"] == runtime["vllm_engine_version"]
    assert (
        f"FROM {recipe['base_image']}@{recipe['base_image_digest']}"
        in BACKEND_DOCKERFILE.read_text(encoding="utf-8")
    )
    assert preflight.validate_runtime_image_recipe(current) == recipe
    # The profile the backend is launched from carries the same binding.
    profile = _json(MODEL_PROFILE)["runtime"]
    assert profile["runtime_image_id"] == runtime["runtime_image_id"]
    assert profile["runtime_image_recipe"] == recipe
    assert profile["runtime_image_ref"] == runtime["runtime_image_ref"]


def test_preflight_rejects_a_stale_or_edited_backend_dockerfile(
    preflight, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    edited = tmp_path / "Dockerfile"
    edited.write_text(BACKEND_DOCKERFILE.read_text(encoding="utf-8") + "RUN true\n")
    monkeypatch.setattr(preflight, "BACKEND_DOCKERFILE", edited)
    monkeypatch.setattr(preflight, "ROOT", tmp_path)
    (tmp_path / "images/backend").mkdir(parents=True)
    (tmp_path / "images/backend/Dockerfile").write_bytes(edited.read_bytes())
    monkeypatch.setattr(preflight, "BACKEND_DOCKERFILE", tmp_path / "images/backend/Dockerfile")
    with pytest.raises(RuntimeError, match="recipe is stale"):
        preflight.validate_runtime_image_recipe(_json(CURRENT_CONTRACT))

    mutated = _json(CURRENT_CONTRACT)
    mutated["runtime"]["runtime_image_id"] = "sha256:" + "0" * 64
    with pytest.raises(RuntimeError, match="not the recipe digest"):
        preflight.validate_runtime_image_recipe(mutated)


def test_preflight_rejects_a_revision_that_changes_the_serving_system(
    preflight, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert preflight.validate_backend_serving_contract_revisions() == (
        preflight.EXPECTED_BACKEND_SERVING_CONTRACT_SHA256
    )

    mutated = _json(CURRENT_CONTRACT)
    mutated["runtime"]["tensor_parallel_size"] = 4
    path = tmp_path / "backend-serving-contract.json"
    path.write_text(json.dumps(mutated), encoding="utf-8")
    monkeypatch.setattr(preflight, "BACKEND_SERVING_CONTRACT", path)
    with pytest.raises(RuntimeError, match="backend serving contract changed"):
        preflight.validate_backend_serving_contract_revisions()


def test_preflight_rejects_a_reintroduced_device_name_whitelist(
    preflight, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mutated = _json(CURRENT_CONTRACT)
    mutated["accelerator_capability"] = dict(_json(LEGACY_CONTRACT)["accelerator_capability"])
    path = tmp_path / "backend-serving-contract.json"
    path.write_text(json.dumps(mutated), encoding="utf-8")
    monkeypatch.setattr(preflight, "BACKEND_SERVING_CONTRACT", path)
    monkeypatch.setattr(
        preflight,
        "EXPECTED_BACKEND_SERVING_CONTRACT_SHA256",
        _canonical_sha256(mutated),
    )

    with pytest.raises(RuntimeError, match="capability predicate"):
        preflight.validate_backend_serving_contract_revisions()


def test_every_trusted_component_shares_one_revision_registry(preflight) -> None:
    judge = load_module("kvbench_capability_judge", ROOT / "judge/run_judge.py")
    public = load_module("kvbench_capability_public", ROOT / "judge/run_public.py")
    final = load_module(
        "kvbench_capability_final", ROOT / "scripts/validate_final_evaluation.py"
    )
    runner = load_module(
        "kvbench_capability_runner",
        ROOT
        / "runtime/frozen/harbor_run_scripts/"
        "concurrent-agent-serving-optimizer-qwen36-27b-swe192-v2.1-gpu45/"
        "run-gpu67-v2.1-pair.py",
    )
    expected = {
        preflight.EXPECTED_BACKEND_SERVING_CONTRACT_SHA256,
        preflight.PREVIOUS_BACKEND_SERVING_CONTRACT_SHA256,
        preflight.LEGACY_BACKEND_SERVING_CONTRACT_SHA256,
    }
    for module in (judge, public, final, runner):
        assert module.ACCEPTED_BACKEND_SERVING_CONTRACT_SHA256 == expected


def test_agent_visible_contract_states_the_envelope_not_a_device_model() -> None:
    contract = _json(RUNTIME_CONTRACT)

    assert "gpu_type" not in contract["runtime"]
    assert contract["runtime"]["accelerator"] == {
        "count": DEVICE_COUNT,
        "min_compute_capability": MIN_COMPUTE_CAPABILITY,
        "minimum_memory_mib": MIN_DEVICE_MEMORY_MIB,
    }
    # The page geometry is a measurement of one backend boot; the contract has
    # to say so rather than present it as a task constant.
    measured = contract["hybrid_cache"]["measured_on_profile"]
    assert "profile_id" not in measured
    assert measured["page_geometry_is_a_profile_measurement"] is True
    assert measured["refreshed_by_launcher_preflight"] is True
    assert measured["comparable_only_within_one_measurement_profile"] is True
    assert measured["reported_by"] == "vllm:cache_config_info"


def test_agent_visible_surfaces_name_no_device_model() -> None:
    for path in (
        RUNTIME_CONTRACT,
        ROOT / "images/work/AGENT_TASK.md",
        CURRENT_CONTRACT,
        ROOT / f"tasks/{TASK_ID}.json",
    ):
        text = path.read_text(encoding="utf-8").lower()
        assert "h100" not in text, path
        assert "hopper" not in text, path


def test_trusted_sources_keep_device_names_only_as_historical_identifiers() -> None:
    """No trusted component may compare a GPU model name.

    The one string that legitimately survives is the boot profile id recorded
    in the pre-contract model provenance: it identifies a historical
    measurement, is never compared against a device, and cannot be renamed
    without invalidating archived Direct evidence.
    """

    allowed = {"qwen3.6-27b-6a9e13bd-h100-nvl-tp2-kv30g-v2"}
    for relative in (
        "judge/run_judge.py",
        "judge/run_public.py",
        "scripts/preflight.py",
        "scripts/sforge_runtime_entry.py",
        "scripts/validate_final_evaluation.py",
        "runtime/frozen/harbor_run_scripts/"
        "concurrent-agent-serving-optimizer-qwen36-27b-swe192-v2.1-gpu45/"
        "run-gpu67-v2.1-pair.py",
        "runtime/frozen/harbor_tasks/"
        "concurrent-agent-serving-optimizer-qwen36-27b-swe192-v2.1-gpu45/"
        "environment/bench-controller/server.py",
        "runtime/frozen/harbor_tasks/"
        "concurrent-agent-serving-optimizer-qwen36-27b-swe192-v2.1-gpu45/"
        "environment/model-api/server.py",
    ):
        path = ROOT / relative
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        literals = {
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and ("h100" in node.value.lower() or "hopper" in node.value.lower())
        }
        assert literals <= allowed, (relative, sorted(literals - allowed))


def test_frozen_profile_admits_any_device_meeting_the_envelope(preflight) -> None:
    gpu = _json(MODEL_PROFILE)["gpu_contract"]

    assert "allowed_names" not in gpu
    assert gpu["count"] == DEVICE_COUNT
    assert gpu["min_compute_capability"] == MIN_COMPUTE_CAPABILITY
    assert gpu["minimum_memory_mib"] == MIN_DEVICE_MEMORY_MIB
    # The qualification device stays recorded as provenance and is read by
    # nothing: the Work-visible contract derived from this profile names no
    # device at all.
    assert gpu["qualified_on_device_names"] == ["NVIDIA H100 NVL"]
    rendered = json.dumps(_json(RUNTIME_CONTRACT))
    for name in FOREIGN_DEVICE_NAMES + ("NVIDIA H100 NVL",):
        assert name not in rendered


def test_live_page_geometry_verifies_or_records_unverified(
    preflight, monkeypatch: pytest.MonkeyPatch
) -> None:
    declared = _json(RUNTIME_CONTRACT)["hybrid_cache"]["raw_gpu_pages"]

    def metrics(body: str):
        def _get(_url: str, path: str) -> bytes:
            assert path == "/metrics"
            return body.encode()

        return _get

    tokens = preflight._oracle_integer_default(
        ROOT / "oracle/reference_gateway/gateway.py",
        constant_name="KV_CAPACITY_TOKENS",
        environment_name="THUNDER_KV_CAPACITY_TOKENS",
    )
    sample = (
        'vllm:cache_config_info{block_size="16",'
        f'kv_cache_size_tokens="{tokens}",num_gpu_blocks="{declared}"}} 1.0\n'
    )
    monkeypatch.setattr(preflight, "_backend_get", metrics(sample))
    record = preflight.live_page_geometry("http://backend:8000")
    assert record["verified_against_live_backend"] is True
    assert record["observed_num_gpu_blocks"] == declared

    # A backend with a different KV geometry stops the launcher instead of
    # handing the agent numbers measured somewhere else.
    drifted = sample.replace(f'num_gpu_blocks="{declared}"', 'num_gpu_blocks="900"')
    monkeypatch.setattr(preflight, "_backend_get", metrics(drifted))
    with pytest.raises(RuntimeError, match="refresh the profile"):
        preflight.live_page_geometry("http://backend:8000")

    # A backend without the endpoint leaves the geometry marked unverified.
    def unreachable(_url: str, _path: str) -> bytes:
        raise RuntimeError("backend /metrics is unreachable")

    monkeypatch.setattr(preflight, "_backend_get", unreachable)
    record = preflight.live_page_geometry("http://backend:8000")
    assert record["verified_against_live_backend"] is False
    assert record["observed_num_gpu_blocks"] is None
    assert record["declared_raw_gpu_pages"] == declared
