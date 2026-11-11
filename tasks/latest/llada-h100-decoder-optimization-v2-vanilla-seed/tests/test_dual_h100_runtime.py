from __future__ import annotations

import importlib.util
import gzip
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess

import pytest


TASK_ID = "llada_h100_decoder_optimization_v2"


def _load_entry(task_root: Path):
    path = task_root / "scripts/sforge_h100_entry.py"
    spec = importlib.util.spec_from_file_location("llada_h100_runtime_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeNetwork:
    attrs = {"Internal": True}

    def reload(self) -> None:
        return None


class _FakeNetworks:
    def __init__(self) -> None:
        self.requested: list[str] = []

    def get(self, name: str) -> _FakeNetwork:
        self.requested.append(name)
        return _FakeNetwork()


class _FakeContainer:
    id = "fake-container-id"
    name = "fake-container"


class _FakeContainers:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def create(self, image: str, **kwargs):
        self.calls.append((image, kwargs))
        return _FakeContainer()


class _FakeImage:
    def __init__(self, image_id: str) -> None:
        self.id = image_id


class _FakeImages:
    def __init__(self, mapping: dict[str, str]) -> None:
        self.mapping = mapping

    def get(self, image: str) -> _FakeImage:
        return _FakeImage(self.mapping[image])


class _FakeClient:
    def __init__(self) -> None:
        self.networks = _FakeNetworks()
        self.containers = _FakeContainers()
        work_tag = f"llada_decoder.work.{TASK_ID}:local"
        judge_tag = f"llada_decoder.judge.{TASK_ID}:local"
        work_id = "sha256:" + "a" * 64
        judge_id = "sha256:" + "b" * 64
        self.images = _FakeImages(
            {
                work_tag: work_id,
                judge_tag: judge_id,
                work_id: work_id,
                judge_id: judge_id,
            }
        )


class _FakeBackend:
    def __init__(self) -> None:
        self.client = _FakeClient()


def _model_repo(tmp_path: Path, revision: str) -> Path:
    repo = tmp_path / "models--GSAI-ML--LLaDA-8B-Instruct"
    (repo / "snapshots" / revision).mkdir(parents=True)
    return repo


@pytest.mark.parametrize(
    "image,name,expected",
    [
        (
            f"llada_decoder.work.{TASK_ID}:local",
            f"sforge.run.{TASK_ID}.formal-run",
            "work",
        ),
        (
            f"llada_decoder.judge.{TASK_ID}:local",
            f"llada_decoder.judge.{TASK_ID}.manual-1",
            "judge",
        ),
        (
            f"llada_decoder.judge.{TASK_ID}:local",
            f"sforge.game.{TASK_ID}.abcdef123456",
            "judge",
        ),
    ],
)
def test_actual_sforge_container_names_classify_by_role(
    task_root: Path, image: str, name: str, expected: str
) -> None:
    entry = _load_entry(task_root)
    assert entry._classify_role(image, name) == expected


def test_ambiguous_or_mismatched_container_role_fails_closed(task_root: Path) -> None:
    entry = _load_entry(task_root)
    work_image = f"llada_decoder.work.{TASK_ID}:local"
    judge_name = f"llada_decoder.judge.{TASK_ID}.manual-1"
    with pytest.raises(RuntimeError, match="role mismatch"):
        entry._classify_role(work_image, judge_name)
    with pytest.raises(RuntimeError, match="unambiguously classify"):
        entry._classify_role("sha256:unknown", "unknown")


@pytest.mark.parametrize(
    "value",
    ["", "0", "2", "all", "void", "none", "GPU-one,GPU-two", " GPU-one, GPU-two "],
)
def test_gpu_selector_rejects_anything_except_one_uuid(
    task_root: Path, value: str
) -> None:
    entry = _load_entry(task_root)
    with pytest.raises(RuntimeError, match="exactly one|is required"):
        entry._single_gpu_uuid(value, field="NVIDIA_VISIBLE_DEVICES")


def test_each_role_gets_exactly_one_matching_nvidia_device_request(
    task_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entry = _load_entry(task_root)
    repo = _model_repo(tmp_path, entry.MODEL_REVISION)
    monkeypatch.setenv(entry.MODEL_REPO_ENV, str(repo))
    monkeypatch.setenv(entry.NETWORK_ENV, "formal-internal")
    monkeypatch.setenv(entry.WORK_UUID_ENV, "GPU-work-uuid")
    monkeypatch.setenv(entry.JUDGE_UUID_ENV, "GPU-judge-uuid")
    monkeypatch.setenv(entry.RUN_SCOPE_ENV, "unit-dual-h100")
    monkeypatch.setenv(entry.WORK_IMAGE_ID_ENV, "sha256:" + "a" * 64)
    monkeypatch.setenv(entry.JUDGE_IMAGE_ID_ENV, "sha256:" + "b" * 64)
    monkeypatch.setattr(
        entry,
        "_judge_checkpoint_volume",
        lambda: "/logs/runs/unit-dual-h100/judge-checkpoints:/logs/verifier/checkpoints:rw",
    )
    backend = _FakeBackend()

    cases = [
        (
            "work",
            f"llada_decoder.work.{TASK_ID}:local",
            f"sforge.run.{TASK_ID}.formal-run",
            "GPU-work-uuid",
        ),
        (
            "judge",
            f"llada_decoder.judge.{TASK_ID}:local",
            f"llada_decoder.judge.{TASK_ID}.manual-1",
            "GPU-judge-uuid",
        ),
    ]
    for role, image, name, uuid in cases:
        entry.role_bound_create_container(
            backend,
            image,
            name,
            environment={
                "NVIDIA_VISIBLE_DEVICES": uuid,
                "CUDA_VISIBLE_DEVICES": "0",
                "LLADA_TARGET": entry.EXECUTION_PROFILE_ID,
                "LLADA_GPU_ROLE": role,
                **(
                    {
                        "LLADA_INTERACTIVE_JUDGE_TIMEOUT": "2400",
                        "LLADA_FINAL_JUDGE_TIMEOUT": "259000",
                    }
                    if role == "judge"
                    else {}
                ),
            },
            extra_hosts={"host.docker.internal": "host-gateway"},
            cap_drop=["NET_RAW"],
            cpu_limit=8,
            mem_limit="48g",
            user="1000:1000" if role == "judge" else None,
            annotations={
                "caller.annotation": "preserved",
                "mlsysforge.run_id": "must-not-override",
            },
        )

    assert len(backend.client.containers.calls) == 2
    work_kwargs = backend.client.containers.calls[0][1]
    judge_kwargs = backend.client.containers.calls[1][1]
    for call, expected_uuid, expected_role, expected_image in (
        (
            backend.client.containers.calls[0],
            "GPU-work-uuid",
            "work",
            "sha256:" + "a" * 64,
        ),
        (
            backend.client.containers.calls[1],
            "GPU-judge-uuid",
            "judge",
            "sha256:" + "b" * 64,
        ),
    ):
        actual_image, kwargs = call
        assert actual_image == expected_image
        requests = kwargs["device_requests"]
        assert len(requests) == 1
        assert requests[0]["Driver"] == "nvidia"
        assert requests[0]["DeviceIDs"] == [expected_uuid]
        assert requests[0]["Capabilities"] == [["gpu"]]
        assert kwargs["environment"]["NVIDIA_VISIBLE_DEVICES"] == expected_uuid
        assert kwargs["environment"]["CUDA_VISIBLE_DEVICES"] == "0"
        assert kwargs["environment"]["LLADA_TARGET"] == entry.EXECUTION_PROFILE_ID
        assert kwargs["environment"]["LLADA_GPU_ROLE"] == expected_role
        assert kwargs["labels"]["mlsysforge.run_id"] == "unit-dual-h100"
        assert kwargs["labels"]["caller.annotation"] == "preserved"
        expected_volumes = [f"{repo}:/opt/models/llada-hf:ro"]
        if expected_role == "judge":
            expected_volumes.append(
                "/logs/runs/unit-dual-h100/judge-checkpoints:"
                "/logs/verifier/checkpoints:rw"
            )
        assert kwargs["volumes"] == expected_volumes
        assert kwargs["security_opt"] == ["no-new-privileges:true"]
    assert work_kwargs["network"] == "formal-internal"
    assert "network_mode" not in work_kwargs
    assert judge_kwargs["network_mode"] == "none"
    assert "network" not in judge_kwargs
    assert backend.client.networks.requested == ["formal-internal"]


@pytest.mark.parametrize(
    "flag,web_search,offline_prompt",
    [(None, "live", False), ("true", "live", False), ("false", "disabled", True)],
)
def test_codex_and_prompt_follow_the_task_internet_flag(
    task_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    flag: str | None,
    web_search: str,
    offline_prompt: bool,
) -> None:
    # run_fixed_budget.sh always passes --enable-internet (so the caller asks
    # for internet=True) and exports the task JSON's flag for the entry.
    from sforge.harness.config import SForgeConfig

    entry = _load_entry(task_root)
    if flag is None:
        monkeypatch.delenv(entry.WORK_INTERNET_ENV, raising=False)
    else:
        monkeypatch.setenv(entry.WORK_INTERNET_ENV, flag)
    assert entry.CodexAgent.format_run_cmd is entry.task_internet_format_run_cmd
    assert entry.run_agent.generate_evolve_prompt is entry.task_internet_generate_prompt
    agent = entry.CodexAgent(SForgeConfig(codex_auth_json_path=tmp_path / "auth.json"))
    command = entry.task_internet_format_run_cmd(agent, "/tmp/prompt.md", internet=True)
    assert f'web_search="{web_search}"' in command
    prompt = entry.task_internet_generate_prompt(
        "Optimize the decoder.",
        submit_paths=["decoder_lab/solutions"],
        internet=True,
    )
    assert ("NO internet access" in prompt) is offline_prompt
    if not offline_prompt:
        assert "internet" not in prompt.lower()
    monkeypatch.setenv(entry.WORK_INTERNET_ENV, "1")
    with pytest.raises(RuntimeError, match=entry.WORK_INTERNET_ENV):
        entry.task_internet_format_run_cmd(agent, "/tmp/prompt.md", internet=True)


def test_runner_wires_the_task_internet_flag(task_root: Path) -> None:
    runner = (task_root / "scripts/run_fixed_budget.sh").read_text(encoding="utf-8")
    assert '.internet | if type == "boolean"' in runner
    assert "restricted_proxy_egress_args=(--allow-public-egress)" in runner
    assert '"${restricted_proxy_egress_args[@]}"' in runner
    assert 'export MLSYSFORGE_WORK_INTERNET="${TASK_INTERNET}"' in runner
    assert '--arg codex_web_search "${codex_web_search}"' in runner
    # Work stays on the internal network and must have no direct egress.
    assert "--internal" in runner and "--enable-internet" in runner
    assert '-fsS https://example.com' in runner
    assert "internal work network unexpectedly has direct public egress" in runner
    # Sidecar policy probes for both modes.
    assert 'connect_probes=("chatgpt.com=200")' in runner
    for probe in (
        '"github.com=200"',
        '"${docker_bridge_gateway}=403"',
        '"host.docker.internal=403"',
        'connect_probes+=("github.com=403")',
    ):
        assert probe in runner


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"NVIDIA_VISIBLE_DEVICES": "GPU-judge-uuid"}, "expected GPU-work-uuid"),
        ({"CUDA_VISIBLE_DEVICES": "1"}, "logical ordinal 0"),
        ({"LLADA_TARGET": "NVIDIA-H100"}, "must be dual-cuda-sm80-34g"),
        ({"LLADA_GPU_ROLE": "judge"}, "identify this container as work"),
    ],
)
def test_role_environment_mismatch_fails_before_docker_create(
    task_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    updates: dict[str, str],
    message: str,
) -> None:
    entry = _load_entry(task_root)
    repo = _model_repo(tmp_path, entry.MODEL_REVISION)
    monkeypatch.setenv(entry.MODEL_REPO_ENV, str(repo))
    monkeypatch.setenv(entry.NETWORK_ENV, "formal-internal")
    monkeypatch.setenv(entry.WORK_UUID_ENV, "GPU-work-uuid")
    environment = {
        "NVIDIA_VISIBLE_DEVICES": "GPU-work-uuid",
        "CUDA_VISIBLE_DEVICES": "0",
        "LLADA_TARGET": entry.EXECUTION_PROFILE_ID,
        "LLADA_GPU_ROLE": "work",
    }
    environment.update(updates)
    backend = _FakeBackend()
    with pytest.raises(RuntimeError, match=message):
        entry.role_bound_create_container(
            backend,
            f"llada_decoder.work.{TASK_ID}:local",
            f"sforge.run.{TASK_ID}.formal-run",
            environment=environment,
        )
    assert backend.client.containers.calls == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("LLADA_INTERACTIVE_JUDGE_TIMEOUT", ""),
        ("LLADA_FINAL_JUDGE_TIMEOUT", "0"),
        ("LLADA_FINAL_JUDGE_TIMEOUT", "nan"),
    ],
)
def test_judge_container_requires_explicit_positive_role_timeouts(
    task_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: str,
) -> None:
    entry = _load_entry(task_root)
    monkeypatch.setenv(entry.JUDGE_UUID_ENV, "GPU-judge-uuid")
    environment = {
        "NVIDIA_VISIBLE_DEVICES": "GPU-judge-uuid",
        "CUDA_VISIBLE_DEVICES": "0",
        "LLADA_TARGET": entry.EXECUTION_PROFILE_ID,
        "LLADA_GPU_ROLE": "judge",
        "LLADA_INTERACTIVE_JUDGE_TIMEOUT": "2400",
        "LLADA_FINAL_JUDGE_TIMEOUT": "259000",
    }
    environment[field] = value
    with pytest.raises(RuntimeError, match=field):
        entry._validated_container_environment("judge", environment)


def test_missing_host_run_scope_fails_before_docker_create(
    task_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entry = _load_entry(task_root)
    repo = _model_repo(tmp_path, entry.MODEL_REVISION)
    monkeypatch.setenv(entry.MODEL_REPO_ENV, str(repo))
    monkeypatch.setenv(entry.NETWORK_ENV, "formal-internal")
    monkeypatch.setenv(entry.WORK_UUID_ENV, "GPU-work-uuid")
    monkeypatch.setenv(entry.WORK_IMAGE_ID_ENV, "sha256:" + "a" * 64)
    monkeypatch.delenv(entry.RUN_SCOPE_ENV, raising=False)
    backend = _FakeBackend()
    with pytest.raises(RuntimeError, match=entry.RUN_SCOPE_ENV):
        entry.role_bound_create_container(
            backend,
            f"llada_decoder.work.{TASK_ID}:local",
            f"sforge.run.{TASK_ID}.formal-run",
            environment={
                "NVIDIA_VISIBLE_DEVICES": "GPU-work-uuid",
                "CUDA_VISIBLE_DEVICES": "0",
                "LLADA_TARGET": entry.EXECUTION_PROFILE_ID,
                "LLADA_GPU_ROLE": "work",
            },
        )
    assert backend.client.containers.calls == []


def test_judge_checkpoint_mount_fails_closed_outside_exact_run_scope(
    task_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entry = _load_entry(task_root)
    logs = tmp_path / "logs"
    outside = tmp_path / "outside"
    logs.mkdir()
    outside.mkdir()
    monkeypatch.setenv("SFORGE_LOG_DIR", str(logs))
    monkeypatch.setenv(entry.RUN_SCOPE_ENV, "unit-run")
    monkeypatch.delenv(entry.JUDGE_CHECKPOINT_DIR_ENV, raising=False)
    with pytest.raises(RuntimeError, match=entry.JUDGE_CHECKPOINT_DIR_ENV):
        entry._judge_checkpoint_volume()
    monkeypatch.setenv(entry.JUDGE_CHECKPOINT_DIR_ENV, str(outside))
    with pytest.raises(RuntimeError, match="outside its exact run scope"):
        entry._judge_checkpoint_volume()


def _run_container_claim_probe(
    task_root: Path,
    tmp_path: Path,
    inspect_payload: dict[str, object],
    uuid: str = "GPU-selected-uuid",
    index: str = "2",
) -> str:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    payload_path = tmp_path / "inspect.json"
    payload_path.write_text(json.dumps([inspect_payload]), encoding="utf-8")
    docker = fake_bin / "docker"
    docker.write_text(
        "#!/usr/bin/env bash\n"
        "if [[ $1 == ps ]]; then echo container-id; exit 0; fi\n"
        "if [[ $1 == inspect ]]; then cat \"$FAKE_INSPECT_JSON\"; exit 0; fi\n"
        "exit 2\n",
        encoding="utf-8",
    )
    docker.chmod(0o755)
    command = f"""
source {task_root / 'scripts/lib.sh'}
if running_container_claims_gpu {uuid} {index}; then
  printf 'claimed:%s\\n' "$GPU_CLAIMING_CONTAINER"
else
  printf 'free\\n'
fi
"""
    result = subprocess.run(
        ["bash", "-c", command],
        check=True,
        text=True,
        capture_output=True,
        env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "FAKE_INSPECT_JSON": str(payload_path),
        },
    )
    return result.stdout.strip()


def _container_inspect(
    *, device_requests=None, environment=None, labels=None
) -> dict[str, object]:
    return {
        "Id": "container-id",
        "Name": "/other-run",
        "HostConfig": {"DeviceRequests": device_requests or []},
        "Config": {"Env": environment or [], "Labels": labels or {}},
    }


@pytest.mark.parametrize(
    "inspect_payload",
    [
        _container_inspect(
            device_requests=[
                {
                    "Driver": "nvidia",
                    "Count": 0,
                    "DeviceIDs": ["GPU-selected-uuid"],
                    "Capabilities": [["gpu"]],
                }
            ]
        ),
        _container_inspect(environment=["NVIDIA_VISIBLE_DEVICES=GPU-selected-uuid"]),
        _container_inspect(environment=["NVIDIA_VISIBLE_DEVICES=2"]),
        _container_inspect(environment=["NVIDIA_VISIBLE_DEVICES=all"]),
        _container_inspect(
            device_requests=[
                {
                    "Driver": "nvidia",
                    "Count": -1,
                    "DeviceIDs": [],
                    "Capabilities": [["gpu"]],
                }
            ]
        ),
    ],
)
def test_running_container_assignment_is_rejected_even_without_compute_process(
    task_root: Path, tmp_path: Path, inspect_payload: dict[str, object]
) -> None:
    assert _run_container_claim_probe(task_root, tmp_path, inspect_payload) == "claimed:other-run"


def test_unrelated_cpu_container_does_not_claim_selected_gpu(
    task_root: Path, tmp_path: Path
) -> None:
    assert _run_container_claim_probe(task_root, tmp_path, _container_inspect()) == "free"


def _run_policy_loader(
    task_root: Path,
    function: str,
    policy: Path,
    *,
    env_updates: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    command = (
        f"source {shlex.quote(str(task_root / 'scripts/lib.sh'))}\n"
        f"{function} {shlex.quote(str(policy))}\n"
        "if [[ $1 == memory ]]; then\n"
        "  printf '%s %s %s\\n' \"$QUALIFIED_MAX_PEAK_GIB\" \"$HEADROOM_MIB\" \"$GPU_ADMISSION_MIN_FREE_MIB\"\n"
        "else\n"
        "  printf '%s %s %s\\n' \"$POLICY_SUBMISSION_COOLDOWN\" \"$POLICY_FINAL_OUTER_TIMEOUT\" \"$POLICY_FINAL_CASE_TIMEOUT\"\n"
        "fi\n"
    )
    return subprocess.run(
        ["bash", "-c", command, "policy-loader", "memory" if function.endswith("memory_policy") else "workload"],
        text=True,
        capture_output=True,
        env={**os.environ, **(env_updates or {})},
    )


def test_hidden_manifest_hash_supports_deterministic_gzip_without_plaintext(
    task_root: Path, tmp_path: Path
) -> None:
    payload = b'[{"case_id":"unit"}]\n'
    plain = tmp_path / "hidden_cases.json"
    compressed = tmp_path / "hidden_cases.json.gz"
    compressed.write_bytes(gzip.compress(payload, mtime=0))
    command = (
        f"source {shlex.quote(str(task_root / 'scripts/lib.sh'))}\n"
        f"canonical_hidden_cases_sha256 {shlex.quote(str(plain))} "
        f"{shlex.quote(str(compressed))}\n"
    )
    result = subprocess.run(["bash", "-c", command], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == hashlib.sha256(payload).hexdigest()

    plain.write_bytes(b"different\n")
    result = subprocess.run(["bash", "-c", command], text=True, capture_output=True)
    assert result.returncode != 0
    assert "plain/compressed hidden case bytes disagree" in result.stderr


def test_formal_memory_policy_gate_requires_frozen_qualified_values(
    task_root: Path, tmp_path: Path
) -> None:
    """The frozen loader exposes provenance, and a floor set by the envelope.

    The admission floor must not follow the peak measured on the qualification
    GPU: the enforced peak cap is derived on-device from the visible device's
    own capacity, so the host floor is the capability envelope.
    """
    policy = tmp_path / "memory-policy.json"
    policy.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "task_id": TASK_ID,
                "qualification_hardware": "NVIDIA H100 NVL",
                "min_compute_capability": "8.0",
                "min_device_memory_mib": 34816,
                "qualification_status": "QUALIFIED",
                "release_status": "FROZEN",
                "qualified_max_peak_gib": 80.0001,
                "headroom_mib": 1024,
                "safety_margin_mib": 512,
                "qualification_run_sha256": "a" * 64,
                "approval_id": "reviewed-memory-v1",
            }
        ),
        encoding="utf-8",
    )
    result = _run_policy_loader(task_root, "load_frozen_memory_policy", policy)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "80.0001 1024 34816"

    # A far larger qualification peak must not raise the admission floor.
    payload = json.loads(policy.read_text(encoding="utf-8"))
    payload["qualified_max_peak_gib"] = 130.0
    policy.write_text(json.dumps(payload), encoding="utf-8")
    result = _run_policy_loader(task_root, "load_frozen_memory_policy", policy)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().split()[-1] == "34816"

    # An explicit operator floor still overrides the envelope default.
    result = _run_policy_loader(
        task_root,
        "load_frozen_memory_policy",
        policy,
        env_updates={"GPU_MIN_FREE_MIB": "40960"},
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().split()[-1] == "40960"

    payload["qualification_status"] = "UNQUALIFIED"
    payload["release_status"] = "PENDING"
    policy.write_text(json.dumps(payload), encoding="utf-8")
    result = _run_policy_loader(task_root, "load_frozen_memory_policy", policy)
    assert result.returncode != 0
    assert "not QUALIFIED/FROZEN" in result.stderr

    payload["qualification_status"] = "QUALIFIED"
    payload["release_status"] = "FROZEN"
    payload["min_device_memory_mib"] = 34816
    del payload["qualification_hardware"]
    policy.write_text(json.dumps(payload), encoding="utf-8")
    result = _run_policy_loader(task_root, "load_frozen_memory_policy", policy)
    assert result.returncode != 0
    assert "not QUALIFIED/FROZEN" in result.stderr


def test_initial_qualification_loaders_require_pending_policies_and_explicit_limits(
    task_root: Path, tmp_path: Path
) -> None:
    # The tracked policy is frozen, so build the bootstrap placeholder the
    # unqualified loader is meant to accept instead of copying release state.
    memory_policy = tmp_path / "memory-policy.json"
    frozen = json.loads((task_root / "memory_policy.json").read_text(encoding="utf-8"))
    placeholder = {
        **frozen,
        "qualification_status": "UNQUALIFIED",
        "release_status": "PENDING",
        "qualified_max_peak_gib": None,
        "headroom_mib": None,
        "qualified_clean_idle_free_mib": None,
        "safety_margin_mib": None,
        "qualification_run_sha256": None,
        "approval_id": None,
        "row_sentinel": {
            **frozen["row_sentinel"],
            "status": "PENDING",
            "max_growth_bytes": None,
            "observed_max_growth_bytes": None,
            "observed_max_repeat_growth_bytes": None,
            "safety_margin_bytes": None,
        },
    }
    memory_policy.write_text(json.dumps(placeholder), encoding="utf-8")
    result = _run_policy_loader(
        task_root,
        "load_unqualified_memory_policy",
        memory_policy,
        env_updates={"LLADA_QUALIFICATION_MIN_FREE_MIB": "91000"},
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "91000"

    # Without an operator override the qualification floor is the envelope.
    result = _run_policy_loader(
        task_root, "load_unqualified_memory_policy", memory_policy
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "34816"

    # Same for the workload policy: the tracked artifact is already qualified.
    workload_policy = tmp_path / "workload-policy.json"
    tracked = json.loads(
        (task_root / "workload_policy.json").read_text(encoding="utf-8")
    )
    workload_placeholder = {
        **tracked,
        "qualification_status": "UNQUALIFIED",
        "release_status": "PENDING",
        "interactive": {
            **tracked["interactive"],
            "status": "PROVISIONAL_UNQUALIFIED",
            "case_timeout_seconds": None,
            "judge_timeout_seconds": None,
            "cooldown_seconds": None,
        },
        "final": {
            **tracked["final"],
            "case_timeout_seconds": None,
            "row_timeout_seconds": None,
            "inner_timeout_seconds": None,
            "outer_timeout_seconds": None,
        },
        "quality_policy": {
            **tracked["quality_policy"],
            "status": "UNQUALIFIED",
            "interactive": "same_as_final",
            "final": "PENDING_CONTROL_QUALIFICATION",
        },
    }
    workload_policy.write_text(json.dumps(workload_placeholder), encoding="utf-8")
    limits = {
        "LLADA_QUALIFICATION_CASE_TIMEOUT": "600",
        "LLADA_QUALIFICATION_INTERACTIVE_JUDGE_TIMEOUT": "2400",
        "LLADA_QUALIFICATION_ROW_TIMEOUT": "4000",
        "LLADA_QUALIFICATION_INNER_TIMEOUT": "7000",
        "LLADA_QUALIFICATION_OUTER_TIMEOUT": "7200",
    }
    result = _run_policy_loader(
        task_root,
        "load_unqualified_workload_policy",
        workload_policy,
        env_updates=limits,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "0 7200 600"

    result = _run_policy_loader(
        task_root,
        "load_unqualified_workload_policy",
        workload_policy,
        env_updates={key: "" for key in limits},
    )
    assert result.returncode != 0
    assert "must be a positive integer" in result.stderr

    payload = json.loads(memory_policy.read_text(encoding="utf-8"))
    payload["release_status"] = "FROZEN"
    memory_policy.write_text(json.dumps(payload), encoding="utf-8")
    result = _run_policy_loader(
        task_root,
        "load_unqualified_memory_policy",
        memory_policy,
        env_updates={"LLADA_QUALIFICATION_MIN_FREE_MIB": "91000"},
    )
    assert result.returncode != 0
    assert "not UNQUALIFIED/PENDING" in result.stderr


def test_formal_workload_policy_gate_uses_frozen_dynamic_limits(
    task_root: Path, tmp_path: Path
) -> None:
    payload = json.loads((task_root / "workload_policy.json").read_text(encoding="utf-8"))
    payload["qualification_status"] = "QUALIFIED"
    payload["release_status"] = "FROZEN"
    payload["quality_policy"].update(
        {
            "status": "QUALIFIED",
            "interactive": "same_as_final",
            "final": "paired_non_inferiority_hard_gate",
            "provisional_parent_sha256": "d" * 64,
        }
    )
    payload["provisional_parent_sha256"] = "d" * 64
    payload["manifest"]["hidden_cases_sha256"] = "b" * 64
    payload["manifest"]["dataset_manifest_sha256"] = "c" * 64
    payload["interactive"].update(
        {
            "status": "QUALIFIED",
            "case_timeout_seconds": 20.5,
            "judge_timeout_seconds": 3999.5,
            "cooldown_seconds": 300.0,
        }
    )
    payload["final"].update(
        {
            "case_timeout_seconds": 20.5,
            "row_timeout_seconds": 300.5,
            "inner_timeout_seconds": 3999.5,
            "outer_timeout_seconds": 4000.0,
        }
    )
    policy = tmp_path / "workload-policy.json"
    policy.write_text(json.dumps(payload), encoding="utf-8")
    result = _run_policy_loader(task_root, "load_frozen_workload_policy", policy)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "300 4000 20.5"

    payload["final"]["outer_timeout_seconds"] = None
    policy.write_text(json.dumps(payload), encoding="utf-8")
    result = _run_policy_loader(task_root, "load_frozen_workload_policy", policy)
    assert result.returncode != 0
    assert "not QUALIFIED/FROZEN" in result.stderr


def test_producer_calibration_requires_provisional_not_release_policy(
    task_root: Path, tmp_path: Path
) -> None:
    payload = json.loads((task_root / "workload_policy.json").read_text(encoding="utf-8"))
    payload["qualification_status"] = "QUALIFIED"
    payload["release_status"] = "PROVISIONAL"
    # The tracked policy is FROZEN with a qualified quality policy; a
    # provisional producer policy carries the pre-control shape.
    payload["quality_policy"] = {
        "final": "PENDING_CONTROL_QUALIFICATION",
        "interactive": "same_as_final",
        "status": "UNQUALIFIED",
    }
    payload["interactive"].update(
        {
            "status": "QUALIFIED",
            "case_timeout_seconds": 20.5,
            "judge_timeout_seconds": 3999.5,
            "cooldown_seconds": 300.0,
        }
    )
    payload["final"].update(
        {
            "case_timeout_seconds": 20.5,
            "row_timeout_seconds": 300.5,
            "inner_timeout_seconds": 3999.5,
            "outer_timeout_seconds": 4000.0,
        }
    )
    policy = tmp_path / "provisional-workload-policy.json"
    policy.write_text(json.dumps(payload), encoding="utf-8")
    result = _run_policy_loader(
        task_root, "load_provisional_workload_policy", policy
    )
    assert result.returncode == 0, result.stderr

    payload["release_status"] = "FROZEN"
    payload["quality_policy"].update(
        {
            "status": "QUALIFIED",
            "final": "paired_non_inferiority_hard_gate",
        }
    )
    policy.write_text(json.dumps(payload), encoding="utf-8")
    result = _run_policy_loader(
        task_root, "load_provisional_workload_policy", policy
    )
    assert result.returncode != 0
    assert "not QUALIFIED/PROVISIONAL" in result.stderr


def test_frozen_policy_parent_chain_must_be_self_consistent(
    task_root: Path, tmp_path: Path
) -> None:
    payload = json.loads((task_root / "workload_policy.json").read_text(encoding="utf-8"))
    payload["qualification_status"] = "QUALIFIED"
    payload["release_status"] = "FROZEN"
    payload["provisional_parent_sha256"] = "a" * 64
    payload["quality_policy"].update(
        {
            "status": "QUALIFIED",
            "final": "paired_non_inferiority_hard_gate",
            "provisional_parent_sha256": "a" * 64,
        }
    )
    payload["interactive"].update(
        {
            "status": "QUALIFIED",
            "case_timeout_seconds": 10,
            "judge_timeout_seconds": 20,
            "cooldown_seconds": 30,
        }
    )
    payload["final"].update(
        {
            "case_timeout_seconds": 10,
            "row_timeout_seconds": 100,
            "inner_timeout_seconds": 1000,
            "outer_timeout_seconds": 1100,
        }
    )
    policy = tmp_path / "frozen.json"
    policy.write_text(json.dumps(payload), encoding="utf-8")
    command = (
        f"source {shlex.quote(str(task_root / 'scripts/lib.sh'))}\n"
        f"frozen_workload_parent_sha256 {shlex.quote(str(policy))}\n"
    )
    result = subprocess.run(["bash", "-c", command], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "a" * 64

    payload["quality_policy"]["provisional_parent_sha256"] = "b" * 64
    policy.write_text(json.dumps(payload), encoding="utf-8")
    result = subprocess.run(["bash", "-c", command], text=True, capture_output=True)
    assert result.returncode != 0
    assert "no consistent provisional parent chain" in result.stderr


def test_quality_control_binding_is_exact_and_fail_closed(
    task_root: Path, tmp_path: Path
) -> None:
    manifest = tmp_path / "quality-controls.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "task_id": TASK_ID,
                "profile": "fast_dllm_v1_llada_12row_stratified520_batch1",
                "status": "PRECOMMITTED",
                "workload_policy_sha256": "a" * 64,
                "hidden_cases_sha256": "b" * 64,
                "positive_controls": [
                    {
                        "id": "positive-v1",
                        "archive_sha256": "c" * 64,
                        "expected_classification": "ACCEPT",
                    }
                ],
                "negative_controls": [],
            }
        ),
        encoding="utf-8",
    )
    command = (
        f"source {shlex.quote(str(task_root / 'scripts/lib.sh'))}\n"
        f"validate_quality_control_binding {shlex.quote(str(manifest))} "
        f"positive-v1 ACCEPT {'c' * 64} {'a' * 64} {'b' * 64}\n"
    )
    result = subprocess.run(["bash", "-c", command], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr

    bad = command.replace("positive-v1 ACCEPT", "positive-v1 REJECT")
    result = subprocess.run(["bash", "-c", bad], text=True, capture_output=True)
    assert result.returncode != 0
    assert "not exactly precommitted" in result.stderr


def test_memory_fixture_binding_executes_and_rejects_changed_bytes(
    task_root: Path, tmp_path: Path
) -> None:
    fixture_root = tmp_path / "fixture-task"
    archives = fixture_root / "qualification-fixtures"
    archives.mkdir(parents=True)
    corrected = archives / "corrected.tar.gz"
    stress = archives / "stress.tar.gz"
    evaluated = tmp_path / "evaluated.tar.gz"
    corrected.write_bytes(b"corrected-archive\n")
    stress.write_bytes(b"stress-archive\n")
    evaluated.write_bytes(corrected.read_bytes())
    workload_sha256 = "a" * 64
    memory_sha256 = "b" * 64
    manifest = fixture_root / "memory_qualification_fixtures.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "task_id": TASK_ID,
                "profile": "fast_dllm_v1_llada_12row_stratified520_batch1",
                "status": "PRECOMMITTED",
                "workload_policy_sha256": workload_sha256,
                "memory_policy_sha256": memory_sha256,
                "approval_id": "memory-review-v1",
                "fixtures": {
                    "corrected_reference": {
                        "kind": "corrected_reference",
                        "archive": "qualification-fixtures/corrected.tar.gz",
                        "archive_sha256": hashlib.sha256(
                            corrected.read_bytes()
                        ).hexdigest(),
                        "archive_size_bytes": corrected.stat().st_size,
                        "stress_extra_mib": None,
                    },
                    "near_cap_stress": {
                        "kind": "near_cap_stress",
                        "archive": "qualification-fixtures/stress.tar.gz",
                        "archive_sha256": hashlib.sha256(
                            stress.read_bytes()
                        ).hexdigest(),
                        "archive_size_bytes": stress.stat().st_size,
                        "stress_extra_mib": 4096,
                    },
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    command = "\n".join(
        (
            f"source {shlex.quote(str(task_root / 'scripts/lib.sh'))}",
            "validate_memory_fixture_binding "
            + " ".join(
                shlex.quote(value)
                for value in (
                    str(manifest),
                    "corrected_reference",
                    str(evaluated),
                    workload_sha256,
                    memory_sha256,
                    str(fixture_root),
                )
            ),
            "printf '%s\\t%s\\t%s\\t%s\\t%s\\n' "
            '"$MEMORY_FIXTURE_MANIFEST_SHA256" '
            '"$MEMORY_FIXTURE_APPROVAL_ID" "$MEMORY_FIXTURE_KIND" '
            '"$MEMORY_FIXTURE_ARCHIVE_SHA256" '
            '"$MEMORY_FIXTURE_ARCHIVE_SIZE_BYTES"',
        )
    )
    completed = subprocess.run(
        ["bash", "-c", command], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stderr
    fields = completed.stdout.strip().split("\t")
    assert fields == [
        hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "memory-review-v1",
        "corrected_reference",
        hashlib.sha256(corrected.read_bytes()).hexdigest(),
        str(corrected.stat().st_size),
    ]

    evaluated.write_bytes(b"changed-after-precommit\n")
    changed = subprocess.run(
        ["bash", "-c", command], capture_output=True, text=True, check=False
    )
    assert changed.returncode != 0
    assert "evaluated archive does not match" in changed.stderr

    evaluated.write_bytes(corrected.read_bytes())
    wrong_policy = command.replace(workload_sha256, "c" * 64)
    rejected = subprocess.run(
        ["bash", "-c", wrong_policy], capture_output=True, text=True, check=False
    )
    assert rejected.returncode != 0
    assert "PRECOMMITTED task/policy binding" in rejected.stderr


def test_runtime_provenance_snapshot_executes_exact_schema_and_file_order(
    task_root: Path, tmp_path: Path
) -> None:
    output = tmp_path / "runtime-provenance.json"
    files = ["sforge/cli.py", "sforge/harness/grading.py"]
    files_json = json.dumps(files)
    digests = [character * 64 for character in "1234567"]
    digests[1] = hashlib.sha256(("\n".join(files) + "\n").encode()).hexdigest()
    command = "\n".join(
        (
            f"source {shlex.quote(str(task_root / 'scripts/lib.sh'))}",
            "write_runtime_provenance_snapshot "
            + " ".join(
                shlex.quote(value)
                for value in (
                    str(output),
                    digests[0],
                    digests[1],
                    str(len(files)),
                    files_json,
                    *digests[2:],
                )
            ),
        )
    )
    completed = subprocess.run(
        ["bash", "-c", command], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "schema_version": 1,
        "sforge_runtime": {
            "scope": "all_sforge_python",
            "sha256": digests[0],
            "files": files,
            "file_count": 2,
            "file_set_sha256": digests[1],
        },
        "judge_invariant_core": {
            "verify_decoder_sha256": digests[2],
            "candidate_worker_sha256": digests[3],
            "cohort_sha256": digests[4],
            "quality_sha256": digests[5],
            "source_admission_sha256": digests[6],
        },
    }
    assert list(tmp_path.glob("runtime-provenance.json.tmp.*")) == []

    invalid = command.replace(files_json, json.dumps(list(reversed(files))))
    rejected = subprocess.run(
        ["bash", "-c", invalid], capture_output=True, text=True, check=False
    )
    assert rejected.returncode != 0
    assert "file set is invalid" in rejected.stderr

    inconsistent = command.replace(digests[1], "f" * 64)
    rejected = subprocess.run(
        ["bash", "-c", inconsistent], capture_output=True, text=True, check=False
    )
    assert rejected.returncode != 0
    assert "file-set digest is inconsistent" in rejected.stderr


def test_eval_runtime_file_json_and_core_hash_helpers_execute_once_in_order(
    task_root: Path, tmp_path: Path
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_docker = fake_bin / "docker"
    fake_docker.write_text(
        """#!/bin/sh
for argument in "$@"; do
  case "$argument" in
    */verify_decoder.py) digit=1 ;;
    */candidate_worker.py) digit=2 ;;
    */cohort.py) digit=3 ;;
    */quality.py) digit=4 ;;
    */source_admission.py) digit=5 ;;
    *) continue ;;
  esac
  printf '%063d%s  %s\\n' 0 "$digit" "$argument"
done
""",
        encoding="utf-8",
    )
    fake_docker.chmod(0o755)
    files = ("sforge/cli.py", "sforge/harness/grading.py")
    command = "\n".join(
        (
            f"source {shlex.quote(str(task_root / 'scripts/lib.sh'))}",
            f"PATH={shlex.quote(str(fake_bin))}:$PATH",
            "payload=$(canonical_sforge_runtime_files_json "
            + " ".join(shlex.quote(item) for item in files)
            + ")",
            "mapfile -t start < <(judge_invariant_core_hashes judge:test)",
            "mapfile -t end < <(judge_invariant_core_hashes judge:test)",
            "printf '%s\\n' \"$payload\"",
            "printf '%s:%s\\n' \"${#start[@]}\" \"${#end[@]}\"",
            "printf '%s\\n' \"${start[*]}\" \"${end[*]}\"",
        )
    )
    completed = subprocess.run(
        ["bash", "-c", command], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stderr
    lines = completed.stdout.splitlines()
    assert json.loads(lines[0]) == list(files)
    assert lines[1] == "5:5"
    expected = " ".join("0" * 63 + digit for digit in "12345")
    assert lines[2:] == [expected, expected]

    evaluation = (task_root / "scripts/eval_archive.sh").read_text(encoding="utf-8")
    assert evaluation.count(
        'canonical_sforge_runtime_files_json "${SFORGE_RUNTIME_FILES[@]}"'
    ) == 1
    assert evaluation.count('judge_invariant_core_hashes "${JUDGE_IMAGE}"') == 2


def test_calibration_manifest_enrichment_branch_executes_once_and_succeeds(
    task_root: Path, tmp_path: Path
) -> None:
    manifest = tmp_path / "evaluation-image-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_digest": "0" * 64,
                "work": {"id": "sha256:" + "1" * 64},
                "judge": {"id": "sha256:" + "2" * 64},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    digests = [character * 64 for character in "3456789a"]
    command = [
        "bash",
        "-c",
        "\n".join(
            (
                f"source {shlex.quote(str(task_root / 'scripts/lib.sh'))}",
                "enrich_calibration_image_manifest "
                + " ".join(
                    shlex.quote(value) for value in (str(manifest), *digests)
                ),
            )
        ),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["judge_runtime_tree_sha256"] == digests[0]
    assert payload["calibration_files"] == {
        "hidden_cases_sha256": digests[1],
        "quality_runtime_sha256": digests[2],
        "verifier_sha256": digests[3],
        "candidate_worker_sha256": digests[4],
        "cohort_sha256": digests[5],
        "workload_policy_sha256": digests[6],
        "memory_policy_sha256": digests[7],
    }
    assert list(tmp_path.glob("evaluation-image-manifest.json.calibration.*")) == []
    evaluation = (task_root / "scripts/eval_archive.sh").read_text(encoding="utf-8")
    assert evaluation.count("enrich_calibration_image_manifest") == 1


def test_resume_identity_is_immutable_and_exact(task_root: Path, tmp_path: Path) -> None:
    run_id = "resume-unit"
    run_root = tmp_path / "runs" / run_id
    run_root.mkdir(parents=True)
    candidate = tmp_path / "identity.json"
    candidate.write_text('{"archive_sha256":"aaa","run_id":"resume-unit"}\n')
    library = shlex.quote(str(task_root / "scripts/lib.sh"))
    initial = (
        f"source {library}\n"
        f"bind_runtime_resume_identity {shlex.quote(str(run_root))} {run_id} 0 "
        f"{shlex.quote(str(candidate))}\n"
    )
    result = subprocess.run(["bash", "-c", initial], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    identity = run_root / "runtime-resume-identity.json"
    assert identity.read_bytes() == candidate.read_bytes()
    assert identity.stat().st_mode & 0o777 == 0o444

    exact_resume = initial.replace(f" {run_id} 0 ", f" {run_id} 1 ")
    result = subprocess.run(
        ["bash", "-c", exact_resume], text=True, capture_output=True
    )
    assert result.returncode == 0, result.stderr
    candidate.write_text('{"archive_sha256":"bbb","run_id":"resume-unit"}\n')
    result = subprocess.run(
        ["bash", "-c", exact_resume], text=True, capture_output=True
    )
    assert result.returncode != 0
    assert "resume identity differs" in result.stderr


def test_quality_control_is_explicit_and_uses_provisional_policy(
    task_root: Path,
) -> None:
    source = (task_root / "scripts/eval_archive.sh").read_text(encoding="utf-8")
    assert 'quality_control_mode="${LLADA_QUALITY_CONTROL:-0}"' in source
    assert "SFORGE_PRIVATE_CONTEXT_QUALITY_CONTROL=1" in source
    assert 'evaluation_mode="quality_control"' in source
    # The workload-policy selection block must reach the provisional loader
    # through the calibration/quality-control branch, and fall through to the
    # frozen loader only for an ordinary release evaluation.
    provisional_branch = source.index('"${quality_control_mode}" == "1" ]]')
    provisional_loader = source.index(
        "load_provisional_workload_policy", provisional_branch
    )
    frozen_loader = source.index("load_frozen_workload_policy", provisional_loader)
    assert provisional_branch < provisional_loader < frozen_loader
    assert 'elif [[ "${qualification_mode}" != "1" ]]; then' in source[
        provisional_loader:frozen_loader
    ]
    assert (
        "evaluation requires a calibrated reference bound to the correct policy parent"
        in source
    )


def test_qualification_qc_parent_and_capacity_paths_are_wired(task_root: Path) -> None:
    scripts = task_root / "scripts"
    evaluation = (scripts / "eval_archive.sh").read_text(encoding="utf-8")
    runner = (scripts / "run_fixed_budget.sh").read_text(encoding="utf-8")
    library = (scripts / "lib.sh").read_text(encoding="utf-8")

    assert 'qualification_mode="${LLADA_QUALIFICATION:-0}"' in evaluation
    assert "load_unqualified_memory_policy" in evaluation
    assert "load_unqualified_workload_policy" in evaluation
    assert "SFORGE_PRIVATE_CONTEXT_QUALIFICATION=1" in evaluation
    assert 'evaluation_mode="qualification"' in evaluation
    assert "SFORGE_PRIVATE_CONTEXT_CONTROL_ID=${control_id}" in evaluation
    assert (
        "SFORGE_PRIVATE_CONTEXT_EXPECTED_CLASSIFICATION=${expected_classification}"
        in evaluation
    )
    assert "validate_quality_control_binding" in evaluation
    for option in (
        "--judge-gpu-uuid",
        "--runtime-provenance",
        "--memory-fixture-manifest",
        "--memory-fixture-kind",
        "--nvml-total-bytes",
        "--nvml-clean-idle-free-bytes",
        "--torch-device-total-bytes",
    ):
        assert option in evaluation
    assert "CONTAINER_GPU_TOTAL_BYTES" in library
    assert ".total_memory_bytes" in library

    assert "frozen_workload_parent_sha256" in runner
    assert (
        ".calibration_provenance.workload_policy_sha256 == "
        "$workload_policy_parent_sha256"
    ) in runner
    assert "frozen_workload_parent_sha256" in evaluation
    assert "enrich_calibration_image_manifest" in evaluation
    assert "calibration_files" in library
    assert "workload_policy_sha256: $workload_policy" in library
    assert "memory_policy_sha256: $memory_policy" in library
    assert "quality_controls.json" in library
    assert "quality_controls_snapshot" in evaluation
    assert "--require-precommitted" in evaluation
    assert "--verify-archive-hashes" in evaluation
    assert "producer_calibration.archive_sha256 == $archive" in evaluation
    assert "quality-control manifest changed during evaluation" in evaluation
    assert "--quality-controls-manifest" in evaluation
    assert (
        "SFORGE_PRIVATE_CONTEXT_QUALITY_CONTROLS_SHA256="
        "${quality_controls_sha256}"
    ) in evaluation
    assert "validate_memory_fixture_binding" in evaluation
    assert "memory-fixture-manifest.json" in evaluation
    assert "memory fixture manifest changed during evaluation" in evaluation
    assert "memory_qualification_fixtures.json" in library


def test_manual_and_formal_runs_bind_complete_runtime_provenance(
    task_root: Path,
) -> None:
    scripts = task_root / "scripts"
    evaluation = (scripts / "eval_archive.sh").read_text(encoding="utf-8")
    runner = (scripts / "run_fixed_budget.sh").read_text(encoding="utf-8")
    private_fields = (
        "RUNTIME_PROVENANCE_SHA256",
        "SFORGE_RUNTIME_SHA256",
        "SFORGE_RUNTIME_FILE_SET_SHA256",
        "SFORGE_RUNTIME_FILE_COUNT",
        "JUDGE_VERIFY_DECODER_SHA256",
        "JUDGE_CANDIDATE_WORKER_SHA256",
        "JUDGE_COHORT_SHA256",
        "JUDGE_QUALITY_SHA256",
        "JUDGE_SOURCE_ADMISSION_SHA256",
    )
    for source in (evaluation, runner):
        assert "write_runtime_provenance_snapshot" in source
        assert "runtime-provenance.json" in source
        assert "SForge Python" in source
        assert "source set changed" in source or "runtime file set changed" in source
        assert "Judge invariant core changed" in source
        for field in private_fields:
            assert f"SFORGE_PRIVATE_CONTEXT_{field}" in source

    assert '--runtime-provenance "${run_runtime_provenance_snapshot}"' in evaluation
    assert 'final_runtime_provenance_snapshot="${final_run_root}/runtime-provenance.json"' in runner
    assert "file_set_sha256: $sforge_runtime_file_set_sha256" in runner
    assert "judge_invariant_core:" in runner


def test_formal_runtime_binds_frozen_quality_controls_end_to_end(
    task_root: Path,
) -> None:
    source = (task_root / "scripts/run_fixed_budget.sh").read_text(
        encoding="utf-8"
    )
    assert "--require-precommitted" in source
    assert "--verify-archive-hashes" in source
    assert ".quality_policy.quality_controls_sha256 == $controls" in source
    assert "quality-controls-manifest.json" in source
    assert (
        "SFORGE_PRIVATE_CONTEXT_QUALITY_CONTROLS_SHA256="
        "${QUALITY_CONTROLS_SHA256}"
    ) in source
    assert "quality-control manifest changed between Work and final evaluation" in source
    assert "quality-control manifest changed during final evaluation" in source


def test_checkpoint_mount_and_resume_are_wired_to_exact_final_identity(
    task_root: Path,
) -> None:
    scripts = task_root / "scripts"
    adapter = (scripts / "sforge_h100_entry.py").read_text(encoding="utf-8")
    evaluation = (scripts / "eval_archive.sh").read_text(encoding="utf-8")
    runner = (scripts / "run_fixed_budget.sh").read_text(encoding="utf-8")
    smoke = (scripts / "smoke_submission_plane.sh").read_text(encoding="utf-8")
    assert 'JUDGE_CHECKPOINT_MOUNT = "/logs/verifier/checkpoints"' in adapter
    assert "Judge checkpoint directory must be root-owned mode 700" in adapter
    assert "_judge_checkpoint_volume()" in adapter
    for source in (evaluation, runner, smoke):
        assert "prepare_root_owned_checkpoint_dir" in source
        assert "SFORGE_PRIVATE_CONTEXT_CHECKPOINT_DIR=/logs/verifier/checkpoints" in source
    assert 'resume_mode="${LLADA_RESUME:-0}"' in evaluation
    assert "bind_runtime_resume_identity" in evaluation
    assert 'submission_round="manual-${submission_index}"' in evaluation
    assert '--submission-root "${submission_root}"' in evaluation
    assert 'export LLADA_RUNTIME_RUN_ID="${final_eval_id}"' in runner
    assert '"${final_run_root}" "${final_eval_id}"' in runner
    assert '--arg final_checkpoint_dir "${final_run_root}/judge-checkpoints"' in runner
    assert '"judge" "${judge_gpu_uuid}" "${final_eval_id}"' in runner


def test_formal_final_has_post_run_model_runtime_and_archive_barriers(
    task_root: Path,
) -> None:
    source = (task_root / "scripts/run_fixed_budget.sh").read_text(encoding="utf-8")
    final_eval = source.index('eval \\\n  --backend docker', source.index('final_eval_id='))
    for marker in (
        "post-final-model-validation.json",
        "model artifact changed during final evaluation",
        "SForge Python source set changed during final evaluation",
        "SForge runtime changed during final evaluation",
        "H100 runtime entry changed during final evaluation",
        "frozen final archive changed during final evaluation",
    ):
        assert source.index(marker) > final_eval


def test_submission_smoke_policy_modes_do_not_mix_bootstrap_and_frozen(
    task_root: Path,
) -> None:
    source = (task_root / "scripts/smoke_submission_plane.sh").read_text(
        encoding="utf-8"
    )
    bootstrap_flag = source.index("SFORGE_PRIVATE_CONTEXT_BOOTSTRAP=1")
    frozen_branch = source.index("if (( workload_frozen == 1 ))")
    bootstrap_else = source.index("else", frozen_branch)
    assert bootstrap_flag > bootstrap_else
    assert "pending, frozen, or explicit pilot policies" in source
    assert "pilot-provisional-workload" in source
    assert "SFORGE_PRIVATE_CONTEXT_PILOT=1" in source
    assert "LLADA_BOOTSTRAP_FINAL_JUDGE_TIMEOUT:-}" in source
    assert "LLADA_BOOTSTRAP_FINAL_JUDGE_TIMEOUT:-}" in source
    for name in (
        "LLADA_BOOTSTRAP_CASE_TIMEOUT",
        "LLADA_BOOTSTRAP_ROW_TIMEOUT",
        "LLADA_BOOTSTRAP_OUTER_TIMEOUT",
    ):
        assert f"{name}:-}}" in source


def test_qualification_and_adapter_timeouts_have_no_hidden_release_fallback(
    task_root: Path,
) -> None:
    library = (task_root / "scripts/lib.sh").read_text(encoding="utf-8")
    evaluation = (task_root / "scripts/eval_archive.sh").read_text(encoding="utf-8")
    verifier = (
        task_root / "images/judge/verifier/verify_decoder.py"
    ).read_text(encoding="utf-8")
    for name in (
        "LLADA_QUALIFICATION_CASE_TIMEOUT",
        "LLADA_QUALIFICATION_ROW_TIMEOUT",
        "LLADA_QUALIFICATION_OUTER_TIMEOUT",
    ):
        assert f'{name}:-}}"' in library
        assert f"{name}=${{POLICY_" in evaluation
        assert f'_positive_env("{name}")' in verifier
    assert 'os.environ.get("LLADA_QUALIFICATION_CASE_TIMEOUT", "600")' not in verifier
    assert 'os.environ.get("LLADA_QUALIFICATION_ROW_TIMEOUT", "100000")' not in verifier
    assert 'os.environ.get("LLADA_QUALIFICATION_OUTER_TIMEOUT", "259000")' not in verifier


def test_runner_keeps_both_uuid_locks_through_frozen_final(task_root: Path) -> None:
    runner = (task_root / "scripts/run_fixed_budget.sh").read_text(encoding="utf-8")
    lock = runner.index("acquire_dual_gpu_locks")
    work_run = runner.index('"${H100_ENTRY}" \\\n  run')
    work_barrier = runner.index('assert_gpu_released "${work_gpu_uuid}"')
    lock_barrier = runner.index("assert_dual_gpu_locks_held", work_barrier)
    final_eval = runner.index('"${H100_ENTRY}" \\\n  eval', lock_barrier)
    final_lock_barrier = runner.index("assert_dual_gpu_locks_held", final_eval)
    completion = runner.index('completion="${RUN_LOG_ROOT}/completion.json"')
    assert lock < work_run < work_barrier < lock_barrier < final_eval
    assert final_eval < final_lock_barrier < completion
    assert "flock -u" not in runner
    assert "exec 8>&-" not in runner
    assert "exec 9>&-" not in runner
    assert 'final_work_lock: "held_until_cleanup"' in runner


def test_abnormal_cleanup_is_scoped_by_task_role_and_gpu_uuid(task_root: Path) -> None:
    scripts = task_root / "scripts"
    library = (scripts / "lib.sh").read_text(encoding="utf-8")
    assert "remove_task_gpu_containers()" in library
    assert 'label=mlsysforge.task_id=${TASK_ID}' in library
    assert 'label=mlsysforge.run_id=${run_id}' in library
    assert 'label=mlsysforge.gpu_role=${role}' in library
    assert 'label=mlsysforge.gpu_uuid=${uuid}' in library
    for name in (
        "run_fixed_budget.sh",
        "eval_archive.sh",
        "smoke_submission_plane.sh",
    ):
        source = (scripts / name).read_text(encoding="utf-8")
        assert "remove_task_gpu_containers" in source


def test_final_judge_receives_complete_frozen_identity_context(task_root: Path) -> None:
    required = {
        "SFORGE_PRIVATE_CONTEXT_ARCHIVE_SHA256",
        "SFORGE_PRIVATE_CONTEXT_WORK_IMAGE_ID",
        "SFORGE_PRIVATE_CONTEXT_JUDGE_IMAGE_ID",
        "SFORGE_PRIVATE_CONTEXT_MODEL_VALIDATION_SHA256",
        "SFORGE_PRIVATE_CONTEXT_RUNTIME_ENTRY_SHA256",
        "SFORGE_PRIVATE_CONTEXT_SOURCE_DIGEST",
    }
    scripts = task_root / "scripts"
    for name in ("run_fixed_budget.sh", "eval_archive.sh"):
        source = (scripts / name).read_text(encoding="utf-8")
        for field in required:
            assert field in source, f"{name} is missing {field}"

    eval_archive = (scripts / "eval_archive.sh").read_text(encoding="utf-8")
    context = eval_archive.index("SFORGE_PRIVATE_CONTEXT_ARCHIVE_SHA256")
    evaluate = eval_archive.index("--archive \"${archive}\"")
    model_recheck = eval_archive.index("model validation identity changed during evaluation")
    runtime_recheck = eval_archive.index("H100 runtime entry changed during evaluation")
    assert context < evaluate < model_recheck < runtime_recheck


def test_submission_plane_smoke_is_explicit_bootstrap(task_root: Path) -> None:
    source = (task_root / "scripts/smoke_submission_plane.sh").read_text(
        encoding="utf-8"
    )
    assert "SFORGE_PRIVATE_CONTEXT_BOOTSTRAP=1" in source
    assert "smoke-only liveness ceilings" in source
    assert "if (( workload_frozen == 1 ))" in source
    assert "KNOWN_VALID_ARCHIVE" in source
    assert '.status == "completed" and .report.valid == true' in source


def test_formal_judge_deadlines_come_only_from_frozen_policy(task_root: Path) -> None:
    # The verifier's case/row/outer deadlines come only from the frozen
    # policy; the adapter kill (run_judge.py) is the host-side 6600 s
    # LLADA_JUDGE_KILL_TIMEOUT (owner decision 2026-10-13), kept at least
    # 600 s beyond the frozen outer deadline so the verifier fires first.
    scripts = task_root / "scripts"
    for name in ("run_fixed_budget.sh", "eval_archive.sh", "smoke_submission_plane.sh"):
        source = (scripts / name).read_text(encoding="utf-8")
        assert "load_frozen_workload_policy" in source
        assert 'LLADA_JUDGE_KILL_TIMEOUT="${LLADA_JUDGE_KILL_TIMEOUT:-6600}"' in source
        assert (
            "(( LLADA_JUDGE_KILL_TIMEOUT >= POLICY_FINAL_OUTER_TIMEOUT + 600 ))"
            in source
        )
        assert (
            "LLADA_INTERACTIVE_JUDGE_TIMEOUT=${LLADA_JUDGE_KILL_TIMEOUT}"
            in source
        )
        assert "LLADA_FINAL_JUDGE_TIMEOUT=${LLADA_JUDGE_KILL_TIMEOUT}" in source
    for name in ("run_fixed_budget.sh", "eval_archive.sh"):
        source = (scripts / name).read_text(encoding="utf-8")
        assert "LLADA_FINAL_JUDGE_TIMEOUT=${POLICY_FINAL_INNER_TIMEOUT}" not in source
    runner = (scripts / "run_fixed_budget.sh").read_text(encoding="utf-8")
    assert (
        "(( POLICY_FINAL_OUTER_TIMEOUT < LLADA_JUDGE_KILL_TIMEOUT \\\n"
        "    && LLADA_JUDGE_KILL_TIMEOUT < TASK_EVAL_TIMEOUT ))"
    ) in runner
    archive = (scripts / "eval_archive.sh").read_text(encoding="utf-8")
    assert "(( LLADA_JUDGE_KILL_TIMEOUT < eval_outer_timeout ))" in archive


def test_judge_kill_order_verifier_then_adapter_then_coordinator(task_root: Path) -> None:
    policy = json.loads((task_root / "workload_policy.json").read_text(encoding="utf-8"))
    task = json.loads(
        (task_root / "tasks/llada_h100_decoder_optimization_v2.json").read_text(encoding="utf-8")
    )
    outer = policy["final"]["outer_timeout_seconds"]
    adapter_kill = 6600
    # ~170 s of Judge admission precede the verifier's persistent deadline;
    # the remaining margin covers worker teardown and result serialization.
    assert outer + 600 <= adapter_kill
    assert adapter_kill + 300 <= task["judge"]["eval_timeout"]


def test_all_launch_surfaces_use_distinct_role_gpus(task_root: Path) -> None:
    scripts = task_root / "scripts"
    runtime_files = [
        scripts / "lib.sh",
        scripts / "run_fixed_budget.sh",
        scripts / "eval_archive.sh",
        scripts / "smoke_test.sh",
        scripts / "smoke_submission_plane.sh",
        # Internal per-model tmux launchers, present in a development checkout only.
        *sorted(scripts.glob("launch_*_tmux.sh")),
        scripts / "status.sh",
    ]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in runtime_files)
    assert "PHYSICAL_GPU" not in combined
    assert "sforge_a6000_entry.py" not in combined
    assert "--device /dev/nvidia0" not in combined
    assert "LLADA_WORK_GPU_UUID" in combined
    assert "LLADA_JUDGE_GPU_UUID" in combined
    assert "running_container_claims_gpu" in combined
    assert "two_uuid_flocks_deterministic_order" in combined

    for path in sorted(scripts.glob("launch_*_tmux.sh")):
        source = path.read_text(encoding="utf-8")
        assert "explicit numeric WORK_GPU" in source
        assert "explicit numeric JUDGE_GPU" in source
        assert '"WORK_GPU=${work_gpu}"' in source
        assert '"JUDGE_GPU=${judge_gpu}"' in source


def test_experiment_yamls_use_only_new_task_identity(task_root: Path) -> None:
    for path in sorted(task_root.glob("experiment-*.yaml")):
        text = path.read_text(encoding="utf-8")
        assert f"  {TASK_ID}: {{}}" in text
        assert "llada_a6000_decoder_optimization_v3" not in text
