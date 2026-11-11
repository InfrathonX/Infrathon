#!/usr/bin/env python3
"""SForge entry with strict Work isolation and trusted Judge host mounts."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
from typing import Any
from urllib.parse import urlsplit

from sforge.harness.agent.claude_code import ClaudeCodeAgent
from sforge.harness.agent.codex import CodexAgent
from sforge.harness.backend.docker_backend import DockerBackend, DockerContainerHandle
from sforge.harness import run_agent


MIGRATION_ROOT = Path(__file__).resolve().parents[1]
NETWORK_ENV = "SFORGE_ISOLATED_DOCKER_NETWORK"
NONCE_SOURCE_ENV = "SFORGE_KVBENCH_NONCE_SOURCE"
DEFAULT_NONCE_SOURCE = MIGRATION_ROOT / "judge/private/nonce_source.txt"
# Where assets/fetch.sh places the same file (assets/manifest.json path).
ASSET_NONCE_SOURCE = "judge/nonce_source.txt"
EVAL_ROOT_ENV = "SFORGE_KVBENCH_EVAL_ROOT"
VLLM_BASE_URL_ENV = "KVBENCH_VLLM_BASE_URL"
HIDDEN_BASELINE_ENV = "KVBENCH_HIDDEN_BASELINE_PATH"
PUBLIC_BASELINE_ENV = "KVBENCH_PUBLIC_BASELINE_PATH"
WEB_SEARCH_MODE_ENV = "CODEX_WEB_SEARCH_MODE"
# "true"/"false" from the task JSON's internet field, exported by
# run_fixed_budget.sh (the same contract as the other H100 task entries).
WORK_INTERNET_ENV = "MLSYSFORGE_WORK_INTERNET"
REPLAY_PREFLIGHT_ENV = "SFORGE_KVBENCH_REPLAY_PREFLIGHT"
REPLAY_PREFLIGHT_SHA_ENV = "SFORGE_KVBENCH_REPLAY_PREFLIGHT_SHA256"
REPLAY_PREFLIGHT_HELPER_SHA_ENV = "SFORGE_KVBENCH_REPLAY_PREFLIGHT_HELPER_SHA256"
PREFLIGHT_HELPER = MIGRATION_ROOT / "scripts/preflight.py"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_replay_guarded_container_ids: set[str] = set()

DOCKER_SOCKET = Path("/var/run/docker.sock")
# The staged corpus (scripts/stage_swebench_corpus.py): the 192 digest-locked
# SWE-bench Lite images in the local Docker store plus manifest.json here.
CORPUS_ROOT = Path(
    os.environ.get("KVBENCH_CORPUS_ROOT")
    or Path.home() / ".cache/harbor-kv-scheduler-v2/swebench-lite-heterogeneous-192-gpu45"
).expanduser()
STATE_ROOT = Path.home() / (
    ".cache/"
    "harbor-kv-qwen36-swe192-v2-1-gpu67-validation"
)
DEFAULT_EVAL_ROOT = (
    Path.home() / ".cache/mlsysforge-kvbench-qwen36-swe192-evals"
)
HOST_LOCK_ROOT = Path(f"/run/user/{os.getuid()}/mlsysforge-kvbench-qwen36-swe192-locks")
CONTAINER_NONCE_SOURCE = "/run/kvbench-secrets/nonce-source.jsonl"
CONTAINER_LOCK_ROOT = "/run/kvbench-locks"
CONTAINER_HIDDEN_BASELINE = "/run/kvbench-baselines/hidden.json"
CONTAINER_PUBLIC_BASELINE = "/run/kvbench-baselines/public.json"
# Contract revision v2.5 states the backend requirement as a capability
# predicate instead of an "NVIDIA H100 NVL" name whitelist.  Evidence recorded
# under the retained v1 revision describes the same serving system and stays
# acceptable; see judge/run_judge.py for the full rationale.
BACKEND_SERVING_CONTRACT_SHA256 = (
    "1cf6d2e6ff7af8448d6b7735590621d083d6a5071252bc60b40c9756ffe328c4"
)
LEGACY_BACKEND_SERVING_CONTRACT_SHA256 = (
    "44e4a3408b298a5fdc53d7bb84f0739a22f047da1974212754a98b5755777ba8"
)
PREVIOUS_BACKEND_SERVING_CONTRACT_SHA256 = (
    "18d27d97c736f90e1d74822e80ce1f212685e26627b31a08e47569a1457f52c6"
)
ACCEPTED_BACKEND_SERVING_CONTRACT_SHA256 = frozenset(
    {
        BACKEND_SERVING_CONTRACT_SHA256,
        LEGACY_BACKEND_SERVING_CONTRACT_SHA256,
        PREVIOUS_BACKEND_SERVING_CONTRACT_SHA256,
    }
)
EXPECTED_PUBLIC_NONCE_COMMITMENT = (
    "4902baae73726c1a2f372e35e9e104f6bb3db7dc7593822359548fba60e1b00c"
)
RUN_BUDGET_PATH = Path("/app/RUN_BUDGET.json")
EXPECTED_HIDDEN_RUNTIME_SECONDS = 2400


def _stable_stat_tuple(
    value: os.stat_result,
) -> tuple[int, int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _read_stable_regular(path: Path, *, private: bool = False) -> bytes:
    """Read one non-symlink file and reject replacement or mutation."""

    try:
        linked_before = os.stat(path, follow_symlinks=False)
    except OSError as exc:
        raise RuntimeError("replay attestation input is unavailable") from exc
    if not stat.S_ISREG(linked_before.st_mode) or linked_before.st_uid != os.getuid():
        raise RuntimeError("replay attestation input is unsafe")
    if private and stat.S_IMODE(linked_before.st_mode) & 0o077:
        raise RuntimeError("replay attestation evidence is not private")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise RuntimeError("replay attestation input is unavailable") from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or (before.st_dev, before.st_ino)
            != (linked_before.st_dev, linked_before.st_ino)
        ):
            raise RuntimeError("replay attestation input changed")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        linked_after = os.stat(path, follow_symlinks=False)
        if (
            _stable_stat_tuple(before) != _stable_stat_tuple(after)
            or (after.st_dev, after.st_ino)
            != (linked_after.st_dev, linked_after.st_ino)
            or not stat.S_ISREG(linked_after.st_mode)
        ):
            raise RuntimeError("replay attestation input changed")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _sha256_stable_regular(path: Path, *, private: bool = False) -> str:
    return hashlib.sha256(_read_stable_regular(path, private=private)).hexdigest()


def _replay_attestation_enabled() -> bool:
    return any(
        os.environ.get(name, "").strip()
        for name in (
            REPLAY_PREFLIGHT_ENV,
            REPLAY_PREFLIGHT_SHA_ENV,
            REPLAY_PREFLIGHT_HELPER_SHA_ENV,
        )
    )


def _verify_replay_live_artifacts() -> None:
    """Repeat the complete formal preflight at each Judge boundary."""

    baseline_raw = os.environ.get(REPLAY_PREFLIGHT_ENV, "").strip()
    expected_baseline_sha = os.environ.get(REPLAY_PREFLIGHT_SHA_ENV, "").strip()
    expected_helper_sha = os.environ.get(
        REPLAY_PREFLIGHT_HELPER_SHA_ENV, ""
    ).strip()
    if not baseline_raw and not expected_baseline_sha and not expected_helper_sha:
        return
    if (
        not baseline_raw
        or _SHA256_RE.fullmatch(expected_baseline_sha) is None
        or _SHA256_RE.fullmatch(expected_helper_sha) is None
    ):
        raise RuntimeError("incomplete KV replay attestation contract")
    baseline_path = Path(baseline_raw)
    if not baseline_path.is_absolute():
        raise RuntimeError("KV replay preflight path must be absolute")
    baseline_bytes = _read_stable_regular(baseline_path, private=True)
    if hashlib.sha256(baseline_bytes).hexdigest() != expected_baseline_sha:
        raise RuntimeError("KV replay preflight evidence changed")
    if _sha256_stable_regular(PREFLIGHT_HELPER) != expected_helper_sha:
        raise RuntimeError("KV replay preflight helper changed")
    try:
        baseline = json.loads(baseline_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("KV replay preflight evidence is invalid") from exc
    mlsysforge_root = baseline.get("mlsysforge_root") if isinstance(baseline, dict) else None
    if not isinstance(mlsysforge_root, str) or not Path(mlsysforge_root).is_absolute():
        raise RuntimeError("KV replay preflight evidence has an invalid root")

    with tempfile.TemporaryDirectory(prefix="sforge-kv-replay-attempt-") as temporary:
        output = Path(temporary) / "preflight.json"
        completed = subprocess.run(
            [
                sys.executable,
                str(PREFLIGHT_HELPER),
                "--nonce-source",
                str(_configured_nonce_source()),
                "--mlsysforge-root",
                mlsysforge_root,
                "--output",
                str(output),
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=900,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError("KV replay live preflight failed")
        fresh_bytes = _read_stable_regular(output, private=True)

    if _sha256_stable_regular(PREFLIGHT_HELPER) != expected_helper_sha:
        raise RuntimeError("KV replay preflight helper changed")
    try:
        fresh = json.loads(fresh_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("KV replay live preflight returned invalid evidence") from exc
    if fresh != baseline:
        raise RuntimeError("KV replay live artifacts changed during the batch")
    if _sha256_stable_regular(baseline_path, private=True) != expected_baseline_sha:
        raise RuntimeError("KV replay preflight evidence changed")


def is_judge_container(name: str) -> bool:
    return ".judge.concurrent_agent_serving_optimizer_qwen36_swe192_v24." in name


def _optional_positive_int_env(name: str) -> int | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a positive integer") from exc
    if value <= 0:
        raise RuntimeError(f"{name} must be a positive integer")
    return value


def run_budget_payload() -> dict[str, Any]:
    """Return the non-secret budget contract injected into untrusted Work."""

    return {
        "schema_version": 1,
        "source": "formal-runner-environment",
        "agent_timeout_seconds": _optional_positive_int_env("TASK_TIMEOUT"),
        "submission_cooldown_seconds": _optional_positive_int_env(
            "SUBMISSION_COOLDOWN"
        ),
        "expected_hidden_runtime_seconds": EXPECTED_HIDDEN_RUNTIME_SECONDS,
        "deadline_instruction": (
            "Do not assume a fixed budget; use the values injected for this "
            "run."
        ),
    }


def run_budget_install_command() -> str:
    payload = json.dumps(
        run_budget_payload(),
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return (
        f"printf '%s\\n' {shlex.quote(payload)} > {RUN_BUDGET_PATH} "
        f"&& chmod 0644 {RUN_BUDGET_PATH}"
    )


def _configured_nonce_source() -> Path:
    """The two-line nonce source: in-tree on the Judge-only path by default.

    ``judge/private/nonce_source.txt`` holds the hidden and public workload
    nonces whose commitments are frozen in this task; the root ``.dockerignore``
    keeps it out of every image build context.  Without the in-repository copy
    the default is the copy ``assets/fetch.sh`` places under the asset root
    (``$INFRATHON_ASSETS_ROOT/<task>/judge/nonce_source.txt``).
    ``SFORGE_KVBENCH_NONCE_SOURCE`` still overrides it for a replay against a
    dedicated copy.
    """

    raw_path = os.environ.get(NONCE_SOURCE_ENV, "").strip()
    if not raw_path:
        if DEFAULT_NONCE_SOURCE.exists() or DEFAULT_NONCE_SOURCE.is_symlink():
            return DEFAULT_NONCE_SOURCE
        assets_root = os.environ.get("INFRATHON_ASSETS_ROOT", "").strip()
        if not assets_root:
            cache = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
            assets_root = str(Path(cache) / "infrathon" / "assets")
        return Path(assets_root).expanduser() / MIGRATION_ROOT.name / ASSET_NONCE_SOURCE
    return Path(raw_path).expanduser().resolve(strict=False)


def _configured_eval_root() -> Path:
    return Path(os.environ.get(EVAL_ROOT_ENV, str(DEFAULT_EVAL_ROOT))).resolve(
        strict=False
    )


def _configured_file(environment_name: str) -> Path:
    raw = os.environ.get(environment_name, "").strip()
    if not raw:
        raise RuntimeError(f"{environment_name} is required")
    return Path(raw).expanduser().resolve(strict=False)


def _configured_public_direct_rate() -> float:
    """Project the selected public baseline into a safe Work-visible scalar."""

    path = _configured_file(PUBLIC_BASELINE_ENV)
    if not path.is_file() or path.is_symlink():
        raise RuntimeError("selected public baseline is unavailable or unsafe")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        baseline = payload["legs"]["baseline"]
        provenance = baseline["model_provenance"]
        rate = baseline["valid_steps_per_min"]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise RuntimeError("selected public baseline has an invalid schema") from exc
    if (
        payload.get("status") != "complete"
        or payload.get("suite") != "public"
        or payload.get("profile") != "public-swe192-distinct"
        or payload.get("workload_nonce_sha256") != EXPECTED_PUBLIC_NONCE_COMMITMENT
        or not isinstance(baseline, dict)
        or baseline.get("mode") != "baseline"
        or not isinstance(provenance, dict)
        or provenance.get("backend_serving_contract_sha256")
        not in ACCEPTED_BACKEND_SERVING_CONTRACT_SHA256
        or isinstance(rate, bool)
        or not isinstance(rate, (int, float))
        or not math.isfinite(float(rate))
        or float(rate) <= 0
    ):
        raise RuntimeError("selected public baseline does not match this run contract")
    return float(rate)


def _configured_vllm_base_url() -> str:
    raw = os.environ.get(VLLM_BASE_URL_ENV, "").strip()
    try:
        parsed = urlsplit(raw)
        port = parsed.port
    except ValueError as exc:
        raise RuntimeError(f"{VLLM_BASE_URL_ENV} is invalid") from exc
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise RuntimeError(
            f"{VLLM_BASE_URL_ENV} must be a credential-free HTTP(S) origin"
        )
    if port is not None and not 1 <= port <= 65535:  # pragma: no cover - urlsplit guards
        raise RuntimeError(f"{VLLM_BASE_URL_ENV} has an invalid port")
    return raw.rstrip("/")


def scratch_dir_for(name: str) -> Path:
    digest = hashlib.sha256(name.encode()).hexdigest()[:24]
    return _configured_eval_root() / f"judge-{digest}"


def mount_plan(name: str) -> dict[str, dict[str, str]]:
    """Return host mounts; Work containers intentionally receive none."""
    if not is_judge_container(name):
        return {}
    scratch = scratch_dir_for(name)
    nonce_source = _configured_nonce_source()
    hidden_baseline = _configured_file(HIDDEN_BASELINE_ENV)
    public_baseline = _configured_file(PUBLIC_BASELINE_ENV)
    return {
        str(DOCKER_SOCKET): {"bind": str(DOCKER_SOCKET), "mode": "rw"},
        str(MIGRATION_ROOT): {"bind": str(MIGRATION_ROOT), "mode": "ro"},
        str(CORPUS_ROOT): {"bind": str(CORPUS_ROOT), "mode": "ro"},
        str(STATE_ROOT): {"bind": str(STATE_ROOT), "mode": "rw"},
        str(HOST_LOCK_ROOT): {"bind": CONTAINER_LOCK_ROOT, "mode": "rw"},
        str(hidden_baseline): {
            "bind": CONTAINER_HIDDEN_BASELINE,
            "mode": "ro",
        },
        str(public_baseline): {
            "bind": CONTAINER_PUBLIC_BASELINE,
            "mode": "ro",
        },
        str(nonce_source): {"bind": CONTAINER_NONCE_SOURCE, "mode": "ro"},
        str(scratch): {"bind": str(scratch), "mode": "rw"},
    }


def _require_real(path: Path, *, kind: str) -> None:
    if kind == "file":
        valid = path.is_file() and not path.is_symlink()
    elif kind == "dir":
        valid = path.is_dir() and not path.is_symlink()
    elif kind == "socket":
        valid = path.exists() and not path.is_symlink()
    else:
        raise ValueError(f"unknown path kind: {kind}")
    if not valid:
        raise RuntimeError(f"required trusted {kind} is unavailable: {path}")


def _prepare_judge_scratch(name: str) -> Path:
    eval_root = _configured_eval_root()
    if eval_root.exists():
        _require_real(eval_root, kind="dir")
    else:
        eval_root.mkdir(mode=0o700, parents=True)
    eval_root.chmod(0o700)
    scratch = scratch_dir_for(name)
    if scratch.exists() or scratch.is_symlink():
        raise RuntimeError(f"refusing stale Judge scratch: {scratch}")
    scratch.mkdir(mode=0o700)
    return scratch


def _validate_judge_host_contract() -> None:
    configured_root = Path(
        os.environ.get("SFORGE_KVBENCH_MIGRATION_ROOT", str(MIGRATION_ROOT))
    ).resolve(strict=True)
    if configured_root != MIGRATION_ROOT:
        raise RuntimeError("configured migration root does not match this runtime adapter")
    _configured_vllm_base_url()
    for path, kind in (
        (DOCKER_SOCKET, "socket"),
        (CORPUS_ROOT, "dir"),
        (STATE_ROOT, "dir"),
        (HOST_LOCK_ROOT, "dir"),
        (_configured_file(HIDDEN_BASELINE_ENV), "file"),
        (_configured_file(PUBLIC_BASELINE_ENV), "file"),
        (_configured_nonce_source(), "file"),
    ):
        _require_real(path, kind=kind)


def isolated_create_container(
    self: DockerBackend,
    image: str,
    name: str,
    *,
    command: str = "tail -f /dev/null",
    environment: dict[str, str] | None = None,
    extra_hosts: dict[str, str] | None = None,
    cap_drop: list[str] | None = None,
    cpu_limit: int | None = None,
    mem_limit: str | None = None,
    user: str | None = None,
    annotations: dict[str, str] | None = None,
    mounts: list[tuple[str, str, str]] | None = None,
) -> DockerContainerHandle:
    del annotations
    is_judge = is_judge_container(name)
    replay_guard = is_judge and _replay_attestation_enabled()
    if replay_guard:
        _verify_replay_live_artifacts()
    kwargs: dict[str, Any] = {}
    if cpu_limit is not None:
        kwargs["nano_cpus"] = int(cpu_limit * 1e9)
    if mem_limit is not None:
        kwargs["mem_limit"] = mem_limit
    if cap_drop:
        kwargs["cap_drop"] = cap_drop
    if extra_hosts:
        kwargs["extra_hosts"] = extra_hosts
    if user:
        kwargs["user"] = user
    if mounts:
        # Only SForge's host-provided agent runtime uses this path: a
        # read-only bind mount into the untrusted Work container.  The Judge
        # mount plan stays exclusively task-owned, and nothing may ever be
        # writable from Work.
        if is_judge:
            raise RuntimeError(
                "Judge containers accept only the trusted task mount plan"
            )
        volumes: dict[str, dict[str, str]] = {}
        for host_path, container_path, mode in mounts:
            if mode != "ro":
                raise RuntimeError(
                    "Work containers accept only read-only bind mounts"
                )
            volumes[host_path] = {"bind": container_path, "mode": mode}
        kwargs["volumes"] = volumes

    container_env = dict(environment or {})
    scratch: Path | None = None
    try:
        if is_judge:
            _validate_judge_host_contract()
            scratch = _prepare_judge_scratch(name)
            kwargs["network_mode"] = "none"
            kwargs["volumes"] = mount_plan(name)
            kwargs["labels"] = {"mlsysforge.kvbench.role": "trusted-judge"}
            container_env.update(
                {
                    "KVBENCH_HOST_EVAL_DIR": str(scratch),
                    "KVBENCH_EVAL_ROOT": str(_configured_eval_root()),
                    "KVBENCH_MIGRATION_ROOT": str(MIGRATION_ROOT),
                    # The frozen validation stack runs inside the Judge and
                    # reaches the corpus and its state through these two
                    # roots, bind-mounted at their host paths (mount_plan).
                    "HARBOR_SWEBENCH_CORPUS_SOURCE": str(CORPUS_ROOT),
                    "HARBOR_KVBENCH_STATE_ROOT": str(STATE_ROOT),
                    "KVBENCH_HIDDEN_NONCE_SOURCE": CONTAINER_NONCE_SOURCE,
                    "KVBENCH_MEASUREMENT_LOCK_PATH": (
                        f"{CONTAINER_LOCK_ROOT}/benchmark-measurement.lock"
                    ),
                    "KVBENCH_HIDDEN_BASELINE_PATH": CONTAINER_HIDDEN_BASELINE,
                    "KVBENCH_PUBLIC_BASELINE_PATH": CONTAINER_PUBLIC_BASELINE,
                    "KVBENCH_VLLM_BASE_URL": _configured_vllm_base_url(),
                    "KVBENCH_BACKEND_SERVING_CONTRACT_SHA256": (
                        BACKEND_SERVING_CONTRACT_SHA256
                    ),
                    "NVIDIA_VISIBLE_DEVICES": "void",
                }
            )
        else:
            # Treat Work as an explicit declassification boundary. Even if a
            # future SForge caller accidentally forwards trusted host values,
            # the untrusted agent container must not learn the raw serving
            # origin, baseline locations, nonce source, or verifier paths.
            for private_name in (
                VLLM_BASE_URL_ENV,
                HIDDEN_BASELINE_ENV,
                PUBLIC_BASELINE_ENV,
                NONCE_SOURCE_ENV,
                EVAL_ROOT_ENV,
                "KVBENCH_BACKEND_SERVING_CONTRACT_SHA256",
                "KVBENCH_HIDDEN_NONCE_SOURCE",
                "KVBENCH_MEASUREMENT_LOCK_PATH",
                "KVBENCH_HOST_EVAL_DIR",
                "KVBENCH_EVAL_ROOT",
                "KVBENCH_MIGRATION_ROOT",
                "SFORGE_KVBENCH_MIGRATION_ROOT",
                "HARBOR_SWEBENCH_CORPUS_SOURCE",
                "HARBOR_KVBENCH_STATE_ROOT",
                "KVBENCH_CORPUS_ROOT",
            ):
                container_env.pop(private_name, None)
            container_env["KVBENCH_PUBLIC_DIRECT_RATE"] = format(
                _configured_public_direct_rate(), ".17g"
            )
            network_name = os.environ.get(NETWORK_ENV, "").strip()
            if not network_name:
                raise RuntimeError(f"{NETWORK_ENV} is required for Work containers")
            network = self.client.networks.get(network_name)
            if not bool(network.attrs.get("Internal")):
                raise RuntimeError(f"refusing non-internal Work network: {network_name}")
            kwargs["network"] = network_name
            kwargs["labels"] = {"mlsysforge.kvbench.role": "untrusted-work"}

        container = self.client.containers.create(
            image,
            name=name,
            detach=True,
            command=command,
            environment=container_env,
            **kwargs,
        )
        handle = DockerContainerHandle(container)
        if replay_guard:
            _replay_guarded_container_ids.add(handle.id)
        return handle
    except Exception:
        if scratch is not None and scratch.parent == _configured_eval_root():
            shutil.rmtree(scratch, ignore_errors=True)
        raise


_original_cleanup_container = DockerBackend.cleanup_container


def isolated_cleanup_container(
    self: DockerBackend,
    handle: DockerContainerHandle | None,
    logger: Any | None = None,
) -> None:
    name = handle.name if handle is not None else ""
    container_id = handle.id if handle is not None else ""
    guarded = container_id in _replay_guarded_container_ids
    try:
        _original_cleanup_container(self, handle, logger)
    finally:
        try:
            if is_judge_container(name):
                scratch = scratch_dir_for(name)
                eval_root = _configured_eval_root()
                if (
                    scratch.parent == eval_root
                    and scratch.name.startswith("judge-")
                    and scratch.exists()
                    and not scratch.is_symlink()
                ):
                    shutil.rmtree(scratch)
        finally:
            if guarded:
                _replay_guarded_container_ids.discard(container_id)
                _verify_replay_live_artifacts()


DockerBackend.create_container = isolated_create_container
DockerBackend.cleanup_container = isolated_cleanup_container

# The Work image is agent-free: SForge bind-mounts a marked host runtime
# (SFORGE_AGENT_RUNTIME_DIR) read-only at /opt/agent-runtime and replaces
# install_cmds with an offline mount probe.  These fallback commands run only
# when no runtime is mounted, where the version assertion fails closed before
# any network installation is attempted; nothing here reaches npm or another
# registry.
CodexAgent.install_cmds = [
    "test \"$(node --version)\" = v20.18.0 && "
    "codex --version | grep -F \"${SFORGE_CODEX_CLI_VERSION:-0.148.0}\"",
    run_budget_install_command(),
]
# Claude Code (MLSYSFORGE_AGENT=claude-code) gets the same registry-free,
# fail-closed fallback: its native install steps would reach nodejs.org and
# npm.  The launcher always mounts the pinned runtime, so the offline probe
# below (with the same budget injection) is what actually runs.
ClaudeCodeAgent.install_cmds = [
    "claude --version | grep -F 2.1.287",
    run_budget_install_command(),
]

# The offline probe replaces install_cmds wholesale, which would silently
# drop the per-run budget injection above and leave the agent reading the
# placeholder RUN_BUDGET.json baked into the image.  Extend the probe with
# the same local, registry-free injection.
_original_runtime_probe_cmds = run_agent.runtime_probe_cmds


def configured_runtime_probe_cmds(runtime: Any) -> list[str]:
    return [*_original_runtime_probe_cmds(runtime), run_budget_install_command()]


run_agent.runtime_probe_cmds = configured_runtime_probe_cmds

_original_format_run_cmd = CodexAgent.format_run_cmd


def _configured_web_search_mode() -> str:
    mode = os.environ.get(WEB_SEARCH_MODE_ENV, "disabled").strip()
    if mode not in {"disabled", "live"}:
        raise RuntimeError(
            f"{WEB_SEARCH_MODE_ENV} must be exactly 'disabled' or 'live'"
        )
    return mode


def _task_internet(requested: Any) -> bool:
    """Apply the task's internet flag to the caller's value.

    The launcher always passes --enable-internet (SForge's own offline mode
    needs sudo iptables); the internal network plus the sidecar is the
    enforcement boundary.  ``false`` forces the offline Codex flags and
    prompt; ``true`` or unset passes the caller's value through.
    """

    value = os.environ.get(WORK_INTERNET_ENV, "").strip().lower()
    if value == "false":
        return False
    if value in {"", "true"}:
        return bool(requested)
    raise RuntimeError(f"{WORK_INTERNET_ENV} must be 'true' or 'false'")


def configured_format_run_cmd(
    self: CodexAgent, *args: Any, **kwargs: Any
) -> str:
    mode = _configured_web_search_mode()
    # CodexAgent maps internet to its explicit web_search setting only; the
    # CODEX_WEB_SEARCH_MODE knob can still turn search off on an online task.
    kwargs["internet"] = (
        _task_internet(kwargs.get("internet", True)) and mode == "live"
    )
    command = _original_format_run_cmd(self, *args, **kwargs)
    if kwargs["internet"]:
        if 'web_search="disabled"' in command:
            raise RuntimeError("live web search command contains disabled override")
        marker = "codex exec"
        if marker not in command:
            raise RuntimeError("cannot locate Codex exec command for live search")
        option = shlex.quote('web_search="live"')
        if f"-c {option}" not in command:
            command = command.replace(marker, f"{marker} -c {option}", 1)
    return command


CodexAgent.format_run_cmd = configured_format_run_cmd


def claude_task_internet(requested: Any) -> bool:
    """Claude Code's web tools follow the same two switches as Codex search.

    sforge/harness/agent/claude_local.py keeps WebSearch/WebFetch when this returns
    true and disallows them otherwise: the task JSON's internet flag
    (``MLSYSFORGE_WORK_INTERNET``) and the CODEX_WEB_SEARCH_MODE knob, which
    the launcher validates for every agent (default ``live``).
    """

    return _task_internet(requested) and _configured_web_search_mode() == "live"

_original_generate_prompt = run_agent.generate_evolve_prompt


def configured_generate_prompt(*args: Any, **kwargs: Any) -> str:
    kwargs["internet"] = _task_internet(kwargs.get("internet", True))
    prompt = _original_generate_prompt(*args, **kwargs)
    return prompt.replace(
        "A background process periodically auto-evaluates your code — if the "
        "submitted files are broken, incomplete, or contain syntax errors at "
        "that moment, the auto-evaluation will fail.",
        "Automatic background evaluation is disabled for this long-running "
        "task. Hidden evaluation occurs only when you run sforge-submit and "
        "once on the immutable timeout-boundary archive after your agent run.",
    )


run_agent.generate_evolve_prompt = configured_generate_prompt

# This task's Judge always emits exactly one delimited result.  Disable the
# generic parser's permissive fallback (which could mistake diagnostic JSON
# from a crashed verifier for an authoritative score).
from sforge.harness.log_parsers import structured_json


def strict_parse_structured_json(output: str) -> dict[str, Any]:
    marker_start = ">>>>> Start Structured Result"
    marker_end = ">>>>> End Structured Result"
    if output.count(marker_start) != 1 or output.count(marker_end) != 1:
        raise ValueError("expected exactly one structured result marker pair")
    start = output.index(marker_start) + len(marker_start)
    end = output.index(marker_end)
    if end <= start:
        raise ValueError("structured result markers are out of order")

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate structured result key: {key}")
            result[key] = value
        return result

    parsed = json.loads(
        output[start:end].strip(),
        parse_constant=reject_constant,
        object_pairs_hook=reject_duplicates,
    )
    if not isinstance(parsed, dict):
        raise ValueError("structured result must be a JSON object")
    return parsed


structured_json.parse_structured_json = strict_parse_structured_json

# The run token is intentionally accepted by SForge's history endpoint as a
# query parameter for compatibility with the generated in-container helper.
# Uvicorn's default access logger would persist that credential verbatim on
# the trusted host, so this formal runtime disables request-target logging.
import uvicorn


_original_uvicorn_run = uvicorn.run


def private_uvicorn_run(*args: Any, **kwargs: Any) -> Any:
    kwargs["access_log"] = False
    return _original_uvicorn_run(*args, **kwargs)


uvicorn.run = private_uvicorn_run

from sforge.cli import main


if __name__ == "__main__":
    # The launcher picks the Agent (MLSYSFORGE_AGENT=codex|claude-code); Claude
    # Code bindings: sforge/harness/agent/claude_local.py.  Codex is untouched.
    from sforge.harness.agent.claude_local import install_from_environment

    install_from_environment(claude_task_internet, sys.argv)
    main()
