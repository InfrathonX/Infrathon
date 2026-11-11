from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field, fields
from pathlib import Path

from sforge.harness.constants import LOG_DIR, TASKS_DIR
from sforge.harness.cpu_affinity import config_values, validate_policy_values
from sforge.harness.llm_judge.config import LLMJudgeConfig, load_llm_judge_config


def validate_judge_network_mode(value: str | None) -> str | None:
    """Accept the one explicit fail-closed Judge network policy."""

    if value is None:
        return None
    if not isinstance(value, str) or value != "none":
        raise ValueError(
            "SFORGE_JUDGE_NETWORK_MODE must be exactly 'none' when set"
        )
    return value


def _env_bool(name: str, value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no"}:
        return False
    raise ValueError(f"{name} must be one of 1/true/yes/0/false/no")


def validate_single_gpu_handoff_config(config: "SForgeConfig") -> None:
    """Validate every deployment input before enabling resource handoff."""

    if not config.single_gpu_handoff:
        return
    # Constructing the allocation performs canonical NVIDIA UUID checks.
    from sforge.harness.backend import AcceleratorAllocation

    if config.backend != "docker":
        raise ValueError(
            "single GPU handoff currently requires the Docker backend"
        )
    if config.accelerator_device_uuid is None:
        raise ValueError(
            "SFORGE_ACCELERATOR_DEVICE_UUID is required for single GPU handoff"
        )
    allocation = AcceleratorAllocation(
        "nvidia", (config.accelerator_device_uuid,)
    )
    allocation_key = config.accelerator_allocation_key or ""
    if (
        not allocation_key.startswith("allocation-sha256:")
        or len(allocation_key) != 82
        or any(c not in "0123456789abcdef" for c in allocation_key[18:])
    ):
        raise ValueError(
            "SFORGE_ACCELERATOR_ALLOCATION_KEY must be a canonical "
            "scheduler-issued allocation-sha256 digest"
        )
    from sforge.harness.resource_handoff import canonical_allocation_key

    if allocation_key != canonical_allocation_key(allocation):
        raise ValueError(
            "SFORGE_ACCELERATOR_ALLOCATION_KEY does not match the canonical "
            "physical device-set digest"
        )
    if config.accelerator_observer != "nvml":
        raise ValueError(
            "single GPU handoff requires SFORGE_ACCELERATOR_OBSERVER=nvml"
        )
    if (
        isinstance(config.accelerator_idle_grace_seconds, bool)
        or not isinstance(config.accelerator_idle_grace_seconds, (int, float))
        or not math.isfinite(float(config.accelerator_idle_grace_seconds))
        or config.accelerator_idle_grace_seconds < 0
    ):
        raise ValueError(
            "SFORGE_ACCELERATOR_IDLE_GRACE_SECONDS must be a finite "
            "non-negative number"
        )
    trusted_lease_id = config.trusted_lease_id or ""
    if (
        re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}", trusted_lease_id
        )
        is None
    ):
        raise ValueError(
            "SFORGE_TRUSTED_LEASE_ID must be a scheduler-issued label-safe identity"
        )


def validate_cpu_affinity_config(config: "SForgeConfig") -> None:
    """Validate the opt-in CPU pinning policy without touching hardware.

    Host checks (online CPUs, NUMA locality, the GPU's slice) happen when a
    container is pinned; see sforge/harness/cpu_affinity.py.
    """

    validate_policy_values(config_values(config), backend=config.backend)


@dataclass
class SForgeConfig:
    """Configuration for SForge harness. All fields injectable via SFORGE_* env vars."""

    # Network proxy
    http_proxy: str | None = None
    https_proxy: str | None = None
    no_proxy: str | None = None

    # Package mirrors
    pypi_index_url: str | None = None
    maven_mirror_url: str | None = None
    go_proxy: str | None = None
    nodejs_mirror_url: str | None = None
    npm_registry_url: str | None = None
    apt_mirror_url: str | None = None

    # Agent configuration
    agent_api_key: str | None = None
    agent_api_base_url: str | None = None
    agent_model: str | None = None
    agent_timeout: int | None = None
    agent_extra_env: dict[str, str] = field(default_factory=dict)

    # Codex-specific local automation settings. auth.json is copied from the
    # host after the agent runtime is installed; it is never injected through
    # an environment variable or written to SForge logs.
    codex_auth_json_path: Path | None = None
    codex_reasoning_effort: str | None = None
    codex_cli_version: str = "0.144.4"

    # Claude Code cache optimization — suppress the random cch attribution
    # header and dynamic system prompt sections that break prompt caching on
    # third-party API proxies.  Enable via SFORGE_CLAUDE_CACHE_OPT=1.
    claude_cache_opt: bool = False

    # Judge extra environment (for LLM-based graders etc.)
    judge_extra_env: dict[str, str] = field(default_factory=dict)
    # Optional Judge-only runtime network policy. Work networking is separate.
    judge_network_mode: str | None = None

    # Explicit opt-in for one exact NVIDIA allocation shared serially by Work
    # and Judge.  Legacy behavior remains unchanged while this is false.  The
    # allocation key is scheduler authority; SForge never derives it from a
    # device alias.  The first production observer is deliberately only NVML.
    single_gpu_handoff: bool = False
    accelerator_device_uuid: str | None = None
    accelerator_allocation_key: str | None = None
    accelerator_observer: str | None = None
    accelerator_idle_grace_seconds: float = 5.0
    trusted_lease_id: str | None = None

    # Git credentials (for private repos in setup_cmds)
    git_user: str | None = None
    git_token: str | None = None

    # Container resource limits (work vs judge independently)
    work_cpu_limit: int | None = None       # CPUs for work containers (e.g. 4)
    work_mem_limit: str | None = None       # memory for work containers (e.g. "8g")
    judge_cpu_limit: int | None = None      # CPUs for judge containers
    judge_mem_limit: str | None = None      # memory for judge containers
    # "pyromind": local Docker Work/Judge/review containers take the task's
    # deployment profile size (deploy/pyromind/RESOURCES.md) over the limits
    # above; "native" keeps them.  load_config() defaults to "pyromind";
    # configs built in code (PyroMind runner, tests) stay "native".
    resource_profile: str = "native"

    # Opt-in GPU-local CPU pinning of Work/Judge containers; off unless a
    # policy is set.  Policy: off | auto | CPULIST[@NODELIST], the per-role
    # value overriding the global one (sforge/harness/cpu_affinity.py).
    cpuset: str | None = None               # every Work/Judge container
    work_cpuset: str | None = None          # Work override (incl. "off")
    judge_cpuset: str | None = None         # Judge override (incl. "off")
    cpuset_mems_mode: str = "bind"          # bind | none
    cpuset_reserve: str | None = None       # CPULIST kept out of auto slices
    cpuset_undersize: str = "refuse"        # refuse | allow budget > slice

    # Docker build extra hosts (DNS overrides)
    extra_hosts: dict[str, str] | None = None

    # Remote container registry
    registry: str | None = None

    # Container backend: "docker" or "k8s"
    backend: str = "docker"

    # K8s backend settings
    k8s_namespace: str = "default"
    k8s_node_selector: dict[str, str] = field(default_factory=dict)
    k8s_image_registry: str = ""
    k8s_kubeconfig: str | None = None
    # Paths
    log_dir: Path = field(default_factory=lambda: LOG_DIR)
    tasks_dir: Path = field(default_factory=lambda: TASKS_DIR)
    version_store_root: Path | None = None

    # Host-provided agent runtime directory, bind-mounted read-only into work
    # containers so the agent CLI (kind and version) stays independent of the
    # task image.  See sforge/harness/agent_runtime.py.
    agent_runtime_dir: Path | None = None

    # Opt-in LLM integrity review that gates submissions before the GPU
    # Judge (SFORGE_LLM_JUDGE_*; off by default).  See docs/LLM_JUDGE.md.
    llm_judge: LLMJudgeConfig = field(default_factory=LLMJudgeConfig)

    def __post_init__(self) -> None:
        validate_judge_network_mode(self.judge_network_mode)
        validate_cpu_affinity_config(self)
        if self.resource_profile not in ("native", "pyromind"):
            raise ValueError(
                "SFORGE_RESOURCE_PROFILE must be native or pyromind, "
                f"not {self.resource_profile!r}"
            )


def load_config(cli_overrides: dict | None = None) -> SForgeConfig:
    """Load config from: defaults < env vars (SFORGE_*) < CLI flags."""
    config = SForgeConfig(resource_profile="pyromind")

    # Map of config field -> env var names to try (in priority order)
    _env_map = {
        "http_proxy": ["SFORGE_HTTP_PROXY", "HTTP_PROXY", "http_proxy"],
        "https_proxy": ["SFORGE_HTTPS_PROXY", "HTTPS_PROXY", "https_proxy"],
        "no_proxy": ["SFORGE_NO_PROXY", "NO_PROXY", "no_proxy"],
        "pypi_index_url": ["SFORGE_PYPI_INDEX_URL"],
        "maven_mirror_url": ["SFORGE_MAVEN_MIRROR_URL"],
        "go_proxy": ["SFORGE_GO_PROXY"],
        "nodejs_mirror_url": ["SFORGE_NODEJS_MIRROR_URL"],
        "npm_registry_url": ["SFORGE_NPM_REGISTRY_URL"],
        "apt_mirror_url": ["SFORGE_APT_MIRROR_URL"],
        "agent_api_key": ["SFORGE_AGENT_API_KEY"],
        "agent_api_base_url": ["SFORGE_AGENT_API_BASE_URL"],
        "agent_model": ["SFORGE_AGENT_MODEL"],
        "agent_timeout": ["SFORGE_AGENT_TIMEOUT"],
        "judge_network_mode": ["SFORGE_JUDGE_NETWORK_MODE"],
        "accelerator_device_uuid": ["SFORGE_ACCELERATOR_DEVICE_UUID"],
        "accelerator_allocation_key": ["SFORGE_ACCELERATOR_ALLOCATION_KEY"],
        "accelerator_observer": ["SFORGE_ACCELERATOR_OBSERVER"],
        "accelerator_idle_grace_seconds": [
            "SFORGE_ACCELERATOR_IDLE_GRACE_SECONDS"
        ],
        "trusted_lease_id": ["SFORGE_TRUSTED_LEASE_ID"],
        "codex_auth_json_path": ["SFORGE_CODEX_AUTH_JSON_PATH"],
        "codex_reasoning_effort": ["SFORGE_CODEX_REASONING_EFFORT"],
        "codex_cli_version": ["SFORGE_CODEX_CLI_VERSION"],
        "git_user": ["SFORGE_GIT_USER"],
        "git_token": ["SFORGE_GIT_TOKEN"],
        "registry": ["SFORGE_REGISTRY"],
        "backend": ["SFORGE_BACKEND"],
        "k8s_namespace": ["SFORGE_K8S_NAMESPACE"],
        "k8s_image_registry": ["SFORGE_K8S_IMAGE_REGISTRY"],
        "k8s_kubeconfig": ["SFORGE_K8S_KUBECONFIG"],
        "work_cpu_limit": ["SFORGE_WORK_CPU_LIMIT"],
        "work_mem_limit": ["SFORGE_WORK_MEM_LIMIT"],
        "cpuset": ["SFORGE_CPUSET"],
        "work_cpuset": ["SFORGE_WORK_CPUSET"],
        "judge_cpuset": ["SFORGE_JUDGE_CPUSET"],
        "cpuset_mems_mode": ["SFORGE_CPUSET_MEMS"],
        "cpuset_reserve": ["SFORGE_CPUSET_RESERVE"],
        "cpuset_undersize": ["SFORGE_CPUSET_UNDERSIZE"],
        "resource_profile": ["SFORGE_RESOURCE_PROFILE"],
        "log_dir": ["SFORGE_LOG_DIR"],
        "tasks_dir": ["SFORGE_TASKS_DIR"],
        "version_store_root": ["SFORGE_VERSION_STORE_ROOT"],
        "agent_runtime_dir": ["SFORGE_AGENT_RUNTIME_DIR"],
    }

    for field_name, env_keys in _env_map.items():
        for env_key in env_keys:
            val = os.environ.get(env_key)
            if val is not None:
                f = next(f for f in fields(config) if f.name == field_name)
                if f.type in ("Path", "Path | None"):
                    setattr(config, field_name, Path(val))
                elif f.type in ("int", "int | None"):
                    setattr(config, field_name, int(val))
                elif f.type in ("float", "float | None"):
                    setattr(config, field_name, float(val))
                else:
                    setattr(config, field_name, val)
                break

    if config.version_store_root is None:
        from sforge.harness.version_store import discover_sibling_version_store

        repository_root = Path(__file__).resolve().parents[2]
        config.version_store_root = discover_sibling_version_store(repository_root)

    # SFORGE_CLAUDE_CACHE_OPT: truthy value enables cache optimization
    if os.environ.get("SFORGE_CLAUDE_CACHE_OPT", "").strip() not in ("", "0", "false", "no"):
        config.claude_cache_opt = True

    handoff_value = os.environ.get("SFORGE_SINGLE_GPU_HANDOFF")
    if handoff_value is not None:
        config.single_gpu_handoff = _env_bool(
            "SFORGE_SINGLE_GPU_HANDOFF", handoff_value
        )

    # Parse SFORGE_AGENT_EXTRA_ENV: "KEY1=VAL1,KEY2=VAL2"
    extra_env_str = os.environ.get("SFORGE_AGENT_EXTRA_ENV", "")
    if extra_env_str:
        for item in extra_env_str.split(","):
            item = item.strip()
            if "=" in item:
                k, v = item.split("=", 1)
                config.agent_extra_env[k.strip()] = v.strip()

    # Parse SFORGE_JUDGE_EXTRA_ENV: "KEY1=VAL1,KEY2=VAL2"
    judge_env_str = os.environ.get("SFORGE_JUDGE_EXTRA_ENV", "")
    if judge_env_str:
        for item in judge_env_str.split(","):
            item = item.strip()
            if "=" in item:
                k, v = item.split("=", 1)
                config.judge_extra_env[k.strip()] = v.strip()

    # Parse SFORGE_EXTRA_HOSTS: "host1:ip1,host2:ip2"
    extra_hosts_str = os.environ.get("SFORGE_EXTRA_HOSTS", "")
    if extra_hosts_str:
        config.extra_hosts = {}
        for item in extra_hosts_str.split(","):
            item = item.strip()
            if ":" in item:
                host, ip = item.rsplit(":", 1)
                config.extra_hosts[host.strip()] = ip.strip()

    # Parse SFORGE_K8S_NODE_SELECTOR: "key1=val1,key2=val2"
    k8s_ns_str = os.environ.get("SFORGE_K8S_NODE_SELECTOR", "")
    if k8s_ns_str:
        for item in k8s_ns_str.split(","):
            item = item.strip()
            if "=" in item:
                k, v = item.split("=", 1)
                config.k8s_node_selector[k.strip()] = v.strip()

    # Override from CLI flags
    if cli_overrides:
        for k, v in cli_overrides.items():
            if v is not None and hasattr(config, k):
                f = next(f for f in fields(config) if f.name == k)
                if f.type in ("Path", "Path | None") and not isinstance(v, Path):
                    v = Path(v)
                setattr(config, k, v)

    config.llm_judge = load_llm_judge_config(
        fallbacks={
            "agent_runtime_dir": config.agent_runtime_dir,
            "codex_auth_json": config.codex_auth_json_path,
        }
    )

    validate_judge_network_mode(config.judge_network_mode)
    validate_single_gpu_handoff_config(config)
    validate_cpu_affinity_config(config)
    return config


def get_env_directives(config: SForgeConfig) -> str:
    """Generate Dockerfile directives for build-time proxy/mirror injection.

    Proxy vars (HTTP_PROXY, etc.) use only ARG so they are available during
    build but NOT baked into the final image.  Docker automatically exposes
    the predefined proxy ARGs to RUN commands.

    Package-mirror vars (PIP_INDEX_URL, etc.) are not Docker-predefined, so
    they need an explicit ARG+ENV pair to be visible in RUN commands.  The
    ENV is intentionally set to the *ARG reference* so that if the build-arg
    is empty the ENV is simply unset.
    """
    lines: list[str] = []

    # Proxy: Docker predefined build args — just declare ARG, Docker handles the rest.
    # These will NOT persist in the final image.
    if config.http_proxy:
        lines.append("ARG http_proxy")
        lines.append("ARG HTTP_PROXY")
    if config.https_proxy:
        lines.append("ARG https_proxy")
        lines.append("ARG HTTPS_PROXY")
    if config.no_proxy:
        lines.append("ARG no_proxy")
        lines.append("ARG NO_PROXY")

    # Package mirrors: ARG only — available during build but NOT persisted in image.
    # Docker ARGs are visible to subsequent RUN commands in the same build stage.
    if config.pypi_index_url:
        lines.append(f"ARG PIP_INDEX_URL={config.pypi_index_url}")
    if config.maven_mirror_url:
        lines.append(f"ARG MAVEN_MIRROR_URL={config.maven_mirror_url}")
    if config.go_proxy:
        lines.append(f"ARG GOPROXY={config.go_proxy}")

    return "\n".join(lines)


def get_build_args(config: SForgeConfig) -> dict[str, str]:
    """Generate Docker build args for proxy/mirror passthrough during image builds."""
    args: dict[str, str] = {}
    if config.http_proxy:
        args["HTTP_PROXY"] = config.http_proxy
        args["http_proxy"] = config.http_proxy
    if config.https_proxy:
        args["HTTPS_PROXY"] = config.https_proxy
        args["https_proxy"] = config.https_proxy
    if config.no_proxy:
        args["NO_PROXY"] = config.no_proxy
        args["no_proxy"] = config.no_proxy
    if config.pypi_index_url:
        args["PIP_INDEX_URL"] = config.pypi_index_url
    if config.maven_mirror_url:
        args["MAVEN_MIRROR_URL"] = config.maven_mirror_url
    if config.go_proxy:
        args["GOPROXY"] = config.go_proxy
    return args


def get_build_secrets(config: SForgeConfig) -> dict[str, str]:
    """Return {secret_id: value} for BuildKit --mount=type=secret."""
    secrets: dict[str, str] = {}
    if config.git_user:
        secrets["git_user"] = config.git_user
    if config.git_token:
        secrets["git_token"] = config.git_token
    return secrets


def get_container_env(config: SForgeConfig, include_judge_extra: bool = False) -> dict[str, str]:
    """Generate environment variables dict for container runtime (docker create --env)."""
    env: dict[str, str] = {}
    if config.http_proxy:
        env["http_proxy"] = config.http_proxy
        env["HTTP_PROXY"] = config.http_proxy
    if config.https_proxy:
        env["https_proxy"] = config.https_proxy
        env["HTTPS_PROXY"] = config.https_proxy
    if config.no_proxy:
        env["no_proxy"] = config.no_proxy
        env["NO_PROXY"] = config.no_proxy
    if config.agent_api_key:
        env["ANTHROPIC_API_KEY"] = config.agent_api_key
    if config.agent_api_base_url:
        env["ANTHROPIC_BASE_URL"] = config.agent_api_base_url
    if include_judge_extra and config.judge_extra_env:
        env.update(config.judge_extra_env)
    return env


def get_container_resource_kwargs(
    config_cpu_limit: int | None,
    config_mem_limit: str | None,
    task_cpu_limit: int | None = None,
    task_mem_limit: str | None = None,
) -> dict:
    """Return Docker container create kwargs for CPU/memory limits.

    Priority: config (CLI/env/experiment) > task.json per-spec defaults.
    """
    kwargs: dict = {}
    cpu = config_cpu_limit if config_cpu_limit is not None else task_cpu_limit
    mem = config_mem_limit if config_mem_limit is not None else task_mem_limit
    if cpu is not None:
        kwargs["nano_cpus"] = int(cpu * 1e9)
    if mem is not None:
        kwargs["mem_limit"] = mem
    return kwargs


def create_backend_from_config(config: SForgeConfig, docker_client=None):
    """Create a ContainerBackend from config."""
    from sforge.harness.backend.factory import create_backend

    return create_backend(
        config.backend,
        docker_client=docker_client,
        k8s_namespace=config.k8s_namespace,
        k8s_node_selector=config.k8s_node_selector,
        k8s_image_registry=config.k8s_image_registry,
        k8s_kubeconfig=config.k8s_kubeconfig,
    )
