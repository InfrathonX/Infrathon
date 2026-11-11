"""Abstract base class for agents."""

from __future__ import annotations

import abc
import logging
from pathlib import Path

from sforge.harness.backend import ContainerBackend, ContainerHandle
from sforge.harness.config import SForgeConfig


class Agent(abc.ABC):
    """Base class for coding agents.

    Subclasses declare agent configuration as class-level attributes
    (``name``, ``install_cmds``, ``run_cmd``, …) and override hook
    methods for agent-specific behavior.
    """

    # -- required (no default) --
    name: str
    install_cmds: list[str]
    run_cmd: str  # template: {prompt_file}, {cwd} placeholders
    api_key_env: str  # env var name the agent expects for API key

    # -- optional --
    api_base_env: str | None = None
    default_api_base_url: str | None = None
    model_env: str | None = None
    default_model: str | None = None
    timeout: int = 3600
    stop_hook: str | None = None
    resume_cmd: str | None = None

    def __init__(self, config: SForgeConfig) -> None:
        self._config = config

    # ------------------------------------------------------------------
    # Overridable hooks
    # ------------------------------------------------------------------

    def augment_env(self, env: dict[str, str], model: str | None) -> None:
        """Mutate *env* with agent-specific environment variables.

        Called **after** the generic env vars (proxy, API key, model,
        mirrors, extra_env) have already been applied.
        """

    def format_run_cmd(
        self,
        prompt_path: str,
        *,
        model: str | None = None,
        cwd: str = "",
        internet: bool = True,
        resume: bool = False,
    ) -> str:
        """Build the shell command to run the agent.

        The default implementation expands ``{prompt_file}`` / ``{cwd}``
        placeholders and handles the resume fallback.  Subclasses that need
        extra CLI flags (``--model``, ``--disallowedTools``, …) should call
        ``super()`` first, then append.
        """
        if resume and self.resume_cmd:
            return self.resume_cmd
        return self.run_cmd.format(prompt_file=prompt_path, cwd=cwd)

    def prepare_runtime(
        self,
        backend: ContainerBackend,
        handle: ContainerHandle,
        log_dir: Path,
        logger: logging.Logger,
    ) -> None:
        """Configure agent credentials/runtime state after installation.

        The default implementation is intentionally empty. Agent adapters can
        use this hook for credential files that must not travel through
        container environment variables.
        """

    def validate_submission_archive(self, archive: bytes) -> None:
        """Reject agent-specific sensitive material before host persistence.

        The default implementation accepts the archive. Adapters that install
        credentials into the work container can add a narrow DLP guard here.
        """

    def install_stop_hook(
        self,
        backend: ContainerBackend,
        handle: ContainerHandle,
        log_dir: Path,
        logger: logging.Logger,
    ) -> None:
        """Install the stop hook into the container.

        Called when the stop hook is enabled.  The default implementation
        logs that no hook is configured (appropriate for agents without one).
        """
        logger.info(
            "No stop hook configured for agent=%s", self.name,
        )
