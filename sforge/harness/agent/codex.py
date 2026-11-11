"""Codex agent — internet blocking and stop hook."""

from __future__ import annotations

import io
import json
import logging
import shlex
import tarfile
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

from sforge.harness.agent.base import Agent
from sforge.harness.backend import ContainerBackend, ContainerHandle
from sforge.harness.container_files import copy_container_file
from sforge.harness.agent.codex_tools import CODEX_TOOL_CATALOG_PATH, load_direct_tool_catalog


class CodexAgent(Agent):

    name = "codex"
    install_cmds = [
        "sudo -E bash -c 'NODE_MIRROR=${SFORGE_NODEJS_MIRROR_URL:-https://nodejs.org/dist} && curl -fsSL $NODE_MIRROR/v20.18.0/node-v20.18.0-linux-x64.tar.xz | tar -xJ -C /usr/local --strip-components=1'",
        "sudo -E npm install -g @openai/codex@${SFORGE_CODEX_CLI_VERSION:-0.144.4}",
        '''if [ -n "$OPENAI_BASE_URL" ]; then
    mkdir -p ~/.codex
    cat > ~/.codex/config.toml << EOF
model_provider = "sforge-proxy"
model_verbosity = "medium"
model = "${CODEX_MODEL:-gpt-5.5}"

[model_providers.sforge-proxy]
name = "sforge-proxy"
base_url = "${OPENAI_BASE_URL}"
env_key = "OPENAI_API_KEY"
EOF
fi''',
    ]
    run_cmd = 'codex exec --dangerously-bypass-approvals-and-sandbox "$(cat {prompt_file})"'
    resume_cmd = 'codex exec resume --last --dangerously-bypass-approvals-and-sandbox "Continue working."'
    api_key_env = "OPENAI_API_KEY"
    api_base_env = "OPENAI_BASE_URL"
    default_api_base_url = "https://api.openai.com"
    model_env = "CODEX_MODEL"
    stop_hook = "codex"

    def augment_env(self, env: dict[str, str], model: str | None) -> None:
        env["SFORGE_CODEX_CLI_VERSION"] = self._config.codex_cli_version
        # File-based ChatGPT credentials use Codex's native ChatGPT transport.
        # In that mode SFORGE_AGENT_API_BASE_URL names a narrowly scoped local
        # reverse proxy, not an OpenAI-compatible API provider.  Keeping
        # OPENAI_BASE_URL here would make the install hook write a custom
        # provider that requires OPENAI_API_KEY and breaks auth.json sessions.
        if self._config.codex_auth_json_path is not None:
            env.pop("OPENAI_BASE_URL", None)
        if not self._config.agent_api_base_url:
            if "OPENAI_API_KEY" in env and "CODEX_API_KEY" not in env:
                env["CODEX_API_KEY"] = env["OPENAI_API_KEY"]

    def prepare_runtime(
        self,
        backend: ContainerBackend,
        handle: ContainerHandle,
        log_dir: Path,
        logger: logging.Logger,
    ) -> None:
        auth_path = self._config.codex_auth_json_path
        if auth_path is None:
            if self._config.agent_api_base_url:
                self._install_provider_config(backend, handle, log_dir, logger)
            return

        auth_path = auth_path.expanduser().resolve()
        if not auth_path.is_file():
            raise RuntimeError(f"Codex auth JSON not found: {auth_path}")

        mode = auth_path.stat().st_mode & 0o777
        if mode & 0o077:
            raise RuntimeError(
                f"Codex auth JSON must not be group/world accessible "
                f"(expected mode 0600, got {mode:04o})"
            )

        try:
            auth_bytes = auth_path.read_bytes()
            auth_data = json.loads(auth_bytes)
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("Codex auth JSON is not readable valid JSON") from exc
        if not isinstance(auth_data, dict) or not (
            auth_data.get("tokens") or auth_data.get("OPENAI_API_KEY")
        ):
            raise RuntimeError("Codex auth JSON does not contain usable credentials")
        sensitive_blobs: set[bytes] = {auth_bytes}

        def collect_secret_values(value, key: str = "") -> None:
            if isinstance(value, dict):
                for child_key, child_value in value.items():
                    collect_secret_values(child_value, str(child_key))
            elif isinstance(value, list):
                for child in value:
                    collect_secret_values(child, key)
            elif isinstance(value, str):
                normalized_key = key.lower().replace("-", "_")
                if (
                    len(value) >= 16
                    and any(
                        marker in normalized_key
                        for marker in ("token", "secret", "api_key", "password")
                    )
                ):
                    sensitive_blobs.add(value.encode())

        collect_secret_values(auth_data)
        self._submission_forbidden_blobs = tuple(sensitive_blobs)

        destination = PurePosixPath("/home/agent/.codex/auth.json")
        result = backend.exec_run(
            handle,
            [
                "/bin/bash",
                "-lc",
                "mkdir -p /home/agent/.codex && "
                "chown agent:agent /home/agent/.codex",
            ],
            user="root",
        )
        if result.exit_code != 0:
            raise RuntimeError(f"Failed to create Codex home: {result.output}")
        backend.copy_to_container(handle, auth_path, destination)
        result = backend.exec_run(
            handle,
            [
                "/bin/bash",
                "-lc",
                f"chown agent:agent {destination} && chmod 600 {destination}",
            ],
            user="root",
        )
        if result.exit_code != 0:
            raise RuntimeError(f"Failed to secure Codex auth JSON: {result.output}")
        logger.info("Installed file-based Codex credentials (mode 0600)")

    def _install_provider_config(
        self,
        backend: ContainerBackend,
        handle: ContainerHandle,
        log_dir: Path,
        logger: logging.Logger,
    ) -> None:
        """Point Codex at an OpenAI-compatible endpoint in API-key mode.

        The network install hook writes this provider block, but a mounted
        agent runtime replaces that hook with an offline probe, and Codex
        0.148 does not read OPENAI_BASE_URL from the environment on its own
        (it keeps talking to api.openai.com).  So write the provider config
        here, before any prompt runs.  Only the Responses wire API exists in
        this Codex generation; the endpoint must serve /v1/responses.
        """
        base_url = self._config.agent_api_base_url.rstrip("/")
        lines = ['model_provider = "sforge-api"']
        model = self._config.agent_model or self.default_model
        if model:
            lines.append(f"model = {json.dumps(model)}")
        lines += [
            "",
            "[model_providers.sforge-api]",
            'name = "sforge-api"',
            f"base_url = {json.dumps(base_url)}",
            f'env_key = "{self.api_key_env}"',
            'wire_api = "responses"',
            "",
        ]
        local_config = log_dir / "_codex_provider_config.toml"
        local_config.write_text("\n".join(lines))
        result = backend.exec_run(
            handle,
            [
                "/bin/bash",
                "-lc",
                "mkdir -p /home/agent/.codex && "
                "chown agent:agent /home/agent/.codex",
            ],
            user="root",
        )
        if result.exit_code != 0:
            raise RuntimeError(f"Failed to create Codex home: {result.output}")
        copy_container_file(
            backend,
            handle,
            local_config,
            PurePosixPath("/home/agent/.codex/config.toml"),
            owner="agent:agent",
            mode=0o600,
            description="Codex model provider config",
        )
        logger.info("Installed Codex API provider config -> %s", base_url)

    def validate_submission_archive(self, archive: bytes) -> None:
        forbidden = getattr(self, "_submission_forbidden_blobs", ())
        if not forbidden:
            return
        try:
            with tarfile.open(fileobj=io.BytesIO(archive), mode="r:*") as bundle:
                for member in bundle.getmembers():
                    if not member.isfile():
                        continue
                    extracted = bundle.extractfile(member)
                    if extracted is None:
                        continue
                    content = extracted.read()
                    if any(marker and marker in content for marker in forbidden):
                        raise RuntimeError(
                            "submission archive contains Codex credential material"
                        )
        except tarfile.TarError as exc:
            raise RuntimeError("submission archive is not a readable tar file") from exc

    def format_run_cmd(
        self,
        prompt_path: str,
        *,
        model: str | None = None,
        cwd: str = "",
        internet: bool = True,
        resume: bool = False,
    ) -> str:
        cmd = super().format_run_cmd(
            prompt_path, model=model, cwd=cwd, internet=internet, resume=resume,
        )

        exec_options: list[str] = []
        effective_model = model or self._config.agent_model or self.default_model
        if effective_model:
            exec_options.extend(["-m", shlex.quote(effective_model)])

        effort = self._config.codex_reasoning_effort
        if effort:
            allowed_efforts = {
                "none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra",
            }
            if effort not in allowed_efforts:
                raise ValueError(
                    f"Unsupported Codex reasoning effort {effort!r}; "
                    f"choose one of {sorted(allowed_efforts)}"
                )
            override = f"model_reasoning_effort={json.dumps(effort)}"
            exec_options.extend(["-c", shlex.quote(override)])

        # Every task allows internet since 2026-10-07 (docs/INTERNET_AND_WEB_SEARCH.md).
        # State the mode explicitly so a Work image's config.toml cannot override it.
        web_search = "live" if internet else "disabled"
        exec_options.extend(["-c", shlex.quote(f'web_search="{web_search}"')])
        # One Agent per run, as on PyroMind: no sub-agent tools (multi_agent_v1)
        # and no code-mode tool surface, for every model and on resume too.
        for feature in ("multi_agent", "code_mode", "code_mode_only"):
            exec_options.extend(["--disable", feature])
        if getattr(self, "_model_catalog_installed", False):
            # Direct shell/patch tools instead of the catalog's code-mode tool
            # surface (some models ship as code_mode_only), as on PyroMind.
            catalog = json.dumps(CODEX_TOOL_CATALOG_PATH)
            exec_options.extend(["-c", shlex.quote(f"model_catalog_json={catalog}")])

        chatgpt_base_override = self._config.agent_extra_env.get(
            "SFORGE_CODEX_CHATGPT_BASE_URL"
        )
        if (
            self._config.codex_auth_json_path is not None
            and self._config.agent_api_base_url
        ):
            chatgpt_base = json.dumps(
                (
                    chatgpt_base_override
                    or self._config.agent_api_base_url
                ).rstrip("/")
            )
            exec_options.extend(
                ["-c", shlex.quote(f"chatgpt_base_url={chatgpt_base}")]
            )

        if exec_options:
            cmd = cmd.replace("codex exec", f"codex exec {' '.join(exec_options)}", 1)

        if chatgpt_base_override and self._config.agent_api_base_url:
            # The generic HTTP(S) proxy in the container is intentionally only
            # usable before iptables is applied (runtime installation).  Run
            # Codex through the separately allowlisted CONNECT proxy, whose
            # destination policy is controlled by the host.
            runtime_proxy = shlex.quote(self._config.agent_api_base_url)
            prefix = (
                "env -u ALL_PROXY -u all_proxy "
                f"HTTP_PROXY={runtime_proxy} http_proxy={runtime_proxy} "
                f"HTTPS_PROXY={runtime_proxy} https_proxy={runtime_proxy} "
            )
            cmd = cmd.replace("codex exec", f"{prefix}codex exec", 1)

        return cmd

    def install_stop_hook(
        self,
        backend: ContainerBackend,
        handle: ContainerHandle,
        log_dir: Path,
        logger: logging.Logger,
    ) -> None:
        hook_path = "/tmp/sforge-codex-stop-hook.sh"

        hook_script = _generate_codex_stop_hook()
        local_hook = log_dir / "_codex-stop-hook.sh"
        local_hook.write_text(hook_script)
        copy_container_file(
            backend,
            handle,
            local_hook,
            PurePosixPath(hook_path),
            owner="root:root",
            mode=0o755,
            description="Codex stop hook",
        )
        logger.info("Installed Codex stop hook")

        hooks_content = _generate_codex_hooks(hook_path)
        local_hooks = log_dir / "_codex_hooks.json"
        local_hooks.write_text(hooks_content)
        result = backend.exec_run_with_exit_code(
            handle, ["mkdir", "-p", "/etc/codex"], timeout=30, user="root",
        )
        if result.timed_out or result.exit_code != 0:
            raise RuntimeError(f"Failed to create /etc/codex: {result.output}")
        copy_container_file(
            backend,
            handle,
            local_hooks,
            PurePosixPath("/etc/codex/hooks.json"),
            owner="root:root",
            mode=0o644,
            description="Codex hooks JSON",
        )

        config_append = _generate_codex_config_append()
        local_config_append = log_dir / "_codex_config_append.toml"
        local_config_append.write_text(config_append)
        copy_container_file(
            backend,
            handle,
            local_config_append,
            PurePosixPath("/tmp/sforge-codex-config-append.toml"),
            owner="root:root",
            mode=0o644,
            description="Codex config append",
        )

        enable_hooks_cmd = r"""
set -e
CONFIG=/home/agent/.codex/config.toml
TMP=$(mktemp)
mkdir -p /home/agent/.codex
touch "$CONFIG"
if grep -Eq '^[[:space:]]*hooks[[:space:]]*=' "$CONFIG"; then
    sed -E 's/^[[:space:]]*hooks[[:space:]]*=.*/hooks = true/' "$CONFIG" > "$TMP"
elif grep -Eq '^[[:space:]]*\[features\][[:space:]]*$' "$CONFIG"; then
    awk '
        /^[[:space:]]*\[features\][[:space:]]*$/ {
            print
            print "hooks = true"
            next
        }
        { print }
    ' "$CONFIG" > "$TMP"
else
    cp "$CONFIG" "$TMP"
    cat /tmp/sforge-codex-config-append.toml >> "$TMP"
fi
mv "$TMP" "$CONFIG"
chown -R agent:agent /home/agent/.codex
"""
        result = backend.exec_run(
            handle, ["/bin/bash", "-lc", enable_hooks_cmd], user="root",
        )
        if result.exit_code != 0:
            raise RuntimeError(f"Failed to configure Codex hooks: {result.output}")
        logger.info("Configured Codex hooks")

    def install_model_catalog(
        self,
        backend: ContainerBackend,
        handle: ContainerHandle,
        log_dir: Path,
        logger: logging.Logger,
        model: str | None,
    ) -> None:
        """Write the pinned runtime's model catalog with direct tools for this model.

        Same catalog as PyroMind (sforge.harness.agent.codex_tools): only the run's
        model changes its tool mode; models absent from the bundled catalog,
        such as provider models behind a relay, keep Codex's fallback metadata.
        """
        runtime_dir = self._config.agent_runtime_dir
        if runtime_dir is None or not model:
            return
        catalog = load_direct_tool_catalog(SimpleNamespace(root=str(runtime_dir)), model)
        local_catalog = log_dir / "_codex_model_catalog.json"
        local_catalog.write_text(catalog)
        copy_container_file(
            backend,
            handle,
            local_catalog,
            PurePosixPath(CODEX_TOOL_CATALOG_PATH),
            owner="agent:agent",
            mode=0o644,
            description="Codex model catalog",
        )
        self._model_catalog_installed = True
        logger.info("Installed Codex direct-tool model catalog for %s", model)


# ---------------------------------------------------------------------------
# Script generators (private to this module)
# ---------------------------------------------------------------------------


def _generate_codex_stop_hook() -> str:
    return r"""#!/bin/bash
cat >/dev/null
echo '{"decision":"block","reason":"Do not stop. Continue working on the implementation."}'
"""


def _generate_codex_hooks(hook_path: str) -> str:
    hooks: dict = {
        "hooks": {
            "Stop": [
                {
                    "hooks": [
                        {
                            "type": "command",
                            "command": hook_path,
                            "timeout": 30,
                            "statusMessage": "SForge stop hook",
                        }
                    ]
                }
            ],
        }
    }
    return json.dumps(hooks, indent=2)


def _generate_codex_config_append() -> str:
    return """\n[features]\nhooks = true\nmulti_agent = false\ncode_mode = false\ncode_mode_only = false\n"""
