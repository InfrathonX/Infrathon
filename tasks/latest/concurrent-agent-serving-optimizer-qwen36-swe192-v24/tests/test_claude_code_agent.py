"""Claude Code (MLSYSFORGE_AGENT=claude-code) beside the unchanged Codex path.

The formal launcher is run for real up to the step that needs built images, with
fake ``docker``/``curl``/``sg`` executables: nothing reaches Docker or the network.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "run_fixed_budget.sh"
CLAUDE_VERSION = "2.1.287"


def _executable(path: Path, body: str) -> None:
    path.write_text("#!/usr/bin/env bash\n" + body, encoding="utf-8")
    path.chmod(0o755)


def _runtime(root: Path, agent: str, version: str) -> Path:
    root.mkdir()
    marker = {
        "kind": "sforge-agent-runtime",
        "agent": agent,
        "version": version,
        "cli": "claude" if agent == "claude-code" else "codex",
        "tree_sha256": "0" * 64,
        "schema_version": 1,
    }
    (root / ".sforge-agent-runtime.json").write_text(json.dumps(marker), encoding="utf-8")
    return root


def _private(path: Path, text: str = "secret-value\n", mode: int = 0o600) -> Path:
    path.write_text(text, encoding="utf-8")
    path.chmod(mode)
    return path


@pytest.fixture()
def launch(tmp_path: Path):
    # The launcher insists on <repository>/.venv/bin/sforge (scripts/lib.sh).
    repository = next(
        (p for p in ROOT.parents if (p / "pyproject.toml").is_file() and (p / "sforge").is_dir()),
        None,
    )
    if repository is None or not os.access(repository / ".venv/bin/sforge", os.X_OK):
        pytest.skip("needs the repository virtualenv (uv sync): .venv/bin/sforge")
    # The launcher reads the Judge-only nonce source: in-tree, else under the
    # asset root (scripts/lib.sh default_nonce_source).
    assets_root = Path(
        os.environ.get("INFRATHON_ASSETS_ROOT")
        or Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
        / "infrathon" / "assets"
    ).expanduser()
    if not (ROOT / "judge/private/nonce_source.txt").is_file() and not (
        assets_root / ROOT.name / "judge/nonce_source.txt"
    ).is_file():
        pytest.skip("Judge-only data: run assets/fetch.sh")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    events = tmp_path / "events.log"
    _executable(
        bin_dir / "docker",
        'printf "docker %s\\n" "$*" >> "$EVENT_LOG"\n'
        '[[ "$1" == info ]] && exit "${DOCKER_INFO_RC:-0}"\nexit 1\n',
    )
    _executable(
        bin_dir / "curl",
        'printf "curl %s\\n" "$*" >> "$EVENT_LOG"\n'
        'printf "%s" "${CURL_CODE:-000}"\nexit "${CURL_RC:-7}"\n',
    )
    _executable(bin_dir / "sg", 'printf "sg %s\\n" "$*" >> "$EVENT_LOG"\nexit 97\n')
    codex_runtime = _runtime(tmp_path / "codex-runtime", "codex", "0.159.3")
    claude_runtime = _runtime(tmp_path / "claude-runtime", "claude-code", CLAUDE_VERSION)
    codex_auth = _private(tmp_path / "auth.json", "{}\n")
    claude_token = _private(tmp_path / "claude-token")
    launches = iter(range(1000))

    def run(agent: str | None, model: str, effort: str = "max", **overrides: str):
        env = {
            key: os.environ[key]
            for key in ("HOME", "USER", "LANG", "LC_ALL", "TMPDIR", "INFRATHON_ASSETS_ROOT", "XDG_CACHE_HOME")
            if key in os.environ
        }
        env.update(
            PATH=f"{bin_dir}:{os.environ['PATH']}",
            EVENT_LOG=str(events),
            LOGS_DIR=str(tmp_path / "logs"),
            RUN_STAMP=f"20261016-{next(launches):06d}",
            MODEL=model,
            REASONING_EFFORT=effort,
            KVBENCH_VLLM_BASE_URL="http://backend.invalid:18000",
            KVBENCH_HIDDEN_BASELINE_PATH=str(tmp_path / "hidden.json"),
            KVBENCH_PUBLIC_BASELINE_PATH=str(tmp_path / "public.json"),
        )
        if agent is not None:
            env["MLSYSFORGE_AGENT"] = agent
        if agent == "claude-code":
            env.update(CLAUDE_OAUTH_TOKEN_FILE=str(claude_token),
                       SFORGE_AGENT_RUNTIME_DIR=str(claude_runtime))
        else:
            env.update(CODEX_AUTH_JSON_PATH=str(codex_auth), CODEX_CLI_VERSION="0.159.3",
                       SFORGE_AGENT_RUNTIME_DIR=str(codex_runtime))
        env.update(overrides)
        env = {key: value for key, value in env.items() if value != "<unset>"}
        if events.exists():
            events.unlink()
        result = subprocess.run(["bash", str(LAUNCHER)], env=env, capture_output=True,
                                text=True, timeout=120)
        log = events.read_text(encoding="utf-8") if events.exists() else ""
        return result, log

    run.paths = {"claude_token": claude_token, "codex_runtime": codex_runtime,
                 "claude_runtime": claude_runtime, "tmp": tmp_path}
    return run


@pytest.mark.parametrize(
    ("agent", "model", "effort", "message"),
    [
        ("codex", "claude-opus-5-5", "max", "claude-opus-5-5 runs only with the claude-code agent"),
        (None, "claude-opus-5-5", "max", "claude-opus-5-5 runs only with the claude-code agent"),
        ("claude-code", "gpt-example-model", "max", "gpt-example-model runs only with the codex agent"),
        ("claude-code", "claude-opus-4-8", "max", "unsupported formal model: claude-opus-4-8"),
        ("claude-code", "claude-opus-5-5", "ultra", "unsupported Claude Code effort: ultra"),
        ("codex", "gpt-example-model", "low", "unsupported reasoning effort: low"),
        ("gemini", "gpt-example-model", "max", "MLSYSFORGE_AGENT must be codex or claude-code"),
    ],
)
def test_agent_model_and_effort_are_paired(launch, agent, model, effort, message) -> None:
    result, log = launch(agent, model, effort)
    assert result.returncode != 0
    assert message in result.stderr
    assert "curl " not in log  # refused before any model API probe


def test_claude_code_needs_its_token_file(launch) -> None:
    result, _ = launch("claude-code", "claude-opus-5-5", CLAUDE_OAUTH_TOKEN_FILE="<unset>")
    assert result.returncode != 0
    assert "CLAUDE_OAUTH_TOKEN_FILE must point to a host-only Claude OAuth token file" in result.stderr


def test_claude_code_refuses_the_docker_group_reexec(launch) -> None:
    result, log = launch("claude-code", "claude-opus-5-5", DOCKER_INFO_RC="1")
    assert result.returncode != 0
    assert "needs Docker access without the docker-group re-exec" in result.stderr
    assert "sg " not in log


@pytest.mark.parametrize(
    ("agent", "model", "url"),
    [
        ("claude-code", "claude-opus-5-5", "https://api.anthropic.com/v1/models"),
        ("codex", "gpt-example-model", "https://api.openai.com/v1/models"),
        (None, "gpt-example-model-mini", "https://api.openai.com/v1/models"),
    ],
)
def test_host_proxy_probe_targets_the_agents_model_api(launch, agent, model, url) -> None:
    result, log = launch(agent, model)
    assert result.returncode != 0
    assert "fixed host proxy cannot reach the model API" in result.stderr
    assert "--proxy http://127.0.0.1:18080" in log and url in log


@pytest.mark.parametrize(
    ("mode", "kind", "message"),
    [
        (0o644, "file", "Claude OAuth token file must have mode 600"),
        (0o600, "empty", "Claude OAuth token file must be a nonempty regular, non-symlink file"),
        (0o600, "symlink", "Claude OAuth token file must be a nonempty regular, non-symlink file"),
    ],
)
def test_claude_token_file_must_be_private(launch, mode, kind, message) -> None:
    tmp = launch.paths["tmp"]
    token = tmp / f"token-{kind}"
    if kind == "symlink":
        token.symlink_to(launch.paths["claude_token"])
    else:
        _private(token, "" if kind == "empty" else "secret-value\n", mode)
    result, _ = launch("claude-code", "claude-opus-5-5", CLAUDE_OAUTH_TOKEN_FILE=str(token),
                       CURL_CODE="401", CURL_RC="0")
    assert result.returncode != 0
    assert message in result.stderr


def test_claude_code_runtime_marker_must_be_the_pinned_claude_runtime(launch) -> None:
    tmp = launch.paths["tmp"]
    old = _runtime(tmp / "claude-old", "claude-code", "2.1.200")
    cases = [
        (launch.paths["codex_runtime"], "a claude-code run requires a claude-code runtime, found: codex"),
        (old, f"agent runtime version 2.1.200 differs from the pinned Claude Code {CLAUDE_VERSION}"),
        # The marked pinned runtime passes; the next step needs the built images.
        (launch.paths["claude_runtime"], "missing build manifest"),
    ]
    for runtime, message in cases:
        result, _ = launch("claude-code", "claude-opus-5-5", SFORGE_AGENT_RUNTIME_DIR=str(runtime),
                           CURL_CODE="401", CURL_RC="0")
        assert result.returncode != 0
        assert message in result.stderr, (runtime, result.stderr)


def test_codex_path_is_unchanged(launch) -> None:
    result, _ = launch("codex", "gpt-example-model", SFORGE_AGENT_RUNTIME_DIR=str(launch.paths["claude_runtime"]),
                       CURL_CODE="401", CURL_RC="0")
    assert "formal run requires a codex runtime, found: claude-code" in result.stderr
    result, _ = launch("codex", "gpt-example-model", CURL_CODE="401", CURL_RC="0", RUN_STAMP="codex")
    assert "missing build manifest" in result.stderr
    completion = json.loads(
        (launch.paths["tmp"] / "logs/runs/kvbench-qwen36-swe192-v24-example-model-max-6h-codex"
         / "completion.json").read_text(encoding="utf-8")
    )
    assert completion["provenance"]["codex_cli_version"] == "0.159.3"

    source = LAUNCHER.read_text(encoding="utf-8")
    for fragment in (
        'assert_private_auth_file "${CODEX_AUTH_JSON_PATH}"',
        'export SFORGE_CODEX_AUTH_JSON_PATH="${CODEX_AUTH_JSON_PATH}"',
        'export SFORGE_CODEX_CLI_VERSION="${CODEX_CLI_VERSION}"',
        'MODEL_API_PROBE_URL="https://api.openai.com/v1/models"',
        "https://chatgpt.com/backend-api/models",
    ):
        assert fragment in source


def test_claude_run_id_and_completion_provenance(launch) -> None:
    result, _ = launch("claude-code", "claude-opus-5-5", CURL_CODE="401", CURL_RC="0", RUN_STAMP="claude")
    assert "missing build manifest" in result.stderr
    run_root = launch.paths["tmp"] / "logs/runs/kvbench-qwen36-swe192-v24-opus-5-5-max-6h-claude"
    completion = json.loads((run_root / "completion.json").read_text(encoding="utf-8"))
    assert completion["provenance"]["model"] == "claude-opus-5-5"
    assert completion["provenance"]["codex_cli_version"] == "none"


def test_sforge_run_and_manifest_carry_the_selected_agent() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    run_section = source[source.index('stage="agent_run"'):]
    assert '--agent "${AGENT}"' in run_section
    assert "--agent codex" not in source
    manifest_section = source[source.index("jq -n "): source.index('stage="agent_run"')]
    assert "agent: $agent," in manifest_section
    assert "CLAUDE_OAUTH_TOKEN_FILE" not in manifest_section
    exports = source[source.index("unset \\\n  SFORGE_AGENT_API_KEY"): source.index("jq -n ")]
    for fragment in (
        'export MLSYSFORGE_CLAUDE_OAUTH_TOKEN_FILE="${CLAUDE_OAUTH_TOKEN_FILE}"',
        'export MLSYSFORGE_AGENT_MODEL="${MODEL}" MLSYSFORGE_AGENT_EFFORT="${REASONING_EFFORT}"',
        'MLSYSFORGE_AGENT_BUDGET_SECONDS="${TASK_TIMEOUT}" MLSYSFORGE_RUN_LOG_ROOT="${RUN_LOG_ROOT}"',
        'export MLSYSFORGE_AGENT="${AGENT}"',
    ):
        assert fragment in exports
    # The Codex-only sidecar check runs only for Codex; Claude Code needs public egress.
    assert 'if [[ "${AGENT}" == codex ]]; then\n  sidecar_chatgpt_code="$(' in source
    assert "claude-code needs the task JSON internet: true" in source


def test_runtime_entry_binds_claude_code_with_the_task_internet_switches(
    runtime_module, monkeypatch
) -> None:
    from sforge.harness.agent.claude_code import ClaudeCodeAgent

    install = ClaudeCodeAgent.install_cmds
    assert install[0] == f"claude --version | grep -F {CLAUDE_VERSION}"
    assert "/app/RUN_BUDGET.json" in install[1]
    assert not any("npm" in step or "nodejs.org" in step for step in install)

    for internet, mode, expected in (
        ("true", "live", True),
        ("true", "disabled", False),
        ("false", "live", False),
        ("", "live", True),
    ):
        monkeypatch.setenv("MLSYSFORGE_WORK_INTERNET", internet)
        monkeypatch.setenv("CODEX_WEB_SEARCH_MODE", mode)
        assert runtime_module.claude_task_internet(True) is expected

    source = (ROOT / "scripts" / "sforge_runtime_entry.py").read_text(encoding="utf-8")
    main_block = source[source.index('if __name__ == "__main__":'):]
    assert "install_from_environment(claude_task_internet, sys.argv)" in main_block
    assert main_block.index("install_from_environment(") < main_block.index("main()")
