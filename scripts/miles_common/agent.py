"""The agent of a Miles task run: Codex (endpoint or ChatGPT login) or Claude Code (OAuth token).

The five Miles launchers (tasks/latest/miles-*/scripts/run_task.py) share this layer. A launch config selects the
agent with AGENT (default codex):

  codex        CODEX_CLI_VERSION + either MODEL_API_BASE_URL/MODEL_API_KEY_FILE (mode "endpoint") or
               CODEX_AUTH_JSON_PATH (mode "chatgpt"); SFORGE_AGENT_RUNTIME_DIR is a codex runtime of that version.
  claude-code  CLAUDE_OAUTH_TOKEN_FILE, a 0600 file holding one `claude setup-token` OAuth token (mode "claude"),
               exclusive of every Codex credential; SFORGE_AGENT_RUNTIME_DIR is a claude-code runtime.

Claude Code runs through the harness's local bindings (sforge/harness/agent/claude_local.py), which the task entry
(runtime/sforge_entry.py) installs from the MLSYSFORGE_* variables run_env() sets: the token is copied into Work as an
Agent-owned 0600 file after creation and never enters a container environment, and WebSearch/WebFetch are on when the
task allows internet. Its API traffic goes through the run relay's public-egress CONNECT proxy, so a Claude Code run
needs a task with "internet": true (the offline relay allowlists only the OpenAI hosts).
"""
from __future__ import annotations

import json
from pathlib import Path

AGENTS = ("codex", "claude-code")
# keys a launch config may set besides the launcher's own; all optional here, load_config checks the combination
OPTIONAL_KEYS = {"AGENT", "CLAUDE_OAUTH_TOKEN_FILE", "CODEX_CLI_VERSION"}


def name(cfg) -> str:
    return "claude-code" if cfg["MODE"] == "claude" else "codex"


def config_mode(cfg, path, require) -> str:
    """The run's model mode from a parsed launch config: "endpoint", "chatgpt" or "claude"."""
    agent = cfg.get("AGENT", "codex")
    require(agent in AGENTS, f"{path}: AGENT must be codex or claude-code, not {agent!r}")
    codex_keys = [k for k in ("CODEX_AUTH_JSON_PATH", "MODEL_API_BASE_URL", "MODEL_API_KEY_FILE") if cfg.get(k)]
    if agent == "claude-code":
        require(cfg.get("CLAUDE_OAUTH_TOKEN_FILE"), f"{path}: AGENT=claude-code needs CLAUDE_OAUTH_TOKEN_FILE")
        require(not codex_keys, f"{path}: CLAUDE_OAUTH_TOKEN_FILE is exclusive of {', '.join(codex_keys)}")
        return "claude"
    require(not cfg.get("CLAUDE_OAUTH_TOKEN_FILE"), f"{path}: CLAUDE_OAUTH_TOKEN_FILE needs AGENT=claude-code")
    require(cfg.get("CODEX_CLI_VERSION"), f"{path}: a Codex run needs CODEX_CLI_VERSION")
    endpoint = bool(cfg.get("MODEL_API_BASE_URL") or cfg.get("MODEL_API_KEY_FILE"))
    require(endpoint != bool(cfg.get("CODEX_AUTH_JSON_PATH")),
            f"{path}: set either MODEL_API_BASE_URL + MODEL_API_KEY_FILE or CODEX_AUTH_JSON_PATH")
    require(not endpoint or (cfg.get("MODEL_API_BASE_URL") and cfg.get("MODEL_API_KEY_FILE")),
            f"{path}: an endpoint needs both MODEL_API_BASE_URL and MODEL_API_KEY_FILE")
    return "endpoint" if endpoint else "chatgpt"


def check_credentials(cfg, require):
    if cfg["MODE"] == "endpoint":
        require(Path(cfg["MODEL_API_KEY_FILE"]).is_file(), "MODEL_API_KEY_FILE missing")
        return
    key = "CLAUDE_OAUTH_TOKEN_FILE" if cfg["MODE"] == "claude" else "CODEX_AUTH_JSON_PATH"
    path = Path(cfg[key]).expanduser()
    require(path.is_file() and not path.is_symlink(), f"{key} missing")
    require(path.stat().st_mode & 0o077 == 0, f"{key} must be mode 0600")
    if cfg["MODE"] == "claude":
        token = path.read_bytes().strip()
        require(token and not any(c <= 32 for c in token), f"{key} must hold exactly one token")


def check_runtime(cfg, require) -> dict:
    """The host agent runtime's marker, which must name the configured agent (and, for Codex, its version)."""
    marker = Path(cfg["SFORGE_AGENT_RUNTIME_DIR"]) / ".sforge-agent-runtime.json"
    require(marker.is_file(), f"agent runtime marker missing: {marker}")
    info = json.loads(marker.read_text())
    if cfg["MODE"] == "claude":
        require(info.get("agent") == "claude-code" and info.get("cli") == "claude",
                f"agent runtime is {info.get('agent')} {info.get('version')}, AGENT=claude-code needs a claude-code runtime")
    else:
        require(info.get("agent") == "codex" and info.get("version") == cfg["CODEX_CLI_VERSION"],
                f"agent runtime is {info.get('agent')} {info.get('version')}, config pins codex {cfg['CODEX_CLI_VERSION']}")
    return info


def check_internet(cfg, internet, require):
    require(cfg["MODE"] != "claude" or internet,
            "Claude Code reaches its API through the relay's public egress: the task must have \"internet\": true")


def manifest_fields(cfg, internet) -> dict:
    """What a launch manifest and a dry-run plan record about the agent."""
    if cfg["MODE"] == "claude":
        return {"agent": "claude-code", "claude_web_tools": "WebSearch,WebFetch" if internet else "disabled"}
    return {"agent": "codex", "codex_web_search": "live" if internet else "disabled",
            "codex_cli_version": cfg["CODEX_CLI_VERSION"]}


def run_env(cfg, run_root) -> dict:
    """Environment of `sforge run` for the agent (credentials other than the Claude token file are added by the
    launcher's mode branches)."""
    env = {"SFORGE_AGENT_MODEL": cfg["MODEL"], "SFORGE_CODEX_REASONING_EFFORT": cfg["REASONING_EFFORT"]}
    if cfg["MODE"] != "claude":
        env["SFORGE_CODEX_CLI_VERSION"] = cfg["CODEX_CLI_VERSION"]
        return env
    # sforge/harness/agent/claude_local.install_from_environment, called by runtime/sforge_entry.py
    env.update(MLSYSFORGE_AGENT="claude-code", MLSYSFORGE_AGENT_MODEL=cfg["MODEL"],
               MLSYSFORGE_AGENT_EFFORT=cfg["REASONING_EFFORT"], MLSYSFORGE_AGENT_BUDGET_SECONDS=cfg["TASK_TIMEOUT"],
               MLSYSFORGE_RUN_LOG_ROOT=str(run_root),
               MLSYSFORGE_CLAUDE_OAUTH_TOKEN_FILE=str(Path(cfg["CLAUDE_OAUTH_TOKEN_FILE"]).expanduser().resolve()))
    return env
