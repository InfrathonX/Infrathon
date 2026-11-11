"""Pinned Claude Code runtime and credentials with SForge's shared submit barrier.

The task image remains agent-free. Controller-owned settings and auth arrive
after Worker creation; public images, candidates and artifact projections never
contain credentials. Supervisor integration uses ``claude_command`` rather than
the Codex command builder. This module does not launch or allocate a Sandbox.
"""
from __future__ import annotations

import json
import math
from pathlib import Path, PurePosixPath
import shlex
from urllib.parse import urlsplit

from sforge.harness.agent_config import (CLAUDE_AUTH_CACHE_PATH, AgentCredentials,
                                        auth_secrets, read_private_file)


CLAUDE_SETTINGS_PATH = "/home/agent/.claude/settings.json"
CLAUDE_KEY_PATH = "/home/agent/.config/sforge/model.key"
CLAUDE_HOOK_PATH = "/opt/sforge/claude-submit-wait-hook.py"
CLAUDE_TOOLS = "Bash,Read,Edit,Write,Glob,Grep"
# Added when the task allows internet (every task since 2026-10-07).
CLAUDE_WEB_TOOLS = "WebSearch,WebFetch"


def parse_claude_oauth_cache(data: bytes) -> dict:
    """Keep the native refreshable cache; an access token alone cannot cover a run.

    ``expiresAt`` is a Unix timestamp in milliseconds. Expired access tokens are
    accepted when the refresh token is present so the native CLI can renew them.
    Errors deliberately contain no credential contents.
    """
    try:
        document = json.loads(data)
    except (ValueError, UnicodeDecodeError):
        raise ValueError("Claude OAuth cache is not valid JSON") from None
    oauth = document.get("claudeAiOauth") if isinstance(document, dict) else None
    if not isinstance(oauth, dict):
        raise ValueError("Claude OAuth cache must contain claudeAiOauth")
    for name in ("accessToken", "refreshToken"):
        value = oauth.get(name)
        if (not isinstance(value, str) or not value.strip()
                or value != value.strip() or any(ord(character) < 32 for character in value)):
            raise ValueError("Claude OAuth cache requires native access and refresh tokens")
    expiry = oauth.get("expiresAt")
    if (isinstance(expiry, bool) or not isinstance(expiry, (int, float))
            or not math.isfinite(expiry) or expiry <= 0):
        raise ValueError("Claude OAuth cache requires a positive finite expiresAt timestamp")
    return document


def load_model_config(path: Path) -> dict[str, str]:
    from dotenv import dotenv_values
    from sforge.harness.agent_runtime import load_agent_runtime

    values = {key: value for key, value in dotenv_values(path, interpolate=False).items()
              if value is not None}
    for key in ("MODEL", "SFORGE_AGENT_RUNTIME_DIR"):
        if not values.get(key):
            raise ValueError("Claude Code model config requires " + key)
    if values.get("CODEX_AUTH_JSON_PATH"):
        raise ValueError("Claude Code cannot use a Codex ChatGPT credential cache")
    cache = bool(values.get("CLAUDE_AUTH_JSON_PATH"))
    api = bool(values.get("MODEL_API_KEY_FILE") or values.get("MODEL_API_BASE_URL"))
    if cache == api:
        raise ValueError("Choose exactly one Claude authentication mode: CLAUDE_AUTH_JSON_PATH or MODEL_API_KEY_FILE + MODEL_API_BASE_URL")
    if cache:
        mode = "claude-oauth"
        credential_name = "CLAUDE_AUTH_JSON_PATH"
        credential_path = Path(values[credential_name]).expanduser().absolute()
        parse_claude_oauth_cache(read_private_file(credential_path))
    else:
        mode = "anthropic-api-key"
        credential_name = "MODEL_API_KEY_FILE"
        if not values.get("MODEL_API_KEY_FILE") or not values.get("MODEL_API_BASE_URL"):
            raise ValueError("Claude API authentication requires MODEL_API_KEY_FILE and MODEL_API_BASE_URL")
        url = urlsplit(values["MODEL_API_BASE_URL"])
        if (url.scheme not in {"http", "https"} or not url.hostname or url.username
                or url.password or url.query or url.fragment):
            raise ValueError("Claude endpoint must be an HTTP(S) URL without credentials or query")
        credential_path = Path(values[credential_name]).expanduser().absolute()
        if not read_private_file(credential_path).strip():
            raise ValueError("Claude model credential file is empty")
    if values.get("AUTH_MODE") not in (None, mode):
        raise ValueError("Claude AUTH_MODE differs from configured credential source")
    values[credential_name] = str(credential_path)
    values["SFORGE_AGENT_RUNTIME_DIR"] = str(Path(values["SFORGE_AGENT_RUNTIME_DIR"]).expanduser().absolute())
    runtime = load_agent_runtime(Path(values["SFORGE_AGENT_RUNTIME_DIR"]))
    if runtime.agent != "claude-code" or runtime.cli != "claude":
        raise ValueError("Claude Code requires a claude-code runtime")
    effort = values.get("REASONING_EFFORT", "max")
    if effort not in {"low", "medium", "high", "xhigh", "max"}:
        raise ValueError("Unsupported Claude reasoning effort")
    values.update(AGENT="claude-code", AUTH_MODE=mode,
                  REASONING_EFFORT=effort, runtime_version=runtime.version)
    return values


def load_credentials(model: dict) -> AgentCredentials:
    if model.get("AUTH_MODE") == "claude-oauth":
        payload = read_private_file(Path(model["CLAUDE_AUTH_JSON_PATH"]))
        return AgentCredentials("claude-oauth", payload,
                                frozenset(auth_secrets(parse_claude_oauth_cache(payload))))
    if model.get("AUTH_MODE") != "anthropic-api-key":
        raise ValueError("Claude Code requires native OAuth or an Anthropic-compatible model key")
    payload = read_private_file(Path(model["MODEL_API_KEY_FILE"])).strip()
    if not payload:
        raise ValueError("Claude model credential file is empty")
    try:
        secret = payload.decode("utf-8")
    except UnicodeDecodeError:
        raise ValueError("Claude model credential must be UTF-8") from None
    return AgentCredentials("anthropic-api-key", payload, frozenset({secret}))


def claude_settings(model: dict, *, submit_wait_timeout_seconds: int = 21_720) -> dict:
    from sforge.harness.submit_wait import submit_wait_hooks_config_fragment

    # Reuse the existing timeout/path validation without inventing a second
    # submission deadline. Only the hook container format differs between CLIs.
    submit_wait_hooks_config_fragment(CLAUDE_HOOK_PATH, timeout_seconds=submit_wait_timeout_seconds)
    hooks = {}
    for name, phase in (("PreToolUse", "pre"), ("PostToolUse", "post"),
                        ("PostToolUseFailure", "failure")):
        hooks[name] = [{"matcher": ".*", "hooks": [{"type": "command",
                       "command": "python3 " + shlex.quote(CLAUDE_HOOK_PATH) + " " + phase,
                       "timeout": submit_wait_timeout_seconds if phase == "pre" else 15}]}]
    environment = {"CLAUDE_CODE_EFFORT_LEVEL": model.get("REASONING_EFFORT", "max")}
    if model.get("AUTH_MODE", "anthropic-api-key") == "anthropic-api-key":
        environment["ANTHROPIC_BASE_URL"] = model["MODEL_API_BASE_URL"].rstrip("/")
    return {"model": model["MODEL"], "hooks": hooks,
            "permissions": {"deny": ["Agent", "Task", "mcp__*"]},
            "env": environment}


def claude_command(model: dict, prompt: str, *, resume: str | None = None,
                   probe: bool = False, web_search: bool = True) -> list[str]:
    # With --verbose the native JSON output is one array, not JSONL. Keep the
    # probe on the same event transport as the task so its terminal-result
    # parser sees each record independently.
    command = ["claude", "-p", "--output-format", "stream-json",
               "--verbose", "--model", model["MODEL"], "--effort",
               model.get("REASONING_EFFORT", "max"), "--settings", CLAUDE_SETTINGS_PATH,
               "--setting-sources", "", "--tools",
               "" if probe else CLAUDE_TOOLS + ("," + CLAUDE_WEB_TOOLS if web_search else ""),
               "--disallowedTools", "Agent,Task,mcp__*", "--strict-mcp-config",
               "--mcp-config", '{"mcpServers":{}}', "--dangerously-skip-permissions"]
    if resume == "continue":
        command += ["--continue"]
    elif resume:
        command += ["--resume", resume]
    return command + [prompt]


def normalize_hook_event(event: dict, phase: str) -> dict:
    """Translate Claude's failure callback to the shared engine's terminal post.

    Claude fires PostToolUseFailure rather than PostToolUse for a failed Bash
    command. Without this translation an ordinary failed experiment would stay
    pending and block every later submit. Runtime background IDs stay pending.
    """
    if not isinstance(event, dict):
        raise ValueError("Claude hook input must be an object")
    value = dict(event)
    if phase == "pre":
        tool_input = value.get("tool_input")
        if isinstance(tool_input, dict) and tool_input.get("run_in_background") is True:
            raise ValueError("SForge tools must finish before another operation; background tools are disabled")
        if str(value.get("tool_name", "")).lower() in {"agent", "task"}:
            raise ValueError("SForge uses one Agent; subagents are disabled")
        return value
    if phase not in {"post", "failure"}:
        raise ValueError("Unknown Claude tool hook phase")
    value["hook_event_name"] = "PostToolUse"
    response = value.get("tool_response")
    if phase == "failure":
        value["tool_response"] = {"status": "failed", "exit_code": 1,
                                  "output": str(value.get("error", "Claude tool failed"))}
    elif isinstance(response, dict) and any(response.get(key) is not None
            for key in ("backgroundTaskId", "background_task_id", "task_id", "session_id", "process_id")):
        value["tool_response"] = {**response, "status": "running"}
    elif isinstance(response, dict) and response.get("status") in {"running", "in_progress", "pending"}:
        value["tool_response"] = response
    else:
        # This event is emitted after synchronous tool success. Stdout may
        # contain arbitrary text resembling an old background status.
        value["tool_response"] = {"status": "completed", "exit_code": 0,
                                  "output": response}
    return value


def generate_claude_hook_script() -> str:
    """Standalone adapter loads the same installed state/execute/replay engine."""
    from sforge.harness.submit_wait import SUBMIT_WAIT_HOOK_PATH

    # Share the normalization implementation, not a copied submission state
    # machine. The generated file needs only the Worker Python standard library.
    import inspect
    normalize = inspect.getsource(normalize_hook_event)
    return ("#!/usr/bin/env python3\nimport io,json,runpy,sys\n"
            + normalize + "\n"
            + "engine=runpy.run_path(" + repr(SUBMIT_WAIT_HOOK_PATH) + ")\n"
            + "def main():\n"
            + " if len(sys.argv)>1 and sys.argv[1] in ('replay','reconcile'): return engine['main']()\n"
            + " phase=sys.argv[1] if len(sys.argv)==2 else 'pre'\n"
            + " try:\n"
            + "  payload=sys.stdin.buffer.read(engine['MAX_INPUT_BYTES']+1)\n"
            + "  if len(payload)>engine['MAX_INPUT_BYTES']: raise ValueError('Input exceeds size limit')\n"
            + "  event=normalize_hook_event(json.loads(payload),phase)\n"
            + " except Exception:\n"
            + "  print(json.dumps(engine['deny']('pre' if phase=='pre' else 'post','SForge cannot validate this Claude tool; background/subagent operations are disabled.'))); return 0\n"
            + " sys.argv[1]='pre' if phase=='pre' else 'post'\n"
            + " class Input:\n"
            + "  buffer=io.BytesIO(json.dumps(event).encode())\n"
            + " sys.stdin=Input()\n"
            + " return engine['main']()\n"
            + "if __name__=='__main__': sys.exit(main())\n")


def install_and_validate_agent(backend, worker, model, runtime, credentials, check,
                               *, submit_wait_timeout_seconds: int = 21_720,
                               tool_catalog=None) -> dict:
    from sforge.harness.submit_wait import (SUBMIT_WAIT_HOOK_PATH, SUBMIT_WAIT_STATE_DIR,
                                           generate_submit_wait_hook_script)

    if runtime.agent != "claude-code" or credentials.mode not in {"anthropic-api-key", "claude-oauth"}:
        raise ValueError("Claude Worker runtime and authentication mode must match")
    backend.copy_to_container(worker, runtime.root, PurePosixPath("/opt/agent-runtime"))
    check(backend, worker, ["mkdir", "-p", "/opt/sforge", SUBMIT_WAIT_STATE_DIR,
                           "/home/agent/.claude", "/home/agent/.config/sforge"], timeout=15)
    backend.write_to_container(worker, generate_submit_wait_hook_script(), PurePosixPath(SUBMIT_WAIT_HOOK_PATH))
    backend.write_to_container(worker, generate_claude_hook_script(), PurePosixPath(CLAUDE_HOOK_PATH))
    destination = CLAUDE_AUTH_CACHE_PATH if credentials.mode == "claude-oauth" else CLAUDE_KEY_PATH
    other = CLAUDE_KEY_PATH if credentials.mode == "claude-oauth" else CLAUDE_AUTH_CACHE_PATH
    check(backend, worker, ["rm", "-f", other], timeout=15)
    check(backend, worker, ["install", "-o", "agent", "-g", "agent", "-m", "600",
                           "/dev/null", destination], timeout=15)
    backend.write_to_container(worker, credentials.payload.decode(), PurePosixPath(destination))
    backend.write_to_container(worker, json.dumps(claude_settings(model,
        submit_wait_timeout_seconds=submit_wait_timeout_seconds)), PurePosixPath(CLAUDE_SETTINGS_PATH))
    check(backend, worker, "chmod -R a+rX /opt/agent-runtime; chown root:root "
          + SUBMIT_WAIT_HOOK_PATH + " " + CLAUDE_HOOK_PATH + "; chmod 644 "
          + SUBMIT_WAIT_HOOK_PATH + " " + CLAUDE_HOOK_PATH + "; chown -R agent:agent "
          + "/home/agent/.claude /home/agent/.config/sforge " + SUBMIT_WAIT_STATE_DIR
          + "; chmod 700 /home/agent/.claude /home/agent/.config/sforge " + SUBMIT_WAIT_STATE_DIR
          + "; chmod 600 " + CLAUDE_SETTINGS_PATH + " " + destination, timeout=30)
    command = claude_command(model, "Reply with exactly READY. Do not use tools or change files.", probe=True)
    probe = '''import json,os,pathlib,subprocess,sys
env=dict(os.environ);env.update(HOME='/home/agent',USER='agent',LOGNAME='agent')
env['PATH']='/opt/agent-runtime/bin:'+env['PATH']
for key in tuple(env):
 if key.startswith('CLAUDE') or key.startswith('ANTHROPIC'):env.pop(key,None)
if sys.argv[1]=='anthropic-api-key':
 env['ANTHROPIC_BASE_URL']=sys.argv[2]
 env['ANTHROPIC_API_KEY']=pathlib.Path('/home/agent/.config/sforge/model.key').read_text().strip()
env['CLAUDE_CODE_EFFORT_LEVEL']=sys.argv[3]
env['CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC']='1'
result=subprocess.run(['runuser','-u','agent','--',*json.loads(sys.argv[4])],cwd='/tmp',env=env,capture_output=True,text=True,timeout=150)
assert result.returncode==0,'Claude Code model preflight failed'
records=[]
for line in result.stdout.splitlines():
 try:records.append(json.loads(line))
 except ValueError:pass
final=[record for record in records if isinstance(record,dict) and record.get('type')=='result']
assert len(final)==1 and final[0].get('subtype')=='success' and final[0].get('is_error') is not True,'Claude Code probe did not complete'
assert final[0].get('result','').strip()=='READY','Claude Code did not return exact READY'
if sys.argv[1]=='claude-oauth':
 assert set(final[0].get('modelUsage',{}))=={sys.argv[5]},'Claude OAuth probe used a different model'
print('Pinned Claude Code CLI and configured model preflight passed')
'''
    print(check(backend, worker, ["python3", "-c", probe, credentials.mode,
          model.get("MODEL_API_BASE_URL", "").rstrip("/"),
          model.get("REASONING_EFFORT", "max"), json.dumps(command), model["MODEL"]], timeout=170), flush=True)
    return {"status": "passed", "auth_mode": credentials.mode, "runtime_uploaded": True,
            "credentials_injected": True, "claude_cli": True,
            "submission_wait_policy": "wait-for-result", "model": model["MODEL"],
            "reasoning_effort": model.get("REASONING_EFFORT", "max")}
