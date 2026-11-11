"""Claude Code bindings for local native SForge runs (task launchers and flux-local).

A local run can drive the pinned Claude Code runtime instead of Codex, with a
long-lived subscription OAuth token (``claude setup-token``) read from a
private 0600 file named by ``CLAUDE_OAUTH_TOKEN_FILE``. The token is never
placed in a container environment, image, log or artifact: it is installed as
an Agent-owned 0600 file inside Work after creation, and a small wrapper reads
it into ``CLAUDE_CODE_OAUTH_TOKEN`` only for the CLI process it execs.

Tool policy matches the PyroMind Claude Worker (sforge/harness/agent/claude_agent.py):
one Agent, no subagents or MCP, the shared synchronous submission hooks, and
WebSearch/WebFetch whenever the task allows internet.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import signal
import subprocess
import tarfile
import tempfile
import time
import uuid

from sforge.harness.agent_runtime import load_agent_runtime
from sforge.harness.agent_config import read_private_file, redact
from sforge.harness.agent.claude_agent import (
    CLAUDE_HOOK_PATH, CLAUDE_SETTINGS_PATH, CLAUDE_TOOLS, CLAUDE_WEB_TOOLS,
    claude_settings, generate_claude_hook_script,
)
from sforge.harness.submit_wait import (
    SUBMIT_WAIT_HOOK_PATH, SUBMIT_WAIT_STATE_DIR, generate_submit_wait_hook_script,
)


CLAUDE_RUNTIME_VERSION = "2.1.287"
TOKEN_MODE = "claude-oauth-token"
TOKEN_PATH = "/home/agent/.config/sforge/claude-oauth.token"
CLI_WRAPPER = "/opt/sforge/local-claude.py"
STOP_HOOK_PATH = "/tmp/sforge-stop-hook.sh"


def _token(path):
    data = read_private_file(Path(path)).strip()
    try:
        token = data.decode("ascii")
    except UnicodeDecodeError:
        raise ValueError("Claude OAuth token must be ASCII") from None
    if not token or any(character.isspace() or ord(character) < 33 for character in token):
        raise ValueError("Claude OAuth token file must hold exactly one token")
    return token


def load_model_config(path, fallback=None):
    """Codex and Claude cache/API configs keep their PyroMind loader (or the caller's)."""
    from dotenv import dotenv_values
    from sforge.harness.agent_config import load_model_config as load_shared

    values = {key: value for key, value in dotenv_values(path, interpolate=False).items() if value is not None}
    if values.get("AGENT") != "claude-code" or not values.get("CLAUDE_OAUTH_TOKEN_FILE"):
        return (fallback or load_shared)(path)
    if values.get("CLAUDE_AUTH_JSON_PATH") or values.get("CODEX_AUTH_JSON_PATH") \
            or values.get("MODEL_API_KEY_FILE") or values.get("MODEL_API_BASE_URL"):
        raise ValueError("CLAUDE_OAUTH_TOKEN_FILE is exclusive of every other Agent credential")
    for key in ("MODEL", "SFORGE_AGENT_RUNTIME_DIR"):
        if not values.get(key):
            raise ValueError("Claude Code model config requires " + key)
    values["CLAUDE_OAUTH_TOKEN_FILE"] = str(Path(values["CLAUDE_OAUTH_TOKEN_FILE"]).expanduser().absolute())
    _token(values["CLAUDE_OAUTH_TOKEN_FILE"])
    values["SFORGE_AGENT_RUNTIME_DIR"] = str(Path(values["SFORGE_AGENT_RUNTIME_DIR"]).expanduser().absolute())
    runtime = load_agent_runtime(Path(values["SFORGE_AGENT_RUNTIME_DIR"]))
    if runtime.agent != "claude-code" or runtime.cli != "claude":
        raise ValueError("Claude Code requires a claude-code runtime")
    values.update(AGENT="claude-code", AUTH_MODE=TOKEN_MODE,
                  REASONING_EFFORT=values.get("REASONING_EFFORT", "max"), runtime_version=runtime.version)
    return values


EFFORTS = ("low", "medium", "high", "xhigh", "max")


def require_local_profile(model, runtime):
    if (model["AGENT"] != "claude-code" or model["AUTH_MODE"] != TOKEN_MODE
            or model["REASONING_EFFORT"] not in EFFORTS or runtime.version != CLAUDE_RUNTIME_VERSION):
        raise RuntimeError("Local Claude Code requires an OAuth token, a supported effort and Claude Code "
                           + CLAUDE_RUNTIME_VERSION)


def model_from_environment(model_name, effort, token_file, runtime_dir):
    """A task launcher's flags as the model dict the bindings take."""
    values = {"AGENT": "claude-code", "MODEL": model_name, "REASONING_EFFORT": effort,
              "AUTH_MODE": TOKEN_MODE, "CLAUDE_OAUTH_TOKEN_FILE": str(Path(token_file).absolute()),
              "SFORGE_AGENT_RUNTIME_DIR": str(Path(runtime_dir).absolute())}
    _token(values["CLAUDE_OAUTH_TOKEN_FILE"])
    runtime = load_agent_runtime(Path(values["SFORGE_AGENT_RUNTIME_DIR"]))
    require_local_profile(values, runtime)
    values["runtime_version"] = runtime.version
    return values, runtime


def install_bytes(backend, handle, path, data, *, mode=0o644, owner="root:root"):
    """Send bytes into a container without writing them to any host artifact."""
    path = PurePosixPath(path)
    result = backend.exec_run_with_exit_code(handle, ["mkdir", "-p", str(path.parent)], timeout=30, user="root")
    if result.exit_code or result.timed_out:
        raise RuntimeError("Cannot prepare Worker runtime directory")
    payload = data.encode() if isinstance(data, str) else data
    memory = io.BytesIO()
    with tarfile.open(fileobj=memory, mode="w") as archive:
        info = tarfile.TarInfo(path.name)
        info.size, info.mode = len(payload), mode
        archive.addfile(info, io.BytesIO(payload))
    if not backend._raw(handle).put_archive(str(path.parent), memory.getvalue()):
        raise RuntimeError("Cannot install Worker runtime bytes")
    result = backend.exec_run_with_exit_code(handle,
        ["/bin/bash", "-c", "chown " + owner + " " + shlex.quote(str(path))
         + "; chmod " + oct(mode)[2:] + " " + shlex.quote(str(path))], timeout=30, user="root")
    if result.exit_code or result.timed_out:
        raise RuntimeError("Cannot secure Worker runtime file")


def _write_private_json(path, value):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def settings_document(model, submit_wait_timeout_seconds, *, stop_hook=True):
    document = claude_settings(model, submit_wait_timeout_seconds=submit_wait_timeout_seconds)
    if stop_hook:
        document["hooks"]["Stop"] = [{"hooks": [{"type": "command", "command": STOP_HOOK_PATH}]}]
    return json.dumps(document, indent=2) + "\n"


def cli_flags(model, *, web_search, settings_path=CLAUDE_SETTINGS_PATH):
    tools = CLAUDE_TOOLS + ("," + CLAUDE_WEB_TOOLS if web_search else "")
    return ["--effort", model["REASONING_EFFORT"], "--settings", settings_path, "--setting-sources", "",
            "--tools", tools, "--disallowedTools", "Agent,Task,mcp__*", "--strict-mcp-config",
            "--mcp-config", '{"mcpServers":{}}']


def cli_wrapper_source():
    return '''#!/usr/bin/env python3
import json,os,pathlib,subprocess,sys
hook='/opt/sforge/submit-wait-hook.py'
state=pathlib.Path('/run/sforge-submit-wait/state.json')
if state.exists():
 result=subprocess.run([sys.executable,hook,'reconcile'],capture_output=True,timeout=30)
 if result.returncode:raise SystemExit('Submission completion proof is unavailable; refusing Agent resume')
 value=json.loads(state.read_text())
 if value.get('schema_version')!=1 or value.get('active') is not None or not isinstance(value.get('active_tools'),dict) or value['active_tools']:
  raise SystemExit('An earlier tool is unfinished; refusing Agent resume')
env=dict(os.environ)
for key in tuple(env):
 if key.startswith(('ANTHROPIC','CLAUDE')) or key in ('OPENAI_API_KEY','OPENAI_BASE_URL','OPENAI_API_BASE','CODEX_API_KEY','PYROMIND_API_KEY','PYROMIND_STORAGE_SECRET_KEY','SFORGE_ADMIN_SECRET'):env.pop(key,None)
env['CLAUDE_CODE_OAUTH_TOKEN']=pathlib.Path(TOKEN_PATH).read_text().strip()
env['CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC']='1'
os.execvpe('claude',['claude',*sys.argv[1:]],env)
'''.replace("TOKEN_PATH", repr(TOKEN_PATH))


def validate_stream(stdout, returncode, model_name, marker=None):
    """Parse Claude's stream-json; optionally require the fail-then-marker tool roundtrip."""
    records = []
    for line in stdout.splitlines():
        try:
            record = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            continue
        if isinstance(record, dict):
            records.append(record)
    final = [record for record in records if record.get("type") == "result"]
    if (returncode or len(final) != 1 or final[0].get("subtype") != "success"
            or final[0].get("is_error") is True):
        raise RuntimeError("Claude Code did not complete a successful turn")
    if final[0].get("result", "").strip() != "READY":
        raise RuntimeError("Claude Code did not return the exact READY reply")
    if set(final[0].get("modelUsage", {})) != {model_name}:
        raise RuntimeError("Claude Code used a model other than the configured one")
    result = {"returncode": returncode, "turn_completed": True, "observed_models": [model_name]}
    if marker is None:
        return result
    tool_results = []
    for record in records:
        content = record.get("message", {}).get("content") if record.get("type") == "user" else None
        for block in content if isinstance(content, list) else ():
            if isinstance(block, dict) and block.get("type") == "tool_result":
                text = block.get("content")
                if isinstance(text, list):
                    text = "".join(part.get("text", "") for part in text if isinstance(part, dict))
                tool_results.append((bool(block.get("is_error")), str(text or "")))
    failed = [index for index, (error, _) in enumerate(tool_results) if error]
    succeeded = [index for index, (error, text) in enumerate(tool_results) if not error and marker in text]
    if not any(first < second for first in failed for second in succeeded):
        raise RuntimeError("Claude Code did not recover from the ordinary failed command to the unique marker")
    result.update(tools=len(tool_results), completed_command_marker_verified=True,
                  ordinary_tool_failure_recovered=True)
    return result


def install_bindings(model, runtime, *, budget_seconds, artifact_root, task_internet,
                     install_bytes=install_bytes, write_private_json=_write_private_json,
                     on_preflight=None, redact_secrets=redact):
    """Patch ClaudeCodeAgent for this process's run.

    ``task_internet(requested) -> bool`` resolves the task's internet flag (the
    task entry's own wrapper); ``on_preflight(runtime)`` records the passed
    in-Worker model preflight for a caller that keeps run state.
    """
    from sforge.harness.agent.claude_code import ClaudeCodeAgent

    require_local_profile(model, runtime)
    token = _token(model["CLAUDE_OAUTH_TOKEN_FILE"])
    wait_timeout = budget_seconds + 120

    def write_settings(backend, handle):
        install_bytes(backend, handle, CLAUDE_SETTINGS_PATH, settings_document(model, wait_timeout),
                      mode=0o600, owner="agent:agent")

    def prepare(self, backend, handle, log_dir, logger):
        install_bytes(backend, handle, SUBMIT_WAIT_HOOK_PATH, generate_submit_wait_hook_script(track_ordinary_tools=False))
        install_bytes(backend, handle, CLAUDE_HOOK_PATH, generate_claude_hook_script())
        install_bytes(backend, handle, CLI_WRAPPER, cli_wrapper_source())
        result = backend.exec_run_with_exit_code(handle,
            ["/bin/bash", "-c", "set -e; rm -f /home/agent/.codex/auth.json /home/agent/.config/sforge/model.key "
             "/home/agent/.claude/.credentials.json; mkdir -p /home/agent/.claude /home/agent/.config/sforge "
             + SUBMIT_WAIT_STATE_DIR + "; chown agent:agent /home/agent/.claude /home/agent/.config/sforge "
             + SUBMIT_WAIT_STATE_DIR + "; chmod 700 /home/agent/.claude /home/agent/.config/sforge "
             + SUBMIT_WAIT_STATE_DIR], timeout=30, user="root")
        if result.exit_code or result.timed_out:
            raise RuntimeError("Cannot prepare local Claude Code state directories")
        install_bytes(backend, handle, TOKEN_PATH, token, mode=0o600, owner="agent:agent")
        write_settings(backend, handle)
        self._submission_forbidden_blobs = tuple(set(
            getattr(self, "_submission_forbidden_blobs", ()) + (token.encode(),)))
        command = ["python3", CLI_WRAPPER, "-p", "--output-format", "stream-json", "--verbose",
                   "--model", model["MODEL"], *cli_flags(model, web_search=False),
                   "--dangerously-skip-permissions", "Reply with exactly READY. Do not use tools or change files."]
        # The probe must not trip the Stop hook, which tells the Agent to keep working.
        install_bytes(backend, handle, CLAUDE_SETTINGS_PATH,
                      settings_document(model, wait_timeout, stop_hook=False), mode=0o600, owner="agent:agent")
        result = backend.exec_run_with_timeout(handle,
            ["/bin/bash", "-c", " ".join(shlex.quote(part) for part in command)],
            timeout=180, user="agent", workdir="/tmp", environment={"HOME": "/home/agent"})
        write_settings(backend, handle)
        proof = {"exit_code": result.exit_code, "timed_out": result.timed_out,
                 "output": redact_secrets(result.output, {token}), "model": model["MODEL"],
                 "reasoning_effort": model["REASONING_EFFORT"], "runtime_version": runtime.version,
                 "auth_mode": TOKEN_MODE}
        write_private_json(Path(artifact_root) / "model-preflight.json", proof)
        if result.timed_out:
            raise RuntimeError("Pinned local model preflight timed out before Agent budget")
        validate_stream(result.output, result.exit_code, model["MODEL"])
        if on_preflight is not None:
            on_preflight(runtime)
        logger.info("Pinned Claude Code model/effort/runtime preflight passed")

    ClaudeCodeAgent.prepare_runtime = prepare
    original_stop = ClaudeCodeAgent.install_stop_hook

    def stop(self, backend, handle, log_dir, logger):
        # The native hook writes its own settings.json; keep its stop script
        # and restore the combined submission + stop settings afterwards.
        original_stop(self, backend, handle, log_dir, logger)
        write_settings(backend, handle)

    ClaudeCodeAgent.install_stop_hook = stop
    original_format = ClaudeCodeAgent.format_run_cmd

    def format_command(self, *args, **kwargs):
        internet = task_internet(kwargs.get("internet", True))
        kwargs["internet"] = internet
        command = original_format(self, *args, **kwargs)
        if not command.startswith("claude "):
            raise RuntimeError("Unexpected native Claude Code command shape")
        flags = " ".join(shlex.quote(part) for part in cli_flags(model, web_search=internet))
        return "python3 " + CLI_WRAPPER + " " + command[len("claude "):] + " " + flags

    ClaudeCodeAgent.format_run_cmd = format_command


AGENT_ENV = "MLSYSFORGE_AGENT"


def install_from_environment(task_internet, argv):
    """Task-entry hook: bind Claude Code when the launcher selected it for a run.

    The task launchers export MLSYSFORGE_AGENT, MLSYSFORGE_AGENT_MODEL,
    MLSYSFORGE_AGENT_EFFORT, MLSYSFORGE_AGENT_BUDGET_SECONDS,
    MLSYSFORGE_RUN_LOG_ROOT, MLSYSFORGE_CLAUDE_OAUTH_TOKEN_FILE and
    SFORGE_AGENT_RUNTIME_DIR. Codex (the default) and non-run commands are untouched.
    """
    agent = os.environ.get(AGENT_ENV, "codex")
    if agent not in ("codex", "claude-code"):
        raise SystemExit(AGENT_ENV + " must be codex or claude-code")
    if agent != "claude-code" or "run" not in argv[1:]:
        return
    model, runtime = model_from_environment(
        os.environ["MLSYSFORGE_AGENT_MODEL"], os.environ["MLSYSFORGE_AGENT_EFFORT"],
        os.environ["MLSYSFORGE_CLAUDE_OAUTH_TOKEN_FILE"], os.environ["SFORGE_AGENT_RUNTIME_DIR"])
    install_bindings(model, runtime, budget_seconds=int(os.environ["MLSYSFORGE_AGENT_BUDGET_SECONDS"]),
                     artifact_root=Path(os.environ["MLSYSFORGE_RUN_LOG_ROOT"]), task_internet=task_internet)


def probe(model_config, output):
    """Host-side tool roundtrip with a private temporary HOME; credentials stay in memory."""
    os.umask(0o077)
    receipt = {"schema_version": 1, "status": "failed", "wire": "claude-stream-json",
               "probe_scope": "local-pinned-Claude-Code-tool-roundtrip-with-isolated-home",
               "remote_worker_validation": "required-at-each-run-start",
               "started_at": datetime.now(timezone.utc).isoformat()}
    started, stage = time.monotonic(), "configuration"
    try:
        model = load_model_config(model_config)
        runtime = load_agent_runtime(Path(model["SFORGE_AGENT_RUNTIME_DIR"]))
        require_local_profile(model, runtime)
        token = _token(model["CLAUDE_OAUTH_TOKEN_FILE"])
        config_digest = hashlib.sha256(Path(model_config).read_bytes()).hexdigest()
        receipt.update(model=model["MODEL"], reasoning_effort=model["REASONING_EFFORT"], agent_auth_mode=TOKEN_MODE,
                       runtime_version=runtime.version, runtime_tree_sha256=runtime.tree_sha256,
                       model_config_sha256=config_digest)
        stage = "private-runtime-setup"
        with tempfile.TemporaryDirectory(prefix="sforge-local-claude-probe-") as directory:
            home = Path(directory)
            state = home / "submit-wait"
            state.mkdir(mode=0o700)
            (home / ".claude").mkdir(mode=0o700)
            submit_hook, claude_hook = home / "submit-wait-hook.py", home / "claude-hook.py"
            submit_hook.write_text(generate_submit_wait_hook_script())
            claude_hook.write_text(generate_claude_hook_script().replace(repr(SUBMIT_WAIT_HOOK_PATH), repr(str(submit_hook))))
            settings = home / ".claude/settings.json"
            settings.write_text(settings_document(model, 240, stop_hook=False).replace(CLAUDE_HOOK_PATH, str(claude_hook)))
            environment = {key: value for key, value in os.environ.items()
                           if key in {"PATH", "LANG", "LC_ALL", "USER", "LOGNAME", "SHELL", "TZ",
                                      "SSL_CERT_FILE", "SSL_CERT_DIR", "HTTPS_PROXY", "HTTP_PROXY", "NO_PROXY",
                                      "https_proxy", "http_proxy", "no_proxy"}}
            environment.update(HOME=str(home), TMPDIR=str(home), SFORGE_SUBMIT_WAIT_STATE_DIR=str(state),
                               SFORGE_SUBMIT_TOOL_TIMEOUT_MS="180000", CLAUDE_CODE_OAUTH_TOKEN=token,
                               CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="1")
            environment["PATH"] = str(runtime.root / "bin") + os.pathsep + environment.get("PATH", os.defpath)
            executable = str(runtime.root / "bin" / "claude")
            version = subprocess.run([executable, "--version"], env=environment, capture_output=True,
                                     text=True, timeout=60, check=True)
            if not version.stdout.strip().startswith(CLAUDE_RUNTIME_VERSION):
                raise RuntimeError("Actual Claude Code executable differs from its pinned runtime marker")
            marker = "SFORGE_LOCAL_MODEL_PROBE_" + uuid.uuid4().hex
            prompt = ("This is a tool completion test. FIRST call the Bash tool with exactly: exit 3. "
                      "That command is intentionally expected to fail. After its tool result returns, "
                      "make a SECOND, separate Bash tool call with exactly: printf '%s\\n' '" + marker + "'. "
                      "Confirm its output matches, then reply with exactly READY. "
                      "Do not run other commands, read files, change files, or access resources.")
            command = [executable, "-p", "--output-format", "stream-json", "--verbose", "--model", model["MODEL"],
                       *cli_flags(model, web_search=True, settings_path=str(settings)),
                       "--dangerously-skip-permissions", prompt]
            stage = "cli-tool-roundtrip"
            process = subprocess.Popen(command, cwd=home, env=environment, stdin=subprocess.DEVNULL,
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                       start_new_session=True)
            try:
                stdout, _stderr = process.communicate(timeout=300)
            except BaseException:
                os.killpg(process.pid, signal.SIGKILL)
                process.communicate(timeout=10)
                raise
            receipt.update(validate_stream(stdout, process.returncode, model["MODEL"], marker))
            stage = "tool-hook-completion"
            proof = json.loads((state / "state.json").read_text())
            if proof.get("schema_version") != 1 or proof.get("active") is not None or proof.get("active_tools") != {}:
                raise RuntimeError("Submission hook did not prove ordinary command completion")
            receipt["tool_hook_completion_verified"] = True
        stage = "input-identity-recheck"
        if hashlib.sha256(Path(model_config).read_bytes()).hexdigest() != config_digest:
            raise RuntimeError("Model configuration changed during the qualification")
        if _token(model["CLAUDE_OAUTH_TOKEN_FILE"]) != token:
            raise RuntimeError("Original credential file changed during the isolated qualification")
        receipt.update(status="passed", auth_source_unchanged=True, temporary_home_removed=True)
    except Exception as exc:
        receipt.update(error_type=type(exc).__name__, error=str(exc)[:500], failed_stage=stage)
    receipt["elapsed_seconds"] = round(time.monotonic() - started, 3)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = output.with_name(output.name + ".tmp-" + uuid.uuid4().hex)
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        stream.write(json.dumps(receipt, indent=2) + "\n")
    os.replace(temporary, output)
    print(json.dumps(receipt))
    return 0 if receipt["status"] == "passed" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    raise SystemExit(probe(arguments.model_config, arguments.output))
