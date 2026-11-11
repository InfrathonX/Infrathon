"""Claude reuses the submission barrier, including its failure and race cases."""
import datetime
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

import pytest

from sforge.harness.agent import claude_agent as CLAUDE
from sforge.harness import agent_config
from sforge.harness import submit_wait


MODEL = {"MODEL": "claude-opus-5-5", "MODEL_API_BASE_URL": "https://model.example.test",
         "REASONING_EFFORT": "max"}


@pytest.fixture
def hook(tmp_path, monkeypatch):
    engine, adapter = tmp_path / "engine.py", tmp_path / "claude-hook.py"
    engine.write_text(submit_wait.generate_submit_wait_hook_script())
    monkeypatch.setattr(submit_wait, "SUBMIT_WAIT_HOOK_PATH", str(engine))
    adapter.write_text(CLAUDE.generate_claude_hook_script())
    binary = tmp_path / "bin"
    binary.mkdir()
    script = binary / "sforge-submit"
    script.write_text("#!" + sys.executable + "\nimport os,pathlib,time\n"
                      "marker=os.environ.get('SUBMIT_MARKER')\n"
                      "if marker:pathlib.Path(marker).write_text('started')\n"
                      "time.sleep(float(os.environ.get('SUBMIT_SLEEP','0')))\n"
                      "print('Score: 180.0')\nraise SystemExit(int(os.environ.get('SUBMIT_EXIT','0')))\n")
    script.chmod(0o755)
    env = {**os.environ, "PATH": str(binary) + os.pathsep + os.environ["PATH"],
           "SFORGE_SUBMIT_WAIT_STATE_DIR": str(tmp_path / "gate")}
    for key in ("SFORGE_SUBMIT_TOOL_TIMEOUT_MS", "SFORGE_AGENT_DEADLINE_AT",
                "SFORGE_AGENT_DEADLINE_MONOTONIC", "SFORGE_REMAINING_SECONDS"):
        env.pop(key, None)

    def run(phase, event, **changes):
        result = subprocess.run([sys.executable, "-I", str(adapter), phase], input=json.dumps(event),
            text=True, capture_output=True, env={**env, **changes}, timeout=10)
        assert result.returncode == 0, result.stderr
        assert not result.stderr
        return json.loads(result.stdout)

    run.script, run.engine, run.env, run.binary = adapter, engine, env, binary
    run.state = Path(env["SFORGE_SUBMIT_WAIT_STATE_DIR"])
    return run


def event(command="sforge-submit", identifier="toolu_submit", tool="Bash"):
    return {"hook_event_name": "PreToolUse", "session_id": "claude-session-1",
            "tool_use_id": identifier, "cwd": str(Path.cwd()),
            "tool_name": tool, "tool_input": {"command": command}}


def state(hook):
    return json.loads((hook.state / "state.json").read_text())


def decision(record):
    return record.get("hookSpecificOutput", {}).get("permissionDecision")


def test_ordinary_failed_bash_releases_pending_before_submit(hook):
    failed = event("false", "toolu_failed")
    assert hook("pre", failed) == {}
    assert state(hook)["active_tools"]
    assert hook("failure", {**failed, "hook_event_name": "PostToolUseFailure", "error": "exit 1"}) == {}
    assert state(hook)["active_tools"] == {}
    assert decision(hook("pre", event())) == "allow"


def test_only_matching_failed_tool_can_release_ordinary_state(hook):
    hook("pre", event("false", "original"))
    hook("failure", {**event("false", "unrelated"), "error": "exit 1"})
    assert state(hook)["active_tools"]
    assert decision(hook("pre", event())) == "deny"


@pytest.mark.parametrize("exit_code", [0, 17])
def test_submit_replays_terminal_result_and_success_or_failure_clears_state(hook, exit_code):
    submitted = event()
    reply = hook("pre", submitted, SUBMIT_EXIT=str(exit_code))
    rewritten = reply["hookSpecificOutput"]["updatedInput"]
    replay = subprocess.run(shlex.split(rewritten["command"]), env=hook.env,
                            text=True, capture_output=True, timeout=5)
    assert replay.returncode == exit_code
    assert replay.stdout == "Score: 180.0\n"
    completed = {**submitted, "tool_input": rewritten}
    if exit_code:
        completed.update(hook_event_name="PostToolUseFailure", error="Command exited 17")
        hook("failure", completed)
    else:
        completed.update(hook_event_name="PostToolUse", tool_response={"stdout": replay.stdout})
        hook("post", completed)
    assert state(hook)["active"] is None
    assert state(hook)["last_completion"]["outcome"] == ("failed" if exit_code else "completed")


def test_submit_pre_hook_stays_blocked_until_result_and_denies_parallel_edit(hook, tmp_path):
    marker = tmp_path / "started"
    process = subprocess.Popen([sys.executable, "-I", str(hook.script), "pre"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        env={**hook.env, "SUBMIT_SLEEP": "0.7", "SUBMIT_MARKER": str(marker)})
    try:
        process.stdin.write(json.dumps(event()))
        process.stdin.close()
        deadline = time.monotonic() + 4
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert marker.exists() and process.poll() is None
        assert decision(hook("pre", event("new content", "edit", "Edit"))) == "deny"
        assert decision(hook("pre", event("sforge-save-version", "save"))) == "deny"
        process.wait(timeout=4)
        reply = json.loads(process.stdout.read())
        assert decision(reply) == "allow"
        assert state(hook)["active"]["result_ready"] is True
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=3)


@pytest.mark.parametrize("tool,inputs", [("Agent", {}), ("Task", {}),
    ("Bash", {"command": "sforge-submit", "run_in_background": True}),
    ("Bash", {"command": "sforge-submit &"})])
def test_background_and_subagent_bypass_rejected(hook, tool, inputs):
    value = {**event(tool=tool), "tool_input": inputs}
    assert decision(hook("pre", value)) == "deny"


def test_actual_background_id_does_not_clear_pending_tool(hook):
    background = event("sleep 100", "background")
    hook("pre", background)
    hook("post", {**background, "tool_response": {"backgroundTaskId": "bg-1", "stdout": "running"}})
    assert state(hook)["active_tools"]
    assert decision(hook("pre", event())) == "deny"


@pytest.mark.parametrize("age_seconds,live,expected", [(901, False, "allow"),
    (899, False, "deny"), (901, True, "deny")])
def test_claude_background_id_recovery_requires_age_and_process_exit(
    hook, tmp_path, age_seconds, live, expected,
):
    background = event("ncu python prof.py", "profile")
    hook("pre", background)
    hook("post", {**background, "tool_response": {"backgroundTaskId": "bg-profile", "stdout": "running"}})
    value = state(hook)
    started = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=age_seconds)
    for tool in value["active_tools"].values():
        tool["started_at"] = started.isoformat(timespec="milliseconds").replace("+00:00", "Z")
    (hook.state / "state.json").write_text(json.dumps(value))
    proc = tmp_path / "proc"
    proc.mkdir()
    if live:
        process = proc / "123456789"
        process.mkdir()
        uid = os.getuid()
        (process / "status").write_text(f"State:\tS\nUid:\t{uid}\t{uid}\t{uid}\t{uid}\n")
    assert decision(hook("pre", event(), SFORGE_SUBMIT_WAIT_PROC_ROOT=str(proc))) == expected
    if expected == "allow":
        assert state(hook)["active_tools"] == {}
        assert state(hook)["active"]["tool_use_id"] == "toolu_submit"
    else:
        assert len(state(hook)["active_tools"]) == 1 and state(hook)["active"] is None


def test_finished_log_printing_old_running_text_does_not_block_submit(hook):
    finished = event("cat old.log", "cat")
    hook("pre", finished)
    hook("post", {**finished, "tool_response": {"stdout": "Process running with session ID 111"}})
    assert state(hook)["active_tools"] == {}
    assert decision(hook("pre", event())) == "allow"


def test_cli_max_and_continue_keep_settings_hooks_and_disable_other_agents():
    command = CLAUDE.claude_command(MODEL, "Continue working", resume="continue")
    assert command[command.index("--effort") + 1] == "max"
    assert "--continue" in command and "--resume" not in command
    assert "--bare" not in command and "--safe-mode" not in command
    assert command[command.index("--tools") + 1] == "Bash,Read,Edit,Write,Glob,Grep,WebSearch,WebFetch"
    offline = CLAUDE.claude_command(MODEL, "continue", web_search=False)
    assert offline[offline.index("--tools") + 1] == "Bash,Read,Edit,Write,Glob,Grep"
    probe = CLAUDE.claude_command(MODEL, "READY", probe=True)
    assert probe[probe.index("--tools") + 1] == ""
    assert command[command.index("--disallowedTools") + 1] == "Agent,Task,mcp__*"
    assert CLAUDE.claude_command(MODEL, "continue", resume="session-1")[-3:] == ["--resume", "session-1", "continue"]


def test_claude_failure_hook_is_configured_and_submission_hook_is_synchronous():
    settings = CLAUDE.claude_settings(MODEL, submit_wait_timeout_seconds=3700)
    assert set(settings["hooks"]) == {"PreToolUse", "PostToolUse", "PostToolUseFailure"}
    hook = settings["hooks"]["PreToolUse"][0]["hooks"][0]
    assert hook["timeout"] == 3700 and hook.get("async") is not True
    assert settings["hooks"]["PostToolUseFailure"][0]["hooks"][0]["command"].endswith(" failure")
    assert "effortLevel" not in settings  # CLI/env accept max; settings reject it.
    assert settings["env"]["CLAUDE_CODE_EFFORT_LEVEL"] == "max"


def test_bad_hook_timeout_rejected():
    with pytest.raises(ValueError):
        CLAUDE.claude_settings(MODEL, submit_wait_timeout_seconds=1)


def test_common_dispatch_selects_pinned_claude_and_keeps_secret_out_of_settings(tmp_path):
    runtime = tmp_path / "runtime"
    (runtime / "bin").mkdir(parents=True)
    (runtime / "bin/claude").write_text("#!/bin/sh\n")
    (runtime / "env.sh").write_text("export PATH=/opt/agent-runtime/bin:$PATH\n")
    (runtime / ".sforge-agent-runtime.json").write_text(json.dumps({
        "schema_version": 1, "kind": "sforge-agent-runtime", "agent": "claude-code",
        "cli": "claude", "version": "2.1.287", "node_version": "v20.18.0",
        "tree_sha256": "a" * 64}))
    key = tmp_path / "private.key"
    key.write_text("fake-test-provider-key")
    key.chmod(0o600)
    configuration = tmp_path / "model.env"
    configuration.write_text("AGENT=claude-code\nMODEL=claude-opus-5-5\n"
        "MODEL_API_BASE_URL=https://model.example.test\nMODEL_API_KEY_FILE=" + str(key)
        + "\nSFORGE_AGENT_RUNTIME_DIR=" + str(runtime) + "\nREASONING_EFFORT=max\n")
    model = agent_config.load_model_config(configuration)
    credentials = agent_config.load_credentials(model)
    assert model["AGENT"] == "claude-code" and model["runtime_version"] == "2.1.287"
    assert credentials.mode == "anthropic-api-key"
    assert credentials.payload == b"fake-test-provider-key"
    assert "fake-test-provider-key" not in repr(credentials)
    assert "fake-test-provider-key" not in json.dumps(CLAUDE.claude_settings(model))
    marker = runtime / ".sforge-agent-runtime.json"
    changed = json.loads(marker.read_text())
    changed["agent"] = "codex"
    marker.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="claude-code runtime"):
        agent_config.load_model_config(configuration)
