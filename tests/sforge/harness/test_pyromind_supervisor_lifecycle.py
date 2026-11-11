"""Execute the generated supervisor against isolated files and fake processes.

The fixture never invokes a command or sends a signal. It exercises the actual
generated control flow, status publication, snapshot creation and resume budget.
"""
from __future__ import annotations

import ast
import hashlib
import json
import signal
import subprocess
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from sforge.pyromind.runner import supervisor_source


class _Clock:
    def __init__(self):
        self.value = 0.0

    def time(self):
        return 10_000 + self.value

    def monotonic(self):
        return self.value

    def sleep(self, seconds):
        assert seconds >= 0
        self.value += seconds


class _Lifecycle:
    def __init__(self, root, scenario, *, tar_fails=False, original_remaining=None,
                 retirement_seconds=0, budget_seconds=30, transcript_noise=b""):
        self.root = root
        self.scenario = scenario
        self.tar_fails = tar_fails
        self.retirement_seconds = retirement_seconds
        self.transcript_noise = transcript_noise
        self.clock = _Clock()
        self.events = []
        self.invocations = []
        self.retirements = []
        self.proc = root / "proc"
        self.proc.mkdir()
        self.runtime = root / "runtime"
        self.runtime.mkdir()
        self.workspace = root / "workspace"
        self.workspace.mkdir()
        (self.workspace / "candidate.py").write_text("answer = 42\n")
        self.wait = root / "wait.json"
        self.set_wait()
        self.key = root / "model.key"
        self.key.write_text("fixture-model-key")
        self.config = {
            "seconds": budget_seconds,
            "auth_mode": "api-key",
            "cwd": str(self.workspace),
            "environment": {},
            "first_command": "FIRST",
            "resume_command": "RESUME",
            "submit_paths": ["candidate.py"],
            "submit_exclude": [],
        }
        if original_remaining is not None:
            self.config["deadline_at"] = self.clock.time() + original_remaining
        (self.runtime / "supervisor.json").write_text(json.dumps(self.config))

    def set_wait(self, *, active=None, ordinary=False):
        self.wait.write_text(json.dumps({
            "schema_version": 1,
            "active": active,
            "active_tools": {"ordinary": {
                "tool_use_id": "ordinary-call", "session_id": "session-fixture",
                "tool_name": "Bash", "started_at": "2026-10-03T00:00:00.000Z",
            }} if ordinary else {},
            "last_completion": None,
        }))

    def path(self, first, *rest):
        mapping = {
            "/run/sforge-agent": self.runtime,
            "/run/sforge-submit-wait/state.json": self.wait,
            "/proc": self.proc,
            "/home/agent/.config/sforge/model.key": self.key,
        }
        return Path(mapping.get(str(first), first), *rest)

    def add_writer(self):
        writer = self.proc / "8123"
        writer.mkdir(exist_ok=True)
        (writer / "status").write_text("Name:\tfixture-writer\nState:\tS\nUid:\t32100\t32100\t32100\t32100\n")

    def kill(self, pid, sig):
        assert pid == 8123, "Only the isolated fake writer may receive a signal"
        self.events.append(("kill", pid, sig))
        writer = self.proc / str(pid)
        if writer.exists():
            (writer / "status").unlink()
            writer.rmdir()

    def killpg(self, *args):
        raise AssertionError("The fake CLI already exited; no group signal is expected")

    def popen(self, command, **kwargs):
        index = len(self.invocations) + 1
        self.events.append(("invoke", index))
        self.invocations.append({"command": command, "environment": dict(kwargs["env"]), "at": self.clock.time()})
        event = {"type": "item.completed", "item": {
            "type": "command_execution", "status": "failed", "exit_code": 1,
        }}
        if self.scenario == "empty-nonzero":
            event = {"type": "turn.failed", "error": {"message": "fixture CLI could not start"}}
        kwargs["stdout"].write(self.transcript_noise)
        kwargs["stdout"].write((json.dumps(event) + "\n").encode())
        if self.scenario in {"repeated-nonzero", "empty-nonzero"}:
            if index <= 3:
                code = 1
            else:
                self.clock.value = self.config.get("deadline_at", 10_000 + self.config["seconds"]) - 10_000
                code = 0
            return SimpleNamespace(pid=9999 + index, returncode=code, poll=lambda: code)
        if index == 1:
            if self.scenario in {"ordinary", "unreadable-writer"}:
                self.set_wait(ordinary=True)
                self.add_writer()
                if self.scenario == "unreadable-writer":
                    (self.proc / "8123" / "status").write_text("Name:\tfixture-writer\nState:\tS\n")
            elif self.scenario == "submission":
                self.set_wait(active={"tool_use_id": "pending-submit", "result_ready": False})
            elif self.scenario == "reconcile-error":
                self.set_wait(ordinary=True)
            code = 1 if self.scenario == "nonzero" else 0
        else:
            # Reach the original absolute deadline on the resumed invocation.
            self.clock.value = self.config.get("deadline_at", 10_000 + self.config["seconds"]) - 10_000
            code = 0
        return SimpleNamespace(pid=9999 + index, returncode=code, poll=lambda: code)

    def run(self, command, **kwargs):
        self.events.append(("run", tuple(command)))
        if command[-1] == "reconcile":
            code = 1 if self.scenario == "reconcile-error" and self.invocations else 0
            return SimpleNamespace(returncode=code, stdout=b"{}", stderr=b"")
        if "retire-ordinary-at-boundary" in command:
            assert command[0] == "python3" and "runuser" not in command
            assert command[-1] == "32100"
            assert not list(self.proc.iterdir()), "Retirement must follow whole-UID writer termination"
            pending = json.loads(self.wait.read_text())
            assert pending["active"] is None, "An unfinished submission can never be retired"
            retired = [{**value, "outcome": "cancelled_at_writer_boundary"}
                       for value in pending["active_tools"].values()]
            self.set_wait()
            result = {"schema_version": 1, "outcome": "cancelled_at_writer_boundary",
                      "retired_tools": retired, "agent_uid": 32100}
            self.retirements.append(result)
            self.clock.value += self.retirement_seconds
            return SimpleNamespace(returncode=0, stdout=json.dumps(result).encode(), stderr=b"")
        if command[0] == "chown":
            return SimpleNamespace(returncode=0)
        if command[:2] == ["tar", "czf"]:
            if self.tar_fails:
                raise subprocess.CalledProcessError(2, command)
            with tarfile.open(command[2], "w:gz") as archive:
                archive.add(self.workspace / "candidate.py", arcname="candidate.py")
            return SimpleNamespace(returncode=0)
        raise AssertionError("Unexpected subprocess command: " + repr(command))

    def execute(self):
        # Remove imports, then supply sandboxed modules. Definitions and the main
        # try/except/finally are the exact generated code, not a test replica.
        tree = ast.parse(supervisor_source())
        tree.body = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
        namespace = {
            "json": json, "hashlib": hashlib, "signal": signal, "time": self.clock,
            "pathlib": SimpleNamespace(Path=self.path),
            "pwd": SimpleNamespace(getpwnam=lambda name: SimpleNamespace(pw_uid=32100)),
            "os": SimpleNamespace(environ={}, kill=self.kill, killpg=self.killpg, geteuid=lambda: 0),
            "subprocess": SimpleNamespace(run=self.run, Popen=self.popen, PIPE=-1, STDOUT=-2,
                                          TimeoutExpired=subprocess.TimeoutExpired),
        }
        exec(compile(tree, "<isolated-generated-supervisor>", "exec"), namespace)
        self.namespace = namespace
        return json.loads((self.runtime / "status.json").read_text())


def test_exited_cli_cancels_detached_ordinary_tool_then_resumes_with_same_deadline(tmp_path):
    fixture = _Lifecycle(tmp_path, "ordinary")
    status = fixture.execute()
    assert status["status"] == "completed"
    assert len(fixture.invocations) == 2
    assert fixture.invocations[1]["command"][-1] == "RESUME"
    assert fixture.retirements[0]["outcome"] == "cancelled_at_writer_boundary"
    assert status["ordinary_tool_recoveries"] == 1
    assert status["last_tool_recovery"]["outcome"] == "cancelled_at_writer_boundary"
    assert status["deadline_at"] == 10_030
    assert fixture.invocations[1]["environment"]["SFORGE_AGENT_DEADLINE_MONOTONIC"] == "30.0"
    assert int(fixture.invocations[1]["environment"]["SFORGE_REMAINING_SECONDS"]) < 30
    order = [event[0] for event in fixture.events]
    assert order.index("kill") < order.index("invoke", order.index("invoke") + 1)
    assert status["no_agent_processes"] is True
    archive = fixture.runtime / "final.tar.gz"
    assert archive.is_file()
    assert status["candidate_archive_available"] is True
    assert status["candidate_archive_sha256"] == hashlib.sha256(archive.read_bytes()).hexdigest()
    assert status["candidate_archive_bytes"] == archive.stat().st_size
    assert not (fixture.runtime / "final.tmp.tar.gz").exists()


def test_unfinished_submission_is_never_cancelled_or_resumed_but_candidate_is_preserved(tmp_path):
    fixture = _Lifecycle(tmp_path, "submission")
    status = fixture.execute()
    assert status["status"] == "failed"
    assert "unfinished submission" in status["error"]
    assert len(fixture.invocations) == 1
    assert fixture.retirements == []
    assert json.loads(fixture.wait.read_text())["active"]["tool_use_id"] == "pending-submit"
    assert status["no_agent_processes"] is True
    assert (fixture.runtime / "final.tar.gz").is_file()
    with tarfile.open(fixture.runtime / "final.tar.gz") as archive:
        assert archive.extractfile("candidate.py").read() == b"answer = 42\n"


def test_failed_snapshot_preserves_original_supervisor_error_and_reports_capture_failure(tmp_path):
    fixture = _Lifecycle(tmp_path, "submission", tar_fails=True)
    status = fixture.execute()
    assert status["status"] == "failed"
    assert "unfinished submission" in status["error"]
    assert status["candidate_preservation_error"] == "CalledProcessError"
    assert not (fixture.runtime / "final.tar.gz").exists()
    assert fixture.retirements == []


def test_one_nonzero_cli_exit_with_tool_work_does_not_abort_the_whole_budget(tmp_path):
    fixture = _Lifecycle(tmp_path, "nonzero")
    status = fixture.execute()
    assert len(fixture.invocations) == 2
    assert status["status"] == "completed"
    assert status["attempts"] == 2
    assert status["deadline_at"] == 10_030


def test_blocked_command_stderr_with_numeric_patch_line_does_not_abort_resume(tmp_path):
    # Codex logs a rejected compound command verbatim on stderr. This patch
    # deletion was parsed as JSON integer -46 in the GLM production failure.
    noise = (b"ERROR codex_core::tools::router: Command blocked by PreToolUse hook\n"
             b"*** Update File: FINAL_VERSION\n@@\n-46\n+52\n"
             b"*** End Patch\nsforge-save-version --label v52\nsforge-submit\n")
    fixture = _Lifecycle(tmp_path, "nonzero", transcript_noise=noise)
    status = fixture.execute()
    assert status["status"] == "completed"
    assert len(fixture.invocations) == 2
    assert fixture.invocations[1]["command"][-1] == "RESUME"
    assert status["tool_turns"] == 2
    assert status["deadline_at"] == 10_030
    assert status["candidate_archive_available"] is True


@pytest.mark.parametrize("record", [
    -46, 52, 1.5, True, False, None, [], "plain stderr", {},
    {"item": 1}, {"item": "not an event"},
    {"type": "assistant", "message": 1},
    {"type": "assistant", "message": {"content": 1}},
    {"type": "assistant", "message": {"content": None}},
])
def test_unexpected_json_log_shapes_do_not_abort_the_agent_budget(tmp_path, record):
    fixture = _Lifecycle(tmp_path, "nonzero",
                         transcript_noise=(json.dumps(record) + "\n").encode())
    status = fixture.execute()
    assert status["status"] == "completed"
    assert status["attempts"] == 2
    assert status["tool_turns"] == 2
    assert status["deadline_at"] == 10_030
    assert status["candidate_archive_available"] is True


def test_writer_boundary_recovery_does_not_replace_parent_absolute_budget(tmp_path):
    fixture = _Lifecycle(tmp_path, "ordinary", original_remaining=12)
    status = fixture.execute()
    assert status["deadline_at"] == 10_012
    assert status["budget_seconds"] == 30
    assert status["remaining_seconds_at_start"] == 12
    assert len(fixture.invocations) == 2
    assert fixture.invocations[1]["environment"]["SFORGE_AGENT_DEADLINE_MONOTONIC"] == "12.0"
    assert int(fixture.invocations[1]["environment"]["SFORGE_REMAINING_SECONDS"]) < 12


def test_invalid_reconciliation_never_retires_tools_and_preserves_failure_snapshot(tmp_path):
    fixture = _Lifecycle(tmp_path, "reconcile-error")
    status = fixture.execute()
    assert status["status"] == "failed"
    assert "Invalid submission wait state" in status["error"]
    assert len(fixture.invocations) == 1
    assert fixture.retirements == []
    assert (fixture.runtime / "final.tar.gz").is_file()


def test_incomplete_proc_identity_cannot_authorize_tool_retirement_or_snapshot(tmp_path):
    fixture = _Lifecycle(tmp_path, "unreadable-writer")
    status = fixture.execute()
    assert status["status"] == "failed"
    assert status["candidate_preservation_error"] == "StopIteration"
    assert len(fixture.invocations) == 1
    assert fixture.retirements == []
    assert status.get("no_agent_processes") is not True
    assert status.get("candidate_archive_available") is not True
    assert not (fixture.runtime / "final.tar.gz").exists()


def test_recovery_work_cannot_admit_another_cli_after_the_original_deadline(tmp_path):
    fixture = _Lifecycle(tmp_path, "ordinary", retirement_seconds=30)
    status = fixture.execute()
    assert len(fixture.invocations) == 1
    assert status["deadline_at"] == 10_030
    assert status["status"] == "completed"
    assert fixture.retirements[0]["outcome"] == "cancelled_at_writer_boundary"
    assert status["candidate_archive_available"] is True


def test_repeated_short_nonzero_cli_exits_with_real_tool_work_are_not_startup_failure(tmp_path):
    fixture = _Lifecycle(tmp_path, "repeated-nonzero")
    status = fixture.execute()
    assert len(fixture.invocations) == 4
    assert status["status"] == "completed"
    assert status["attempts"] == 4
    assert status["tool_turns"] == 4
    assert status["deadline_at"] == 10_030
    assert all(invocation["environment"]["SFORGE_AGENT_DEADLINE_MONOTONIC"] == "30.0"
               for invocation in fixture.invocations)
    assert int(fixture.invocations[-1]["environment"]["SFORGE_REMAINING_SECONDS"]) == 15
    assert status["candidate_archive_available"] is True


def test_repeated_short_nonzero_cli_exits_without_tool_work_remain_startup_failure(tmp_path):
    fixture = _Lifecycle(tmp_path, "empty-nonzero", budget_seconds=300)
    status = fixture.execute()
    assert len(fixture.invocations) == 3
    assert status["status"] == "failed"
    assert "repeatedly failed before doing work" in status["error"]
    assert status["deadline_at"] == 10_300
    assert status.get("tool_turns", 0) == 0
    assert status["candidate_archive_available"] is True
    assert status["no_agent_processes"] is True
