from __future__ import annotations

import os
import time
import uuid

import pytest

from sforge.harness.backend import docker_backend
from sforge.harness.backend.docker_backend import DockerBackend
from sforge.harness.run_agent import (
    CODEX_PROCESS_GROUP_FILE,
    _agent_exec_command,
)


pytestmark = pytest.mark.skipif(
    os.environ.get("SFORGE_RUN_DOCKER_TESTS") != "1"
    or not os.environ.get("SFORGE_DOCKER_TEST_IMAGE"),
    reason=(
        "set SFORGE_RUN_DOCKER_TESTS=1 and SFORGE_DOCKER_TEST_IMAGE "
        "to run Docker integration tests"
    ),
)


def test_timeout_stops_a_term_ignoring_descendant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = os.environ["SFORGE_DOCKER_TEST_IMAGE"]
    backend = DockerBackend()
    name = f"sforge-process-group-test-{uuid.uuid4().hex[:10]}"
    handle = backend.create_container(
        image,
        name,
        command="tail -f /dev/null",
    )
    backend.start_container(handle)
    monkeypatch.setattr(
        docker_backend, "EXEC_TERM_GRACE_SECONDS", 0.2
    )
    monkeypatch.setattr(
        docker_backend, "EXEC_KILL_GRACE_SECONDS", 2.0
    )
    try:
        result = backend.exec_run_with_timeout(
            handle,
            _agent_exec_command(
                type("Codex", (), {"name": "codex"})(),
                (
                    "echo \"$$ $(ps -o pgid= -p $$ | xargs)\" "
                    "> /tmp/sforge-process-ids; "
                    "trap '' TERM; "
                    "(trap '' TERM; "
                    "while true; do "
                    "printf x >> /tmp/sforge-sentinel; "
                    "sleep 0.05; "
                    "done) & "
                    "wait"
                ),
            ),
            timeout=1,
            environment={
                "SFORGE_EXEC_PROCESS_GROUP_FILE": (
                    CODEX_PROCESS_GROUP_FILE
                )
            },
        )
        assert result.timed_out is True

        first_size = int(
            backend.exec_run(
                handle,
                ["/bin/sh", "-c", "wc -c < /tmp/sforge-sentinel"],
            ).output.strip()
        )
        time.sleep(0.25)
        second_size = int(
            backend.exec_run(
                handle,
                ["/bin/sh", "-c", "wc -c < /tmp/sforge-sentinel"],
            ).output.strip()
        )
        if second_size != first_size:
            ids = backend.exec_run(
                handle,
                ["/bin/sh", "-c", "cat /tmp/sforge-process-ids"],
            ).output.strip()
            processes = backend.exec_run(
                handle,
                [
                    "/bin/sh",
                    "-c",
                    "ps -eo pid,ppid,pgid,sid,stat,args",
                ],
            ).output
            pytest.fail(
                f"sentinel kept growing ({first_size}->{second_size}); "
                f"shell/pgid={ids}\n{processes}"
            )
    finally:
        backend.cleanup_container(handle)
