from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
import uuid

import pytest

from sforge.harness.backend.docker_backend import DockerBackend
from sforge.harness.container_files import copy_container_file


def test_umask_077_runtime_files_are_usable_by_agent_in_real_container(
    tmp_path: Path,
) -> None:
    image = os.environ.get("SFORGE_RUNTIME_PERMISSION_TEST_IMAGE")
    if not image:
        pytest.skip("set SFORGE_RUNTIME_PERMISSION_TEST_IMAGE for Docker integration")

    previous = os.umask(0o077)
    try:
        prompt = tmp_path / "prompt.md"
        prompt.write_text("authoritative-prompt\n", encoding="utf-8")
        submit = tmp_path / "sforge-submit"
        submit.write_text("#!/bin/sh\nprintf 'submit-ok\\n'\n", encoding="utf-8")
        hooks = tmp_path / "hooks.json"
        hooks.write_text('{"hooks":{}}\n', encoding="utf-8")
    finally:
        os.umask(previous)
    assert all(path.stat().st_mode & 0o777 == 0o600 for path in (prompt, submit, hooks))

    backend = DockerBackend()
    name = f"sforge-runtime-permissions-{uuid.uuid4().hex[:12]}"
    handle = backend.create_container(image, name)
    try:
        backend.start_container(handle)
        copy_container_file(
            backend,
            handle,
            prompt,
            PurePosixPath("/tmp/agent_prompt.md"),
            owner="agent:agent",
            mode=0o600,
            description="agent prompt",
        )
        copy_container_file(
            backend,
            handle,
            submit,
            PurePosixPath("/usr/local/bin/sforge-submit"),
            owner="root:root",
            mode=0o755,
            description="sforge-submit",
        )
        copy_container_file(
            backend,
            handle,
            hooks,
            PurePosixPath("/tmp/hooks.json"),
            owner="root:root",
            mode=0o644,
            description="hooks JSON",
        )

        result = backend.exec_run(
            handle,
            [
                "/bin/sh",
                "-c",
                (
                    "test \"$(cat /tmp/agent_prompt.md)\" = authoritative-prompt "
                    "&& test \"$(sforge-submit)\" = submit-ok "
                    "&& test \"$(cat /tmp/hooks.json)\" = '{\"hooks\":{}}' "
                    "&& test \"$(stat -c '%U:%G:%a' /tmp/agent_prompt.md)\" "
                    "= agent:agent:600 "
                    "&& test \"$(stat -c '%U:%G:%a' /usr/local/bin/sforge-submit)\" "
                    "= root:root:755 "
                    "&& test \"$(stat -c '%U:%G:%a' /tmp/hooks.json)\" "
                    "= root:root:644"
                ),
            ],
            user="agent",
        )
        assert result.exit_code == 0, result.output
    finally:
        backend.cleanup_container(handle)
