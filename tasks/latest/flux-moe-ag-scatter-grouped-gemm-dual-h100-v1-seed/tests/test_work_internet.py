"""Work egress follows the task JSON's ``internet`` flag (docs/INTERNET_AND_WEB_SEARCH.md).

The launcher always passes --enable-internet (SForge's internet=false path
needs sudo iptables) and keeps Work on an internal Docker network behind the
borrowed llada sidecar; the task flag reaches the entry as
MLSYSFORGE_WORK_INTERNET and the sidecar as --allow-public-egress.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from conftest import TASK_ROOT, load_module


@pytest.mark.parametrize(
    "flag,web_search,offline_prompt",
    [(None, "live", False), ("true", "live", False), ("false", "disabled", True)],
)
def test_codex_and_prompt_follow_the_task_internet_flag(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    flag: str | None,
    web_search: str,
    offline_prompt: bool,
) -> None:
    from sforge.harness.config import SForgeConfig

    entry = load_module("flux_dual_entry_internet_test", "scripts/sforge_dual_h100_entry.py")
    if flag is None:
        monkeypatch.delenv(entry.WORK_INTERNET_ENV, raising=False)
    else:
        monkeypatch.setenv(entry.WORK_INTERNET_ENV, flag)
    assert entry.CodexAgent.format_run_cmd is entry.task_internet_format_run_cmd
    assert entry.run_agent.generate_evolve_prompt is entry.task_internet_generate_prompt
    agent = entry.CodexAgent(SForgeConfig(codex_auth_json_path=tmp_path / "auth.json"))
    command = entry.task_internet_format_run_cmd(agent, "/tmp/prompt.md", internet=True)
    assert f'web_search="{web_search}"' in command
    prompt = entry.task_internet_generate_prompt("Optimize.", submit_paths=["solution"], internet=True)
    assert ("NO internet access" in prompt) is offline_prompt
    monkeypatch.setenv(entry.WORK_INTERNET_ENV, "1")
    with pytest.raises(RuntimeError, match=entry.WORK_INTERNET_ENV):
        entry.task_internet_format_run_cmd(agent, "/tmp/prompt.md", internet=True)


def test_launcher_wires_the_task_internet_flag() -> None:
    source = (TASK_ROOT / "scripts/launch_fixed_budget_after_gpu_pair_idle.sh").read_text(encoding="utf-8")
    assert '"${TASK_ROOT}/tasks/${TASK_ID}.json"' in source
    assert "PROXY_EGRESS_ARGS=(--allow-public-egress)" in source
    assert '"${PROXY_EGRESS_ARGS[@]}"' in source
    assert 'export MLSYSFORGE_WORK_INTERNET="${TASK_INTERNET}"' in source
    assert "--internal" in source and "--enable-internet" in source
    assert "llada-h100-decoder-optimization-v2-vanilla-seed/scripts/restricted_connect_proxy.py" in source
