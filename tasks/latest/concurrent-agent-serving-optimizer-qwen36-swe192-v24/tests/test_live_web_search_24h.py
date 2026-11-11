from __future__ import annotations

from pathlib import Path
import re
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_fixed_budget_runner_accepts_24h_live_search_runs() -> None:
    runner = _read("scripts/run_fixed_budget.sh")

    assert '[[ "${MODEL}" =~ ^[A-Za-z0-9][A-Za-z0-9_./:-]*$ ]]' in runner
    assert "high|max|xhigh|ultra" in runner
    assert "TASK_TIMEOUT must be minute-aligned and at least 3600 seconds" in runner
    assert "--disable-auto-eval" in runner


def test_24h_profiles_are_exact_and_manual_eval_only() -> None:
    """experiment-*-24h.yaml run profiles and their tmux wrappers, checked only
    when the tree carries them."""
    for path in sorted(ROOT.glob("experiment-*-24h.yaml")):
        profile = path.read_text(encoding="utf-8")
        model = re.search(r"^\s*model: (\S+)$", profile, re.M)
        assert model, path.name
        assert "timeout: 86400" in profile
        assert "disable_auto_eval: true" in profile
        assert "internet: true" in profile
        assert "CODEX_WEB_SEARCH_MODE: live" in profile
        stem = path.stem.removeprefix("experiment-").replace("-", "_")
        wrapper_path = ROOT / "scripts" / f"launch_{stem}_tmux.sh"
        if wrapper_path.is_file():
            wrapper = wrapper_path.read_text(encoding="utf-8")
            assert f'export MODEL="{model.group(1)}"' in wrapper
            assert 'export TASK_TIMEOUT="86400"' in wrapper
            assert 'export CODEX_WEB_SEARCH_MODE="live"' in wrapper


def test_live_search_is_explicit_and_attested() -> None:
    runtime = _read("scripts/sforge_runtime_entry.py")
    runner = _read("scripts/run_fixed_budget.sh")
    launcher = _read("scripts/launch_fixed_budget_tmux.sh")

    assert "_configured_web_search_mode" in runtime
    assert "_task_internet(kwargs.get(\"internet\", True)) and mode == \"live\"" in runtime
    assert 'WORK_INTERNET_ENV = "MLSYSFORGE_WORK_INTERNET"' in runtime
    assert "web_search=\"live\"" in runtime
    assert "web_search=\"disabled\"" in runtime
    assert "codex_web_search: $web_search" in runner
    assert '"CODEX_WEB_SEARCH_MODE=${WEB_SEARCH_VALUE}"' in launcher
    assert "codex_web_search:$codex_web_search" in launcher


def test_runtime_generates_explicit_live_search_command(
    runtime_module: Any, monkeypatch: Any
) -> None:
    monkeypatch.setenv("CODEX_WEB_SEARCH_MODE", "live")
    monkeypatch.setattr(
        runtime_module,
        "_original_format_run_cmd",
        lambda _self, *args, **kwargs: "codex exec --example",
    )

    command = runtime_module.configured_format_run_cmd(object())

    assert "web_search=\"live\"" in command
    assert "web_search=\"disabled\"" not in command


def test_public_advice_does_not_invent_hidden_promotion_gates() -> None:
    instructions = _read("images/work/AGENT_TASK.md")

    assert "It is an advisory" in instructions
    assert "measurement, not a promotion gate" in instructions
    assert "Do not invent additional numeric prerequisites" in instructions
    assert "is not a hidden-submission eligibility gate" in instructions
    assert "The task has no general internet access" not in instructions
    # Briefs never advertise internet or search.
    assert "web search" not in instructions.lower()
    assert "Arbitrary shell egress" not in instructions
