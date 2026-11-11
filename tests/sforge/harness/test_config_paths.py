from pathlib import Path

import pytest

from sforge.harness.config import load_config


def test_cli_path_overrides_are_coerced_to_paths(tmp_path: Path) -> None:
    config = load_config(
        {
            "log_dir": str(tmp_path / "output"),
            "tasks_dir": str(tmp_path / "tasks"),
        }
    )

    assert config.log_dir == tmp_path / "output"
    assert config.tasks_dir == tmp_path / "tasks"
    assert isinstance(config.log_dir, Path)
    assert isinstance(config.tasks_dir, Path)


def test_judge_network_mode_accepts_only_explicit_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SFORGE_JUDGE_NETWORK_MODE", "none")
    assert load_config().judge_network_mode == "none"

    monkeypatch.setenv("SFORGE_JUDGE_NETWORK_MODE", "bridge")
    with pytest.raises(ValueError, match="SFORGE_JUDGE_NETWORK_MODE|network mode"):
        load_config()
