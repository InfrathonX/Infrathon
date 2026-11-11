from __future__ import annotations

import importlib.util
import json
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_task_and_codex_defaults_enable_native_web_search() -> None:
    task = json.loads(
        (ROOT / "tasks/quant_videogen_cross_model_kv_codec_h100_scratch_v4.json").read_text()
    )
    config = tomllib.loads((ROOT / "images/work/codex-config.toml").read_text())
    assert task["internet"] is True
    assert config["web_search"] == "live"


def test_runtime_command_enables_native_search_without_direct_work_egress() -> None:
    path = ROOT / "scripts/codex_search.py"
    spec = importlib.util.spec_from_file_location("qvg_codex_search", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    command = module.enable_native_web_search(
        "codex exec -m example-model --dangerously-bypass-approvals-and-sandbox prompt"
    )
    assert command.startswith("codex --search exec ")
    assert 'web_search="disabled"' not in command

