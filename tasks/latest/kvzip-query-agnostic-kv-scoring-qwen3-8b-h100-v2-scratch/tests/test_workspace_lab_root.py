"""Workspace admission must accept the exact lab root the Work image ships.

The public controller runs `validate_submission(..., workspace_mode=True)` against
`/app/workdir/kv_scoring_lab` inside the Work image. That root carries the
task-owned files the Dockerfile bakes in plus the agent's scratch `build/`, none
of which reach the Judge (`submit_paths` keeps only candidates/FINAL_VERSION/
notes.md). A missing entry in the workspace allowlist breaks the participant's
only feedback loop, so bind the allowlist to the image layout here.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from conftest import TASK_ROOT
from source_admission import validate_submission

WORK_LAB = TASK_ROOT / "images/work/workdir/kv_scoring_lab"
DOCKERFILE = (TASK_ROOT / "images/work/Dockerfile").read_text(encoding="utf-8")
SEED = "".join(
    [
        "import torch\n",
        "class KVScorer:\n",
        " def __init__(self, spec, assets): self.spec = spec\n",
        " def score(self, layer_idx, hidden_states, queries, keys, values,"
        " position_embeddings, weights, ctx_range):\n",
        "  return keys[:,:,ctx_range[0]:ctx_range[1]].float().norm(dim=-1)\n",
    ]
)


def image_lab_root_entries() -> set[str]:
    """Entry names the Work image's lab root holds: repo tree + Dockerfile additions."""
    entries = {item.name for item in WORK_LAB.iterdir() if item.name != "__pycache__"}
    for match in re.finditer(r"^COPY\s+\S+\s+/app/workdir/kv_scoring_lab/(\S+)$", DOCKERFILE, re.M):
        entries.add(Path(match.group(1)).parts[0])
    for match in re.finditer(r"/app/workdir/kv_scoring_lab/([A-Za-z0-9_.-]+)", DOCKERFILE):
        entries.add(match.group(1))
    return {name for name in entries if name and name != "kv_scoring_lab"}


def build_lab(root: Path, entries: set[str]) -> None:
    candidate = root / "candidates" / "scorer_v001"
    candidate.mkdir(parents=True)
    (candidate / "submission.py").write_text(SEED, encoding="utf-8")
    (root / "FINAL_VERSION").write_text("1\n", encoding="utf-8")
    (root / "notes.md").write_text("notes\n", encoding="utf-8")
    for name in entries - {"candidates", "FINAL_VERSION", "notes.md"}:
        if Path(name).suffix:
            (root / name).write_text("x\n", encoding="utf-8")
        else:
            (root / name).mkdir(exist_ok=True)


def test_image_lab_root_is_admitted_in_workspace_mode(tmp_path: Path) -> None:
    entries = image_lab_root_entries()
    # The regression this test exists for: these two only appear inside the image.
    assert {"AGENT_TASK.md", "build"} <= entries
    root = tmp_path / "kv_scoring_lab"
    root.mkdir()
    build_lab(root, entries)
    result = validate_submission(root, workspace_mode=True)
    assert result["ok"], result["errors"]


def test_judge_mode_still_rejects_the_workspace_only_entries(tmp_path: Path) -> None:
    root = tmp_path / "kv_scoring_lab"
    root.mkdir()
    build_lab(root, image_lab_root_entries())
    result = validate_submission(root)
    assert not result["ok"]
    assert any("AGENT_TASK.md" in error for error in result["errors"])


def test_submit_paths_strip_every_workspace_only_entry() -> None:
    spec = json.loads(
        (TASK_ROOT / "tasks/kvzip_query_agnostic_kv_scoring_qwen3_8b_h100_scratch_v2.json").read_text(
            encoding="utf-8"
        )
    )
    assert set(spec["submit_paths"]) == {"candidates", "FINAL_VERSION", "notes.md"}
    workspace_only = image_lab_root_entries() - set(spec["submit_paths"])
    assert workspace_only, "the workspace allowlist must be wider than submit_paths"
    for name in workspace_only:
        assert name not in spec["submit_paths"]
