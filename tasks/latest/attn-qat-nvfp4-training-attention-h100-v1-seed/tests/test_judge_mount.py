"""The real layers reach the Judge only as a checked, read-only run-time mount."""
from __future__ import annotations

import hashlib
import json

import pytest

import conftest
from conftest import TASK_ROOT

load_module = getattr(conftest, "load_module", None) or conftest._load
entry = load_module("attn_qat_judge_entry", "scripts/sforge_judge_entry.py")
JUDGE = f"attn_qat_training_attention.judge.{entry.TASK_ID}:local"
WORK = f"attn_qat_training_attention.work.{entry.TASK_ID}:local"


def capture(tmp_path, *, layers=None, recipe=None):
    """A fake capture directory holding both recipes' manifests and their layer files."""
    for index, (recipe_path, manifest_name, count) in enumerate(entry.CAPTURES):
        wanted = count if layers is None or index else layers
        names = {f"src{index}-b{i:02d}-s010": {} for i in range(wanted)}
        for name in names:
            (tmp_path / f"{name}.pt").write_bytes(b"x")
        sha = recipe if recipe is not None and index == 0 else hashlib.sha256(recipe_path.read_bytes()).hexdigest()
        (tmp_path / manifest_name).write_text(json.dumps({"recipe": {"sha256": sha}, "files": names}))
    return tmp_path


def test_real_layer_directory_is_checked(tmp_path, monkeypatch):
    monkeypatch.delenv(entry.REAL_LAYERS_ENV, raising=False)
    with pytest.raises(RuntimeError, match="required"):
        entry.real_layers_directory()
    monkeypatch.setenv(entry.REAL_LAYERS_ENV, str(capture(tmp_path)))
    assert entry.real_layers_directory() == str(tmp_path.resolve())
    assert entry.EXPECTED_LAYERS == 78
    assert [manifest for _recipe, manifest, _count in entry.CAPTURES] == [
        "real_layers.manifest.json", "real_layers_v2.manifest.json",
    ]


def test_wrong_recipe_or_count_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv(entry.REAL_LAYERS_ENV, str(capture(tmp_path, recipe="0" * 64)))
    with pytest.raises(RuntimeError, match="REAL_LAYERS.json"):
        entry.real_layers_directory()
    for stale in tmp_path.iterdir():
        stale.unlink()
    capture(tmp_path, layers=31)
    monkeypatch.setenv(entry.REAL_LAYERS_ENV, str(tmp_path))
    with pytest.raises(RuntimeError, match="32 layers"):
        entry.real_layers_directory()


def test_only_the_judge_gets_the_mount(tmp_path, monkeypatch):
    monkeypatch.setenv(entry.REAL_LAYERS_ENV, str(capture(tmp_path)))
    calls = []
    monkeypatch.setattr(
        entry, "_create_container",
        lambda _self, image, name, **kwargs: calls.append((image, kwargs.get("mounts"))),
    )
    backend = object.__new__(entry.DockerBackend)
    entry.create_container(backend, JUDGE, "sforge.x")
    entry.create_container(backend, WORK, f"sforge.run.{entry.TASK_ID}.x")
    assert calls[0][1] == [(str(tmp_path.resolve()), entry.REAL_LAYERS_MOUNT, "ro")]
    assert calls[1][1] is None


def test_real_layers_are_never_baked_into_an_image():
    for dockerfile in ("images/judge/Dockerfile", "images/judge/Dockerfile.private", "images/work/Dockerfile"):
        assert "real_layers/" not in (TASK_ROOT / dockerfile).read_text().replace(
            "real_layers/REAL_LAYERS.json", ""
        ).replace("real_layers/REAL_LAYERS_V2.json", "")
    assert 'real_layers=' not in (TASK_ROOT / "scripts/build_images.sh").read_text()


def test_every_judge_command_goes_through_the_entry():
    for script in ("run_fixed_budget.sh", "eval_archive.sh", "replay_candidates.sh"):
        text = (TASK_ROOT / "scripts" / script).read_text()
        assert '"${JUDGE_ENTRY}"' in text, script
        for command in (" serve ", " eval ", ' "${subcommand}"'):
            assert f'"${{SFORGE_BIN}}"{command}' not in text, (script, command)
