"""scripts/infrathon/assets.py: task asset manifests stage, fetch and verify by SHA-256."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("infrathon_assets", REPOSITORY_ROOT / "scripts/infrathon/assets.py")
assert SPEC is not None and SPEC.loader is not None
assets = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(assets)


def _task(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, files: dict[str, bytes]) -> str:
    tasks_root = tmp_path / "tasks"
    task = "demo-task-v1"
    entries = []
    for relative, data in files.items():
        origin = tasks_root / task / "judge/private" / Path(relative).name
        origin.parent.mkdir(parents=True, exist_ok=True)
        origin.write_bytes(data)
        entries.append({
            "path": relative,
            "sha256": hashlib.sha256(data).hexdigest(),
            "source": {"kind": "infrathon-dataset"},
            "staged_from": str(origin.relative_to(tasks_root / task)),
        })
    manifest = {"schema_version": 1, "task_directory": task, "assets": entries}
    (tasks_root / task / "assets").mkdir(parents=True)
    (tasks_root / task / "assets/manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(assets, "TASKS_ROOT", tasks_root)
    return task


def test_stage_then_fetch_from_a_local_copy_verifies_every_file(tmp_path, monkeypatch):
    task = _task(tmp_path, monkeypatch, {"judge/final_cases.json": b'{"cases": []}'})
    staged = tmp_path / "staged"
    assets.stage(task, staged)
    target = staged / task / "judge/final_cases.json"
    assert target.read_bytes() == b'{"cases": []}'
    # Judge-only inputs stay private to the host user.
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert stat.S_IMODE(target.parent.stat().st_mode) == 0o700
    monkeypatch.setenv("INFRATHON_ASSETS_SOURCE", str(staged))
    fetched = tmp_path / "fetched"
    assets.fetch(task, fetched)
    assert assets.check(task, fetched) == []


def test_check_reports_a_changed_or_missing_file(tmp_path, monkeypatch):
    task = _task(tmp_path, monkeypatch, {"judge/final_cases.json": b"frozen"})
    root = tmp_path / "root"
    assert assets.check(task, root)  # missing
    assets.stage(task, root)
    (root / task / "judge/final_cases.json").write_bytes(b"edited")
    assert assets.check(task, root)


def test_a_source_that_differs_from_the_manifest_is_refused(tmp_path, monkeypatch):
    task = _task(tmp_path, monkeypatch, {"judge/final_cases.json": b"frozen"})
    (tmp_path / "tasks" / task / "judge/private/final_cases.json").write_bytes(b"drifted")
    with pytest.raises(assets.AssetError, match="does not match the manifest"):
        assets.stage(task, tmp_path / "root")
    assert not (tmp_path / "root" / task / "judge/final_cases.json").exists()


@pytest.mark.parametrize("path", ["../escape.json", "/abs.json", "private/x.json"])
def test_manifest_paths_stay_inside_the_task_asset_tree(tmp_path, monkeypatch, path):
    task = _task(tmp_path, monkeypatch, {"judge/a.json": b"x"})
    manifest_path = tmp_path / "tasks" / task / "assets/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["assets"][0]["path"] = path
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(assets.AssetError):
        assets.load_manifest(task)


def test_every_committed_task_manifest_matches_its_staged_from_file():
    for task in assets.all_tasks():
        manifest = assets.load_manifest(task)
        for asset in manifest["assets"]:
            origin = asset.get("staged_from")
            if not origin or Path(origin).is_absolute() or assets.remote_origin(origin) or "$" in origin:
                continue  # host-side sources are checked by `stage` on the host
            path = assets.TASKS_ROOT / task / origin
            actual = assets.tree_digest(path) if assets.is_tree(asset) else hashlib.sha256(path.read_bytes()).hexdigest()
            assert actual == asset["sha256"], (task, origin)


def _tree_task(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[str, Path]:
    """A task with one listed file and one listed directory under judge/private/."""
    tasks_root = tmp_path / "tasks"
    task = "tree-task-v1"
    origin = tmp_path / "host-assets/private"
    (origin / "games/a").mkdir(parents=True)
    (origin / "games/a/1.json").write_bytes(b"one")
    (origin / "games/2.json").write_bytes(b"two")
    (origin / "eval.jsonl").write_bytes(b"eval")
    (origin / "private_seed.txt").write_bytes(b"secret seed")  # never enters a context
    manifest = {"schema_version": 1, "task_directory": task, "assets": [
        {"path": "judge/private/eval.jsonl", "sha256": hashlib.sha256(b"eval").hexdigest(),
         "source": {"kind": "infrathon-dataset"}, "staged_from": str(origin / "eval.jsonl")},
        {"path": "judge/private/games/", "sha256": assets.tree_digest(origin / "games"),
         "source": {"kind": "infrathon-dataset"}, "staged_from": str(origin / "games")},
    ]}
    (tasks_root / task / "assets").mkdir(parents=True)
    (tasks_root / task / "assets/manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(assets, "TASKS_ROOT", tasks_root)
    return task, origin


def test_directory_assets_stage_fetch_and_check_by_tree_digest(tmp_path, monkeypatch):
    task, _ = _tree_task(tmp_path, monkeypatch)
    staged = tmp_path / "staged"
    assets.stage(task, staged)
    assert (staged / task / "judge/private/games/a/1.json").read_bytes() == b"one"
    assert assets.check(task, staged) == []
    monkeypatch.setenv("INFRATHON_ASSETS_SOURCE", str(staged))
    fetched = tmp_path / "fetched"
    assets.fetch(task, fetched)
    assert assets.check(task, fetched) == []
    (fetched / task / "judge/private/games/extra.json").write_bytes(b"planted")
    assert assets.check(task, fetched)  # an added file changes the tree digest


def test_context_copies_listed_assets_and_refuses_unlisted_unless_allowed(tmp_path, monkeypatch):
    task, origin = _tree_task(tmp_path, monkeypatch)
    with pytest.raises(assets.AssetError, match="unlisted"):
        assets.build_context(task, "judge/private", origin, tmp_path / "ctx")
    assets.build_context(task, "judge/private", origin, tmp_path / "ctx", allow_unlisted=True)
    copied = sorted(p.relative_to(tmp_path / "ctx").as_posix() for p in (tmp_path / "ctx").rglob("*") if p.is_file())
    assert copied == ["eval.jsonl", "games/2.json", "games/a/1.json"]
    assert not (tmp_path / "ctx/private_seed.txt").exists()


def test_context_of_one_listed_file(tmp_path, monkeypatch):
    task, origin = _tree_task(tmp_path, monkeypatch)
    assets.build_context(task, "judge/private/eval.jsonl", origin / "eval.jsonl", tmp_path / "ctx")
    assert (tmp_path / "ctx/eval.jsonl").read_bytes() == b"eval"
    (origin / "eval.jsonl").write_bytes(b"drifted")
    with pytest.raises(assets.AssetError, match="does not match"):
        assets.build_context(task, "judge/private/eval.jsonl", origin / "eval.jsonl", tmp_path / "ctx")


def test_hardware_assets_are_fetched_for_this_host_only(tmp_path, monkeypatch):
    task = _task(tmp_path, monkeypatch, {
        "judge/final_cases.json": b"final",
        "hw/h100-94gb-400w/goldens_nvl.json": b"nvl",
        "hw/h100-80gb-700w/judge/goldens_sxm.json": b"sxm",
    })
    staged = tmp_path / "staged"
    assets.stage(task, staged)  # the maintainer stages every hardware type
    sxm = staged / task / "hw/h100-80gb-700w/judge/goldens_sxm.json"
    assert stat.S_IMODE(sxm.stat().st_mode) == 0o600  # Judge-only under hw/<hw>/judge/
    nvl = staged / task / "hw/h100-94gb-400w/goldens_nvl.json"
    assert stat.S_IMODE(nvl.stat().st_mode) == 0o644
    monkeypatch.setenv("INFRATHON_ASSETS_SOURCE", str(staged))
    fetched = tmp_path / "fetched"
    with pytest.raises(assets.AssetError, match="hardware-specific"):
        assets.fetch(task, fetched, None)
    assets.fetch(task, fetched, "h100-94gb-400w")
    assert (fetched / task / "hw/h100-94gb-400w/goldens_nvl.json").read_bytes() == b"nvl"
    assert not (fetched / task / "hw/h100-80gb-700w").exists()
    assert assets.check(task, fetched, "h100-94gb-400w") == []
    assert assets.check(task, fetched, "h100-80gb-700w")


def test_hardware_assets_name_a_known_hardware_type(tmp_path, monkeypatch):
    task = _task(tmp_path, monkeypatch, {"hw/b200-180gb-1000w/goldens.json": b"x"})
    with pytest.raises(assets.AssetError, match="hw/<one of"):
        assets.load_manifest(task)


def test_hardware_comes_from_the_environment_or_the_gpu(monkeypatch):
    monkeypatch.setenv("INFRATHON_HARDWARE", "h100-80gb-700w")
    assert assets.detect_hardware() == "h100-80gb-700w"
    monkeypatch.setenv("INFRATHON_HARDWARE", "h100")
    with pytest.raises(assets.AssetError):
        assets.detect_hardware()
    monkeypatch.delenv("INFRATHON_HARDWARE")

    class Completed:
        returncode = 0
        stdout = "NVIDIA H100 NVL\n"

    monkeypatch.setattr(assets.subprocess, "run", lambda *a, **k: Completed())
    assert assets.detect_hardware() == "h100-94gb-400w"
    Completed.stdout = "NVIDIA B200\n"
    assert assets.detect_hardware() is None


def _infrathon_tasks():
    spec = importlib.util.spec_from_file_location("infrathon_tasks", REPOSITORY_ROOT / "scripts/infrathon/tasks.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_release_task_list_covers_every_current_task_and_the_readme_table_is_current():
    tasks = _infrathon_tasks()
    document = tasks.load()
    current = {p.name for p in (REPOSITORY_ROOT / "tasks/latest").iterdir() if p.is_dir()}
    assert set(document["tasks"]) == current
    readme = (REPOSITORY_ROOT / "scripts/infrathon/README.md").read_text(encoding="utf-8")
    assert tasks.table(document) in readme
    gkd = "qwen3-4b-to-1p7b-gkd-distillation-optimizer-dual-h100-v1-official-trl-seed"
    assert tasks.check(document, gkd, "h100-80gb-700w", allow_untested=False) is None
    blocked = json.loads(json.dumps(document))
    blocked["tasks"][gkd]["hardware"]["h100-80gb-700w"] = "unsupported"
    assert tasks.check(blocked, gkd, "h100-80gb-700w", allow_untested=True)
    assert tasks.check(document, "miles-olmo3-7b-model-integration-3xh100-v2-seed", "h100-94gb-400w", True)
    assert tasks.check(document, "eagle3-spec-decode-engine-qwen3-8b-h100-v1-seed", "h100-80gb-700w", False) is None


def _hf_task(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, path: str = "public/models/tiny/",
             revision: str = "a" * 40) -> tuple[str, Path]:
    """A task with one hf asset whose 'cache snapshot' links each file to a blob, like huggingface_hub."""
    blobs = tmp_path / "hf-cache/blobs"
    snapshot = tmp_path / "hf-cache/snapshots" / revision
    (snapshot / "sub").mkdir(parents=True)
    blobs.mkdir(parents=True)
    for name, data in {"config.json": b"{}", "sub/model.safetensors": b"weights"}.items():
        blob = blobs / hashlib.sha256(data).hexdigest()
        blob.write_bytes(data)
        (snapshot / name).symlink_to(blob)
    tasks_root = tmp_path / "tasks"
    task = "hf-task-v1"
    manifest = {"schema_version": 1, "task_directory": task, "assets": [{
        "path": path, "sha256": assets.tree_digest(snapshot, follow_symlinks=True),
        "source": {"kind": "hf", "repo_id": "org/tiny", "repo_type": "model", "revision": revision,
                   "layout": "flat"}}]}
    (tasks_root / task / "assets").mkdir(parents=True)
    (tasks_root / task / "assets/manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(assets, "TASKS_ROOT", tasks_root)
    monkeypatch.setattr(assets, "_hf_snapshot", lambda asset: snapshot)
    return task, blobs


def test_hf_assets_are_laid_out_as_plain_files_linked_from_the_cache(tmp_path, monkeypatch):
    task, blobs = _hf_task(tmp_path, monkeypatch)
    root = tmp_path / "root"
    assets.fetch(task, root)
    weights = root / task / "public/models/tiny/sub/model.safetensors"
    assert not weights.is_symlink() and weights.read_bytes() == b"weights"
    assert weights.stat().st_ino == (blobs / hashlib.sha256(b"weights").hexdigest()).stat().st_ino
    assert assets.check(task, root) == []
    assert (root / ".infrathon-verified" / task / "public/models/tiny.json").is_file()
    assets.stage(task, tmp_path / "staged")  # public models are not part of our dataset
    assert not (tmp_path / "staged" / task).exists()


def test_a_stamped_directory_is_rehashed_when_a_file_changes_or_with_full(tmp_path, monkeypatch):
    task, _ = _hf_task(tmp_path, monkeypatch)
    root = tmp_path / "root"
    assets.fetch(task, root)
    config = root / task / "public/models/tiny/config.json"
    stat_before = config.stat()
    config.unlink()
    config.write_bytes(b"[]")  # same size, new content
    os.utime(config, ns=(stat_before.st_atime_ns, stat_before.st_mtime_ns))
    assert assets.check(task, root) == []  # the stamp trusts unchanged sizes and mtimes
    assert assets.check(task, root, full=True)
    config.write_bytes(b"{ }")  # size changes: the stamp no longer matches
    assert assets.check(task, root)


@pytest.mark.parametrize("path, revision", [("judge/models/tiny/", "a" * 40), ("public/models/tiny/", "main"),
                                            ("public/models/tiny.json", "a" * 40)])
def test_hf_assets_are_public_directories_at_a_commit(tmp_path, monkeypatch, path, revision):
    task, _ = _hf_task(tmp_path, monkeypatch, path=path, revision=revision)
    with pytest.raises(assets.AssetError, match="hf asset"):
        assets.load_manifest(task)


def test_cache_layout_hf_assets_stay_in_the_hub_cache(tmp_path, monkeypatch):
    task, _ = _hf_task(tmp_path, monkeypatch)
    manifest_path = tmp_path / "tasks" / task / "assets/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["assets"][0]["source"]["layout"]  # cache is the default
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    hub = tmp_path / "hub"
    snapshot = hub / "models--org--tiny/snapshots" / ("a" * 40)
    snapshot.parent.mkdir(parents=True)
    (tmp_path / "hf-cache/snapshots" / ("a" * 40)).rename(snapshot)
    monkeypatch.setenv("HF_HUB_CACHE", str(hub))
    monkeypatch.setattr(assets, "_hf_snapshot", lambda asset: snapshot)
    root = tmp_path / "root"
    assert assets.check(task, root) == []  # an existing cache is used as it is
    assets.fetch(task, root)
    assert not (root / task).exists()  # nothing laid out under the asset root
    assert assets.hub_cache() == hub
    # A commit download writes no ref; fetch adds refs/main for launchers that resolve it.
    ref = hub / "models--org--tiny/refs/main"
    assert ref.read_text(encoding="utf-8") == "a" * 40
    ref.write_text("b" * 40, encoding="utf-8")
    assets.fetch(task, root)
    assert ref.read_text(encoding="utf-8") == "b" * 40  # an existing ref is left alone
    (snapshot / "config.json").unlink()
    assert assets.check(task, root)


def test_a_readable_judge_asset_keeps_public_modes(tmp_path, monkeypatch):
    task = _task(tmp_path, monkeypatch, {"judge/train_private.jsonl": b"rows"})
    manifest_path = tmp_path / "tasks" / task / "assets/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["assets"][0]["readable"] = True
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assets.stage(task, tmp_path / "root")
    assert stat.S_IMODE((tmp_path / "root" / task / "judge/train_private.jsonl").stat().st_mode) == 0o644


def test_owner_root_is_only_for_judge_only_assets(tmp_path, monkeypatch):
    task = _task(tmp_path, monkeypatch, {"public/data.json": b"x"})
    manifest_path = tmp_path / "tasks" / task / "assets/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["assets"][0]["owner"] = "root"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(assets.AssetError, match="owner"):
        assets.load_manifest(task)


def test_own_hands_a_verified_asset_to_root_through_a_container(tmp_path, monkeypatch):
    task, origin = _tree_task(tmp_path, monkeypatch)
    manifest_path = tmp_path / "tasks" / task / "assets/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["assets"][1]["owner"] = "root"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    root = tmp_path / "root"
    assets.stage(task, root)
    calls = []
    monkeypatch.setattr(assets.subprocess, "run", lambda command, check: calls.append(command))
    assets.own(task, root, None, "judge-image:local")
    assert len(calls) == 1 and calls[0][:2] == ["docker", "run"] and "judge-image:local" in calls[0]
    assert f"{root / task / 'judge/private/games'}:/target" in calls[0]


def _owner_root_task(tmp_path, monkeypatch) -> tuple[str, Path, Path]:
    """A tree task whose games/ directory is owner-root, staged as a source and fetched from it."""
    task, _ = _tree_task(tmp_path, monkeypatch)
    manifest_path = tmp_path / "tasks" / task / "assets/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["assets"][1]["owner"] = "root"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assets.stage(task, tmp_path / "staged")
    monkeypatch.setenv("INFRATHON_ASSETS_SOURCE", str(tmp_path / "staged"))
    root = tmp_path / "root"
    return task, root, root / task / "judge/private/games"


def test_fetch_hands_an_asset_the_agent_uid_owns_to_root_through_a_local_image(tmp_path, monkeypatch):
    task, root, games = _owner_root_task(tmp_path, monkeypatch)
    monkeypatch.setattr(assets, "AGENT_UID", os.getuid())  # this host user is the container agent's uid
    monkeypatch.setattr(assets, "_local_image", lambda task: "judge-image:local")
    calls = []
    monkeypatch.setattr(assets.subprocess, "run", lambda command, check: calls.append(command))
    assets.fetch(task, root)
    assert len(calls) == 1 and "judge-image:local" in calls[0] and f"{games}:/target" in calls[0]
    calls.clear()
    assets.fetch(task, root, image="other-image:local")  # an --image wins over the task's images
    assert len(calls) == 1 and "other-image:local" in calls[0]


def test_fetch_without_a_local_image_prints_the_own_command(tmp_path, monkeypatch, capsys):
    task, root, games = _owner_root_task(tmp_path, monkeypatch)
    monkeypatch.setattr(assets, "AGENT_UID", os.getuid())
    monkeypatch.setattr(assets, "_local_image", lambda task: None)
    monkeypatch.setattr(assets.subprocess, "run", lambda *a, **k: pytest.fail("no container without an image"))
    assets.fetch(task, root, "h100-80gb-700w")
    err = capsys.readouterr().err
    assert f"own {task} --image <image> --hardware h100-80gb-700w" in err and f"INFRATHON_ASSETS_ROOT={root}" in err
    assert "sudo" not in err.replace("no sudo", "")


def test_fetch_leaves_an_asset_another_uid_owns_and_skips_a_root_owned_one(tmp_path, monkeypatch):
    task, root, games = _owner_root_task(tmp_path, monkeypatch)
    monkeypatch.setattr(assets, "AGENT_UID", os.getuid() + 1)  # fetched by another user: already safe
    monkeypatch.setattr(assets, "_local_image", lambda task: pytest.fail("no image needed"))
    assets.fetch(task, root)
    monkeypatch.setattr(assets, "_root_owned", lambda path: path == games)
    monkeypatch.setattr(assets, "_download", lambda task, asset: pytest.fail(f"re-downloaded {asset['path']}"))
    shutil.rmtree(games)  # unreadable once root-owned: fetch must not re-hash or replace it
    games.mkdir()
    assets.fetch(task, root)


def test_own_agent_owned_only_touches_assets_the_agent_uid_owns(tmp_path, monkeypatch):
    task, root, games = _owner_root_task(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(assets.subprocess, "run", lambda command, check: calls.append(command))
    assets.own(task, root, None, "judge-image:local", agent_owned=True)  # not fetched yet: left to fetch
    monkeypatch.setattr(assets, "AGENT_UID", os.getuid() + 1)
    assets.fetch(task, root)
    assets.own(task, root, None, "judge-image:local", agent_owned=True)
    assert calls == []
    monkeypatch.setattr(assets, "AGENT_UID", os.getuid())
    assets.own(task, root, None, "judge-image:local", agent_owned=True)
    assert len(calls) == 1 and f"{games}:/target" in calls[0]


def test_the_local_image_is_the_tasks_judge_then_work_image_docker_has(tmp_path, monkeypatch):
    task, root, games = _owner_root_task(tmp_path, monkeypatch)
    assert assets._local_image(task) is None  # no images/release.json
    (tmp_path / "tasks" / task / "images").mkdir()
    (tmp_path / "tasks" / task / "images/release.json").write_text(json.dumps(
        {"local_tags": {"work": "w:local", "judge": "j:local"}}), encoding="utf-8")
    present = {"w:local"}
    monkeypatch.setattr(assets.subprocess, "run", lambda command, **kwargs: subprocess.CompletedProcess(
        command, 0 if command[-1] in present else 1))
    assert assets._local_image(task) == "w:local"
    present.add("j:local")
    assert assets._local_image(task) == "j:local"


def test_export_dataset_holds_only_our_assets(tmp_path, monkeypatch):
    task = _task(tmp_path, monkeypatch, {"judge/final.json": b"final", "hw/h100-94gb-400w/g.json": b"g"})
    root = tmp_path / "root"
    assets.stage(task, root)
    (root / ".infrathon-verified").mkdir(exist_ok=True)
    out = tmp_path / "dataset"
    assets.export_dataset([task], root, out)
    files = sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file())
    assert files == [f"{task}/hw/h100-94gb-400w/g.json", f"{task}/judge/final.json"]


def test_export_dataset_copies_a_root_owned_asset_out_through_a_container(tmp_path, monkeypatch):
    task, origin = _tree_task(tmp_path, monkeypatch)
    manifest_path = tmp_path / "tasks" / task / "assets/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["assets"][1]["owner"] = "root"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    root = tmp_path / "root"
    assets.stage(task, root)
    owned = root / task / "judge/private/games"
    monkeypatch.setattr(assets, "_root_owned", lambda path: path == owned)
    with pytest.raises(assets.AssetError, match="--image"):
        assets.export_dataset([task], root, tmp_path / "dataset")

    def container(command, check):  # what the container's cp -r + chown leaves behind
        assert "judge-image:local" in command and f"{owned}:/source:ro" in command
        shutil.copytree(owned, tmp_path / "dataset" / task / "judge/private/games")

    monkeypatch.setattr(assets.subprocess, "run", container)
    assets.export_dataset([task], root, tmp_path / "dataset", "judge-image:local")
    exported = tmp_path / "dataset" / task / "judge/private/games"
    assert assets.tree_digest(exported) == manifest["assets"][1]["sha256"]


def _upstream_asset(path: str, source: dict, data_root: Path | None = None) -> dict:
    asset = {"path": path, "sha256": "0" * 64, "source": source}
    if data_root is not None:
        asset["sha256"] = assets.tree_digest(data_root, exclude=assets._records(asset))
    return asset


def test_git_assets_check_out_the_commit_and_keep_git_out_of_the_digest(tmp_path):
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", "-C", str(upstream)]
    subprocess.run(["git", "init", "-q", str(upstream)], check=True)
    (upstream / "hubconf.py").write_text("x = 1\n", encoding="utf-8")
    subprocess.run([*git, "add", "."], check=True)
    subprocess.run([*git, "commit", "-q", "-m", "one"], check=True)
    commit = subprocess.run([*git, "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    (upstream / "hubconf.py").write_text("x = 2\n", encoding="utf-8")  # a later commit is not wanted
    subprocess.run([*git, "commit", "-q", "-am", "two"], check=True)
    expected = tmp_path / "expected"
    expected.mkdir()
    (expected / "hubconf.py").write_text("x = 1\n", encoding="utf-8")
    asset = _upstream_asset("public/dino/", {"kind": "git", "url": str(upstream), "commit": commit}, expected)
    root = tmp_path / "root"
    target = root / "demo" / "public/dino"
    assets._install_upstream_tree(root, "demo", asset, target, assets._stamp_path(root, "demo", asset))
    assert (target / "hubconf.py").read_text(encoding="utf-8") == "x = 1\n"
    assert (target / ".git").is_dir()  # kept: consumers check the commit there
    assert assets._current(target, asset, full=True)
    assert not (root / assets.PARTIAL_DIRECTORY / "demo/public/dino").exists()


def test_url_assets_are_installed_only_with_the_manifest_sha256(tmp_path, monkeypatch):
    payload = b"weights"
    monkeypatch.setattr(assets.urllib.request, "urlopen", lambda request, timeout: io.BytesIO(payload))
    asset = {"path": "public/w.pth", "sha256": hashlib.sha256(payload).hexdigest(),
             "source": {"kind": "url", "url": "https://example.invalid/w.pth"}}
    root = tmp_path / "root"
    assets._install_url(root, "demo", asset, root / "demo/public/w.pth")
    assert (root / "demo/public/w.pth").read_bytes() == payload
    asset["sha256"] = "0" * 64
    with pytest.raises(assets.AssetError, match="does not match"):
        assets._install_url(root, "demo", asset, root / "demo/public/other.pth")
    assert not (root / "demo/public/other.pth").exists()


def test_local_dir_hf_assets_keep_their_revision_records_out_of_the_digest(tmp_path, monkeypatch):
    huggingface_hub = pytest.importorskip("huggingface_hub")

    def snapshot_download(**kwargs):
        local_dir = Path(kwargs["local_dir"])
        (local_dir / "config.json").write_text("{}", encoding="utf-8")
        records = local_dir / ".cache/huggingface/download"
        records.mkdir(parents=True, exist_ok=True)
        (records / "config.json.metadata").write_text(f"{kwargs['revision']}\netag\n1.0\n", encoding="utf-8")
        return str(local_dir)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    expected = tmp_path / "expected"
    expected.mkdir()
    (expected / "config.json").write_text("{}", encoding="utf-8")
    source = {"kind": "hf", "repo_id": "org/tiny", "repo_type": "model", "revision": "a" * 40, "layout": "local-dir"}
    asset = _upstream_asset("public/models/tiny/", source, expected)
    root = tmp_path / "root"
    target = root / "demo/public/models/tiny"
    assets._install_upstream_tree(root, "demo", asset, target, assets._stamp_path(root, "demo", asset))
    assert (target / ".cache/huggingface/download/config.json.metadata").read_text().startswith("a" * 40)
    assert assets._current(target, asset, full=True)


@pytest.mark.parametrize("path, source", [
    ("public/dino/", {"kind": "git", "url": "git@github.com:org/dino.git", "commit": "a" * 40}),
    ("public/dino/", {"kind": "git", "url": "https://github.com/org/dino.git", "commit": "main"}),
    ("judge/dino/", {"kind": "git", "url": "https://github.com/org/dino.git", "commit": "a" * 40}),
    ("public/w/", {"kind": "url", "url": "https://example.invalid/w.pth"}),
    ("public/w.pth", {"kind": "url", "url": "http://example.invalid/w.pth"}),
])
def test_git_and_url_assets_are_public_and_pinned(tmp_path, monkeypatch, path, source):
    tasks_root = tmp_path / "tasks"
    manifest = {"schema_version": 1, "task_directory": "demo-task-v1",
                "assets": [{"path": path, "sha256": "0" * 64, "source": source}]}
    (tasks_root / "demo-task-v1/assets").mkdir(parents=True)
    (tasks_root / "demo-task-v1/assets/manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(assets, "TASKS_ROOT", tasks_root)
    with pytest.raises(assets.AssetError, match="(git|url) asset"):
        assets.load_manifest("demo-task-v1")
