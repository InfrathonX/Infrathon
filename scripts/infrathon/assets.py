#!/usr/bin/env python3
"""Infrathon task assets: stage, fetch and check.

Every task lists the files it needs beyond the repository in
``tasks/latest/<task directory>/assets/manifest.json``.  They live under one
asset root (``INFRATHON_ASSETS_ROOT``) as ``<task directory>/<path>``:

    public/...         readable by Work and Judge (mounted, or copied into public images)
    judge/...          Judge-only inputs (holdouts); copied into the host-built Judge
                       holdout layer, never into a published image
    hw/<hw>/...        inputs that differ by hardware type (e.g. GPU-generated goldens);
    hw/<hw>/judge/...  the Judge-only ones among them

Judge-only files are installed 0600 (directories 0700) for the host user,
except an asset marked ``"readable": true``: data a task mounts into its Judge
for the candidate's own code to read there (ORPO's and GKD's private training
partitions), installed 0644/0755 like public files.  An asset marked
``"owner": "root"`` is a Judge-only directory a task mounts and refuses when the
container's agent uid owns it (the Miles diffusion private roots): when the host
user has that uid, ``own`` hands it to root (through a container, no sudo), after
which ``fetch`` and ``check`` report it as root-owned rather than re-hashing it.
``fetch`` runs that step itself for an asset it leaves owned by the agent uid, through
``--image`` or the task's local Judge or Work image (images/release.json), and prints
the ``own`` command when neither is there; ``pull_images.sh`` runs it again once the
images are pulled (``own --agent-owned``).

``<hw>`` is one of ``HARDWARE_TYPES``.  ``fetch`` and ``check`` take only the
``hw/`` assets of this host's hardware type (``--hardware``, ``INFRATHON_HARDWARE``,
or detected from GPU 0; ``hardware`` prints it); ``stage`` takes all of them.

An asset is one file, or a directory when its path ends in ``/``; a
directory's ``sha256`` is the digest of its ``sha256sum``-style listing (see
``tree_digest``).

``source.kind`` says where ``fetch`` gets it: ``infrathon-dataset`` (our dataset), or
``hf`` for a public Hugging Face repository at a pinned commit
(``repo_id``, ``repo_type`` model|dataset, 40-hex ``revision``, optional
``allow_patterns``, ``layout``).  An ``hf`` asset is downloaded through the normal
Hugging Face cache (``HF_HUB_CACHE`` / ``HF_HOME``, so an existing cache is
reused) and is never part of our dataset.  ``layout`` ``cache`` (the default)
leaves it there: tasks mount the cache's repository folder
(``<hub cache>/models--org--name``; ``hub-cache`` prints the directory) and the
digest is that of the snapshot at ``revision``.  ``flat`` also lays it out under
the asset root as plain files, hard-linked from the cache blobs when on the same
file system, for consumers that cannot follow the cache's links.  ``local-dir``
downloads it straight into the asset root, as ``hf download --local-dir`` does,
keeping the download records under ``.cache/huggingface/`` for consumers that read
the revision there (Quant-VideoGen); the cache is not used.  ``git`` is a public git
repository checked out at a pinned commit (https ``url``, 40-hex ``commit``, ``lfs``
to fetch its Git LFS files), with its ``.git`` kept; ``url`` is one public file
downloaded from an https ``url``.  None of these are part of our dataset.  The digest
of a ``local-dir`` or ``git`` directory leaves out its ``.cache/huggingface/`` or
``.git/`` records.  A verified directory
leaves a stamp (file sizes and mtimes) under ``<asset root>/.infrathon-verified/``, so
later checks of a large model skip re-hashing it unless ``--full``; that
directory is local bookkeeping, not part of the dataset, and so is
``<asset root>/.infrathon-partial/``, where an interrupted ``local-dir`` or ``git``
download is kept and resumed.

``fetch`` fills the asset root from the Infrathon Hugging Face dataset
(``INFRATHON_HF_DATASET``) or from a local copy of it (``INFRATHON_ASSETS_SOURCE``) and
verifies every SHA-256.  ``stage`` is the maintainer side: it copies each
asset from where the task keeps it today (``staged_from``) into the asset
root, which is then uploaded as the dataset.  ``check`` verifies without
changing anything.

    scripts/infrathon/assets.py fetch <task directory>... | --all  [--hardware <hw>] [--image IMAGE]
    scripts/infrathon/assets.py check <task directory>... | --all  [--hardware <hw>]
    scripts/infrathon/assets.py stage <task directory>... | --all
    scripts/infrathon/assets.py hardware
    scripts/infrathon/assets.py hub-cache              the Hugging Face hub cache directory
    scripts/infrathon/assets.py own <task directory> --image IMAGE [--hardware <hw>] [--agent-owned]
    scripts/infrathon/assets.py digest <directory> [--exclude .git/]
                                                 maintainers: the sha256 of a directory asset
    scripts/infrathon/assets.py export-dataset <dest> [<task directory>... | --all] [--image IMAGE]
                                                 maintainers: the dataset upload tree (--image
                                                 copies root-owned assets out)
    scripts/infrathon/assets.py context <task directory> <prefix> <source> <dest> [--allow-unlisted]

``context`` is for a task's build_images.sh: it copies exactly the manifest's
assets under one prefix (e.g. ``judge/private_data``) from ``source`` into a
fresh build context.  ``--allow-unlisted`` lets ``source`` hold other files,
which are left out (a host asset directory that also keeps a seed, say).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TASKS_ROOT = REPOSITORY_ROOT / "tasks" / "latest"
DEFAULT_HF_DATASET = "Infrathon/Infrathon"
SCHEMA_VERSION = 1
VISIBILITIES = ("public", "judge", "hw")
SOURCE_KINDS = ("infrathon-dataset", "hf", "git", "url")
# Fetched from their own upstream, never part of our dataset.
UPSTREAM_KINDS = ("hf", "git", "url")
# Hardware type -> the GPU name nvidia-smi reports for it.
HARDWARE_TYPES = {
    "h100-94gb-400w": "NVIDIA H100 NVL",
    "h100-80gb-700w": "NVIDIA H100 80GB HBM3",
}
# uid of ``agent`` in the task images: candidate code runs as it, so an owner-root asset must not be its.
AGENT_UID = 1001


class AssetError(RuntimeError):
    pass


def assets_root() -> Path:
    value = os.environ.get("INFRATHON_ASSETS_ROOT")
    if value:
        return Path(value).expanduser()
    cache = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(cache) / "infrathon" / "assets"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_files(root: Path, follow_symlinks: bool = False, exclude: str | None = None) -> list[str]:
    """Regular files under ``root`` (POSIX relative paths, sorted); refuses symlinks,
    except, with ``follow_symlinks``, links to regular files (a Hugging Face cache
    snapshot links every file to its blob).  ``exclude`` leaves out one directory of
    download records (``.git/``, ``.cache/huggingface/``)."""
    files = []
    for path in root.rglob("*"):
        if exclude and (path.relative_to(root).as_posix() + "/").startswith(exclude):
            continue
        if path.is_symlink():
            if not (follow_symlinks and path.resolve().is_file()):
                raise AssetError(f"symlink in asset directory: {path}")
            files.append(path.relative_to(root).as_posix())
        elif path.is_file():
            files.append(path.relative_to(root).as_posix())
    return sorted(files)


def tree_digest(root: Path, follow_symlinks: bool = False, exclude: str | None = None) -> str:
    """SHA-256 of ``<sha256>  <relative path>\\n`` lines, sorted by path (``sha256sum`` format)."""
    listing = "".join(f"{sha256_file(root / name)}  {name}\n" for name in tree_files(root, follow_symlinks, exclude))
    return hashlib.sha256(listing.encode("utf-8")).hexdigest()


STAMP_DIRECTORY = ".infrathon-verified"
PARTIAL_DIRECTORY = ".infrathon-partial"


def _stamp_path(root: Path, task: str, asset: dict) -> Path:
    return root / STAMP_DIRECTORY / task / (asset["path"].rstrip("/") + ".json")


def _stat_listing(root: Path, follow_symlinks: bool = False, exclude: str | None = None) -> list[list]:
    listing = []
    for name in tree_files(root, follow_symlinks, exclude):
        st = (root / name).stat()
        listing.append([name, st.st_size, st.st_mtime_ns])
    return listing


def _write_stamp(stamp: Path | None, target: Path, digest: str, follow_symlinks: bool = False,
                 exclude: str | None = None) -> None:
    if stamp is None:
        return
    stamp.parent.mkdir(parents=True, exist_ok=True)
    stamp.write_text(json.dumps({"sha256": digest, "files": _stat_listing(target, follow_symlinks, exclude)}),
                     encoding="utf-8")


def hub_cache() -> Path:
    """The Hugging Face hub cache huggingface_hub uses (HF_HUB_CACHE, HF_HOME/hub, ~/.cache/huggingface/hub)."""
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"]).expanduser()
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]).expanduser() / "hub"
    cache = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(cache) / "huggingface" / "hub"


def _cached(asset: dict) -> bool:
    return asset["source"]["kind"] == "hf" and asset["source"].get("layout", "cache") == "cache"


def _records(asset: dict) -> str | None:
    """The download records a directory asset keeps beside its content, left out of its digest."""
    source = asset["source"]
    if source["kind"] == "git":
        return ".git/"
    if source["kind"] == "hf" and source.get("layout") == "local-dir":
        return ".cache/huggingface/"
    return None


def _hex40(value: str) -> bool:
    return len(value) == 40 and all(c in "0123456789abcdef" for c in value)


def _hf_cache_snapshot(asset: dict) -> Path:
    source = asset["source"]
    folder = ("models--" if source["repo_type"] == "model" else "datasets--") + source["repo_id"].replace("/", "--")
    return hub_cache() / folder / "snapshots" / source["revision"]


def _record_main_ref(asset: dict) -> None:
    """Give a cache-layout snapshot a refs/main when its repository has none.

    snapshot_download at a commit writes no ref, but launchers that take a model repository
    folder (kvpress, the Miles tasks) resolve refs/main -> snapshots/<commit>.  An existing
    ref, for example from another tool, is left as it is."""
    if not _cached(asset):
        return
    snapshot = _hf_cache_snapshot(asset)
    ref = snapshot.parent.parent / "refs" / "main"
    if snapshot.is_dir() and not ref.exists():
        ref.parent.mkdir(parents=True, exist_ok=True)
        ref.write_text(asset["source"]["revision"], encoding="utf-8")


def is_tree(asset: dict) -> bool:
    return asset["path"].endswith("/")


def load_manifest(task: str) -> dict:
    path = TASKS_ROOT / task / "assets" / "manifest.json"
    if not path.is_file():
        raise AssetError(f"{task}: no assets/manifest.json")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise AssetError(f"{path}: schema_version must be {SCHEMA_VERSION}")
    if manifest.get("task_directory") != task:
        raise AssetError(f"{path}: task_directory must be {task!r}")
    for asset in manifest.get("assets", []):
        relative = Path(asset["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise AssetError(f"{path}: asset path must stay inside the task: {relative}")
        if relative.parts[0] not in VISIBILITIES:
            raise AssetError(f"{path}: asset path must start with one of {VISIBILITIES}: {relative}")
        if relative.parts[0] == "hw" and (len(relative.parts) < 3 or relative.parts[1] not in HARDWARE_TYPES):
            raise AssetError(f"{path}: hw/ assets must be under hw/<one of {sorted(HARDWARE_TYPES)}>/: {relative}")
        source = asset.get("source", {})
        if source.get("kind") not in SOURCE_KINDS:
            raise AssetError(f"{path}: unsupported source for {relative}: {source}")
        if source["kind"] == "hf":
            if (not asset["path"].endswith("/") or _judge_only(asset["path"])
                    or source.get("layout", "cache") not in ("cache", "flat", "local-dir")
                    or source.get("repo_type") not in ("model", "dataset") or "/" not in source.get("repo_id", "")
                    or not _hex40(source.get("revision", ""))):
                raise AssetError(f"{path}: an hf asset is a public directory with repo_id, repo_type and a "
                                 f"40-hex commit revision: {relative}")
        if source["kind"] == "git" and (not asset["path"].endswith("/") or _judge_only(asset["path"])
                                        or not source.get("url", "").startswith("https://")
                                        or not _hex40(source.get("commit", ""))):
            raise AssetError(f"{path}: a git asset is a public directory with an https url and a 40-hex "
                             f"commit: {relative}")
        if source["kind"] == "url" and (asset["path"].endswith("/") or _judge_only(asset["path"])
                                        or not source.get("url", "").startswith("https://")):
            raise AssetError(f"{path}: a url asset is one public file with an https url: {relative}")
        if len(asset.get("sha256", "")) != 64:
            raise AssetError(f"{path}: {relative} needs a 64-hex sha256")
        if asset.get("owner") not in (None, "root") or (asset.get("owner") and not _judge_only(asset["path"])):
            raise AssetError(f"{path}: owner may only be \"root\", on a Judge-only asset: {relative}")
    return manifest


def all_tasks() -> list[str]:
    return sorted(p.parent.parent.name for p in TASKS_ROOT.glob("*/assets/manifest.json"))


def _judge_only(asset_path: str) -> bool:
    parts = Path(asset_path).parts
    return parts[0] == "judge" or (parts[0] == "hw" and len(parts) > 3 and parts[2] == "judge")


def detect_hardware() -> str | None:
    """``INFRATHON_HARDWARE``, or the hardware type of GPU 0 (None when unknown)."""
    value = os.environ.get("INFRATHON_HARDWARE")
    if value:
        if value not in HARDWARE_TYPES:
            raise AssetError(f"INFRATHON_HARDWARE must be one of {sorted(HARDWARE_TYPES)}: {value}")
        return value
    try:
        completed = subprocess.run(["nvidia-smi", "--id=0", "--query-gpu=name", "--format=csv,noheader"],
                                   capture_output=True, text=True, check=False, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    name = completed.stdout.strip() if completed.returncode == 0 else ""
    return next((hw for hw, gpu in HARDWARE_TYPES.items() if gpu == name), None)


def for_hardware(assets: list[dict], hardware: str | None, task: str) -> list[dict]:
    """The assets a host of ``hardware`` needs: everything outside hw/, plus hw/<hardware>/."""
    if hardware is None and any(a["path"].startswith("hw/") for a in assets):
        raise AssetError(f"{task} has hardware-specific assets: pass --hardware or set INFRATHON_HARDWARE "
                         f"to one of {sorted(HARDWARE_TYPES)}")
    return [a for a in assets if not a["path"].startswith("hw/") or a["path"].startswith(f"hw/{hardware}/")]


def _install_file(source: Path, target: Path, expected: str, judge_only: bool) -> None:
    """Copy source to target atomically, then verify the digest."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if judge_only:
        os.chmod(target.parent, 0o700)
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        shutil.copyfile(source, temporary)
        actual = sha256_file(temporary)
        if actual != expected:
            raise AssetError(f"{source}: sha256 {actual} does not match the manifest {expected}")
        os.chmod(temporary, 0o600 if judge_only else 0o644)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _install_tree(source: Path, target: Path, expected: str, judge_only: bool, from_cache: bool = False,
                  stamp: Path | None = None, follow_symlinks: bool = False) -> None:
    """Copy the directory ``source`` to ``target`` (replacing it), then verify its digest.

    ``from_cache``: ``source`` is a Hugging Face cache snapshot; its file links are
    followed and the blobs hard-linked when possible (the cache keeps them read-only
    in practice; the asset root holds plain files a container can mount).
    ``follow_symlinks``: ``source`` is a dataset snapshot in the Hugging Face cache; its
    links are followed too, and Judge-only files are copied, not linked, so they get
    their own 0600 mode."""
    if not source.is_dir() or source.is_symlink():
        raise AssetError(f"asset directory source must be a directory: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if judge_only:
        os.chmod(target.parent, 0o700)
    temporary = Path(tempfile.mkdtemp(dir=target.parent, prefix=f".{target.name}."))
    follow = from_cache or follow_symlinks
    linked: set[Path] = set()
    try:
        for name in tree_files(source, follow_symlinks=follow):
            destination = temporary / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            origin = (source / name).resolve() if follow else source / name
            try:
                if not (from_cache or (follow_symlinks and not judge_only)):
                    raise OSError
                os.link(origin, destination)
                linked.add(destination)
            except OSError:
                shutil.copyfile(origin, destination)
        actual = tree_digest(temporary)
        if actual != expected:
            raise AssetError(f"{source}: tree sha256 {actual} does not match the manifest {expected}")
        for path in [temporary, *temporary.rglob("*")]:
            if path.is_dir():
                os.chmod(path, 0o700 if judge_only else 0o755)
            elif path not in linked:  # a hard link shares the cache blob's mode
                os.chmod(path, 0o600 if judge_only else 0o644)
        if target.exists():
            shutil.rmtree(target)
        os.replace(temporary, target)
        _write_stamp(stamp, target, expected)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _owner_only(asset: dict) -> bool:
    return _judge_only(asset["path"]) and not asset.get("readable", False)


def _install(source: Path, target: Path, asset: dict, stamp: Path | None = None,
             follow_symlinks: bool = False) -> None:
    if is_tree(asset):
        _install_tree(source, target, asset["sha256"], _owner_only(asset), stamp=stamp,
                      follow_symlinks=follow_symlinks)
    else:
        _install_file(source, target, asset["sha256"], _owner_only(asset))


def _target(root: Path, task: str, asset: dict) -> Path:
    if _cached(asset):
        return _hf_cache_snapshot(asset)
    return root / task / asset["path"].rstrip("/")


def _current(path: Path, asset: dict, stamp: Path | None = None, full: bool = False) -> bool:
    follow = _cached(asset)  # a cache snapshot links every file to its blob
    records = _records(asset)
    if path.is_symlink():
        return False
    if not is_tree(asset):
        return path.is_file() and sha256_file(path) == asset["sha256"]
    if not path.is_dir():
        return False
    if stamp is not None and not full:
        try:
            recorded = json.loads(stamp.read_text(encoding="utf-8"))
            if (recorded.get("sha256") == asset["sha256"]
                    and recorded.get("files") == _stat_listing(path, follow, records)):
                return True
        except (OSError, ValueError, AssetError):
            pass
    if tree_digest(path, follow, records) != asset["sha256"]:
        return False
    _write_stamp(stamp, path, asset["sha256"], follow, records)
    return True


def _root_owned(path: Path) -> bool:
    try:
        st = path.lstat()
    except OSError:
        return False
    return path.is_dir() and st.st_uid == 0 and not os.access(path, os.R_OK | os.X_OK)


def check(task: str, root: Path, hardware: str | None = None, full: bool = False) -> list[str]:
    problems = []
    for asset in for_hardware(load_manifest(task)["assets"], hardware, task):
        target = _target(root, task, asset)
        if asset.get("owner") == "root" and _root_owned(target):
            print(f"note: {task}/{asset['path']} is root-owned (assets.py own); not re-hashed", file=sys.stderr)
            continue
        if not _current(target, asset, _stamp_path(root, task, asset), full):
            problems.append(f"{task}: {asset['path']} missing or wrong sha256 at {target}")
    return problems


def _agent_owned(path: Path) -> bool:
    try:
        return path.lstat().st_uid == AGENT_UID
    except OSError:
        return False


def own(task: str, root: Path, hardware: str | None, image: str, agent_owned: bool = False) -> None:
    """Hand the task's owner-root assets to root:root, go-rwx, through ``image`` (no sudo).  With
    ``agent_owned``, only those the agent uid owns; missing ones are left to ``fetch``."""
    for asset in for_hardware(load_manifest(task)["assets"], hardware, task):
        if asset.get("owner") != "root":
            continue
        target = _target(root, task, asset)
        if _root_owned(target):
            print(f"ok      {task}/{asset['path']} (root-owned)")
            continue
        if agent_owned and not _agent_owned(target):
            continue
        if not _current(target, asset, _stamp_path(root, task, asset)):
            raise AssetError(f"{task}: {asset['path']} is missing or wrong at {target}; fetch it first")
        subprocess.run(["docker", "run", "--rm", "--pull=never", "--network", "none", "--user", "0",
                        "--entrypoint", "sh", "-v", f"{target}:/target", image, "-c",
                        "chown -R 0:0 /target && chmod -R go-rwx /target"], check=True)
        print(f"owned   {task}/{asset['path']} -> root:root, go-rwx")


def _hf_snapshot(asset: dict) -> Path:
    """The pinned snapshot in the Hugging Face cache, downloading what is missing."""
    source = asset["source"]
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise AssetError("fetching from Hugging Face needs huggingface_hub") from exc
    try:
        return Path(snapshot_download(repo_id=source["repo_id"], repo_type=source["repo_type"],
                                      revision=source["revision"], allow_patterns=source.get("allow_patterns")))
    except Exception as exc:  # huggingface_hub raises many transport/auth types
        raise AssetError(f"cannot download {source['repo_id']}@{source['revision'][:12]} from Hugging Face "
                         f"(gated repositories need `hf auth login` first): {type(exc).__name__}: {exc}") from exc


def _fill_hf_local_dir(asset: dict, directory: Path) -> None:
    """``hf download --local-dir``: the files and their revision records, resuming what is there."""
    source = asset["source"]
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise AssetError("fetching from Hugging Face needs huggingface_hub") from exc
    try:
        snapshot_download(repo_id=source["repo_id"], repo_type=source["repo_type"], revision=source["revision"],
                          allow_patterns=source.get("allow_patterns"), local_dir=directory)
    except Exception as exc:  # huggingface_hub raises many transport/auth types
        raise AssetError(f"cannot download {source['repo_id']}@{source['revision'][:12]} from Hugging Face "
                         f"(gated repositories need `hf auth login` first): {type(exc).__name__}: {exc}") from exc


def _fill_git(asset: dict, directory: Path) -> None:
    """Check ``directory`` out at the pinned commit: a fresh clone, or the clone an earlier
    attempt left; then the Git LFS files when ``lfs``."""
    source = asset["source"]
    if source.get("lfs") and shutil.which("git-lfs") is None:
        raise AssetError(f"{source['url']} keeps files in Git LFS: install git-lfs (https://git-lfs.com) first")

    def git(*args: str, smudge: bool = False) -> str:
        environment = dict(os.environ, GIT_TERMINAL_PROMPT="0")
        if not smudge:
            environment["GIT_LFS_SKIP_SMUDGE"] = "1"
        completed = subprocess.run(["git", "-C", str(directory), *args], env=environment,
                                   capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            raise AssetError(f"git {args[0]} failed for {source['url']}: {completed.stderr.strip()[-400:]}")
        return completed.stdout.strip()

    if not (directory / ".git").is_dir():
        git("clone", "--no-checkout", source["url"], ".")
    git("checkout", "--force", "--detach", source["commit"])
    if source.get("lfs"):
        git("lfs", "install", "--local")
        git("lfs", "pull", smudge=True)
    if git("rev-parse", "HEAD") != source["commit"]:
        raise AssetError(f"{source['url']}: the checkout is not at {source['commit']}")


def _install_upstream_tree(root: Path, task: str, asset: dict, target: Path, stamp: Path | None) -> None:
    """Fill a resumable staging directory from the asset's upstream, verify it, then move it
    to ``target``.  A ``target`` that is no longer current is taken as the starting point."""
    partial = root / PARTIAL_DIRECTORY / task / asset["path"].rstrip("/")
    if not partial.exists() and target.is_dir() and not target.is_symlink():
        partial.parent.mkdir(parents=True, exist_ok=True)
        os.replace(target, partial)
    partial.mkdir(parents=True, exist_ok=True)
    (_fill_git if asset["source"]["kind"] == "git" else _fill_hf_local_dir)(asset, partial)
    records = _records(asset)
    actual = tree_digest(partial, exclude=records)
    if actual != asset["sha256"]:
        raise AssetError(f"{task}: {asset['path']}: tree sha256 {actual} does not match the manifest "
                         f"{asset['sha256']} (remove {partial} to start over)")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        shutil.rmtree(target)
    os.replace(partial, target)
    _write_stamp(stamp, target, asset["sha256"], exclude=records)


def _install_url(root: Path, task: str, asset: dict, target: Path) -> None:
    """Download a ``url`` asset and install it once its sha256 matches."""
    url = asset["source"]["url"]
    download = root / PARTIAL_DIRECTORY / task / asset["path"]
    download.parent.mkdir(parents=True, exist_ok=True)
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "infrathon-assets"})
        with urllib.request.urlopen(request, timeout=120) as response, download.open("wb") as handle:
            shutil.copyfileobj(response, handle, 1 << 20)
        actual = sha256_file(download)
        if actual != asset["sha256"]:
            raise AssetError(f"{task}: {url}: sha256 {actual} does not match the manifest {asset['sha256']}")
        os.chmod(download, 0o644)
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(download, target)
    except OSError as exc:  # urllib's errors are OSErrors
        raise AssetError(f"{task}: cannot download {url}: {type(exc).__name__}: {exc}") from exc
    finally:
        download.unlink(missing_ok=True)


def _download(task: str, asset: dict) -> Path:
    relative = asset["path"].rstrip("/")
    source_root = os.environ.get("INFRATHON_ASSETS_SOURCE")
    if source_root:
        source = Path(source_root).expanduser() / task / relative
        if not (source.is_dir() if is_tree(asset) else source.is_file()):
            raise AssetError(f"{task}: {asset['path']} is not in INFRATHON_ASSETS_SOURCE={source_root}")
        return source
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise AssetError("fetching from Hugging Face needs huggingface_hub") from exc
    repo_id = os.environ.get("INFRATHON_HF_DATASET", DEFAULT_HF_DATASET)
    revision = os.environ.get("INFRATHON_HF_REVISION")
    try:
        if is_tree(asset):
            snapshot = snapshot_download(repo_id=repo_id, repo_type="dataset", revision=revision,
                                         allow_patterns=[f"{task}/{relative}/**"])
            return Path(snapshot) / task / relative
        return Path(hf_hub_download(repo_id=repo_id, repo_type="dataset", revision=revision,
                                    filename=f"{task}/{relative}"))
    except Exception as exc:  # huggingface_hub raises many transport/auth types
        raise AssetError(
            f"{task}: cannot download {asset['path']} from dataset {repo_id}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def _local_image(task: str) -> str | None:
    """The task's local Judge or Work image (images/release.json local_tags), if Docker has one."""
    try:
        tags = json.loads((TASKS_ROOT / task / "images/release.json").read_text(encoding="utf-8"))["local_tags"]
    except (OSError, ValueError, KeyError):
        return None
    for role in ("judge_public", "judge", "work"):
        if tags.get(role) and subprocess.run(["docker", "image", "inspect", tags[role]], stdout=subprocess.DEVNULL,
                                             stderr=subprocess.DEVNULL, check=False).returncode == 0:
            return tags[role]
    return None


def _own_agent_owned(task: str, root: Path, hardware: str | None, image: str | None) -> None:
    """After ``fetch``: hand the owner-root assets the agent uid owns (a host user with that uid
    fetched them) to root, or say how."""
    owned = [asset for asset in for_hardware(load_manifest(task)["assets"], hardware, task)
             if asset.get("owner") == "root" and _agent_owned(_target(root, task, asset))]
    if not owned:
        return
    try:
        image = image or _local_image(task)
    except OSError:  # no docker client
        image = None
    if image is not None:
        own(task, root, hardware, image, agent_owned=True)
        return
    for asset in owned:
        print(f"note: {task}/{asset['path']} is owned by uid {AGENT_UID}, the uid candidate code runs as; "
              "the task refuses it until it is handed to root", file=sys.stderr)
    print(f"note: no local image of {task} to do that with: scripts/infrathon/pull_images.sh {task} does it once it "
          f"has pulled the images, or run, with any local image that has sh, chown and chmod (no sudo needed):\n"
          f"  INFRATHON_ASSETS_ROOT={root} {sys.executable} {Path(__file__).resolve()} own {task} --image <image>"
          + (f" --hardware {hardware}" if hardware else ""), file=sys.stderr)


def fetch(task: str, root: Path, hardware: str | None = None, image: str | None = None) -> None:
    for asset in for_hardware(load_manifest(task)["assets"], hardware, task):
        target, stamp = _target(root, task, asset), _stamp_path(root, task, asset)
        if asset.get("owner") == "root" and _root_owned(target):
            print(f"ok      {task}/{asset['path']} (root-owned, not re-hashed)")
            continue
        if _current(target, asset, stamp):
            _record_main_ref(asset)
            print(f"ok      {task}/{asset['path']}")
            continue
        if _cached(asset):
            snapshot = _hf_snapshot(asset)
            if not _current(snapshot, asset, stamp, full=True):
                raise AssetError(f"{task}: {asset['path']}: the downloaded snapshot {snapshot} does not match "
                                 f"the manifest sha256 {asset['sha256']}")
            _record_main_ref(asset)
        elif asset["source"]["kind"] == "git" or asset["source"].get("layout") == "local-dir":
            _install_upstream_tree(root, task, asset, target, stamp)
        elif asset["source"]["kind"] == "hf":
            _install_tree(_hf_snapshot(asset), target, asset["sha256"], judge_only=False, from_cache=True, stamp=stamp)
        elif asset["source"]["kind"] == "url":
            _install_url(root, task, asset, target)
        else:
            # From Hugging Face the download is a cache snapshot, whose files are links to blobs.
            _install(_download(task, asset), target, asset, stamp,
                     follow_symlinks=not os.environ.get("INFRATHON_ASSETS_SOURCE"))
        print(f"fetched {task}/{asset['path']}")
    _own_agent_owned(task, root, hardware, image)


def remote_origin(origin: str) -> bool:
    """``host:/path`` names a file on another host (copied into the root by hand)."""
    head = origin.split("/", 1)[0]
    return ":" in head


def stage(task: str, root: Path) -> None:
    for asset in load_manifest(task)["assets"]:
        if asset["source"]["kind"] in UPSTREAM_KINDS:
            print(f"public  {task}/{asset['path']} (fetched from its upstream, not part of the dataset)")
            continue
        target, stamp = _target(root, task, asset), _stamp_path(root, task, asset)
        if _current(target, asset, stamp):
            print(f"ok      {task}/{asset['path']}")
            continue
        origin = asset.get("staged_from")
        if not origin:
            raise AssetError(f"{task}: {asset['path']} has no staged_from")
        if remote_origin(origin):
            raise AssetError(f"{task}: {asset['path']} lives on another host; copy it first: "
                             f"scp -r {origin} {target}")
        # ``$VAR/...`` names a host asset root (e.g. the Miles roots that
        # scripts/miles_common/host_config.py resolves); it must be set to stage.
        expanded = os.path.expandvars(origin)
        if "$" in expanded:
            raise AssetError(f"{task}: set the host variable(s) in staged_from first: {origin}")
        source = Path(expanded).expanduser()
        if not source.is_absolute():
            source = TASKS_ROOT / task / source
        if source.is_symlink() or not (source.is_dir() if is_tree(asset) else source.is_file()):
            raise AssetError(f"{task}: staged_from must be a regular {'directory' if is_tree(asset) else 'file'}: {source}")
        _install(source, target, asset, stamp)
        print(f"staged  {task}/{asset['path']}")


def _export_root_owned(source: Path, target: Path, asset: dict, image: str | None) -> tuple[int, int]:
    """Copy a root-owned (``assets.py own``) directory out through ``image`` as the host
    user's files, then verify the copy: the host user can neither read nor link the original."""
    if not image:
        raise AssetError(f"{source} is root-owned; export-dataset needs --image (a local image with sh and cp)")
    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["docker", "run", "--rm", "--pull=never", "--network", "none", "--user", "0",
                    "--entrypoint", "sh", "-v", f"{source}:/source:ro", "-v", f"{target.parent}:/export", image,
                    "-c", f'cp -r /source "/export/{target.name}" && chown -R {os.getuid()}:{os.getgid()} '
                          f'"/export/{target.name}" && chmod -R u+rwX "/export/{target.name}"'], check=True)
    if tree_digest(target) != asset["sha256"]:
        raise AssetError(f"the export of root-owned {source} does not match the manifest sha256")
    names = tree_files(target)
    return len(names), sum((target / name).stat().st_size for name in names)


def export_dataset(tasks: list[str], root: Path, destination: Path, image: str | None = None) -> None:
    """Lay out exactly the ``infrathon-dataset`` assets of ``tasks`` (every hardware type) under
    ``destination`` as hard links into the asset root, verified first: the tree to upload as the
    Hugging Face dataset.  Public Hugging Face assets and the stamps are left out.  A root-owned
    owner-root asset is copied out through ``image`` instead and verified after the copy."""
    count = size = 0
    for task in tasks:
        for asset in load_manifest(task)["assets"]:
            if asset["source"]["kind"] != "infrathon-dataset":
                continue
            source = _target(root, task, asset)
            if asset.get("owner") == "root" and _root_owned(source):
                files, nbytes = _export_root_owned(source, destination / task / asset["path"].rstrip("/"),
                                                   asset, image)
                count += files
                size += nbytes
                continue
            if not _current(source, asset, _stamp_path(root, task, asset)):
                raise AssetError(f"{task}: {asset['path']} is missing or wrong at {source}; stage it first")
            names = tree_files(source) if is_tree(asset) else [""]
            for name in names:
                origin = source / name if name else source
                target = destination / task / asset["path"].rstrip("/") / name if name else destination / task / asset["path"]
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    try:
                        os.link(origin, target)
                    except OSError:
                        shutil.copyfile(origin, target)
                count += 1
                size += origin.stat().st_size
    print(f"exported {count} files, {size / 2**30:.1f} GiB, to {destination}")


def build_context(task: str, prefix: str, source: Path, destination: Path,
                  allow_unlisted: bool = False) -> None:
    """Copy exactly the manifest's assets under ``prefix`` from ``source`` into a
    fresh Judge-only build context, verifying every SHA-256.

    ``prefix`` names one listed file (``source`` is then that file and it lands
    as ``<destination>/<name>``) or a directory of listed files and directories
    (``source`` holds them relative to ``prefix``: the task's committed copy, a
    host asset directory, or ``<asset root>/<task>/<prefix>``).  A missing asset,
    a symlink, or, without ``allow_unlisted``, a file the manifest does not
    cover is refused, so the context is exactly the reviewed asset set.
    """
    assets = load_manifest(task)["assets"]
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    os.chmod(destination, 0o700)
    single = next((a for a in assets if not is_tree(a) and a["path"] == prefix.rstrip("/")), None)
    if single is not None:
        if not source.is_file() or source.is_symlink():
            raise AssetError(f"{task}: {prefix} source must be a regular file: {source}")
        _install_file(source, destination / Path(single["path"]).name, single["sha256"], judge_only=True)
        print(f"staged 1 Judge-only file from {source} into {destination}")
        return
    prefix = prefix.rstrip("/") + "/"
    selected = [a for a in assets if a["path"].startswith(prefix) and a["path"] != prefix]
    if not selected:
        raise AssetError(f"{task}: the manifest lists no assets under {prefix}")
    if not source.is_dir() or source.is_symlink():
        raise AssetError(f"{task}: {prefix} source must be a directory: {source}")
    covered_files = {a["path"][len(prefix):] for a in selected if not is_tree(a)}
    covered_trees = [a["path"][len(prefix):] for a in selected if is_tree(a)]
    present = tree_files(source)
    missing = sorted(name for name in covered_files if name not in present)
    missing += sorted(tree for tree in covered_trees if not (source / tree).is_dir())
    unlisted = sorted(name for name in present
                      if name not in covered_files and not any(name.startswith(t) for t in covered_trees))
    if missing or (unlisted and not allow_unlisted):
        raise AssetError(f"{task}: {source} does not match the manifest under {prefix}: "
                         f"unlisted {unlisted[:10]}{'...' if len(unlisted) > 10 else ''}, missing {missing}")
    for asset in selected:
        relative = asset["path"][len(prefix):].rstrip("/")
        if is_tree(asset):
            _install_tree(source / relative, destination / relative, asset["sha256"], judge_only=True)
        else:
            _install_file(source / relative, destination / relative, asset["sha256"], judge_only=True)
    print(f"staged {len(selected)} Judge-only asset(s) from {source} into {destination}"
          + (f" ({len(unlisted)} unlisted file(s) left out)" if unlisted else ""))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("fetch", "check", "stage", "context", "hardware", "hub-cache", "digest", "own",
                                            "export-dataset"))
    parser.add_argument("tasks", nargs="*", help="task directory names under tasks/latest")
    parser.add_argument("--all", action="store_true", help="every task that has assets/manifest.json")
    parser.add_argument("--allow-unlisted", action="store_true", help="context: leave out unlisted source files")
    parser.add_argument("--hardware", choices=sorted(HARDWARE_TYPES),
                        help="fetch/check: this host's hardware type (default INFRATHON_HARDWARE, else detected)")
    parser.add_argument("--full", action="store_true", help="check: re-hash directories even when stamped")
    parser.add_argument("--image", help="own, export-dataset, fetch: a local image with sh, cp, chown and chmod "
                                        "(e.g. the task's Judge image) to handle root-owned assets")
    parser.add_argument("--agent-owned", action="store_true",
                        help=f"own: only the assets uid {AGENT_UID} owns; skip missing ones")
    parser.add_argument("--exclude", help="digest: leave out this directory of download records (.git/, "
                                          ".cache/huggingface/)")
    args = parser.parse_args()
    try:
        if args.command == "hub-cache":
            print(hub_cache())
            return 0
        if args.command == "export-dataset":
            if not args.tasks:
                parser.error("export-dataset takes <destination> [<task directory>... | --all]")
            destination, chosen = Path(args.tasks[0]), (all_tasks() if args.all else args.tasks[1:])
            if not chosen:
                parser.error("name task directories or pass --all")
            export_dataset(chosen, assets_root(), destination, args.image)
            return 0
        if args.command == "digest":
            if args.all or len(args.tasks) != 1:
                parser.error("digest takes one directory")
            directory = Path(args.tasks[0])
            files = tree_files(directory, follow_symlinks=True, exclude=args.exclude)
            size = sum((directory / name).resolve().stat().st_size for name in files)
            print(json.dumps({"sha256": tree_digest(directory, follow_symlinks=True, exclude=args.exclude),
                              "files": len(files), "size": size}))
            return 0
        if args.command == "hardware":
            hardware = args.hardware or detect_hardware()
            if hardware is None:
                print(f"ERROR: cannot tell the hardware type; set INFRATHON_HARDWARE to one of "
                      f"{sorted(HARDWARE_TYPES)}", file=sys.stderr)
                return 1
            print(hardware)
            return 0
        if args.command == "context":
            if args.all or len(args.tasks) != 4:
                parser.error("context takes <task directory> <asset prefix> <source> <destination>")
            task, prefix, source, destination = args.tasks
            build_context(task, prefix, Path(source), Path(destination), args.allow_unlisted)
            return 0
        tasks = all_tasks() if args.all else args.tasks
        if not tasks:
            parser.error("name at least one task directory or pass --all")
        root = assets_root()
        hardware = args.hardware or (detect_hardware() if args.command in ("fetch", "check", "own") else None)
        if args.command == "own":
            if not args.image:
                parser.error("own needs --image (a local image with sh, chown and chmod)")
            for task in tasks:
                own(task, root, hardware, args.image, args.agent_owned)
            return 0
        if args.command == "check":
            problems = [problem for task in tasks for problem in check(task, root, hardware, args.full)]
            for problem in problems:
                print(problem, file=sys.stderr)
            if problems:
                return 1
            print(f"assets current under {root}: {', '.join(tasks)}")
            return 0
        for task in tasks:
            if args.command == "fetch":
                fetch(task, root, hardware, args.image)
            else:
                stage(task, root)
    except AssetError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
