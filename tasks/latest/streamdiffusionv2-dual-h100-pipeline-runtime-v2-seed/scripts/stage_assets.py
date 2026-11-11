#!/usr/bin/env python3
"""Assemble ``$SDV2_ASSETS_ROOT`` from public pins and in-tree inputs.

Every evaluation input of this task is one of three P1 classes, and this script
is the only definition of how the asset root is built:

* **public-pinned** — Wan2.1-T2V-1.3B and the causal/DMD v2v checkpoint from
  Hugging Face at pinned revisions, the prebuilt flash-attn wheel from the
  upstream GitHub release, the StreamDiffusionV2 source at a pinned commit, the
  Blender Foundation films (through ``fetch_clips.py``);
* **generated** — the six media clips, cut from the films and verified against
  ``assets/clips/clips_manifest.json``;
* **in-tree** — the prompt table, the 6 public and 12 hidden umT5 prompt
  embeddings, the checkpoint manifest and the smoke loader.  The hidden
  embeddings live on the Judge-only path ``judge/private/`` and are placed
  only in the Judge's build context.

Layout produced (default root ``build/assets``)::

    weights/{checkpoint_manifest.sha256, ckpts/..., wan_models/...}
    wheels/<flash-attn wheel>
    upstream/StreamDiffusionV2/{models/, LICENSE, PROVENANCE.txt, smoke_load.py}
    clips/*.npy + clips_manifest.json + THIRD_PARTY_NOTICES.md
    embeddings/{public,hidden}/*.pt
    prompts/public_prompts.json
    work-context/{prompts, embeddings/public, clips/<public clips only>}

``work-context/`` is the ``assets`` build context of the Work image, so the
hidden embeddings and hidden clips never enter it.  Every file is verified
against its recorded SHA-256; a mismatch fails the build.
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

TASK_ROOT = Path(__file__).resolve().parents[1]
IN_TREE = TASK_ROOT / "assets"
JUDGE_PRIVATE = TASK_ROOT / "judge" / "private"

HF_WEIGHTS = (
    # (repo, revision, repo path, destination relative to weights/)
    ("Wan-AI/Wan2.1-T2V-1.3B", "37ec512624d61f7aa208f7ea8140a131f93afc9a",
     "config.json", "wan_models/Wan2.1-T2V-1.3B/config.json"),
    ("Wan-AI/Wan2.1-T2V-1.3B", "37ec512624d61f7aa208f7ea8140a131f93afc9a",
     "diffusion_pytorch_model.safetensors", "wan_models/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors"),
    ("Wan-AI/Wan2.1-T2V-1.3B", "37ec512624d61f7aa208f7ea8140a131f93afc9a",
     "Wan2.1_VAE.pth", "wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"),
    ("jerryfeng/StreamDiffusionV2", "2373eb2b39278b3a1aa174964a724ee78ead96f0",
     "wan_causal_dmd_v2v/model.pt", "ckpts/wan_causal_dmd_v2v/model.pt"),
)

# The Dockerfiles name this wheel file explicitly; keep the three in step.
FLASH_WHEEL = {
    "name": "flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl",
    "url": "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/"
           "flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl",
    "sha256": "ffe17686fa1a0f288de9eae7c32af209d32a27b037ef28614f042b377af5b15a",
}

UPSTREAM_REPO = "https://github.com/chenfengxu714/StreamDiffusionV2.git"
UPSTREAM_COMMIT = "2f8ee1c5b6616b115662d190d5213677882d85d7"
UPSTREAM_PROVENANCE = (
    "Copied from vendor/StreamDiffusionV2 at pinned upstream commit "
    f"{UPSTREAM_COMMIT} for in-container model/VAE smoke imports.\n"
)

PUBLIC_CLIPS = ("bbb_480p_129.npy", "bbb_square_257_cut.npy", "sintel_480p_129.npy")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        target.unlink()
    try:
        os.link(source, target)
    except OSError:
        shutil.copyfile(source, target)


def stage_weights(root: Path) -> None:
    from huggingface_hub import hf_hub_download

    manifest = {}
    for line in (IN_TREE / "weights" / "checkpoint_manifest.sha256").read_text().splitlines():
        digest, _, relative = line.partition("  ")
        manifest[relative.strip()] = digest.strip()
    weights = root / "weights"
    for repo, revision, repo_path, relative in HF_WEIGHTS:
        target = weights / relative
        expected = manifest[relative]
        if target.is_file() and sha256_file(target) == expected:
            print(f"REUSED weights/{relative}", flush=True)
            continue
        print(f"fetching {repo}@{revision[:12]} {repo_path}", flush=True)
        cached = Path(hf_hub_download(repo, repo_path, revision=revision))
        observed = sha256_file(cached)
        if observed != expected:
            raise SystemExit(f"{repo}/{repo_path}: sha256 {observed} != {expected}")
        link_or_copy(cached.resolve(), target)
        print(f"BUILT  weights/{relative}", flush=True)
    shutil.copyfile(IN_TREE / "weights" / "checkpoint_manifest.sha256", weights / "checkpoint_manifest.sha256")
    subprocess.run(["sha256sum", "--check", "--strict", "--quiet", "checkpoint_manifest.sha256"],
                   cwd=weights, check=True)


def stage_wheel(root: Path) -> None:
    target = root / "wheels" / FLASH_WHEEL["name"]
    if target.is_file() and sha256_file(target) == FLASH_WHEEL["sha256"]:
        print(f"REUSED wheels/{FLASH_WHEEL['name']}", flush=True)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"fetching {FLASH_WHEEL['url']}", flush=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as handle:
        temporary = Path(handle.name)
        with urllib.request.urlopen(FLASH_WHEEL["url"]) as response:
            shutil.copyfileobj(response, handle, length=1 << 20)
    observed = sha256_file(temporary)
    if observed != FLASH_WHEEL["sha256"]:
        temporary.unlink()
        raise SystemExit(f"{FLASH_WHEEL['name']}: sha256 {observed} != {FLASH_WHEEL['sha256']}")
    temporary.replace(target)
    print(f"BUILT  wheels/{FLASH_WHEEL['name']}", flush=True)


def stage_upstream(root: Path, source_dir: Path) -> None:
    target = root / "upstream" / "StreamDiffusionV2"
    stamp = target / "PROVENANCE.txt"
    if stamp.is_file() and stamp.read_text() == UPSTREAM_PROVENANCE and (target / "models").is_dir():
        print("REUSED upstream/StreamDiffusionV2", flush=True)
        return
    checkout = source_dir / "StreamDiffusionV2"
    if not (checkout / ".git").is_dir():
        subprocess.run(["git", "clone", "--quiet", "--no-checkout", UPSTREAM_REPO, str(checkout)], check=True)
    subprocess.run(["git", "-C", str(checkout), "fetch", "--quiet", "origin", UPSTREAM_COMMIT], check=True)
    subprocess.run(["git", "-C", str(checkout), "checkout", "--quiet", "--force", UPSTREAM_COMMIT], check=True)
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    shutil.copytree(checkout / "models", target / "models",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copyfile(checkout / "LICENSE", target / "LICENSE")
    shutil.copyfile(IN_TREE / "upstream" / "smoke_load.py", target / "smoke_load.py")
    stamp.write_text(UPSTREAM_PROVENANCE)
    print("BUILT  upstream/StreamDiffusionV2", flush=True)


def stage_clips(root: Path, source_dir: Path, ffmpeg: str | None) -> None:
    command = [sys.executable, str(TASK_ROOT / "scripts" / "fetch_clips.py"),
               "--output", str(root / "clips"), "--source-dir", str(source_dir / "blender")]
    if ffmpeg:
        command += ["--ffmpeg", ffmpeg]
    subprocess.run(command, check=True)


def hidden_embeddings() -> Path:
    """The Judge-only hidden embeddings: the in-repository copy, else the dataset copy
    at $INFRATHON_ASSETS_ROOT/<task>/judge/embeddings/hidden (assets/fetch.sh)."""
    if JUDGE_PRIVATE.is_dir():
        return JUDGE_PRIVATE / "embeddings" / "hidden"
    assets = Path(os.environ.get("INFRATHON_ASSETS_ROOT") or Path.home() / ".cache" / "infrathon" / "assets")
    return assets / TASK_ROOT.name / "judge" / "embeddings" / "hidden"


def stage_in_tree(root: Path) -> None:
    for source, relative in (
        (IN_TREE / "prompts" / "public_prompts.json", "prompts/public_prompts.json"),
        *((p, f"embeddings/public/{p.name}") for p in sorted((IN_TREE / "embeddings" / "public").glob("*.pt"))),
        *((p, f"embeddings/hidden/{p.name}") for p in sorted(hidden_embeddings().glob("*.pt"))),
    ):
        link_or_copy(source, root / relative)
    if len(list((root / "embeddings" / "public").glob("*.pt"))) != 6:
        raise SystemExit("expected 6 public prompt embeddings in-tree")
    if len(list((root / "embeddings" / "hidden").glob("*.pt"))) != 12:
        raise SystemExit(f"expected 12 hidden prompt embeddings under {hidden_embeddings()} (run assets/fetch.sh)")


def stage_work_context(root: Path) -> None:
    work = root / "work-context"
    if work.exists():
        shutil.rmtree(work)
    link_or_copy(root / "prompts" / "public_prompts.json", work / "prompts" / "public_prompts.json")
    for path in sorted((root / "embeddings" / "public").glob("*.pt")):
        link_or_copy(path, work / "embeddings" / "public" / path.name)
    for name in PUBLIC_CLIPS:
        link_or_copy(root / "clips" / name, work / "clips" / name)
    private = [p for p in work.rglob("*") if "hidden" in p.parts]
    if private:
        raise SystemExit(f"work-context contains hidden inputs: {private}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path,
                        default=Path(os.environ.get("SDV2_ASSETS_ROOT") or TASK_ROOT / "build" / "assets"))
    parser.add_argument("--source-dir", type=Path,
                        help="scratch for the ~2 GiB film downloads and the upstream checkout (default: <root>/.sources)")
    parser.add_argument("--ffmpeg", help="ffmpeg binary for fetch_clips.py (default: PATH)")
    parser.add_argument("--skip", action="append", default=[],
                        choices=["weights", "wheel", "upstream", "clips"],
                        help="skip a public-pinned stage (repeatable)")
    args = parser.parse_args()
    root = args.root.resolve()
    source_dir = (args.source_dir or root / ".sources").resolve()
    root.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)
    stage_in_tree(root)
    if "weights" not in args.skip:
        stage_weights(root)
    if "wheel" not in args.skip:
        stage_wheel(root)
    if "upstream" not in args.skip:
        stage_upstream(root, source_dir)
    if "clips" not in args.skip:
        stage_clips(root, source_dir, args.ffmpeg)
    stage_work_context(root)
    print(json.dumps({"status": "ready", "root": str(root), "skipped": sorted(args.skip)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
