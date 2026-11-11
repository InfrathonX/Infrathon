#!/usr/bin/env python3
"""Regenerate the six benchmark media clips from the public Blender films.

The clips are P1 *generated* inputs: uint8 ``[T, H, W, 3]`` RGB arrays cut from
the Blender Foundation 1080p downloads (CC BY 3.0) at the frame ranges recorded
in ``assets/clips/clips_manifest.json`` and ``THIRD_PARTY_NOTICES.md``.  The
transform is fixed:

* accurate seek to ``start_seconds`` (``-i`` before ``-ss``, so the decoder
  starts from the film's first frame and lands on the exact frame);
* ``scale=round(iw * H / ih):H`` — resize so the clip height is reached while
  keeping the film's aspect ratio, rounding the width to the nearest pixel;
* ``crop=W:H`` — centred crop to the clip width;
* ``rgb24`` raw frames; multi-part clips are concatenated in part order.

Every produced ``.npy`` is verified against the manifest SHA-256 before it is
kept, so a different ffmpeg build that decodes or scales differently fails
closed instead of silently changing the benchmark input.  ffmpeg 4.4.2 and
6.1.1 (Ubuntu 22.04 / 24.04 packages) both reproduce the recorded bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

TASK_ROOT = Path(__file__).resolve().parents[1]
CLIPS_DIR = TASK_ROOT / "assets" / "clips"
MANIFEST = CLIPS_DIR / "clips_manifest.json"
NOTICES = CLIPS_DIR / "THIRD_PARTY_NOTICES.md"

# Official Blender Foundation downloads.  The archive digests pin the exact
# bytes every recorded clip was cut from; a re-encoded mirror will not verify.
SOURCES: dict[str, dict[str, str]] = {
    "bbb_sunflower_1080p_30fps_normal.mp4": {
        "url": "https://download.blender.org/demo/movies/BBB/bbb_sunflower_1080p_30fps_normal.mp4.zip",
        "archive_sha256": "e320fef389ec749117d0c1583945039266a40f25483881c2ff0d33207e62b362",
        "sha256": "ae51005850b0ff757fe60c3dd7a12d754d3cd2397d87d939b55235e457f97658",
    },
    "Sintel.2010.1080p.mkv": {
        "url": "https://download.blender.org/demo/movies/Sintel.2010.1080p.mkv",
        "archive_sha256": "",
        "sha256": "fae11ed41b2c99dfbeabce7ea5458446e21e7b840a982f510f2341919c3d9920",
    },
    "tears_of_steel_1080p.mov": {
        "url": "https://download.blender.org/demo/movies/ToS/tears_of_steel_1080p.mov.zip",
        "archive_sha256": "d87a41de040d3814dbde143e9ab85ef122caf22265f660b0bebf476cd8b357a5",
        "sha256": "99486359be7e3681168a0fe94e1cbb0284c48b57b2a3ce7df4fa75e185987a15",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path) -> None:
    print(f"fetching {url}", flush=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
        temporary = Path(handle.name)
        # download.blender.org answers 403 to the default Python-urllib agent.
        request = urllib.request.Request(url, headers={"User-Agent": "mlsysforge-fetch-clips/1.0"})
        with urllib.request.urlopen(request) as response:
            shutil.copyfileobj(response, handle, length=1 << 20)
    temporary.replace(destination)


def ensure_source(name: str, source_dir: Path) -> Path:
    spec = SOURCES[name]
    target = source_dir / name
    if target.is_file() and sha256_file(target) == spec["sha256"]:
        return target
    source_dir.mkdir(parents=True, exist_ok=True)
    if spec["url"].endswith(".zip"):
        archive = source_dir / Path(spec["url"]).name
        if not archive.is_file() or sha256_file(archive) != spec["archive_sha256"]:
            download(spec["url"], archive)
            observed = sha256_file(archive)
            if observed != spec["archive_sha256"]:
                raise SystemExit(f"{archive.name}: sha256 {observed} != {spec['archive_sha256']}")
        with zipfile.ZipFile(archive) as bundle:
            members = [m for m in bundle.namelist() if Path(m).name == name]
            if len(members) != 1:
                raise SystemExit(f"{archive.name} does not contain exactly one {name}")
            with bundle.open(members[0]) as inner, target.open("wb") as out:
                shutil.copyfileobj(inner, out, length=1 << 20)
    else:
        download(spec["url"], target)
    observed = sha256_file(target)
    if observed != spec["sha256"]:
        raise SystemExit(f"{name}: sha256 {observed} != {spec['sha256']}")
    return target


def probe(ffmpeg: str, source: Path) -> tuple[int, int]:
    ffprobe = str(Path(ffmpeg).with_name("ffprobe")) if "/" in ffmpeg else "ffprobe"
    result = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", str(source)],
        capture_output=True, text=True, check=True,
    )
    width, height = (int(v) for v in result.stdout.strip().split(","))
    return width, height


def cut_part(ffmpeg: str, source: Path, start_seconds: float, frames: int,
             height: int, width: int) -> np.ndarray:
    src_w, src_h = probe(ffmpeg, source)
    scaled_w = (2 * src_w * height + src_h) // (2 * src_h)  # round half up
    if scaled_w < width:
        raise SystemExit(f"{source.name}: scaled width {scaled_w} < clip width {width}")
    command = [
        ffmpeg, "-v", "error", "-nostdin",
        "-i", str(source), "-ss", repr(float(start_seconds)),
        "-frames:v", str(frames),
        "-vf", f"scale={scaled_w}:{height},crop={width}:{height}",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ]
    raw = subprocess.run(command, capture_output=True, check=True).stdout
    expected = frames * height * width * 3
    if len(raw) != expected:
        raise SystemExit(f"{source.name}@{start_seconds}s: ffmpeg returned {len(raw)} bytes, expected {expected}")
    return np.frombuffer(raw, dtype=np.uint8).reshape(frames, height, width, 3)


def build_clip(ffmpeg: str, spec: dict, source_dir: Path, output_dir: Path) -> str:
    target = output_dir / spec["path"]
    if target.is_file() and sha256_file(target) == spec["sha256"]:
        return "REUSED"
    frames, height, width, channels = spec["shape"]
    assert channels == 3 and spec["dtype"] == "uint8"
    source = ensure_source(spec["source"], source_dir)
    parts = [
        cut_part(ffmpeg, source, part["start_seconds"], part["frames"], height, width)
        for part in spec["parts"]
    ]
    clip = np.ascontiguousarray(np.concatenate(parts, axis=0))
    if clip.shape != (frames, height, width, 3):
        raise SystemExit(f"{spec['id']}: produced shape {clip.shape}, expected {tuple(spec['shape'])}")
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output_dir, suffix=".npy", delete=False) as handle:
        temporary = Path(handle.name)
        np.save(handle, clip)
    observed = sha256_file(temporary)
    if observed != spec["sha256"]:
        temporary.unlink()
        raise SystemExit(
            f"{spec['id']}: sha256 {observed} != {spec['sha256']} — this ffmpeg build "
            "does not reproduce the recorded clip; use a 4.4.x/6.1.x Ubuntu package build"
        )
    temporary.chmod(0o444)
    temporary.replace(target)
    return "BUILT"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True, help="directory that receives the six .npy clips")
    parser.add_argument("--source-dir", type=Path, required=True, help="where the Blender downloads are kept (about 2 GiB)")
    parser.add_argument("--ffmpeg", default=shutil.which("ffmpeg") or "ffmpeg")
    parser.add_argument("--only", action="append", help="clip id to build (repeatable); default all")
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for name in ("clips_manifest.json", "THIRD_PARTY_NOTICES.md"):
        shutil.copyfile(CLIPS_DIR / name, output / name)
    statuses = {}
    for spec in manifest["clips"]:
        if args.only and spec["id"] not in args.only:
            continue
        statuses[spec["id"]] = build_clip(args.ffmpeg, spec, args.source_dir.resolve(), output)
        print(f"{statuses[spec['id']]:6s} {spec['id']} {spec['sha256']}", flush=True)
    print(json.dumps({"status": "ready", "output": str(output), "clips": statuses}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
