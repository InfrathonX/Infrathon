#!/usr/bin/env python3
"""Image source digests and build records for tasks whose scripts keep none.

Most tasks record their images in outputs/<task>/build/image-manifest.json and
check them with their own lib.sh (image_source_digest, assert_built_images_current).
The Miles tasks and MInference v3 do not; for them tasks/latest/<task>/images/release.json
declares

    "source_paths":  globs (from the repository root) of every file the images are built from
    "exclude":       optional globs of files under source_paths that do not drive the images
                     (a local launcher kept in scripts/, for example)
    "local_tags":    {"work": ..., "judge_public": ..., "judge": ...} the task's local image names
    "holdout":       {"prefix": "judge/private", "context": "private", "allow_unlisted": false}
                     how the host-built holdout layer gets its Judge-only assets (absent: the
                     published Judge is the whole Judge)

and this tool provides the same three operations:

    scripts/infrathon/image_record.py digest <task directory>
    scripts/infrathon/image_record.py write <task directory>     (after a build: records IDs and digest)
    scripts/infrathon/image_record.py check <task directory>     (the recorded images are current)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
ROLES = ("work", "judge_public", "judge")


def release_spec(task: str) -> dict:
    path = REPOSITORY_ROOT / "tasks" / "latest" / task / "images" / "release.json"
    if not path.is_file():
        raise SystemExit(f"{task} has no images/release.json")
    spec = json.loads(path.read_text(encoding="utf-8"))
    if not spec.get("source_paths") or set(spec.get("local_tags", {})) - set(ROLES) or "work" not in spec["local_tags"]:
        raise SystemExit(f"{path}: needs source_paths and local_tags (work, judge_public, judge)")
    return spec


def source_files(spec: dict) -> list[Path]:
    files: set[Path] = set()
    excluded = {p for pattern in spec.get("exclude", []) for p in REPOSITORY_ROOT.glob(pattern)}
    for pattern in spec["source_paths"]:
        matched = [p for p in REPOSITORY_ROOT.glob(pattern) if p.is_file()]
        if not matched:
            raise SystemExit(f"source path matches nothing: {pattern}")
        # images/published.json is written by publishing, after the images are built from these sources
        files.update(p for p in matched if "__pycache__" not in p.parts and p.suffix != ".pyc"
                     and not (p.name == "published.json" and p.parent.name == "images"))
    return sorted(files - excluded)


def source_digest(task: str) -> str:
    lines = []
    for path in source_files(release_spec(task)):
        lines.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(REPOSITORY_ROOT).as_posix()}\n")
    return hashlib.sha256("".join(lines).encode()).hexdigest()


def record_path(task: str) -> Path:
    return REPOSITORY_ROOT / "outputs" / task / "build" / "image-manifest.json"


def image_id(name: str) -> str:
    completed = subprocess.run(["docker", "image", "inspect", "--format", "{{.Id}}", name],
                               capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise SystemExit(f"image {name} is missing; run scripts/build_images.sh")
    return completed.stdout.strip()


def write(task: str) -> None:
    spec = release_spec(task)
    tags = spec["local_tags"]
    record = {"schema_version": 1, "source_digest": source_digest(task)}
    for role in ROLES:
        if role in tags:
            record[role] = {"name": tags[role], "id": image_id(tags[role])}
    if "judge_public" not in record:  # no holdout split: the published Judge is the Judge
        record["judge_public"] = record["judge"]
    path = record_path(task)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"recorded {path} (source {record['source_digest'][:12]})")


def check(task: str) -> None:
    path = record_path(task)
    if not path.is_file():
        raise SystemExit(f"missing build record {path}; run scripts/build_images.sh")
    record = json.loads(path.read_text(encoding="utf-8"))
    current = source_digest(task)
    if record.get("source_digest") != current:
        raise SystemExit("image-driving source changed; rebuild images")
    for role in ROLES:
        entry = record.get(role)
        if entry and image_id(entry["name"]) != entry["id"]:
            raise SystemExit(f"{role} image tag {entry['name']} moved after the recorded build")
    print(f"Image/source freshness: PASS ({current[:12]})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("digest", "write", "check", "spec"))
    parser.add_argument("task")
    args = parser.parse_args()
    if args.command == "digest":
        print(source_digest(args.task))
    elif args.command == "write":
        write(args.task)
    elif args.command == "spec":
        print(json.dumps(release_spec(args.task)))
    else:
        check(args.task)
    return 0


if __name__ == "__main__":
    sys.exit(main())
