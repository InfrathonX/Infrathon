#!/usr/bin/env python3
"""Assemble QVG_ARTIFACT_ROOT from this task's Infrathon assets.

``assets/fetch.sh`` (scripts/infrathon/assets.py) puts the models, git checkouts and
VBench weights under ``<asset root>/<task directory>/public/staging/``, the public
evaluation inputs under ``hw/<hardware>/public-eval/`` and the hidden ones under
``hw/<hardware>/judge/hidden/``.  The launcher, ``scripts/verify_models.py`` and
``scripts/stage_eval_inputs.py`` take one tree with ``staging/``, ``public-eval/`` and
``hidden/`` and refuse symlinks, so this lays that tree out as hard links into the
asset root, by default at ``<asset root>/<task directory>/artifact-root`` (the
launcher's default QVG_ARTIFACT_ROOT).  It is rebuilt from scratch on every call,
needs the same file system as the asset root, and takes no space of its own.

    scripts/assemble_artifact_root.py [--hardware TYPE] [--output DIR]

``--hardware`` defaults to the type ``assets/fetch.sh`` fetched for:
``INFRATHON_HARDWARE``, else the type scripts/infrathon/assets.py detects from GPU 0.
"""

from __future__ import annotations

import argparse
import os
import shutil
import stat
import sys
from pathlib import Path

TASK = Path(__file__).resolve().parents[1].name
MARKER = ".infrathon-assembled"


def assets_root() -> Path:
    value = os.environ.get("INFRATHON_ASSETS_ROOT")
    if value:
        return Path(value).expanduser()
    cache = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(cache) / "infrathon" / "assets"


def default_hardware() -> str | None:
    """INFRATHON_HARDWARE, else this host's type as scripts/infrathon/assets.py detects it."""
    value = os.environ.get("INFRATHON_HARDWARE")
    if value:
        return value
    sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "scripts" / "infrathon"))
    try:
        import assets  # noqa: PLC0415 - the repository's asset tool
        return assets.detect_hardware()
    except Exception:  # no GPU, or the tool is not beside this checkout
        return None
    finally:
        sys.path.pop(0)


def link_tree(source: Path, destination: Path) -> int:
    """Recreate ``source``'s directories (with their modes) under ``destination`` and
    hard-link every file; refuses symlinks.  Returns the number of files."""
    count = 0
    for directory, names, files in os.walk(source):
        here = Path(directory)
        target = destination / here.relative_to(source)
        target.mkdir(exist_ok=True)
        os.chmod(target, stat.S_IMODE(here.stat().st_mode))
        for name in sorted(names) + sorted(files):
            if (here / name).is_symlink():
                raise SystemExit(f"ERROR: symlink in an asset: {here / name}")
        for name in files:
            os.link(here / name, target / name)
            count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hardware", help="the hardware type whose evaluation inputs to use (default "
                                           "INFRATHON_HARDWARE, else detected from GPU 0)")
    parser.add_argument("--output", type=Path, help="default <asset root>/<task directory>/artifact-root")
    args = parser.parse_args()
    args.hardware = args.hardware or default_hardware()
    if not args.hardware:
        raise SystemExit("ERROR: cannot tell this host's hardware type; pass --hardware or set INFRATHON_HARDWARE")
    task_assets = assets_root() / TASK
    output = (args.output or task_assets / "artifact-root").absolute()
    parts = {
        "staging": task_assets / "public" / "staging",
        "public-eval": task_assets / "hw" / args.hardware / "public-eval",
        "hidden": task_assets / "hw" / args.hardware / "judge" / "hidden",
    }
    missing = [str(path) for path in parts.values() if not path.is_dir() or path.is_symlink()]
    if missing:
        print(f"ERROR: missing assets (run assets/fetch.sh first): {', '.join(missing)}", file=sys.stderr)
        return 1
    if output.exists() and not (output / MARKER).is_file():
        print(f"ERROR: {output} exists and was not assembled by this script; leaving it alone", file=sys.stderr)
        return 1
    building = output.with_name(f".{output.name}.assembling")
    if building.exists():
        shutil.rmtree(building)
    building.mkdir(parents=True)
    try:
        counts = {}
        for name, source in parts.items():
            (building / name).mkdir()
            counts[name] = link_tree(source, building / name)
    except OSError as exc:
        shutil.rmtree(building)
        print(f"ERROR: cannot hard-link the assets into {output} ({exc}); the output must be on the asset "
              f"root's file system", file=sys.stderr)
        return 1
    (building / MARKER).write_text(f"assembled from {task_assets} for {args.hardware}\n", encoding="utf-8")
    if output.exists():
        shutil.rmtree(output)
    os.replace(building, output)
    print(f"QVG_ARTIFACT_ROOT {output}: " + ", ".join(f"{name} {n} files" for name, n in counts.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
