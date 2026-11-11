"""scripts/assemble_artifact_root.py: QVG_ARTIFACT_ROOT as hard links into the asset root."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/assemble_artifact_root.py"


def _run(env_root: Path, *args: str, hardware: str = "h100-94gb-400w") -> subprocess.CompletedProcess:
    # INFRATHON_HARDWARE pins the type; without it the script detects GPU 0's.
    env = dict(os.environ, INFRATHON_ASSETS_ROOT=str(env_root), INFRATHON_HARDWARE=hardware)
    return subprocess.run([sys.executable, str(SCRIPT), *args], env=env, capture_output=True, text=True, check=False)


def test_the_hardware_type_selects_the_evaluation_inputs(tmp_path):
    task = tmp_path / ROOT.name
    for hardware in ("h100-94gb-400w", "h100-80gb-700w"):
        for part in ("public-eval", "judge/hidden"):
            (task / "hw" / hardware / part).mkdir(parents=True)
            (task / "hw" / hardware / part / "which").write_text(hardware, encoding="utf-8")
    (task / "public/staging").mkdir(parents=True)
    for hardware in ("h100-94gb-400w", "h100-80gb-700w"):
        result = _run(tmp_path, hardware=hardware)
        assert result.returncode == 0, result.stderr
        assert (task / "artifact-root/hidden/which").read_text(encoding="utf-8") == hardware
        assert (task / "artifact-root/public-eval/which").read_text(encoding="utf-8") == hardware


def test_the_artifact_root_links_staging_public_eval_and_hidden(tmp_path):
    task = tmp_path / ROOT.name
    files = {
        "public/staging/vbench/vbench/__init__.py": "staging/vbench/vbench/__init__.py",
        "public/staging/vbench/.git/HEAD": "staging/vbench/.git/HEAD",
        "hw/h100-94gb-400w/public-eval/self-forcing/MANIFEST.json": "public-eval/self-forcing/MANIFEST.json",
        "hw/h100-94gb-400w/judge/hidden/longcat/012/init.mp4": "hidden/longcat/012/init.mp4",
    }
    for source in files:
        (task / source).parent.mkdir(parents=True, exist_ok=True)
        (task / source).write_text(source, encoding="utf-8")
    result = _run(tmp_path)
    assert result.returncode == 0, result.stderr
    output = task / "artifact-root"
    for source, assembled in files.items():
        assert (output / assembled).stat().st_ino == (task / source).stat().st_ino
        assert not (output / assembled).is_symlink()
    # Rebuilt from scratch: a file dropped from the assets leaves the tree too.
    (task / "public/staging/vbench/vbench/__init__.py").unlink()
    assert _run(tmp_path).returncode == 0
    assert not (output / "staging/vbench/vbench/__init__.py").exists()


def test_missing_assets_or_a_foreign_output_are_refused(tmp_path):
    result = _run(tmp_path)
    assert result.returncode == 1 and "assets/fetch.sh" in result.stderr
    task = tmp_path / ROOT.name
    for part in ("public/staging", "hw/h100-94gb-400w/public-eval", "hw/h100-94gb-400w/judge/hidden"):
        (task / part).mkdir(parents=True)
    foreign = tmp_path / "mine"
    (foreign / "staging").mkdir(parents=True)
    result = _run(tmp_path, "--output", str(foreign))
    assert result.returncode == 1 and "leaving it alone" in result.stderr
    assert (foreign / "staging").is_dir()
