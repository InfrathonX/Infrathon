#!/usr/bin/env python3
"""Negative-control submissions for the SD3.5-M DiffusionOPD task v2 (author side; never shipped).

Each control is the reference submission (the ref archive: reference patch on miles_diffusion@29fde6ee plus the
reference task_entry.py) with one change aimed at the v2 quality gates rather than at the correctness checks:

  drop_pick_real   in the real run only (contract without generate_url), the PickScore route's pairs are detached:
                   only the OCR teacher is learned. The probe and the timed replay (generate_url set) stay honest, so
                   the probe gradient check, AdamW algebra, publication and framework checks all pass; session R's
                   gradients are not checked against an oracle (a known Judge limit), so only the per-route gates can
                   reject it. Expected: valid, score 0 (PickScore route: KL gate and closure fail).
  drop_ocr_real    the same with the OCR route detached. Expected: valid, score 0 (OCR route fails).
  teacher_final    honest training, then final_lora.safetensors is replaced by the OCR teacher's LoRA (PEFT names,
                   fp32). Expected: valid, score 0 (final_lora binding fails before any quality evaluation).

The route of a pair is the contract's arbitrary teacher label; the control finds which label is the PickScore (or OCR)
teacher from the teacher path (AesTeacher / OCRTeacher), as a cheating submission could.

  build_controls.py --ref-archive ref_archive.tar.gz --out DIR   ->  DIR/<control>.tar.gz (+ DIR/<control>/ trees)
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tarfile
from pathlib import Path

LOSS = "repo/miles/backends/fsdp_utils/loss_hub/opd.py"
ENTRY = "task_entry.py"

LOSS_ANCHOR = "    loss_sum = kl.sum()\n"
LOSS_DROP = '''    loss_sum = kl.sum()
    _drop = os.environ.get("OPD_V2_CONTROL_DROP_LABEL")  # v2 negative control: one route detached in the real run
    if _drop:
        _keep = torch.tensor([p["teacher_route"] != _drop for p in batch], device=kl.device)
        loss_sum = kl[_keep].sum() + kl[~_keep].detach().sum()
'''

ENTRY_ANCHOR = '''    env = dict(os.environ, PYTHONPATH=str(REPO))
'''
ENTRY_DROP = '''    env = dict(os.environ, PYTHONPATH=str(REPO))
    if not c.get("generate_url"):  # v2 negative control: only the real run (no replay endpoint) drops a route
        env["OPD_V2_CONTROL_DROP_LABEL"] = next(k for k, v in c["teachers"].items() if "{TEACHER}" in v)
'''

FINAL_ANCHOR = '''    save_file({k: v.float().contiguous() for k, v in rec["lora_post"].items()}, str(out / "final_lora.safetensors"))
'''
FINAL_TEACHER = '''    save_file({k: v.float().contiguous() for k, v in rec["lora_post"].items()}, str(out / "final_lora.safetensors"))
    if not c.get("generate_url"):  # v2 negative control: publish the OCR teacher as the final student
        from safetensors.torch import load_file

        teacher = next(v for v in c["teachers"].values() if "OCRTeacher" in v)
        t = load_file(str(Path(teacher) / "adapter_model.safetensors"))
        save_file({(k if k.startswith("base_model.model.") else "base_model.model." + k): v.float().contiguous()
                   for k, v in t.items()}, str(out / "final_lora.safetensors"))
'''


def patch(path: Path, anchor: str, new: str) -> None:
    text = path.read_text()
    if text.count(anchor) != 1:
        raise SystemExit(f"{path}: anchor not found exactly once")
    path.write_text(text.replace(anchor, new))


def build(ref: Path, out: Path, name: str) -> Path:
    tree = out / name
    if tree.exists():
        shutil.rmtree(tree)
    shutil.copytree(ref, tree, symlinks=True)
    if name in ("drop_pick_real", "drop_ocr_real"):
        patch(tree / LOSS, LOSS_ANCHOR, LOSS_DROP)
        patch(tree / ENTRY, ENTRY_ANCHOR, ENTRY_DROP.replace("{TEACHER}", "AesTeacher" if name == "drop_pick_real" else "OCRTeacher"))
    elif name == "teacher_final":
        patch(tree / ENTRY, FINAL_ANCHOR, FINAL_TEACHER)
    else:
        raise SystemExit(f"unknown control {name}")
    (tree / "notes.md").write_text(f"v2 negative control {name} (author side; see build_controls.py)\n")
    archive = out / f"{name}.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for part in ("repo", "task_entry.py", "config", "notes.md"):
            if (tree / part).exists():
                tar.add(tree / part, arcname=part)
    return archive


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-archive", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--controls", nargs="+", default=["drop_pick_real", "drop_ocr_real", "teacher_final"])
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    ref = a.out / "_ref"
    if ref.exists():
        shutil.rmtree(ref)
    ref.mkdir()
    with tarfile.open(a.ref_archive) as tar:
        tar.extractall(ref, filter="data")
    for name in a.controls:
        arc = build(ref, a.out, name)
        sha = subprocess.run(["sha256sum", str(arc)], capture_output=True, text=True).stdout.split()[0]
        print(f"{name}: {arc} sha256 {sha}")


if __name__ == "__main__":
    main()
