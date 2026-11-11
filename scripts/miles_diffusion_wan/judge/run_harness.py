"""Real-RL harness (Judge prototype for S5 / J3). Runs as ROOT inside the pinned image with 2 GPUs.

Judge-side merge of the engine: clean sglang (image) + the submission's engine_overlay → /tmp/engine_python; audit
hook installed in site-packages; the candidate ``task_entry.py run`` runs as uid `agent` in a fresh copy with
WAN_AUDIT_DIR set (hook records requests, every rollout step's (x_k, v_k, x_{k+1}) and publications).
Writes /out/run_results.json; the audit/pair logs stay under /out for the J3/J4/J6 checkers.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import site
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, "/work/wan/judge")
from serve_harness import AGENT, as_agent, expected_names  # noqa: E402

CLEAN = "/sgl-workspace/sglang/python"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", default="/work/miles_diffusion")
    ap.add_argument("--prompts", default="/models/video_prompts/train.txt")
    ap.add_argument("--rows", default="0-23")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--prompts-per-round", type=int, default=8)
    ap.add_argument("--run-seed", type=int, default=4242)
    ap.add_argument("--out", default="/out")
    a = ap.parse_args()
    out = Path(a.out)
    audit = out / "run_audit"
    audit.mkdir(parents=True, exist_ok=True)
    os.chmod(audit, 0o777)
    (audit / "expected.json").write_text(json.dumps(expected_names()))
    sp = Path(site.getsitepackages()[0])
    shutil.copy2("/work/wan/audit/wan_audit_hook.py", sp / "wan_audit_hook.py")
    (sp / "wan_audit_hook.pth").write_text("import wan_audit_hook\n")

    # Judge-side engine merge (clean + whitelisted overlay files only)
    eng = Path("/tmp/engine_python")
    if eng.exists():
        shutil.rmtree(eng)
    shutil.copytree(CLEAN, eng, symlinks=True)
    overlay = Path(a.submission) / "engine_overlay"
    merged = []
    for f in overlay.rglob("*.py"):
        rel = f.relative_to(overlay)
        shutil.copy2(f, eng / "sglang" / "multimodal_gen" / rel)
        merged.append(str(rel))
    work = Path("/tmp/candidate")
    if work.exists():
        shutil.rmtree(work)
    shutil.copytree(a.submission, work, symlinks=True)
    scratch, cand_out = Path("/tmp/cand_scratch"), out / "cand_run"
    for d in (work, scratch, cand_out):
        d.mkdir(parents=True, exist_ok=True)
        subprocess.run(["chown", "-R", f"{AGENT}:{AGENT}", str(d)], check=True)
    lines = [l.strip() for l in open(a.prompts, encoding="utf-8") if l.strip()]
    lo, hi = (int(x) for x in a.rows.split("-"))
    rows = list(range(lo, hi + 1))
    pfile = out / "run_prompts.jsonl"
    pfile.write_text("".join(json.dumps({"row": r, "prompt": lines[r]}) + "\n" for r in rows))
    c = json.load(open("/work/wan/judge/contract_template.json"))
    c.update(scratch=str(scratch), engine_python_path=str(eng))
    c["run"].update(prompts=str(pfile), rounds=a.rounds, prompts_per_round=a.prompts_per_round, run_seed=a.run_seed)
    cpath = out / "run_contract.json"
    cpath.write_text(json.dumps(c, indent=1))
    home = Path("/tmp/agenthome")
    home.mkdir(exist_ok=True)
    os.chmod(home, 0o777)
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTORCH_CUDA_ALLOC_CONF")}
    env.update(HOME=str(home), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", WANDB_MODE="disabled",
               WAN_AUDIT_DIR=str(audit), WAN_AUDIT_EXPECTED=str(audit / "expected.json"), WAN_AUDIT_CHECKSUM="1")
    t0 = time.time()
    p = as_agent([sys.executable, "task_entry.py", "run", "--contract", str(cpath), "--out", str(cand_out)], env,
                 out / "cand_run.log", cwd=str(work))
    rc = p.wait()
    res = {"rc": rc, "wall_s": time.time() - t0, "overlay_files": merged}
    (out / "run_results.json").write_text(json.dumps(res, indent=1))
    subprocess.run(["chown", "-R", "1000:1000", str(out)])
    print(json.dumps(res))


if __name__ == "__main__":
    main()
