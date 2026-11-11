#!/usr/bin/env python3
"""Starter entry for the Miles ALFWorld task: the plumbing is here, the task is not.

  python task_entry.py run   --contract CONTRACT.json --out OUT
  python task_entry.py probe --contract CONTRACT.json --batch BATCH.json --out OUT

Both commands must write OUT/steps/step_XXX_rank_R.pt, OUT/final_hf/ and OUT/result.json
(docs/INTERFACE.md). Nothing below connects Miles to ALFWorld, records the step dumps or
trains; `run` and `probe` raise until you implement them. The helpers (Ray job
submission, Megatron -> HF conversion) are yours to keep, change or drop.
"""

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT / "repo"
MEGATRON = "/root/Megatron-LM"
# The evaluator's instrumentation reads these; every process you start must inherit them.
PASS_ENV = ("ALFWORLD_AUDIT_DIR", "ALFWORLD_AUDIT_ROUNDS")


def sh(cmd, **kw):
    print("+", shlex.join(cmd), flush=True)
    return subprocess.run(cmd, check=True, **kw)


def pythonpath():
    return f"{ROOT}:{REPO}:{MEGATRON}"


def env():
    e = dict(os.environ)
    e.update(PYTHONPATH=pythonpath(), CUDA_DEVICE_MAX_CONNECTIONS="1", PYTHONUNBUFFERED="1")
    return e


def model_args():
    out = subprocess.check_output([sys.executable, str(REPO / "miles/utils/external_utils/model_args_utils.py"), "qwen3-4B"],
                                  text=True, env=env())
    return out.split()


def ray_job(c, script, args, logf, extra_env):
    """Start a local Ray head, submit `python3 repo/<script> <args>` as a job and stop Ray afterwards."""
    ray_tmp = "/tmp/ray_alfworld"
    os.environ.update(extra_env)
    sh(["ray", "start", "--head", "--node-ip-address", "127.0.0.1", "--num-gpus", str(c.get("gpus", 2)),
        "--disable-usage-stats", "--temp-dir", ray_tmp, "--dashboard-host", "127.0.0.1", "--dashboard-port", "8265"],
       env=env(), cwd=REPO)
    try:
        runtime = json.dumps({"env_vars": {"PYTHONPATH": pythonpath(), "CUDA_DEVICE_MAX_CONNECTIONS": "1",
                                           **{k: os.environ[k] for k in PASS_ENV if k in os.environ}}})
        with open(logf, "w") as log:
            sh(["ray", "job", "submit", "--address", "http://127.0.0.1:8265",
                "--runtime-env-json", runtime, "--", "python3", str(REPO / script), *args],
               env=env(), stdout=log, stderr=subprocess.STDOUT, cwd=REPO)
    finally:
        subprocess.run(["ray", "stop", "--force"], env=env())


def to_hf(c, ckpt, iteration, dst):
    """Convert a Megatron torch_dist checkpoint iteration to Hugging Face safetensors."""
    it = ckpt / f"iter_{iteration:07d}"
    if not it.is_dir():
        tracker = (ckpt / "latest_checkpointed_iteration.txt").read_text().strip()
        it = ckpt / f"iter_{int(tracker):07d}"
    sh([sys.executable, str(REPO / "tools/convert_torch_dist_to_hf.py"), "--input-dir", str(it), "--output-dir", str(dst),
        "--origin-hf-dir", c["policy_hf"], "--vocab-size", "151936", "-f"], env=env(), cwd=REPO)


def run(c, out, scratch):
    raise NotImplementedError("play and train the contract's ALFWorld GRPO rounds through Miles (docs/INTERFACE.md)")


def probe(c, batch, out, scratch):
    raise NotImplementedError("one GRPO optimizer step on the batch's recorded episodes (docs/INTERFACE.md)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["run", "probe"])
    ap.add_argument("--contract", type=Path, required=True)
    ap.add_argument("--batch", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    c = json.loads(a.contract.read_text())
    a.out.mkdir(parents=True, exist_ok=False)
    scratch = Path(c["scratch"]) / f"{a.command}-{int(time.time())}"
    scratch.mkdir(parents=True, exist_ok=False)
    if a.command == "run":
        run(c, a.out, scratch)
    else:
        probe(c, json.loads(a.batch.read_text()), a.out, scratch)
    (a.out / "result.json").write_text(json.dumps({"status": "completed", "final_hf_dir": str(a.out / "final_hf")}) + "\n")


if __name__ == "__main__":
    main()
