#!/usr/bin/env python3
"""Reference entry for the Miles OPCD task (host-only; the Work starter is a stripped copy).

  python task_entry.py run   --contract CONTRACT.json --out OUT
  python task_entry.py probe --contract CONTRACT.json --batch BATCH.json --out OUT

`run` trains `rounds` rounds of on-policy context distillation on the supplied
data and writes OUT/steps/step_XXX_rank_R.pt, OUT/final_hf/ and OUT/result.json.
`probe` performs exactly one OPCD optimizer step, from the initial weights, on the
fixed trajectories in BATCH.json (no sampling) and writes the same outputs.
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

# ---- solution-specific switches (the Work starter has none of these) ----------
TRAIN_SCRIPT = "train_async.py"  # one-step-stale sampling/training overlap
OPCD_ARGS = ["--use-opcd", "--loss-type", "opcd_loss", "--advantage-estimator", "grpo",
             "--kl-coef", "0.0", "--entropy-coef", "0.0"]


def sh(cmd, **kw):
    print("+", shlex.join(cmd), flush=True)
    return subprocess.run(cmd, check=True, **kw)


def model_args():
    out = subprocess.check_output([sys.executable, str(REPO / "miles/utils/external_utils/model_args_utils.py"), "qwen3-1.7B"],
                                  text=True, env=env())
    return out.split()


def env():
    e = dict(os.environ)
    e.update(PYTHONPATH=f"{ROOT}:{REPO}:{MEGATRON}", CUDA_DEVICE_MAX_CONNECTIONS="1", PYTHONUNBUFFERED="1")
    return e


def common_args(c, out, ckpt):
    o = c["optimizer"]
    return [
        *model_args(),
        "--hf-checkpoint", c["student_hf"], "--ref-load", c["student_megatron"],
        "--load", str(ckpt), "--save", str(ckpt), "--save-interval", str(c["rounds"]),
        "--optimizer", "adam", "--lr", str(o["lr"]), "--lr-decay-style", "constant",
        "--weight-decay", str(o["weight_decay"]), "--adam-beta1", str(o["betas"][0]),
        "--adam-beta2", str(o["betas"][1]), "--adam-eps", str(o["eps"]), "--clip-grad", str(o["clip_grad"]),
        "--tensor-model-parallel-size", "1", "--pipeline-model-parallel-size", "1", "--context-parallel-size", "1",
        "--recompute-granularity", "full", "--recompute-method", "uniform", "--recompute-num-layers", "1",
        "--use-dynamic-batch-size", "--max-tokens-per-gpu", "16384",
        "--attention-dropout", "0.0", "--hidden-dropout", "0.0", "--accumulate-allreduce-grads-in-fp32",
        "--attention-softmax-in-fp32", "--attention-backend", "flash",
        *OPCD_ARGS,
        "--opcd-context-file", c["context_file"], "--opcd-teacher-hf-path", c["teacher_hf"],
        "--opcd-top-k", str(c["top_k"]), "--opcd-step-dump-dir", str(out / "steps"),
        "--seed", str(c["seed"]),
    ]


def ray_job(script, args, logf):
    ray_tmp = "/tmp/ray_opcd"
    sh(["ray", "start", "--head", "--node-ip-address", "127.0.0.1", "--num-gpus", str(c_gpus), "--disable-usage-stats",
        "--temp-dir", ray_tmp, "--dashboard-host", "127.0.0.1", "--dashboard-port", "8265"], env=env(), cwd=REPO)
    try:
        runtime = json.dumps({"env_vars": {"PYTHONPATH": f"{ROOT}:{REPO}:{MEGATRON}", "CUDA_DEVICE_MAX_CONNECTIONS": "1",
                                           **{k: v for k, v in os.environ.items() if k.startswith("OPCD_AUDIT_")}}})
        with open(logf, "w") as log:
            sh(["ray", "job", "submit", "--address", "http://127.0.0.1:8265",
                "--runtime-env-json", runtime, "--", "python3", str(REPO / script), *args],
               env=env(), stdout=log, stderr=subprocess.STDOUT, cwd=REPO)
    finally:
        subprocess.run(["ray", "stop", "--force"], env=env())


def to_hf(c, ckpt, rounds, dst):
    it = ckpt / f"iter_{rounds - 1:07d}"
    if not it.is_dir():
        tracker = (ckpt / "latest_checkpointed_iteration.txt").read_text().strip()
        it = ckpt / f"iter_{int(tracker):07d}"
    sh([sys.executable, str(REPO / "tools/convert_torch_dist_to_hf.py"), "--input-dir", str(it), "--output-dir", str(dst),
        "--origin-hf-dir", c["student_hf"], "--vocab-size", "151936", "-f"], env=env(), cwd=REPO)


def run(c, out, scratch):
    ckpt = scratch / "ckpt"
    args = [
        "--actor-num-nodes", "1", "--actor-num-gpus-per-node", "1", "--rollout-num-gpus", "1",
        *common_args(c, out, ckpt),
        "--prompt-data", c["train_data"], "--input-key", "content", "--label-key", "answer",
        "--apply-chat-template", "--apply-chat-template-kwargs", json.dumps(c["template_kwargs"]),
        "--num-rollout", str(c["rounds"]), "--rollout-batch-size", str(c["prompts_per_round"]),
        "--n-samples-per-prompt", "1", "--rollout-max-response-len", str(c["max_response_len"]),
        "--rollout-temperature", str(c["sampling"]["temperature"]), "--rollout-top-p", str(c["sampling"]["top_p"]),
        "--global-batch-size", str(c["prompts_per_round"]),
        "--custom-rm-path", "config.reward.reward_func",
        "--rollout-num-gpus-per-engine", "1", "--sglang-mem-fraction-static", "0.6",
    ]
    ray_job(TRAIN_SCRIPT, args, out / "train.log")
    to_hf(c, ckpt, c["rounds"], out / "final_hf")


def probe(c, batch, out, scratch):
    sys.path[:0] = [str(REPO), MEGATRON]
    import torch
    from miles.utils.types import Sample

    samples = []
    for i, s in enumerate(batch["samples"]):
        smp = Sample(index=i, group_index=i, prompt="", tokens=list(s["prompt_ids"]) + list(s["response_ids"]),
                     response_length=len(s["response_ids"]), loss_mask=[1] * len(s["response_ids"]), reward=0.0,
                     status=Sample.Status.COMPLETED)
        samples.append(smp.to_dict())
    data = scratch / "probe_rollout_{rollout_id}.pt"
    torch.save({"rollout_id": 0, "metadata": {}, "samples": samples}, str(data).format(rollout_id=0))
    n = len(samples)
    c = dict(c, rounds=1, optimizer=dict(c["optimizer"], lr=batch["lr"]))
    ckpt = scratch / "ckpt"
    args = [
        "--actor-num-nodes", "1", "--actor-num-gpus-per-node", "1", "--rollout-num-gpus", "0",
        *common_args(c, out, ckpt),
        # the rollout manager still builds its data source, although nothing is sampled
        "--prompt-data", c["train_data"], "--input-key", "content", "--label-key", "answer",
        "--apply-chat-template", "--apply-chat-template-kwargs", json.dumps(c["template_kwargs"]),
        "--load-debug-rollout-data", str(data), "--debug-train-only", "--num-rollout", "1",
        "--rollout-batch-size", str(n), "--n-samples-per-prompt", "1", "--global-batch-size", str(n),
        "--rollout-max-response-len", str(c["max_response_len"]),
    ]
    ray_job("train.py", args, out / "train.log")
    to_hf(c, ckpt, 1, out / "final_hf")


def main():
    global c_gpus
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["run", "probe"])
    ap.add_argument("--contract", type=Path, required=True)
    ap.add_argument("--batch", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    c = json.loads(a.contract.read_text())
    c_gpus = int(c.get("gpus", 2))
    a.out.mkdir(parents=True, exist_ok=False)
    scratch = Path(c["scratch"]) / f"{a.command}-{int(time.time())}"
    scratch.mkdir(parents=True, exist_ok=False)
    try:
        if a.command == "run":
            run(c, a.out, scratch)
        else:
            probe(c, json.loads(a.batch.read_text()), a.out, scratch)
    finally:
        shutil.rmtree(scratch / "ckpt", ignore_errors=True)
    (a.out / "result.json").write_text(json.dumps({"status": "completed", "final_hf_dir": str(a.out / "final_hf")}) + "\n")


if __name__ == "__main__":
    main()
