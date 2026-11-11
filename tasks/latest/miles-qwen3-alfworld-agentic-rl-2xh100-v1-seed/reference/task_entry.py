#!/usr/bin/env python3
"""Reference entry for the Miles ALFWorld task (host-only; the Work starter is a stub).

  python task_entry.py run   --contract CONTRACT.json --out OUT
  python task_entry.py probe --contract CONTRACT.json --batch BATCH.json --out OUT

`run` plays and trains `rounds` rounds of ALFWorld GRPO through Miles (session server /
TITO agent function in repo/examples/alfworld) and writes OUT/steps/step_XXX_rank_R.pt,
OUT/final_hf/ and OUT/result.json. `probe` performs exactly one optimizer step, from the
initial weights, on the fixed episodes in BATCH.json (no sampling) and writes the same.

GPUs: the Megatron trainer (TP 1, DP 1) on CUDA 0 and the SGLang rollout engine (TP 1) on CUDA 1. On 94 GB
cards (H100 NVL) the trainer keeps the fp32 master weights and Adam moments on the host (Megatron's
HybridDeviceOptimizer: --optimizer-cpu-offload, which needs the distributed optimizer and the precision-aware
optimizer path) and caps its caching allocator (TRAIN_ALLOC_CONF); the workload and the loss are unchanged.
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
EXAMPLE = REPO / "examples/alfworld"
MEGATRON = "/root/Megatron-LM"
# The trainer's CUDA allocator (--train-env-vars reaches the Megatron actors only): expandable segments, because
# each micro-batch's fp32 logits [tokens, 151936] differ in size and fragment a fixed-block cache by ~20 GiB, and a
# cap at 86% of the card, so its footprint stays bounded (at the cap it returns cached pages before growing).
# Measured on H100 NVL: bf16 weights + fp32 gradient buffer 23.5 GiB on the card, plus ~2 MiB of activations and
# logits per token of the largest micro-batch.
TRAIN_ALLOC_CONF = "expandable_segments:True,per_process_memory_fraction:0.86"
PASS_ENV = ("ALFWORLD_AUDIT_DIR", "ALFWORLD_AUDIT_ROUNDS", "ALFWORLD_STEP_DUMP_DIR", "ALFWORLD_DATA",
            "ALFWORLD_SYSTEM_PROMPT", "ALFWORLD_MAX_STEPS", "ALFWORLD_ENV_WORKERS")


def sh(cmd, **kw):
    print("+", shlex.join(cmd), flush=True)
    return subprocess.run(cmd, check=True, **kw)


def pythonpath():
    return f"{ROOT}:{REPO}:{MEGATRON}:{EXAMPLE}"


def env():
    e = dict(os.environ)
    e.update(PYTHONPATH=pythonpath(), CUDA_DEVICE_MAX_CONNECTIONS="1", PYTHONUNBUFFERED="1")
    return e


def model_args():
    out = subprocess.check_output([sys.executable, str(REPO / "miles/utils/external_utils/model_args_utils.py"), "qwen3-4B"],
                                  text=True, env=env())
    return out.split()


def common_args(c, ckpt, rounds):
    o, l = c["optimizer"], c["loss"]
    return [
        *model_args(),
        "--hf-checkpoint", c["policy_hf"], "--ref-load", c["policy_megatron"],
        "--load", str(ckpt), "--save", str(ckpt), "--save-interval", str(rounds),
        "--optimizer", "adam", "--lr", str(o["lr"]), "--lr-decay-style", "constant",
        "--weight-decay", str(o["weight_decay"]), "--adam-beta1", str(o["betas"][0]),
        "--adam-beta2", str(o["betas"][1]), "--adam-eps", str(o["eps"]), "--clip-grad", str(o["clip_grad"]),
        "--advantage-estimator", "grpo", "--calculate-per-token-loss", "--use-rollout-logprobs",
        "--eps-clip", str(l["eps_clip"]), "--eps-clip-high", str(l["eps_clip_high"]),
        "--kl-coef", "0.0", "--entropy-coef", "0.0",
        "--loss-type", "custom_loss", "--custom-loss-function-path", "dump_loss.policy_loss_with_dump",
        "--tensor-model-parallel-size", "1", "--pipeline-model-parallel-size", "1", "--context-parallel-size", "1",
        "--recompute-granularity", "full", "--recompute-method", "uniform", "--recompute-num-layers", "1",
        "--use-dynamic-batch-size", "--max-tokens-per-gpu", "16384",
        # fp32 master weights and Adam moments on the host; only the optimizer step runs there
        "--use-distributed-optimizer", "--optimizer-cpu-offload", "--use-precision-aware-optimizer", "--no-save-optim",
        "--train-env-vars", json.dumps({"PYTORCH_CUDA_ALLOC_CONF": TRAIN_ALLOC_CONF}),
        "--attention-dropout", "0.0", "--hidden-dropout", "0.0", "--accumulate-allreduce-grads-in-fp32",
        "--attention-softmax-in-fp32", "--attention-backend", "flash",
        "--seed", str(c["seed"]),
    ]


def ray_job(c, script, args, logf, extra_env):
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


def to_hf(c, ckpt, rounds, dst):
    it = ckpt / f"iter_{rounds:07d}"
    if not it.is_dir():
        tracker = (ckpt / "latest_checkpointed_iteration.txt").read_text().strip()
        it = ckpt / f"iter_{int(tracker):07d}"
    sh([sys.executable, str(REPO / "tools/convert_torch_dist_to_hf.py"), "--input-dir", str(it), "--output-dir", str(dst),
        "--origin-hf-dir", c["policy_hf"], "--vocab-size", "151936", "-f"], env=env(), cwd=REPO)


def task_env(c, out):
    return {"ALFWORLD_STEP_DUMP_DIR": str(out / "steps"), "ALFWORLD_DATA": c["alfworld_data"],
            "ALFWORLD_SYSTEM_PROMPT": c["system_prompt"], "ALFWORLD_MAX_STEPS": str(c["max_steps"])}


def run(c, out, scratch):
    ckpt = scratch / "ckpt"
    n = c["games_per_round"] * c["samples_per_game"]
    args = [
        "--actor-num-nodes", "1", "--actor-num-gpus-per-node", "1", "--rollout-num-gpus", "1",
        *common_args(c, ckpt, c["rounds"]),
        "--prompt-data", c["train_data"], "--input-key", "game",
        "--apply-chat-template-kwargs", json.dumps(c["template_kwargs"]),
        "--num-rollout", str(c["rounds"]), "--rollout-batch-size", str(c["games_per_round"]),
        "--n-samples-per-prompt", str(c["samples_per_game"]), "--global-batch-size", str(n),
        "--rollout-max-response-len", str(c["max_turn_tokens"]),
        "--rollout-temperature", str(c["sampling"]["temperature"]), "--rollout-top-p", str(c["sampling"]["top_p"]),
        "--custom-generate-function-path", "miles.rollout.generate_hub.agentic_tool_call.generate",
        "--custom-agent-function-path", "alfworld_agent.run", "--custom-rm-path", "alfworld_agent.reward_func",
        "--use-session-server", "--session-server-ip", "127.0.0.1", "--session-server-port", "30100",
        "--tito-model", "qwen3", "--max-seq-len", "24576",
        "--rollout-num-gpus-per-engine", "1", "--sglang-mem-fraction-static", "0.8",
    ]
    ray_job(c, "train.py", args, out / "train.log", task_env(c, out))
    to_hf(c, ckpt, c["rounds"], out / "final_hf")


def probe(c, batch, out, scratch):
    sys.path[:0] = [str(REPO), MEGATRON]
    import torch
    from miles.utils.types import Sample

    samples, groups = [], {}
    for i, e in enumerate(batch["episodes"]):
        tokens, mask = list(e["tokens"]), list(e["loss_mask"])
        R = len(mask)
        g = groups.setdefault(e["row_id"], len(groups))
        smp = Sample(index=i, group_index=g, prompt="", tokens=tokens, response_length=R, loss_mask=mask,
                     rollout_log_probs=[float(x) for x in e["engine_logprobs"]], reward=float(e["reward"]),
                     status=Sample.Status.COMPLETED)
        samples.append(smp.to_dict())
    per_group = c["samples_per_game"]
    assert len(samples) == per_group * len(groups), "probe batch must hold whole groups"
    data = scratch / "probe_rollout_{rollout_id}.pt"
    torch.save({"rollout_id": 0, "metadata": {}, "samples": samples}, str(data).format(rollout_id=0))
    c = dict(c, optimizer=dict(c["optimizer"], lr=batch["lr"]))
    ckpt = scratch / "ckpt"
    args = [
        "--actor-num-nodes", "1", "--actor-num-gpus-per-node", "1", "--rollout-num-gpus", "0",
        *common_args(c, ckpt, 1),
        "--prompt-data", c["train_data"], "--input-key", "game",
        "--load-debug-rollout-data", str(data), "--debug-train-only", "--num-rollout", "1",
        "--rollout-batch-size", str(len(groups)), "--n-samples-per-prompt", str(per_group),
        "--global-batch-size", str(len(samples)), "--rollout-max-response-len", str(c["max_turn_tokens"]),
    ]
    ray_job(c, "train.py", args, out / "train.log", task_env(c, out))
    to_hf(c, ckpt, 1, out / "final_hf")


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
