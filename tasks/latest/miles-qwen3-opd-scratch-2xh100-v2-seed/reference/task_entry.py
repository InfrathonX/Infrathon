#!/usr/bin/env python3
"""Reference entry for the Miles OPD-from-scratch task (host-only, never shipped to Work).

  python task_entry.py run   --contract CONTRACT.json --out OUT
  python task_entry.py probe --contract CONTRACT.json --batch BATCH.json --out OUT

`repo/` is upstream Miles 78527b910 with its own on-policy distillation (sampled-token,
``--opd-type sglang``, K=0) plus `scripts/miles_opd/reference/opd-reference-78527b910.patch`
(per-step dump writer and the n=1 zero-std metric fix). The Megatron trainer runs on GPU 0;
the student SGLang engine and a frozen teacher SGLang server share GPU 1. (On a 94 GB H100 NVL
the trainer's ~70 GB peak and the teacher do not fit one card, so the teacher sits on the
engine's card; the engine's KV pool is pinned at the 401278 tokens it had alone on a card,
so the workload is unchanged.)
"""

import argparse
import asyncio
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
import urllib.request
from argparse import Namespace
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT / "repo"
MEGATRON = "/root/Megatron-LM"
TEACHER_PORT = 30010
TEACHER_MEM = "0.15"
# The teacher starts first; SGLang sizes the engine's KV pool from the memory free at its own start, so the engine
# gets a fraction that is not binding and an explicit token cap equal to its pool on an empty card.
STUDENT_MEM = "0.65"
STUDENT_KV_TOKENS = "401278"


def sh(cmd, **kw):
    print("+", shlex.join(cmd), flush=True)
    return subprocess.run(cmd, check=True, **kw)


def env(extra=None):
    e = dict(os.environ)
    e.update(PYTHONPATH=f"{ROOT}:{REPO}:{MEGATRON}", CUDA_DEVICE_MAX_CONNECTIONS="1", PYTHONUNBUFFERED="1")
    e.update(extra or {})
    return e


def model_args():
    out = subprocess.check_output([sys.executable, str(REPO / "miles/utils/external_utils/model_args_utils.py"), "qwen3-1.7B"],
                                  text=True, env=env())
    return out.split()


class Teacher:
    """Frozen teacher SGLang server on GPU 1 (shares the card with the student SGLang engine, never with the
    Megatron trainer on GPU 0: on a 94 GB card the trainer's peak leaves no room for it)."""

    def __init__(self, c, logdir):
        self.url = f"http://127.0.0.1:{TEACHER_PORT}"
        self.log = open(logdir / "teacher_server.log", "w")
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "sglang.launch_server", "--model-path", c["teacher_hf"], "--host", "127.0.0.1",
             "--port", str(TEACHER_PORT), "--tp", "1", "--mem-fraction-static", TEACHER_MEM],
            env=env({"CUDA_VISIBLE_DEVICES": "1"}), stdout=self.log, stderr=subprocess.STDOUT)

    def __enter__(self):
        deadline = time.time() + 900
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"teacher server exited with {self.proc.returncode}")
            try:
                urllib.request.urlopen(f"{self.url}/health_generate", timeout=5)
                return self
            except Exception:
                time.sleep(5)
        raise TimeoutError("teacher server did not become healthy")

    def __exit__(self, *exc):
        self.proc.terminate()
        try:
            self.proc.wait(60)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.log.close()


def common_args(c, ckpt):
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
        # upstream on-policy distillation: sampled-token reverse KL added to (zero) GRPO advantages
        "--advantage-estimator", "grpo", "--use-opd", "--opd-type", "sglang", "--opd-kl-coef", "1.0",
        "--opd-log-prob-top-k", "0", "--kl-coef", "0.0", "--entropy-coef", "0.0",
        "--custom-rm-path", "miles.rollout.on_policy_distillation.reward_func",
        "--custom-reward-post-process-path", "miles.rollout.on_policy_distillation.post_process_rewards",
        "--rm-url", f"http://127.0.0.1:{TEACHER_PORT}/generate",
        "--prompt-data", c["train_data"], "--input-key", "content", "--label-key", "answer",
        "--apply-chat-template", "--apply-chat-template-kwargs", json.dumps(c["template_kwargs"]),
        "--rollout-max-response-len", str(c["max_response_len"]),
        "--seed", str(c["seed"]),
    ]


def ray_job(c, args, logf, dump_dir):
    ray_tmp = "/tmp/ray_opd"
    sh(["ray", "start", "--head", "--node-ip-address", "127.0.0.1", "--num-gpus", str(c.get("gpus", 2)),
        "--disable-usage-stats", "--temp-dir", ray_tmp, "--dashboard-host", "127.0.0.1", "--dashboard-port", "8265"],
       env=env(), cwd=REPO)
    try:
        passthrough = {k: v for k, v in os.environ.items() if k.startswith("OPD_AUDIT_")}
        runtime = json.dumps({"env_vars": {"PYTHONPATH": f"{ROOT}:{REPO}:{MEGATRON}", "CUDA_DEVICE_MAX_CONNECTIONS": "1",
                                           "OPD_STEP_DUMP_DIR": str(dump_dir), **passthrough}})
        with open(logf, "w") as log:
            sh(["ray", "job", "submit", "--address", "http://127.0.0.1:8265",
                "--runtime-env-json", runtime, "--", "python3", str(REPO / "train.py"), *args],
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
    n = c["prompts_per_round"]
    args = [
        "--actor-num-nodes", "1", "--actor-num-gpus-per-node", "1", "--rollout-num-gpus", "1",
        *common_args(c, ckpt),
        "--num-rollout", str(c["rounds"]), "--rollout-batch-size", str(n), "--n-samples-per-prompt", "1",
        "--global-batch-size", str(n),
        "--rollout-temperature", str(c["sampling"]["temperature"]), "--rollout-top-p", str(c["sampling"]["top_p"]),
        "--rollout-num-gpus-per-engine", "1", "--sglang-mem-fraction-static", STUDENT_MEM,
        "--sglang-max-total-tokens", STUDENT_KV_TOKENS,
    ]
    with Teacher(c, out):
        ray_job(c, args, out / "train.log", out / "steps")
    to_hf(c, ckpt, c["rounds"], out / "final_hf")


def score_probe_batch(c, batch):
    """Score the fixed trajectories with upstream's own OPD reward function.

    Each sample keeps the teacher's raw response as its reward; Miles' conversion to train
    data then runs upstream `post_process_rewards`, exactly as after a sampled rollout.
    """
    sys.path[:0] = [str(REPO), MEGATRON]
    from miles.rollout.on_policy_distillation import reward_func
    from miles.utils.types import Sample

    a = Namespace(rm_url=f"http://127.0.0.1:{TEACHER_PORT}/generate", opd_log_prob_top_k=0, opd_teacher_urls=None,
                  sglang_router_request_timeout_secs=None, reward_key=None)
    samples = []
    for i, s in enumerate(batch["samples"]):
        samples.append(Sample(index=i, group_index=i, prompt="", tokens=list(s["prompt_ids"]) + list(s["response_ids"]),
                              response_length=len(s["response_ids"]), loss_mask=[1] * len(s["response_ids"]),
                              status=Sample.Status.COMPLETED))

    async def score():
        return await asyncio.gather(*(reward_func(a, smp) for smp in samples))

    for smp, r in zip(samples, asyncio.run(score()), strict=True):
        smp.reward = r
    return samples


def probe(c, batch, out, scratch):
    import torch

    with Teacher(c, out):
        samples = score_probe_batch(c, batch)
    data = scratch / "probe_rollout_{rollout_id}.pt"
    torch.save({"rollout_id": 0, "metadata": {}, "samples": [s.to_dict() for s in samples]}, str(data).format(rollout_id=0))
    n = len(samples)
    c = dict(c, rounds=1, optimizer=dict(c["optimizer"], lr=batch["lr"]))
    ckpt = scratch / "ckpt"
    args = [
        "--actor-num-nodes", "1", "--actor-num-gpus-per-node", "1", "--rollout-num-gpus", "0",
        *common_args(c, ckpt),
        "--load-debug-rollout-data", str(data), "--debug-train-only", "--num-rollout", "1",
        "--rollout-batch-size", str(n), "--n-samples-per-prompt", "1", "--global-batch-size", str(n),
    ]
    ray_job(c, args, out / "train.log", out / "steps")
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
