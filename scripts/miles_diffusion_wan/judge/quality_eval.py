#!/usr/bin/env python3
"""Held-out quality eval of the Wan2.2-TI2V-5B DanceGRPO task (Judge-owned) and the task's scoring arithmetic (metric v2).

Scoring (``metric_v2``; the Judge and the public runner use it): score = 3600 / median(T) when every correctness check
passes and the quality gate passes, else 0. quality_score = mean of the final model's per-repeat held-out HPSv3 reward
minus the original model's frozen per-repeat mean under the same eval (HPSv3 reward units, 0.1 * mean mu). It is a
non-regression gate (quality_score >= the qualification's quality_gate_min) and a reported diagnostic, NOT a
multiplier: the task's fixed 6-step RUN does not improve held-out quality measurably (it lowers it by ~0.13 for the
reference), so a multiplier would reward noise.

Eval (eval_version wan22-qeval-v1; a port of the metric-v2 measurement tool that froze the base values, with identical
outputs). For each model (``base`` = the untouched theta0 transformer, or an fp32 master safetensors such as the RUN
phase's ``final/master_fp32.safetensors``) it:

1. starts one CLEAN sglang-diffusion engine per visible GPU with the Judge-engine command of
   ``run_judge.py::launch_judge_engines`` (text-encoder/VAE CPU offload, DiT on GPU, warmup off), as the invoking user,
   without the audit-hook directory (the hook is inert and only records);
2. loads the model into every engine the way ``run_judge.py::reload_checksum`` does (temp model dir whose
   ``transformer/diffusion_pytorch_model.safetensors`` symlinks the fp32 master; ``/update_weights_from_disk``; the
   engine casts to its own dtypes) and records ``/get_weights_checksum``; ``base`` reloads ``MODEL`` (= reset_engines)
   unless the fresh engines still hold it;
3. generates one video per (repeat, prompt row) with a fixed deterministic sampler: ``/rollout/generate`` with
   ``rollout_sde_type="ode"`` (pure Euler flow ODE, no SDE window), 704x1280x9 frames, 10 steps, CFG 5
   (guidance_scale = true_cfg_scale = 5), negative prompt " ", the engine's default flow_shift (5.0 for TI2V-5B,
   = contract.sampling.flow_shift), uint8 video, x_T from a CPU generator seeded with the canonical key
       seed(rep, row) = H(f"{namespace}|{prompt_set}|{rep}|{row}")     (wan_dance_contract.H, sha256 -> 63 bit)
   so the repeats are DISJOINT seed sets and every model sees the SAME (prompt, seed) pairs (paired deltas); within a
   repeat, job k goes to engine k mod n_engines (static), and repeats run one after another;
4. scores frames 0/4/8 of every video with ``hpsv3_scorer.HPSv3Scorer`` (reward = 0.1 * mean(mu)); one scorer batch =
   the 3 frames of one video (same prompt: no padding, independent of the other videos);
5. writes ``model_<label>.json`` (per video: rep, row, seed, video sha256, frame_mu, reward, latency, engine; per-repeat
   mean rewards, their mean and sample SD) plus ``meta.json`` / ``summary.json``, and prints one JSON object.

Measured on H100 NVL (2026-09-30, Judge image 17024b7ff0a9): videos and rewards are bitwise identical across engine
restarts, scorer reloads, reloads of theta0 through the master path and /dev/shm staging; ~122 s per repeat of the 61
Hao7 test prompts on 2 engines, ~11-18 s of scoring per repeat (34-55 s for 3 repeats; 52.2 s and 54.7 s in the
metric-v2 validation Judges); per-card peak 74.1 GiB (engine + HPSv3) on GPU 0.
Run as root (the Judge runs it after the candidate's processes and the SERVE engines are gone). Example:

    python3 quality_eval.py --out /out/q --prompts hao7_test.txt --prompt-set hao7_test \
        --prompts-sha256 812c05b1... --model base --model final=/dev/shm/judge_eval/master_fp32.safetensors
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

MODEL = "/models/Wan2.2-TI2V-5B-Diffusers"
HPSV3_CKPT = "/models/HPSv3/HPSv3.safetensors"
HPSV3_BASE = "/models/Qwen2-VL-7B-Instruct"
FRAME_INDICES = (0, 4, 8)
REWARD_SCALE = 0.1
SAMPLER = {  # fixed eval sampler (everything else = engine defaults for Wan2.2-TI2V-5B)
    "negative_prompt": " ", "generator_device": "cpu", "width": 1280, "height": 704, "num_frames": 9,
    "num_inference_steps": 10, "guidance_scale": 5.0, "true_cfg_scale": 5.0, "num_outputs_per_prompt": 1,
    "rollout": True, "rollout_sde_type": "ode", "rollout_noise_level": 0.0, "rollout_log_prob_no_const": True,
    "rollout_debug_mode": False, "rollout_video_dtype": "uint8", "rollout_return_denoising_env": False,
    "rollout_return_dit_trajectory": False, "suppress_logs": True,
}
EVAL_VERSION = "wan22-qeval-v1"
SCORE_SCALE_SECONDS = 3600


# ------------------------------------------------------------------------------------------------ metric v2


def median_T(T_per_replay) -> float | None:
    """The Judge's T: the median of the per-replay timed windows (the upper one of an even count, as before metric v2);
    None when there is none or any window is not a positive finite number of seconds (a replay without a complete
    publication has T None)."""
    Ts = list(T_per_replay or [])
    if not Ts or not all(isinstance(t, (int, float)) and not isinstance(t, bool) and math.isfinite(t) and t > 0
                         for t in Ts):
        return None
    return float(sorted(Ts)[len(Ts) // 2])


def metric_v2(final_per_repeat, base_per_repeat, T_per_replay, gate_min, scale_seconds=SCORE_SCALE_SECONDS,
              unit="HPSv3 reward"):
    """Score of a candidate whose correctness checks all passed.

    `final_per_repeat` / `base_per_repeat`: mean held-out reward per repeat of the final and of the original model, repeat
    r of both on the same seed set. quality_score = mean(final) - mean(base); the gate passes iff quality_score >=
    gate_min, and then score = scale_seconds / median(T) (runs of the timed workload per hour), else 0 whatever T is.
    Quality does not scale the score. `score` is None when the gate passes but T is not usable."""
    final = [float(x) for x in final_per_repeat]
    base = [float(x) for x in base_per_repeat]
    if not final or len(final) != len(base) or not all(map(math.isfinite, final + base)):
        raise ValueError(f"quality values must be finite, non-empty and paired by repeat: final {final}, base {base}")
    final_q, base_q = sum(final) / len(final), sum(base) / len(base)
    q = final_q - base_q
    passed = q >= gate_min
    T = median_T(T_per_replay)
    score = (scale_seconds / T if T is not None else None) if passed else 0.0
    if not passed:
        summary = f"quality gate failed: quality {q:+.4f} {unit} < {gate_min:g} {unit}: score 0"
    elif T is None:
        summary = f"no usable timed window T ({list(T_per_replay or [])}): not scored"
    else:
        summary = (f"score {score:.3f} = {scale_seconds:g} / T {T:.1f} s "
                   f"(quality gate passed: quality {q:+.4f} {unit} >= {gate_min:g})")
    return {"quality_repeats": len(final), "final_quality_per_repeat": final, "final_quality": final_q,
            "base_quality_per_repeat": base, "base_quality": base_q, "quality_score": q,
            "quality_delta_per_repeat": [f - b for f, b in zip(final, base)], "quality_unit": unit,
            "quality_gate_min": gate_min, "quality_gate_passed": passed, "T_seconds": T,
            "T_per_replay": list(T_per_replay or []), "score": score,
            "score_formula": f"{scale_seconds:g} / T if correct and quality_score >= {float(gate_min)!r} else 0 "
                             "(quality_score is a gate, not a multiplier)",
            "summary": summary}


# ------------------------------------------------------------------------------------------------ eval inputs


def H(s: str) -> int:
    """Canonical 63-bit key (same definition as wan_dance_contract.H)."""
    return int.from_bytes(hashlib.sha256(s.encode("utf-8")).digest()[:8], "big") % (1 << 63)


def eval_seed(namespace: str, prompt_set: str, rep: int, row: int) -> int:
    return H(f"{namespace}|{prompt_set}|{rep}|{row}")


def parse_rows(spec: str | None, n: int) -> list[int]:
    if not spec:
        return list(range(n))
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b) + 1)) if b else [int(a)]
    return out


def read_prompts(path: str) -> list[str]:
    # identical to the task's handling of Hao7 train.txt rows: one prompt per non-empty line, stripped, verbatim
    # (literal "\n" escape sequences are kept, as in run_prompts.jsonl / the replay dumps)
    return [line.strip() for line in open(path, encoding="utf-8") if line.strip()]


# ------------------------------------------------------------------------------------------------ engines


def launch_engine(gpu: int, port: int, log: Path):
    """The Judge-engine command of run_judge.py::launch_judge_engines, as the invoking user and without the hook dir."""
    env = {"PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"), "HOME": os.environ.get("HOME", "/root"),
           "LANG": "C.UTF-8", "CUDA_VISIBLE_DEVICES": str(gpu), "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
           "PYTHONUNBUFFERED": "1", "TMPDIR": os.environ.get("TMPDIR", "/tmp")}
    cmd = [sys.executable, "-m", "sglang.multimodal_gen.runtime.launch_server", "--model-path", MODEL,
           "--port", str(port), "--host", "127.0.0.1", "--num-gpus", "1", "--warmup-mode", "off",
           "--trust-remote-code", "--master-port", str(port + 1000), "--scheduler-port", str(port + 2000),
           "--text-encoder-cpu-offload", "true", "--vae-cpu-offload", "true", "--dit-cpu-offload", "false"]
    fh = open(log, "w")
    return subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True), cmd


def wait_healthy(port: int, proc, timeout=1200) -> float:
    import requests

    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"engine on {port} exited rc={proc.returncode}")
        try:
            if requests.get(f"http://127.0.0.1:{port}/health_generate", timeout=5).status_code == 200:
                return time.time() - t0
        except Exception:
            pass
        time.sleep(2)
    raise TimeoutError(f"engine {port}")


def kill_engines(procs):
    import signal

    for p in procs:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def engine_checksum(port: int) -> str | None:
    import requests

    r = requests.post(f"http://127.0.0.1:{port}/get_weights_checksum", json={"module_names": ["transformer"]},
                      timeout=900)
    return r.json().get("transformer") if r.status_code == 200 else None


def load_model(port: int, master: str | None, tag: str, tmp_root: Path) -> dict:
    """base (master=None): reload MODEL (run_judge.reset_engines). Else the run_judge.reload_checksum pattern."""
    import requests

    t0 = time.time()
    if master is None:
        path = MODEL
    else:
        tmp = tmp_root / f"reload_{tag}_{port}"
        if tmp.exists():
            shutil.rmtree(tmp)
        (tmp / "transformer").mkdir(parents=True)
        for item in Path(MODEL).iterdir():
            if item.name != "transformer":
                (tmp / item.name).symlink_to(item)
        shutil.copy2(Path(MODEL) / "transformer" / "config.json", tmp / "transformer" / "config.json")
        (tmp / "transformer" / "diffusion_pytorch_model.safetensors").symlink_to(master)
        path = str(tmp)
    r = requests.post(f"http://127.0.0.1:{port}/update_weights_from_disk",
                      json={"model_path": path, "target_modules": ["transformer"]}, timeout=1800)
    if r.status_code != 200:
        raise RuntimeError(f"update_weights_from_disk {port} {tag}: HTTP {r.status_code} {r.text[:500]}")
    t1 = time.time()
    return {"checksum": engine_checksum(port), "load_s": t1 - t0, "checksum_s": time.time() - t1}


def decode_tensor(v):
    from safetensors.torch import load as st_load

    if isinstance(v, list):
        v = v[0]
    if isinstance(v, dict) and v.get("__tensor__"):
        return st_load(v["data"])["t"]
    raise TypeError(type(v))


def generate(port: int, body: dict):
    import msgpack
    import requests

    t0 = time.time()
    r = requests.post(f"http://127.0.0.1:{port}/rollout/generate", json=body, timeout=1800)
    if r.status_code != 200:
        raise RuntimeError(f"generate {port}: HTTP {r.status_code}: {r.text[:1000]}")
    lat = time.time() - t0
    return msgpack.unpackb(r.content, raw=False)[0], lat


def video_from_body(b) -> np.ndarray:
    v = decode_tensor(b["generated_output"])
    v = v.numpy() if hasattr(v, "numpy") else np.asarray(v)
    # engine returns [C, T, H, W] or [T, H, W, C] uint8 depending on path; normalise to [T, H, W, 3]
    if v.ndim == 5:
        v = v[0]
    if v.shape[-1] != 3 and v.shape[0] == 3:
        v = np.transpose(v, (1, 2, 3, 0))
    if v.dtype != np.uint8 or v.shape != (9, 704, 1280, 3):
        raise RuntimeError(f"unexpected video {v.dtype} {v.shape}")
    return np.ascontiguousarray(v)


class GpuMem(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.peak: dict[int, int] = {}
        self.stop = False

    def run(self):
        while not self.stop:
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=10).stdout
                for line in out.strip().splitlines():
                    i, m = (int(x) for x in line.split(","))
                    self.peak[i] = max(self.peak.get(i, 0), m)
            except Exception:
                pass
            time.sleep(1.0)


# ------------------------------------------------------------------------------------------------ eval


def sample_sd(xs):
    return statistics.stdev(xs) if len(xs) > 1 else float("nan")


def evaluate_model(label, master, ports, jobs, prompts, scorer, args, tmp_root: Path, out_dir: Path,
                   engines_hold_base: bool = False):
    res = {"label": label, "master": master, "eval_version": EVAL_VERSION}
    t0 = time.time()
    staged = None
    if master is not None and args.stage_dir:
        ts = time.time()
        Path(args.stage_dir).mkdir(parents=True, exist_ok=True)
        staged = str(Path(args.stage_dir) / f"{label}.safetensors")
        shutil.copyfile(master, staged)
        res["stage_s"] = time.time() - ts
        master_src, master = master, staged
    if master is None and engines_hold_base:  # fresh engines already hold theta0: no reload (checksum re-read)
        loads = [{"checksum": engine_checksum(p), "load_s": 0.0, "checksum_s": None, "skipped_reload": True}
                 for p in ports]
    else:
        with cf.ThreadPoolExecutor(len(ports)) as ex:
            loads = list(ex.map(lambda p: load_model(p, master, label, tmp_root), ports))
    res["load"] = loads
    res["load_wall_s"] = time.time() - t0
    cks = {x["checksum"] for x in loads}
    if len(cks) != 1:
        raise RuntimeError(f"engines disagree on checksum after loading {label}: {cks}")
    res["engine_checksum"] = cks.pop()
    if staged:
        os.unlink(staged)
        res["master"] = master_src
    print(f"[qeval] {label}: loaded in {res['load_wall_s']:.1f}s checksum {res['engine_checksum']}", flush=True)

    frames: dict[int, np.ndarray] = {}
    per: list[dict] = [None] * len(jobs)  # type: ignore[list-item]

    t_gen = time.time()
    rep_t: dict[int, float] = {}
    # repeats are generated one after another so that per-repeat wall time is measurable
    for rep in sorted({j[0] for j in jobs}):
        idx = [i for i, j in enumerate(jobs) if j[0] == rep]
        sub = [jobs[i] for i in idx]
        tr = time.time()

        def work_rep(ei):
            port = ports[ei]
            for k in range(ei, len(sub), len(ports)):
                ji = idx[k]
                _, row, seed = jobs[ji]
                body = dict(SAMPLER, prompt=prompts[row], seed=seed)
                b, lat = generate(port, body)
                v = video_from_body(b)
                frames[ji] = v[list(FRAME_INDICES)].copy()
                per[ji] = {"rep": rep, "row": row, "seed": seed, "engine": ei, "gen_s": lat,
                           "inference_time_s": b.get("inference_time_s"),
                           "video_sha256": hashlib.sha256(v.tobytes()).hexdigest()}

        with cf.ThreadPoolExecutor(len(ports)) as ex:
            list(ex.map(work_rep, range(len(ports))))
        rep_t[rep] = time.time() - tr
        print(f"[qeval] {label}: repeat {rep} generated {len(sub)} videos in {rep_t[rep]:.1f}s", flush=True)
    res["gen_wall_s"] = time.time() - t_gen
    res["gen_wall_per_repeat_s"] = {str(k): v for k, v in rep_t.items()}

    t_sc = time.time()
    import torch

    from hpsv3_scorer import resize_frame

    for ji in range(len(jobs)):
        _, row, _ = jobs[ji]
        # == HPSv3Scorer.score_videos for one video: frames 0/4/8, PIL bicubic resize, one batch of 3
        imgs = [resize_frame(np.ascontiguousarray(f)) for f in frames[ji]]
        mu = scorer.score_images(imgs, [prompts[row]] * len(imgs), batch_size=len(imgs))
        per[ji]["frame_mu"] = [float(x) for x in mu.tolist()]
        per[ji]["reward"] = float(REWARD_SCALE * mu.double().mean())
    res["score_wall_s"] = time.time() - t_sc
    del frames
    torch.cuda.empty_cache()

    reps = sorted({p["rep"] for p in per})
    means = [float(np.mean([p["reward"] for p in per if p["rep"] == r])) for r in reps]
    res["per_repeat_mean_reward"] = {str(r): m for r, m in zip(reps, means)}
    res["mean_reward"] = float(np.mean(means))
    res["sd_reward_across_repeats"] = sample_sd(means)
    res["per_video"] = per
    res["wall_s"] = time.time() - t0
    print(f"[qeval] {label}: per-repeat {['%.5f' % m for m in means]} mean {res['mean_reward']:.5f} "
          f"(gen {res['gen_wall_s']:.0f}s, score {res['score_wall_s']:.0f}s)", flush=True)
    (out_dir / f"model_{label}.json").write_text(json.dumps(res, indent=1))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--prompts", required=True, help="one prompt per line (Hao7 video_prompts test.txt)")
    ap.add_argument("--prompt-set", required=True, help="name mixed into the seed key, e.g. hao7_test")
    ap.add_argument("--prompts-sha256", default=None, help="refuse a prompt file with another sha256")
    ap.add_argument("--rows", default=None, help="row spec '0-60,70' (default: all rows)")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--rep-offset", type=int, default=0, help="first repeat index (repeat r uses seed set r)")
    ap.add_argument("--namespace", default=EVAL_VERSION)
    ap.add_argument("--model", action="append", required=True,
                    help="'base' or 'label=/path/master_fp32.safetensors'; repeatable, evaluated in order")
    ap.add_argument("--ports", default="30500,30510", help="one engine per visible GPU, in GPU order")
    ap.add_argument("--wan-dir", default=str(Path(__file__).resolve().parents[1]),
                    help="dir holding hpsv3_scorer.py (default: the parent of this file's directory)")
    ap.add_argument("--score-device", default="cuda:0")
    ap.add_argument("--tmp", default="/scratch/qeval_tmp", type=Path)
    ap.add_argument("--stage-dir", default="/dev/shm/qeval_stage",
                    help="copy each master here before /update_weights_from_disk ('' = load in place)")
    a = ap.parse_args()

    prompt_sha = hashlib.sha256(Path(a.prompts).read_bytes()).hexdigest()
    if a.prompts_sha256 and prompt_sha != a.prompts_sha256:
        raise SystemExit(f"{a.prompts}: sha256 {prompt_sha}, expected {a.prompts_sha256}")
    a.out.mkdir(parents=True, exist_ok=True)
    a.tmp.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, a.wan_dir)
    from hpsv3_scorer import HPSv3Scorer  # noqa: E402

    prompts = read_prompts(a.prompts)
    rows = parse_rows(a.rows, len(prompts))
    jobs = [(rep, row, eval_seed(a.namespace, a.prompt_set, rep, row))
            for rep in range(a.rep_offset, a.rep_offset + a.repeats) for row in rows]
    ports = [int(p) for p in a.ports.split(",")]
    models = []
    for m in a.model:
        if m == "base":
            models.append(("base", None))
        else:
            lab, _, path = m.partition("=")
            if not Path(path).is_file():
                raise SystemExit(f"master not found: {path}")
            models.append((lab, path))
    meta = {"eval_version": EVAL_VERSION, "namespace": a.namespace, "prompt_set": a.prompt_set, "prompts": a.prompts,
            "prompts_sha256": prompt_sha, "rows": rows,
            "repeats": list(range(a.rep_offset, a.rep_offset + a.repeats)), "sampler": SAMPLER,
            "engine_offload": True, "ports": ports, "models": models, "argv": sys.argv,
            "seed_rule": "H(f'{namespace}|{prompt_set}|{rep}|{row}') ; H = sha256[:8] big-endian mod 2**63",
            "scorer": {"ckpt": HPSV3_CKPT, "base": HPSV3_BASE, "frames": FRAME_INDICES, "reward_scale": REWARD_SCALE,
                       "batch": "3 frames of one video"}, "hostname": os.uname().nodename}
    (a.out / "meta.json").write_text(json.dumps(meta, indent=1))

    mem = GpuMem()
    mem.start()
    t_start = time.time()
    procs, cmds = [], []
    for g, port in enumerate(ports):
        p, cmd = launch_engine(g, port, a.out / f"engine{g}.log")
        procs.append(p)
        cmds.append(cmd)
    summary = {"meta": meta, "engine_cmds": cmds, "models": {}}
    try:
        with cf.ThreadPoolExecutor(len(ports)) as ex:
            cold = list(ex.map(lambda pp: wait_healthy(pp[0], pp[1]), zip(ports, procs)))
        summary["engine_cold_start_s"] = cold
        summary["mem_after_engine_load_mb"] = dict(mem.peak)
        t = time.time()
        scorer = HPSv3Scorer(HPSV3_CKPT, HPSV3_BASE, device=a.score_device)
        summary["scorer_load_s"] = time.time() - t
        # one throw-away video per engine (first request compiles / warms caches; not part of any repeat)
        tw = time.time()
        with cf.ThreadPoolExecutor(len(ports)) as ex:
            wl = list(ex.map(lambda ei: generate(ports[ei], dict(SAMPLER, prompt=prompts[rows[0]],
                                                                  seed=H(f"{a.namespace}|warmup|{ei}")))[1],
                             range(len(ports))))
        summary["warmup_s"] = {"per_engine": wl, "wall": time.time() - tw}
        holds_base = True
        for label, master in models:
            r = evaluate_model(label, master, ports, jobs, prompts, scorer, a, a.tmp, a.out, holds_base)
            holds_base = master is None
            summary["models"][label] = {k: v for k, v in r.items() if k != "per_video"}
            summary["mem_peak_mb"] = dict(mem.peak)
            (a.out / "summary.json").write_text(json.dumps(summary, indent=1))
    finally:
        mem.stop = True
        kill_engines(procs)
        summary["mem_peak_mb"] = dict(mem.peak)
        summary["total_wall_s"] = time.time() - t_start
        (a.out / "summary.json").write_text(json.dumps(summary, indent=1))
        shutil.rmtree(a.tmp, ignore_errors=True)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk in ("per_repeat_mean_reward", "mean_reward",
                                                                  "engine_checksum", "gen_wall_s")}
                      for k, v in summary["models"].items()}, indent=1))


if __name__ == "__main__":
    main()
