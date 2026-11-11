#!/usr/bin/env python3
"""Judge (MVP) for the miles_diffusion Wan2.2-TI2V-5B DanceGRPO + HPSv3 task.

Runs as root in the Judge container with 2 GPUs; candidate code runs as uid `agent` and is never imported.
Phases (any gate failure -> score 0; Judge/infrastructure errors -> valid=false):
  J0   static: engine_overlay ⊆ whitelist + AST policy; legacy sde/cps/ode bitwise vs the clean engine
  J2   `score` on the private score set vs the official-HPSv3 oracle (per-frame μ, reward definition)
  RUN  `run` under the in-engine hook -> J3 (params, canonical keys/windows, lag 0, publications, pair set), J4
  SERVE Judge-owned engines (clean engine, uid agent, hook) + candidate `serve`:
       J1 probe vs the fp32 oracle; then per hidden dump: T (dump readable -> last engine publication_complete),
       R1 per-frame μ vs oracle, R2 advantages, R3 π_old vs fp32 oracle, R4 self-consistency + composition,
       R5 one complete publication per engine, the per-pair log already final at `log_complete` (R5_log_final: the
       publication records are candidate-writable); R6' published == what each engine reports holding after `done`
       == engine_cast(dumped master) (reload + engine checksum);
       J6: the run's final fp32 master (staged on tmpfs) reloaded into a Judge engine == the run's last publication checksum
       (these engines run as the candidate's uid: a candidate that replaces one can answer the checksum queries itself)
  QUALITY (only when every check above passed; after the candidate and the SERVE engines are gone): the run's final
       master on fresh root engines (quality_eval.py: 61 Hao7 test prompts x 3 seed sets): their checksum == J6's
       (J6.eval_engines), then held-out HPSv3 -> quality_gate
score = 3600 / median(T)  if every check passes and quality_score >= quality_gate_min, else 0
        quality_score = mean of the 3 repeats - the original model's frozen 3-repeat mean (HPSv3 reward); a
        non-regression gate and a reported diagnostic, not a multiplier (quality_eval.metric_v2; the "metric" block of
        runtime/qualification.json, baked into the image as metric.json)
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import re
import shutil
import site
import stat
import subprocess
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import checks_probe  # noqa: E402
import checks_replay  # noqa: E402
import checks_run  # noqa: E402
import checks_score  # noqa: E402
import quality_eval  # noqa: E402

AGENT = 1001
MODEL = "/models/Wan2.2-TI2V-5B-Diffusers"
CLEAN_ENGINE = Path("/sgl-workspace/sglang/python")
WHITELIST = {
    "configs/post_training/rl_rollout.py", "configs/sample/sampling_params.py",
    "runtime/post_training/scheduler_rl_mixin.py", "runtime/post_training/rl_dataclasses.py",
    "runtime/pipelines_core/stages/latent_preparation.py",
    "runtime/entrypoints/post_training/io_struct.py", "runtime/entrypoints/post_training/rollout_api.py",
}
FORBIDDEN_IMPORTS = {"sys", "importlib", "inspect", "ctypes", "gc", "builtins"}
STAGE = Path("/dev/shm/judge_stage")        # hidden dumps, staged per replay (root 0711; each dump go-rwx until go)
INPUTS = Path("/scratch/judge_inputs")      # agent-readable copies of the private inputs a phase hands to the candidate
# PUBLIC (run_public.py in the Work container): same checks, run as the agent itself on public data. No setpriv/chown,
# and only the process groups this runner started are killed (never `pkill -u agent`: that would kill the agent).
PUBLIC = False
PGIDS: list = []
MASTER_DIR = Path("/dev/shm/judge_master")  # candidate writes the fp32 master here after the clock stops
# The hook runs inside engines as uid agent, so its directory must be reachable by them. The out dir /logs/verifier is
# root 0700 in the Judge image, so a dir under it makes the hook silently inert.
AUDIT_ROOT = Path("/scratch/judge_audit")
# metric v2: runtime/qualification.json "metric" and the held-out prompts, baked into the image next to judge/ by
# build_images.sh (the private root keeps the replay list, tau_pi_old and R6 thresholds)
METRIC_FILE = HERE.parent / "metric.json"
# Directories the Judge relies on after candidate code ran live on the container's own /dev/shm (sticky, root-owned) and
# are created root 0755 by main() before any candidate code: /scratch and /tmp are host bind mounts the candidate's uid
# can write (on hosts whose launching user has uid 1001 it even owns them), so it could plant files or symlinks there.
EVAL_DIR = Path("/dev/shm/judge_eval")      # the run's final master staged for J6 and QUALITY; QUALITY's temp dirs
RELOAD_DIR = Path("/dev/shm/judge_reload")  # the model dirs of the R6' / J6 reloads (symlinks to MODEL and a master)
STAGE_TIMEOUT_S = 600  # copying the 20 GB final master (measured 9-12 s); longer (e.g. a FIFO) -> J6 fails
# measured ~9-10 min (QUALITY phase 552 s and 582 s in the validation Judges): engines ~50-58 s, HPSv3 ~8-10 s, warmup
# ~37-45 s, master load ~14-37 s, 3 x ~122 s generation, ~34-55 s scoring. Capped so that the slowest known candidate
# Judge before metric v2 (2,627 s: one run's 3rd submission, whose J6 still read the master from its disk, the read the
# 9-12 s staging copy replaces) + wait_gpus_free (<= 180 s) + this stays < 4,200 s. A candidate that stretches its own
# copy towards STAGE_TIMEOUT_S can exceed that, as it can by stalling any protocol event (up to 1,800 s each)
QUALITY_TIMEOUT_S = 2100  # the 50-step eval of one model measured ~970 s (10 steps: ~450 s, cap 1200)


class JudgeError(Exception):
    """A Judge-side failure inside a phase: valid=false (a RuntimeError there would count as the candidate's)."""


# ------------------------------------------------------------------------------------------------ plumbing


class Ctx:
    def __init__(self, a):
        self.a = a
        self.out = a.out
        self.private = a.private
        self.qual = json.loads((a.private / "qualification.json").read_text())
        self.metric = json.loads(METRIC_FILE.read_text()) if METRIC_FILE.exists() else None
        self.quality: dict | None = None  # quality_eval.metric_v2 of the QUALITY phase
        self.gates: dict = {}
        self.timings: dict = {}
        self.home = Path("/scratch/agenthome")

    def gate(self, name, res: dict):
        self.gates[name] = res
        ok = all(v.get("ok", True) for v in res.values() if isinstance(v, dict)) if "ok" not in res else res["ok"]
        print(f"[judge] gate {name}: {'PASS' if ok else 'FAIL'}", flush=True)
        return ok


def agent_env(ctx: Ctx, extra=None):
    env = {"PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"), "HOME": str(ctx.home), "LANG": "C.UTF-8",
           "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0,1"), "HF_HUB_OFFLINE": "1",
           "TRANSFORMERS_OFFLINE": "1", "WANDB_MODE": "disabled", "RAY_USAGE_STATS_ENABLED": "0",
           "PYTHONUNBUFFERED": "1", "TMPDIR": "/scratch/tmp"}
    env.update(extra or {})
    return env


def as_agent_cmd(cmd):
    return list(cmd) if PUBLIC else ["setpriv", "--reuid", str(AGENT), "--regid", str(AGENT), "--clear-groups"] + cmd


def reap_agent():
    if PUBLIC:
        import signal

        for pg in PGIDS:
            try:
                os.killpg(pg, signal.SIGKILL)
            except ProcessLookupError:
                pass
        PGIDS.clear()
        subprocess.run(["ray", "stop", "--force"], check=False, capture_output=True)
    else:
        subprocess.run(["pkill", "-KILL", "-u", str(AGENT)], check=False)
    time.sleep(3)


def spawn(cmd, **kw):
    """Start a candidate/engine process in its own session and remember its process group (public-mode cleanup)."""
    p = subprocess.Popen(as_agent_cmd(cmd), start_new_session=True, **kw)
    PGIDS.append(p.pid)
    return p


def run_candidate(cmd, timeout, **kw):
    p = spawn(cmd, **kw)
    try:
        return p.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        reap_agent()
        raise


def readable_copy(src: Path, name: str) -> Path:
    """Copy a private input the candidate must read into INPUTS (the private root itself is a read-only mount)."""
    dst = INPUTS / name
    if dst.exists():
        shutil.rmtree(dst) if dst.is_dir() else dst.unlink()
    (shutil.copytree if src.is_dir() else shutil.copy2)(src, dst)
    subprocess.run(["chmod", "-R", "a+rX", str(dst)], check=True)
    return dst


def fresh_submission(ctx: Ctx, name: str) -> Path:
    d = Path("/scratch/judge") / name
    if d.exists():
        shutil.rmtree(d)
    shutil.copytree(ctx.a.submission, d / "repo", symlinks=True)
    (d / "out").mkdir()
    (d / "scratch").mkdir()
    if not PUBLIC:
        subprocess.run(["chown", "-R", f"{AGENT}:{AGENT}", str(d)], check=True)
    return d


def audit_dir(ctx: Ctx, name: str) -> Path:
    root = ctx.out if PUBLIC else AUDIT_ROOT
    root.mkdir(parents=True, exist_ok=True)
    if not PUBLIC:
        os.chmod(root, 0o711)
    return root / name


def install_hook(audit: Path):
    audit.mkdir(parents=True, exist_ok=True)
    os.chmod(audit, 0o777)
    probe = subprocess.run(as_agent_cmd([sys.executable, "-c", f"import os; assert os.access({str(audit)!r}, os.W_OK | os.X_OK)"]),
                           capture_output=True)
    if probe.returncode != 0:  # Judge bug, not the candidate's: the hook would record nothing
        raise OSError(f"audit dir {audit} is not writable by the candidate uid")
    sp = Path(site.getsitepackages()[0])
    if not (sp / "wan_audit_hook.pth").exists():  # the task images bake it in; spikes install it here
        shutil.copy2(HERE.parent / "audit" / "wan_audit_hook.py", sp / "wan_audit_hook.py")
        (sp / "wan_audit_hook.pth").write_text("import wan_audit_hook\n")
    (audit / "expected.json").write_text(json.dumps(expected_names()))


def expected_names():
    import struct

    out = {}
    for f in sorted((Path(MODEL) / "transformer").glob("*.safetensors")):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            h = json.loads(fh.read(n))
        for k, v in h.items():
            if k != "__metadata__":
                out[k] = int(math.prod(v["shape"])) if v["shape"] else 1
    return out


# ------------------------------------------------------------------------------------------------ J0


def ast_policy(src: str) -> list[str]:
    bad = []
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            bad += [f"import {a.name}" for a in node.names if a.name.split(".")[0] in FORBIDDEN_IMPORTS]
        elif isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] in FORBIDDEN_IMPORTS:
            bad.append(f"from {node.module} import")
        elif isinstance(node, ast.Call) and getattr(node.func, "id", None) in ("__import__", "setattr", "exec", "eval"):
            bad.append(f"call {node.func.id}")
        elif isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute) and \
                node.value.attr == "environ" and isinstance(node.ctx, ast.Store):
            bad.append("os.environ write")
    return bad


def phase_j0(ctx: Ctx) -> Path:
    overlay = ctx.a.submission / "engine_overlay"
    # the starter ships engine_overlay/README.md (documentation only; ignored); every other file must be whitelisted
    files = sorted(str(f.relative_to(overlay)) for f in overlay.rglob("*")
                   if f.is_file() and str(f.relative_to(overlay)) != "README.md") if overlay.exists() else []
    outside = [f for f in files if f not in WHITELIST]
    # the policy applies to what the candidate ADDED: violations already present in the clean file are not counted
    from collections import Counter

    policy = {}
    for f in files:
        if f.endswith(".py"):
            clean = CLEAN_ENGINE / "sglang" / "multimodal_gen" / f
            base = Counter(ast_policy(clean.read_text())) if clean.exists() else Counter()
            policy[f] = sorted((Counter(ast_policy((overlay / f).read_text())) - base).elements())
    eng = Path("/scratch/engine_python")
    if eng.exists():
        shutil.rmtree(eng)
    shutil.copytree(CLEAN_ENGINE, eng, symlinks=True)
    for f in files:
        if f in WHITELIST:
            shutil.copy2(overlay / f, eng / "sglang" / "multimodal_gen" / f)
    subprocess.run(["chmod", "-R", "a+rX", str(eng)], check=True)
    # legacy branches bitwise vs the clean engine (CPU)
    env = dict(os.environ, PRISTINE_SGLANG_PYTHON=str(CLEAN_ENGINE), OVERLAY_SGLANG_PYTHON=str(eng),
               PYTHONPATH=f"{eng}:{HERE.parent}", ALLOW_ANY_ENGINE="1")
    p = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                        str(HERE.parent / "tests" / "test_engine_overlay.py::test_legacy_branches_unchanged_vs_pristine_engine")],
                       env=env, capture_output=True, text=True, timeout=900)
    legacy_ok = p.returncode == 0
    ctx.gate("J0", {"whitelist": {"ok": not outside, "outside": outside, "files": files},
                    "ast_policy": {"ok": not any(policy.values()), "violations": {k: v for k, v in policy.items() if v}},
                    "legacy_bitwise": {"ok": legacy_ok, "tail": (p.stdout + p.stderr)[-800:]}})
    return eng


# ------------------------------------------------------------------------------------------------ J2 score


def phase_j2(ctx: Ctx):
    d = fresh_submission(ctx, "score")
    ss = ctx.private / "score_set"
    vis = readable_copy(ss / "videos.npz", "score_videos.npz"), readable_copy(ss / "prompts.jsonl", "score_prompts.jsonl")
    t0 = time.time()
    with open(ctx.out / "score.log", "w") as log:
        rc = run_candidate([sys.executable, "task_entry.py", "score", "--videos", str(vis[0]), "--prompts", str(vis[1]),
                            "--out", str(d / "out")], 1800, cwd=d / "repo", env=agent_env(ctx), stdout=log,
                           stderr=subprocess.STDOUT)
    reap_agent()
    ctx.timings["score_s"] = time.time() - t0
    if rc != 0 or not (d / "out" / "scores.json").exists():
        return ctx.gate("J2", {"ok": False, "rc": rc, "tail": (ctx.out / "score.log").read_text()[-1500:]})
    return ctx.gate("J2", checks_score.check_scores(json.loads((d / "out" / "scores.json").read_text()),
                                                   json.loads((ss / "hpsv3_oracle.json").read_text())))


# ------------------------------------------------------------------------------------------------ RUN


def phase_run(ctx: Ctx, eng: Path):
    d = fresh_submission(ctx, "run")
    audit = audit_dir(ctx, "run_audit")
    install_hook(audit)
    c = json.loads((ctx.private / "contract.json").read_text())
    c.update(scratch=str(d / "scratch"), engine_python_path=str(eng))
    c["run"]["prompts"] = str(readable_copy(ctx.private / "run_prompts.jsonl", "run_prompts.jsonl"))
    cpath = d / "contract.json"
    cpath.write_text(json.dumps(c, indent=1))
    env = agent_env(ctx, {"WAN_AUDIT_DIR": str(audit), "WAN_AUDIT_EXPECTED": str(audit / "expected.json"),
                          "WAN_AUDIT_CHECKSUM": "1"})
    t0 = time.time()
    with open(ctx.out / "run.log", "w") as log:
        rc = run_candidate([sys.executable, "task_entry.py", "run", "--contract", str(cpath), "--out", str(d / "out")],
                           int(ctx.qual["run_timeout_s"]), cwd=d / "repo", env=env, stdout=log,
                           stderr=subprocess.STDOUT)
    reap_agent()
    ctx.timings["run_s"] = time.time() - t0
    if rc != 0:
        ctx.gate("J3", {"ok": False, "rc": rc})
        return None
    res = checks_run.check_run(c, audit, d / "out" / "pairs", j4_samples=ctx.qual.get("j4_samples"))
    ctx.gate("J3_J4", res)
    # the engine holds engine_cast(master): most parameters RNE-bf16, but the engine keeps some (affine residual-norm
    # weights) in fp32, so an all-bf16 file is NOT what it holds (ref_v0_1 J6 false fail) -> reload the fp32 master
    return {"final_master": d / "out" / "final" / "master_fp32.safetensors",
            "final_checksums": res.get("final_publication_checksums")}


# ------------------------------------------------------------------------------------------------ SERVE


def launch_judge_engines(ctx: Ctx, audit: Path, ports):
    engines = []
    for g, port in enumerate(ports):
        env = agent_env(ctx, {"CUDA_VISIBLE_DEVICES": str(g), "WAN_AUDIT_DIR": str(audit),
                              "WAN_AUDIT_EXPECTED": str(audit / "expected.json"), "WAN_AUDIT_CHECKSUM": "1"})
        cmd = [sys.executable, "-m", "sglang.multimodal_gen.runtime.launch_server", "--model-path", MODEL,
               "--port", str(port), "--host", "127.0.0.1", "--num-gpus", "1", "--warmup-mode", "off",
               "--trust-remote-code", "--master-port", str(port + 1000), "--scheduler-port", str(port + 2000),
               "--text-encoder-cpu-offload", "true", "--vae-cpu-offload", "true", "--dit-cpu-offload", "false"]
        log = open(ctx.out / f"judge_engine{g}.log", "w")
        engines.append(spawn(cmd, env=env, stdout=log, stderr=subprocess.STDOUT))
    return engines


def wait_healthy(port, proc, timeout=1200):
    import requests

    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"judge engine on {port} exited rc={proc.returncode}")
        try:
            if requests.get(f"http://127.0.0.1:{port}/health_generate", timeout=5).status_code == 200:
                return time.time() - t0
        except Exception:
            pass
        time.sleep(2)
    raise TimeoutError(f"judge engine {port}")


def hook_records(audit: Path, kind: str):
    return [r for r in checks_run.records(audit) if r.get("kind") == kind]


def reload_checksum(port: int, transformer_file: Path, tag: str) -> str | None:
    """Reload a Judge engine's transformer from a safetensors file (engine casts to bf16) and return its checksum."""
    import requests

    tmp = RELOAD_DIR / tag
    if tmp.exists():
        shutil.rmtree(tmp)
    (tmp / "transformer").mkdir(parents=True)
    for item in Path(MODEL).iterdir():
        if item.name != "transformer":
            (tmp / item.name).symlink_to(item)
    shutil.copy2(Path(MODEL) / "transformer" / "config.json", tmp / "transformer" / "config.json")
    (tmp / "transformer" / "diffusion_pytorch_model.safetensors").symlink_to(transformer_file)
    subprocess.run(["chmod", "-R", "a+rX", str(tmp)])
    r = requests.post(f"http://127.0.0.1:{port}/update_weights_from_disk",
                      json={"model_path": str(tmp), "target_modules": ["transformer"]}, timeout=900)
    if r.status_code != 200:
        return None
    return requests.post(f"http://127.0.0.1:{port}/get_weights_checksum", json={"module_names": ["transformer"]},
                         timeout=900).json().get("transformer")


def engine_checksums(ports) -> list:
    """What each Judge engine reports holding now (None for an engine that does not answer). The Judge queries the
    engines itself, which catches a publication recorded but not applied; the engines run as the candidate's uid, so
    it is not evidence against a candidate that replaces an engine and answers on its port."""
    from concurrent.futures import ThreadPoolExecutor

    def one(port):
        try:
            return quality_eval.engine_checksum(port)
        except Exception:  # noqa: BLE001 — an engine the candidate broke fails the gate, not the Judge
            return None

    with ThreadPoolExecutor(len(ports)) as ex:
        return list(ex.map(one, ports))


def judge_dir(p: Path):
    """`p` must be what main() created before any candidate code ran: a real directory (not a symlink) owned by the
    Judge's uid with mode 0755, so the candidate can neither write into it nor swap it."""
    st = os.lstat(p)
    if not stat.S_ISDIR(st.st_mode) or st.st_uid != os.getuid() or stat.S_IMODE(st.st_mode) != 0o755:
        raise JudgeError(f"{p}: mode {oct(st.st_mode)} uid {st.st_uid}, expected a 0755 directory of uid {os.getuid()}")


def stage_master(src: Path) -> Path | None:
    """Root-owned tmpfs copy of the run's final fp32 master, taken after the timed replays. J6 reloads it in seconds
    (the Run:ai streamer needs ~340 s for 20 GB from a cold scratch disk) and QUALITY evaluates exactly these bytes,
    which the candidate can no longer change. The candidate's file is read as the candidate's uid (it may be a
    symlink to anything); the copy is created exclusively, without following links, in the Judge's own EVAL_DIR. None
    when that uid cannot read a non-empty master there within STAGE_TIMEOUT_S (J6 then fails)."""
    judge_dir(EVAL_DIR)
    dst = EVAL_DIR / "master_fp32.safetensors"
    dst.unlink(missing_ok=True)
    fd = os.open(dst, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        try:
            rc = subprocess.run(as_agent_cmd(["cat", "--", str(src)]), stdout=fd, stderr=subprocess.DEVNULL,
                                timeout=STAGE_TIMEOUT_S).returncode
        except subprocess.TimeoutExpired:  # e.g. a FIFO: the candidate would decide when the copy ends
            rc = None
        size = os.fstat(fd).st_size
        if rc == 0 and size > 0:
            os.fchmod(fd, 0o644)
    finally:
        os.close(fd)
    if rc != 0 or size == 0:
        dst.unlink()
        return None
    return dst


def reset_engines(ports):
    import requests

    for port in ports:
        requests.post(f"http://127.0.0.1:{port}/update_weights_from_disk",
                      json={"model_path": MODEL, "target_modules": ["transformer"]}, timeout=900)


class Serve:
    """FIFO protocol client (non-blocking reads, liveness checks, timeouts)."""

    def __init__(self, proc, ev: Path, ctl: Path):
        self.proc, self.ctl_path, self.buf, self.ctl = proc, ctl, b"", None
        self.ev_fd = os.open(ev, os.O_RDONLY | os.O_NONBLOCK)

    def wait(self, kind, timeout=1800):
        t0 = time.time()
        while time.time() - t0 < timeout:
            while b"\n" in self.buf:
                line, self.buf = self.buf.split(b"\n", 1)
                if line.strip():
                    e = json.loads(line)
                    if e.get("event") == "error":
                        raise RuntimeError(e)
                    if e.get("event") == kind:
                        return e
            try:
                chunk = os.read(self.ev_fd, 65536)
            except BlockingIOError:
                chunk = b""
            if chunk:
                self.buf += chunk
                continue
            if self.proc.poll() is not None:
                raise RuntimeError(f"candidate exited rc={self.proc.returncode} waiting for {kind}")
            time.sleep(0.1)
        raise TimeoutError(kind)

    def send(self, obj):
        if self.ctl is None:
            t0 = time.time()
            while True:
                try:
                    fd = os.open(self.ctl_path, os.O_WRONLY | os.O_NONBLOCK)
                    os.set_blocking(fd, True)
                    self.ctl = os.fdopen(fd, "w", buffering=1)
                    break
                except OSError:
                    if self.proc.poll() is not None or time.time() - t0 > 300:
                        raise RuntimeError("candidate never opened the control FIFO")
                    time.sleep(0.2)
        self.ctl.write(json.dumps(obj) + "\n")
        self.ctl.flush()


def replay_gates(ctx: Ctx, dump: Path, pair_dir: Path, oracle: Path) -> dict:
    meta = json.loads((dump / "meta.json").read_text())
    res = checks_replay.check_replay(meta, pair_dir, int(ctx.qual["replay_optimizer_steps"]))
    # R1: per-frame μ the candidate trained with (pair log) vs the official scorer oracle
    ev = checks_replay.load_events(pair_dir)
    orc = json.loads((oracle / "hpsv3_oracle.json").read_text())
    rows = {f"{meta['run_seed']}|{meta['round']}|{r}": g for g, r in enumerate(meta["rows"])}
    cand = {}
    for e in ev:
        if e["kind"] == "step" and e.get("frame_mu") is not None:
            cand[(rows[e["group"]], e["j"])] = e["frame_mu"]
    import torch

    if len(cand) == len(meta["rows"]) * int(meta["n"]):
        c = torch.tensor([cand[(g, j)] for g in range(len(meta["rows"])) for j in range(int(meta["n"]))],
                         dtype=torch.float64)
        o = torch.tensor(orc["frame_mu"], dtype=torch.float64).reshape(-1, 3)
        res["R1_reward"] = checks_score.check_scores({"frame_mu": c.tolist(), "reward": (0.1 * c.mean(-1)).tolist()},
                                                     {"frame_mu": o.tolist()})["J2_frame_mu"]
    else:
        res["R1_reward"] = {"ok": False, "reason": "frame_mu missing from pair log", "n": len(cand)}
    # R3: all π_old vs the fp32 oracle (distribution gate)
    po = json.loads((oracle / "train_oracle_pi_old.json").read_text())["pi_old"]
    got = {}
    for e in ev:
        key = f"{e['group']}|{e['j']}|{e['k']}"
        if e["kind"] == "pi_old" or (e["kind"] == "step" and e["optim_step"] == 0):
            got[key] = e["lp_old"]
    sd = torch.tensor([got[k] - v for k, v in po.items() if k in got], dtype=torch.float64)
    d = sd.abs()
    tau = ctx.qual["tau_pi_old"]
    # signed lower bound: honest bf16 is +5.7e-5 (fp32 ~0); transition-math mean shifts are negative (S6d, checks_probe)
    signed_min = tau.get("signed_min", -5e-5)
    res["R3_pi_old"] = {"ok": len(d) == len(po) and float(d.mean()) <= tau["mean"] and float(d.max()) <= tau["max"]
                        and float(sd.mean()) >= signed_min,
                        "n": int(d.numel()), "mean": float(d.mean()) if d.numel() else None,
                        "max": float(d.max()) if d.numel() else None,
                        "signed_mean": float(sd.mean()) if d.numel() else None}
    return res


_THETA0_U = {}


def r6_traj(ctx: Ctx, master: Path, name: str, i: int) -> dict:
    """R6 trajectory: the candidate's cumulative update on the fixed coordinate sample U vs the precomputed fp32
    4-step oracle trajectory. RECORD-ONLY until S6 calibrates thresholds from honest variants and planted bugs."""
    import torch
    import traj_delta as T

    try:
        if not _THETA0_U:
            files = T.st_files(Path(MODEL) / "transformer")
            names, numels, U = T.coord_sample(T.safetensors_layout(files))
            _THETA0_U.update(names=names, numels=numels, U=U, theta0=T.gather(files, names, numels, U))
        z = _THETA0_U
        dc = T.gather(T.st_files(master), z["names"], z["numels"], z["U"]) - z["theta0"]
        torch.save({"U_seed": T.U_SEED, "delta_U": dc.float()}, ctx.out / f"delta_U_r{i}_{name}.pt")
        res = {"ok": True, "record_only": True, "moved_frac": float((dc != 0).double().mean()),
               "absmean": float(dc.abs().mean())}
        orc = ctx.private / "oracles" / name / "train_oracle_traj.pt"
        if orc.exists():
            o = torch.load(orc, map_location="cpu", weights_only=True)
            assert torch.equal(o["U"], z["U"]), "oracle U differs from the Judge's U"
            res.update(T.update_metrics(dc, o["delta_U"].double()))
            if "delta_U_steps" in o:
                res.update(T.step1_decomposition(dc, o["delta_U_steps"][0].double()))
                res["oracle_step1_resid"] = T.step1_decomposition(o["delta_U"].double(),
                                                                  o["delta_U_steps"][0].double())["step1_resid"]
        else:
            res["oracle"] = "missing"
        return res
    except Exception as e:  # noqa: BLE001 — a master that cannot be read on U fails the gate
        return {"ok": False, "error": repr(e)[:500]}


# S6 calibration (4 honest implementations x 3 dumps vs 6 R6 plants; docs/.../wan22-dancegrpo/README.md 2026-09-24):
# honest medians norm 0.90-0.98, corr 0.41-0.85, step1_resid 0.83-0.89, step1_coef 1.44-1.64; every plant fails >= 2.
R6_DEFAULT = {"norm_lo": 0.75, "norm_hi": 1.30, "corr_min": 0.28, "resid_min": 0.70, "coef_lo": 1.20, "coef_hi": 2.20}


def r6_aggregate(ctx: Ctx) -> dict:
    """R6 gate: medians over the hidden dumps of the per-dump trajectory metrics (one dump can be chaotic: its clip set
    sits on the 1e-4 boundary). norm ratio catches skipped/partial training and wrong optimizers, corr catches noise
    and sub-sampling, the step-1 residual catches updates that are a multiple of the first step, and the projection
    coefficient on the oracle's first-step update (honest 4-step AdamW: ~1.5, the most stable metric) catches all
    six R6 plants on its own (S6c/S6e calibration)."""
    th = {**R6_DEFAULT, **ctx.qual.get("r6", {})}
    per = [v["R6_traj"] for v in ctx.gates.values() if isinstance(v, dict) and isinstance(v.get("R6_traj"), dict)]
    need = ("norm_ratio", "corr", "step1_resid", "step1_coef")
    if not per or any(any(m not in r for m in need) for r in per):
        return {"ok": False, "reason": "R6 metrics missing", "per_dump": per}

    def med(m):
        xs = sorted(float(r[m]) for r in per)
        return xs[len(xs) // 2]

    agg = {m: med(m) for m in need}
    checks = {"norm": th["norm_lo"] <= agg["norm_ratio"] <= th["norm_hi"], "corr": agg["corr"] >= th["corr_min"],
              "step1_resid": agg["step1_resid"] >= th["resid_min"],
              "step1_coef": th["coef_lo"] <= agg["step1_coef"] <= th["coef_hi"]}
    return {"ok": all(checks.values()), **agg, "checks": checks, "thresholds": th}


def phase_serve(ctx: Ctx, run_info):
    ports = [30500, 30510]
    audit = audit_dir(ctx, "judge_audit")
    install_hook(audit)
    engines = launch_judge_engines(ctx, audit, ports)
    d = fresh_submission(ctx, "serve")
    c = json.loads((ctx.private / "contract.json").read_text())
    c.update(scratch=str(d / "scratch"))
    c["replay"]["engine_urls"] = [f"http://127.0.0.1:{p}" for p in ports]
    cpath = d / "contract.json"
    cpath.write_text(json.dumps(c, indent=1))
    ctl, ev = d / "ctl.fifo", d / "events.fifo"
    for f in (ctl, ev):
        os.mkfifo(f)
        os.chmod(f, 0o666)
    t0 = time.time()
    log = open(ctx.out / "serve.log", "w")
    cand = spawn([sys.executable, "task_entry.py", "serve", "--contract", str(cpath), "--ctl", str(ctl), "--events",
                  str(ev), "--out", str(d / "out")], cwd=d / "repo", env=agent_env(ctx), stdout=log,
                 stderr=subprocess.STDOUT)
    s = Serve(cand, ev, ctl)
    try:
        ctx.timings["judge_engines_cold_s"] = [wait_healthy(p, e) for p, e in zip(ports, engines)]
        s.wait("ready", timeout=1500)
        ctx.timings["serve_cold_s"] = time.time() - t0
        # J1 probe
        probe = ctx.private / "probe"
        probe_dump = STAGE / "probe"
        if probe_dump.exists():
            shutil.rmtree(probe_dump)
        shutil.copytree(probe / "dump", probe_dump)
        subprocess.run(["chmod", "-R", "a+rX", str(probe_dump)], check=True)
        s.send({"cmd": "probe", "id": "probe", "dump": str(probe_dump), "prompts": 1,
                "theta1_path": str(d / "out" / "probe_theta1.safetensors")})
        s.wait("done")
        ctx.gate("J1", checks_probe.check_probe(probe / "oracle", d / "out" / "pairs" / "probe00",
                                                d / "out" / "probe_theta1.safetensors",
                                                Path(MODEL) / "transformer"))
        s.send({"cmd": "reset", "id": "probe_reset"})
        s.wait("done")
        Ts, T_log, T_done = [], [], []
        for i, name in enumerate(ctx.qual["replay_dumps"][: ctx.a.max_replays]):
            # stage the hidden dump on tmpfs before the clock (untimed; root-only until the go), so T measures the
            # candidate and not the shared /data disk (other sessions' checkpoint I/O slowed disk reads 8 s -> 103 s in S5f)
            dump = STAGE / f"{i:02d}_{name}"
            if dump.exists():
                shutil.rmtree(dump)
            shutil.copytree(ctx.private / "dumps" / name, dump)
            if not PUBLIC:
                subprocess.run(["chmod", "-R", "go-rwx", str(dump)], check=True)
            pair_dir = d / "out" / "pairs" / f"replay{i + 1:02d}"
            n_before = len(hook_records(audit, "publication_complete"))
            t_go = time.time()
            subprocess.run(["chmod", "-R", "go+rX", str(dump)], check=True)  # the clock starts: dump readable
            s.send({"cmd": "replay", "id": f"r{i}", "dump": str(dump)})
            s.wait("log_complete")
            t_log = time.time()
            log_at_t_log = checks_replay.log_digest(pair_dir)  # a few hundred kB: the log R1-R4 check must be final now
            s.wait("done")
            t_done = time.time()
            deadline = time.time() + 120
            while len(hook_records(audit, "publication_complete")) < n_before + len(ports) and time.time() < deadline:
                time.sleep(0.5)
            pubs = sorted(hook_records(audit, "publication_complete"), key=lambda r: r["t_complete"])[n_before:]
            held = engine_checksums(ports)  # untimed; before the R6' reload replaces ports[0]'s weights
            # the clock stops at the later of: every Judge engine holds the full publication, and the candidate has
            # reported its per-pair log complete (so the log cannot be deferred past the timed window)
            T = max([p["t_complete"] for p in pubs] + [t_log]) - t_go if len(pubs) == len(ports) else None
            gates = replay_gates(ctx, dump, pair_dir, ctx.private / "oracles" / name)
            gates.update(checks_replay.check_publications(pubs, len(ports), t_go, None))
            gates.update(checks_replay.check_log_final(log_at_t_log, checks_replay.log_digest(pair_dir)))
            master = MASTER_DIR / f"master_r{i}.safetensors"  # tmpfs: the R6' reload does not touch the disk
            s.send({"cmd": "dump_master", "id": f"m{i}", "path": str(master)})
            s.wait("done")
            ck = reload_checksum(ports[0], master, f"r{i}")
            gates["R6_equal"] = {"ok": ck is not None and all(p.get("checksum") == ck for p in pubs)
                                 and all(h == ck for h in held),
                                 "master": ck, "published": [p.get("checksum") for p in pubs], "held_at_done": held}
            gates["R6_traj"] = r6_traj(ctx, master, name, i)
            ctx.gate(f"R{i}_{name}", gates)
            Ts.append(T)
            T_log.append(t_log - t_go)
            T_done.append(t_done - t_go)
            master.unlink(missing_ok=True)
            shutil.rmtree(dump, ignore_errors=True)
            reset_engines(ports)
            s.send({"cmd": "reset", "id": f"x{i}"})
            s.wait("done")
            print(f"[judge] replay {name}: T={T}", flush=True)
        ctx.timings["T"] = Ts
        # the Judge clock's own view of each T: dump readable -> log_complete (a lower bound: the per-pair log is final
        # then) and -> `done` (after the publication, per the protocol); between them T rests on the hook's records
        ctx.timings["T_log"], ctx.timings["T_done"] = T_log, T_done
        if Ts:  # the public `probe` suite runs no replay
            ctx.gate("R6_traj_agg", r6_aggregate(ctx))
        if run_info and run_info.get("final_checksums"):
            t0 = time.time()
            staged = stage_master(run_info["final_master"])
            ctx.timings["stage_master_s"] = time.time() - t0
            ck = reload_checksum(ports[0], staged, "runfinal") if staged else None
            ctx.gate("J6", {"ok": ck is not None and all(x == ck for x in run_info["final_checksums"]),
                            "reloaded": ck, "run_last_publication": run_info["final_checksums"],
                            **({} if staged else {"reason": "no final/master_fp32.safetensors readable by the "
                                                            f"candidate uid within {STAGE_TIMEOUT_S} s"})})
            run_info.update(staged_master=staged, j6_checksum=ck)
        s.send({"cmd": "stop"})
    finally:
        for p in [cand] + engines:
            try:
                os.killpg(p.pid, 9)
            except Exception:
                pass
        reap_agent()


# ------------------------------------------------------------------------------------------------ QUALITY


def agent_alive() -> list[int]:
    """Live processes of the candidate's uid. Zombies do not count: the container's PID 1 (`tail`) never reaps the
    orphans the killed candidate leaves behind, and they hold no GPU memory and cannot act."""
    alive = []
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:
            status = dict(line.split(":", 1) for line in (d / "status").read_text().splitlines() if ":" in line)
        except OSError:  # exited meanwhile
            continue
        if status.get("Uid", "").split()[:1] == [str(AGENT)] and not status.get("State", "").strip().startswith("Z"):
            alive.append(int(d.name))
    return alive


def wait_gpus_free(timeout=180):
    """Until no live process of the candidate's uid is left (nothing of it can reach the eval engines' ports) and every
    visible GPU is (nearly) empty. A SIGKILLed process stays alive (R/D) until the kernel has released its GPU and
    pinned host memory, which for the SERVE engines and trainer (~50 GB each) can take longer than one kill pass, so
    the uid is killed again until nothing is left; anything still alive after `timeout` is the Judge's problem (no
    process survives SIGKILL; repeated passes also catch processes forked during a pass)."""
    t0 = time.time()
    while not PUBLIC and agent_alive():
        if time.time() - t0 >= timeout:
            raise JudgeError(f"processes of uid {AGENT} still alive {timeout} s after SIGKILL: {agent_alive()[:8]}")
        reap_agent()
    while True:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True).stdout.split()
        if out and all(int(x) <= 2048 for x in out):
            return time.time() - t0
        if time.time() - t0 >= timeout:
            raise JudgeError(f"GPUs still hold memory {timeout} s after the candidate and the SERVE engines stopped")
        time.sleep(2)


def phase_quality(ctx: Ctx, run_info: dict):
    """Held-out HPSv3 of the run's final master: the staged bytes J6 bound to the run's last publication, on FRESH
    Judge-owned engines (the SERVE engines took the candidate's publications; engine + HPSv3 need ~74 GiB on GPU 0, so
    this runs after the candidate stopped). quality_eval.py is the measured eval that froze the base values, run as
    root on both GPUs with its temp files in the Judge's own EVAL_DIR (the candidate could have planted files in /tmp
    and /scratch). A failure of the eval is the Judge's (valid=false). Its engines are the first of the Judge's that
    no candidate process could impersonate (the SERVE engines run as the candidate's uid): their checksum of the
    staged master must equal J6's reload, else J6 fails (J6.eval_engines) and there is no quality gate."""
    m = ctx.metric
    proto = m["quality_protocol"]
    ctx.timings["quality_wait_gpus_s"] = wait_gpus_free()
    judge_dir(EVAL_DIR)
    tmp_root = EVAL_DIR / "tmp"  # only this process writes in EVAL_DIR
    shutil.rmtree(tmp_root, ignore_errors=True)
    tmp_root.mkdir(mode=0o700)
    out = ctx.out / "quality"
    cmd = [sys.executable, str(HERE / "quality_eval.py"), "--out", str(out),
           "--prompts", str(HERE.parent / proto["prompts_file"]), "--prompt-set", proto["prompt_set"],
           "--prompts-sha256", proto["prompts_sha256"], "--namespace", proto["namespace"],
           "--repeats", str(m["quality_repeats"]), "--steps", str(proto["num_inference_steps"]),
           "--model", f"final={run_info['staged_master']}",
           "--stage-dir", "", "--tmp", str(tmp_root / "reload"), "--wan-dir", str(HERE.parent)]
    try:
        with open(ctx.out / "quality_eval.log", "w") as log:
            # cwd too: the eval engines start with `python -m`, which puts the working directory on sys.path
            rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=QUALITY_TIMEOUT_S, cwd=tmp_root,
                                env=dict(os.environ, TMPDIR=str(tmp_root))).returncode
    except subprocess.TimeoutExpired as e:
        raise JudgeError(f"quality eval exceeded {QUALITY_TIMEOUT_S} s") from e
    finally:  # its engines have their own sessions: a killed eval would leave them behind
        subprocess.run(["pkill", "-KILL", "-f", "sglang.multimodal_gen.runtime.launch_server"], check=False)
        run_info["staged_master"].unlink(missing_ok=True)
        shutil.rmtree(tmp_root, ignore_errors=True)
    res_file = out / "model_final.json"
    if rc != 0 or not res_file.exists():
        raise JudgeError(f"quality eval failed (rc={rc}; {ctx.out / 'quality_eval.log'})")
    res = json.loads(res_file.read_text())
    bound = res["engine_checksum"] == run_info["j6_checksum"]
    ctx.gates["J6"]["eval_engines"] = {"ok": bound, "checksum": res["engine_checksum"],
                                       "j6_reloaded": run_info["j6_checksum"]}
    print(f"[judge] gate J6.eval_engines: {'PASS' if bound else 'FAIL'}", flush=True)
    if not bound:
        return
    per = [res["per_repeat_mean_reward"][str(r)] for r in range(int(m["quality_repeats"]))]
    q = quality_eval.metric_v2(per, m["base_quality_per_repeat"], ctx.timings.get("T"), m["quality_gate_min"],
                               m["score_scale_seconds"], m["quality_unit"])
    ctx.quality = q
    ctx.timings["quality_eval"] = {k: res.get(k) for k in ("load_wall_s", "gen_wall_s", "gen_wall_per_repeat_s",
                                                           "score_wall_s", "wall_s", "engine_checksum")}
    ctx.gate("quality_gate", {"ok": q["quality_gate_passed"], "quality_score": q["quality_score"],
                              "quality_gate_min": q["quality_gate_min"], "unit": q["quality_unit"],
                              "final_quality_per_repeat": q["final_quality_per_repeat"],
                              "base_quality_per_repeat": q["base_quality_per_repeat"]})


# ------------------------------------------------------------------------------------------------ main


def check_private(private: Path):
    """The private root (read-only mount; host copy go-rwx) is never modified: whatever the candidate must read is
    copied out when its phase starts (readable_copy / STAGE). The candidate (uid agent) must not be able to open it."""
    if PUBLIC:
        return
    st = private.stat()
    if st.st_mode & 0o007 or (st.st_uid == AGENT) or (st.st_gid == AGENT and st.st_mode & 0o070):
        raise RuntimeError(f"private root {private} is accessible to the candidate uid {AGENT}")



def all_pass(gates: dict) -> bool:
    def ok(v):
        if isinstance(v, dict):
            if "ok" in v:
                return bool(v["ok"]) and all(ok(x) for k, x in v.items() if isinstance(x, dict))
            return all(ok(x) for x in v.values() if isinstance(x, dict))
        return True
    return all(ok(v) for v in gates.values())


def failed_gates(gates: dict, prefix: str = "") -> list[str]:
    out = []
    for k, v in gates.items():
        if not isinstance(v, dict):
            continue
        name = f"{prefix}{k}"
        if "ok" in v and not v["ok"]:
            out.append(name + (f" (rc={v['rc']})" if isinstance(v.get("rc"), int) and v["rc"] else ""))
        out += failed_gates(v, name + ".")
    return out


def final_score(ctx: Ctx) -> float:
    """metric v2: 3600 / median(T) when every check, quality_gate included, passed; else 0. Without a QUALITY phase
    (partial --phases, the public suites) nothing is scored."""
    if ctx.quality is None or not all_pass(ctx.gates):
        return 0.0
    if ctx.quality["score"] is None:  # every check passed, so each replay has a T > 0
        raise JudgeError(ctx.quality["summary"])
    return ctx.quality["score"]


def finish(result: dict, ctx: Ctx, phase: str | None):
    """metrics, summary and details of the structured result. The agent sees it, so it carries gate names, statuses
    and numbers only: gate records can hold candidate output."""
    correct = bool(ctx.gates) and not result.get("error") and \
        all_pass({k: v for k, v in ctx.gates.items() if k != "quality_gate"})
    result["metrics"] = {**metrics_of(ctx, correct), "judge_wall_s": result.get("judge_wall_s")}
    result["summary"] = agent_summary(result, ctx.gates, phase)
    if result["valid"] and ctx.quality is not None and not result.get("error"):
        result["summary"] = f"{ctx.quality['summary']}; {result['summary']}"
    result["details"] = [{"name": k, "status": "PASSED" if all_pass({k: v}) else "FAILED"} for k, v in ctx.gates.items()]
    for det in result["details"]:
        if det["name"] == "quality_gate":
            det["message"] = json.dumps(ctx.gates["quality_gate"])


def metrics_of(ctx: Ctx, correct: bool) -> dict:
    """The structured result's metrics: the metric-v2 fields (quality ones None when QUALITY did not run), T and the
    Judge's wall. correctness_gate_passed excludes quality_gate."""
    m = ctx.metric or {}
    q = {k: v for k, v in (ctx.quality or {}).items() if k not in ("score", "summary")}
    Ts = ctx.timings.get("T") or []
    out = {"score_formula": m.get("formula"), "correctness_gate_passed": correct,
           "quality_gate_passed": bool((ctx.gates.get("quality_gate") or {}).get("ok")),
           "quality_gate_min": m.get("quality_gate_min"), "quality_repeats": m.get("quality_repeats"),
           "final_quality_per_repeat": None, "final_quality": None,
           "base_quality_per_repeat": m.get("base_quality_per_repeat"), "base_quality": m.get("base_quality"),
           "quality_score": None, "quality_unit": m.get("quality_unit"),
           "T_seconds": quality_eval.median_T(Ts), "T_per_replay": Ts}
    out.update(q)
    out.update(T_log_per_replay=ctx.timings.get("T_log"), T_done_per_replay=ctx.timings.get("T_done"),
               quality_eval=ctx.timings.get("quality_eval"),
               phase_seconds={k[6:-2]: v for k, v in ctx.timings.items() if k.startswith("phase_")})
    return out


def agent_summary(result: dict, gates: dict, phase: str | None) -> str:
    """What the submitter sees: gate names, pass/fail, the aborting phase and an exit code; never candidate text
    (an error event or stderr could carry hidden prompts or dump contents)."""
    fails = failed_gates(gates)
    s = f"gates: {len(gates)} checked, failed: {', '.join(fails) if fails else 'none'}"
    err = result.get("error") or ""
    if err:
        m = re.search(r"candidate exited rc=(-?\d+) waiting for (\w+)", err)
        why = (f"candidate exited rc={m.group(1)} before '{m.group(2)}'" if m else
               "timeout" if err.startswith(("TimeoutError", "TimeoutExpired")) else
               "candidate error" if result.get("valid") else "Judge infrastructure error")
        s += f"; aborted in phase {phase}: {why}"
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", type=Path, default=Path("/app/workdir/miles_lab/repo"))
    ap.add_argument("--private", type=Path, default=Path("/judge_private"))
    ap.add_argument("--out", type=Path, default=Path("/logs/verifier"))
    ap.add_argument("--phases", default="J0,J2,RUN,SERVE,QUALITY")
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--public", action="store_true", help="run as the agent on public data (run_public.py)")
    ap.add_argument("--max-replays", type=int, default=None)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    global PUBLIC
    PUBLIC = a.public
    for p in ("/scratch/judge", "/scratch/tmp", str(INPUTS)):
        Path(p).mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(p, 0o1777 if p != str(INPUTS) else 0o755)
        except PermissionError:  # public mode: /scratch/tmp is created by the launcher (host uid), already 1777
            if not PUBLIC:
                raise
    ctx = Ctx(a)
    check_private(a.private)
    ctx.home.mkdir(parents=True, exist_ok=True)
    os.chmod(ctx.home, 0o777)
    for p, mode in ((STAGE, 0o711), (MASTER_DIR, 0o1777), (EVAL_DIR, 0o755), (RELOAD_DIR, 0o755)):
        p.mkdir(parents=True, exist_ok=True)
        os.chmod(p, mode)
    result = {"valid": True, "score": 0.0}
    phase = [None]
    t_all = time.time()
    try:
        if a.plan_only:
            need = ["qualification.json", "contract.json", "run_prompts.jsonl", "score_set/videos.npz",
                    "score_set/hpsv3_oracle.json", "probe/dump/meta.json", "probe/oracle/train_oracle_probe.json"]
            need += [f"dumps/{n}/meta.json" for n in ctx.qual["replay_dumps"]]
            need += [f"oracles/{n}/{f}" for n in ctx.qual["replay_dumps"]
                     for f in ("hpsv3_oracle.json", "train_oracle_pi_old.json", "train_oracle_traj.pt")]
            missing = [n for n in need if not (a.private / n).exists()]
            if ctx.metric is None:
                missing.append(str(METRIC_FILE))
            elif not (HERE.parent / ctx.metric["quality_protocol"]["prompts_file"]).is_file():
                missing.append(str(HERE.parent / ctx.metric["quality_protocol"]["prompts_file"]))
            shm = shutil.disk_usage("/dev/shm")
            result.update(plan_only=True, missing=missing, submission_ok=(a.submission / "task_entry.py").exists(),
                          shm_total_gb=shm.total / 2**30, shm_ok=shm.total >= 96 * 2**30)
            return
        phases = a.phases.split(",")
        if "QUALITY" in phases and not PUBLIC:
            if ctx.metric is None:
                raise JudgeError(f"{METRIC_FILE} missing: the image was not built by build_images.sh")
            if ctx.metric["quality_protocol"]["eval_version"] != quality_eval.EVAL_VERSION:
                raise JudgeError(f"metric.json froze eval {ctx.metric['quality_protocol']['eval_version']}, "
                                 f"quality_eval.py is {quality_eval.EVAL_VERSION}")

        def timed(tag, fn, *args):
            phase[0] = tag
            t0 = time.time()
            try:
                return fn(*args)
            finally:
                ctx.timings[f"phase_{tag}_s"] = time.time() - t0

        eng = timed("J0", phase_j0, ctx) if "J0" in phases else CLEAN_ENGINE
        if "J2" in phases:
            timed("J2", phase_j2, ctx)
        run_info = timed("RUN", phase_run, ctx, eng) if "RUN" in phases else None
        if "SERVE" in phases:
            timed("SERVE", phase_serve, ctx, run_info)
        # QUALITY needs every correctness check passed (J6 included: it staged the master QUALITY evaluates)
        if "QUALITY" in phases and ctx.gates and all_pass(ctx.gates) and (run_info or {}).get("staged_master"):
            timed("QUALITY", phase_quality, ctx, run_info)
        result["score"] = final_score(ctx)
        result["gates_pass"] = all_pass(ctx.gates)
    except (RuntimeError, TimeoutError, subprocess.TimeoutExpired) as e:  # candidate failure
        result.update(score=0.0, error=repr(e), tb=traceback.format_exc()[-3000:])
    except Exception as e:  # noqa: BLE001 — Judge/infra failure
        result.update(valid=False, score=None, error=repr(e), tb=traceback.format_exc()[-3000:])
    finally:
        result.update(gates=ctx.gates, timings=ctx.timings, judge_wall_s=time.time() - t_all)
        finish(result, ctx, phase[0])
        (a.out / "wan_judge.json").write_text(json.dumps(result, indent=1, default=str))
        print(">>>>> Start Structured Result")
        print(json.dumps({k: result[k] for k in ("valid", "score", "summary", "metrics", "details") if k in result},
                         default=str))
        print(">>>>> End Structured Result")


if __name__ == "__main__":
    main()
