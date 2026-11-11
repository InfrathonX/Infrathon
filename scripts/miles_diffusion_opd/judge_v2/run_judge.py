#!/usr/bin/env python3
"""Judge for the SD3.5-M DiffusionOPD task v2. Runs as root in the Judge container; candidate code runs as the
unprivileged `agent` user and is never imported here. Two candidate launches share the leased GPU pair:

  P  one process: round 0 = the probe fixture (hidden rows, nonzero fixture LoRA as student init, random route labels),
     rounds 1..18 = three timed replay segments; every generate request goes to the Judge's replay server
     -> probe_check (G1-G4) on step 0, AdamW algebra on every step, replay protocol (incl. the untimed setup cap:
        replay server start -> release of round 1), per-round publication (hook)
  R  the scored-contract real on-policy run under the in-engine audit hook, with a deadline
     -> audit W1-W6, AdamW algebra on every step (theta_0 == student_init), final_lora.safetensors == the last
        step's lora_post bit for bit
     (the AdamW checks of P and R also require every record's optimizer settings to equal the contract's)
  K  per-route KL gate: the Judge's held-out teacher-student KL of the final student (quality_eval.py, v1's
     evaluator), one case seed per GPU; on each route ln(KL_base / KL_final) must reach the frozen minimum (half the
     reference's route gain)
  Q  task metric (metric v3): the final student generates one image per held-out prompt (task_quality_eval.py: the
     official DiffusionOPD evaluator's 512x512, 40-step ODE pipeline at the frozen CFG), prompts dealt over both
     GPUs; OCR accuracy on the OCR route, PickScore on the PickScore route;
     closure_r = (S_r - B_r) / (T_r - B_r) with the original model B_r and the route's teacher T_r frozen
  score = mean_r closure_r * 3600 / T   if every check passes, the KL gate passes and every closure_r >= the gate,
          else 0; T = median(segment T)
A candidate failure (a check or a quality gate) is valid=true, score=0; valid=false only for Judge/infrastructure
errors. Every candidate-written step record is read with torch.load(weights_only=True).
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import pwd
import random
import shutil
import statistics
import string
import subprocess
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
# No torch in this process before the last candidate launch: candidate() forks with preexec_fn, which is unsafe once
# a library has started threads (importing torch starts dozens). adamw_check / quality_eval are imported where used.
from replay_server import segment_times, setup_seconds, warmup_seconds  # noqa: E402

PY = sys.executable
AGENT = pwd.getpwnam("agent")
WORK = Path(os.environ.get("OPD_JUDGE_WORK", "/scratch/judge"))


def as_agent():
    os.setgid(AGENT.pw_gid)
    os.setgroups([])
    os.setuid(AGENT.pw_uid)


def agent_env(extra=None):
    env = {"PATH": "/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
           "HOME": "/home/agent", "USER": "agent", "LANG": "C.UTF-8", "TMPDIR": "/tmp",
           "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0,1"),
           "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
           "WANDB_MODE": "disabled", "RAY_USAGE_STATS_ENABLED": "0", "PYTHONUNBUFFERED": "1",
           "PYTHONDONTWRITEBYTECODE": "1"}
    env.update(extra or {})
    return env


def reap_agent():
    subprocess.run(["pkill", "-KILL", "-u", "agent"], check=False)
    time.sleep(3)


def write_json(p: Path, obj) -> None:
    p.write_text(json.dumps(obj, indent=1, allow_nan=False))


def run_logged(cmd, log: Path, *, cwd=None, env=None, timeout=None, preexec=None):
    t0 = time.monotonic()
    with open(log, "w") as f:
        try:
            p = subprocess.run(cmd, cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=timeout,
                               preexec_fn=preexec)
            code = p.returncode
        except subprocess.TimeoutExpired:
            code = "timeout"
    return code, time.monotonic() - t0


def run_tool(args, log: Path, out: Path):
    code, _ = run_logged([PY, *[str(a) for a in args]], log, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
    return json.loads(out.read_text()) if out.is_file() else None


def run_tools_parallel(jobs):
    """run_tool for every (args, log, out, extra_env) at the same time (one per GPU); their outputs in job order."""
    running = []
    for args, log, out, extra in jobs:
        f = open(log, "w")
        p = subprocess.Popen([PY, *[str(a) for a in args]], stdout=f, stderr=subprocess.STDOUT,
                             env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1", **extra))
        running.append((p, f, out))
    for p, f, _ in running:
        p.wait()
        f.close()
    return [json.loads(out.read_text()) if out.is_file() else None for _, _, out in running]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def quality_repeats(student: Path, prompts: Path, seeds: list[int], samples: int, Q: dict, out_dir: Path) -> list[dict]:
    """The Judge's held-out KL of `student` for each case seed: the seeds are dealt round-robin over the visible GPUs,
    one quality_eval.py process per GPU (it loads the encoders and DiT copies once and runs its seeds one after
    another; a seed's result does not depend on the process or GPU it ran on). {case_seed, per_route, kl_pooled} per
    seed in `seeds` order; an evaluator that does not return every seed is a Judge error."""
    devices = [x.strip() for x in os.environ.get("CUDA_VISIBLE_DEVICES", "0,1").split(",") if x.strip()] or ["0"]
    groups = [seeds[k::len(devices)] for k in range(min(len(devices), len(seeds)))]
    jobs = []
    for i, group in enumerate(groups):
        out = out_dir / f"quality_{i}.json"
        jobs.append(([HERE / "quality_eval.py", "--prompts", prompts, "--student", student, "--samples", samples,
                      "--case-seeds", *group, "--out", out, "--model", Q["model"], "--teachers", Q["teachers_dir"]],
                     out_dir / f"quality_{i}.log", out, {"CUDA_VISIBLE_DEVICES": devices[i]}))
    got = {r["case_seed"]: r for res in run_tools_parallel(jobs) for r in (res or {}).get("repeats", [])}
    if sorted(got) != sorted(seeds):
        raise RuntimeError(f"quality evaluator returned case seeds {sorted(got)}, expected {sorted(seeds)}")
    return [got[c] for c in seeds]


def positive_finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and x > 0


def route_gain(repeats: list[dict], base: list[dict], route: str):
    """Mean over the case seeds of ln(KL_base / KL_final) on one route (reported, not scored); None when a KL of the
    final student is not positive and finite."""
    pairs = [(b["per_route"][route]["kl"], r["per_route"][route]["kl"]) for r, b in zip(repeats, base)]
    if not all(positive_finite(x) for pair in pairs for x in pair):
        return None
    return sum(math.log(b / f) for b, f in pairs) / len(pairs)


def paired_with_base(repeats: list[dict], base: list[dict]) -> None:
    """Repeat c must be measured like the frozen base's repeat c: same case seed, same routes, same pair counts."""
    for r, b in zip(repeats, base, strict=True):
        counts = {k: v["n"] for k, v in r["per_route"].items()}
        if r["case_seed"] != b["case_seed"] or counts != {k: v["n"] for k, v in b["per_route"].items()}:
            raise RuntimeError(f"case seed {r['case_seed']}: routes / pair counts {counts} differ from the frozen "
                               f"base's (case seed {b['case_seed']})")


def kl_route_gate(repeats: list[dict], base: list[dict], min_gain: dict) -> dict:
    """Per-route KL gate: on every route the mean over the case seeds of ln(KL_base / KL_final) must reach
    min_gain[route] (a non-finite final KL fails)."""
    gains = {r: route_gain(repeats, base, r) for r in min_gain}
    ok = all(g is not None and g >= min_gain[r] for r, g in gains.items())
    return {"passed": ok, "route_gain": gains, "min_route_gain": min_gain}


def task_metric(student: Path, prompts: Path, TQ: dict, Q: dict, out_dir: Path) -> dict:
    """The Judge's task metric of `student`: the held-out prompts are dealt over the visible GPUs by index (one
    task_quality_eval.py process per GPU), the shard outputs merged; {route: mean} plus the merged result. A shard
    that returns nothing is a Judge error."""
    from task_quality_eval import merge  # stdlib only

    devices = [x.strip() for x in os.environ.get("CUDA_VISIBLE_DEVICES", "0,1").split(",") if x.strip()] or ["0"]
    jobs = []
    for i, dev in enumerate(devices):
        out = out_dir / f"task_{i}.json"
        jobs.append(([HERE / "task_quality_eval.py", "eval", "--model", Q["model"],
                      "--pickscore-model", Q["scorers"]["pickscore_model"],
                      "--pickscore-processor", Q["scorers"]["pickscore_processor"],
                      "--prompts", prompts, "--lora", student, "--cfg", TQ["cfg"], "--case-seed", TQ["case_seed"],
                      "--steps", TQ["steps"], "--res", TQ["res"], "--batch", TQ["batch"],
                      "--ocr-workers", TQ["ocr_workers_per_gpu"], "--shard", i, "--num-shards", len(devices),
                      "--out", out], out_dir / f"task_{i}.log", out, {"CUDA_VISIBLE_DEVICES": dev}))
    parts = run_tools_parallel(jobs)
    if any(p is None for p in parts):
        raise RuntimeError(f"task metric shard(s) {[i for i, p in enumerate(parts) if p is None]} returned nothing")
    merged = merge([str(out) for _, _, out, _ in jobs])
    res = merged["results"][0]
    write_json(out_dir / "task_metric.json", {k: v for k, v in merged.items() if k != "results"} |
               {"result": {k: v for k, v in res.items() if k != "items"}})
    counts = {r: v["n"] for r, v in res["per_route"].items()}
    if counts != TQ["route_counts"]:
        raise RuntimeError(f"task metric route counts {counts} differ from the frozen protocol's {TQ['route_counts']}")
    return {r: v["mean"] for r, v in res["per_route"].items()}, res


def final_lora_verdict(final: Path, dump: Path) -> dict:
    """The LoRA the quality evaluation reads is the student after the run's last optimizer step, bit for bit (the
    last record's lora_post); with quality in the score, training on after the last step must not pay. It reads
    candidate-written files: anything that goes wrong reading them fails the check (it is not a Judge error)."""
    from adamw_check import final_lora_issues  # torch: only after the last candidate launch (see the imports)

    try:
        issues = final_lora_issues(str(final), str(dump))
    except Exception as e:  # noqa: BLE001
        issues = [f"the final LoRA or the last step record could not be checked ({type(e).__name__}: {e})"[:300]]
    return {"passed": not issues, "issues": issues}


# ------------------------------------------------------------------------------------------ static
FORBIDDEN_SUBSTRINGS = ("--rollout-patch-group", "rollout_patch_group")


def engine_mutations(text: str) -> set[str]:
    """Assignment / setattr targets that reach into sglang (multimodal_gen) objects."""
    out: set[str] = set()
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return out
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AugAssign)):
            for tgt in (node.targets if isinstance(node, ast.Assign) else [node.target]):
                src = ast.unparse(tgt)
                if "multimodal_gen" in src or src.startswith("sglang."):
                    out.add(src)
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "setattr" and node.args:
            first = ast.unparse(node.args[0])
            if "sglang" in first or "multimodal_gen" in first:
                out.add("setattr(" + first + ")")
    return out


def static_checks(app: Path, pristine: Path) -> dict:
    """Submission hygiene. Engine-patching rules apply to what the submission added or changed relative to the pristine
    miles_diffusion@29fde6ee tree (upstream itself defines rollout patch groups and its own engine monkey patches)."""
    res = {"passed": True, "problems": [], "changed_py_files": 0}
    if not (app / "task_entry.py").is_file() or not (app / "repo/train_diffusion.py").is_file():
        res["problems"].append("task_entry.py or repo/train_diffusion.py missing")
    for p in sorted(app.rglob("*")):
        rel = p.relative_to(app)
        if p.is_symlink():
            res["problems"].append(f"symlink {rel}")
            continue
        if not p.is_file():
            continue
        if p.suffix in (".safetensors", ".pt", ".bin", ".ckpt", ".msgpack"):
            res["problems"].append(f"weights-like file {rel}")
            continue
        if p.suffix != ".py":
            continue
        if rel.parts[0] == "repo":
            base = pristine / Path(*rel.parts[1:])
            if base.is_file() and base.read_bytes() == p.read_bytes():
                continue
        res["changed_py_files"] += 1
        text = p.read_text(errors="ignore")
        base_text = ""
        if rel.parts[0] == "repo":
            base = pristine / Path(*rel.parts[1:])
            base_text = base.read_text(errors="ignore") if base.is_file() else ""
        for s_ in FORBIDDEN_SUBSTRINGS:
            if text.count(s_) > base_text.count(s_):
                res["problems"].append(f"new use of {s_} in {rel}")
        new_targets = engine_mutations(text) - engine_mutations(base_text)
        if new_targets:
            res["problems"].append(f"new assignments into sglang internals in {rel}: {sorted(new_targets)[:5]}")
    res["passed"] = not res["problems"]
    return res


# ------------------------------------------------------------------------------------------ launches
def relabel(rows: list[dict], mapping: dict[str, str]) -> list[dict]:
    out = []
    for r in rows:
        r = json.loads(json.dumps(r))
        r["metadata"]["data_source"] = mapping[r["metadata"]["data_source"]]
        out.append(r)
    return out


def candidate(app: Path, name: str, contract: dict, timeout: float, extra_env=None):
    d = WORK / name
    shutil.copytree(app, d / "app", symlinks=True)
    for sub in ("out", "scratch", "tmp", "home"):
        (d / sub).mkdir()
    contract = dict(contract, scratch=str(d / "scratch"))
    write_json(d / "contract.json", contract)
    subprocess.run(["chown", "-R", "agent:agent", str(d)], check=True)
    # HOME and TMPDIR on /scratch: the container rootfs sits on the host root disk (caches of Ray, Triton, compile)
    env = agent_env(dict(extra_env or {}, TMPDIR=str(d / "tmp"), HOME=str(d / "home")))
    try:
        code, secs = run_logged([PY, "task_entry.py", "run", "--contract", str(d / "contract.json"), "--out", str(d / "out/run")],
                                d / "task_entry.log", cwd=d / "app", env=env,
                                timeout=timeout, preexec=as_agent)
    finally:
        reap_agent()
    return d, code, secs


def start_replay_server(workload: Path, log: Path, port: int):
    p = subprocess.Popen([PY, str(HERE / "replay_server.py"), "--workload", str(workload), "--port", str(port), "--log", str(log)],
                         stdout=open(log.with_suffix(".out"), "w"), stderr=subprocess.STDOUT,
                         env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
    for _ in range(120):
        try:
            import urllib.request

            urllib.request.urlopen(f"http://127.0.0.1:{port}/replay/status", timeout=1).read()
            return p
        except Exception:
            time.sleep(0.5)
    p.kill()
    raise RuntimeError("replay server did not come up")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--app", type=Path, default=Path("/app/workdir/miles_lab"))
    ap.add_argument("--private", type=Path, default=Path("/judge_private"))
    ap.add_argument("--qualification", type=Path, default=HERE / "qualification.json")
    ap.add_argument("--contract", type=Path, default=HERE / "contract.json")
    ap.add_argument("--log-dir", type=Path, default=Path("/logs/verifier"))
    ap.add_argument("--pristine", type=Path, default=Path(os.environ.get("OPD_PRISTINE_MD", "/opt/opd_judge/md_pristine")))
    a = ap.parse_args()
    Q = json.loads(a.qualification.read_text())
    TH = Q["thresholds"]
    M = Q["metric"]
    PRIV = a.private
    a.log_dir.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(WORK, ignore_errors=True)
    WORK.mkdir(parents=True)
    os.chmod(WORK, 0o755)
    checks, metrics = {}, {"score_formula": M["formula"]}
    result = {"valid": True, "score": 0.0}
    summary = None
    rng = random.Random()
    try:
        checks["static"] = static_checks(a.app, a.pristine)
        if not checks["static"]["passed"]:
            raise StopIteration
        base = json.loads(a.contract.read_text())
        labels = ["".join(rng.choice(string.ascii_lowercase) for _ in range(7)) for _ in range(2)]
        mapping = {"ocr": labels[0], "pickscore": labels[1]}
        teachers = {labels[0]: base["teachers"]["ocr"], labels[1]: base["teachers"]["pickscore"]}

        # ---------------------------------------------------------------- P: probe round + timed replay
        # The server's workload (recorded responses of every round) stays root-only: a candidate that could read
        # future rounds' responses would move teacher scoring out of the timed window. The candidate only gets the
        # rows (prompts, seeds, relabelled routes) and the probe student init.
        wl = WORK / "session_p_server"
        wl.mkdir()
        os.chmod(wl, 0o700)
        manifest = json.loads((PRIV / "session_p/manifest.json").read_text())
        rows = relabel([json.loads(line) for line in open(PRIV / "session_p/rows.jsonl")], mapping)
        for f in (PRIV / "session_p").glob("row_*"):
            os.symlink(f, wl / f.name)
        write_json(wl / "manifest.json", manifest)
        cand = WORK / "session_p_inputs"
        cand.mkdir()
        (cand / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        shutil.copy2(PRIV / "session_p/student_init.safetensors", cand / "student_init.safetensors")
        subprocess.run(["chmod", "-R", "a+rX", str(cand)], check=True)
        rlog = WORK / "replay.log"
        server = start_replay_server(wl, rlog, Q["replay_port"])
        audit_p = WORK / "audit_p"
        audit_p.mkdir()
        subprocess.run(["chown", "agent:agent", str(audit_p)], check=True)
        cp = dict(base, teachers=teachers, rows=str(cand / "rows.jsonl"), student_init=str(cand / "student_init.safetensors"),
                  rounds=len(manifest["rounds"]), generate_url=f"http://127.0.0.1:{Q['replay_port']}", probe_steps=1)
        try:
            d, code, secs = candidate(a.app, "session_p", cp, Q["timeouts"]["session_p"], {"OPD_AUDIT_DIR": str(audit_p)})
        finally:
            server.terminate()
        checks["session_p_exit"] = {"passed": code == 0, "code": code, "seconds": round(secs, 1)}
        if code != 0:
            raise StopIteration
        rev = [json.loads(line) for line in open(rlog)]
        viol = [e for e in rev if e["kind"] == "violation"]
        published = sum(1 for e in rev if e["kind"] == "published")
        setup = setup_seconds(rlog, manifest["segments"])
        checks["replay_protocol"] = {"passed": not viol and published == len(manifest["rounds"]) and setup is not None
                                     and setup <= Q["setup_cap_seconds"], "violations": viol[:5], "published": published,
                                     "warmup_seconds": warmup_seconds(rlog), "setup_seconds": setup,
                                     "setup_cap_seconds": Q["setup_cap_seconds"]}
        dump_p = d / "out/run/steps"
        fx = WORK / "probe_fixture"
        fx.mkdir()
        probe_rows = [r for r in rows if r["metadata"]["row_id"] in set(manifest["rounds"][0])]
        (fx / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in probe_rows[: len(probe_rows) // 2]))
        for r in probe_rows[: len(probe_rows) // 2]:
            for ext in ("json", "msgpack"):
                shutil.copy2(wl / f"row_{r['metadata']['row_id']:06d}.{ext}", fx)
        shutil.copy2(PRIV / "session_p/student_init.safetensors", fx / "student_lora.safetensors")
        write_json(fx / "fixture.json", {"labels": {"ocr": labels[0], "aes": labels[1]}})
        step0 = WORK / "probe_step0"
        step0.mkdir()
        shutil.copy2(dump_p / "step_00000.pt", step0 / "step_00000.pt")
        pr = run_tool([HERE / "probe_check.py", "--fixture", fx, "--dump", step0, "--out", WORK / "probe_check.json",
                       "--model", Q["model"], "--teachers", Q["teachers_dir"]], WORK / "probe_check.log", WORK / "probe_check.json")
        rep = (pr or {}).get("report", {})
        checks["probe"] = {"passed": not probe_failures(rep, TH), "failures": probe_failures(rep, TH), **rep}
        ad = run_tool([HERE / "adamw_check.py", "--dump", dump_p, "--student-init", PRIV / "session_p/student_init.safetensors",
                       "--optimizer", json.dumps(base["optimizer"]), "--out", WORK / "adamw_p.json"],
                      WORK / "adamw_p.log", WORK / "adamw_p.json")
        checks["adamw_p"] = adamw_verdict(ad, TH, expect_steps=cp["rounds"] * cp["steps_per_round"])
        checks["publication_p"] = publication_verdict(run_tool(
            [HERE / "publication_check.py", "--audit", audit_p, "--dump", dump_p, "--rounds", cp["rounds"],
             "--steps-per-round", cp["steps_per_round"], "--replay-log", rlog, "--out", WORK / "publication_p.json"],
            WORK / "publication_p.log", WORK / "publication_p.json"))
        seg = segment_times(rlog, manifest["segments"])
        metrics["segment_seconds"] = seg
        checks["framework_p"] = framework_verdict(run_tool(
            [HERE / "framework_check.py", "--audit", audit_p, "--rounds", cp["rounds"], "--replay-log", rlog,
             "--out", WORK / "framework_p.json"], WORK / "framework_p.log", WORK / "framework_p.json"))
        if not all(checks[k]["passed"] for k in ("replay_protocol", "probe", "adamw_p", "publication_p", "framework_p")):
            raise StopIteration

        # ---------------------------------------------------------------- R: real run under audit
        audit_r = WORK / "audit_r"
        audit_r.mkdir()
        subprocess.run(["chown", "agent:agent", str(audit_r)], check=True)
        rows_r = relabel([json.loads(line) for line in open(PRIV / "run/rows.jsonl")], mapping)
        rr = WORK / "run_rows.jsonl"
        rr.write_text("".join(json.dumps(r) + "\n" for r in rows_r))
        os.chmod(rr, 0o644)
        init_r = WORK / "run_student_init.safetensors"
        shutil.copy2(PRIV / "run/student_init.safetensors", init_r)
        os.chmod(init_r, 0o644)
        cr = dict(base, teachers=teachers, rows=str(rr), student_init=str(init_r))
        d, code, secs = candidate(a.app, "session_r", cr, Q["timeouts"]["session_r"], {"OPD_AUDIT_DIR": str(audit_r)})
        checks["session_r_exit"] = {"passed": code == 0, "code": code, "seconds": round(secs, 1)}
        if code != 0:
            raise StopIteration
        au = run_tool([HERE / "audit_check.py", "--audit", audit_r, "--rows", rr, "--rounds", cr["rounds"],
                       "--rows-per-round", cr["rows_per_round"], "--dump", d / "out/run/steps", "--out", WORK / "audit_r.json"],
                      WORK / "audit_r.log", WORK / "audit_r.json")
        checks["audit_r"] = {"passed": bool(au) and not au["violations"], "violations": (au or {}).get("violations", ["no report"])[:10]}
        ad = run_tool([HERE / "adamw_check.py", "--dump", d / "out/run/steps", "--student-init", PRIV / "run/student_init.safetensors",
                       "--optimizer", json.dumps(base["optimizer"]), "--out", WORK / "adamw_r.json"],
                      WORK / "adamw_r.log", WORK / "adamw_r.json")
        checks["adamw_r"] = adamw_verdict(ad, TH, expect_steps=cr["rounds"] * cr["steps_per_round"])
        checks["publication_r"] = publication_verdict(run_tool(
            [HERE / "publication_check.py", "--audit", audit_r, "--dump", d / "out/run/steps", "--rounds", cr["rounds"],
             "--steps-per-round", cr["steps_per_round"], "--out", WORK / "publication_r.json"],
            WORK / "publication_r.log", WORK / "publication_r.json"))
        final = d / "out/run/final_lora.safetensors"
        checks["final_lora"] = final_lora_verdict(final, d / "out/run/steps")
        checks["framework_r"] = framework_verdict(run_tool(
            [HERE / "framework_check.py", "--audit", audit_r, "--rounds", cr["rounds"], "--out", WORK / "framework_r.json"],
            WORK / "framework_r.log", WORK / "framework_r.json"))
        if not all(checks[k]["passed"] for k in ("audit_r", "adamw_r", "publication_r", "framework_r", "final_lora")):
            raise StopIteration

        # ---------------------------------------------------------------- K: per-route KL gate
        from task_quality_eval import metric_v3

        KG = M["kl_gate"]
        prompts = PRIV / "heldout_kl.jsonl"
        if sha256_file(prompts) != KG["prompts_sha256"]:
            raise RuntimeError("the private KL held-out prompts are not the ones the frozen base was measured on")
        qe = quality_repeats(final, prompts, KG["case_seeds"], KG["samples"], Q, WORK)
        frozen_kl = KG["base_kl_per_repeat"]
        paired_with_base(qe, frozen_kl)
        kg = kl_route_gate(qe, frozen_kl, KG["min_route_gain"])
        checks["kl_gate"] = kg
        metrics["kl_route_gain"] = kg["route_gain"]
        metrics["final_kl_per_route_per_repeat"] = [{k: v["kl"] if math.isfinite(v["kl"]) else None
                                                    for k, v in r["per_route"].items()} for r in qe]
        T = statistics.median(seg)
        TQ = M["task_quality"]
        frozen = {r: {"base": v["base"], "teacher": v["teacher"]} for r, v in TQ["routes"].items()}
        if kg["passed"]:
            # ------------------------------------------------------------ Q: task metric of the final student
            tprompts = PRIV / "heldout_task.jsonl"
            if sha256_file(tprompts) != TQ["prompts_sha256"]:
                raise RuntimeError("the private task held-out prompts are not the ones the frozen values were measured on")
            student_metric, _ = task_metric(final, tprompts, TQ, Q, WORK)
        else:  # the quality gate has already failed: skip the image generation (~20-30 minutes)
            student_metric = {r: math.nan for r in frozen}
        m = metric_v3(student_metric, frozen, kg["passed"], T, M["closure_gate_min"], M["score_scale_seconds"],
                      M["quality_unit"])
        checks["quality_gate"] = {"passed": m["quality_gate_passed"], "quality_score": m["quality_score"],
                                  "closure_per_route": m["closure_per_route"], "closure_gate_min": m["closure_gate_min"],
                                  "kl_gate_passed": m["kl_gate_passed"], "unit": m["quality_unit"]}
        metrics.update({k: v for k, v in m.items() if k not in ("score", "summary", "score_formula")}, t_seconds=T)
        if m["score"] is None:
            raise RuntimeError(f"the replay passed but gave no usable timed window: {m['summary']}")
        result["score"] = m["score"]
        summary = m["summary"]
    except StopIteration:
        pass
    except Exception:
        result["valid"] = False
        metrics["judge_error"] = traceback.format_exc()[-4000:]
    finally:
        reap_agent()
    correct = bool(checks) and all(v.get("passed") for k, v in checks.items() if k not in ("quality_gate", "kl_gate"))
    result["metrics"] = {**metrics, "correctness_gate_passed": correct,
                         "quality_gate_passed": bool(checks.get("quality_gate", {}).get("passed"))}
    result["details"] = [{"name": k, "status": "PASSED" if v.get("passed") else "FAILED",
                          "message": json.dumps(v, default=str)[:2000]} for k, v in checks.items()]
    if not result["valid"]:
        result["summary"] = "evaluator error: not scored"
    elif not correct:
        result["summary"] = "correctness checks failed: score 0"
    else:
        result["summary"] = summary or f"score {result['score']:.3f}"
    write_json(a.log_dir / "opd_private.json", {**result, "checks": checks})
    print(">>>>> Start Structured Result")
    print(json.dumps(result, allow_nan=False, default=str))
    print(">>>>> End Structured Result")


def probe_failures(rep: dict, TH: dict) -> list[str]:
    """Probe verdict: structural checks, G3 transition-mean errors, G4 gradient direction AND size (a gradient that
    drops or detaches part of the pairs keeps its direction but not its size), update moved."""
    if not rep:
        return ["no probe report"]
    fails = [k for k, ok in rep.get("checks", {}).items() if not ok]
    g3, g4 = rep.get("G3", {}), rep.get("G4", {})
    for key in ("e_T_median", "e_T_p90", "e_S_median", "e_S_p90"):
        if not g3.get(key, 9) <= TH[key]:
            fails.append(key)
    if not g4.get("cos", 0) >= TH["grad_cos"]:
        fails.append("grad_cos")
    if not g4.get("rel_l2", 9) <= TH["grad_rel_l2"]:
        fails.append("grad_rel_l2")
    if not g4.get("sign_agree_top_half", 0) >= TH["sign_agree"]:
        fails.append("sign_agree")
    if not g4.get("update_nonzero"):
        fails.append("update_nonzero")
    return fails


def framework_verdict(fc) -> dict:
    """framework_check.py: the session ran through miles_diffusion's driver, RolloutManager rollout path and FSDP
    training actor (engine requests and weight updates inside those calls)."""
    if not fc:
        return {"passed": False, "error": "no framework report"}
    return {"passed": not fc["violations"], **fc}


def publication_verdict(pc) -> dict:
    """publication_check.py: engines hold exactly the dumped student after every round (fp16-rounded fingerprint
    match), and in replay sessions `published` comes only after every engine applied it."""
    if not pc:
        return {"passed": False, "error": "no publication report"}
    return {"passed": not pc["violations"] and pc.get("fingerprint_kind") == "fingerprint16",
            **{k: (v[:10] if k == "violations" else v) for k, v in pc.items()}}


def adamw_verdict(ad, TH, expect_steps: int) -> dict:
    """adamw_check.py, run with --student-init and --optimizer (the contract's): a report without either check fails."""
    if not ad:
        return {"passed": False, "error": "no AdamW report"}
    ok = (not ad["violations"] and ad["steps"] == expect_steps and ad.get("init_matches_student_init", False)
          and ad.get("optimizer_matches_contract", False)
          and (ad.get("max_v_rel") or 0) <= TH["adam_v_rel"] and (ad.get("max_clip_rel") or 0) <= TH["adam_clip_rel"]
          and (ad.get("max_theta_rel_to_update") or 0) <= TH["adam_theta_rel"])
    return {"passed": ok, **{k: v for k, v in ad.items() if k != "per_step"}}


if __name__ == "__main__":
    main()
