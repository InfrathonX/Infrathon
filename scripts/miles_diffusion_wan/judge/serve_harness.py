"""Fixed-replay harness (Judge prototype for S5). Runs as ROOT inside the pinned image with 2 GPUs.

1. Installs the audit hook .pth into site-packages (container-local), launches two PRISTINE engines (clean sglang,
   no overlay) as uid `agent` (setpriv) with the text encoder offloaded — the Judge-owned publication targets.
2. Writes the serve contract (engine URLs), creates control/event FIFOs, starts the candidate
   ``task_entry.py serve`` as uid `agent` in a fresh copy of the submission.
3. probe (1 step, theta_1 export) → for each replay: make the dump readable, send `replay`, T = (last engine
   `publication_complete` time) − (go time), then `dump_master` (untimed), Judge engines reload base, `reset`.
Writes /out/serve_results.json and keeps pair logs / hook logs under /out.
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

import requests

sys.path.insert(0, "/work/wan/spikes")
from s1_engine_rollout import wait_healthy  # noqa: E402

AGENT = 1001
MODEL = "/models/Wan2.2-TI2V-5B-Diffusers"


def expected_names():
    import struct

    import numpy as np

    out = {}
    for f in sorted((Path(MODEL) / "transformer").glob("*.safetensors")):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            h = json.loads(fh.read(n))
        for k, v in h.items():
            if k != "__metadata__":
                out[k] = int(np.prod(v["shape"])) if v["shape"] else 1
    return out


def as_agent(cmd, env, log, cwd=None):
    pre = ["setpriv", "--reuid", str(AGENT), "--regid", str(AGENT), "--clear-groups"]
    return subprocess.Popen(pre + cmd, env=env, stdout=open(log, "w"), stderr=subprocess.STDOUT, cwd=cwd,
                            start_new_session=True)


def hook_records(audit: Path, kind: str):
    recs = []
    for f in audit.glob("audit-*.jsonl"):
        for line in f.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("kind") == kind:
                    recs.append(r)
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", default="/work/miles_diffusion")
    ap.add_argument("--probe-dump", required=True)
    ap.add_argument("--replay-dumps", required=True, help="comma list; repeated entries = repeated reps")
    ap.add_argument("--out", default="/out")
    a = ap.parse_args()
    out = Path(a.out)
    audit = out / "judge_audit"
    audit.mkdir(parents=True, exist_ok=True)
    os.chmod(audit, 0o777)
    (audit / "expected.json").write_text(json.dumps(expected_names()))
    # hook for every process in this container (the Judge image ships it in site-packages)
    sp = Path(site.getsitepackages()[0])
    shutil.copy2("/work/wan/audit/wan_audit_hook.py", sp / "wan_audit_hook.py")
    (sp / "wan_audit_hook.pth").write_text("import wan_audit_hook\n")
    home = Path("/tmp/agenthome")
    home.mkdir(exist_ok=True)
    os.chmod(home, 0o777)
    base_env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTORCH_CUDA_ALLOC_CONF")}
    base_env.update(HOME=str(home), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", WANDB_MODE="disabled")

    # 1. Judge-owned engines
    ports = [30500, 30510]
    engines = []
    for g, p in enumerate(ports):
        env = dict(base_env, CUDA_VISIBLE_DEVICES=str(g), WAN_AUDIT_DIR=str(audit),
                   WAN_AUDIT_EXPECTED=str(audit / "expected.json"), WAN_AUDIT_CHECKSUM="1")
        cmd = [sys.executable, "-m", "sglang.multimodal_gen.runtime.launch_server", "--model-path", MODEL,
               "--port", str(p), "--host", "127.0.0.1", "--num-gpus", "1", "--warmup-mode", "off",
               "--trust-remote-code", "--master-port", str(p + 1000), "--scheduler-port", str(p + 2000),
               "--text-encoder-cpu-offload", "true", "--vae-cpu-offload", "true", "--dit-cpu-offload", "false"]
        engines.append(as_agent(cmd, env, out / f"judge_engine{g}.log"))
    res = {"engine_cold_s": [wait_healthy(p, e) for p, e in zip(ports, engines)]}

    # 2. candidate serve process
    work = Path("/tmp/candidate")
    if work.exists():
        shutil.rmtree(work)
    shutil.copytree(a.submission, work, symlinks=True)
    subprocess.run(["chown", "-R", f"{AGENT}:{AGENT}", str(work)], check=True)
    cand_out = out / "cand"
    cand_out.mkdir(exist_ok=True)
    scratch = Path("/tmp/cand_scratch")
    scratch.mkdir(exist_ok=True)
    for d in (cand_out, scratch):
        subprocess.run(["chown", "-R", f"{AGENT}:{AGENT}", str(d)], check=True)
    contract = json.load(open("/work/wan/judge/contract_template.json"))
    contract["scratch"] = str(scratch)
    contract["replay"]["engine_urls"] = [f"http://127.0.0.1:{p}" for p in ports]
    cpath = out / "contract.json"
    cpath.write_text(json.dumps(contract, indent=1))
    ctl, ev = out / "ctl.fifo", out / "events.fifo"
    for f in (ctl, ev):
        if f.exists():
            f.unlink()
        os.mkfifo(f)
        os.chmod(f, 0o666)
    cand_env = dict(base_env)
    t_start = time.time()
    cand = as_agent([sys.executable, "task_entry.py", "serve", "--contract", str(cpath), "--ctl", str(ctl),
                     "--events", str(ev), "--out", str(cand_out)], cand_env, out / "cand_serve.log", cwd=str(work))
    # Non-blocking FIFO handling: never block on an open/read the candidate may never complete (it can die before
    # opening either FIFO); liveness is checked via the process, and every wait has a timeout.
    ev_fd = os.open(ev, os.O_RDONLY | os.O_NONBLOCK)
    pending = {"buf": b"", "ctl": None}

    def wait_event(kind, timeout=1800):
        t0 = time.time()
        while time.time() - t0 < timeout:
            while b"\n" in pending["buf"]:
                line, pending["buf"] = pending["buf"].split(b"\n", 1)
                if not line.strip():
                    continue
                e = json.loads(line)
                if e.get("event") == "error":
                    raise RuntimeError(e)
                if e.get("event") == kind:
                    return e
            try:
                chunk = os.read(ev_fd, 65536)
            except BlockingIOError:
                chunk = b""
            if chunk:
                pending["buf"] += chunk
                continue
            if cand.poll() is not None:
                raise RuntimeError(f"candidate exited rc={cand.returncode} while waiting for {kind}")
            time.sleep(0.2)
        raise TimeoutError(kind)

    def send(obj):
        if pending["ctl"] is None:
            t0 = time.time()
            while True:
                try:
                    fd = os.open(ctl, os.O_WRONLY | os.O_NONBLOCK)
                    os.set_blocking(fd, True)
                    pending["ctl"] = os.fdopen(fd, "w", buffering=1)
                    break
                except OSError:
                    if cand.poll() is not None or time.time() - t0 > 300:
                        raise RuntimeError("candidate never opened the control FIFO")
                    time.sleep(0.2)
        pending["ctl"].write(json.dumps(obj) + "\n")
        pending["ctl"].flush()

    try:
        wait_event("ready", timeout=1500)
    except Exception as e:  # noqa: BLE001
        res["error"] = f"serve start failed: {e!r}"
        for p in [cand] + engines:
            try:
                os.killpg(p.pid, 9)
            except Exception:
                pass
        (out / "serve_results.json").write_text(json.dumps(res, indent=1))
        subprocess.run(["chown", "-R", "1000:1000", str(out)])
        raise
    res["cand_cold_s"] = time.time() - t_start

    # 3a. probe
    t0 = time.time()
    send({"cmd": "probe", "id": "probe", "dump": a.probe_dump, "prompts": 1,
          "theta1_path": str(cand_out / "probe_theta1.safetensors")})
    wait_event("done")
    res["probe_s"] = time.time() - t0
    send({"cmd": "reset", "id": "probe_reset"})  # the probe took an optimizer step: back to theta_0
    wait_event("done")

    # 3b. replays
    reps = []
    for i, dump in enumerate(a.replay_dumps.split(",")):
        n_before = len(hook_records(audit, "publication_complete"))
        t_go = time.time()
        send({"cmd": "replay", "id": f"r{i}", "dump": dump})
        wait_event("log_complete")
        t_log = time.time()
        wait_event("done")
        t_done = time.time()
        deadline = time.time() + 120
        while True:
            pubs = hook_records(audit, "publication_complete")
            if len(pubs) >= n_before + len(ports) or time.time() > deadline:
                break
            time.sleep(0.5)
        new = sorted(pubs, key=lambda r: r["t_complete"])[n_before:]
        rep = {"dump": dump, "t_log_s": t_log - t_go, "t_done_s": t_done - t_go, "n_pub": len(new),
               "T_s": (max(r["t_complete"] for r in new) - t_go) if len(new) == len(ports) else None,
               "pub_first_bucket_s": [r["t_first_bucket"] - t_go for r in new]}
        t0 = time.time()
        send({"cmd": "dump_master", "id": f"m{i}", "path": str(cand_out / f"master_r{i}.safetensors")})
        wait_event("done")
        rep["dump_master_s"] = time.time() - t0
        # R6' equality: reload engine 0 from the candidate's fp32 master (engine casts to bf16 like a publication)
        # and compare its checksum with the one the hook recorded right after the timed publication.
        t0 = time.time()
        tmp = Path(f"/tmp/master_model_r{i}")
        if tmp.exists():
            shutil.rmtree(tmp)
        (tmp / "transformer").mkdir(parents=True)
        for item in Path(MODEL).iterdir():
            if item.name != "transformer":
                (tmp / item.name).symlink_to(item)
        shutil.copy2(Path(MODEL) / "transformer" / "config.json", tmp / "transformer" / "config.json")
        (tmp / "transformer" / "diffusion_pytorch_model.safetensors").symlink_to(cand_out / f"master_r{i}.safetensors")
        subprocess.run(["chmod", "-R", "a+rX", str(tmp)])
        rr = requests.post(f"http://127.0.0.1:{ports[0]}/update_weights_from_disk",
                           json={"model_path": str(tmp), "target_modules": ["transformer"]}, timeout=900)
        ck = requests.post(f"http://127.0.0.1:{ports[0]}/get_weights_checksum",
                           json={"module_names": ["transformer"]}, timeout=900).json()
        pub0 = [r for r in new if r.get("checksum")]
        rep["r6_reload_ok"] = rr.status_code == 200
        rep["r6_checksum_master"] = ck.get("transformer")
        rep["r6_checksum_published"] = [r.get("checksum") for r in new]
        rep["r6_equal"] = bool(pub0) and all(r["checksum"] == ck.get("transformer") for r in pub0)
        rep["r6_s"] = time.time() - t0
        t0 = time.time()
        for p in ports:  # Judge engines back to theta_0
            requests.post(f"http://127.0.0.1:{p}/update_weights_from_disk", json={"model_path": MODEL, "target_modules": ["transformer"]}, timeout=600)
        send({"cmd": "reset", "id": f"x{i}"})
        wait_event("done")
        rep["reset_s"] = time.time() - t0
        reps.append(rep)
        print(json.dumps(rep), flush=True)
    res["replays"] = reps
    send({"cmd": "stop"})
    try:
        wait_event("stopped", timeout=60)
    except Exception:
        pass
    for p in [cand] + engines:
        try:
            os.killpg(p.pid, 9)
        except Exception:
            pass
    (out / "serve_results.json").write_text(json.dumps(res, indent=1))
    subprocess.run(["chown", "-R", "1000:1000", str(out)])
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
