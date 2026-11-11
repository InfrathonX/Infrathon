#!/usr/bin/env python3
"""Re-run the final Judge on an existing run's final archive (after a fix to the task environment, or to repeat it).

  rescore.py --config launch/miles-diffusion-wan/<model>.env --run-id mdwan-<model>-max-6h-02 --suffix rescore1
             [--judge-gpus C,D | --shared-gpus]   (the run's own --judge-gpus / --shared-gpus when it was given one)
  rescore.py --config launch/miles-diffusion-wan/<model>.env --archive ref.tar.gz --label qual-ref-02   (any archive of
             repo/ + notes.md, e.g. the reference: this is the qualification path, identical to a run's final Judge)

Same Judge command, images, GPU pair (under the configured lease holder, else this process takes the pair's flocks),
host locations (scripts/miles_common/host_config.py), pinned core count (MDWAN_CPU_CORES), CPU set and data roots as the
run's own final evaluation: the Judge pair of --judge-gpus or the config's JUDGE_GPU_INDICES with JUDGE_CPUSET (a
separate Judge pair), else GPU_INDICES with CPUSET (--shared-gpus, JUDGE_GPU_INDICES=shared or unset). Without
JUDGE_CPUSET and a pinning policy a separate Judge takes, as in the run, the first cores local to its pair that Work's
cpuset does not use: Work's cpuset recorded in the run's launch-manifest.json (--run-id, same Judge pair), else Work's
cpuset from the config (CPUSET, or the first cores local to GPU_INDICES). A --run-id whose launch manifest names
another Judge pair is reported on stderr and in the manifest (source_judge_gpu_indices). The pair must meet the task's
execution profile; the qualification guard of run_task.py is not applied (this is how a reference Judge qualifies a Judge change, e.g. while runtime/qualification.json is PENDING),
but the manifest records whether the qualification matches the profile and core count used. Output: $MILES_SCRATCH_ROOT/<task>/runs/<run-id>-<suffix>/ (eval.log,
rescore-manifest.json, the SForge report).
"""
from __future__ import annotations

import argparse
import json
import os
import secrets
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_task as R  # noqa: E402


def judge_pair(cfg, judge_gpus=None, shared_gpus=False):
    """(GPU indices, cpuset config key) of the Judge: --judge-gpus or the config's separate Judge pair, else (also with
    --shared-gpus) the shared Work pair."""
    if shared_gpus:
        return cfg["GPU_INDICES"], "CPUSET"
    spec = (judge_gpus or cfg.get("JUDGE_GPU_INDICES") or "").strip()
    if spec and spec != R.SHARED:
        R.require(R.re.fullmatch(r"\d+,\d+", spec), f"JUDGE_GPU_INDICES={spec!r} must be two GPU indices C,D or 'shared'")
        return spec, "JUDGE_CPUSET"
    return cfg["GPU_INDICES"], "CPUSET"


def source_manifest(run_id):
    """The launch-manifest.json of the run being re-scored, {} when there is none (an --archive, or an older layout)."""
    path = R.OUTPUT / "runs" / run_id / "launch-manifest.json" if run_id else None
    return json.loads(path.read_text()) if path and path.is_file() else {}


def work_cores(cfg, source, judge_gpus, count):
    """Work's cores, which a separate Judge's default cpuset skips as in run_task.judge_cpuset(): the run's recorded
    cpuset when it used this Judge pair, else the config's CPUSET or the first `count` cores local to GPU_INDICES."""
    if source.get("judge_placement") == "separate" and source.get("judge_gpu_indices") == judge_gpus \
            and source.get("cpuset"):
        return set(R.parse_cpulist(source["cpuset"]))
    if cfg.get("CPUSET"):
        return set(R.parse_cpulist(cfg["CPUSET"]))
    return set(R.parse_cpulist(R.numa_cpuset(R.gpu_pair(cfg["GPU_INDICES"])[1], count)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--run-id")
    src.add_argument("--archive", type=Path)
    ap.add_argument("--suffix", default="rescore1")
    ap.add_argument("--label", help="with --archive: output run id mdwan-<label>")
    judge = ap.add_mutually_exclusive_group()
    judge.add_argument("--judge-gpus", metavar="C,D", help="the Judge pair (overrides JUDGE_GPU_INDICES)")
    judge.add_argument("--shared-gpus", action="store_true", help="the Judge on GPU_INDICES with CPUSET")
    a = ap.parse_args()
    cfg = R.load_config(a.config)
    try:
        host = R.resolve_host(cfg)
        gpus, cpuset_key = judge_pair(cfg, a.judge_gpus, a.shared_gpus)
        indices, uuids, _ = R.gpu_pair(gpus)
        profile_id = R.admit(R.TASK_JSON, indices)["execution_profile_id"]
        pinning = R.cpu_pinning(cfg, uuids)  # SForge's opt-in CPU pinning policy, None when off
        if pinning:
            host["cpu_cores"] = pinning["cpu_count"]
        review = R.review_settings(cfg)
        review_identity = R.check_review(cfg, review)
    except (R.NotReady, R.NotConfigured, R.ProfileMismatch, R.PinningError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 78
    try:
        R.check_qualification(host["MDWAN_PRIVATE_DATA"], profile_id, host["cpu_cores"], pinning and pinning["policy_id"])
        qualification = "matches"
    except R.NotReady as exc:
        qualification = str(exc)
    if a.run_id:
        archive = R.OUTPUT / "runs" / a.run_id / R.TASK_ID / "final_archive.tar.gz"
        rid = f"{a.run_id}-{a.suffix}"
    else:
        R.require(a.label and R.re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", a.label), "--archive needs a safe --label")
        archive, rid = a.archive.resolve(), f"mdwan-{a.label}"
    R.require(archive.is_file() and archive.stat().st_size > 0, f"no archive at {archive}")
    run_root = R.OUTPUT / "runs" / rid
    R.require(not run_root.exists(), f"{run_root} already exists")
    ids = {role: R.docker("image", "inspect", "--format", "{{.Id}}", f"{R.BENCH}.{role}.{R.TASK_ID}:local").stdout.strip()
           for role in ("work", "judge")}
    source = source_manifest(a.run_id)
    source_judge = source.get("judge_gpu_indices", source.get("gpu_indices"))
    if source_judge and source_judge != gpus:
        print(f"warning: {a.run_id} ran its Judge on GPUs {source_judge}, this rescore on {gpus} (pass --judge-gpus "
              f"{source_judge} or --shared-gpus to match)", file=sys.stderr)
    exclude = ()
    if cpuset_key == "JUDGE_CPUSET" and not pinning and not cfg.get("JUDGE_CPUSET"):
        exclude = work_cores(cfg, source, gpus, host["cpu_cores"])
    cpuset = R.choose_cpuset(cfg, uuids, rid, host["cpu_cores"], pinning, key=cpuset_key, exclude=exclude)
    with R.pair_lease(uuids, cfg.get("LEASE_HOLDER")):
        R.require(R.gpus_idle(uuids), "GPU pair has active compute processes")
        run_root.mkdir(parents=True)
        manifest = {"task_id": R.TASK_ID, "rescore_of": a.run_id or None, "run_id": rid, "archive": str(archive),
                    "images": ids, "gpu_indices": gpus, "gpu_uuids": uuids, "execution_profile_id": profile_id, "cpuset": cpuset,
                    "judge_placement": "separate" if cpuset_key == "JUDGE_CPUSET" else "shared",
                    **({"source_judge_gpu_indices": source_judge} if source_judge and source_judge != gpus else {}),
                    "cpu_cores": host["cpu_cores"], "qualification": qualification, "host": host,
                    "integrity_review": review_identity,
                    **({"cpu_pinning": pinning["record"]} if pinning else {}),
                    "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        env = {k: os.environ[k] for k in ("PATH", "HOME", "USER", "LOGNAME", "LANG", "LC_ALL", "TZ") if k in os.environ}
        env.update({"PYTHONPATH": cfg["HARNESS_ROOT"], "SFORGE_TASKS_DIR": str(R.TASK / "tasks"),
                    "SFORGE_LOG_DIR": str(R.OUTPUT), "SFORGE_VERSION_STORE_ROOT": cfg["SFORGE_VERSION_STORE_ROOT"],
                    "SFORGE_AGENT_RUNTIME_DIR": cfg["SFORGE_AGENT_RUNTIME_DIR"],
                    "SFORGE_ADMIN_SECRET": secrets.token_urlsafe(48), "SFORGE_JUDGE_NETWORK_MODE": "none",
                    "MDWAN_GPU_UUIDS": ",".join(uuids), "MDWAN_RUN_ID": rid, "MDWAN_RUN_ROOT": str(run_root),
                    "MDWAN_ASSETS": host["MDWAN_ASSETS"], "MDWAN_PUBLIC_DATA": host["MDWAN_PUBLIC_DATA"],
                    "MDWAN_PRIVATE_DATA": host["MDWAN_PRIVATE_DATA"], "MDWAN_CPUSET": cpuset})
        env.update(review)  # `sforge eval` reviews the archive before the GPU Judge (llm_judge/POLICY.md)
        if pinning:
            env.update(pinning["record"]["policy_env"])
            if pinning["cpuset_mems"]:
                env["MDWAN_CPUSET_MEMS"] = pinning["cpuset_mems"]
        cmd = [str(Path(cfg["HARNESS_ROOT"]) / ".venv/bin/python"), str(R.ENTRY), "eval", "--backend", "docker",
               "--task", R.TASK_ID, "--archive", str(archive), "--run-id", rid, "--timeout", str(R.JUDGE_TIMEOUT),
               "--json"]
        with open(run_root / "eval.log", "w") as log:
            code = subprocess.call(cmd, env=env, cwd=cfg["HARNESS_ROOT"], stdout=log, stderr=subprocess.STDOUT)
        manifest["eval_exit"] = code
        reports = sorted(run_root.rglob("report.json"))
        if reports:
            rep = json.loads(reports[-1].read_text())
            manifest.update(report=str(reports[-1]), valid=rep.get("valid"), score=rep.get("score"),
                            metrics=rep.get("metrics"))
        manifest["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        (run_root / "rescore-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps(manifest, indent=2))
    return 0 if code == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
