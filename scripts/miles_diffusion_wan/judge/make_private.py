#!/usr/bin/env python3
"""Assemble a Judge private directory (the layout `run_judge.py` expects under /judge/private) from authoring outputs.

  private/
    qualification.json        tolerances, replay list, run timeout (root-only at Judge time; the metric v2 block
                              is baked into the images from runtime/qualification.json instead)
    contract.json             contract_template.json (agent-readable)
    run_prompts.jsonl         prompts of the `run` phase (agent-readable)
    score_set/                videos.npz, prompts.jsonl (agent-readable) + hpsv3_oracle.json (root-only)
    probe/dump/               J1 probe dump (agent-readable)   probe/oracle/  fp32 probe oracle + HPSv3 oracle
    dumps/<name>/             hidden replay dumps (meta.json, frames.npy, latents/cond.safetensors; staged at go)
    oracles/<name>/           hpsv3_oracle.json, train_oracle_pi_old.json, build/ (dump build logs, requests)

Files are copied, not linked: the Judge chowns/chmods the whole tree.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
DUMP_FILES = ("meta.json", "frames.npy", "latents.safetensors", "cond.safetensors")


def copy_dump(src: Path, dst: Path, build: Path | None):
    dst.mkdir(parents=True)
    for f in DUMP_FILES:
        shutil.copy2(src / f, dst / f)
    if build is not None:
        build.mkdir(parents=True, exist_ok=True)
        for f in src.iterdir():
            if f.name not in DUMP_FILES:
                shutil.copy2(f, build / f.name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--run-prompts", type=Path, required=True)
    ap.add_argument("--score-set", type=Path, required=True, help="dir with videos.npz, prompts.jsonl")
    ap.add_argument("--score-oracle", type=Path, required=True, help="hpsv3_oracle.json of the score set")
    ap.add_argument("--probe-dump", type=Path, required=True)
    ap.add_argument("--probe-oracle", type=Path, required=True)
    ap.add_argument("--replay", action="append", required=True, help="name=dump_dir:oracle_dir (repeatable)")
    ap.add_argument("--tau-pi-old-mean", type=float, required=True)
    ap.add_argument("--tau-pi-old-max", type=float, required=True)
    ap.add_argument("--run-timeout-s", type=int, default=2400)
    ap.add_argument("--j4-samples", type=int, default=None)
    ap.add_argument("--note", default="")
    ap.add_argument("--tau-pi-old-signed-min", type=float, default=-5e-5)
    ap.add_argument("--run-seed", type=int, default=None, help="contract run.run_seed (default: template)")
    ap.add_argument("--public", action="store_true",
                    help="public root for the Work container (world-readable); default: Judge private root (0700)")
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit(f"{a.out} exists; refusing to overwrite")
    a.out.mkdir(parents=True)

    contract = json.loads((HERE / "contract_template.json").read_text())
    if a.run_seed is not None:
        contract["run"]["run_seed"] = a.run_seed
    (a.out / "contract.json").write_text(json.dumps(contract, indent=1))
    shutil.copy2(a.run_prompts, a.out / "run_prompts.jsonl")
    (a.out / "score_set").mkdir()
    for f in ("videos.npz", "prompts.jsonl"):
        shutil.copy2(a.score_set / f, a.out / "score_set" / f)
    shutil.copy2(a.score_oracle, a.out / "score_set" / "hpsv3_oracle.json")
    copy_dump(a.probe_dump, a.out / "probe" / "dump", a.out / "probe" / "oracle" / "build")
    shutil.copytree(a.probe_oracle, a.out / "probe" / "oracle", dirs_exist_ok=True)

    names = []
    for spec in a.replay:
        name, rest = spec.split("=", 1)
        dump, oracle = (Path(x) for x in rest.split(":", 1))
        copy_dump(dump, a.out / "dumps" / name, a.out / "oracles" / name / "build")
        for f in ("hpsv3_oracle.json", "train_oracle_pi_old.json"):
            shutil.copy2(oracle / f, a.out / "oracles" / name / f)
        for f in ("train_oracle_traj.pt", "train_oracle_traj.json"):  # R6 trajectory oracle, when precomputed
            if (oracle / f).exists():
                shutil.copy2(oracle / f, a.out / "oracles" / name / f)
        names.append(name)

    qual = {"tau_pi_old": {"mean": a.tau_pi_old_mean, "max": a.tau_pi_old_max, "signed_min": a.tau_pi_old_signed_min},
            "run_timeout_s": a.run_timeout_s, "replay_dumps": names,
            "replay_optimizer_steps": json.loads((HERE / "contract_template.json").read_text())["replay"]["optimizer_steps"],
            "j4_samples": a.j4_samples, "note": a.note}
    import importlib.util

    spec = importlib.util.spec_from_file_location("run_judge", HERE / "run_judge.py")
    rj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rj)
    qual["r6"] = dict(rj.R6_DEFAULT)
    (a.out / "qualification.json").write_text(json.dumps(qual, indent=1))
    import subprocess

    if a.public:
        subprocess.run(["chmod", "-R", "a+rX,go-w", str(a.out)], check=True)
    else:  # the candidate (uid agent) must not be able to open it; the Judge (root) reads it through a read-only mount
        subprocess.run(["chmod", "-R", "go-rwx", str(a.out)], check=True)
    print(json.dumps(qual, indent=1))


if __name__ == "__main__":
    main()
