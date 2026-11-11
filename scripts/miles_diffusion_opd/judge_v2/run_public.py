#!/usr/bin/env python3
"""Public checks for the SD3.5-M on-policy distillation task (run by the agent inside the Work container; no score).
They only get a pipeline working: their sizes are not the scored run's, and no timing or quality here predicts a score.

  --suite probe   a probe round (nonzero fixture LoRA, random labels, recorded engine responses)
                  -> transition means and gradient against the float32 oracle
  --suite small   4 real rounds under the in-engine audit hook -> audit W1-W6, AdamW algebra (with every record's
                  optimizer settings == the contract's), publication integrity (engines hold exactly the dumped student
                  after every round), framework path (the run went through train_diffusion.py, the RolloutManager
                  rollout path and the FSDP actor), final_lora.safetensors == the last step's lora_post bit for bit
  --suite perf    a replay workload (1 untimed round, 2 segments of 4 rounds) -> segment times and their median T, AdamW
                  algebra, publication integrity and order (`published` only after every engine applied the round's
                  student); add --quality to also run the real contract, check its final LoRA and report its
                  held-out KL and task metric (generated images: OCR accuracy, PickScore) on the public held-out prompts,
                  next to the original model's and the teachers' values in published_reference.json. A run this short
                  shows that the evaluation works on your LoRA, not how much the student learns.
Exit code 0 means every check of the suite passed.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import statistics
import string
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from replay_server import segment_times, setup_seconds, warmup_seconds  # noqa: E402
from public_common import (adamw_verdict, final_lora_verdict, framework_verdict, paired_with_base,  # noqa: E402
                           probe_failures, publication_verdict, quality_repeats, relabel, route_gain, run_logged,
                           run_tool, sha256_file, start_replay_server, task_metric, write_json)

PY = sys.executable
PUBLIC = Path(os.environ.get("OPD_PUBLIC_DATA", "/data/public"))
APP = Path(os.environ.get("OPD_APP", "/app/workdir/miles_lab"))


def launch(out: Path, name: str, contract: dict, timeout: float, extra_env=None):
    d = out / name
    for sub in ("out", "scratch", "tmp"):
        (d / sub).mkdir(parents=True)
    contract = dict(contract, scratch=str(d / "scratch"))
    write_json(d / "contract.json", contract)
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}  # the candidate brings its own repo
    env.update(extra_env or {})
    code, secs = run_logged([PY, "task_entry.py", "run", "--contract", str(d / "contract.json"), "--out", str(d / "out/run")],
                            d / "task_entry.log", cwd=APP, env=env, timeout=timeout)
    subprocess.run(["bash", "-c", "ray stop --force >/dev/null 2>&1; pkill -9 -f sgl_diffusion; true"], check=False)
    return d, code, secs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=["probe", "small", "perf"], required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--quality", action="store_true",
                    help="perf: also run the real contract, check its final LoRA and report its held-out KL and task "
                         "metric on the public held-out prompts")
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    # checkers parse engine responses with a pristine miles_diffusion copy, never the candidate's repo
    os.environ["PYTHONPATH"] = os.environ.get("OPD_PRISTINE_MD", "/opt/opd_public/md_pristine")
    Q = json.loads((HERE / "qualification.json").read_text())
    TH = Q["thresholds"]
    base = json.loads((HERE / "contract.json").read_text())
    rng = random.Random()
    labels = ["".join(rng.choice(string.ascii_lowercase) for _ in range(7)) for _ in range(2)]
    mapping = {"ocr": labels[0], "pickscore": labels[1]}
    teachers = {labels[0]: base["teachers"]["ocr"], labels[1]: base["teachers"]["pickscore"]}
    rep: dict = {"suite": a.suite, "checks": {}}
    C = rep["checks"]

    def replay_session(name, src: Path, rounds_subset=None, probe_steps=0):
        wl = a.output / f"{name}_workload"
        wl.mkdir()
        manifest = json.loads((src / "manifest.json").read_text())
        rows = relabel([json.loads(line) for line in open(src / "rows.jsonl")], mapping)
        (wl / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        for f in src.glob("row_*"):
            os.symlink(f, wl / f.name)
        write_json(wl / "manifest.json", manifest)
        log = a.output / f"{name}_replay.log"
        server = start_replay_server(wl, log, Q["replay_port"])
        audit = a.output / f"{name}_audit"
        audit.mkdir()
        c = dict(base, teachers=teachers, rows=str(wl / "rows.jsonl"), student_init=str(src / "student_init.safetensors"),
                 rounds=len(manifest["rounds"]), rows_per_round=len(manifest["rounds"][0]), generate_url=f"http://127.0.0.1:{Q['replay_port']}", probe_steps=probe_steps)
        try:
            d, code, secs = launch(a.output, name, c, Q["timeouts"]["session_p"], {"OPD_AUDIT_DIR": str(audit)})
        finally:
            server.terminate()
        ev = [json.loads(line) for line in open(log)]
        C[f"{name}_exit"] = {"passed": code == 0, "code": code, "seconds": round(secs, 1)}
        viol = [e for e in ev if e["kind"] == "violation"]
        pub = sum(1 for e in ev if e["kind"] == "published")
        setup = setup_seconds(log, manifest.get("segments", []))
        C[f"{name}_protocol"] = {"passed": code == 0 and not viol and pub == len(manifest["rounds"]) and setup is not None
                                 and setup <= Q["setup_cap_seconds"], "violations": viol[:5], "published": pub,
                                 "warmup_seconds": warmup_seconds(log), "setup_seconds": setup,
                                 "setup_cap_seconds": Q["setup_cap_seconds"]}
        C[f"{name}_framework"] = framework_verdict(run_tool(
            [HERE / "framework_check.py", "--audit", audit, "--rounds", len(manifest["rounds"]), "--replay-log", log,
             "--out", a.output / f"{name}_framework.json"], a.output / f"{name}_framework.log",
            a.output / f"{name}_framework.json"))
        return d, c, manifest, rows, log, wl

    if a.suite == "probe":
        d, c, manifest, rows, _, wl = replay_session("probe", PUBLIC / "probe", probe_steps=1)
        if C["probe_exit"]["passed"]:
            fx = a.output / "probe_fixture"
            fx.mkdir()
            half = rows[: len(manifest["rounds"][0]) // 2]
            (fx / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in half))
            for r in half:
                for ext in ("json", "msgpack"):
                    shutil.copy2(wl / f"row_{r['metadata']['row_id']:06d}.{ext}", fx)
            shutil.copy2(PUBLIC / "probe/student_init.safetensors", fx / "student_lora.safetensors")
            write_json(fx / "fixture.json", {"labels": {"ocr": labels[0], "aes": labels[1]}})
            step0 = a.output / "probe_step0"
            step0.mkdir()
            shutil.copy2(d / "out/run/steps/step_00000.pt", step0 / "step_00000.pt")
            pr = run_tool([HERE / "probe_check.py", "--fixture", fx, "--dump", step0, "--out", a.output / "probe_check.json",
                           "--model", Q["model"], "--teachers", Q["teachers_dir"]], a.output / "probe_check.log",
                           a.output / "probe_check.json")
            r = (pr or {}).get("report", {})
            C["probe"] = {"passed": not probe_failures(r, TH), "failures": probe_failures(r, TH), **r}
    elif a.suite == "small":
        audit = a.output / "audit"
        audit.mkdir()
        rows = relabel([json.loads(line) for line in open(PUBLIC / "rows.jsonl")], mapping)
        rounds = 4
        rows = rows[: rounds * base["rows_per_round"]]
        rr = a.output / "rows.jsonl"
        rr.write_text("".join(json.dumps(r) + "\n" for r in rows))
        c = dict(base, teachers=teachers, rows=str(rr), student_init=str(PUBLIC / "student_init.safetensors"), rounds=rounds)
        d, code, secs = launch(a.output, "small", c, Q["timeouts"]["session_r"], {"OPD_AUDIT_DIR": str(audit)})
        C["small_exit"] = {"passed": code == 0, "code": code, "seconds": round(secs, 1)}
        if code == 0:
            au = run_tool([HERE / "audit_check.py", "--audit", audit, "--rows", rr, "--rounds", rounds,
                           "--rows-per-round", c["rows_per_round"], "--dump", d / "out/run/steps", "--out", a.output / "audit.json"], a.output / "audit.log", a.output / "audit.json")
            C["audit"] = {"passed": bool(au) and not au["violations"], "violations": (au or {}).get("violations", ["no report"])[:10]}
            ad = run_tool([HERE / "adamw_check.py", "--dump", d / "out/run/steps", "--student-init",
                           PUBLIC / "student_init.safetensors", "--optimizer", json.dumps(base["optimizer"]),
                           "--out", a.output / "adamw.json"], a.output / "adamw.log", a.output / "adamw.json")
            C["adamw"] = adamw_verdict(ad, TH, expect_steps=rounds * c["steps_per_round"])
            C["framework"] = framework_verdict(run_tool(
                [HERE / "framework_check.py", "--audit", audit, "--rounds", rounds, "--out", a.output / "framework.json"],
                a.output / "framework.log", a.output / "framework.json"))
            C["publication"] = publication_verdict(run_tool(
                [HERE / "publication_check.py", "--audit", audit, "--dump", d / "out/run/steps", "--rounds", rounds,
                 "--steps-per-round", c["steps_per_round"], "--out", a.output / "publication.json"],
                a.output / "publication.log", a.output / "publication.json"))
            C["final_lora"] = final_lora_verdict(d / "out/run/final_lora.safetensors", d / "out/run/steps")
    else:
        d, c, manifest, rows, log, _ = replay_session("perf", PUBLIC / "replay")
        published = json.loads((HERE / "published_reference.json").read_text())
        seg, quality = None, None
        if C["perf_exit"]["passed"]:
            seg = segment_times(log, manifest["segments"])
            rep["segment_seconds"] = seg
            rep["t_median"] = statistics.median(seg)
            ad = run_tool([HERE / "adamw_check.py", "--dump", d / "out/run/steps", "--student-init",
                           PUBLIC / "replay/student_init.safetensors", "--optimizer", json.dumps(base["optimizer"]),
                           "--out", a.output / "adamw.json"], a.output / "adamw.log", a.output / "adamw.json")
            C["adamw"] = adamw_verdict(ad, TH, expect_steps=c["rounds"] * c["steps_per_round"])
            C["publication"] = publication_verdict(run_tool(
                [HERE / "publication_check.py", "--audit", a.output / "perf_audit", "--dump", d / "out/run/steps",
                 "--rounds", c["rounds"], "--steps-per-round", c["steps_per_round"], "--replay-log", log,
                 "--out", a.output / "publication.json"], a.output / "publication.log", a.output / "publication.json"))
        if a.quality:
            rows = relabel([json.loads(line) for line in open(PUBLIC / "rows.jsonl")], mapping)
            rr = a.output / "rows_full.jsonl"
            rr.write_text("".join(json.dumps(r) + "\n" for r in rows[: base["rounds"] * base["rows_per_round"]]))
            cf = dict(base, teachers=teachers, rows=str(rr), student_init=str(PUBLIC / "student_init.safetensors"))
            d, code, secs = launch(a.output, "full", cf, Q["timeouts"]["session_r"])
            C["full_exit"] = {"passed": code == 0, "code": code, "seconds": round(secs, 1)}
            if code == 0:
                final = d / "out/run/final_lora.safetensors"
                C["full_final_lora"] = final_lora_verdict(final, d / "out/run/steps")
                KG, TQ = published["public_kl"], published["public_task_quality"]
                held = PUBLIC / "heldout.jsonl"
                try:
                    if not sha256_file(held) == KG["prompts_sha256"] == TQ["prompts_sha256"]:
                        raise RuntimeError("public held-out prompts differ from the published values'")
                    repeats = quality_repeats(final, held, KG["case_seeds"], KG["samples"], Q, a.output)
                    paired_with_base(repeats, KG["base_kl_per_repeat"])
                    gains = {r: route_gain(repeats, KG["base_kl_per_repeat"], r) for r in TQ["routes"]}
                    student, _ = task_metric(final, held, TQ, Q, a.output)
                    quality = {"kl_route_gain": gains, "student": student}
                    C["public_quality"] = {"passed": True}
                except RuntimeError as e:
                    quality = None
                    C["public_quality"] = {"passed": False, "error": str(e)}
        rep["performance"] = performance(seg, quality, published)
    passed = bool(C) and all(v.get("passed") for v in C.values())
    rep["passed"] = passed
    write_json(a.output / "report.json", rep)
    print(json.dumps({k: v for k, v in rep.items() if k != "checks"}, indent=1))
    print(json.dumps({k: v.get("passed") for k, v in C.items()}, indent=1))
    sys.exit(0 if passed else 1)


def performance(seg, quality, published):
    """The perf report: T (median replay segment) and, after --quality, the held-out KL and task metric next to the
    original model's and the teachers' public values."""
    perf = {"T_seconds": statistics.median(seg) if seg else None, "segment_seconds": seg,
            "note": "for getting the pipeline working: this is not the scored workload, so nothing here predicts a score; "
                    "only sforge-submit measures it"}
    if quality is not None:
        routes = published["public_task_quality"]["routes"]
        perf.update(public_kl_route_gain=quality["kl_route_gain"], public_task_metric_per_route=quality["student"],
                    public_original_and_teacher_per_route={r: {"base": v["base"], "teacher": v["teacher"]} for r, v in routes.items()})
    return perf


if __name__ == "__main__":
    main()
