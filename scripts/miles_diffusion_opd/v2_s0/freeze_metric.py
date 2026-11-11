#!/usr/bin/env python3
"""Freeze metric v3 of the SD3.5-M DiffusionOPD task v2 into runtime/qualification.json and
runtime/published_reference.json from the S0 stage-B measurements (author side).

  freeze_metric.py --task TASK_DIR --full s0.json,s1.json --pub s0.json,s1.json --cfg 1.0 --closure-gate G \
                   --public-kl public_kl.json --s0-summary summary.json [--status STATUS]

--full / --pub: the task_quality_eval.py shard outputs of base_cfg*, ocrT_cfg*, aesT_cfg* (and the reference) on the
private heldout_task.jsonl / the public heldout.jsonl. --public-kl: quality_eval.py --case-seeds 0 1 outputs of the
original model and of the public reference LoRA on the public held-out prompts ({"base": {...}, "reference": {...}}).
v1's metric and validation records move under "v1_history"; everything run_judge.py reads stays at the top level.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "judge_v2"))
from task_quality_eval import merge  # noqa: E402

ROUTE_TEACHER = {"ocr": "ocrT", "pickscore": "aesT"}
ROUTE_METRIC = {"ocr": "ocr", "pickscore": "pickscore"}
KL_MIN = {"ocr": 0.364, "pickscore": 0.690}
KL_MIN_SOURCE = ("half the mean per-route ln(KL_base / KL_final) of the 4 H100 reference final LoRAs on the private "
                 "256 KL held-out prompts, case seeds 0 and 1 (ref-01..04: OCR 0.7282 / 0.7276 / 0.7266 / 0.7303, mean "
                 "0.7282; PickScore 1.3804 / 1.3793 / 1.3803 / 1.3768, mean 1.3792; $MILES_SCRATCH_ROOT/metric-v2/sd35m/raw/"
                 "private). The 17 honest H100 final LoRAs (4 reference, 13 agent submissions) score OCR 0.7240-0.7361, "
                 "PickScore 1.3717-1.3822; on the ref-01 trajectory the OCR route first passes 0.364 at round 48 (0.389; "
                 "r40 0.212, r32 0.270, r24 -0.314) and the PickScore route at round 16 (0.833)")


def results(group: str) -> dict:
    parts = group.split(",")
    merged = merge(parts) if len(parts) > 1 else json.load(open(parts[0]))
    return {r["name"].split("_cfg")[0]: r for r in merged["results"]}


def frozen_routes(res: dict) -> dict:
    out = {}
    for route, teacher in ROUTE_TEACHER.items():
        b, t = res["base"]["per_route"][route], res[teacher]["per_route"][route]
        out[route] = {"metric": ROUTE_METRIC[route], "base": b["mean"], "teacher": t["mean"], "base_se": b["se"],
                      "teacher_se": t["se"], "n": b["n"]}
        assert b["n"] == t["n"]
    return out


def closure(res: dict, name: str, routes: dict) -> dict:
    return {r: (res[name]["per_route"][r]["mean"] - v["base"]) / (v["teacher"] - v["base"]) for r, v in routes.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=Path, required=True)
    ap.add_argument("--full", required=True)
    ap.add_argument("--pub", required=True)
    ap.add_argument("--cfg", type=float, required=True)
    ap.add_argument("--closure-gate", type=float, required=True)
    ap.add_argument("--closure-gate-source", required=True)
    ap.add_argument("--public-kl", type=Path, required=True)
    ap.add_argument("--s0-summary", type=Path, required=True)
    ap.add_argument("--status", default="PENDING: metric v3 frozen from S0; no v2 reference Judge yet")
    a = ap.parse_args()
    qpath, ppath = a.task / "runtime/qualification.json", a.task / "runtime/published_reference.json"
    q, pub = json.loads(qpath.read_text()), json.loads(ppath.read_text())
    full, pubres = results(a.full), results(a.pub)
    routes = frozen_routes(full)
    kl_v1 = q["metric"]["base_kl_per_repeat"] if "base_kl_per_repeat" in q.get("metric", {}) else \
        q["v1_history"]["metric"]["base_kl_per_repeat"]
    v1_hist = q.get("v1_history") or {k: q[k] for k in ("metric", "metric_v2_validation", "h100_qualification",
                                                         "metric_revision", "identity_migration", "status") if k in q}
    task_quality = {
        "evaluator": "task_quality_eval.py (Judge copy judge/task_quality_eval.py) on session R's final_lora.safetensors, "
                     "which check final_lora binds bit for bit to the last step's lora_post; the official DiffusionOPD "
                     "evaluation pipeline (ali-vilab/DiffusionOPD@0c85bf0 scripts/evaluation.py): diffusers "
                     "StableDiffusion3Pipeline, transformer and text encoders fp16, VAE fp32, fp16 autocast, LoRA merged "
                     "(fp64 W0 + alpha/r B A over the fp32 checkpoint, cast to fp16)",
        "prompts": "private heldout_task.jsonl: the official ocr/test.txt and pickscore/test.txt minus the 14 prompts any "
                   "contract row uses",
        "prompts_sha256": "144d1d672e0326c23d1adb5b77c6d9128a4108b54143e82e641ed6d030c8c46a",
        "route_counts": {r: v["n"] for r, v in routes.items()},
        "cfg": a.cfg, "steps": 40, "res": 512, "case_seed": 0, "batch": 16, "ocr_workers_per_gpu": 8,
        "seeds": "image i uses a CPU generator seeded from sha256('sd35m-mopd-v2:<case_seed>:<i>') (task_quality_eval.item_seed); "
                 "the prompts are dealt over the visible GPUs by index (i mod n), batches of 16 within a route",
        "routes": routes,
        "source": "S0 stage B (H100 NVL GPU 0/1, 2026-10-07, image miles-diffusion-base:sgl859a278e-md29fde6ee, "
                  "$MDOPD_V2_ROOT/s0/out/stageBfull_s{0,1}): the original SD3.5-M and the official OCR / Aes teachers "
                  "(quanhaol/DiffusionOPD@38686355) under this protocol; values are means over the route's prompts "
                  "(OCR accuracy; raw PickScore, logit_scale * cos)",
    }
    q.update({
        "status": a.status,
        "task_id": "miles_diffusion_sd35m_mopd_2xh100_v2",
        "scorers": {"pickscore_model": "/models/scorers/PickScore_v1",
                    "pickscore_processor": "/models/scorers/CLIP-ViT-H-14-processor"},
        "metric": {
            "version": 3,
            "formula": f"mean_r closure_r * 3600 / T if correct, the per-route KL gate passes and every closure_r >= "
                       f"{a.closure_gate!r} else 0; closure_r = (S_r - B_r) / (T_r - B_r)",
            "score_scale_seconds": 3600,
            "quality_unit": "mean per-route closure of the original-to-teacher gap in task metric",
            "closure_gate_min": a.closure_gate,
            "closure_gate_source": a.closure_gate_source,
            "task_quality": task_quality,
            "kl_gate": {
                "evaluator": "quality_eval.py (v1's Judge evaluator, unchanged) on the final LoRA, one case seed per GPU",
                "prompts": "private heldout_kl.jsonl (v1's heldout.jsonl: 128 ocr + 128 pickscore)",
                "prompts_sha256": "312f7fa7ecc3638eb8b6fca268d7cc6c7d421eb90b4d155e5c56a3255d3830c0",
                "case_seeds": [0, 1], "samples": 2,
                "base_kl_per_repeat": [b for b in kl_v1 if b["case_seed"] in (0, 1)],
                "base_source": "v1 metric v2: the original model's per-seed KL measured 2026-09-30 (H100 NVL), seeds 0 and 1",
                "min_route_gain": KL_MIN, "min_route_gain_source": KL_MIN_SOURCE,
            },
        },
        "s0": json.loads(a.s0_summary.read_text()),
        "v1_history": v1_hist,
    })
    for k in ("metric_v2_validation", "h100_qualification", "metric_revision", "identity_migration"):
        q.pop(k, None)
    qpath.write_text(json.dumps(q, indent=1) + "\n")

    proutes = frozen_routes(pubres)
    kl = json.loads(a.public_kl.read_text())
    base_rep = kl["base"]["repeats"]
    ref_rep = kl["reference"]["repeats"]
    ref_gain = {r: sum(math.log(b["per_route"][r]["kl"] / f["per_route"][r]["kl"]) for b, f in zip(base_rep, ref_rep)) / len(base_rep)
                for r in ("ocr", "pickscore")}
    pub_new = {
        "status": f"MEASURED on H100 NVL GPU 0/1, 2026-10-07: public KL gate and task-metric values on the 128 public "
                  f"held-out prompts; the public replay reference T is v1's (same replay workload)",
        "public_replay_reference_seconds": pub["public_replay_reference_seconds"],
        "public_replay_reference_source": pub["public_replay_reference_source"],
        "closure_gate_min": a.closure_gate,
        "public_heldout": "/data/public/heldout.jsonl: 64 ocr + 64 pickscore prompts from the end of the training split "
                          "that no contract row uses and that are not in either test split",
        "public_kl_gate": {"prompts_sha256": "65e7d293969ab1a7b64f0432bb3e088cbdc91583b25084bef6f6e9fb66ad087d",
                           "case_seeds": [0, 1], "samples": 2, "base_kl_per_repeat": base_rep, "min_route_gain": KL_MIN,
                           "public_reference_route_gain": ref_gain},
        "public_task_quality": {"prompts_sha256": "65e7d293969ab1a7b64f0432bb3e088cbdc91583b25084bef6f6e9fb66ad087d",
                                "route_counts": {r: v["n"] for r, v in proutes.items()}, "cfg": a.cfg, "steps": 40,
                                "res": 512, "case_seed": 0, "batch": 16, "ocr_workers_per_gpu": 8, "routes": proutes},
        "public_reference": {"lora": "final LoRA of the reference's full public run h100sd-ref-perfq-03 (2026-09-28)",
                             "closure_per_route": closure(pubres, "pubref", proutes),
                             "kl_route_gain": ref_gain, "T_seconds": pub["public_reference_quality"]["T_seconds"]},
        "note": "perf --quality: full public 64-round run -> final LoRA binding -> per-route KL gate on the public "
                "held-out prompts (case seeds 0, 1; the Judge's minimum gains) -> task metric on the same prompts -> "
                "provisional closure and score against these public values. Provisional: public rows, replay and prompts.",
    }
    ppath.write_text(json.dumps(pub_new, indent=1) + "\n")
    print(json.dumps({"routes": routes, "public_routes": proutes, "public_reference": pub_new["public_reference"]}, indent=1))


if __name__ == "__main__":
    main()
