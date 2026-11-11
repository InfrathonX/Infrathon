"""Judge-owned quality evaluation for the SD3.5-M DiffusionOPD task: held-out teacher-student KL.

For every held-out prompt (route label -> teacher) and sample j, the Judge runs its own copy of the contract sampler
with the *student* (a LoRA, explicitly merged; ``None`` = the untouched base, i.e. KL_init): x_T and the CPS noise come
from canonical keys (case, j, window start, step); exactly as in the contract, steps before the 3-step window are
ODE (Euler) moves and the window steps are CPS eta 0.8 moves along the student's own mean, and at every window step
the routed teacher scores the same state. All three window starts (0, 1, 2) are evaluated, so the window position is
integrated out. The per-pair KL/D is averaged over window steps, starts and samples per route.

    gain = ln(KL_init / KL_final), KL = mean over all held-out pairs of both routes (pooled; the training objective)
    or over one route's pairs (per route; used for the per-route KL gate)

A repeat is one case seed (all noise keys start with it). --case-seeds evaluates several repeats in one process: the
text encoders run and the DiT copies are merged once, then each seed runs the same per-seed loop on the same weights,
so each result equals its own single-seed run bit for bit (also across GPUs and processes).

The per-route KL gate averages the per-route gain over the case seeds, repeat c of the final student paired with
repeat c of the original model (frozen in the qualification record).

Candidate code and the candidate's engine are not involved; text conditioning is the Judge's own fp32 encoding.
DiT forwards run in fp16 autocast over fp16 weights.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "oracle"))
from sd3_oracle import Lora, SD3Oracle, SD3TextEncoder, canonical_normal, cps_mean, cps_std, kl_over_d, sd3_schedule  # noqa: E402

ETA = 0.8
WINDOW = 3
WINDOW_STARTS = (0, 1, 2)


def merged(model_dir: str, lora: Lora | None, dtype=torch.float16) -> SD3Oracle:
    o = SD3Oracle(model_dir, dtype=dtype)
    if lora is not None:
        o._merge(lora)  # permanent explicit merge for this copy
    return o


@torch.no_grad()
def evaluate(model_dir: str, prompts: list[dict], teachers: dict[str, Lora], student: Lora | None,
             samples: int = 2, micro: int = 16, case_seed: int = 0) -> dict:
    return evaluate_seeds(model_dir, prompts, teachers, student, [case_seed], samples, micro)[0]


@torch.no_grad()
def evaluate_seeds(model_dir: str, prompts: list[dict], teachers: dict[str, Lora], student: Lora | None,
                   case_seeds: list[int], samples: int = 2, micro: int = 16) -> list[dict]:
    """evaluate() for each case seed, with the encoders run and the DiT copies merged once."""
    text = SD3TextEncoder(model_dir)
    enc = [text.encode([p["input"]]) for p in prompts]
    del text
    torch.cuda.empty_cache()
    timesteps, sigmas = sd3_schedule(model_dir, 10)
    s_model = merged(model_dir, student)
    t_models = {route: merged(model_dir, lora) for route, lora in teachers.items()}
    return [_evaluate_seed(prompts, enc, timesteps, sigmas, s_model, t_models, samples, micro, c) for c in case_seeds]


def _evaluate_seed(prompts, enc, timesteps, sigmas, s_model, t_models, samples, micro, case_seed) -> dict:
    per_route: dict[str, list[float]] = {}
    items = [(ci, j) for ci in range(len(prompts)) for j in range(samples)]
    for route in t_models:
        route_items = [(ci, j) for ci, j in items if prompts[ci]["route"] == route]
        for lo in range(0, len(route_items), micro):
            chunk = route_items[lo:lo + micro]
            pe = torch.cat([enc[ci][0] for ci, _ in chunk]).cuda()
            pooled = torch.cat([enc[ci][1] for ci, _ in chunk]).cuda()
            # the contract sampler: ODE (Euler) steps before the 3-step CPS window; every window start 0..2 is
            # evaluated so the window position is integrated out exactly rather than sampled
            for start in WINDOW_STARTS:
                x = torch.stack([canonical_normal((16, 64, 64), case_seed, ci, j, -1) for ci, j in chunk]).cuda()
                kls = torch.zeros(len(chunk), dtype=torch.float64, device="cuda")
                for i in range(start + WINDOW):
                    t, s, sn = timesteps[i], float(sigmas[i]), float(sigmas[i + 1])
                    v_s = s_model.velocity(x, t, pe, pooled, autocast_dtype=torch.float16)
                    if i < start:
                        x = (x.double() + (sn - s) * v_s.double()).float()
                        continue
                    v_t = t_models[route].velocity(x, t, pe, pooled, autocast_dtype=torch.float16)
                    mu_s, mu_t = cps_mean(x, v_s, s, sn, ETA), cps_mean(x, v_t, s, sn, ETA)
                    kls += kl_over_d(mu_s, mu_t, sn, ETA)
                    noise = torch.stack([canonical_normal((16, 64, 64), case_seed, ci, j, 100 * start + i)
                                         for ci, j in chunk]).cuda()
                    x = (mu_s + cps_std(sn, ETA) * noise.double()).float()
                per_route.setdefault(route, []).extend((kls / WINDOW).tolist())
    return {route: {"kl": sum(v) / len(v), "n": len(v)} for route, v in per_route.items()}


def pooled(res: dict) -> float:
    return sum(v["kl"] * v["n"] for v in res.values()) / sum(v["n"] for v in res.values())


def gain(init: dict, final: dict) -> float:
    return math.log(pooled(init) / pooled(final))


QUALITY_UNIT = "nats of pooled teacher-student KL reduction"


def metric_v2(final_kl_per_repeat, base_kl_per_repeat, T, gate_min, scale_seconds=3600, unit=QUALITY_UNIT) -> dict:
    """Score of a candidate whose correctness checks all passed (metric v2; the Judge and public perf use it).

    `final_kl_per_repeat` / `base_kl_per_repeat`: pooled held-out KL of the final student and of the original model,
    one per case seed, in the same seed order (repeat c is paired with repeat c). The quality of a repeat is
    -ln(KL) in nats, so quality_score = mean(final) - mean(base) = mean over seeds of ln(KL_base,c / KL_final,c).
    The quality gate passes iff quality_score >= gate_min, and then score = quality_score * scale_seconds / T (nats
    per hour of timed training), else 0 whatever T is. `score` is None when the gate passes but T is not a positive
    finite number of seconds. A final KL that is not positive and finite (a diverged or degenerate student: NaN,
    inf, 0) has no quality: the gate fails (quality_score None, score 0; the values come back JSON-safe, non-finite
    ones as None). An unpaired or non-positive / non-finite base is an evaluator error (ValueError)."""
    final_kl = [float(x) for x in final_kl_per_repeat]
    base_kl = [float(x) for x in base_kl_per_repeat]
    if not final_kl or len(final_kl) != len(base_kl) or not all(math.isfinite(x) and x > 0 for x in base_kl):
        raise ValueError(f"base KL values must be positive, finite and paired per repeat: final {final_kl}, base {base_kl}")
    final = [-math.log(x) if math.isfinite(x) and x > 0 else None for x in final_kl]
    base = [-math.log(x) for x in base_kl]
    measured = None not in final
    base_q = sum(base) / len(base)
    final_q = sum(final) / len(final) if measured else None
    q = final_q - base_q if measured else None
    passed = measured and q >= gate_min
    T = float(T) if isinstance(T, (int, float)) and not isinstance(T, bool) else None
    usable = T is not None and math.isfinite(T) and T > 0
    score = (q * scale_seconds / T if usable else None) if passed else 0.0
    if not measured:
        summary = f"quality gate failed: the final student's held-out KL {final_kl} is not positive and finite: score 0"
    elif not passed:
        summary = f"quality gate failed: quality {q:+.3f} {unit} < {gate_min:g} {unit}: score 0"
    elif not usable:
        summary = f"no usable timed window T ({T}): not scored"
    else:
        summary = f"score {score:.3f} = quality {q:+.3f} {unit} x {scale_seconds:g} / T {T:.1f} s"
    return {"quality_repeats": len(final), "final_quality_per_repeat": final, "final_quality": final_q,
            "base_quality_per_repeat": base, "base_quality": base_q, "quality_score": q, "quality_unit": unit,
            "final_kl_per_repeat": [x if math.isfinite(x) else None for x in final_kl], "base_kl_per_repeat": base_kl,
            "gain_per_repeat": [math.log(b / f) if fq is not None else None for b, f, fq in zip(base_kl, final_kl, final)],
            "quality_gate_min": gate_min, "quality_gate_passed": passed,
            "T_seconds": T if T is not None and math.isfinite(T) else None, "score": score,  # JSON-safe
            "score_formula": f"quality_score * {scale_seconds:g} / T if correct and quality_score >= {float(gate_min)!r} else 0",
            "summary": summary}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/assets/sd35m/stable-diffusion-3.5-medium")
    ap.add_argument("--teachers", default="/assets/teachers/DiffusionOPD-38686355")
    ap.add_argument("--prompts", required=True, help="jsonl rows {input, metadata: {data_source}} (held-out)")
    ap.add_argument("--student", help="PEFT-named LoRA safetensors; omit for the base (KL_init)")
    ap.add_argument("--init-json", help="frozen KL_init result to compute the gain against")
    ap.add_argument("--labels", default='{"ocr": "ocr", "aes": "pickscore"}')
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--case-seed", type=int, default=0)
    ap.add_argument("--case-seeds", type=int, nargs="+",
                    help="several repeats in this process (models loaded once); output: repeats[]")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.case_seeds and args.init_json:
        ap.error("--init-json takes a single --case-seed")
    labels = json.loads(args.labels)
    teachers = {labels["ocr"]: Lora.load(f"{args.teachers}/OCRTeacher/lora"),
                labels["aes"]: Lora.load(f"{args.teachers}/AesTeacher/lora")}
    prompts = [{"input": r["input"], "route": r["metadata"]["data_source"]}
               for r in (json.loads(line) for line in open(args.prompts))]
    student = Lora.load(args.student) if args.student else None
    seeds = args.case_seeds or [args.case_seed]
    per_seed = evaluate_seeds(args.model, prompts, teachers, student, seeds, args.samples)
    res = {"student": args.student, "prompts": args.prompts, "samples": args.samples,
           "repeats": [{"case_seed": c, "per_route": r, "kl_pooled": pooled(r)} for c, r in zip(seeds, per_seed)]}
    if not args.case_seeds:  # the single-seed schema of earlier callers
        res.update(res["repeats"][0])
        if args.init_json:
            res["gain"] = gain(json.load(open(args.init_json))["per_route"], res["per_route"])
    json.dump(res, open(args.out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
