"""Probe checks G1-G4 for the SD3.5-M DiffusionOPD task (Judge side; public copy for `probe`).

Inputs: the fixture dir (rows.jsonl, fixture.json, student_lora.safetensors, recorded engine responses
row_*.{json,msgpack}) and the candidate's probe dump (exactly one ``step_00000.pt``). Engine responses are
parsed with a pristine miles_diffusion@29fde6ee copy on PYTHONPATH (never the candidate's code).

G1 exact contract: dumped pair set == {(row, output, window step)} of the fixture; routes == fixture labels.
G2 fp64 self-consistency: per-pair KL and the step loss recomputed from the dumped mu_S / mu_T and the pair's own
   sigma_{i+1} (CPS std) agree with the dumped values (rel <= 1e-3).
G3 independent fp32 oracle (explicit LoRA merge, fp64 CPS) at the engine's x_i with the engine's conditioning:
   e_T = ||mu_T^cand - mu_T^or|| / ||mu_T^or - mu_base^or||, e_S likewise (fixed-scale denominator).
G4 gradient: g_cand from the first AdamW moment (m1 / (1 - beta1), clip undone with the dumped pre-clip norm) vs
   the oracle gradient of the pair-mean KL w.r.t. the student LoRA (cos, rel-L2); first-step sign agreement on
   the larger half of |g_or|; the update is nonzero (no skipped step).
Thresholds are applied by the caller (calibrated on honest fp16 / fp32 implementations and planted bugs).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

import msgpack
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "oracle"))
from sd3_oracle import PEFT_PREFIX, Lora, SD3Oracle, cps_mean, cps_std, lora_grad_scope, strict_fp32  # noqa: E402

ETA = 0.8


def load_fixture(fx: str):
    from miles.utils.diffusion_rollout_response import apply_rollout_image_response
    from miles.utils.types import Sample

    rows = [json.loads(line) for line in open(os.path.join(fx, "rows.jsonl"))]
    pairs = {}
    for row in rows:
        rid = row["metadata"]["row_id"]
        rec = json.load(open(os.path.join(fx, f"row_{rid:06d}.json")))
        bodies = msgpack.unpackb(open(os.path.join(fx, f"row_{rid:06d}.msgpack"), "rb").read(), raw=False)
        window = rec["payload"]["rollout_sde_step_indices"]
        for j, body in enumerate(bodies):
            s = apply_rollout_image_response(Sample(prompt=row["input"]), body)
            traj, env = s.dit_trajectory, s.denoising_env
            pos = ({int(i): p for p, i in enumerate(traj.latent_step_indices.tolist())}
                   if traj.latent_step_indices is not None else {i: i for i in range(traj.latents.shape[0])})
            pe = env.pos_cond_kwargs.encoder_hidden_states[0]
            pooled = env.pos_cond_kwargs.pooled_projections[0]
            for i in window:
                pairs[(rid, j, int(i))] = {
                    "x": traj.latents[pos[int(i)]].float(), "t": float(traj.timesteps[i]),
                    "t_next": float(traj.timesteps[i + 1]),
                    "pe": pe if pe.ndim == 3 else pe.unsqueeze(0),
                    "pooled": pooled if pooled.ndim == 2 else pooled.unsqueeze(0),
                    "route": row["metadata"]["data_source"],
                }
    return rows, pairs


def batches(keys, pairs, micro):
    for lo in range(0, len(keys), micro):
        chunk = keys[lo:lo + micro]
        yield (chunk, torch.stack([pairs[k]["x"] for k in chunk]).cuda(),
               torch.tensor([pairs[k]["t"] for k in chunk]),
               torch.cat([pairs[k]["pe"] for k in chunk]).cuda(),
               torch.cat([pairs[k]["pooled"] for k in chunk]).cuda())


def flat(d: dict[str, torch.Tensor], keys: list[str]) -> torch.Tensor:
    return torch.cat([d[k].double().flatten() for k in keys])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", required=True)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--model", default="/assets/sd35m/stable-diffusion-3.5-medium")
    ap.add_argument("--teachers", default="/assets/teachers/DiffusionOPD-38686355")
    ap.add_argument("--micro", type=int, default=None,
                    help="pairs per oracle forward/backward (default: 8; 4 on cards below 80 GiB, where the fp32 G4 "
                         "backward at 8 does not fit; G4 sums the micro-batch gradients, so only fp32 summation order changes)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.micro is None:
        args.micro = 8 if torch.cuda.get_device_properties(0).total_memory >= 80 * 2**30 else 4
    strict_fp32()
    fixture = json.load(open(os.path.join(args.fixture, "fixture.json")))
    labels = fixture["labels"]
    teacher_by_label = {labels["ocr"]: Lora.load(f"{args.teachers}/OCRTeacher/lora"),
                        labels["aes"]: Lora.load(f"{args.teachers}/AesTeacher/lora")}
    student = Lora.load(os.path.join(args.fixture, "student_lora.safetensors"))
    _, pairs = load_fixture(args.fixture)
    report: dict = {"checks": {}, "oracle_micro": args.micro}

    steps = sorted(f for f in os.listdir(args.dump) if f.startswith("step_") and f.endswith(".pt"))
    report["checks"]["exactly_one_step"] = len(steps) == 1
    if not steps:
        report["checks"]["dump_present"] = False
        json.dump({"report": report}, open(args.out, "w"), indent=1); print(json.dumps(report)); return
    rec = torch.load(os.path.join(args.dump, steps[0]), weights_only=True)
    dumped = {(int(r[0]), int(r[1]), int(r[3])): {"route": r[4], "kl": float(r[5]), "idx": n}
              for n, r in enumerate(rec["pairs"])}
    # ------------------------------------------------------------------------------------------ G1
    report["checks"]["G1_pair_set"] = set(dumped) == set(pairs) and len(rec["pairs"]) == len(pairs)
    report["checks"]["G1_routes"] = all(dumped[k]["route"] == pairs[k]["route"] for k in pairs if k in dumped)
    means = rec.get("means")
    report["checks"]["means_present"] = means is not None and len(means) == len(rec["pairs"])
    if not (report["checks"]["G1_pair_set"] and report["checks"]["means_present"]):
        report["missing"] = [list(k) for k in sorted(set(pairs) - set(dumped))][:10]
        report["extra"] = [list(k) for k in sorted(set(dumped) - set(pairs))][:10]
        json.dump({"report": report}, open(args.out, "w"), indent=1); print(json.dumps(report, indent=1)); return
    # ------------------------------------------------------------------------------------------ G2
    kl64, g2 = {}, []
    for k, d in dumped.items():
        mu_s, mu_t = means[d["idx"]]
        std = cps_std(pairs[k]["t_next"] / 1000.0, ETA)
        kl64[k] = ((mu_s.double() - mu_t.double()) ** 2).mean().item() / (2 * std * std)
        g2.append(abs(kl64[k] - d["kl"]) / max(abs(kl64[k]), 1e-300))
    loss64 = sum(kl64.values()) / len(kl64)
    report["G2"] = {"kl_rel_max": max(g2), "loss_fp64_from_means": loss64}
    report["checks"]["G2"] = max(g2) <= 1e-3
    # ------------------------------------------------------------------------------------------ G3
    oracle = SD3Oracle(args.model, dtype=torch.float32)
    e_t, e_s, per_pair, mu_t_or = [], [], {}, {}
    for label, teacher in teacher_by_label.items():
        ks = sorted(k for k in pairs if pairs[k]["route"] == label)
        for chunk, x, t, pe, pooled in batches(ks, pairs, args.micro):
            with torch.no_grad():
                with oracle.lora_scope(None):
                    v_b = oracle.velocity(x, t, pe, pooled)
                with oracle.lora_scope(teacher):
                    v_t = oracle.velocity(x, t, pe, pooled)
                with oracle.lora_scope(student):
                    v_s = oracle.velocity(x, t, pe, pooled)
            for n, k in enumerate(chunk):
                s, sn = pairs[k]["t"] / 1000.0, pairs[k]["t_next"] / 1000.0
                mb, mt, ms = (cps_mean(x[n:n + 1], v[n:n + 1], s, sn, ETA)[0].cpu() for v in (v_b, v_t, v_s))
                mu_t_or[k] = mt
                denom = (mt - mb).norm().item()
                cand_s, cand_t = means[dumped[k]["idx"]]
                e_t.append((cand_t.double() - mt).norm().item() / denom)
                e_s.append((cand_s.double() - ms).norm().item() / denom)
                kl_or = ((ms - mt) ** 2).mean().item() / (2 * cps_std(sn, ETA) ** 2)
                per_pair[str(k)] = {"e_T": e_t[-1], "e_S": e_s[-1], "kl_or": kl_or, "kl_cand": dumped[k]["kl"]}
    q = lambda v, p: sorted(v)[int(p * (len(v) - 1))]  # noqa: E731
    report["G3"] = {"e_T_median": statistics.median(e_t), "e_T_p90": q(e_t, 0.9), "e_T_max": max(e_t),
                    "e_S_median": statistics.median(e_s), "e_S_p90": q(e_s, 0.9), "e_S_max": max(e_s),
                    "kl_rel_median_diag": statistics.median(abs(v["kl_cand"] - v["kl_or"]) / v["kl_or"]
                                                           for v in per_pair.values())}
    # ------------------------------------------------------------------------------------------ G4
    n_pairs = len(pairs)
    with lora_grad_scope(oracle, student) as params:
        for chunk, x, t, pe, pooled in batches(sorted(pairs), pairs, args.micro):
            v_s = oracle.velocity(x, t, pe, pooled)
            loss = 0.0
            for n, k in enumerate(chunk):
                s, sn = pairs[k]["t"] / 1000.0, pairs[k]["t_next"] / 1000.0
                ms = cps_mean(x[n:n + 1], v_s[n:n + 1], s, sn, ETA)[0]
                loss = loss + ((ms - mu_t_or[k].cuda()) ** 2).mean() / (2 * cps_std(sn, ETA) ** 2)
            (loss / n_pairs).backward()
        g_or = {}
        for name, (a, b) in params.items():
            g_or[f"{PEFT_PREFIX}{name}.lora_A.weight"] = a.grad.detach().cpu()
            g_or[f"{PEFT_PREFIX}{name}.lora_B.weight"] = b.grad.detach().cpu()
    keys = sorted(g_or)
    report["checks"]["G4_keys"] = set(keys) == set(rec["exp_avg"]) == set(rec["lora_pre"])
    beta1 = rec["betas"][0]
    gnorm = rec["grad_norm_preclip"]
    clip_coef = min(1.0, rec["clip_grad"] / (gnorm + 1e-6))
    g_c = flat(rec["exp_avg"], keys) / (1 - beta1) / clip_coef
    g_o = flat(g_or, keys)
    cos = torch.nn.functional.cosine_similarity(g_c, g_o, dim=0).item()
    rel_l2 = ((g_c - g_o).norm() / g_o.norm()).item()
    report["G4"] = {"cos": cos, "rel_l2": rel_l2, "grad_norm_cand_preclip": gnorm, "grad_norm_oracle": g_o.norm().item(),
                    "adam_step": rec["adam_step"]}
    if rec.get("lora_post") is not None:
        delta = flat(rec["lora_post"], keys) - flat(rec["lora_pre"], keys)
        big = g_o.abs() > g_o.abs().median()
        report["G4"]["sign_agree_top_half"] = ((torch.sign(delta[big]) == -torch.sign(g_o[big])).double().mean().item())
        report["G4"]["update_nonzero"] = bool(delta.abs().max().item() > 0)
    # the student init used by the candidate must be the fixture LoRA
    report["checks"]["init_is_fixture"] = all(
        torch.equal(rec["lora_pre"][f"{PEFT_PREFIX}{n}.lora_A.weight"].float(), student.pairs[n][0]) and
        torch.equal(rec["lora_pre"][f"{PEFT_PREFIX}{n}.lora_B.weight"].float(), student.pairs[n][1]) for n in student.pairs)
    json.dump({"report": report, "per_pair": per_pair}, open(args.out, "w"), indent=1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
