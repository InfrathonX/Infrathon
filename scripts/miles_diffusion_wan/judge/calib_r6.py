#!/usr/bin/env python3
"""S6 calibration summary for the Judge runs of honest variants and planted bugs.

  calib_r6.py --runs-root <Judge runs root> --honest ref_v1_1,serve_h2_efficient,... \
              --plants serve_plant_skip_backward,... [--out summary.json]

Per run and hidden dump: the Judge's R6 trajectory metrics against the fp32 oracle, plus J1/R3/T. Then, from the saved
Δθ_c on U (verifier/delta_U_r{i}_{dump}.pt):
  honest-vs-honest: every ordered pair of honest runs (how different honest implementations are from each other)
  plant-vs-honest:  every plant against every honest run
using the same update_metrics, so thresholds can be set between the two distributions.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import torch

import traj_delta as T


def run_summary(d: Path) -> dict:
    f = d / "verifier" / "wan_judge.json"
    if not f.exists():
        return {"missing": True}
    j = json.loads(f.read_text())
    out = {"valid": j.get("valid"), "score": j.get("score"), "gates_pass": j.get("gates_pass"),
           "T": j.get("timings", {}).get("T"), "error": j.get("error"), "replays": {}}
    g = j.get("gates", {})
    j1 = g.get("J1", {})
    out["J1"] = {k: j1[k] for k in j1 if isinstance(j1[k], dict)}
    for key, v in g.items():
        if key.startswith("R") and "_" in key:
            out["replays"][key] = {"R3": v.get("R3_pi_old"), "R6_equal": (v.get("R6_equal") or {}).get("ok"),
                                   "R6_traj": v.get("R6_traj"),
                                   "fails": sorted(k for k, x in v.items() if isinstance(x, dict) and not x.get("ok", True))}
    return out


def deltas(d: Path) -> dict:
    # key "r{i}_{dump}", e.g. "r0_replay0"
    return {p.stem[len("delta_U_"):]: torch.load(p)["delta_U"].double()
            for p in sorted((d / "verifier").glob("delta_U_r*_*.pt"))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", type=Path, required=True)
    ap.add_argument("--honest", required=True)
    ap.add_argument("--plants", default="")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--oracle-root", type=Path, default=None,
                    help="dir with <dump>/train_oracle_traj.pt (with delta_U_steps): recompute all R6 metrics offline, "
                         "including the step-1 decomposition, and the aggregate gate under run_judge.R6_DEFAULT")
    ap.add_argument("--table-only", action="store_true", help="with --oracle-root: skip the pairwise comparisons")
    a = ap.parse_args()
    honest = [x for x in a.honest.split(",") if x]
    plants = [x for x in a.plants.split(",") if x]
    res = {"runs": {n: run_summary(a.runs_root / n) for n in honest + plants}}
    D = {n: deltas(a.runs_root / n) for n in honest + plants}
    if a.oracle_root:
        import importlib.util

        spec = importlib.util.spec_from_file_location("run_judge", Path(__file__).resolve().parent / "run_judge.py")
        rj = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rj)
        orc = {}
        for key in sorted({k for n in D for k in D[n]}):
            dump = key.split("_", 1)[1]
            o = torch.load(a.oracle_root / dump / "train_oracle_traj.pt")
            orc[key] = (o["delta_U"].double(), o["delta_U_steps"][0].double())
        table = {}
        print("== offline R6 vs oracle (with step-1 decomposition); aggregate = medians over dumps")
        for n in honest + plants:
            per = []
            for key in sorted(D[n]):
                do, d1 = orc[key]
                m = {**T.update_metrics(D[n][key], do), **T.step1_decomposition(D[n][key], d1)}
                per.append(m)
                print(f"{n:<30} {key:<12} sign {m['sign_agree_top']:.3f} corr {m['corr']:.3f} norm {m['norm_ratio']:.2f} "
                      f"step1_resid {m['step1_resid']:.3f} coef {m['step1_coef']:.2f}")
            import types
            fake = types.SimpleNamespace(qual={}, gates={f"R{i}_x": {"R6_traj": m} for i, m in enumerate(per)})
            agg = rj.r6_aggregate(fake)
            table[n] = {"per_dump": per, "aggregate": agg}
            print(f"{n:<30} AGG norm {agg.get('norm_ratio', float('nan')):.2f} corr {agg.get('corr', float('nan')):.3f} "
                  f"resid {agg.get('step1_resid', float('nan')):.3f} -> {'PASS' if agg['ok'] else 'FAIL'} {agg.get('checks')}")
        res["offline_vs_oracle"] = table
        if a.table_only:
            if a.out:
                a.out.write_text(json.dumps(res, indent=1, default=str))
            return
    hh, ph = [], []
    for x, y in itertools.permutations(honest, 2):
        for key in sorted(set(D[x]) & set(D[y])):
            hh.append({"cand": x, "ref": y, "replay": key, **T.update_metrics(D[x][key], D[y][key])})
    for p in plants:
        for h in honest:
            for key in sorted(set(D[p]) & set(D[h])):
                ph.append({"cand": p, "ref": h, "replay": key, **T.update_metrics(D[p][key], D[h][key])})
    res["honest_vs_honest"], res["plant_vs_honest"] = hh, ph

    def fmt(r):
        return (f"{r['cand']:<28} vs {r['ref']:<20} {r['replay']:<12} sign {r['sign_agree_top']:.3f} "
                f"corr {r['corr']:.3f} medratio {r['median_ratio_top']:.2f} norm {r['norm_ratio']:.2f}")

    print("== vs fp32 oracle (Judge R6_traj)")
    for n, s in res["runs"].items():
        for key, v in (s.get("replays") or {}).items():
            t = v.get("R6_traj") or {}
            if "sign_agree_top" in t:
                print(f"{n:<28} {key:<12} sign {t['sign_agree_top']:.3f} corr {t['corr']:.3f} "
                      f"medratio {t['median_ratio_top']:.2f} norm {t['norm_ratio']:.2f} fails {v['fails']}")
            else:
                print(f"{n:<28} {key:<12} {t}")
    print("== honest vs honest")
    for r in hh:
        print(fmt(r))
    print("== plant vs honest")
    for r in ph:
        print(fmt(r))
    if a.out:
        a.out.write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
