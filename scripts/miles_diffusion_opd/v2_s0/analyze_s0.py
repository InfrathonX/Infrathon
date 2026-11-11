#!/usr/bin/env python3
"""S0 analysis for the SD3.5-M DiffusionOPD task v2: merge task_quality_eval.py shard outputs and report, per CFG, every
model's per-route task metric (mean, SE) and its closure of the original-to-teacher gap.

  analyze_s0.py --out summary.json GROUP [GROUP ...]
GROUP is one evaluation's shard outputs, comma-separated (stageA_s0/res.json,stageA_s1/res.json).
Config names are '<model>_cfg<scale>'; the original model is 'base', the teachers 'ocrT' (OCR route) and 'aesT'
(PickScore route).

Paired SE of a closure: the same prompts and seeds are used for every model, so the SE of S - B is computed from
per-prompt differences; the teacher-gap uncertainty is propagated to first order.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "judge_v2"))
from task_quality_eval import merge  # noqa: E402

TEACHER = {"ocr": "ocrT", "pickscore": "aesT"}


def paired(a: dict, b: dict) -> tuple[float, float]:
    """mean and SE of per-prompt a - b on the common indices."""
    ia = {it["index"]: it["score"] for it in a}
    d = [ia[it["index"]] - it["score"] for it in b if it["index"] in ia]
    m = sum(d) / len(d)
    sd = math.sqrt(sum((x - m) ** 2 for x in d) / (len(d) - 1))
    return m, sd / math.sqrt(len(d))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    configs = {}
    for group in a.runs:
        parts = group.split(",")
        merged = merge(parts) if len(parts) > 1 else json.load(open(parts[0]))
        for r in merged["results"]:
            configs[r["name"]] = r
    out = {}
    for cfg in sorted({n.split("_cfg")[1] for n in configs}):
        names = {n.split("_cfg")[0]: configs[n] for n in configs if n.split("_cfg")[1] == cfg}
        if "base" not in names:
            continue
        rows = {}
        for model, res in sorted(names.items()):
            row = {}
            for route, v in res["per_route"].items():
                items = [it for it in res["items"] if it["route"] == route]
                base_items = [it for it in names["base"]["items"] if it["route"] == route]
                row[route] = {"mean": v["mean"], "se": v["se"], "n": v["n"]}
                t = names.get(TEACHER[route])
                if t is not None and model != "base":
                    t_items = [it for it in t["items"] if it["route"] == route]
                    gap, gap_se = paired(t_items, base_items)
                    gain, gain_se = paired(items, base_items)
                    c = gain / gap
                    row[route].update(closure=c, closure_se=abs(c) * math.sqrt((gain_se / gain) ** 2 + (gap_se / gap) ** 2)
                                      if gain else gain_se / abs(gap), gain=gain, gain_se=gain_se, gap=gap, gap_se=gap_se)
            rows[model] = row
        out[f"cfg{cfg}"] = rows
    json.dump(out, open(a.out, "w"), indent=1)
    for cfg, rows in out.items():
        print(f"== {cfg}")
        for model, row in rows.items():
            print(f"  {model:18s}", "  ".join(
                f"{route}: {v['mean']:.4f}±{v['se']:.4f}" + (f" closure {v['closure']:+.3f}±{v['closure_se']:.3f}" if "closure" in v else "")
                for route, v in sorted(row.items())))


if __name__ == "__main__":
    main()
