#!/usr/bin/env python3
"""Record authoritative manual-evaluation feedback and the independent final point."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("task_root", type=Path)
    parser.add_argument("--final-eval", type=Path, required=True)
    parser.add_argument("--interval-minutes", type=int, required=True)
    parser.add_argument("--final-minutes", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    final = json.loads(args.final_eval.read_text(encoding="utf-8"))
    report = final["report"]
    history = json.loads((args.task_root / "run_history.json").read_text(encoding="utf-8"))
    experiment = json.loads(
        (args.task_root.parent / "experiment-manifest.json").read_text(encoding="utf-8")
    )
    started_at = dt.datetime.fromisoformat(experiment["started_at"].replace("Z", "+00:00")).timestamp()
    points = []
    for entry in history.get("entries", []):
        submitted_at = float(entry.get("submitted_at", started_at))
        points.append(
            {
                "minutes": max(0.0, (submitted_at - started_at) / 60.0),
                "round": entry.get("round"),
                "final_only": False,
                "valid": bool(entry.get("valid")),
                "score": float(entry.get("score") or 0.0),
            }
        )
    points.append(
        {
            "minutes": args.final_minutes,
            "round": "independent-final",
            "final_only": True,
            "valid": bool(report.get("valid")),
            "score": float(report.get("score", 0.0)),
        }
    )
    curve = {
        "schema_version": 1,
        "manual_evaluations_during_work": len(history.get("entries", [])),
        "points": points,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "curve.json").write_text(
        json.dumps(curve, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "curve.md").write_text(
        "# Iterative evaluation curve\n\n"
        f"Manual evaluations during Work: {len(history.get('entries', []))}.\n\n"
        + "".join(
            f"- {point['round']} at {point['minutes']:.1f} min: "
            f"valid={point['valid']}, score={point['score']:.6f}\n"
            for point in points[:-1]
        )
        + "\n"
        f"- timeout boundary: {args.final_minutes} minutes\n"
        f"- valid: {bool(report.get('valid'))}\n"
        f"- throughput score: {float(report.get('score', 0.0)):.6f} tok/s\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
