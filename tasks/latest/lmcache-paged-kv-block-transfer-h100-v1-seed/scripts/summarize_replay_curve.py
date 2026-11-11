#!/usr/bin/env python3
"""Summarise a per-version replay into a candidate score curve.

Reads the `measurement.json` files a `replay_candidates.sh --per-version`
batch writes and prints one row per implementation, in replay order, with the
group breakdown.  Rows still in flight are shown as pending rather than as
failures -- a measurement file is written before the judge finishes, so a
half-written record is progress, not a defect.

Usage:
  python3 scripts/summarize_replay_curve.py --replay-root <run dir> [--json out.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("measurement.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        report = payload.get("report") or {}
        implementation = payload.get("implementation") or {}
        judge_report = path.parent / "judge" / "report.json"
        metrics: dict[str, Any] = {}
        if judge_report.is_file():
            try:
                metrics = (json.loads(judge_report.read_text(encoding="utf-8"))
                           .get("metrics") or {})
            except (OSError, ValueError):
                metrics = {}
        rows.append(
            {
                "implementation": path.parent.name,
                "label": implementation.get("comment") or implementation.get("label") or "",
                "status": str(payload.get("status", "?")),
                "valid": bool(report.get("valid")),
                "score": float(report.get("score") or 0.0),
                "device_gbps": float(metrics.get("device_gbps") or 0.0),
                "host_gbps": float(metrics.get("host_gbps") or 0.0),
                "trace_gbps": float(metrics.get("trace_gbps") or 0.0),
                "runtime_seconds": float(report.get("runtime_seconds") or 0.0),
                "started_at": str(payload.get("started_at", "")),
            }
        )
    rows.sort(key=lambda row: row["started_at"])
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)

    rows = load_rows(args.replay_root)
    if not rows:
        print("no measurements found yet")
        return 0

    print(f"{'implementation':28s} {'score':>8s} {'device':>8s} {'host':>6s} "
          f"{'trace':>6s} {'sec':>5s}  label")
    best = None
    for row in rows:
        if row["status"] != "completed":
            print(f"{row['implementation']:28s} {'pending':>8s}"
                  f"{'':22s}  {row['label']}")
            continue
        if not row["valid"]:
            print(f"{row['implementation']:28s} {'INVALID':>8s}"
                  f"{'':22s}  {row['label']}")
            continue
        print(f"{row['implementation']:28s} {row['score']:8.2f} "
              f"{row['device_gbps']:8.1f} {row['host_gbps']:6.1f} "
              f"{row['trace_gbps']:6.1f} {row['runtime_seconds']:5.0f}  {row['label']}")
        if best is None or row["score"] > best["score"]:
            best = row

    completed = [row for row in rows if row["status"] == "completed" and row["valid"]]
    print()
    print(f"completed {len(completed)} of {len(rows)} measurements")
    if best is not None:
        print(f"best: {best['implementation']} at {best['score']:.2f} GB/s "
              f"(device {best['device_gbps']:.1f}, host {best['host_gbps']:.1f}, "
              f"trace {best['trace_gbps']:.1f})")

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(rows, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
