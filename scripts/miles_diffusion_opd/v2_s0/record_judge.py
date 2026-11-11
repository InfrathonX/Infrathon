#!/usr/bin/env python3
"""Append a v2 qualification Judge's readings to runtime/qualification.json (v2_qualification.runs) (author side).

  record_judge.py --task TASK_DIR --qual-dir $MDOPD_V2_ROOT/qual RUN_ID [RUN_ID ...]
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from miles_common.host_config import HostConfig  # noqa: E402

HOST = HostConfig()


def peaks(nvml: Path) -> dict:
    m = {}
    for row in csv.reader(open(nvml)):
        if len(row) >= 3 and row[2].strip()[:1].isdigit():
            g, v = row[1].strip(), int(row[2].strip().split()[0])
            m[g] = max(m.get(g, 0), v)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=Path, required=True)
    ap.add_argument("--qual-dir", type=Path, required=True)
    ap.add_argument("runs", nargs="+")
    a = ap.parse_args()
    qp = a.task / "runtime/qualification.json"
    q = json.loads(qp.read_text())
    vq = q.setdefault("v2_qualification", {"host": "H100 NVL GPU 0/1 (NVLink pair), SFORGE_CPUSET=auto (0-7,32-39, "
                                                  "NUMA 0, quota 24), LLM review off (no review credentials on the host)",
                                          "runner": "$MDOPD_V2_ROOT/qual/judge.sh (runtime/sforge_entry.py eval with the "
                                                    "built images)", "runs": {}})
    for rid in a.runs:
        log = a.qual_dir / "logs" / f"{rid}.eval.log"
        t = log.read_text()
        d = json.loads(t[t.index("{"):])
        meta = (a.qual_dir / "logs" / f"{rid}.meta").read_text().splitlines()
        m = d.get("metrics", {})
        checks = {x["name"]: x["status"] for x in d.get("details", [])}
        for var in ("MILES_SCRATCH_ROOT", "MDOPD_V2_ROOT"):  # host paths are recorded by name, not value
            meta = [line.replace(HOST.value(var), "$" + var) for line in meta]
        vq["runs"][rid] = {
            "archive": meta[1].split()[1], "archive_sha256": meta[1].split()[2], "started": meta[2],
            "images": {"work": meta[3].split()[1], "judge": meta[4].split()[1]}, "end": meta[-1],
            "valid": d.get("valid"), "score": d.get("score"), "summary": d.get("summary"),
            "checks_passed": sum(v == "PASSED" for v in checks.values()), "checks": checks,
            **{k: m.get(k) for k in ("t_seconds", "segment_seconds", "quality_score", "closure_per_route",
                                     "student_metric_per_route", "kl_route_gain") if k in m},
            "gpu_peak_mib": peaks(a.qual_dir / "logs" / f"{rid}.nvml.csv"),
        }
    qp.write_text(json.dumps(q, indent=1) + "\n")
    print(json.dumps({r: {k: vq["runs"][r][k] for k in ("valid", "score", "summary", "checks_passed")} for r in a.runs}, indent=1))


if __name__ == "__main__":
    main()
