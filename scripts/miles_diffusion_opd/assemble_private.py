"""Assemble the Judge's private data root for the SD3.5-M DiffusionOPD task.

  <out>/session_p/  manifest.json (round 0 = probe round, rounds 1..R = replay rounds; segments over the replay rounds),
                    rows.jsonl (canonical labels "ocr"/"pickscore"), row_*.{json,msgpack} (recorded engine responses),
                    student_init.safetensors (= the probe fixture's nonzero LoRA)
  <out>/run/        rows.jsonl (real-run rows, canonical labels, per-row seeds), student_init.safetensors
  <out>/heldout.jsonl, <out>/kl_init.json  (quality: held-out prompts, frozen KL of the untouched base)
The Judge randomises route labels per submission; the data here keeps the canonical ones.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil


def link(src: str, dst: str) -> None:
    """Copy (not hard-link): the private root must get its own inodes so it can be made unreadable to the candidate
    without touching the recordings it came from (which may be root-owned)."""
    if not os.path.exists(dst):
        shutil.copyfile(src, dst)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-fixture", required=True, help="make_probe_fixture output (rows, fixture.json, student_lora)")
    ap.add_argument("--probe-record", required=True, help="engine responses recorded from the fixture LoRA")
    ap.add_argument("--replay-workload", required=True, help="build_workload output (manifest, rows, responses)")
    ap.add_argument("--run-rows", required=True)
    ap.add_argument("--run-rounds", type=int, required=True)
    ap.add_argument("--rows-per-round", type=int, default=8)
    ap.add_argument("--student-init", required=True)
    ap.add_argument("--heldout", required=True)
    ap.add_argument("--kl-init", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sp = os.path.join(args.out, "session_p")
    os.makedirs(sp, exist_ok=True)
    os.makedirs(os.path.join(args.out, "run"), exist_ok=True)

    fx = json.load(open(os.path.join(args.probe_fixture, "fixture.json")))
    canon = {fx["labels"]["ocr"]: "ocr", fx["labels"]["aes"]: "pickscore"}
    probe_rows = [json.loads(line) for line in open(os.path.join(args.probe_fixture, "rows.jsonl"))]
    for r in probe_rows:
        r["metadata"]["data_source"] = canon[r["metadata"]["data_source"]]
    wl_manifest = json.load(open(os.path.join(args.replay_workload, "manifest.json")))
    wl_rows = [json.loads(line) for line in open(os.path.join(args.replay_workload, "rows.jsonl"))]
    rounds = [[r["metadata"]["row_id"] for r in probe_rows]] + wl_manifest["rounds"]
    segments = [[a + 1, b + 1] for a, b in wl_manifest["segments"]]
    ids = [row for rnd in rounds for row in rnd]
    assert len(ids) == len(set(ids)), "probe and replay row ids collide"
    with open(os.path.join(sp, "rows.jsonl"), "w") as f:
        for r in probe_rows + wl_rows:
            f.write(json.dumps(r) + "\n")
    for r in probe_rows:
        rid = r["metadata"]["row_id"]
        for ext in ("json", "msgpack"):
            link(os.path.join(args.probe_record, f"row_{rid:06d}.{ext}"), os.path.join(sp, f"row_{rid:06d}.{ext}"))
    for rid in (row for rnd in wl_manifest["rounds"] for row in rnd):
        for ext in ("json", "msgpack"):
            link(os.path.join(args.replay_workload, f"row_{rid:06d}.{ext}"), os.path.join(sp, f"row_{rid:06d}.{ext}"))
    shutil.copy2(os.path.join(args.probe_fixture, "student_lora.safetensors"), os.path.join(sp, "student_init.safetensors"))
    json.dump({"rounds": rounds, "segments": segments}, open(os.path.join(sp, "manifest.json"), "w"), indent=1)

    run_rows = [json.loads(line) for line in open(args.run_rows)][: args.run_rounds * args.rows_per_round]
    assert len(run_rows) == args.run_rounds * args.rows_per_round
    with open(os.path.join(args.out, "run/rows.jsonl"), "w") as f:
        for r in run_rows:
            f.write(json.dumps(r) + "\n")
    shutil.copy2(args.student_init, os.path.join(args.out, "run/student_init.safetensors"))
    shutil.copy2(args.heldout, os.path.join(args.out, "heldout.jsonl"))
    shutil.copy2(args.kl_init, os.path.join(args.out, "kl_init.json"))
    print(f"session_p: {len(rounds)} rounds ({len(segments)} timed segments); run: {len(run_rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
