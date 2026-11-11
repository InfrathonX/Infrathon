"""Assemble the public data root of the SD3.5-M DiffusionOPD task (baked into the Work image at /data/public).

  <out>/probe/    manifest.json (one round), rows.jsonl (canonical labels), row_*.{json,msgpack} recorded from the public
                  fixture LoRA, student_init.safetensors (= the public fixture LoRA)          -> run_public --suite probe
  <out>/replay/   build_workload output of the public reference run (18 rounds, 3 segments)  -> run_public --suite perf
  <out>/rows.jsonl, <out>/student_init.safetensors   public 64-round contract                -> --suite small / --quality
  <out>/heldout.jsonl, <out>/kl_init.json            public held-out prompts, frozen KL of the untouched base
Everything is disjoint from the Judge's private data (other prompts, row ids, init seed, fixture LoRA).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-fixture", required=True)
    ap.add_argument("--probe-record", required=True)
    ap.add_argument("--replay-workload", required=True)
    ap.add_argument("--contract", required=True, help="dir with the public rows.jsonl and student_init.safetensors")
    ap.add_argument("--heldout", required=True)
    ap.add_argument("--kl-init", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out, "probe"))

    fx = json.load(open(os.path.join(args.probe_fixture, "fixture.json")))
    canon = {fx["labels"]["ocr"]: "ocr", fx["labels"]["aes"]: "pickscore"}
    rows = [json.loads(line) for line in open(os.path.join(args.probe_fixture, "rows.jsonl"))]
    for r in rows:
        r["metadata"]["data_source"] = canon[r["metadata"]["data_source"]]
        for ext in ("json", "msgpack"):
            name = f"row_{r['metadata']['row_id']:06d}.{ext}"
            shutil.copyfile(os.path.join(args.probe_record, name), os.path.join(args.out, "probe", name))
    with open(os.path.join(args.out, "probe/rows.jsonl"), "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    json.dump({"rounds": [[r["metadata"]["row_id"] for r in rows]], "segments": []},
              open(os.path.join(args.out, "probe/manifest.json"), "w"), indent=1)
    shutil.copyfile(os.path.join(args.probe_fixture, "student_lora.safetensors"),
                    os.path.join(args.out, "probe/student_init.safetensors"))

    shutil.copytree(args.replay_workload, os.path.join(args.out, "replay"))  # copies, not the source's hard links
    m = json.load(open(os.path.join(args.out, "replay/manifest.json")))
    m.pop("source_record", None)
    json.dump(m, open(os.path.join(args.out, "replay/manifest.json"), "w"), indent=1)

    shutil.copyfile(os.path.join(args.contract, "rows.jsonl"), os.path.join(args.out, "rows.jsonl"))
    shutil.copyfile(os.path.join(args.contract, "student_init.safetensors"), os.path.join(args.out, "student_init.safetensors"))
    shutil.copyfile(args.heldout, os.path.join(args.out, "heldout.jsonl"))
    kl = json.load(open(args.kl_init))
    json.dump({k: kl[k] for k in ("per_route", "kl_pooled", "samples", "case_seed", "prompts") if k in kl},
              open(os.path.join(args.out, "kl_init.json"), "w"), indent=1)
    for root, dirs, files in os.walk(args.out):  # mounted read-only into Work, read there by the agent user
        for name in dirs + files:
            path = os.path.join(root, name)
            os.chmod(path, 0o755 if os.path.isdir(path) else 0o644)
    os.chmod(args.out, 0o755)
    print(f"public root -> {args.out}: probe {len(rows)} rows, replay {len(m['rounds'])} rounds")


if __name__ == "__main__":
    main()
