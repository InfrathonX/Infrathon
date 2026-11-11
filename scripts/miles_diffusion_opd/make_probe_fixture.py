"""Build a probe fixture for the SD3.5-M DiffusionOPD task (public or private flavour).

A fixture = 4 contract rows (2 per teacher) whose window starts cover 0, 1 and 2 (start 0 means sigma = 1.0), one
of them a long prompt past both the CLIP 77- and T5 256-token limits, route labels drawn at random (the candidate
gets the mapping only through ``--teacher-model LABEL=PATH``), and a nonzero student LoRA far from both teachers
(A ~ N(0, 1/r^2); B random with, per module, the Frobenius norm of the OCR teacher's B) so that "teacher leaks
into the student adapter" and "base DiT as teacher" are O(1) errors. The engine responses for these rows are
recorded afterwards from a real run started from this LoRA (record_generate), and the Judge replays them.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import string

import numpy as np
import torch
from safetensors.torch import load_file, save_file


# MDOPD_ASSETS: SD3.5-M assets root (scripts/miles_common/host_config.py; default <MILES_DATA_ROOT>/models/miles-diffusion-assets).
ASSETS = os.environ.get("MDOPD_ASSETS")


def window_start(seed: int) -> int:
    return int(np.random.default_rng(seed).integers(0, 3))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=not ASSETS,
                    default=ASSETS and f"{ASSETS}/datasets/miles-diffusion-datasets-0e8958fa")
    ap.add_argument("--teachers", required=not ASSETS, default=ASSETS and f"{ASSETS}/teachers/DiffusionOPD-38686355")
    ap.add_argument("--split", default="test")
    ap.add_argument("--offset", type=int, default=0, help="prompt offset in each split (public/private disjoint)")
    ap.add_argument("--rollout-seed", type=int, default=42)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--rows", type=int, default=8, help="probe round size (4 OCR + 4 Aes)")
    ap.add_argument("--row-id-base", type=int, default=900000, help="keeps probe row ids apart from contract rows")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    os.makedirs(args.out, exist_ok=True)

    def prompts(sub):
        return [json.loads(line)["input"].strip() for line in open(f"{args.data}/{sub}/{args.split}.jsonl")]

    ocr, pick = prompts("flowgrpo_ocr")[args.offset:], prompts("flowgrpo_pickscore")[args.offset:]
    labels = ["".join(rng.choice(string.ascii_lowercase) for _ in range(6)) for _ in range(2)]
    long_prompt = " ".join(ocr[10:16]) + " " + " ".join(pick[10:14])
    rows_src = [(ocr[0], labels[0]), (pick[0], labels[1]), (long_prompt, labels[0]), (pick[1], labels[1])]
    for k in range(args.rows - 4):  # the rest of the probe round alternates teachers too
        rows_src.append((ocr[1 + k // 2], labels[0]) if k % 2 == 0 else (pick[2 + k // 2], labels[1]))
    # MD seeds a row with rollout_seed + group_index * n; the probe rows are group indices 0..3. Choose prompts'
    # order is fixed; check the window starts cover {0, 1, 2}; otherwise shift the rollout seed deterministically.
    rollout_seed = args.rollout_seed
    # the first 4 rows form optimizer step 0 (the probed step): their window starts must cover {0, 1, 2}, row 0 at 0
    while {window_start(rollout_seed + g * args.n) for g in range(4)} != {0, 1, 2} or window_start(rollout_seed) != 0:
        rollout_seed += 1
    rows = [{"input": p, "metadata": {"data_source": lab, "row_id": args.row_id_base + g, "seed": rollout_seed + g * args.n}}
            for g, (p, lab) in enumerate(rows_src)]
    with open(os.path.join(args.out, "rows.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    ocr_b = load_file(f"{args.teachers}/OCRTeacher/lora/adapter_model.safetensors")
    g = torch.Generator().manual_seed(args.seed)
    lora = {}
    for key in sorted(ocr_b):
        if key.endswith("lora_A.weight"):
            lora[key] = torch.randn(ocr_b[key].shape, generator=g) / 32.0
        else:
            b = torch.randn(ocr_b[key].shape, generator=g)
            lora[key] = b * (ocr_b[key].norm() / b.norm())
    save_file(lora, os.path.join(args.out, "student_lora.safetensors"))
    save_file(lora, os.path.join(args.out, "student_init.safetensors"))  # same bytes; the name the launcher mounts
    meta = {"labels": {"ocr": labels[0], "aes": labels[1]}, "rollout_seed": rollout_seed, "n": args.n,
            "window_starts": [window_start(rollout_seed + gi * args.n) for gi in range(args.rows)],
            "long_prompt_words": len(long_prompt.split()), **vars(args)}
    json.dump(meta, open(os.path.join(args.out, "fixture.json"), "w"), indent=1)
    print(json.dumps({k: meta[k] for k in ("labels", "rollout_seed", "window_starts", "long_prompt_words")}))


if __name__ == "__main__":
    main()
