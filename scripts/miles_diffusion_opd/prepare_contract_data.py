"""Build contract prompt rows and the student LoRA initial state for the SD3.5-M DiffusionOPD task.

Rows: jsonl ``{"input": prompt, "metadata": {"data_source": <label>, "row_id": k}}``, alternating
OCR / PickScore (row k is OCR when k is even), each prompt drawn in file order from its split, so every
4-row half-round holds 2 OCR + 2 PickScore rows. Labels are parameters so the Judge can randomise them.

student_init: PEFT-named LoRA (``base_model.model.<module>.lora_{A,B}.weight``, fp32) with the module set
and shapes of the official teacher adapters; A ~ N(0, (1/r)^2) (PEFT ``init_lora_weights="gaussian"``)
from a seeded CPU generator, B = 0.
"""

from __future__ import annotations

import argparse
import json
import os

import torch
from safetensors.torch import load_file, save_file


# MDOPD_ASSETS: SD3.5-M assets root (scripts/miles_common/host_config.py; default <MILES_DATA_ROOT>/models/miles-diffusion-assets).
ASSETS = os.environ.get("MDOPD_ASSETS")


def read_prompts(path: str) -> list[str]:
    return [json.loads(line)["input"].strip() for line in open(path) if line.strip()]


def build_rows(ocr: list[str], pick: list[str], n_rows: int, ocr_offset: int, pick_offset: int,
               labels: tuple[str, str] = ("ocr", "pickscore"), row_id_base: int = 0, seed_base: int = 42) -> list[dict]:
    rows = []
    for k in range(n_rows):
        if k % 2 == 0:
            prompt, label = ocr[ocr_offset + k // 2], labels[0]
        else:
            prompt, label = pick[pick_offset + k // 2], labels[1]
        rid = row_id_base + k
        rows.append({"input": prompt, "metadata": {"data_source": label, "row_id": rid, "seed": seed_base + 8 * rid}})
    return rows


def student_init(template_adapter: str, seed: int, r: int = 32) -> dict[str, torch.Tensor]:
    template = load_file(template_adapter)
    g = torch.Generator(device="cpu").manual_seed(seed)
    out = {}
    for key in sorted(template):
        shape = template[key].shape
        if key.endswith(".lora_A.weight"):
            out[key] = torch.randn(shape, generator=g, dtype=torch.float32) * (1.0 / r)
        elif key.endswith(".lora_B.weight"):
            out[key] = torch.zeros(shape, dtype=torch.float32)
        else:
            raise ValueError(key)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=not ASSETS,
                    default=ASSETS and f"{ASSETS}/datasets/miles-diffusion-datasets-0e8958fa")
    ap.add_argument("--teacher-adapter", required=not ASSETS,
                    default=ASSETS and f"{ASSETS}/teachers/DiffusionOPD-38686355/OCRTeacher/lora/adapter_model.safetensors")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", type=int, default=512)
    ap.add_argument("--ocr-offset", type=int, default=0)
    ap.add_argument("--pick-offset", type=int, default=0)
    ap.add_argument("--split", default="train")
    ap.add_argument("--init-seed", type=int, default=20260923)
    ap.add_argument("--row-id-base", type=int, default=0, help="public and private row ids (hence seeds) are disjoint")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    ocr = read_prompts(f"{args.data}/flowgrpo_ocr/{args.split}.jsonl")
    pick = read_prompts(f"{args.data}/flowgrpo_pickscore/{args.split}.jsonl")
    rows = build_rows(ocr, pick, args.rows, args.ocr_offset, args.pick_offset, row_id_base=args.row_id_base)
    with open(os.path.join(args.out, "rows.jsonl"), "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    save_file(student_init(args.teacher_adapter, args.init_seed), os.path.join(args.out, "student_init.safetensors"))
    json.dump(vars(args), open(os.path.join(args.out, "provenance.json"), "w"), indent=1)
    print(f"{len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
