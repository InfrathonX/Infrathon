#!/usr/bin/env python3
"""Build the public and private data assets of the Miles OPCD task v3 from MedMCQA (run in the task image).

Every question is a MedMCQA training row (openlifescienceai/medmcqa @ MEDMCQA_REVISION, Apache-2.0; the
MedMCQA GitHub release is MIT) rendered with the LMOps OPCD prompt template (TEMPLATE). medmcqa_row_map.json
fixes which rows, in which order and split: the same questions, byte for byte, as the v2 assets, which were
cut from the unlicensed ytz20/sys_medmcqa_* tables (derive_row_map.py). The context c is LMOps'
opcd/system_prompts/medmcqa.txt (MIT, from MetaSPO).

work-public/  the data in the Work image
  train_public.jsonl   512 rows: public/train_public.jsonl and the medmcqa_work_rows.json rows; the perf suite uses
                       them all (8 rounds x 64), the small suite the first 64 (4 rounds x 16)
  test_public.jsonl, context.txt, probe_batch.json: as in public/
public/   the data in the Judge image (/data/public); none of its files but those three reach Work
  train.jsonl          16384 training rows; round r uses lines [512r, 512r+512)
  train_public.jsonl   the first 64 rows of work-public/train_public.jsonl
  test_public.jsonl    the first 64 questions of the paper's 500-question test set (indicative accuracy only)
  context.txt          the system prompt c, given to the teacher only
  probe_batch.json     8 fixed trajectories for the public one-step probe
private/  Judge only
  eval.jsonl           the 2436 held-out scoring questions (deduplicated against every question visible in Work)
  paper_test.jsonl     the paper's 500-question test set, for the report-only paper-protocol diagnostic
  probe_batch.json     8 fixed trajectories from hidden rows for the Judge probe
Each directory gets manifest.json with the source revision, row provenance and the SHA256 of every file.

The probe batches are the v2 ones (student samples; the teacher plays no part in them), carried over with
--probe-from after checking that each trajectory's prompt is exactly the rendering of its mapped row.
"""

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

MEDMCQA_REVISION = "91c6572c454088bf71b679ad90aa8dffcd0d5868"
TRAIN_PARQUET = "data/train-00000-of-00001.parquet"
TRAIN_PARQUET_SHA256 = "b119434ba551517a6ec0ba1f7e0b4c029165ed284a4704f262ce37c791c493c5"
TEMPLATE = ("Given the following question and candidate answers, choose the best answer. <Question>{q}\nChoices:\n"
            "A. {a}\nB. {b}\nC. {c}\nD. {d}\n</Question>\n"
            "At the end present your answer in <answer> and </answer> using an uppercase letter (A, B, C, or D).")


def render(r):
    """(user message, answer letter) of a MedMCQA row."""
    return TEMPLATE.format(q=r["question"], a=r["opa"], b=r["opb"], c=r["opc"], d=r["opd"]), "ABCD"[r["cop"]]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def question_key(text):
    return " ".join(text.split()).lower()


def stem_key(row):
    """A MedMCQA row's question stem without case, punctuation or layout: near-duplicates share it."""
    return " ".join(re.sub(r"[\W_]+", " ", row["question"].lower()).split())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--medmcqa", type=Path, required=True, help=f"snapshot dir of openlifescienceai/medmcqa @ {MEDMCQA_REVISION}")
    ap.add_argument("--row-map", type=Path, default=Path(__file__).with_name("medmcqa_row_map.json"))
    ap.add_argument("--work-rows", type=Path, default=Path(__file__).with_name("medmcqa_work_rows.json"))
    ap.add_argument("--context", type=Path, required=True, help="LMOps opcd/system_prompts/medmcqa.txt")
    ap.add_argument("--probe-from", type=Path, required=True, help="the v2 asset dir (public/ and private/ probe_batch.json)")
    ap.add_argument("--tokenizer", required=True, help="the pinned Qwen3-1.7B snapshot")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    parquet = args.medmcqa / TRAIN_PARQUET
    assert sha(parquet) == TRAIN_PARQUET_SHA256, f"{parquet} is not the pinned MedMCQA revision"
    rows = pq.read_table(parquet).to_pylist()
    row_map = json.loads(args.row_map.read_text())
    assert row_map["source_sha256"] == TRAIN_PARQUET_SHA256
    pub, priv = args.out / "public", args.out / "private"
    pub.mkdir(parents=True, exist_ok=False)
    priv.mkdir(parents=True, exist_ok=False)

    def item(i):
        prompt, answer = render(rows[i])
        return {"row_id": f"medmcqa:{rows[i]['id']}", "content": [{"role": "user", "content": prompt}], "answer": answer}

    def write_jsonl(path, key):
        with open(path, "w", encoding="utf-8") as f:
            for i in row_map[key]:
                f.write(json.dumps(item(i), ensure_ascii=False) + "\n")

    for d, key in ((pub, "train"), (pub, "train_public"), (pub, "test_public"), (priv, "eval"), (priv, "paper_test")):
        write_jsonl(d / f"{key}.jsonl", key)
    (pub / "context.txt").write_bytes(args.context.read_bytes())

    # The extra Work rows share no question stem with the Judge's data. The first 64 rows, the public test questions and
    # the public probe predate the split: what they share with the Judge's rows is reported below.
    work_rows = json.loads(args.work_rows.read_text())
    assert work_rows["source_sha256"] == TRAIN_PARQUET_SHA256 and len(work_rows["added"]) == 448
    def stems(*names):
        return {stem_key(rows[i]) for k in names for i in row_map[k]}
    added = {stem_key(rows[i]) for i in work_rows["added"]}
    assert len(added) == 448 and not added & stems("train", "train_public", "test_public", "eval", "paper_test",
                                                   "probe_public", "probe_private"), "an extra Work row repeats a question"
    judge_rows = {"train": stems("train"), "eval": stems("eval"), "probe_private": stems("probe_private")}
    overlap = {f"{public}_stems_in_{judge}": len(stems(public) & shared) for public in ("train_public", "test_public", "probe_public")
               for judge, shared in judge_rows.items()}

    visible = [render(rows[i])[0] for k in ("train", "train_public", "test_public") for i in row_map[k]]
    seen = {question_key(p) for p in visible}
    hidden = [question_key(render(rows[i])[0]) for i in row_map["eval"]]
    assert len(set(hidden)) == len(hidden) and not seen & set(hidden), "eval questions repeat or are visible in Work"
    assert len(row_map["train"]) == 16384 and len(row_map["eval"]) == 2436 and len(row_map["paper_test"]) == 500

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    for d, key in ((pub, "probe_public"), (priv, "probe_private")):
        batch = json.loads((args.probe_from / d.name / "probe_batch.json").read_text())
        for s, i in zip(batch["samples"], row_map[key], strict=True):
            text = tok.apply_chat_template([{"role": "user", "content": render(rows[i])[0]}], tokenize=False,
                                           add_generation_prompt=True, enable_thinking=False)
            assert tok.encode(text, add_special_tokens=False) == s["prompt_ids"], f"{key}: probe prompt != row {i}"
            s["row_id"] = f"medmcqa:{rows[i]['id']}"
        (d / "probe_batch.json").write_text(json.dumps(batch, indent=1))

    work = args.out / "work-public"
    work.mkdir(exist_ok=False)
    for name in ("test_public.jsonl", "context.txt", "probe_batch.json"):
        (work / name).write_bytes((pub / name).read_bytes())
    with open(work / "train_public.jsonl", "w", encoding="utf-8") as f:
        for i in row_map["train_public"] + work_rows["added"]:
            f.write(json.dumps(item(i), ensure_ascii=False) + "\n")

    for d in (pub, priv, work):
        files = {p.name: {"bytes": p.stat().st_size, "sha256": sha(p)} for p in sorted(d.iterdir()) if p.name != "manifest.json"}
        (d / "manifest.json").write_text(json.dumps({
            "schema_version": 2, "source": f"openlifescienceai/medmcqa @ {MEDMCQA_REVISION} ({TRAIN_PARQUET}, "
            f"sha256 {TRAIN_PARQUET_SHA256}), Apache-2.0; rendered with the LMOps OPCD template",
            "row_map_sha256": sha(args.row_map), **({"work_rows_sha256": sha(args.work_rows)} if d == work else {}), "context": "microsoft/LMOps opcd/system_prompts/medmcqa.txt (MIT)",
            "files": files}, indent=2) + "\n")
    print(json.dumps({"public": sorted(os.listdir(pub)), "private": sorted(os.listdir(priv)),
                      "work": sorted(os.listdir(work)), "eval_rows": len(row_map["eval"]), "overlap": overlap}, indent=2))


if __name__ == "__main__":
    main()
