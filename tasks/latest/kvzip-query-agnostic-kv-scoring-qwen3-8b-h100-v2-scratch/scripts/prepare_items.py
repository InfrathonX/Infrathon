#!/usr/bin/env python3
"""Build the KVzip v2 public rows and the frozen final-holdout manifest.

The holdout follows the upstream KVzip evaluation protocol
(snu-mllab/KVzip @ 5d84729, ``eval.py --num 100`` with ``data/load.py``) at
the scale agreed for v2, with SCBench's ``_short`` variants because Qwen3-8B
has a 40,960-token window (upstream does the same for Gemma3-12B):

======================  ====================================================  =====
group                   rows                                                  count
======================  ====================================================  =====
retrieval               NIAH (KVzip generator, 3 private needles x 3 lengths    613
                        x 11 depths), scbench_kv_short (100 upstream + 50
                        generated), scbench_prefix_suffix_short (100 + 50),
                        scbench_repoqa_short (22), RULER niah x 8 tasks x
                        8k/16k x 12
contextual QA           SQuAD (first 250 train contexts), GSM8K (first 200      602
                        eligible test rows), RULER qa_1/qa_2 x 8k/16k x 38
high redundancy         scbench_mf_short (100 + 150 generated),                 600
                        scbench_many_shot (54), scbench_summary_short (62),
                        RULER vt x 8k/16k x 117
======================  ====================================================  =====

Public rows never share a source row with the holdout: SQuAD validation,
GSM8K train, NIAH with the upstream needle over the other half of the essay
files, freshly generated SCBench-style kv / prefix_suffix / mf rows, and other
RULER rows.  Generated rows follow the upstream formats byte for byte and are
seeded, so they rebuild identically.  ICL.ManyShot, En.Summary and
Code.RepoQA have too few upstream rows to split and stay holdout-only.

``build`` writes ``corpus/public/items_public.jsonl`` (+ MANIFEST.json) and
``judge/private/final_manifest.json``; ``materialize`` rebuilds
``final_items.jsonl.gz`` from the pinned sources and checks every row's
SHA-256; ``verify`` rebuilds in memory and compares bytes.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import random
import sys
import uuid
from pathlib import Path
from typing import Any, Iterable, Mapping

TASK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK_ROOT / "runtime"))
import kv_scoring_runtime as rt  # noqa: E402

SUITE_SEED = "kvzip-scoring-v2"
PINS = {
    "rajpurkar/squad": "7b6d24c440a36b6815f21b70d25016731768db1f",
    "openai/gsm8k": "740312add88f781978c0658806c59bc2815b9866",
    "Jang-Hyun/SCBench-preprocessed": "a079be919d3131822c202180ffd4dc322968de45",
    "simonjegou/ruler": "24adceac8a0e6532936e8d721cd9e9084d2e4686",
}
KVZIP_COMMIT = "5d8472975d286e3da6b52178af39d18904330c0d"
ESSAY_DIR = "data/needle/PaulGrahamEssays"
NIAH_LENGTHS = (500, 2000, 8000)
NIAH_DEPTHS = tuple(range(0, 101, 10))
UPSTREAM_NEEDLE = {"needle": "\nThe best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\n",
                   "question": "Based on the content of the book, the best thing to do in San Francisco is: ",
                   "answer": "Eat a sandwich and sit in Dolores Park on a sunny day."}
HOLDOUT = {"squad": 250, "gsm": 200, "kv_generated": 50, "prefix_suffix_generated": 50, "mf_generated": 150,
           "ruler_niah": 12, "ruler_qa": 38, "ruler_vt": 117}
PUBLIC = {"squad": 10, "gsm": 10, "kv": 8, "prefix_suffix": 8, "mf": 8}
PUBLIC_NIAH = ((500, 0), (500, 50), (500, 100), (2000, 10), (2000, 40), (2000, 70), (2000, 90), (8000, 20), (8000, 50), (8000, 80), (8000, 100))
PUBLIC_RULER = (("niah_single_1", "8192"), ("niah_multikey_2", "8192"), ("niah_multivalue", "8192"), ("niah_multiquery", "16384"),
                ("qa_1", "8192"), ("qa_2", "16384"), ("vt", "8192"), ("vt", "16384"))
RULER_KIND = {"niah_single_1": "ruler_niah", "niah_single_2": "ruler_niah", "niah_single_3": "ruler_niah", "niah_multikey_1": "ruler_niah",
              "niah_multikey_2": "ruler_niah", "niah_multikey_3": "ruler_niah", "niah_multivalue": "ruler_niah", "niah_multiquery": "ruler_niah",
              "qa_1": "ruler_qa", "qa_2": "ruler_qa", "vt": "ruler_vt"}
PREFIX_SUFFIX_HEADER = ("Given a list of words called 'dictionary', return the word in the dictionary that has both the prefix 'pref' and the suffix "
                        "'suff'. If there is more than one valid answer, return the latest of them. \n\nFor example:\ndictionary = ")
PREFIX_SUFFIX_ALPHABET = "+-ABCDEFGHIJKLMNOPQRSTUVWXYZ^_abcdefghijklmnopqrstuvwxyz"
MF_QUESTIONS = (("largest number of the list", lambda v: True, max), ("smallest number of the list", lambda v: True, min),
                ("largest number with 7 in it", lambda v: "7" in str(v), max), ("smallest number with 7 in it", lambda v: "7" in str(v), min),
                ("largest number with 3 at the end", lambda v: str(v).endswith("3"), max), ("smallest number with 3 at the end", lambda v: str(v).endswith("3"), min))


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _rng(*labels: str) -> random.Random:
    return random.Random(int.from_bytes(hashlib.sha256("|".join((SUITE_SEED, *labels)).encode()).digest()[:8], "big"))


def _snapshot(hf_hub: Path, repo: str) -> Path:
    path = hf_hub / f"datasets--{repo.replace('/', '--')}" / "snapshots" / PINS[repo]
    if not path.is_dir(): raise SystemExit(f"missing pinned snapshot {repo}@{PINS[repo]} under {hf_hub}")
    return path


def _parquet(path: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq
    return pq.read_table(path).to_pylist()


def _row(task: str, context: str, questions: list[str], answers: list[Any], meta: Mapping[str, Any], tok: Any) -> dict[str, Any]:
    return {"task": task, "context": context, "questions": list(questions), "answers": list(answers),
            "meta": {**meta, "context_tokens": len(tok.encode(context, add_special_tokens=False)), "question_count": len(questions)}}


# ------------------------------------------------------------------ upstream rows

def squad_rows(hf_hub: Path, split: str, n: int, tok: Any) -> list[dict[str, Any]]:
    """KVzip load_squad: group questions by context, keep the first n contexts of at least MIN_CTX_TOKENS tokens."""
    root = _snapshot(hf_hub, "rajpurkar/squad"); rel = f"plain_text/{split}-00000-of-00001.parquet"
    groups: dict[str, dict[str, Any]] = {}; skipped: set[str] = set()
    for index, row in enumerate(_parquet(root / rel)):
        if row["context"] in skipped: continue
        group = groups.get(row["context"])
        if group is None:
            if len(tok.encode(row["context"], add_special_tokens=False)) < rt.MIN_CTX_TOKENS: skipped.add(row["context"]); continue
            if len(groups) == n: break
            group = groups[row["context"]] = {"questions": [], "answers": [], "first_row": index}
        group["questions"].append(row["question"]); group["answers"].append(row["answers"]["text"][0])
    return [_row("squad", ctx, g["questions"], g["answers"], {"source": {"repo": "rajpurkar/squad", "file": rel, "row": g["first_row"]}}, tok) for ctx, g in groups.items()]


def gsm_rows(hf_hub: Path, split: str, tok: Any, limit: int | None = None) -> list[dict[str, Any]]:
    """KVzip load_gsm: drop the last sentence into the question; skip contexts under 72 tokens."""
    root = _snapshot(hf_hub, "openai/gsm8k"); rel = f"main/{split}-00000-of-00001.parquet"; out = []
    for index, row in enumerate(_parquet(root / rel)):
        sentences = row["question"].split(". "); context = ". ".join(sentences[:-1]).strip() + "."
        if len(tok.encode(context, add_special_tokens=False)) < 72: continue
        out.append(_row("gsm", context, [sentences[-1].strip()], [row["answer"]], {"source": {"repo": "openai/gsm8k", "file": rel, "row": index}}, tok))
        if limit is not None and len(out) == limit: break
    return out


def scbench_rows(hf_hub: Path, name: str, task: str, tok: Any) -> list[dict[str, Any]]:
    """KVzip load_scbench: prompts[0] is the context, prompts[1:] the questions; list answers are comma-joined."""
    root = _snapshot(hf_hub, "Jang-Hyun/SCBench-preprocessed"); rel = f"scbench_{name}.parquet"; out = []
    for index, row in enumerate(_parquet(root / rel)):
        answers = [", ".join(gt) if isinstance(gt, list) else str(gt) for gt in row["ground_truth"]]
        meta: dict[str, Any] = {"source": {"repo": "Jang-Hyun/SCBench-preprocessed", "file": rel, "row": index}}
        if task == "scbench_repoqa":
            meta["repoqa"] = {"lang": row["lang"], "repo": row["repo"], "func_names": list(row["func_name"]), "needles": list(row["ground_truth"])}
            answers = list(row["func_name"])
        out.append(_row(task, row["prompts"][0], list(row["prompts"][1:]), answers, meta, tok))
    return out


def ruler_cells(hf_hub: Path, tok: Any) -> dict[tuple[str, str], list[dict[str, Any]]]:
    root = _snapshot(hf_hub, "simonjegou/ruler"); cells: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for length in ("8192", "16384"):
        rel = f"{length}/test-00000-of-00001.parquet"
        for index, row in enumerate(_parquet(root / rel)):
            kind = RULER_KIND.get(row["task"])
            if kind is None: continue
            cells[(row["task"], length)].append(_row(kind, row["context"], [row["question"].strip()], [list(row["answer"])],
                {"source": {"repo": "simonjegou/ruler", "file": rel, "row": index}, "ruler_task": row["task"], "answer_prefix": row["answer_prefix"]}, tok))
    return cells


# ------------------------------------------------------------------ generated rows

def _uuid(rng: random.Random) -> str:
    return str(uuid.UUID(int=rng.getrandbits(128), version=4))


def kv_row(rng: random.Random, tok: Any, label: str) -> dict[str, Any]:
    """SCBench Retr.KV short format: a JSON object of UUID pairs, 10 keys asked."""
    n_pairs = rng.randint(361, 365); data = {}
    while len(data) < n_pairs: data[_uuid(rng)] = _uuid(rng)
    keys = rng.sample(list(data), 10)
    questions = [f'\nKey: "{k}"\nThe value associated with the specified key is: ' for k in keys]
    return _row("scbench_kv", "JSON data:\n" + json.dumps(data), questions, [data[k] for k in keys], {"generated": label}, tok)


def prefix_suffix_row(rng: random.Random, tok: Any, label: str) -> dict[str, Any]:
    """SCBench Retr.PrefixSuffix short format: ~1,007 random 25-character words, 5 unique prefix/suffix queries."""
    n_words = rng.randint(1004, 1010); words = ["".join(rng.choice(PREFIX_SUFFIX_ALPHABET) for _ in range(25)) for _ in range(n_words)]
    questions, answers, chosen = [], [], set()
    while len(questions) < 5:
        target = rng.randrange(n_words)
        if target in chosen: continue
        word = words[target]; pre, suf = word[:5], word[-5:]
        if sum(w.startswith(pre) and w.endswith(suf) for w in words) != 1: continue
        chosen.add(target)
        questions.append(f'Given prefix = "{pre}"\nsuffix = "{suf}"\n\nWhat is the word in the dictionary that has both the prefix and the suffix?\n'); answers.append(word)
    return _row("scbench_prefix_suffix", PREFIX_SUFFIX_HEADER + repr(words) + "\n", questions, answers, {"generated": label}, tok)


def mf_row(rng: random.Random, tok: Any, label: str) -> dict[str, Any]:
    """SCBench Math.Find short format: ~6,000 integers from a random range, six questions in random order."""
    lo = rng.randint(10, 60); hi = rng.randint(lo + 150, 999); nums = [rng.randint(lo, hi) for _ in range(rng.randint(6002, 6008))]
    order = list(range(len(MF_QUESTIONS))); rng.shuffle(order); questions, answers = [], []
    for i in order:
        text, keep, agg = MF_QUESTIONS[i]; pool = [v for v in nums if keep(v)]
        if not pool: return mf_row(rng, tok, label)
        questions.append(f"What is the {text}?\n\nYou should answer with only one number, no other words. The {text} is: "); answers.append(str(agg(pool)))
    return _row("scbench_mf", repr(nums), questions, answers, {"generated": label}, tok)


# ------------------------------------------------------------------ NIAH

def essay_files(kvzip_repo: Path) -> list[Path]:
    files = sorted((kvzip_repo / ESSAY_DIR).glob("*.txt"))
    if len(files) != 49: raise SystemExit(f"expected 49 essay files at KVzip {KVZIP_COMMIT[:7]}, found {len(files)}")
    return files


def essay_split(files: list[Path]) -> tuple[list[Path], list[Path]]:
    public = [f for f in files if int(_sha256(f.name.encode()), 16) % 2 == 0]
    return public, [f for f in files if f not in public]


class Needle:
    """Port of KVzip data/needle/data.py (MIT) with a sorted, explicit file list."""

    def __init__(self, tok: Any, files: list[Path]) -> None:
        self.tok = tok; self.files = files
        self.period = set(tok.encode(".", add_special_tokens=False) + tok.encode(".\n", add_special_tokens=False))
        self.haystack = ""
        while len(tok.encode(self.haystack)) < max(NIAH_LENGTHS):
            for f in files: self.haystack += f.read_text(encoding="utf-8")
        self.haystack_ids = tok.encode(self.haystack, add_special_tokens=False); self.trimmed: dict[int, list[int]] = {}

    def context(self, needle: str, length: int, depth: int) -> str:
        if length not in self.trimmed:  # encode_and_trim: decode the first `length` tokens and re-encode
            ids = self.haystack_ids
            self.trimmed[length] = self.tok.encode(self.tok.decode(ids[:length]), add_special_tokens=False) if len(ids) > length else list(ids)
        tokens = list(self.trimmed[length])
        needle_ids = self.tok.encode(needle, add_special_tokens=False)
        if len(tokens) + len(needle_ids) > length: tokens = tokens[:length - len(needle_ids)]
        if depth == 100: new = tokens + needle_ids
        else:
            point = int(len(tokens) * depth / 100); before = tokens[:point]
            while before and before[-1] not in self.period: point -= 1; before = tokens[:point]
            new = before + needle_ids + tokens[point:]
        return self.tok.decode(new).strip()


def niah_rows(tok: Any, files: list[Path], needles: list[Mapping[str, str]], grid: Iterable[tuple[int, int]], label: str) -> list[dict[str, Any]]:
    """One row per (needle, length, depth); a depth that lands on the same sentence
    boundary as an earlier depth (upstream's insertion backs up to the previous
    period) yields an identical row and is dropped."""
    gen = Needle(tok, files); out = []; seen: set[str] = set()
    for n_index, needle in enumerate(needles):
        for length, depth in grid:
            context = gen.context(needle["needle"], length, depth)
            if context in seen: continue
            seen.add(context)
            out.append(_row("needle", context, [needle["question"]], [needle["answer"]],
                            {"generated": label, "niah": {"length": length, "depth": depth, "needle": n_index}}, tok))
    return out


# ------------------------------------------------------------------ assembly

def _take(pool: list[dict[str, Any]], n: int, *labels: str) -> list[dict[str, Any]]:
    if n > len(pool): raise SystemExit(f"cell {labels} needs {n}, has {len(pool)}")
    order = list(range(len(pool))); _rng(*labels).shuffle(order)
    return [pool[i] for i in sorted(order[:n])]


def build(hf_hub: Path, model_dir: Path, kvzip_repo: Path, private_needles: Path) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, local_files_only=True)
    needles = json.loads(private_needles.read_text(encoding="utf-8"))
    if len(needles) != 3: raise SystemExit("the holdout needs exactly three private needles")
    public_files, holdout_files = essay_split(essay_files(kvzip_repo))
    public: list[dict[str, Any]] = []; final: list[dict[str, Any]] = []
    # contextual QA
    public += squad_rows(hf_hub, "validation", PUBLIC["squad"], tok); final += squad_rows(hf_hub, "train", HOLDOUT["squad"], tok)
    # GSM8K: only 148 test rows pass the 72-token context rule, so the holdout
    # takes all of them (the upstream protocol's first 100 included) and then
    # the eligible train rows right after the public ones.
    gsm_test = gsm_rows(hf_hub, "test", tok); gsm_train = gsm_rows(hf_hub, "train", tok, PUBLIC["gsm"] + HOLDOUT["gsm"])
    public += gsm_train[:PUBLIC["gsm"]]
    final += (gsm_test + gsm_train[PUBLIC["gsm"]:])[:HOLDOUT["gsm"]]
    # retrieval / redundancy upstream SCBench rows (holdout only)
    for name, task in (("kv_short", "scbench_kv"), ("prefix_suffix_short", "scbench_prefix_suffix"), ("repoqa_short", "scbench_repoqa"),
                       ("mf_short", "scbench_mf"), ("many_shot", "scbench_many_shot"), ("summary_short", "scbench_summary")):
        final += [r for r in scbench_rows(hf_hub, name, task, tok) if r["context"].strip()]
    # generated SCBench-style rows
    for side, plan in (("public", PUBLIC), ("final", HOLDOUT)):
        for kind, maker in (("kv", kv_row), ("prefix_suffix", prefix_suffix_row), ("mf", mf_row)):
            n = plan.get(kind) if side == "public" else plan.get(f"{kind}_generated"); rng = _rng(side, kind)
            rows = [maker(rng, tok, f"{side}-{kind}-{i:03d}") for i in range(n)]
            (public if side == "public" else final).extend(rows)
    # NIAH
    public += niah_rows(tok, public_files, [UPSTREAM_NEEDLE], PUBLIC_NIAH, "public-niah")
    final += niah_rows(tok, holdout_files, needles, [(l, d) for l in NIAH_LENGTHS for d in NIAH_DEPTHS], "final-niah")
    # RULER
    cells = ruler_cells(hf_hub, tok); used: dict[tuple[str, str], set[int]] = collections.defaultdict(set)
    for task, length in PUBLIC_RULER:
        row = _take(cells[(task, length)], 1, "public", "ruler", task, length)[0]; public.append(row); used[(task, length)].add(row["meta"]["source"]["row"])
    for (task, length), pool in sorted(cells.items()):
        rest = [r for r in pool if r["meta"]["source"]["row"] not in used[(task, length)]]
        final += _take(rest, HOLDOUT[RULER_KIND[task]], "final", "ruler", task, length)
    # checks and identities
    for row in public + final: rt.validate_item({"item_id": "x", **row}, tok)
    public_contexts = {rt.context_sha256(r["context"]) for r in public}
    if public_contexts & {rt.context_sha256(r["context"]) for r in final}: raise SystemExit("public and holdout share a context")
    _rng("final", "order").shuffle(final)
    public_items = [{"item_id": f"public-{i:03d}-{r['task']}", **r} for i, r in enumerate(public)]
    final_items, entries = [], []
    for i, r in enumerate(final):
        item = {"item_id": f"f{i:04d}", **{k: v for k, v in r.items() if k != "meta"}, "meta": {k: v for k, v in r["meta"].items() if k not in ("source", "generated", "niah", "ruler_task")}}
        final_items.append(item)
        entries.append({"item_id": item["item_id"], "task": r["task"], "sha256": rt.item_sha256(item), "context_tokens": r["meta"]["context_tokens"],
                        "origin": r["meta"].get("source") or {"generated": r["meta"]["generated"], **({"niah": r["meta"]["niah"]} if "niah" in r["meta"] else {})},
                        **({"ruler_task": r["meta"]["ruler_task"]} if "ruler_task" in r["meta"] else {})})
    counts = collections.Counter(e["task"] for e in entries)
    manifest = {"schema_version": 1, "suite_id": rt.FINAL_SUITE_ID, "generator": "kvzip-v2-prepare-items", "seed": SUITE_SEED, "pins": PINS,
                "kvzip_commit": KVZIP_COMMIT, "private_needles_sha256": _sha256(private_needles.read_bytes()), "row_count": len(entries),
                "question_count": sum(len(x["questions"]) for x in final_items), "counts_per_task": dict(sorted(counts.items())), "items": entries}
    return public_items, manifest, final_items


def _jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return "".join(json.dumps(r, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for r in rows).encode()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("build", "verify", "materialize"))
    ap.add_argument("--hf-hub", type=Path, required=True); ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--kvzip-repo", type=Path, required=True, help=f"snu-mllab/KVzip checkout at {KVZIP_COMMIT[:7]} (NIAH haystack)")
    ap.add_argument("--private-needles", type=Path, default=TASK_ROOT / "judge/private/needle_private.json")
    ap.add_argument("--out", type=Path, help="materialize: output final_items.jsonl.gz")
    ap.add_argument("--manifest", type=Path, help="materialize: the final manifest (default judge/private/final_manifest.json)")
    args = ap.parse_args()
    public_path = TASK_ROOT / "corpus/public/items_public.jsonl"; manifest_path = TASK_ROOT / "judge/private/final_manifest.json"
    public_items, manifest, final_items = build(args.hf_hub, args.model_dir, args.kvzip_repo, args.private_needles)
    public_bytes = _jsonl_bytes(public_items); manifest_bytes = (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=1) + "\n").encode()
    if args.command == "verify":
        problems = [str(p) for p, b in ((public_path, public_bytes), (manifest_path, manifest_bytes)) if not p.is_file() or p.read_bytes() != b]
        if problems: raise SystemExit("rebuild differs from committed files: " + ", ".join(problems))
        print("verify: public rows and final manifest rebuild byte-for-byte"); return 0
    if args.command == "materialize":
        if args.out is None: raise SystemExit("materialize needs --out")
        committed = args.manifest or manifest_path
        if not committed.is_file() or committed.read_bytes() != manifest_bytes: raise SystemExit("committed final manifest differs from the rebuild")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with gzip.GzipFile(args.out, "wb", mtime=0) as handle: handle.write(_jsonl_bytes(final_items))
        print(json.dumps({"rows": len(final_items), "sha256": _sha256(args.out.read_bytes())})); return 0
    public_path.parent.mkdir(parents=True, exist_ok=True); public_path.write_bytes(public_bytes)
    (public_path.parent / "MANIFEST.json").write_text(json.dumps({"schema_version": 2, "files": {public_path.name: _sha256(public_bytes)}}, indent=2, sort_keys=True) + "\n")
    manifest_path.write_bytes(manifest_bytes)
    print(json.dumps({"public_rows": len(public_items), "final_rows": manifest["row_count"], "final_questions": manifest["question_count"],
                      "counts_per_task": manifest["counts_per_task"], "public_sha256": _sha256(public_bytes), "manifest_sha256": _sha256(manifest_bytes)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
