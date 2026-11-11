#!/usr/bin/env python3
"""Build the kvpress v2 public items and the frozen final-holdout manifest.

Every item comes from the long-context evaluation sets that kvpress itself
registers (``evaluation/evaluate_registry.py`` @ 7f359b84): RULER, LongBench,
LooGLE and Zero-SCROLLS, read from the pinned Hugging Face parquet files
below.  Items keep the upstream ``context`` / ``question`` / ``answer_prefix``
fields verbatim; the trusted runtime appends ``question + assistant prefix +
answer_prefix`` after compression, as the kvpress pipeline does.

Selection rules (all deterministic):

* licence filter: NC / NC-ND / research-only sources and sources with unclear
  licences are never read (samsum, repobench-p, multi_news, book_sum_sort,
  lcc, trec, multifieldqa_en, summ_screen_fd, space_digest, LooGLE
  shortdep_cloze); screenplay documents are dropped from LooGLE longdep_qa
  and NarrativeQA; only the AMI and ICSI meetings of QMSum are used.
* one item per distinct document: LooGLE, Qasper, QuALITY, SQuALITY and
  NarrativeQA ask several questions about one document, so rows are grouped
  by document and the lowest row of each document is kept; documents that
  recur across LongBench and Zero-SCROLLS are merged by 13-word shingles.
* source split: a corpus appears on one side only.  Public real documents
  come from LongBench hotpotqa, triviaqa, qmsum and passage_count; the
  holdout never uses HotpotQA, TriviaQA, QMSum or passage_count documents.
  RULER is on both sides with disjoint rows.
* length: holdout contexts are 6,144-28,672 Qwen3 tokens including the chat
  template; nothing is truncated.
* sampling: ``random.Random(sha256(seed-label))`` per cell, so every quota is
  reproducible; quotas are the constants below.

``build`` writes ``corpus/public/items_public.jsonl`` (+ MANIFEST.json) and
``judge/private/final_manifest.json``; ``materialize`` turns the manifest back
into ``final_items.jsonl.gz`` and checks every item's SHA-256; ``verify``
rebuilds in memory and compares bytes with the committed files.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

TASK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK_ROOT / "runtime"))
import kv_eviction_runtime as rt  # noqa: E402

SUITE_SEED = "kvpress-kv-eviction-v2"
PINS = {
    "simonjegou/ruler": "24adceac8a0e6532936e8d721cd9e9084d2e4686",
    "Xnhyacinth/LongBench": "2e9ade51ebf45d98942056c0716234f9d5d257d5",
    "simonjegou/loogle": "6e3eaf28d24ce34933170c85de36be2448d0d2a7",
    "simonjegou/zero_scrolls": "1994930942afa252f7707cb9c74253b774d7ed8d",
}
HOLDOUT_TOKENS = (6144, rt.MAX_CONTEXT_TOKENS)
PUBLIC_REAL_TOKENS = (6144, rt.MAX_CONTEXT_TOKENS)
FINAL_ITEM_COUNT = 5000
FAMILY_TARGET = 1250  # qa target; retrieval/aggregation share what generation and qa leave

RULER_TASKS = {
    "niah_single_1": "retrieval", "niah_single_2": "retrieval", "niah_single_3": "retrieval",
    "niah_multikey_1": "retrieval", "niah_multikey_2": "retrieval", "niah_multikey_3": "retrieval",
    "niah_multivalue": "retrieval", "niah_multiquery": "retrieval",
    "vt": "aggregation", "cwe": "aggregation", "fwe": "aggregation",
    "qa_1": "qa", "qa_2": "qa",
}
# (repo, parquet path inside the snapshot, row task name) -> (side, family, item task)
LONGBENCH = "Xnhyacinth/LongBench"
REAL_SOURCES: tuple[tuple[str, str, str, str, str], ...] = (
    # public side
    ("public", "qa", "longbench.hotpotqa", LONGBENCH, "hotpotqa"),
    ("public", "qa", "longbench.hotpotqa", LONGBENCH, "hotpotqa_e"),
    ("public", "qa", "longbench.triviaqa", LONGBENCH, "triviaqa"),
    ("public", "qa", "longbench.triviaqa", LONGBENCH, "triviaqa_e"),
    ("public", "generation", "longbench.qmsum", LONGBENCH, "qmsum"),
    ("public", "aggregation", "longbench.passage_count", LONGBENCH, "passage_count"),
    ("public", "aggregation", "longbench.passage_count", LONGBENCH, "passage_count_e"),
    # holdout side
    ("final", "retrieval", "longbench.passage_retrieval_en", LONGBENCH, "passage_retrieval_en"),
    ("final", "retrieval", "longbench.passage_retrieval_en", LONGBENCH, "passage_retrieval_en_e"),
    ("final", "qa", "longbench.2wikimqa", LONGBENCH, "2wikimqa"),
    ("final", "qa", "longbench.2wikimqa", LONGBENCH, "2wikimqa_e"),
    ("final", "qa", "longbench.musique", LONGBENCH, "musique"),
    ("final", "qa", "qasper", LONGBENCH, "qasper"),
    ("final", "qa", "qasper", LONGBENCH, "qasper_e"),
    ("final", "qa", "narrativeqa", LONGBENCH, "narrativeqa"),
    ("final", "generation", "gov_report", LONGBENCH, "gov_report"),
    ("final", "generation", "gov_report", LONGBENCH, "gov_report_e"),
    ("final", "qa", "loogle.shortdep_qa", "simonjegou/loogle", "shortdep_qa"),
    ("final", "qa", "loogle.longdep_qa", "simonjegou/loogle", "longdep_qa"),
    ("final", "generation", "loogle.longdep_summarization", "simonjegou/loogle", "longdep_summarization"),
    ("final", "qa", "zero_scrolls.quality", "simonjegou/zero_scrolls", "quality"),
    ("final", "qa", "qasper", "simonjegou/zero_scrolls", "qasper"),
    ("final", "qa", "narrativeqa", "simonjegou/zero_scrolls", "narrative_qa"),
    ("final", "generation", "gov_report", "simonjegou/zero_scrolls", "gov_report"),
    ("final", "generation", "zero_scrolls.squality", "simonjegou/zero_scrolls", "squality"),
)
ZERO_SCROLLS_FILES = ("data/test-00000-of-00002.parquet", "data/test-00001-of-00002.parquet")
SCREENPLAY_FILTERED = {"loogle.longdep_qa", "narrativeqa"}
PUBLIC_RULER = {"4096": 1, "8192": 2, "16384": 2}  # rows per task per length
PUBLIC_REAL_PER_TASK = 5
HOLDOUT_RULER_LENGTHS = ("8192", "16384")


# ------------------------------------------------------------------ helpers

def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _rng(*labels: str) -> random.Random:
    return random.Random(int.from_bytes(hashlib.sha256("|".join((SUITE_SEED, *labels)).encode()).digest()[:8], "big"))


def _snapshot(hf_hub: Path, repo: str) -> Path:
    path = hf_hub / f"datasets--{repo.replace('/', '--')}" / "snapshots" / PINS[repo]
    if not path.is_dir(): raise SystemExit(f"missing pinned snapshot {repo}@{PINS[repo]} under {hf_hub}")
    return path


def _parquet_rows(path: Path, columns: Iterable[str]) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq
    return pq.read_table(path, columns=list(columns)).to_pylist()


def screenplay_like(text: str) -> bool:
    return len(re.findall(r"(?m)^\s*(INT\.|EXT\.|INT/EXT|CUT TO|FADE (IN|OUT)|DISSOLVE TO)", text)) >= 3


def qmsum_domain(text: str) -> str:
    low = text.lower()
    score = {"ami": sum(low.count(w) for w in ("remote control", "industrial designer", "project manager", "user interface")),
             "icsi": sum(low.count(w) for w in ("meeting recorder", "transcri", "recogni", "digits", "smartkom", "the corpus")),
             "committee": sum(low.count(w) for w in ("committee", "minister", "chair", "assembly", "parliament", "government"))}
    return max(("ami", "icsi", "committee"), key=lambda k: score[k])


_WORD = re.compile(r"\w+")


def shingles(text: str, every: int = 25, width: int = 13) -> set[int]:
    """Sampled 13-word shingles under a stable hash (Python's hash() is salted per process)."""
    words = _WORD.findall(text.lower())
    return {int.from_bytes(hashlib.blake2b(" ".join(words[i:i + width]).encode(), digest_size=8).digest(), "big")
            for i in range(0, max(0, len(words) - width + 1), every)}


class Tokens:
    def __init__(self, model_dir: Path) -> None:
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)

    def context_tokens(self, contexts: list[str]) -> list[int]:
        rendered = []
        for context in contexts:
            sep = "#" * (len(context) + 10)
            text = self.tokenizer.apply_chat_template([{"role": "user", "content": context + sep}], add_generation_prompt=True, tokenize=False, enable_thinking=False)
            rendered.append(text.split(sep)[0])
        return [len(x) for x in self.tokenizer(rendered, add_special_tokens=False)["input_ids"]]


# ---------------------------------------------------------------- candidates

def load_ruler(hf_hub: Path, tokens: Tokens) -> dict[tuple[str, str], list[dict[str, Any]]]:
    root = _snapshot(hf_hub, "simonjegou/ruler"); cells: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for length in ("4096", "8192", "16384"):
        rel = f"{length}/test-00000-of-00001.parquet"
        rows = _parquet_rows(root / rel, ("context", "question", "answer_prefix", "task"))
        counts = tokens.context_tokens([r["context"] for r in rows])
        for index, (row, n) in enumerate(zip(rows, counts)):
            if row["task"] not in RULER_TASKS: raise SystemExit(f"unexpected RULER task {row['task']}")
            cells[(row["task"], length)].append({"source": {"repo": "simonjegou/ruler", "file": rel, "row": index}, "family": RULER_TASKS[row["task"]],
                "task": f"ruler.{row['task']}.{length}", "context": row["context"], "question": row["question"], "answer_prefix": row["answer_prefix"] or "", "tokens": n})
    return cells


def _real_rows(hf_hub: Path, repo: str, name: str) -> list[tuple[str, int, dict[str, Any]]]:
    root = _snapshot(hf_hub, repo); cols = ("context", "question", "answer_prefix", "task")
    if repo == "simonjegou/zero_scrolls":
        out = []
        for rel in ZERO_SCROLLS_FILES:
            out.extend((rel, i, r) for i, r in enumerate(_parquet_rows(root / rel, cols)) if r["task"] == name)
        return out
    rel = f"{name}/test-00000-of-00001.parquet"
    return [(rel, i, r) for i, r in enumerate(_parquet_rows(root / rel, cols))]


def load_real(hf_hub: Path, tokens: Tokens) -> list[dict[str, Any]]:
    """One candidate per distinct document, licence and screenplay filters applied."""
    out = []
    for side, family, corpus, repo, name in REAL_SOURCES:
        rows = _real_rows(hf_hub, repo, name); seen: set[str] = set(); kept = []
        for rel, index, row in rows:
            digest = _sha256_bytes(row["context"].encode())
            if digest in seen: continue  # several questions per document: keep the first row
            seen.add(digest)
            if corpus in SCREENPLAY_FILTERED and screenplay_like(row["context"]): continue
            if corpus == "longbench.qmsum" and qmsum_domain(row["context"]) == "committee": continue
            kept.append((rel, index, row))
        counts = tokens.context_tokens([r["context"] for _, _, r in kept])
        head, tail = template_span([r["context"] for _, _, r in kept])
        for (rel, index, row), n in zip(kept, counts):
            body = row["context"][head:len(row["context"]) - tail] or row["context"]
            out.append({"side": side, "corpus": corpus, "source": {"repo": repo, "file": rel, "row": index}, "family": family,
                        "task": f"{repo.split('/')[-1].lower()}.{name}", "context": row["context"], "question": row["question"],
                        "answer_prefix": row["answer_prefix"] or "", "tokens": n, "body": body})
    return out


def template_span(contexts: list[str]) -> tuple[int, int]:
    """Lengths of the instruction text every context of one source starts and ends with.

    The upstream builders wrap each document in a fixed per-task instruction;
    overlap checks compare document bodies, not those shared templates.
    """
    if len(contexts) < 2: return 0, 0
    def common(a: str, b: str) -> int:
        n = 0
        while n < min(len(a), len(b)) and a[n] == b[n]: n += 1
        return n
    head = common(min(contexts), max(contexts)); rev = [c[::-1] for c in contexts]; tail = common(min(rev), max(rev))
    shortest = min(len(c) for c in contexts)
    return min(head, shortest), min(tail, max(0, shortest - head))


def merge_documents(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop later candidates whose document repeats an earlier one (>=20% shared shingles)."""
    index: dict[int, list[int]] = collections.defaultdict(list); kept: list[dict[str, Any]] = []; sigs: list[set[int]] = []
    for cand in candidates:
        sig = shingles(cand["body"]); hits = collections.Counter(j for s in sig for j in index.get(s, ()))
        if any(n >= 0.2 * min(len(sig), len(sigs[j])) for j, n in hits.items() if sigs[j] and sig): continue
        for s in sig: index[s].append(len(kept))
        kept.append(cand); sigs.append(sig)
    return kept


# ------------------------------------------------------------------ sampling

def _take(pool: list[dict[str, Any]], n: int, *labels: str) -> list[dict[str, Any]]:
    if n > len(pool): raise SystemExit(f"cell {labels} needs {n} items, pool has {len(pool)}")
    chosen = list(pool); _rng(*labels).shuffle(chosen)
    return sorted(chosen[:n], key=lambda c: (c["source"]["file"], c["source"]["row"]))


def _water_fill(sizes: Mapping[str, int], total: int) -> dict[str, int]:
    """Equal shares across cells, capped by each cell's size; remainder to larger cells by name order."""
    alloc = {k: 0 for k in sizes}; remaining = total; open_cells = sorted(sizes)
    while remaining > 0 and open_cells:
        share = max(1, remaining // len(open_cells)); progressed = False
        for key in list(open_cells):
            if remaining == 0: break
            add = min(share, sizes[key] - alloc[key], remaining)
            if add > 0: alloc[key] += add; remaining -= add; progressed = True
            if alloc[key] >= sizes[key]: open_cells.remove(key)
        if not progressed: break
    if remaining: raise SystemExit(f"pools too small by {remaining} items")
    return alloc


def build(hf_hub: Path, model_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    tokens = Tokens(model_dir)
    ruler = load_ruler(hf_hub, tokens)
    real = merge_documents(load_real(hf_hub, tokens))
    public: list[dict[str, Any]] = []; used_ruler: dict[tuple[str, str], set[int]] = collections.defaultdict(set)
    for (task, length), pool in sorted(ruler.items()):
        n = PUBLIC_RULER[length]
        for cand in _take(pool, n, "public", "ruler", task, length):
            public.append(cand); used_ruler[(task, length)].add(cand["source"]["row"])
    lo, hi = PUBLIC_REAL_TOKENS
    for corpus in sorted({c["corpus"] for c in real if c["side"] == "public"}):
        pool = [c for c in real if c["side"] == "public" and c["corpus"] == corpus and lo <= c["tokens"] <= hi]
        public.extend(_take(pool, PUBLIC_REAL_PER_TASK, "public", corpus))
    # Any holdout document sharing even one sampled 13-word shingle of its body
    # with any public real document (all public-side documents, not only the
    # ones sampled) is dropped.
    public_sigs = set().union(*(shingles(c["body"]) for c in real if c["side"] == "public"))
    # holdout pools
    lo, hi = HOLDOUT_TOKENS
    holdout_real = [c for c in real if c["side"] == "final" and lo <= c["tokens"] <= hi and not (shingles(c["body"]) & public_sigs)]
    by_family: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for c in holdout_real: by_family[c["family"]].append(c)
    final: list[dict[str, Any]] = []
    # generation: every eligible real document
    final.extend(sorted(by_family["generation"], key=lambda c: (c["corpus"], c["source"]["file"], c["source"]["row"])))
    n_generation = len(by_family["generation"])
    # qa: every eligible real document, then RULER qa to reach the family target
    final.extend(sorted(by_family["qa"], key=lambda c: (c["corpus"], c["source"]["file"], c["source"]["row"])))
    ruler_pool = {key: [c for c in pool if c["source"]["row"] not in used_ruler[key] and lo <= c["tokens"] <= hi]
                  for key, pool in ruler.items() if key[1] in HOLDOUT_RULER_LENGTHS}
    qa_cells = {f"{t}@{l}": len(p) for (t, l), p in ruler_pool.items() if RULER_TASKS[t] == "qa"}
    for key, n in _water_fill(qa_cells, max(0, FAMILY_TARGET - len(by_family["qa"]))).items():
        task, length = key.split("@"); final.extend(_take(ruler_pool[(task, length)], n, "final", "ruler", task, length))
    rest = FINAL_ITEM_COUNT - len(final); retrieval_total = rest // 2; aggregation_total = rest - retrieval_total
    retrieval_cells = {f"{t}@{l}": len(p) for (t, l), p in ruler_pool.items() if RULER_TASKS[t] == "retrieval"}
    retrieval_cells["longbench.passage_retrieval_en"] = len(by_family["retrieval"])
    for key, n in _water_fill(retrieval_cells, retrieval_total).items():
        if "@" in key:
            task, length = key.split("@"); final.extend(_take(ruler_pool[(task, length)], n, "final", "ruler", task, length))
        else:
            final.extend(_take(by_family["retrieval"], n, "final", key))
    aggregation_cells = {f"{t}@{l}": len(p) for (t, l), p in ruler_pool.items() if RULER_TASKS[t] == "aggregation"}
    for key, n in _water_fill(aggregation_cells, aggregation_total).items():
        task, length = key.split("@"); final.extend(_take(ruler_pool[(task, length)], n, "final", "ruler", task, length))
    if len(final) != FINAL_ITEM_COUNT: raise SystemExit(f"holdout has {len(final)} items, expected {FINAL_ITEM_COUNT}")
    _rng("final", "order").shuffle(final)
    public_items = []
    for i, c in enumerate(public):
        public_items.append({"item_id": f"public-{i:03d}-{c['task'].replace('.', '-')}", "family": c["family"], "task": c["task"], "context": c["context"],
                             "question": c["question"], "answer_prefix": c["answer_prefix"],
                             "meta": {"context_tokens": c["tokens"], "source": c["source"]}})
    final_items, entries = [], []
    for i, c in enumerate(final):
        item = {"item_id": f"f{i:05d}", "family": c["family"], "task": c["task"], "context": c["context"], "question": c["question"],
                "answer_prefix": c["answer_prefix"], "meta": {"context_tokens": c["tokens"]}}
        final_items.append(item)
        entries.append({"item_id": item["item_id"], "family": c["family"], "task": c["task"], "source": c["source"], "context_tokens": c["tokens"], "sha256": rt.item_sha256(item)})
    counts_family = collections.Counter(e["family"] for e in entries); counts_task = collections.Counter(e["task"] for e in entries)
    manifest = {"schema_version": 1, "suite_id": rt.FINAL_SUITE_ID, "generator": "kvpress-v2-prepare-items", "seed": SUITE_SEED,
                "pins": PINS, "context_tokens": list(HOLDOUT_TOKENS), "item_count": len(entries),
                "counts_per_family": dict(sorted(counts_family.items())), "counts_per_task": dict(sorted(counts_task.items())),
                "generation_documents": n_generation, "items": entries}
    return public_items, manifest, final_items


# ------------------------------------------------------------------ writers

def _jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return "".join(json.dumps(r, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for r in rows).encode()


def _manifest_bytes(manifest: Mapping[str, Any]) -> bytes:
    return (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=1) + "\n").encode()


def materialize(manifest: Mapping[str, Any], hf_hub: Path, model_dir: Path | None = None) -> list[dict[str, Any]]:
    """Rebuild holdout items from their pinned source rows and check every SHA-256."""
    cache: dict[tuple[str, str], list[dict[str, Any]]] = {}; items = []
    for entry in manifest["items"]:
        src = entry["source"]; key = (src["repo"], src["file"])
        if key not in cache:
            if manifest["pins"].get(src["repo"]) != PINS[src["repo"]]: raise SystemExit(f"pin differs for {src['repo']}")
            cache[key] = _parquet_rows(_snapshot(hf_hub, src["repo"]) / src["file"], ("context", "question", "answer_prefix"))
        row = cache[key][src["row"]]
        item = {"item_id": entry["item_id"], "family": entry["family"], "task": entry["task"], "context": row["context"], "question": row["question"],
                "answer_prefix": row["answer_prefix"] or "", "meta": {"context_tokens": entry["context_tokens"]}}
        if rt.item_sha256(item) != entry["sha256"]: raise SystemExit(f"materialised text differs for {entry['item_id']}")
        items.append(item)
    return items


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("build", "verify", "materialize"))
    ap.add_argument("--hf-hub", type=Path, required=True, help="Hugging Face hub cache directory holding the pinned dataset snapshots")
    ap.add_argument("--model-dir", type=Path, help="Qwen3-8B snapshot (tokenizer) for build/verify")
    ap.add_argument("--out", type=Path, help="materialize: output final_items.jsonl.gz")
    ap.add_argument("--manifest", type=Path, help="materialize: the final manifest (default judge/private/final_manifest.json)")
    args = ap.parse_args()
    public_path = TASK_ROOT / "corpus/public/items_public.jsonl"; manifest_path = TASK_ROOT / "judge/private/final_manifest.json"
    if args.command == "materialize":
        if args.out is None: raise SystemExit("materialize needs --out")
        items = materialize(json.loads((args.manifest or manifest_path).read_text()), args.hf_hub)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with gzip.GzipFile(args.out, "wb", mtime=0) as handle: handle.write(_jsonl_bytes(items))
        print(json.dumps({"items": len(items), "sha256": _sha256_bytes(args.out.read_bytes())})); return 0
    if args.model_dir is None: raise SystemExit("build/verify need --model-dir")
    public_items, manifest, _final_items = build(args.hf_hub, args.model_dir)
    public_bytes = _jsonl_bytes(public_items); manifest_bytes = _manifest_bytes(manifest)
    if args.command == "verify":
        problems = [p for p, b in ((public_path, public_bytes), (manifest_path, manifest_bytes)) if not p.is_file() or p.read_bytes() != b]
        if problems: raise SystemExit("rebuild differs from committed files: " + ", ".join(str(p) for p in problems))
        print("verify: public items and final manifest rebuild byte-for-byte"); return 0
    public_path.parent.mkdir(parents=True, exist_ok=True); public_path.write_bytes(public_bytes)
    (public_path.parent / "MANIFEST.json").write_text(json.dumps({"schema_version": 2, "files": {public_path.name: _sha256_bytes(public_bytes)}}, indent=2, sort_keys=True) + "\n")
    manifest_path.parent.mkdir(parents=True, exist_ok=True); manifest_path.write_bytes(manifest_bytes)
    print(json.dumps({"public_items": len(public_items), "final_items": manifest["item_count"], "counts_per_family": manifest["counts_per_family"],
                      "public_sha256": _sha256_bytes(public_bytes), "manifest_sha256": _sha256_bytes(manifest_bytes)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
