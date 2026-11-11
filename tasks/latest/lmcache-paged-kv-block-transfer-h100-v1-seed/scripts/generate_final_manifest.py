#!/usr/bin/env python3
"""Generate the v2 public case table and the Judge-only final manifest.

Host-only.  Nothing here ships in an image.

Workloads come from serving traces, not from a synthetic grid:

1. Per-GPU shapes are real model configurations under tensor parallelism
   (layers, KV heads per GPU, head size; HF ``config.json``), for every TP in
   {1, 2, 4, 8} whose weights fit an 80 GB GPU with room for a pool.  The
   Judge uses the models the LMCache paper evaluates (arXiv 2510.09665 §8.1,
   tier ``paper``) and other open models (tier ``other``); the public table
   uses a third set of models.  No (layout, NL, NH, HS) is shared.  Every
   (shape, layout) is a stratum; the score weights strata and tiers equally.
2. The engine pool size follows vLLM's sizing rule
   (``gpu_memory_utilization`` 0.92 of an 80 GB H100, minus the weights per
   GPU and a fixed activation reserve), capped so one case fits the Judge.
3. Block-id tables come from replaying a Mooncake FAST'25 trace
   (github.com/kvcache-ai/Mooncake, FAST25-release/traces) through a
   simplified vLLM block manager (LRU free queue, prefix-hash reuse, reverse
   free order) and an LMCache store: a request's GPU-resident prefix is
   skipped, chunks already in the store are retrieved, new chunks (including
   a partial tail chunk) are stored, and completed decode chunks are stored.
   The public table replays the synthetic trace; the Judge replays the
   conversation and tool-agent traces.

The replay is a model of a real engine, not a capture of one; the block-id
tables carry its run structure, not a recorded vLLM run.

Usage:
  generate_final_manifest.py --traces DIR --public-out PATH --final-out PATH --redraw
The final manifest belongs in the Judge-only asset root
(<INFRATHON_ASSETS_ROOT>/<task>/judge/final_cases.json), never in the repo.

The draw is seeded, never from a system random source.  The default seed is
derived from the suite identity alone (the first eight bytes of
``sha256("<suite_id>|<task_id>")``, big-endian), so the frozen instance is
reproducible from the traces on any host.  ``--redraw`` is the required
acknowledgement that writing a manifest mints the suite instance whose
``frozen_instance`` record (portable/final-suite-manifest.json) must be
updated before any score is compared against it.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import heapq
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Iterable

SCRIPT_DIR = Path(__file__).resolve().parent


def _load_trusted_runtime():
    """Load runtime/kv_transfer_runtime.py by path under a private name, so an
    already-imported public runtime of the same module name is never reused."""
    import importlib.util

    path = SCRIPT_DIR.parent / "runtime" / "kv_transfer_runtime.py"
    spec = importlib.util.spec_from_file_location("lmcache_trusted_runtime_for_generator", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runtime = _load_trusted_runtime()

SUITE_ID = runtime.FINAL_SUITE_ID
TASK_ID = "lmcache_paged_kv_block_transfer_h100_seed_v1"


def default_seed() -> int:
    digest = hashlib.sha256(f"{SUITE_ID}|{TASK_ID}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


DEFAULT_SEED = default_seed()
BS = runtime.BS
HASH_TOKENS = 512
GPU_BYTES = 85.0e9
GPU_MEMORY_UTILIZATION = 0.92
ACTIVATION_RESERVE = 2.5e9
POOL_CAP_BYTES = 40.0e9
PUBLIC_POOL_CAP_BYTES = 12.0e9

# Base models: name, HF id, layers per call, KV heads, head size, weight
# bytes (all GPUs), tier.  Layers per call is the layer count, except gpt-oss:
# its sliding and full attention layers form two KV groups of half the layers.
# KV heads per GPU = KV heads / TP, or 1 once TP exceeds them (vLLM
# replicates a KV head across ranks).
JUDGE_MODELS = (
    ("llama31-8b", "meta-llama/Llama-3.1-8B-Instruct", 32, 8, 128, 16.1e9, "paper"),
    ("qwen25-coder-32b", "Qwen/Qwen2.5-Coder-32B-Instruct", 64, 8, 128, 65.5e9, "paper"),
    ("llama31-70b", "meta-llama/Llama-3.1-70B-Instruct (Qwen2.5-72B: same shape)", 80, 8, 128, 141.2e9, "paper"),
    ("qwen3-coder-480b", "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8", 62, 8, 128, 480.0e9, "paper"),
    ("qwen25-14b", "Qwen/Qwen2.5-14B-Instruct (InternLM2.5-20B: same shape)", 48, 8, 128, 29.5e9, "other"),
    ("qwen3-235b", "Qwen/Qwen3-235B-A22B", 94, 4, 128, 470.0e9, "other"),
    ("mixtral-8x22b", "mistralai/Mixtral-8x22B-Instruct-v0.1", 56, 8, 128, 281.2e9, "other"),
    ("mistral-large-2411", "mistralai/Mistral-Large-Instruct-2411", 88, 8, 128, 245.2e9, "other"),
    ("yi15-34b", "01-ai/Yi-1.5-34B-Chat", 60, 8, 128, 68.8e9, "other"),
    ("gpt-oss-120b", "openai/gpt-oss-120b", 18, 8, 64, 65.0e9, "other"),
    ("llama32-1b", "meta-llama/Llama-3.2-1B-Instruct", 16, 8, 64, 2.5e9, "other"),
)
PUBLIC_MODELS = (
    ("qwen3-8b", "Qwen/Qwen3-8B", 36, 8, 128, 16.4e9, "public"),
    ("qwen25-3b", "Qwen/Qwen2.5-3B-Instruct", 36, 2, 128, 6.2e9, "public"),
    ("qwen25-7b", "Qwen/Qwen2.5-7B-Instruct", 28, 4, 128, 15.2e9, "public"),
    ("qwen25-15b", "Qwen/Qwen2.5-1.5B-Instruct", 28, 2, 128, 3.1e9, "public"),
    ("llama32-3b", "meta-llama/Llama-3.2-3B-Instruct", 28, 8, 128, 6.4e9, "public"),
    ("mistral-nemo-12b", "mistralai/Mistral-Nemo-Instruct-2407 (Qwen3-14B: same shape)", 40, 8, 128, 24.5e9, "public"),
    ("phi-4", "microsoft/phi-4", 40, 10, 128, 29.3e9, "public"),
    ("glm4-9b", "THUDM/glm-4-9b-chat", 40, 2, 128, 18.8e9, "public"),
    ("olmo2-13b", "allenai/OLMo-2-1124-13B-Instruct", 40, 40, 128, 27.4e9, "public"),
    ("gpt-oss-20b", "openai/gpt-oss-20b", 12, 8, 64, 13.8e9, "public"),
)
TP_CHOICES = (1, 2, 4, 8)
MIN_POOL_BYTES = 8.0e9


def expand_shapes(models: tuple) -> list[tuple]:
    """(name, model, tp, nl, nh, hs, weights per GPU, tier), one per distinct per-GPU shape."""
    shapes, seen = [], set()
    for name, model, nl, kv, hs, weights, tier in models:
        for tp in TP_CHOICES:
            per_gpu = weights / tp
            if GPU_BYTES * GPU_MEMORY_UTILIZATION - per_gpu - ACTIVATION_RESERVE < MIN_POOL_BYTES:
                continue
            nh = max(1, kv // tp)
            if nh > runtime.NH_RANGE[1] or (nl, nh, hs) in seen:
                continue
            seen.add((nl, nh, hs))
            shapes.append((f"{name}-tp{tp}", model, tp, nl, nh, hs, per_gpu, tier))
    return shapes


JUDGE_SHAPES = tuple(expand_shapes(JUDGE_MODELS))
PUBLIC_SHAPES = tuple(expand_shapes(PUBLIC_MODELS))
TRIALS = runtime.DEFAULT_TRIALS
LARGE_WAVES = 16
SMALL_WAVES = 64
TRACE_WAVES = 32
OVERLAP_WAVES = 12
DUPLEX_WAVES = 16
MIN_DIRECT_WAVES = 12
# Every case moves at least this much per trial, well above CUDA-event resolution.
MIN_CASE_BYTES = 32 * 1024 * 1024
# Cases per (shape, layout) stratum and family: fractional quotas are met on
# average by alternating across strata.
JUDGE_QUOTA = {"device-large": 3, "device-small": 2.5, "host-large": 2, "host-small": 2.5,
               "host-duplex": 2, "trace": 1.5, "overlap": 2}
PUBLIC_QUOTA = {"device-large": 1, "device-small": 1, "host-large": 0.5, "host-small": 1,
                "host-duplex": 0.5, "trace": 0.5, "overlap": 0.5}


def block_bytes(nl: int, nh: int, hs: int) -> int:
    return 2 * nl * nh * BS * hs * runtime.ESIZE


def pool_blocks(shape: tuple, cap: float) -> int:
    _name, _model, _tp, nl, nh, hs, weights, _tier = shape
    available = GPU_BYTES * GPU_MEMORY_UTILIZATION - weights - ACTIVATION_RESERVE
    per_block = block_bytes(nl, nh, hs)
    return int(min(available // per_block, cap // per_block, runtime.NB_RANGE[1]))


# --------------------------------------------------------------------------
# Trace replay
# --------------------------------------------------------------------------
class BlockManager:
    def __init__(self, nb: int) -> None:
        self.free = collections.OrderedDict((b, None) for b in range(nb))
        self.key_of: dict[int, tuple] = {}
        self.block_of: dict[tuple, int] = {}
        self.ref = [0] * nb

    def allocate(self) -> int:
        block, _ = self.free.popitem(last=False)
        key = self.key_of.pop(block, None)
        if key is not None:
            self.block_of.pop(key, None)
        self.ref[block] = 1
        return block

    def touch(self, block: int) -> None:
        if self.ref[block] == 0:
            self.free.pop(block, None)
        self.ref[block] += 1

    def release(self, blocks: list[int]) -> None:
        for block in reversed(blocks):
            self.ref[block] -= 1
            if self.ref[block] == 0:
                self.free[block] = None

    def cache(self, block: int, key: tuple) -> None:
        if key not in self.block_of:
            self.block_of[key] = block
            self.key_of[block] = key


def replay(rows: list[dict], nb: int, chunk: int, *, max_requests: int,
           warmup_requests: int, base_s: float = 2.0, tpot_s: float = 0.03) -> list[dict]:
    """Transfer events of an LMCache-backed engine replaying ``rows``."""
    bm = BlockManager(nb)
    stored: set[tuple] = set()
    per_chunk = chunk // BS
    blocks_per_hash = HASH_TOKENS // BS
    pending: collections.deque = collections.deque()
    running: list = []
    events: list[dict] = []
    clock = 0.0
    index = 0
    seq = 0
    admitted = 0

    def emit(direction: str, groups: list[tuple[list[int], int]], skip: int, when: float,
             record: bool) -> None:
        # groups: (chunk block ids, chunk tokens); one call carries <= 4 equal chunks
        start = 0
        while start < len(groups):
            tokens = groups[start][1]
            batch = [groups[start]]
            while (len(batch) < 4 and start + len(batch) < len(groups)
                   and groups[start + len(batch)][1] == tokens):
                batch.append(groups[start + len(batch)])
            if record:
                ids = [b for chunk_ids, _t in batch for b in chunk_ids]
                events.append({"direction": direction, "chunk_tokens": tokens,
                               "num_objects": len(batch), "ids": ids,
                               "skip": skip if start == 0 else 0, "t": when})
            start += len(batch)
            skip = 0

    while (index < len(rows) or pending) and admitted < max_requests:
        next_arrival = rows[index]["timestamp"] / 1000.0 if index < len(rows) else math.inf
        next_finish = running[0][0] if running else math.inf
        if next_arrival <= next_finish and index < len(rows):
            clock = next_arrival
            pending.append(rows[index])
            index += 1
        elif running:
            clock, _seq, blocks, decode_groups, record = heapq.heappop(running)
            emit("store", decode_groups, 0, clock, record)
            bm.release(blocks)
        else:
            break
        while pending:
            req = pending[0]
            prompt_blocks = req["input_length"] // BS
            total = math.ceil((req["input_length"] + req["output_length"]) / BS)
            hashes = req["hash_ids"]
            keys = []
            for j in range(prompt_blocks):
                h = j // blocks_per_hash
                keys.append((hashes[h], j % blocks_per_hash) if h < len(hashes) else ("u", seq, j))
            hit = 0
            while hit < prompt_blocks and keys[hit] in bm.block_of:
                hit += 1
            idle = sum(1 for j in range(hit) if bm.ref[bm.block_of[keys[j]]] == 0)
            if total - hit + idle > len(bm.free):
                if not running:
                    pending.popleft()
                    continue
                break
            pending.popleft()
            blocks = []
            for j in range(hit):
                block = bm.block_of[keys[j]]
                bm.touch(block)
                blocks.append(block)
            for j in range(hit, total):
                block = bm.allocate()
                blocks.append(block)
                if j < prompt_blocks:
                    bm.cache(block, keys[j])
            record = admitted >= warmup_requests
            # Chunk the prompt; chunks fully inside the GPU prefix move nothing.
            retrieve: list[tuple[list[int], int]] = []
            store: list[tuple[list[int], int]] = []
            retrieve_skip = 0
            for c0 in range(0, prompt_blocks, per_chunk):
                c1 = min(c0 + per_chunk, prompt_blocks)
                if c1 <= hit:
                    continue
                ids = blocks[c0:c1]
                tokens = (c1 - c0) * BS
                if all(keys[j] in stored for j in range(c0, c1)):
                    if not retrieve:
                        retrieve_skip = max(0, hit - c0)
                    retrieve.append((ids, tokens))
                else:
                    store.append((ids, tokens))
                    stored.update(keys[j] for j in range(c0, c1))
            emit("retrieve", retrieve, retrieve_skip, clock, record)
            emit("store", store, 0, clock, record)
            out_blocks = blocks[prompt_blocks:]
            decode_groups = [(out_blocks[k:k + per_chunk], chunk)
                             for k in range(0, len(out_blocks) - per_chunk + 1, per_chunk)]
            finish = clock + base_s + req["output_length"] * tpot_s
            seq += 1
            admitted += 1
            heapq.heappush(running, (finish, seq, blocks, decode_groups, record))
    return events


def map_stats(events: Iterable[dict]) -> dict[str, float]:
    adj = tot = 0
    for ev in events:
        ids = ev["ids"]
        for a, b in zip(ids, ids[1:]):
            tot += 1
            adj += abs(b - a) == 1
    return {"adjacent_fraction": adj / max(1, tot)}


# --------------------------------------------------------------------------
# Case assembly
# --------------------------------------------------------------------------
def wave_of(shape: tuple, layout: str, nb: int, ev: dict) -> dict:
    _name, _model, _tp, nl, nh, hs, _w, _tier = shape
    return {"layout": layout, "nl": nl, "nh": nh, "hs": hs,
            "chunk_tokens": ev["chunk_tokens"], "num_objects": ev["num_objects"],
            "direction": ev["direction"], "skip": ev["skip"], "nb": nb, "ids": list(ev["ids"])}


def logical(shape: tuple, layout: str, nb: int, ev: dict) -> int:
    return runtime.wave_logical_bytes(wave_of(shape, layout, nb, ev))


def pick_direct(events: list[dict], shape: tuple, layout: str, nb: int, *, direction: str,
                small: bool, waves: int, rng: random.Random, start_frac: float) -> list[dict] | None:
    """``waves`` events of one call class, consecutive in the replay where possible."""
    def fits(ev: dict) -> bool:
        if ev["direction"] != direction:
            return False
        if logical(shape, layout, nb, ev) == 0:
            return False
        return (logical(shape, layout, nb, ev) < runtime.SMALL_CALL_BYTES) == small

    candidates = [ev for ev in events if fits(ev)]
    if not candidates:
        return None
    classes = collections.Counter((ev["chunk_tokens"], ev["num_objects"]) for ev in candidates)
    ranked = [key for key, _n in classes.most_common()]
    # A timed case needs enough calls to sit well above timer resolution:
    # prefer call classes that occur at least ``waves`` times in the replay.
    plenty = [key for key in ranked if classes[key] >= waves]
    pool = plenty[:3] if plenty else ranked[:1]
    key = pool[0] if not small else rng.choice(pool)
    if classes[key] < MIN_DIRECT_WAVES:
        return None
    chosen = [ev for ev in candidates if (ev["chunk_tokens"], ev["num_objects"]) == key]
    begin = int(start_frac * max(0, len(chosen) - waves))
    picked = chosen[begin:begin + waves]
    return picked if picked else None


CASE_PAYLOAD_BUDGET = int(2.5 * 1024**3)


def make_case(case_id: str, family: str, arm: str, waves: list[dict], seed: int,
              stratum: str, tier: str, foreground: str | None = None) -> dict:
    while len(waves) > (2 if arm == "duplex" else 1) and runtime.case_payload_bytes({"waves": waves}) > CASE_PAYLOAD_BUDGET:
        cut = max(1, len(waves) * 3 // 4)
        waves = waves[: max(2, cut - cut % 2)] if arm == "duplex" else waves[:cut]
    case = {"case_id": case_id, "family": family, "arm": arm, "foreground": foreground,
            "stratum": stratum, "tier": tier,
            "waves": waves, "trials": TRIALS, "seed": seed}
    runtime.validate_case(case, scored=True)
    return case


def pick_duplex(events: list[dict], shape: tuple, layout: str, nb: int, *,
                waves: int) -> list[dict] | None:
    """Alternating store and retrieve calls of one call class over disjoint blocks."""
    large = [ev for ev in events if logical(shape, layout, nb, ev) >= runtime.SMALL_CALL_BYTES]
    classes = collections.Counter((ev["chunk_tokens"], ev["num_objects"]) for ev in large)
    for key, _n in classes.most_common():
        stores = [ev for ev in large if (ev["chunk_tokens"], ev["num_objects"]) == key
                  and ev["direction"] == "store"][: waves // 2]
        used = {b for ev in stores for b in ev["ids"]}
        loads = [ev for ev in large if (ev["chunk_tokens"], ev["num_objects"]) == key
                 and ev["direction"] == "retrieve" and not used & set(ev["ids"])][: waves // 2]
        count = min(len(stores), len(loads))
        if count * 2 >= MIN_DIRECT_WAVES:
            out = []
            for a, b in zip(stores[:count], loads[:count]):
                out += [a, b]
            return out
    return None


def _replay_job(args: tuple) -> tuple:
    trace, nb, chunk, rows, max_requests = args
    return (trace, nb, chunk), replay(rows, nb, chunk, max_requests=max_requests,
                                      warmup_requests=max_requests // 3)


def build_suite(shapes: tuple, traces: dict[str, list[dict]], trace_names: tuple[str, ...],
                *, prefix: str, seed: int, cap: float, foregrounds: tuple[str, ...],
                quota: dict[str, float], max_requests: int, jobs: int,
                neutral_strata: bool) -> tuple[list[dict], dict]:
    import multiprocessing

    rng = random.Random(seed)
    cases: list[dict] = []
    meta: dict[str, Any] = {"deployments": [], "strata": []}
    chunk_choices = (256, 256, 512, 128, 1024)
    plans = []
    for s_index, shape in enumerate(shapes):
        nb = pool_blocks(shape, cap)
        chunk = chunk_choices[s_index % len(chunk_choices)]
        plans.append((shape, nb, chunk))
    work = sorted({(trace, nb, chunk) for _shape, nb, chunk in plans for trace in trace_names})
    with multiprocessing.Pool(max(1, jobs)) as pool:
        replays = dict(pool.map(_replay_job, [(t, nb, c, traces[t], max_requests) for t, nb, c in work]))
    for key in work:
        meta["deployments"].append({"trace": key[0], "nb": key[1], "chunk": key[2],
                                    "events": len(replays[key]), **map_stats(replays[key])})

    def new_seed() -> int:
        return rng.randrange(1, 1 << 62)

    owed = collections.Counter()
    issued = collections.Counter()
    stratum_index = 0
    for shape, nb, chunk in plans:
        name, _model, _tp, _nl, _nh, _hs, _w, tier = shape
        for layout in ("split", "fused"):
            stratum = f"s{stratum_index:03d}" if neutral_strata else f"{name}-{layout}"
            stratum_index += 1
            meta["strata"].append({"stratum": stratum, "shape": name, "layout": layout, "nb": nb,
                                   "chunk": chunk, "tier": tier})
            for family, share in quota.items():
                owed[(stratum, family)] = 0
                count = int(share + owed.get(("carry", family), 0.0))
                owed[("carry", family)] = owed.get(("carry", family), 0.0) + share - count
                for k in range(count):
                    trace = trace_names[(k + stratum_index) % len(trace_names)]
                    events = replays[(trace, nb, chunk)]
                    tag = f"{prefix}-{name}-{layout}-{trace[:4]}-{family}-{k}"
                    arm = runtime.FAMILIES[family]
                    if family == "trace":
                        begin = rng.randrange(0, max(1, len(events) - 2 * TRACE_WAVES))
                        window = [ev for ev in events[begin:begin + TRACE_WAVES * 2]
                                  if logical(shape, layout, nb, ev) > 0][:TRACE_WAVES]
                        if len(window) < 8:
                            continue
                        waves = [wave_of(shape, layout, nb, ev) for ev in window]
                        cases.append(make_case(tag, family, arm, waves, new_seed(), stratum, tier))
                        continue
                    if family == "host-duplex":
                        start = rng.randrange(0, max(1, len(events) // 2))
                        picked = pick_duplex(events[start:], shape, layout, nb, waves=DUPLEX_WAVES)
                        if picked is None:
                            continue
                        waves = [wave_of(shape, layout, nb, ev) for ev in picked]
                        cases.append(make_case(tag, family, arm, waves, new_seed(), stratum, tier))
                        continue
                    issued[family] += 1
                    direction = ("store", "retrieve")[issued[family] % 2]
                    small = family.endswith("-small")
                    count_waves = SMALL_WAVES if small else (OVERLAP_WAVES if family == "overlap" else LARGE_WAVES)
                    picked = pick_direct(events, shape, layout, nb, direction=direction, small=small,
                                         waves=count_waves, rng=rng, start_frac=rng.random())
                    if picked is None:
                        other = "retrieve" if direction == "store" else "store"
                        picked = pick_direct(events, shape, layout, nb, direction=other, small=small,
                                             waves=count_waves, rng=rng, start_frac=rng.random())
                    if picked is None:
                        continue
                    waves = [wave_of(shape, layout, nb, ev) for ev in picked]
                    foreground = (foregrounds[(issued[family] // 2) % len(foregrounds)]
                                  if family == "overlap" else None)
                    if sum(runtime.wave_logical_bytes(w) for w in waves) < MIN_CASE_BYTES:
                        continue
                    cases.append(make_case(tag, family, arm, waves, new_seed(), stratum, tier, foreground))
    return cases, meta


def load_trace(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_table(path: Path, suite_id: str, cases: list[dict], meta: dict, mode: int) -> str:
    payload = {"schema_version": 2, "suite_id": suite_id, "cases": cases}
    text = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    path.chmod(mode)
    digest = hashlib.sha256(text.encode()).hexdigest()
    meta_path = path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps({**meta, "sha256": digest, "cases": len(cases)}, indent=1) + "\n")
    return digest


def summarize(cases: list[dict]) -> dict:
    by_family = collections.Counter(case["family"] for case in cases)
    return {"cases": len(cases), "families": dict(sorted(by_family.items())),
            "logical_bytes": sum(runtime.case_logical_bytes(c) for c in cases)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--traces", type=Path, required=True,
                        help="directory with Mooncake FAST'25 *_trace.jsonl files")
    parser.add_argument("--public-out", type=Path, required=True)
    parser.add_argument("--final-out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-requests", type=int, default=3000)
    parser.add_argument("--jobs", type=int, default=16, help="parallel trace replays")
    parser.add_argument("--redraw", action="store_true",
                        help="acknowledge that the output is a new suite instance")
    args = parser.parse_args(argv)
    if not args.redraw:
        parser.error("writing a manifest mints a suite instance; pass --redraw to acknowledge it")
    traces = {}
    digests = {}
    for name in ("conversation", "toolagent", "synthetic"):
        path = args.traces / f"{name}_trace.jsonl"
        traces[name] = load_trace(path)
        digests[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    public, public_meta = build_suite(
        PUBLIC_SHAPES, traces, ("synthetic",), prefix="public", seed=args.seed,
        cap=PUBLIC_POOL_CAP_BYTES, foregrounds=("gemm",), quota=PUBLIC_QUOTA,
        max_requests=args.max_requests // 2, jobs=args.jobs, neutral_strata=False)
    final, final_meta = build_suite(
        JUDGE_SHAPES, traces, ("conversation", "toolagent"), prefix="final",
        seed=args.seed ^ 0x5EED, cap=POOL_CAP_BYTES, foregrounds=("gemm", "hbm"),
        quota=JUDGE_QUOTA, max_requests=args.max_requests, jobs=args.jobs, neutral_strata=True)
    # Judge case ids are neutral: the descriptive name (model, trace, chunk
    # size) would name the hidden shapes.  It is kept in the root-only meta file.
    names = []
    for index, case in enumerate(final):
        names.append({"case_id": f"final-{index:04d}-{case['family']}", "name": case["case_id"]})
        case["case_id"] = names[-1]["case_id"]
    final_meta["case_names"] = names
    check_disjoint(public, final)
    meta = {"traces": digests, "generator_seed": args.seed}
    public_sha = write_table(args.public_out, runtime.PUBLIC_SUITE_ID, public,
                             {**meta, **public_meta, "shapes": [s[:8] for s in PUBLIC_SHAPES]},
                             0o644)
    final_sha = write_table(args.final_out, runtime.FINAL_SUITE_ID, final,
                            {**meta, **final_meta, "shapes": [s[:8] for s in JUDGE_SHAPES]},
                            0o600)
    print(json.dumps({"public": {**summarize(public), "sha256": public_sha},
                      "final": {**summarize(final), "sha256": final_sha}}, indent=1))
    return 0


def shape_key(wave: dict) -> tuple:
    return (wave["layout"], wave["nl"], wave["nh"], wave["hs"])


def check_disjoint(public: list[dict], final: list[dict]) -> None:
    pub_shapes = {shape_key(w) for c in public for w in c["waves"]}
    fin_shapes = {shape_key(w) for c in final for w in c["waves"]}
    if pub_shapes & fin_shapes:
        raise SystemExit(f"public and final share shapes: {sorted(pub_shapes & fin_shapes)}")
    if {c["seed"] for c in public} & {c["seed"] for c in final}:
        raise SystemExit("public and final share seeds")
    if {c["case_id"] for c in public} & {c["case_id"] for c in final}:
        raise SystemExit("public and final share case ids")
    for cases, label in ((public, "public"), (final, "final")):
        families = {c["family"] for c in cases}
        if families != set(runtime.FAMILIES):
            raise SystemExit(f"{label} misses families {set(runtime.FAMILIES) - families}")


if __name__ == "__main__":
    raise SystemExit(main())
