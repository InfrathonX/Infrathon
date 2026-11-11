#!/usr/bin/env python3
"""Trim raw router captures into the routing fixtures the suites read.

``scripts/capture_routing.py`` records, for an open MoE model, every MoE layer's
router top-k expert ids over ~164k tokens of mixed real text (encyclopedia prose,
chat, source code, math word problems and reading-comprehension QA, interleaved
sequence by sequence).  This script keeps FIXTURE_LAYERS evenly spaced layers and
the first FIXTURE_TOKENS tokens of each capture (the interleaving puts every source
in that window), stores them as uint8 ``[layers, tokens, topk]`` ``.npy`` files
(.npy format 1.0, no pickle) and writes a JSON provenance record beside each one,
including the sha256 a captured-routing case pins.  Judge-only fixtures go to
``judge/private/routing`` (mode 0600), public ones to
``images/work/benchmark/routing``.

    scripts/build_routing_fixtures.py --raw <capture output dir>
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path

import numpy as np

TASK_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_LAYERS = 8
FIXTURE_TOKENS = 49152
SIDES = {
    "judge": ("gpt-oss-20b", "phi-3.5-moe", "mixtral-8x7b", "granite-3.1-3b-a800m"),
    "public": ("olmoe-1b-7b", "qwen3-30b-a3b"),
}
OUTPUT = {"judge": TASK_ROOT / "judge/private/routing", "public": TASK_ROOT / "images/work/benchmark/routing"}
MODE = {"judge": 0o600, "public": 0o644}


def encode(table: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.lib.format.write_array(buffer, table, version=(1, 0), allow_pickle=False)
    return buffer.getvalue()


def trim(raw: Path, meta: dict) -> tuple[bytes, dict]:
    data = raw.read_bytes()
    table = np.load(io.BytesIO(data), allow_pickle=False)
    layers, tokens, topk = table.shape
    if (layers, tokens, topk) != (meta["layers"], meta["tokens"], meta["topk"]):
        raise SystemExit(f"{raw}: shape {table.shape} differs from its record")
    if layers < FIXTURE_LAYERS or tokens < FIXTURE_TOKENS:
        raise SystemExit(f"{raw}: capture smaller than the fixture window")
    if int(table.min()) < 0 or int(table.max()) >= min(int(meta["experts"]), 256):
        raise SystemExit(f"{raw}: expert ids outside [0, {meta['experts']})")
    kept = [round(i * (layers - 1) / (FIXTURE_LAYERS - 1)) for i in range(FIXTURE_LAYERS)]
    fixture = np.ascontiguousarray(table[kept, :FIXTURE_TOKENS, :].astype(np.uint8))
    encoded = encode(fixture)
    sequences = []
    source_tokens: dict[str, int] = {}
    for row in meta["sequences"]:
        start, count = int(row["start"]), int(row["tokens"])
        if start >= FIXTURE_TOKENS:
            break
        count = min(count, FIXTURE_TOKENS - start)
        sequences.append({"source": row["source"], "start": start, "tokens": count})
        source_tokens[row["source"]] = source_tokens.get(row["source"], 0) + count
    record = {
        "name": meta["name"],
        "repo": meta["repo"],
        "revision": meta["revision"],
        "experts": int(meta["experts"]),
        "topk": int(topk),
        "model_moe_layers": int(layers),
        "kept_layers": kept,
        "layers": FIXTURE_LAYERS,
        "tokens": FIXTURE_TOKENS,
        "dtype": "uint8",
        "shape": list(fixture.shape),
        "source_tokens": source_tokens,
        "datasets": meta["datasets"],
        "sequences": sequences,
        "seq_len": meta["seq_len"],
        "method": meta["method"],
        "capture_sha256": hashlib.sha256(data).hexdigest(),
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }
    return encoded, record


def write(path: Path, payload: bytes, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_bytes(payload)
    tmp.chmod(mode)
    os.replace(tmp, path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", type=Path, required=True, help="capture_routing.py output directory")
    args = parser.parse_args(argv)
    summary = {}
    for side, names in SIDES.items():
        for name in names:
            meta = json.loads((args.raw / f"{name}.json").read_text(encoding="utf-8"))
            encoded, record = trim(args.raw / f"{name}.npy", meta)
            write(OUTPUT[side] / f"{name}.npy", encoded, MODE[side])
            write(OUTPUT[side] / f"{name}.json", (json.dumps(record, indent=1, sort_keys=True) + "\n").encode(), MODE[side])
            summary[name] = {"side": side, "sha256": record["sha256"], "bytes": len(encoded),
                             "source_tokens": record["source_tokens"]}
    print(json.dumps(summary, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
