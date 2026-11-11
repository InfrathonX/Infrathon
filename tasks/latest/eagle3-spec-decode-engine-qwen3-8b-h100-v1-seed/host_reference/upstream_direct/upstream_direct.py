"""Thin token-level harness around the pinned vendored EaModel implementation."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch

from runtime.api import truncate_generation
from runtime.loader import TARGET_REVISION, build_contract, load_tokenizer
from runtime.prompts import load_pool, render


REPO_ROOT = Path(__file__).resolve().parents[5]
VENDOR = REPO_ROOT / "vendor/EAGLE"


def target_snapshot() -> Path:
    """Resolve the exact offline target snapshot used by the vendored route."""
    default_root = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-8B"
    root = Path(os.environ.get("EAGLE3_TARGET_REPO", str(default_root))).resolve()
    snapshot = (
        root
        if root.name == TARGET_REVISION
        else root / "snapshots" / TARGET_REVISION
    )
    if not snapshot.is_dir() or not (snapshot / "config.json").is_file():
        raise FileNotFoundError(f"frozen target snapshot is unavailable: {snapshot}")
    return snapshot


def load_upstream(head: Path) -> Any:
    """Load the pinned vendored EAGLE implementation offline."""
    if str(VENDOR) not in sys.path:
        sys.path.insert(0, str(VENDOR))
    from eagle.model.ea_model import EaModel
    model = EaModel.from_pretrained(
        use_eagle3=True,
        base_model_path=str(target_snapshot()),
        ea_model_path=str(head),
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        total_token=60,
        depth=7,
        top_k=10,
        threshold=1.0,
    )
    return model.eval()


@torch.inference_mode()
def run(
    model: Any,
    prompts: list[tuple[str, torch.Tensor]],
    max_new_tokens: int = 512,
) -> list[dict[str, Any]]:
    """Generate and time one upstream stream at a time."""
    rows = []
    for prompt_id, prompt in prompts:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        output, new_token, last_idx = model.eagenerate(
            prompt.unsqueeze(0),
            temperature=0,
            max_new_tokens=max_new_tokens,
            max_length=4096,
            log=True,
        )
        torch.cuda.synchronize()
        wall = time.perf_counter() - start
        output = truncate_generation(
            output[0],
            prompt.numel(),
            max_new_tokens,
            (151645, 151643),
        )
        rows.append(
            {
                "prompt_id": prompt_id,
                "prompt_len": prompt.numel(),
                "tokens": output.tolist(),
                "generated_tokens": output.numel() - prompt.numel(),
                "upstream_new_token": int(new_token),
                "cycles": int(last_idx) + 1,
                "wall_s": wall,
                "peak_memory_bytes": torch.cuda.max_memory_allocated(),
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    """Run the host-only upstream parity route and persist full evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--max-new", type=int, default=512)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest_path = args.head / "MANIFEST.json"
    if not manifest_path.is_file():
        manifest_path = args.head / "sha256s.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    contract = build_contract(manifest["revision"])
    tokenizer = load_tokenizer(contract)
    pool = load_pool(args.pool)[:args.n]
    prompts = [
        (
            row["prompt_id"],
            render(row["messages"], tokenizer).to(contract.device),
        )
        for row in pool
    ]
    model = load_upstream(args.head)
    rows = run(model, prompts, args.max_new)
    payload = {
        "implementation": "vendor/EAGLE@cb7e0841fe0c206c6ed74a197ad5e2a1f13f5a2b",
        "rows": rows,
        "total_tokens": sum(row["generated_tokens"] for row in rows),
        "total_wall_s": sum(row["wall_s"] for row in rows),
    }
    payload["tokens_per_s"] = payload["total_tokens"] / payload["total_wall_s"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items() if k != "rows"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
