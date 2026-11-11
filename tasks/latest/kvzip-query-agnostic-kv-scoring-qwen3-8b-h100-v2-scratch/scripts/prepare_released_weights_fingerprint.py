#!/usr/bin/env python3
"""Build the Judge-only fingerprint of the released Fast KVzip Qwen3-8B gates.

The fingerprint (file/tensor SHA-256 digests, a 256-wide random sketch of each
4096-wide row and the sketch matrix) is what ``released_weights_gate.py``
compares submitted assets with; it holds no weights.  ``build_images.sh``
writes it 0600 into the Judge-only build context, next to the holdout rows.
Its outputs are pinned by SHA-256; the repository virtualenv's CPU torch
reproduces them byte for byte (checked with torch 2.8 and 2.14).
"""
from __future__ import annotations

import argparse
import hashlib
import io
import sys
from pathlib import Path

TASK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK_ROOT / "images/judge")); sys.path.insert(0, str(TASK_ROOT / "scripts"))
import prepare_oracle_assets as oracle  # noqa: E402
import released_weights_gate as gate  # noqa: E402

# Every released gate for Qwen3-8B at the pinned revision.  The FP8-model gate
# is a near-copy of the BF16 one (row |cos| ~0.999), so only the BF16 gate's
# rows are sketched; both contribute file and tensor digests.
RELEASED = (
    {"repo": oracle.REPO_ID, "revision": oracle.REVISION, "filename": oracle.FILENAME, "sha256": oracle.UPSTREAM_SHA256,
     "repacked_sha256": oracle.OUTPUT_SHA256, "sketch": True},
    {"repo": oracle.REPO_ID, "revision": oracle.REVISION, "filename": "qwen3-8b-fp8/q4_dim16_sink16.pt",
     "sha256": "aa6eb0daab67239ded49e13128bf269cc05101efc0b0f70d9b69c2433b9a4df8", "sketch": False},
)


def build() -> dict:
    import torch
    from huggingface_hub import hf_hub_download
    checkpoints = []; sources = []
    for entry in RELEASED:
        path = Path(hf_hub_download(entry["repo"], filename=entry["filename"], revision=entry["revision"]))
        digest = oracle.sha256_file(path)
        if digest != entry["sha256"]: raise RuntimeError(f"{entry['filename']} SHA-256 differs: {digest}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        tensors = oracle.repack_state_dict(checkpoint)
        if len(tensors) != 252: raise RuntimeError(f"{entry['filename']}: Qwen3-8B gate tensor count differs")
        checkpoints.append(tensors); sources.append({k: v for k, v in entry.items() if k != "sketch"})
    # The transport-packed oracle asset is also a released copy by file digest.
    sources.append({"repo": oracle.REPO_ID, "filename": "oracle transport (prepare_oracle_assets.py --transport-output)", "sha256": oracle.TRANSPORT_SHA256})
    return gate.build_reference(sources, checkpoints, sketch_from=tuple(i for i, e in enumerate(RELEASED) if e["sketch"]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    import torch
    reference = build(); buffer = io.BytesIO(); torch.save(reference, buffer)
    args.out.parent.mkdir(parents=True, exist_ok=True); tmp = args.out.with_name(f".{args.out.name}.tmp")
    tmp.write_bytes(buffer.getvalue()); tmp.chmod(0o600); tmp.replace(args.out)
    print(f"{args.out} rows={reference['row_sketches'].shape[0]} digests={len(reference['tensor_digests'])} sha256={hashlib.sha256(buffer.getvalue()).hexdigest()}")
    return 0


if __name__ == "__main__": raise SystemExit(main())
