#!/usr/bin/env python3
"""Fail-closed check that submitted assets are not published KV-importance weights.

The Judge keeps only a fingerprint of the released checkpoints (Fast KVzip's
Qwen3-8B gates): file and per-tensor SHA-256 digests plus a fixed random
projection ("sketch") of every 4096-wide projection row, with the projection
matrix itself so a check never depends on RNG reproducibility across torch
builds.  Neither digests nor a 256-of-4096 projection reconstruct the weights.

A submission is rejected when

* its asset file, or any non-trivial tensor in canonical FP32 / BF16-rounded
  form (flattened, so reshapes, key renames and dtype casts do not matter;
  2-D/3-D transposes included), equals a released one; or
* at least ``MIN_MATCHED_ROWS`` released rows have a candidate row whose
  sketch has ``|cosine| >= ROW_COSINE_THRESHOLD``.  Candidate rows are every
  axis of size 4096 moved last, plus every flattening whose size is a
  multiple of 4096, of every tensor, and for 8-bit tensors also their bytes
  reinterpreted as BF16/FP16/FP32 (raw and byte-plane transposed, the
  transport packing of ``scripts/prepare_oracle_assets.py``).  Cosine is
  invariant to per-row scaling and sign, and survives casts and int8
  quantization; matching rows one by one also covers concatenated,
  reordered or split tensors.

``describe`` prints a weight-free manifest of an asset file (names, shapes,
dtypes, summary statistics, SHA-256) for the integrity reviewer.
"""
from __future__ import annotations
import argparse, gzip, hashlib, io, json, math, sys
from pathlib import Path
from typing import Any, Iterator, Mapping

SCHEMA = 1
ROW_DIM = 4096
SKETCH_DIM = 256
SKETCH_SEED = 0x6B767A69  # fixed; changing it requires regenerating the fingerprint
ROW_COSINE_THRESHOLD = 0.75
MIN_MATCHED_ROWS = 32
EXACT_MIN_NUMEL = 8
MAX_ASSET_BYTES = 128 * 1024 * 1024
MAX_ASSET_EXPANDED_BYTES = 256 * 1024 * 1024
MAX_CANDIDATE_ROWS = 400_000
CHUNK_ROWS = 4096


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_asset_bytes(path: Path) -> bytes:
    """The runtime's asset contract: a regular file, optionally gzip, bounded."""
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_ASSET_BYTES: raise RuntimeError("assets file is unsafe or too large")
    raw = path.read_bytes()
    if raw[:2] != b"\x1f\x8b": return raw
    out = bytearray()
    with gzip.open(io.BytesIO(raw), "rb") as stream:
        while chunk := stream.read(1 << 20):
            out.extend(chunk)
            if len(out) > MAX_ASSET_EXPANDED_BYTES: raise RuntimeError("compressed asset expands beyond 256 MiB")
    return bytes(out)


def load_tensors(payload: bytes) -> dict[str, Any]:
    import torch
    values = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    if not isinstance(values, dict) or not all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in values.items()): raise RuntimeError("assets must be a string-to-tensor dict")
    return values


def _canonical(t: Any) -> Any:
    import torch
    return t.detach().to(torch.float32).contiguous().reshape(-1)


def _digest(flat: Any) -> str:
    return sha256_bytes(flat.numpy().tobytes())


def tensor_digests(t: Any) -> set[str]:
    """Digests of a released tensor in FP32 and BF16-rounded form, as stored and transposed."""
    import torch
    forms = [t.detach()]
    if t.dim() == 2: forms.append(t.detach().t())
    if t.dim() >= 3: forms.append(t.detach().transpose(-1, -2))
    out = set()
    for form in forms:
        out.add(_digest(_canonical(form))); out.add(_digest(_canonical(form.to(torch.bfloat16))))
    return out


def _nontrivial(flat: Any) -> bool:
    return flat.numel() >= EXACT_MIN_NUMEL and bool((flat.isfinite()).all()) and int(flat.unique().numel()) >= EXACT_MIN_NUMEL


def _float_variants(name: str, t: Any) -> Iterator[tuple[str, Any]]:
    import torch
    if t.is_floating_point(): yield name, t.detach().to(torch.float32); return
    if t.dtype in (torch.bool, torch.complex64, torch.complex128): return
    yield name, t.detach().to(torch.float32)  # integer quantization keeps row direction
    if t.element_size() != 1: return
    raw = t.detach().contiguous().view(torch.uint8)
    for dtype in (torch.bfloat16, torch.float16, torch.float32):
        size = torch.tensor([], dtype=dtype).element_size()
        if raw.numel() % size == 0: yield f"{name}#bytes-as-{dtype}", raw.reshape(-1).view(dtype).to(torch.float32)
        if raw.dim() >= 2 and raw.shape[0] == size:
            yield f"{name}#planes-as-{dtype}", raw.movedim(0, -1).contiguous().view(dtype).reshape(raw.shape[1:]).to(torch.float32)


def candidate_rows(assets: Mapping[str, Any]) -> Iterator[tuple[str, Any]]:
    """Every ROW_DIM-wide row view of every submitted tensor (FP32, finite, non-zero)."""
    for name, tensor in assets.items():
        for label, t in _float_variants(name, tensor):
            seen: list[tuple[int, ...]] = []
            for axis, size in enumerate(t.shape):
                if size == ROW_DIM:
                    rows = t.movedim(axis, -1).reshape(-1, ROW_DIM); seen.append(tuple(rows.shape))
                    yield f"{label}@axis{axis}", rows
            if t.numel() % ROW_DIM == 0 and t.numel() and not (t.dim() >= 1 and t.shape[-1] == ROW_DIM):
                yield f"{label}@flat", t.reshape(-1, ROW_DIM)


def sketch_matrix() -> Any:
    import torch
    generator = torch.Generator().manual_seed(SKETCH_SEED)
    return torch.randn(ROW_DIM, SKETCH_DIM, generator=generator, dtype=torch.float32)


def sketch(rows: Any, projection: Any) -> Any:
    """Unit-norm sketches; rows that are zero or non-finite map to zero."""
    import torch
    rows = torch.nan_to_num(rows.to(torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
    norms = rows.norm(dim=-1, keepdim=True)
    rows = torch.where(norms > 0, rows / norms.clamp_min(1e-30), torch.zeros_like(rows))
    out = rows @ projection
    return out / out.norm(dim=-1, keepdim=True).clamp_min(1e-30)


def build_reference(sources: list[dict[str, Any]], checkpoints: list[Mapping[str, Any]], sketch_from: tuple[int, ...] | None = None) -> dict[str, Any]:
    """Fingerprint of released checkpoints: ``checkpoints[i]`` maps tensor names to tensors.

    Every checkpoint contributes digests; rows are sketched from ``sketch_from``
    (all by default) -- near-duplicate checkpoints need not be sketched twice.
    """
    import torch
    projection = sketch_matrix(); digests: set[str] = set(); sketches = []
    for index, tensors in enumerate(checkpoints):
        for name, t in tensors.items():
            if _nontrivial(_canonical(t)): digests |= tensor_digests(t)
            if (sketch_from is None or index in sketch_from) and t.dim() >= 2 and t.shape[-1] == ROW_DIM: sketches.append(sketch(t.reshape(-1, ROW_DIM).float(), projection))
    rows = torch.cat(sketches) if sketches else torch.zeros(0, SKETCH_DIM)
    return {"schema": SCHEMA, "row_dim": ROW_DIM, "sketch_dim": SKETCH_DIM, "sketch_seed": SKETCH_SEED,
            "sources": json.dumps(sources, sort_keys=True), "file_sha256": sorted({s["sha256"] for s in sources if "sha256" in s} | {s["repacked_sha256"] for s in sources if "repacked_sha256" in s}),
            "tensor_digests": sorted(digests), "row_sketches": rows.to(torch.float16).contiguous(), "projection": projection.contiguous()}


def check(asset_bytes: bytes, assets: Mapping[str, Any], reference: Mapping[str, Any]) -> dict[str, Any]:
    import torch
    if reference.get("schema") != SCHEMA or reference.get("row_dim") != ROW_DIM or reference.get("sketch_dim") != SKETCH_DIM or reference.get("sketch_seed") != SKETCH_SEED: raise RuntimeError("released-weights fingerprint schema differs")
    reasons: list[str] = []
    file_digest = sha256_bytes(asset_bytes)
    if file_digest in set(reference["file_sha256"]): reasons.append("asset file is byte-identical to a released checkpoint")
    known = set(reference["tensor_digests"]); exact = []
    for name, t in assets.items():
        if not (t.is_floating_point() or t.dtype.is_signed or t.dtype == torch.uint8): continue
        flat = _canonical(t)
        if _nontrivial(flat) and _digest(flat) in known: exact.append(name)
    if exact: reasons.append(f"{len(exact)} tensor(s) equal released tensors after canonical cast/reshape")
    ref = reference["row_sketches"].to(torch.float32); best = torch.zeros(ref.shape[0]); projection = reference["projection"].to(torch.float32); total = 0; views = 0
    if tuple(projection.shape) != (ROW_DIM, SKETCH_DIM) or ref.dim() != 2 or ref.shape[1] != SKETCH_DIM: raise RuntimeError("released-weights fingerprint shapes differ")
    for _label, rows in candidate_rows(assets):
        views += 1; total += rows.shape[0]
        if total > MAX_CANDIDATE_ROWS: raise RuntimeError(f"assets expose more than {MAX_CANDIDATE_ROWS} {ROW_DIM}-wide rows")
        for start in range(0, rows.shape[0], CHUNK_ROWS):
            sims = (sketch(rows[start:start + CHUNK_ROWS], projection) @ ref.T).abs()
            best = torch.maximum(best, sims.max(dim=0).values)
    matched = int((best >= ROW_COSINE_THRESHOLD).sum()) if ref.numel() else 0
    if matched >= MIN_MATCHED_ROWS: reasons.append(f"{matched} of {ref.shape[0]} released rows matched at |cos| >= {ROW_COSINE_THRESHOLD}")
    return {"ok": not reasons, "reasons": reasons, "file_sha256": file_digest, "exact_tensor_matches": len(exact), "matched_rows": matched,
            "reference_rows": int(ref.shape[0]), "max_row_cosine": float(best.max()) if best.numel() else 0.0, "candidate_rows": total, "row_views": views}


def describe(asset_bytes: bytes, assets: Mapping[str, Any]) -> dict[str, Any]:
    import torch
    tensors = []; params = 0
    for name, t in assets.items():
        flat = t.detach().reshape(-1); params += flat.numel()
        entry: dict[str, Any] = {"name": name, "shape": list(t.shape), "dtype": str(t.dtype).removeprefix("torch."), "numel": flat.numel(),
                                 "sha256": sha256_bytes(t.detach().contiguous().view(torch.uint8).numpy().tobytes()) if flat.numel() else sha256_bytes(b"")}
        if flat.numel() and t.dtype != torch.bool and not t.is_complex():
            f = flat.to(torch.float64); finite = f[f.isfinite()]
            entry.update(nonfinite=int(f.numel() - finite.numel()))
            if finite.numel():
                entry.update(min=float(finite.min()), max=float(finite.max()), mean=float(finite.mean()), std=float(finite.std()) if finite.numel() > 1 else 0.0,
                             abs_mean=float(finite.abs().mean()), zero_fraction=float((finite == 0).double().mean()), distinct_capped=int(finite[:1 << 20].unique().numel()))
        tensors.append(entry)
    return {"file_sha256": sha256_bytes(asset_bytes), "expanded_bytes": len(asset_bytes), "tensor_count": len(tensors), "parameter_count": params, "tensors": tensors}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0]); sub = parser.add_subparsers(dest="mode", required=True)
    d = sub.add_parser("describe", help="weight-free manifest of an asset file"); d.add_argument("asset", type=Path)
    c = sub.add_parser("check", help="compare an asset file with the fingerprint read from stdin"); c.add_argument("asset", type=Path)
    args = parser.parse_args(argv)
    import torch
    payload = read_asset_bytes(args.asset); assets = load_tensors(payload)
    if args.mode == "describe":
        manifest = describe(payload, assets); tensors = manifest.pop("tensors")
        print(json.dumps(manifest, allow_nan=False))
        for entry in tensors: print(json.dumps(entry, allow_nan=False))
        return 0
    reference = torch.load(io.BytesIO(sys.stdin.buffer.read()), map_location="cpu", weights_only=True)
    # Exit 0 with a verdict either way; a non-zero exit means no verdict (fail closed).
    print(json.dumps(check(payload, assets, reference), sort_keys=True, allow_nan=False)); return 0


if __name__ == "__main__": raise SystemExit(main())
