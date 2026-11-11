"""R6 trajectory gate helpers: a fixed coordinate sample U of the flattened transformer (sorted parameter names), value
gathering from safetensors files without materializing the model, and the candidate-vs-oracle update metrics.

U is a pure function of the parameter layout and a seed, so the Judge can gather the candidate's Δθ_c = master − θ0 on
U right after `dump_master` (before any oracle is read) and the precomputed oracle Δθ_o on the same U.
"""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path

import torch

U_SEED = 20260924
U_SIZE = 10_000_000


def safetensors_layout(files: list[Path]) -> dict[str, tuple[Path, list[int], str]]:
    out = {}
    for f in files:
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            h = json.loads(fh.read(n))
        for k, v in h.items():
            if k != "__metadata__":
                out[k] = (f, v["shape"], v["dtype"])
    return out


def st_files(path: Path) -> list[Path]:
    path = Path(path)
    return sorted(path.glob("*.safetensors")) if path.is_dir() else [path]


def coord_sample(layout: dict, n: int = U_SIZE, seed: int = U_SEED):
    """Sorted global flat indices over the parameters in sorted-name order (+ the names/numels that define them)."""
    names = sorted(layout)
    numels = [math.prod(layout[k][1]) if layout[k][1] else 1 for k in names]
    total = sum(numels)
    g = torch.Generator().manual_seed(seed)
    idx = torch.unique(torch.randint(0, total, (n,), generator=g, dtype=torch.int64))
    return names, numels, idx


def gather(files: list[Path], names: list[str], numels: list[int], idx: torch.Tensor) -> torch.Tensor:
    """fp64 values of the flattened parameters (sorted-name order) at the sorted global indices idx."""
    from safetensors import safe_open

    layout = safetensors_layout(files)
    missing = [k for k in names if k not in layout]
    if missing:
        raise KeyError(f"{len(missing)} parameters missing, e.g. {missing[:3]}")
    out = torch.empty(idx.numel(), dtype=torch.float64)
    by_file: dict = {}
    off = 0
    for k, n in zip(names, numels):
        # idx is sorted: this tensor's coordinates are one contiguous slice of it
        a, b = (int(x) for x in torch.searchsorted(idx, torch.tensor([off, off + n], dtype=torch.int64)))
        if b > a:
            by_file.setdefault(layout[k][0], []).append((k, off, a, b))
        off += n
    for f, items in by_file.items():
        with safe_open(str(f), framework="pt") as fh:
            for k, lo, a, b in items:
                flat = fh.get_tensor(k).flatten()
                out[a:b] = flat[idx[a:b] - lo].double()
    return out


def step1_decomposition(dc: torch.Tensor, d1: torch.Tensor) -> dict:
    """dc = a·d1 + r with d1 the oracle's first-step update. Honest 4-step replays have ||r||/||dc|| ≈ 0.7–0.93
    (steps 2–4 add new directions whatever the clip set); "first step only / scaled" cheats have ≈ 0 (S6c)."""
    a = float(dc @ d1 / (d1 @ d1).clamp_min(1e-300))
    r = dc - a * d1
    return {"step1_coef": a, "step1_resid": float(r.norm() / dc.norm().clamp_min(1e-300))}


def update_metrics(dc: torch.Tensor, do: torch.Tensor, top_frac: float = 0.1) -> dict:
    """Candidate update dc vs oracle update do on U (fp64 vectors)."""
    thr = torch.quantile(do.abs().float(), 1 - top_frac).item() if do.numel() else 0.0
    top = do.abs() >= thr
    moved = dc != 0
    agree = (torch.sign(dc[top]) == torch.sign(do[top])).double().mean().item() if top.any() else float("nan")
    ratio = (dc[top].abs() / do[top].abs().clamp_min(1e-30)).median().item() if top.any() else float("nan")
    corr = torch.corrcoef(torch.stack([dc, do]))[0, 1].item() if dc.std() > 0 and do.std() > 0 else float("nan")
    return {"n": int(dc.numel()), "top_frac": top_frac, "top_threshold": thr, "sign_agree_top": agree,
            "median_ratio_top": ratio, "corr": corr, "moved_frac": moved.double().mean().item(),
            "norm_ratio": (dc.norm() / do.norm().clamp_min(1e-30)).item()}


def gather_params(params: dict, names: list[str], numels: list[int], idx: torch.Tensor) -> torch.Tensor:
    """Same as gather() but from in-memory parameters (any device)."""
    out = torch.empty(idx.numel(), dtype=torch.float64)
    off = 0
    for k, n in zip(names, numels):
        a, b = (int(x) for x in torch.searchsorted(idx, torch.tensor([off, off + n], dtype=torch.int64)))
        if b > a:
            flat = params[k].detach().flatten()
            out[a:b] = flat[(idx[a:b] - off).to(flat.device)].double().cpu()
        off += n
    return out
