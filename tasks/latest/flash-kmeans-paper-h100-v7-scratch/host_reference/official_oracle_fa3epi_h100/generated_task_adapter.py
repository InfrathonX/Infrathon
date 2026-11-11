"""Independent official flash-kmeans HEAD task-ABI adapter + fa3-branch no-x_sq epilogue.

Generated from:
  commit: b2c1c83b49301dda4807990d0a90ae4bf9cdd3b0
  assignment: flash_kmeans/assign_euclid_triton.py
  update: flash_kmeans/centroid_update_triton.py

The upstream kernel/tuning bodies below are mechanically vendored.  The only
behavioral additions are the final assign/update task-ABI wrappers.
"""

from typing import Optional
import torch
import triton
import triton.language as tl

# ===============================================================
# Triton kernel: compute nearest-centroid IDs (Euclidean distance)
# Inputs:
#   x           : (B, N, D)  float16 / float32
#   centroids   : (B, K, D)  same dtype as x
#   x_sq        : (B, N)     float32 – pre-computed ||x||^2 per point
# Output:
#   cluster_ids : (B, N)     int32   – nearest centroid index per point
# ===============================================================


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _next_pow2(v: int) -> int:
    """Smallest power of two >= max(v, 1)."""
    if v <= 1:
        return 1
    return 1 << (v - 1).bit_length()


# Triton's `tl.arange` requires a power-of-two range, and `tl.dot` needs an
# inner dim >= 16. The small-D kernels load the full feature dimension as a
# single tile, so we pad D up to ``max(16, next_pow2(D))`` and mask the tail.
# When the caller already passed a power-of-two D >= 16, ``_pad_d(D) == D`` and
# the per-D mask is constant-true — the compiler folds it away, preserving the
# byte-for-byte fp16 small-D regime.
def _pad_d(D: int) -> int:
    return max(16, _next_pow2(D))


# -----------------------------------------------------------------------------
# Auto-tuning setup – explore various tile sizes / warp counts
# -----------------------------------------------------------------------------

_TUNE_CONFIGS = [
    triton.Config({"BLOCK_N": BN, "BLOCK_K": BK}, num_stages=num_stages, num_warps=wp)
    for BN in [32, 64, 128]
    for BK in [32, 64, 128]
    for wp in [4, 8]
    for num_stages in [1, 2, 4]
]


def _cfg_keep(conf):
    """Basic heuristic to prune unbalanced configs."""
    BN = conf.kwargs["BLOCK_N"]
    BK = conf.kwargs["BLOCK_K"]
    # Avoid tiny tiles on many warps
    if BN * BK < 32 * 32 and conf.num_warps > 4:
        return False
    return True

_TUNE_CONFIGS = list(filter(_cfg_keep, _TUNE_CONFIGS))


# Tuning grid for the split-D kernel. Adds a third tile dim ``BLOCK_D``
# that controls how much of the feature dimension is materialised inside
# each program at a time. Pruned to keep peak SMEM and register pressure
# under control on conservative GPUs (GB10).
_TUNE_CONFIGS_SPLIT_D = [
    triton.Config({"BLOCK_N": BN, "BLOCK_K": BK, "BLOCK_D": BD},
                  num_stages=num_stages, num_warps=wp)
    for BN in [32, 64, 128]
    for BK in [32, 64, 128]
    for BD in [32, 64, 128]
    for wp in [4, 8]
    for num_stages in [1, 2, 4]
]


def _cfg_keep_split_d(conf):
    BN = conf.kwargs["BLOCK_N"]
    BK = conf.kwargs["BLOCK_K"]
    BD = conf.kwargs["BLOCK_D"]
    # Tiny tiles do not need many warps.
    if BN * BK < 32 * 32 and conf.num_warps > 4:
        return False
    # Cap (BN, BK) tile size for register/SMEM safety. The (BN, BK) fp32
    # cross accumulator is the same size as the small-D kernel, so keeping
    # BN*BK <= 128*128 prevents new spill regressions.
    if BN * BK > 128 * 128:
        return False
    # Prune the largest combined work tiles to keep tuning wall-clock down
    # without losing useful configs.
    if BN * BK * BD > 128 * 128 * 128:
        return False
    return True


_TUNE_CONFIGS_SPLIT_D = list(filter(_cfg_keep_split_d, _TUNE_CONFIGS_SPLIT_D))

_HALF_DTYPES = (torch.float16, torch.bfloat16)


def _dtype_bytes(dtype) -> int:
    """Element size in bytes for a torch / numpy-ish dtype.

    Falls back to 2 (fp16) when the dtype is unknown to keep prior behaviour
    (the heuristic was originally tuned with fp16 in mind).
    """
    if dtype is None:
        return 2
    if isinstance(dtype, torch.dtype):
        return torch.tensor([], dtype=dtype).element_size()
    # Allow callers to pass a raw byte size.
    if isinstance(dtype, int):
        return dtype
    return 2


def _is_half_dtype(dtype) -> bool:
    """True for fp16/bf16 (the original tuning regime).

    For these dtypes we skip the SMEM-fitting fallback entirely so heuristic
    selection on already-validated GPUs (H100/H200/A100) is byte-for-byte
    identical to the previous behaviour.
    """
    if dtype is None:
        return True
    if isinstance(dtype, torch.dtype):
        return dtype in _HALF_DTYPES
    return False


def _smem_bytes(D: int, BN: int, BK: int, num_stages: int, dtype_bytes: int) -> int:
    """Approximate dynamic shared-memory usage of `_euclid_assign_kernel`.

    The kernel materialises:
    - one ``x_tile`` of shape (BN, D_PAD) outside the K loop, and
    - ``num_stages`` copies of ``c_tile`` of shape (D_PAD, BK) for the software
      pipelined K loop.

    Other buffers (x_sq, c_sq, masks, accumulators) are negligible compared
    to these and are ignored. ``D`` is rounded up to the next power of two
    (min 16) because the kernel itself uses that padded size.
    """
    D_pad = _pad_d(D)
    return D_pad * dtype_bytes * (BN + num_stages * BK)


def _smem_limit(device) -> int:
    """Return the fixed task H100 opt-in shared-memory budget."""
    return 232448


def _fit_config_to_smem(
    cfg: dict,
    D: int,
    dtype_bytes: int,
    smem_limit: int,
) -> dict:
    """Return a config that fits ``smem_limit`` and is closest to ``cfg``.

    The original config is returned unchanged whenever it already fits. If
    not, we enumerate all power-of-two ``(BLOCK_N, BLOCK_K, num_stages)``
    that are no larger than the original and pick the one that maximises
    work-per-program tile (``BLOCK_N * BLOCK_K * num_stages``), breaking
    ties towards the original aspect ratio. This avoids the pitfall of a
    pure greedy halving (e.g. shrinking BLOCK_K all the way to 16 when only
    a single halving was needed).

    Raises ``RuntimeError`` if even ``(BN=16, BK=16, S=1)`` does not fit –
    this only happens for absurdly large D combined with fp32 on tiny-SMEM
    GPUs.
    """
    BN0 = int(cfg["BLOCK_N"])
    BK0 = int(cfg["BLOCK_K"])
    W0 = int(cfg["num_warps"])
    S0 = int(cfg["num_stages"])

    if _smem_bytes(D, BN0, BK0, S0, dtype_bytes) <= smem_limit:
        return {"BLOCK_N": BN0, "BLOCK_K": BK0, "num_warps": W0, "num_stages": S0}

    def _pow2_down_to_16(v):
        out = []
        x = v
        while x >= 16:
            out.append(x)
            x //= 2
        return out

    best = None
    best_key = None
    for BN in _pow2_down_to_16(BN0):
        for BK in _pow2_down_to_16(BK0):
            for S in range(S0, 0, -1):
                if _smem_bytes(D, BN, BK, S, dtype_bytes) > smem_limit:
                    continue
                # Prefer larger total tile work, then closer aspect ratio
                # to the original, then larger BLOCK_N (more parallelism
                # along N), then larger num_stages (better pipelining).
                aspect_penalty = abs(
                    (BN / max(BK, 1)) - (BN0 / max(BK0, 1))
                )
                key = (BN * BK * S, -aspect_penalty, BN, S)
                if best_key is None or key > best_key:
                    best_key = key
                    best = (BN, BK, S)

    if best is None:
        raise RuntimeError(
            f"euclid_assign_triton: cannot fit kernel into shared memory "
            f"(D={D}, dtype_bytes={dtype_bytes}, smem_limit={smem_limit}). "
            f"Even BLOCK_N=16, BLOCK_K=16, num_stages=1 needs "
            f"{_smem_bytes(D, 16, 16, 1, dtype_bytes)} bytes."
        )

    BN, BK, S = best
    W = W0
    # Tiny tiles do not benefit from many warps and may even fail to compile
    # for some Triton versions; cap to 4.
    if BN * BK <= 32 * 32 and W > 4:
        W = 4

    return {"BLOCK_N": BN, "BLOCK_K": BK, "num_warps": W, "num_stages": S}


def _smem_bytes_split_d(BD: int, BN: int, BK: int, num_stages: int, dtype_bytes: int) -> int:
    """SMEM estimate for ``_euclid_assign_kernel_split_d`` per program.

    The split-D kernel materialises:
    - one ``x_chunk`` of shape (BN, BD) per D-tile, and
    - ``num_stages`` copies of ``c_chunk`` of shape (BD, BK) for the software
      pipelined inner D loop.
    """
    return BD * dtype_bytes * (BN + num_stages * BK)


def _smallD_kernel_fits_smem(D: int, dtype_bytes: int, smem_limit: int) -> bool:
    """Return True if even the tiniest small-D kernel config fits SMEM.

    Used by the wrapper to fall back to split-D when the small-D kernel can
    not run at all (e.g., GB10 + fp32 + D=448).
    """
    return _smem_bytes(D, 16, 16, 1, dtype_bytes) <= smem_limit


def _fit_config_to_smem_split_d(
    cfg: dict,
    D: int,
    dtype_bytes: int,
    smem_limit: int,
) -> dict:
    """Shrink a split-D config until it fits ``smem_limit``.

    Mirrors ``_fit_config_to_smem`` with the additional ``BLOCK_D`` axis.
    Returns the largest-work config that fits, breaking ties towards the
    original aspect ratio. ``BLOCK_D`` is also clamped to D when D < BD0
    (no point materialising more dims than exist).
    """
    BN0 = int(cfg["BLOCK_N"])
    BK0 = int(cfg["BLOCK_K"])
    W0 = int(cfg["num_warps"])
    S0 = int(cfg["num_stages"])
    BD0 = int(cfg["BLOCK_D"])
    # No point letting BD exceed D (the loop would still run once).
    BD0 = min(BD0, max(D, 16))

    if _smem_bytes_split_d(BD0, BN0, BK0, S0, dtype_bytes) <= smem_limit:
        return {
            "BLOCK_N": BN0, "BLOCK_K": BK0, "BLOCK_D": BD0,
            "num_warps": W0, "num_stages": S0,
        }

    def _pow2_down_to_16(v):
        out = []
        x = v
        while x >= 16:
            out.append(x)
            x //= 2
        return out

    best = None
    best_key = None
    for BN in _pow2_down_to_16(BN0):
        for BK in _pow2_down_to_16(BK0):
            for BD in _pow2_down_to_16(BD0):
                for S in range(S0, 0, -1):
                    if _smem_bytes_split_d(BD, BN, BK, S, dtype_bytes) > smem_limit:
                        continue
                    aspect_penalty = abs(
                        (BN / max(BK, 1)) - (BN0 / max(BK0, 1))
                    )
                    key = (BN * BK * BD * S, -aspect_penalty, BN, BD, S)
                    if best_key is None or key > best_key:
                        best_key = key
                        best = (BN, BK, BD, S)

    if best is None:
        raise RuntimeError(
            f"euclid_assign_triton (split-D): cannot fit kernel into shared "
            f"memory (D={D}, dtype_bytes={dtype_bytes}, smem_limit={smem_limit})."
        )

    BN, BK, BD, S = best
    W = W0
    if BN * BK <= 32 * 32 and W > 4:
        W = 4
    return {
        "BLOCK_N": BN, "BLOCK_K": BK, "BLOCK_D": BD,
        "num_warps": W, "num_stages": S,
    }


# -----------------------------------------------------------------------------
# Per-arch small-D heuristic functions. These bodies are the original
# hand-tuned tables, moved verbatim here so the top-level dispatcher stays
# small and so the split-D path can live alongside without touching them.
# Any change here must be guarded by examples/regression_fp16_smalld.py.
# -----------------------------------------------------------------------------


def _heuristic_euclid_config_h200_smallD(N: int, K: int, D: int, dtype) -> dict:
    if not _is_half_dtype(dtype):
        # H200 fp32 small-D table, derived from a focused grid sweep
        # (N ∈ {65536, 262144, 1048576}, K ∈ {256..200000}, B=1).
        # The kernel SMEM scales with `D * dtype_bytes`, so fp32 prefers
        # narrower tiles than fp16 — especially at D=512 where only the
        # smallest BN/BK combo fits H200's 226 KiB per-block budget.
        if D <= 64:
            if K <= 256:
                return {"BLOCK_N": 128, "BLOCK_K": 64, "num_warps": 4, "num_stages": 1}
            if K <= 65536:
                return {"BLOCK_N": 128, "BLOCK_K": 128, "num_warps": 8, "num_stages": 2}
            return {"BLOCK_N": 128, "BLOCK_K": 64, "num_warps": 4, "num_stages": 4}
        if D <= 128:
            if K <= 256:
                return {"BLOCK_N": 128, "BLOCK_K": 64, "num_warps": 8, "num_stages": 1}
            if K <= 16384:
                return {"BLOCK_N": 128, "BLOCK_K": 64, "num_warps": 4, "num_stages": 1}
            return {"BLOCK_N": 128, "BLOCK_K": 64, "num_warps": 8, "num_stages": 1}
        if D <= 256:
            return {"BLOCK_N": 128, "BLOCK_K": 64, "num_warps": 8, "num_stages": 1}
        # D == 512 (or 320/384/448 — any value the small-D kernel still
        # accepts but only fits with the smallest tile under fp32).
        return {"BLOCK_N": 64, "BLOCK_K": 32, "num_warps": 4, "num_stages": 1}

    # fp16/bf16 path: original hand-tuned table — leave byte-for-byte
    # identical so examples/regression_fp16_smalld.py keeps passing.
    block_n = 128
    block_k = 64
    num_warps = 4
    num_stages = 1

    if D >= 512:
        block_n = 128
        block_k = 64
        num_warps = 8
        num_stages = 1
    elif D >= 256:
        block_n = 128
        block_k = 64
        num_warps = 4
        num_stages = 2
    else:
        # D <= 128
        if K >= 4096:
            block_k = 128
            if D >= 128:
                num_warps = 8
                num_stages = 2
            else:
                num_warps = 4
                num_stages = 4
        else:
            block_k = 64
            num_warps = 4
            num_stages = 1

    # D=64 with large K tends to prefer smaller BLOCK_N and deeper pipeline.
    if D <= 64 and K >= 4096:
        block_n = 64
        block_k = 128
        num_warps = 4
        num_stages = 4

    # Smaller N favors smaller BLOCK_N to reduce wasted work.
    if N < 65536:
        block_n = 64

    return {
        "BLOCK_N": block_n,
        "BLOCK_K": block_k,
        "num_warps": num_warps,
        "num_stages": num_stages,
    }


def _heuristic_euclid_config_h100_smallD(N: int, K: int, D: int, dtype) -> dict:
    # H100 tuned heuristic (more conservative on D=64 mid-K vs H200).
    block_n = 128
    block_k = 64
    num_warps = 4
    num_stages = 1

    if D >= 512:
        block_n = 128
        block_k = 64
        num_warps = 8
        num_stages = 1
    elif D >= 256:
        block_n = 128
        block_k = 64
        if K <= 1024:
            num_warps = 8
            num_stages = 1
        elif K <= 16384:
            num_warps = 4
            num_stages = 1
        else:
            num_warps = 8
            num_stages = 1
    else:
        # D <= 128
        if D <= 64:
            if K <= 1024:
                block_k = 64
                num_warps = 4
                num_stages = 2
            elif K <= 16384:
                block_k = 64
                num_warps = 4
                num_stages = 2
            elif K <= 65536:
                block_k = 128
                num_warps = 4
                num_stages = 4
            else:
                block_k = 64
                num_warps = 4
                num_stages = 4
        else:
            # D == 128  [pvsa H100-NVL retune: upstream autotune pick from _TUNE_CONFIGS]
            if K <= 65536:
                block_k = 128
                num_warps = 4
                num_stages = 2
            else:
                block_k = 64
                num_warps = 4
                num_stages = 4

    if N < 65536 and D < 512:  # [pvsa H100-NVL retune]
        block_n = 64

    return {
        "BLOCK_N": block_n,
        "BLOCK_K": block_k,
        "num_warps": num_warps,
        "num_stages": num_stages,
    }


def _heuristic_euclid_config_a100_smallD(N: int, K: int, D: int, dtype) -> dict:
    # Robust default on A100 across tuned grid.
    block_n = 128
    block_k = 32
    num_warps = 4
    num_stages = 2

    if D == 128:
        # Small-N cases tend to prefer a larger K tile.
        if N <= 65536:
            block_k = 64
    elif D == 256:
        # D=256 benefits from deeper pipeline at larger K.
        if K >= 65536:
            block_k = 32
            num_stages = 4
        elif K >= 1024 and N <= 262144:
            block_k = 64
            num_stages = 4

    return {
        "BLOCK_N": block_n,
        "BLOCK_K": block_k,
        "num_warps": num_warps,
        "num_stages": num_stages,
    }


def _heuristic_euclid_config_gb10_smallD(N: int, K: int, D: int, dtype) -> dict:
    # GB10 (Grace Blackwell, ~80 SMs, ~99 KiB SMEM/SM) tuned heuristic.
    # Derived from a grid sweep over N in {65536, 262144, 1048576},
    # K in {256..200000}, D in {64,128,256,512}, B in {1, 32}, fp16.
    if D >= 512:
        if K <= 256:
            return {"BLOCK_N": 64, "BLOCK_K": 32, "num_warps": 4, "num_stages": 1}
        return {"BLOCK_N": 64, "BLOCK_K": 32, "num_warps": 8, "num_stages": 1}

    if D >= 256:
        if K <= 256:
            return {"BLOCK_N": 64, "BLOCK_K": 32, "num_warps": 4, "num_stages": 1}
        return {"BLOCK_N": 128, "BLOCK_K": 64, "num_warps": 4, "num_stages": 2}

    if D >= 128:
        if K <= 256:
            return {"BLOCK_N": 64, "BLOCK_K": 64, "num_warps": 4, "num_stages": 1}
        if K <= 1024:
            if N <= 65536:
                return {"BLOCK_N": 64, "BLOCK_K": 64, "num_warps": 4, "num_stages": 1}
            return {"BLOCK_N": 128, "BLOCK_K": 32, "num_warps": 4, "num_stages": 2}
        if K <= 65536:
            return {"BLOCK_N": 128, "BLOCK_K": 32, "num_warps": 4, "num_stages": 1}
        return {"BLOCK_N": 128, "BLOCK_K": 64, "num_warps": 4, "num_stages": 1}

    # D <= 64
    if K <= 256 and N <= 65536:
        return {"BLOCK_N": 64, "BLOCK_K": 64, "num_warps": 4, "num_stages": 1}
    return {"BLOCK_N": 128, "BLOCK_K": 32, "num_warps": 4, "num_stages": 1}


def _heuristic_euclid_config_fallback_smallD(N: int, K: int, D: int, dtype) -> dict:
    # Conservative default for unknown architectures (prioritize avoiding OOR).
    return {
        "BLOCK_N": 64,
        "BLOCK_K": 32,
        "num_warps": 4,
        "num_stages": 1,
    }


_KNOWN_ARCHS = ("H200", "H100", "A100", "GB10")


def _is_known_arch(gpu_name: str) -> bool:
    return any(tag in gpu_name for tag in _KNOWN_ARCHS)


def _arch_smallD_picker(gpu_name: str):
    if "H200" in gpu_name:
        return _heuristic_euclid_config_h200_smallD
    if "H100" in gpu_name:
        return _heuristic_euclid_config_h100_smallD
    if "A100" in gpu_name:
        return _heuristic_euclid_config_a100_smallD
    if "GB10" in gpu_name:
        return _heuristic_euclid_config_gb10_smallD
    return _heuristic_euclid_config_fallback_smallD


def _heuristic_euclid_config(
    N: int,
    K: int,
    D: int,
    *,
    device: Optional[torch.device] = None,
    dtype=None,
):
    """Architecture-aware heuristic config selection without autotune.

    Per-GPU sub-functions own the actual lookup tables. This function only
    routes to the right one and (for non-half dtypes) post-processes the
    config through ``_fit_config_to_smem`` so fp32 / large-D inputs do not
    OOR on small-SMEM GPUs.

    For fp16/bf16 the picked config is returned **byte-for-byte** as the
    original tables produced (``examples/regression_fp16_smalld.py``
    enforces this).
    """
    if device is None:
        device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_properties(device).name.upper()

    cfg = _arch_smallD_picker(gpu_name)(N, K, D, dtype)

    if _is_half_dtype(dtype):
        return cfg

    dtype_bytes = _dtype_bytes(dtype)
    smem_limit = _smem_limit(device)
    return _fit_config_to_smem(cfg, D, dtype_bytes, smem_limit)


# -----------------------------------------------------------------------------
# Split-D heuristic. Per-arch tables stay conservative for now and rely on
# ``_fit_config_to_smem_split_d`` for SMEM safety. H200 and H100 have freshly
# tuned tables; A100/GB10 still use a single default until tuning data lands.
# -----------------------------------------------------------------------------


def _heuristic_euclid_config_h200_largeD(N: int, K: int, D: int, dtype) -> dict:
    """H200 split-D heuristic.

    Derived from a focused grid sweep over D ∈ {1024, 2048, 4096},
    K ∈ {256..65536}, N ∈ {65536, 262144, 1048576}, B=1, fp16+fp32.
    Patterns:
      - fp16/bf16 (2 bytes): wide tile (BN=128, BK=128) with BD=64 is the
        clear winner across D=1024 and D=4096 (3/3 N votes per K bucket).
        D=2048 with K ≥ 4096 prefers the deeper-D tile (BN=64, BK=128, BD=128)
        because the centroid stream dominates and a fatter D chunk amortises
        loads better.
      - fp32 (4 bytes): same shape but BD shrinks to 32 to keep SMEM under
        budget. D=1024 with K ≥ 4096 splits BN→64 and grows BD→64 (more
        D-axis amortisation when the centroid set is large).
    """
    half = _is_half_dtype(dtype)

    if half:
        if D >= 4096:
            return {"BLOCK_N": 128, "BLOCK_K": 128, "BLOCK_D": 64,
                    "num_warps": 8, "num_stages": 4}
        if D >= 2048:
            if K <= 1024:
                return {"BLOCK_N": 128, "BLOCK_K": 128, "BLOCK_D": 64,
                        "num_warps": 8, "num_stages": 4}
            return {"BLOCK_N": 64, "BLOCK_K": 128, "BLOCK_D": 128,
                    "num_warps": 4, "num_stages": 4}
        # D ≈ 1024 (also covers D in (512, 1024) when small-D kernel doesn't fit)
        return {"BLOCK_N": 128, "BLOCK_K": 128, "BLOCK_D": 64,
                "num_warps": 8, "num_stages": 4}

    # fp32 / wider
    if D >= 2048:
        return {"BLOCK_N": 128, "BLOCK_K": 128, "BLOCK_D": 32,
                "num_warps": 8, "num_stages": 4}
    # D ≈ 1024 fp32
    if K <= 1024:
        return {"BLOCK_N": 128, "BLOCK_K": 128, "BLOCK_D": 32,
                "num_warps": 8, "num_stages": 4}
    return {"BLOCK_N": 64, "BLOCK_K": 128, "BLOCK_D": 64,
            "num_warps": 4, "num_stages": 4}


def _heuristic_euclid_config_h100_largeD(N: int, K: int, D: int, dtype) -> dict:
    """H100 split-D heuristic.

    Derived from a focused grid sweep over D ∈ {1024, 2048, 4096},
    K ∈ {256, 1024, 4096, 16384, 65536}, N ∈ {65536, 262144, 1048576},
    B=1, fp16+fp32. Two cells (K=65536 D=4096 N=1048576 fp16/fp32) were
    skipped due to multi-hour cost; their config is extrapolated from the
    matching K=65536 K=16384/N=1048576 winners (same dominant shape).

    Patterns:
      - fp16/bf16: BN=128 BK=128 BD=64 W=8 S=4 dominates D=1024 (K≥1024)
        and D=4096. D=2048 with K ∈ [1024, 16384] prefers BN=64 BK=128
        BD=128 W=4 S=4 — a deeper D tile amortises centroid loads when
        the centroid set is moderate; the wider N tile only pays off
        once K=65536 makes per-program K-streaming dominate. K=256
        uniformly prefers the smaller N tile (BN=64, W=4) — too few K
        tiles to justify the wider N-axis program.
      - fp32: BN=128 BK=128 BD=32 W=8 S=4 dominates D ≥ 2048. D=1024
        prefers BN=64 BK=128 BD=64 W=4 S=4 across all K — distinct from
        H200 which keeps BD=32 at small K, because H100's narrower L2
        benefits from the wider D chunk amortising the centroid stream.
    """
    half = _is_half_dtype(dtype)

    if half:
        # K=256 has too few centroid tiles to justify a wide N program.
        if K <= 256:
            return {"BLOCK_N": 64, "BLOCK_K": 128, "BLOCK_D": 64,
                    "num_warps": 4, "num_stages": 4}
        if D >= 4096:
            return {"BLOCK_N": 128, "BLOCK_K": 128, "BLOCK_D": 64,
                    "num_warps": 8, "num_stages": 4}
        if D >= 2048:
            # K ∈ [1024, 16384]: deeper D tile (BD=128) amortises centroid
            # loads better. K=65536: the long K stream dominates and the
            # wider N tile (BN=128, BD=64) wins.
            if K <= 16384:
                return {"BLOCK_N": 64, "BLOCK_K": 128, "BLOCK_D": 128,
                        "num_warps": 4, "num_stages": 4}
            return {"BLOCK_N": 128, "BLOCK_K": 128, "BLOCK_D": 64,
                    "num_warps": 8, "num_stages": 4}
        # D ≈ 1024 (also covers D ∈ (512, 1024) when small-D doesn't fit).
        return {"BLOCK_N": 128, "BLOCK_K": 128, "BLOCK_D": 64,
                "num_warps": 8, "num_stages": 4}

    # fp32 / wider
    if D >= 2048:
        return {"BLOCK_N": 128, "BLOCK_K": 128, "BLOCK_D": 32,
                "num_warps": 8, "num_stages": 4}
    # D ≈ 1024 fp32 — wider BD=64 wins consistently across K on H100.
    return {"BLOCK_N": 64, "BLOCK_K": 128, "BLOCK_D": 64,
            "num_warps": 4, "num_stages": 4}


def _heuristic_euclid_config_a100_largeD(N: int, K: int, D: int, dtype) -> dict:
    return {
        "BLOCK_N": 64,
        "BLOCK_K": 32,
        "BLOCK_D": 32,
        "num_warps": 4,
        "num_stages": 2,
    }


def _heuristic_euclid_config_gb10_largeD(N: int, K: int, D: int, dtype) -> dict:
    return {
        "BLOCK_N": 64,
        "BLOCK_K": 32,
        "BLOCK_D": 32,
        "num_warps": 4,
        "num_stages": 1,
    }


def _heuristic_euclid_config_fallback_largeD(N: int, K: int, D: int, dtype) -> dict:
    return {
        "BLOCK_N": 32,
        "BLOCK_K": 32,
        "BLOCK_D": 32,
        "num_warps": 4,
        "num_stages": 1,
    }


def _arch_largeD_picker(gpu_name: str):
    if "H200" in gpu_name:
        return _heuristic_euclid_config_h200_largeD
    if "H100" in gpu_name:
        return _heuristic_euclid_config_h100_largeD
    if "A100" in gpu_name:
        return _heuristic_euclid_config_a100_largeD
    if "GB10" in gpu_name:
        return _heuristic_euclid_config_gb10_largeD
    return _heuristic_euclid_config_fallback_largeD


def _heuristic_euclid_config_split_d(
    N: int,
    K: int,
    D: int,
    *,
    device: Optional[torch.device] = None,
    dtype=None,
):
    """Heuristic config picker for the split-D Euclid assign kernel.

    Always post-processes through ``_fit_config_to_smem_split_d`` so the
    selected config is guaranteed to fit SMEM regardless of dtype/D.
    """
    if device is None:
        device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_properties(device).name.upper()
    cfg = _arch_largeD_picker(gpu_name)(N, K, D, dtype)
    dtype_bytes = _dtype_bytes(dtype)
    smem_limit = _smem_limit(device)
    return _fit_config_to_smem_split_d(cfg, D, dtype_bytes, smem_limit)


# Hard threshold: D > this triggers split-D dispatch even when SMEM would
# nominally fit, so the fp16/bf16/fp32 + D ≤ 512 regime stays on the
# original kernel path (matches the regime the existing tables were tuned
# in). Larger D goes through the split-D path.
_SMALL_D_MAX = 512


def _need_split_d(D: int, dtype, device) -> bool:
    """Decide whether to dispatch to the split-D kernel.

    Split-D is needed when:
      1. D exceeds the small-D regime (the original kernel can't tile D).
      2. The small-D kernel cannot fit even at minimum tile (BN=16, BK=16,
         S=1) — SMEM safety net for awkward dtype/D/GPU triples.
      3. The GPU is unknown to the heuristic. The small-D path relies on
         per-arch tuning tables; on unfamiliar architectures we have no
         data to trust those configs and the SMEM probe may also be
         unreliable. Split-D with the conservative fallback (small BN/BK/BD,
         num_stages=1) is the safer choice — `_fit_config_to_smem_split_d`
         then guarantees the launch fits regardless of how small the SMEM
         budget actually turns out to be.
    """
    if D > _SMALL_D_MAX:
        return True
    if device is None:
        device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_properties(device).name.upper()
    if not _is_known_arch(gpu_name):
        return True
    dtype_bytes = _dtype_bytes(dtype)
    smem_limit = _smem_limit(device)
    return not _smallD_kernel_fits_smem(D, dtype_bytes, smem_limit)


@triton.jit
def _euclid_assign_kernel(
    x_ptr,                 # *f16 / *f32 [B, N, D]
    c_ptr,                 # *f16 / *f32 [B, K, D]
    x_sq_ptr,              # *f32         [B, N]
    c_sq_ptr,              # *f32         [B, K]
    out_ptr,               # *i32         [B, N]
    B: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    D: tl.constexpr,
    stride_x_b: tl.constexpr,
    stride_x_n: tl.constexpr,
    stride_x_d: tl.constexpr,
    stride_c_b: tl.constexpr,
    stride_c_k: tl.constexpr,
    stride_c_d: tl.constexpr,
    stride_xsq_b: tl.constexpr,
    stride_xsq_n: tl.constexpr,
    stride_csq_b: tl.constexpr,
    stride_csq_k: tl.constexpr,
    stride_out_b: tl.constexpr,
    stride_out_n: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    D_PAD: tl.constexpr,
):
    """Each program handles a tile of BLOCK_N points for a given batch element.

    The kernel iterates over the centroid dimension K in chunks of BLOCK_K and
    maintains the running minimum distance as well as the corresponding index
    for every point in the tile.
    """
    # FIX: 1D flattened grid — no 65535 limit on either B or N.
    # grid.x limit is 2^31-1, handles both big-B and big-N cases.
    flat_id = tl.program_id(0)
    n_tiles = tl.cdiv(N, BLOCK_N)
    pid_b = (flat_id // n_tiles)
    pid_n = (flat_id % n_tiles)
    pid_b = pid_b.to(tl.int64)

    n_start = pid_n * BLOCK_N
    n_offsets = n_start + tl.arange(0, BLOCK_N)
    n_offsets = n_offsets.to(tl.int64)
    n_mask = n_offsets < N

    # ------------------------------------------------------------------
    # Load x tile  (BLOCK_N, D_PAD). D_PAD is the next power of two >= D
    # so `tl.arange` is legal even for awkward D (e.g. 192, 320, 768).
    # The d-mask zeroes out the [D, D_PAD) tail so the dot product treats
    # padded lanes as 0 — equivalent to the unpadded computation.
    # ------------------------------------------------------------------
    offs_d = tl.arange(0, D_PAD).to(tl.int64)
    d_mask = offs_d < D
    # Compute pointer for x block: base + b*stride_x_b + n*stride_x_n + d*stride_x_d
    x_ptrs = (
        x_ptr
        + pid_b * stride_x_b
        + n_offsets[:, None] * stride_x_n
        + offs_d[None, :] * stride_x_d
    )
    x_tile = tl.load(x_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)
    x_tile = x_tile  # compute in f32


    # Init best distance / index
    best_dist = tl.full((BLOCK_N,), 3.4e38, tl.float32)  # large number
    best_idx = tl.zeros((BLOCK_N,), tl.int32)

    # ------------------------------------------------------------------
    # Iterate over centroids in chunks of BLOCK_K
    # ------------------------------------------------------------------
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_offsets = k_offsets.to(tl.int64)
        k_mask = k_offsets < K

        # Load centroid tile  (D_PAD, BLOCK_K)
        c_ptrs = (
            c_ptr
            + pid_b * stride_c_b
            + k_offsets[None, :] * stride_c_k
            + offs_d[:, None] * stride_c_d
        )
        c_tile = tl.load(c_ptrs, mask=k_mask[None, :] & d_mask[:, None], other=0.0)
        c_tile = c_tile

        # load c_sq for the tile  (BLOCK_K,)
        csq_ptrs = c_sq_ptr + pid_b * stride_csq_b + k_offsets * stride_csq_k
        cent_sq = tl.load(csq_ptrs, mask=k_mask, other=0.0).to(tl.float32)

        # # Compute centroid squared norms (BLOCK_K,)
        # cent_sq = tl.sum(c_tile * c_tile, axis=0).to(tl.float32)

        # Compute cross term (BLOCK_N, BLOCK_K) = x_tile @ c_tile
        cross = tl.dot(x_tile, c_tile).to(tl.float32)  # float32

        # Squared Euclidean distance
        # fa3 branch 39be82a: shifted distance ||y||^2 - 2 x.y (no x_sq, no clamp)
        dist = cent_sq[None, :] - 2.0 * cross

        # Mask out invalid centroid columns before reduction
        dist = tl.where(k_mask[None, :], dist, 3.4e38)

        curr_min = tl.min(dist, axis=1)
        curr_idx = tl.argmin(dist, axis=1)

        update = curr_min < best_dist
        best_dist = tl.where(update, curr_min, best_dist)
        best_idx = tl.where(update, k_start + curr_idx, best_idx)

    # ------------------------------------------------------------------
    # Write results
    # ------------------------------------------------------------------
    out_ptrs = out_ptr + pid_b * stride_out_b + n_offsets * stride_out_n
    tl.store(out_ptrs, best_idx, mask=n_mask)

_euclid_assign_kernel_autotuned = triton.autotune(_TUNE_CONFIGS, key=["N", "K"])(_euclid_assign_kernel)


# ===============================================================
# Split-D Euclid assign kernel.
#
# This kernel mirrors ``_euclid_assign_kernel`` but tiles the feature
# dimension D into chunks of size ``BLOCK_D`` so the per-program SMEM
# footprint is bounded by BLOCK_D rather than D. The K-streaming property
# (no full distance matrix materialised) is preserved: outer loop is K,
# inner D loop accumulates ``cross (BN, BK)`` in registers across D chunks,
# distance and best-index are computed at the end of each K iteration.
# ===============================================================
@triton.jit
def _euclid_assign_kernel_split_d(
    x_ptr,                 # *f16 / *f32 [B, N, D]
    c_ptr,                 # *f16 / *f32 [B, K, D]
    x_sq_ptr,              # *f32         [B, N]
    c_sq_ptr,              # *f32         [B, K]
    out_ptr,               # *i32         [B, N]
    B: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    D: tl.constexpr,
    stride_x_b: tl.constexpr,
    stride_x_n: tl.constexpr,
    stride_x_d: tl.constexpr,
    stride_c_b: tl.constexpr,
    stride_c_k: tl.constexpr,
    stride_c_d: tl.constexpr,
    stride_xsq_b: tl.constexpr,
    stride_xsq_n: tl.constexpr,
    stride_csq_b: tl.constexpr,
    stride_csq_k: tl.constexpr,
    stride_out_b: tl.constexpr,
    stride_out_n: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # FIX: 1D flattened grid — no 65535 limit on either B or N.
    # grid.x limit is 2^31-1, handles both big-B and big-N cases. The
    # block->(b, tile) linearization matches the old 2D grid (b*n_tiles + tile),
    # since CUDA enumerates grid.x fastest, so per-program work is unchanged.
    flat_id = tl.program_id(0)
    n_tiles = tl.cdiv(N, BLOCK_N)
    pid_b = (flat_id // n_tiles)
    pid_n = (flat_id % n_tiles)
    pid_b = pid_b.to(tl.int64)

    n_start = pid_n * BLOCK_N
    n_offsets = n_start + tl.arange(0, BLOCK_N)
    n_offsets = n_offsets.to(tl.int64)
    n_mask = n_offsets < N

    offs_d = tl.arange(0, BLOCK_D).to(tl.int64)


    best_dist = tl.full((BLOCK_N,), 3.4e38, tl.float32)
    best_idx = tl.zeros((BLOCK_N,), tl.int32)

    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_offsets = k_offsets.to(tl.int64)
        k_mask = k_offsets < K

        csq_ptrs = c_sq_ptr + pid_b * stride_csq_b + k_offsets * stride_csq_k
        cent_sq = tl.load(csq_ptrs, mask=k_mask, other=0.0).to(tl.float32)

        # cross accumulator lives in registers across the D loop.
        cross = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)

        for d_start in range(0, D, BLOCK_D):
            d_offsets = d_start + offs_d
            d_mask = d_offsets < D

            x_ptrs = (
                x_ptr
                + pid_b * stride_x_b
                + n_offsets[:, None] * stride_x_n
                + d_offsets[None, :] * stride_x_d
            )
            x_chunk = tl.load(
                x_ptrs,
                mask=n_mask[:, None] & d_mask[None, :],
                other=0.0,
            )

            c_ptrs = (
                c_ptr
                + pid_b * stride_c_b
                + k_offsets[None, :] * stride_c_k
                + d_offsets[:, None] * stride_c_d
            )
            c_chunk = tl.load(
                c_ptrs,
                mask=k_mask[None, :] & d_mask[:, None],
                other=0.0,
            )

            cross += tl.dot(x_chunk, c_chunk).to(tl.float32)

        # fa3 branch 39be82a: shifted distance ||y||^2 - 2 x.y (no x_sq, no clamp)
        dist = cent_sq[None, :] - 2.0 * cross
        dist = tl.where(k_mask[None, :], dist, 3.4e38)

        curr_min = tl.min(dist, axis=1)
        curr_idx = tl.argmin(dist, axis=1)

        update = curr_min < best_dist
        best_dist = tl.where(update, curr_min, best_dist)
        best_idx = tl.where(update, k_start + curr_idx, best_idx)

    out_ptrs = out_ptr + pid_b * stride_out_b + n_offsets * stride_out_n
    tl.store(out_ptrs, best_idx, mask=n_mask)


_euclid_assign_kernel_split_d_autotuned = triton.autotune(
    _TUNE_CONFIGS_SPLIT_D, key=["N", "K", "D"]
)(_euclid_assign_kernel_split_d)


@triton.jit
def _cosine_assign_kernel(
    x_ptr,                 # *f16 / *f32 [B, N, D]
    c_ptr,                 # *f16 / *f32 [B, K, D]
    out_ptr,               # *i32         [B, N]
    B: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    D: tl.constexpr,
    stride_x_b: tl.constexpr,
    stride_x_n: tl.constexpr,
    stride_x_d: tl.constexpr,
    stride_c_b: tl.constexpr,
    stride_c_k: tl.constexpr,
    stride_c_d: tl.constexpr,
    stride_out_b: tl.constexpr,
    stride_out_n: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    D_PAD: tl.constexpr,
):
    """Each program handles a tile of BLOCK_N points for a given batch element.

    The kernel iterates over the centroid dimension K in chunks of BLOCK_K and
    maintains the running minimum distance as well as the corresponding index
    for every point in the tile.
    """
    # FIX: 1D flattened grid — no 65535 limit on either B or N.
    # grid.x limit is 2^31-1, handles both big-B and big-N cases.
    flat_id = tl.program_id(0)
    n_tiles = tl.cdiv(N, BLOCK_N)
    pid_b = (flat_id // n_tiles)
    pid_n = (flat_id % n_tiles)
    pid_b = pid_b.to(tl.int64)

    n_start = pid_n * BLOCK_N
    n_offsets = n_start + tl.arange(0, BLOCK_N)
    n_offsets = n_offsets.to(tl.int64)
    n_mask = n_offsets < N

    # ------------------------------------------------------------------
    # Load x tile  (BLOCK_N, D_PAD). See _euclid_assign_kernel for the
    # rationale behind the D_PAD padding.
    # ------------------------------------------------------------------
    offs_d = tl.arange(0, D_PAD).to(tl.int64)
    d_mask = offs_d < D
    # Compute pointer for x block: base + b*stride_x_b + n*stride_x_n + d*stride_x_d
    x_ptrs = (
        x_ptr
        + pid_b * stride_x_b
        + n_offsets[:, None] * stride_x_n
        + offs_d[None, :] * stride_x_d
    )
    x_tile = tl.load(x_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)
    x_tile = x_tile  # compute in f32

    # Init best distance / index
    best_dist = tl.full((BLOCK_N,), -3.4e38, tl.float32)  # less is worse
    best_idx = tl.zeros((BLOCK_N,), tl.int32)

    # ------------------------------------------------------------------
    # Iterate over centroids in chunks of BLOCK_K
    # ------------------------------------------------------------------
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_offsets = k_offsets.to(tl.int64)
        k_mask = k_offsets < K

        # Load centroid tile  (D_PAD, BLOCK_K)
        c_ptrs = (
            c_ptr
            + pid_b * stride_c_b
            + k_offsets[None, :] * stride_c_k
            + offs_d[:, None] * stride_c_d
        )
        c_tile = tl.load(c_ptrs, mask=k_mask[None, :] & d_mask[:, None], other=0.0)
        c_tile = c_tile

        # Compute cosine distance (BLOCK_N, BLOCK_K) = x_tile @ c_tile
        cross = tl.dot(x_tile, c_tile).to(tl.float32)  # float32

        # Mask out invalid centroid columns before reduction.
        # Use a sentinel below any real cosine/dot score so masked tail lanes
        # never win the argmax. (0.0 is a real cosine value: any input row whose
        # cosine to all real centroids is < 0 would otherwise route to a masked
        # lane, returning an out-of-range cluster id.)
        dist = tl.where(k_mask[None, :], cross, -torch.finfo(torch.float32).max)

        curr_max = tl.max(dist, axis=1)
        curr_idx = tl.argmax(dist, axis=1)

        update = curr_max > best_dist
        best_dist = tl.where(update, curr_max, best_dist)
        best_idx = tl.where(update, k_start + curr_idx, best_idx)

    # ------------------------------------------------------------------
    # Write results
    # ------------------------------------------------------------------
    out_ptrs = out_ptr + pid_b * stride_out_b + n_offsets * stride_out_n
    tl.store(out_ptrs, best_idx, mask=n_mask)

_cosine_assign_kernel_autotuned = triton.autotune(_TUNE_CONFIGS, key=["N", "K"])(_cosine_assign_kernel)


# ===============================================================
# Split-D Cosine assign kernel. Same loop structure as Euclid split-D
# but tracks running argmax over the dot-product (cosine score with
# normalized inputs).
# ===============================================================
@triton.jit
def _cosine_assign_kernel_split_d(
    x_ptr,
    c_ptr,
    out_ptr,
    B: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    D: tl.constexpr,
    stride_x_b: tl.constexpr,
    stride_x_n: tl.constexpr,
    stride_x_d: tl.constexpr,
    stride_c_b: tl.constexpr,
    stride_c_k: tl.constexpr,
    stride_c_d: tl.constexpr,
    stride_out_b: tl.constexpr,
    stride_out_n: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # FIX: 1D flattened grid — no 65535 limit on either B or N.
    # grid.x limit is 2^31-1, handles both big-B and big-N cases. The
    # block->(b, tile) linearization matches the old 2D grid (b*n_tiles + tile),
    # since CUDA enumerates grid.x fastest, so per-program work is unchanged.
    flat_id = tl.program_id(0)
    n_tiles = tl.cdiv(N, BLOCK_N)
    pid_b = (flat_id // n_tiles)
    pid_n = (flat_id % n_tiles)
    pid_b = pid_b.to(tl.int64)

    n_start = pid_n * BLOCK_N
    n_offsets = n_start + tl.arange(0, BLOCK_N)
    n_offsets = n_offsets.to(tl.int64)
    n_mask = n_offsets < N

    offs_d = tl.arange(0, BLOCK_D).to(tl.int64)

    best_dist = tl.full((BLOCK_N,), -3.4e38, tl.float32)
    best_idx = tl.zeros((BLOCK_N,), tl.int32)

    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_offsets = k_offsets.to(tl.int64)
        k_mask = k_offsets < K

        cross = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)

        for d_start in range(0, D, BLOCK_D):
            d_offsets = d_start + offs_d
            d_mask = d_offsets < D

            x_ptrs = (
                x_ptr
                + pid_b * stride_x_b
                + n_offsets[:, None] * stride_x_n
                + d_offsets[None, :] * stride_x_d
            )
            x_chunk = tl.load(
                x_ptrs,
                mask=n_mask[:, None] & d_mask[None, :],
                other=0.0,
            )

            c_ptrs = (
                c_ptr
                + pid_b * stride_c_b
                + k_offsets[None, :] * stride_c_k
                + d_offsets[:, None] * stride_c_d
            )
            c_chunk = tl.load(
                c_ptrs,
                mask=k_mask[None, :] & d_mask[:, None],
                other=0.0,
            )

            cross += tl.dot(x_chunk, c_chunk).to(tl.float32)

        # Mask invalid centroid columns to a sentinel below any real score.
        dist = tl.where(k_mask[None, :], cross, -torch.finfo(torch.float32).max)

        curr_max = tl.max(dist, axis=1)
        curr_idx = tl.argmax(dist, axis=1)

        update = curr_max > best_dist
        best_dist = tl.where(update, curr_max, best_dist)
        best_idx = tl.where(update, k_start + curr_idx, best_idx)

    out_ptrs = out_ptr + pid_b * stride_out_b + n_offsets * stride_out_n
    tl.store(out_ptrs, best_idx, mask=n_mask)


_cosine_assign_kernel_split_d_autotuned = triton.autotune(
    _TUNE_CONFIGS_SPLIT_D, key=["N", "K", "D"]
)(_cosine_assign_kernel_split_d)


# ---------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------

def euclid_assign_triton(
    x: torch.Tensor,
    centroids: torch.Tensor,
    x_sq: torch.Tensor,
    out: torch.Tensor = None,
    c_sq: torch.Tensor = None,
    *,
    BLOCK_N: int = 128,
    BLOCK_K: int = 128,
    num_warps: Optional[int] = None,
    num_stages: Optional[int] = None,
    config: Optional[dict] = None,
    use_heuristic: bool = True,
) -> torch.Tensor:
    """Return nearest-centroid indices using Triton kernel.

    Args:
        x         : (B, N, D) float16 / float32 (on CUDA)
        centroids : (B, K, D) same dtype/device as x
        x_sq      : (B, N)    float32 – ||x||^2 per point (on CUDA)
        out       : (B, N)    int32   – (option) pre-allocated output tensor (on CUDA)
        c_sq      : (B, K)    float32 – (option) ||centroids||^2 per centroid (on CUDA)

    Returns:
        cluster_ids (B, N) int32 (callers can cast to int64 if desired)
    Extra:
        config        : {"BLOCK_N","BLOCK_K","num_warps","num_stages"} to force a config
        use_heuristic : use a fixed heuristic config instead of autotune
    """
    assert x.is_cuda and centroids.is_cuda and x_sq.is_cuda, "All tensors must be on CUDA"
    # assert x.dtype in (torch.float16, torch.float32), "x must be fp16/fp32"
    assert centroids.dtype == x.dtype, "centroids dtype mismatch"

    B, N, D = x.shape
    K = centroids.shape[1]
    assert centroids.shape == (B, K, D), "centroids shape mismatch"
    assert x_sq.shape == (B, N), "x_sq shape mismatch"

    # x = x.contiguous()
    # centroids = centroids.contiguous()
    # x_sq = x_sq.contiguous()

    if out is None:
        out = torch.empty((B, N), device=x.device, dtype=torch.int32)
    if c_sq is None:
        c_sq = (centroids.to(torch.float32) ** 2).sum(-1)

    # Strides (in elements)
    stride_x_b, stride_x_n, stride_x_d = x.stride()
    stride_c_b, stride_c_k, stride_c_d = centroids.stride()
    stride_xsq_b, stride_xsq_n = x_sq.stride()
    stride_csq_b, stride_csq_k = c_sq.stride()
    stride_out_b, stride_out_n = out.stride()

    grid = lambda META: (B * triton.cdiv(N, META["BLOCK_N"]),)
    # split-D kernels now decode the same 1D flattened grid (B * n_tiles,) and
    # recover (b, tile) from program_id(0), so they also launch for B > 65535.
    grid_split_d = grid

    use_split_d = _need_split_d(D, x.dtype, x.device)

    selected_config = None
    if config is not None:
        selected_config = config
    elif num_warps is not None or num_stages is not None:
        if num_warps is None or num_stages is None:
            raise ValueError("num_warps and num_stages must be set together")
        selected_config = {
            "BLOCK_N": BLOCK_N,
            "BLOCK_K": BLOCK_K,
            "num_warps": num_warps,
            "num_stages": num_stages,
        }
        if use_split_d and "BLOCK_D" not in selected_config:
            # Caller supplied a small-D-shaped config but D demands split-D.
            # Use a sane default for BLOCK_D and let SMEM fitter shrink.
            selected_config = dict(selected_config)
            selected_config["BLOCK_D"] = 64
            selected_config = _fit_config_to_smem_split_d(
                selected_config,
                D,
                _dtype_bytes(x.dtype),
                _smem_limit(x.device),
            )
    elif use_heuristic:
        if use_split_d:
            selected_config = _heuristic_euclid_config_split_d(
                N, K, D, device=x.device, dtype=x.dtype
            )
        else:
            selected_config = _heuristic_euclid_config(
                N, K, D, device=x.device, dtype=x.dtype
            )

    if use_split_d:
        if selected_config is None:
            _euclid_assign_kernel_split_d_autotuned[grid_split_d](
                x, centroids, x_sq, c_sq, out,
                B, N, K, D,
                stride_x_b, stride_x_n, stride_x_d,
                stride_c_b, stride_c_k, stride_c_d,
                stride_xsq_b, stride_xsq_n,
                stride_csq_b, stride_csq_k,
                stride_out_b, stride_out_n,
            )
        else:
            _euclid_assign_kernel_split_d[grid_split_d](
                x, centroids, x_sq, c_sq, out,
                B, N, K, D,
                stride_x_b, stride_x_n, stride_x_d,
                stride_c_b, stride_c_k, stride_c_d,
                stride_xsq_b, stride_xsq_n,
                stride_csq_b, stride_csq_k,
                stride_out_b, stride_out_n,
                BLOCK_N=selected_config["BLOCK_N"],
                BLOCK_K=selected_config["BLOCK_K"],
                BLOCK_D=selected_config["BLOCK_D"],
                num_warps=selected_config["num_warps"],
                num_stages=selected_config["num_stages"],
            )
        return out

    D_pad = _pad_d(D)
    if selected_config is not None:
        _euclid_assign_kernel[grid](
            x,
            centroids,
            x_sq,
            c_sq,
            out,
            B,
            N,
            K,
            D,
            stride_x_b,
            stride_x_n,
            stride_x_d,
            stride_c_b,
            stride_c_k,
            stride_c_d,
            stride_xsq_b,
            stride_xsq_n,
            stride_csq_b,
            stride_csq_k,
            stride_out_b,
            stride_out_n,
            BLOCK_N=selected_config["BLOCK_N"],
            BLOCK_K=selected_config["BLOCK_K"],
            D_PAD=D_pad,
            num_warps=selected_config["num_warps"],
            num_stages=selected_config["num_stages"],
        )
    else:
        _euclid_assign_kernel_autotuned[grid](
            x,
            centroids,
            x_sq,
            c_sq,
            out,
            B,
            N,
            K,
            D,
            stride_x_b,
            stride_x_n,
            stride_x_d,
            stride_c_b,
            stride_c_k,
            stride_c_d,
            stride_xsq_b,
            stride_xsq_n,
            stride_csq_b,
            stride_csq_k,
            stride_out_b,
            stride_out_n,
            D_PAD=D_pad,
        )
    return out


def cosine_assign_triton(x: torch.Tensor, centroids: torch.Tensor, out: torch.Tensor = None,
                         *, BLOCK_N: int = 128, BLOCK_K: int = 128) -> torch.Tensor:
    """Return nearest(cosine similarity)-centroid indices using Triton kernel.

    Args:
        x         : (B, N, D) float16 / float32 (on CUDA)
        centroids : (B, K, D) same dtype/device as x

    Returns:
        cluster_ids (B, N) int32 (callers can cast to int64 if desired)
    """
    assert x.is_cuda and centroids.is_cuda, "All tensors must be on CUDA"
    # assert x.dtype in (torch.float16, torch.float32), "x must be fp16/fp32"
    assert centroids.dtype == x.dtype, "centroids dtype mismatch"

    B, N, D = x.shape
    K = centroids.shape[1]
    assert centroids.shape == (B, K, D), "centroids shape mismatch"

    # x = x.contiguous()
    # centroids = centroids.contiguous()
    # x_sq = x_sq.contiguous()

    if out is None:
        out = torch.empty((B, N), device=x.device, dtype=torch.int32)

    # Strides (in elements)
    stride_x_b, stride_x_n, stride_x_d = x.stride()
    stride_c_b, stride_c_k, stride_c_d = centroids.stride()
    stride_out_b, stride_out_n = out.stride()

    grid = lambda META: (B * triton.cdiv(N, META["BLOCK_N"]),)
    # split-D kernel decodes the same 1D flattened grid (see euclid path).
    grid_split_d = grid

    if _need_split_d(D, x.dtype, x.device):
        _cosine_assign_kernel_split_d_autotuned[grid_split_d](
            x,
            centroids,
            out,
            B,
            N,
            K,
            D,
            stride_x_b,
            stride_x_n,
            stride_x_d,
            stride_c_b,
            stride_c_k,
            stride_c_d,
            stride_out_b,
            stride_out_n,
        )
        return out

    # Small-D path. The existing autotune sweep includes configs that may
    # not fit SMEM (e.g. fp16 D=512 BN=64 BK=64 S=4 → 320 KiB > H200 limit;
    # fp32 D=512 BN=64 BK=32 S=4 → 393 KiB). The cosine kernel has the
    # same tile shapes as euclid, so reuse the SMEM-aware euclid heuristic
    # for config selection regardless of dtype. This keeps fp16/bf16 small-D
    # behaviour byte-identical to the euclid path's heuristic (verified by
    # examples/regression_fp16_smalld.py).
    cfg = _heuristic_euclid_config(N, K, D, device=x.device, dtype=x.dtype)
    _cosine_assign_kernel[grid](
        x,
        centroids,
        out,
        B,
        N,
        K,
        D,
        stride_x_b,
        stride_x_n,
        stride_x_d,
        stride_c_b,
        stride_c_k,
        stride_c_d,
        stride_out_b,
        stride_out_n,
        BLOCK_N=cfg["BLOCK_N"],
        BLOCK_K=cfg["BLOCK_K"],
        D_PAD=_pad_d(D),
        num_warps=cfg["num_warps"],
        num_stages=cfg["num_stages"],
    )
    return out

# ---------------------------------------------------------------



def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _next_pow2(v: int) -> int:
    if v <= 1:
        return 1
    return 1 << (v - 1).bit_length()


def _pad_d(D: int) -> int:
    """Pad D up to the next power of two (min 16) for kernels that load the
    whole feature dimension as a single tile. ``tl.arange`` requires a
    power-of-two range, so non-power-of-two D (e.g. 192, 320, 768) needs to
    be expanded and the tail masked out at load/store time.
    """
    return max(16, _next_pow2(D))


def _dtype_bytes(dtype) -> int:
    """Element size in bytes for a torch dtype (fallback 4 for the unknown)."""
    if isinstance(dtype, torch.dtype):
        return torch.tensor([], dtype=dtype).element_size()
    if isinstance(dtype, int):
        return dtype
    return 4


# Per-program feature-tile budget (bytes) for the chunk kernel. The kernel
# materialises a ``[BLOCK_N, BLOCK_D]`` feature tile in registers/local memory
# for each cluster run; capping its byte size keeps the driver's per-launch
# local-memory scratch bounded. Without this cap a large D combined with a
# wide dtype (fp32/fp64) forces a huge ``[BLOCK_N, next_pow2(D)]`` tile that
# spills heavily and can fail ``cuLaunchKernel`` with CUDA OOM (issue #19).
# 256 KiB per program keeps D<=512 (all dtypes) and D<=1024 (fp16) in a single
# D pass -- byte-for-byte identical to the pre-split-D kernel -- while tiling
# only the genuinely large tiles that used to spill.
_CHUNK_TILE_BUDGET_BYTES = 256 * 1024


def _choose_block_d(D: int, BLOCK_N: int, dtype_bytes: int,
                    budget_bytes: int = _CHUNK_TILE_BUDGET_BYTES) -> int:
    """Pick the feature-dim tile ``BLOCK_D`` for ``_centroid_update_chunk_kernel``.

    Returns ``_pad_d(D)`` (a single D pass, identical to the original kernel)
    whenever the full padded feature vector fits the per-program tile budget.
    Otherwise returns the largest power-of-two ``BLOCK_D`` (>= 16) whose
    ``[BLOCK_N, BLOCK_D]`` tile fits the budget, so the inner D loop keeps the
    register/local-memory footprint bounded regardless of D or dtype.
    """
    D_pad = _pad_d(D)
    if BLOCK_N * D_pad * dtype_bytes <= budget_bytes:
        return D_pad
    bd = D_pad
    while bd > 16 and BLOCK_N * bd * dtype_bytes > budget_bytes:
        bd //= 2
    return max(bd, 16)


@triton.jit
def _centroid_update_kernel(
    x_ptr,                # *f16 / *f32 [B, N, D]
    cluster_ptr,          # *i32        [B, N]
    sum_ptr,              # *f32        [B, K, D]
    count_ptr,            # *i32        [B, K]
    # --- strides (elements) ---
    stride_x_b, stride_x_n, stride_x_d,
    stride_sum_b, stride_sum_k, stride_sum_d,
    stride_count_b, stride_count_k,
    B: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    K: tl.constexpr,
    BLOCK_D: tl.constexpr,   # number of dims processed per program
):
    """Each program processes 1 token across BLOCK_D dims using atomics with general strides."""
    pid = tl.program_id(axis=0)
    token_idx = pid  # range: [0, B*N)

    # Derive (b, n)
    b = (token_idx // N).to(tl.int64)
    n = (token_idx % N).to(tl.int64)

    # pointer to this token's feature vector
    x_offset = b * stride_x_b + n * stride_x_n
    x_tok_ptr = x_ptr + x_offset

    cluster_idx = tl.load(cluster_ptr + b * N + n)
    cluster_idx = tl.where(cluster_idx < K, cluster_idx, 0)
    cluster_idx = cluster_idx.to(tl.int64)

    # base ptr for centroid accum array
    centroid_base = b * stride_sum_b + cluster_idx * stride_sum_k

    offs = tl.arange(0, BLOCK_D).to(tl.int64)
    for d_start in range(0, D, BLOCK_D):
        mask = offs + d_start < D
        feats = tl.load(x_tok_ptr + (d_start + offs) * stride_x_d, mask=mask, other=0.0)
        feats = feats.to(tl.float32)
        dest_ptr = sum_ptr + centroid_base + (d_start + offs) * stride_sum_d
        tl.atomic_add(dest_ptr, feats, mask=mask)

    tl.atomic_add(count_ptr + b * stride_count_b + cluster_idx * stride_count_k, 1)


def triton_centroid_update_cosine(x_norm: torch.Tensor, cluster_ids: torch.Tensor, old_centroids: torch.Tensor):
    """Compute centroids using custom Triton kernel.

    Args:
        x_norm (Tensor): (B, N, D) normalized input vectors (float16/float32)
        cluster_ids (LongTensor): (B, N) cluster assignment per point
        old_centroids (Tensor): (B, K, D) previous centroids (same dtype as x_norm)

    Returns:
        Tensor: (B, K, D) updated and L2-normalized centroids (dtype == x_norm.dtype)
    """
    assert x_norm.is_cuda and cluster_ids.is_cuda, "Input tensors must be on CUDA device"
    B, N, D = x_norm.shape
    K = old_centroids.shape[1]
    assert cluster_ids.shape == (B, N)

    # Allocate accumulation buffers
    centroid_sums = torch.zeros((B, K, D), device=x_norm.device, dtype=torch.float32)
    centroid_counts = torch.zeros((B, K), device=x_norm.device, dtype=torch.int32)

    # Launch Triton kernel – one program per token
    total_tokens = B * N
    BLOCK_D = 128  # tuneable

    grid = (total_tokens,)
    _centroid_update_kernel[grid](
        x_norm,
        cluster_ids.to(torch.int32),
        centroid_sums,
        centroid_counts,
        x_norm.stride(0), x_norm.stride(1), x_norm.stride(2),
        centroid_sums.stride(0), centroid_sums.stride(1), centroid_sums.stride(2),
        centroid_counts.stride(0), centroid_counts.stride(1),
        B, N, D, K,
        BLOCK_D=BLOCK_D,
    )

    # Compute means; keep old centroid if empty cluster
    counts_f = centroid_counts.to(torch.float32).unsqueeze(-1).clamp(min=1.0)
    centroids = centroid_sums / counts_f

    # For clusters with zero count, revert to old centroids
    zero_mask = (centroid_counts == 0).unsqueeze(-1)
    centroids = torch.where(zero_mask, old_centroids.to(torch.float32), centroids)

    centroids = centroids.to(x_norm.dtype)
    centroids = F.normalize(centroids, p=2, dim=-1)
    return centroids


def torch_loop_centroid_update_cosine(x_norm: torch.Tensor, cluster_ids: torch.Tensor, old_centroids: torch.Tensor):
    """Reference Python implementation (double for-loop)"""
    B, N, D = x_norm.shape
    K = old_centroids.shape[1]
    new_centroids = torch.zeros_like(old_centroids)
    for b in range(B):
        for k in range(K):
            mask = cluster_ids[b] == k
            if mask.any():
                new_centroids[b, k] = F.normalize(x_norm[b][mask].mean(dim=0, dtype=x_norm.dtype), p=2, dim=0)
            else:
                new_centroids[b, k] = old_centroids[b, k]
    return new_centroids


def triton_centroid_update_euclid(x: torch.Tensor, cluster_ids: torch.Tensor, old_centroids: torch.Tensor):
    """Compute centroids for Euclidean KMeans using Triton.

    Args:
        x (Tensor): (B, N, D) input vectors (float16/float32)
        cluster_ids (LongTensor): (B, N) cluster assignment per point
        old_centroids (Tensor): (B, K, D) previous centroids (same dtype as x)

    Returns:
        Tensor: (B, K, D) updated centroids (dtype == x.dtype)
    """
    assert x.is_cuda and cluster_ids.is_cuda, "Input tensors must be on CUDA device"
    B, N, D = x.shape
    K = old_centroids.shape[1]
    assert cluster_ids.shape == (B, N)

    # Allocate accumulation buffers
    centroid_sums = torch.zeros((B, K, D), device=x.device, dtype=torch.float32)
    centroid_counts = torch.zeros((B, K), device=x.device, dtype=torch.int32)

    total_tokens = B * N
    BLOCK_D = 128  # tuneable
    grid = (total_tokens,)

    _centroid_update_kernel[grid](
        x,
        cluster_ids.to(torch.int32),
        centroid_sums,
        centroid_counts,
        x.stride(0), x.stride(1), x.stride(2),
        centroid_sums.stride(0), centroid_sums.stride(1), centroid_sums.stride(2),
        centroid_counts.stride(0), centroid_counts.stride(1),
        B, N, D, K,
        BLOCK_D=BLOCK_D,
    )

    # Compute means; keep old centroid if empty cluster
    counts_f = centroid_counts.to(torch.float32).unsqueeze(-1).clamp(min=1.0)
    centroids = centroid_sums / counts_f

    # For clusters with zero count, revert to old centroids
    zero_mask = (centroid_counts == 0).unsqueeze(-1)
    centroids = torch.where(zero_mask, old_centroids.to(torch.float32), centroids)

    return centroids.to(x.dtype)


# ------------------------------ NEW: chunk-wise centroid update (sorted ids) ------------------------------

@triton.jit
def _centroid_update_chunk_kernel(
    x_ptr,                # *f16 / *f32 [B, N, D] – ORIGINAL ORDER
    sorted_idx_ptr,       # *i32        [B, N]    – indices after sort
    sorted_cluster_ptr,   # *i32        [B, N]    – cluster ids in sorted order
    sum_ptr,              # *f32        [B, K, D]
    count_ptr,            # *i32        [B, K]
    # strides
    stride_x_b, stride_x_n, stride_x_d,
    stride_idx_b, stride_idx_n, stride_cluster_b, stride_cluster_n,
    stride_sum_b, stride_sum_k, stride_sum_d,
    stride_count_b, stride_count_k,
    B: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    K: tl.constexpr,
    BLOCK_N: tl.constexpr,   # how many tokens (points) each program processes
    BLOCK_D: tl.constexpr,   # feature-dim tile; inner loop streams D in BLOCK_D chunks
):
    """Each program processes **BLOCK_N consecutive, already-sorted tokens**.

    Because the tokens are sorted by cluster id, identical ids appear in
    contiguous runs.  We therefore accumulate a local sum/count for the
    current run and perform **a single atomic update per run**, instead of
    per-token.

    The feature dimension is streamed in ``BLOCK_D`` chunks (split-D), so the
    per-program ``[BLOCK_N, BLOCK_D]`` tile — and therefore the register /
    local-memory footprint — is bounded by ``BLOCK_D`` rather than D. This
    avoids the launch-time OOM that the old single-tile ``[BLOCK_N, next_pow2(D)]``
    load hit for large D + wide dtypes (issue #19), and drops the
    next-power-of-two padding waste (the inner loop iterates over the real D and
    masks only the final partial tile).
    """
    # FIX: 1D flattened grid — no 65535 limit on either B or N.
    flat_id = tl.program_id(axis=0)
    n_tiles = tl.cdiv(N, BLOCK_N)
    pid_b     = flat_id // n_tiles
    pid_chunk = flat_id % n_tiles

    b = pid_b.to(tl.int64)
    chunk_start = (pid_chunk * BLOCK_N).to(tl.int64)

    # Nothing to do – out of range
    if chunk_start >= N:
        return

    # base pointers for this batch
    idx_batch_base     = sorted_idx_ptr + b * stride_idx_b
    cid_batch_base     = sorted_cluster_ptr + b * stride_cluster_b
    x_batch_base       = x_ptr + b * stride_x_b  # for pointer arithmetic

    offs_token = tl.arange(0, BLOCK_N).to(tl.int64)
    # Hoisted feature-lane base; the (constexpr) d_start offset is added per
    # D-tile below. Keeping the arange out of the runtime cluster loop avoids
    # rebuilding it every iteration (matters for the single-pass small-D case).
    base_dim   = tl.arange(0, BLOCK_D).to(tl.int64)

    # first token index & validity mask
    token_idx  = chunk_start + offs_token
    valid_tok  = token_idx < N
    first_token_idx = chunk_start
    last_token_idx = tl.minimum(chunk_start + BLOCK_N, N) - 1

    # Load first cluster id to initialise the running accumulator
    first_id = tl.load(cid_batch_base + first_token_idx)
    last_id = tl.load(cid_batch_base + last_token_idx)
    all_ids = tl.load(cid_batch_base + token_idx * stride_cluster_n, mask=valid_tok, other=-1)

    all_tokens_idxs = tl.load(idx_batch_base + token_idx * stride_idx_n, mask=valid_tok, other=-1) # [BLOCK_N]
    all_tokens_idxs = all_tokens_idxs.to(tl.int64)

    for cid in range(first_id, last_id + 1):
        cluster_mask = all_ids == cid
        cluster_size = tl.sum(cluster_mask.to(tl.int32))
        if cluster_size != 0:
            for d_start in range(0, D, BLOCK_D):
                offs_dim = d_start + base_dim
                d_mask = offs_dim < D
                row_ptrs = x_batch_base + all_tokens_idxs[:, None] * stride_x_n + offs_dim[None, :] * stride_x_d
                cluster_feats = tl.load(
                    row_ptrs,
                    mask=cluster_mask[:, None] & d_mask[None, :],
                    other=0.0,
                )  # [BLOCK_N, BLOCK_D]
                cluster_feats = cluster_feats.to(tl.float32)
                sum_feats = tl.sum(cluster_feats, axis=0)  # [BLOCK_D]
                dest_ptr = sum_ptr + b * stride_sum_b + cid * stride_sum_k + offs_dim * stride_sum_d
                tl.atomic_add(dest_ptr, sum_feats, mask=d_mask)
            tl.atomic_add(count_ptr + b * stride_count_b + cid * stride_count_k, cluster_size)


# ---------------------------------------------------------------------------------------------

def triton_centroid_update_sorted_cosine(x_norm: torch.Tensor, cluster_ids: torch.Tensor, old_centroids: torch.Tensor,
                                         *, BLOCK_N: int = 256):
    """Fast centroid update assuming **cluster_ids are sorted along N**.

    This helper will sort the assignments (together with `x_norm`) and launch the
    chunk kernel above.  Compared to the naive per-token kernel it performs *one
    atomic add per run of identical ids* instead of per token, providing large
    speed-ups when clusters are reasonably sized.
    """
    assert x_norm.is_cuda and cluster_ids.is_cuda, "Inputs must be on CUDA"
    B, N, D = x_norm.shape
    K = old_centroids.shape[1]
    assert cluster_ids.shape == (B, N)

    # -------- sort per-batch --------
    sorted_cluster_ids, sorted_idx = torch.sort(cluster_ids, dim=-1)
    sorted_idx_int = sorted_idx.to(torch.int32)

    # accumulation buffers
    centroid_sums = torch.zeros((B, K, D), device=x_norm.device, dtype=torch.float32)
    centroid_cnts = torch.zeros((B, K),    device=x_norm.device, dtype=torch.int32)

    grid = (B * triton.cdiv(N, BLOCK_N),)
    _centroid_update_chunk_kernel[grid](
        x_norm,
        sorted_idx_int,
        sorted_cluster_ids.to(torch.int32),
        centroid_sums,
        centroid_cnts,
        x_norm.stride(0), x_norm.stride(1), x_norm.stride(2),
        sorted_idx_int.stride(0), sorted_idx_int.stride(1),
        sorted_cluster_ids.stride(0), sorted_cluster_ids.stride(1),
        centroid_sums.stride(0), centroid_sums.stride(1), centroid_sums.stride(2),
        centroid_cnts.stride(0), centroid_cnts.stride(1),
        B, N, D, K,
        BLOCK_N=BLOCK_N,
        BLOCK_D=_choose_block_d(D, BLOCK_N, _dtype_bytes(x_norm.dtype)),
    )

    # finalise – convert to means, handle empty clusters, renormalise
    counts_f = centroid_cnts.to(torch.float32).unsqueeze(-1).clamp(min=1.0)
    centroids = centroid_sums / counts_f
    empty_mask = (centroid_cnts == 0).unsqueeze(-1)
    centroids = torch.where(empty_mask, old_centroids.to(torch.float32), centroids)
    centroids = centroids.to(x_norm.dtype)
    centroids = F.normalize(centroids, p=2, dim=-1)
    return centroids

def triton_centroid_update_sorted_euclid(x: torch.Tensor, cluster_ids: torch.Tensor, old_centroids: torch.Tensor,
                                         *, BLOCK_N: int = 256, centroid_sums: torch.Tensor = None, centroid_cnts: torch.Tensor = None, calculate_new: bool = True):
    """Fast centroid update for *Euclidean* KMeans assuming cluster IDs are pre-sorted.

    Parameters
    ----------
    x : Tensor [B, N, D]
        Input feature vectors (no normalization assumed).
    cluster_ids : LongTensor [B, N]
        Cluster assignment for each point.
    old_centroids : Tensor [B, K, D]
        Previous centroids (used to fill empty clusters).
    BLOCK_N : int, optional
        Tokens per Triton program (affects occupancy/perf).
    centroid_sums : Tensor [B, K, D], optional
        Pre-allocated accumulation buffer for sums.  If None, a new buffer is created.
    centroid_cnts : Tensor [B, K], optional
        Pre-allocated accumulation buffer for counts.  If None, a new buffer is created.
    calculate_new : bool, default=True
        If True, compute and return the new centroids.  If False, only update the
        accumulation buffers.

    Returns
    _________
        centroids_new : Tensor [B, K, D] or None
            Updated centroids if `calculate_new` is True; otherwise None.
    """
    assert x.is_cuda and cluster_ids.is_cuda, "Inputs must be on CUDA device"
    B, N, D = x.shape
    K = old_centroids.shape[1]

    # Batch-wise sort of cluster assignments
    sorted_cluster_ids, sorted_idx = torch.sort(cluster_ids, dim=-1)
    sorted_idx_int = sorted_idx.to(torch.int32)

    if centroid_sums is None:
        centroid_sums = torch.zeros((B, K, D), device=x.device, dtype=torch.float32)
    else:
        assert centroid_sums.shape == (B, K, D)
    
    if centroid_cnts is None:
        centroid_cnts = torch.zeros((B, K),    device=x.device, dtype=torch.int32)
    else:
        assert centroid_cnts.shape == (B, K)

    grid = (B * triton.cdiv(N, BLOCK_N),)
    _centroid_update_chunk_kernel[grid](
        x,                       # original features
        sorted_idx_int,          # gather indices
        sorted_cluster_ids.to(torch.int32),
        centroid_sums,
        centroid_cnts,
        x.stride(0), x.stride(1), x.stride(2),
        sorted_idx_int.stride(0), sorted_idx_int.stride(1),
        sorted_cluster_ids.stride(0), sorted_cluster_ids.stride(1),
        centroid_sums.stride(0), centroid_sums.stride(1), centroid_sums.stride(2),
        centroid_cnts.stride(0), centroid_cnts.stride(1),
        B, N, D, K,
        BLOCK_N=BLOCK_N,
        BLOCK_D=_choose_block_d(D, BLOCK_N, _dtype_bytes(x.dtype)),
    )

    if calculate_new:
        # Convert sums to means; replace empty clusters with old centroids
        counts_f = centroid_cnts.to(torch.float32).unsqueeze(-1).clamp(min=1.0)
        centroids = centroid_sums / counts_f
        empty_mask = (centroid_cnts == 0).unsqueeze(-1)
        centroids = torch.where(empty_mask, old_centroids.to(torch.float32), centroids)
        return centroids.to(x.dtype)
    else:
        return None
# ------------------------------ END new implementation ------------------------------

def assign(
    x: torch.Tensor,
    centroids: torch.Tensor,
    x_sq: torch.Tensor,
) -> torch.Tensor:
    return euclid_assign_triton(x, centroids, x_sq)


def update(
    x: torch.Tensor,
    labels: torch.Tensor,
    old_centroids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, _, dimension = x.shape
    clusters = old_centroids.shape[1]
    centroid_sums = torch.zeros(
        (batch, clusters, dimension),
        device=x.device,
        dtype=torch.float32,
    )
    centroid_counts = torch.zeros(
        (batch, clusters),
        device=x.device,
        dtype=torch.int32,
    )
    triton_centroid_update_sorted_euclid(
        x,
        labels,
        old_centroids,
        centroid_sums=centroid_sums,
        centroid_cnts=centroid_counts,
        calculate_new=False,
    )
    denominator = centroid_counts.to(torch.float32).unsqueeze(-1).clamp(min=1.0)
    centroids = centroid_sums / denominator
    empty = (centroid_counts == 0).unsqueeze(-1)
    centroids = torch.where(
        empty,
        old_centroids.to(torch.float32),
        centroids,
    )
    return centroid_counts, centroids.to(x.dtype)
