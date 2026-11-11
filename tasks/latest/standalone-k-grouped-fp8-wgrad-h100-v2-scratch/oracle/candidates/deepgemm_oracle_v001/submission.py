"""Upstream DeepGEMM reference under this task's ABI (host-side oracle only).

DeepGEMM 2.6.1 (`559d79f`) implements this exact workload, but its SM90 entry
point is `k_grouped_fp8_gemm_nt_contiguous`, which wants each group's operands
K-major and a host-side `ks` list.  This task's ABI hands over the MN-major
layout of the SM100 `..._tn_contiguous` entry point plus a device-side `ks`,
so the adapter pays for the conversion inside the measured call.

This file never enters the Work or Judge image; `source_admission.py` forbids
`import deep_gemm` in an agent submission by design.
"""

from __future__ import annotations

import deep_gemm
from deep_gemm.utils import set_mk_alignment_for_contiguous_layout


_RECIPE = (1, 1, 128)

# One-time process setup, outside every measured call.
set_mk_alignment_for_contiguous_layout(128)


def _to_k_major(tensor, ks_cpu, width):
    """Repack `[K_total, width]` into per-group `[width, K_g]` blocks."""
    packed = tensor.new_empty((tensor.numel(),))
    prefix = 0
    for group_k in ks_cpu:
        if group_k:
            block = tensor[prefix : prefix + group_k]
            packed[prefix * width : (prefix + group_k) * width] = block.T.flatten()
        prefix += group_k
    return packed


def k_grouped_fp8_gemm_tn_contiguous(
    a, b, a_scale, b_scale, ks, c, expected_k_hint, out
) -> None:
    m = a.shape[1]
    n = b.shape[1]

    # The SM90 API needs the group sizes on the host; the ABI only provides
    # them on the device, so this device-to-host copy is part of the cost.
    ks_cpu = [int(value) for value in ks.tolist()]

    deep_gemm.k_grouped_fp8_gemm_nt_contiguous(
        (_to_k_major(a, ks_cpu, m), a_scale.T),
        (_to_k_major(b, ks_cpu, n), b_scale.T),
        out,
        ks_cpu,
        ks,
        c,
        recipe=_RECIPE,
    )
