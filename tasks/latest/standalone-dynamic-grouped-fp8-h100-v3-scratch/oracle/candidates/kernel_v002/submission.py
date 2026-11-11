"""Upstream DeepGEMM under this task's ABI (paper solution; host-side oracle only).

Upstream: deepseek-ai/DeepGEMM @ 559d79fb6994a58b8a15b4b93bf13ccc16edf247 (v2.6.1),
the exact commit the task's workload, grid and calc_diff check were ported from.

Both entry points map 1:1 onto upstream APIs; nothing in the kernels is changed:

* masked  -> deep_gemm.m_grouped_fp8_gemm_nt_masked
             (SM90: sm90_m_grouped_fp8_gemm_masked_1d2d, GemmType::MGroupedMasked)
* psum    -> deep_gemm.m_grouped_fp8_gemm_nt_contiguous(use_psum_layout=True)
             (SM90: sm90_m_grouped_fp8_gemm_contiguous_1d2d,
              GemmType::MGroupedContiguousWithPsumLayout)

ABI match (checked against csrc/apis/gemm.hpp and tests/generators.py at 559d79f):

* a [G, M_max, K] / [G*M_max, K] FP8 E4M3, K-major          == upstream `a.first`
* a_scale FP32 [.., K/128] per row and 128-K block (K-major) == upstream `a.second`
  before `transform_sf_pair_into_required_layout`; on SM90 upstream itself
  transposes it into an MN-major, TMA-aligned copy (`get_mn_major_tma_aligned_tensor`,
  one `transpose_fp32` JIT kernel) inside the call.  That transpose is part of the
  measured time here, exactly as it is part of any upstream call that receives
  K-major scales.  (Upstream's own benchmark `bench_kineto(..., 'gemm_')` times
  only the GEMM kernel; `oracle/paper_reference/oracle_runner.py --mode hoisted` reports
  that variant.)
* b [G, N, K] FP8 + b_scale FP32 [G, N/128, K/128]           == upstream `b`
* m_sizes int32 [G]                                          == upstream `masked_m`
* group_ends int32 [G], start_g = align(end_{g-1}, 128)      == upstream `psum_m`
  (generate_m_grouped_masked: psum_m[j] = align(psum_m[j-1], 128) + masked_m[j];
  SM90 mk alignment is fixed at 128, heuristics/runtime.hpp)
* expected_m_hint = int(1.2 * expected_m)                    == the value upstream's
  test_m_grouped_gemm_masked passes as `expected_m` / `expected_m_for_psum_layout`
* out BF16 [G, M_max, N] / [G*M_max, N], N-major             == upstream `d`

Upstream's own tuned configs are used as-is: SM90 `get_best_config` (block_m
{64,128} for masked, 128 for psum, block_n 16..192, 1x2/2x1 TMA multicast,
stage count from shared memory), recomputed on every call from the shapes and
`expected_m`, exactly as upstream does.

Forced-oracle glue (oracle/candidates/kernel_v002): identical to
oracle/paper_reference/candidates/deepgemm_v001/submission.py except for the
DG_USE_TEMP_CUBLASLT_WORKSPACE setting below.

This file can NOT be scored by the task Judge: `images/judge/source_admission.py`
admits only the import allowlist {__future__, collections, dataclasses, enum,
functools, itertools, math, torch, triton, typing}; `import deep_gemm` is rejected
("selected source imports a forbidden module"), and the brief forbids DeepGEMM /
CUTLASS / compiled extensions.  Use oracle/paper_reference/ instead.
"""

from __future__ import annotations

import os

# Library configuration only (no kernel or heuristic change): upstream's own
# compute-sanitizer setting (csrc/jit/device_runtime.hpp at 559d79f: "Enabled by
# compute-sanitizer tests, which trigger `cudaErrorCudartUnloading` when the
# workspace tensor is destructed after CUDA driver shutdown").  Without it the
# process-exit destructor of DeepGEMM's static DeviceRuntime frees its cached
# 32 MiB cuBLASLt workspace after the driver has shut down under compute-sanitizer
# and aborts the worker after all cases passed.  The workspace is used only by
# DeepGEMM's cuBLASLt routes, never by the two FP8 grouped GEMMs called here.
os.environ.setdefault("DG_USE_TEMP_CUBLASLT_WORKSPACE", "1")

import deep_gemm  # noqa: E402


def grouped_fp8_gemm_nt_masked(a, b, a_scale, b_scale, m_sizes, expected_m_hint, out) -> None:
    deep_gemm.m_grouped_fp8_gemm_nt_masked(
        (a, a_scale), (b, b_scale), out, m_sizes, int(expected_m_hint)
    )


def grouped_fp8_gemm_nt_psum(a, b, a_scale, b_scale, group_ends, expected_m_hint, out) -> None:
    deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
        (a, a_scale), (b, b_scale), out, group_ends,
        use_psum_layout=True,
        expected_m_for_psum_layout=int(expected_m_hint),
    )
