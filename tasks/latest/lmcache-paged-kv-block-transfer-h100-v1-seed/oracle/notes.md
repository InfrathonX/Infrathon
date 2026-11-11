# Oracle notes

`FINAL_VERSION` = 1. The oracle is host-only qualification material; it is
never a build input and never visible to agents.

## kernel_v001 -- seed control (selected; byte-identical to the Work starter)

The faithful upstream port of `multi_layer_block_kv_transfer` (LMCache @
f9addd2e) restricted to the task's two layout families: same launch geometry,
same streaming-cache vectorised copies, host chunks read/written zero-copy over
PCIe. This is upstream's engine-driven SHM/inline transport
(`transfer_context/base.py` gather_paged_kv_to_cpu / scatter_cpu_to_paged_kv,
`platform/native_block_transfer.py` "Pinned chunks keep the asynchronous
path"). The Work copy carries no comments; it derives from LMCache under Apache-2.0.
`tests/test_contract.py` pins it to
`images/work/workdir/kv_transfer_lab/candidates/kernel_v001/submission.cu`.

## kernel_v002 -- upstream staged path (measured, not selected)

LMCache's other upstream transport for the two scored HND layouts, ported to
the task ABI (sha256 91498485655db5bcc8ac86a34eb2a8443a84c13490833168c4ee1d49b765be3f).

- Upstream: https://github.com/LMCache/LMCache @
  ebe3edde1df6813a14a5d69ae36a5109572f0b10 (main, 2026-10-29). Device code of
  `multi_layer_block_transfer_kernel` (`csrc/cuda/mp_mem_kernels_detail.cuh`)
  and launch arithmetic of `multi_layer_block_kv_transfer_templated`
  (`csrc/cuda/mp_mem_kernels.cu`) are verbatim and unchanged vs the pinned
  f9addd2e for HND.
- Host chunk objects follow upstream's staged executor
  (`_run_object_group_transfer_plan` + `execute_object_group_transfer` +
  `lmcache_memcpy_async`, the lmcache-driven / CUDA-IPC transport that is
  upstream's default on NVIDIA): H2D = whole-object `cudaMemcpyAsync` into GPU
  temp buffers, then the kernel; D2H = kernel into GPU temp buffers, then
  `cudaMemcpyAsync` out. The direct copy-engine path (`cudaMemcpyBatchAsync`)
  is ineligible for HND upstream, so HND always takes this staged path there.
- ABI glue only (no tuning, no per-shape dispatch): temp buffers = the Judge
  scratch arena (batch = whole objects that fit); device-resident chunks run
  the kernel directly; the D2H copy-out skips each slab's skipped token prefix
  (task contract: the skipped prefix must stay untouched; upstream only uses
  skip for retrieve) via `cudaMemcpy2DAsync`; `TORCH_CHECK` -> early return;
  objects that do not fit the arena fall back to the zero-copy kernel (never
  on the scored grid).
- Admission: passes `images/judge/source_admission.py` (allowlisted CUDA
  runtime symbols only: `cudaLaunchKernel`, `cudaMemcpyAsync`,
  `cudaMemcpy2DAsync`); builds with the frozen `-arch=sm_90a` command.

## Measured (final-holdout Judge, H100 NVL 94 GB, LLM review off, native profile)

Same card, same session, alternating order (GPU 5, Judge cpuset 20-23,52-55, 2026-10-29):

| candidate | run ids (`oracleup-lmcache-*-gpu5-*`) | scores | mean | device | host | trace |
|---|---|---|---|---|---|---|
| v001 zero-copy (selected) | old-194403, old2/old3-195100 | 233.88, 234.06, 234.98 | 234.31 | ~821.6 | ~51.1 | ~50.3 |
| v002 staged | new-194403, new2/new3-195100 | 234.05, 233.78, 234.29 | 234.04 | ~824.2 | ~48.7 | ~51.0 |

The archive built from this tree by `scripts/make_oracle_archive.sh` (both
candidates, `FINAL_VERSION` 1) was judged once more: 234.77, valid
(`oracleup-lmcache-final-gpu5-20261029-201235`).

Earlier session (GPU 3, Judge cpuset 24-31): v001 226.02
(`pvsa-lmcache-oracle-1-gpu3-20261029-120619`), v002 234.83 / 235.03
(`pvsa-lmcache-paper-{1,2}-gpu3-20261029-120619`; single-candidate archive whose
`kernel_v001/submission.cu` is byte-identical to this `kernel_v002`). That v001
run had degraded zero-copy host/trace cases (trace 43-47 GB/s vs 49-51 on GPU 5);
on GPU 5 the two paths tie within run-to-run noise, so `FINAL_VERSION` stays 1.

Per group the paths trade off: staged wins the trace cases (+1-2 GB/s) and
host-split ct64, zero-copy wins most host chunk cases (e.g. host-split-ct256
51.8 vs 45.8); device-resident cases run the same kernel.

## Caveats

- Both candidates are upstream-faithful; neither is a ceiling. Agent finals
  reached 287-290 on the GPU 3 session.
- The zero-copy path is more sensitive to host/PCIe state than the staged
  path (see the GPU 3 run). Compare scores only within one card and session
  (cross-card spread ~1.7%, same-card CV ~0.3%).

## Suite v2 qualification, first cut (2026-11-08, H100 SXM 80 GB)

Superseded by the 843-case suite below; kept as the record of the first cut.
The v1 numbers above are on the retired 20-case holdout and are not comparable
with v2. Suite v2's first cut (127 scored cases plus 23 edge gates) was run directly through `runtime/kv_transfer_runtime.py`
(not through the Docker Judge flow) on three free GPUs in parallel, each
bound to its NUMA node. Every case and edge was bit-exact, and
`images/judge/verify_kv_transfer.py` recomputed every case score and the
primary score from the raw records to the same value.

| Candidate | Score | device-large | device-small | host-large | host-small | trace | overlap | Suite time | Device peak |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v001 seed (zero-copy) | 82.8 | 859 | 571 | 38.0 | 43.4 | 43.7 | 9.0 | 250 s | 46.7 GB |
| v002 staged | 116.0 | 864 | 569 | 51.0 | 43.6 | 49.4 | 35.6 | 246 s | 46.7 GB |
| v020, head-size fix (qualification only) | 131.4 | 1323 | 696 | 52.1 | 44.3 | 50.7 | 36.5 | 244 s | 46.7 GB |

- Overlap: the zero-copy seed costs the concurrent foreground 38-86 % of its
  time while it transfers; the staged path costs 0-11 %. In isolation the two
  move host data at the same rate.
- Room left for an agent: device small calls reach about 48 % of the
  1456 GB/s device envelope and host small calls about 80 % of the 55 GB/s
  host envelope; large device calls reach 91 %.
- v020 and v004 were written for v1's domain (head size 128 only) and
  fail the v2 head-size-64 edge fixtures. The qualification variant (template
  width from `hs`) lives outside the task with the qualification data.
- `FINAL_VERSION` stays 1 (the seed is the task's starting point; v002 is the
  stronger upstream reference on v2).
- Planted violations: ignoring the skipped prefix (caught by 4 edges), moving
  half the layers (20 edges, every scored case tried) and assuming a
  contiguous block map (21 edges, every scored case tried, then an illegal
  access that fails the worker).

## Suite v2 qualification, 843 cases (2026-11-08, H100 SXM 80 GB)

`lmcache-kv-transfer-final-holdout-v2` as frozen: 843 scored cases over 60
(shape, layout) strata plus 25 edge gates, seven families including
`host-duplex`, stratified tier-balanced family scores and weights device 0.30,
host 0.25, trace 0.25, overlap 0.20. Run directly through
`runtime/kv_transfer_runtime.py` on three free GPUs in parallel, each bound to
its NUMA node. Every case and edge was bit-exact; the verifier recomputed every
case and the primary score to the same value. The confidence interval resamples
strata within each family and tier.

| Candidate | Score | 95 % CI | device-large | device-small | host-large | host-small | host-duplex | trace | overlap | Suite time | Device peak |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v001 seed (zero-copy) | 70.3 | ±2.6 % | 838 | 513 | 41.2 | 45.1 | 40.4 | 44.2 | 8.3 | 1238 s | 46.9 GB |
| v002 staged | 102.2 | ±1.8 % | 842 | 511 | 50.6 | 42.8 | 70.8 | 49.4 | 35.1 | 1203 s | 46.9 GB |
| v020, head-size fix | 114.8 | ±1.7 % | 1281 | 632 | 51.9 | 44.9 | 71.7 | 50.9 | 36.2 | 1204 s | 46.9 GB |

- Duplex: the same-run duplex envelope was about 75 GB/s against 55 GB/s one
  way. Staged copies reach about 71 GB/s; the zero-copy seed stays near 40.
- The public suite (155 cases) runs in 2 min 47 s for the seed; quick runs in
  14 s.
- Host cases measured identically with a 0.25 s and a 0.10 s settle barrier;
  the barrier is 0.10 s.
