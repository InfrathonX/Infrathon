# Performance oracle (host-only)

**Summary (H100 NVL, TFLOPS, higher is better).** `oracle/FINAL_VERSION` (3)
selects `candidates/deepgemm_oracle_v003`, upstream DeepGEMM 2.6.1. It is the
strongest paper solution, scored by the real private Judge through the local
forced-oracle image in `forced_judge/`. No paper candidate is admissible, so
anything that must pass the normal Judge uses `baseline/`.

| role | implementation | Judge | gates | score | run id |
|---|---|---|---|---|---|
| admissible control (no admissible paper candidate exists) | `baseline/` kernel_v001, Triton | normal | valid, all pass | 91.08 | `pvsa-kgrouped-wgrad-baseline-r1` |
| previous oracle number | `deepgemm_oracle_v002`, `run_oracle.sh` harness (no admission, sanitizer or edge cases) | harness only | not judged | 279.1-279.5 | see "Measured" |
| **forced strongest paper** | `deepgemm_oracle_v003` (code = v002) | forced: admission and LLM review bypassed; sanitizer and edges isolated per case, recorded, not fatal | **INVALID**: `memory_safety_ok` (sanitizer memcheck 1/8 passed; 3 edge cases fault) and `edge_correctness_ok` (50/160 pass) | scored grid **283.83 / 285.79** (Judge score 0) | `forced-kgrouped-wgrad-deepgemm-v003-r2-g5-20261029-215241`, `-r1-g5-20261029-213159` |

This directory measures **upstream DeepGEMM** on the trusted runtime and the
frozen scored-grid manifest the Judge uses, so the task has a real
performance oracle to compare candidates against.  It is ported to the
current runtime (`runtime/k_grouped_wgrad_runtime.py`) and was last measured
on 2026-10-29 (numbers below).

It is not the qualification control.  That is `baseline/`: a masked Triton
positive control that covers the whole support domain, proves the task is
solvable, and proves the Judge accepts a correct implementation.  It goes
through the scored path (source admission, the sanitizer pass, the edge
cases, the scored grid), and the scoring contract applies no baseline ratio.
DeepGEMM cannot take that path: `source_admission.py` admits only an import
allowlist and forbids `import deep_gemm` in a submission by design.  An
archive of either DeepGEMM candidate sent through `scripts/eval_archive.sh`
is INVALID at `source_ok` (verified: run ids `pvsa-kgrouped-wgrad-oracle-r1`,
`pvsa-kgrouped-wgrad-paper-r1`, `pvsa-kgrouped-wgrad-paper-r2`, all score 0).

The oracle can only ever cover the **scored grid**.  All seven scored shapes
are aligned (M and N multiples of 256, no empty groups), while the edge
cases include M and N that are not multiples of 16; DeepGEMM's FP32 D tensor
map needs N % 4 == 0 and tiny M/N configurations fail its JIT static
asserts, so upstream fails most of the edge gate (an earlier per-case run of
the v002 glue on 2026-10-29 passed 50 of the 160 private edge cases).
Driving the edge cases through `runtime.run_suite` left a kernel spinning on
the GPU for more than 20 minutes, so `run_oracle.sh` does not run them.  The
edge gate is a `baseline/` concern only.

This oracle is deliberately **outside the scored task**:

- it is never copied into the Work or the Judge image: `scripts/build_images.sh`
  builds only `images/work` and `images/judge`, neither build context reaches
  this directory, and `scripts/validate_layout.py` fails if either image build
  names `oracle/` or `baseline/`;
- it does not change `image_source_digest()`, which hashes only `images/work`,
  `images/judge`, `runtime/`, and `judge/run_judge.py`;
- it is not referenced by `tasks/<task-id>.json` or by any launch entrypoint;
- the DeepGEMM source tree is **not** vendored here (`validate_layout.py`
  forbids an upstream tree inside a standalone task).  `run_oracle.sh` points a
  Docker build context at a checkout you supply via `DEEPGEMM_SOURCE` instead.

## Provenance

- Upstream: https://github.com/deepseek-ai/DeepGEMM @
  `559d79fb6994a58b8a15b4b93bf13ccc16edf247` (v2.6.1), the commit the task's
  runtime, scored grid (`enumerate_k_grouped_contiguous`) and generator are
  derived from.  Submodules `third-party/cutlass` @ `f3fde583`,
  `third-party/fmt` @ `553ec11e`.
- Kernel: `deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d1d.cuh`
  (`KGroupedContiguous`), host `csrc/jit_kernels/impls/sm90_fp8_gemm_1d1d.hpp`,
  API `k_grouped_fp8_gemm_nt_contiguous`; configuration from upstream's own
  SM90 heuristic (`heuristics/sm90.hpp`).  Nothing is overridden.

## Candidates

| candidate | glue around the upstream call |
| --- | --- |
| `candidates/deepgemm_oracle_v001` | upstream's test-generator repack (`tests/generators.py:468-474`: two torch copies per group per operand), new packed buffers each call |
| `candidates/deepgemm_oracle_v002` | lean glue: byte-identical K-major slabs from one Triton launch for both operands, launched directly through the cached compiled kernel, one grow-only workspace |
| `candidates/deepgemm_oracle_v003` | **the oracle (`FINAL_VERSION` 3).** Code identical to v002; the docstring is rewritten for the forced Judge. sha256 `1c19d0ffcdbf8e644331863c5523021f1e57cb9756f671c0f588300c96f34afd` |

Both pay, inside the timed call, what the task ABI forces on the SM90 API:
this task's ABI matches DeepGEMM's `k_grouped_fp8_gemm_tn_contiguous`
(A `[K_total, M]`, MN-major, device `ks`), which upstream dispatches **only
on SM100**.  On SM90 the workload goes through
`k_grouped_fp8_gemm_nt_contiguous`, which needs each group's operands
transposed to K-major plus a host-side `ks` list (`ks.tolist()` D2H sync),
and DeepGEMM's epilogue supports only `C == D`, so it runs `d.copy_(c)`
itself.

## What the numbers mean

| mode (`run_oracle.sh` label) | what is timed | comparable to a Judge score? |
| --- | --- | --- |
| `deepgemm-abi` / `deepgemm-abi-v001` | glue (repack + `ks` D2H) + the DeepGEMM call, `runtime.run_suite` | yes: same ABI, same protocol as a candidate (no source admission, no sanitizer pass, no LLM review) |
| `deepgemm-hoisted` | the DeepGEMM call only; repack and `ks` D2H outside the timer | no: upper bound on upstream fed K-major operands |
| `deepgemm-kernel-only` | as hoisted, plus `C -> out` outside the timer and `d == c` | no: upstream's own benchmark condition (`tests/generators.py:446`) |

## Measured (H100 NVL 94 GB, frozen final-holdout seeds, 5 trials, 1 warm-up)

Scored metric = geomean(coarse, fine) TFLOPS.

| implementation | 2026-10-29 in-repo `run_oracle.sh`, GPU 7, run 1 / run 2 | earlier harness runs, same protocol, GPU 6 |
| --- | --- | --- |
| baseline-triton (qualification control) | 89.62 / 89.83 | 90.89; Judge run `pvsa-kgrouped-wgrad-baseline-r1`: 91.08 (valid) |
| deepgemm-abi-v001 (upstream generator glue) | 151.17 / 152.08 | 135.07 |
| **deepgemm-abi (v002 lean glue)** | **279.51 / 279.12** (coarse 487.6 / 489.2, fine 160.2 / 159.3) | 283.68, 281.61 |
| deepgemm-hoisted | 326.54 / 327.56 | 299.26 |
| deepgemm-kernel-only | 368.10 / 377.20 | 381.43 |

Per-case `deepgemm-abi` on run 1: g8_k128 (G 8, 768x2048)
28.1 TFLOPS (162 us, launch/glue bound), ep64 m4096 540.7, ep64 m7168 493.7,
ep32 m4096 489.8, ep32 m7168 432.2, ep16 m4096 396.1, ep16 m7168 369.9.
The fine family is dominated by the launch-bound g8_k128 case, where the
glue (repack launch, `ks.tolist()` sync, `d.copy_(c)`) costs more than the
GEMM.

Caveats: agent submissions that implement their own kernel on the MN-major
layout avoid the repack and the D2H sync and can beat this oracle on the
scored grid (Judge-valid agent finals reached 325-333 TFLOPS on the same
cards); the oracle is the upstream reference, not a ceiling.  Compare
numbers only within one card and session.

## Forced Judge (real private Judge, admission bypassed; 2026-10-29)

`forced_judge/` (see its README) builds `<private judge image>:forced-oracle`
FROM `standalone_k_grouped_wgrad.judge.standalone_k_grouped_fp8_wgrad_h100_scratch_v2:local`
(`b70d98ac2e7b`, which holds the holdout), with DeepGEMM 559d79f built for
sm_90a. It then runs the same `sforge eval` as `scripts/eval_archive.sh` on
the archive `candidates/kernel_v003/submission.py` + `FINAL_VERSION` 3
(sha256 `299b53ab...`). Run on GPU 5 (H100 NVL), Judge cpuset
20-23,52-55, `--network none`, with the LLM review skipped.

| run id | forced image | Judge verdict | sanitizer (per case) | edge cases (per case, 240 s limit) | scored grid coarse / fine / score |
|---|---|---|---|---|---|
| `forced-kgrouped-wgrad-deepgemm-v003-r2-g5-20261029-215241` | `fe8b520f50f0` | INVALID, score 0, first failing gate `memory_safety_ok`; `forced_failed_gates` = edge_correctness_ok, memory_safety_ok | 1/8 passed (5 memcheck errors, 2 failed) | 50/160 passed | 493.62 / 163.21 / **283.83** |
| `forced-kgrouped-wgrad-deepgemm-v003-r1-g5-20261029-213159` | `a6ddbcde0f48` (earlier `forced_verify.py` that listed only the first gate) | same | 1/8 | 50/160 | 496.32 / 164.56 / **285.79** |

Each run took about 19 minutes. Edge failures:

- 97 raise `CUDA_ERROR_INVALID_VALUE` while building DeepGEMM's TMA
  descriptors (unaligned N, or M/N too small);
- 8 fail DeepGEMM's JIT compile static asserts (tiny M/N);
- 3 fault the context with "unspecified launch failure" (memory_safety_ok):
  `edge-061`, `edge-072` and `edge-131`;
- 2 hang and are killed at 240 s: `edge-046-v1-contract` (G4 2176x2944) and
  `edge-064-v1-contract` (G4 640x1920).

The scored grid passes every scored-case check: correctness, guard bands,
sentinel and plausibility. Per band (r2, TFLOPS): g8_k128 28.3,
ep64_m4096 550.7, ep64_m7168 501.2, ep32_m4096 483.6, ep32_m7168 444.8,
ep16_m4096 405.4, ep16_m7168 379.5. These match the `run_oracle.sh` harness
(279.1-279.5) to within about 2%.

## Running it

The oracle image is built from an upstream DeepGEMM checkout, which this
repository does not vendor.  Clone it once at the pinned commit:

```bash
git clone --recursive https://github.com/deepseek-ai/DeepGEMM /path/to/DeepGEMM
git -C /path/to/DeepGEMM checkout 559d79fb6994a58b8a15b4b93bf13ccc16edf247
git -C /path/to/DeepGEMM submodule update --init --recursive
```

Then, from the task root.  `run_oracle.sh` reads the frozen scored seeds from
`judge/private/final_cases.json` if the repository copy exists, else from the
Judge-only dataset asset that `assets/fetch.sh` places under
`${INFRATHON_ASSETS_ROOT:-~/.cache/infrathon/assets}/<task directory>/judge/`,
and checks them against `assets/manifest.json`:

```bash
export DEEPGEMM_SOURCE=/path/to/DeepGEMM
flock /path/to/gpu<idx>.lock oracle/run_oracle.sh --gpu <idx>              # build + measure
flock /path/to/gpu<idx>.lock oracle/run_oracle.sh --gpu <idx> --skip-build
```

The 2026-10-29 numbers above were produced with
`DEEPGEMM_SOURCE=<checkout at 559d79f> ORACLE_CPUSET=28-31,60-63 ORACLE_MEMS=1 oracle/run_oracle.sh --gpu 7`
(fresh build; the DeepGEMM build takes about 10-20 min on a cold Docker cache).

`run_oracle.sh` measures, in order: `baseline-triton`, `deepgemm-abi-v001`,
`deepgemm-abi`, `deepgemm-hoisted`, `deepgemm-kernel-only`.  It fails immediately if `DEEPGEMM_SOURCE`
is unset or not at the pinned commit, if `third-party/cutlass/include` is
missing (the submodules were not initialised), if a frozen input does not
match `assets/manifest.json`, or if the GPU already has 1 GiB or more in use.
It does not take the repository's GPU lock files; hold the card's lock
yourself on a shared host.  `ORACLE_CPUSET` / `ORACLE_MEMS` optionally pin
the measurement container.

The scored manifest is copied to a mode-0600 file under `OUTPUT_DIR`
(default `oracle/results/`, gitignored and excluded from the oracle image's
build context) and mounted read-only.  Results land in `OUTPUT_DIR`.
Nothing here writes into the version store or the run logs.
