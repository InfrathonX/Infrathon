# Paper reference: upstream DeepGEMM (host-only, not admissible)

This directory measures the paper solution, **upstream DeepGEMM**, on this
task's ABI with the Judge's own trusted runtime and the frozen
final-holdout-v3 seeds.  It exists so we can see how far the scored oracle
(`../candidates/kernel_v001`, a self-contained Triton kernel) is from the
upstream implementation.  It is never scored and never reaches an agent.

**Update (forced oracle):** `oracle/FINAL_VERSION` now selects
`oracle/candidates/kernel_v002` (this directory's `deepgemm_v001` adapter plus
upstream's compute-sanitizer setting `DG_USE_TEMP_CUBLASLT_WORKSPACE=1`),
scored by the full Judge pipeline in a local-only derived Judge image that
admits it (`forced_judge/`, README there): **166.76 / 164.94 TFLOPS, valid**, sanitizer
passed, 158/158 edge (`forced-dyn-grouped-fp8-deepgemm-r2-g6`, `-r3-g6`).  The verbatim
`deepgemm_v001` adapter measured 166.12 on the scored cases there but failed
`memory_safety_ok` because DeepGEMM aborts at process exit under
compute-sanitizer (`forced-dyn-grouped-fp8-deepgemm-r1-g6`).
`oracle/ADMISSIBLE_VERSION` keeps `kernel_v001` for the normal Judge; the
`oracle-triton-*` runs below time that admissible candidate.  The sections
below describe this harness as it was before the forced oracle.

## Why it is not the scored oracle

`images/judge/source_admission.py` admits only the import allowlist
{`__future__`, `collections`, `dataclasses`, `enum`, `functools`,
`itertools`, `math`, `torch`, `triton`, `typing`}, and the task brief forbids
DeepGEMM / CUTLASS / compiled extensions.  An archive with
`candidates/deepgemm_v001/submission.py` as its selected candidate is INVALID
at `source_ok` on the real Judge ("forbidden import: 'deep_gemm'"; run ids
`pvsa-dyn-grouped-fp8-paper-r1-120736`, `pvsa-dyn-grouped-fp8-paper-r2-120736`,
score 0).  The Judge image also has no DeepGEMM.  So the scored oracle stays
`kernel_v001`, and this directory is a separate host-side harness.

## Isolation

- Lives under `oracle/`, which is not a build input of either image
  (`scripts/validate_layout.py` checks the Work context and that neither
  task Dockerfile copies DeepGEMM); it does not change
  `image_source_digest()` (images/work, images/judge, runtime/,
  judge/run_judge.py).
- `scripts/make_oracle_archive.sh` packs only `candidates`, `FINAL_VERSION`
  and `notes.md` (`--admissible`: `FINAL_VERSION` = `ADMISSIBLE_VERSION`), so
  nothing here enters the oracle archive.
- The DeepGEMM source tree is not vendored (`validate_layout.py` forbids an
  upstream tree in a standalone task); you pass a checkout as a Docker build
  context.
- Results default to `oracle/paper_reference/results/` (gitignored, excluded
  from the image build context by `.dockerignore`).

## Provenance

- Upstream: https://github.com/deepseek-ai/DeepGEMM @
  `559d79fb6994a58b8a15b4b93bf13ccc16edf247` (v2.6.1), the commit the task's
  workload, scored grid and `calc_diff` check were ported from.  Submodules
  `third-party/cutlass` @ `f3fde58372d33e9a5650ba7b80fc48b3b49d40c8`,
  `third-party/fmt` @ `553ec11ec06fbe0beebfbb45f9dc3c9eabd83d28`.
- `candidates/deepgemm_v001/submission.py`: both task entry points map 1:1
  onto upstream public APIs, nothing in the kernels or heuristics changed:
  masked -> `deep_gemm.m_grouped_fp8_gemm_nt_masked`
  (`sm90_m_grouped_fp8_gemm_masked_1d2d`); psum ->
  `deep_gemm.m_grouped_fp8_gemm_nt_contiguous(use_psum_layout=True,
  expected_m_for_psum_layout=hint)`.  The task ABI is the upstream ABI (same
  tensors, psum alignment 128, hint = int(1.2 * expected_m)).  The only
  adaptation cost is upstream's own SM90 SFA transpose (K-major FP32 ->
  MN-major TMA-aligned), which DeepGEMM performs inside the call.
- `oracle_runner.py`: imports the trusted runtime baked into the Judge image
  (`/opt/standalone_grouped_fp8_worker/grouped_fp8_runtime.py`, identical to
  `runtime/` here).  Modes:
  - `suite`: `runtime.run_suite` exactly as the Judge's worker calls it
    (edge cases optional, then the 32 scored cases: 1 warm-up, decoy
    pre-call, 5 fresh-tensor trials, guard bands, FP64 + calc_diff checks,
    sentinel, plausibility bound, time = max(host wall, CUDA event)).
    Directly comparable to a Judge score; not applied: source admission, the
    compute-sanitizer pass, the LLM integrity review.
  - `hoisted`: DeepGEMM with the SFA transpose outside the timer (upper bound
    for "DeepGEMM fed the layout DeepEP produces"); not a task score.
  - `kernel`: GPU time only (CUDA events around 20 back-to-back calls),
    the analogue of upstream's `bench_kineto`; excludes host launch cost.
- `Dockerfile`: `FROM` the task's Judge image (same torch 2.8.0 / CUDA 12.9 /
  Triton 3.4.0) + DeepGEMM built in place with `develop.sh`.

## Build and run

```bash
git clone --recursive https://github.com/deepseek-ai/DeepGEMM /path/to/DeepGEMM
git -C /path/to/DeepGEMM checkout 559d79fb6994a58b8a15b4b93bf13ccc16edf247
git -C /path/to/DeepGEMM submodule update --init --recursive
scripts/build_images.sh            # builds or pulls the task's Judge image (the base)

# from the task root; reads judge/private/{final,edge}_cases.json if present, else the
# fetched dataset assets (assets/fetch.sh), and checks them against assets/manifest.json
flock /path/to/gpu<idx>.lock env GPU=<idx> DEEPGEMM_SOURCE=/path/to/DeepGEMM \
  oracle/paper_reference/run_paper_reference.sh
```

Optional: `JUDGE_IMAGE=<tag>` (base image), `IMAGE`, `RESULTS`, `JIT_CACHE`,
`HARNESS_CPUSET` / `HARNESS_MEMS` (pin the container), `SKIP_BUILD=1`.
The script refuses a GPU with 1 GiB or more in use and does not take the
repository's GPU lock itself.

## Measured (H100 NVL 94 GB, frozen final-holdout-v3 seeds)

Score = sqrt(masked x psum) of the per-family geomean TFLOPS, as the Judge.

| implementation | 2026-10-29 rerun from this directory (GPU 6, cpuset 24-27,56-59) | earlier run, same harness (GPU 5) | real Judge |
| --- | --- | --- | --- |
| DeepGEMM, Judge protocol (`deepgemm-abi-suite`) | 164.44 (masked 166.5, psum 162.5) | 166.42 | INVALID (source_ok) |
| same, repeated after the oracle (`-r2`) | 166.99 | 171.95 | |
| same + 158 private edge cases first (`-edge`) | 165.81, edge 158/158 passed | 170.38, 158/158 | |
| DeepGEMM, SFA transpose hoisted | 191.90 | 194.23 | |
| DeepGEMM, kernel-only | 261.50 | 261.32 | |
| scored oracle `kernel_v001` (Triton), Judge protocol | 128.74 | 128.20 | 124.44 valid (`pvsa-dyn-grouped-fp8-oracle-120736`) |
| scored oracle `kernel_v001`, kernel-only | 193.97 | 194.02 | |

Paper number under the Judge protocol: about 165-170 TFLOPS (mean of the six
suite runs 167.7), i.e. about 1.3x the scored oracle.  Most of the gap
between the protocol number and the kernel-only number is host launch cost
and the in-call SFA transpose (about 14% of protocol time).

Also measured once (earlier session, GPU 5, not re-run): the newest upstream,
DeepGEMM `5081f8513af57e61696de4f4f04b0e707ffb503e` (v2.8.2, DeepJIT host
runtime; same SM90 kernel and heuristics), with gcc-13: protocol 164.08 /
167.52, edge 158/158, hoisted 188.21, kernel 260.73.  It is not faster than
the pinned 2.6.1 here, so the pinned commit is the reference.

Caveats: candidate sha256 of `deepgemm_v001/submission.py` changed only in its
docstring between the two sessions (code identical).  Compare numbers only
within one card and session; agent finals reached about 195-197 on the real
Judge on these cards, so neither the oracle nor this reference is a ceiling.
