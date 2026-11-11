# Forced-oracle Judge: DeepGEMM through the full Judge pipeline (LOCAL ONLY)

The oracle does not have to follow the rules imposed on agent submissions
(source-admission import allowlist, no DeepGEMM / compiled extensions, LLM
integrity review).  This directory scores the strongest official paper
solution, upstream DeepGEMM, with the task's **real** Judge pipeline by
building a local-only Judge image that admits it.  The result is a normal
Judge `report.json` (plus the private per-case artifact) for
`oracle/candidates/kernel_v002/submission.py`, which `oracle/FINAL_VERSION`
selects.

Never use this image or script for an agent archive, never push the image,
and never retag it `:local`.

## What differs from the normal Judge

| | normal Judge | forced-oracle Judge |
|---|---|---|
| image | `standalone_grouped_fp8.judge.standalone_dynamic_grouped_fp8_h100_scratch_v3:local` (private: public Judge + frozen holdouts) | `...:forced-oracle`, `FROM` that image (`Dockerfile`) |
| DeepGEMM | absent | upstream tree built in place (`develop.sh`) at `/opt/deepgemm`, importable through a site-packages symlink `deep_gemm -> /opt/deepgemm/deep_gemm`; kernels JIT-compile with the image's `nvcc` on first use into the worker's own `~/.deep_gemm` (cold cache every Judge run; the compile time falls in warm-up/edge calls, not in timed calls) |
| source admission | `source_admission.py` (allowlist + bans) | `forced_source_admission.py` installed as `source_admission.py`; the original runs unchanged as `source_admission_enforcing.py` and makes every structural decision (layout, `FINAL_VERSION`, selected path, SHA-256), only its verdict is overridden; its errors are kept as warnings |
| `protected_hashes.txt` | original | the `judge:source_admission.py` line updated to the override's hash; every other record unchanged and still checked |
| LLM integrity review | runs for agent archives | skipped (`TRUSTED_CONTROL=1`, as for every trusted oracle smoke; `SFORGE_LLM_JUDGE_MODE=off`) |
| everything else | | identical: `verify_grouped.py`, `run_judge.py`, `candidate_worker.py`, trusted runtime, frozen final holdout (32 cases) and 158 private edge cases, compute-sanitizer memcheck pass (6 cases), 5 fresh-tensor trials + 1 warm-up, decoy pre-call, guard bands, sentinel, plausibility bound, aggregation |

The run is the task's own `scripts/eval_archive.sh` (`sforge eval --backend
docker`: GPU pinned by UUID, the per-UUID GPU lock, `--network none`, 8 CPUs /
32 GiB, optional `SFORGE_JUDGE_CPUSET`), with `TASKS_DIR` pointing at a
temporary copy of `tasks/standalone_dynamic_grouped_fp8_h100_scratch_v3.json`
whose only change is `judge.image_tag = forced-oracle`.

## Provenance

- Upstream: https://github.com/deepseek-ai/DeepGEMM @
  `559d79fb6994a58b8a15b4b93bf13ccc16edf247` (v2.6.1, the commit the task was
  ported from); `third-party/cutlass` @ `f3fde58372d33e9a5650ba7b80fc48b3b49d40c8`,
  `third-party/fmt` @ `553ec11ec06fbe0beebfbb45f9dc3c9eabd83d28`.  Kernels and
  heuristics unchanged (SM90 `sm90_m_grouped_fp8_gemm_masked_1d2d` /
  `sm90_m_grouped_fp8_gemm_contiguous_1d2d` + upstream `transpose_fp32` for the
  SFA layout).
- Build: `../Dockerfile` (FROM the private Judge image, `develop.sh`) produces
  the paper image; `Dockerfile` here copies `/opt/deepgemm` from it into the
  private Judge image.  The upstream tree is supplied as the `deepgemm` named
  build context (`DEEPGEMM_SOURCE`), never vendored; `run_forced_oracle.sh`
  checks the paper image's `/opt/deepgemm` is at `559d79f`.
- Candidate: `oracle/candidates/kernel_v002/submission.py`, sha256
  `e401a21653807130c3f91b6eaddef9edc2fd5a9ca04fa0e23cb48192d946fcd4` =
  `../candidates/deepgemm_v001/submission.py` (the 1:1 public-API adapter,
  sha256 `cb8dc853...0f6d7c`) plus one line of library configuration,
  `os.environ.setdefault("DG_USE_TEMP_CUBLASLT_WORKSPACE", "1")`, before
  `import deep_gemm`.  This is upstream's own compute-sanitizer setting
  (`csrc/jit/device_runtime.hpp`: "Enabled by compute-sanitizer tests, which
  trigger `cudaErrorCudartUnloading` when the workspace tensor is destructed
  after CUDA driver shutdown"); the workspace is used only by DeepGEMM's
  cuBLASLt route (`smxx_cublaslt.hpp`), never by the two grouped FP8 GEMMs.
  Without it, run r1 below: every scored and edge case passed (scored cases
  166.12), but the compute-sanitizer worker aborted at process exit
  (`terminate ... c10::AcceleratorError: CUDA error: driver shutting down`
  from `deep_gemm::DeviceRuntime::~DeviceRuntime`, memcheck
  `ERROR SUMMARY: 0 errors`), which the Judge correctly fails as
  `memory_safety_ok` ("the compute-sanitizer pass did not complete").
- SFA-transpose hoisting is not usable as glue: the ABI hands the candidate a
  K-major FP32 `a_scale` on every call (fresh tensors per trial, decoy
  pre-call), so the MN-major TMA-aligned copy SM90 DeepGEMM needs must be made
  inside the call.  `../oracle_runner.py --mode hoisted` (191.9 on the
  harness) moves it outside the timer and is not a task score.

## Build and run

```bash
git clone --recursive https://github.com/deepseek-ai/DeepGEMM /path/to/DeepGEMM
git -C /path/to/DeepGEMM checkout 559d79fb6994a58b8a15b4b93bf13ccc16edf247
git -C /path/to/DeepGEMM submodule update --init --recursive
scripts/build_images.sh        # the private Judge image this derives from

# from the task root; builds the paper image first if it is missing
DEEPGEMM_SOURCE=/path/to/DeepGEMM oracle/paper_reference/forced_judge/run_forced_oracle.sh --build-only
# one full Judge run on an idle H100 (about 20 min, mostly cold-cache JIT compiles)
PHYSICAL_GPU=<idx> SFORGE_JUDGE_CPUSET=<cpus of that GPU's slice> \
  oracle/paper_reference/forced_judge/run_forced_oracle.sh --skip-build <RUN_ID>
```

`PAPER_IMAGE=<tag>` reuses an existing paper image (the runs below used
`dyn-grouped-fp8-paper-harness:local`, built from `../Dockerfile` at the same
commit).  `EVAL_TASK_ROOT=<task tree>` runs the Judge from another checkout of
this task whose build manifest matches the local `:local` images (the runs
below used a development checkout of the same task; the oracle archive is
always packed from this tree).  The report lands at
`<outputs>/runs/<RUN_ID>/standalone_dynamic_grouped_fp8_h100_scratch_v3/submissions/manual-1/report.json`.

## Measured (H100 NVL 94 GB, 2026-10-29)

Score = sqrt(masked x psum) of the per-family geomean TFLOPS (higher is better).

| run id | candidate | Judge | GPU / cpuset | valid | score | masked | psum | sanitizer | edge |
|---|---|---|---|---|---|---|---|---|---|
| `forced-dyn-grouped-fp8-deepgemm-r2-g6` | `kernel_v002` (sha `e401a216`) | forced-oracle | 6 / 24-27,56-59 | yes | **166.76** | 169.90 | 163.68 | ran, passed (6) | 158/158 |
| `forced-dyn-grouped-fp8-deepgemm-r3-g6` | `kernel_v002` (sha `e401a216`) | forced-oracle | 6 / 24-27,56-59 | yes | 164.94 | 169.21 | 160.79 | ran, passed (6) | 158/158 |
| `forced-dyn-grouped-fp8-deepgemm-r1-g6` | verbatim `deepgemm_v001` (sha `cb8dc853`) | forced-oracle | 6 / 24-27,56-59 | **no**, `memory_safety_ok` | 0 (scored cases: 166.12) | (168.40) | (163.86) | aborted at exit, 0 memcheck errors | 158/158 |
| `forced-dyn-grouped-fp8-admissible-v001-r1-g7` | `kernel_v001` (Triton, `--admissible` archive) | normal | 7 / 28-31,60-63 | yes | 126.88 | 127.28 | 126.47 | ran, passed (6) | 158/158 |

Forced image `...:forced-oracle` = `sha256:ffcbc8b8f0b4183944de266ff5eae267127488c9a291130642063510495f2fcd`, built from
private Judge `...:local` = `sha256:2631448c7fbd74e510b636bf54b6a50574c285ff9cb6b01865e51f96d06d062b` and paper image
`dyn-grouped-fp8-paper-harness:local` (`285268820e51`, DeepGEMM at 559d79f).
Gates bypassed: source admission verdict (bypassed errors:
`candidates/kernel_v002/submission.py:52: forbidden import: 'os'` and
`...:64: forbidden import: 'deep_gemm'`) and the LLM integrity review.  No other gate was bypassed.  The
normal Judge rejects DeepGEMM at `source_ok`
(`pvsa-dyn-grouped-fp8-paper-r1-120736`).
