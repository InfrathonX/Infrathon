# Forced-oracle Judge: upstream DeepGEMM through the real Judge (local only)

This directory scores upstream DeepGEMM, which the task's grid and generator
are taken from, with the task's **own private Judge**: the frozen final
holdout, the 160 private edge cases, the compute-sanitizer subset, the 7
scored cases × 5 trials, and the Judge's record checks and aggregation.
Only the rules that apply to agent submissions are forced open. The result
is the reference number for the paper, not an admissible submission. The
admissible qualification control is still `baseline/` (Triton, about 91
TFLOPS, Judge-valid).

`oracle/FINAL_VERSION` (3) selects `oracle/candidates/deepgemm_oracle_v003`.
`run_forced_oracle.sh` packs it as `candidates/kernel_v003/submission.py`
with `FINAL_VERSION` 3.

## Provenance

- Upstream: https://github.com/deepseek-ai/DeepGEMM @
  `559d79fb6994a58b8a15b4b93bf13ccc16edf247` (v2.6.1); `third-party/cutlass`
  @ `f3fde58372d33e9a5650ba7b80fc48b3b49d40c8`, `third-party/fmt` @
  `553ec11ec06fbe0beebfbb45f9dc3c9eabd83d28`. It is built in place with
  upstream's `develop.sh`. The kernels are JIT-compiled for `sm_90a` at run
  time and cached in the worker's `$HOME/.deep_gemm`, which is cold in every
  Judge container.
- Kernel and configuration: SM90 `k_grouped_fp8_gemm_nt_contiguous`
  (`sm90_fp8_gemm_1d1d`, `KGroupedContiguous`) with upstream's own heuristic.
  Nothing is overridden.
- Candidate: `oracle/candidates/deepgemm_oracle_v003/submission.py`, sha256
  `1c19d0ffcdbf8e644331863c5523021f1e57cb9756f671c0f588300c96f34afd`. Its
  code is identical to `deepgemm_oracle_v002` (the lean glue, about 279
  TFLOPS on the `run_oracle.sh` harness); only the docstring differs.
- The upstream tree is not vendored. It is passed as the named build context
  `deepgemm=${DEEPGEMM_SOURCE}`. Stage 1 of the `Dockerfile` repeats the head
  of `oracle/Dockerfile` instruction for instruction, so a matching oracle
  image build can be reused from the cache.

## What is forced, and what is not

The image is `<private judge image>:forced-oracle`, built `FROM` the task's
private Judge image (`scripts/lib.sh` `JUDGE_IMAGE`, the `:local` tag that
holds the holdout). The build adds the following:

- `/opt/deepgemm` and a `.pth` file. The Judge's worker clears `PYTHONPATH`.
- `forced_admission.py`, which runs the Judge's `source_admission.py`
  unchanged, keeps its decision, and admits a rejected submission anyway. It
  records the original errors (`forbidden import: 'deep_gemm'`) under
  `forced.original_admission`.
- `forced_verify.py`, which imports the Judge's `verify_wgrad.py` unchanged
  and replaces only `evaluate()`, with **per-case isolation and strict
  timeouts**. Upstream's SM90 kernel does not support most edge shapes: some
  raise, some fault the context, and some spin on the GPU. In the Judge, a
  single hang would kill the whole worker and leave no result. So:
  1. each of the 8 sanitizer-flagged cases runs alone under the Judge's own
     compute-sanitizer command (`FORCED_SANITIZER_CASE_TIMEOUT`, 300 s);
  2. each of the 160 edge cases runs alone in the Judge's own
     `candidate_worker.py` (`FORCED_EDGE_CASE_TIMEOUT`, 240 s, then SIGKILL);
  3. the scored grid runs in one more Judge worker, with the Judge's trials,
     warm-ups and 2400 s timeout.

  A failed sanitizer or edge case is recorded, and the run continues. If
  every gate passes, the combined record goes through the Judge's own
  `validate_runtime_result`, and the result is the Judge's valid result.
  Otherwise the result is the Judge's trusted gate failure, with score 0 and
  the failing gate named. The scored numbers are kept under
  `forced_scored_*` (after the Judge's per-case validation and aggregation),
  and the per-case edge outcomes are kept under `forced_edge_cases`.
- Image `ENV`: the verifier total timeout is 13800 s and the Judge timeout is
  14100 s, against the Judge's 3300 s and 3400 s. Per-case process start-up
  and DeepGEMM's per-shape JIT make the isolated run long. The trial and
  warm-up counts stay the Judge's.

The Judge's own files are not changed, so the protected-hash check still
runs and passes: `verify_wgrad.py`, `candidate_worker.py`,
`source_admission.py`, `run_judge.py`, `protected_hashes.txt` and the
runtime. `run_forced_oracle.sh` runs the same `sforge eval` that
`scripts/eval_archive.sh` runs: the docker backend, the task id, the 8-CPU /
32 GB limits, GPU binding by UUID, the per-UUID GPU lock, and the idle and
container-pin checks. It changes four things:

- It uses a temporary copy of `tasks/` in which `judge.image_tag` is the
  forced tag, `eval_cmd` passes `--verifier forced_verify.py`, and
  `eval_timeout` is `FINAL_EVAL_TIMEOUT` (15000 s).
- It passes `--skip-integrity-review` and sets `SFORGE_LLM_JUDGE_MODE=off`.
- It sets `SFORGE_JUDGE_NETWORK_MODE=none`.

Gates bypassed: **source admission** and the **LLM integrity review**.
Changed: **sanitizer and edge isolation** (per case, recorded, not fatal) and
the **total timeouts**. Unchanged: the protected hashes, the profile, the
manifests, the hardware check, and every scored-case record check.

## Run

```bash
git clone --recursive https://github.com/deepseek-ai/DeepGEMM /path/to/DeepGEMM
git -C /path/to/DeepGEMM checkout 559d79fb6994a58b8a15b4b93bf13ccc16edf247
git -C /path/to/DeepGEMM submodule update --init --recursive
SFORGE_ROOT=/path/to/mlsys_forge LOGS_DIR=/path/to/outputs/<task dir> \
DEEPGEMM_SOURCE=/path/to/DeepGEMM SFORGE_JUDGE_CPUSET=20-23,52-55 \
  oracle/forced_judge/run_forced_oracle.sh --gpu <idx> [--skip-build] [--run-id ID]
```

The script never retags or overwrites `:local` and never pushes. It writes
`report.json`, `private_judge_artifacts.tar.gz` (the verifier's full
result, including `forced_*`) and `forced_receipt.txt` under
`${LOGS_DIR}/runs/<run id>/`.

## Measured (H100 NVL 94 GB, GPU 5, cpuset 20-23,52-55, 2026-10-29)

| run id | Judge verdict | sanitizer | edge | scored grid coarse / fine / score (TFLOPS) |
|---|---|---|---|---|
| `forced-kgrouped-wgrad-deepgemm-v003-r2-g5-20261029-215241` (image `fe8b520f50f0`, committed `forced_verify.py`) | INVALID, score 0; first failing gate `memory_safety_ok`, also `edge_correctness_ok` | 1/8 passed | 50/160 passed | 493.62 / 163.21 / **283.83** |
| `forced-kgrouped-wgrad-deepgemm-v003-r1-g5-20261029-213159` (image `a6ddbcde0f48`, earlier `forced_verify.py`; only `forced_failed_gates` differed) | same | 1/8 | 50/160 | 496.32 / 164.56 / **285.79** |

The archive sha256 is
`299b53abd5f5f83fd87cedefd195580234a162f313874d8b2692cc5c2aaf8668`. The
private Judge image is `b70d98ac2e7b`, and DeepGEMM was built from a
checkout at 559d79f. The per-case breakdown (TMA-descriptor errors, JIT
static asserts, 3 context faults, 2 hangs killed at 240 s) and per-band
numbers are in the "Forced Judge" section of `oracle/README.md`. To
reproduce:

```bash
SFORGE_ROOT=<mlsys_forge> LOGS_DIR=<mlsys_forge>/outputs/standalone-k-grouped-fp8-wgrad-h100-v2-scratch \
DEEPGEMM_SOURCE=<DeepGEMM checkout> SFORGE_JUDGE_CPUSET=20-23,52-55 \
  oracle/forced_judge/run_forced_oracle.sh --gpu 5 --run-id <id>
```
