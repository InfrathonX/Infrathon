# Forced-oracle Judge: FlashAttention-3 through the real Judge (local only)

This directory scores the task's source-paper implementation, upstream
FlashAttention-3, with the task's **own private Judge**: the frozen final
holdout, the 160 private edge cases, the compute-sanitizer pass, the 10
scored cases × 7 trials, and the Judge's aggregation and record checks.
Only the rules that apply to agent submissions are forced open. The result
is the reference number for the paper, not an admissible submission.

`oracle/FINAL_VERSION` selects the forced candidate `kernel_v003`. The
admissible Oracle, used by `scripts/smoke_test.sh --oracle`,
`scripts/make_oracle_archive.sh` and the tests, is
`oracle/ADMISSIBLE_VERSION` (`kernel_v002`, vLLM Triton). See `oracle/notes.md`.

## Provenance

- Upstream: https://github.com/Dao-AILab/flash-attention @
  `0251105a2fb19d2957484b7f023cd8c115286ced` (`hopper/`, `flash_attn_3`),
  submodule `csrc/cutlass` @ `7127592069c2fe01b041e174ba4345ef9b279671`.
  It is built for SM90 (`sm_90a`) by `../Dockerfile`, which compiles only the
  variants this ABI uses.
- Candidate: `oracle/candidates/kernel_v003/submission.py`, sha256
  `b501a7a41dbf73e4b5bc7389ad2f132d55b196c2b19144903becaa56ca29015f`.
  Its code is identical to `../fa3_entry.py`; only the docstring differs. It
  makes one call to `torch.ops.flash_attn_3.fwd` with `out_=out`,
  `seqused_k=cache_seqlens`, `page_table`, `causal=True`, `num_splits=0` and
  `pack_gqa=None`, so FA3 picks its own split and pack-GQA settings. This is
  the call `flash_attn_with_kvcache` makes.
- The upstream tree is not vendored. `FA3_IMAGE` is the paper-reference image
  that `../run_fa3_reference.sh` builds from `FA3_SOURCE`. If that image is
  missing, `run_forced_oracle.sh` builds it from `FA3_SOURCE`.

## What is forced, and what is not

The image is `<private judge image>:forced-oracle`, built `FROM` the task's
private Judge image (`scripts/lib.sh` `JUDGE_IMAGE`, the `:local` tag that
holds the holdout). The build adds the following:

- the `flash_attn_3` and `einops` packages, copied from `FA3_IMAGE` into
  site-packages (the base is the same: pytorch 2.8.0 / CUDA 12.9 / Python
  3.11);
- `forced_admission.py`, which runs the Judge's `source_admission.py`
  unchanged, keeps its decision, and admits a rejected submission anyway. It
  records the original errors under `forced.original_admission`;
- `forced_worker.py`, which is the Judge's `candidate_worker.py`, loaded by
  path and run unchanged, except that `run_suite` gets
  `stop_on_edge_failure=False`;
- `forced_verify.py`, which imports the Judge's `verify_paged_gqa.py`
  unchanged and replaces only `evaluate()`. The sanitizer pass runs exactly
  as in the Judge, but a failure is recorded instead of ending the run. The
  worker is `forced_worker.py`. If every gate passes, the result is the
  Judge's own valid result. If a gate fails, the result is the Judge's
  trusted gate failure, with score 0 and the failing gate named. The scored
  numbers that were still measured are then kept under `forced_scored_*`,
  after the Judge's own per-case validation and aggregation.

The Judge's own files are not changed, so the protected-hash check still
runs and passes: `verify_paged_gqa.py`, `candidate_worker.py`,
`source_admission.py`, `run_judge.py`, `protected_hashes.txt` and the
runtime. `run_forced_oracle.sh` runs the same `sforge eval` that
`scripts/eval_archive.sh` runs: the docker backend, the task id, the 8-CPU /
32 GB limits, GPU binding by UUID, the per-UUID GPU lock, and the idle and
container-pin checks. It changes three things:

- It uses a temporary copy of `tasks/` in which `judge.image_tag` is the
  forced tag and `eval_cmd` passes `--verifier forced_verify.py`.
- It passes `--skip-integrity-review` and sets `SFORGE_LLM_JUDGE_MODE=off`,
  so the LLM integrity review is bypassed.
- It sets `SFORGE_JUDGE_NETWORK_MODE=none`.

Gates bypassed: **source admission** and the **LLM integrity review**.
Recorded but not fatal: **compute-sanitizer** and the **edge-case** gate.
Unchanged: the protected hashes, the profile, the manifests, the hardware
check, every scored-case record check, and the timeouts.

## Run

```bash
# once: the FA3 paper-reference image (CPU-only build, tens of minutes)
FA3_SOURCE=/path/to/flash-attention oracle/paper_reference/run_fa3_reference.sh --gpu <idx> --tuned
# forced Judge run (needs SFORGE_ROOT pointing at an sforge checkout with .venv)
SFORGE_ROOT=/path/to/mlsys_forge LOGS_DIR=/path/to/outputs/<task dir> \
SFORGE_JUDGE_CPUSET=12-15,44-47 FA3_IMAGE=standalone_paged_gqa_decode.paper-fa3:0251105a \
  oracle/paper_reference/forced_judge/run_forced_oracle.sh --gpu <idx> [--skip-build] [--run-id ID]
```

The script never retags or overwrites `:local` and never pushes. It writes
`report.json`, `private_judge_artifacts.tar.gz` (the verifier's full
result, including `forced_*`) and `forced_receipt.txt` (image ids, archive
sha256, GPU and cpuset) under `${LOGS_DIR}/runs/<run id>/`.

## Measured (H100 NVL 94 GB, GPU 3, cpuset 12-15,44-47, 2026-10-29)

Forced image `525e09988b24`, built FROM the private Judge image
`f5461236c1ce` with FA3 from `paged-gqa-fa3-oracle:0251105a`
(`d700e1078853`, built with `../Dockerfile`). The archive sha256 is
`80414f28cfa7c650b484a95b72d25994981a9b2cebadef2d8ba11bb991d3337f`
(`FINAL_VERSION` 3).

| run id | Judge verdict | failing gate | edge | sanitizer | scored grid q1 / q16 / score (TFLOPS) |
|---|---|---|---|---|---|
| `forced-paged-gqa-fa3-v003-r2-g3-20261029-215854` | INVALID, score 0 | `edge_correctness_ok` | 3/160 fail | passed | 15.52 / 184.97 / **53.58** |
| `forced-paged-gqa-fa3-v003-r3-g3-20261029-220005` | INVALID, score 0 | `edge_correctness_ok` | 3/160 fail | passed | 15.51 / 185.15 / **53.58** |
| `forced-paged-gqa-fa3-v003-r1-g3-20261029-214509` (earlier `forced_verify.py`) | INVALID, score 0 | `edge_correctness_ok` | 3/160 fail | passed | 15.61 / 185.25 / 53.78, recomputed from per-case records |

The failing edge cases are `edge-061-lengths`, `edge-098-heads` and
`edge-099-heads`. In each one the output "contains ... unwritten, NaN or
infinite elements": NaN/Inf in unused KV page slots leaks into the output.
The scored grid passes every scored-case check (correctness, guard bands
and sentinel). The forced scored number agrees with the `../` harness
(53.7-55.2). For comparison, the admissible Oracle `kernel_v002` scores
24.03 on the normal Judge on the same card. To reproduce:

```bash
SFORGE_ROOT=<mlsys_forge> LOGS_DIR=<mlsys_forge>/outputs/standalone-paged-gqa-decode-bf16-h100-v2-scratch \
SFORGE_JUDGE_CPUSET=12-15,44-47 FA3_IMAGE=paged-gqa-fa3-oracle:0251105a \
  oracle/paper_reference/forced_judge/run_forced_oracle.sh --gpu 3 --skip-build --run-id <id>
```

(The image was built without `--skip-build` once, with the same `FA3_IMAGE`.)
