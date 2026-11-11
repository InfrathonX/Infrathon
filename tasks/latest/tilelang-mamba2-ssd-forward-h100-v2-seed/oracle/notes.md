# Oracle

Positive control for the task, not a target shown to participants. It is
host-only and never copied into an image.

`FINAL_VERSION` selects `kernel_v002`, the strongest admissible upstream (paper)
solution measured on H100. `kernel_v001`, the original positive control, stays
in place.

## kernel_v001: original positive control

`kernel_v001` is a legal, source-only TileLang implementation of the same
`ssd_forward` entry point as the seed. It folds the seed's separate
chunk-cumsum/decay stage into the chunk-state kernel, so the forward runs as
three kernels instead of four:

1. per-chunk state, with the chunk decay computed in place;
2. inter-chunk state passing;
3. the intra-chunk scan, with the entering-state contribution and the `D` skip
   term.

The inter-chunk states are kept in BF16. All dense products use `T.gemm` with
BF16 operands and FP32 accumulators. It goes through the same source admission
and Judge path as a submission (`scripts/make_oracle_archive.sh`,
`scripts/eval_archive.sh`).

## kernel_v002: upstream SSD decomposition in TileLang, offline-tuned (selected)

- Upstream algorithm: state-spaces/mamba, https://github.com/state-spaces/mamba,
  commit `e9594ce1c732d97440f0332fdc43170a2294dbfa`,
  `_mamba_chunk_scan_combined_fwd` (`ssd_combined.py:374-387`). Mamba-2 paper
  (Dao & Gu, "Transformers are SSMs", ICML 2024). The stages are chunk cumsum,
  chunk state (FP32), state passing (BF16 out), CB = C B^T, and chunk scan.
- Upstream kernels: tile-ai/tilelang, https://github.com/tile-ai/tilelang,
  commit `f935b2ad27152c9d970b15c666acaf3ed73e5f2a`:
  - chunk state: `examples/linear_attention/example_mamba_chunk_state.py`
    (`chunk_state_fwd`);
  - chunk scan: `benchmark/mamba2/benchmark_mamba_chunk_scan.py`
    (`chunk_scan_fwd`), with TL_ENABLE_FAST_MATH and block_Dstate = dstate.
  - The glue stages (cumsum, state passing, bmm) have no TileLang upstream.
    They are direct TileLang transcriptions of the mamba_ssm Triton kernels.
- Changes vs upstream, ABI and dtype only:
  - out-parameter ABI, BF16 x/B/C/out, FP32 dt/A/D;
  - dt and dA_cumsum are FP32, as in mamba_ssm, where upstream TileLang uses
    FP16;
  - chunk states are FP32 and entering states BF16 (mamba_ssm
    `states_in_fp32=True`);
  - the elementwise passes on the CB tile run as one FP32 pass, rounded to BF16
    once before `T.gemm`;
  - no `tilelang.autotune` inside the submission, which is forbidden.
- Tuning: offline, which POLICY allows ("offline autotuning ... followed by
  bounded, static shape-based kernel choices"), over upstream's own autotune
  search spaces. It ran on an H100 NVL in the Work image at L=4096 per family,
  and the result was baked into the literal `TUNED` table:
  - Mamba-2 and Codestral: chunk 256, BF16 CB; chunk_state
    64/128/64 with 3 stages; chunk_scan 64/64/64 with 2 stages (upstream
    default).
  - Falcon-H1 (dstate 256): chunk 256, FP32 CB; chunk_state 128/128/64 with
    2 stages; chunk_scan 128/64/64 with 2 stages.
  - Unlisted geometries fall back to the upstream default configs.
- Caveats:
  - Static-shape JIT costs about 12 s of warm-up per new shape. The budget gate
    still passes.
  - BF16 CB fails two ragged-tail edge fixtures for Falcon (dstate 256), so
    Falcon keeps FP32 CB. The first tuned bake, which used BF16 CB for Falcon,
    was invalid (`pvsa-tilelang-mamba2-paper-v003tuned-r1`).
  - TileLang 0.1.13 deadlocks intermittently on some Falcon chunk_scan configs
    (block_N=32 / block_K=64). Those configs were blacklisted during tuning and
    are not in the table.
  - Source admission passes. Evaluate it as a trusted control
    (`TRUSTED_CONTROL=1`, `SFORGE_LLM_JUDGE_MODE=off`), because it
    intentionally copies upstream code.
- sha256(submission.py) = `c5ea001793aa462f171e6976f7c9cd80abd17cd8ef85ca5bd0e54f93014b447a`.
  This equals `candidates/kernel_v003/submission.py` of the paper archive
  `paper_submission_v003_tunedfix.tar.gz` (sha256 `c60062d4...adfae`), which
  was measured earlier.

## Judge scores (H100 NVL 94 GB, 400 W, final suite)

All runs used the real Judge (`scripts/eval_archive.sh`, Docker, 8 CPUs and
32 GB). Every run passed 11/11 gates: correctness on 18 scored cases plus 18
edge fixtures, fresh input/output, budget, tilelang_only and hardware.

| archive | run id | GPU / Judge cpuset | score | mamba2 | codestral | falcon |
| --- | --- | --- | --- | --- | --- | --- |
| kernel_v002 (new oracle) | oracleup-tilelang-mamba2-new-r1-gpu1-20261029-194526 | 1 / 4-7,36-39 | **61.97** | 56.48 | 53.76 | 78.36 |
| kernel_v002 (new oracle) | oracleup-tilelang-mamba2-new-r2-gpu1-20261029-194526 | 1 / 4-7,36-39 | **61.95** | 56.46 | 53.73 | 78.36 |
| kernel_v001 (old oracle) | oracleup-tilelang-mamba2-old-r1-gpu1-20261029-194526 | 1 / 4-7,36-39 | 48.56 | 45.18 | 46.25 | 54.80 |
| kernel_v002, same `submission.py` bytes (paper v003 tunedfix) | pvsa-tilelang-mamba2-paper-v003tunedfix-r1 / -r2 | 2 / 16-23 | 62.51 / 62.44 | 57.08 | 54.31 | 78.80 |
| kernel_v001 | pvsa-tilelang-mamba2-oracle-r1 | 2 / 16-23 | 48.73 | 45.26 | 46.44 | 55.06 |

On the same card (GPU 1, back to back), kernel_v002 is 1.28x kernel_v001. Other upstream variants measured with the
same protocol were weaker:

- the untuned TileLang port (upstream default configs): 57.5;
- upstream's production mamba_ssm Triton kernels: 46.4 at best (chunk 256 for
  Falcon). These were measured in a separate harness with the task runtime;
  they are not admissible because the task is TileLang-only.

## v3 suite pilot (H100 SXM 80 GB, 2026-11-08)

The v3 final suite (`tilelang-ssd-final-v3`: 138 scored cases, 24 edge fixtures,
3 trials) run end to end with the trusted runtime in the Work image, TileLang
cache disabled, 8 CPUs, one GPU each, then checked with the Judge verifier
(`validate_runtime_result`). The oracle was then evaluated through the real Judge
(`scripts/eval_archive.sh`, private image with the 24 layers, 8 CPUs, 32 GB):
valid, every gate passed, score 52.51 (batch 55.59, serving 33.55, long 66.73,
real 61.09) in 30.1 min (`mamba2-v3-oracle-eval-r1`).

| kernel | wall | score | batch | serving | long | real | worst real rel L2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| kernel_v001 (seed) | 32.5 min | 27.77 | 30.88 | 21.29 | 35.88 | 25.21 | 5.23e-3 |
| kernel_v002 (oracle) | 30.4 min | 52.35 | 55.96 | 33.32 | 66.43 | 60.61 | 5.23e-3 |
| r1 best (agent-8 v058) | 36.0 min | 61.20 | 78.37 | 36.06 | 80.47 | 61.68 | 5.24e-3 |

- Compilation is 75-80% of the wall (one cold compile per distinct shape, 12-16 s
  per shape for the seed and the oracle, up to 21 s for v058 on its batch-1 path).
  Edge fixtures take 5.5 min, references 40 s, the timed calls under 2 s in total.
- v058 scored 91.56 on v2 and 61.20 here: its persistent path needs high
  batch x heads, and the serving family (batch 1) is its weakest.
- On the v3 suite the oracle (52.35) is 1.89x the seed (27.77).
- The real-layer limit (1.3e-2) is 2.5x the worst honest value above.
