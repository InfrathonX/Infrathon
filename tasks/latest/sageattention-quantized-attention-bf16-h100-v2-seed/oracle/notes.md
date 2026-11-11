# Oracle

Host-only positive control and performance reference. It is never copied into
the Work or Judge image and is not shown to participants.

| selector | candidate | what | Judge score (GPU 2, final-holdout-v2) | gates |
| --- | --- | --- | --- | --- |
| seed / old oracle | `kernel_v001` | the Work seed (SageAttention-1-derived) | 297.67 (`oracleup-sageattention-old-r1-gpu2-20261029-194528`) | all pass |
| `ADMISSIBLE_VERSION` = 2 | `kernel_v002` | upstream SageAttention-1 Triton, best **admissible** paper path | 350.74 / 348.85 (`oracleup-sageattention-new-r1/-r2-gpu2-20261029-194528`) | all pass, normal Judge |
| `FINAL_VERSION` = 4 | `kernel_v004` | upstream SageAttention2 sm90 CUDA, per_warp quantizer, **forced** | **581.73 / 577.88 / 581.80** (`forced-sageattention-v004-r1-gpu2-20261029-211128`, `-v004-r2-gpu2-20261029-211539`, `-v004-r3final-gpu2-20261029-215207`, the `make_oracle_archive.sh` archive) | source admission **bypassed**; every other gate passes (8 edge, 62 real layers, correctness, fresh input/output) |
| (not selected) | `kernel_v003` | the same, per_thread quantizer (upstream default), **forced** | 549.72 / 545.53 (`forced-sageattention-v003-r1-gpu2-20261029-211336`, `-v003-r2-gpu2-20261029-211746`) | admission bypassed; all other gates pass |

`FINAL_VERSION` selects the strongest paper candidate, which is not
admissible (it imports the compiled `sageattention` extension). Anything that
must pass the normal, admission-enforcing Judge uses `ADMISSIBLE_VERSION`:
`scripts/smoke_test.sh --oracle` and the public-runner smoke, through
`ORACLE_SELECTOR=ADMISSIBLE_VERSION scripts/make_oracle_archive.sh`
(`artifacts/oracle-admissible.tar.gz`). `scripts/make_oracle_archive.sh` with
no selector packs `FINAL_VERSION`; that archive scores only through
`paper_reference/forced_judge/run_forced_oracle.sh`.

## kernel_v001: seed control

The admitted SageAttention-1-derived positive control. It is byte-identical to
`images/work/workdir/quant_attention_lab/candidates/kernel_v001/submission.py`,
and `scripts/validate_layout.py` and `tests/test_contract.py` enforce that. It
uses a Triton K mean, per-block INT8 Q/K, exp2 online softmax, and FP16 P·V
accumulated into FP32. It deliberately leaves fusion, finer scales, FP8 and
Hopper scheduling as optimization headroom.

## kernel_v002: upstream SageAttention-1 Triton (ADMISSIBLE_VERSION)

- Upstream: thu-ml/SageAttention, https://github.com/thu-ml/SageAttention,
  commit `d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5` (2026-01-17), Apache-2.0.
  Paper: SageAttention (ICLR 2025). It is the
  `sageattn_qk_int8_pv_fp16_triton` path, the only attention path upstream
  ships in Triton.
- Algorithm: smooth-K, per-block INT8 Q (128 tokens) and K (64 tokens), exp2
  online softmax, and FP16 P·V with an FP16 MMA accumulator added into FP32.
  The kernel bodies are copied verbatim from `sageattention/triton/*.py`, with
  upstream's tile, warp and stage configs.
- Adapted for the task ABI, glue only:
  - the K mean, the K-mean subtraction and the V-to-FP16 cast moved from host
    torch into Triton kernels;
  - the output is written straight into `out`;
  - `+1e-7` on the scales;
  - LSE is disabled.
- Source admission passes. The Judge archive must be evaluated as a trusted
  control (`TRUSTED_CONTROL=1` / `SFORGE_LLM_JUDGE_MODE=off`), because this
  file intentionally copies upstream code, which the LLM integrity policy
  forbids for agent submissions.
- Why it beats kernel_v001 although the attention kernel is the same: the K
  mean is reduced in parallel. kernel_v001 reduces it with one program per
  head, serially over L, and L is 12k-29k tokens. This explanation was not
  profiled further.
- sha256(submission.py) = `a08f34ce268d308b74b8309152d0504992f847fd3a0467a9b1e5fdbafa30e9c3`.

## Judge scores (H100 NVL 94 GB, 400 W, final-holdout-v2)

All runs used the real Judge (`scripts/eval_archive.sh`, Docker, 8 CPUs and
32 GB). Every run passed 10/10 gates: 8 edge cases and 62 real layers.

| archive | run id | GPU / Judge cpuset | score | dit | causal | D1 / D2 / D3 / L1 / L2 |
| --- | --- | --- | --- | --- | --- | --- |
| kernel_v002 (new oracle) | oracleup-sageattention-new-r1-gpu2-20261029-194528 | 2 / 8-11,40-43 | **350.74** | 341.59 | 360.13 | 353.0 / 361.8 / 312.0 / 369.4 / 351.1 |
| kernel_v002 (new oracle) | oracleup-sageattention-new-r2-gpu2-20261029-194528 | 2 / 8-11,40-43 | **348.85** | 337.59 | 360.49 | 351.2 / 356.3 / 307.4 / 372.4 / 349.0 |
| kernel_v001 (old oracle = seed) | oracleup-sageattention-old-r1-gpu2-20261029-194528 | 2 / 8-11,40-43 | 297.67 | 304.04 | 291.43 | 306.8 / 319.1 / 287.1 / 303.3 / 280.0 |
| kernel_v002, same `submission.py` bytes | pvsa-sageattention-paper-triton-gpu0-r1 / -r2 | 0 / 0-7 | 343.66 / 340.69 | | | |
| kernel_v001 | pvsa-sageattention-oracle-gpu0-r1 | 0 / 0-7 | 289.93 | | | |

On the same card, kernel_v002 is 1.18x kernel_v001. On this card the
same-run cuDNN BF16 SDPA reference measured 439-448 TFLOPS, so the admissible
paper path still runs at about 0.79x SDPA. All gate ratios sit at about 0.50
for both candidates, because the gates are defined relative to a Sage1
emulation.

## Strongest paper path (not admissible): forced Judge

Upstream's own H100 path is SageAttention2/2++ on the sm90 wgmma CUDA path
(`sageattn_qk_int8_pv_fp8_cuda_sm90`, what `sageattn()` dispatches to on
sm90). It is a compiled extension, which `source_admission.py` forbids by
design.

### kernel_v003 / kernel_v004: forced (FINAL_VERSION = 4)

- Upstream: thu-ml/SageAttention @ `d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5`,
  compiled for sm_90a (`TORCH_CUDA_ARCH_LIST=9.0`) by `paper_reference/Dockerfile`.
- The two files are `paper_reference/paper_sm90_candidate.py` with the
  quantizer fixed instead of read from an environment variable (the Judge
  passes no environment to the candidate): INT8 QK, FP8 PV,
  `pv_accum_dtype="fp32+fp32"`, smooth-K, the CUDA kernel writing straight
  into `out`. All host-side preprocessing stays inside the timed call.
  - `kernel_v003`: `qk_quant_gran="per_thread"`, upstream's default (Triton
    quantizer). sha256 `381eeb20c74c5e217a505459e2997ec2c61766b3511a55b35e3d8aad014fa79c`.
  - `kernel_v004`: `qk_quant_gran="per_warp"`, an upstream option (upstream's
    own CUDA `per_warp_int8`). It is about 6% faster on H100 and passes the
    same gates, so it is the selected forced oracle. sha256
    `92dbdb5249fd6477bfb17e2bff72ba06cbe8b25c70fb3ace568fa092a542efda`.
- Scored by the task's normal Judge pipeline (`sforge eval` -> `run_judge.py`
  -> `verify_quant_attention.py` -> `candidate_worker.py` -> trusted
  runtime: frozen final-holdout-v2, 8 edge cases, 62 real layers, 1 warm-up +
  7 fresh-input trials, aggregation) in the local-only image built by
  `paper_reference/forced_judge/`, whose only change is that source admission
  records and bypasses its errors. The LLM integrity review is off. See
  `paper_reference/forced_judge/README.md` for the command, the bypassed
  admission errors and the per-cell numbers.
- On the same card the same-run cuDNN BF16 SDPA reference measured
  435-445 TFLOPS, so the forced oracle runs at about 1.31x SDPA, against about
  0.79x for the admissible `kernel_v002`.

`paper_reference/` also keeps the round-1 host harness (`run_paper_reference.sh`),
which calls the trusted runtime's `run_suite` directly; its numbers (per_warp
581.3, per_thread 551.8 on GPU 2) agree with the forced Judge runs. Nothing in
`paper_reference/` reaches a Judge archive or a published image:
`scripts/make_oracle_archive.sh` packs only `FINAL_VERSION`, `notes.md` and the
selected `submission.py`.

## final-holdout-v3 (2026-11-08): pilot on H100 SXM 80 GB

`final-holdout-v3`: 774 scored cases (video 199, image 44, llm 392, real 139) and 8 edge
cases; `scripts/generate_final_manifest.py`, frozen instance sha256 `fbb864de…7fc7`.
Run host-side through the trusted runtime (`scripts/pilot_suite.py`, then
`scripts/pilot_verify.py` re-derives the score with the Judge verifier's own result
validation), one candidate per idle GPU, torch 2.8.0+cu129 / Triton 3.4.0. Not yet run through
the Docker Judge (the host's Docker daemon was unresponsive); the Judge adds container start,
the edge cases and the private BF16 SDPA line on the 14 recipe cases, about 1-2 minutes.

| candidate | wall (774 cases) | score | video | image | llm | real | gates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| kernel_v001 (seed) | 862 s | 312.25 | 348.56 | 254.68 | 304.04 | 352.21 | all pass |
| kernel_v002 (admissible oracle) | 813 s | 351.27 | 394.99 | 271.70 | 352.48 | 402.50 | all pass |

- Device-memory peak 44.2 GiB (262K-token cases); hence the 64 GiB profile.
- Trial noise on an idle, cool GPU: median CV 0.45-0.49 %, p90 1.3-2.2 %.
- Every gate ratio sits near the Sage1 reference's own error (RelL1 about 0.35 of its limit,
  row p99.9 about 0.4-0.5 of its limit), on every real source including the 262K prefill.
- llm by context band (seed / v002, TFLOPS): 2K 96 / 90, 8K 263 / 314, 32K 348 / 423,
  128K 384 / 443, 256K 383 / 432. Neither kernel slows down at long context; short prompts and
  narrow per-rank head slices (TP8 8/1, 12/1) are the weak spot, and image cases (1.4K-17K
  tokens) are the weakest family.
- Warm-up (compilation included) at most 2.8 s per case for both, far below the 60 s cap.
- The forced Sage2 CUDA oracle (kernel_v004) was not re-run on v3: it needs the
  `paper_reference/forced_judge` image.
- One card of the host (GPU 7, 67 °C idle against 28-31 °C) throttles under load and gave
  20-60 % lower and noisy numbers; measure on cool cards only.
