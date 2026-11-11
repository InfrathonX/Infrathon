# Paper reference: SageAttention2 sm90 CUDA (not admissible)

This directory scores the strongest upstream path of the paper on H100:
SageAttention2/2++ on the sm90 wgmma CUDA path.

The kernel is a compiled CUDA extension (`import sageattention`,
`sm90_compile`). `images/judge/source_admission.py` admits only `math`,
`typing`, `torch`, `triton` and `triton.language`, by design, so the kernel
can never pass the normal Judge. Oracles are not bound by that rule, so it is
the oracle's `FINAL_VERSION` (`../candidates/kernel_v004`, per_warp) and is
scored two ways:

- **`forced_judge/`, the authoritative oracle score.** The task's full Judge
  pipeline (`sforge eval` -> `run_judge.py` -> verifier -> worker -> trusted
  runtime) runs in a local-only derived Judge image whose source admission is
  bypassed. It scored **581.73 / 577.88** (per_warp) and 549.72 / 545.53
  (per_thread), and every other gate passed. See `forced_judge/README.md`.
- **This directory's host harness (round 1).** `run_paper_sm90.py` calls the
  trusted runtime's `run_suite` directly. It scored 581.30 (per_warp) and
  551.84 (per_thread), consistent with the forced runs.

The best **admissible** oracle stays `../candidates/kernel_v002` (upstream
SageAttention-1 Triton), selected by `../ADMISSIBLE_VERSION`. The smoke test and
anything else that uses the normal Judge use it. See `../notes.md`.

## Isolation

- Nothing here is copied into the Work or Judge image.
- Nothing here feeds `image_source_digest()`, which hashes only `images/`,
  `runtime/`, `judge/run_judge.py` and the fixture recipe.
- `scripts/make_oracle_archive.sh` packs only `FINAL_VERSION`, `notes.md` and
  the selected `submission.py`, so nothing here reaches a Judge archive.
- `forced_judge/` builds a separate local-only image tagged `:forced-oracle`.
  It never retags `:local` and never pushes.
- `scripts/smoke_test.sh --oracle` mounts only `candidates/`, `ADMISSIBLE_VERSION`
  (as `FINAL_VERSION`) and `notes.md`.
- Upstream is not vendored. `run_paper_reference.sh` passes a checkout you
  supply as a named Docker build context.

## Provenance

- Repository: https://github.com/thu-ml/SageAttention, Apache-2.0.
- Pinned commit: `d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5` (2026-01-17,
  "Update README"). This is the same commit kernel_v002 is ported from.
- Kernel: `sageattention/core.py::sageattn_qk_int8_pv_fp8_cuda_sm90`, which is
  what upstream's `sageattn()` dispatches to on sm90. It uses INT8 QK, FP8 PV
  and `pv_accum_dtype="fp32+fp32"`, the only PV accumulation upstream allows on
  sm90, with smooth-K and no LSE.
- QK quantizer, the `PAPER_SM90_QK_GRAN` setting:
  - `per_thread` is upstream's default: the Triton quantizer.
  - `per_warp` is upstream's own CUDA quantizer (`sageattention.quant.per_warp_int8`).
    It is faster on H100, so it is reported as the paper's best configuration.
- ABI adaptation, in `paper_sm90_candidate.py`: the CUDA kernel writes straight
  into the caller's `out` instead of a fresh `torch.empty`. Everything upstream
  does on the host stays inside the timed call, as it does for any submission:
  the K mean, quantization, V zero-padding to a multiple of 128, and the V
  transpose/pad/permute plus FP8 quant.

## Files

| file | role |
| --- | --- |
| `Dockerfile` | private Judge image + upstream compiled with `TORCH_CUDA_ARCH_LIST=9.0` (sm_90a); checks that `_qattn_sm90`, `_qattn_sm89` and `_fused` were built |
| `check_ext.py` | build-time check that the three extensions exist (it does not import them, because libcuda is only present at run time) |
| `paper_sm90_candidate.py` | upstream sm90 host flow behind the task ABI `quantized_attention_bf16(q, k, v, is_causal, softmax_scale, out)` |
| `run_paper_sm90.py` | calls `quant_attention_runtime.run_suite` exactly as `candidate_worker.py` does: 8 edge cases, 62 real layers, the frozen final-holdout-v2 cases, 1 warm-up + 7 fresh-input trials with decoy, max(host, CUDA event), median, family geomeans and a same-card cuDNN BF16 reference. Only source admission, the LLM review and the privilege drop are skipped. |
| `kernel_only_bench.py` | optional and **not a score**: the attention kernel alone on pre-quantized inputs, with CUDA events and warm L2 |
| `run_paper_reference.sh` | builds the image and runs the scored harness on one idle GPU, holding the Judge's per-GPU lock |

## Build and run

```bash
# Prerequisite: the private Judge image, built by scripts/build_images.sh.
git clone https://github.com/thu-ml/SageAttention /path/to/SageAttention
git -C /path/to/SageAttention checkout d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5
SAGE_SOURCE=/path/to/SageAttention \
  oracle/paper_reference/run_paper_reference.sh --gpu 2 --cpuset 8-11,40-43
# Results go to artifacts/paper_reference/paper_sm90_{per_warp,per_thread}_gpu<N>_<stamp>.json.
# Score fields: .score_tflops, .dit_tflops, .causal_tflops, .cases, .worst_gate_ratios.
# Options: --skip-build, --gran per_warp, --kernel-only, OUTPUT_DIR=..., HARNESS_IMAGE=...
# MAX_JOBS / EXT_PARALLEL / NVCC_THREADS lower the build parallelism; the compiled code is the same.
```

The build is CPU only. It took about 3 minutes at the defaults
(`MAX_JOBS=16`) and about 10 minutes at `MAX_JOBS=4 EXT_PARALLEL=1
NVCC_THREADS=2`. Each scored run takes about 1.5 minutes.

## Measured (H100 NVL 94 GB, 400 W, final-holdout-v2, Judge runtime protocol)

All runs passed every gate: 8 edge cases and 62 real layers. The worst gate
ratio was always on `real-cogvideox2b-l01`.

| date (UTC) | GPU / cpuset | image | quantizer | score | dit | causal | D1 / D2 / D3 / L1 / L2 | worst gate | cuDNN BF16 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-10-29 20:05 | 2 / 8-11,40-43 | this Dockerfile (`sha256:87c5ea06...`) | per_warp | **581.30** | 567.19 | 595.77 | 647.5 / 643.6 / 437.9 / 647.3 / 548.3 | 0.932 | 447.6 |
| 2026-10-29 20:05 | 2 / 8-11,40-43 | this Dockerfile | per_thread (upstream default) | 551.84 | 538.55 | 565.47 | 629.1 / 623.1 / 398.5 / 620.7 / 515.1 | 0.893 | 445.7 |
| 2026-10-29 13:30 | 0 / 0-7 | equivalent earlier build (`sha256:e7821536...`) | per_warp, r1 / r2 | 557.83 / 560.28 | 543.77 / 545.70 | 572.25 / 575.26 | 609.0 / 617.9 / 427.3 / 621.0 / 527.3 (r1) | 0.932 | 430.3 / 433.3 |
| 2026-10-29 13:28 | 0 / 0-7 | equivalent earlier build | per_thread, r1 / r2 | 534.69 / 532.23 | 524.55 / 519.65 | 545.02 / 545.12 | 607.7 / 605.4 / 392.3 / 602.2 / 493.3 (r1) | 0.893 | 435.3 / 430.2 |

The earlier build used the same upstream tree (a `git archive` of the pinned
commit) and the same compile flags. It copied the tree from its build context
instead of the named `sage_src` context. GPU 2 measures about 4% faster than
GPU 0 for every implementation, including cuDNN.

For comparison, on the same card (GPU 2) the scored admissible oracle
`kernel_v002` (Sage1 Triton) scored 348.9-350.7 and the seed `kernel_v001`
scored 297.7. The upstream SageAttention2++ mma.sync kernel (`_qattn_sm89`
built for sm_90a, fp32+fp16 accumulation) scored only 293.1 on GPU 0, which
confirms that sm90 wgmma is the paper's best H100 path.
