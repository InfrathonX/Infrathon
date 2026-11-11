# Forced Judge: SageAttention2 sm90 CUDA through the real Judge pipeline

This directory scores the strongest upstream (paper) path, which is not
admissible, with the task's **normal Judge pipeline**. It runs in a local-only
derived Judge image whose only change is that source admission is bypassed.
The result is a real `report.json` and private artifact, not a harness number.

Oracles are not bound by the agent-submission rules: the source-admission
import allowlist, the compiled-extension ban and the LLM integrity review.
This is how `oracle/FINAL_VERSION` (= 4) gets its score.

## What is changed and what is not

| step | in this run |
| --- | --- |
| LLM integrity review (host-side `sforge` step) | **off** (`SFORGE_LLM_JUDGE_MODE=off`), as for every trusted control |
| source admission (`source_admission.validate_submission`) | **bypassed**. `admission_override.py` wraps it: the original errors are recorded and the result is forced to `ok` whenever `FINAL_VERSION` selects an existing `submission.py`. `protected_hashes.txt` is updated so the verifier's self-check covers the patched file. |
| everything else: `run_judge.py`, `verify_quant_attention.py`, `candidate_worker.py`, `quant_attention_runtime.py`, the frozen final-holdout-v2, the 8 edge cases, the 62 real layers, 1 warm-up + 7 fresh-input trials, timing, aggregation, privilege drop | unchanged (the bytes of the private `:local` Judge image) |
| docker run (`sforge eval --backend docker`, 8 CPUs / 32 GB, timeout 3600 s, per-GPU Judge lock, idle check, `NVIDIA_VISIBLE_DEVICES` = the GPU's UUID) | identical to `scripts/eval_archive.sh`. Only the image differs: a scratch copy of the task JSON selects tag `forced-oracle` instead of `local`. |

The task has no compute-sanitizer stage, so none was skipped.

The image is
`sageattention_quantized_attention.judge.sageattention_quantized_attention_bf16_h100_seed_v2:forced-oracle`.
It is built `FROM` the paper-reference image (`../Dockerfile`: the private
Judge image plus upstream compiled for sm_90a), and that in turn sits on the
`:local` private Judge image. `run_forced_oracle.sh` checks this layer
lineage before every run. It never retags `:local` and never pushes.

## Provenance

- Upstream: https://github.com/thu-ml/SageAttention, commit
  `d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5` (Apache-2.0). It is supplied as
  the named build context `sage_src` and is not vendored.
- Build: `TORCH_CUDA_ARCH_LIST=9.0 pip install --no-build-isolation --no-deps .`
  (`../Dockerfile`). This builds `_qattn_sm90` (wgmma), `_qattn_sm89` and
  `_fused`.
- Candidates, byte-identical to what was judged:
  - `../../candidates/kernel_v004/submission.py`: `qk_quant_gran="per_warp"`.
    sha256 `92dbdb5249fd6477bfb17e2bff72ba06cbe8b25c70fb3ace568fa092a542efda`.
    **Selected** (`FINAL_VERSION`).
  - `../../candidates/kernel_v003/submission.py`: `qk_quant_gran="per_thread"`,
    upstream's default. sha256
    `381eeb20c74c5e217a505459e2997ec2c61766b3511a55b35e3d8aad014fa79c`.

  Both are `../paper_sm90_candidate.py` with the quantizer fixed in the file,
  because the Judge passes no environment to the candidate. per_warp is
  upstream's own CUDA `per_warp_int8` quantizer, an option upstream exposes
  (`sageattn_qk_int8_pv_fp8_cuda_sm90(..., qk_quant_gran="per_warp")`).
  per_thread is the literal default.
- Original admission errors (the bypassed gate), from the unmodified Judge
  (16 for kernel_v004):
  - forbidden from-imports: `sageattention`, `sageattention.core`,
    `sageattention.quant` (twice) and `sageattention.triton.quant_per_thread`;
  - forbidden host calls: `torch.cuda.set_device`, `k.mean`, `torch.cat`,
    `torch.zeros`, `v.size`, `per_warp_int8_cuda`, `per_thread_int8_triton`,
    `per_channel_fp8` and
    `sm90_compile.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf`;
  - the K mean is a forbidden host tensor data access;
  - "entrypoint call graph must launch a declared @triton.jit kernel".

  `run_forced_oracle.sh` stores the full list as
  `original_source_admission.json` next to each run's `report.json`.

## Run

```bash
# Prerequisite: the private Judge image (scripts/build_images.sh).
git clone https://github.com/thu-ml/SageAttention /path/to/SageAttention
git -C /path/to/SageAttention checkout d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5
SAGE_SOURCE=/path/to/SageAttention \
  oracle/paper_reference/forced_judge/run_forced_oracle.sh --gpu 2 --cpuset 8-11,40-43 --version 4
# Builds the paper-reference image if PAPER_IMAGE (default
# sageattention-paper-sm90-harness:d1a57a5) is missing, then the forced image,
# then packs oracle/{FINAL_VERSION=V, notes.md, candidates/kernel_vVVV} like
# scripts/make_oracle_archive.sh and runs sforge eval.
# Options: --skip-build, --rebuild-paper, --run-id ID; env LOGS_DIR, SFORGE_BIN.
```

The report lands in
`<LOGS_DIR>/runs/<run id>/sageattention_quantized_attention_bf16_h100_seed_v2/submissions/manual-1/report.json`,
next to `private_judge_artifacts.tar.gz` (the per-cell results, the 62 real
layers and the edge cases) and `original_source_admission.json`.

The recorded runs used `PAPER_IMAGE=sageattention-paper-sm90-harness:d1a57a5-oracleup`
(`sha256:87c5ea06...`, built from `../Dockerfile` with the `sage_src`
context). They also used the dev-tree SForge
(`SFORGE_ROOT=<mlsys_forge checkout>`), on whose Judge build the `:local`
images were recorded.

## Measured (H100 NVL 94 GB, GPU 2, Judge cpuset 8-11,40-43, 2026-10-29 UTC)

Every run is `valid: true`, 10/10 report gates. The r3final run judged the archive that `scripts/make_oracle_archive.sh` produced from the committed `oracle/` at that moment (FINAL_VERSION=4; sha256 `48b6df914a3f4e2b7908051abf16f7641e4becd33f09601d8ab9ebb72e8585f2`, byte-identical to the run's `submission.tar.gz`). Since then `notes.md` has changed only to add this run's row, which moves the archive sha but not the candidate identity (`notes.md` is excluded from it, and `submission.py` is unchanged). It ran with the forced image rebuilt by the committed `run_forced_oracle.sh` (image id `sha256:ef32fd47...`). The earlier four runs used the same `submission.py` bytes; only `notes.md` was older. `source_ok` passed only
because admission was bypassed; every other gate passed on its own: profile,
suite, hardware, correctness, 8/8 edge, 62/62 real layers, fresh input,
fresh output. 

| candidate | run id | score | dit | causal | D1 / D2 / D3 / L1 / L2 | cuDNN BF16 ref |
| --- | --- | --- | --- | --- | --- | --- |
| kernel_v004 per_warp | `forced-sageattention-v004-r1-gpu2-20261029-211128` | **581.73** | 568.52 | 595.24 | 648.9 / 646.7 / 437.9 / 646.0 / 548.5 | 444.4 |
| kernel_v004 per_warp | `forced-sageattention-v004-r2-gpu2-20261029-211539` | **577.88** | 566.31 | 589.69 | 642.4 / 641.2 / 440.9 / 643.1 / 540.7 | 435.4 |
| kernel_v004 per_warp, final archive | `forced-sageattention-v004-r3final-gpu2-20261029-215207` | **581.80** | 567.10 | 596.89 | 646.0 / 644.8 / 437.9 / 647.0 / 550.7 | 449.2 |
| kernel_v003 per_thread | `forced-sageattention-v003-r1-gpu2-20261029-211336` | 549.72 | 537.68 | 562.03 | 622.8 / 625.1 / 399.3 / 619.0 / 510.3 | 444.8 |
| kernel_v003 per_thread | `forced-sageattention-v003-r2-gpu2-20261029-211746` | 545.53 | 537.36 | 553.83 | 623.4 / 621.5 / 400.4 / 610.7 / 502.3 | 439.3 |

Each run takes about 105 s of Judge time. The numbers agree with the round-1
`run_paper_reference.sh` harness on the same card (per_warp 581.30, per_thread
551.84). For comparison on GPU 2: admissible `kernel_v002` scored 350.7 /
348.9 and the seed `kernel_v001` scored 297.7.
