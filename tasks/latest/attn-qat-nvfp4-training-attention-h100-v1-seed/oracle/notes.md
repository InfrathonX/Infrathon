# Oracle

Host-only positive control and performance reference. It is never copied into
the Work or Judge image and is not shown to participants.

`FINAL_VERSION` selects `kernel_v002`, the strongest admissible upstream (paper)
configuration measured on H100. `kernel_v001` stays in place, byte-identical
to the Work seed, as the seed control.

## v3 contract (causal + grouped-query attention, D = 256)

The `final-holdout-v3` suite adds an LLM domain, so both kernels now take the six- and
eleven-argument ABI (`causal` after `softmax_scale`), read `k`/`v` with `H_kv` heads, and accept
`D = 256`:

- fake quantization of `q`, `k`, `v` goes through one row kernel per tensor (`k`, `v` at
  `H_kv` heads);
- the statistic pre-pass and the main forward map query head `h` to key/value head
  `h // (H / H_kv)` and, under `causal`, stop at the diagonal block and mask `j > i`;
- the backward is a per-query-head dQ kernel and a per-key/value-head dK/dV kernel that loops
  over its group's query heads (the old fused self-attention kernel is gone); masked and
  out-of-range entries of `P` are zeroed before the fake quantization, so the tail 16-group of
  an unaligned `S_kv` now matches the golden exactly;
- `D = 256` runs the 32x32 tiles at 8 warps.

kernel_v002 is kernel_v001 with the forward main pass at 64x32 / 4 warps / 3 stages for
`D` in {64, 128}; its backward is kernel_v001's. On the full v3 suite (H100 SXM, one run each):
kernel_v001 74.2 TFLOPS in 1233 s, kernel_v002 93.7 TFLOPS in 1194 s.

## kernel_v001: seed control (upstream SM90 route)

kernel_v001 is the admitted FastVideo-derived seed: the upstream SM90 Attn-QAT
training attention route, brought to this task's two-call ABI, with pinned
configurations:

- fake quantization at 32-row tiles;
- a 32x32 statistic pre-pass at 8 warps / 4 stages;
- the main forward at BLOCK_M = BLOCK_N = 32 with 4 warps and 2 stages, over
  device-side tensor descriptors;
- a 128-row backward preprocess and a combined 32/32/32/32 backward at 4 warps
  and 3 stages;
- the two cross-attention backward kernels.

It deliberately leaves tile shape, stage counts, warp specialization, TMA on
the backward, fusing the quantization prologue and collapsing the backward's
duplicated QK/dP recomputation as headroom. It is byte-identical to
`images/work/workdir/attn_qat_lab/candidates/kernel_v001/submission.py`,
enforced by `scripts/validate_layout.py` and `tests/test_contract.py`.

## kernel_v002: best upstream configuration on H100 (selected)

- Upstream: hao-ai-lab/FastVideo, https://github.com/hao-ai-lab/FastVideo,
  Apache-2.0. Attn-QAT kernels in
  `fastvideo-kernel/python/fastvideo_kernel/triton_kernels/{attn_qat_train,quant_utils,nvfp4_utils}.py`:
  - the forward and the SM90 combined backward come from commit
    `7f8316423355a2f07db533a5a339ac16e0da31ff` (the seed's pin);
  - the split dQ / dK-dV backward bodies come from main
    `7203b4976eefcd70a52728ca1281797ce96734eb` (#1899, SPLIT_FULL_TILES). They
    are present in the file but switched off (see below).
- What it runs:
  - Forward: 64x32 tiles, 4 warps, 3 stages, warp specialization off, for both
    head dims. Upstream's Hopper autotune space for this kernel is commented
    out upstream, and runtime autotuning is forbidden by the task policy, so
    the config was chosen statically from that space. It is the best of the
    configs measured on H100.
  - Statistic: upstream's exact-M 32x32 pre-pass, which the task's final-row-max
    contract requires.
  - Backward: upstream's own SM90 combined 32x32 / 4-warp / 3-stage backward
    everywhere. The SM100-style split backward from #1899 measured slower on
    H100 (C1 65 vs 95, C4 56 vs 68.5 TFLOPS), so its route is disabled with
    `if False and ...`.
  - The kernel bodies are unchanged upstream code. Only the launch configs and
    the routing differ from kernel_v001.
- ABI glue, the same as the seed:
  - production flags are baked in (IS_QAT, fake_quant_P, use_high_prec_o; no
    smooth_k / smooth_q / two-level / global scales);
  - the dequant destination dtype is pinned to bf16, which is exact;
  - no autograd;
  - saved = (Qf, Kf, Vf, O', M).
- Caveat: the module docstring was inherited from the SM100-route variant this
  file was derived from (paper archive "v002"). The docstring still describes
  the 128x128 forward and the split backward, but the code runs the route
  described above. The file was kept byte-identical to the measured archive
  rather than editing the docstring.
- Source admission passes. Evaluate it as a trusted control
  (`SFORGE_LLM_JUDGE_MODE=off`), because it intentionally copies upstream code.
- sha256(submission.py) = `cb95ab6775fdd5b03680d282c64553ba978e2f15fdb390a62572cb4eef4a6283`.
  This equals `candidates/kernel_v005/submission.py` of the paper archive
  `paper_submission_v005.tar.gz` (sha256 `754c2907...d28b`), which was measured
  earlier.

## Judge scores (H100 NVL 94 GB, 400 W, retired four-case final-holdout-v1)

These predate `final-holdout-v2` and are not comparable with it; re-measure on the new suite.

All runs used the real Judge (`scripts/eval_archive.sh`, Docker, 8 CPUs and
32 GB). Every run passed all gates: correctness, the 11 edge cases, fresh
input/output, saved_cap, hardware and worker. Per-case values are
fwd / bwd / case TFLOPS.

| archive | run id | GPU / Judge cpuset | score | wan13b | ltx2 | kand5 | C1 | C2 | C3 | C4 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| kernel_v002 (new oracle) | oracleup-attnqat-new-r1-gpu0-20261029-194523 | 0 / 0-3,32-35 | **88.92** | 103.30 | 96.61 | 70.46 | 109.1 / 95.5 / 103.3 | 107.6 / 97.0 / 103.3 | 99.3 / 92.8 / 96.6 | 71.9 / 68.3 / 70.5 |
| kernel_v002 (new oracle) | oracleup-attnqat-new-r2-gpu0-20261029-194523 | 0 / 0-3,32-35 | **89.10** | 103.45 | 97.06 | 70.45 | 109.1 / 95.3 / 103.3 | 108.1 / 97.0 / 103.6 | 99.6 / 93.1 / 97.1 | 71.9 / 68.3 / 70.5 |
| kernel_v001 (old oracle = seed) | oracleup-attnqat-old-r1-gpu0-20261029-194523 | 0 / 0-3,32-35 | 67.80 | 76.96 | 69.17 | 58.56 | 73.2 / 95.2 / 80.3 | 64.1 / 97.0 / 73.7 | 59.6 / 93.0 / 69.2 | 53.8 / 68.3 / 58.6 |
| kernel_v002, same `submission.py` bytes (paper v005) | pvsa-attnqat-paper5-gpu1-20261029-122459 / paper5b | 1 / unpinned | 88.54 / 88.61 | 102.67 | 95.58 | 70.72 | | | | |
| kernel_v001 (old oracle = seed) | pvsa-attnqat-oracle-gpu1-20261029-120523 / oracle2 | 1 / unpinned | 67.38 / 67.43 | 76.31 | 68.28 | 58.70 | 72.7 / 94.7 / 79.9 | 63.4 / 96.0 / 72.9 | 58.8 / 91.8 / 68.3 | 53.9 / 68.5 / 58.7 |

On the same card (GPU 0, back to back), kernel_v002 is 1.31x kernel_v001. Nearly all of the gain is in the forward:
the D128 forward goes from about 59-73 to 98-109 TFLOPS, and the D64 forward
from 54 to 72. The backward is upstream's own and stays at about 68-97. Error
levels match the seed: o 0.0017-0.0019, dq 0.006-0.011, dk 0.003-0.004, dv
0.007-0.017.

Other upstream configurations measured with the same protocol, all weaker or
invalid:

- upstream main's SM100 "fast" route moved to H100 (128x128 / 8 warps
  forward, split backward): 48.6;
- 64x32 forward with the split backward: 83.6;
- 64x64 forward with the split backward: 70.3;
- 64x32 with 2 stages: 87.9;
- D64 with warp specialization on: invalid, non-finite dq.
