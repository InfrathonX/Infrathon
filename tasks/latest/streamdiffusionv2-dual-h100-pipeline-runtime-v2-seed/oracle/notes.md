# Oracle notes

FINAL_VERSION=2 selects `runtime_v002` (split_by_steps). It is the seed with the
StreamDiffusionV2 paper's block balancing (§4.3 dynamic block scheduler; upstream
`--schedule_block` / `compute_balanced_split`, chenfengxu714/StreamDiffusionV2 @ 2f8ee1c5,
Apache-2.0) as a static per-step split. Rank 0 holds DiT blocks [0, split) with
split = {1: 24, 2: 22, 3: 18, 4: 16}[steps]. Everything else is the seed unchanged.

`runtime_v001` is the seed: a static 15/15 two-rank lag ring with per-(chunk,step) noise,
matched NCCL drain, clean rank-1 join, decode overlap, and exact UpstreamKVPolicy
behavior. It is byte-identical to the Work seed, and the Judge re-measures it in every run.

Final suite (12 hidden cases), H100 NVL pair, latency-weighted FPS:

| candidate | score | FPS geomean | seed FPS geomean (same run) | gates | parity | run |
|---|---:|---:|---:|---|---|---|
| runtime_v001 seed (previous oracle) | 28.89 | 28.87 | 28.96 | all 12 pass | 12/12 | v2 qualification, 2026-10-06 |
| runtime_v002 split_by_steps | 32.80 | 31.72 | 28.81 | all 12 pass | 12/12 | v2 qualification, 2026-10-06 |
| runtime_v002 split_by_steps (this archive) | 33.119 | 31.91 | 29.12 | all 12 pass | 12/12 | forced-sdv2-oracle-v002-20261029-231223 |

runtime_v002 is admissible: it passes source admission and the normal Judge, so no forced
run is needed. The scores are raw FPS and compare only within one hardware type.
