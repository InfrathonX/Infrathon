# runtime_v002: split_by_steps (paper block balancing, static)

The pristine seed (`runtime_v001`) with one change: the default DiT block split. Rank 0
holds blocks `[0, split)` with `split = {1: 24, 2: 22, 3: 18, 4: 16}[steps]` (15 for any
other step count) instead of the fixed 15/15 split.

It is a static form of the StreamDiffusionV2 paper's dynamic block scheduler (§4.3;
upstream `--schedule_block`, `streamv2v/inference_pipe.py` `_maybe_schedule_blocks` and
`streamv2v/communication/utils.py` `compute_balanced_split`, commit `2f8ee1c5`,
Apache-2.0). That scheduler moves DiT blocks toward rank 0 because rank 1 also runs the
VAE decode. These are the splits it converges to on an H100 pair, from the v1 split sweep.
