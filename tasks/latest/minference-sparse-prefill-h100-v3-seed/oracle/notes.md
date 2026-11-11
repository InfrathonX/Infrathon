v3 seed: upstream last-64 vertical/slash, real variable N, shape-only bounded scratch. Quality and numeric policies require fresh pilot and independent holdout; no v2 qualification is inherited.

## Candidates

| candidate | what it is | H100 NVL Judge (2026-10-29) |
|---|---|---|
| `kernel_v001` | the v3 seed above, byte-identical to the Work starter | about 197 (196.52 / 197.05) |
| `kernel_v002` (selected) | MInference 1.0 (microsoft/MInference @ a4eb395f) upstream default vertical_and_slash config for Llama-3.1-8B (512/2048, last-64 estimator, upstream Triton sparse kernel, the CUDA index converter ported to Triton) plus the seed's probe guard at band NRMSE 0.30 | valid, 364.04 (`oracleup-minference-guarded`) and 363.54 (`oracleup-minference-paperoracle-v002`) |

The strict upstream port without the guard is invalid: on `final-niah_single_2-32768-r0-L1` the
tail worst-head NRMSE is 0.934 against the 0.50 limit. The guard comes from the seed, so
`kernel_v002` is upstream plus one task-side repair, not pure upstream. Full provenance and the
other measured variants are in `../paper_reference/README.md`.
