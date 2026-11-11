Version 2 is the Oracle (FINAL_VERSION=2). It ports kvpress `ExpectedAttentionPress`
(NVIDIA/kvpress @ 7f359b84, Apache-2.0). It takes the query mean and covariance of the
supplied pre-RoPE queries and rotates them by the RoPE matrix averaged over the next 512
positions. Each key's expected attention is averaged over the query heads of its KV head
and scaled by (score + 0.01) * ||v||. Four sinks are always kept, and budgets are uniform.
Its bytes are identical to oracle/references/expected_attention.

Version 1 is the task seed, byte-identical to the Work image starter. It keeps the first
four cached positions and the most recent ones, and computes no score.

| version | policy | final holdout (5,000 items) | gates | run |
|---|---|---:|---|---|
| 1 | seed: 4 sinks + recency | 40.790466 | all 11 pass | q-kvpress-v2-20261005-063132-822146 (8x H100 NVL) |
| 2 | kvpress ExpectedAttentionPress port | 77.737238 | all 11 pass | kvpress-v2-qual-ea-n4-20261004-193111 (4x H100 NVL) |

Version 2 is admissible. It passes source admission and the integrity review, so the
normal Judge scores it unchanged.
