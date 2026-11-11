Version 1 ports kvpress `ExpectedAttentionPress` (NVIDIA/kvpress @ 7f359b84,
Apache-2.0): query mean and covariance from the supplied pre-RoPE queries,
rotated by the RoPE matrix averaged over the next 512 positions; expected
attention per key, averaged over the query heads of a KV head and scaled by
(score + 0.01) * ||v||; four sinks always kept; uniform budgets.
