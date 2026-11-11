# Baseline notes

Version 1 is a self-contained Triton positive control for the v2 entry point.
It is the **qualification control**, not a performance target: it proves the
task is solvable over the whole v2 support domain and that the Judge accepts a
correct implementation; the scoring contract applies no baseline ratio.  For
headroom, see `oracle/`, which measures upstream DeepGEMM on the scored grid.

One kernel implements the declared per-channel FP8 block-scale math with FP32
accumulation and the `C` addend.  It masks the M and N tails, so any
`1 <= M, N <= 8192` works; a group with `K_g = 0` stores `C` unchanged; a
problem with `K_total = 0` is a plain copy of `C`.  Group starts come from a
device-side `cumsum` of `ks`.  FP8 tiles are converted to BF16 (exactly) for
`tl.dot` with FP32 accumulation, and one scale row is applied per 128-row K
block.  It calls no existing GEMM API and is never copied into either task
image.
