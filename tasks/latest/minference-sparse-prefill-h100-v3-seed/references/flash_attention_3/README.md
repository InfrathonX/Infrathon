# Pinned author-only FlashAttention-3 baseline

This baseline uses the Hopper implementation from official FlashAttention
commit `060c9188beec3a8b62b33a3bfa6d5d2d44975fab` (tag `v2.8.3`), with CUTLASS
commit `dc4817921edda44a549197ff3a9dcf5df0636e7b`. The wheel distribution calls
itself `flash_attn_3 3.0.0b1`; its exact source commit and binary hashes, rather
than that version string alone, identify this baseline.

[The pinned public function](https://github.com/Dao-AILab/flash-attention/blob/060c9188beec3a8b62b33a3bfa6d5d2d44975fab/hopper/flash_attn_interface.py#L502-L578)
accepts BNHD Q/K/V and native grouped-query attention. The benchmark calls the
top-level `flash_attn_interface.flash_attn_func` with BF16, head dimension 128,
32 query heads, 8 KV heads, `causal=True`, and `num_splits=1`. BHND-to-BNHD views
preserve the real sequence length and KV head count. Its output copy into the
five-argument task ABI and the immediate snapshot are inside the timing region.
This pin does not accept a `return_attn_probs` keyword.

[The pinned build script](https://github.com/Dao-AILab/flash-attention/blob/060c9188beec3a8b62b33a3bfa6d5d2d44975fab/hopper/setup.py#L377-L443)
uses a vendored CUDA 12.6.85 nvcc and CUDA 12.8.93 ptxas when building from this
CUDA 12.9 base. Their exact binary hashes and the system compiler are recorded
in `BUILD.json`. The specialized build keeps causal BF16 forward attention,
head dimension 128, native GQA and Hopper code; unrelated backward, local,
paged-KV, FP8, FP16 and other head-dimension variants are disabled.

`build_source.sh OUTPUT_DIRECTORY` rebuilds from the pinned official source in
a CPU-only Docker container limited to 8 CPUs and 64 GiB, using four compiler
jobs and two nvcc threads. It forces compilation instead of downloading a
guessed prebuilt wheel. The Ubuntu git package is pinned to the version used
in the recorded build. Upstream CUDA compiler downloads remain network
dependencies; their installed binaries must match `BUILD.json` before a new
build is admitted as this baseline. Rebuilding a wheel need not produce an
identical compressed wheel hash, so retained wheel, interface and extension
hashes identify the actual measured artifact.

Install the retained wheel only into the author baseline image that has
Torch `2.8.0+cu129`, Triton `3.4.0`, and Transformers `4.57.1`. Candidate Work
and Judge images do not contain this extension. Record all three exact image
IDs when measuring; the baseline image has an additional extension and is
therefore a distinct image. `BUILD.json` records CPU compilation and import
checks, not GPU numerical or performance qualification.

Run `scripts/benchmark_dense_v3.py --help` for arguments. The script accepts
only a complete public v3 manifest: 48 groups, each with three real trials
paired with same-shape layer-2 warm-ups. It authenticates each fixture, uses
the task's trusted `measure_call` and full-output/input-integrity checks,
compares 512 uniform and 64 tail rows against the FP32 reference, and reports
per-group medians and equal-weight geometric means. Its public probe seed is
reproducible and independent of private evaluation nonces. Baseline results
are diagnostic; downstream terminal QA and its frozen policy determine task
validity separately.
