# Dynamic-Grouped FP8 Lab

Every `sforge-submit` runs the evaluation on hidden test cases. `sforge-submit --details` returns the gate states, the score and the four family scores. Hidden cases and per-case results stay on the Judge. Your final result is the best valid score you reach within the time budget, so keep `FINAL_VERSION` on your strongest candidate.

The editable surface is:

```text
candidates/kernel_vNNN/submission.py
FINAL_VERSION
notes.md
```

The initial `kernel_v001/submission.py` is zero bytes.  There is no starter
kernel elsewhere in the Work image.  A selected module must implement both
`grouped_fp8_gemm_nt_masked` and `grouped_fp8_gemm_nt_psum` exactly as described
in `AGENT_TASK.md`; it may contain Triton kernels and Python dispatch helpers.

Start with a simple correct tiled kernel for each layout.  Both layouts share
the same 128-wide K-block FP8/scaling math.  The masked ABI gets a padded
`[G,M_max,K]` A tensor and per-group row counts.  The prefix-sum ABI gets a
`[G*M_max,K]` row buffer whose groups begin after 128-row-aligned previous
ends.  A shared Triton kernel with a small layout-specific row mapping is a
reasonable baseline; it is not required.

Correctness must hold over the whole support domain, not only the public cases:
`G` from 1 to 64, `N` and `K` from 128 to 8192 in steps of 128 (odd numbers of
128-blocks included), `M_max` up to the per-layout limit in `AGENT_TASK.md`, any
row count from 0 to `M_max` per group, and hints that say nothing reliable about
the actual rows.  The usual ways to fail are a grid
sized with floor division (`N // BLOCK_N`, `M_max // BLOCK_M`) when the block
does not divide the size, a store without a row or column mask, and a grid
bounded by the hint.  Tensors can sit between guard bands, invalid A rows hold
arbitrary values (often NaN), and inputs must stay unchanged.

Development commands:

```bash
./tools/run_one.sh --version 1
./tools/run_public_tests.sh --quick --candidate-version 1
./tools/run_public_tests.sh --candidate-version 1
./tools/run_public_tests.sh --edge --candidate-version 1
./tools/run_public_tests.sh --sanitize --candidate-version 1   # memcheck subset; exits 2 where compute-sanitizer cannot attach
```

Allocation, quantization and reference calculation are outside measured time.
Untimed calls run first with the same shapes and hints but different row counts;
compilation during a timed call counts as time.  Inputs, including `b`, can repeat
across calls, and row counts change from call to call.

Keep a known-correct version before tuning.  Run the full public suite, the
public edge suite and the sanitizer suite before selecting `FINAL_VERSION`.

Hidden cases use other shapes and larger batches, and there are many more of
them, so make optimizations general rather than fitting particular shapes.

Before each `sforge-submit --details`, wait for all public-test and compiler
workers to exit.
