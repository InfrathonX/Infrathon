# K-grouped FP8 weight-gradient GEMM lab

Every `sforge-submit` runs the evaluation on hidden test cases. `sforge-submit --details` returns the gate states, the score and the four family scores. Hidden cases and per-case results stay on the Judge. Your final result is the best valid score you reach within the time budget, so keep `FINAL_VERSION` on your strongest candidate.

The editable surface is:

```text
candidates/kernel_vNNN/submission.py
FINAL_VERSION
notes.md
```

The initial `kernel_v001/submission.py` is zero bytes.  There is no starter
kernel elsewhere in the Work image.  A selected module must implement
`k_grouped_fp8_gemm_tn_contiguous` exactly as described in `AGENT_TASK.md`;
it may contain Triton kernels and Python dispatch helpers.

Start with a simple correct tiled kernel that masks the M and N tails.  Each
output tile belongs to one group and accumulates over that group's K rows only;
group boundaries are the running sums of `ks` and always fall on 128-row K
blocks, so the scale row for global K rows `[128q, 128q + 128)` is simply `q`.
A group with `ks[g] == 0` must still write `out[g] = c[g]`, and a call with
`K_total == 0` (zero-row `a` and `b`) must still write every output element.
`M` and `N` can be any size from 1 to 8192, so a grid built with floor division
or a store without a mask silently drops or overruns the tail.

Development commands:

```bash
./tools/run_one.sh --version 1
./tools/run_public_tests.sh --quick --candidate-version 1
./tools/run_public_tests.sh --candidate-version 1
./tools/run_public_tests.sh --family large --candidate-version 1
./tools/run_public_tests.sh --edge --candidate-version 1
./tools/run_public_tests.sh --sanitize --candidate-version 1
```

Allocation, quantization and reference calculation are outside measured time;
compilation during a timed call is not.  Do not rely on the prior contents of
`out`.  Public runs check the four correctness layers, guard bands and
unchanged inputs.

Keep a known-correct version before tuning.  The four families (large batch,
many groups, small batch, sequence) are described in `AGENT_TASK.md`; the
public cases cover all four.  Run
the full public suite, the edge suite and the sanitizer suite once before
selecting `FINAL_VERSION`.  Public tools never run the hidden cases.

Before each `sforge-submit --details`, wait for all public-test and compiler
workers to exit.
