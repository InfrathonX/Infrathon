# Sparse-MLA Prefill Lab

## Submission evaluation policy

Every `sforge-submit` runs the evaluation on the complete hidden suite. `sforge-submit --details` returns the gate states, the score and the four family scores. Hidden cases and per-case results stay on the Judge. Your final result is the best valid score you reach within the time budget, so keep `FINAL_VERSION` on your strongest candidate.

Read `AGENT_TASK.md` before editing. The only submitted surface is:

```text
candidates/kernel_vNNN/submission.py
FINAL_VERSION
notes.md
```

`kernel_v001/submission.py` starts at zero bytes. The selected module must
expose the exact nine-argument `sparse_mla_prefill` ABI, write `out`,
`max_logits` and `lse` in place on the current stream, return `None`, write
nothing outside those three outputs and its own scratch, and leave all inputs
unchanged.

The support domain is wider than the public cases and is a hard gate: `d_qk`
of 512 or 576 at any `h_q` that is a multiple of 16 from 16 to 128, any `s_q`
from 1 to 70000, any `s_kv` from 1 to 1048576 (also below
`topk`), `topk` any multiple of 128 up to 2048, invalid indices interleaved anywhere, duplicated keys, rows with no
valid key, `+-inf` sinks, `topk_length` from 0 to `topk`, and any positive
`sm_scale`. A sensible first version is one program per (query row, head
block) with an FP32 online softmax that already follows four rules:

1. put `s_q` on grid axis 0 and compute row offsets in 64 bits (`s_q` can be
   70000 and `q` can hold more than `2**31` elements);
2. mask every slot individually: a slot is invalid when `j >= topk_length[i]`,
   or the index is negative or `>= s_kv`; never stop a scan at the first
   invalid slot and never deduplicate keys; slots past `topk_length` may hold
   stale indices that look valid;
3. return the raw `max_logits` and `lse`; fold the sink only into the `out`
   normaliser; an empty row gives `out = 0`, `max_logits = -inf`, `lse = +inf`;
4. mask the `s_q` tail of every block; `s_q = 1` is its own Triton
   specialization and must be correct too.

Shape-keyed fast
paths are fine as long as every path is correct for every input that can
reach it. Hidden streams time every call after a short warm-up, so a kernel
that compiles a new specialization for every `(s_q, s_kv)` pays for each
compilation.

Development commands:

```bash
./tools/run_one.sh --version 1
./tools/run_public_tests.sh --quick --candidate-version 1
./tools/run_public_tests.sh --candidate-version 1
./tools/run_public_tests.sh --case public-040-sweep --perf-only --candidate-version 1
./tools/run_public_tests.sh --edge --candidate-version 1
./tools/run_public_tests.sh --sanitize --candidate-version 1
```

The first three run one public case, the first case of each of the eight
groups, and all 108 public cases; `--case` runs named cases and `--perf-only`
skips the reference comparison. `--edge` runs the 73 public edge cases natively
with guard bands and input checks and reports each case; `--sanitize` runs the
seven `sanitize` cases under memcheck (it prints
`MEMCHECK UNAVAILABLE` and exits 2 where the sanitizer cannot attach, and
`MEMCHECK INCONCLUSIVE` and exits 3 when a tool-level error stops the pass).
Run `--sanitize` before `--edge` whenever a kernel is new: memcheck reports an
out-of-bounds access before it reaches the GPU. Public output provides
disclosed shapes, correctness summaries, per-case edge results and performance
aggregates only; it never runs the hidden cases.

Multiply operands are BF16 or FP16 and accumulation and softmax statistics
FP32; FP8, INT8 and FP4 count as shortcuts even within tolerance. Each call
computes only its own outputs: no caching across calls, no work for later
calls, no work after the call returns, and no patching of imported modules,
clocks, CUDA events or global numeric settings. Do not name a function so that
it starts with `_Z` or contains `gemm`, `xmma`, `nvjet`, `cutlass` or `cublas`
(admission rejects the names).

Keep a known-correct version before changing schedules. Record measurements in
`notes.md`, then point `FINAL_VERSION` to the strongest clean candidate and
run the full public suite, the edge suite and the sanitizer suite. Only Torch
metadata/scratch allocation and self-contained Triton are allowed in the submission: no network
use, external source, existing attention or GEMM kernel, private stream,
FP8/INT8/FP4 path, dynamic loading, or evaluator introspection.

Before each `sforge-submit --details`, wait for all public-test and compiler
workers to exit.
