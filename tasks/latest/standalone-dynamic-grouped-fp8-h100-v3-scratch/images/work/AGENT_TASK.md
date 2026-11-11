# Dynamic-grouped FP8 GEMM

Work in `/app/workdir/grouped_fp8_lab` on the NVIDIA GPU allocated to you.  The
task starts with an exactly empty candidate, and no GEMM kernel or reference
implementation is supplied.  Write the complete computation and group scheduling
in self-contained Triton code.

Every selected `submission.py` must define both callables:

```python
def grouped_fp8_gemm_nt_masked(
    a, b, a_scale, b_scale, m_sizes, expected_m_hint, out
) -> None:
    ...

def grouped_fp8_gemm_nt_psum(
    a, b, a_scale, b_scale, group_ends, expected_m_hint, out
) -> None:
    ...
```

Both calls run on the current CUDA stream and return `None`.

## Tensor and math contract

Common tensors are contiguous on the current device; they may be views with a
non-zero storage offset, so always address them through `data_ptr()` and their
shapes:

- `b`: `[G, N, K]`, `torch.float8_e4m3fn`;
- `b_scale`: `[G, N // 128, K // 128]`, FP32;
- `out`: BF16.

For each valid `(g, m, n)`, let `q` select a 128-element K block and let
`r = n // 128`.  Compute the FP32 accumulation

```text
sum_q a_scale[row,q] * b_scale[g,r,q]
      * sum_{k in block q} FP32(a[row,k]) * FP32(b[g,n,k])
```

and cast once to BF16.  Inputs were quantized from BF16 standard-normal source
tensors with scale `max(abs(block), 1e-4) / 448`; A uses one scale per row and
K block, while B uses one scale per 128-by-128 `(N,K)` block.

### Masked entry point

- `a`: `[G, M_max, K]`, FP8 E4M3FN;
- `a_scale`: `[G, M_max, K // 128]`, FP32;
- `m_sizes`: `[G]`, CUDA int32;
- `out`: `[G, M_max, N]`, BF16.

Only rows `0 <= m < m_sizes[g]` are required.

### Prefix-sum entry point

- `a`: `[G * M_max, K]`, FP8 E4M3FN;
- `a_scale`: `[G * M_max, K // 128]`, FP32;
- `group_ends`: `[G]`, CUDA int32;
- `out`: `[G * M_max, N]`, BF16.

Group zero starts at row zero.  For `g > 0`, the start is
`align(group_ends[g - 1], 128)`; every group ends at `group_ends[g]`, so an
empty group has `group_ends[g] == align(group_ends[g - 1], 128)`.

### Support domain

Every call satisfies:

- `1 <= G <= 64`;
- `M_max` is a multiple of 128.  For the masked entry point,
  `128 <= M_max <= 4096`.  For the prefix-sum entry point,
  `128 <= M_max <= 131072` and `G * M_max * (K + 2 * N) <= 2^34`;
- `N` and `K` are multiples of 128 with `128 <= N, K <= 8192` (one-block and
  odd-block-count sizes are in scope; there are no N/K tails);
- **each group has any number of valid rows `0 <= M_g <= M_max`**, including
  empty groups, single rows, full capacity, and counts that are not multiples
  of any tile size;
- **`expected_m_hint` is a positive integer used only for scheduling.**  It is an
  estimate from earlier calls and can be stale or badly wrong: rows beyond any
  bound you derive from it must still be computed, and those calls count toward
  the score.

Invalid A rows (masked rows `m >= m_sizes[g]`, prefix-sum alignment gaps and
trailing capacity) and their `a_scale` entries contain arbitrary values, often
NaN.  Rows of `out` outside the valid region are unspecified: you may leave them
or overwrite them.
There are no aliases, non-contiguous tensors, N/K tails, multiple GPUs, or
private streams.

## Correctness and memory safety

For every non-empty group:

1. the output must match the exact FP8 math above: the normalized error
   `1 - 2*dot(c, r) / (sum(c^2) + sum(r^2))` against the FP64 product of the
   dequantized operands must be below `5e-5`.  Accumulate in FP32: BF16 or FP16
   accumulation and casting partial blocks to a lower precision are not allowed;
2. the same normalized error against the unquantized BF16 source GEMM must be
   below `1e-3` for every group with at least 16,384 output elements, and for
   all smaller groups of a case pooled together when they reach 16,384 elements;
3. every valid output element must be finite and written by the call.  Do not
   rely on the prior contents of `out`.

Every call must write only inside `out`'s valid-or-unspecified region and must
leave every input byte unchanged.  Out-of-bounds reads and writes are hard
failures.  On platforms where compute-sanitizer cannot attach to a CUDA process,
the public `--sanitize` suite stops with `MEMCHECK UNAVAILABLE` (exit status 2)
instead of hanging.

## Workloads, feedback, and score

Scored workloads model one rank of a mixture-of-experts model in serving, in four
families:

- **Prefill:** large batches in the prefix-sum layout.
- **Decode:** small batches in both layouts.  Row counts within one call can be
  very uneven, some groups can be empty, and some hints are poor estimates.
- **Latency:** a single group, or very few rows across mostly empty groups.
- **Sequence:** many consecutive calls alternating between two `(N, K)` shapes,
  with different row counts on every call and no synchronization between calls.
  Inputs, including `b`, can repeat across calls.

**Public tests.**  These cover the four families and are meant for development
and profiling.  A good public number does not guarantee a good Judge score; write
your own benchmarks for performance work.

**Hidden tests.**  Your score comes from hidden test cases on the Judge.  They use
different `N` and `K` values and larger batches than the public tests, and there
are many more of them.  Keep your kernel correct for any input the contract above
allows, and make optimizations general rather than fitting particular shapes.

**Measurement.**  Time is the larger of the synchronized host wall time and the
CUDA-event time.  Before each timed single call, an untimed call runs with the
same shapes and hint but different row counts.  Each sequence gets one untimed
round with the same shapes and hints.  Compilation during timed calls counts as
time.

```text
effective TFLOPS = 2 * sum(actual M) * N * K / seconds / 1e12
```

A single-call case is the median TFLOPS of its trials.  A sequence case is the
median over its rounds of total work divided by total time.  Cases are
geometrically averaged within each family, and the primary score is the geometric
mean of the four family scores.  There is no minimum performance threshold.
Correctness, edge-case correctness, memory safety, harness integrity, admitted
source, and a supported GPU are hard gates; any failure makes the submission
invalid.

Useful commands:

```bash
./tools/run_one.sh --version 1                                # one public case
./tools/run_public_tests.sh --quick --candidate-version 1      # one case per layout
./tools/run_public_tests.sh --candidate-version 1              # all 60 public cases
./tools/run_public_tests.sh --family decode --candidate-version 1  # one family
./tools/run_public_tests.sh --edge --candidate-version 1       # public edge cases
./tools/run_public_tests.sh --sanitize --candidate-version 1   # memcheck subset
```

Put versions at `candidates/kernel_vNNN/submission.py`, keep three-digit
version directories, and write the selected integer to `FINAL_VERSION`.
Keep concise measurements in `notes.md`.

## Submission evaluation policy

Every `sforge-submit` runs the evaluation on hidden test cases. `sforge-submit --details` returns the gate states, the score and the four family scores. Hidden cases and per-case results stay on the Judge. Your final result is the best valid score you reach within the time budget, so keep `FINAL_VERSION` on your strongest candidate.

## Submitting

Wait for every public test and compiler process to finish, then run
`sforge-submit --details`.  Each Judge run uses part of your budget, so run the
public edge and sanitizer suites first.

## Allowed implementation surface

Use `torch` for tensor metadata, scratch allocation and current-stream dispatch,
and `triton` / `triton.language` for submitted kernels.

Source admission accepts imports of `torch`, `triton`, `math`, `functools`,
`itertools`, `typing`, `dataclasses`, `collections`, `enum` and `__future__`
only, and rejects:

- an existing GEMM through `torch.mm`, `matmul`, `bmm`, `_scaled_mm`, `linear`,
  `einsum`, `@`, `torch.ops`, a GEMM library, or another compiled extension;
- assigning to, or deleting, any attribute of an imported module, class or
  object (for example `time.perf_counter = ...`,
  `torch.cuda.Event.elapsed_time = ...`, `torch.backends... = ...`), and
  `setattr`, `delattr`, `getattr`, `vars`, `globals`, `locals`, `eval`, `exec`,
  `compile`, `open`, `__import__` and double-underscore attributes other than
  `__init__` and `__name__`;
- `torch.utils`, `torch.backends`, `torch.jit`, `torch.library`, `torch._*`,
  `triton.compiler`, `triton.tools` and similar internal namespaces;
  `torch.set_*`, RNG seeding, `torch.save` / `torch.load`, CUDA graphs, private
  streams, process spawning and binary loading.

A run that replaces clocks, CUDA events, checking functions or a global numeric
setting is invalid.

Do not fetch or load external implementations, replay outputs, specialize for
exact seeds, values, evaluation order, or benchmark presence, or import or
inspect the runtime, public runner, worker, or Judge.  Cache only compiled code
and scheduling metadata; do not keep anything derived from tensor contents across
calls.  The visible benchmark is trusted test infrastructure, not an
implementation library.

Before finishing, point `FINAL_VERSION` at a clean candidate and run the full
public suite, the public edge suite and the sanitizer suite once.
