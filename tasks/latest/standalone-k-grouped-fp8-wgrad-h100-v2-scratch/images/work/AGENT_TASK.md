# K-grouped FP8 weight-gradient GEMM

Work in `/app/workdir/k_grouped_wgrad_lab` on the NVIDIA GPU allocated to you.
The task starts with an exactly empty candidate, and no GEMM kernel or reference
implementation is supplied.  Write the complete computation and K-group
scheduling in self-contained Triton code.

Every selected `submission.py` must define this callable:

```python
def k_grouped_fp8_gemm_tn_contiguous(
    a, b, a_scale, b_scale, ks, c, expected_k_hint, out
) -> None:
    ...
```

The call runs on the current CUDA stream and returns `None`.

## Tensor and math contract

This is the K-grouped weight-gradient GEMM: `G` groups of token rows are
concatenated along K, and each group accumulates one independent
`D_g = C_g + A_g^T @ B_g` product into an FP32 output.  All tensors are
contiguous on the current device; they may be views with a non-zero storage
offset, so always address them through `data_ptr()` and their shapes:

- `a`: `[K_total, M]`, `torch.float8_e4m3fn`;
- `b`: `[K_total, N]`, `torch.float8_e4m3fn`;
- `a_scale`: `[K_total // 128, M]`, FP32;
- `b_scale`: `[K_total // 128, N]`, FP32;
- `ks`: `[G]`, CUDA int32; every entry is 0 or a positive multiple of 128 and
  `sum(ks) == K_total`;
- `c`: `[G, M, N]`, FP32, read-only addend;
- `out`: `[G, M, N]`, FP32; every element must be written by the call.

Group `g` owns the K rows `[sum(ks[:g]), sum(ks[:g]) + ks[g])`.  Groups are
concatenated without gaps, so every 128-row K block lies inside exactly one
group and scale row `q` covers global K rows `[128q, 128q + 128)`.

For each `(g, m, n)`, with `Q_g` the set of global 128-row K blocks of group
`g`, compute the FP32 accumulation

```text
out[g,m,n] = c[g,m,n]
           + sum_{q in Q_g} a_scale[q,m] * b_scale[q,n]
             * sum_{k in block q} FP32(a[k,m]) * FP32(b[k,n])
```

and store FP32.  A group with `ks[g] == 0` has `out[g] = c[g]`.  Inputs were
quantized from BF16 standard-normal source tensors with scale
`max(abs(block), 1e-4) / 448`, one scale per (128-row K block, column) pair for
both operands — the `(1, 1, 128)` weight-gradient recipe.  `C` is a standard
normal times 32.

### Support domain

Every call satisfies:

- `1 <= G <= 128`;
- **`M` and `N` are any integers from 1 to 8192** — not only multiples of 128
  or 8: sizes such as 1, 3, 7, 129 or 8191 are in scope;
- **every group K is 0 or a positive multiple of 128**, `K_total <= 262144`;
  empty groups and an empty problem (`K_total == 0`, so `a`, `b` and the
  scales have zero rows) are in scope;
- `G * M * N <= 2^31 - 1`;
- **`expected_k_hint` is a positive integer used only for scheduling.**  It is
  an estimate from earlier steps and can be stale or badly wrong: K rows beyond
  any bound you derive from it must still be computed, and those calls count
  toward the score.

Any call inside this domain can appear in the hidden tests.  Keep every path
correct across the whole domain; do not assume tighter bounds than the ones
stated here.

There are no aliases, non-contiguous tensors, K tails inside a 128-row block,
multiple GPUs, or private streams.  When `M` is not a multiple of 16, rows of
`a` are not 16-byte aligned (and likewise `b` for `N`), so paths that need
alignment must have a fallback for those shapes.

## Correctness and memory safety

Let `E` be the exact result (the formula above evaluated in FP64 from the
dequantized FP8 operands) and `R[g,m,n] = sqrt(sum_k (a s_a)^2 (b s_b)^2)` the
magnitude scale of the element's products.  Every call must satisfy:

1. every output element is finite and written by the call.  Do not rely on the
   prior contents of `out`;
2. **element-wise:** `|out - E| <= 0.05 * R + 2^-20 * |c|` for every element.
   Correct FP32 accumulation stays far below this; a dropped or duplicated K block, a tile left at `c`, a
   wrong scale row, or a single wrong element fails;
3. **precision:** per group, the normalized error
   `1 - 2*dot(out, E) / (sum(out^2) + sum(E^2))` is below `2e-7`.  Correct FP32
   accumulation stays near `1e-8`; accumulating in
   FP16/BF16, or rounding the block dot, the scale product or the dequantized
   operands to 16 bits, fails on deep-K groups;
4. a check against the unquantized BF16 source GEMM,
   `calc_diff < 1e-3`, is applied to every group with at least 16,384 output
   elements, and to all smaller groups of a case pooled together when they
   reach 16,384 elements.

Every call must write only inside `out` and must leave every input byte
unchanged (`a`, `b`, both scales, `ks` and `c`).  The Judge checks memory safety
and every input after every call, so out-of-bounds reads and writes are hard
failures.  Mask the M and N tails of loads and stores.

On platforms where compute-sanitizer cannot attach to a CUDA process, the public
`--sanitize` suite stops with `MEMCHECK UNAVAILABLE` (exit status 2) instead of
hanging.  Out-of-bounds reads and writes remain contract violations there.

## Workloads, feedback, and score

Scored workloads model one rank of a mixture-of-experts training step: the
weight gradient of the expert linear layers, where each group is one local
expert and its K is the number of tokens routed to it.  They come in four
families:

- **Large batch:** many tokens per expert and deep K, with routing from
  balanced to strongly uneven.
- **Many groups:** dozens to more than a hundred local experts, many of them
  small or empty.
- **Small batch:** few tokens per expert; most groups hold one or two K blocks,
  or none.
- **Sequence:** one backward pass through many layers, alternating between two
  `(M, N)` shapes, with different group sizes on every call and no
  synchronization between calls.  Inputs, including `c` and `out`, can be
  reused across calls.

**Public tests.**  These cover the four families and are meant for development
and profiling.  A good public number does not guarantee a good Judge score;
write your own benchmarks for performance work.

**Hidden tests.**  Your score comes from hidden test cases on the Judge.  They
use different `(M, N)` shapes and wider ranges of K, G and routing imbalance
than the public tests, and there are many more of them.  Keep your kernel
correct for any input the contract above allows, and make optimizations general
rather than fitting particular shapes or hints.

**Measurement.**  Time is the larger of the synchronized host wall time and the
CUDA-event time.  Before each timed single call, an untimed call runs with the
same shapes and hint but different group sizes.  Each sequence gets one untimed
round with the same shapes and hint.  Compilation during timed calls counts as
time.

```text
effective TFLOPS = 2 * sum(ks) * M * N / seconds / 1e12
```

A single-call case is the median TFLOPS of its trials.  A sequence case is the
median over its rounds of total work divided by total time.  Within a family,
cases are geometrically averaged per model and then across models, so
every model counts equally regardless of how many cases it has.  The primary
score is the geometric mean of the four family scores.  There is no minimum
performance threshold.  Correctness, edge-case correctness, memory safety,
harness integrity, admitted source, and a supported GPU are hard gates; any
failure makes the submission invalid.

Useful commands:

```bash
./tools/run_one.sh --version 1                                # one public case
./tools/run_public_tests.sh --quick --candidate-version 1      # two public cases
./tools/run_public_tests.sh --candidate-version 1              # all 123 public cases
./tools/run_public_tests.sh --family large --candidate-version 1  # one family
./tools/run_public_tests.sh --edge --candidate-version 1       # public edge cases
./tools/run_public_tests.sh --sanitize --candidate-version 1   # memcheck subset
```

`run_one.sh` runs one small-batch case, quick adds one large-batch case, and the
full suite runs all public cases with three measured trials each (sequence cases
one round; one and quick use one trial).  The public cases are listed in
`/workspace/benchmark/public_cases.json`.  `--edge` runs the 43 public edge cases in
`/workspace/benchmark/public_edge_cases.json` natively with guard bands and input
checks and reports every case; each is untimed and called twice.  `--sanitize`
runs the `sanitize` ones under memcheck.  Public tools never run or reveal hidden
cases.

Put versions at `candidates/kernel_vNNN/submission.py`, keep three-digit
version directories, and write the selected integer to `FINAL_VERSION`.
Keep concise measurements in `notes.md`.

## Submission evaluation policy

Every `sforge-submit` runs the evaluation on hidden test cases. `sforge-submit --details` returns the gate states, the score and the four family scores. Hidden cases and per-case results stay on the Judge. Your final result is the best valid score you reach within the time budget, so keep `FINAL_VERSION` on your strongest candidate.

## Submitting

Wait for every public test and compiler process to finish, then run
`sforge-submit --details`.  Each Judge run uses part of your budget.
Run the public sanitizer and edge suites first (the sanitizer suite before the
edge suite whenever a kernel is new: an out-of-bounds access under memcheck is
reported instead of reaching the GPU).

## Allowed implementation surface

Use `torch` for tensor metadata, scratch allocation, small metadata
computations (for example a `cumsum` of `ks`) and current-stream dispatch, and
`triton` / `triton.language` for submitted kernels.  Any technique inside
self-contained Triton is allowed, including separate paths for different shapes
as long as every path is correct for the inputs that reach it.  Keep scratch
allocations within the size of `out`: at the domain limit `c` and `out` are
each about 8 GiB on the device.

Source admission accepts imports of `torch`, `triton`, `math`, `functools`,
`itertools`, `typing`, `dataclasses`, `collections`, `enum` and `__future__`
only, and rejects:

- an existing GEMM through `torch.mm`, `matmul`, `bmm`, `_scaled_mm`, `linear`,
  `einsum`, `@`, `torch.ops`, cuBLAS, cuBLASLt, or another compiled library or
  extension;
- assigning to, or deleting, any attribute of an imported module, class or
  object (for example `time.perf_counter = ...`,
  `torch.cuda.Event.elapsed_time = ...`, `torch.backends... = ...`), and
  `setattr`, `delattr`, `getattr`, `vars`, `globals`, `locals`, `eval`, `exec`,
  `compile`, `open`, `__import__` and double-underscore attributes other than
  `__init__` and `__name__`;
- `torch.utils`, `torch.backends`, `torch.jit`, `torch.library`, `torch._*`,
  `torch.cuda.memory`, `triton.compiler`, `triton.tools` and similar internal
  namespaces; `torch.set_*`, RNG seeding, `torch.save` / `torch.load`, CUDA
  graphs, private streams, threads, process spawning and binary loading.

A run that replaces clocks, CUDA events, checking functions or a global numeric
setting fails `harness_integrity_ok`.

Cache only compiled kernels and metadata that depends on shapes and
`expected_k_hint`; recompute anything that depends on the values of `ks` or any
other tensor on every call.  Do not fetch or load external
implementations, replay outputs, compute results for a call other than the
current one, specialize for exact seeds, values,
evaluation order, or benchmark presence, or import or inspect the runtime,
public runner, worker, or Judge.  The visible benchmark is trusted test
infrastructure, not an implementation library.

Before finishing, point `FINAL_VERSION` at a clean candidate and run the full
public suite, the public edge suite and the sanitizer suite once.
