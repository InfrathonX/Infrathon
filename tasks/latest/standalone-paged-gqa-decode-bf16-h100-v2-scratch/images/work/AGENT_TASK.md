# BF16 paged-GQA decode

Work in `/app/workdir/paged_gqa_decode_lab` on the NVIDIA GPU allocated to you (compute capability 8.0 or newer, at least 32 GiB of device memory). The
starter `candidates/kernel_v001/submission.py` is exactly empty. Implement the
whole paged gather, scaled dot products, stable softmax, and value accumulation
in self-contained Triton. No existing attention kernel, compiled attention
library or earlier candidate is present.

The selected `submission.py` must define exactly:

```python
def paged_gqa_decode_bf16(
    q, k_cache, v_cache, page_table, cache_seqlens, softmax_scale, out
) -> None:
    ...
```

Launch on the current CUDA stream, overwrite every element of `out`, return
`None`, write nothing outside `out` and scratch tensors the call allocated
itself, and leave `q`, both caches, `page_table`, and `cache_seqlens`
bit-identical.

## Tensor and addressing contract

All tensors are contiguous CUDA tensors on the current device, and
`softmax_scale` is always the Python float `1 / sqrt(128)`.

- `q`: `[B, S_q, H_q, 128]`, `torch.bfloat16`
- `k_cache`, `v_cache`: `[P, page_size, H_kv, 128]`, `torch.bfloat16`
- `page_table`: `[B, W]`, `torch.int32`
- `cache_seqlens`: `[B]`, `torch.int32`
- `out`: `[B, S_q, H_q, 128]`, `torch.bfloat16`, NaN-poisoned before
  each call

### Support domain

Every call satisfies:

- **`B` is any integer from 1 to 256 and `S_q` any integer from 1 to 16**;
- head dimension `D` is always 128; **`H_q <= 128` and `H_q % H_kv == 0`
  with any group size `H_q // H_kv`**, a power of two or not, including MHA
  (`H_q == H_kv`) and MQA (`H_kv == 1`);
- **`page_size` is any power of two from 1 to 256**;
- every `L_b = cache_seqlens[b]` is any integer with
  `S_q <= L_b <= L_max <= 1048576`;
- **the page table can be wider than needed**: `W >= ceil(L_max / page_size)`;
- **the page pool can be larger than `B * W`, and page ids can repeat**:
  several requests may read the same physical page (a shared prefix).
  One cache can hold more than 2^31 elements, so compute cache offsets with
  64-bit integers;
- **contents you must not rely on**: table entries at index
  `>= ceil(L_b / page_size)` of row `b` are arbitrary valid page ids in
  `[0, P)`, and cache slots no query may see -- positions `j >= L_b` in a
  request's last page, and pages no request uses -- can hold any BF16 bit
  pattern, **including NaN and Inf**. The output may depend only on visible
  keys.

There are no aliases, non-contiguous tensors, rows without a visible key
(`L_b < S_q`), multiple GPUs, or private streams.

For request `b`, logical key position `j`, and KV head `g`, the physical
cache element is:

```text
physical_page = page_table[b, j // page_size]
slot          = j % page_size
K[b,j,g,:]    = k_cache[physical_page, slot, g, :]
V[b,j,g,:]    = v_cache[physical_page, slot, g, :]
```

Do not assume an identity, monotone or one-to-one page table, a table width
of exactly `ceil(L_max / page_size)`, or a pool of exactly `B * W` pages.

## Attention contract

Query head `h` uses the contiguous GQA group:

```text
kv_head = h // (H_q // H_kv)
```

Bottom-right causal alignment means query row `i` of request `b` may attend
exactly the logical keys:

```text
0 <= j <= L_b - S_q + i
```

For those visible keys:

```text
logit[j] = softmax_scale * dot(FP32(q[b,i,h,:]), FP32(K[b,j,kv_head,:]))
prob     = stable_softmax(logit)
out[b,i,h,:] = sum_j prob[j] * FP32(V[b,j,kv_head,:])
```

Every visible key of every query row, head and request contributes; online
softmax must remain stable for extreme positive and extreme negative logits.

**Precision.** Compute in BF16 and FP32 only: BF16 tensor-core products with
FP32 accumulation, FP32 logits and softmax statistics, and BF16 or FP32
probabilities in the P·V product are all fine. FP16, FP8, INT8 and FP4 are
approximations of this contract wherever they appear, even when the result
stays within tolerance.

### Implementation notes for the support domain

1. `tl.arange` and `tl.dot` block shapes are compile-time powers of two
   (`tl.dot` needs at least 16 along each dimension). Pad a group size or
   `S_q` that is not a power of two up to the next power of two and mask the
   padded rows on load and store. Triton also compiles separate
   variants for integer arguments equal to 1 or divisible by 16 and for
   16-byte-aligned pointers, so a kernel tried only on aligned shapes has
   variants that were never compiled or run.
2. Use `page_table.shape[1]` (or `page_table.stride(0)`) as the row stride of
   the page table, never `k_cache.shape[0] // B`: the pool can be larger than
   `B * W`, and the table wider than needed.
3. Zero or mask V for `j >= L_b` before the P·V product (a masked load with
   `other=0`): a probability of exactly 0 times a NaN slot is still NaN.
4. Mask K logits with `tl.where`, not by adding a `-inf` bias:
   `NaN + (-inf)` is NaN. Visible logits can be as low as about `-4.6e4`, so a
   finite mask value must be far below that (for example `-1e30`).

## Correctness and memory safety

The Judge computes an independent FP32 de-paged reference `ref` over the exact
BF16 inputs, and a BF16 reference `bref` (logits and the
P·V product rounded to BF16, softmax in FP32). Every scored and edge call must
satisfy:

```text
max_abs(out - ref)          <= 2.0 * max_abs(bref - ref) + max(1e-5, 2 * rounding_probe)
mean_abs(out - ref)         <= 1.5 * mean_abs(bref - ref)
||out - ref||_2 / ||ref||_2 <= 5e-3
```

with `rounding_probe = max_abs(((ref + 0.3) - 0.3) - ref)`. The max is element-wise, so a single wrong request,
head or row fails. The comparisons use `<=`, so a zero limit demands an exact
answer: when every visible context has one key, `bref` equals `ref` and `out`
must reproduce `V` exactly (a correct kernel computes `exp(0) = 1` and
divides by 1). Every output element must be finite and written by the call.

Each call may write only to `out` and to scratch tensors it allocated itself,
and must leave every input byte unchanged, even temporarily. Reads outside an
input's extent are violations too. The Judge checks memory safety and every
input after every call. An invalid access, a changed input, or a CUDA fault
fails `memory_safety_ok`; a CUDA fault also stops the run at once.

On platforms where compute-sanitizer cannot attach to a CUDA process, the
public `--sanitize` suite stops with `MEMCHECK UNAVAILABLE` (exit status 2)
instead of hanging. Out-of-bounds reads and writes remain contract violations
there.

The Judge's gates are `profile_ok`, `suite_ok`, `source_ok`, `hardware_ok`,
`correctness_ok` (scored cases), `edge_correctness_ok` (edge cases), `memory_safety_ok` and `harness_integrity_ok`; any
failure makes the submission invalid.

## Workloads, feedback, and measurement

Scored workloads model one tensor-parallel rank of a serving engine during
decode, in four families:

- **Decode:** one decode step of a busy server, `S_q = 1`. Lengths within a
  batch are very uneven (the longest request is often several times the mean),
  prompts share prefix pages, the page pool is fragmented and larger than the
  batch needs, and the table is wider than needed. Hidden slots can hold NaN.
  Some batches share pages far more heavily: parallel samples of one prompt,
  beams that share most of their history, or many requests behind one short
  common prompt.
- **Speculative:** 2 to 16 query tokens per request, as when a server verifies
  draft tokens, at small batch sizes.
- **Long context:** one or a few requests with up to about one million cached
  tokens, sometimes one long request among many short ones.
- **Sequence:** hundreds of consecutive decode steps of one server, some of them
  with several query tokens per request. Requests join and leave between steps,
  so `B` and every length change on every call while the table width and the
  page pool stay fixed.

In some cases Q, K and V contain rare outliers about ten times larger than the
rest of the values.

**Public tests.** `/workspace/benchmark/public_cases.json` lists 127 public
cases that cover the four families. A good public number does not guarantee a
good Judge score; write your own benchmarks for performance work across the
support domain.

**Hidden tests.** Your score comes from hidden test cases on the Judge. They
stay inside the support domain above but use different head configurations,
different request traces and page sizes, larger contexts, and there are many
more of them. A hidden call typically holds more requests and more KV than
most public calls. Judge a design by how it performs at that scale: compare
candidates on your own benchmarks with large batches and long contexts, and
use the public suite to check correctness and catch regressions, not as the
target to tune to. A change that helps the small public shapes can lose on the
Judge. Keep every path correct across the whole domain, and make
optimizations hold up at larger scale rather than fitting the public shapes.

**Measurement.** Time is the larger of the synchronized host wall time and the
CUDA-event time of one call on fresh tensors. Before each timed single call, an
untimed call runs with the same `B`, `S_q`, heads, page size, table width and
pool but different lengths and data. Each sequence starts after 16 untimed
calls on nearby shapes; after that every step is timed, and **compilation or
autotuning during these timed calls counts as time.** A kernel that recompiles
whenever `B`, a length or the table width changes therefore loses most of its
sequence score. The timed calls of one sequence may take at most 120 seconds in
total; a sequence that runs longer fails the correctness gate.

Useful commands:

```bash
./tools/run_one.sh --version 1                                # one public case
./tools/run_public_tests.sh --quick --candidate-version 1      # one case per call family
./tools/run_public_tests.sh --candidate-version 1              # all 127 public cases
./tools/run_public_tests.sh --family stream --candidate-version 1  # one family
./tools/run_public_tests.sh --edge --candidate-version 1       # public edge cases
./tools/run_public_tests.sh --sanitize --candidate-version 1   # memcheck subset
```

`run_one.sh` runs `public-0012-decode`; quick runs `public-0012-decode`,
`public-0044-spec` and `public-0084-long` with one measured trial each; the full
suite and `--family` run three measured trials per single-call case.
`--edge` runs the 44 public edge cases in
`/workspace/benchmark/public_edge_cases.json` natively with guard bands and
input checks and reports every case; each is untimed and called twice.
`--sanitize` runs the `sanitize` ones under memcheck. Public tools never run or
reveal hidden cases.

## Metric

For a call whose requests have realized lengths `L_b`:

```text
pairs_b         = S_q * L_b - S_q * (S_q - 1) / 2
effective_flops = 4 * 128 * H_q * sum_b(pairs_b)
```

A single-call case scores `effective_flops / measured_seconds / 1e12`, the
median over three fresh trials. A sequence case scores its total effective
FLOPs over all steps divided by the total time of all its timed steps. The
Judge geometrically averages the case TFLOPS inside each family and computes:

```text
score_tflops = geomean(decode, speculative, long context, sequence)
```

`sforge-submit --details` reports the score and the four family scores. The raw
TFLOPS score is uncapped and maximize-only. It is not divided by any baseline.
Correctness (including the sequence time limit), edge-case correctness, full
writes, memory safety, harness integrity, source admission, suite identity,
and one GPU meeting the hardware requirement above are hard gates.

## Version workflow

Keep experiments at `candidates/kernel_vNNN/submission.py` using three-digit
directories. Only the version named by the canonical integer in
`FINAL_VERSION` is evaluated; older versions may remain for comparison.
Keep concise decisions and public measurements in `notes.md`. Before
finishing, select the strongest clean candidate and run the full
public suite, the public edge suite and the sanitizer suite once.

## Submission evaluation policy

Every `sforge-submit` runs the evaluation on hidden test cases. `sforge-submit --details` returns the gate states, the score and the four family scores. Hidden cases and per-case results stay on the Judge. Your final result is the best valid score you reach within the time budget, so keep `FINAL_VERSION` on your strongest candidate.

## Submitting

Wait for every public test and compiler process to finish, then run
`sforge-submit --details`. A Judge run takes 15 to 40 minutes of your budget.
Run the public sanitizer and edge suites first (the sanitizer suite before the
edge suite whenever a kernel is new: an out-of-bounds access under memcheck is
reported instead of reaching the GPU, where a wild address can hang the driver).

## Allowed implementation surface

The selected source may import only Python `math`/`typing`, `torch`,
`triton`, and `triton.language`. Use Torch only for tensor metadata and
safe same-device scratch allocation. Use one or more reachable
`@triton.jit` kernels for every tensor data path. Triton loads, stores,
`tl.dot`, reductions, atomics, persistent/split-K schedules, autotuning, and
compiler meta-parameters are allowed.

### Inside `@triton.jit` kernels

Ordinary Triton is available, including `for`, `tl.static_range`, `tl.range`,
nested control flow, `tl.dot`, `tl.atomic_*`, and `tl.math.*`. The rules in
the next subsection do **not** apply to kernel bodies.

### Host code is a small deliberate subset

Source admission reads the selected module statically, so host code -- the
entrypoint, every helper it calls, and any callable grid -- is restricted.
This is a fixed contract, not a bug; write to it from the first version rather
than discovering it through rejections.

- Statements: assignment, annotated assignment, `assert`, a bare call, `if` /
  `else`, `pass`, and `return`. There is no host `for`, `while`, `try`,
  `with`, `del`, `raise`, `import`, `global`, `nonlocal`, nested `def`, or
  `class`. Unroll host loops or move the iteration into the kernel.
- Expressions: no comprehensions, generators, `lambda`, f-strings, `%`/
  `.format` string building, `*`/`**` unpacking, or walrus assignment.
- No default argument values on any function.
- No `is` / `is not` comparison in host code.
- Torch metadata only: the `.shape`, `.dtype`, `.device`, `.ndim`, and
  `.is_cuda` attributes and the `.dim()`, `.size()`, `.stride()`, `.numel()`,
  and `.is_contiguous()` methods. `.view`, `.reshape`, `.contiguous`,
  `.element_size`, `.data_ptr`, and `torch.cuda.get_device_properties` are
  not available; hard-code or derive what you need instead.
- Torch allocation only through `torch.empty`, `torch.empty_like`,
  `torch.empty_strided`, `torch.full`, `torch.full_like`, `torch.zeros`, and
  `torch.zeros_like`.
- Other calls are limited to `abs`, `bool`, `float`, `int`, `len`, `max`,
  `min`, `round`, `tuple`, `math.ceil`, `math.floor`, `math.log2`,
  `math.sqrt`, `triton.cdiv`, your own host helpers, and kernel launches.
- Every host function must be reachable from the entrypoint. Delete helpers
  you stop calling; dead host code is rejected.
- Kernel decorators are `@triton.jit` with no arguments and
  `@triton.autotune(configs=..., key=..., warmup=..., rep=...)` built from
  `triton.Config({...}, num_warps=..., num_stages=..., num_ctas=...,
  maxnreg=...)`. `@triton.heuristics`, `prune_configs_by`, `do_bench`,
  `pre_hook`, and `launch_metadata` are not available: compute meta-parameters
  in host code and pass them as `tl.constexpr` arguments.
- A callable grid is allowed, as in `kernel[my_grid](...)`, and `my_grid`
  obeys every host rule above.

Do not perform attention, GEMM, softmax, tensor arithmetic, input mutation, or
output copying through host Torch. Do not use FP16, FP8, INT8 or FP4, private
CUDA streams, `torch.ops`, extension compilation, dynamic imports/code
execution, filesystem/process/network/environment-variable APIs,
dunder/frame/traceback introspection, or module-dictionary mutation. `setattr`, `getattr`,
`delattr`, `eval`, `exec`, `compile`, `globals`, `locals`, `vars`, and `open`
are rejected anywhere in the module, as is any dunder name, including one
written as a string. Do not load existing kernels from any attention
library, another compiled library, the visible benchmark, or
the Judge, and do not import, read or inspect the runtime, public runner,
public edge manifest, worker or Judge from the submission: the visible
benchmark is trusted test infrastructure, not an implementation library.
Do not copy existing implementations into the submission.

### Integrity rules

- **Shape-keyed paths must be correct everywhere they can be selected.**
  Choosing tile sizes, kernels, schedules and fast paths from `B`, `S_q`,
  `H_q`, `H_kv`, the page size, the table width or the pool size is welcome,
  as is an aligned fast path beside a masked general path. A path that is
  correct only for particular shapes -- for example one that drops the running
  maximum, drops tail or causal masks, or relies on a one-to-one page layout
  (`P == B * W`, finite hidden slots) -- is a contract violation even when no
  test catches it.
- **Each call computes only its own output.** Do not cache or replay outputs
  across calls, compute during one call (an untimed one included) anything a
  later call reuses, inspect allocator state or device memory that was not
  passed to the call, or reconstruct seeds or future inputs. Do not specialize
  for seeds, values, call order, untimed versus measured calls, or the
  presence of the benchmark, guard bands or compute-sanitizer.
- **All work belongs to the call.** Launch every kernel on the current CUDA
  stream before returning; no background threads, processes, other streams,
  or work deferred past the call's return.
- **Leave the environment alone.** Do not assign to, wrap or replace any
  attribute of an imported module, class or function (for example
  `time.perf_counter` or `torch.cuda.Event.elapsed_time`), and do not change
  global numeric settings (TF32 flags, matmul precision, the default dtype).
  A run that does so fails `harness_integrity_ok`.
