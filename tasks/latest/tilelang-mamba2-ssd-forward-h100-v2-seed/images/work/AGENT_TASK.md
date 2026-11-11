# TileLang Mamba-2 SSD forward

Work in `/app/workdir/mamba2_ssd_lab` on the NVIDIA GPU allocated to you.  A **correct
but deliberately unoptimised** four-stage TileLang implementation is already
supplied as `candidates/kernel_v001/submission.py`.  Your job is to make it
faster without changing what it computes.

The score is raw effective throughput.  There is no threshold to clear, no
reference implementation to beat, and no cap: every improvement moves the
number.

## Entry point

Every selected `submission.py` must define this callable:

```python
def ssd_forward(x, dt, A, B, C, D, out) -> None:
    ...
```

It runs on the current CUDA stream, writes `out` completely, leaves every input
unchanged, and returns `None`.

## Tensor contract

All tensors are contiguous and live on the current device.  `H % G == 0` and
`g(h) = h // (H // G)`.

- `x`:   `[batch, L, H, P]`, `torch.bfloat16`;
- `dt`:  `[batch, L, H]`, FP32, strictly positive, **already preprocessed** —
  the kernel applies no softplus, no `dt_bias`, and no `dt_limit` clamp;
- `A`:   `[H]`, FP32, strictly negative;
- `B`:   `[batch, L, G, N]`, `torch.bfloat16`;
- `C`:   `[batch, L, G, N]`, `torch.bfloat16`;
- `D`:   `[H]`, FP32;
- `out`: `[batch, L, H, P]`, `torch.bfloat16`; every element must be written.

### Support domain

Every call satisfies:

- **`batch` is any integer from 1 to 16 and `L` any positive integer**, including
  lengths below one chunk and long lengths that are not multiples of 64, 128 or
  256;
- **`H` is any integer from 1 to 256 and `G` any divisor of `H`**: a group can
  hold one head or all of them, and `H // G` need not be a power of two;
- **`P` is a multiple of 16 from 16 to 256, and `N` is a power of two from 16 to
  256**;
- no tensor reaches `2**31` elements.

A tail chunk that is handled by padding must leave the padded positions
completely inert.

## Math contract

Per `(b, h)`, with state `S` in `R^{P x N}` starting at zero:

```text
for t = 0 .. L-1:
    dA           = exp(A[h] * dt[b,t,h])
    S            = dA * S + dt[b,t,h] * outer(x[b,t,h,:], B[b,t,g,:])
    out[b,t,h,:] = S @ C[b,t,g,:] + D[h] * x[b,t,h,:]
```

Exponentials, the state, and every accumulation are FP32 semantics; bfloat16 is
storage and tensor-core operand format only.  There is no `z` gate, no returned
final state, and no backward pass.

**The chunk size is not part of this contract.**  How you decompose the scan is
your choice.

`out` is compared against an independent blocked FP32 reference with TF32
disabled.  For generated inputs, an element is accepted when **either** its absolute error is within
`1.5e-2` times the reference tensor's own maximum magnitude **or** its relative
error is within `3.01/128`; in addition the whole-tensor relative L2 must stay
within `6e-3`.

## Workloads and score

Scored workloads come in four families:

- **Batch:** training and padded batches, batch 2 to 16.
- **Serving:** batch 1, with lengths drawn from real request traces.
- **Long context:** batch 1, long sequences.
- **Real:** layers captured from real models on real text, each timed once.

**Public tests.** These cover the four families and are meant for development and
profiling.  A good public number does not guarantee a good Judge score; write
your own benchmarks for performance work across the support domain.

| case | family | `H` | `P` | `N` | `G` | batch | `L` |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| P1 | batch | 48 | 64 | 256 | 1 | 8 | 1280 |
| P2 | batch | 96 | 80 | 128 | 4 | 4 | 7000 |
| P3 | serving | 40 | 64 | 64 | 1 | 1 | 24000 |
| P4 | batch | 256 | 48 | 128 | 16 | 2 | 5000 |
| P5 | batch | 48 | 64 | 128 | 1 | 8 | 6144 |
| P6 | batch | 64 | 128 | 256 | 4 | 16 | 1393 |
| P7 | batch | 24 | 64 | 128 | 1 | 2 | 8222 |
| P8 | serving | 48 | 64 | 128 | 1 | 1 | 6166 |
| P9 | serving | 48 | 64 | 128 | 1 | 1 | 12535 |
| P10 | serving | 48 | 64 | 256 | 1 | 1 | 3170 |
| P11 | serving | 48 | 64 | 256 | 1 | 1 | 10128 |
| P12 | serving | 80 | 64 | 64 | 1 | 1 | 3175 |
| P13 | serving | 80 | 64 | 64 | 1 | 1 | 1632 |
| P14 | serving | 24 | 64 | 128 | 1 | 1 | 6242 |
| P15 | serving | 24 | 64 | 128 | 1 | 1 | 10318 |
| P16 | serving | 64 | 128 | 256 | 4 | 1 | 5424 |
| P17 | serving | 64 | 128 | 256 | 4 | 1 | 15919 |
| P18 | serving | 96 | 80 | 128 | 4 | 1 | 6346 |
| P19 | serving | 96 | 80 | 128 | 4 | 1 | 16890 |
| P20 | long | 48 | 64 | 128 | 1 | 1 | 65500 |
| P21 | long | 48 | 64 | 256 | 1 | 1 | 131000 |
| P22 | long | 24 | 32 | 64 | 1 | 1 | 240000 |
| P23 | long | 64 | 128 | 256 | 4 | 1 | 100003 |
| P24 | real | 48 | 32 | 128 | 1 | 1 | 8123 |

P24 is a layer captured from a real model on real text; its tensors ship in
`/workspace/benchmark/public_layer/`.

**Hidden tests.**  Your score comes from hidden cases in the
same four families as the public tests, on different geometries, batches and
lengths.  Every hidden case stays inside the support domain above; exact shapes
and seeds are hidden.  Some cases are layers captured from real models.

In generated cases `A` is `-exp(u)` with `u` uniform on `[log 1, log 16]` and `dt`
is log-uniform on `[1e-3, 1e-1]`.  Captured layers carry the values a real model
produced.  A captured layer is compared on whole-tensor relative L2 alone, which
must stay within `1.3e-2`; the elementwise rule and the `6e-3` limit apply to
generated cases.

**Measurement.** Time is the larger of the synchronized host wall time and the
CUDA-event time of a single call on fresh tensors.  An untimed warm-up absorbs
compilation, but compiling counts toward the evaluation's 4500 s limit, so a kernel that takes
more than about 30 s to compile for each distinct shape fails.  For every measured trial:

```text
effective TFLOPS = 4 * batch * L * H * P * N / seconds / 1e12
```

A case is the median over its trials; a real layer is timed once.  Cases are
geometrically averaged within each family, and the score is the equal-weight
geometric mean of the four family scores.  Every measured trial draws fresh
inputs and is checked for complete, finite writes and unchanged inputs; any
call, including the warm-up, may be compared against its own reference.

Hard gates: source admission, a supported GPU, the edge
cases, numerical correctness on every compared call, complete writes with no NaN,
unmodified inputs, a per-case warm-up wall under 300 s, transient device memory
under four times the case's own input plus output bytes and never above 24 GiB,
and the TileLang-only operator record described below.

## Allowed implementation surface

Submitted compute must be TileLang.  Use `tilelang.jit`, `tilelang.language`,
`tilelang.layout`, and `tilelang.PassConfigKey`.

Torch is only for allocating your own buffers and describing them.  The torch
names a submission may use are `torch.empty`, `torch.empty_like`,
`torch.empty_strided`, `torch.zeros`, `torch.zeros_like`, `torch.full`,
`torch.full_like`, dtype constants, `torch.device`, `torch.Size`,
`torch.Tensor`, `torch.no_grad`, `torch.inference_mode`, and the
`torch.cuda` queries `current_stream`, `current_device`, `device_count`,
`get_device_capability`, `get_device_properties`, `is_available`, and
`synchronize`.  Tensor views (`view`, `reshape` of a contiguous tensor,
`permute`, `transpose`, slicing, `unsqueeze`, `squeeze`, `expand`) are fine.

**TileLang-only operator record.**  Every evaluation records the torch operators
each untimed call dispatches.  Anything other than an allocation or a view — a GEMM, an
elementwise or scan operator, a copy or dtype conversion, `.item()` — fails
the `tilelang_only_ok` gate.  TileLang kernels launch through TileLang's own
adapter and are not affected.  The public runner prints the operators it saw.

Tiling, block and warp shapes, thread counts, `T.Pipelined` depth,
`T.annotate_layout` and swizzled layouts, `T.use_swizzle`, warp specialisation,
register budget, atomics, persistent schedules, how many kernels you launch,
which intermediates you materialise, and the internal chunk size are all yours.
`T.alloc_tmem` is a storage declaration and is allowed.

Not allowed, and rejected by source admission before anything runs:

- `torch.matmul`, `mm`, `bmm`, `addmm`, `einsum`, `linear`, the `@` operator,
  `scaled_dot_product_attention`, or any other existing GEMM/attention/SSM
  kernel, including `mamba_ssm`, `fla`, Triton, FlashAttention, and FlashInfer;
- any torch name outside the list above, including `torch.linalg`,
  `torch.nn`, `torch.utils`, and every `torch._*` name;
- every route out of the tile DSL: `T.import_source`, `T.CUDASourceCodeKernel`,
  `T.call_extern`, `T.call_packed`, `T.call_kernel`, `T.call_intrin`,
  `prelude=`, `compile_flags=`, `execution_backend=`, `target=`,
  the `tl.device_compile_flags` and `tirx.add_lower_pass` pass keys,
  `register_*_postproc`, `JITKernel.from_database`, `tilelang.contrib.*`,
  and the bundled `tvm`;
- raw hardware intrinsics — anything named `T.ptx_*`, `T.mma_*`,
  `T.tcgen05_*`.  `T.gemm` and `T.copy` already lower to the target's
  tensor-core and bulk-copy paths; the task is the tile level, so reaching past
  it is out of scope;
- double-underscore attribute access (`x.__matmul__`, `T.__dict__`) and
  dynamic attribute access (`getattr`, `operator.attrgetter`,
  `operator.methodcaller`, or passing such builtins around as values);
- `tilelang.autotune` **inside a submission**.  Search all you like in your own
  scratch scripts during work; the submitted file must resolve to a bounded set
  of kernels, and the warm-up wall cap is enforced;
- `ctypes`, `cffi`, `torch.utils.cpp_extension`, `torch.ops`, `subprocess`,
  `eval`/`exec`/`type`, environment mutation, private CUDA streams,
  importing the runtime or runner, and specialising on seeds, values,
  evaluation order, or benchmark presence.

Imports are allowlisted.  The permitted roots are `tilelang`, `torch`, and a
small standard-library set (`math`, `functools`, `itertools`, `dataclasses`,
`typing`, `collections`, `enum`, `abc`, `operator`, `numbers`, `contextlib`,
`warnings`).

## The seed

`kernel_v001` is a correct but unoptimised first draft.  Read it.

## Useful commands

```bash
./tools/run_one.sh --version 1
./tools/run_public_tests.sh --quick --candidate-version 1
./tools/run_public_tests.sh --candidate-version 1
./tools/run_public_tests.sh --family serving --candidate-version 1
./tools/run_public_tests.sh --case P21 --perf-only --candidate-version 1
```

`run_one` measures P1, quick measures P1 and P20, and the full suite measures all
24 public cases with three trials each.  `--case` and `--family` select cases, and
`--perf-only` times them without the reference comparison.  Only the full suite
runs the 21 public edge fixtures first, which check correctness over the support
domain and are never timed.  Compiling all public shapes takes about ten minutes
the first time and seconds after that, because the Work image keeps a TileLang
cache.

Put versions at `candidates/kernel_vNNN/submission.py`, keep three-digit
version directories, and write the selected integer to `FINAL_VERSION`.  Keep
concise measurements in `notes.md`.

## Submission evaluation policy

Every `sforge-submit` runs the evaluation on hidden test cases. `sforge-submit --details` returns the gate states, the score and the four family scores. Hidden cases and per-case results stay on the Judge. Your final result is the best valid score you reach within the time budget, so keep `FINAL_VERSION` on your strongest candidate.

## Submitting

Wait for every public test and compiler process to finish, then run
`sforge-submit --details`.  A Judge run takes about half an hour of your budget.

Before finishing, point `FINAL_VERSION` at a clean candidate and run the full
public suite once.
