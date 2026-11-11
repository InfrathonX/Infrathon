# BF16 sparse-MLA prefill

Work in `/app/workdir/sparse_mla_prefill_lab` on the NVIDIA GPU allocated to you. The starter `candidates/kernel_v001/submission.py` is exactly empty. No attention kernel or reference implementation is supplied. Write the complete gather, score, softmax, and value accumulation in self-contained Triton.

The selected `submission.py` must define exactly:

```python
def sparse_mla_prefill(
    q, kv, indices, sm_scale, attn_sink, topk_length, out, max_logits, lse
) -> None:
    ...
```

Launch on the current CUDA stream, write every element of all three outputs, return `None`, write nothing outside `out`, `max_logits`, `lse` and scratch tensors the call allocated itself, and leave every input bit-identical.

## Tensor contract

`h_kv = 1` and `d_v = 512` always. `d_qk` is 576 (a 512-channel latent plus a 64-channel RoPE tail) or 512, and `h_q` is any multiple of 16 from 16 to 128: a whole model's query heads, or the heads left on one tensor-parallel rank.

All tensors are contiguous and live on the current device:

- `q`: `[s_q, h_q, d_qk]`, `torch.bfloat16`;
- `kv`: `[s_kv, 1, d_qk]`, `torch.bfloat16`;
- `indices`: `[s_q, 1, topk]`, `torch.int32`;
- `sm_scale`: Python `float`, any positive finite value (not only 0.5, not
  only a power of two);
- `attn_sink`: `[h_q]`, FP32, or `None`; values may be `+inf` or `-inf`;
- `topk_length`: `[s_q]`, `torch.int32` with values in `[0, topk]`, or `None`;
- `out`: `[s_q, h_q, 512]`, `torch.bfloat16`, NaN-poisoned before each call;
- `max_logits`: `[s_q, h_q]`, FP32, NaN-poisoned before each call;
- `lse`: `[s_q, h_q]`, FP32, NaN-poisoned before each call.

When `d_qk` is 576, the first 512 channels are the value channels and the trailing 64 are the RoPE tail; only the first 512 channels of a gathered key are its value vector.

### Support domain

Every call satisfies all of:

- **`d_qk` is 512 or 576 and `h_q` is a multiple of 16 from 16 to 128**, with `h_kv = 1` and `d_v = 512`;
- **`topk` is a multiple of 128 from 128 to 2048**;
- **`s_q` is any integer from 1 to 70000**, with `s_q * h_q * d_qk <= 2**32` and `s_q * topk <= 2**25`;
- **`s_kv` is any integer from 1 to 1048576**, including `s_kv < topk` (the row then holds invalid padding);
- any index value an `int32` can hold, valid or not, and any `topk_length` in `[0, topk]`.

### Implementation notes for the support domain

- Put `s_q` on grid axis 0. Triton grid axes 1 and 2 stop at 65535, and `s_q` can reach 70000.
- Compute row offsets with 64-bit integers: `q` and `out` can hold more than `2**31` elements (`s_q = 70000` at `d_qk = 512`, `h_q = 64` is 2.29e9 elements), and a key offset reaches `2**20 * 576` elements.
- Mask the `s_q` tail of every block. Triton compiles an integer argument equal to 1 into a separate specialization, so `s_q = 1` is its own compiled kernel and must be correct too; odd `s_q` and odd `s_kv` are in scope.

## Math contract

For query row `i`, slot `j` is **invalid** when `j >= topk_length[i]` (if `topk_length` is given) or `indices[i,0,j] < 0` or `indices[i,0,j] >= s_kv`. The length test is applied first; an index at an expired slot is ignored whatever its value. Otherwise slot `j` attends key `k = indices[i,0,j]`.

```text
P[i,h,j] = sm_scale * sum_c FP32(q[i,h,c]) * FP32(kv[k,0,c])      valid slot
P[i,h,j] = -inf                                                   invalid slot

max_logits[i,h] = max_j P[i,h,j]
raw_lse[i,h]    = log sum_j exp(P[i,h,j])

norm[i,h] = raw_lse[i,h]                                          no sink
norm[i,h] = log(exp(raw_lse[i,h]) + exp(attn_sink[h]))            with sink
norm[i,h] = +inf   whenever the value above is -inf

out[i,h,:] = sum_j exp(P[i,h,j] - norm[i,h]) * FP32(kv[k,0,:512])
lse[i,h]   = raw_lse[i,h], or +inf when raw_lse[i,h] is -inf
```

Three points are easy to get wrong:

- the returned `max_logits` and `lse` are the **raw** statistics. The attention sink changes only the normaliser used for `out`, never `lse` and never `max_logits`;
- a row with no valid slot returns `out = 0`, `max_logits = -inf`, `lse = +inf`. These infinities must appear at exactly the right positions;
- a `+inf` sink drives that head's `out` to zero and changes neither `max_logits` nor `lse`; a `-inf` sink has no effect.

Invalid slots may be interleaved with valid ones anywhere in a row, not just in a tail, and a valid key index may repeat inside one row. Neither may end a scan early, and duplicated keys are separate attention positions that must both contribute.

### Precision rule

Multiply operands are BF16 or FP16; accumulation, the softmax statistics (running maximum and denominator), `max_logits` and `lse` are FP32. FP8, INT8 and FP4 in any step are a shortcut even when a case happens to stay within tolerance; source admission rejects 8-bit and 4-bit floating-point types and scaled dots. Rounding `P` to BF16 or FP16 for the `PV` product with FP32 accumulation is allowed.

## Correctness and memory safety

All three outputs are compared against an independent chunked FP32 reference with these limits:

```text
out:        every finite element |a-r| < 8e-4 or |a-r|/(|r|+1e-6) < 3.01/128; |cosine diff| <= 7e-6
max_logits: 1e-6*max(1, sm_scale/0.5) or 2.01/65536;   |cosine diff| <= 1e-7
lse:        1e-6*max(1, sm_scale/0.5) or 2.01/65536;   |cosine diff| <= 1e-7
```

The positions of `+inf`, `-inf` and `NaN` must match the reference exactly; a `NaN` anywhere (an unwritten element included) fails.

Every call is also checked for memory safety. Writes or reads outside a tensor's extent and any change to an input fail `memory_safety_ok`; a fatal CUDA error also stops the run at once. Do not name a function so that it starts with `_Z` or contains `gemm`, `xmma`, `nvjet`, `cutlass` or `cublas`; source admission rejects those names.

On platforms where compute-sanitizer cannot attach to a CUDA process, the public `--sanitize` suite stops with `MEMCHECK UNAVAILABLE` (exit status 2) instead of hanging. Out-of-bounds reads and writes remain contract violations there.

## Workloads, feedback, and measurement

The calls are prefill attention as a serving engine issues it. Most calls look like this:

- **Causal indices.** Row `i` of a call is at position `s_kv - s_q + i` of its context and selects keys at or before it. A row with fewer such keys than `topk` holds them first and `-1` after, and `topk_length` counts them.
- **Index locality.** A learned indexer's keys are rarely uniform: many may sit just before the row, at the first keys of the context, or in blocks that neighbouring rows share, and a row's keys may arrive sorted or in any order.
- **Slots past `topk_length` are not cleared.** They may hold a stale index that looks valid, so apply `topk_length` before reading a slot.
- **Several requests in one call.** One call can cover a run of requests packed into one KV workspace; each row selects keys only from its own request.
- **Shards.** A decode-context-parallel shard holds only a fraction of a row's keys, so its `topk_length` is short, and it can differ widely from row to row.
- **Streams.** Consecutive calls of one prefill or of a stream of requests each have their own `s_q` and `s_kv`.
- **Inputs.** `sm_scale` is usually the model's `qk_head_dim ** -0.5` but may be any positive value, and `attn_sink` may be finite, `+-inf` or absent.

**Public tests.** `/workspace/benchmark/public_cases.json` holds 108 public cases in eight groups:

| group | what it runs |
| --- | --- |
| `basic` | six single calls at `s_q` 1536 or 2048: inputs A (every index valid and distinct, `+-inf` sinks, no `topk_length`) or B (invalid indices mixed in, `topk_length`, finite sinks) |
| `sweep` | causal single calls, `s_q` from 1 to about 20000 and `s_kv` up to about 225000, aligned and unaligned |
| `pattern` | causal single calls whose indices are local, sink-and-window, block-shared or sorted |
| `heads` | causal single calls at fewer heads than the whole model |
| `length` | shards whose `topk_length` is about `topk / 2` or `topk / 4`, even or ragged across rows |
| `multi` | one call over 2 to 16 requests |
| `replay` | one context's chunked prefill as a stream, up to 262144 keys |
| `trace` | the prefill calls of a synthetic request stream |

**Hidden tests.** Your score comes from hidden cases on the Judge. They cover every kind of call listed above, reach the limits of the support domain, and use `topk` values, tensor-parallel head counts, sizes, chunkings, request traces and seeds that differ from the public cases. None of them is disclosed. Keep every path correct across the whole support domain, and write your own benchmarks for the regimes the public cases cover only lightly.

**Measurement.** Time is the larger of the synchronized host wall time and the CUDA-event time of a single call on fresh tensors.

- **Single calls.** An untimed warm-up at the same shape comes first, then the Judge takes the median of 3 trials.
- **Streams.** 16 untimed warm-up calls at shapes close to the stream's own come first; every later call is timed, compilation included. A kernel that compiles a new specialization for every `(s_q, s_kv)` therefore pays for it, while the few specializations a size-agnostic kernel needs are compiled during the warm-up.
- **Stream time limit.** A stream stops after 8 timed seconds and scores the work it finished.

```text
positions        = s_q * topk, or sum(topk_length) when topk_length is given
effective TFLOPS = 2 * positions * h_q * (d_qk + 512) / seconds / 1e12
stream TFLOPS    = total effective FLOPs of its timed calls / their total seconds
```

## Metric

Every hidden case gives one effective TFLOPS value. The cases fall into four families, one per deployed model geometry, and inside a family into kinds of workload. Each kind is the geometric mean of its cases; the kinds form two halves of equal weight, serving workloads (reference shapes, request streams, chunked prefills, multi-request calls, indexer-shaped indices) and size sweeps (single calls over sizes, head counts and short rows), each the geometric mean of its kinds:

```text
family       = sqrt(serving half * sweep half)
score_tflops = geomean over the four families
```

`sforge-submit --details` reports `score_tflops` and the four family scores (`v32_tflops`, `glm5_tflops`, `v4_flash_tflops`, `v4_pro_tflops`).

The raw TFLOPS score is uncapped and maximize-only; there is no minimum. Correctness, edge-case correctness, complete writes, memory safety, harness integrity, source admission, suite identity, and a supported GPU are hard gates.

Useful commands:

```bash
./tools/run_one.sh --version 1                                 # one public case
./tools/run_public_tests.sh --quick --candidate-version 1       # one case per group
./tools/run_public_tests.sh --candidate-version 1               # all 108 public cases
./tools/run_public_tests.sh --case public-040-sweep --perf-only --candidate-version 1
./tools/run_public_tests.sh --edge --candidate-version 1        # public edge cases
./tools/run_public_tests.sh --sanitize --candidate-version 1    # memcheck subset
```

`run_one.sh` runs `public-002-basic`, quick runs the first case of every group, and the full suite runs all 108 public cases with three measured trials per single call (one and quick use one). `--case ID` (repeatable) runs only the named cases; `--perf-only` skips the reference comparison and only checks that every output element is written. `--edge` runs the 73 public edge cases in `/workspace/benchmark/public_edge_cases.json` natively with guard bands and input checks and reports every case; each is untimed and called twice. `--sanitize` runs the seven `sanitize` ones under memcheck. Run `--sanitize` before `--edge` whenever a kernel is new: under memcheck an out-of-bounds access is reported instead of reaching the GPU, where a wild address can hang the driver. Public tools never run or reveal hidden cases.

## Version workflow

Keep experiments at `candidates/kernel_vNNN/submission.py` using three-digit directories. Only the version named by the canonical integer in `FINAL_VERSION` is evaluated; older versions may remain for comparison. Keep concise decisions and public measurements in `notes.md`. Source admission, in the public tools and on the Judge, rejects a lab that holds more than 64 `kernel_vNNN` directories, a `submission.py` over 512 KiB, a `notes.md` over 64 KiB, or more than 4 MiB in all, so retire old versions and keep the notes short. Before finishing, select the strongest clean candidate and run the full public suite, the public edge suite and the sanitizer suite once.

## Submission evaluation policy

Every `sforge-submit` runs the evaluation on the complete hidden suite. `sforge-submit --details` returns the gate states, the score and the four family scores. Hidden cases and per-case results stay on the Judge. Your final result is the best valid score you reach within the time budget, so keep `FINAL_VERSION` on your strongest candidate.

## Submitting

Wait for every public test and compiler process to finish, then run `sforge-submit --details`. A Judge run takes about ten minutes of your budget, and longer when kernels are slow or compile for every shape, so submit when a candidate is worth measuring rather than after every change. Run the public sanitizer and edge suites first.

## Allowed implementation surface

The selected source may import only `torch`, `triton` and their submodules (not `torch._*`, `torch.backends`, `torch.utils`, `torch.cuda.memory`, `torch.cuda.graphs`, `torch.ops`, `triton.backends`, `triton.compiler`, `triton.tools`, `triton.knobs` and the other internal namespaces), plus `math`, `functools`, `itertools`, `typing`, `dataclasses`, `collections`, `enum` and `__future__`. Use `torch` for tensor metadata, scratch allocation and current-stream dispatch, and `triton` / `triton.language` (Gluon included) for kernels. Triton loads, stores, `tl.dot`, reductions, atomics, persistent schedules, autotuning and compiler meta-parameters are allowed.

Source admission rejects: any other import; assigning to an imported module, class or function or to anything reached through one; `getattr`, `setattr`, `delattr`, `globals`, `locals`, `vars`, `eval`, `exec`, `compile`, `open`, `__import__` and dunder names; `torch.compile`, CUDA graphs, RNG seeding, `torch.set_*`, `torch.load`/`save`; process, dynamic-loading and private-stream APIs; existing GEMM routes (`torch.matmul`, `mm`, `bmm`, `addmm`, `einsum`, `linear`, `_scaled_mm`, the `@` operator, ...) and existing attention kernels (`scaled_dot_product_attention`, ...); 8-bit and 4-bit floating-point types; `functools.wraps`; and function names that the sanitizer would skip (see above). Do not import, read or inspect the runtime, public runner, public edge manifest, worker or Judge: the visible benchmark is trusted test infrastructure, not an implementation library. External implementation material is not provided; do not fetch or copy solution material into the submission.

### Integrity rules

- **Shape-keyed paths must be correct everywhere they can be selected.** Choosing tile sizes, kernels, schedules and fast paths from `s_q`, `s_kv`, `topk`, `h_q`, `d_qk` or whether `attn_sink` / `topk_length` are present is welcome, as is an aligned fast path beside a masked general path. A path that is correct only for particular shapes or inputs -- for example one that skips invalid-index masking, hard-codes `sm_scale`, assumes no `topk_length`, finite sinks or no duplicated keys -- is a contract violation even when no test catches it.
- **Each call computes only its own outputs.** Do not cache or replay outputs across calls, compute during one call (an untimed one included) anything a later call reuses, inspect allocator state or device memory that was not passed to the call, or reconstruct seeds or future inputs. Do not specialize for seeds, values, call order, untimed versus measured calls, or the presence of the benchmark, guard bands or compute-sanitizer.
- **All work belongs to the call.** Launch every kernel on the current CUDA stream before returning; no background threads, processes, other streams, or work deferred past the call's return.
- **Leave the environment alone.** Do not assign to, wrap or replace any attribute of an imported module, class or function (for example `time.perf_counter` or `torch.cuda.Event.elapsed_time`), and do not change global numeric settings (TF32 flags, matmul precision, the default dtype). A run that does so fails `harness_integrity_ok`.
- **Write only your outputs.** Nothing outside `out`, `max_logits`, `lse` and your own scratch may be written, and no input may change.
- **Code paths compiled only for other GPUs do not count.** The Judge runs on the same kind of GPU as this run; a kernel or path that only compiles or only runs on another architecture (for example Blackwell-only instructions) must not be the path that produces your results here.
