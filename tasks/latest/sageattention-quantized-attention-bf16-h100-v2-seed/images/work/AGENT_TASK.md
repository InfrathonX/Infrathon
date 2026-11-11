# Quantized attention BF16 optimization

Optimize the seeded self-contained Triton implementation while preserving:

```python
def quantized_attention_bf16(q, k, v, is_causal, softmax_scale, out) -> None:
    ...
```

`q` and `out` are contiguous `[B,L,H_q,D]` BF16 tensors; `k` and `v` are
contiguous `[B,L,H_kv,D]` BF16 tensors. `D` is 64 or 128, `H_q % H_kv == 0`,
query head `h` uses KV head `h // (H_q // H_kv)`, and `softmax_scale` is the
Python float `1/sqrt(D)`. The call must overwrite every NaN-poisoned output
element on the current stream, return `None`, and leave all inputs byte-exact.
For causal cases use the ordinary lower triangle; otherwise every query sees
every key.

The FP32 semantics are `softmax(softmax_scale * q @ k.T) @ v`. The golden
upcasts the BF16 inputs, disables TF32, and processes 1024 query rows at a time.
Accuracy is differential against a reference with these numerics:
Q is symmetric INT8 fake-quantized per 128-token block; K first has its
per-(batch, KV-head, channel) token mean subtracted, then is symmetric INT8
fake-quantized per 64-token block; scales are `absmax/127 + 1e-7`, rounding is
half away from zero, and softmax/PV are FP32.

| gate | required result |
| --- | --- |
| finite/write | every output element finite |
| immutable | all inputs byte-identical |
| global RelL1 | `<= max(3 * reference, 0.025)` |
| global CosSim | `>= 0.998` |
| worst-head RelL1 | `<= max(3 * reference, 0.04)` |
| row relative-L2 p99.9 | `<= max(2 * reference, 0.02)` |
| row relative-L2 maximum | `<= max(2.5 * reference, 0.05)` |

Row error is `||Orow-O'row||2 / max(||Orow||2, 1e-6)`.

The contract allows `B` from 1 to 4, up to 64 query heads, and `L` up to 262,144,
with the inputs of one call at most 6 GiB: `2*B*L*(H_q + 2*H_kv)*D <= 6 * 2**30`
bytes. Every output row must be right: the golden may compare any of them.

### Real-model activation layers

Real-model attention activations are part of the workload. Each real layer is
measured like any other case (below), on copies of the captured tensors, and must
pass the gates above, except that the row relative-L2 p99.9 limit is
`max(2.5 * reference, 0.02)`. Real layers are scored like any other case.

## Workloads, feedback, and score

Scored workloads come in four families:

- **video:** non-causal self attention of video diffusion transformers.
- **image:** non-causal joint attention of image diffusion transformers.
- **llm:** causal prefill of language models with grouped-query attention.
- **real:** attention layers captured from real models.

**Public tests.** The public suite covers the four families with its own
geometries, lengths and models; it is for development and profiling. A good public
number does not guarantee a good Judge score; write your own benchmarks for
performance work.

**Hidden tests.** Your score comes from hidden test cases on the Judge. They use
other geometries, lengths, models and seeds than the public tests, they are larger,
and there are many more of them. Keep your kernel correct for any input the
contract above allows, and make optimizations general rather than fitting
particular shapes. Approximate or sparse strategies must pass every gate on the
hidden cases.

**Measurement.** Time is the larger of the synchronized host wall time and the
CUDA-event time of a single call on fresh tensors. Each case runs one untimed
warm-up call, then three timed calls, each preceded by an untimed call on other
data of the same shape; the case value is the median. Compilation and autotuning
happen in the warm-up and are not timed, but a case's warm-up may take at most
60 seconds and the whole evaluation has a fixed time limit: a kernel that
recompiles or re-tunes for every distinct length can exhaust it.

**Metric.** Effective work is `4*B*H_q*D*L^2` for noncausal cases and
`4*B*H_q*D*L*(L+1)/2` for causal cases. A case's value is its effective TFLOPS.
The video, image and llm families each consist of two equally weighted groups; the
family score is the geometric mean of the two groups' case geometric means. The
real family is the geometric mean over source models of each model's per-layer
geometric mean. The primary score is the geometric mean of the four family scores. It is uncapped and
maximize-only.

A Judge run takes about 15 minutes for a kernel as fast as the seed, longer for a
slower or recompiling kernel, and `sforge-submit` waits for it, so iterate on the
public tests and your own benchmarks first.

Public development commands:

```bash
./tools/run_one.sh --version 1
./tools/run_public_tests.sh --quick --candidate-version 1
./tools/run_public_tests.sh --candidate-version 1
./tools/run_public_tests.sh --case public-llm-h24-d128-l190001 --perf-only --candidate-version 1
```

`run_one.sh` runs the edge cases and one public case. Quick adds a second case, a
long timing-only case and the two public real-model layers; full runs every public case
(three timed trials each) and the two real-model layers. `--case ID` runs one case, `--perf-only` skips the
golden comparison and only times. Public results show geometry, warm-up time,
candidate and reference metrics, realized limits, medians, and family and primary
aggregates. Public tools never run or reveal hidden cases.

## Allowed implementation surface

The selected source may import only Python `math`/`typing`, `torch`, `triton`,
and `triton.language`. Use Torch only for tensor metadata and safe same-device
scratch allocation. Use one or more reachable `@triton.jit` kernels for every
tensor data path. Triton loads, stores, `tl.dot`, reductions, atomics,
autotuning, and compiler meta-parameters are allowed.

Inside Triton kernels, ordinary Triton control flow and FP16, INT8, FP8,
BF16, and FP32 compute are allowed. Host code is deliberately restricted:
assignment, assert, bare calls, if/else, pass and return only; no host loops,
try/with, comprehensions, lambda, dynamic imports, extensions, private CUDA
streams, filesystem/process/network access, dunder introspection, or module
mutation. Torch host use is metadata plus `empty`, `empty_like`,
`empty_strided`, `full`, `full_like`, `zeros`, and `zeros_like`; permitted
scratch dtypes are INT8, FP8 e4m3, FP16, BF16, and FP32. Host tensor math,
including `k.mean(...)`, attention, GEMM, softmax, input mutation, or copying,
is forbidden. Every host helper must be reachable. Decorator callbacks,
unsafe autotune hooks, dead host code, imported-module mutation, string-dunder
routes, and callable-grid bypasses fail admission.

Keep versioned candidates at `candidates/kernel_vNNN/submission.py`, notes in
`notes.md`, and the selected canonical integer in `FINAL_VERSION`. Before
finishing, select the strongest clean candidate and run the full public suite once.

## Submission evaluation policy

Every `sforge-submit` runs the evaluation on hidden test cases. `sforge-submit --details` returns the gate states, the score and the four family scores. Hidden cases and per-case results stay on the Judge. Your final result is the best valid score you reach within the time budget, so keep `FINAL_VERSION` on your strongest candidate.

## Submitting

Wait for every public test and compiler process to finish, then run
`sforge-submit --details`. Each Judge run uses part of your budget, so run the
public tests first.
