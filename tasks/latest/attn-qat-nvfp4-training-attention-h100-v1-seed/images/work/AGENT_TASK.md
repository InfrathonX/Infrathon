# NVFP4 training attention (forward + backward)

## Task

Work in `/app/workdir/attn_qat_lab` on the NVIDIA GPU allocated to you.

You are optimizing one training step of a 4-bit **quantization-aware-training** attention
layer: a forward call that fake-quantizes `q`, `k`, `v` and the attention probabilities to
NVFP4, and a backward call that recomputes those probabilities in the same low precision.
Both calls are timed. The same operator serves two kinds of training: full attention in video
diffusion transformers, and causal grouped-query attention in language models. The seed in
`candidates/kernel_v001/submission.py` is a correct, self-contained Triton implementation; your
job is to make it faster without changing what it computes.

## 1. The operator

`q, o, do, dq` are contiguous **bf16 `[B, H, S_q, D]`** and `k, v, dk, dv` are contiguous
**bf16 `[B, H_kv, S_kv, D]`**, with `D` in {64, 128, 256}, `H % H_kv == 0` and
`softmax_scale = D ** -0.5`.

* **Grouped-query attention.** Query head `h` reads key/value head `h // (H / H_kv)`;
  `H_kv == H` is ordinary multi-head attention. `dk` and `dv` sum over the `H / H_kv` query
  heads of each group.
* **Causal masking.** With `causal=True`, `S_q == S_kv` and query row `i` attends to keys
  `0..i`. A masked score is `-inf`: its `P~` is exactly 0, it adds nothing to `l`, and it
  enters its 16-group of `P` as a zero. With `causal=False` every key is visible, and `S_q`
  may differ from `S_kv` (cross attention).

### 1.1 Fake quantization `FQ`

`FQ` acts on groups of **16 consecutive elements along one axis**, aligned at 0:

```text
a  = max |x|            over the group
sd = e4m3(a / 6)        round-to-nearest FP8 e4m3, saturating at 448
x_hat = e2m1_rtne(x / sd)   FP4 e2m1 grid {0, +-0.5, +-1, +-1.5, +-2, +-3, +-4, +-6},
                            round-to-nearest-EVEN, saturating
FQ(x) = bf16(x_hat * sd)
```

There is no global scale factor, no two-level P quantization and no Q/K smoothing. Three
consequences you must reproduce exactly:

* **Ties round to even.** `0.25 -> 0`, `0.75 -> 1`, `1.25 -> 1`, `1.75 -> 2`, `2.5 -> 2`,
  `3.5 -> 4`, `5 -> 4` when `sd = 1`.
* **A group of zeros round-trips through a NaN.** `a = 0` gives `sd = 0`, `1 / sd = inf`,
  `x / sd = NaN`, e2m1 saturates the NaN to 6, and `6 * 0 = 0`. Reordering that computation
  (special-casing the zero group, or dividing after the encode) produces NaN output.
* **e4m3's smallest magnitude is `2^-9`,** so any group whose maximum is below
  `6 * 2^-10 = 5.86e-3` gets block scale zero and is flushed to zero entirely.

### 1.2 Forward

```text
Qf = FQ_D(q)   Kf = FQ_D(k)   Vf = FQ_D(v)      groups of 16 along D, per token
S  = Qf Kf^T * softmax_scale * log2(e)          bf16 operands, fp32 accumulate
m  = the FINAL row maximum of S
P~ = exp2(S - m)                in (0, 1]       fp32
Pq = FQ_KV(P~)                                  groups of 16 along S_kv
l  = sum_j bf16(P~_ij)
o  = (sum_j Pq_ij Vf_j) / l                     the returned output
O' = (sum_j P~_ij Vf_j) / l                     the auxiliary high-precision output
M  = m + log2(l)
```

`m` is the **final** row maximum, not a running one, and `P~` is **unnormalized** (its row
maximum is exactly 1). Both facts are load-bearing:

* Quantizing the *normalized* probabilities would be wrong. At `S >= 10^3` a typical
  probability is far below the `5.86e-3` block-scale floor of section 1.1, so most of the
  row would be flushed to zero and `o` would be destroyed. Keeping `P~` unnormalized puts
  the per-row maximum at exactly 1 and divides by `l` in the epilogue instead.
* Quantizing against a *running* maximum would make `Pq` depend on the order in which you
  visit key blocks, so two honest kernels with different tile shapes would disagree by more
  than a wrong kernel does. Against the final maximum, every tiling produces the same `Pq`.
  The row sum `l` can still differ in its last bits between a kernel that rescales a running
  sum and one that sums against the final maximum; the gates allow for that, and the backward
  quantizes against `M`, so a group of `P` near the block-scale floor may land on either side
  of it.

The identity `P_i^T dP_i = dO_i^T O_i` assumes the forward `O` was computed from the same `P`
the backward uses; once the forward quantizes `P`, that is false, so the row statistic
`Delta` must come from an output built with the *unquantized* probabilities.

### 1.3 Backward

```text
Delta = rowsum(dO * bf16(O'))
P  = exp2(S - M)                                the true normalized probabilities
Pq = FQ_KV(P)                                   deterministic: M is the final statistic
dV = Pq^T dO
dP = dO Vf^T
dS = P * (dP - Delta)
dK = softmax_scale * dS^T Qf
dQ = softmax_scale * dS Kf
```

`dS` uses the **unquantized** `P` (the straight-through estimator), while `dV` uses the
quantized `Pq`. The backward quantizes the *normalized* `P`, which is what the golden
reference implements.

## 2. ABI

```python
def attn_qat_fwd(q, k, v, softmax_scale, causal, o) -> tuple[torch.Tensor, ...]:
    """Overwrite `o` in full; return the `saved` bundle the backward will read."""

def attn_qat_bwd(q, k, v, softmax_scale, causal, o, saved, do, dq, dk, dv) -> None:
    """Overwrite `dq`, `dk`, `dv` in full."""
```

* `causal` is a Python `bool`.
* `saved` is any tuple of tensors on the same device as the inputs. New storages reachable
  from it may total at most
  `bytes(q) + bytes(k) + bytes(v) + bytes(o) + 4 * B * H * S_q`
  -- exactly `Qf, Kf, Vf, O'` in bf16 plus the fp32 row statistic `M`. Storages that
  already belong to `q, k, v, o` are free. No callables in the bundle.
* The backward also receives `q, k, v`, so you may re-quantize instead of saving.
* All four outputs are **NaN-poisoned** before each call and must be written in full and be
  finite, on the current stream.
* `q, k, v, do` must be byte-identical after both calls.
* `o` is byte-identical between the end of the forward and the end of the backward. The
  test is byte identity, not absence of a write: the backward may re-derive `o` and store
  the same bytes back into it -- recomputing the forward instead of saving `O'` is a legal
  strategy -- but any difference fails the case, so the forward's work cannot be moved into
  the backward's timer.
* No autograd, no `torch.autograd.Function`: the harness drives both calls itself.

## 3. Score

```text
pairs     = S_q * S_kv,  or S * (S + 1) / 2 when causal
base      = B * H * pairs * D
work_fwd  = 8 * base        two QK-type passes, two PV-type products
work_bwd  = 10 * base       QK recompute, dV, dP, dK, dQ
work      = 2 * work_fwd + work_bwd = 26 * base
time      = 2 * t_fwd + t_bwd
case TFLOPS = work / median(time over the trials)
group   = geometric mean over a group's cases
family  = geometric mean over the family's groups
domain  = geometric mean over the domain's families   (video, LLM)
primary = geometric mean over the two domains
```

Each trial measures one forward and one backward, each as `max(host wall, CUDA event)`, on
fresh tensors; an untimed warm-up absorbs compilation, but compiling counts toward the
evaluation's 3600 s limit. The forward is weighted twice because
training with full activation checkpointing executes it twice per step. The score is
**uncapped and maximize-only**; `fwd_tflops = 8 * base / t_fwd` and
`bwd_tflops = 10 * base / t_bwd` are reported for information.

## 4. Workloads

**Public tests.** `S` is `S_q x S_kv` for cross attention; `H` is `H/H_kv` for grouped-query
attention.

| cases | domain | family | B | H | D | causal | S |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| `p5`, `p1`, `p6`, `p7` | video | d128 | 1 | 4 | 128 | no | 2784, 9984, 26208, 57440 |
| `p2`, `p8`, `p9`, `p10` | video | d128 | 1 | 8 | 128 | no | 6120, 12384, 26208, 57440 |
| `p3`, `p11`, `p12`, `p13` | video | d128 | 1 | 24 | 128 | no | 2728, 9696, 18016, 26208 |
| `p4`, `p14`, `p15` | video | d64 | 1 | 16 | 64 | no | 9216, 20480, 40960 |
| `p16`, `p17` | video | d64 | 1 | 24 | 64 | no | 20480, 40960 |
| `p18`, `p19`, `p20` | video | cross | 1 | 16 | 64 | no | 9216, 20480, 40960 x 320 |
| `pr` | video | real | 1 | 24 | 128 | no | 27280 |
| `p21`, `p22`, `p23` | LLM | llm | 4, 2, 1 | 28/4 | 128 | yes | 3584, 7168, 14336 |
| `p24`, `p25` | LLM | llm | 2, 1 | 14/2 | 128 | yes | 7168, 28672 |
| `p26`, `p27` | LLM | llm_d256 | 4, 1 | 20/4 | 256 | yes | 3584, 14336 |
| `prl` | LLM | real_llm | 1 | 16/8 | 128 | yes | 7168 |

Case ids are `public-<name>`. Every trial of a synthetic case draws fresh `q, k, v, do` from the
generator in `/workspace/benchmark/attn_qat_runtime.py`. `pr` (a video diffusion model) and
`prl` (a language model on real text) are layers captured from real training steps, timed once.

**Hidden tests.** Your score comes from hidden test cases, not from these public tests. The
hidden cases have different shapes and can be much larger, but they always stay inside the
contract of sections 1 and 2. Make your kernel correct and fast across that whole contract,
not just on the public shapes: a gate failure on any hidden case scores 0.

## 5. Accuracy gates

Global relative L2 against a chunked fp32 golden that implements section 1 exactly (TF32
off, `Pq` against the final row statistic in both passes, bit-exact fake quantization).

| output | timed cases | real video layers | real LLM layers |
| --- | ---: | ---: | ---: |
| `o` | `<= 1.0e-2` | `<= 1.0e-2` | `<= 1.0e-2` |
| `dq` | `<= 8.0e-2` | `<= 1.2e-1` | `<= 1.5e-1` |
| `dk` | `<= 3.0e-2` | `<= 3.5e-1` | `<= 8.0e-2` |
| `dv` | `<= 1.2e-1` | `<= 1.0e-1` to `<= 3.0e-1`, by source model | `<= 8.0e-2` |

The public edge cases have fixed inputs, so each one carries its own limits, several of them
much tighter.

| edge case | `o` | `dq` | `dk` | `dv` |
| --- | ---: | ---: | ---: | ---: |
| e1 | 1.0e-02 | 1.5e-02 | 1.5e-02 | 2.0e-02 |
| e2 | 8.0e-03 | 3.0e-02 | 1.0e-02 | 8.0e-02 |
| e3 | 8.0e-03 | 8.0e-02 | 2.0e-02 | 3.0e-02 |
| e4 | 1.0e-02 | 3.0e-02 | 1.5e-02 | 8.0e-02 |
| e5 | 5.0e-03 | 1.5e-02 | 1.0e-02 | 2.0e-02 |
| e6 | 8.0e-03 | 2.0e-02 | 2.0e-02 | 5.0e-02 |
| e7 | 8.0e-03 | 5.0e-02 | 1.5e-02 | 2.0e-02 |
| e8 | 5.0e-03 | 2.0e-02 | 1.0e-02 | 2.0e-02 |
| e9 | 5.0e-03 | 1.5e-02 | 1.0e-02 | 2.0e-02 |
| e10 | 1.0e-02 | 2.0e-02 | 3.0e-02 | 5.0e-02 |
| e11 | 8.0e-03 | 5.0e-02 | 1.5e-02 | 3.0e-02 |
| e12 | 8.0e-03 | 1.5e-02 | 1.5e-02 | 3.0e-02 |
| e13 | 5.0e-03 | 1.5e-02 | 1.5e-02 | 2.0e-02 |
| e14 | 8.0e-03 | 1.5e-02 | 1.5e-02 | 3.0e-02 |
| e15 | 8.0e-03 | 2.0e-02 | 1.5e-02 | 3.0e-02 |

Where the golden output is exactly zero -- for example `dq` and `dv` when `k = 0` makes the
softmax uniform and `Kf` identically zero -- the gate is `max |candidate| <= 1e-4` instead. The worst-`(b, h)` relative L2 is reported but not gated. Every gate is evaluated
on every case, so a failure report names each gate that tripped.

Changing what is quantized, when, or against which statistic is a contract violation even
when the result stays inside these limits.

## 6. Public edge cases

Correctness only, no timing, run before any timed case. `H=8/2` means 8 query heads sharing 2
key/value heads.

| id | shape | what it checks |
| --- | --- | --- |
| e1 | `S=48, H=2, D=128` | a shape smaller than one tile |
| e2 | `S=1003, H=3, D=128` | an unaligned sequence length |
| e3 | `S_q=2048, S_kv=768, H=4, D=64` | cross attention |
| e4 | `S_q=1203, S_kv=517, B=2, H=4, D=128` | unaligned cross attention, `S_kv % 16 != 0` |
| e5 | `S=1536, H=3, D=128`, `k = 0`, zeroed `v` groups | uniform softmax and the zero-through-NaN path |
| e6 | `S=768, H=3, D=128`, e2m1 tie set | round-to-nearest-even |
| e7 | `S=6136, H=8, D=128` | a long unaligned sequence |
| e8 | `S=5120, H=8, D=128`, forced sink | one-hot softmax rows |
| e9 | `S=3072, H=3, D=128`, `do` at rms 1e-6 | gradient magnitude extremes |
| e10 | `S=160, H=24, D=128` | many heads on a short sequence |
| e11 | `S=1536, B=2, H=5, D=64` | the second head dimension |
| e12 | `S=1203, B=2, H=8/2, D=128`, causal | causal grouped-query attention at an unaligned length |
| e13 | `S=3072, H=6/1, D=128`, causal, forced sink | one-hot rows under the causal mask, one key/value head |
| e14 | `S=1280, B=2, H=4/2, D=256`, causal | the third head dimension |
| e15 | `S=901, H=3, D=256` | D = 256 without a mask, unaligned |

## 7. Allowed surface

The selected `candidates/kernel_vNNN/submission.py` is admitted by a static policy before it
is ever imported. It must:

* import only `math`, `typing`, `torch`, `triton`, `triton.language as tl`, and optionally
  `from triton.tools.tensor_descriptor import TensorDescriptor`;
* define both entry points with exactly the parameter tuples of section 2;
* route every tensor data path through a reachable `@triton.jit` kernel -- each entry point's
  own call graph must launch one;
* keep host Torch to metadata plus
  `empty / empty_like / empty_strided / full / full_like / zeros / zeros_like`.

Allowed inside kernels: FP16/BF16/FP8/INT8/FP32 `tl.dot`, `tl.make_tensor_descriptor` and
host-side `TensorDescriptor` (TMA), `tl.range(..., warp_specialize=...)`, atomics, persistent
schedules, and `num_warps / num_stages / num_ctas / maxnreg` at the launch site. Method forms
of Triton language operations (`x.reshape(...)`, `desc.load(...)`, `desc.store(...)`,
`x.to(...)`, `trans`, `permute`, `expand_dims`, `broadcast_to`, `atomic_add`, `gather`,
`scatter`) are admitted; other method calls inside a kernel are not.

`triton.set_allocator(fn)` -- needed for device-side TMA descriptors -- is admitted only when
`fn` is a module-level function of exactly three parameters whose entire body -- after an
optional docstring -- is
`return torch.empty(<first parameter>, dtype=torch.int8, device=...)`. Anything else in that
callback is a rejected callback route. Note that a `global` statement is not part of the host
subset, so install the allocator unconditionally rather than behind a module-level flag.

Rejected: `@triton.autotune` and `@triton.heuristics` are **not** permitted (both run
participant code at launch time, outside every static pass, and an evaluation-time sweep
would land inside a timed trial); `tl.inline_asm_elementwise` is not permitted (the target GPU has no
e2m1 conversion instruction to reach with it, and inline PTX is an unreviewable data path);
`triton.language.target_info` is not permitted; `torch.autograd` is not permitted; host
tensor math (`torch.softmax`, `torch.matmul`, `scaled_dot_product_attention`, `copy_` into
an output, ...), host tensor indexing, `torch.compile`, `torch.load`, private streams,
filesystem, process and network access, dunder introspection and module mutation are all
rejected. Whether a reduced-precision product survives the accuracy gates is a numerical
question, not an admission one.

## 8. Working here

```bash
./tools/run_one.sh --version 1
./tools/run_public_tests.sh --quick --candidate-version 1
./tools/run_public_tests.sh --candidate-version 1
./tools/run_public_tests.sh --case P10 --perf-only --candidate-version 1
```

`run_one.sh` runs the fifteen public edge cases plus `public-p1`; `--quick` runs them plus `p2`,
`p4`, `p24` and a timing-only `p10`; the full run covers all 29 cases. `--case`
times one case after the edge cases; `--perf-only` skips that case's golden (real layers are always checked).
Keep a known-good version while you experiment: create `candidates/kernel_v002/`, and leave
`FINAL_VERSION` on the strongest candidate that passes every gate. Only `candidates/`,
`FINAL_VERSION` and `notes.md` are submitted. Public tools never run or reveal hidden cases.

Run the complete public controller once on the selected version before you finish.

## Submission evaluation policy

Every `sforge-submit` runs the complete hidden suite. `sforge-submit --details` returns the gate states and the score. Hidden cases and per-case results stay on the Judge. Your final result is the best valid score you reach within the time budget, so keep `FINAL_VERSION` on your strongest candidate.

## Submitting

Wait for every public test and compiler process to finish, then run
`sforge-submit --details`. A Judge run uses part of your budget, so check candidates with the
public tests first.
