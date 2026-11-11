# Qwen3-8B KV eviction policy lab (v2)

Choose which cached key/value positions survive fixed compression so that the
frozen Qwen3-8B keeps its full-cache next-token behaviour. The score covers
50%, 75% and 87.5% eviction on long-context items in four families: synthetic
retrieval and aggregation tasks, question answering over real documents, and
long-form generation such as summaries.

## Evaluation

Every scored `sforge-submit` evaluates the complete frozen private suite,
whose items differ from the public ones and can have
much longer contexts, with the same metric, gates and limits as the evaluation
after the deadline. The final
evaluation repeats that suite on the frozen archive. The Judge spreads the
items over several GPUs; one submission takes on the order of an hour, and
`sforge-submit` waits for it, so use the public suite for iteration and submit
deliberately. Each submission first passes an integrity review of the code
against this brief (see "What a policy may use").

Your final result is the best valid score you reach within the time budget, so keep `FINAL_VERSION` on your strongest candidate.

Feedback is aggregate only: the primary score, the three ratio scores, the four
family scores, the suite and per-item latency ratios, and every gate. No
per-item results or texts are returned.

The public suite (`/workspace/benchmark/corpus/items_public.jsonl`) contains
RULER items at 4k, 8k and 16k tokens and four LongBench tasks (hotpotqa,
triviaqa, qmsum, passage_count). Private items differ from the public ones but
stay in the same four families.

## Candidate ABI

```python
class EvictionPolicy:
    def __init__(self, spec: dict, compression_ratio: float, assets: dict) -> None: ...
    def layer_budgets(self, seq_len: int, n_kept_total: int) -> list[int] | None: ...
    def select(self, layer_idx: int, hidden_states, queries, keys, values,
               position_embeddings, weights, n_kept: int) -> torch.Tensor: ...
```

`spec` contains `model_id`, `n_layers`, `n_heads`, `n_kv_heads`, `head_dim`,
`hidden_size`, `rope_theta`, `n_sink_floor=4`, and `min_keep=64`.
`compression_ratio` is the fraction evicted: 0.5, 0.75 or 0.875. Optional
`assets.pt` is loaded as a string-to-tensor dict with `weights_only=True` and
is limited to 64 MiB.

`layer_budgets` returns `None` for uniform `int(L*(1-r))` per layer, or 36
integers in `[64,L]` whose sum is `36*int(L*(1-r))`.

`select` receives read-only BF16 CUDA tensors: hidden states `[1,L,4096]`
(the layer's attention input after RMSNorm), pre-RoPE post-q_norm queries
`[1,32,L,128]`, cached post-RoPE keys and values `[1,8,L,128]`, `(cos,sin)`
each `[1,L,128]`, and a read-only weight dict for `q_proj`, `k_proj`, `v_proj`,
`o_proj`, `q_norm`, and `k_norm`. Return CUDA `int64 [1,8,n_kept]`, unique per
row and in `[0,L)`.

A fresh `EvictionPolicy` is constructed for every item and ratio. For each
item, every ratio's policy first runs one untimed `select` pass over all 36
layers on random tensors with the item's shapes and strides (this absorbs
one-off costs such as kernel compilation or autotuning for a new length);
then the timed calls receive freshly copied item tensors, and each layer is
charged the slowest of its three ratio calls, so work shared across ratios
saves no measured time. A policy must compute each call from that call's
inputs: do not carry results from one call, ratio or item to another, do not
recognise repeated inputs (by tensor identity, address, shape and layer, or
content) to skip work, and do not treat the untimed calls differently from
the timed ones. Input-independent state is fine: loaded assets, positional
tables, compiled kernels and scratch buffers.

The candidate sees the context only. The question and the dataset's answer
prefix are appended after selection by the trusted replay, exactly as in the
kvpress pipeline: chat-templated context, then `question + assistant prefix +
answer_prefix`.

## What a policy may use

Score positions from what the model computes over the context: attention
estimates built from the supplied queries and keys, key/value/query and
hidden-state norms, statistics and similarities, positions, sinks and recency,
observation windows, and layer or head budget heuristics. Small learned
importance scorers over these activations, trained on public data and shipped
in `assets.pt`, are allowed.

Token identity is not allowed, in any form. A policy must not work out which
vocabulary token, or which kind of token, occupies a position — digits,
punctuation, separators, newlines, labels or markers, special or chat-template
tokens, particular words — and use that to score, protect or budget positions.
That covers hand-written rules, comparing activations with rows of the
embedding or unembedding matrix or with stored token vectors, token-id tables,
and learned components that act as token lookups or token-class detectors. It
also covers generic token-class heuristics such as always keeping separators
or punctuation. A statistic that merely correlates with token identity (a key
norm, an attention mass) is fine; identifying tokens is not. The integrity
review rejects submissions that use token identity.

## Score and gates

Per item and ratio, `fid = exp(-mean_token_KL(full || compressed))` over the
reference continuation. Family fidelity is the arithmetic mean over that
family's items; the ratio score is the geometric mean of the four family
fidelities (retrieval, aggregation, qa, generation); the primary score is 100
times the mean of the three ratio scores. Families are weighted equally
whatever their item counts.

All gates are fail-closed: admitted layout/source, valid unique indices,
conserved layer budget, immutable inputs, unchanged CPU/CUDA RNG, suite-wide
summed policy latency at most 30% of the summed trusted plain-prefill time with
every single (item, ratio) at most 60% of its own prefill, peak policy
allocation at most 8 GiB on the inputs' GPU, and finite trusted replay
fidelity with at least 16 reference tokens.
The Judge runs one shard per GPU in a single container and leaves every GPU
visible; a shard's GPU is the current CUDA device and holds every input
tensor. Allocating memory on, computing on, or switching to any other GPU
fails the run.

Before the first measured item, the Judge runs one untimed `select` pass over
all 36 layers for each ratio on a 2,048-token random decoy capture; the
ordinary validity, immutability, RNG, budget, and memory gates apply to the
warm-up. Measured policy latency for a layer call is the maximum of
synchronized host wall time and CUDA-event time; an (item, ratio) is charged,
layer by layer, the slowest of that layer's three ratio calls, summed across
layers. An item over the per-item cap is timed once more in a fresh worker
process (only when at most 64 items exceed the cap); the faster measurement counts only
if it reproduces every selection of that item bit for bit. Reference generation suppresses
every EOS id for the first 16 tokens, stops at EOS afterwards, and is capped at
48 tokens. Reference log-probabilities are recomputed in one teacher-forced
pass through the same per-layer masked replay path.

The public JSON reports per-item fidelity, mean KL, top-1 agreement, NLL delta,
reference-token count, select timing, plain-prefill timing, memory and every
gate, plus family means, per-task means, ratio scores, primary score, suite
latency ratio and maximum item latency ratio.

## Development

```bash
./tools/run_one.sh --version 1
./tools/run_public_tests.sh --quick --candidate-version 1
./tools/run_public_tests.sh --candidate-version 1
```

Keep versions at `candidates/policy_vNNN/submission.py`, optional tensors at
`assets.pt`, notes in `notes.md`, and the chosen canonical integer in
`FINAL_VERSION`.

Allowed: Torch, optional Triton, `math`, `typing`, and `dataclasses`.
Forbidden: filesystem/process/network access, dynamic code, load/save, dunder
or frame access, interpreter/module mutation, global RNG mutation, private
CUDA streams/graphs, `torch.compile`, and native extension loading.

Version 1 keeps the first four cached positions and the most recent ones; it
consults no query, key, value or weight. No other policy ships with this task
and no results for other methods are disclosed: every input that `select`
receives is yours to use within the rules above, and the public suite is the
instrument for deciding what works.

Keep changes inside the admitted ABI and measure every gate.

## Submitting

This lab has one GPU; the Judge runs on separate GPUs. Wait for every public
test and compiler process to finish before `sforge-submit --details`, so the
archive is the version you intend. Only your manual submissions run the Judge
while you work; the archive at the deadline is judged after Work ends.
