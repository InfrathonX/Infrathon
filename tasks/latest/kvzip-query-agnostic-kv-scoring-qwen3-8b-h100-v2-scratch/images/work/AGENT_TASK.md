# Query-agnostic KV importance scoring for frozen Qwen3-8B

Your objective is to produce one query-agnostic importance score per cached
context KV pair so that retaining the globally highest 10%, 20%, and 30%
preserves the frozen model's answers. Version 1 is intentionally empty: you
must write the whole scorer.

## Interface

Create `candidates/scorer_vNNN/submission.py` with exactly:

```python
class KVScorer:
    def __init__(self, spec: dict, assets: dict) -> None: ...
    def score(self, layer_idx: int, hidden_states, queries, keys, values,
              position_embeddings, weights, ctx_range: tuple[int, int]) -> torch.Tensor: ...
```

`spec` has `model_id`, `n_layers`, `n_heads`, `n_kv_heads`, `head_dim`,
`hidden_size`, `rope_theta`, `ratio_ladder`, and `max_ctx_tokens`. `assets` is
`{}` or the string-to-tensor dictionary loaded from your asset file.

The call occurs per layer during one context-only prefill. Inputs are
read-only BF16 CUDA tensors:

- `hidden_states [1,S,4096]`: normalized attention-layer input.
- `queries [1,32,S,128]`: pre-RoPE, post-q_norm.
- `keys`, `values [1,8,S,128]`: cached K/V; keys are post-RoPE and post-k_norm.
- `position_embeddings=(cos,sin)`, each `[1,S,128]`.
- `weights`: read-only `q_proj`, `k_proj`, `v_proj`, `o_proj`, `q_norm`, `k_norm` tensors.
- `ctx_range=(start,end)`, where `end == S`; positions before `start` are the protected system prompt.

Return a finite CUDA `torch.float32 [1,8,end-start]` tensor. Higher means
keep. Scores must be calibrated across every layer and KV head.

A fresh `KVScorer` is constructed for every row, and every layer is scored
in three passes. First, untimed, each layer on random tensors with the row's
shapes and strides (this absorbs one-off costs such as kernel compilation or
autotuning for a new length). Second, timed, each layer on the row's own
tensors; these scores are used and only these calls count toward the
overhead check. Third, untimed, each layer on fresh copies of the row's
tensors; the scores must match the second pass (reported as determinism).
No untimed call sees the row's content before every timed call has
returned. Compute each call from that call's inputs: returning a stored
result when the same inputs come back (recognised by tensor identity,
address, shape and layer, or content), carrying scores or intermediates from
one row to the next, or treating the untimed calls differently from the
timed ones is not allowed, and the integrity review rejects it.
Input-independent state (loaded assets, tables, compiled kernels, scratch
buffers) may be reused.

## What a scorer may use

Score positions from what the model computes over the context: attention
estimates built from the supplied queries and keys, key/value/query and
hidden-state norms, statistics and similarities, positions, sinks and recency,
observation windows, cross-layer and cross-head calibration.

Learned assets. If you want a learned component, design and train it
yourself in this run and ship its parameters in assets (for example a small
scorer trained on targets from a reconstruction oracle you implement on public
text). Do not take weights that someone else has already trained --
downloaded, ported, cast or re-quantized, fine-tuned from or distilled from --
and use them in your scorer or assets. Reading papers and code, or downloading
such weights only to study or compare locally, is fine. Say in `notes.md` how
each asset was produced (script, data, seed).

Token identity is not allowed, in any form. A scorer must not work out which
vocabulary token, or which kind of token, occupies a position — digits,
punctuation, separators, newlines, labels or markers, special or chat-template
tokens, particular words — and use that to raise or lower scores. That covers
hand-written rules, comparing activations with rows of the embedding or
unembedding matrix or with stored token vectors, token-id tables, and learned
components that act as token lookups or token-class detectors. It also covers
generic token-class heuristics such as always keeping separators or
punctuation. A statistic that merely correlates with token identity (a key
norm, an attention mass) is fine; identifying tokens is not.

## Evaluation

The Judge stacks scores in `(layer, kv_head, context_position)` order. At
ratio `r`, it keeps exactly `round(r * 36 * 8 * n_context)` globally highest
entries. Equal scores keep lower flattened indices first. The same score
tensor serves every ratio.

Questions are appended only after scoring, so no question or answer is visible
to your code. System-prompt and appended question/answer positions always
remain visible. Full-cache and compressed answers use the same head-specific
masked-attention path and greedy decoding.

**Public tests.** These tests are for getting your scorer working and checking
correctness, with rough profiling. A good public number does not guarantee a
good Judge score; build your own validation data for tuning.

| group | public task | metric | max tokens | stress |
|---|---|---|---:|---|
| retrieval | needle | normalized answer inclusion | 48 | a planted fact in essays |
| retrieval | scbench_kv, scbench_prefix_suffix | inclusion | 96 / 128 | key-value and exact-string recall |
| retrieval | ruler_niah | all reference values present | 64 | needle variants |
| contextual QA | squad, gsm | inclusion / numeric inclusion | 256 / 512 | semantic evidence and reasoning context |
| contextual QA | ruler_qa | any reference present | 32 | multi-document QA |
| redundancy | scbench_mf | exact match | 48 | finding extremes in long number lists |
| redundancy | ruler_vt | all reference values present | 48 | variable tracking |

Some rows start the answer with a fixed answer prefix.

**Hidden tests.** Your score comes from hidden test cases, not from these public
tests. The hidden suite is much larger than the public one, its rows differ, and
it can include workloads the public tests do not cover. Every context stays
between 64 and 32,000 tokens. Make your scorer work across that whole range and
across kinds of context, not just on the public rows.

A submission takes on the order of half an hour to an hour and `sforge-submit`
waits for it, so iterate on the public suite and submit deliberately. Each
submission first passes an integrity review of the code against this brief.
Feedback is aggregate only: the primary score, the three ratio scores, the three
group scores, the overhead ratios and every check.

For task `t`, `rel(t,r)=acc(t,r)/acc_full(t)`. The Judge averages task
relatives within each of retrieval, contextual-QA, and redundancy; averages
the three group values per ratio; then reports
`score_rel_auc = 100 * mean(ratio_010, ratio_020, ratio_030)`. Maximize it.

Every check is fail-closed:

| check | requirement |
|---|---|
| source/layout | exact ABI and admitted version tree |
| assets | not a copy of published pretrained KV-importance weights |
| scores | exact dtype/shape/device; all finite |
| immutability | every input checksum unchanged |
| RNG | global CPU and CUDA RNG unchanged |
| overhead | total score time ≤25% of plain-prefill time; each row ≤ max(50% of its prefill, 10x the shortest plain prefill in the suite) |
| memory | peak scorer allocation delta ≤8 GiB, on the inputs' GPU only |
| metrics/reference | all finite; every full-cache task accuracy positive |

The Judge runs one shard per GPU in a single container and leaves every GPU
visible; a shard's GPU is the current CUDA device and holds every input
tensor. Allocating memory on, computing on, or switching to any other GPU
fails the run.

Two untimed warm-ups, of 2,048 and 128 tokens, run across all layers with
every check active before the first row. The overhead check counts the
second-pass (timed) calls only; each is the larger of synchronized host wall
time and CUDA-event time. A row over the per-item cap is timed once more in a
fresh worker process (only when at most 64 rows are over the cap); the faster measurement
counts only if it reproduces that row's scores bit for bit.

## Development

```bash
./tools/run_one.sh --version 1
./tools/run_public_tests.sh --quick --candidate-version 1
./tools/run_public_tests.sh --candidate-version 1
./tools/check_submission_size.sh
```

`--quick` takes the first two public rows of every task. The JSON reports
task/full accuracies, relatives, group and ratio scores, overhead ratios and
item violations, deterministic-item telemetry, memory, checks, per-row
decoding counts, and Phase A/B seconds including per-task Phase B sums.

Keep every attempt as `scorer_vNNN`; `FINAL_VERSION` contains the selected
integer. A version may contain `submission.py` and optionally one of
`assets.pt` (string-to-tensor, at most 128 MiB) or `assets.ref` (one line
naming an earlier `scorer_vNNN` with a real asset file; no chains). The harness
rejects any submission whose compressed archive exceeds 128 MiB, covering all
versions together plus `FINAL_VERSION` and `notes.md`; BF16 tensors barely
compress, so a large asset can exist in one version only and other versions
should use `assets.ref`. A rejected archive is a lost round. Keep assets at or
below 64 MiB when possible.

Allowed imports: `math`, `typing`, `dataclasses`, `torch`,
`torch.nn.functional`, `triton`, `triton.language`. Filesystem, processes,
network, dynamic code, serialization calls, interpreter/module mutation,
dunder/frame access, native extensions, private CUDA streams/graphs,
`torch.compile`, and global RNG mutation are forbidden.

Do not modify the frozen model, runtime, evaluator, or task-owned tools.

## Submission evaluation policy

Every `sforge-submit` runs the evaluation on hidden test cases. `sforge-submit --details` returns the gate states and the score. Hidden cases and per-case results stay on the Judge. Your final result is the best valid score you reach within the time budget, so keep `FINAL_VERSION` on your strongest candidate.

## Submitting

Wait for every public test and compiler process to finish before
`sforge-submit --details`, so the archive is the version you intend. Only your
manual submissions run the Judge while you work.
