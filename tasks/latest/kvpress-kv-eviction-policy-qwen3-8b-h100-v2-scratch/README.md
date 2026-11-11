# KV-cache eviction policy for Qwen3-8B (kvpress-derived, scratch)

The agent designs a KV-cache eviction policy for a frozen Qwen3-8B: after the long context has
been prefilled, the policy chooses which cached key/value positions survive when 50%, 75% or
87.5% of them are evicted, so that the model's next-token distributions stay as close as possible
to those of the full cache. Items are long-context evaluation items taken verbatim from the
datasets that NVIDIA kvpress registers (RULER, LongBench, LooGLE, Zero-SCROLLS). This is a
"scratch" task: the only starting point is a trivial position-only baseline (four sink positions
plus the most recent positions) that reads no query, key, value or weight.

## The task

The Work container's lab root is `/app/workdir/kv_eviction_lab`. The agent writes versioned
candidates at `candidates/policy_vNNN/submission.py` (with an optional `assets.pt`), selects one
with the integer in `FINAL_VERSION`, and keeps notes in `notes.md`. Only `candidates/`,
`FINAL_VERSION` and `notes.md` are submitted. The full brief is
[`images/work/AGENT_TASK.md`](images/work/AGENT_TASK.md).

Each candidate defines:

```python
class EvictionPolicy:
    def __init__(self, spec: dict, compression_ratio: float, assets: dict) -> None: ...
    def layer_budgets(self, seq_len: int, n_kept_total: int) -> list[int] | None: ...
    def select(self, layer_idx: int, hidden_states, queries, keys, values,
               position_embeddings, weights, n_kept: int) -> torch.Tensor: ...
```

- `select` receives read-only BF16 CUDA tensors for one of the 36 layers (normalized hidden
  states, pre-RoPE queries, post-RoPE cached keys and values, RoPE `cos`/`sin`, and the layer's
  attention projection and norm weights) and returns `int64 [1, 8, n_kept]` indices, unique per KV head and in range.
- `layer_budgets` returns `None` (uniform budget) or 36 per-layer budgets, each at least 64,
  that conserve the total.
- `assets.pt` is an optional string-to-tensor dict, loaded with `weights_only=True`, at most
  64 MiB.
- The policy sees the context only. The question and answer prefix are appended after selection
  by the trusted replay, as in the kvpress pipeline.

Rules (enforced by static source admission, runtime gates and an integrity review):

- Allowed: PyTorch, optional Triton, `math`, `typing`, `dataclasses`. Forbidden: filesystem,
  process or network access, dynamic code, load/save, dunder or frame access, interpreter or
  module mutation, global RNG mutation, private CUDA streams or graphs, `torch.compile`, native
  extension loading.
- No token identity in any form: a policy may not determine which token, or which kind of token
  (digits, punctuation, separators, special or chat-template tokens, particular words), occupies
  a position, including through embedding-matrix comparisons or learned token detectors.
  Statistics of the activations (norms, attention estimates, similarities), positions, sinks,
  recency, observation windows and layer/head budget heuristics are allowed, as are small learned
  scorers over activations trained on public data.
- Each call must be computed from its own inputs: no reuse of results across calls, ratios or
  items, no recognition of repeated inputs, and no special treatment of untimed warm-up calls.
  Input-independent state (assets, positional tables, compiled kernels, scratch buffers) is fine.
- The policy runs on the current CUDA device only; touching another GPU fails the run.

The Work container has one GPU and a public suite of 85 items
(`/workspace/benchmark/corpus/items_public.jsonl`: RULER at 4k/8k/16k tokens and LongBench
hotpotqa, triviaqa, qmsum and passage_count) with `tools/run_one.sh --version N` and
`tools/run_public_tests.sh [--quick] --candidate-version N` for iteration. Internet access is available through a
restricted proxy. The launcher's default agent budget is 24 hours.

## Evaluation

The score is:

```text
fid(item, r)       = exp(-mean_token KL(p_full || p_compressed))   over the reference continuation
family_fid(r, f)   = mean over the items of family f of fid(item, r)
ratio_score(r)     = geomean over f in {retrieval, aggregation, qa, generation} of family_fid(r, f)
score              = 100 * mean over r in {0.5, 0.75, 0.875} of ratio_score(r)
```

The score is in fidelity points (higher is better, 100 = lossless). The reference continuation is
generated greedily from the full cache (EOS suppressed for the first 16 tokens, at most 48
tokens) and replayed teacher-forced through the same masked path for both caches. Families are
weighted equally regardless of item count.

The Judge runs in two phases. Phase A (unprivileged) imports the candidate, gives it the context
only, and records validated selections. Phase B (trusted, no candidate code) recomputes prefill,
reference and replay and measures fidelity. A submission is valid only if every gate passes; all
gates are fail-closed and run-wide:

- admitted layout and source; valid, unique, in-range indices; conserved layer budgets;
- inputs unmodified; CPU and CUDA RNG state unchanged;
- latency: summed policy time at most 30% of the summed trusted plain-prefill time, and every
  (item, ratio) at most 60% of its own prefill. Each item first gets an untimed `select` pass on
  random tensors of its shapes; timed calls get fresh input copies, and each layer is charged the
  slowest of its three ratio calls. An item over the per-item cap is re-timed once in a fresh
  process (at most 64 items) and the faster time counts only if it reproduces the selection bit
  for bit;
- policy peak memory at most 8 GiB on the inputs' GPU;
- finite fidelity with at least 16 reference tokens;
- identical Judge GPUs, each compute capability 8.0 or higher with at least 56 GiB.

Every scored `sforge-submit` during Work evaluates the complete frozen private suite: 5,000 items
with contexts of 6k to 28k tokens, drawn from RULER rows and long-document sources that do not
appear in the public suite. One submission takes about an hour on the seven Judge GPUs (see
[Hardware](#hardware)). Feedback is aggregate
only: primary score, the three ratio scores, the four family scores, latency ratios and gates.
Scheduled automatic evaluation is off; the default cooldown between submissions is 1,800 s, and
each submission first passes the integrity review ([`llm_judge/POLICY.md`](llm_judge/POLICY.md)).
When the budget ends, the archive selected by `FINAL_VERSION` is frozen and evaluated once more
on the same suite; the run's result is the best valid score across the run's submissions and this final evaluation. The task JSON's Judge timeout is
11,000 s.

## Hardware

This task uses 1 Work GPU and 7 identical Judge GPUs (8 per run), on separate cards: the benchmark
setup. By default Work uses GPU 0 and the Judge GPUs 1-7; `--judge-gpus` picks other GPUs, and the
launchers refuse any count other than seven. The Judge shards the items across its seven GPUs; the
final evaluation after the budget ends takes the same time again. Each GPU needs compute
capability 8.0 or higher and at least 56 GiB.

| hardware type | status |
|---|---|
| `h100-94gb-400w` (H100 NVL 94 GB) | qualified |
| `h100-80gb-700w` (H100 SXM 80 GB) | qualified |

Other host needs: at least 150 GB free on Docker's data root, at least 12 GB free in the Judge
container's `/tmp` (the Phase A to Phase B handoff for 5,000 items is about 8 GB), and Judge
CPU/memory of 56 CPUs and 88 GiB (`8 x N` CPUs and `32 + 8 x N` GiB for the `N` = 7 Judge GPUs).

## Running

1. Assets: `tasks/latest/kvpress-kv-eviction-policy-qwen3-8b-h100-v2-scratch/assets/fetch.sh`
   (files under `INFRATHON_ASSETS_ROOT`). It downloads `Qwen/Qwen3-8B` and the four source datasets
   (`simonjegou/ruler`, `Xnhyacinth/LongBench`, `simonjegou/loogle`, `simonjegou/zero_scrolls`)
   from their Hugging Face repositories at pinned revisions into the Hugging Face cache, and the
   Judge-only frozen holdout manifest from the dataset's `judge/` part (not in the repository;
   the build reads it from `INFRATHON_ASSETS_ROOT`). The holdout items are
   third-party text and are not redistributed: the image build rebuilds them from the pinned
   datasets and checks every item's SHA-256 against the manifest.
2. Images: `INFRATHON_PULL=1 tasks/latest/kvpress-kv-eviction-policy-qwen3-8b-h100-v2-scratch/scripts/build_images.sh`
   pulls the published Work and public Judge images and builds the holdout layer
   (`images/judge/Dockerfile.private`) locally; without `INFRATHON_PULL=1` it builds everything from
   source. `KV_EVICTION_MODEL_REPO` (the cache folder of `Qwen/Qwen3-8B`) and
   `KV_EVICTION_HF_HUB` (the hub cache with the datasets) default to the Hugging Face cache.
3. Run:

   ```bash
   launch/kvpress-kv-eviction-scratch/run.sh --model <model> \
     [--work-gpu 0 --judge-gpus 1,2,3,4,5,6,7] --dry-run
   ```

   Drop `--dry-run` to start. Defaults: Work GPU 0, Judge GPUs 1-7 (the benchmark setup),
   24 hours, effort `max`, Judge port 18621; `--work-gpu`, `--judge-gpus`, `--hours`,
   `--effort` and `--judge-port` change them (always exactly seven Judge GPUs). Options include `--agent codex|claude-code`, `--model-repo DIR`,
   `--cpuset auto` and `--tmux`. `--eval-archive FILE [--public-smoke]` runs no agent and
   evaluates an archive through the final-Judge path. `launch/kvpress-kv-eviction-scratch/replay.sh`
   re-evaluates saved candidate versions of a finished run, and
   `JUDGE_GPUS=1,2,3,4,5,6,7 scripts/eval_archive.sh ARCHIVE.tar.gz` evaluates a single archive;
   use exactly seven Judge GPUs there too (`replay.sh` refuses another count).

The launcher checks for the agent runtime and its credentials (a Codex `auth.json`, a Claude
Code OAuth token, or `MODEL_API_BASE_URL` for another endpoint), a configured integrity review
(`SFORGE_LLM_JUDGE_MODE=enforce`, see `launch/mlsysforge.env.example`), the images, the model
repository, and that the requested GPUs are free. Outputs go to
`outputs/<task directory>/runs/<run id>/`; `scripts/status.sh <run id>` summarizes a run.

## Files

- `images/work/`: Work image, agent brief (`AGENT_TASK.md`), public benchmark runner and the lab
  root with the baseline candidate (`workdir/kv_eviction_lab/`).
- `images/judge/`: public Judge (verifier, sharding, Phase A worker, trusted replay, source
  admission) and `Dockerfile.private` for the holdout layer.
- `runtime/`: the shared model runtime used by Work and Judge.
- `corpus/public/`: the 85 public items and their manifest.
- `scripts/`: image build, item preparation (`prepare_items.py`), launch entry, replay and
  archive evaluation, layout validation.
- `tasks/`: the SForge task spec. `portable/`: portable contract manifests and a rebuild guide.
- `llm_judge/`: integrity policy.
- `oracle/`: host-only reference policies used for qualification. `tests/`: host tests.

Oracle (`oracle/FINAL_VERSION` = 2; full 5,000-item final holdout, H100 NVL):

| oracle | policy | score | gates |
|---|---|---:|---|
| seed (`policy_v001`) | 4 sinks + recency | 40.790466 | all pass |
| best admissible paper (`policy_v002`, selected) | kvpress `ExpectedAttentionPress` port (@ `7f359b84`) | 77.737238 | all pass |

## Upstream and license

- NVIDIA kvpress at commit `7f359b84` (Apache-2.0): the evaluation pipeline the task follows and
  the dataset registry its items come from. kvpress itself is not installed in the images.
- `Qwen/Qwen3-8B` at revision `b968826d9c46dd6066d109eabc6255188de91218`, Apache-2.0 (per its
  model card).
- Datasets at pinned revisions:

  | Hugging Face dataset | revision | license on the Hub |
  |---|---|---|
  | [`simonjegou/ruler`](https://huggingface.co/datasets/simonjegou/ruler) | `24adceac` | none declared; follows the upstream source (RULER) |
  | [`Xnhyacinth/LongBench`](https://huggingface.co/datasets/Xnhyacinth/LongBench) | `2e9ade51` | MIT |
  | [`simonjegou/loogle`](https://huggingface.co/datasets/simonjegou/loogle) | `6e3eaf28` | none declared; follows the upstream source (LooGLE) |
  | [`simonjegou/zero_scrolls`](https://huggingface.co/datasets/simonjegou/zero_scrolls) | `19949309` | none declared; follows the upstream source (Zero-SCROLLS) |

  `scripts/prepare_items.py` never reads non-commercial, no-derivatives, research-only or
  unclear-license sources: LongBench samsum, repobench-p, multi_news, lcc, trec and
  multifieldqa_en (and their `_e` variants); Zero-SCROLLS book_sum_sort, summ_screen_fd and
  space_digest; LooGLE shortdep_cloze. It also drops screenplay documents from LooGLE
  longdep_qa and NarrativeQA, and keeps only the AMI and ICSI meetings (CC BY 4.0) of QMSum.
- The task's own code is under the repository's root `LICENSE` (PolyForm Noncommercial 1.0.0).
