# Query-Agnostic KV Cache Scoring for Qwen3-8B (KVzip protocol)

The agent writes a scorer that assigns one query-agnostic importance score to every cached
context key/value pair of frozen Qwen3-8B. A trusted runtime keeps the globally highest 10%,
20% and 30% of those pairs and measures how much of the full-cache answer quality survives on
the KVzip evaluation tasks plus RULER, with contexts of 64 to 32,000 tokens. This is a
scratch task: `scorer_v001` is an empty file, and the agent writes the whole scorer.

## The task

The Work container's working directory is `/app/workdir/kv_scoring_lab`. The agent creates
versioned candidates `candidates/scorer_vNNN/submission.py`, names the selected version in
`FINAL_VERSION`, and documents its work in `notes.md`. Only `candidates/`, `FINAL_VERSION`
and `notes.md` are submitted. Each version may also hold one `assets.pt` (a string-to-tensor
dictionary, at most 128 MiB) or an `assets.ref` naming an earlier version that has a real
asset file; the whole compressed submission must stay under 128 MiB.

`submission.py` defines exactly:

```python
class KVScorer:
    def __init__(self, spec: dict, assets: dict) -> None: ...
    def score(self, layer_idx: int, hidden_states, queries, keys, values,
              position_embeddings, weights, ctx_range: tuple[int, int]) -> torch.Tensor: ...
```

`score` is called once per layer (36 layers) during a context-only prefill, with read-only
BF16 CUDA tensors: the normalized attention input, pre-RoPE queries `[1,32,S,128]`, cached
keys and values `[1,8,S,128]`, the RoPE tables, the layer's projection and norm weights, and
`ctx_range=(start, end)` (positions before `start` are the protected system prompt). It
returns a finite `float32 [1,8,end-start]` CUDA tensor; higher means keep. Scores must be
comparable across all layers and KV heads, because one score tensor serves all three
retention ratios. Questions are appended only after scoring, so the scorer never sees them.

Rules (stated in `images/work/AGENT_TASK.md`, enforced by static source admission, the
Judge's runtime checks and an LLM integrity review):

- Allowed imports: `math`, `typing`, `dataclasses`, `torch`, `torch.nn.functional`, `triton`,
  `triton.language`. Filesystem, process and network access, dynamic code, serialization,
  interpreter or module mutation, frame access, native extensions, private CUDA streams or
  graphs, `torch.compile` and global RNG mutation are forbidden.
- No token identity in any form: the scorer may not determine which token or kind of token
  occupies a position (digits, punctuation, separators, special or template tokens, words),
  whether by rules, embedding comparisons, token tables or learned token-class detectors.
- Each call computes from its own inputs. A fresh `KVScorer` is built for every row; caching
  results keyed on tensor identity, address or content, carrying state between rows, or
  behaving differently in untimed calls is rejected. Input-independent state (assets,
  compiled kernels, scratch buffers) may be reused.
- Learned assets must be designed and trained by the agent during the run, with their recipe
  (script, data, seed) in `notes.md`. Weights that someone else trained (downloaded, ported,
  re-quantized, fine-tuned from or distilled from) may not be used; a Judge gate rejects close
  copies of published KV-importance weights.
- The frozen model, runtime, evaluator and task-owned tools may not be modified. The scorer
  must use only the GPU that holds its inputs.

The Work container has one GPU, the frozen model mounted read-only, and outbound access to
public hosts through a restricted proxy. The launcher's default agent budget is 24 hours.

## Evaluation

For each row the Judge runs two phases. In Phase A, an unprivileged worker loads the
candidate and scores every layer in three passes: untimed on random tensors with the row's
shapes (warm-up), timed on the row's own tensors (these scores are used), and untimed on
fresh copies of the row's tensors (the scores must match: determinism). In Phase B, trusted
code that never imports candidate code keeps exactly `round(r * 36 * 8 * n_context)` of the
flattened `(layer, kv_head, position)` scores for each ratio `r` in {0.1, 0.2, 0.3} (ties keep
lower indices; system-prompt and appended positions always stay visible) and decodes the
full-cache and compressed answers greedily through the same masked-attention path.

Workloads, in three equally weighted groups:

| group | tasks |
|---|---|
| retrieval | needle-in-a-haystack, SCBench kv and prefix_suffix, SCBench RepoQA, RULER niah |
| contextual QA | SQuAD, GSM8K, RULER qa |
| redundancy | SCBench Math.Find, SCBench many-shot ICL, SCBench summary, RULER vt |

Per-task metrics follow upstream KVzip and RULER: normalized answer inclusion, numeric
inclusion, exact match, option-letter match, ROUGE-L, RepoQA BLEU >= 0.8, and RULER string
match. The score is

```text
rel(task, r)  = acc(task, r) / acc(task, full)
score_rel_auc = 100 * mean_r( mean_group( mean_task_in_group( rel(task, r) ) ) )
```

in relative-accuracy points (100 = lossless), maximized.

A submission is valid only if every check passes (all fail-closed): admitted source and
layout; assets not a copy of published KV-importance weights; scores of the exact dtype,
shape and device and all finite; inputs unchanged; global CPU and CUDA RNG unchanged;
overhead (total timed score time <= 25% of plain-prefill time, and each row <= max(50% of
its prefill, 10x the shortest prefill in the suite), with a row over the per-row cap re-timed
once in a fresh process, counted only if its scores reproduce bit for bit); peak scorer
memory <= 8 GiB on the inputs' GPU; identical Judge GPUs; finite metrics and positive
full-cache accuracy for every task.

Data: the public suite (63 rows from SQuAD validation, GSM8K train, a public needle,
generated SCBench-format rows and RULER) ships in the Work image and shares no row with the
held-out suite. Every scored `sforge-submit` during Work and the final evaluation run the same
held-out suite of 1,812 rows (about 7,000 questions) covering all twelve tasks, with private
needle statements. The held-out rows are rebuilt at image build time from pinned public
datasets and checked against a Judge-only manifest. Feedback is aggregate only: the primary
score, the three ratio scores, the three group scores, overhead ratios and every check.

Scheduled evaluation is off; the agent submits manually with `sforge-submit`, which waits for
the result, with a cooldown between submissions (`EVAL_INTERVAL`, default 300 s). One
evaluation takes well under an hour on the seven Judge GPUs. When the
budget closes, the harness freezes one final archive and evaluates it once on the same suite;
the run's result is the best valid score across the run's submissions and this final evaluation. An evaluation still running when the budget closes finishes
first.

## Hardware

This task uses 1 Work GPU and 7 identical Judge GPUs (8 per run), on separate cards: the benchmark
setup. By default Work uses GPU 0 and the Judge GPUs 1-7; `--judge-gpus` picks other GPUs, and the
launchers refuse any count other than seven. The Judge shards the rows across its seven GPUs (one
Phase A worker and one Phase B decode per card). Each card needs at least 79 GiB of memory. The runtime sets the PyTorch
allocator option `expandable_segments:True` on every card type, so one code path serves both.
The Judge also needs free disk space for the Phase A to Phase B score handoff (about 27 GB for
the full suite; the Judge requires 1.25 x that plus 1 GiB free); it refuses to start otherwise.

| hardware type | status |
|---|---|
| `h100-94gb-400w` (H100 NVL 94 GB) | qualified |
| `h100-80gb-700w` (H100 SXM 80 GB) | qualified |

## Running

1. Assets. `tasks/latest/kvzip-query-agnostic-kv-scoring-qwen3-8b-h100-v2-scratch/assets/fetch.sh`
   fetches into `INFRATHON_ASSETS_ROOT` (default `~/.cache/infrathon/assets`) and verifies:
   - `Qwen/Qwen3-8B` (the frozen model, Work and Judge), pinned revision, from Hugging Face;
   - the build-time sources of the held-out rows: `rajpurkar/squad`, `openai/gsm8k`,
     `Jang-Hyun/SCBench-preprocessed` and `simonjegou/ruler`, pinned revisions;
   - `Jang-Hyun/Fast-KVzip`, used only at build time to compute the weight-free fingerprint
     for the released-weights gate;
   - the Judge-only held-out manifest and private needles, from the dataset's `judge/` part
     (not in the repository; the build reads them from `INFRATHON_ASSETS_ROOT`).

   The build also clones `snu-mllab/KVzip` at the pinned commit for its needle generator and
   essay files (`KV_SCORING_KVZIP_REPO` to use an existing checkout).
2. Images. `INFRATHON_PULL=1 tasks/latest/kvzip-query-agnostic-kv-scoring-qwen3-8b-h100-v2-scratch/scripts/build_images.sh`
   pulls the published Work and public Judge images and builds the held-out Judge layer
   locally; without `INFRATHON_PULL=1` it builds everything from source. In both modes it rebuilds
   the held-out rows and checks each against the manifest.
3. Run.

   ```bash
   launch/kvzip-kv-scoring/run.sh --model MODEL \
     [--work-gpu 0 --judge-gpus 1,2,3,4,5,6,7] [--tmux]
   ```

   Defaults: Work GPU 0, Judge GPUs 1-7 (the benchmark setup), 24 hours, effort `max`, Judge
   port 18622; `--work-gpu`, `--judge-gpus`, `--hours`, `--effort` and `--judge-port` change
   which GPUs are used (always exactly seven Judge GPUs).
   `--agent codex` (default; Codex `auth.json` via `--auth`, or an OpenAI-compatible endpoint
   via `MODEL_API_BASE_URL` and `MODEL_API_KEY_FILE`) or `--agent claude-code`
   (`--claude-token FILE`). `--dry-run` checks and prints the plan without starting anything;
   `--tmux` detaches into a tmux session named `kvzip-v2-<label>-<timestamp>` (`<label>` is `--label`, else `run`; not the run ID)
   that logs to `launch/kvzip-kv-scoring/logs/run-<session>.log`;
   `--config FILE` loads a launch config (see `launch/mlsysforge.env.example`). The launcher
   refuses to start unless the integrity review is configured with
   `SFORGE_LLM_JUDGE_MODE=enforce` and the `SFORGE_LLM_JUDGE_*` reviewer settings; it also
   checks that the GPUs are idle and the Judge GPUs identical.

   Eval only: `JUDGE_GPUS=1,2,3,4,5,6,7 tasks/latest/kvzip-query-agnostic-kv-scoring-qwen3-8b-h100-v2-scratch/scripts/eval_archive.sh ARCHIVE.tar.gz`.
   The evaluation runs the LLM integrity review first and needs the `SFORGE_LLM_JUDGE_*`
   settings; for a trusted archive (the oracle, a reference scorer) set
   `SFORGE_LLM_JUDGE_MODE=off`. The host-only oracle: `scripts/make_oracle_archive.sh`, then
   `KV_SCORING_HOST_ORACLE=1 SFORGE_LLM_JUDGE_MODE=off JUDGE_GPUS=... scripts/eval_archive.sh artifacts/oracle.tar.gz`.
   The oracle's `FINAL_VERSION` is 2 (`scorer_v002`):

   | oracle | what | H100 NVL 94 GB score | gates |
   |---|---|---|---|
   | `scorer_v001` | Fast KVzip released gate, FastKVzip repo window rule | 82.75 (82.83 on H100 SXM 80 GB) | valid: released-weights gate exempted, all other gates pass |
   | `scorer_v002` (strongest paper solution) | same gate and weights, NVIDIA kvpress window rule | 92.81 | valid: released-weights gate exempted, all other gates pass |

   Both oracles use the released Fast-KVzip weights. They pass only because
   `KV_SCORING_HOST_ORACLE=1` exempts this host-only reference asset from the released-weights
   gate (the LLM integrity review is also off for them); they are not admissible for agents.
   `ORACLE_VERSION=1 scripts/make_oracle_archive.sh` packs `scorer_v001` as `artifacts/oracle-v001.tar.gz`.
   `oracle/notes.md` has the details.
   Replay saved candidates of a finished run:
   `JUDGE_GPUS=<seven GPUs> launch/kvzip-kv-scoring/replay.sh --origin-run RUN_ID --measurement-set NAME [--per-version]`.

Lightweight checks without a GPU: `scripts/validate_layout.py <task dir>` and
`pytest tasks/latest/kvzip-query-agnostic-kv-scoring-qwen3-8b-h100-v2-scratch/tests`.

## Files

| path | contents |
|---|---|
| `images/work/` | Work image: `AGENT_TASK.md` (the agent's brief), the lab (`workdir/kv_scoring_lab`: empty `scorer_v001`, `tools/` for public runs and size checks), the public runner |
| `images/judge/` | public Judge (`run_judge.py`, `verify_kv_scoring.py`, candidate worker, trusted replay, sharding, source admission, released-weights gate) and `Dockerfile.private` for the held-out layer |
| `runtime/` | shared trusted runtime and metrics (prefill capture, global top-r allocation, masked decode) |
| `corpus/public/` | the 63-row public suite |
| `oracle/` | host-only reference scorers used for qualification; never shipped to Work |
| `scripts/` | image build, item preparation, launcher entry, archive evaluation, replay, layout validation |
| `tasks/` | SForge task spec (judge command, timeouts, resources, submit paths) |
| `portable/` | portable task contract manifests |
| `llm_judge/POLICY.md` | integrity-review policy |
| `tests/` | CPU tests of the task contract |

## Upstream and license

- Evaluation protocol and needle generator: `snu-mllab/KVzip` at
  `5d8472975d286e3da6b52178af39d18904330c0d` (MIT code; Paul Graham essays as haystack).
- Model: `Qwen/Qwen3-8B` at revision `b968826d9c46dd6066d109eabc6255188de91218`.
  Apache-2.0 (per its model card).
- Data (pinned revisions, see `assets/manifest.json`): `Jang-Hyun/SCBench-preprocessed` (MIT),
  `rajpurkar/squad` (CC BY-SA 4.0), `openai/gsm8k` (MIT), [`simonjegou/ruler`](https://huggingface.co/datasets/simonjegou/ruler)
  (no license declared on the Hub; it follows the upstream RULER source, whose QA tasks draw on
  SQuAD and HotpotQA). The held-out rows are third-party text and are
  not redistributed; every host rebuilds them.
- `Jang-Hyun/Fast-KVzip` (MIT) is used only to build the weight-free fingerprint and the
  host-only oracle; submissions may not use it (see the rules above).

The task's own code is under the repository's root `LICENSE` (PolyForm Noncommercial 1.0.0).
