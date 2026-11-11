# Miles Qwen3-1.7B On-Policy Distillation (math, 2×H100)

The agent works in a checkout of the Miles RL framework (SGLang rollout, Megatron-Core training,
Ray) with on-policy distillation removed. It must implement sampled-token on-policy distillation
inside Miles: the Qwen3-1.7B student samples, a frozen Qwen3-4B teacher scores the same tokens, and
the student is trained toward the teacher on its own trajectories. Then it makes the complete
fixed-size training run fast while the student keeps learning. The starting point is a seed: the
starter launches a native Miles training job and exports weights, but has no teacher feedback
(zero advantage) and writes none of the required per-step records.

## The task

The agent edits, under the Work root `/app/workdir/miles_lab`:

- `repo/`: the Miles checkout (upstream Miles with its on-policy distillation code removed);
- `task_entry.py`: the entry point, with two commands defined in `docs/INTERFACE.md`:
  `task_entry.py run --contract CONTRACT.json --out DIR` and
  `task_entry.py probe --contract CONTRACT.json --batch BATCH.json --out DIR`;
- `config/` and `notes.md` (design notes).

These four paths are the submission. Both commands must write `result.json`, the trained student
as Hugging Face safetensors in `final_hf/` (original parameter names and shapes), and one
`steps/step_{k:03d}_rank_{r}.pt` record per optimizer step and data-parallel rank (tokens, loss
masks, student and teacher log-probabilities, advantages, loss).

The training math is fixed. Per round the student samples one response per prompt (temperature 1,
top-p 1, chat template with `enable_thinking=false`); the teacher returns full-vocabulary
log-probabilities `log q_t` of the sampled tokens; the pre-update student log-probabilities
`log p_old_t` come from a Megatron training-side forward (not from the sampling engine); the
advantage is `A_t = -(log p_old_t - log q_t)` with no other term; the loss is the PPO-clipped
objective (clip 0.2), averaged per sequence and then over sequences; one AdamW step per round
(lr 5e-6, betas 0.9/0.999, eps 1e-8, weight decay 0.01, gradient clipping 1.0), then the new
weights are published to the sampling engine.

Scored workload: 32 rounds of 128 DAPO-Math-17k problems, one response each, at most 4096 response
tokens (Work holds neither these rows nor this size). Round `r` uses lines `128r .. 128r+127` of the
training file; every response is
trained on exactly once, truncated ones included; staleness at most one step; one publication per
step.

Allowed: any change inside the Miles job, such as where the teacher runs (its own SGLang server or
inside Megatron, on either GPU), batching, overlap of sampling, scoring and training,
micro-batching, data parallelism, SGLang engine configuration and weight publication.

Not allowed (enforced by the Judge's checks and an LLM integrity review): replacing the Ray /
`MegatronTrainRayActor` / rollout manager / weight updater path with a custom loop; bringing in
another Megatron, SGLang or Ray; changing models, tokenizer, prompt rendering, data, row
assignment, sampling parameters, advantage, loss, optimizer hyper-parameters or workload; updating
the teacher; computing anything below BF16 (no FP8/INT8 weights, activations, GEMMs or KV caches);
student SGLang tensor parallelism other than 1 or Megatron TP/PP other than 1; any difference in
implementation between `probe` and `run`; serving or exporting weights that differ from
Megatron's optimizer output. The implementation must be written natively in the checkout; copying
in an existing on-policy distillation implementation (including the upstream one) is not allowed.
The `OPD_AUDIT_*` environment variables must reach every process the run starts.

The Work container has the public validation tool `/opt/miles_opd_public/run_public.py` with
three suites, only for getting the pipeline working: `probe` (one step on a fixed batch), `small`
(4 rounds of 16 prompts) and `perf` (8 rounds of 32 prompts, `runtime/public_contract.json`, with timed
window `T` and the accuracy on 256 public test problems, and no score). Apart from one problem in the `small`
suite's rows, Work's 256 training rows share no problem with the Judge's rows, even ignoring case, punctuation
and layout.
The agent must keep its use of the shared `/scratch` disk under 150 GB.

The agent budget is set by `TASK_TIMEOUT` in the launch config; the example configs use 86400 s
(24 hours).

## Evaluation

The Judge runs `probe` on a hidden fixed batch and then the full `run`, and applies these checks
(all must pass):

- **probe**: the step record matches the batch; advantages and loss are consistent with the dumped
  log-probabilities; student and teacher log-probabilities agree with an independent float32
  recomputation; the weight change follows the distillation gradient and agrees with the task's
  reference implementation on the same batch; exactly one Megatron-Core optimizer step with the
  contract's AdamW settings.
- **provenance**: one Megatron-Core optimizer step per round per training process, driven by
  Miles' `MegatronTrainRayActor`, from the pinned initial checkpoint; every publication is
  bit-identical to Megatron's weights; `final_hf` matches, tensor by tensor, both Megatron's
  weights after the last step and the weights the engines served last (SHA-256 digests).
- **audit**: every SGLang request is recorded inside the engine; each round has exactly 128
  student generations with the contract's parameters on the assigned rows, each trained once
  within the staleness limit; one publication per step; `old_logprobs` come from a training-side
  forward.
- **steps** and **same path**: per-step records are self-consistent, and the run's numerics-relevant
  Megatron configuration equals the probe's.
- **oracles**: step 0 is recomputed in float32 from the initial student; the served weights before
  four randomly chosen publications are copied and those steps' log-probabilities, loss and gradient
  are recomputed in float32 and compared with Megatron's reduced gradient; teacher log-probabilities
  of random rows of two more steps are recomputed.
- **final weights**: `final_hf` loads and is complete.

Score (maximized):

```
quality_score = mean accuracy of the final student over 3 seeded repeats
                - frozen mean accuracy of the original Qwen3-1.7B (same problems, protocol, seeds)
                (percentage points)
score         = quality_score × 3600 / T   if every check passes and quality_score >= 1.6
              = 0                          otherwise
```

Accuracy is measured on 2,000 hidden DAPO-Math-17k problems, sampled with the training contract
(temperature 1, top-p 1, at most 4096 new tokens, thinking disabled), one fixed seed per problem
and repeat, graded by upstream Miles' held-out DAPO scorer (last `\boxed{}` or last `Answer:`
line). MATH-500 is evaluated with the same protocol and reported only. The base accuracies per
repeat and the 1.6-point gate are fixed in the profile's qualification record. `T` is the audited wall
clock from the first student generation request to the end of the last weight publication;
model loading, the final checkpoint conversion and the evaluator's weight copies and digests
(at most 36 s in total) are not counted. A submission that fails a check or the gate scores 0
and stays a valid evaluation; an evaluator error makes the result invalid.

Submissions: there is no scheduled evaluation. Each `sforge-submit` from Work runs the full Judge
(on its own GPU pair by default, while Work keeps running; with `--shared-gpus` Work is paused
while it runs, because both share one pair) and returns the report to the agent. After the budget
ends, the supervisor evaluates `final_archive.tar.gz` once more on the Judge pair; the run's
result is the best valid score across the run's submissions and this final evaluation. Every evaluation is preceded by the LLM integrity review
(`llm_judge/POLICY.md`). The Judge's time limit is 10800 s (run 5400 s, probe 1800 s).

## Hardware

Four H100 GPUs per run by default, as two NVLink pairs: Work runs on one pair (`GPU_INDICES`) and
the Judge on the other (`JUDGE_GPU_INDICES`, or `--judge-gpus C,D`), and Work keeps running while
the Judge measures. `JUDGE_GPU_INDICES=shared` or `--shared-gpus` puts both on one pair, used
serially, with Work paused during each Judge. Work and Judge each get 24 CPUs and 256 GiB of host
memory; with `SFORGE_CPUSET=auto` each role's CPUs are pinned next to its own GPU pair.

The qualification Judges ran with no Work container next to them.
Scores from the default separate mode are not directly comparable with shared-mode
runs: Work keeps running during Judge rounds and gets more wall time, and the Judge times its
window while Work trains on the other pair.

| hardware type | support | execution profile | memory floor per card |
|---|---|---|---|
| `h100-94gb-400w` (H100 NVL 94 GB) | qualified | `dual-cuda-sm90-nccl` | 80 GiB |
| `h100-80gb-700w` (H100 SXM 80 GB) | qualified | `dual-cuda-sm90-nccl-80g` | 79 GiB |

Each profile has its own qualification record (`runtime/qualification.json`,
`runtime/qualification.dual-cuda-sm90-nccl-80g.json`); scores are comparable only within one
profile. The reference Judge peaks close to the full card on both types (the SGLang engines
reserve a fixed fraction of the card). The launcher refuses a formal run on a profile whose
qualification status is pending.

## Running

1. **Assets.** `INFRATHON_ASSETS_ROOT=<dir> tasks/latest/miles-qwen3-opd-scratch-2xh100-v2-seed/assets/fetch.sh`
   downloads `Qwen/Qwen3-1.7B` (student) and `Qwen/Qwen3-4B` (teacher) from Hugging Face at pinned
   revisions into the Hugging Face cache, and from the Infrathon dataset the initial student
   in Megatron `torch_dist` format, the public task data built into the images (Work has 256 training rows; the Judge has the
   scored rows) and the Judge-only
   data (hidden evaluation set, MATH-500, hidden probe batch) from the dataset's `judge/` part. The
   Judge-only data is not in the repository.
2. **Images.** `INFRATHON_ASSETS_ROOT=<dir> scripts/infrathon/pull_images.sh miles-qwen3-opd-scratch-2xh100-v2-seed`
   pulls the published Work and public Judge images for this host's hardware type and builds the
   Judge holdout layer locally (`images/judge/Dockerfile.private`). To build from source instead,
   first build the base image (`scripts/miles_base_images/build.sh model-flat`, see
   `scripts/miles_base_images/README.md`), then run
   `tasks/latest/miles-qwen3-opd-scratch-2xh100-v2-seed/scripts/build_images.sh` with
   `MILES_UPSTREAM` pointing at a radixark/miles clone that contains the pinned commit.
3. **Run.** Keep `INFRATHON_ASSETS_ROOT` (and `HF_HUB_CACHE`, if step 1 used one) exported in the
   launching shell, or set them in `launch/mlsysforge.env`: the Miles host configuration reads both
   from the environment, else from that file, not from the run config. Without them the dry run stops with a
   "... is not configured" error and the model cache falls back to `~/.cache/huggingface/hub`.
   Where a GPU pair has fewer local CPUs than the 24-CPU quota (for example 16 on some H100 NVL hosts), the
   launcher sets `SFORGE_CPUSET_UNDERSIZE=allow` itself and records it: the quota stays and the
   threads are confined to the pair's CPUs. The H100 NVL qualification was recorded this way, and a formal
   run must pin under its qualification's policy, so on an H100 NVL host whose pairs have 24 or more
   local CPUs the launcher refuses a formal run.

   ```bash
   # copy one of launch/miles-opd/*.env.example to run.env and fill in paths and credentials
   SFORGE_CPUSET=auto bash launch/miles-opd/run.sh --config launch/miles-opd/run.env --dry-run
   SFORGE_CPUSET=auto bash launch/miles-opd/run.sh --config launch/miles-opd/run.env   # detaches into tmux
   ```

   Options: `--judge-gpus C,D` or `--shared-gpus` (override `JUDGE_GPU_INDICES`), `--run-id ID`,
   `--dry-run`, `--foreground`. The config sets the agent and model
   (Codex with a ChatGPT login, Codex with an OpenAI-compatible endpoint and key file, or Claude
   Code with `AGENT=claude-code`), the agent runtime directory, the budget, the submission
   cooldown, Work's GPU pair (`GPU_INDICES`), the Judge's pair (`JUDGE_GPU_INDICES`, or `shared`)
   and the Judge port (the examples use 86400 s, `GPU_INDICES=0,1`, `JUDGE_GPU_INDICES=2,3` and
   `JUDGE_PORT=18635`). The launcher checks that both pairs meet the same execution profile, that a qualification exists for that profile and CPU pinning
   policy, that the LLM integrity review (`SFORGE_LLM_JUDGE_*` settings) can run, and that the
   scratch root has at least 200 GiB free. Outputs go to
   `$MILES_SCRATCH_ROOT/miles-qwen3-opd-scratch-2xh100-v2-seed/runs/<run id>/`. The launcher has
   no separate eval-only or replay entry.

Work runs on an internal network; the model API, the Judge and (because the task enables
internet) public HTTP(S) hosts are reached only through the run's relay. The Judge has no network
and never receives API credentials.

## Files

| path | contents |
|---|---|
| `images/work/` | Work Dockerfile, the agent instructions (`AGENT_TASK.md`), the starter `task_entry.py`, `config/` and `notes.md` |
| `images/judge/` | public Judge Dockerfile (checks, reference implementation, no hidden data) and `Dockerfile.private` (adds the Judge-only data) |
| `images/release.json` | image source paths and local image names for `scripts/infrathon/pull_images.sh` |
| `docs/INTERFACE.md` | commands, contract fields, required outputs and what the checks look at |
| `tasks/` | SForge task spec (submit paths, Judge command and limits, execution profiles) and the base image pin |
| `runtime/` | the Judge's contract and Work's `public_contract.json`, qualification records per profile (metric, base accuracy, gate, limits), public reference numbers for the `perf` suite, SForge binding (mounts, network, Judge pauses Work) |
| `reference/task_entry.py` | the reference entry point; built only into the Judge image |
| `llm_judge/POLICY.md` | the integrity review policy |
| `scripts/` | `build_images.sh`, `prepare_assets.py` (public/private data split), `make_work_data.py` (the Work image's data), `run_task.py` (run supervisor used by the launcher), `qualify.py` |
| `assets/` | `manifest.json` and `fetch.sh` |
| `tests/` | CPU-only tests of the Work network policy |

Shared code outside the task directory: `scripts/miles_opd/` (starter and reference patches,
checks, audit hooks, math evaluation and grader, data split) and `scripts/miles_common/`
(supervisor helpers, host configuration, CPU pinning).

## Upstream and license

- Miles ([radixark/miles](https://github.com/radixark/miles)) at
  `78527b9102b3a5f15891128cbec3ac6c2dc4089e`, Apache-2.0. The Work checkout is this commit minus its
  on-policy distillation code (`scripts/miles_opd/starter/`); the Judge's reference is the unmodified
  upstream implementation plus a launcher and step records (`scripts/miles_opd/reference/`).
- Training and hidden evaluation data:
  [BytedTsinghua-SIA/DAPO-Math-17k](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k)
  at `65877096`, Apache-2.0, deduplicated on prompt text and split with the permutation of Miles'
  self-distillation example (`scripts/miles_opd/v2/make_split.py`). The grader in
  `scripts/miles_opd/v2/math_grader/` is copied from upstream Miles (Apache-2.0).
- MATH-500 (reported only): [openai/prm800k](https://github.com/openai/prm800k)
  `math_splits/test.jsonl` at `7ecc7947`, MIT.
- Models: `Qwen/Qwen3-1.7B` at `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` and `Qwen/Qwen3-4B` at
  `1cfa9a7208912126459214e8b04321603b3df60c`, both Apache-2.0 (per their model cards).
- Base image: `radixark/miles` container image pinned by digest, rebuilt by
  `scripts/miles_base_images/`.
- The task's own code is under the repository's root `LICENSE` (PolyForm Noncommercial 1.0.0).
