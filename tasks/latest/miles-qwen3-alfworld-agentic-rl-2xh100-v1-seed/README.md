# ALFWorld Agentic GRPO for Qwen3-4B in Miles

Miles (SGLang rollouts, Megatron-Core training, Ray) already trains multi-turn agents through its session server
with token-in-token-out, GRPO and weight publication, but it has no ALFWorld integration. The agent integrates the
ALFWorld text household environment into Miles so that Qwen3-4B trains with multi-turn GRPO on whether it wins each
game. The agent then makes the training run fast while it keeps learning. This is a seed task: the
starting point is Miles at `78527b9` with a starter `task_entry.py` that has the Ray and checkpoint plumbing only
(`run` and `probe` raise, so it scores 0).

## The task

The Work container's root is `/app/workdir/miles_lab`. The agent edits and submits `repo/` (Miles),
`task_entry.py`, `config/` and `notes.md`. The Judge runs:

```text
python task_entry.py run   --contract CONTRACT.json --out NEW_DIR
python task_entry.py probe --contract CONTRACT.json --batch BATCH.json --out NEW_DIR
```

`docs/INTERFACE.md` defines the contract, the outputs (`final_hf/` and one record per optimizer step and rank) and
the checks.

The training is fixed by the contract (`runtime/contract.json`):

- **Episodes.** Games load through TextWorld with `alfworld==0.4.2`. Each observation is rendered with its
  admissible actions, and the conversation uses the Qwen3 chat template with `enable_thinking=false`. The full
  history is kept, token-in-token-out: each request's prompt is exactly the previous prompt plus the generated
  tokens plus the new user turn.
- **Actions and reward.** The action is the text inside the first `<action>...</action>`. An episode ends on a win
  or after 30 actions; the reward is 1 for a win and 0 otherwise.
- **Sampling.** Temperature 1, top-p 1, at most 256 new tokens per turn, with the sampled token's log-probability
  only.
- **Loss.** GRPO with token mean over a group of the 8 episodes of a game: asymmetric clip 0.2 / 0.28, ratio
  against the engine's log-probabilities, no KL, no entropy term.
- **Optimizer and publication.** One AdamW step per round (lr 2e-6) by Megatron-Core inside Miles'
  `MegatronTrainRayActor`, then a publication to the sampling engines.
- **Workload.** The scored run is 12 rounds x 16 games x 8 episodes (Work holds neither the rows nor the size). Staleness is at most one weight version, and a single model turn
  never spans a publication.

Rules the Judge enforces:

- The run stays a Miles job: Megatron-Core in `MegatronTrainRayActor`, and SGLang engines started by Miles' Ray
  actors.
- Policy engines run at tensor-parallel size 1, and the trainer at tensor- and pipeline-parallel size 1.
- The `ALFWORLD_AUDIT_*` environment reaches every process.
- `probe` and `run` use the same implementation.
- `final_hf` and every served publication are exactly the weights Megatron's optimizer steps produced.
- The model, games, rendering, action rule, reward, loss, optimizer settings and quota stay fixed. ALFWorld and its
  game files stay unmodified.
- The implementation is written natively in the checkout.

The agent may change scheduling, environment hosting and parallelism, SGLang settings, prefix caching, overlap of
sampling and training within the staleness rule, batching, data parallelism, GPU placement, CPU optimizer offload
and the publication path.

The Judge stops `probe` after 1800 s and `run` after 7300 s, start-up included. Each submission also passes an LLM
integrity review (`llm_judge/POLICY.md`) before the GPU Judge. The agent's time budget is `TASK_TIMEOUT` in the
launch config; the example configs use 86400 s (24 h).

## Evaluation

The Judge runs three stages:

1. `probe` on hidden recorded episodes;
2. `run` on hidden training rows;
3. a greedy evaluation of the final policy (temperature 0, the same rendering and action rule) on 1134 hidden
   games, three times.

Checks:

- **probe.** The dump matches the batch, the GRPO advantages are correct, and the loss and the trainer's
  log-probabilities agree with fp64 and independent float32 recomputations. Exactly one Megatron-Core optimizer
  step produces `final_hf`.
- **run, provenance.** One optimizer step per round inside `MegatronTrainRayActor`, with publications bit-identical
  to Megatron's weights. `final_hf` must equal, tensor by tensor and by SHA-256, both Megatron's weights after the
  last step and what every engine served after its last publication.
- **run, audit.** Every engine request is recorded inside the engine. Episodes chain by exact token prefix and are
  replayed in the Judge's own ALFWorld. The checks cover sampling parameters, quotas, staleness, one publication per
  step, dumps equal to the engine records, and each step's loss.
- **run, oracle.** Step 0 and one step that starts from randomly copied served weights are recomputed in float32
  (trainer log-probabilities and loss).

```
quality_score = mean greedy success of the final policy over 3 repeats
                - mean success of the original Qwen3-4B (same games and protocol, measured once and frozen),
                in percentage points
score         = quality_score x 3600 / T     if every check passes and quality_score >= 15
              = 0                             otherwise (valid)
```

`T` is the wall clock of `run`, from the first training generation request to the end of the last weight
publication, taken from the Judge's records. The Judge's own weight copies and digests are removed from it, at most
120 s per interval and 300 s in total. Start-up is not timed. There is no reference time and no cap. Sampling is
stochastic, so the same code gains different amounts from run to run. A score of 0 from failed checks or gates is
still a valid result. Judge or infrastructure errors make the result invalid.

The public suites in the Work container, `python /opt/miles_alfworld_public/run_public.py --suite {cpu,probe,small,perf} --output DIR`,
run the Judge's checks on public data:

- `small` runs 3 rounds x 4 games and `perf` 4 rounds x 8 games (`runtime/public_contract.json`, 32 public rows);
- `perf` reports `T` and greedy success on 8 public games in 3 repeats, and no score;
- they are only for getting the pipeline working.

Scheduled auto-evaluation is off. Each `sforge-submit` runs the full Judge (on its own GPU pair by default, while
Work keeps running; with `--shared-gpus` the Judge pauses Work because both use one pair) and returns its report to
the agent. After the budget ends, the supervisor evaluates the final archive once more on the Judge pair, and the
run's result is the best valid score across the run's submissions and this final evaluation. The Judge timeout is 14400 s.

## Hardware

Four NVIDIA H100 GPUs by default, as two NVLink pairs: Work runs on one pair (`GPU_INDICES`) and the Judge on the
other (`JUDGE_GPU_INDICES`, or `--judge-gpus C,D`), and Work keeps running while the Judge measures.
`JUDGE_GPU_INDICES=shared` or `--shared-gpus` runs both on one pair one after the other: the Judge pauses Work while
it runs. Containers get a 24-CPU quota and 256 GiB of host memory. A formal run pins each role to the
CPUs local to its pair (`SFORGE_CPUSET=auto`), as in the qualification.

The qualification Judges ran with no Work container next to them. Scores from
the default separate mode are not directly comparable with shared-mode runs: Work keeps running during Judge rounds and
gets more wall time, and the Judge times its window while Work trains on the other pair.

| hardware type | status | execution profile | memory floor per card |
|---|---|---|---|
| `h100-94gb-400w` (H100 NVL 94 GB) | qualified | `dual-cuda-sm90-nccl` | 90 GiB |
| `h100-80gb-700w` (H100 SXM 80 GB) | qualified | `dual-cuda-sm90-nccl-80g` | 79 GiB |

Each card type has its own qualification record (`runtime/qualification.json` for NVL,
`runtime/qualification.dual-cuda-sm90-nccl-80g.json` for SXM). The launcher admits each pair to a profile (both pairs must meet the same one) and refuses
a formal run without a matching record. Scores are comparable only within one profile. The reference places the
trainer (with CPU optimizer offload) on one card and the policy engine on the other.

## Running

1. **Assets.** `INFRATHON_ASSETS_ROOT=<dir> tasks/latest/miles-qwen3-alfworld-agentic-rl-2xh100-v1-seed/assets/fetch.sh`
   downloads:
   - `Qwen/Qwen3-4B` at a pinned commit, into the Hugging Face cache (`HF_HUB_CACHE` / `HF_HOME`; the task mounts it
     as `ALFW_HF_CACHE`);
   - from the Infrathon dataset: the initial Megatron checkpoint (about 8 GB, converted from Qwen3-4B with
     Miles' `tools/convert_hf_to_torch_dist.py`), the public task data that is built into the images, and the
     Judge-only data (hidden training rows, hidden evaluation games and a hidden probe batch) from the dataset's
     `judge/` part; it is not in the repository.
2. **Images.** `INFRATHON_ASSETS_ROOT=<dir> scripts/infrathon/pull_images.sh miles-qwen3-alfworld-agentic-rl-2xh100-v1-seed`
   pulls the published Work and public Judge for this host's hardware type. It then builds the Judge's holdout
   layer locally from the Judge-only assets (`images/judge/Dockerfile.private`).

   To build from source instead:
   - build the base images with `scripts/miles_base_images/build.sh model-flat`, then
     `scripts/miles_base_images/build.sh alfworld`;
   - set `MILES_UPSTREAM` to a Miles clone that contains `78527b9`;
   - run `tasks/latest/miles-qwen3-alfworld-agentic-rl-2xh100-v1-seed/scripts/build_images.sh`.
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
   SFORGE_CPUSET=auto launch/miles-alfworld/run.sh --config <config.env> [--judge-gpus C,D | --shared-gpus] [--run-id ID] [--dry-run] [--foreground]
   ```

   Start the config from `launch/miles-alfworld/*.env.example`. It sets:
   - the SForge version store and the agent runtime (built with `launch/build-agent-runtime.sh`);
   - the agent and model credentials: a Codex ChatGPT login, an OpenAI-compatible endpoint, or Claude Code;
   - `MODEL`, `REASONING_EFFORT`, `TASK_TIMEOUT`, `SUBMISSION_COOLDOWN`, `GPU_INDICES` (Work's NVLink pair),
     `JUDGE_GPU_INDICES` (the Judge's NVLink pair, or `shared`) and `JUDGE_PORT`. The examples use `TASK_TIMEOUT=86400`,
     `GPU_INDICES=0,1`, `JUDGE_GPU_INDICES=2,3` and `JUDGE_PORT=18631`.

   `--dry-run` prints the plan, or the first failed precondition. The launcher checks:
   - the agent runtime, the credentials and the mandatory LLM review settings (`SFORGE_LLM_JUDGE_*`);
   - the GPU profile of both pairs, its qualification record, and the CPU pinning policy;
   - both images;
   - at least 300 GiB free under `MILES_SCRATCH_ROOT`.

   To judge an existing submission archive outside a launcher run, use
   `tasks/latest/miles-qwen3-alfworld-agentic-rl-2xh100-v1-seed/scripts/judge_archive.sh ARCHIVE NAME [GPU_PAIR]`.
   It uses one pair, and the caller must hold it.

During a run, Work is on an internal network. It reaches the model API and public web hosts (ports 80/443) only
through the run's relay proxy. The Judge has no network.

## Files

| path | contents |
|---|---|
| `images/work/` | Work Dockerfile, `AGENT_TASK.md` (the agent's brief), the starter `workdir/miles_lab/` (`task_entry.py`, `config/`, `notes.md`) |
| `images/judge/` | public Judge Dockerfile and `Dockerfile.private` (the host-built holdout layer) |
| `docs/INTERFACE.md` | commands, contract fields, step-record schema, what each check looks at |
| `runtime/` | contract, qualification records per profile (metric, frozen base quality, gate, limits), SForge binding (`sforge_entry.py`) |
| `llm_judge/POLICY.md` | integrity-review policy |
| `reference/task_entry.py` | reference entry (host only); the reference Miles changes are in `scripts/miles_alfworld/reference/` |
| `scripts/` | `build_images.sh`, `prepare_assets.py` (public and private rows, evaluation sets, game files), `run_task.py` (run supervisor), `judge_archive.sh` |
| `assets/` | `manifest.json` and `fetch.sh` |
| `tests/` | CPU tests of the Work network policy |

Checks, oracle, evaluator, public runner and audit hooks: `scripts/miles_alfworld/`. Shared Miles tooling:
`scripts/miles_common/`.

## Upstream and license

- Miles, `radixark/miles` at `78527b9102b3`: Apache-2.0. The base image starts from `radixark/miles`, pinned by
  digest (`scripts/miles_base_images/README.md`).
- ALFWorld, the `alfworld` package 0.4.2: MIT (package metadata). It installs TextWorld 1.7.0; see the TextWorld
  project for its license.
- The games are the solvable train, `valid_seen` and `valid_unseen` games of ALFWorld's `json_2.1.1` release, split
  by `scripts/prepare_assets.py`. See the upstream ALFWorld repository for the license of the game files.
- `Qwen/Qwen3-4B` at `1cfa9a72`: Apache-2.0 (per its model card).
- The task's own code is under the repository's root `LICENSE` (PolyForm Noncommercial 1.0.0).
