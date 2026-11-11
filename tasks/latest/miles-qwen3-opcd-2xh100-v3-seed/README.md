# On-Policy Context Distillation for Qwen3-1.7B in Miles

Miles (SGLang rollouts, Megatron-Core training, Ray) supports sampled-token on-policy distillation but cannot
express context distillation. The agent implements on-policy context distillation (OPCD) inside Miles. The setting
is the OPCD paper's system-prompt distillation with a same-size frozen teacher:

- the teacher is the student's own initial Qwen3-1.7B weights, given a fixed medical system prompt `c`;
- the student learns, without `c`, to match the teacher through an analytic top-256 reverse KL;
- the agent then makes the fixed-size training run fast while it keeps learning.

This is a seed task: the starting point is Miles at `78527b9`, with a starter that launches Miles on the contract
and converts the final checkpoint. Its loss (`config/opcd_loss.py`) is a placeholder that raises, and it writes no
step records.

## The task

The Work container's root is `/app/workdir/miles_lab`. The agent edits and submits `repo/` (Miles),
`task_entry.py`, `config/` and `notes.md`. The Judge runs:

```text
python task_entry.py run   --contract CONTRACT.json --out NEW_DIR
python task_entry.py probe --contract CONTRACT.json --batch BATCH.json --out NEW_DIR
```

`docs/INTERFACE.md` defines the contract, the outputs (`final_hf/` and one record per optimizer step and rank, with
the top-256 ids and the student and teacher log-probabilities) and the checks.

The training is fixed by the contract (`runtime/contract.json`):

1. **Sample.** The student samples one response per prompt without `c` (temperature 1, top-p 1, at most 512 tokens,
   `enable_thinking=false`).
2. **Top-k set.** The 256 highest-probability tokens of the pre-update student at every response position come from
   a training-side forward pass. They may not come from the sampling engine.
3. **Teacher.** The teacher is the frozen initial snapshot. It scores the same tokens with `c` as the system
   message and never follows the student's updates.
4. **Loss.** The mean over all response tokens of the round of `sum_{v in top256} p(v) (log p(v) - log q(v))`.
   There is no reward or policy-gradient term.
5. **Update and publish.** One AdamW step per round (lr 5e-6), by Megatron-Core inside Miles'
   `MegatronTrainRayActor`, then a publication to the SGLang engine.

The scored workload is 32 rounds x 512 hidden MedMCQA prompts (Work holds neither the rows nor the size), every training row exactly once, in its round. Sampling may be at
most one step stale.

Rules the Judge enforces:

- The run stays a Miles Ray job: Megatron-Core from the installed `/root/Megatron-LM` in `MegatronTrainRayActor`,
  and SGLang engines started by Miles. Megatron-Core, SGLang and Ray must not be vendored, shadowed or edited;
  patching them at run time from Miles code is allowed.
- Student engines run at tensor-parallel size 1, and the trainer at tensor- and pipeline-parallel size 1. A
  data-parallel trainer holds complete weights when the step returns.
- The `OPCD_AUDIT_*` environment reaches every process.
- The teacher forward and the training GEMMs run in bf16 or wider (no FP8/INT8/FP4).
- The student never sees `c`.
- `probe` and `run` share one implementation.
- Served weights and `final_hf` are exactly the bf16 weights Megatron's steps produced.
- Models, tokenizer, context, rendering, data, loss, sampling, optimizer and workload stay fixed.
- The implementation is written natively in the checkout.

The agent may change teacher placement and batching, how the top-k set is produced, overlap within the staleness
limit, micro-batching, parallelism, kernels, SGLang configuration and the publication path.

The Judge stops `probe` after 1800 s and `run` after 3600 s. Each submission also passes an LLM integrity review
(`llm_judge/POLICY.md`) before the GPU Judge. The agent's time budget is `TASK_TIMEOUT` in the launch config; the
example configs use 86400 s (24 h).

## Evaluation

The Judge runs a probe on hidden fixed trajectories, then the full run on training rows Work does not hold, under
SGLang audit and provenance hooks.

Checks:

- **probe.** The dump matches the batch. The loss matches its float64 recomputation, and the loss, top-256 coverage
  and teacher log-probabilities match an independent float32 computation. Exactly one Megatron-Core optimizer step
  produces `final_hf`.
- **run, provenance.** One optimizer step per round inside `MegatronTrainRayActor`, starting from the pinned initial
  student, with publications bit-identical to Megatron's weights. Fingerprints of the weights and optimizer state
  must chain from step to step. `final_hf` must equal, by SHA-256 per tensor, both Megatron's last weights and what
  the engine served last.
- **run, random steps.** The steps that started from four randomly copied served weights are recomputed in float32:
  loss, top-256 set, teacher log-probabilities, and the gradient against the gradient Megatron applied. Teacher
  log-probabilities of two more random steps are recomputed from the frozen snapshot with `c`.
- **run, audit.** Every engine request is recorded inside the engine. The checks cover the sampling parameters (no
  top-log-probability options), 512 generations per round on the assigned rows without `c`, staleness, one
  publication per step, and every step's loss.

The final student, without `c`, then answers 2436 hidden held-out MedMCQA questions three times. The protocol:

- sampling at temperature 1 and top-p 1, at most 512 new tokens, `enable_thinking=false`;
- per-request seeds and deterministic inference, with repeat `r` seeded with `r`;
- a response is correct when its last `<answer>X</answer>` tag holds the right letter.

```
quality_score = mean accuracy of the final student over the 3 repeats
                - mean accuracy of the original Qwen3-1.7B (same questions and protocol, frozen: 22.73),
                in percentage points
score         = quality_score x 3600 / T     if every check passes and quality_score >= 6.5
              = 0                             otherwise (valid)
```

`T` is the wall clock of `run`, from the first student generation request to the end of the last weight
publication, taken from the Judge's records. Model loading, the final checkpoint conversion, and the Judge's weight
copies and digests (at most 60 s in total) are not counted. There is no reference time and no cap.

The Judge also reports, without scoring them:

- the answer-tag rate;
- the original model's accuracy with `c` (the in-context reference);
- a diagnostic under the paper's own evaluation protocol on its 500 test questions.

A score of 0 from failed checks or gates is still a valid result. Judge or infrastructure errors make the result
invalid.

The public suites in the Work container, `python /opt/miles_opcd_public/run_public.py --suite {probe,small,perf} --output DIR`,
run the Judge's checks on small public data, only to get the pipeline working. `small` runs 4 rounds x 16 prompts and
`perf` 8 rounds x 64 (`runtime/public_contract.json`, 512 rows that share no question with the Judge's rows). `perf`
reports `T` and the accuracy and answer-tag rate on 64 public questions, and no score.

Scheduled auto-evaluation is off. Each `sforge-submit` runs the full Judge (on its own GPU pair by default, while
Work keeps running; with `--shared-gpus` the Judge pauses Work because both use one pair) and returns its report to
the agent. After the budget ends, the supervisor evaluates the final archive once more on the Judge pair, and the
run's result is the best valid score across the run's submissions and this final evaluation. The Judge timeout is 10800 s.

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
| `h100-94gb-400w` (H100 NVL 94 GB) | qualified | `dual-cuda-sm90-nccl` | 80 GiB |
| `h100-80gb-700w` (H100 SXM 80 GB) | qualified | `dual-cuda-sm90-nccl-80g` | 79 GiB |

Each card type has its own qualification record
(`runtime/qualification.json` for NVL, `runtime/qualification.dual-cuda-sm90-nccl-80g.json` for SXM). Scores are
comparable only within one profile. The reference places the Megatron trainer and the frozen teacher on one card
and the student SGLang engine on the other.

## Running

1. **Assets.** `INFRATHON_ASSETS_ROOT=<dir> tasks/latest/miles-qwen3-opcd-2xh100-v3-seed/assets/fetch.sh` downloads:
   - `Qwen/Qwen3-1.7B` at a pinned commit, into the Hugging Face cache (`HF_HUB_CACHE` / `HF_HOME`), used as both
     student and teacher;
   - from the Infrathon dataset: the initial Megatron checkpoint (about 3.4 GB, converted with Miles'
     `tools/convert_hf_to_torch_dist.py`), the public task data that is built into the images (for Work: 512 training rows, `c`,
     public probe batch and questions; for the Judge: the same plus the scored training rows), and the Judge-only data
     (the held-out questions, the paper's test set and a hidden probe batch) from the dataset's `judge/` part; the Judge-only data is not in the repository.
2. **Images.** `INFRATHON_ASSETS_ROOT=<dir> scripts/infrathon/pull_images.sh miles-qwen3-opcd-2xh100-v3-seed` pulls the published
   Work and public Judge for this host's hardware type. It then builds the Judge's holdout layer locally from the
   Judge-only assets (`images/judge/Dockerfile.private`).

   To build from source instead:
   - build the base image with `scripts/miles_base_images/build.sh model-flat`;
   - set `MILES_UPSTREAM` to a Miles clone that contains `78527b9`;
   - run `tasks/latest/miles-qwen3-opcd-2xh100-v3-seed/scripts/build_images.sh`.
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
   SFORGE_CPUSET=auto launch/miles-opcd/run.sh --config <config.env> [--judge-gpus C,D | --shared-gpus] [--run-id ID] [--dry-run] [--foreground]
   ```

   Start the config from `launch/miles-opcd/*.env.example`. It sets:
   - the SForge version store and the agent runtime (built with `launch/build-agent-runtime.sh`);
   - the agent and model credentials: a Codex ChatGPT login, an OpenAI-compatible endpoint, or Claude Code;
   - `MODEL`, `REASONING_EFFORT`, `TASK_TIMEOUT`, `SUBMISSION_COOLDOWN`, `GPU_INDICES` (Work's NVLink pair),
     `JUDGE_GPU_INDICES` (the Judge's NVLink pair, or `shared`) and `JUDGE_PORT`. The examples use `TASK_TIMEOUT=86400`,
     `GPU_INDICES=0,1`, `JUDGE_GPU_INDICES=2,3` and `JUDGE_PORT=18634`.

   `--dry-run` prints the plan, or the first failed precondition. The launcher checks:
   - the agent runtime, the credentials and the mandatory LLM review settings (`SFORGE_LLM_JUDGE_*`);
   - the GPU profile of both pairs and its qualification record (which must not be pending), and the CPU pinning
     policy;
   - both images;
   - at least 200 GiB free under `MILES_SCRATCH_ROOT`.

During a run, Work is on an internal network. It reaches the model API and public web hosts (ports 80/443) only
through the run's relay proxy. The Judge has no network.

## Files

| path | contents |
|---|---|
| `images/work/` | Work Dockerfile, `AGENT_TASK.md` (the agent's brief), the starter `workdir/miles_lab/` (`task_entry.py`, `config/` with the placeholder loss, `notes.md`) |
| `images/judge/` | public Judge Dockerfile and `Dockerfile.private` (the host-built holdout layer) |
| `docs/INTERFACE.md` | commands, contract fields, step-record schema, what each check looks at |
| `runtime/` | the Judge's contract and Work's `public_contract.json`, qualification records per profile (metric, frozen base quality, gate, timeouts, gradient limits), public reference values, SForge binding (`sforge_entry.py`) |
| `llm_judge/POLICY.md` | integrity-review policy |
| `reference/task_entry.py` | reference entry (host only) |
| `scripts/` | `build_images.sh`, `prepare_assets.py` and `medmcqa_row_map.json` (task data from MedMCQA), `derive_row_map.py`, `run_task.py` (run supervisor), `qualify.py` |
| `assets/` | `manifest.json` and `fetch.sh` |
| `tests/` | CPU tests of the Work network policy |

Checks, oracle, evaluator and public runner: `scripts/miles_opcd/v3/`. Instrumentation hooks:
`scripts/miles_opcd/audit/`. Shared Miles tooling: `scripts/miles_common/`.

## Upstream and license

- Paper: On-Policy Context Distillation for Language Models, Ye et al.,
  [arXiv:2602.12275](https://arxiv.org/abs/2602.12275).
- Miles, `radixark/miles` at `78527b9102b3`: Apache-2.0. The base image starts from `radixark/miles`, pinned by
  digest (`scripts/miles_base_images/README.md`).
- MedMCQA, `openlifescienceai/medmcqa` at `91c6572c`: Apache-2.0 (the MedMCQA GitHub release is MIT). Questions are
  rendered with the LMOps OPCD prompt template.
- The context `c` is `opcd/system_prompts/medmcqa.txt` from
  [microsoft/LMOps](https://github.com/microsoft/LMOps/tree/main/opcd) (MIT, originally from MetaSPO). The task's
  OPCD protocol was checked against LMOps commit `6f53ef41c951de0962bbf27fe3f5819eabc8ce0e`.
- `Qwen/Qwen3-1.7B` at `70d244cc`: Apache-2.0 (per its model card).
- The task's own code is under the repository's root `LICENSE` (PolyForm Noncommercial 1.0.0).
