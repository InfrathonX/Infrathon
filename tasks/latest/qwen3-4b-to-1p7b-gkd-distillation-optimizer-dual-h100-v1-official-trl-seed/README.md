# Qwen3-4B to Qwen3-1.7B GKD Distillation Step Optimization (two H100)

This task is about the training step of off-policy generalized knowledge distillation (GKD).
A frozen BF16 Qwen3-4B teacher is distilled into a fully trainable BF16 Qwen3-1.7B student on
two GPUs, with the student sharded by FSDP (FULL_SHARD, world size 2). The agent must raise
end-to-end training-step throughput and stay numerically equivalent to the official
Hugging Face TRL 0.12.1 `GKDTrainer.generalized_jsd_loss`. This is a seed task: the Work
container starts from a working baseline that calls the pinned official TRL method, and the
agent improves it.

## The task

One timed training step contains the teacher forward pass (eval mode, no gradients,
replicated on each rank), the student forward pass, both vocabulary projections, the
generalized Jensen-Shannon divergence loss, backward, and AdamW. All of these steps count
toward throughput. The teacher forward pass is inside the timed window, so making it faster
is part of the task.

The workspace in the Work container is `/app/workdir/gkd_lab`. The agent edits:

- `solutions/objective_vNNN/candidate.py`. This file must export `build_objective(contract)`,
  which returns an objective that follows the callable contract in `runtime.api`. The
  objective takes the student and teacher head weights, their hidden states, completion-only
  labels (`-100` elsewhere) and optional biases, and returns a `GKDResult`. The harness does
  not perform either vocabulary projection. The candidate owns both projections and may fuse
  them with the loss.
- Optionally, `build_model_patch(contract)` (student) and/or `build_teacher_patch(contract)`
  (teacher). Each returns the declarative `Qwen3ModelPatch` from `runtime.api`, and the
  runtime installs it. Candidates must not export `optimize_model` or mutate model objects.
- `FINAL_VERSION`, which holds the integer of the selected version, and `notes.md`.

Only `solutions/`, `FINAL_VERSION` and `notes.md` are captured. `objective_v001` is the
baseline. It must stay unchanged as a recovery point, and each experiment goes in a new,
complete version directory.

Rules the Judge enforces:

- The candidate must keep the full observable behavior: the loss, all nine auxiliary outputs,
  the student gradients, full-parameter AdamW updates and distributed execution. It must
  support every `beta` in [0, 1] and every positive `temperature`.
- Gradients may flow only into student tensors. Teacher parameters must stay bit-for-bit
  unchanged.
- Computation must use BF16 or FP32 (TF32 accumulation is allowed). FP8, INT8, FP4 or any
  other quantized format counts as an approximate result. So do skipped terms, dropped tokens
  or vocabulary entries, top-k approximations, fabricated gradients and shortened training.
- Source admission: each version directory may contain only `candidate.py` and `__init__.py`.
  These must be small, auditable Python/Triton files that use only allowlisted imports. File
  access, network access, subprocesses, dynamic imports, encoded payloads, frame inspection
  and runtime monkey-patching are all rejected. The Work container and the Judge use the
  same policy module (`runtime/_source_policy_impl.py`).
- The trusted runtime owns the models, data, FSDP, optimizer, seeds, the amount of work,
  timing and memory probes. The candidate cannot change any of these.
- An LLM integrity review checks the submitted code against `llm_judge/POLICY.md`.

The launcher gives the agent a default budget of 24 hours (`--hours` / `--timeout`).

Public checks inside the Work container:

```bash
bash /workspace/public_tests/run_public_tests.sh --cpu-only   # CPU objective cases
bash /workspace/public_tests/run_public_tests.sh --quick      # short two-GPU smoke
bash /workspace/public_tests/run_public_tests.sh              # full public development run
```

## Evaluation

The Judge (`judge/run_judge.py`, with verifier code in `images/judge/verifier/`) evaluates
the version named by `FINAL_VERSION` on unseen workloads:

1. **Source admission.** The Judge applies the same source policy that the Work container
   uses.
2. **Objective correctness.** The Judge runs hidden CPU objective cases with different
   shapes, masks, biases, `beta` values and temperatures. Each result is compared with the
   pinned official TRL method.
3. **End-to-end formal run.** The Judge measures five private training configurations that
   vary sequence length and per-rank batch size. The records are held-out TL;DR records that
   do not overlap the public partition. For each configuration, the official TRL reference
   and the candidate run as fresh processes in four counterbalanced pairs
   (`RC, CR, CR, RC`), each with 5 warm-up steps and 10 timed steps on the same two GPUs.
   Each pair is checked for:
   - the loss trajectory, within a frozen tolerance of the paired reference;
   - auxiliary outputs, a global gradient norm computed in FP64, the AdamW state topology and
     `exp_avg`/`exp_avg_sq` witnesses, and non-zero parameter updates;
   - a frozen-teacher signature: an FP64 signature over all teacher parameters must be
     identical across ranks, unchanged from start to end, and equal between the reference
     and candidate runs;
   - memory: the candidate's peak reserved memory may be at most 1.05 times that of its
     paired reference, and at most `min(per-rank total memory) - 4 GiB`.

Throughput is the number of scored (non-ignored, shifted) completion tokens across both
ranks, divided by the slowest rank's full-step wall time. The controller counts the tokens;
the candidate cannot report them. For each configuration `r`:

```text
ratio_r = median over pairs of (TPS_candidate / TPS_official_TRL_reference)
score   = 100 * geometric_mean(ratio_r over the five configurations)
```

The score is a dimensionless speedup times 100 (100 = parity with the official reference),
and higher is better. All five configurations carry equal weight. A submission is valid only
when all ten fail-closed gates pass: source, correctness, protocol, schema, execution, loss,
memory, correctness evidence, and the two manifest gates `formal_ready` and
`formal_score_enabled` (the private workload manifest is marked as qualified and enabled for
scoring). Any failed gate makes the score zero. The reference is measured
again on the same devices in every Judge run, so the score does not depend on a frozen
device constant.

During a run the agent submits with `sforge-submit`. There is no limit on the number of
submissions, but there is a 300-second cooldown between them, and auto-evaluation is turned
off. A managed submission returns only aggregate feedback. When the agent's budget ends, the
launcher runs a standalone final Judge on the final archive, on the Judge pair (the one
shared pair in shared mode). The run's result is the best valid score across the run's submissions and this final evaluation.

## Hardware

Each run uses four GPUs by default: the Work container runs on one pair and the Judge on
another, and Work keeps running while the Judge measures. `--shared-gpus` runs both on one
pair, and the Judge pauses Work during each Judge round. Containers always see their pair as
CUDA devices `0,1`. The task does not pin a pair: the launcher uses physical GPUs 0,1 for
Work and 2,3 for the Judge unless `--work-gpus A,B` and `--judge-gpus C,D` name others.
Scores from separate mode are not directly comparable with shared-mode runs, because Work
keeps running during Judge rounds and gets more wall time.

| Hardware type | Support |
| --- | --- |
| `h100-94gb-400w` (H100 NVL 94 GB) | qualified |
| `h100-80gb-700w` (H100 SXM 80 GB) | qualified |

The capability envelope, which `scripts/preflight.py` (on each pair) and the Judge controller
both enforce, requires:

- compute capability 8.0 or newer;
- at least 79 GiB of total memory per card;
- both cards idle, with no other compute owner;
- NCCL;
- NVLink (the launcher always requires it; `scripts/preflight.py --allow-non-nvlink` waives it
  only when you run the preflight by hand).

On the largest configurations the official reference peaks at about 75,500 MiB reserved per
rank on an H100 SXM pair (up to about 76,865 MiB across the H100 NVL qualification runs),
because each rank holds a replicated frozen 4B teacher alongside the sharded student. The
second memory bound (`min(per-rank total memory) - 4 GiB`) depends on the card: on an H100
NVL 94 GB the 1.05 ratio is the binding limit, but on an H100 SXM 80 GB (PyTorch reports
81,081 MiB per card) the ceiling is 76,985 MiB. On the largest configuration the reference
measured about 75,454 MiB on an H100 SXM pair, so a candidate may reserve only about 2% more
than the reference there.

## Running

1. **Assets.** Run `tasks/latest/qwen3-4b-to-1p7b-gkd-distillation-optimizer-dual-h100-v1-official-trl-seed/assets/fetch.sh`.
   It fetches into `INFRATHON_ASSETS_ROOT` (default `~/.cache/infrathon/assets`) and verifies
   every file:
   - `Qwen/Qwen3-1.7B` (student) at revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` and
     `Qwen/Qwen3-4B` (teacher) at revision `1cfa9a7208912126459214e8b04321603b3df60c`, from
     their Hugging Face repositories into the Hugging Face cache;
   - from the Infrathon dataset, the pretokenized TL;DR GKD data `tldr_gkd_v1/`. Work
     mounts its public partition and the Judge mounts its private partitions. The dataset
     also supplies, in its `judge/` part, the Judge-only workload manifest
     `judge/private_data/hidden_cases.json`; Judge-only files are not in the repository.

   You can override the mounted data paths with `GKD_DATA_HOST_ROOT` and
   `GKD_PRIVATE_DATA_HOST_ROOT`; the models are always mounted from the Hugging Face cache.
2. **Images.** Run
   `INFRATHON_PULL=1 tasks/latest/qwen3-4b-to-1p7b-gkd-distillation-optimizer-dual-h100-v1-official-trl-seed/scripts/build_images.sh`.
   This pulls the published Work image and public Judge image and builds the holdout Judge
   layer locally. Without `INFRATHON_PULL`, everything builds from source: the Work image is based
   on `pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel` and verifies the TRL 0.12.1 wheel and
   source hashes.
3. **Run.**

   ```bash
   launch/qwen3-gkd/run.sh --model MODEL [--work-gpus A,B --judge-gpus C,D] \
       [--agent codex|claude-code] [--dry-run]
   ```

   - Defaults: Work on GPUs 0,1, Judge on GPUs 2,3, 24 hours, effort `max`, Judge port
     18614; `--work-gpus`, `--judge-gpus`, `--hours`, `--effort` and `--judge-port` change
     them. The four GPUs must be distinct.
   - `--shared-gpus` runs Work and Judge on the Work pair, with Work paused during each Judge
     round; `--gpus A,B` is the same as `--shared-gpus` on pair A,B. In a launch config or
     the environment, `GKD_GPU_INDICES` is the Work pair and `GKD_JUDGE_GPU_INDICES` is the
     Judge pair or `shared`.
   - `--dry-run` resolves and checks the plan without touching the GPUs.
   - There is no start delay unless `MIN_DELAY_SECONDS` is exported. The launcher waits for
     all of the run's GPUs to be idle. Concurrent runs on one host need their own GPUs and
     `--judge-port`; an 8-GPU host fits two runs in the default mode.
   - To evaluate an archive with the final Judge only, on one pair, run
     `GKD_GPU_INDICES=A,B tasks/latest/qwen3-4b-to-1p7b-gkd-distillation-optimizer-dual-h100-v1-official-trl-seed/scripts/eval_archive.sh ARCHIVE.tar.gz [RUN_ID]`.
   - To measure saved candidate versions again as one serial series, run
     `launch/qwen3-gkd/replay.sh`. It wraps `scripts/replay_candidates.sh`, which takes
     `--origin-run`, `--measurement-set`, `--replay-run-id`, `--candidate` / `--from-v` / `--to-v`, `--dry-run`
     and `--resume`.

   The launcher checks these prerequisites:
   - the repository virtualenv (`.venv`);
   - a host agent runtime (`SFORGE_AGENT_RUNTIME_DIR`, built with
     `launch/build-agent-runtime.sh`);
   - model credentials: a Codex `auth.json` (`--auth`), a Claude Code OAuth token file
     (`--claude-token`), or an OpenAI-compatible endpoint (`MODEL_API_BASE_URL` with
     `MODEL_API_KEY_FILE`);
   - integrity-review settings (`SFORGE_LLM_JUDGE_MODE=enforce` and the
     `SFORGE_LLM_JUDGE_*` reviewer variables).

   Launch settings can also come from `--config FILE` (see
   `launch/mlsysforge.env.example`).

## Files

| Path | Contents |
| --- | --- |
| `tasks/` | SForge task spec: Judge command, timeouts, resources, submit paths, capability envelope |
| `images/work/` | Work Dockerfile, `AGENT_TASK.md`, the seed workspace `workdir/gkd_lab/` (`runtime/` trusted ABI and harness, `solutions/objective_v001/` baseline), and `public_tests/` |
| `images/judge/` | Public Judge Dockerfile, `Dockerfile.private` (holdout layer), and `verifier/`: formal controller, correctness checks, source admission |
| `judge/` | Judge entry point (`run_judge.py`) |
| `llm_judge/POLICY.md` | Integrity-review policy: permitted techniques and rejected shortcuts |
| `assets/` | Asset manifest and `fetch.sh` |
| `scripts/` | Image build, preflight, final-archive and replay evaluation, data preparation (`prepare_tldr_gkd_data.py`), layout and protected-hash checks |
| `host_reference/` | Host-only reference variants and the oracle (`FINAL_ORACLE`, `ADMISSIBLE_ORACLE`, `forced_judge/`), used to show that headroom exists. It is not part of either image or of scoring |
| `tests/` | Maintainer tests for the task contract, the Judge and the launchers |

## Oracle

GKD has no official paper code. The repo-optimized solution is Liger-Kernel
(commit `780e76b8`). Its stock JSD loss does not match TRL when
`beta != 0.5`, so both oracle arms use a TRL-exact, Liger-style chunked
objective.

| Candidate | Score | Gate status | Hardware |
| --- | ---: | --- | --- |
| seed / official TRL reference | about 100 (99.53 to 100.07) | all PASS | H100 NVL pairs |
| best admissible candidate: `host_reference/gkd_oracle` (objective only) | 105.18 | 10/10 PASS on the normal Judge | H100 NVL pair |
| strongest candidate: `host_reference/full_liger_gkd` (plus Liger Qwen3 kernels for student and teacher); not admissible, because it imports `liger_kernel` | 117.73 / 117.64 in two Judge runs | 10/10 PASS with the source-admission scan and LLM review turned off (`host_reference/forced_judge/`) | H100 NVL pair |

## Upstream and license

| Component | Source | Pin | License |
| --- | --- | --- | --- |
| Hugging Face TRL (baseline loss) | `https://github.com/huggingface/trl`, tag `v0.12.1` | commit `c1e9ea8ecfdaf1486e2fc84bc55379b2338d5320`; wheel SHA-256 `502b958a…5c5a` | Apache-2.0 |
| Qwen3-1.7B (student) | `Qwen/Qwen3-1.7B` | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` | Apache-2.0 |
| Qwen3-4B (teacher) | `Qwen/Qwen3-4B` | `1cfa9a7208912126459214e8b04321603b3df60c` | Apache-2.0 |
| TL;DR data | [`trl-lib/tldr-preference`](https://huggingface.co/datasets/trl-lib/tldr-preference) | `7ae482b332979f8ce40d2a6b5f78f9eddf137efc` | No license declared on the Hub; follows the upstream source |

The data is derived with `scripts/prepare_tldr_gkd_data.py` and the pinned Qwen3 tokenizer.
Each record is one prompt plus its chosen completion. Prompt and padding positions are
masked, and rejected completions are not used. The derived pretokenized tensors are published
in the Infrathon dataset.

The task's own code is under the repository's root `LICENSE` (PolyForm Noncommercial 1.0.0).
