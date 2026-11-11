# Qwen3-1.7B ORPO Training Optimizer (two GPUs)

This task measures how much an agent can speed up one complete ORPO (Odds Ratio
Preference Optimization) training step for `Qwen/Qwen3-1.7B`: BF16
full-parameter training (not LoRA) under FSDP1 `FULL_SHARD` on two GPUs, with a
task-owned AdamW optimizer. The agent replaces the ORPO objective and may
supply a declarative patch for some Qwen3 layers; the model, data, distributed
setup, optimizer and timing stay under the trusted runtime. It is a seed task:
the workspace starts from a working baseline that calls the official Hugging
Face TRL 0.12.1 ORPO methods, and the agent improves on it.

## The task

The agent works in `/app/workdir/orpo_lab` inside the Work container and reads
`AGENT_TASK.md` there. The evaluated file is
`solutions/objective_vNNN/candidate.py`, where `NNN` is the number in
`FINAL_VERSION`. It must export:

```python
def build_objective(contract): ...          # required
def build_model_patch(contract): ...        # optional
```

`build_objective` returns a module that obeys the types and callable contract
in `runtime/api.py`: given the tied output weight, the final hidden states of
the concatenated chosen and rejected sequences, completion-only labels for the
odds-ratio term and a separate full-sequence target for the chosen NLL term, it
returns a differentiable loss and nine auxiliary values. `build_model_patch`
may return a `Qwen3ModelPatch` with RoPE, RMSNorm-forward and SwiGLU-forward
callables; the runtime installs it and re-checks the model topology and tied
weights. The starting solution `objective_v001` only selects the pinned TRL
baseline and must be kept unchanged as a recovery point; each experiment goes
in a new complete version directory.

Rules the Judge enforces:

- Preserve the complete observable behavior: loss, all auxiliary values,
  gradients for every parameter, optimizer updates and distributed execution.
  Bitwise identity with the baseline is not required.
- Compute in BF16 or FP32 (TF32 accumulation and FP32 upcasts are allowed). FP8,
  INT8, FP4 or other quantized formats count as approximation.
- No skipped terms, partial vocabularies or sequences, fabricated or rescaled
  gradients, replayed outputs, or shortened training. No specialization to
  public shapes or data.
- Source admission: each version directory may hold only `candidate.py` and
  `__init__.py`, using the imports the public source-admission check accepts.
  No file or network access, subprocesses, dynamic imports, encoded payloads,
  frame inspection, monkey-patching, or access to evaluator state. Candidates
  must not mutate model attributes or export `optimize_model`.
- Only `solutions/`, `FINAL_VERSION` and `notes.md` are captured. The agent
  records hypotheses and public results in `notes.md`.

The Work container has the public tests at `/workspace/public_tests`
(`run_public_tests.sh --cpu-only`, `--quick` for a short two-GPU smoke test,
no flag for the full public run, `--candidate-version N` to test a retained
version). The task JSON enables internet access for Work through a restricted
proxy. The launcher's default agent budget is 24 hours.

An LLM integrity review (`llm_judge/POLICY.md`) checks submissions for scope
violations such as the ones listed above.

## Evaluation

The Judge runs the selected candidate and the official TRL reference in the
same image, harness and lifecycle on the same two GPUs, in fresh processes, and
compares them. Each scored workload runs five warm-up steps and ten timed full
optimizer steps (forward, objective, backward with FSDP communication, AdamW
step) for four reference/candidate pairs in `RC, CR, CR, RC` order.

Gates (any failure makes the submission invalid with score 0):

- source admission, protocol and execution integrity;
- loss trajectory, all nine auxiliary outputs, complete finite FSDP gradient
  coverage and global gradient norm, AdamW state topology and moments, and
  non-zero parameter updates at warm-up and at the final step, each within
  frozen tolerances;
- memory: candidate peak reserved memory at most 1.02 times the paired
  reference, and at most the smallest card's total memory minus 4 GiB. Both
  limits are measured on the devices that ran the pair.

Throughput for each role is the number of scored (shifted, non-ignored)
completion tokens across both ranks divided by the slowest rank's full-step
wall time. For each workload, the ratio is the median of candidate/reference
throughput across the pairs, and

```text
score = 100 * geometric_mean(ratio over all private workloads)
```

with equal weights. An unmodified seed scores about 100; higher is better.
The private workloads are end-to-end training rows on held-out TL;DR preference
records that vary the stored sequence length and the per-rank pair batch; they
are not the public development workloads.

Interactive submissions: auto-evaluation is off. The agent submits with
`sforge-submit` as often as it likes, subject to a 300-second cooldown, and
gets aggregate feedback only. By default the Judge has its own GPU pair and
Work keeps running while it evaluates. With `--shared-gpus`, Work and Judge
share two GPUs, and a managed Judge pauses the Work container during
evaluation and resumes it afterwards. When the run ends, the launcher runs a
fresh Judge on the final archive on the Judge pair (the one shared pair in
shared mode); the run's result is the best valid score across the run's submissions and this final evaluation.
The Judge timeout is 12,000 seconds.

## Hardware

One run uses four GPUs by default: Work runs on one pair and the Judge on
another. `--shared-gpus` runs both on one pair, with Work paused during each
Judge round. Scores from separate mode are not directly comparable with
shared-mode runs, because Work keeps running during Judge rounds and gets more
wall time. The capability envelope for each pair is two CUDA GPUs with compute capability 8.0 or newer, at least 60 GiB of memory each,
BF16 and NCCL. The 60 GiB floor covers the largest scored workload with
headroom. NVLink is used for
qualification, and the launcher's preflight requires it (`scripts/preflight.py
--allow-non-nvlink` waives it only when you run the preflight by hand). Work and Judge each get 16 CPUs and
96 GB of host memory.

| Hardware type | Status |
| --- | --- |
| `h100-94gb-400w` (H100 NVL 94 GB) | qualified |
| `h100-80gb-700w` (H100 SXM 80 GB) | qualified |

The score is a ratio against the reference measured live on the same devices,
so it carries no device-specific constant. Absolute throughput is comparable
only on the same hardware.

## Running

1. Assets: `tasks/latest/qwen3-1p7b-orpo-training-optimizer-dual-h100-v4-cross-matrix-seed/assets/fetch.sh`
   downloads into `INFRATHON_ASSETS_ROOT` (default `~/.cache/infrathon/assets`):
   - `Qwen/Qwen3-1.7B` at revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`
     from Hugging Face, into the Hugging Face cache (`HF_HUB_CACHE`, not the
     asset root; the whole repository folder is mounted read-only);
   - the public pretokenized TL;DR preference data (`public/tldr_preference_v1`)
     and the Judge-only private partitions (`judge/tldr_preference_cross_v4`;
     "cross" is the cross matrix of sequence lengths x per-rank pair batches,
     "v4" the task version) and `judge/private_data/hidden_cases.json` from
     the Infrathon dataset. The Judge-only files come from the dataset's
     `judge/` part and are not in the repository.

   The launcher reads the data from `INFRATHON_ASSETS_ROOT` when it is set; the
   locations can also be given as `ORPO_MODEL_HOST_PATH`,
   `ORPO_DATA_HOST_ROOT` and `ORPO_PRIVATE_DATA_HOST_ROOT`.
2. Images: `INFRATHON_PULL=1 tasks/latest/qwen3-1p7b-orpo-training-optimizer-dual-h100-v4-cross-matrix-seed/scripts/build_images.sh`
   pulls the published Work and public Judge images and builds the holdout
   Judge layer locally. Without `INFRATHON_PULL=1` it builds everything from source
   (base image `pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel`).
3. Run:

   ```bash
   launch/qwen3-orpo/run.sh --model MODEL [--work-gpus A,B --judge-gpus C,D] [--hours N]
   ```

   Defaults: Work on GPUs 0,1, Judge on GPUs 2,3, 24 hours, effort `max`,
   Judge port 18613; `--work-gpus`, `--judge-gpus`, `--hours`, `--effort` and
   `--judge-port` change them. The four GPUs must be distinct. `--shared-gpus`
   runs Work and Judge on the Work pair, with Work paused during each Judge
   round; `--gpus A,B` is the same as `--shared-gpus` on pair A,B.
   `--work-gpus` (or `--gpus`) sets `ORPO_GPU_INDICES` and looks up
   `ORPO_GPU_UUIDS` itself; `--judge-gpus` sets `ORPO_JUDGE_GPU_INDICES`, and
   the launcher looks up the Judge UUIDs unless `ORPO_JUDGE_GPU_UUIDS` is set.
   `ORPO_JUDGE_GPU_INDICES=shared` in a launch config selects shared mode.
   `--agent codex`
   (default) or `--agent claude-code` with `--claude-token FILE`; `--auth FILE`
   for a Codex `auth.json`; `--config FILE` for a launch config
   (`launch/mlsysforge.env.example`). There is no start delay unless
   `MIN_DELAY_SECONDS` is exported; the launcher waits for all of the run's GPUs
   to be idle before it takes per-GPU locks. Runs in parallel on one host need
   different GPUs and `--judge-port`; an 8-GPU host fits two runs in the
   default mode. The launcher checks for Docker, `nvidia-smi`,
   `jq`, `flock`, the repository virtualenv, the built images, the pinned agent
   runtime and the agent credentials, and runs a frozen preflight on each GPU
   pair and the data. The LLM integrity review is configured with the `SFORGE_LLM_JUDGE_*`
   settings in the launch config.

   Eval only: `tasks/latest/qwen3-1p7b-orpo-training-optimizer-dual-h100-v4-cross-matrix-seed/scripts/eval_archive.sh ARCHIVE.tar.gz [RUN_ID]`
   runs one Judge on a submission archive on the one pair in `ORPO_GPU_INDICES`
   (also set `ORPO_GPU_UUIDS` to that pair's two GPU UUIDs; same data variables).
   `launch/qwen3-orpo/replay.sh` replays saved candidate versions of a run
   from a version store as one serial curve (`--origin-run`,
   `--measurement-set`, `--replay-run-id`, `--candidate`/`--from-v`/`--to-v`,
   `--dry-run`, `--resume`).

## Files

| Path | Contents |
| --- | --- |
| `tasks/` | SForge task spec (`qwen3_1p7b_orpo_training_optimizer_dual_h100_v4.json`) and base-image declaration |
| `images/work/` | Work image: `AGENT_TASK.md`, the `orpo_lab` workspace (trusted `runtime/`, seed `solutions/`), public tests |
| `images/judge/` | public Judge image, verifier and controller, holdout layer (`Dockerfile.private`), protected-file hashes |
| `judge/` | Judge adapter |
| `llm_judge/POLICY.md` | integrity review policy |
| `scripts/` | image build, launch engine, preflight, archive evaluation, candidate replay, dataset preparation |
| `assets/` | asset manifest and `fetch.sh` |
| `host_reference/` | maintainer-only positive controls and the oracle (`FINAL_ORACLE`, `forced_judge/`); not in any image |
| `tests/` | task tests (`pytest`) |

## Oracle

The oracle is the repo's own optimized solution, `host_reference/full_liger_qwen3`
(the Liger fused-linear ORPO loss plus Liger Qwen3 RoPE/RMSNorm/SwiGLU, Liger-Kernel
commit `780e76b8`). It imports `liger_kernel`, so participant source admission
rejects it. It is measured on the full formal Judge in a local-only derived image
that bypasses only the source-admission scan and the LLM review
(`host_reference/forced_judge/`).

| Candidate | Score | Gate status | Runs and hardware |
| --- | ---: | --- | --- |
| seed / official TRL reference | about 100 (99.56 to 100.29) | all gates pass | 5 Judge runs (qualification and seed replays) |
| best admissible candidate | none (every Liger variant imports `liger_kernel`) | n/a | n/a |
| full Liger (`FINAL_ORACLE`) | 123.10 / 127.11 (mean 125.10) | all other gates pass; source scan and LLM review turned off | 2 Judge runs, H100 NVL pairs |

## Upstream and license

- Hugging Face TRL `v0.12.1` (commit `c1e9ea8ecfdaf1486e2fc84bc55379b2338d5320`),
  Apache-2.0. The baseline calls the installed
  `ORPOTrainer.get_batch_logps` and `ORPOTrainer.odds_ratio_loss`; the image
  build verifies the wheel and source hashes. The timed workload is these
  official objective methods inside a task-owned tensor and FSDP harness, not
  the full upstream `ORPOTrainer` lifecycle.
- `Qwen/Qwen3-1.7B` at revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`,
  Apache-2.0.
- Preference data: `trl-lib/tldr-preference` at revision
  `7ae482b332979f8ce40d2a6b5f78f9eddf137efc`, pretokenized with the pinned Qwen3
  tokenizer (`scripts/prepare_tldr_data.py`,
  `scripts/prepare_tldr_cross_matrix.py`). The source repository declares no
  license on the Hugging Face Hub
  ([dataset page](https://huggingface.co/datasets/trl-lib/tldr-preference));
  its license follows the upstream source. The derived pretokenized tensors
  are published in the Infrathon dataset.
- `host_reference/` uses Liger-Kernel at commit
  `780e76b8115bfd7b78592569853b245a33cc7bf0` as a maintainer positive control
  ([linkedin/Liger-Kernel](https://github.com/linkedin/Liger-Kernel), BSD-2-Clause).
- The task's own code is under the repository's root `LICENSE` (PolyForm Noncommercial 1.0.0).
