# miles_diffusion: multi-teacher on-policy distillation for SD3.5-Medium on two GPUs

Hardware: two NVIDIA H100 GPUs on one NVLink pair, assigned to this run. Their memory and the CPUs you get depend on the host: check them with `nvidia-smi` and `nproc` and size your configuration from what you see. The container has 256 GiB of host memory.

## Goal

`repo/` is a checkout of miles_diffusion: RL post-training for diffusion models with FSDP2 training and
sglang-diffusion rollout engines (Ray). For Stable Diffusion 3.5 Medium it already trains a LoRA student with
Flow-GRPO or DiffusionNFT against reward models, including CPS stochastic sampling, SDE-step windows, and
publishing the LoRA to the engines after every rollout. It has no teacher path at all: its only reference models are
the frozen base under the adapter or an EMA of the student.

Implement **multi-teacher on-policy distillation** natively in miles_diffusion (command-line options, a recipe under `repo/scripts/`,
docs): two frozen LoRA teachers, one for OCR and one for aesthetics, are distilled into one student LoRA
on the student's own sampling trajectories. The score is how close the run brings the student to its teachers on the
teachers' own tasks, per hour of the timed training loop (see Score).

## Starting point

- `task_entry.py run`, `config/` and `repo/` are yours; `docs/INTERFACE.md` defines the command, the contract file,
  the replay protocol, the required outputs and per-step records, and the precise checks.
- The starter maps the contract onto miles_diffusion's existing options and stops with `NotImplementedError`: there
  is no teacher path, no distillation loss and no record writing yet.
- Models are read-only under `/models`: the SD3.5-Medium base and the two teacher adapters (exact paths are in the
  contract), and the task-metric scorers under `/models/scorers`. Public data (rows, student initialisation, recorded
  probe and replay responses) is under `/data/public`.
- The sandbox denies `pidfd_getfd`: CUDA-IPC weight sync from memory allocated with
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` fails with "Operation not permitted". Leave expandable segments
  off for anything you share with the engines.

## The training you must implement

Every round uses `rows_per_round` contract rows `{input, metadata: {data_source, row_id, seed}}`:

1. **Sample (student, on-policy).** One engine request per row with `num_outputs_per_prompt` 8: the row's prompt
   verbatim, the row's `seed`, 512x512, 10 steps, guidance 1.0 (no CFG, no negative prompt, no true CFG),
   `rollout_sde_type="cps"`, noise level (eta) 0.8, `rollout_log_prob_no_const=True`, return the DiT trajectory and
   the denoising environment, SDE window `[s, s+1, s+2]` with `s = numpy.random.default_rng(seed).integers(0, 3)`
   (steps outside the window are ODE), `extra_sampling_params` unset. The 8 outputs of a row share its window.
   The engine runs the student's current weights.
2. **Pairs.** Each output gives 3 pairs `(x_i, t_i, t_{i+1}, cond)`, one per window step `i`, with `x_i`, the step's
   timesteps and the conditioning exactly as the engine returned them (you may re-encode the prompt instead of
   reusing the engine's conditioning if you reproduce the engine's truncation).
3. **Teacher.** The row's `data_source` selects its teacher through the command line
   (`--teacher-model LABEL=PATH`, repeatable; labels are arbitrary strings — do not hard-code them). The teacher
   predicts a velocity at the **same** `x_i`, timestep and conditioning, at guidance 1.0. Teachers are frozen and the
   student's adapter must not be active in a teacher forward.
4. **Loss.** Both velocities go through the same CPS transition. With `sigma = sigma_i`, `sigma' = sigma_{i+1}` (the
   engine's own schedule), `s_t = sigma' * sin(0.8 * pi / 2)` and
   `mu(v) = (x_i - sigma v)(1 - sigma') + (x_i + (1 - sigma) v) * sqrt(sigma'^2 - s_t^2)`:

   ```
   KL_i = mean over the D latent elements of (mu(v_student) - mu(v_teacher))^2 / (2 s_t^2)
   ```

   (the exact Gaussian KL between the two transitions divided by D). Gradients flow only through the student. An
   optimizer step's loss is the mean of `KL_i` over all its pairs across data-parallel ranks.
5. **Update.** Student = LoRA r 32, alpha 64 on the attention projections listed in the contract, initialised from the
   contract's `student_init`. AdamW lr 1e-4, betas (0.9, 0.999), eps 1e-8, weight decay 1e-4, gradient clipping 1.0,
   fp32 optimizer state. Two optimizer steps per round: the first half of the round's rows, then the second half.
   **Every optimizer step must update** (a gradient scaler skipping a step is a violation).
6. **Publish.** After each round's second step, publish the student (only the student) to every rollout engine, once:
   the engines must receive exactly the student's LoRA after that step (fp32, or its fp16 rounding). Round `r+1` may
   only be requested after round `r`'s publication is applied to all engines (lag 0).

The contract gives the rows, `rounds` and `rows_per_round`; the public and the scored runs differ in all three, so read them
from the contract and assume no fixed size. Every row exactly once, in order; no other generation requests.

## Timed part: replay

In the scored timing the engine does not generate. With `generate_url` set in the contract, send every generate
request to that URL instead of your engines; it answers with the exact bytes a real engine produced for that request.
After a round's publication is applied to all of your (still real) engines, `POST {generate_url}/replay/published?round=r`;
the next round's responses are released only then. Everything between a round's release and that call is yours:
parsing responses, teacher scoring, the two optimizer steps, dumps, publication. Generation speed is not scored.

Setup is not timed but is capped: from the replay server's start to the release of the first timed round (process
start-up, model and engine loading, compilation, CUDA graphs, caches, and the untimed probe round 0) you have at most
600 s, otherwise the submission scores 0. Timed rounds count in full, including any recompilation or other one-time
work that lands in them.

## What you may change

**The run must stay a miles_diffusion training job.** It is launched by `repo/train_diffusion.py` (the MD driver);
generation goes through its Ray `RolloutManager` rollout path (`miles.rollout.base_types.call_rollout_fn` calling
your rollout function); training happens in its FSDP training actor (`FSDPTrainRayActor.train`), and every
publication to the engines is made by `FSDPTrainRayActor.update_weights`. OPD is switched on through MD's arguments
(e.g. a new `--loss-type`, `--teacher-model LABEL=PATH`). A run outside this path scores 0: the point is that
miles_diffusion users get the feature and its speed.

Inside that path you may change anything in `repo/`, `task_entry.py` and `config/`: new loss / teacher /
data-conversion modules, the actor, the rollout manager, where and how the teachers are hosted (merged copies,
adapters, other GPU), batching and overlap of teacher and student work, colocation, memory residency and offload,
kernels and compilation, checkpointing, response parsing, the publication path, and engine flags from this list:
number and placement of engines, `--sglang-server-concurrency`, offload/residency, VAE slicing.

You may not:

- replace that path with a separate trainer that drives the engines itself, even one that lives in `repo/`;
- change the engine source or its pinned settings (fp16 DiT, default attention backend, LoRA merge mode `auto`,
  batching 1 / 0 ms, no caching/TeaCache/Spectrum env or flags), use rollout patch groups or monkey-patch engine
  internals;
- change the models, teachers, student initialisation, data, row order, sampling parameters, the loss, optimizer,
  number of steps and rounds, or the publication schedule.

**Your own implementation.** You may read papers and other projects' code to understand the method, but the
implementation you submit is written natively in this checkout. Do not copy in an existing implementation of the
feature from elsewhere (another framework, or another version of this one), whether as copied files, a vendored
package or a wrapper that calls it.

## Correctness requirements

- **Precision.** Training precision is **fp16 or fp32** (TF32 allowed); bf16 is too far from the evaluator's float32
  reference and fails the correctness checks.
- **Instrumentation.** Both launches run with `OPD_AUDIT_DIR` set; keep it in the environment of every process you
  start (Ray workers, engine servers and their workers).
- **Exact weights.** `final_lora.safetensors` must be bit for bit the student after
  the run's last optimizer step, i.e. the last step record's `lora_post`, which holds exactly the trained tensors (the
  keys and shapes of that record's `lora_pre`). Training or changing the LoRA after the last step, folding extra
  training into a step, or adding adapter tensors fails the checks.
- **Correct steps.** Every step record (`docs/INTERFACE.md`) is what was trained on, with the contract's optimizer
  settings, and the probe step (pairs, transition means, KL, gradient) must agree with an independent float32
  recomputation.
- **Training signal.** The contract's KL objective only: no image reward, OCR or PickScore signal in training, and
  both routes' pairs count equally in every step.
- **Records.** Every step record is read with `torch.load(weights_only=True)`: plain tensors, numbers, strings,
  lists, tuples and dicts only.
- **Submitted tree.** It contains `task_entry.py` and `repo/train_diffusion.py`, no symlinks and no weight-like files
  (`.safetensors`, `.pt`, `.bin`, `.ckpt`, `.msgpack`) anywhere.
- **Time limits.** Setup within 600 s (above), and each launch within its deadline (`docs/INTERFACE.md`).

## Public validation

```bash
python /opt/opd_public/run_public.py --suite probe --output /scratch/public/probe-001
python /opt/opd_public/run_public.py --suite small --output /scratch/public/small-001
python /opt/opd_public/run_public.py --suite perf  --output /scratch/public/perf-001
```

`probe` checks a probe round against the float32 oracle, `small` a 4-round real run, and `perf` a small replay workload
with its `T`; `perf --quality` adds a short real run and reports its held-out KL and task metric on public prompts. They
run the Judge's checks on small public data and report no score: a run this short shows that the pipeline works, not
how fast it will be at scale or how much the student will learn. The Judge repeats the checks on hidden data.

Suite outputs rotate. The output of the suite you run and of the suite before it stay complete; when a suite starts,
older suite outputs in the same directory keep their reports, logs and JSON records but lose their bulky files
(checkpoints, weights and weight copies, step tensors, generated images). Only directories these suites created are
touched. Add `--keep-artifacts` to pin a suite's output so it is never rotated; it then counts against your disk
budget until you delete it.

## Score

The evaluator measures the final student of the real run on held-out OCR and PickScore prompts. Each prompt
gets one 512x512 image (40-step flow-matching ODE, guidance 1.0, a fixed seed per prompt), scored by OCR
accuracy `1 - edit distance / length` of the quoted text (PaddleOCR) or by PickScore_v1. The original model and each
route's teacher were measured the same way once and frozen.

If every check passes:

```
closure_r     = (S_r - B_r) / (T_r - B_r)         r in {OCR route, PickScore route}
                S_r: the final student's mean metric on route r; B_r: the original model; T_r: route r's teacher
quality_score = mean over the two routes of closure_r
score         = quality_score × 3600 / T     if the KL gate passes and closure_r ≥ 0.25 on both routes
              = 0                             otherwise
```

- `closure_r` is the share of the original-to-teacher gap closed on that route (not capped); each route must clear
  0.25 on its own.
- KL gate: on each route, the evaluator's held-out teacher-student KL of the final student must have dropped by at
  least `ln(KL_base / KL_final) ≥ 0.364` nats on the OCR route and `≥ 0.690` on the PickScore route (averaged over
  the evaluator's sampling seeds; `KL_base` is the original model's, frozen).
- `T`: median over hidden replay segments of the time from the segment's first release to its
  last `published` call, on the evaluator's clock.
- A faster loop and a student that learns more both count; there is no reference time and no cap.

## Submission evaluation policy

**The public suites are only for getting your pipeline working and passing its checks.** The scored run is a different
and much larger workload, so public timings, memory use and quality do not predict your score. Think about how your
pipeline behaves at a larger scale and how to optimize it, and measure real speed, memory use and learning with
submissions.

Each `sforge-submit` runs your code through the whole pipeline on hidden data, from a clean copy of your submission: a
launch with a probe round and timed replay, a complete real training run, the correctness checks, and the evaluation of
the final student. A failure at any stage ends the evaluation with a score of 0, and the report lists the failed checks
(submit with `sforge-submit --details` to see each check's message). The Judge stops the first launch after 1500 s and
the second after 2700 s, start-up included, and untimed setup in the first launch is capped at 600 s. A submission takes
far longer than a public suite and uses part of your time budget, so plan them. Keep your strongest version in the
workspace when the budget ends.

Your final result is the best valid score you reach within the time budget.

## Submission workflow

1. Get the pipeline working with the public suites.
2. Stop every GPU process (Ray, engines).
3. Submit with `sforge-submit`.

Write large files under `/scratch` (the per-step records of a long run take tens of GB), never into the submitted tree.
Keep design notes and measurements in `notes.md`.

**Disk.** `/scratch` is on a disk shared with the other runs on this host. Keep what you store there under
**250 GB** at any time (check with `du -sh /scratch`): delete checkpoints, weight copies, Ray spill and
public-suite outputs you no longer need. Filling the shared disk breaks your run and the others.
