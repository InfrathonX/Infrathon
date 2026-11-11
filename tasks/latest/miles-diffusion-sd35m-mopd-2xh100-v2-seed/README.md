# SD3.5-Medium Multi-Teacher On-Policy Distillation in miles_diffusion

miles_diffusion is an RL post-training framework for diffusion models (FSDP2 training, sglang-diffusion rollout
engines, Ray). For Stable Diffusion 3.5 Medium it trains a LoRA student with Flow-GRPO or DiffusionNFT, but it has no
teacher path. The agent implements multi-teacher on-policy distillation (DiffusionOPD) natively in miles_diffusion:
the official DiffusionOPD OCR and aesthetics LoRA teachers are distilled into one student LoRA on the student's own
CPS (coefficients-preserving sampling) trajectories. It then makes the timed training loop fast. This is a seed
task: the starting point is a working miles_diffusion checkout whose task entry maps the contract onto existing
options and then stops with `NotImplementedError`.

## The task

The Work container's root is `/app/workdir/miles_lab`. The agent edits and submits `repo/` (miles_diffusion),
`task_entry.py`, `config/` and `notes.md`. The Judge runs
`python task_entry.py run --contract CONTRACT.json --out NEW_DIR`; `docs/INTERFACE.md` defines the contract,
the replay protocol, the outputs (`final_lora.safetensors`, one record per optimizer step) and the checks.

The training is fixed by the contract (the Judge's `runtime/contract.json`; Work gets the smaller
`runtime/public_contract.json`, and its brief does not state the scored sizes):

- 64 rounds of 8 rows. Each row is sampled 8 times by the student on the rollout engines (512x512, 10 steps,
  guidance 1.0, CPS with eta 0.8, a 3-step SDE window chosen from the row seed).
- Each output gives 3 pairs. The row's `data_source` selects a frozen teacher, which is passed on the command line
  as `--teacher-model LABEL=PATH` with arbitrary labels. The loss is the exact Gaussian KL between the student's and
  the teacher's CPS transitions at the same state, averaged over latent elements.
- The student is a rank-32, alpha-64 LoRA initialised from `student_init`, trained with AdamW (lr 1e-4).
  Each round has two optimizer steps, and the student alone is published to every engine after the second one
  (lag 0).

Rules the Judge enforces:

- The run stays a miles_diffusion training job: `repo/train_diffusion.py`, generation through
  `call_rollout_fn`, training in `FSDPTrainRayActor.train`, publication through
  `FSDPTrainRayActor.update_weights`. The checks record this inside the processes.
- The engine source and its pinned settings stay unchanged: fp16 DiT, default attention backend, LoRA merge mode
  `auto`, no caches, no rollout patch groups, no monkey-patching of engine internals.
- Models, data, row order, sampling parameters, loss, optimizer and schedule stay fixed.
- Training runs in fp16 or fp32; bf16 fails the checks.
- `final_lora.safetensors` is bit for bit the last step's post-update LoRA.
- No image-reward, OCR or PickScore signal enters training.
- The submitted tree contains no symlinks and no weight-like files.
- The implementation is written natively in the checkout, not copied from another framework.

Everything inside that path may change: teacher hosting, batching and overlap, memory residency and offload,
kernels and compilation, response parsing, the publication path, and the number and placement of engines.

Each submission also passes an LLM integrity review (`llm_judge/POLICY.md`) before the GPU Judge. The agent's
time budget is `TASK_TIMEOUT` in the launch config; the example configs use 86400 s (24 h).

## Evaluation

The Judge launches the submission twice. The first launch replays recorded engine responses through a replay
server (a probe round plus the timed segments) and is stopped after 1500 s. The second is the real 64-round run
under an in-engine audit and is stopped after 2700 s. Untimed setup, from the replay server's start to the first
timed release, is capped at 600 s.

Correctness checks:

- static: the tree's layout, and a code policy on files changed from the pristine checkout;
- framework: the required miles_diffusion path ran for every request, round and publication;
- probe: on hidden rows, a hidden non-zero student init and random teacher labels, the step's pairs, transition
  means, KL and gradient agree with an independent float32 oracle;
- replay and run: byte-identical payloads, AdamW algebra of every step, and the published weights seen from
  inside every engine after every round;
- a final-weights check against the last step record.

After the checks pass, the Judge measures the final LoRA:

- **KL gate.** On each route, the Judge's own held-out teacher-student KL of the final student must drop below
  the original model's frozen KL by at least `ln(KL_base / KL_final) >= 0.364` nats (OCR route) and `>= 0.690`
  (PickScore route).
- **Task metric.** One 512x512 image per held-out test prompt of Flow-GRPO / DiffusionOPD (about 1,000 OCR and
  2,000 PickScore prompts), generated with the DiffusionOPD evaluation pipeline (40-step ODE, guidance 1.0, fixed
  seed per prompt). Images are scored by OCR accuracy (PaddleOCR) or PickScore_v1.

```
closure_r     = (S_r - B_r) / (T_r - B_r)      S_r: final student, B_r: original model, T_r: route r's teacher
quality_score = mean over the two routes of closure_r
score         = quality_score x 3600 / T       if every check passes, the KL gate passes and closure_r >= 0.25 on both routes
              = 0                              otherwise (valid)
```

`B_r` and `T_r` were measured once with the same protocol and frozen. `T` is the median, over three hidden replay
segments of 6 rounds each, of the time from a segment's first release to its last `published` call, on the Judge's
clock. There is no reference time and no cap. A score of 0 from failed checks or gates is still a valid result.
Judge or infrastructure errors make the result invalid.

The public suites in the Work container, `python /opt/opd_public/run_public.py --suite {probe,small,perf}`, run the
Judge's checks on small public data, only to get the pipeline working. `small` runs 4 real rounds of 4 rows and `perf`
replays 1 untimed round and 2 segments of 4 rounds of 4 rows. `perf --quality` adds a real run of 8 rounds of 4 rows and
reports the KL and task metric on 128 public held-out prompts (`runtime/published_reference.json`), with no gate and no
score. The Judge repeats the checks on hidden data: hidden contract rows, a hidden probe fixture, three hidden replay
segments and private KL-gate prompts. The held-out test prompts are the public test splits named above.

Scheduled auto-evaluation is off. Each `sforge-submit` runs the full Judge (on its own GPU pair by default, while
Work keeps running; with `--shared-gpus` the Judge pauses Work because both use one pair) and returns its report to
the agent. After the budget ends, the supervisor evaluates the final archive once more on the Judge pair, and the
run's result is the best valid score across the run's submissions and this final evaluation. The Judge timeout is 7500 s.

## Hardware

Four NVIDIA H100 GPUs by default, as two NVLink pairs: Work runs on one pair (`GPU_INDICES`) and the Judge on the
other (`JUDGE_GPU_INDICES`, or `--judge-gpus C,D`), and Work keeps running while the Judge measures.
`JUDGE_GPU_INDICES=shared` or `--shared-gpus` runs both on one pair one after the other: the Judge pauses Work while
it runs. Containers get 256 GiB of host memory. A formal run pins Work and the Judge to the CPUs local to
their pairs (`SFORGE_CPUSET=auto`), as in the qualification; an unpinned
run is refused.

The qualification Judges ran with no Work container next to them. Scores from
the default separate mode are not directly comparable with shared-mode runs: Work keeps running during Judge rounds and
gets more wall time, and the Judge times its window while Work trains on the other pair.

| hardware type | status | execution profile | memory floor per card |
|---|---|---|---|
| `h100-94gb-400w` (H100 NVL 94 GB) | qualified | `dual-cuda-sm90-nccl` | 80 GiB |
| `h100-80gb-700w` (H100 SXM 80 GB) | qualified | `dual-cuda-sm90-nccl-80g` | 79 GiB |

Each card type has its own qualification record: `runtime/qualification.json` for NVL and
`runtime/qualification.dual-cuda-sm90-nccl-80g.json` for SXM. The launcher admits each pair to a profile (both pairs must meet the same one) and refuses
a formal run without a matching record. Scores are comparable only within one profile. On cards below 80 GiB, the
probe check's float32 oracle runs in smaller micro-batches; only the summation order changes.

## Running

1. **Assets.** `INFRATHON_ASSETS_ROOT=<dir> tasks/latest/miles-diffusion-sd35m-mopd-2xh100-v2-seed/assets/fetch.sh`
   downloads:
   - from their Hugging Face repositories at pinned commits: Stable Diffusion 3.5 Medium (an ungated mirror whose
     files equal `stabilityai/stable-diffusion-3.5-medium`), the DiffusionOPD teacher LoRAs
     (`quanhaol/DiffusionOPD`), PickScore_v1, and the CLIP-ViT-H-14 processor files;
   - from the Infrathon dataset: the public data (rows, student init, public held-out prompts, probe and small replay
     recordings, about 3 GB) and the Judge's private root (about 11 GB).

   The private root must not be owned by uid 1001 (the uid candidate code runs as) and must be `go-rwx`.
   When this host's user is uid 1001, `fetch.sh` hands it to root through a container (no sudo) if the task's
   images are already local, and `pull_images.sh` (step 2) does it otherwise. By hand:
   `scripts/infrathon/assets.py own miles-diffusion-sd35m-mopd-2xh100-v2-seed --image <a local image with sh>`.
   With `INFRATHON_ASSETS_ROOT` set, `scripts/miles_common/host_config.py` maps the task's host locations (`MDOPD_*`)
   onto the asset root.
2. **Images.** `INFRATHON_ASSETS_ROOT=<dir> scripts/infrathon/pull_images.sh miles-diffusion-sd35m-mopd-2xh100-v2-seed` pulls the published Work and
   Judge for this host's hardware type. The Judge carries no private data: its private root is mounted read-only at
   run time.

   To build from source instead:
   - build the base image `miles-diffusion-base:sgl859a278e-md29fde6ee` with
     `scripts/miles_base_images/build.sh diffusion` (see `scripts/miles_base_images/README.md`);
   - set `MD_UPSTREAM` to a miles_diffusion clone that contains `29fde6ee`;
   - set `MILES_SCRATCH_ROOT` to a directory with about 50 GB free;
   - run `tasks/latest/miles-diffusion-sd35m-mopd-2xh100-v2-seed/scripts/build_images.sh`.
3. **Run.** Keep `INFRATHON_ASSETS_ROOT` (and `HF_HUB_CACHE`, if step 1 used one) exported in the
   launching shell, or set them in `launch/mlsysforge.env`: the Miles host configuration reads both
   from the environment, else from that file, not from the run config. Without them the dry run stops with a
   "... is not configured" error and the model cache falls back to `~/.cache/huggingface/hub`.

   ```bash
   SFORGE_CPUSET=auto launch/miles-diffusion-opd/run.sh --config <config.env> [--judge-gpus C,D | --shared-gpus] [--run-id ID] [--dry-run] [--foreground]
   ```

   Start the config from `launch/miles-diffusion-opd/*.env.example`. It sets:
   - the SForge version store and the agent runtime (`SFORGE_AGENT_RUNTIME_DIR`, built with
     `launch/build-agent-runtime.sh`);
   - the agent and model: Codex with a ChatGPT login (`CODEX_AUTH_JSON_PATH`) or an OpenAI-compatible endpoint
     (`MODEL_API_BASE_URL`, `MODEL_API_KEY_FILE`), or Claude Code with `AGENT=claude-code`;
   - `MODEL`, `REASONING_EFFORT`, `TASK_TIMEOUT`, `SUBMISSION_COOLDOWN`, `GPU_INDICES` (Work's NVLink pair),
     `JUDGE_GPU_INDICES` (the Judge's NVLink pair, or `shared`) and `JUDGE_PORT`. The examples use `TASK_TIMEOUT=86400`,
     `GPU_INDICES=0,1`, `JUDGE_GPU_INDICES=2,3` and `JUDGE_PORT=18632`.

   `--dry-run` prints the plan, or the first failed precondition, without starting anything. The launcher checks:
   - the agent runtime and credentials;
   - the mandatory LLM integrity review settings (`SFORGE_LLM_JUDGE_*`, which default to the run's own Codex login
     in ChatGPT mode);
   - the GPU profile of both pairs and its qualification record, CPU pinning, and that all four GPUs are idle;
   - both images;
   - at least 500 GiB free under `MILES_SCRATCH_ROOT`.

   The launcher has no separate eval-only mode.

During a run, Work is on an internal network. It reaches the model API and public web hosts (ports 80/443) only
through the run's relay proxy. The Judge has no network.

## Files

| path | contents |
|---|---|
| `images/work/` | Work Dockerfile, `AGENT_TASK.md` (the agent's brief), the starter `workdir/miles_lab/` (`task_entry.py`, `config/`, `notes.md`) |
| `images/judge/` | Judge Dockerfile (checks at `/opt/opd_judge`; private root mounted at `/judge_private`) |
| `docs/INTERFACE.md` | command, contract fields, replay protocol, step-record schema, what each check looks at |
| `runtime/` | contract, qualification records per profile, public reference values, SForge binding (`sforge_entry.py`: mounts, GPU pair, network relay) |
| `llm_judge/POLICY.md` | integrity-review policy |
| `reference/` | reference task entry and the reference miles_diffusion changes as a patch on `29fde6ee` (host only, never in an image) |
| `scripts/` | `build_images.sh`, `run_task.py` (run supervisor) |
| `assets/` | `manifest.json` (every external file with its SHA-256) and `fetch.sh` |
| `tests/` | CPU tests of the Work network policy |

Checkers, oracle, replay server and audit hook: `scripts/miles_diffusion_opd/` (`judge_v2/`, `oracle/`,
`judge/audit/`). Shared Miles tooling: `scripts/miles_common/`.

## Upstream and license

- miles_diffusion, `radixark/miles_diffusion` at `29fde6ee`: Apache-2.0.
- sglang-diffusion from SGLang, `sgl-project/sglang` at `859a278e`, built into the diffusion base image: Apache-2.0.
- Stable Diffusion 3.5 Medium. The task uses `adamo1139/stable-diffusion-3.5-medium-ungated` at `1cb9becf`, a mirror
  whose files equal `stabilityai/stable-diffusion-3.5-medium` at `b940f670`. It is under the Stability AI Community
  License (`LICENSE.md` in the snapshot); the official repository is gated, and users must comply with Stability
  AI's license.
- DiffusionOPD teacher LoRAs, `quanhaol/DiffusionOPD` at `38686355`. The repository is tagged CC BY 4.0; its model
  card states "This project is released for academic research only".
- PickScore_v1, [`yuvalkirstain/PickScore_v1`](https://huggingface.co/yuvalkirstain/PickScore_v1) at `a4e4367c`.
  No license is declared on the Hub; its license follows the upstream source (the PickScore code repository is MIT). The processor files come from `laion/CLIP-ViT-H-14-laion2B-s32B-b79K` at `1c2b8495` (MIT).
- Training and test prompts come from the `dataset/` directory of `ali-vilab/DiffusionOPD` at `0c85bf0`
  (Apache-2.0); they are identical to Flow-GRPO's (MIT).
- The base image also installs PaddleOCR 2.9.1 (Apache-2.0) and HPSv2 (`tgxs002/HPSv2` at `866735ecaae9`,
  Apache-2.0).
- The task's own code is under the repository's root `LICENSE` (PolyForm Noncommercial 1.0.0).
