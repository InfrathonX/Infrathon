# Wan2.2 DanceGRPO with an HPSv3 Video Reward in miles_diffusion

miles_diffusion (FSDP2 diffusion RL with sglang-diffusion rollout engines and Ray) trains Flow-GRPO with the flow-SDE
and CPS (coefficients-preserving sampling) transitions. It cannot yet run DanceGRPO for video. The agent implements
DanceGRPO for Wan2.2-TI2V-5B in text-to-video mode with full-parameter training. That means three additions:

- the Dance-SDE transition in the rollout engine;
- seed groups that share the initial noise but keep the SDE noise independent;
- the official HPSv3 video reward.

The recipe is fixed. The score is how often per hour the agent's trainer completes a fixed replay workload: reward
scoring, the old-log-prob recompute, CFG training forward and backward, AdamW, and full-weight publication. This is
a seed task: the starting point is miles_diffusion at `29fde6ee` with a stub `task_entry.py`, whose three commands
parse their arguments and stop.

## The task

The Work container's root is `/app/workdir/miles_lab`. The agent submits `repo/` and `notes.md`:

- `repo/` is miles_diffusion plus `repo/task_entry.py`, which provides the commands `score`, `run` and `serve`;
- `repo/engine_overlay/` holds the agent's versions of whitelisted sglang-diffusion files, under the same relative
  paths as in `python/sglang/multimodal_gen/`.

The Judge copies `repo/` fresh for every phase and runs it offline as the unprivileged user `agent`.
`docs/INTERFACE.md` defines the commands, the contract, the replay protocol, the per-pair log and the checks.

The fixed recipe (`runtime/contract.json`):

- **Sampling.** 704x1280, 9 frames, 10 steps, CFG 5 with negative prompt `" "`, flow shift 5, bf16 DiT, no step
  skipping or caching of any kind.
- **Seed groups.** Groups of 8 samples share an initial noise derived from canonical SHA-256 keys. Each sample has
  its own 2-step SDE window and its own SDE noise.
- **Dance-SDE transition.** eta 1.2, with the Dance log-prob.
- **Reward.** HPSv3 (official MizzenAI semantics) on frames 0, 4 and 8. The video reward is 0.1 x the mean of the
  three.
- **Advantages.** Group-normalised and clipped to +-5.
- **Loss.** PPO-clip (clip range 1e-4, no KL), with pi_old recomputed by the trainer on the pre-update weights.
- **Optimizer and publication.** Full-parameter AdamW (lr 1e-5) with an fp32 master. The full transformer is
  published through `update_weights_from_tensor` after every round.
- **`run`.** 3 rounds x 8 prompts x 8 samples, lag 0, 2 optimizer steps per round.

Rules the Judge enforces:

- Engine changes are limited to the whitelisted overlay files, and the existing `sde` / `cps` / `ode` behaviour
  stays bitwise identical.
- No other engine files, weights or contract may change, and nothing about what is sampled or trained.
- The trainer computes in bf16 or wider (no fp8/fp4 GEMMs).
- The `WAN_AUDIT_*` environment reaches every engine the agent starts.
- `run` uses exactly two engines; `serve` uses the Judge's two engines and starts none.
- `serve` writes `ready` within 25 minutes and answers each later command within 30 minutes.
- The implementation is written natively in the checkout.

Within `repo/`, the agent may change anything that keeps these semantics: placement, offload, CFG batching,
attention backend, `torch.compile`, reward batching, overlap, and the publication transport.

Each submission also passes an LLM integrity review (`llm_judge/POLICY.md`) before the GPU Judge. The agent's
time budget is `TASK_TIMEOUT` in the launch config; the example configs use 86400 s (24 h).

## Evaluation

The Judge runs the commands on hidden data:

- `score` on a video set checked per frame against the official HPSv3 inferencer;
- `run` (the real RL, audited inside the engines, untimed);
- `serve` against two engines the Judge starts itself: a probe step, then one timed replay for each of three hidden
  replay dumps. Each dump holds 16 prompts x 8 videos, with the frames, latents, sigma grid and conditioning.

For each replay, `T` runs from the moment the dump becomes readable to the later of two events: the last Judge
engine holding a complete publication, and the agent's `log_complete` event. `T` is taken on the Judge's clock.
Loading, the probe, `dump_master` and `reset` are not timed.

The correctness checks, in the order the Judge runs them; the ids are the check names (or their prefixes) in the
Judge's report, and their thresholds are Judge-private:

- **J0:** overlay whitelist, AST policy on added code, and bitwise legacy branches.
- **J2:** HPSv3 per frame.
- **J3:** the `run` request audit.
- **J4:** transitions recomputed from the engine's own tensors.
- **J1:** the probe's pi_old and first AdamW step against float32 oracles.
- **R1-R5, per replay:** rewards, advantages, all pi_old, log self-consistency and composition, one complete
  publication, and a final log at `log_complete`.
- **R6_equal and R6_traj:** published weights equal what each engine holds (bitwise), and the cumulative update
  matches a float32 4-step oracle.
- **J6:** the last `run` publication equals a reload of the final fp32 master.

After every check passes, a quality gate runs. The Judge loads the final master of its own `run` into two fresh
engines. It generates one video per prompt of the public Hao7/video_prompts `test.txt` (61 prompts, used by no
training input) for each of 3 fixed seed sets, using a deterministic 50-step flow-ODE sampler. It scores the videos
with HPSv3 and compares the result with the original model's frozen value (0.7891).

```
quality_score = mean HPSv3 of the final model - frozen original-model value
score         = 3600 / median(T over the 3 hidden dumps)   if every check passes and quality_score >= -0.28
              = 0                                           otherwise (valid)
```

Quality is a non-degradation gate, not a multiplier: the contract's six AdamW steps are not expected to improve
held-out HPSv3. There is no reference time and no cap. A score of 0 from failed checks or gates is still a valid
result. Judge or infrastructure errors make the result invalid.

The public suites in the Work container, `python /opt/wan_public/run_public.py --suite {cpu,score,probe,small,perf,full}`,
run the Judge's code on public data:

- `perf` runs the probe plus one timed replay of the public dump `pub0` (8 prompts, 2 optimizer steps; the public
  data has no R6 trajectory oracle, so R6's trajectory check is not run on it) and reports its `T`, with no score;
- no suite runs the quality evaluation.

Scheduled auto-evaluation is off. Each `sforge-submit` runs the full Judge (on its own GPU pair by default, while
Work keeps running; with `--shared-gpus` the Judge pauses Work because both use one pair) and returns the score or
the failing checks, with no logs. After the budget ends, the supervisor evaluates the final archive once more on the
Judge pair, and the run's result is the best valid score across the run's submissions and this final evaluation. The Judge timeout is 5000 s.

## Hardware

Four NVIDIA H100 GPUs by default, as two NVLink pairs: Work runs on one pair (`GPU_INDICES`) and the Judge on the
other (`JUDGE_GPU_INDICES`, or `--judge-gpus C,D`), and Work keeps running while the Judge measures.
`JUDGE_GPU_INDICES=shared` or `--shared-gpus` runs both on one pair one after the other: the Judge pauses Work while
it runs. Containers get 256 GiB of host memory and 128 GiB of `/dev/shm`. A formal run pins Work and the
Judge to the CPUs local to their pairs (`SFORGE_CPUSET=auto`), as in the qualification; without a pinning policy,
`CPUSET` and `JUDGE_CPUSET` name the cores and must not overlap.

The qualification Judges ran with no Work container next to them. Scores from
the default separate mode are not directly comparable with shared-mode runs: Work keeps running during Judge rounds and
gets more wall time, and the Judge times its window while Work trains on the other pair.

| hardware type | status | execution profile | memory floor per card |
|---|---|---|---|
| `h100-94gb-400w` (H100 NVL 94 GB) | qualified | `dual-cuda-sm90-nccl` | 90 GiB |
| `h100-80gb-700w` (H100 SXM 80 GB) | qualified | `dual-cuda-sm90-nccl-80g` | 79 GiB |

Each card type has its own qualification record (`runtime/qualification.json` for NVL,
`runtime/qualification.dual-cuda-sm90-nccl-80g.json` for SXM). Each also has its own Judge-only private root, and
the two roots differ only in their `qualification.json`, which holds the Judge thresholds for that profile. The
launcher admits each pair to a profile (both pairs must meet the same one) and refuses a formal run without a matching record. Scores are comparable
only within one profile.

The reference implementation needs memory-placement changes to fit each card; `reference/*.patch` changes where
memory goes, not what is sampled or trained.

## Running

1. **Assets.** `INFRATHON_ASSETS_ROOT=<dir> tasks/latest/miles-diffusion-wan22-dancegrpo-2xh100-v2-seed/assets/fetch.sh`
   downloads:
   - from their Hugging Face repositories at pinned commits: `Wan-AI/Wan2.2-TI2V-5B-Diffusers`, `MizzenAI/HPSv3`,
     the configuration and processor files of `Qwen/Qwen2-VL-7B-Instruct` (no weights), and `Hao7/video_prompts`
     (its `test.txt` is the quality-gate prompt set, built into the Judge image);
   - from the Infrathon dataset: the public data root (about 4.3 GB) and the private Judge root for this
     host's hardware type (about 16 GB; set `INFRATHON_HARDWARE` or let it be detected).

   The private root must not be owned by uid 1001 (the uid candidate code runs as) and must be `go-rwx`.
   When this host's user is uid 1001, `fetch.sh` hands it to root through a container (no sudo) if the task's
   images are already local, and `pull_images.sh` (step 2) does it otherwise. By hand:
   `scripts/infrathon/assets.py own miles-diffusion-wan22-dancegrpo-2xh100-v2-seed --image <a local image with sh>`.
   With `INFRATHON_ASSETS_ROOT` set, `scripts/miles_common/host_config.py` maps `MDWAN_*` onto the asset root.
2. **Images.** `INFRATHON_ASSETS_ROOT=<dir> scripts/infrathon/pull_images.sh miles-diffusion-wan22-dancegrpo-2xh100-v2-seed` pulls the published Work
   and Judge for this host's hardware type. The Judge carries no private data: its private root is mounted
   read-only at run time.

   To build from source instead:
   - build the base image `miles-diffusion-base:sgl859a278e-md29fde6ee` with
     `scripts/miles_base_images/build.sh diffusion`;
   - set `MD_UPSTREAM` to a miles_diffusion clone that contains `29fde6ee`;
   - set `MILES_SCRATCH_ROOT` to a directory with about 50 GB free;
   - run `tasks/latest/miles-diffusion-wan22-dancegrpo-2xh100-v2-seed/scripts/build_images.sh`.
3. **Run.** Keep `INFRATHON_ASSETS_ROOT` (and `HF_HUB_CACHE`, if step 1 used one) exported in the
   launching shell, or set them in `launch/mlsysforge.env`: the Miles host configuration reads both
   from the environment, else from that file, not from the run config. Without them the dry run stops with a
   "... is not configured" error and the model cache falls back to `~/.cache/huggingface/hub`.

   ```bash
   SFORGE_CPUSET=auto launch/miles-diffusion-wan/run.sh --config <config.env> [--judge-gpus C,D | --shared-gpus] [--run-id ID] [--dry-run] [--foreground]
   ```

   Start the config from `launch/miles-diffusion-wan/*.env.example`. It sets:
   - the SForge version store and the agent runtime (built with `launch/build-agent-runtime.sh`);
   - the agent and model credentials: a Codex ChatGPT login, an OpenAI-compatible endpoint, or Claude Code;
   - `MODEL`, `REASONING_EFFORT`, `TASK_TIMEOUT`, `SUBMISSION_COOLDOWN`, `GPU_INDICES` (Work's NVLink pair),
     `JUDGE_GPU_INDICES` (the Judge's NVLink pair, or `shared`) and `JUDGE_PORT`. The examples use `TASK_TIMEOUT=86400`,
     `GPU_INDICES=0,1`, `JUDGE_GPU_INDICES=2,3` and `JUDGE_PORT=18633`.

   `--dry-run` prints the plan, or the first failed precondition. The launcher checks:
   - the agent runtime, the credentials and the mandatory LLM review settings (`SFORGE_LLM_JUDGE_*`);
   - the GPU profile, the private root's qualification record for that profile and core count, and CPU pinning;
   - both images and the metric definition built into them;
   - at least 300 GiB free under `MILES_SCRATCH_ROOT`.

   The launcher need not run as the private root's owner: when it cannot read the
   qualification record itself, it reads it through a short-lived root container of the Judge image.

   To evaluate an existing submission archive (`repo/` + `notes.md`) with the same Judge, images and GPU pair as a
   run's final evaluation (the Judge pair of `--judge-gpus` or `JUDGE_GPU_INDICES`, else `GPU_INDICES`):

   ```bash
   python3 tasks/latest/miles-diffusion-wan22-dancegrpo-2xh100-v2-seed/scripts/rescore.py --config <config.env> --archive <archive.tar.gz> --label <name>
   ```

During a run, Work is on an internal network. It reaches the model API and public web hosts (ports 80/443) only
through the run's relay proxy. The Judge has no network.

## Files

| path | contents |
|---|---|
| `images/work/` | Work Dockerfile, `AGENT_TASK.md` (the agent's brief), the starter (`starter/task_entry.py`, `starter/engine_overlay/`) and `workdir/miles_lab/notes.md` |
| `images/judge/` | Judge Dockerfile (checks root-only at `/opt/wan_judge`; private root mounted at `/judge_private`) |
| `docs/INTERFACE.md` | commands, contract, replay protocol and dump layout, per-pair log schema, check summary |
| `runtime/` | contract, qualification records per profile (metric, thresholds), public reference times (context only), SForge binding (`sforge_entry.py`) |
| `llm_judge/POLICY.md` | integrity-review policy |
| `reference/` | how the reference implementation is rebuilt: patches on miles_diffusion `29fde6ee` and on the SGLang `python/` tree at `859a278e`, plus the per-card memory patches (host only, never in an image) |
| `scripts/` | `build_images.sh`, `run_task.py` (run supervisor), `rescore.py` (Judge an archive) |
| `assets/` | `manifest.json` and `fetch.sh` |
| `tests/` | CPU tests of the Work network policy |

Checks, oracles, the HPSv3 scorer and the audit hook: `scripts/miles_diffusion_wan/` (`judge_v2/`, `audit/`,
`hpsv3_scorer.py`, `oracle_dance_fp32.py`). Shared Miles tooling: `scripts/miles_common/`.

## Upstream and license

- miles_diffusion, `radixark/miles_diffusion` at `29fde6ee`: Apache-2.0.
- sglang-diffusion from SGLang, `sgl-project/sglang` at `859a278e` (in the diffusion base image): Apache-2.0.
- The recipe follows the Wan2.2 DanceGRPO default CUDA configuration of
  [verl-project/verl-omni](https://github.com/verl-project/verl-omni) at
  `b8b67c9ab1ee45397d429af23c26789425d3fce6`
  (`examples/dancegrpo_trainer/wan22/run_wan22_5b_t2v_hpsv3_v1.sh`); the HPSv3 scorer is
  transcribed from its `verl_omni/utils/reward_score/hpsv3_reward.py` (Apache-2.0).
- Models and data, pinned by commit in `assets/manifest.json`:
  - `Wan-AI/Wan2.2-TI2V-5B-Diffusers` at `b8fff731`: Apache-2.0;
  - `MizzenAI/HPSv3` at `4f81e3e0`: Apache-2.0;
  - `Qwen/Qwen2-VL-7B-Instruct` at `eed13092` (configuration and processor only): Apache-2.0;
  - `Hao7/video_prompts` at `bc7d7a44`: MIT.

  Licenses are as declared on the Hugging Face model and dataset cards.
- The task's own code is under the repository's root `LICENSE` (PolyForm Noncommercial 1.0.0).
