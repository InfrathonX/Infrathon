# StreamDiffusionV2 Dual-GPU Pipeline Runtime

This task runs a live video-to-video stream through frozen StreamDiffusionV2 / Wan2.1-T2V-1.3B
math on two CUDA GPUs. The agent rewrites the pipeline runtime around that math (stage
placement, transfers, compute/communication overlap, rolling-KV bookkeeping, lane scheduling,
buffers, CUDA graphs) to maximize the stream's latency-weighted frames per second, while every
output stays bit-exact. It is a seed task: the starting point is a working baseline runtime
(`runtime_v001`, a static two-rank pipeline that splits the 30 DiT blocks 15/15) to improve.

## The task

The agent works in `/app/workdir/sdv2_pipeline_lab` inside the Work container and submits
`candidates/`, `FINAL_VERSION` and `notes.md`. Each attempt is a new package
`candidates/runtime_vNNN/`; `FINAL_VERSION` holds the integer of the version to evaluate. The
package's `runtime.py` must define the `Runtime` class with this surface:

```
Runtime(core_loader, devices, plan_meta, sink, *, master_port=None)
start(prompt_embeds, first_chunk_frames) -> None
push(chunk_idx, frames) -> None
flush() -> None
close() -> RunLedger
```

`plan_meta` is a frames-free `StreamPlanMeta` (chunk descriptors, indexed noise, cache
geometry); frames reach the runtime only through `start` (chunk 0) and `push`. The trusted
controller timestamps each push immediately before calling `start`/`push`; the sink records
emissions. `close` returns only after all communication is matched, child processes have exited
and both devices are idle.

Frozen (owned by the root-owned `sdv2_core` and the Judge): weights and all numerical math
(patchification, all 30 Wan DiT blocks, attention projections, head, flow-to-x0/add-noise,
streaming VAE), input clips, prompt tensors, indexed noise, plan metadata, output sink, clock,
goldens and gate thresholds. The stream protocol is 5 prepare frames followed by 4-frame
chunks, denoising steps `[700,500,400,200][:S]`, BF16 math, and a fixed rolling/sink/refresh
KV-cache policy.

Candidate-owned: placement, KV/policy migration, activation and state transport, streams and
events, VAE/DiT/transfer overlap, rolling-KV bookkeeping, lane scheduling, buffers and CUDA
graphs. The candidate may spawn child processes. Any use of the two GPUs is allowed, including
running everything on one GPU; running independent copies of the stream does not help because
each case is one stream whose chunks depend on earlier chunks' caches.

Rules the Judge and the integrity policy (`llm_judge/POLICY.md`) enforce:

- Every chunk runs every requested denoising step through all DiT blocks and the frozen VAE;
  step order, chunk-indexed noise, per-step causal cache visibility and the rolling/sink/refresh
  policy stay exact. Migrations move all K/V, indices, cross-attention state and policy
  counters at a quiescent boundary.
- No replacing or monkey-patching the frozen math, weights or inputs, no independent
  reimplementation that routes around the frozen core, no deliberate weakening of
  bit-exactness. Fetched implementations or pretrained assets may not enter a submission.
- Source admission: a candidate package is a regular `runtime_vNNN/` directory with
  `runtime.py` (defining `Runtime`) and `__init__.py`, only `.py`/`.md`/`.txt`/`.json` files,
  no links, at most 128 files, 2 MiB per file and 8 MiB in total.

The launcher's default agent budget is 24 hours (`--hours`/`--timeout` change it). Formal
submissions (`sforge-submit`) are unlimited with a 300-second cooldown; automatic evaluation is
disabled. Work has internet access through a restricted egress sidecar.

## Evaluation

Each Judge run first measures the pristine seed (`runtime_v001`, hash-protected in the Judge
image) over the suite on the same two devices, then the candidate. All seed-relative quantities
come from that in-run pass, so the score has no frozen device constant.

Per case (warm-up chunk W = 12):

- `steady_fps` = output frames of chunks W..last / (emission of the last chunk - push of
  chunk W).
- latency(i) = emission(i) - push(i); p99 over chunks >= W.
- case score = `steady_fps * (seed_p99 / candidate_p99) ** 0.25`.

The suite score `latency_weighted_fps` (unit fps, maximized, uncapped) is the geometric mean of
the case scores. Results also report raw FPS geomean, the p99 ratio, construction time, per-rank
NVML peak memory, host RSS and parity diagnostics per case.

A submission is valid only if all twelve gates pass (fail-closed): profile, protected hashes,
source admission, in-run seed reference, worker completion, complete causal ledgers, parity,
liveness, resources, TTFF, p99 and construction. In particular:

- Parity: every x0 latent and decoded float32 frame equals the in-run seed output bit for bit
  (no tolerance).
- TTFF <= max(2.5 x seed TTFF, seed TTFF + 4 s); p99 <= 3 x seed p99; construction <=
  max(2 x seed construction, seed construction + 60 s).
- GPU memory per rank <= the larger of the seed's two per-rank peaks + 8 GiB; host RSS <=
  65536 MiB per worker session; after `close`, at most 1024 MiB above each rank's pre-run
  occupancy.
- Causality: no chunk is emitted before it is pushed.

A failure in the seed reference pass is reported as a trusted failure, not charged to the
candidate.

Test data. The Work image ships 6 public cases (832x480 and 512x512, 2 and 4 denoising
steps, 129 or 225 frames) with public tools (`tools/run_one.sh`, `tools/run_public_tests.sh`)
for local diagnostics. Every formal submission and the final evaluation run the same frozen
private suite of 12 cases, which also covers 1- and 3-step settings, three resolutions not in
the public set and longer streams, with hidden prompts. Each evaluation gives the seed pass
and the candidate pass the same per-pass limit of 1800 s inside a 4200 s Judge limit.
Interactive submissions evaluate the version named by `FINAL_VERSION`. After the agent
finishes, the launcher runs a fresh Judge of the sealed final snapshot on the Judge pair (the
one shared pair in shared mode); that report gives the agent no feedback, and the run's result is the best valid score across the run's submissions and this final evaluation.

Scores are raw latency-weighted FPS and are comparable only within one measurement profile
(same hardware type).

Oracle (`oracle/FINAL_VERSION` = 2), final suite, H100 NVL pair:

| oracle | candidate | score | gates |
|---|---|---:|---|
| seed | `runtime_v001`, static 15/15 block split, two-rank pipeline | 28.89 (qualification run) | all 12 pass, 12/12 parity |
| best admissible paper candidate (= `oracle/FINAL_VERSION`) | `runtime_v002` split_by_steps: paper §4.3 block balancing as a static per-step split | 33.12 (seed 29.12 in the same run) | all 12 pass, 12/12 parity |
| non-admissible paper candidate | none: the paper candidate is admissible, so no run with the admission checks turned off is needed | n/a | n/a |

## Hardware

Each run uses four GPUs by default: Work runs on one GPU pair and the Judge on another, and Work
is never paused. `--shared-gpus` runs both on one pair instead, and the Judge pauses the Work
container while it evaluates. Each Judge measures the seed and the candidate on its own pair in
the same evaluation, in either mode. Scores from separate mode are not directly comparable with
shared-mode runs, because Work keeps running during Judge rounds and gets more wall time. The
capability envelope (checked by the trusted worker and the
Judge) is two CUDA devices with compute capability >= 8.0, at least 26 GiB each, at least
64 GiB usable host RAM, and NCCL across the pair, on the pinned cu124 / torch 2.6.0 /
flash-attn 2.7.4.post1 software stack. The launcher's preflight requires each pair to be
NVLink-connected. Work and Judge containers are each limited to 16 CPUs and 96 GB of memory.

| hardware type | status |
|---|---|
| `h100-94gb-400w` (H100 NVL 94 GB) | qualified |
| `h100-80gb-700w` (H100 SXM 80 GB) | qualified |

The Judge's frozen goldens are measured on the card, so they are kept per hardware type
(`hw/<hardware>/` in the assets, `images/judge/protected_hashes/<hardware>.txt`). They enter
only the locally built holdout layer, so the published Work and Judge images are the same for
both hardware types. Parity gates against the in-run seed; the frozen goldens are compared as
a recorded diagnostic.

## Running

1. Assets. `tasks/latest/streamdiffusionv2-dual-h100-pipeline-runtime-v2-seed/assets/fetch.sh`
   downloads this task's files from the Infrathon dataset into `INFRATHON_ASSETS_ROOT`: the
   Judge-only final case definitions, hidden prompt embeddings and video clips (from the
   dataset's `judge/` part; they are not in the repository), and the Judge goldens of this
   host's hardware type. Building the images from source additionally needs
   the asset root that `scripts/stage_assets.py` assembles (`SDV2_ASSETS_ROOT`): Wan2.1-T2V-1.3B
   and the causal/DMD v2v checkpoint from their Hugging Face repositories at pinned revisions,
   the flash-attn 2.7.4.post1 wheel from its GitHub release, the StreamDiffusionV2 source at a
   pinned commit, and clips cut from the Blender Foundation films (`scripts/fetch_clips.py`).
   `fetch.sh` does not download these source-build inputs; `scripts/stage_assets.py` does.
2. Images. `INFRATHON_PULL=1 tasks/latest/streamdiffusionv2-dual-h100-pipeline-runtime-v2-seed/scripts/build_images.sh`
   pulls the published Work and public Judge images for this host's hardware type and builds
   the holdout layer locally (it checks that the goldens match this hardware type). Without
   `INFRATHON_PULL=1` it builds everything from source. `INFRATHON_HARDWARE` overrides the detected
   hardware type.
3. Run. Launch an agent run:

   ```bash
   launch/sdv2-pipeline/run.sh --model <model-id> [--work-gpus A,B --judge-gpus C,D] [--hours N]
   ```

   Defaults: Work on GPUs 0,1, Judge on GPUs 2,3, 24 hours, effort `max`, Judge port 18612;
   `--work-gpus`, `--judge-gpus`, `--hours`, `--effort` and `--judge-port` change them. The four
   GPUs must be distinct. `--shared-gpus` runs Work and Judge on the Work pair, with Work paused
   during each Judge round; `--gpus A,B` is the same as `--shared-gpus` on pair A,B. In a launch
   config or the environment, `SDV2_GPU_INDICES` is the Work pair and `SDV2_JUDGE_GPU_INDICES` is
   the Judge pair or `shared`. Options include `--agent codex|claude-code` (default codex),
   `--auth FILE` (Codex `auth.json`, default `~/.codex/auth.json`), `--claude-token FILE` (a
   mode-0600 file with a `claude setup-token` OAuth token, required for claude-code),
   `--cpuset auto|off|CPULIST` and `--config FILE` (a `.env` or `.json` launch config). There is
   no start delay unless `MIN_DELAY_SECONDS` is exported. The launcher waits until all of the
   run's GPUs are idle, locks them, runs a preflight on each pair, runs the agent and then the
   final Judge on the Judge pair. Concurrent runs on one host each need their own GPUs and
   `--judge-port`; an 8-GPU host fits two runs in the default mode.

   Evaluate a saved archive (`submission.tar.gz` or `final_archive.tar.gz`) without an agent,
   on one pair:

   ```bash
   SDV2_GPU_INDICES=A,B tasks/latest/streamdiffusionv2-dual-h100-pipeline-runtime-v2-seed/scripts/eval_archive.sh ARCHIVE.tar.gz [RUN_ID]
   ```

   Replay saved candidates of a finished run: `launch/sdv2-pipeline/replay.sh --origin-run
   RUN_ID [--candidate vN] [--dry-run]` (needs the SForge version store,
   `SFORGE_VERSION_STORE_ROOT`).

Prerequisites the launcher checks: Docker, the repository's `.venv` with SForge, a CUDA build
of `torch` in the first `python3` on `PATH`, used by the GPU-pair preflight (see
`docs/HOST_PREREQUISITES.md`),
the built Work and Judge images, idle NVLink GPU pairs, and agent credentials. Every
submission goes through the LLM integrity review against `llm_judge/POLICY.md` before the
Judge runs. It is on by default (`SFORGE_LLM_JUDGE_MODE=enforce`); its credentials come from
the `SFORGE_LLM_JUDGE_*` launch settings (`docs/LLM_JUDGE.md`).

## Files

```
tasks/          SForge task spec (Judge command, timeouts, resources, submit paths)
images/work/    Work image: agent brief, lab directory with seed candidate, public tools
images/judge/   Judge image: verifier, candidate worker, source admission,
                per-hardware protected hashes, holdout layer Dockerfile.private
judge/          Judge entry (Judge-only inputs come from the assets)
runtime/        frozen sdv2_core, case definitions, scoring, capability envelope,
                evaluation protocol, public case manifest
oracle/         pristine seed runtime_v001, paper split runtime_v002, FINAL_VERSION=2
portable/       suite identity and selection protocol (no private case parameters)
assets/         asset manifest and fetch.sh, prompts, public embeddings, clip and
                checkpoint manifests, third-party notices
scripts/        image build, asset staging, goldens, preflight, launch, eval, replay
tests/          CPU contract, scoring and security tests
llm_judge/      integrity review policy
```

## Upstream and license

- Paper: StreamDiffusionV2, Feng et al., [arXiv:2511.07399](https://arxiv.org/abs/2511.07399).
- StreamDiffusionV2 (<https://github.com/chenfengxu714/StreamDiffusionV2>), commit
  `2f8ee1c5b6616b115662d190d5213677882d85d7`, Apache-2.0. The frozen `sdv2_core` is derived
  from it.
- Wan2.1-T2V-1.3B (Hugging Face `Wan-AI/Wan2.1-T2V-1.3B`), revision
  `37ec512624d61f7aa208f7ea8140a131f93afc9a`, Apache-2.0.
- Causal/DMD v2v checkpoint (Hugging Face `jerryfeng/StreamDiffusionV2`, file
  `wan_causal_dmd_v2v/model.pt`), revision `2373eb2b39278b3a1aa174964a724ee78ead96f0`,
  Apache-2.0 (per its model card).
- FlashAttention 2.7.4.post1 prebuilt wheel (Dao-AILab/flash-attention release).
- Video clips cut from Big Buck Bunny, Sintel and Tears of Steel, (c) Blender Foundation,
  CC BY 3.0; source URLs and frame ranges are in `assets/clips/THIRD_PARTY_NOTICES.md`.
- The task's own code is under the repository's root `LICENSE` (PolyForm Noncommercial 1.0.0).
