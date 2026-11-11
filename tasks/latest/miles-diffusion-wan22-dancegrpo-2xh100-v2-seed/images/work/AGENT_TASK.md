# miles_diffusion: Wan2.2-TI2V-5B DanceGRPO with an HPSv3 video reward on two GPUs

Hardware: two NVIDIA H100 GPUs on one NVLink pair, assigned to this run. Their memory and the CPUs you get depend on the host: check them with `nvidia-smi` and `nproc` and size your configuration from what you see. The container has 256 GiB of host memory. By default the Judge runs on its own pair of GPUs; only in shared mode does it use this pair, pausing your
workspace while it runs.

## Goal

`repo/` (in `/app/workdir/miles_lab`) is miles_diffusion `29fde6ee` (FSDP2 diffusion RL with sglang-diffusion rollout
engines, Ray). It already trains Flow-GRPO with the flow-SDE and CPS transitions and ships PickScore/HPSv2/OCR rewards.

Implement **DanceGRPO** for Wan2.2 in it: Wan2.2-TI2V-5B in text-to-video mode, the
**DanceGRPO SDE**, seed groups, and the **HPSv3** video reward, full-parameter. miles_diffusion cannot run it today:
the rollout engine has no Dance transition, cannot share the initial noise within a group while keeping the SDE
noise independent, and there is no HPSv3 reward. (The miles docs claim DanceGRPO is "the sde backend with a tuned
η"; it is not — the mean and the std differ.)

The recipe below is fixed (transition, RNG keys, reward, advantage, loss, optimizer). Implementing it exactly is the
entry condition, not the score. Then make the part you own — reward scoring, the old-log-prob recompute, the CFG
training forward/backward, AdamW and the full-weight publication — fast on a fixed replay workload: the score is how
many times per hour that workload completes (`3600 / T`).

## Starting point

- `repo/` is yours: `repo/task_entry.py` (the commands `score`, `run`, `serve`) and `repo/engine_overlay/` (your
  versions of the whitelisted engine files, § What you may change). `docs/INTERFACE.md` defines the commands, the
  contract file, the replay protocol, the required outputs and the precise checks.
- The starter's three commands parse their arguments and the contract, then stop; its docstring points to the useful
  pieces of the checkout. The engine needs a Dance branch and a separate SDE-noise generator; the trainer needs the
  Dance log-prob, the seed-group-aware rollout, HPSv3, the replay path and publication to the Judge's engines.
- Policy: `/models/Wan2.2-TI2V-5B-Diffusers` (single `transformer`, 5.0 B params, UMT5 + VAE frozen). T2V mode only
  (no image in requests). The model already resolves in miles via the `wan2_2` family; only one expert exists.
- Reward model: `/models/HPSv3/HPSv3.safetensors` with `/models/Qwen2-VL-7B-Instruct` (config/processor only).
- Engine: sglang-diffusion at `859a278e` (clean copy `/sgl-workspace/sglang/python`).
- Public data (`/data/public`, read-only), in the Judge's layout:
  - `prompts.jsonl` (`{"row": int, "prompt": str}`, rows 0–399 of Hao7/video_prompts train; the Judge uses other
    prompts) and `hpsv3_template.txt`;
  - `run_prompts.jsonl` (for `run`) and `contract.json`;
  - `score_set/`: videos, prompts and the official HPSv3 per-frame oracle;
  - `probe/`: a probe dump and its fp32 oracle;
  - `dumps/pub0` + `oracles/pub0`: a replay dump of 8 public prompts (2 optimizer steps in `contract.json`), with its
    HPSv3 and π_old oracles;
  - `qualification.json`: public thresholds.

## The recipe you must implement

Sampling (checked on every request, inside the engine):
704×1280, 9 frames, 10 steps, CFG 5 (`guidance_scale` = `true_cfg_scale` = 5), negative prompt `" "` (one space),
flow shift 5, bf16 DiT, the engine's own rollout σ grid, uint8 video output. No step skipping or caching of any kind
(TeaCache, cache-dit, `cfg_gate_step < 1`, sparse/quantised attention, progressive resolution).

1. **Seed group and canonical RNG.** Group `g = (run_seed, round, row)`, sample `j ∈ [0, 8)`.
   `H(s)` = first 8 bytes of `sha256(s)`, big-endian, mod 2^63; `base = f"{run_seed}|{round}|{row}"`.
   - initial noise `x_T = torch.randn((1,48,3,44,80), generator=torch.Generator("cpu").manual_seed(H(f"init|{base}")),
     dtype=torch.float32)` — **shared by the 8 samples of the group**;
   - SDE window of sample `j`: start `w_j = H(f"win|{base}|{j}") mod 4`, window `{w_j, w_j+1}`; other steps are Euler
     (`x' = x + (σ'−σ)·v`) and draw no noise;
   - SDE noise of sample `j`: one `torch.randn((1,48,3,44,80), dtype=float32)` per window step, in step order, from
     `torch.Generator("cpu").manual_seed(H(f"sde|{base}|{j}"))`.
2. **Dance-SDE transition** (DanceGRPO), η = 1.2, with σ = σ_k, σ' = σ_{k+1}, v the
   CFG-combined velocity: `x̂0 = x − σ·v`, `μ = x + (σ'−σ)·v + ½η²·(x − x̂0·(1−σ))/σ²·(σ'−σ)`, `s = η·√(σ−σ')`,
   `x' = μ + s·ε`. Log-prob `lp = mean over elements of −(x'−μ)²/(2s²)` (constants dropped). The window may contain
   the σ = 1 step.
3. **Reward = HPSv3** (official MizzenAI semantics): frames 0, 4, 8 of the engine's uint8 video; each frame
   PIL-bicubic resized from the raw frame to `smart_resize(h, w, factor 28, min=max=256·28·28)` (704×1280 → 308×588);
   one user turn `[image, INSTRUCTION.format(text_prompt=prompt) + "\nPlease provide the overall ratings of this
   image: <|Reward|>\n\nEND\n"]` (template file `/data/public/hpsv3_template.txt`), `apply_chat_template(...,
   add_generation_prompt=True)`, right padding, no truncation; model = Qwen2-VL-7B-Instruct config + tokenizer extended
   by `<|Reward|>`, ranknet head, HPSv3 weights loaded strictly; **sequential 1-D position ids for all tokens (the
   official inferencer's forward; no M-RoPE)**; μ = head output at the `<|Reward|>` token, column 0.
   Video reward = 0.1 × mean of the three μ. (The HPSv3 repo's own `config.json` names Qwen2.5-VL; it does not match
   the weights.)
4. **Advantages**: per group mean; std over all raw rewards of the round (unbiased) + 1e-4; clip to ±5.
5. **Loss**: PPO-clip, clip range 1e-4, mean over pairs, no KL. A pair is `(g, j, k)` for `k` in sample `j`'s window;
   both window steps of every sample are trained, each exactly once. **π_old is recomputed by the trainer on the
   pre-update weights** (engine log-probs must not enter the loss); the first optimizer step may reuse its own forward.
   Training forward in bf16 or wider (no fp8/fp4 GEMMs), with CFG (both branches) exactly as in sampling.
6. **Optimizer**: full-parameter AdamW, lr 1e-5, betas 0.9/0.999, eps 1e-8, weight decay 1e-4, global grad-clip 1.0,
   fp32 master. Publish the full transformer after every round (`update_weights_from_tensor`; legacy CUDA IPC —
   `expandable_segments` cannot be used for published tensors here).

## Entry points (`repo/task_entry.py`)

- `score --videos V.npz --prompts P.jsonl --out O` → `O/scores.json` with per-frame μ `[N][3]` and rewards.
- `run --contract C --out O` — the real RL: R = 3 rounds × 8 prompts × 8 samples; round `r` uses rows `8r..8r+7`,
  each once; lag 0 (round `r` sampled with the weights published after round `r−1`); 2 optimizer steps per round
  (64 pairs each); one publication per round. Write the per-pair log (`O/pairs/…jsonl`) and `O/final/`
  (`master_fp32.safetensors`: the fp32 master whose publication was the last one; the Judge loads it into a clean
  engine and compares the engine checksum with your last publication, then measures its held-out quality, § Score).
- `serve --contract C --ctl F --events E --out O` — fixed replay (below).

## Fixed replay (the timed part)

The Judge starts **its own** two sglang-diffusion engines (clean `859a278e`, your uid, not your child processes)
before your `serve` process and passes their URLs in the contract. `serve` loads the trainer at θ₀, then answers
JSON lines: `probe` (one step on 16 pairs, write fp32 θ₁, no publish), `replay` (timed), `dump_master`, `reset`,
`stop`. A replay dump holds a number of prompts × 8 videos, set in the contract (`replay.prompts_per_dump`,
`replay.optimizer_steps`; uint8 frames, `x_w, x_{w+1}, x_{w+2}` per sample, σ grid, pos/neg conditioning, canonical
ids; no engine log-probs). For each replay: HPSv3 on frames 0, 4 and 8 of every video → advantages → π_old for every
pair → one optimizer step per 4 dump prompts (step `k` = the pairs of dump prompts `4k..4k+3`) → write and fsync the
per-pair log → **one** full publication to every Judge engine.

**T** = from the moment the dump becomes readable to the later of: the last Judge engine having received a publication
covering every parameter, and your `log_complete` (Judge clock). Process start, model loads, `dump_master`, `reset` are
not timed. The per-pair log must be complete on disk when you send `log_complete` (the Judge reads it then and again
after `done`), and after your `done` each Judge engine must hold the published weights (the Judge queries their
checksums).

## What you may change

Allowed: anything in `repo/` (placement, offload, batching, CFG batching, attention backend, `torch.compile`,
merging identical forwards — samples of a group that share `x_w` at their first window step may share that forward
and backward — reward batching and GPU preprocessing that reproduces the definition,
overlapping reward with training, publication transport through `update_weights_from_tensor`).

In the engine you may edit **only** these files, submitted as `repo/engine_overlay/<same relative path>` under
`python/sglang/multimodal_gen/`: `configs/post_training/rl_rollout.py`, `configs/sample/sampling_params.py` (new
fields only), `runtime/post_training/scheduler_rl_mixin.py`, `runtime/post_training/rl_dataclasses.py`,
`runtime/pipelines_core/stages/latent_preparation.py` (RNG source only),
`runtime/entrypoints/post_training/io_struct.py` and `rollout_api.py` (field pass-through only).
The existing `sde` / `cps` / `ode` behaviour must stay bitwise identical.

Not allowed: other engine files, weights, the contract, network access, precision below bf16 in the trainer, any change to what is
sampled or trained.

**Your own implementation.** You may read papers and other projects' code to understand the method, but the
implementation you submit is written natively in this checkout. Do not copy in an existing implementation of the
feature from elsewhere (another framework, or another version of this one), whether as copied files, a vendored
package or a wrapper that calls it.

## Correctness requirements

- **Precision.** The engines sample with a bf16 DiT; the trainer's forward and backward run in bf16 or wider (no
  fp8/fp4 GEMMs).
- **Instrumentation.** `run` is started with `WAN_AUDIT_*` environment variables (the Judge's in-engine audit) that
  must reach every engine process you start, and each of your engines runs with the contract's `engine_python_path`
  first on `PYTHONPATH`. `run` samples with exactly two engines; `serve` uses the Judge's two engines and starts none.
- **Correct requests and steps.** Every engine request of `run` carries the sampling fields and canonical keys of
  `docs/INTERFACE.md`, and its transitions must match a recomputation from the engine's own records. The per-pair log
  is what was trained on; the probe step, π_old of every replay pair and each replay's cumulative update must agree
  with independent fp32 oracles.
- **Exact weights.** What each engine holds after a publication is the engine's own cast of your fp32 master (bf16 for
  most parameters, fp32 for those the engine keeps in fp32): `dump_master` writes the master you just published, and
  `final/master_fp32.safetensors` is the master of the run's last publication.
- **Start-up.** `serve` must write `ready` within 25 minutes after the Judge's engines are up, and answer each later
  command within 30 minutes.

The Judge's own checksum queries and weight reloads are not counted in `T`; `docs/INTERFACE.md` lists every check.

## Public validation

```bash
python /opt/wan_public/run_public.py --suite SUITE   # SUITE: cpu, score, probe, small, perf or full
```

`cpu` checks the overlay whitelist, the code policy and the bitwise legacy branches; `score` your `score` command
against the official HPSv3 oracle; `probe` one `serve` probe step against the fp32 oracle; `small` a 3-round `run`
under the in-engine audit; `perf` the probe plus one timed replay of `pub0`, with its `T` and no score; `full` all
of them. No suite runs the quality eval. The suites run the Judge's own code as you on public data, start their own
engines and `serve`, and write results under `/scratch/public/`; the Judge repeats the checks on hidden data.

Suite outputs rotate. The output of the suite you run and of the suite before it stay complete; when a suite starts,
older suite outputs in the same directory keep their reports, logs and JSON records but lose their bulky files
(checkpoints, weights and weight copies, step tensors, generated videos). Only directories these suites created are
touched. Add `--keep-artifacts` to pin a suite's output so it is never rotated; it then counts against your disk
budget until you delete it.

## Score

Every check passes and the quality gate passes → `score = 3600 / median(T over the hidden dumps)`: how many times per
hour your implementation completes the fixed replay workload (`T` as defined in § Fixed replay). Any check fails, or
the quality gate fails → 0. There is no reference time in the score and no cap. Your final result is the best valid score you reach within
the time budget.

**Quality gate** (a gate, not a multiplier). The Judge loads the fp32 master of its own `run`
(`O/final/master_fp32.safetensors`) into two clean engines and generates one video per prompt of the public
Hao7/video_prompts `test.txt` (61 prompts, used by no training input) for each of 3 fixed seed sets, with a
deterministic flow-ODE sampler (704×1280, 9 frames, 50 steps, CFG 5, negative prompt `" "`). It scores frames 0/4/8
with HPSv3 (0.1 × mean μ, as in training) and compares the 3-repeat mean with the original model's, frozen on the same
prompts and seeds (0.7891). `quality_score` = final − original must be **≥ −0.28**. The contract's 6 AdamW steps do not
improve held-out HPSv3; the gate rejects a damaged final model. `/opt/wan_public/metric.json` holds the formula, the
frozen values, the gate and the eval protocol.

## Submission workflow

1. Run the public suites before submitting: correctness first (`cpu`, `score`, `probe`, `small`), then speed (`perf`).
2. In shared mode, where the Judge uses your two GPUs and pauses your workspace while it runs, stop every GPU
   process (your engines, Ray, `serve`) first. By default the Judge has its own pair.
3. Submit with `sforge-submit`. The result gives the score or why it is 0 (failed checks, or the phase where the
   Judge stopped), `T` per dump, the quality numbers and each check's status, but no logs.

Put Ray, temp files, checkpoints and other large files under `/scratch`, never into the submitted tree: the container
root disk is tiny. `/dev/shm` has 128 GiB and counts against the 256 GiB of host memory. In the Judge only `repo/`,
`/models/*` and `/data/public` exist (`docs/INTERFACE.md`). Keep design notes and measurements in `notes.md`.

**Disk.** `/scratch` is on a disk shared with the other runs on this host. Keep what you store there under
**200 GB** at any time (check with `du -sh /scratch`): delete checkpoints, weight copies, Ray spill and
public-suite outputs you no longer need. Filling the shared disk breaks your run and the others.
