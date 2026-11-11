# Executable interface — Wan2.2-TI2V-5B DanceGRPO + HPSv3 (`miles_diffusion_wan22_dancegrpo_2xh100_v2`)

Your submission is the directory `repo/`: miles_diffusion plus `repo/task_entry.py` (the three commands below) and
`repo/engine_overlay/` (your versions of the whitelisted sglang-diffusion files, same relative paths under
`python/sglang/multimodal_gen/`). The Judge copies `repo/` fresh for every phase and runs it as the unprivileged user
`agent`, offline, with `cwd = repo/`. Nothing of yours is imported by the Judge. The Judge container mounts the same
read-only `/models/*` and `/data/public` as your workspace. Nothing else from your workspace is there: not your
`/scratch`, home or `/tmp` contents, and no files outside `repo/`. Write to the contract's `scratch`. The clean
engine is at
`/sgl-workspace/sglang/python`. The Judge lays your overlay over a copy of it and passes the result in the contract as
`engine_python_path`. Put that path first on `PYTHONPATH` for every engine process you start, so your engines run with
your overlay.

## Contract (`--contract C`, JSON)

```
model              /models/Wan2.2-TI2V-5B-Diffusers
hpsv3_checkpoint   /models/HPSv3/HPSv3.safetensors
hpsv3_base         /models/Qwen2-VL-7B-Instruct            (config / processor / tokenizer only)
engine_python_path the overlaid engine (run only; null in serve: the Judge owns the engines there)
scratch            a writable directory for this phase (put Ray / temp here)
sampling  {height 704, width 1280, num_frames 9, steps 10, guidance 5.0, negative_prompt " ", flow_shift 5.0,
           eta 1.2, window_size 2, window_range [0, 5]}
optim     {lr 1e-5, weight_decay 1e-4, betas [0.9, 0.999], eps 1e-8, grad_clip 1.0, clip_range 1e-4, adv_clip 5.0}
run       {prompts <jsonl of {"row", "prompt"}>, rounds 3, prompts_per_round 8, n 8, run_seed <int>,
           optimizer_steps_per_round 2}
replay    {prompts_per_dump P, n 8, optimizer_steps P/4, probe_optimizer_steps 1, engine_urls [<url>, <url>]}
```

## `score --videos V.npz --prompts P.jsonl --out O`

`V.npz` holds `videos` uint8 `[N, 9, 704, 1280, 3]`. `P.jsonl` has one `{"prompt": ...}` line per video, in order.
Write `O/scores.json`:
`{"frame_mu": [[mu0, mu4, mu8], ...N], "reward": [0.1 * mean(mu), ...N]}`. Each entry is the official HPSv3 μ of frames
0, 4 and 8 (AGENT_TASK.md §3).
Checked per frame against the official inferencer (J2). The set includes natural videos, videos whose frames 0/4/8
come from different videos, and flat/black/white frames.

## `run --contract C --out O` (real RL, audited; not timed)

- **Rounds.** R = 3 rounds; round r uses contract prompt rows `[8r, 8r+8)` in file order. Each prompt gets 8 samples,
  one engine request per sample, with the canonical keys of AGENT_TASK.md §1 (`run_seed` from the contract, `round`
  = r, `row` = the prompt's `row` field).
- **Publications.** Publish the base weights to your engines before round 0, then again after each round's training.
  That is R+1 = 4 full publications per engine, through `update_weights_from_tensor`.
- **Lag 0.** Every request of round r must be issued after publication r completed on both engines, and before
  publication r+1 starts.
- **Optimizer steps.** 2 per round, each over 64 pairs. A pair is `(group, j, k)` for k in sample j's window.
  Optimizer step s of round r trains the pairs of prompts `[4s, 4s+4)` of that round.
- **Outputs.**
  - `O/pairs/`: the per-pair log below; one file per round and rank, named `rollout{r:03d}_rank{rank}.jsonl`.
  - `O/final/master_fp32.safetensors`: the fp32 master (diffusers parameter names of `transformer`) whose publication
    was the last one.
- **Engine requests.** The Judge's hook records every engine request inside the engine (merged SamplingParams), every
  denoising step `(x_k, v_k, σ_k, σ_{k+1}, x_{k+1})` and every publication (coverage of all parameter names and
  numels, plus a checksum). It checks them against the contract (J3/J4).
  - Sampling fields: `width 1280`, `height 704`, `num_frames 9`, `num_inference_steps 10`, `guidance_scale 5`,
    `true_cfg_scale 5`, `negative_prompt " "`, `rollout true`, `rollout_sde_type "dance"`, `rollout_noise_level 1.2`,
    `generator_device "cpu"`, `num_outputs_per_prompt 1`.
  - Keys: `seed = H(init|base)`, `rollout_sde_noise_seed = H(sde|base|j)`,
    `rollout_sde_step_indices = [w_j, w_j+1]`.
  - No explicit sigmas and no caches.

## `serve --contract C --ctl F --events E --out O` (fixed replay; one timed window per replay)

The Judge starts two sglang-diffusion engines (clean engine, one per GPU, your uid) before your process and puts their
URLs in `replay.engine_urls`. Do not start, stop or restart them. Load your trainer at θ0 and do all heavy setup. Then
write `{"event": "ready"}` to `E` and read JSON lines from `F`:

| command | you do | events you write |
|---|---|---|
| `{"cmd":"probe","id":..,"dump":D,"prompts":1,"theta1_path":P}` | one optimizer step on the dump's single prompt (16 pairs), no publication; write the fp32 θ1 to `P` (safetensors, diffusers names) | `log_complete` then `done` |
| `{"cmd":"replay","id":..,"dump":D}` | **timed**: HPSv3 on frames 0/4/8 of every video → advantages → π_old of every pair → one optimizer step per 4 dump prompts (step s = pairs of dump prompts `[4s, 4s+4)`) → fsync the per-pair log → **one** full publication of the transformer to every engine URL | `{"event":"log_complete","id":..}` after the log is on disk, then `{"event":"done","id":..}` |
| `{"cmd":"dump_master","id":..,"path":P}` | write your fp32 master to `P` (untimed) | `done` |
| `{"cmd":"reset","id":..}` | restore θ0 and a fresh AdamW state (untimed) | `done` |
| `{"cmd":"stop"}` | exit | `{"event":"stopped"}` |

Every event is one JSON line: `{"event": ..., "id": <the command's id>}`. `{"event": "error", "message": ...}`
aborts the Judge.
Per-pair logs:
- the probe → `O/pairs/probe00/`;
- the i-th replay (1-based, in the order of `replay` commands) → `O/pairs/replay{i:02d}/`.

Each is written as `rollout*_rank*.jsonl`.

The replay size comes from the contract (`replay.prompts_per_dump`, `replay.optimizer_steps`):
`/data/public/contract.json` gives the public dump's (8 prompts, 2 steps), and the Judge passes its own.

**T** runs from the moment the dump becomes readable to the later of two events: the last engine confirming a complete
publication, and your `log_complete`. The clock is the Judge's. The Judge reloads its engines to θ0 between replays.
The score is `3600 / median(T)` over the hidden dumps (AGENT_TASK.md § Score). The replay's per-pair log must already
be complete, byte for byte, when `log_complete` arrives (the Judge reads it then and again after `done`; R5), and after
`done` each engine's weight checksum, which the Judge queries, must equal the published one (R6′).

### Replay dump `D/`

`P` = the contract's `replay.prompts_per_dump`.

| file | content |
|---|---|
| `meta.json` | `run_seed`, `round`, `rows` [P], `prompts` [P], `n` 8, `windows` [P][8] (window start w per sample), `sigmas` [11], `timesteps` [11] |
| `frames.npy` | uint8 `[P, 8, 9, 704, 1280, 3]`, the engine's decoded videos |
| `latents.safetensors` | `x` fp32 `[P, 8, 3, 48, 3, 44, 80]` = `(x_w, x_{w+1}, x_{w+2})` per sample, as the engine produced them |
| `cond.safetensors` | `pos`, `neg` fp32 `[P, 512, 4096]` UMT5 conditioning (positive prompt, negative `" "`) |

The canonical ids of sample `(g, j)` are `group = f"{run_seed}|{round}|{rows[g]}"`, `j`, and `k ∈ {w, w+1}`. Pair
`(g, j, k)` goes from `x_k` to `x_{k+1}` with σ = `sigmas[k]` → `sigmas[k+1]`. No engine log-probs are given: π_old is
yours to compute on θ0 (the first optimizer step may reuse its own forward).

## Per-pair log (run and serve), one JSON object per line

```
{"kind":"pi_old","rollout":r,"group":"<run_seed>|<round>|<row>","j":j,"k":k,"lp_old":float}
{"kind":"step","rollout":r,"optim_step":s,"group":..,"j":j,"k":k,"adv":A,"raw_reward":R,"frame_mu":[mu0,mu4,mu8],
 "lp_old":float,"lp_new":float,"ratio":float,"clipped":bool,"loss":float}
```

- Every pair appears in exactly one `step` event per call, in the optimizer step that the composition rule assigns it.
- `pi_old` events cover the pairs whose π_old you recomputed separately. For pairs of the first optimizer step, whose
  π_old comes from the same forward, `lp_old` of the step event is used.
- `lp` is the Dance log-prob of AGENT_TASK.md §2 (mean over elements, constants dropped).
- The logged values must be self-consistent: `ratio = exp(lp_new − lp_old)`, `loss = max(−A·r, −A·clip(r, 1±1e-4))`,
  and `clipped = |r − 1| > 1e-4`.
- They are also compared against independent oracles: HPSv3 per frame, fp32 π_old of all pairs, fp32 first
  AdamW step, and the cumulative update of the replay's optimizer steps.

## What is compared how (summary; thresholds are Judge-private)

J0 overlay whitelist, AST policy on added code, legacy sde/cps/ode bitwise · J2 per-frame HPSv3 · J3 run audit (exact)
· J4 transitions from the engine's own tensors (x_T bitwise; ε̂ vs canonical ε; η² and velocity coefficients; Euler
residual) · J1 probe: π_old, AdamW first-step sign/magnitude · R1–R5 per replay: rewards, advantages, all π_old (fp32
oracle, signed and absolute), self-consistency, composition, one complete publication, the log final at
`log_complete` · R6′ (`R6_equal`) published weights == what each engine holds after `done` == engine cast of the
dumped master (bitwise) · R6 the master's cumulative update vs the fp32 oracle of the replay's optimizer steps (medians over
the hidden dumps) · J6 run's last publication == reload of `final/master_fp32` (also on the Judge's fresh quality-eval engines) ·
quality gate (after every check passed and `serve` stopped): that master's held-out HPSv3 on the 61 public Hao7
`test.txt` prompts × 3 fixed seed sets (fresh Judge engines, deterministic 50-step flow-ODE sampler) minus the
original model's frozen value must be ≥ −0.28.
