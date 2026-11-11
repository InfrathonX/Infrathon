# Miles × ALFWorld: integrate the environment, then make agentic GRPO fast (two GPUs)

Hardware: two NVIDIA H100 GPUs on one NVLink pair, assigned to this run. Their memory and the CPUs you get depend on the host: check them with `nvidia-smi` and `nproc` and size your configuration from what you see. The container has 256 GiB of host memory.

## Goal

`repo/` is Miles (SGLang + Megatron RL) at commit `78527b9`. It already trains multi-turn
agents: an agent function can talk to the policy through Miles' session server
(OpenAI-compatible, token-in-token-out), and GRPO, weight publication and asynchronous
training are all there. It has no ALFWorld.

Users want **ALFWorld** — a text household environment widely used for agentic RL — as a Miles environment: the policy plays a game over many turns,
reads each observation, emits one action per turn, and is trained with GRPO on whether it
won. Your job has two parts:

1. **Integrate** ALFWorld into Miles so that the training below runs correctly.
2. **Make it fast, and keep it learning**: the score is the hidden-set success the complete
   training run adds to the policy, per hour of timed training (see Score): a shorter run and a policy that learns more both raise it.

## Starting point

- `task_entry.py run|probe`, `config/` and `repo/` are yours; `docs/INTERFACE.md` defines the
  commands, the contract file, the required outputs and the precise checks.
- The starter `task_entry.py` has the Ray/checkpoint plumbing only; `run` and `probe` raise.
  It scores 0.
- `alfworld==0.4.2` (TextWorld backend) is installed; the public game files are under
  `/data/public/games`. Models are under `/models` (the initial policy is Qwen3-4B), the
  initial Megatron checkpoint under `/data/megatron`, public data under `/data/public`.
- The container has a 24-CPU quota on the CPUs it is pinned to (`nproc`).

## The training you must implement

**Episodes.** For training row `r` (`{row_id, game}`), load `alfworld_data/<game>` with
TextWorld (`AlfredDemangler(shuffle=False)`, `AlfredInfos`, `EnvInfos(won, admissible_commands)`,
`max_episode_steps = max_steps`). The conversation is:

- system: the contract's `system_prompt`;
- user: the observation, rendered as
  `obs.strip() + "\n\nAdmissible actions: " + ", ".join(adm')`, where `adm'` is the
  admissible commands in the engine's order without `look` and `inventory`, followed by
  `look` and `inventory`;
- assistant: the policy's reply; then the next observation as a user turn, and so on.

Render with the Qwen3 chat template and `enable_thinking=false`. The whole history is kept
(no truncation, no summarisation) and is **token-in-token-out**: each request's prompt is
exactly the previous request's prompt + the tokens the engine generated + the rendering of
the new user turn. Never re-tokenize what the model generated.

**Actions.** Decode the reply (skipping special tokens), take the text inside the first
`<action>…</action>`, strip it and send it to the game as is (an empty string if there is
none). The episode ends when the game is won or after `max_steps` (30) actions. Reward:
1 if won, else 0.

**Sampling.** Temperature 1, top-p 1, at most `max_turn_tokens` (256) new tokens per turn;
request the sampled token's log-probability and nothing else (no top-k log-probs, logit
bias, stop strings or other constraints).

**Loss (GRPO, token mean).** One trained sequence per episode: the last request's prompt +
its generated tokens. Loss mask 1 exactly on generated tokens. Group = the `samples_per_game` episodes of a
row: `A_i = (R_i − mean) / (std + 1e-6)` with the unbiased std, `A_i = 0` if all rewards in
the group are equal. `ρ_t = exp(log π_θ(y_t) − log π_engine(y_t))` where `log π_engine` is
the log-probability SGLang returned when it sampled `y_t`.

```
L = Σ_t m_t · max(−A ρ_t, −A · clip(ρ_t, 1 − 0.2, 1 + 0.28)) / Σ_t m_t      (all tokens of the step's sequences)
```

No KL term, no entropy term, no reward shaping.

**Optimizer and publication.** Exactly one AdamW update per round (lr 2e-6, betas (0.9, 0.999),
eps 1e-8, weight decay 0.01, gradient clipping 1.0, constant lr), done by Megatron-Core
inside Miles' `MegatronTrainRayActor`, starting from the given initial weights
(`policy_megatron` / `policy_hf`). After every update, publish the weights to the sampling
engine(s).

## Workload contract

- The contract gives `train_data`, `alfworld_data`, `rounds`, `games_per_round` (`G`) and
  `samples_per_game` (`S`). The public and the scored runs differ in rows, games, rounds and
  games per round, so read these from the contract and assume no fixed size.
- Round `r` plays rows `[r*G, (r+1)*G)` of `train_data`, `S` episodes per row; all of them are
  trained at step `r`, each exactly once, each row used once, in file order.
- **Staleness.** The engine's weight version is the number of publications so far (0 =
  initial). Every turn of an episode trained at step `s` must be generated entirely under
  version `s−1` or `s`: an episode may continue across a publication, a single model turn may
  not.

## What you may change

You may change anything about how this runs: Miles itself (repo/), how environments are
hosted and parallelised, SGLang engine settings, request routing and prefix caching,
overlap of sampling and training (within the staleness rule), how long episodes are
handled, batching, parallelism and GPU placement, and how weights are published. CPU optimizer offload,
recomputation and batching are allowed while preserving the required optimizer, precision
and loss.

You may not change the model, the games or their order, the rendering, the action rule,
the reward, the loss, the optimizer settings or the quota; you may not modify ALFWorld or
its game files; and the training must stay a Miles job — Megatron-Core inside Miles'
training actor, SGLang engines started by Miles.

**Your own implementation.** You may read papers and other projects' code to understand the method, but the
implementation you submit is written natively in this checkout. Do not copy in an existing implementation of the
feature from elsewhere (another framework, or another version of this one), whether as copied files, a vendored
package or a wrapper that calls it.

## Correctness requirements

- **Instrumentation.** Both commands run with `ALFWORLD_AUDIT_*` environment variables that
  must reach every process you start (Ray workers, SGLang servers). Policy SGLang engines run
  at tensor-parallel size 1 and the Megatron trainer at tensor- and pipeline-parallel size 1;
  data parallelism and GPU placement are yours.
- **Same implementation.** `probe` and `run` use the same implementation; only the data
  source differs.
- **Exact weights.** The weights every policy engine serves after each publication, and
  `final_hf`, are exactly the weights Megatron's optimizer steps produced: `final_hf` is
  Megatron's weights after the last optimizer step and what every engine served after its
  last publication. No training or weight change after the last step or publication, and no
  publishing of weights other than the trainer's.
- **Correct steps.** Every episode is played in ALFWorld exactly as specified above, and
  every step's records (`docs/INTERFACE.md`) are what was trained on. The step itself (the
  trainer's log-probabilities and the loss) must agree with an independent float32
  recomputation from the weights the step started from.

The evaluator copies and digests weights during the run to check this; that time is not
counted in `T` (limits in `docs/INTERFACE.md`).

## Public validation

```bash
python /opt/miles_alfworld_public/run_public.py --suite cpu   --output /scratch/public/cpu-001
python /opt/miles_alfworld_public/run_public.py --suite probe --output /scratch/public/probe-001
python /opt/miles_alfworld_public/run_public.py --suite small --output /scratch/public/small-001 --probe-output /scratch/public/probe-001
python /opt/miles_alfworld_public/run_public.py --suite perf  --output /scratch/public/perf-001  --probe-output /scratch/public/probe-001
```

`cpu` checks rendering, action parsing and the observation format on a recorded run (no
GPU); `probe` one GRPO step on fixed episodes; `small` a short run (3 rounds × 4 games); and
`perf` a longer one (4 rounds × 8 games) with its `T` and greedy success on 8 public games in
3 repeats. They run the Judge's checks on
small public data and report no score: a run this short shows that the pipeline works, not how
fast it will be at scale or how much the policy will learn. The Judge repeats the checks on
hidden data. When a GPU
suite ends it deletes the output's `scratch/`, `final_hf/`, checkpoints and served-weight
copies (reports, logs and step dumps stay); add `--keep-artifacts` to keep them, and delete
them yourself afterwards.

## Score

The Judge runs `probe` on hidden recorded episodes, then `run` on hidden training rows, then
evaluates the final policy greedily (temperature 0, the same rendering and action rule) on
hidden games, three times.

```
quality_score = mean success of the final policy over the 3 repeats
                − mean success of the original Qwen3-4B (same games and protocol, measured once),
                in percentage points
score         = quality_score × 3600 / T     if every check passes and quality_score ≥ 15
              = 0                             otherwise
```

- `T`: wall clock of your `run`, from the first training generation request to the end of the
  last weight publication, taken from the evaluator's own records, minus the evaluator's
  weight copies and digests. Start-up (model loading, compilation, CUDA graphs) is not timed.
- Training faster and learning more both count; there is no reference time and no cap.
  Sampling is stochastic, so the same code gains different amounts from run to run.

## Submission evaluation policy

**The public suites are only for getting your pipeline working and passing its checks.** The scored run is a different
and much larger workload, so public timings, memory use and success do not predict your score. Think about how your
pipeline behaves at a larger scale and how to optimize it, and measure real speed, memory use and learning with
submissions.

Each `sforge-submit` runs your code through the whole pipeline on hidden data, from a clean copy of your submission:
`probe`, a complete `run` from the initial weights, the correctness checks, and the evaluation of the final policy. A
failure at any stage ends the evaluation with a score of 0, and the report lists the failed checks (submit with
`sforge-submit --details` to see each check's message). The Judge stops `probe` after 1800 s and `run` after 7300 s,
start-up included. A submission takes far longer than a public suite and uses part of your time budget, so plan them.
Keep your strongest version in the workspace when the budget ends.

Your final result is the best valid score you reach within the time budget.

## Submission workflow

1. Get the pipeline working with the public suites.
2. Stop every GPU process (including Ray and SGLang).
3. Submit with `sforge-submit`.

Write only under `/scratch` and `/app/workdir`; put large files under `/scratch`, never into
the submitted tree. Keep design notes and measurements in `notes.md`.

**Disk.** `/scratch` is on a disk shared with the other runs on this host. Keep what you store there under
**200 GB** at any time (check with `du -sh /scratch`): delete checkpoints, weight copies, Ray spill and
public-suite outputs you no longer need. Filling the shared disk breaks your run and the others.
