# Miles ALFWorld task: executable interface

## Commands

```text
python task_entry.py run   --contract CONTRACT.json --out NEW_DIR
python task_entry.py probe --contract CONTRACT.json --batch BATCH.json --out NEW_DIR
```

The working directory is the submission root (`/app/workdir/miles_lab`). `--out` does not
exist beforehand. Both commands exit 0 only after writing every output below. The Judge
runs them as an unprivileged user with the two GPUs visible as CUDA devices 0 and 1 and no
network, and stops `probe` after 1800 s and `run` after 7300 s (start-up included).

Both commands run with `ALFWORLD_AUDIT_DIR` and `ALFWORLD_AUDIT_ROUNDS` in their
environment. Keep them in the environment of every process you start: Ray workers inherit
them from `ray start`, and a `runtime_env` you pass should carry them too. Without them the
evaluator has no evidence and the checks fail. Policy SGLang engines must run with
tensor-parallel size 1, and the Megatron trainer with tensor- and pipeline-parallel size 1.

### Contract (public copy: `/opt/miles_alfworld_public/contract.json`)

| field | meaning |
| --- | --- |
| `policy_hf` | Hugging Face snapshot directory of Qwen3-4B (the initial policy) |
| `policy_megatron` | the initial policy converted to Megatron `torch_dist` |
| `alfworld_data` | root of the game files; a row's game is `alfworld_data/<row.game>` |
| `train_data` | JSONL rows `{row_id, game}`; round `r` uses lines `[r*G, (r+1)*G)` |
| `rounds`, `games_per_round` (G), `samples_per_game` | the size of the run; rounds and games per round differ between the public and the scored run |
| `max_steps`, `max_turn_tokens` | 30 actions per episode, 256 new tokens per model turn |
| `system_prompt` | the system message of every episode |
| `sampling` | `temperature` 1.0, `top_p` 1.0 |
| `max_staleness` | 1 |
| `loss` | `eps_clip` 0.2, `eps_clip_high` 0.28 |
| `optimizer` | AdamW `lr`, `betas`, `eps`, `weight_decay`, `clip_grad` |
| `template_kwargs` | `{"enable_thinking": false}` for every chat-template rendering |
| `seed`, `gpus` | 1234, 2 |
| `scratch` | a writable directory for checkpoints and temporary files |

The public `perf` suite runs that contract as it is (4 rounds of 8 games); `small` uses its
first 3 rounds of 4 games. Implementations must not assume any fixed size. The Judge passes its
own contract, with larger sizes and with `train_data` and `alfworld_data` pointing at hidden
rows and their game files.

### `probe`

`BATCH.json` is `{"lr": float, "episodes": [{row_id, tokens, loss_mask, engine_logprobs, reward}]}`:
recorded episodes, `samples_per_game` per `row_id` (a GRPO group). `tokens` is the full
token sequence of the episode; `loss_mask` and `engine_logprobs` cover its response region
(every token after the first prompt; `engine_logprobs` is 0 where `loss_mask` is 0). Perform
exactly one optimizer step from the initial weights on exactly these episodes, with `lr` from
the batch and everything else from the contract. Do not sample.

### Outputs (both commands)

- `OUT/result.json`: `{"status": "completed", "final_hf_dir": "<OUT>/final_hf"}`
- `OUT/final_hf/`: the trained policy as Hugging Face safetensors with the original
  parameter names and shapes (other files are ignored; the Judge uses the pinned config
  and tokenizer).
- `OUT/steps/step_{k:03d}_rank_{r}.pt`: one `torch.save` dict per optimizer step `k` and
  data-parallel rank `r`, holding what that rank trained on. Per trained sequence `i`
  (`R_i` = response length):

| key | type | content |
| --- | --- | --- |
| `step` | int | `k` |
| `loss` | float | the loss the backward pass used for step `k`: the global token mean over all ranks |
| `tokens` | list of int64 tensors `[T_i]` | the full token sequence of each trained episode |
| `response_lengths` | list of int | `R_i`: the loss region is the last `R_i` tokens |
| `loss_masks` | list of tensors `[R_i]` | 1 exactly on tokens the policy generated |
| `engine_logprobs` | list of float tensors `[R_i]` | the SGLang log-probability of each generated token (0 elsewhere) |
| `train_logprobs` | list of float tensors `[R_i]` | the trainer's log-probability of each token under the pre-update weights (the values the loss used) |
| `advantages` | list of float | the sequence's GRPO advantage |

## What the checks look at

`run_public.py` (the Judge runs the same checks on hidden data, with its own random choices):

- **probe**: every batch episode trained exactly once with its mask and engine log-probs;
  advantages are GRPO of the batch rewards per row; `loss` equals the fp64 recomputation
  from the dump; an independent fp32 model reproduces `train_logprobs` and the loss;
  provenance: exactly one Megatron-Core optimizer step driven by Miles'
  `MegatronTrainRayActor` with the contract's AdamW and the batch's lr, whose result is
  `final_hf`.
- **run, provenance**: exactly one Megatron-Core optimizer step per round inside Miles'
  `MegatronTrainRayActor` from the pinned initial policy; before the first training
  generation the engine serves those weights, and its k-th publication after that is
  bit-identical (fingerprint of the input embedding) to Megatron's weights after step k-1;
  `final_hf` is Megatron's weights after the last step; the engine was started by a Ray actor.
  `final_hf` is also, tensor by tensor, exactly Megatron's weights right after the last
  optimizer step and exactly what every policy engine served after its last publication. The
  hook records the SHA-256 of every weight tensor: in the training process before the last
  step returns (Megatron's names, layout and dtype), and in each engine after each publication
  that can be the last (served layout and dtype). Every engine is digested and must match; a
  missing digest on any engine fails. Every tensor in every `final_hf/*.safetensors`
  must match the trainer's digest and each engine's. Converted to Megatron's layout it must
  match the trainer's: q/k/v interleaved per query group, `linear_fc1` = [gate; up], the
  RMSNorm weights where Megatron keeps them; this needs whole, unpadded tensors in the trainer.
  Converted to the served dtype and fused as SGLang serves it (q/k/v, gate/up), it must match
  each engine's. A tied `lm_head.weight` must equal the embedding, and no other tensor may be
  present. The trainer's digest and those after each engine's last two publications are
  removed from `T`. Each such interval, and each snapshot copy, must lie where the hook takes it and last
  at most 120 s.
- **run, audit** (every request is recorded inside the engine; in the Judge, the timed
  window must also lie inside the run by the Judge's own clock, with at most 300 s of
  evaluator intervals removed from it): requests chain into
  episodes by exact token prefix; each episode starts with the rendering of an assigned
  game's first observation; every model turn, parsed with the action rule and replayed in
  the evaluator's own ALFWorld, gives exactly the next request's appended tokens; episodes
  end exactly on a win or at `max_steps`. Sampling uses the contract's parameters, at most
  `max_turn_tokens` per turn, returns the sampled token's log-prob and nothing else. Every
  row is played `samples_per_game` times in its own round; every episode is trained exactly
  once, at that round's step; every turn of an episode trained at step s ran entirely under
  weight version s-1 or s (no publication inside a turn); one publication per step. The
  dumped sequences, masks, engine log-probs and advantages equal the engine's records and
  GRPO of the replayed rewards; every step's `loss` equals its fp64 recomputation.
- **run, oracle**: before two publications chosen at random, each
  policy engine's served weights are copied (about 8 GB each, under the audit directory, time
  removed from `T`); at most two copies per engine, and at least one must hold the weights a
  step in 1..rounds-1 started from. Step 0 and one random copied step are recomputed in fp32
  (the trainer's log-probs and the loss). With `--probe-output` (always in the Judge), the
  numerics-relevant Megatron model configuration of the run equals the probe's.
- **perf**: the public contract, the audited timed window `T`, and greedy success on the 8
  public evaluation games in 3 repeats. It is for getting the pipeline working and reports no
  score: the run has too few steps to show learning, 8 games say nothing about quality, and `T`
  does not predict the scored run's.

Reports go to `OUTPUT/report.json`; exit code 0 means every check of the suite passed.
