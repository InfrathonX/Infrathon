# Miles OPD task: executable interface

## Commands

```text
python task_entry.py run   --contract CONTRACT.json --out NEW_DIR
python task_entry.py probe --contract CONTRACT.json --batch BATCH.json --out NEW_DIR
```

The working directory is the submission root (`/app/workdir/miles_lab`). `--out` does not
exist beforehand. Both commands exit 0 only after writing every output below. The Judge
runs them as an unprivileged user with the two GPUs visible as CUDA devices 0 and 1 and no
network.

Both commands run with `OPD_AUDIT_DIR` and `OPD_AUDIT_ROUNDS` in their environment.
Keep them in the environment of every process you start: Ray workers inherit them from
`ray start`, and a `runtime_env` you pass should carry them too. Without them the
evaluator has no evidence and the checks fail. Student SGLang engines must run with
tensor-parallel size 1, and the Megatron trainer with tensor- and pipeline-parallel size 1.

### Contract (public copy: `/opt/miles_opd_public/contract.json`)

The Judge passes its own contract, with other `train_data` and larger sizes.

| field | meaning |
| --- | --- |
| `student_hf`, `teacher_hf` | Hugging Face snapshot directories of Qwen3-1.7B and Qwen3-4B |
| `student_megatron` | the initial student converted to Megatron `torch_dist` |
| `train_data` | JSONL rows `{row_id, content: [{role: user, content}], answer}` (DAPO-Math-17k problems; `answer` is never needed for training); round `r` uses lines `[r*n, (r+1)*n)` |
| `rounds`, `prompts_per_round`, `max_response_len` | the size of the run; rounds and prompts per round differ between the public and the scored run |
| `sampling` | `temperature` 1.0, `top_p` 1.0 |
| `max_staleness` | 1 |
| `kl_coef` | `β` = 1.0 |
| `ppo_clip` | 0.2 |
| `optimizer` | AdamW `lr`, `betas`, `eps`, `weight_decay`, `clip_grad` |
| `template_kwargs` | `{"enable_thinking": false}` for every chat-template rendering |
| `seed`, `gpus` | 1234, 2 |
| `scratch` | a writable directory for checkpoints and temporary files |

The public `perf` suite runs that contract as it is (8 rounds of 32 prompts); `small` uses its
first 4 rounds of 16 prompts. Implementations must not assume any fixed size.

### `probe`

`BATCH.json` is `{"lr": float, "samples": [{"row_id", "prompt_ids": [...], "response_ids": [...]}]}`.
Perform exactly one on-policy-distillation optimizer step from the initial weights on
exactly these sequences (prompt ids already rendered; every response token counts), with
`lr` from the batch and the other hyper-parameters from the contract. The teacher scores
these sequences as in `run`. Do not sample.

### Outputs (both commands)

- `OUT/result.json`: `{"status": "completed", "final_hf_dir": "<OUT>/final_hf"}`
- `OUT/final_hf/`: the trained student as Hugging Face safetensors with the original
  parameter names and shapes (other files are ignored; the Judge uses the pinned config
  and tokenizer).
- `OUT/steps/step_{k:03d}_rank_{r}.pt`: one `torch.save` dict per optimizer step `k` and
  data-parallel rank `r`, holding what that rank trained on:

| key | type | content |
| --- | --- | --- |
| `step` | int | `k` |
| `loss` | float | `L` of step `k` at the pre-update weights: `(1/N) sum_i mean_t (log p_old_t - log q_t)` over all `N` sequences of the step (all ranks) |
| `tokens` | list of int64 tensors `[P+R]` | exact prompt + response ids of each trained sequence |
| `response_lengths` | list of int | `R` per sequence |
| `loss_masks` | list of tensors `[R]` | 1 for tokens in the loss |
| `old_logprobs` | list of float32 tensors `[R]` | `log p_old_t`: pre-update student log-probability of each sampled response token (training side, full-vocabulary normalisation) |
| `teacher_logprobs` | list of float32 tensors `[R]` | `log q_t`: teacher log-probability of each sampled response token |
| `advantages` | list of float32 tensors `[R]` | `A_t` as used by the loss |

For the first round of a run and for the probe, `old_logprobs` are the initial weights'
log-probabilities.

## What the checks look at

`run_public.py` (the Judge runs the same checks on hidden data, with its own copies and its own
random choices):

- **probe**: the dump matches the batch; `advantages == -β (old_logprobs - teacher_logprobs)`
  and `loss` equals the value recomputed in float64 from the dumped tensors; the student
  and teacher log-probabilities agree with an independent float32 computation; the weight
  change follows the on-policy-distillation gradient and (Judge) agrees with the change
  made by the task's reference implementation on the same batch (on the weights with the
  largest gradients); provenance: exactly one
  Megatron-Core optimizer step, driven by Miles' `MegatronTrainRayActor`, with the
  contract's AdamW settings and the batch's lr, whose result is `final_hf`.
- **run, provenance**: exactly one Megatron-Core optimizer step per round per training
  process, each driven by Miles' `MegatronTrainRayActor`, with Megatron-Core loaded from
  `/root/Megatron-LM` and the contract's AdamW settings. Megatron starts from the pinned
  initial student. Before the first training generation the student engine serves those
  initial weights, and its k-th publication after that is bit-identical (as a fingerprint
  of the input embedding) to Megatron's weights after step k-1. `final_hf` is Megatron's
  weights after the last step. The student engine process was started by a Ray actor.
  `final_hf` is also, tensor by tensor, exactly Megatron's weights right after the last
  optimizer step and exactly what each student engine served after its last publication. The
  hook records the SHA-256 of every weight tensor twice: in the training process before the
  last step returns (the GPTModel that step updated, in Megatron's names, layout and dtype), and
  in the engine after each publication that can be the last (served layout and dtype).
  Every tensor in every `final_hf/*.safetensors` must match both digests. Converted to
  Megatron's layout it must match the first: q/k/v interleaved per query group, `linear_fc1` =
  [gate; up], the RMSNorm weights where Megatron keeps them; this needs whole, unpadded tensors
  in the trainer. Converted to the served dtype and fused as SGLang serves it (q/k/v, gate/up),
  it must match the second. A tied `lm_head.weight` must equal the embedding, and no other
  tensor may be present. The trainer's digest and the digest after the engine's last
  publication are removed from `T`. With an initial push the engine also digests after the
  publication before its last one; that interval is removed only until the trainer starts its
  last step (its first grad-enabled forward after the step before), because from then on it
  runs beside training. Each digest must start within 5 s of the optimizer step or publication
  it follows and last at most 12 s, and the
  earlier engine digest must end before the last publication completes.
- **run, random steps**: before four publications chosen at random, the student engine's
  served weights are copied (about 3.4 GB
  each, under the audit directory). The steps those weights started are recomputed in
  float32 (the public suite recomputes the first two copies, the Judge all four): `old_logprobs` under those weights,
  `teacher_logprobs`, `loss`, and the gradient of `L` at those weights with the dumped
  `advantages` (at ratio 1: `(1/N) sum_i (1/|R_i|) sum_t -A_t grad log π_θ(y_t)`). That
  gradient must match the reduced gradient Megatron held just before its optimizer step
  (direction and magnitude on the first rows of the input embedding, and the total
  gradient norm Megatron reported). This check fails when more than half of the recomputed
  steps miss these limits (one step can miss them through bf16 noise alone; a different loss
  misses them on every step). The teacher log-probabilities of random rows of two
  more random steps are also recomputed. The copies' time is removed from `T`. Each copy
  must lie in its own engine's publication timeline, after the engine's previous publication
  completed and before the publication it precedes completed, and hold the version the engine
  served then; at most four per engine, each lasting at most 12 s. A copy record anywhere
  else fails the run.
- **run, same path**: with `--probe-output` (always in the Judge), the numerics-relevant
  Megatron model configuration of the run equals the probe's.
- **run, audit** (in the Judge, the timed window must also lie inside the run by the
  Judge's own clock, with at most 36 s of evaluator intervals removed from it in total;
  see the end of this item): every
  request to an SGLang engine is recorded inside the engine. Each
  round has exactly `prompts_per_round` student generations with the contract's sampling parameters whose
  prompts are the assigned rows, each generation is trained on exactly once at a step
  within the staleness limit, and there is one weight publication per step (a publication
  is a group of weight-update calls with no generation in between). The dumped
  `old_logprobs` are compared with the log-probabilities the sampling engine returned for
  the same tokens: they must come from a training-side forward, so they may not copy the
  engine's values. Every step's
  `advantages` and `loss` must be consistent with its dumped log-probabilities, and step 0
  is recomputed in float32 from the initial student (`old_logprobs`, `teacher_logprobs`
  and `loss`). The 36 s bound is for the whole run, not per student engine. One student
  engine's copies and digests take about 16 s. Each
  further student engine adds its own copies and digests: two student engines come close to
  the bound, and three or more exceed it, which fails the run.
- **perf**: the public contract, the audited timed window `T` (minus the evaluator's weight
  copies and digests), and the final student's accuracy and answer-format rates on the 256 public
  test problems (`test_public.jsonl`) with the Judge's protocol (3 seeded repeats at temperature 1,
  at most `max_response_len` new tokens, the Judge's grader), next to the original student's accuracy
  on those problems (`published_reference.json`). It is for getting the pipeline working and reports
  no score: the run is far shorter than the scored one and one problem is 0.39 points, so neither its
  accuracy nor its `T` predicts the scored run's.

Reports go to `OUTPUT/report.json`; exit code 0 means every check of the suite passed.
