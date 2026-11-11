# Miles OPCD task: executable interface

## Commands

```text
python task_entry.py run   --contract CONTRACT.json --out NEW_DIR
python task_entry.py probe --contract CONTRACT.json --batch BATCH.json --out NEW_DIR
```

The working directory is the submission root (`/app/workdir/miles_lab`). `--out` does not
exist beforehand. Both commands exit 0 only after writing every output below. The Judge
runs them as an unprivileged user with the two GPUs visible as CUDA devices 0 and 1 and no
network.

Both commands run with `OPCD_AUDIT_DIR` and `OPCD_AUDIT_ROUNDS` in their environment.
Keep them in the environment of every process you start: Ray workers inherit them from
`ray start`, and a `runtime_env` you pass should carry them too. Without them the
evaluator has no evidence and the checks fail. Student SGLang engines must run with
tensor-parallel size 1, and the Megatron trainer with tensor- and pipeline-parallel size 1.

### Contract (public copy: `/opt/miles_opcd_public/contract.json`)

The Judge passes its own contract, with hidden `train_data` and larger sizes.

| field | meaning |
| --- | --- |
| `student_hf`, `teacher_hf` | the same Hugging Face snapshot directory of Qwen3-1.7B: the student starts from it, and the teacher is that snapshot, frozen, with `c` |
| `student_megatron` | the initial student converted to Megatron `torch_dist` |
| `context_file` | the system prompt `c` (teacher only) |
| `train_data` | JSONL rows `{row_id, content: [{role: user, content}], answer}`; round `r` uses lines `[r*n, (r+1)*n)` |
| `rounds`, `prompts_per_round`, `max_response_len` | the size of the run; rounds and prompts per round differ between the public and the scored run |
| `sampling` | `temperature` 1.0, `top_p` 1.0 |
| `max_staleness` | 1 |
| `top_k` | 256 |
| `optimizer` | AdamW `lr`, `betas`, `eps`, `weight_decay`, `clip_grad` |
| `template_kwargs` | `{"enable_thinking": false}` for every chat-template rendering |
| `seed`, `gpus` | 1234, 2 |
| `scratch` | a writable directory for checkpoints and temporary files |

The public `perf` suite runs that contract as it is (8 rounds of 64 prompts); `small` uses its
first 4 rounds of 16 prompts. Implementations must not assume any fixed size.

### `probe`

`BATCH.json` is `{"lr": float, "samples": [{"row_id", "prompt_ids": [...], "response_ids": [...]}]}`.
Perform exactly one OPCD optimizer step from the initial weights on exactly these
sequences (prompt ids already rendered without `c`; every response token counts), with
`lr` from the batch and the other hyper-parameters from the contract. Do not sample.

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
| `loss` | float | the loss the backward pass used for step `k`: the global token mean over all ranks |
| `tokens` | list of int64 tensors `[P+R]` | exact prompt + response ids of each trained sequence |
| `response_lengths` | list of int | `R` per sequence |
| `loss_masks` | list of tensors `[R]` | 1 for tokens in the loss |
| `opcd_topk_ids` | list of int64 tensors `[R, 256]` | the pre-update student's top-256 ids per response position |
| `opcd_old_topk_logprobs` | list of float32 tensors `[R, 256]` | pre-update student log-probabilities of those ids (full-vocabulary normalisation) |
| `opcd_teacher_topk_logprobs` | list of float32 tensors `[R, 256]` | teacher log-probabilities of those ids given `c` |

For the first round of a run and for the probe, `opcd_old_topk_logprobs` are the initial
weights' log-probabilities.

## What the checks look at

`run_public.py` (the Judge runs the same checks on hidden data, with its own copies and its own
random choices):

- **probe**: the dump matches the batch; `loss` equals the value recomputed in float64
  from the dumped tensors; the loss, top-256 coverage and teacher log-probabilities agree
  with an independent float32 computation; the weight change follows the OPCD gradient;
  provenance: exactly one Megatron-Core optimizer step, driven by Miles'
  `MegatronTrainRayActor`, with the contract's AdamW settings and the batch's lr, whose
  result is `final_hf`.
- **run, provenance**: exactly one Megatron-Core optimizer step per round per training
  process, each driven by Miles' `MegatronTrainRayActor`, with Megatron-Core loaded from
  `/root/Megatron-LM` and the contract's AdamW settings. Megatron starts from the pinned
  initial student. Before the first training generation the engine serves those initial
  weights, and its k-th publication after that is bit-identical (as a fingerprint of the
  input embedding) to Megatron's weights after step k-1. `final_hf` is Megatron's weights
  after the last step. The engine process was started by a Ray actor.
  Nothing between two optimizer steps changes the student's weights or the optimizer's state:
  at every step's entry and return the hook fingerprints every student parameter and every
  tensor the optimizer holds (fp32 main params, moments) at random positions, drawn afresh at
  each return, and each step must enter with the fingerprint the previous one returned with
  (a small cost per step, inside `T`). A data-parallel trainer must hold complete weights when
  the step returns (no `--overlap-param-gather`).
  `final_hf` is also, tensor by tensor, exactly Megatron's weights right after the last
  optimizer step and exactly what the engine served after its last publication. The hook
  records the SHA-256 of every weight tensor twice: in the training process before the last
  step returns (Megatron's names, layout and dtype), and in the engine after each publication
  that can be the last (served layout and dtype). Each digest takes a few seconds.
  Every tensor in every `final_hf/*.safetensors` must match both digests. Converted to
  Megatron's layout it must match the first: q/k/v interleaved per query group, `linear_fc1` =
  [gate; up], the RMSNorm weights where Megatron keeps them; this needs whole, unpadded tensors
  in the trainer. Converted to the served dtype and fused as SGLang serves it (q/k/v, gate/up),
  it must match the second. A tied `lm_head.weight` must equal the embedding, and no other
  tensor may be present. The trainer's digest and those after every engine publication that
  can be the last are removed from `T`. Each digest must start within 5 s of the record it
  follows and end before the engine's next publication completes; each weight copy below must
  lie in its engine's publication timeline (after the previous publication completed, before
  its own completed); each lasts at most 60 s. `T` loses at most 8 s of each, and nothing of
  the time the trainer spends in a training step (from its first grad-enabled forward to the
  optimizer step's return).
- **run, random steps**: before four publications chosen at random, the engine's served
  weights are copied (about 3.4 GB each,
  under the audit directory). The steps those weights started are recomputed in float32
  (the public suite recomputes the first two copies, the Judge all four): loss, top-256 set,
  teacher log-probabilities, and the gradient of the specified loss. That gradient must
  match the reduced gradient Megatron held just before its optimizer step (direction and
  magnitude on the first rows of the input embedding, and the total gradient norm Megatron
  reported). This check fails when more than half of the recomputed steps miss these limits
  (one step can miss them through bf16 noise alone; a different loss misses them on every
  step). The teacher log-probabilities of random rows of two more random steps are also
  recomputed from the frozen initial snapshot with `c`, so a teacher that drifts with the
  student fails. The copies' time is removed from `T` as above (at most four copies per
  engine).
- **run, same path**: with `--probe-output` (always in the Judge), the numerics-relevant
  Megatron model configuration of the run equals the probe's.
- **run, audit** (in the Judge, the timed window must also lie inside the run by the Judge's own
  clock, with at most 60 s of evaluator intervals removed from it, and `T` must be positive):
  every request to an SGLang engine is recorded inside the engine. The student
  engine may only receive generation requests with the contract's sampling parameters and
  no top-log-probability options. Each round has exactly `prompts_per_round` generations whose prompts are
  the assigned rows (no context), each generation is trained on exactly once at a step
  within the staleness limit, and there is one weight publication per step (a publication
  is a group of weight-update calls with no generation in between). Every step's `loss`
  must equal the value recomputed from its dumped tensors, and step 0 is checked against an
  independent float32 computation. The 60 s bound is for the whole run, not per student
  engine: each student engine adds its own copies and digests.
- **perf**: the public contract, the audited timed window `T`, and the final student's accuracy
  and answer-tag rate on the 64 public questions with the Judge's protocol (three seeded sampled
  repeats; like the Judge, it loads `final_hf`'s safetensors with the pinned config and tokenizer),
  next to the original model's accuracy on those questions (`published_reference.json`). It is for
  getting the pipeline working and reports no score: the run has too few steps to show learning, 64
  questions move by several points between runs of the same model, and `T` does not predict the
  scored run's.

The Judge also reports, without scoring them: your student's answer-tag rate and its accuracy
on tagged answers, the original model **with** `c` (the teacher itself), and a diagnostic under a
different evaluation protocol (top-p 0.95, top-k 20, one sample).

Reports go to `OUTPUT/report.json`; exit code 0 means every check of the suite passed.
