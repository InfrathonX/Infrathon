# SD3.5-M on-policy distillation task: executable interface

## Command

```text
python task_entry.py run --contract CONTRACT.json --out NEW_DIR
```

The working directory is the submission root (`/app/workdir/miles_lab`). `--out` does not exist beforehand. Exit 0
only after writing every output below. The evaluator runs it as the unprivileged `agent` user with the two GPUs
visible as CUDA 0 and 1 and no network, twice per submission: once with `generate_url` set (probe round + timed
replay) and once without (the real run under the in-engine audit). The first launch is stopped after
1500 s and the second after 2700 s; a launch that runs out of time, or exits nonzero, scores 0.

### Contract

| field | meaning |
| --- | --- |
| `model` | SD3.5-M diffusers directory (student base; read-only) |
| `teachers` | `{label: PATH}`, PATH a PEFT LoRA adapter dir on top of `model`; labels are arbitrary per evaluation |
| `teacher_key` | row metadata field that selects the teacher (`data_source`) |
| `rows` | JSONL `{input, metadata: {data_source, row_id, seed}}`; round `r` uses lines `[r*n, (r+1)*n)`, `n` = `rows_per_round` |
| `student_init` | PEFT-named LoRA safetensors (`base_model.model.<module>.lora_{A,B}.weight`); load exactly this |
| `rounds`, `rows_per_round`, `steps_per_round`, `samples_per_row` | the size of the run; rounds and rows per round differ between the public and the scored run |
| `sampling` | 10 steps, 512x512, guidance 1.0, `cps`, eta 0.8, 3-step window, start in `[0, 3)` from the row seed |
| `optimizer`, `lora` | AdamW and LoRA settings (see AGENT_TASK) |
| `train_precision` | allowed training precisions: fp16, fp32 |
| `generate_url` | `null`, or the base URL that must receive every `POST /rollout/generate` (replay / probe) |
| `probe_steps` | the first N optimizer steps must also record transition means and the post-step LoRA |
| `scratch` | a writable directory |

The Judge passes its own contract, with hidden data and larger sizes. The public copy is `/opt/opd_public/judge/contract.json`; the suites run it with their own data and sizes. Implementations must not assume any fixed size.

### Replay protocol (`generate_url` set)

- Send each round's requests to `{generate_url}/rollout/generate` with exactly the payload a real engine would get
  (same JSON body you would send to your engines). The response body is the engine's msgpack response.
- After the round's publication is applied to **every** local engine: `POST {generate_url}/replay/published?round=r`.
  The next round is released only after that call. Requests for an unreleased round wait.
- Untimed setup — the replay server's start to the release of the first timed round (start-up, loading, compilation,
  the probe round) — is capped at 600 s; the timed segments start after it.

### Outputs

- `OUT/result.json`: `{"status": "completed"}`
- `OUT/final_lora.safetensors`: the final student LoRA, PEFT names, fp32: bit for bit the last step's `lora_post`
  (same keys and values)
- `OUT/steps/step_{k:05d}.pt`: one `torch.save` dict per optimizer step `k` (all ranks together):

| key | content |
| --- | --- |
| `pairs` | list of `(row_id, output_idx, request_id, step_index, route_label, kl)` for every pair of the step |
| `lora_pre` | `{peft_key: fp32 tensor}` student LoRA before the update |
| `exp_avg`, `exp_avg_sq` | `{peft_key: fp32}` AdamW moments after the update; `adam_step` its step count |
| `grad_norm_preclip` | total gradient norm before clipping |
| `lr`, `betas`, `eps`, `weight_decay`, `clip_grad` | the optimizer settings used: the contract's `optimizer` values (checked, rel 1e-6) |
| `means` | first `probe_steps` steps only: per pair `(mu_student, mu_teacher)`, fp32 `[16, 64, 64]` |
| `lora_post` | first `probe_steps` steps and the run's last step: the student after the update, the same keys and shapes as `lora_pre` |

Every record is read with `torch.load(weights_only=True)`: plain tensors, numbers, strings, lists, tuples and dicts
only (no numpy objects, no custom classes). A record that does not load that way fails the check that reads it.

## What the checks look at

- **static** (before any launch): the submitted tree has `task_entry.py` and `repo/train_diffusion.py`, no symlinks,
  and no weight-like files (`.safetensors`, `.pt`, `.bin`, `.ckpt`, `.msgpack`) anywhere. Python files that differ
  from the pristine miles_diffusion tree may not add uses of `--rollout-patch-group` / `rollout_patch_group` or new
  assignments (or `setattr`) into sglang / `multimodal_gen` objects.

- **framework** (every session): a process ran `train_diffusion.py`; every generation request (engine or replay
  server) falls inside a `miles.rollout.base_types.call_rollout_fn` training-rollout call; `FSDPTrainRayActor.train`
  ran for every round; every engine weight update falls inside `FSDPTrainRayActor.update_weights` (recorded inside the
  processes, same host clock). A trainer outside miles_diffusion's framework fails this.

- **probe** (hidden rows, a nonzero hidden student init, random labels): the pair set and routes of step 0; the
  logged KL against the dumped means in float64; the dumped means against an independent float32 oracle (errors
  normalised by the teacher's distance from the base); the step's gradient, recovered from the first AdamW moment,
  against the oracle gradient (direction and size); the update moved every step.
- **replay**: payloads byte-identical to the contract, each row once, one `published` per round after all its rows;
  untimed setup within 600 s;
  AdamW algebra of every step with the contract's optimizer settings; from inside the engines: every engine holds
  exactly the dumped student after each round (fp32, or its fp16 rounding) and each `published` call comes only after
  every engine applied that round's student; timing as in AGENT_TASK.
- **run**: every engine request and publication recorded inside the engines: rows once each, sampling parameters,
  window, accelerator switches off, lag 0 on every engine, one publication per round with identical weights on all
  engines, only student tensors published and equal to the dumped student after each round (fp32 or its fp16
  rounding); AdamW algebra of every step from `student_init` with the contract's optimizer settings; pair sets per
  step; `final_lora.safetensors` equal to the last step's `lora_post` bit for bit, and that `lora_post` holding the
  keys and shapes of the step's `lora_pre`.
- **KL gate**: the evaluator's own sampler measures the held-out teacher-student KL of `final_lora.safetensors`
  per route (one case seed per GPU) against the original model's frozen KL for the same seeds; each route's
  `ln(KL_base / KL_final)`, averaged over the case seeds, must reach its frozen minimum (see AGENT_TASK: Score).
- **task metric**: the evaluator generates one image per held-out prompt with `final_lora.safetensors` merged into
  SD3.5-M (512x512, 40-step ODE, the frozen guidance scale, a fixed seed per prompt; prompts dealt over both GPUs),
  scores OCR accuracy on the OCR route and PickScore on the PickScore route, and computes each route's closure of
  the original-to-teacher gap against the frozen original-model and teacher values (see AGENT_TASK: Score).
