# Task: optimize deterministic masked-token generation

You are given a real BF16 language-model checkpoint and a correct but
deliberately slow generation engine. Improve generation throughput while
preserving answer quality, determinism, request isolation, and the required
interface.

The implementation strategy is entirely your choice. Inspect the supplied
starter, runtime, model interface, and visible tests; form hypotheses, measure
them, and keep the strongest locally validated implementation.

## Objective

Reduce the wall-clock latency of `engine.generate()` while continuing to meet
the interface, determinism, isolation, and visible quality requirements.
Returning less content or weakening correctness is not a valid optimization.
Checkpoint loading and prompt construction happen before the candidate method
is called.

The supplied local tests are the only detailed, example-level development
feedback. Do not specialize to their prompts, answers, token sequences, or
example shapes.

The independent final uses hidden requests that differ from the visible ones
and may use any spec `runtime.api` accepts.
Interactive submissions execute the identical complete frozen request
cohort, score rows, quality/memory gates and final deadlines.
Complete Judge evaluation can take substantially longer than public tests.
The default visible GPU suite uses
40 public targets (10 per dataset);
its quick mode runs one target per dataset.
The visible suite runs its own, shorter generation lengths than the scored
evaluation. Each reported row shows the generation length it ran; a length
inside a visible case name is only a label.

## Initialized workspace

The workspace begins with:

```text
solutions/decoder_v001/candidate.py
```

This starter is a correct deterministic implementation for debugging the
interface. It repeatedly evaluates the full model and is deliberately slow.
Keep it unchanged as a recovery point; create new version directories for
experiments.

The runtime loads the checkpoint once and calls:

```python
engine = build(model, tokenizer)
```

The same read-only model and tokenizer may be reused for multiple requests.
You may inspect all code under `runtime/` to understand the model interface.

## Required interface

The selected source is:

```text
solutions/decoder_vNNN/candidate.py
```

`FINAL_VERSION` contains the selected version as one decimal integer from 1
through 999. Every retained version directory must contain a complete,
importable candidate.

The selected module must export:

```python
def build(model, tokenizer):
    return engine
```

The returned object must provide:

```python
@torch.inference_mode()
def generate(self, prompt_ids, spec):
    return runtime.api.GenerationResult(token_ids=output, trace={})
```

### Input

`prompt_ids` is a contiguous `torch.int64` tensor shaped `[1, prompt_length]`
on the device supplied by the runtime.

`spec` is a validated `runtime.api.GenerationSpec` and provides the requested
generation length, block length, temperature, and mask-token ID. Handle every
valid specification accepted by `runtime.api`; do not hard-code visible-test
values.

### Output

`GenerationResult.token_ids` must:

- remain on the prompt device with dtype `torch.int64`;
- have shape `[1, prompt_length + spec.gen_length]`;
- preserve the prompt prefix bit-for-bit;
- fill every requested generation slot;
- contain only valid vocabulary IDs;
- contain no remaining `spec.mask_id` tokens.

Complete the full requested buffer even if an EOS token appears. Every
generated slot must hold a token selected from the supplied model's
predictions for that request: do not pad, overwrite or remap slots, and do not
suppress or replace EOS predictions to change the measured output length.
Useful output length is checked against a reference decoder in both
directions. Repeated calls with identical inputs must produce identical
output. Mutable state from one request must not affect another request,
including key/value, hidden-state or logit caches of a prompt prefix shared
with other requests (such as few-shot examples); rebuild such caches within
each request. Every request must actually use the supplied model.

`trace` is optional development metadata.

## Runtime environment

```text
checkpoint              GSAI-ML/LLaDA-8B-Instruct
checkpoint revision     6059b30c9531767ce47e0ca6a8161a6bb0c67693
dtype                    BF16
parameters               8,015,581,184
layers                   32
hidden size              4096
attention heads          32
vocabulary size          126,464
maximum total sequence   4096
batch size               1
device                   one CUDA GPU, compute capability >= 8.0,
                         >= 34 GiB device memory
```

The checkpoint is mounted read-only. Do not train, quantize, replace, reload,
or mutate model weights. Do not load another model.

Exact token equality with the starter is not required. A different generation
path is acceptable when it satisfies the interface and preserves semantic
quality.

## Visible tests

Run the source-policy and CPU interface/state checks:

```bash
bash /workspace/public_tests/run_public_tests.sh --cpu-only
```

The sanitized allocation contract is available at
`/workspace/public_tests/memory_contract.json`. Its limits are frozen
(qualified on an NVIDIA H100 NVL): compute capability 8.0 or newer, at least
34,816 MiB of device memory, and peak allocated memory below
`floor(total_device_memory_mib) - memory_reserve_mib` with a 7,168 MiB
reserve, evaluated on the device the Judge runs on.

Quality is checked against a reference decoder.

Run a short real-checkpoint smoke test:

```bash
bash /workspace/public_tests/run_public_tests.sh --quick
```

Run the complete visible development suite:

```bash
bash /workspace/public_tests/run_public_tests.sh
```

Test a retained version without changing `FINAL_VERSION`:

```bash
bash /workspace/public_tests/run_public_tests.sh --quick --candidate-version 2
```

Visible tests can report interface status, semantic results on visible
examples, elapsed candidate time, observed model calls, useful output length,
visible-example throughput, and peak allocated memory. These are local
development measurements, not a complete performance characterization. The
CPU test uses a small synthetic model and cannot establish real-model quality
or performance.

During a managed run, the session wrapper may also provide a run-scoped
`sforge-submit` command. It submits the captured paths and waits for a redacted
aggregate validity/score result subject to the stated cooldown; it does not
return inputs, outputs, or per-example details. Validate locally before using
it, keep working files in a runnable state while it waits, and treat small
score differences cautiously. Manual aggregate submissions are uncapped, with
a host-frozen minimum interval between successful submission starts.

## Submitting on the single-accelerator profile

By default Work and the Judge run on separate GPUs and Work keeps running
while the Judge evaluates. In the shared-GPU (single-accelerator) profile,
Work and Judge share the one CUDA GPU allocated to this run, and
`sforge-submit` automatically invokes the
root-owned `.sforge/prepare_submit.sh` hook immediately before archive/HTTP
admission and refuses submission if that hook fails. Wait for every
public-test and compiler process to finish first. You may run the hook
manually to diagnose remaining GPU holders. Its success is not GPU-idle proof:
the host handoff coordinator still pauses this exact Work container and
attests the scheduler allocation before the Judge starts, and resumes it
afterwards.

The managed run performs no automatic aggregate evaluation while
you work: only your manual `sforge-submit --details` submissions run the
Judge, and the captured final state is evaluated after the work session ends
without sending feedback to you.

Your final result is the best valid score you reach within the time budget, so keep `FINAL_VERSION` on your strongest candidate.

## Versioning workflow

1. Keep `decoder_v001` unchanged.
2. Copy it to a new complete version directory.
3. Make one coherent experimental change.
4. Record the change and visible result in `notes.md`.
5. Run CPU tests after semantic changes.
6. Run real-checkpoint tests for promising complete implementations.
7. Preserve a known valid version before risky experiments.
8. Set `FINAL_VERSION` to the strongest locally validated version before the
   work session ends.

Only these paths are captured:

```text
solutions/
FINAL_VERSION
notes.md
```

Changes to the runtime, visible tests, instructions, mounted checkpoint, or
installed packages are not part of the submission.

## Source policy

Submission files must be small, regular, auditable Python or Triton source.
Only `candidate.py` and `__init__.py` are allowed under complete
`solutions/decoder_vNNN/` directories. Do not use symlinks, generated binaries,
encoded source, external data files, or dynamically loaded code.
The complete retained source tree must not exceed 2 MiB, and `notes.md` must
not exceed 128 KiB.

Allowed import roots are `__future__`, `collections`, `contextlib`,
`dataclasses`, `functools`, `itertools`, `math`, `runtime`, `torch`,
`triton`, and `typing`. Only `runtime.api` may be imported from `runtime`.

The starter's following fixed three-argument feature probe is allowed:

```python
getattr(obj, "mlsysforge_allow_cpu", default)
```

Other reflection, dynamic attribute names, aliases of `getattr`, dynamic
imports, file access, subprocesses, frame or hook access, and dunder
introspection are not allowed. Do not replace `model.forward`, audit hooks, or
trusted runtime objects. Do not conceal source or attempt to identify
evaluation inputs.

## General policy

- Candidate code uses only the supplied local resources.
- Do not access protected runtime files, credentials, host files, or other runs.
- Do not hard-code visible prompts, answers, token sequences, or example shapes.
- Do not execute model-generated code from candidate code.
- Keep request-specific mutable state local to the request or reset it fully.

Use the visible tests for detailed debugging and any run-scoped aggregate
validation only as a coarse signal. Leave the selected implementation complete
and runnable when the work session ends.
