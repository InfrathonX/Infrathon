# Task: optimize ORPO post-training

## Goal

Improve the end-to-end training throughput of the supplied ORPO implementation
for full-parameter BF16 Qwen3-1.7B training on two CUDA GPUs (compute
capability 8.0 or newer, at least 60 GiB of device memory each, NCCL).

Your result must remain behaviorally equivalent to the supplied baseline. Loss
results, auxiliary results, autograd behavior, model updates, and distributed
execution must remain correct. The evaluator also enforces a GPU-memory limit.

This is full-parameter training, not LoRA.

## Starting point

The initialized solution is:

```text
solutions/objective_v001/candidate.py
```

It selects the pinned official Hugging Face TRL baseline described in
`UPSTREAM_HF_TRL.md`. Keep `objective_v001` unchanged so it remains a recovery
point. Put each retained experiment in a new, complete version directory.

## Candidate interface

The selected file is `solutions/objective_vNNN/candidate.py`, where `NNN` is the
decimal value in `FINAL_VERSION`.

It must export:

```python
def build_objective(contract):
    ...
```

The returned module must obey the types and callable contract in `runtime.api`.
Use that module, the initialized solution, and the public tests as the
authoritative executable specification. Do not copy or replace trusted runtime
types.

A candidate may additionally export:

```python
def build_model_patch(contract):
    ...
```

If used, this function must return the declarative `Qwen3ModelPatch` defined in
`runtime.api`. The runtime owns patch installation and model structure. Do not
export `optimize_model` or mutate model/module attributes from candidate code.

The optional interface is not required: an objective-only solution is valid.

## Correctness requirements

- Preserve the complete observable behavior defined by `runtime.api` and the
  upstream baseline, including all returned values and required gradients.
- Use values supplied through the contract and inputs; do not specialize to a
  public shape, token pattern, mask, or expected output.
- Perform the complete requested work. Approximate results, skipped terms,
  fabricated gradients, replayed outputs, or shortened training are invalid.
- Compute in BF16 or FP32 (TF32 accumulation and FP32 upcasts are fine).
  FP8, INT8, FP4 or other quantized formats for any matmul, weight or
  activation of the objective or of a model patch are approximate results,
  even when they stay within tolerance.
- Keep candidate state deterministic and isolated between calls.

Operator-for-operator or bitwise identity with the eager implementation is not
required when the observable numerical contract is preserved.

## Public validation

Fast CPU checks:

```bash
bash /workspace/public_tests/run_public_tests.sh --cpu-only
```

Short two-GPU smoke test:

```bash
bash /workspace/public_tests/run_public_tests.sh --quick
```

Complete visible development run:

```bash
bash /workspace/public_tests/run_public_tests.sh
```

To test a retained version without changing `FINAL_VERSION`:

```bash
bash /workspace/public_tests/run_public_tests.sh --quick --candidate-version 2
```

These commands use public development workloads. Their output is diagnostic;
do not assume that private evaluation uses the same inputs or configuration.

## Evaluation

Private evaluation first checks behavioral correctness and resource compliance
on unseen workloads, then measures complete training-step throughput relative
to the official reference. Managed submissions return only aggregate feedback.

Implement the general contract rather than attempting to infer private inputs
or evaluator behavior.

## Submission workflow

1. Keep `objective_v001` unchanged.
2. Create a complete `solutions/objective_vNNN/` for each retained experiment.
3. Run public correctness checks before GPU measurements.
4. Record the hypothesis and observed public results in `notes.md`.
5. Set `FINAL_VERSION` to the strongest validated version.

Only these paths are captured:

```text
solutions/
FINAL_VERSION
notes.md
```

Edits elsewhere in the workspace are not part of the submission.

Your final result is the best valid score you reach within the time budget, so keep `FINAL_VERSION` on your strongest candidate.

## Source and security policy

Each version directory may contain only `candidate.py` and `__init__.py` as
small, regular, auditable source files. Use only the imports accepted by the
public source-admission check. Do not use files, network access, subprocesses,
dynamic imports, encoded payloads, frame/stack inspection, runtime
monkey-patching, or evaluator/controller state.

Candidates use only the supplied model, data and runtime. Leave every retained
version complete and importable.
