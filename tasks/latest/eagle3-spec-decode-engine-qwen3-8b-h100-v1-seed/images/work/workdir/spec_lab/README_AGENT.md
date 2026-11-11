# EAGLE-3 engine workspace

Read `AGENT_TASK.md` and `runtime/api.py` before editing. This
directory is the task cwd. Only `solutions/`, `FINAL_VERSION`, and `notes.md`
are submitted; `runtime/` and `public_prompts/` are trusted read-only inputs,
and native build products belong under `build/`.

## Version discipline

The seed is `solutions/engine_v001`. For a new hypothesis, copy the current
best to the next unused monotonically numbered `engine_vNNN`, change that new
directory only, and update `FINAL_VERSION` after it is runnable. Never rewrite
a version that has produced a public or private measurement. Before its first
score-producing run:

```bash
sforge-save-version --label descriptive-hypothesis
```

Record the hypothesis, implementation changes, public command, per-cell rates,
oracle summary, memory, and decision in `notes.md`. Keep negative results; they
prevent later repetition and make the version curve interpretable.

## Public checks

```bash
/workspace/public_tests/run_public_tests.sh --cpu-only
/workspace/public_tests/run_public_tests.sh --quick --candidate-version 1
/workspace/public_tests/run_public_tests.sh --full --candidate-version 1
```

CPU-only checks source layout, import contract, prompt/return validation, and
admission. Quick runs the first batch of each of the six public cells; full runs
all 38 public prompts. The public cells (`chat_bs2`, `math_bs4`, `code_bs3`,
`longctx_bs2`, `think_bs2`, `mixed_bs6`) deliberately differ from the Judge cells
in batch size, budget and prompt source. Both GPU modes build the chosen engine, warm each cell shape,
synchronize around timed generation, teacher-force outputs, rerun determinism
checks, and report the same score formula as Judge.

Public policy is calibrated, not provisional: eager BF16 reference, hard margin
24, global soft rate 0.03284789994157488, and eight disclosed CDF tiers.

## What not to infer

`public_prompts/reference_outputs.json` contains seed streams for evidence.
Exact stream match is not a gate because legitimate BF16 reduction-order
changes can choose another near-tie token and continue on a different valid
trajectory. Optimize throughput under teacher-forced own-trajectory margins;
do not target stored tokens.

An improved mean acceptance length does not imply improved speed. A larger tree
can accept more tokens while spending more target time. Compare total generated
tokens divided by synchronized wall time, per cell and geometrically aggregated.
Changes near the documented noise floor should be interleaved against the prior
best.

## Seed orientation

The seed uses target layer-input features 2/18/33 and a 60-token, depth-7,
top-k-10 tree.

Preserve prompt prefixes, EOS inclusion, exact no-EOS budgets, vocabulary
range, int64 output, request independence, and deterministic reruns. Ensure all
CUDA streams and background threads are complete before returning.

## Native code and source policy

Python, CUDA/C++, headers, JSON, Markdown, text, and TOML are admitted within
the documented size limits. Torch/Triton and compilation from shipped source
are allowed. Network/process libraries, pickle loading, dynamic eval/import,
frame inspection, environment-based run routing, trusted-runtime mutation,
outside-root file access, symlinks, binaries, and oversized files fail source
admission before `build`.

The target and head mounts are immutable inputs. You may construct optimized
copies inside `build`, including draft-side lower precision. Target-side
numerical changes remain subject to the complete margin CDF, not just the hard
cap.

## Private evaluation

`sforge-submit` evaluates the complete frozen final prompt set. Automatic scheduled
final evaluation is disabled. After Work closes, the frozen archive is scored
against that same complete frozen final prompt set. Host-side replay may request an older saved
`engine_vNNN`; interactive rounds always follow `FINAL_VERSION`.

By default Work and the Judge run on separate GPUs. When a run shares one GPU
between them (the single-accelerator profile), wait for all public-test and
compiler workers to exit before each `sforge-submit --details`: the root-owned
`.sforge/prepare_submit.sh` drain hook runs before admission and refuses the
submission if any same-UID process still holds the device, and the host pauses
this Work container while the Judge runs.
