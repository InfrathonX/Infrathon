# Optimize one K-Means iteration on the allocated CUDA accelerator

Work in `/app/workdir/flash_kmeans_lab`.

The workspace intentionally starts with no implementation
or starter code. Create version `v001` yourself and optimize both operations
below for the accelerator allocated to this run:

```python
assign(x, centroids, x_sq) -> labels
update(x, labels, old_centroids) -> (counts, new_centroids)
```

You may use PyTorch, Triton, or hand-written Triton kernels.

## Required behavior

- Inputs are contiguous CUDA tensors.
- `x` is FP16 `[B,N,D]`.
- `centroids` and `old_centroids` are FP16 `[B,K,D]`.
- `x_sq` is FP32 `[B,N]` and equals
  `x.float().square().sum(-1)`.
- `assign` returns CUDA integer labels `[B,N]` for the nearest centroid under
  squared Euclidean distance.
- Exact ties select the smallest cluster id.
- `update` returns exact CUDA integer counts `[B,K]` and FP16 centroids
  `[B,K,D]`, accumulated in FP32.
- An empty cluster retains its old centroid exactly.
- Neither function may mutate an input.
- Do not materialize the complete `[B,N,K]` distance tensor.

Correctness, input integrity, output contracts, completion, and bounded extra
GPU memory are hard gates.

## Files to create

Create a complete first implementation at:

```text
kernels/flash_kmeans_v001.py
```

It must export both `assign` and `update`. Write `1` followed by a newline to
`FINAL_VERSION`.

Put later complete experiments in:

```text
kernels/flash_kmeans_vNNN.py
```

Keep `FINAL_VERSION` on the best implementation that has passed Judge. Every
retained version is source-checked, so remove incomplete or abandoned kernel
files.

## Optimization loop

The Work image contains public tests, but no K-Means reference implementation,
starter, baseline, oracle, or formal performance cases.
Public correctness uses literal golden inputs/outputs plus sampled
expected values without shipping a reusable K-Means implementation, and its
timing rows execute only your candidate.

The public deployment-like probes run in isolated fresh Python processes with
an empty per-case compilation cache. Their first call includes module import,
the first synchronized candidate call, and any compilation. First-call time is
diagnostic; warmed calls use fresh input pointers and seeds. Public sampled
checks cover both the first and warmed calls, including assignments, counts,
selected centroids, input integrity, output contracts, and bounded memory.

The full public run includes a control workload plus small/large scale ladders
for `large-K` and `large-N/high-D`. It reports raw candidate-only effective
assignment TFLOP/s plus relative scale-ladder throughput retention. These are
development diagnostics. Use the
ladders to detect superlinear growth, excessive kernel launches, and tiling or
streaming choices that stop scaling.

Run a quick public check with:

```bash
bash /workspace/public_tests/run_public_tests.sh --quick
```

Run its larger diagnostic probes with:

```bash
bash /workspace/public_tests/run_public_tests.sh
```

You may test an unselected complete version with `--candidate-version N`.
Public timings are absolute candidate-only diagnostics and do not produce a
score. Scaling ratios and quality buckets are diagnostics under the same
candidate-only rule. Hidden shapes differ from the public ones. Do not
hard-code them. Passing public tests does not imply passing Judge.

For a complete candidate, run:

```bash
sforge-submit --details
```

Your final result is the best valid score you reach within the time budget, so keep `FINAL_VERSION` on your strongest candidate.

Wait for every public test and compiler process to finish before you submit.
If Work and Judge share one GPU, `sforge-submit` automatically invokes a
cleanup hook first.

Judge evaluates additional shapes and returns structured stage-level
correctness, input-integrity, memory, fresh-process, and performance feedback.
For an authoritative invalid submission, deployment feedback is limited to a
coarse `passed`, `failed`, or `not_reached` status for each of the `large-K`
and `large-N/high-D` families. It does not return family latency.
After a fully valid evaluation, each family may additionally receive a coarse
warmed-latency band and a categorical trend against the best earlier valid
band from a manual submission in this run. A first valid result need not have a
trend. Infrastructure-invalid evaluations omit family feedback entirely.

Feedback never discloses exact formal shapes, per-case or exact family
latencies, per-case weights, seeds, case names, family membership, or member
counts.
Manual submissions are count-unlimited with a 300-second cooldown. Use the
feedback to repair or optimize a new version, select it in `FINAL_VERSION`,
and submit again. Do not submit incomplete or knowingly broken sources.

If Judge invalidates a version, restore `FINAL_VERSION` to the best previously
Judge-passing version. If none has passed, select your simplest locally correct
implementation and repair it in a new version.

Only your manual submissions run the Judge while you work. The workspace is
evaluated once more when the time budget ends, so keep the best known valid
version selected.

After all hard gates pass, the Judge score is a family-first weighted geometric
mean of candidate latency in milliseconds. Lower is better:

```text
score = assign_GM^0.10
      * update_GM^0.10
      * full_GM^0.20
      * control_fresh_process^0.10
      * large_K_fresh_process_GM^0.25
      * large_N_high_D_fresh_process_GM^0.25
```

Rows are first combined within their private group, then the six public group
weights above are applied. Exact group membership and per-case weights remain
hidden. The first-run latency is diagnostic; warmed fresh-process latency is
scored. No baseline or oracle is timed as part of the score.

Optimize shape-generically across all stages. In particular, reason explicitly
about how work, memory traffic, temporary storage, launch count, and compilation
behavior scale when `N`, `K`, or `D` becomes very large, including combinations
of large `N` with high `D`. A design that is fast only for small or medium
parameters is not sufficient.

## Source policy

This is a clean-room task. Do not copy, vendor or install another K-Means
implementation.

Candidate code may use the PyTorch and Triton already installed, but may not
call another K-Means implementation. Do not forge results, modify evaluation
state, spawn subprocesses, dynamically load code, reset CUDA measurement
state, or perform filesystem/network I/O from candidate code.

Only `kernels/`, `FINAL_VERSION`, and `notes.md` are submitted. Record useful
measurements and failed ideas in `notes.md`.
