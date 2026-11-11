# MInference sparse prefill kernel optimization

Work only in `candidates/kernel_vNNN/submission.py`, `FINAL_VERSION` and `notes.md`. Preserve:

```python
def minference_sparse_prefill_bf16(q, k, v, softmax_scale, out) -> None:
    ...
```

The inputs are BF16 causal native-GQA tensors with 32 query heads, 8 KV heads and head dimension 128. `N` is any integer with `64 <= N <= 131072`, and it need not be a multiple of 64. Do not modify the inputs. Write every element of `out` on the caller's CUDA stream.

## Correctness

The kernel may approximate attention. Each call is compared with an FP32 exact causal native-GQA reference on sampled query rows, separately for 512 uniformly sampled rows and for the last 64 rows. For both samples:

```text
NRMSE = sqrt(sum((out - ref)^2) / max(sum(ref^2), 1e-12))
global NRMSE <= 0.25
worst-head NRMSE <= 0.50
```

Small lengths are checked on the full output. Every output element must be finite. Any failure makes the submission invalid, with score 0.

## Public tests

These tests exist to get your kernel working and to check correctness.

| task | context budgets | timed layers | warm-up layer |
| --- | ---: | ---: | ---: |
| niah_multikey_2 | 8K, 16K, 24K | 8, 20 | 5 |
| cwe | 8K, 16K, 24K | 8, 20 | 5 |
| fwe | 8K, 16K, 24K | 8, 20 | 5 |

Inputs are real Q/K/V activations captured from Llama-3-8B-Instruct-262k on one RULER-style prompt per task and budget. Each case has one untimed warm-up and three timed calls. Boundary lengths use random inputs.

```bash
tools/run_public_tests.sh --candidate-version N                    # boundary lengths and all 18 cases
tools/run_public_tests.sh --quick --candidate-version N            # boundary lengths and the three 8K prompts on one layer
tools/run_public_tests.sh --boundaries-only --candidate-version N  # boundary lengths only; no model load
tools/check_source.sh --candidate-version N                        # CPU source admission
```

## Hidden tests

Your score comes from hidden test cases, not from these public tests. The hidden cases use a different model, prompts, lengths and layers, and can be much larger, but they always stay inside the contract above. Like the public tests, they use real Q/K/V activations, plus random inputs at small lengths. Make your kernel meet the error budget and run fast across that whole contract; do not tune it to the public model or prompts.

## Measurement and metric

For one call:

```text
tflops = 2 * 32 * 128 * N * (N + 1) / max(host wall time, CUDA event time) / 1e12
```

The timed region covers pattern estimation, index conversion, attention, any dense repair, the output write and synchronization. An untimed warm-up absorbs compilation. The Judge takes the median of the trials in each group, then the geometric mean over all groups.

## Submission evaluation policy

Every `sforge-submit` runs the evaluation on hidden test cases. `sforge-submit --details` returns the validity and the score. Hidden cases and per-case results stay on the Judge. Your final result is the best valid score you reach within the time budget, so keep `FINAL_VERSION` on your strongest candidate.

## Rules

- All input-dependent work belongs to each call. Only compiled code and shape-only empty scratch may persist across calls. Do not reuse outputs or patterns from a warm-up or an earlier call.
- A self-written Triton dense fallback and fixed input-independent sparse patterns are allowed if the correctness checks pass.
- Do not use existing vendor attention kernels, dynamic or native loading, file I/O, timing or stream manipulation, or `torch.compile`.
- Do not inspect the Judge context or private data, or hard-code case identities.
- Do not read or alter protected runtime, benchmark or controller files.

Before each submission, wait for your public tests to finish. A Judge run takes about half an hour of your budget.
