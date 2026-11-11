# MInference H100 v3: single-GPU kernel microbenchmark

Scope: optimize a complete single-GPU kernel call within an explicit attention-vector error budget. The contestant submits only a five-argument native-GQA BF16 attention function; no full-model answer-quality gate is run. Submit and final use the same private suite.

| Item | Contract |
| --- | --- |
| ABI | `minference_sparse_prefill_bf16(q, k, v, softmax_scale, out) -> None`; Q heads 32 / KV heads 8, D=128, N>=64 |
| Kernel inputs | 4 text categories x 32K/64K/128K x layers 1/4/15/27 = 48 groups; each group uses three different real prompts, 144 calls in total |
| Warm-up | Real layer 2 at the same shape; reusing its pattern or output is forbidden |
| Numeric gate | uniform-512 and tail-64 query samples are checked separately: NRMSE over all samples <= 0.25 and worst-head NRMSE <= 0.50; output must also be fully written, finite, and inputs unmodified |
| Reference | Sampled-query FP32 dense causal native-GQA; 11 small-N boundary cases (N from 64 to 8193) additionally get a full-output check and are not timed; dense FA3 serves as the speed/noise control |
| Score | Per trial `8192*N*(N+1)/max(host wall, CUDA event)` with the real N; median of three trials; equal-weight geometric mean over the 48 groups |
| Timed region | Pattern estimation, conversions, attention, dense repair, writing/snapshotting `out`, and synchronization; model/QKV preparation and reference computation are excluded |
| Resources | Worker and Judge each get 8 vCPU, 48 GiB host memory, 1 H100; one GPU may be reused serially for author validation |
| Persistent inputs | One model plus two kernel text sets (public and private); QKV is materialized one prompt at a time, <= 8 GiB, and released after use |
| Full-model QA | Not part of any Judge run and not a release requirement |

`NRMSE = sqrt(sum((candidate - FP32_dense)^2) / max(sum(FP32_dense^2), 1e-12))`, computed per head; the worst head is the maximum. The absolute budget is uniform and frozen after public calibration and before private acceptance; it is not a multiple of the seed's error. The 25% / 50% figures are an explicit engineering budget for approximate kernels, not an upstream official threshold, not a rounding tolerance for exact attention, and not a guarantee of full-model semantic quality. The budget must not be loosened based on private results.

The 12 public stratified prompts cover all 48 task/length/layer points. On each call the starter estimates per-head error from 128 distributed queries plus 64 tail queries in segments; if any 16-row segment exceeds the 0.10 guard, that head is repaired with a self-contained Triton dense kernel. The guard and the repair count toward measured time, and no input-dependent result is cached. The 192-row guard is required: a smaller (64-row) guard can pass public calibration while exceeding the budget on private inputs. A dense x0.6 control exceeds the overall budget, so the gate rejects trivially degraded outputs.

The reference derives from Microsoft MInference at the fixed commit `a4eb395f949ea39e871f9bc586d683390692c6be`: its last-64 vertical/slash estimator and sparse kernel. The original budgets 512/2048 are scaled by 1.5 to 768/3072 and adapted for native GQA and N that is not a multiple of the block size. The error budget, the 48-group input matrix and the speed score are this task's own design. That commit's `tests/test_e2e.py` is a full-model forward smoke test (100000-token input), not a numeric kernel benchmark; the upstream repository does not provide this 48-group standard. Original sources, URLs and SHAs are listed in `references/upstream/manifest.json`.

The context protocol is `minference-kernel-prefill-v3`. Results from a multi-GPU full-model protocol cannot be relabeled as single-GPU passes.

Integrity is checked by static admission, an LLM review of the submitted source against `llm_judge/POLICY.md`, and a bounded set of GPU controls, which together check for cached replay, input modification, missing writes / non-finite output, and access to the hidden reference or its device memory. The bounded controls and source review do not prove isolation against arbitrary malicious GPU code. The Judge mounts only two read-only trees (private texts and the model); the 25 protected files are all source code or small metadata, with no QA projection. Proof of the real single-GPU image/device/mount is a bootstrap requirement; if the platform backend cannot provide that proof, deployment stays blocked, and it must not be faked through request parameters. If a deployment allocates one GPU each to Worker and Judge, author validation may use one GPU serially.

Code submission, partial builds, author calibration and full-entry qualification are recorded separately; CPU-only success or a full-model QA result does not count as a pass under this protocol.

Rebuild using `scripts/build_images.sh` with the model/text contracts and the frozen kernel numeric policy only. Generate the 25 protected hashes and refresh source/task digests (`scripts/refresh_portable_manifests.py`). No QUALITY_MANIFEST / QUALITY_POLICY or full-model QA worker is a build prerequisite.

A quick public test is `tools/run_public_tests.sh --candidate-version 1 --boundaries-only`: it checks only the 11 small-N cases, does not load the full model for real QKV preparation, and produces no formal score. The default public test runs all 48 groups. The conservative starter is not demonstrated to be faster than dense FA3 on the same QKV; it is only a valid starting point, and any optimization must still satisfy the fixed numeric gate. A Judge run includes real model/QKV preparation and an independent reference check; only the kernel calls are scored.
