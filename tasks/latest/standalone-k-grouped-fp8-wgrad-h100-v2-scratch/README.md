# K-Grouped FP8 Weight-Gradient GEMM (from scratch)

An agent writes a K-grouped, per-channel-scaled FP8 weight-gradient GEMM in self-contained
Triton. This is the training-side GEMM that DeepGEMM provides for mixture-of-experts layers:
groups of token rows are concatenated along K, and each group accumulates
`D_g = C_g + A_g^T @ B_g` into an FP32 output. The kernel must be correct and memory-safe over
the declared domain, including arbitrary M and N and empty groups, and as fast as possible on
one rank of a mixture-of-experts training step across 12 public MoE model shapes. This is a
**scratch** task: the starting
`submission.py` is an empty file.

## The task

The agent works in `/app/workdir/k_grouped_wgrad_lab` inside the Work container. It writes
versions to `candidates/kernel_vNNN/submission.py` (three-digit directories), puts the selected
version number in `FINAL_VERSION`, and keeps notes in `notes.md`. Only these three paths are
submitted. The full brief is `images/work/AGENT_TASK.md`.

The selected `submission.py` must define this function. It launches its work on the current
CUDA stream, writes `out` in place, and returns `None`:

```python
def k_grouped_fp8_gemm_tn_contiguous(a, b, a_scale, b_scale, ks, c, expected_k_hint, out) -> None: ...
```

- **Math.** `a` is `[K_total, M]` and `b` is `[K_total, N]`, both FP8 E4M3FN. Each operand has
  one FP32 scale per (128-row K block, column) pair. `ks` holds the per-group K sizes. `c` is a
  read-only FP32 addend `[G, M, N]`, and `out` is FP32 `[G, M, N]`. Each 128-row block's partial
  product is scaled and accumulated in FP32 on top of `c`. A group with `K = 0` gets
  `out[g] = c[g]`.
- **Support domain.**
  - `1 <= G <= 128`.
  - `M` and `N` are any integers from 1 to 8192, not only multiples of 8 or 128.
  - Every group K is 0 or a positive multiple of 128. `K_total <= 262144`, and an empty problem
    is in scope.
  - `G * M * N <= 2^31 - 1`.
  - `expected_k_hint` is only a scheduling hint, and it can be stale or badly wrong.
- **Allowed.** `torch` for metadata, scratch allocation, small metadata computations (such as a
  `cumsum` of `ks`) and dispatch on the current stream. `triton` / `triton.language` for the
  kernels: `tl.dot`, atomics, split-K, persistent schedules, autotuning, and TMA descriptors.
  Aligned fast paths are allowed next to a general masked path.
- **Not allowed** (rejected by source admission, the Judge or the integrity review):
  - any existing GEMM: `torch.mm`/`matmul`/`bmm`/`_scaled_mm`/`einsum`, `@`, `torch.ops`,
    cuBLAS, CUTLASS/CuTe, DeepGEMM, or another compiled extension;
  - imports outside a small allowlist;
  - patching any module, class or object attribute, and `getattr`/`setattr`/`eval`/`exec`;
  - CUDA graphs, private streams and threads;
  - caching or replaying outputs across calls;
  - specializing for seeds or for benchmark presence;
  - using `expected_k_hint` to skip K rows;
  - writing outside `out` or modifying any input;
  - fetching or copying external implementations.

  The Work image contains PyTorch, Triton, the public contract and public test tools. It does
  not contain DeepGEMM, a GEMM kernel, a reference solution or Judge code.
- **Time budget.** The launcher's default is 12 hours (`--hours`/`--timeout`).

## Evaluation

Every Judge run performs the same complete evaluation. Any failed gate makes the submission
invalid.

- **Sanitizer pass, first.** A small subset of edge cases runs under
  `compute-sanitizer --tool memcheck` before any native call. On a platform where the sanitizer
  cannot attach, this pass is skipped and recorded as `sanitizer_ran = 0`.
- **Edge cases (correctness, no score weight).** 166 Judge-only cases cover the whole support
  domain:
  - tiny `M`/`N` and sizes that are not multiples of 8, 16 or 128;
  - `M` or `N` at 8191/8192, and 1 to 128 groups;
  - empty groups and `K_total = 0`, and a single group up to `K = 49152`;
  - deep K: one to four groups with `K_total` up to 262144;
  - unbalanced K and hints far from the real K.

  Each case is called twice. Forty-three public cases from the same generator ship in the Work
  image.
- **Correctness.** Each call is checked against `E`, the exact FP64 result from the dequantized
  operands, computed in 8192-row K chunks so the check's memory does not grow with K:
  - every element must be finite and written, because `out` is filled with NaN first;
  - every element is within `0.05 * R + 2^-20 * |c|` of `E`, where `R` is the magnitude scale
    of the element's products;
  - the per-group normalized error against `E` is below `2e-7`;
  - DeepGEMM's `calc_diff < 1e-3` against the BF16 source GEMM holds on groups (or pooled small
    groups) with at least 16,384 outputs.
- **Memory safety.** Each tensor sits between 1 MiB guard bands. Edge-case inputs are compared
  byte for byte after every call; scored-case inputs are compared by fingerprint.
- **Harness integrity.** The Judge captures its clocks, CUDA events and checking functions
  before it imports the candidate. It fails the run if any of them changes. It also rejects
  throughput above the device's dense FP8 peak.
- **Score.** 1082 Judge-only cases model one rank of a mixture-of-experts training step (the
  weight gradient of the expert linears) over 12 public MoE models, 24 model layers and 22
  `(M, N)` shapes. Each case is an enumerated combination of expert parallelism, tokens per rank
  and routing imbalance; group sizes are Dirichlet splits of a fixed `K_total`, redrawn every call
  (a calibrated synthetic load, not a routing capture). Four families:
  - **large** (504): many tokens per expert (K per group >= 1024), tokens per rank 4096 to 32768,
    late- and early-training imbalance;
  - **many_groups** (189): 16 to 128 local experts, many small or empty;
  - **small_batch** (307): K per group below 1024, tokens per rank 64 to 1024;
  - **sequence** (82): one 58-layer backward pass, alternating the two expert linears with fresh
    group sizes on every call and no synchronization between calls.

  96 of the single-call cases carry a stale or badly wrong hint (x0.05 to x8). The 123 public
  cases use four other models' shapes and different token counts, plus seven fixed-shape cases
  that are not tied to a model and draw uniform group sizes.
  - A single-call case has one warm-up and three measured trials. Before each measured call, the
    Judge makes one untimed call with the same shapes and hint but redrawn group sizes on
    different tensors. A sequence case has one untimed round and three timed rounds.
  - Time is the larger of host wall time and CUDA-event time.
  - `effective TFLOPS = 2 * sum(ks) * M * N / seconds / 1e12`, summed over the calls of a round.
  - Each case is its median trial or round. Within a family, cases are geometrically averaged per
    model and then across models; the score is the geometric mean of the four family scores, in
    effective TFLOPS. Higher is better. It is uncapped, with no baseline ratio and no threshold.
- **Interactive and final evaluation.** Scheduled auto-evaluation is off. The agent can submit
  with `sforge-submit`, at most once per 300 s cooldown. It gets back the gate states, the
  score, the four family scores and the number of failed edge cases. Private shapes, seeds and
  per-case results are never returned. When the budget ends, the workspace is archived. A fresh
  Judge scores that archive once; the run's result is the best valid score across the run's submissions and this final evaluation. Each submission first
  passes an LLM integrity review against `llm_judge/POLICY.md`.

## Hardware

- **GPUs:** two GPUs per run by default. Work runs on GPU 0 and the Judge on GPU 1 (`--work-gpu`,
  `--judge-gpu`), so the Judge measures on its own GPU while Work keeps running. `--shared-gpu`
  puts both on one GPU instead, and the host pauses the Work container while the Judge runs.
- **Execution profile:** one CUDA GPU with compute capability >= 8.9 (FP8 E4M3) and at least
  48 GiB of memory. Work and Judge containers are limited to 8 CPUs and 32 GB of RAM each.

| hardware type | support |
|---|---|
| `h100-94gb-400w` (H100 NVL 94 GB) | qualified |
| `h100-80gb-700w` (H100 SXM 80 GB) | qualified |

TFLOPS scores can only be compared within one hardware type.

## Running

1. **Assets.** Run `tasks/latest/standalone-k-grouped-fp8-wgrad-h100-v2-scratch/assets/fetch.sh`.
   It downloads two Judge-only case lists into `INFRATHON_ASSETS_ROOT` (default
   `~/.cache/infrathon/assets`) from the dataset's `judge/` part: the scored-grid seeds
   and the private edge cases. They are not in the repository. No models or datasets are
   needed. With `INFRATHON_ASSETS_ROOT` set, `build_images.sh` reads both from the asset root.
2. **Images.** Build them in one of two ways:
   - Pull the published Work and public Judge images, then build the Judge holdout layer locally:
     ```bash
     INFRATHON_PULL=1 tasks/latest/standalone-k-grouped-fp8-wgrad-h100-v2-scratch/scripts/build_images.sh
     ```
   - Build everything from source by running the same script without `INFRATHON_PULL`. Both images
     start from `pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel` and use Triton 3.4.0.
3. **Run.**
   ```bash
   launch/standalone-kgrouped-wgrad/run.sh --model <model> --auth /path/to/auth.json \
     [--work-gpu N --judge-gpu M] [--tmux]
   ```
   - Defaults: Work on GPU 0, Judge on GPU 1, 12 hours, effort `max`, Judge port 18604.
     `--work-gpu`, `--judge-gpu`, `--hours`, `--effort` and `--judge-port` change them.
     `--shared-gpu` runs Work and Judge on one GPU, chosen with `--gpu N` (default 0) or
     `--device-uuid UUID`; either flag alone also selects shared mode. `--allocation-key` and
     `--lease-id` are optional scheduler IDs for shared mode, derived when they are omitted.
   - Other options: `--run-id`, `--cpuset`, and `--config FILE` for a launch config (see
     `launch/mlsysforge.env.example`).
   - `--agent claude-code` with `--claude-token FILE` runs Claude Code instead of Codex.
   - `--model` takes any Codex or Claude model ID, or any model that `MODEL_API_BASE_URL`
     serves.
   - `--dry-run` checks the model, runtime, images, integrity review, GPU and port, prints the
     plan and exits. It takes no lock and starts no container.
   - **Eval only:** `PHYSICAL_GPU=<index> tasks/latest/standalone-k-grouped-fp8-wgrad-h100-v2-scratch/scripts/eval_archive.sh ARCHIVE.tar.gz [RUN_ID]`
     evaluates one submission archive in the Judge.
     `PHYSICAL_GPU=<index> scripts/smoke_test.sh --baseline`, run from the task directory,
     checks the images with the reference solution in `baseline/`.
   - **Replay:** `launch/standalone-kgrouped-wgrad/replay.sh --origin-run RUN_ID --measurement-set NAME`
     re-measures the saved candidate versions of a finished run. Add `--dry-run` to preview
     without using a GPU.

**Prerequisites the launcher checks:**

- the SForge harness;
- a version store (`SFORGE_VERSION_STORE_ROOT`, with this task registered);
- an agent runtime directory (`SFORGE_AGENT_RUNTIME_DIR`, the Codex or Claude Code CLI built by
  `launch/build-agent-runtime.sh`, at the version pinned in `launch/common.sh`);
- model credentials: a Codex `auth.json` (mode 0600), a Claude Code token file, or an
  OpenAI-compatible endpoint through `MODEL_API_BASE_URL`/`MODEL_API_KEY_FILE`;
- if the host reaches the internet through a proxy, a proxy address that containers can reach.

The task spec allows internet access. Work reaches public hosts on ports 80/443 only through a
restricted proxy sidecar, and Codex web search is enabled. The LLM integrity review is
configured with the `SFORGE_LLM_JUDGE_*` keys. It runs in `enforce` mode
(required).

## Files

| path | contents |
|---|---|
| `images/work/` | Work image: `AGENT_TASK.md`, the empty starter lab (`workdir/k_grouped_wgrad_lab/`), the public runner, 43 public edge cases and the source-admission checker (`benchmark/`) |
| `images/judge/` | public Judge image (verifier, candidate worker, source admission) and `Dockerfile.private`, the locally built holdout layer |
| `runtime/` | trusted input generator and references, shared by both images |
| `judge/` | SForge Judge adapter (the Judge-only case lists come from the assets, not the repository) |
| `tasks/` | SForge task spec (`*.json`): Judge command, timeouts, resources, submit paths |
| `portable/` | portable contract: suite manifests, software ABI, calibration and selection protocols |
| `assets/` | asset manifest and `fetch.sh` |
| `scripts/` | image build, fixed-budget run, eval, replay, smoke test, manifest generators and validators |
| `baseline/` | host-only reference Triton solution covering the whole domain, never copied into an image |
| `oracle/` | host-only upstream DeepGEMM 2.6.1 reference, never copied into an image. `FINAL_VERSION` 3 selects `candidates/deepgemm_oracle_v003`, which is not admissible (`import deep_gemm`). For reference only, it was scored by the private Judge with source admission and the LLM review turned off (`oracle/forced_judge/`), on an earlier, smaller suite on H100 NVL: 283.8-285.8 TFLOPS on the scored cases, but invalid, because it fails the memory-safety gate (sanitizer 1/8) and the edge gate (50 of that suite's 160 edge cases pass). `oracle/run_oracle.sh` times the same kernel without the gates (about 280 TFLOPS on that suite). `baseline/` is the admissible reference solution (91 TFLOPS, valid, same suite and hardware). See `oracle/README.md` |
| `llm_judge/POLICY.md` | integrity-review policy |
| `tests/` | host-side tests of the task contract |

## Upstream and license

The input generator, quantization recipe (`(1, 1, 128)` per-channel), correctness metric,
scored grid (`enumerate_k_grouped_contiguous`) and FLOP count are a small Judge port of
[DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) at commit
`559d79fb6994a58b8a15b4b93bf13ccc16edf247`. No DeepGEMM code enters either image or runs at
evaluation time. This task defines the following:

- the entry point's ABI: an MN-major layout with a CUDA `ks` tensor and an `expected_k_hint`;
- the support-domain bounds and the edge cases;
- the exact FP64 reference and the element-wise and per-group bounds;
- the memory-safety checks.

DeepGEMM is MIT-licensed. The task's own code is under the repository's root `LICENSE`
(PolyForm Noncommercial 1.0.0).
