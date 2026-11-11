# Dynamic Grouped FP8 GEMM (from scratch)

An agent writes a grouped, block-scaled FP8 GEMM, the mixture-of-experts GEMM that DeepGEMM
provides, in self-contained Triton. The kernel has two entry points: a padded/masked layout
and a packed prefix-sum layout. It must be correct and memory-safe for every per-group row
count in the declared domain, and as fast as possible on a hidden suite that models one rank
of mixture-of-experts serving. This is a **scratch** task: the starting
`submission.py` is an empty file.

## The task

The agent works in `/app/workdir/grouped_fp8_lab` inside the Work container. It writes versions
to `candidates/kernel_vNNN/submission.py` (three-digit directories), puts the selected version
number in `FINAL_VERSION`, and keeps notes in `notes.md`. Only these three paths are submitted.
The full brief is `images/work/AGENT_TASK.md`.

The selected `submission.py` must define both functions. Each launches its work on the current
CUDA stream, writes `out` in place, and returns `None`:

```python
def grouped_fp8_gemm_nt_masked(a, b, a_scale, b_scale, m_sizes, expected_m_hint, out) -> None: ...
def grouped_fp8_gemm_nt_psum(a, b, a_scale, b_scale, group_ends, expected_m_hint, out) -> None: ...
```

- **Math.** Operands are FP8 E4M3FN. A has one FP32 scale per row and 128-wide K block. B has
  one FP32 scale per 128 x 128 (N, K) block. Each 128-wide K block's partial product is scaled,
  accumulated in FP32, and cast once to BF16.
- **Masked layout.** `a` is `[G, M_max, K]`. Group `g` has `m_sizes[g]` valid rows.
- **Prefix-sum layout.** `a` is `[G * M_max, K]`. Each group starts at the 128-aligned end of
  the previous group and ends at `group_ends[g]`.
- **Support domain.** `1 <= G <= 64`. `N` and `K` are multiples of 128 from 128 to 8192.
  `M_max` is a multiple of 128 from 128 to 4096 (masked) or to 131072 (prefix-sum), with
  `G * M_max * (K + 2 * N) <= 2^34`. A group can have any row count from 0 to `M_max`.
  `expected_m_hint` is only a scheduling hint; it can be stale or badly wrong.
- **Allowed.** `torch` for metadata, scratch allocation and dispatch on the current stream.
  `triton` / `triton.language` for the kernels: `tl.dot`, atomics, persistent schedules,
  autotuning, and TMA descriptors.
- **Not allowed** (rejected by source admission, the Judge or the integrity review):
  - any existing GEMM: `torch.mm`/`matmul`/`bmm`/`_scaled_mm`/`einsum`, `@`, `torch.ops`,
    cuBLAS, CUTLASS/CuTe, DeepGEMM, or another compiled extension;
  - imports outside a small allowlist;
  - patching any module, class or object attribute, and `getattr`/`setattr`/`eval`/`exec`;
  - CUDA graphs and private streams;
  - caching or replaying outputs, or anything derived from tensor contents, across calls;
  - specializing for seeds or for benchmark presence;
  - using `expected_m_hint` to skip rows;
  - writing outside `out` or modifying any input;
  - fetching or copying external implementations.

  The Work image contains PyTorch, Triton, the public contract and public test tools. It does
  not contain DeepGEMM, a GEMM kernel, a reference solution or Judge code.
- **Time budget.** The launcher's default is 12 hours (`--hours`/`--timeout`).

## Evaluation

Every Judge run performs the same complete evaluation. Any failed gate makes the submission
invalid.

- **Edge cases (correctness, no score weight).** 188 Judge-only cases cover the whole support
  domain: one-block and odd-block N/K, N or K at 8192, 1 to 64 groups, empty, single-row, full
  and tile-boundary groups, hints far from the real row counts, tight capacities, and
  prefix-sum row offsets past 2^31 bytes of A, M_max up to 131072, single groups, one hot
  group, all-empty calls and extreme hints. Each case is called twice. Thirty-two public cases
  ship in the Work image, including one past the 2^31-byte offset.
- **Correctness.** Every non-empty group is checked against two references:
  - the FP64 product of the dequantized FP8 operands: normalized error below `5e-5`;
  - DeepGEMM's `calc_diff < 1e-3` against the BF16 source GEMM, on groups (or pooled small
    groups) with at least 16,384 outputs.

  Valid output rows are filled with NaN before each single call (before each round for
  sequence cases, whose last writer of every output buffer is checked against the exact
  reference), so every valid element must be written and finite.
- **Memory safety.**
  - Each tensor sits between guard bands.
  - Inputs are compared after every call.
  - A subset of edge cases runs under `compute-sanitizer --tool memcheck`. On a platform where
    the sanitizer cannot attach, this pass is skipped and recorded as `sanitizer_ran = 0`.
- **Harness integrity.** The Judge captures its clocks, CUDA events and checking functions
  before it imports the candidate. It fails the run if any of them changes. It also rejects
  throughput above the device's dense FP8 peak.
- **Score.** 1000 Judge-only cases in four families:
  - **prefill** (200): prefix-sum, large batches, Dirichlet-skewed per-group rows;
  - **decode** (500): both layouts, small batches, skewed rows with empty groups; 100 cases
    carry a hint 1/8 to 8 times off;
  - **latency** (200): one group, or very few rows across mostly empty groups;
  - **sequence** (100): 58 layers x two linears = 116 consecutive calls per round, weights from
    an 8-layer pool, fresh row counts per call, no synchronization between calls.

  The 20 hidden `(N, K)` pairs come from public MoE model configurations and are disjoint from
  the 10 public pairs. Row counts are synthetic (calibrated Dirichlet), not a captured trace,
  and every per-call row count is a pure function of the case seed, which the Judge recomputes.
  - A single-call case has one warm-up and five measured trials. Before each measured call, the
    Judge makes one untimed call with the same shapes and hint but different row counts and
    tensors. A sequence case has one untimed round and three timed rounds.
  - Time is the larger of host wall time and CUDA-event time (per call, or per round).
  - `effective TFLOPS = 2 * sum(actual M) * N * K / seconds / 1e12`, summed over a round's calls.
  - A single-call case is its median trial, a sequence case its median round. Cases are
    geometrically averaged within each family and the score is the geometric mean of the four
    family scores, in effective TFLOPS. Higher is better. It is uncapped, with no baseline ratio
    and no threshold.
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
  56 GiB of memory (the Judge's largest prefill case peaks near 48 GB with its references). Work and Judge containers are limited to 8 CPUs and 32 GB of RAM each.

| hardware type | support |
|---|---|
| `h100-94gb-400w` (H100 NVL 94 GB) | qualified |
| `h100-80gb-700w` (H100 SXM 80 GB) | qualified |

TFLOPS scores can only be compared within one hardware type.

## Running

1. **Assets.** Run `tasks/latest/standalone-dynamic-grouped-fp8-h100-v3-scratch/assets/fetch.sh`.
   It downloads two Judge-only case lists into `INFRATHON_ASSETS_ROOT` (default
   `~/.cache/infrathon/assets`) from the dataset's `judge/` part: the 1000 scored cases
   and the private edge cases. They are not in the repository. No models or datasets are
   needed. With `INFRATHON_ASSETS_ROOT` set, `build_images.sh` reads both from the asset root.
2. **Images.** Build them in one of two ways:
   - Pull the published Work and public Judge images, then build the Judge holdout layer locally:
     ```bash
     INFRATHON_PULL=1 tasks/latest/standalone-dynamic-grouped-fp8-h100-v3-scratch/scripts/build_images.sh
     ```
   - Build everything from source by running the same script without `INFRATHON_PULL`. Both images
     start from `pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel` and use Triton 3.4.0.
3. **Run.**
   ```bash
   launch/standalone-fp8/run.sh --model <model> --auth /path/to/auth.json \
     [--work-gpu N --judge-gpu M] [--tmux]
   ```
   - Defaults: Work on GPU 0, Judge on GPU 1, 12 hours, effort `max`, Judge port 18603.
     `--work-gpu`, `--judge-gpu`, `--hours`, `--effort` and `--judge-port` change them.
     `--shared-gpu` runs Work and Judge on one GPU, chosen with `--gpu N` (default 0) or
     `--device-uuid UUID`; either flag alone also selects shared mode. `--allocation-key` and
     `--lease-id` are optional scheduler IDs for shared mode, derived when they are omitted.
   - Other options: `--run-id`, `--cpuset`, and `--config FILE` for a launch config (see
     `launch/mlsysforge.env.example`).
   - `--model` takes any Codex model ID, or any model that `MODEL_API_BASE_URL` serves.
   - **Eval only:** `PHYSICAL_GPU=<index> tasks/latest/standalone-dynamic-grouped-fp8-h100-v3-scratch/scripts/eval_archive.sh ARCHIVE.tar.gz [RUN_ID]`
     evaluates one submission archive in the Judge.
     `PHYSICAL_GPU=<index> scripts/smoke_test.sh --oracle`, run from the task directory,
     checks the images with the reference solution.
   - **Replay:** `launch/standalone-fp8/replay.sh --origin-run RUN_ID --measurement-set NAME`
     re-measures the saved candidate versions of a finished run. Add `--dry-run` to preview
     without using a GPU.

**Prerequisites the launcher checks:**

- the SForge harness;
- a version store (`SFORGE_VERSION_STORE_ROOT`, with this task registered);
- an agent runtime directory (`SFORGE_AGENT_RUNTIME_DIR`, the Codex CLI built by
  `launch/build-agent-runtime.sh`, at the version pinned in `launch/common.sh`);
- model credentials: a Codex `auth.json` (mode 0600), or an OpenAI-compatible endpoint
  through `MODEL_API_BASE_URL`/`MODEL_API_KEY_FILE`;
- if the host reaches the internet through a proxy, a proxy address that containers can reach.

The task spec allows internet access. Work reaches public hosts on ports 80/443 only through a
restricted proxy sidecar, and Codex web search is enabled. The LLM integrity review is
configured with the `SFORGE_LLM_JUDGE_*` keys. It runs in `enforce` mode
(required).

## Files

| path | contents |
|---|---|
| `images/work/` | Work image: `AGENT_TASK.md`, the empty starter lab (`workdir/grouped_fp8_lab/`), the public runner, 60 public cases, 32 public edge cases and the source-admission checker (`benchmark/`) |
| `images/judge/` | public Judge image (verifier, candidate worker, source admission) and `Dockerfile.private`, the locally built holdout layer |
| `runtime/` | trusted input generator and references, shared by both images |
| `judge/` | SForge Judge adapter (the Judge-only case lists come from the assets, not the repository) |
| `tasks/` | SForge task spec (`*.json`): Judge command, timeouts, resources, submit paths |
| `portable/` | portable contract: suite manifests, software ABI, calibration and selection protocols |
| `assets/` | asset manifest and `fetch.sh` |
| `scripts/` | image build, fixed-budget run, eval, replay, smoke test, manifest generators and validators |
| `oracle/` | host-only oracle, never copied into an image; `scripts/make_oracle_archive.sh` packs it. `FINAL_VERSION` = 2 selects the paper solution, upstream DeepGEMM 2.6.1 (`kernel_v002`), which is not admissible (`import deep_gemm`). For reference only, it was scored by the private Judge with source admission and the LLM review turned off and every other gate enforced (`oracle/paper_reference/forced_judge/`), on an earlier suite on H100 NVL: 166.76 / 164.94 TFLOPS in two runs, with the sanitizer and all 158 edge cases of that suite passing. `ADMISSIBLE_VERSION` = 1 selects the Triton control `kernel_v001` (126.88 TFLOPS, valid, on the same earlier suite and hardware; no admissible paper port exists), used by `make_oracle_archive.sh --admissible` and `smoke_test.sh --oracle`. `oracle/paper_reference/` also holds the DeepGEMM trusted-runtime harness (not admissible: `import deep_gemm`) |
| `llm_judge/POLICY.md` | integrity-review policy |
| `tests/` | host-side tests of the task contract |

## Upstream and license

The tensor layouts, quantization, correctness metric and FLOP count are a small Judge port
of
[DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) at commit
`559d79fb6994a58b8a15b4b93bf13ccc16edf247`: `tests/generators.py`, `tests/test_fp8_fp4.py`,
`tests/test_sanitizer.py` and `csrc/apis/gemm.hpp`; the public suite keeps its masked
benchmark grid and contiguous shapes. No DeepGEMM code enters either image or runs at
evaluation time. This task defines the support-domain bounds, the edge cases, the serving
workload, the exact FP64 reference and the memory-safety checks. DeepGEMM is MIT-licensed.

The task's own code is under the repository's root `LICENSE` (PolyForm Noncommercial 1.0.0).
