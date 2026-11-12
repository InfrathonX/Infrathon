# Paged GQA Decode Attention, BF16 (from scratch)

An agent writes a paged-KV grouped-query attention kernel for decode and short-query prefill in
self-contained Triton. The kernel follows FlashAttention's `flash_attn_with_kvcache` semantics:
paged K/V, GQA/MQA, and bottom-right causal masking. It must be correct and memory-safe over a
support domain drawn from the upstream tests of FlashAttention, vLLM, FlashInfer and SGLang.
It must also be as fast as possible on head configurations from FlashAttention's model
benchmark table. This is a **scratch** task: the starting `submission.py` is an empty file.

## The task

The agent works in `/app/workdir/paged_gqa_decode_lab` inside the Work container. It writes
versions to `candidates/kernel_vNNN/submission.py` (three-digit directories), puts the selected
version number in `FINAL_VERSION`, and keeps notes in `notes.md`. Only these three paths are
submitted. The full brief is `images/work/AGENT_TASK.md`.

The selected `submission.py` must define this function. It launches its work on the current
CUDA stream, overwrites every element of `out`, and returns `None`:

```python
def paged_gqa_decode_bf16(q, k_cache, v_cache, page_table, cache_seqlens, softmax_scale, out) -> None: ...
```

- **Tensors.**
  - `q` and `out` are `[B, S_q, H_q, 128]` BF16.
  - `k_cache` and `v_cache` are `[P, page_size, H_kv, 128]` BF16.
  - `page_table` is `[B, W]` int32, and `cache_seqlens` is `[B]` int32.
  - Logical key `j` of request `b` is at `page_table[b, j // page_size]`, slot
    `j % page_size`.
  - Query head `h` uses KV head `h // (H_q // H_kv)`.
  - Query row `i` sees keys `0 <= j <= L_b - S_q + i`.
- **Support domain.**
  - `D = 128`; `1 <= B <= 256` and `1 <= S_q <= 16`.
  - Any GQA group size with `H_q <= 128`.
  - The page size is any power of two from 1 to 256.
  - `S_q <= L_b <= L_max <= 1048576`.
  - Page tables can be wider than needed, and page pools larger than `B x W`, with repeated
    page ids.
  - Cache slots that no query may see can hold any BF16 bit pattern, including NaN and Inf.
  - A cache can exceed 2^31 elements, so offsets must be 64-bit.
- **Precision.** BF16 and FP32 only. FP16, FP8, INT8 and FP4 are not allowed anywhere.
- **Allowed.** One or more `@triton.jit` kernels for every tensor data path, with ordinary
  Triton inside them (`tl.dot`, atomics, split-K, autotuning). Host code is a small,
  statically checked subset:
  - no host loops, comprehensions or lambdas;
  - only tensor metadata calls and `torch.empty`/`zeros`/`full` allocation;
  - `@triton.jit` and `@triton.autotune` only.

  `AGENT_TASK.md` lists the exact rules.
- **Not allowed** (rejected by source admission, the Judge or the integrity review):
  - attention, GEMM or softmax through host Torch, or any existing attention library;
  - `torch.ops` and extension compilation;
  - dynamic code and `getattr`/`setattr`/`eval`/`exec`;
  - patching module attributes or global numeric settings;
  - private streams;
  - caching or replaying outputs across calls;
  - fast paths that are correct only on the scored layouts;
  - specializing for seeds or for benchmark presence;
  - writing outside `out` and the call's own scratch, or modifying any input.

  The Work image contains PyTorch, Triton, the public contract and public test tools. It does
  not contain FlashAttention, FlashMLA, CUTLASS/CuTe, a reference solution or Judge code.
- **Time budget.** The launcher's default is 12 hours (`--hours`/`--timeout`).

## Evaluation

Every Judge run performs the same complete evaluation. Any failed gate makes the submission
invalid.

- **Sanitizer pass, first.** A small subset of edge cases runs under
  `compute-sanitizer --tool memcheck` before any native call. This subset includes a cache
  larger than 2^31 elements. On a platform where the sanitizer cannot attach, this pass is
  skipped and recorded as not run.
- **Edge cases (correctness, no score weight).** 168 Judge-only cases cover the support domain.
  Their categories are batch, lengths, `S_q`, heads, page size, page-pool and page-table
  layouts, value hazards (extreme logits, NaN/Inf in hidden slots), long contexts up to
  1,048,576 tokens and the sanitizer subset. Each case is called twice. Forty-four public cases
  from the same generator ship in the Work image.
- **Correctness.** The Judge compares `out` with an FP32 de-paged reference `ref` and with the
  upstream test's BF16 reference `bref`:
  - `max_abs(out - ref) <= 2 * max_abs(bref - ref)` plus a rounding floor;
  - `mean_abs(out - ref) <= 1.5 * mean_abs(bref - ref)`;
  - normalized L2 at most `5e-3`;
  - every element finite and written, because `out` is filled with NaN first.
- **Memory safety.** Each tensor sits between 1 MiB guard bands. The bands around the page
  table and the lengths hold valid values, so a stray read gives a wrong answer instead of a
  wild address. Every input is compared byte for byte after every call. A CUDA fault stops the
  run.
- **Harness integrity.** The Judge captures its clocks, CUDA events and checking functions
  before it imports the candidate. It fails the run if any of them changes. It also rejects
  throughput above the device's dense BF16 peak.
- **Score.** 1114 scored cases model one tensor-parallel rank of a serving engine
  during decode, in four families. Head configurations come from public models split across
  tensor-parallel ranks; lengths, batch composition and shared-prefix blocks come from a
  simulated server replaying the Mooncake conversation and tool-agent request traces at
  several load levels and time windows, with a per-rank KV budget computed from each
  model's configuration.
  - `decode` (440): one decode step of a busy server, `S_q = 1`, very uneven lengths, shared
    prefix pages, a fragmented pool larger than needed, a table wider than needed, NaN in
    hidden slots for a quarter of the cases; plus the parallel-sampling (n = 2, 4, 6),
    beam-search (width 2, 4, 6) and shared few-shot prompt (80 and 341 tokens) workloads of
    the PagedAttention paper (sections 6.3-6.4), shared at 16-token granularity.
  - `spec` (374): 2 to 16 query tokens per request, batches of 1 to 64 requests.
  - `long` (220): contexts up to each model's maximum position (up to 1,048,576), single
    requests or one long request among a busy batch, plus the four configurations of the
    split-KV decode benchmark the task semantics come from (64/8 heads, 131072 tokens).
  - `stream` (80): 512 consecutive decode steps of one server, ten of them speculative
    (`S_q` 2-5), covering every hidden head configuration and page size at four load levels; requests join and leave
    between steps, pages are reserved on arrival and returned on departure, the table width
    and pool stay fixed. 16 untimed calls on nearby shapes come first; after that every
    step is timed and compilation counts. The timed calls of one sequence may take at most
    120 s.
  - A quarter of the single calls draw Q, K and V from `N(0,1) + N(0,100) * Bernoulli(0.001)`,
    the outlier distribution of the FlashAttention-3 numerics study (section 4.3).
  - Each single-call case has one warm-up and three measured trials. Before each measured
    call the Judge makes one untimed call with the same batch, heads, page size, table width
    and pool but different lengths and data. Time is the larger of host wall time and
    CUDA-event time. The golden check runs on every trial, and every 64 steps (plus the last)
    of a sequence.
  - `effective_flops = 4 * 128 * H_q * sum_b(S_q * L_b - S_q * (S_q - 1) / 2)`. A single-call
    case is its median trial; a sequence is total work over total time. The score is the
    geometric mean of the four family geometric means, in effective TFLOPS. Higher is better.
    It is uncapped, with no baseline ratio and no threshold.
  - The hidden cases are frozen in the Judge-only asset `judge/final_cases.json`; the generator,
    traces and frozen plan that produced them are the Judge-only asset `judge/source/`, not
    part of the repository. The public suite has 127 cases over the same
    four families with disjoint head configurations, a different request trace and different
    page sizes (`images/work/benchmark/public_cases.json`).
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
- **Execution profile:** one CUDA GPU with compute capability >= 8.0 and at least 32 GiB of
  memory. Work and Judge containers are limited to 8 CPUs and 32 GB of RAM each.

| hardware type | support |
|---|---|
| `h100-94gb-400w` (H100 NVL 94 GB) | qualified |
| `h100-80gb-700w` (H100 SXM 80 GB) | qualified |

TFLOPS scores can only be compared within one hardware type.

## Running

1. **Assets.** Run `tasks/latest/standalone-paged-gqa-decode-bf16-h100-v2-scratch/assets/fetch.sh`.
   It downloads the Judge-only files into `INFRATHON_ASSETS_ROOT` (default
   `~/.cache/infrathon/assets`) from the dataset's `judge/` part: the scored-case
   lengths and seeds, the private edge cases, and `judge/source/`. They are not in the repository. No models
   or datasets are needed. With
   `INFRATHON_ASSETS_ROOT` set, `build_images.sh` reads both from the asset root.
2. **Images.** Build them in one of two ways:
   - Pull the published Work and public Judge images, then build the Judge holdout layer locally:
     ```bash
     INFRATHON_PULL=1 tasks/latest/standalone-paged-gqa-decode-bf16-h100-v2-scratch/scripts/build_images.sh
     ```
   - Build everything from source by running the same script without `INFRATHON_PULL`. Both images
     start from `pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel` and use Triton 3.4.0.
3. **Run.**
   ```bash
   launch/standalone-paged-gqa-v2/run.sh --model <model> --auth /path/to/auth.json \
     [--work-gpu N --judge-gpu M] [--tmux]
   ```
   - Defaults: Work on GPU 0, Judge on GPU 1, 12 hours, effort `max`, Judge port 18605.
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
   - **Eval only:** `PHYSICAL_GPU=<index> tasks/latest/standalone-paged-gqa-decode-bf16-h100-v2-scratch/scripts/eval_archive.sh ARCHIVE.tar.gz [RUN_ID]`
     evaluates one submission archive in the Judge.
     `PHYSICAL_GPU=<index> scripts/smoke_test.sh --oracle`, run from the task directory,
     checks the images with the reference solution.
   - **Replay:** `launch/standalone-paged-gqa-v2/replay.sh --origin-run RUN_ID --measurement-set NAME`
     re-measures the saved candidate versions of a finished run. Add `--dry-run` to preview
     without using a GPU.

**Prerequisites the launcher checks:**

- the SForge harness;
- a version store (`SFORGE_VERSION_STORE_ROOT`, with this task registered);
- an agent runtime directory (`SFORGE_AGENT_RUNTIME_DIR`, the Codex or Claude Code CLI built by
  `launch/build-agent-runtime.sh`, at the version pinned in `launch/common.sh`);
- model credentials: a Codex `auth.json` (mode 0600), a Claude Code token file, or an
  OpenAI-compatible endpoint through `MODEL_API_BASE_URL`/`MODEL_API_KEY_FILE`;
- a ready LLM integrity review: `SFORGE_LLM_JUDGE_MODE=enforce` and the other
  `SFORGE_LLM_JUDGE_*` reviewer settings. The launcher refuses to start
  without them;
- if the host reaches the internet through a proxy, a proxy address that containers can reach.

The task spec allows internet access. Work reaches public hosts on ports 80/443 only through a
restricted proxy sidecar, and Codex web search is enabled.

Untrusted kernels can produce wild pointers. Run candidates only through the Judge (sanitizer
pass first) or under `compute-sanitizer`, never natively in ad-hoc probes.

## Files

| path | contents |
|---|---|
| `images/work/` | Work image: `AGENT_TASK.md`, the empty starter lab (`workdir/paged_gqa_decode_lab/`), the public runner, 44 public edge cases and the source-admission checker (`benchmark/`) |
| `images/judge/` | public Judge image (verifier, candidate worker, source admission) and `Dockerfile.private`, the locally built holdout layer |
| `runtime/` | trusted input generator and references, shared by both images |
| `judge/` | SForge Judge adapter (the Judge-only case lists come from the assets, not the repository) |
| `tasks/` | SForge task spec (`*.json`): Judge command, timeouts, resources, submit paths |
| `portable/` | portable contract: suite manifests, software ABI, calibration and selection protocols |
| `assets/` | asset manifest and `fetch.sh` |
| `scripts/` | image build, fixed-budget run, eval, replay, smoke test, manifest generators and validators |
| `oracle/` | host-only Oracle, never copied into an image. `FINAL_VERSION` selects `kernel_v003`, upstream FlashAttention-3 (0251105a), the strongest paper solution: not admissible (`import flash_attn_3`). For reference only, it was scored by the private Judge with source admission and the LLM review turned off (`oracle/paper_reference/forced_judge/`), on an earlier ten-case suite on H100 NVL: 53.58 TFLOPS on the scored cases, but invalid, because 3 of that suite's 160 edge cases fail (the sanitizer passes). `ADMISSIBLE_VERSION` selects `kernel_v002`, upstream vLLM unified Triton paged attention (glue-only port; valid; 24.0-24.4 TFLOPS on the same earlier suite and H100 NVL, 3.63 TFLOPS on the current suite on H100 SXM), which `make_oracle_archive.sh`, `smoke_test.sh --oracle` and the tests use. `kernel_v001` is the straightforward Triton control (13.3 TFLOPS on the earlier suite, H100 NVL). See `oracle/notes.md` |
| `llm_judge/POLICY.md` | integrity-review policy |
| `tests/` | host-side tests of the task contract |

## Upstream and license

The semantics, reference practice, tolerances and scored head configurations come from
[FlashAttention](https://github.com/Dao-AILab/flash-attention) at commit
`0251105a2fb19d2957484b7f023cd8c115286ced` (BSD 3-Clause). The specific sources are
`hopper/flash_attn_interface.py`, `hopper/paged_kv.h`, `hopper/test_flash_attn.py` and
`hopper/benchmark_split_kv.py`.

The support domain takes shapes, layouts and numeric hazards from the tests of four
projects. Only values are taken; no code is copied:

- vLLM `6bbad6a` (Apache-2.0);
- FlashInfer `ea8135c` (Apache-2.0);
- SGLang `f3b5a28` (Apache-2.0);
- the PagedAttention paper (arXiv 2309.06180).

No upstream source or compiled package is in either image. The trusted runtime is a
task-authored PyTorch reference and input generator.

The task's own code is under the repository's root `LICENSE` (PolyForm Noncommercial 1.0.0).
