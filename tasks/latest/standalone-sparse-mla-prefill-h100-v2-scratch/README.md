# Sparse MLA Prefill Attention, BF16 (from scratch)

An agent writes the sparse-attention prefill kernel of DeepSeek Sparse Attention in
self-contained Triton. The kernel follows FlashMLA's `flash_mla_sparse_fwd` semantics: each
query attends a top-k set of indexed keys, with an optional attention sink and per-row top-k
length. It must be correct and memory-safe over the declared support domain, and as fast as
possible on prefill as a serving engine issues it: FlashMLA's own performance cases plus causal
chunks, tensor-parallel head counts, short shards, multi-request chunks and streams of calls up
to each model's context length. This is a **scratch** task: the starting `submission.py` is an
empty file.

## The task

The agent works in `/app/workdir/sparse_mla_prefill_lab` inside the Work container. It writes
versions to `candidates/kernel_vNNN/submission.py` (three-digit directories), puts the selected
version number in `FINAL_VERSION`, and keeps notes in `notes.md`. Only these three paths are
submitted. The full brief is `images/work/AGENT_TASK.md`.

The selected `submission.py` must define this function. It launches its work on the current
CUDA stream, writes every element of `out`, `max_logits` and `lse`, and returns `None`:

```python
def sparse_mla_prefill(q, kv, indices, sm_scale, attn_sink, topk_length, out, max_logits, lse) -> None: ...
```

- **Tensors.**
  - `q` is `[s_q, h_q, d_qk]` BF16 and `kv` is `[s_kv, 1, d_qk]` BF16.
  - `indices` is `[s_q, 1, topk]` int32.
  - `attn_sink` is `[h_q]` FP32 or `None`, and `topk_length` is `[s_q]` int32 or `None`.
  - The outputs are `out` `[s_q, h_q, 512]` BF16, and `max_logits` and `lse` `[s_q, h_q]` FP32.
  - `h_kv = 1` and `d_v = 512`. `d_qk` is one of the DeepSeek-V3.2 (576, 128 heads), GLM-5
    (576, 64), DeepSeek-V4-Flash (512, 64) and DeepSeek-V4-Pro (512, 128) geometries, and
    `h_q` is the geometry's head count or any multiple of 16 below it (one tensor-parallel rank).
- **Semantics.**
  - A slot is invalid when it is past `topk_length` or its index is outside `[0, s_kv)`.
    Invalid slots can appear anywhere in a row, slots past `topk_length` may hold stale
    valid-looking indices, and duplicated keys count separately.
  - The sink changes only the normalizer of `out`. `max_logits` and `lse` are raw statistics.
  - A row with no valid slot returns `out = 0`, `max_logits = -inf` and `lse = +inf`.
- **Support domain.**
  - `topk` is a multiple of 128 up to 2048.
  - `s_q` is any integer from 1 to 70000, with `s_q * h_q * d_qk <= 2^32` and
    `s_q * topk <= 2^25`. Row offsets therefore need 64-bit integers.
  - `s_kv` is any integer from 1 to 1048576.
  - Any int32 index value, valid or not, is allowed.
  - Sinks can be `+-inf`.
  - `sm_scale` is any positive value.
- **Precision.** Multiply operands are BF16 or FP16. Accumulation and softmax statistics are
  FP32. FP8, INT8 and FP4 are not allowed anywhere.
- **Allowed.** `torch` for metadata, scratch allocation and dispatch on the current stream.
  `triton` / `triton.language` (including Gluon) for the kernels: `tl.dot`, atomics,
  persistent schedules, and autotuning.
- **Not allowed** (rejected by source admission, the Judge or the integrity review):
  - imports outside an allowlist;
  - existing GEMM or attention routes (`matmul`, `@`, `scaled_dot_product_attention`,
    FlashMLA, FlashAttention);
  - 8-bit and 4-bit float types;
  - `torch.compile` and CUDA graphs;
  - patching attributes or global numeric settings, and `getattr`/`setattr`/`eval`/`exec`;
  - private streams;
  - kernel names that the sanitizer would skip;
  - caching or replaying outputs across calls;
  - fast paths that are correct only on scored inputs;
  - specializing for seeds or for benchmark presence;
  - writing outside the outputs and the call's own scratch, or modifying any input.

  The Work image contains PyTorch, Triton, the public contract and public test tools. It does
  not contain FlashMLA, FlashAttention, CUTLASS/CuTe, a reference solution or Judge code.
- **Time budget.** The launcher's default is 12 hours (`--hours`/`--timeout`).

## Evaluation

Every Judge run performs the same complete evaluation. Any failed gate makes the submission
invalid.

- **Sanitizer pass, first.** 17 Judge-only edge cases run in one process under
  `compute-sanitizer --tool memcheck` with 4 MiB padding, before any native call. They include
  one per DeepSeek geometry, the `s_q = 70000` case, a 2^20-key case, a causal case and a
  tensor-parallel head count. An invalid access fails the submission,
  and nothing else runs.
- **Edge cases (correctness, no score weight).** 470 Judge-only cases:
  - FlashMLA's upstream correctness, with-features and corner lists inside the DeepSeek
    geometries;
  - the 17 anchor shapes with full semantics: invalid and duplicated indices, `+-inf` sinks,
    `topk_length`, empty rows, and a non-power-of-two `sm_scale`;
  - zero-bias precision probes at the anchor shapes;
  - an odd-size sweep over the four families;
  - causal indices with stale slots past `topk_length`, keys at each family's context limit,
    and tensor-parallel head counts.

  Each case is called twice. 73 public cases with other shapes ship in the Work image.
- **Correctness.** All three outputs are compared with an independent chunked FP32 reference,
  using FlashMLA's limits:
  - `out`: every finite element within `abs < 8e-4` or `rel < 3.01/128`, and cosine
    difference `<= 7e-6`;
  - `max_logits` and `lse`: `1e-6` (scaled with `sm_scale`) or `2.01/65536`, and cosine
    difference `<= 1e-7`;
  - the positions of `+inf`, `-inf` and NaN must match exactly.

  The outputs are filled with NaN before each call, so every element must be written.
- **Memory safety.** Each tensor sits between guard bands of at least 1 MiB or 128 rows. The
  bands around `indices` and `topk_length` hold valid values, so a stray read gives a wrong
  answer instead of a wild address. Every input is compared byte for byte after every call.
- **Harness integrity.** The Judge captures its clocks, CUDA events and checking functions
  before it imports the candidate. It fails the run if any of them changes. It also rejects
  throughput above the device's dense BF16 peak.
- **Score.** 997 Judge-only scored cases over the four families (about 250 each), in eight groups:

  | group | cases | what it times |
  | --- | ---: | --- |
  | `anchor` | 17 | FlashMLA's `performance_cases` (`s_q = 4096`, upstream inputs, `sm_scale = 0.5`) and the same template at the GLM-5 geometry |
  | `trace` | 80 | streams of up to 100 prefill calls of consecutive requests of three Mooncake FAST'25 traces (conversation, tool-agent, synthetic), chunked at 2048 to 16384 rows |
  | `replay` | 60 | streams of one whole-context chunked prefill, contexts log-spaced from 16384 keys to the model's context length (1048576 for V4) |
  | `multi` | 100 | one call over 2 to 128 requests packed into one KV workspace (vLLM's prefill workspace), lengths drawn from the traces, a third with one long-context request |
  | `pattern` | 120 | causal single calls whose indices carry a learned indexer's locality: local window, sink plus window, blocks shared by 64-row tiles, sorted |
  | `sweep` | 400 | causal single calls stratified on a log grid, `s_q` 1 to the native cap over 256 keys to the model's context length, half aligned and half ragged |
  | `heads` | 120 | causal single calls at every tensor-parallel split of the model's heads (16 to 64) |
  | `length` | 100 | decode-context-parallel shards with `topk_length` near `topk / 2`, `/ 4` or `/ 8`, even or ragged across rows |

  Non-anchor calls use causal (or shard, per-request or patterned) indices with `topk_length`,
  half of the slots past `topk_length` hold stale indices, `sm_scale = qk_head_dim ** -0.5`, and
  the sink is finite or absent. Shapes, trace windows and seeds are Judge-only; the suite plan
  that lists them is not in the repository.
  - A single call: one warm-up, then three trials, each a decoy call on other tensors, a 256 MiB
    L2 flush and one timed call. The reference checks the first trial of every anchor and
    multi-request and pattern case and of a hidden half of the sweep-tier cases; every other call is checked
    for complete writes.
  - A stream: 16 untimed warm-up calls at shapes next to the stream's own, then every call is
    timed with compilation included; every eighth call and the last are checked against the
    reference. A stream stops after 8 timed seconds and scores the work it finished.
  - Time is the larger of host wall time and CUDA-event time.
  - `effective TFLOPS = 2 * positions * h_q * (d_qk + 512) / seconds / 1e12`, with positions
    `s_q * topk` or `sum(topk_length)`; a stream scores total FLOPs over total seconds.
  - Each group is the geometric mean of its cases in one family; the evidence tier (anchor,
    trace, replay, multi, pattern) and the sweep tier (sweep, heads, length) are the geometric means of
    their groups; a family scores `sqrt(evidence * sweep)`; the score is the geometric mean of
    the four family scores in effective TFLOPS. Higher is better. It is uncapped, with no
    baseline ratio and no threshold.
- **Interactive and final evaluation.** Scheduled auto-evaluation is off. The agent can submit
  with `sforge-submit`, at most once per 300 s cooldown. It gets back the gate states, the
  score, the four family scores and the number of failed edge cases. Private seeds, inputs and
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

1. **Assets.** Run `tasks/latest/standalone-sparse-mla-prefill-h100-v2-scratch/assets/fetch.sh`.
   It downloads two Judge-only case lists into `INFRATHON_ASSETS_ROOT` (default
   `~/.cache/infrathon/assets`) from the dataset's `judge/` part: the scored-case seeds
   and the private edge cases. They are not in the repository. No models or datasets are
   needed. With `INFRATHON_ASSETS_ROOT` set, `build_images.sh` reads both from the asset root.
2. **Images.** Build them in one of two ways:
   - Pull the published Work and public Judge images, then build the Judge holdout layer locally:
     ```bash
     INFRATHON_PULL=1 tasks/latest/standalone-sparse-mla-prefill-h100-v2-scratch/scripts/build_images.sh
     ```
   - Build everything from source by running the same script without `INFRATHON_PULL`. Both images
     start from `pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel` and use Triton 3.4.0.
3. **Run.**
   ```bash
   launch/standalone-sparse-mla-v2/run.sh --model <model> --auth /path/to/auth.json \
     [--work-gpu N --judge-gpu M] [--tmux]
   ```
   - Defaults: Work on GPU 0, Judge on GPU 1, 12 hours, effort `max`, Judge port 18606.
     `--work-gpu`, `--judge-gpu`, `--hours`, `--effort` and `--judge-port` change them.
     `--shared-gpu` runs Work and Judge on one GPU, chosen with `--gpu N` (default 0) or
     `--device-uuid UUID`; either flag alone also selects shared mode. `--allocation-key` and
     `--lease-id` are optional scheduler IDs for shared mode, derived when they are omitted.
   - Other options: `--run-id`, `--cpuset`, and `--config FILE` for a launch config (see
     `launch/mlsysforge.env.example`).
   - `--model` takes any Codex model ID, or any model that `MODEL_API_BASE_URL` serves.
   - `--dry-run` checks the model, runtime, images, integrity review, GPU and port, prints the
     plan and exits. It takes no lock and starts no container.
   - **Eval only:** `PHYSICAL_GPU=<index> tasks/latest/standalone-sparse-mla-prefill-h100-v2-scratch/scripts/eval_archive.sh ARCHIVE.tar.gz [RUN_ID]`
     evaluates one submission archive in the Judge.
     `PHYSICAL_GPU=<index> scripts/smoke_test.sh --oracle`, run from the task directory,
     checks the images with the reference solution.
   - **Replay:** `launch/standalone-sparse-mla-v2/replay.sh --origin-run RUN_ID --measurement-set NAME`
     re-measures the saved candidate versions of a finished run. Add `--dry-run` to preview
     without using a GPU.

**Prerequisites the launcher checks:**

- the SForge harness;
- a version store (`SFORGE_VERSION_STORE_ROOT`, with this task registered);
- an agent runtime directory (`SFORGE_AGENT_RUNTIME_DIR`, the Codex CLI built by
  `launch/build-agent-runtime.sh`, at the version pinned in `launch/common.sh`);
- model credentials: a Codex `auth.json` (mode 0600), or an OpenAI-compatible endpoint
  through `MODEL_API_BASE_URL`/`MODEL_API_KEY_FILE`;
- a ready LLM integrity review: `SFORGE_LLM_JUDGE_MODE=enforce` and the other
  `SFORGE_LLM_JUDGE_*` reviewer settings. The launcher refuses to start
  without them;
- if the host reaches the internet through a proxy, a proxy address that containers can reach.

The task spec allows internet access. Work reaches public hosts on ports 80/443 only through a
restricted proxy sidecar, and Codex web search is enabled (the task spec sets
`internet: true`). Fetching upstream solution material is still against the task rules.

Untrusted kernels can produce wild pointers. Run candidates only through the Judge (sanitizer
pass first) or under `compute-sanitizer`, one process at a time, never natively in ad-hoc
probes.

## Files

| path | contents |
|---|---|
| `images/work/` | Work image: `AGENT_TASK.md`, the empty starter lab (`workdir/sparse_mla_prefill_lab/`), the public runner, 73 public edge cases and the source-admission checker (`benchmark/`) |
| `images/judge/` | public Judge image (verifier, candidate worker, source admission) and `Dockerfile.private`, the locally built holdout layer |
| `runtime/` | trusted input generator and chunked FP32 reference, shared by both images |
| `judge/` | SForge Judge adapter (the Judge-only case lists come from the assets, not the repository) |
| `tasks/` | SForge task spec (`*.json`): Judge command, timeouts, resources, submit paths |
| `portable/` | portable contract: suite manifests, software ABI, calibration and selection protocols |
| `assets/` | asset manifest and `fetch.sh` |
| `scripts/` | image build, fixed-budget run, eval, replay, smoke test, manifest generators and validators |
| `oracle/` | host-only oracle, never copied into an image; `artifacts/oracle.tar.gz` (`scripts/make_oracle_archive.sh`) is its submission archive. `FINAL_VERSION` = 2 selects the paper solution, upstream FlashMLA SM90 (`kernel_v002`), which is not admissible (it imports `flash_mla.cuda`). For reference only, it was scored by the private Judge with source admission and the LLM review turned off and every other gate enforced (`oracle/paper_reference/forced_judge/`), on an earlier suite on H100 NVL: 386.37 / 381.59 TFLOPS in two runs, with the sanitizer and all 401 edge cases of that suite passing; these numbers are not comparable with scores on the current suite. `ADMISSIBLE_VERSION` = 1 selects the in-task Triton control (`kernel_v001`, 146.0 TFLOPS, valid, on the same earlier suite and hardware; no admissible paper port exists), which `make_oracle_archive.sh --admissible` and `smoke_test.sh --oracle` use. `oracle/paper_reference/` also holds the FlashMLA trusted-runtime harness |
| `llm_judge/POLICY.md` | integrity-review policy |
| `tests/` | host-side tests of the task contract |

## Upstream and license

The semantics, test lists, input generator, tolerances, scored suite and FLOP count come from
[FlashMLA](https://github.com/deepseek-ai/FlashMLA) (MIT) at commit
`ba89a3466e9470ad08ab39738d4e7bb66989e1e7`, the last commit with Hopper support. The specific sources are `tests/test_flash_mla_sparse_prefill.py`,
`tests/lib.py`, `tests/ref.py`, `tests/kernelkit/` and the SM90 sparse-prefill kernel's
assertions. The reference is a port of upstream's `ref_sparse_attn_fwd`.

The model geometries come from the Hugging Face configs of `deepseek-ai/DeepSeek-V3.2`, GLM-5,
`deepseek-ai/DeepSeek-V4-Flash` and `deepseek-ai/DeepSeek-V4-Pro`. No upstream source is
fetched by any script or present in either image.

The task's own code is under the repository's root `LICENSE` (PolyForm Noncommercial 1.0.0).
