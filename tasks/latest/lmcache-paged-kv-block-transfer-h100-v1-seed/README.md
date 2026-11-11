# LMCache Paged-KV Block Transfer

LMCache moves KV-cache blocks between a serving engine's paged per-layer buffers and its own
contiguous, token-major chunk objects. This task is about that transfer kernel
(`multi_layer_block_kv_transfer`). The agent gets a working, upstream-faithful CUDA port of the
kernel and must make it faster. Every byte it moves must stay bit-exact. This is a seed task:
`candidates/kernel_v001/submission.cu` is the correct baseline, and the agent optimizes from it.

## The task

The agent works in `/app/workdir/kv_transfer_lab` inside the Work container. Each candidate is
a single CUDA C++ translation unit at `candidates/kernel_vNNN/submission.cu` (three-digit
version directories). `FINAL_VERSION` holds the integer of the selected candidate, and the
agent keeps short measurement notes in `notes.md`. These three paths are submitted.
`transfer_abi.h`, `tools/` and the build output are excluded from the submission.

The candidate must export exactly this function, written against the read-only header
`transfer_abi.h`:

```c
extern "C" void lmcache_paged_kv_block_transfer(const TransferParams* params,
                                                cudaStream_t stream);
```

One call moves whole 16-token blocks for 1–4 chunk objects. It covers both directions
(`store`: engine pages to chunks, `retrieve`: chunks to engine pages) and takes an arbitrary
unique block-id map and an optional skipped prefix. Two HND engine layouts are scored: `split`
(`[2, NB, NH, BS, HS]`) and `fused` (`[NB, NH, BS, 2·HS]`), with head size 64 or 128, 1–128
layers, 1–32 KV heads and up to 65,536 blocks. Elements are 2 bytes (bf16 storage). In the
host, trace and overlap families, the chunk objects live in mapped pinned host memory; the kernel
reads or writes them over PCIe, or copies them with stream-ordered `cudaMemcpy*Async`. The
Judge provides a 256 MiB device scratch arena and re-poisons it before every call.

The agent is free to change launch geometry, tiling, vector width, cache hints, the order of
zero-copy accesses, and how it uses the scratch arena. The Judge enforces these rules (source
admission plus an audit of undefined symbols in the compiled library):

- Allowed includes: `transfer_abi.h`, `cuda_runtime.h`, `stdint.h`/`cstdint`,
  `stddef.h`/`cstddef`, `type_traits`.
- Allowed CUDA runtime calls: kernel launches (including cooperative launches),
  `cudaMemcpy*Async`/`cudaMemset*Async` on the given stream, error queries, occupancy and
  attribute introspection, and `cudaFuncSetAttribute`. Inline PTX is allowed only in device
  code.
- Forbidden: any allocation (including `__managed__`), private streams or events,
  synchronization calls, CUDA graphs and stream capture, the driver API, synchronous
  `cudaMemcpy`/`cudaMemset`, peer access, host callbacks, threads, files, processes, loaders,
  environment access, keeping results across calls, and reading `params` after the function
  returns.
- The compile command is fixed:
  `nvcc -O3 -std=c++17 -arch=<native sm of the GPU> --shared --cudart=shared`, with a 300 s
  limit. The agent does not control the flags.

The Work container reaches the internet only through a restricted proxy sidecar: public hosts
on ports 80/443. The agent must not copy external implementations or solution material. A
required LLM integrity review (`llm_judge/POLICY.md`) checks submissions for shortcuts such as
copying only sampled bytes, replaying earlier payloads, detecting the measurement phase, or
tampering with the harness.

Inside Work, the agent runs the disclosed public suite with `./tools/run_one.sh --version N` and
`./tools/run_public_tests.sh [--quick] --candidate-version N`. The launcher's default budget is
12 hours (`TASK_TIMEOUT=43200`).

## Evaluation

**Workloads.** Every case models one GPU of a serving engine, following the
LMCache paper (arXiv 2510.09665):

- **Shapes** are per-GPU slices of real models under tensor parallelism (layers, KV heads per
  GPU, head size, from each model's `config.json`), at every TP in {1, 2, 4, 8} whose weights
  fit an 80 GB GPU with room for a pool. The Judge has 30 such shapes: the paper's evaluation
  models (tier `paper`, §8.1: Llama-3.1-8B, Llama-3.1-70B / Qwen2.5-72B, Qwen2.5-Coder-32B,
  Qwen3-Coder-480B) and other open models (tier `other`: Qwen2.5-14B, Qwen3-235B,
  Mixtral-8x22B, Mistral-Large-2411, Yi-1.5-34B, gpt-oss-120b, Llama-3.2-1B; gpt-oss's
  sliding and full attention layers form two KV groups of half the layers). The public table
  has 19 shapes of other models; no (layout, NL, NH, HS) is shared. Every (shape, layout) is a
  stratum: 60 in the Judge, 38 in public.
- **Pools** are sized by vLLM's rule (0.92 of an 80 GB device minus weights per GPU and a
  2.5 GB reserve), capped at 40 GB on the Judge and 12 GB in public.
- **Block maps** come from replaying the Mooncake FAST'25 traces through a simplified
  vLLM-style block manager (LRU free queue, prefix-hash reuse, reverse free order) and an
  LMCache store: the GPU-resident prefix is skipped, stored chunks are retrieved, new chunks
  (including a partial tail chunk) are stored, and completed decode chunks are stored
  (paper §5.1). The replay is a model of an engine, not a recorded vLLM run. The Judge replays
  the conversation and tool-agent traces; the public table replays the synthetic trace.
- **Families:** `device-large` / `device-small` and `host-large` / `host-small` (a call is
  small below 16 MiB: a tail chunk, a decode-time chunk, one chunk of a deep tensor-parallel
  split), `host-duplex` (host stores and retrieves at once on two streams over disjoint
  blocks, each stream with its own scratch arena; paper §5.1: parallel store and load over
  full-duplex PCIe), `trace` (a window of mixed calls of one deployment against one pool),
  and `overlap` (host calls while a trusted foreground — an 8192³ bf16 GEMM or a 1 GiB device
  copy — runs on a second stream; paper §5 and §5.2: transfers run alongside inference).

**Correctness gate (bit-exact, zero tolerance).** Each case allocates one engine pool shared by
all its calls. Before every pass the pool is refilled layer by layer from a seeded generator,
retrieve sources get fresh payloads, store destinations are poisoned, and the whole block map
is rotated by one offset (mod NB). After every pass the Judge regenerates each pool layer,
replays the case's calls in order, and compares every stored chunk, every retrieved block,
every skipped block and every other pool byte bit for bit; it also checks that retrieve
sources, block-id tables and the pointer array are unchanged. No pool is cloned, so pools of
tens of gigabytes fit. Twenty-five disclosed edge fixtures (support-domain limits: 128
layers, 32 heads, 65,536 blocks, head size 64, tail chunks, objects larger than the scratch,
skipped prefixes, reversed and pool-spanning maps, repeated blocks, duplex calls) run before
every suite;
they gate validity but never count toward the score.

**Timing.** A case is a list of back-to-back calls ("waves") on one stream. Each case runs one
untimed, verified pre-call pass, then three timed, verified trials. A trial's time is
`max(CUDA event time, synchronized host wall time)` over the whole wave list; the case uses
the median. Overlap cases also time the transfer alone and the foreground alone; the
foreground runs 3× the transfer's own time (doubled until it covers the transfer), and the
loss is the foreground's extra time divided by the transfer time, clamped to [0, 0.95].

**Score** (GB/s, higher is better, no cap and no baseline ratio):

```text
case GB/s     = logical payload bytes / median trial seconds (duplex: both directions)
overlap case  = case GB/s × (1 − median foreground loss)
stratum       = geometric mean of the stratum's cases
family score  = mean over tiers (paper, other) of the mean log stratum score, exponentiated
device        = geometric mean of device-large and device-small
host          = geometric mean of host-large, host-small and host-duplex
score_gbps    = device^0.35 × host^0.25 × trace^0.10 × overlap^0.30
```

Every shape stratum counts once within its tier and each tier counts once, so neither the
number of cases per shape nor the number of non-paper shapes moves the score.

Logical bytes do not depend on the implementation. Staging through scratch moves more physical
bytes but only earns credit for the payload. The trusted runtime probes a physical bandwidth
envelope on the same GPU and host before it loads the candidate: a contiguous device-to-device
copy, pinned H2D/D2H copies, an SM zero-copy stream, and concurrent H2D plus D2H copies (the
duplex envelope). A case that implies more than 1.10 × its envelope invalidates the run.

**Valid** means the verifier completed, every gate passed, and the result records, schema,
suite identities and aggregate are consistent. An invalid run scores 0.

**Suites.** The public suite inside Work has 155 scored cases (every family, both layouts,
both head sizes, 38 strata) plus the 25 edge fixtures; quick runs one case per family. Every
interactive submission (`sforge-submit --details`) and the final evaluation run the same
Judge-only final suite: 843 scored cases (device-large 156, device-small 143, host-large 104,
host-small 142, host-duplex 104, trace 90, overlap 104; 60 strata, both tiers in every family)
plus the 25 edge gates. The count is sized so a family score moves by about 1–2 % when the
cases are redrawn, and the suite runs in about 25 minutes on one H100. Interactive
submissions return the score, the four group scores, the seven family scores and the gate
states, never which cases a family holds or per-case results; there is a cooldown between
them (`EVAL_INTERVAL`, default 300 s). Scheduled auto-evaluation is off. When Work ends, the
archive captured at the time limit is judged on its own, using the candidate that
`FINAL_VERSION` selects; the run's result is the best valid score across submissions and this
final evaluation. The Judge timeout is 4800 s.

**Regenerating the suites.** The suites are generated deterministically from the three
Mooncake trace files by `scripts/generate_final_manifest.py` (usage in the script); a redraw
is a new suite identity.

## Hardware

By default a run uses **two GPUs**: Work runs on GPU 0 and the Judge on GPU 1 (`--work-gpu`,
`--judge-gpu`), so the Judge measures on its own GPU while Work keeps running. With
`--shared-gpu` Work and Judge share one GPU one at a time, and the Work container is paused
while the Judge runs. The execution profile requires compute capability 8.0 or
newer and at least 64 GiB (68,719,476,736 bytes) of total and free device memory: the largest
Judge pool is 40 GB, and a 4 GiB pinned host arena holds the host-side chunks. The kernel is
compiled for the GPU's native architecture.

| hardware type | status |
|---|---|
| `h100-94gb-400w` (H100 NVL 94 GB) | qualified |
| `h100-80gb-700w` (H100 SXM 80 GB) | qualified |

## Running

1. **Assets.** `tasks/latest/lmcache-paged-kv-block-transfer-h100-v1-seed/assets/fetch.sh`
   (into `INFRATHON_ASSETS_ROOT`). The task needs no models or datasets. The only asset is the
   Judge-only final-case table (`judge/final_cases.json`, 843 cases, about 10 MB), downloaded
   from the dataset's `judge/` part. It is not in the repository; the build reads it from
   `INFRATHON_ASSETS_ROOT`.
2. **Images.** Pull the published Work and public Judge images, then build the Judge holdout
   layer locally:

   ```bash
   INFRATHON_PULL=1 tasks/latest/lmcache-paged-kv-block-transfer-h100-v1-seed/scripts/build_images.sh
   ```

   To build everything from source, omit `INFRATHON_PULL=1`. The Work image is based on
   `pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel`, which provides the full CUDA 12.9 toolkit.
3. **Run.**

   ```bash
   launch/lmcache-kv-transfer/run.sh --model MODEL [--work-gpu N --judge-gpu M] \
       [--agent codex|claude-code] [--dry-run] [--tmux]
   ```

   Defaults: Work on GPU 0, Judge on GPU 1, 12 hours, effort `max`, Judge port 18496;
   `--work-gpu`, `--judge-gpu`, `--hours`, `--effort` and `--judge-port` change them.
   `--shared-gpu` runs Work and Judge on one GPU, chosen with `--gpu N` (default 0) or
   `--device-uuid UUID`; either flag alone also selects shared mode. `--gpu` takes a physical
   index and the launcher looks up its UUID; `--device-uuid` names one whole GPU by UUID
   (MIG identities are not accepted). `--allocation-key` and `--lease-id` are optional
   scheduler IDs and apply only to shared mode. `--dry-run` checks the model, agent runtime,
   images, integrity-review settings, GPU and port, prints the plan, and exits without starting
   anything.
   - Agent credentials: Codex reads `--auth auth.json`. Claude Code reads `--claude-token FILE`,
     a 0600 file holding a `claude setup-token` token. Any model ID that an endpoint set by
     `MODEL_API_BASE_URL` serves also works.
   - Shared settings come from `--config FILE` (see `launch/mlsysforge.env.example`). This
     includes the required LLM integrity review (`SFORGE_LLM_JUDGE_*`).
   - Evaluate a saved archive only:
     `tasks/latest/lmcache-paged-kv-block-transfer-h100-v1-seed/scripts/eval_archive.sh ARCHIVE.tar.gz [RUN_ID]`
     (GPU chosen with `PHYSICAL_GPU`, default 0).
   - Replay the saved candidates of a finished run: `launch/lmcache-kv-transfer/replay.sh
     --source-run-id RUN_ID --measurement-set NAME [--candidate vN | --from-v N --to-v N]
     [--dry-run]`.

## Files

| path | contents |
|---|---|
| `images/work/` | Work image: `AGENT_TASK.md`, the lab (`workdir/kv_transfer_lab`: seed candidate, `README_AGENT.md`, `tools/`), public runner, public case table and submit hook (`benchmark/`) |
| `images/judge/` | public Judge image (verifier, candidate worker, source admission, public case table, protected hashes); `Dockerfile.private` adds the Judge-only cases |
| `runtime/` | the Judge's trusted runtime (`kv_transfer_runtime.py`) and the frozen ABI header (`transfer_abi.h`); the Work runtime in `images/work/benchmark/` is generated from it by `scripts/make_public_runtime.py` |
| `judge/run_judge.py` | Judge adapter that turns verifier output into the structured result |
| `tasks/*.json` | task spec: Work/Judge commands, resources, submit paths, suites, metric |
| `portable/` | resource, suite, oracle, selection, software-ABI and calibration manifests |
| `llm_judge/POLICY.md` | integrity-review policy |
| `oracle/` | reference candidates for qualification (not a build input): `kernel_v001` (selected) = the seed, LMCache's upstream zero-copy path; `kernel_v002` = LMCache's upstream staged path; see `oracle/notes.md` |
| `assets/` | asset manifest and `fetch.sh` |
| `scripts/` | image build, run entrypoints, archive evaluation, replay, layout checks, the suite generator (`generate_final_manifest.py`) and the public-runtime generator |
| `tests/` | contract, scoring, admission and portability tests |

## Upstream and license

- [LMCache/LMCache](https://github.com/LMCache/LMCache) at commit
  `f9addd2e4e074f21f26cbda84c3abd932d32ef33`, Apache-2.0. The seed kernel ports
  `multi_layer_block_kv_transfer` from `csrc/cuda/mp_mem_kernels.cu`. Only the two scored layout
  families are kept, upstream's launch geometry is unchanged, and `TORCH_CHECK` validation and
  the stream lookup are replaced by the ABI. The seed file carries an SPDX line and a
  derivation notice.
- Paper: "LMCache: An Efficient KV Cache Layer for Enterprise-Scale LLM Inference",
  arXiv 2510.09665.
- Workload sources: model `config.json` files (shapes only) and the Mooncake FAST'25 traces
  ([kvcache-ai/Mooncake](https://github.com/kvcache-ai/Mooncake), `FAST25-release/traces`,
  Apache-2.0 repository) — only block-id tables derived from them ship, in the case tables.
- Everything else was written for this task: the ABI header, runtime, verifier, Judge, case
  tables and scripts. The task ships no model weights, tokenizers or datasets.

The task-original files are under the repository's root `LICENSE` (PolyForm Noncommercial 1.0.0).
