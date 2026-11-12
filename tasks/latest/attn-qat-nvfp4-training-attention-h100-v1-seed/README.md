# Attn-QAT NVFP4 training attention (forward + backward)

The workload is one training step of a 4-bit quantization-aware-training (Attn-QAT) attention
layer, as used for QAT fine-tuning of video diffusion models (Wan2.1, LTX-2, Kandinsky 5, CogVideoX,
HunyuanVideo) and of language models (the Attn-QAT paper fine-tunes Qwen3-14B and Llama 3.1-70B).
The forward fake-quantizes `q`, `k`, `v` and the attention probabilities to NVFP4 (groups of 16
values, FP8 e4m3 block scales, FP4 e2m1 values). The backward recomputes those probabilities in
the same low precision. Both calls are timed. This is a **seed** task: the agent starts from a
correct, self-contained Triton port of the upstream FastVideo SM90 kernels and has to make it
faster without changing what it computes.

## The task

The agent works in `/app/workdir/attn_qat_lab` inside the Work container. The full specification
is `AGENT_TASK.md` (in this repository at `images/work/AGENT_TASK.md`). It defines the
fake-quantization function, the forward and backward math, the ABI, the gates and the score.

- **What is submitted:** `candidates/kernel_vNNN/submission.py` (versioned candidates), `FINAL_VERSION`
  (which candidate is evaluated) and `notes.md`. Nothing else is submitted. The seed is
  `candidates/kernel_v001`.
- **ABI:** two entry points on contiguous bf16 tensors, `q, o, do, dq` as `[B, H, S_q, D]` and
  `k, v, dk, dv` as `[B, H_kv, S_kv, D]`, with `D` in {64, 128, 256}, `H % H_kv == 0` (grouped-query
  attention; `dk`, `dv` sum over each group) and `softmax_scale = D ** -0.5`. `causal=True` masks
  the keys after each query row (`S_q == S_kv`); `causal=False` allows `S_q != S_kv`:
  - `attn_qat_fwd(q, k, v, softmax_scale, causal, o)` overwrites `o` and returns a `saved` tuple
    of tensors;
  - `attn_qat_bwd(q, k, v, softmax_scale, causal, o, saved, do, dq, dk, dv)` overwrites `dq`,
    `dk`, `dv`.
- **Contract the Judge enforces:**
  - all outputs are NaN-poisoned before each call and must be written in full, with finite
    values, on the current stream;
  - `q, k, v, do` stay byte-identical, and `o` is byte-identical between the end of the
    forward and the end of the backward;
  - new storage reachable from `saved` totals at most
    `bytes(q) + bytes(k) + bytes(v) + bytes(o) + 4 * B * H * S_q`, and `saved` holds no
    callables;
  - no autograd: the harness drives both calls itself.
- **Admission policy (static, before import):**
  - Allowed imports are `math`, `typing`, `torch`, `triton`, `triton.language` and
    `triton.tools.tensor_descriptor.TensorDescriptor`.
  - Every tensor data path must go through a reachable `@triton.jit` kernel. Host Torch is
    limited to metadata and `empty`/`zeros`/`full`-style allocations.
  - These are rejected: `@triton.autotune`, `@triton.heuristics`, `tl.inline_asm_elementwise`,
    `triton.language.target_info`, `torch.autograd`, host tensor math or indexing,
    `torch.compile`, private streams, and filesystem, process and network access.
  - `triton.set_allocator` is admitted only in one fixed form, described in `AGENT_TASK.md`
    section 7.
- **Allowed in kernels:** fusion, reordering, other tile shapes, TMA descriptors, warp
  specialization, persistent schedules and reduced-precision `tl.dot` are all allowed, as long
  as the accuracy gates hold. Changing what is quantized, when it is quantized, or against which
  statistic is not allowed.
- **Integrity review:** before the GPU Judge runs, an LLM reviewer checks each submission
  against `llm_judge/POLICY.md`. It rejects submissions that substitute a different
  quantization recipe, fabricate or defer outputs, cache results across calls, or special-case
  fixtures and seeds.
- **Environment:** the Work and Judge images are built on
  `pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel` (torch 2.8.0, Triton 3.4.0). FastVideo,
  FlashAttention, SageAttention, FlashInfer and xFormers are not installed. The Work container
  reaches the internet only through a proxy that allows public hosts. The agent is told not to
  copy upstream implementations into the submission.
- **Time budget:** the launcher defaults to 43200 s (12 h) of agent time. Use `--hours` or
  `--timeout` to change it.

Inside the container the agent has `tools/run_one.sh` (edge fixtures plus one public case) and
`tools/run_public_tests.sh` (`--quick`, or the full public controller over all 29 cases; `--case` and `--perf-only` narrow it).

## Evaluation

- **Work per case:** `base = B * H * pairs * D`, where `pairs = S_q * S_kv`, or `S * (S + 1) / 2`
  under a causal mask. The effective work is `26 * base`: the forward counts `8 * base` and is
  weighted twice (training runs with full activation checkpointing), and the backward counts
  `10 * base`.
- **Case score:** each trial times one forward and one backward. Each call is timed as
  `max(host wall time, CUDA event time)`. The measured time is `2 * t_fwd + t_bwd`, and case
  TFLOPS is `26 * base / median(time)` over 3 fresh-input trials, after 1 warm-up. A real layer
  is one timed step on its captured tensors.
- **Primary score:** effective TFLOPS (maximize, uncapped): the geometric mean of two domain
  scores, video and LLM, which weigh equally. A domain's score is the geometric mean of its
  families; a family's score is the geometric mean of its `recipe` and `sweep` tiers, each the
  geometric mean of its cases, so the many extrapolated cases cannot outvote the few recipe cases.
  Real families have a single tier.

  | domain | family | D | cases | source |
  |---|---|---:|---:|---|
  | video | wan13b | 128 | 3 recipe + 15 sweep | Wan2.1 1.3B stage-1 and stage-2 recipes |
  | video | wan14b | 128 | 1 recipe + 22 sweep | the paper's Wan2.1 14B 720p fine-tune; 480p and other splits extrapolated |
  | video | ltx2 | 128 | 1 recipe + 11 sweep | LTX-2 recipe; other lengths and H = 16 extrapolated |
  | video | kandinsky5 | 64 | 2 recipe + 16 sweep | Kandinsky5 recipe, self and cross attention |
  | video | cogvideox | 64 | 2 recipe + 18 sweep | CogVideoX-2B/5B at 49 frames (226 text + 1350 tokens per latent frame); other lengths and sequence-parallel splits extrapolated |
  | video | hunyuan | 128 | 15 sweep | HunyuanVideo 544x960 (256 text + 2040 tokens per latent frame) under sequence parallelism |
  | video | real_video | 64, 128 | 42 | captured training steps: 10 blocks x 3 noise levels of Wan2.1 1.3B, 2 blocks of 14B, 10 blocks of CogVideoX-2B |
  | LLM | llm_pretrain | 128 | 1 recipe + 32 sweep | the paper's Qwen3-14B at 8192 tokens; Qwen3, Qwen3-MoE and Llama 3.1 head layouts and tensor-parallel splits at 1k-8k tokens in 16k-token packed batches |
  | LLM | llm_long | 128 | 14 sweep | single sequences of 16k-64k tokens |
  | LLM | llm_sft | 128 | 36 sweep | padded fine-tuning batches: drawn lengths (multiples of 64, 1k-12k) and batch sizes |
  | LLM | llm_d256 | 256 | 20 sweep | full-attention layers of Qwen3.6-27B and Qwen3.5-4B and their splits |
  | LLM | real_llm | 128, 256 | 36 | captured training steps on real text: 14 layers x 2 texts of Qwen3-14B, 8 full-attention layers of Qwen3.5-4B |

  The exact geometries are in the Judge-only manifest; `scripts/generate_final_manifest.py`
  draws the seeds, the fine-tuning lengths and batch sizes, and the hidden half of each family's
  sweep cases that runs the golden.
- **Inputs:** video lengths come from each family's token lattice; LLM lengths are padded to a
  multiple of 64. Synthetic inputs are drawn from a parametric family calibrated on real
  Wan2.1-T2V-1.3B activations and training-step gradients, with fresh values in every trial;
  grouped-query cases share each group's keys and values. The real families run captured layers
  from `fixtures/recipe/real_layers/`.
- **Accuracy gates:** the Judge compares `o`, `dq`, `dk`, `dv` against a chunked FP32 golden by
  global relative L2 on one hidden trial of each gated case: every recipe and real case, and a
  frozen hidden half of each family's sweep cases. Scored cases use flat limits (`o` 1e-2,
  `dq` 8e-2, `dk` 3e-2, `dv` 1.2e-1); real layers use the limits in `AGENT_TASK.md` section 5,
  2.5x the largest error that correct reference kernels reach on those layers.
  Every trial of every case still checks full, finite writes, unchanged inputs and the
  saved-bundle cap.
- **Edge fixtures:** fifteen disclosed hard-gate fixtures each carry their own, often tighter,
  limits. They cover sub-tile and unaligned lengths, cross attention, zero groups, e2m1 ties,
  a forced sink, tiny gradients, all three head dimensions, causal grouped-query attention (one
  with a forced sink and a single key/value head), and one real Wan2.1-1.3B capture (e7). They
  never enter the score.
- **Validity:** a submission is valid only if:
  - it passes admission and the integrity review;
  - the worker completes;
  - every hard gate (writes, immutability, saved-bundle cap, accuracy) passes on every case;
  - the result records are well-formed and match the frozen suite identity.

  An invalid submission scores 0.
- **Interactive and final evaluation:**
  - Each `sforge-submit --details` from the agent runs the complete frozen suite: 287 scored
    cases plus the 15 fixtures, about 20 minutes on H100 SXM (measured with the seed and the oracle).
  - After the run ends, the sealed final archive is evaluated once more, in a fresh Judge, on
    the same suite with the same gates and timing.
  - Scheduled automatic evaluation is off.
  - The agent sees the score and the gate outcomes, but not domain or family scores, per-case
    results, or the hidden seeds, lengths or tensors.
- **Public cases:** the public controller in Work has its own 29 cases (27 synthetic, one real
  Wan2.2-TI2V-5B layer and one real Qwen3-1.7B layer) and fifteen edge cases in
  `images/work/benchmark/attn_qat_runtime.py`. They cover the same regimes as the scored cases but
  share no `(H, H_kv, D)` geometry or length with them, and the Judge's lengths are not in the
  Work image.
- **Judge timeout:** 3600 s for the candidate worker, 4200 s for a standalone final evaluation.

## Hardware

- **GPUs:** two GPUs per run by default. Work runs on GPU 0 and the Judge on GPU 1 (`--work-gpu`,
  `--judge-gpu`), so the Judge measures on its own GPU while Work keeps running. `--shared-gpu`
  puts both on one GPU instead, and the host pauses the Work container while the Judge runs.
- **Execution profile:** one CUDA GPU with compute capability >= 8.9, FP8 e4m3 and bf16
  support, and at least 40 GiB of memory on the Judge (a 720p case peaks near 26 GiB before the
  candidate's own scratch). The public tests need 16 GiB.

| hardware type | support |
|---|---|
| `h100-94gb-400w` (H100 NVL 94 GB) | qualified |
| `h100-80gb-700w` (H100 SXM 80 GB) | qualified |

TFLOPS scores can only be compared within one hardware type.

## Running

1. **Assets.** Run `tasks/latest/attn-qat-nvfp4-training-attention-h100-v1-seed/assets/fetch.sh`.
   It downloads into `INFRATHON_ASSETS_ROOT` (default `~/.cache/infrathon/assets`) from the
   Infrathon dataset:
   - the e7 edge-fixture capture and its manifest (public; built into the Judge image);
   - the Judge-only final-suite case list;
   - the Judge-only real layers (`judge/real_layers/`, 78 layers and two capture manifests, about
     24 GB, mounted read-only into the Judge at run time by `scripts/sforge_judge_entry.py`, never
     in an image) and the two public layers (`public/public_layer/`, part of the Work image).
     `fixtures/recipe/real_layers/build_real_layers.sh` re-captures them; `REAL_LAYERS_DIR` and
     `PUBLIC_LAYER_DIR` override the locations.

   No models are downloaded at run time. With `INFRATHON_ASSETS_ROOT` set, `build_images.sh` reads
   both from the asset root, so the build needs the downloaded capture. `fixtures/recipe/` is the
   recipe it was captured with (re-capturing needs `FIXTURE_GPU`, the `hf` CLI, and a download of
   `Wan-AI/Wan2.1-T2V-1.3B` at a pinned revision).
2. **Images.** Build them in one of two ways:
   - Pull the published Work and public Judge images for this host's hardware type, then build
     the Judge holdout layer locally:
     ```bash
     INFRATHON_PULL=1 tasks/latest/attn-qat-nvfp4-training-attention-h100-v1-seed/scripts/build_images.sh
     ```
   - Build everything from source by running the same script without `INFRATHON_PULL`.
3. **Run.**
   ```bash
   launch/attn-qat-train/run.sh --model <model> --auth /path/to/auth.json \
     [--work-gpu N --judge-gpu M] [--tmux]
   ```
   - Defaults: Work on GPU 0, Judge on GPU 1, 12 hours, effort `max`, Judge port 18601.
     `--work-gpu`, `--judge-gpu`, `--hours`, `--effort` and `--judge-port` change them.
     `--shared-gpu` runs Work and Judge on one GPU, chosen with `--gpu N` (default 0) or
     `--device-uuid UUID`; either flag alone also selects shared mode. `--allocation-key` and
     `--lease-id` are optional scheduler IDs for shared mode, derived when they are omitted.
   - `--agent claude-code` with `--claude-token FILE` runs Claude Code instead of Codex.
   - `--model` takes any Codex or Claude model ID, or any model that `MODEL_API_BASE_URL`
     serves.
   - Other options: `--run-id`, `--cpuset`, and `--config FILE` for a launch config (see
     `launch/mlsysforge.env.example`).
   - `--dry-run` checks the model, runtime, images, integrity review, GPU and port, prints the
     plan and exits. It takes no lock and starts no container.
   - **Eval only:** `PHYSICAL_GPU=<index> tasks/latest/<task>/scripts/eval_archive.sh ARCHIVE.tar.gz [RUN_ID]`
     evaluates a submission archive in the Judge.
   - **Replay:** `launch/attn-qat-train/replay.sh --origin-run RUN_ID --measurement-set NAME`
     re-measures the saved candidate versions of a finished run. Add `--dry-run` to preview
     without using a GPU.

**Prerequisites the launcher checks:**

- the SForge harness;
- a version store (`SFORGE_VERSION_STORE_ROOT`, with this task registered);
- an agent runtime directory (`SFORGE_AGENT_RUNTIME_DIR`, the Codex or Claude Code CLI built by
  `launch/build-agent-runtime.sh`);
- model credentials: a Codex `auth.json` (mode 0600), a Claude Code token file, or an
  OpenAI-compatible endpoint through `MODEL_API_BASE_URL`/`MODEL_API_KEY_FILE`;
- if the host reaches the internet through a proxy, a proxy address that containers can reach.

The LLM integrity review is configured with the `SFORGE_LLM_JUDGE_*` keys. It runs in `enforce`
mode (required), and it falls back to the agent runtime and credentials when its own are not set.

## Files

| path | contents |
|---|---|
| `images/work/` | Work image: `AGENT_TASK.md`, the seed lab (`workdir/attn_qat_lab`), public test tools, the submit hook |
| `images/judge/` | public Judge image (verifier, candidate worker, source-admission policy, protected hashes) and `Dockerfile.private` for the holdout layer |
| `runtime/attn_qat_runtime.py` | shared input generator, FP32 golden, gate limits and timing, used by both images |
| `judge/` | Judge adapter (`run_judge.py`); the Judge-only final-suite case list is not in the repository: it ships in the dataset's `judge/` part, and the build reads it from `INFRATHON_ASSETS_ROOT` |
| `fixtures/recipe/` | frozen recipe and capture script for the e7 Wan2.1 edge fixture; `real_layers/` holds the recipes and capture of the scored real layers and the public layer |
| `oracle/` | host-only reference: `kernel_v001` is the seed control (byte-identical to the Work seed); `FINAL_VERSION` selects `kernel_v002`, the best upstream FastVideo configuration on H100 (93.7 TFLOPS vs 74.2 for the seed on the full scored suite, H100 SXM; see `oracle/notes.md`) |
| `llm_judge/POLICY.md` | integrity-review policy |
| `tasks/` | SForge task spec (`*.json`) and `BENCHMARK.yaml` |
| `portable/` | portable contract: suite, oracle, calibration, software-ABI and resource manifests |
| `assets/` | `manifest.json` (assets with SHA-256) and `fetch.sh` |
| `scripts/` | image build, run, eval, replay, smoke-test and validation scripts |
| `tests/` | CPU/static contract and security tests |

## Upstream and license

- **Kernel:** the seed is a self-contained port of the SM90 route of `attn_qat_train.py` in
  [hao-ai-lab/FastVideo](https://github.com/hao-ai-lab/FastVideo) (Apache-2.0), pinned at
  commit `7f8316423355a2f07db533a5a339ac16e0da31ff`, together with its `quant_utils.py`
  prologues and `nvfp4_utils.py` e2m1/e4m3 encode and decode. Blackwell inline-PTX branches and
  autotune wrappers are removed, and upstream's causal handling is replaced by the task's own
  causal mask. The forward quantizes probabilities against the final row maximum (upstream's
  exact-M path) instead of a running maximum.
- **Algorithm:** "Attn-QAT: 4-Bit Attention With Quantization-Aware Training", arXiv 2603.00040.
- **Real layers:** captures from `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`, `Wan-AI/Wan2.1-T2V-14B-Diffusers`,
  `zai-org/CogVideoX-2b`, `Qwen/Qwen3-14B` and `Qwen/Qwen3.5-4B` (scored), and
  `Wan-AI/Wan2.2-TI2V-5B-Diffusers` and `Qwen/Qwen3-1.7B` (public), with the LLM layers run on
  texts from the `Xnhyacinth/LongBench` dataset, at the revisions in `fixtures/recipe/real_layers/`.
- **e7 fixture:** a capture from `Wan-AI/Wan2.1-T2V-1.3B` (revision
  `37ec512624d61f7aa208f7ea8140a131f93afc9a`) run with
  [Wan-Video/Wan2.1](https://github.com/Wan-Video/Wan2.1) at commit
  `9737cba9c1c3c4d04b33fcad41c111989865d315`.

- **Licenses:** `Wan-AI/Wan2.1-T2V-1.3B` is Apache-2.0 (per its model card); the e7 capture
  in the Infrathon dataset is derived from it. The Wan2.1 and Wan2.2 real-layer captures are
  Apache-2.0 (as recorded in `assets/manifest.json`). For `zai-org/CogVideoX-2b`,
  `Qwen/Qwen3-14B`, `Qwen/Qwen3.5-4B`, `Qwen/Qwen3-1.7B` and `Xnhyacinth/LongBench`, see the
  model's or dataset's page. For the Wan2.1 code, see the upstream repository for its license.
- The task's own code is under the repository's root `LICENSE` (PolyForm Noncommercial 1.0.0).
