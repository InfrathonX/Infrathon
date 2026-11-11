# SageAttention-style Quantized Attention, BF16 (seed)

An agent optimizes a quantized attention forward pass in self-contained Triton. The kernel takes
contiguous BF16 Q/K/V in `[B, L, H, D]` layout and must do everything inside the timed call: K
smoothing, quantization, softmax and P·V. Accuracy is judged against SageAttention-1 numerics.
Speed and accuracy are both measured on the attention calls of real deployments: video and image
diffusion transformers at their resolutions, clip lengths and sequence-parallel head splits,
long-context LLM prefill per tensor-parallel rank up to 262,144 tokens, and 139 attention layers
captured from seven real models.
This is a **seed** task: the starting kernel is a working SageAttention-1-style Triton port (K mean
smoothing, per-block INT8 Q/K, exp2 online softmax, FP16 P·V) to improve.

## The task

The agent works in `/app/workdir/quant_attention_lab` inside the Work container. It writes
versions to `candidates/kernel_vNNN/submission.py` (three-digit directories), puts the selected
version number in `FINAL_VERSION`, and keeps notes in `notes.md`. Only these three paths are
submitted. The full brief is `images/work/AGENT_TASK.md`.

The selected `submission.py` must define this function. It runs on the current CUDA stream,
overwrites every element of `out`, returns `None` and leaves all inputs byte-identical:

```python
def quantized_attention_bf16(q, k, v, is_causal, softmax_scale, out) -> None: ...
```

- **Tensors.** `q` and `out` are contiguous `[B, L, H_q, D]` BF16; `k` and `v` are contiguous
  `[B, L, H_kv, D]` BF16. `D` is 64 or 128 and `H_q % H_kv == 0`; query head `h` uses KV head
  `h // (H_q // H_kv)`. `softmax_scale` is `1/sqrt(D)`. Causal cases use the ordinary lower
  triangle. `B <= 4`, `H_q <= 64`, `L <= 262,144`, and one call's inputs are at most 6 GiB
  (`2*B*L*(H_q + 2*H_kv)*D` bytes, `MAX_CASE_INPUT_BYTES`).
- **Semantics.** `softmax(softmax_scale * q @ k.T) @ v`. Approximation and sparsity are allowed
  as long as every accuracy gate passes on hidden inputs.
- **Allowed.** One or more reachable `@triton.jit` kernels for every tensor data path, with
  ordinary Triton inside them (`tl.dot`, reductions, atomics, persistent or split-K schedules,
  autotuning) and FP16, INT8, FP8 (e4m3, compute capability 8.9 and newer), BF16 and FP32
  compute. Imports are limited to `math`, `typing`, `torch`, `triton` and `triton.language`.
  Host code is a small, statically checked subset:
  - assignment, `assert`, bare calls, `if`/`else`, `pass` and `return` only;
  - Torch only for tensor metadata and `empty`/`zeros`/`full` (and their `_like`/`_strided`
    forms) scratch in INT8, FP8 e4m3, FP16, BF16 or FP32.

  `AGENT_TASK.md` lists the exact rules.
- **Not allowed** (rejected by source admission, the Judge or the integrity review):
  - host tensor math, including `k.mean(...)`, attention, GEMM, softmax, copies or input
    mutation;
  - host loops, comprehensions, lambdas, `try`/`with`, dynamic imports and extensions;
  - private CUDA streams, file, process or network access, dunder introspection, module
    mutation, unsafe autotune hooks and callable-grid tricks;
  - copying an upstream implementation into the submission.

  The Work image contains PyTorch 2.8.0 and Triton 3.4.0, the seed, the public runner and two
  real-model layers of its own. It does not contain SageAttention, FlashAttention, FlashMLA, FlashInfer,
  xformers, DeepGEMM or any Judge code.
- **Public tools.** `./tools/run_one.sh --version N` and
  `./tools/run_public_tests.sh [--quick] [--case ID] [--perf-only] --candidate-version N` run the
  public suite: 19 synthetic cases over the same four kinds of workload as the Judge (up to
  190,001 tokens), eight edge cases and two real-model layers of models the Judge does not use
  (Wan2.2-TI2V-5B, Qwen3-8B). Quick runs three cases, one of them timing-only. Public geometries,
  lengths, models and input distributions all differ from the Judge's; full takes about two
  minutes.
- **Time budget.** The launcher's default is 12 hours (`--hours`/`--timeout`).

## Evaluation

Every Judge run performs the same complete evaluation of the frozen final suite:
hardware check, 8 edge cases, then 774 scored cases (635 synthetic, 139 real layers). Any failed
gate makes the submission invalid.

- **Accuracy gates.** Output is compared with an FP32 golden (TF32 off, 1024 query rows at a
  time) and with a trusted FP32 emulation of SageAttention-1 numerics: per-128-token INT8 Q,
  per-64-token INT8 K after subtracting the per-channel token mean, symmetric absmax scales,
  half-away-from-zero rounding, FP32 softmax and P·V. Limits are relative to that emulation
  ("reference"), with absolute floors:
  - every output element finite and written (`out` is NaN-filled first), all inputs unchanged;
  - global relative L1 `<= max(3 * reference, 0.025)` and cosine similarity `>= 0.998`;
  - worst-head relative L1 `<= max(3 * reference, 0.04)`;
  - row relative-L2 p99.9 `<= max(2 * reference, 0.02)` and maximum
    `<= max(2.5 * reference, 0.05)`.
- **Edge cases (no score weight).** Eight disclosed cases: odd lengths, GQA, MQA, large K
  outliers, flat and peaked logits, and a real Wan2.1-T2V-1.3B self-attention slice (8,192
  tokens).
- **Scored suite (Judge-only, `scripts/generate_final_manifest.py`).** Four families:
  - **video** (199 cases): non-causal self attention of Wan2.1 1.3B and 14B (480p and 720p,
    up to 75,600 tokens), HunyuanVideo (up to 118,800 tokens), Mochi-1, CogVideoX 2B/5B/1.5 and
    LTX-Video, at their clip lengths, sequence-parallel head splits (Ulysses/xDiT) and CFG batch.
  - **image** (44 cases): joint attention of SD3.5 Medium/Large and FLUX.1-dev at several
    resolutions and aspect ratios (about 1,400 to 17,000 tokens).
  - **llm** (392 cases): causal GQA prefill per tensor-parallel rank of 14 head layouts
    (Llama-3.x 3B/8B/70B/405B, Qwen2.5/Qwen3, Qwen3-235B, Mistral-Large-2) over context bands
    from 2K to 256K tokens, four hidden lengths per band, none a multiple of 128.
  - **real** (139 cases): captured layers of CogVideoX-2B (denoising steps 0, 10, 40), Wan2.1
    1.3B (480p, three steps) and 14B (720p), SD3.5 Medium (1024², every layer), and prefills of
    Llama-3.1-8B (16K tokens), Llama-3.1-8B-Instruct (64K and 128K) and Llama-3-8B-Instruct-262k
    (262K). Recipes are under
    `fixtures/recipe/real_layers/`; real layers use a row p99.9 limit of `max(2.5 * reference, 0.02)`.

  Each video, image and llm case is `recipe` (the model's published default setting, 14 cases)
  or `sweep` (its other supported settings).
- **Protocol.** Each synthetic case runs one untimed warm-up, then three timed trials on fresh
  inputs and outputs, each after an untimed call on other data of the same shape; the case value
  is the median; a real layer is timed the same way on fresh copies of its tensors. Recipe and real cases, and a
  frozen hidden half of each family's sweep cases, are compared against the golden on one hidden
  trial (from `L = 65,536` on, a stratified eighth of the 1024-row chunks, always the first and
  the last); every trial is checked for complete, finite writes and unchanged inputs. A case's
  warm-up (compilation and autotuning) may take at most 60 s.
  - Time is the larger of host wall time and CUDA-event time.
  - Effective work is `4*B*H_q*D*L^2` (non-causal) or `4*B*H_q*D*L*(L+1)/2` (causal).
  - Score: per family, `sqrt(geomean(recipe) * geomean(sweep))` for video, image and llm, and
    the geomean over sources of each source's layer geomean for real; the primary score is the
    geomean of the four family scores. Higher is better; uncapped, no baseline, no threshold.
- **Valid** means all nine gates pass: profile, suite, source admission, hardware, correctness,
  edges, real-model layers, fresh inputs and fresh outputs.
- **Interactive and final evaluation.** Scheduled auto-evaluation is off. The agent can submit
  with `sforge-submit`, at most once per 300 s cooldown; each submission runs the complete
  final suite and returns safe aggregate feedback (gate states, score, family scores). Private
  case identities, seeds, inputs and per-case timings stay in the Judge. The run's result is
  the best valid score reached within the budget. Each submission first passes an LLM integrity review against `llm_judge/POLICY.md`.

## Hardware

- **GPUs:** two GPUs per run by default. Work runs on GPU 0 and the Judge on GPU 1 (`--work-gpu`,
  `--judge-gpu`), so the Judge measures on its own GPU while Work keeps running. `--shared-gpu`
  puts both on one GPU instead, and the host pauses the Work container while the Judge runs.
- **Execution profile:** one CUDA GPU with compute capability >= 8.0 and at least 64 GiB of
  memory for the Judge (the largest cases peak near 44 GiB); the launcher also wants 14 GiB free
  before it starts. Work and Judge containers are
  limited to 8 CPUs and 32 GB of RAM each.

| hardware type | support |
|---|---|
| `h100-94gb-400w` (H100 NVL 94 GB) | qualified |
| `h100-80gb-700w` (H100 SXM 80 GB) | qualified |

TFLOPS scores can only be compared within one hardware type.

## Running

1. **Assets.** Run `tasks/latest/sageattention-quantized-attention-bf16-h100-v2-seed/assets/fetch.sh`.
   It downloads about 69 GB from the Infrathon dataset into `INFRATHON_ASSETS_ROOT` (default
   `~/.cache/infrathon/assets`):
   - `judge/`: the frozen final suite (`final_cases.json`) and the 139 scored real-layer
     captures (`real_layers/`, about 68 GB). The real layers are not built into any image:
     `scripts/sforge_judge_entry.py` mounts `REAL_LAYERS_DIR` (default `judge/real_layers/`)
     read-only into every Judge container;
   - `public/`: the two public real-model layers (`public_layers/`), the Wan2.1 edge-case capture, and
     `licenses/` (the notice and model licenses for the captures).

   No model or dataset download is needed. The capture recipes under `fixtures/recipe/` can
   re-capture any of them on your own GPU (optional; they download the pinned public model
   snapshots).
2. **Images.** Build them in one of two ways:
   - Pull the published Work and public Judge images, then build the two local layers:
     ```bash
     INFRATHON_PULL=1 tasks/latest/sageattention-quantized-attention-bf16-h100-v2-seed/scripts/build_images.sh
     ```
     The local Work layer adds the two public real-model layers; the local Judge layer adds the
     Judge-only final suite. Neither is published.
   - Build everything from source by running the same script without `INFRATHON_PULL`. Both images
     start from `pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel`.
3. **Run.**
   ```bash
   launch/sageattention-qattn/run.sh --model <model> --auth /path/to/auth.json \
     [--work-gpu N --judge-gpu M] [--tmux]
   ```
   - Defaults: Work on GPU 0, Judge on GPU 1, 12 hours, effort `max`, Judge port 18602.
     `--work-gpu`, `--judge-gpu`, `--hours`, `--effort` and `--judge-port` change them.
     `--shared-gpu` runs Work and Judge on one GPU, chosen with `--gpu N` (default 0) or
     `--device-uuid UUID`; either flag alone also selects shared mode. `--allocation-key` and
     `--lease-id` are optional scheduler IDs for shared mode, derived when they are omitted.
   - `--agent claude-code` with `--claude-token FILE` runs Claude Code instead of Codex.
   - `--model` takes any Codex or Claude model ID, or any model that `MODEL_API_BASE_URL`
     serves.
   - Other options: `--run-id`, `--calibration-epoch` (optional clock/power calibration campaign
     ID), `--cpuset`, and `--config FILE` for a launch config (see `launch/mlsysforge.env.example`).
   - `--dry-run` checks the model, runtime, images, integrity review, GPU and port, prints the
     plan and exits. It takes no lock and starts no container.
   - **Eval only:** `PHYSICAL_GPU=<index> tasks/latest/sageattention-quantized-attention-bf16-h100-v2-seed/scripts/eval_archive.sh ARCHIVE.tar.gz [RUN_ID]`
     evaluates one submission archive in the Judge. `PHYSICAL_GPU=<index> scripts/smoke_test.sh --oracle`,
     run from the task directory, checks the images with the admissible reference solution
     (`oracle/`, `kernel_v002`).
   - **Replay:** `launch/sageattention-qattn/replay.sh --origin-run RUN_ID --measurement-set NAME`
     re-measures the saved candidate versions of a finished run.

**Prerequisites the launcher checks:**

- the SForge harness;
- a version store (`SFORGE_VERSION_STORE_ROOT`, with this task registered);
- an agent runtime directory (`SFORGE_AGENT_RUNTIME_DIR`, built by
  `launch/build-agent-runtime.sh`);
- model credentials: a Codex `auth.json` (mode 0600), a Claude Code token file, or an
  OpenAI-compatible endpoint through `MODEL_API_BASE_URL`/`MODEL_API_KEY_FILE`;
- a ready LLM integrity review: `SFORGE_LLM_JUDGE_MODE=enforce` and the other
  `SFORGE_LLM_JUDGE_*` reviewer settings. The launcher refuses to start without them;
- if the host reaches the internet through a proxy, a proxy address that containers can reach.

The task spec allows internet access. Work reaches public hosts on ports 80/443 only through a
restricted proxy sidecar, and Codex web search is enabled.

## Files

| path | contents |
|---|---|
| `images/work/` | Work image: `AGENT_TASK.md`, the seed lab (`workdir/quant_attention_lab/`), the public runner and source-admission checker (`benchmark/`), and `Dockerfile.local`, the locally built layer with the two public real-model layers |
| `images/judge/` | public Judge image (verifier, candidate worker, source admission) and `Dockerfile.private`, the locally built layer with the Judge-only final suite (the real layers are mounted at run time) |
| `runtime/` | trusted input generator, golden and SageAttention-1 emulation, timing and gates, shared by both images |
| `fixtures/recipe/` | capture recipes (pinned models, data, capture points, sanity bands) and the optional capture scripts for the Wan2.1 edge case and the real-model layers |
| `judge/` | SForge Judge adapter (the Judge-only cases come from the assets, not the repository) |
| `tasks/` | SForge task spec (`*.json`): Judge command, timeouts, resources, submit paths |
| `portable/` | portable contract: suite manifests, software ABI, calibration and selection protocols, rebuild guide (`REBUILD_PROMPT.md`) |
| `assets/` | asset manifest and `fetch.sh` |
| `scripts/` | image build, fixed-budget run, eval, replay, smoke test, holdout generator and validators |
| `oracle/` | host-only, never copied into an image. All numbers below are Judge scores on an earlier suite on H100 NVL. `kernel_v001` is the seed control (297.7 TFLOPS). `ADMISSIBLE_VERSION` selects `kernel_v002`, upstream SageAttention-1 Triton, the best admissible paper path (348.9-350.7 TFLOPS in two runs, all gates); the smoke test and the normal Judge use it. `FINAL_VERSION` selects `kernel_v004`, the strongest paper path: upstream SageAttention2 sm90 CUDA with the per_warp quantizer. It is not admissible (it imports the compiled `sageattention` extension); for reference only, it was scored by the full Judge pipeline in a local-only image with source admission and the LLM review turned off (581.7 / 577.9 / 581.8 TFLOPS in three runs, all other gates pass; `kernel_v003`, upstream's default per_thread quantizer, 549.7 / 545.5 TFLOPS in two runs). See `oracle/notes.md` and `oracle/paper_reference/forced_judge/` |
| `llm_judge/POLICY.md` | integrity-review policy |
| `tests/` | host-side tests of the task contract |

## Upstream and license

The seed kernel is a Triton port of the quantized attention in
[SageAttention](https://github.com/thu-ml/SageAttention) at commit
`d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5` (Apache-2.0): `sageattention/core.py` and its
per-block quantization and attention Triton paths. The accuracy reference and the layer-wise
real-activation protocol follow the SageAttention papers. No upstream source or compiled package
is in either image.

The activation captures are derived from public models and data, and carry their terms. The
dataset's `public/licenses/` holds a notice and copies of the model licenses.

- **Llama-3.1-8B layers** (`meta-llama/Llama-3.1-8B` at `d04e592b`, on WikiText-2 text) and
  **Llama-3.1-8B-Instruct layers** (`NousResearch/Meta-Llama-3.1-8B-Instruct` at `d10aef79`, on
  WikiText-103 text): **Built with Llama.** Llama 3.1 is licensed under the Llama 3.1 Community
  License, Copyright © Meta Platforms, Inc. All Rights Reserved. Use of these layers is subject
  to that license and the Llama 3.1 Acceptable Use Policy. Before an optional re-capture, accept
  the license on the original `meta-llama` model page. By default the re-capture downloads the
  base weights from `NousResearch/Meta-Llama-3.1-8B` at `1f47e50c`; that mirror is used only
  because its files have the same SHA-256 as the pinned snapshot, which the capture checks
  (`REAL_LAYER_NO_MIRROR=1` downloads the original instead).
- **Llama-3-8B-Instruct-262k layers** (`gradientai/Llama-3-8B-Instruct-262k` at `5c5269d5`, on
  WikiText-103 text): Meta Llama 3 Community License (Built with Meta Llama 3).
- **WikiText-2 and WikiText-103** (`Salesforce/wikitext` at `b08601e0`): CC BY-SA 3.0.
- **CogVideoX-2B layers** (`zai-org/CogVideoX-2b` at `1137dacf`): Apache-2.0.
- **Wan2.1 layers** (`Wan-AI/Wan2.1-T2V-1.3B-Diffusers` at `0fad780a` and
  `Wan-AI/Wan2.1-T2V-14B-Diffusers` at `38ec498c`): Apache-2.0.
- **SD3.5 Medium layers** (`stabilityai/stable-diffusion-3.5-medium`, files pinned by SHA-256 in
  the recipe): Stability AI Community License. Before an optional re-capture, accept the license
  on the original model page; the capture reads the files from the mirror
  `adamo1139/stable-diffusion-3.5-medium-ungated` at `1cb9becf` only because they match the
  pinned SHA-256 values.
- **Public layers** (Work image only): `Wan-AI/Wan2.2-TI2V-5B-Diffusers` at `b8fff731`
  (Apache-2.0) and `Qwen/Qwen3-8B` on WikiText-2 text (Apache-2.0).
- **Wan2.1 edge capture** (`Wan-AI/Wan2.1-T2V-1.3B` at `37ec5126`, pipeline from
  [Wan2.1](https://github.com/Wan-Video/Wan2.1) at `9737cba9`): Apache-2.0.

The task's own code is under the repository's root `LICENSE` (PolyForm Noncommercial 1.0.0).
