# TileLang Mamba-2 SSD Forward on H100

This task optimizes the Mamba-2 SSD (state-space dual) forward pass, written in
TileLang, on one H100. The scored operator is `mamba_chunk_scan_combined`
without the `z` gate and without a returned final state: the forward used by
Mamba-2, Codestral Mamba, Nemotron-H, Falcon-H1, Bamba/Granite-4.0-H and Zamba2.
It is a **seed** task: the agent starts from a correct but deliberately
unoptimized four-stage TileLang implementation and must make it faster without
changing what it computes.

## The task

The agent works in `/app/workdir/mamba2_ssd_lab` inside the Work container. The
brief it receives is `AGENT_TASK.md` (contract, workloads, score, allowed
surface) plus `README_AGENT.md` (working mechanics).

- **Entry point.** Each candidate is `candidates/kernel_vNNN/submission.py`
  (three-digit version) and defines
  `ssd_forward(x, dt, A, B, C, D, out) -> None`. It runs on the current CUDA
  stream, writes all of `out`, leaves every input unchanged and returns `None`.
  `FINAL_VERSION` holds the integer version to be judged.
- **Tensors.** `x` `[batch, L, H, P]` bf16; `dt` `[batch, L, H]` FP32, positive
  and already preprocessed (no softplus, bias or clamp); `A` `[H]` FP32,
  negative; `B`, `C` `[batch, L, G, N]` bf16; `D` `[H]` FP32; `out`
  `[batch, L, H, P]` bf16. `L` is any positive integer.
- **Math.** Per batch and head, the exact linear recurrence
  `S = exp(A*dt) * S + dt * outer(x, B)`, `out = S @ C + D * x`, with FP32
  semantics for exponentials, state and accumulation. The internal chunk size is
  not part of the contract; how the scan is decomposed is the agent's choice.
- **Seed.** `candidates/kernel_v001` runs four separate kernels (chunk
  cumulative decay, chunk state, inter-chunk state passing, chunk scan) at a
  fixed chunk size of 256, with FP32 intermediates through global memory and no
  layout annotation, deep pipelining or tuning.
- **Submitted files.** Only `candidates/`, `FINAL_VERSION` and `notes.md` are
  submitted; `tools/`, `build/` and scratch files stay local.

Rules (enforced by source admission in both the public runner and the Judge,
and by a runtime operator record):

- Compute must be TileLang (`tilelang.jit`, `tilelang.language`,
  `tilelang.layout`, `tilelang.PassConfigKey`). Torch may only allocate buffers,
  make views and query the device and current stream; any other torch name is
  rejected.
- Rejected: `torch.matmul`/`@`/`einsum` and other existing GEMM, attention or SSM
  kernels (`mamba_ssm`, `fla`, Triton, FlashAttention, FlashInfer); every route
  out of the tile DSL (raw source, external/packed calls, raw `T.ptx_*`/`T.mma_*`
  intrinsics, compiler-flag or backend redirection, prebuilt kernels, the bundled
  TVM); double-underscore and dynamic attribute access; `tilelang.autotune`
  inside a submission (autotuning in scratch scripts is fine); `ctypes`,
  `subprocess`, `eval`/`exec`, environment mutation, private streams; and
  specializing on seeds, values, call order or benchmark presence. Imports are
  allowlisted (`tilelang`, `torch` and a small standard-library set).
- Every untimed decoy and edge-fixture call runs under a torch dispatch record;
  any ATen operator other than an allocation or a view fails the
  `tilelang_only_ok` gate.
- Copying external implementations or solution material is not allowed.
- A required LLM integrity review (`llm_judge/POLICY.md`) reads the submission
  against the brief before the GPU Judge.

The launcher's default agent budget is 12 hours (`TASK_TIMEOUT=43200`).

## Evaluation

The Judge runs a frozen suite of 162 cases: 24 correctness-only edge fixtures and
138 scored cases.  Every `sforge-submit` and the final evaluation run the same
complete suite.

- **Scored cases.** 15 Mamba-2 head geometries read from published model configs
  (Mamba-2, Codestral, Nemotron-H, Nemotron 3 Nano, Falcon-H1, Granite-4.0-H,
  Bamba, Zamba2), five of them the per-rank layouts of tensor-parallel serving,
  in four workload families:

  | family | cases | what it models |
  | --- | ---: | --- |
  | `batch` | 42 | training (`L = 8192`, the upstream TileLang benchmark grid) and padded batches, batch 2 to 16 |
  | `serving` | 54 | batch 1 at real request lengths (Mooncake traces), capped by each model's context limit |
  | `long` | 18 | batch 1 from 60,000 tokens to each model's limit, up to 262,144 |
  | `real` | 24 | layers captured from Granite-4.0-H-Micro and Bamba-9B-v2 on WikiText-103, each timed once |

  The case list lives only in the Judge-only frozen instance
  (`judge/final_cases.json`); `portable/final-suite-manifest.json` records its
  SHA-256, families, tiers and weights but not its shapes.  The public suite in
  the Work container is separate: 24 cases in the same four families on other
  geometries and lengths, including one captured layer of Granite-4.0-H-350M.
- **Edge fixtures.** 24 fixtures (down to `L = 1`, chunk-boundary and
  ragged-tail lengths to `L = 262143`, extreme decays, isolated `D` and state
  terms, the largest-state and single-group geometries) run before every suite
  and never enter the score.
- **Correctness.** Outputs are compared with a blocked FP32 reference (TF32
  off). An element passes if its absolute error is within `1.5e-2 * max|ref|` or
  its relative error is within `3.01/128`; the whole-tensor relative L2 must be
  at most `6e-3`. A captured real layer is checked on whole-tensor relative L2
  alone, at most `1.3e-2`. NaN or Inf fails.
- **Timing.** Per case: one warm-up call (compilation, correctness, workspace
  measurement), then 3 measured trials (a real case: 1 trial on the captured
  layer after a warm-up on generated tensors of its shape). Each trial draws
  fresh inputs after an untimed decoy call with different data and addresses
  and writes into a NaN-poisoned output; every trial is checked for complete,
  finite writes and unchanged inputs, and the warm-up plus one hidden trial per
  case are compared against their own reference. Time is the larger of the
  CUDA-event and synchronized host-wall measurement. Compilation is cold in
  every Judge phase (fresh, uncached TileLang directories), so the suite's
  about 140 distinct shapes are compiled inside the evaluation time limit.
- **Score.** `effective TFLOPS = 4 * batch * L * H * P * N / seconds / 1e12`
  (chunk-size independent). Each case is the median of its trials; cases are
  geometrically averaged per workload family, and the score is the equal-weight
  geometric mean of the four family means, in effective TFLOPS, higher is
  better. There is no threshold, baseline ratio or cap.
- **Valid.** A result is valid only if every hard gate passes: source admission,
  one GPU meeting the capability envelope, all edge fixtures, correctness on
  every compared call, complete writes with no NaN, unmodified inputs, the
  fresh-tensor protocol, a per-case warm-up wall of at most 300 s, transient
  device memory at most 4x the case's input plus output bytes and at most
  24 GiB, and `tilelang_only_ok`. The run's result is the best valid score.
- **Interactive vs final.** Scheduled auto-evaluation is off. The agent may run
  `sforge-submit` (300-second cooldown); each submission runs the complete frozen
  suite and returns the aggregate, the four family scores and gate states. After
  the budget ends, the Judge independently reruns the same suite on the sealed
  final archive. Shapes, seeds, per-case Judge timings and tensors stay in the
  Judge. The Judge timeout is 4500 s for the candidate worker and 5400 s for
  the whole Judge run.


## Hardware

Two GPUs per run by default: Work runs on GPU 0 and the Judge on GPU 1
(`--work-gpu`, `--judge-gpu`), so the Judge measures on its own GPU while Work
keeps running. With `--shared-gpu` Work and Judge share one GPU: before each
Judge run the host pauses the Work container and resumes it afterwards (a
root-owned `prepare_submit.sh` hook first checks that no Work process still
holds the device). The execution profile requires compute capability 8.0 or newer and at
least 73 GiB of device memory.

| hardware type | support |
| --- | --- |
| `h100-94gb-400w` (H100 NVL 94 GB) | qualified |
| `h100-80gb-700w` (H100 SXM 80 GB) | qualified |

Work and Judge containers are limited to 8 CPUs and 48 GB of host memory each.

## Running

1. **Assets.** `tasks/latest/tilelang-mamba2-ssd-forward-h100-v2-seed/assets/fetch.sh`
   downloads into `INFRATHON_ASSETS_ROOT` the Judge-only final-suite case list
   (`judge/final_cases.json`), the 24 Judge-only captured layers
   (`judge/real_layers/`, 8.6 GB) and the public captured layer
   (`public/public_layer/`, 31 MB); none of them is in the repository. To
   recapture the layers instead, run
   `CAPTURE_GPU=<uuid> OUT=<dir> HF_HUB_CACHE=<hub> fixtures/recipe/real_layers/build_real_layers.sh judge`
   (and `public`); it downloads the pinned models and WikiText-103 split named in
   `fixtures/recipe/real_layers/REAL_LAYERS.json`. `REAL_LAYERS_DIR` and
   `PUBLIC_LAYER_DIR` override the asset-root locations.
2. **Images.** `INFRATHON_PULL=1 tasks/latest/tilelang-mamba2-ssd-forward-h100-v2-seed/scripts/build_images.sh`
   pulls the published Work and public Judge images and builds the Judge holdout
   layer locally (`images/judge/Dockerfile.private`: the case list and the 24
   layers). Without `INFRATHON_PULL=1` it builds both images from source (base
   `pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel`, `tilelang==0.1.13`); the Work
   image takes the public layer. With `INFRATHON_ASSETS_ROOT` set, the build reads
   all three assets from the asset root.
3. **Run.**

   ```bash
   launch/tilelang-mamba2-ssd-v2/run.sh --model <codex-model> \
     --auth <path-to-codex-auth.json> [--work-gpu N --judge-gpu M] [--tmux]
   ```

   Defaults: Work on GPU 0, Judge on GPU 1, 12 hours, effort `max`, Judge port
   18607; `--work-gpu`, `--judge-gpu`, `--hours`, `--effort` and `--judge-port`
   change them. `--shared-gpu` runs Work and Judge on one GPU, chosen with
   `--gpu N` (default 0) or `--device-uuid UUID` (from `nvidia-smi -L`); either
   flag alone also selects shared mode. `--dry-run` resolves and checks the plan
   (model, agent runtime, images, integrity review, GPU, port) without starting anything. Host
   settings can come from `--config FILE` (see `launch/mlsysforge.env.example`).
   `--agent claude-code` with `--claude-token FILE` (a 0600 file holding one
   `claude setup-token` token) and a Claude `--model` runs Claude Code instead of Codex.

   Replay saved candidates of a finished run through the Judge:

   ```bash
   launch/tilelang-mamba2-ssd-v2/replay.sh --origin-run <RUN_ID> --measurement-set <NAME> [--per-version]
   ```

   A single archive can be evaluated with
   `tasks/latest/tilelang-mamba2-ssd-forward-h100-v2-seed/scripts/eval_archive.sh ARCHIVE.tar.gz [RUN_ID]`
   (GPU chosen with `PHYSICAL_GPU`, default 0).

Prerequisites the launcher checks: a Codex or Claude Code agent runtime directory
(`SFORGE_AGENT_RUNTIME_DIR`, mounted read-only into the agent-free Work image),
Codex credentials (`--auth`) or a Claude Code token file (`--claude-token`), a
proxy reachable from the containers, and a
version store (`SFORGE_VERSION_STORE_ROOT`) that registers this task so
submitted candidates can be replayed. The LLM integrity review is configured
with the `SFORGE_LLM_JUDGE_*` settings (see `launch/mlsysforge.env.example`).

## Files

| path | contents |
| --- | --- |
| `images/work/` | Work image: `AGENT_TASK.md`, the lab directory with the seed candidate and `README_AGENT.md`, public runner and source admission (`benchmark/`) |
| `images/judge/` | Judge image: `run_judge.py`, `verify_ssd.py`, candidate worker, source admission, protected hashes; `Dockerfile.private` adds the holdout layer |
| `runtime/` | trusted runtime shared by the public runner and the Judge (reference, workload generator, timing, operator record) |
| `judge/` | host-side Judge adapter |
| `tasks/` | SForge task spec (`*.json`) and `BENCHMARK.yaml` |
| `portable/` | portable contract: suite, oracle, selection, calibration and resource manifests |
| `assets/` | asset manifest and `fetch.sh` |
| `oracle/` | host-only reference candidates used to check the Judge: `kernel_v001` is a three-kernel variant of the seed, used as a positive control; `FINAL_VERSION` selects `kernel_v002`, the offline-tuned upstream TileLang SSD port (see `oracle/notes.md`) |
| `llm_judge/` | integrity review policy |
| `scripts/` | image build, run, replay, archive evaluation, smoke test and validation scripts |
| `tests/` | CPU tests of the contract, scoring, source admission and launchers |

## Upstream and license

- **TileLang** `0.1.13`, installed from the published wheel in both images
  (paired with `torch 2.8.0+cu129` and `triton 3.4.0`). The seed's TileLang
  idioms follow upstream's `examples/linear_attention/example_mamba_chunk_scan.py`
  and `example_mamba_chunk_state.py`; the scored lengths and batch come from
  upstream's `benchmark/mamba2`. The reference source tree consulted is TileLang
  commit `f935b2ad27152c9d970b15c666acaf3ed73e5f2a`; no upstream file is shipped
  or executed. TileLang's root license is MIT text with an additional sentence
  on collaboration terms with Microsoft for a fixed period; the wheel bundles
  CUTLASS headers.
- **Operator definition:** Mamba-2 SSD, from *Transformers are SSMs:
  Generalized Models and Efficient Algorithms Through Structured State Space
  Duality* (Dao and Gu, ICML 2024), as implemented in `mamba_ssm`
  (`mamba_chunk_scan_combined`). `mamba_ssm` is not used or shipped.
- **Head geometries** are read from the published configs of Mamba-2-2.7B and
  -1.3B, Mamba-Codestral-7B-v0.1, Nemotron-H-8B, -47B and -56B, Nemotron 3 Nano,
  Falcon-H1-3B, -7B and -34B, Granite-4.0-H-Small and -Micro, Bamba-9B-v2 and
  Zamba2-1.2B and -7B; the geometries need only these configs (for the models'
  licenses, see each model's page).
- **Real layers** are captured by running `ibm-granite/granite-4.0-h-micro` and
  `ibm-ai-platform/Bamba-9B-v2` (Judge) and `ibm-granite/granite-4.0-h-350m`
  (public) on WikiText-103 (`Salesforce/wikitext`), at the revisions in
  `fixtures/recipe/real_layers/REAL_LAYERS.json`. The three models are
  Apache-2.0 and the WikiText-103 text is CC BY-SA 3.0 and GFDL, as recorded
  there; the captured layers in the Infrathon dataset are derived from them.
- The images also carry NVIDIA CUDA toolchain components under NVIDIA's
  licenses.

The published images contain these third-party components under their own
licenses; see the TileLang repository for its license terms and third-party
notices.

The task code is part of this repository and under its root `LICENSE` (PolyForm
Noncommercial 1.0.0).
