<h1 align="center">Infrathon</h1>

<p align="center"><b>Benchmarking language agents for real-world, long-horizon AI infra optimization</b></p>

<p align="center">
  <a href="https://arxiv.org/abs/2611.11995"><img src="https://img.shields.io/badge/arXiv-2611.11995-b31b1b?logo=arxiv&logoColor=white" alt="arXiv paper"></a>
  <a href="https://infrathonx.github.io"><img src="https://img.shields.io/badge/Website-infrathonx.github.io-0969da?logo=googlechrome&logoColor=white" alt="Website"></a>
  <a href="https://huggingface.co/datasets/Infrathon/Infrathon"><img src="https://img.shields.io/badge/Dataset-Infrathon%2FInfrathon-ffcc4d?logo=huggingface&logoColor=black" alt="Hugging Face dataset"></a>
  <a href="https://hub.docker.com/r/infrathon/infrathon"><img src="https://img.shields.io/badge/Docker%20Hub-infrathon%2Finfrathon-2496ED?logo=docker&logoColor=white" alt="Docker Hub images"></a>
  <a href="https://github.com/InfrathonX/Infrathon-build"><img src="https://img.shields.io/badge/Build%20tasks-Infrathon--build-24292f?logo=github&logoColor=white" alt="Infrathon-build"></a>
</p>

<p align="center"><b>25 tasks</b> · <b>12–24 hours</b> per run · <b>6 frontier agents</b> · <b>150 runs, 10,000+ H100 GPU-hours</b></p>

<p align="center"><img src="docs/assets/overview.svg" width="100%" alt="How a run works: the task gives a workspace, public tests, private tests, resources and a time budget. The agent edits and tests in its Work container and submits; the Controller's LLM integrity review passes or rejects the submission; the Judge runs the held-out workloads on its own GPUs and scores it; the feedback goes back to the agent. The run's result is its best valid score within the budget."></p>

Infrathon gives a coding agent a GPU workstation, a real ML system and 12 to 24 hours to make it
better: GPU kernels, inference engines and training systems taken from open-source projects and
research code. An independent Judge, in its own container on its own GPUs, checks every submission
for correctness and fidelity on held-out workloads and scores its performance; before that, an LLM
integrity review rejects submissions that break the task's rules. A run's result is the best valid
score it reaches within its budget.

## What agents do with 24 hours

<p align="center"><img src="docs/assets/discovery.svg" width="100%" alt="Quant-VideoGen KV codec, 24 hours: Claude Opus 5.5's first five submissions are invalid; it then passes the human codec's score with 4-bit quantization, PCA and entropy coding, quantization-step tuning, trellis quantization, and metadata compression with per-head quantization, ending at a score of 28.45 against the human codec's 6.82. The other five agents end between 5.82 and 11.34."></p>

On the Quant-VideoGen KV codec task, Claude Opus 5.5 starts from a stub and keeps redesigning the
codec for the whole day. It raises KV-cache compression from the human codec's **6.82× to 19.62×**,
with higher PSNR and SSIM and lower LPIPS.

<p align="center"><img src="docs/assets/results.svg" width="100%" alt="Best valid score over the budget for six agents on SageAttention, Qwen3 ORPO, Miles Qwen3 OPCD and LLaDA decoding, and the rank score over all 25 tasks: Claude Opus 5.5 91.2 (17 wins), GPT-6 Astra 75.2 (6), GPT-6.1 Sol 54.4 (1), Qwen 3.8 Flash 40.8 (1), DeepSeek V4.1 Flash 30.4 (0), GLM 5.3 Flash 8.0 (0)."></p>

Agents keep finding improvements many hours into a run, and they differ widely. At equal GPU-hours,
one long run beats the best of two or four shorter runs on module and system tasks, while shorter
runs stay competitive on operators. Every task's trajectories and scores are on the
[website](https://infrathonx.github.io); the analysis is in the [paper](https://arxiv.org/abs/2611.11995).

## Tasks

<p align="center"><img src="docs/assets/tasks.svg" width="100%" alt="The 25 tasks: 11 operators (12 hours each), 5 modules and 9 systems (24 hours each), for training and for inference."></p>

`seed` tasks start from a working baseline to improve; `scratch` tasks start from a stub. Each task
directory has its own README: what the agent must do, how it is scored, its hardware, and how to run
it. The launchers default to the paper's setup: its budgets, reasoning effort `max`, and the GPU
counts below; [launch/README.md](launch/README.md) lists every default and the flags that change it.

<details>
<summary><b>All 25 tasks: directories, launchers, budgets and GPUs</b></summary>
<br>

| Task | Launcher | Level | Budget | GPUs per run (Work + Judge) |
|---|---|---|---|---|
| [`attn-qat-nvfp4-training-attention-h100-v1-seed`](tasks/latest/attn-qat-nvfp4-training-attention-h100-v1-seed) | [`launch/attn-qat-train/`](launch/attn-qat-train) | training operator | 12 h | 1 + 1 |
| [`standalone-k-grouped-fp8-wgrad-h100-v2-scratch`](tasks/latest/standalone-k-grouped-fp8-wgrad-h100-v2-scratch) | [`launch/standalone-kgrouped-wgrad/`](launch/standalone-kgrouped-wgrad) | training operator | 12 h | 1 + 1 |
| [`standalone-dynamic-grouped-fp8-h100-v3-scratch`](tasks/latest/standalone-dynamic-grouped-fp8-h100-v3-scratch) | [`launch/standalone-fp8/`](launch/standalone-fp8) | inference operator | 12 h | 1 + 1 |
| [`standalone-sparse-mla-prefill-h100-v2-scratch`](tasks/latest/standalone-sparse-mla-prefill-h100-v2-scratch) | [`launch/standalone-sparse-mla-v2/`](launch/standalone-sparse-mla-v2) | inference operator | 12 h | 1 + 1 |
| [`standalone-paged-gqa-decode-bf16-h100-v2-scratch`](tasks/latest/standalone-paged-gqa-decode-bf16-h100-v2-scratch) | [`launch/standalone-paged-gqa-v2/`](launch/standalone-paged-gqa-v2) | inference operator | 12 h | 1 + 1 |
| [`sageattention-quantized-attention-bf16-h100-v2-seed`](tasks/latest/sageattention-quantized-attention-bf16-h100-v2-seed) | [`launch/sageattention-qattn/`](launch/sageattention-qattn) | inference operator | 12 h | 1 + 1 |
| [`lmcache-paged-kv-block-transfer-h100-v1-seed`](tasks/latest/lmcache-paged-kv-block-transfer-h100-v1-seed) | [`launch/lmcache-kv-transfer/`](launch/lmcache-kv-transfer) | inference operator | 12 h | 1 + 1 |
| [`flash-kmeans-paper-h100-v7-scratch`](tasks/latest/flash-kmeans-paper-h100-v7-scratch) | [`launch/flash-kmeans/`](launch/flash-kmeans) | inference operator | 12 h | 1 + 1 |
| [`tilelang-mamba2-ssd-forward-h100-v2-seed`](tasks/latest/tilelang-mamba2-ssd-forward-h100-v2-seed) | [`launch/tilelang-mamba2-ssd-v2/`](launch/tilelang-mamba2-ssd-v2) | inference operator | 12 h | 1 + 1 |
| [`minference-sparse-prefill-h100-v3-seed`](tasks/latest/minference-sparse-prefill-h100-v3-seed) | [`launch/minference-v3/`](launch/minference-v3) | inference operator | 12 h | 1 + 1 |
| [`eagle3-spec-decode-engine-qwen3-8b-h100-v1-seed`](tasks/latest/eagle3-spec-decode-engine-qwen3-8b-h100-v1-seed) | [`launch/eagle3-spec-decode/`](launch/eagle3-spec-decode) | inference system | 24 h | 1 + 1 |
| [`llada-h100-decoder-optimization-v2-vanilla-seed`](tasks/latest/llada-h100-decoder-optimization-v2-vanilla-seed) | [`launch/llada-h100/`](launch/llada-h100) | inference system | 24 h | 1 + 1 |
| [`qwen3-1p7b-orpo-training-optimizer-dual-h100-v4-cross-matrix-seed`](tasks/latest/qwen3-1p7b-orpo-training-optimizer-dual-h100-v4-cross-matrix-seed) | [`launch/qwen3-orpo/`](launch/qwen3-orpo) | training module | 24 h | 2 + 2 |
| [`qwen3-4b-to-1p7b-gkd-distillation-optimizer-dual-h100-v1-official-trl-seed`](tasks/latest/qwen3-4b-to-1p7b-gkd-distillation-optimizer-dual-h100-v1-official-trl-seed) | [`launch/qwen3-gkd/`](launch/qwen3-gkd) | training module | 24 h | 2 + 2 |
| [`miles-qwen3-opcd-2xh100-v3-seed`](tasks/latest/miles-qwen3-opcd-2xh100-v3-seed) | [`launch/miles-opcd/`](launch/miles-opcd) | training system | 24 h | 2 + 2 |
| [`miles-qwen3-opd-scratch-2xh100-v2-seed`](tasks/latest/miles-qwen3-opd-scratch-2xh100-v2-seed) | [`launch/miles-opd/`](launch/miles-opd) | training system | 24 h | 2 + 2 |
| [`miles-qwen3-alfworld-agentic-rl-2xh100-v1-seed`](tasks/latest/miles-qwen3-alfworld-agentic-rl-2xh100-v1-seed) | [`launch/miles-alfworld/`](launch/miles-alfworld) | training system | 24 h | 2 + 2 |
| [`miles-diffusion-sd35m-mopd-2xh100-v2-seed`](tasks/latest/miles-diffusion-sd35m-mopd-2xh100-v2-seed) | [`launch/miles-diffusion-opd/`](launch/miles-diffusion-opd) | training system | 24 h | 2 + 2 |
| [`miles-diffusion-wan22-dancegrpo-2xh100-v2-seed`](tasks/latest/miles-diffusion-wan22-dancegrpo-2xh100-v2-seed) | [`launch/miles-diffusion-wan/`](launch/miles-diffusion-wan) | training system | 24 h | 2 + 2 |
| [`streamdiffusionv2-dual-h100-pipeline-runtime-v2-seed`](tasks/latest/streamdiffusionv2-dual-h100-pipeline-runtime-v2-seed) | [`launch/sdv2-pipeline/`](launch/sdv2-pipeline) | inference system | 24 h | 2 + 2 |
| [`flux-moe-ag-scatter-grouped-gemm-dual-h100-v1-seed`](tasks/latest/flux-moe-ag-scatter-grouped-gemm-dual-h100-v1-seed) | [`launch/flux-moe-ag/`](launch/flux-moe-ag) | training operator | 12 h | 8, shared by Work and the Judge |
| [`concurrent-agent-serving-optimizer-qwen36-swe192-v24`](tasks/latest/concurrent-agent-serving-optimizer-qwen36-swe192-v24) | [`launch/concurrent-serving/`](launch/concurrent-serving) | inference system | 24 h | none (a 2-GPU vLLM backend on the host serves both) |
| [`quant-videogen-cross-model-kv-codec-h100-v4-scratch`](tasks/latest/quant-videogen-cross-model-kv-codec-h100-v4-scratch) | [`launch/quant-videogen/`](launch/quant-videogen) | inference module | 24 h | 1 + 6 |
| [`kvpress-kv-eviction-policy-qwen3-8b-h100-v2-scratch`](tasks/latest/kvpress-kv-eviction-policy-qwen3-8b-h100-v2-scratch) | [`launch/kvpress-kv-eviction-scratch/`](launch/kvpress-kv-eviction-scratch) | inference module | 24 h | 1 + 7 |
| [`kvzip-query-agnostic-kv-scoring-qwen3-8b-h100-v2-scratch`](tasks/latest/kvzip-query-agnostic-kv-scoring-qwen3-8b-h100-v2-scratch) | [`launch/kvzip-kv-scoring/`](launch/kvzip-kv-scoring) | inference module | 24 h | 1 + 7 |

- **Work and the Judge use separate GPUs**, so the Judge measures while Work keeps going. Flux is
  the exception: both use the same eight GPUs of one host (peer access between every pair) and
  Work is paused for each Judge round.
- **One GPU for both.** The launchers of the 1 + 1 tasks (all but MInference) take `--shared-gpu`;
  those of ORPO, GKD and StreamDiffusionV2 take `--shared-gpus`; a Miles config takes
  `JUDGE_GPU_INDICES=shared`. The host then pauses Work while the Judge measures.
- **Fixed Judge counts.** quant-videogen (1 + 6), kvpress and kvzip (1 + 7) shard the Judge over a
  fixed number of GPUs; their launchers let you choose which GPUs, but refuse any other count.
- **Totals.** One run of every task holds 89 GPUs if all 25 run at once, and takes 1,800
  GPU-hours (each run's GPUs for its whole budget, not counting the final evaluation after it).

**Hardware.** All 25 tasks run on both NVIDIA H100 NVL 94 GB (`h100-94gb-400w`) and NVIDIA H100 SXM
80 GB (`h100-80gb-700w`): reference and oracle submissions were judged valid on both with the current
task version. The Docker image tags start with the hardware type (`<hardware>-<task>-work`,
`<hardware>-<task>-judge`). Today the two tags of a task point at the same image; the prefix is
there so that images for other GPUs can be added later. On an H100 NVL host Flux moves data
between its NVLink pairs over PCIe, so its scores are lower there and compare only with other
H100 NVL runs.

</details>

## Quick start

You need a Linux host with Docker, the NVIDIA Container Toolkit and H100 GPUs, [uv](https://docs.astral.sh/uv/)
with Python 3.12, access to the gated [Infrathon dataset](https://huggingface.co/datasets/Infrathon/Infrathon)
(request it on the dataset page), and an agent: Codex or Claude Code.

```bash
git clone https://github.com/InfrathonX/Infrathon && cd Infrathon
uv sync --frozen --dev --python 3.12   # the SForge harness and tools
.venv/bin/hf auth login                # once your access to Infrathon/Infrathon is approved
uv pip install --python .venv/bin/python numpy==2.5.2
uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cpu torch==2.14.1
launch/build-agent-runtime.sh --agent codex   # host-side agent CLI at the pinned version (see launch/README.md)
cp launch/mlsysforge.env.example launch/mlsysforge.env   # credentials, INFRATHON_ASSETS_ROOT, ...
set -a; . launch/mlsysforge.env; set +a   # the asset and image scripts read INFRATHON_ASSETS_ROOT / HF_HUB_CACHE from the environment

T=tasks/latest/lmcache-paged-kv-block-transfer-h100-v1-seed
$T/assets/fetch.sh                     # task data and models into INFRATHON_ASSETS_ROOT
INFRATHON_PULL=1 $T/scripts/build_images.sh  # pull the published images, build the Judge holdout layer
# (the five Miles tasks and MInference: scripts/infrathon/pull_images.sh "$(basename "$T")" instead)
launch/lmcache-kv-transfer/run.sh --help
```

Each task README gives the exact commands, including evaluation of a single archive without an
agent run.

<details>
<summary><b>Requirements in full</b></summary>
<br>

- Linux with Docker, the NVIDIA Container Toolkit and recent NVIDIA drivers; NVIDIA H100 GPUs
  (2, 4, 7 or 8 per run, see [Tasks](#tasks); each task README gives its memory floor). [docs/HOST_PREREQUISITES.md](docs/HOST_PREREQUISITES.md)
  lists disk, memory and Docker settings; [docs/NEW_HARDWARE.md](docs/NEW_HARDWARE.md) lists what it takes
  to qualify the tasks on another GPU.
- [uv](https://docs.astral.sh/uv/) and Python 3.12 (`pyproject.toml` accepts 3.10-3.13; the release was
  tested with 3.12). Host-side checks of several tasks import CPU
  `torch` and `numpy`, which the lock file leaves out; install them into the venv as above.
- Access to the [Infrathon dataset](https://huggingface.co/datasets/Infrathon/Infrathon) on Hugging
  Face, which holds the task data and the held-out Judge inputs: request access on the dataset
  page, then log in with `hf auth login` (the `hf` CLI is listed in
  [docs/HOST_PREREQUISITES.md](docs/HOST_PREREQUISITES.md)).
- An agent: Codex (a ChatGPT login `auth.json`, or any model behind `MODEL_API_BASE_URL`) or
  Claude Code (on the launchers that list `--agent`; see `run.sh --help`). Agent CLIs are mounted
  into the Work container from a host runtime directory (`launch/build-agent-runtime.sh`).
- A model for the integrity review: set `SFORGE_LLM_JUDGE_MODE=enforce` and
  `SFORGE_LLM_JUDGE_MODEL`. The reviewer uses the run's Codex runtime and `auth.json` unless other
  `SFORGE_LLM_JUDGE_*` settings say otherwise; a Claude Code reviewer needs an API key file (see
  [docs/LLM_JUDGE.md](docs/LLM_JUDGE.md)).

</details>

<details>
<summary><b>Images, data and hardware types</b></summary>
<br>

- **Images.** Each task publishes a Work image and a public Judge image per supported hardware
  type to Docker Hub ([`infrathon/infrathon`](https://hub.docker.com/r/infrathon/infrathon)`:<hardware>-<task>-{work,judge}`;
  pulls are by digest, so `INFRATHON_REGISTRY_REPOSITORY` can point at a copy elsewhere);
  the concurrent-serving task also builds its nested benchmark images and vLLM backend image
  on your host. The public Judge holds no held-out data: for most tasks the held-out inputs are
  added by a small layer built on your host (`images/judge/Dockerfile.private`; the SageAttention
  task adds its captured model activations to both images the same way); concurrent-serving,
  MInference, SD3.5 and Wan2.2 mount their Judge-only data at run time instead. Tasks can also be built entirely
  from source.
- **Data.** `tasks/latest/<task>/assets/manifest.json` lists every file a task needs beyond this
  repository with its SHA-256. Task data, including the Judge-only held-out sets, comes from the
  [Infrathon dataset](https://huggingface.co/datasets/Infrathon/Infrathon) on Hugging Face (access on
  request); public models and datasets come from their own
  Hugging Face repositories at pinned commits. `assets/fetch.sh` downloads and verifies them.
- **Hardware types.** `INFRATHON_HARDWARE` names the host's type (`h100-94gb-400w` or
  `h100-80gb-700w`); otherwise it is detected from GPU 0.

[scripts/infrathon/README.md](scripts/infrathon/README.md) describes images, assets and hardware types in
detail.

</details>

<details>
<summary><b>Repository layout</b></summary>
<br>

| Path | Contents |
|---|---|
| `tasks/latest/<task>/` | one task: Work and Judge images, runtime, Judge, task spec, assets manifest, scripts, tests |
| `sforge/` | the SForge harness (`sforge/harness/`): Work and Judge orchestration, agents, submission server, integrity review, and an optional host model relay for providers without the Responses API (`sforge/harness/relay/`) |
| `sforge/pyromind/` | the PyroMind platform adapter (Sandbox runner, deployments, qualification); not used by local Docker runs |
| `launch/` | one launcher per task (`launch/<dir>/run.sh`) and shared launcher code |
| `scripts/infrathon/` | release-wide tools: assets (`assets.py`), image pulling, layer splitting |
| `scripts/miles_*` | shared code of the Miles tasks and their base-image recipes |
| `docs/` | integrity review, CPU affinity, host prerequisites, internet and web search, bringing the tasks to new hardware; `docs/assets/make_figures.py` draws the figures in this README |
| `tests/` | harness and tooling tests (`uv run pytest tests`) |

</details>

## Integrity review

Every submission is reviewed by an LLM agent before the GPU Judge runs, against each task's
`llm_judge/POLICY.md` (for example: no copying of external implementations, no caching of
results across calls, no detection of the measurement). Configure it with the
`SFORGE_LLM_JUDGE_*` settings; see [docs/LLM_JUDGE.md](docs/LLM_JUDGE.md).

## Building new tasks

[Infrathon-build](https://github.com/InfrathonX/Infrathon-build) turns an expert's task definition
(what the agent optimizes, the public and private test cases, the metric, the checks, the resources
and the time budget) into a runnable Infrathon task. A builder agent implements the task in an
Infrathon checkout, using the tasks here as examples; the tool checks it from a clean clone, runs
trials in which another agent attempts it through the real launcher, and delivers a patch for this
repository with a report for the expert.

## Citation

```bibtex
@article{hu2026infrathon,
  title   = {Infrathon: Benchmarking Language Agents for Real-World, Long-Horizon AI Infra Optimization},
  author  = {Hu, Zhengyu and Xiao, Zheyuan and Wang, Yike and Ivison, Hamish and Wang, Haoyu and
             Yuan, Yige and Shao, Rulin and Li, Yuetai and Lyu, Niqi and Ding, Yicheng and Yang, Shu and
             Zeng, Zhichen and Zhu, Banghua and Pei, Jiaxin and Ding, Kaize and Dasigi, Pradeep and
             Roth, Dan and Hajishirzi, Hannaneh and Smith, Noah A. and Xiao, Teng and Poovendran, Radha},
  journal = {arXiv preprint arXiv:2611.11995},
  year    = {2026}
}
```

## License

Infrathon is released under the [PolyForm Noncommercial License 1.0.0](LICENSE): free for
research, teaching, personal use and use by noncommercial organizations. Any commercial use, such
as selling it, offering it as a paid service or using it in a commercial product, needs a separate
license from the authors. This covers the tasks (`tasks/`), their oracles and judges, the scripts,
the launchers and the documentation.

<details>
<summary>Exceptions, which keep their own licenses</summary>
<br>

- The SForge harness (`sforge/`, packaged as `sforge`) and its tests (`tests/sforge/`) are under
  the Apache License 2.0 ([sforge/LICENSE](sforge/LICENSE), [sforge/NOTICE](sforge/NOTICE)).
- Any file that carries its own copyright or license header, or sits under a directory with its
  own `LICENSE`, `THIRD_PARTY_LICENSE` or `THIRD_PARTY_NOTICES` file (for example vendored
  upstream code inside a task directory), keeps that license.
- Models, datasets and upstream projects that tasks use keep their own licenses; each task README
  lists them.

</details>

## Acknowledgements

SForge is adapted from the evaluation harness released with
[EdgeBench](https://github.com/ByteDance-Seed/EdgeBench) by ByteDance Seed. We thank the EdgeBench
team for making it available.
