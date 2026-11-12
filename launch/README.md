# Launchers

One entrypoint per task for starting agent runs and replaying saved
candidates, with the cross-task knobs (agent, model, reasoning effort, budget,
agent CLI pin, version store root) in one place. Every script here is a thin
delegate: the task-side hardening — GPU locks, image currency checks, formal
profile whitelists, provenance pinning — runs inside
`tasks/latest/<task>/scripts/`, and nothing here bypasses it.

Before a first run on a host, see
[docs/HOST_PREREQUISITES.md](../docs/HOST_PREREQUISITES.md) and, for images,
assets and hardware types, [scripts/infrathon/README.md](../scripts/infrathon/README.md).

## Launch directories

Called with only `--model` (and the agent's credentials), every launcher uses the paper's
budget for the task's level (12 h for operator tasks, 24 h for module and system tasks),
reasoning effort `max`, and the GPU layout below. The default GPUs
and Judge port suit one run at a time on an 8-GPU host; to run several tasks side by side,
give each run its own GPUs and Judge port. In the default layout the Judge has GPUs of its
own and measures while Work keeps going (Flux excepted: Work and the Judge share its eight
GPUs and Work is paused during Judge rounds, see [Flux](#flux-eight-gpus)). The one-GPU launchers (except MInference v3) and
the two-GPU launchers also have a shared mode, in which Work and the Judge use the same GPUs
one after the other: the host pauses the Work container while the Judge measures.

| directory | task directory (`tasks/latest/…`) | GPUs per run | default GPUs | default budget | default Judge port |
| --- | --- | --- | --- | --- | --- |
| `attn-qat-train/` | `attn-qat-nvfp4-training-attention-h100-v1-seed` | 2: 1 Work + 1 Judge | `--work-gpu 0 --judge-gpu 1` | 12 h | 18601 |
| `sageattention-qattn/` | `sageattention-quantized-attention-bf16-h100-v2-seed` | 2: 1 Work + 1 Judge | `--work-gpu 0 --judge-gpu 1` | 12 h | 18602 |
| `standalone-fp8/` | `standalone-dynamic-grouped-fp8-h100-v3-scratch` | 2: 1 Work + 1 Judge | `--work-gpu 0 --judge-gpu 1` | 12 h | 18603 |
| `standalone-kgrouped-wgrad/` | `standalone-k-grouped-fp8-wgrad-h100-v2-scratch` | 2: 1 Work + 1 Judge | `--work-gpu 0 --judge-gpu 1` | 12 h | 18604 |
| `standalone-paged-gqa-v2/` | `standalone-paged-gqa-decode-bf16-h100-v2-scratch` | 2: 1 Work + 1 Judge | `--work-gpu 0 --judge-gpu 1` | 12 h | 18605 |
| `standalone-sparse-mla-v2/` | `standalone-sparse-mla-prefill-h100-v2-scratch` | 2: 1 Work + 1 Judge | `--work-gpu 0 --judge-gpu 1` | 12 h | 18606 |
| `tilelang-mamba2-ssd-v2/` | `tilelang-mamba2-ssd-forward-h100-v2-seed` | 2: 1 Work + 1 Judge | `--work-gpu 0 --judge-gpu 1` | 12 h | 18607 |
| `flash-kmeans/` | `flash-kmeans-paper-h100-v7-scratch` | 2: 1 Work + 1 Judge | `--work-gpu 0 --judge-gpu 1` | 12 h | 18284 |
| `lmcache-kv-transfer/` | `lmcache-paged-kv-block-transfer-h100-v1-seed` | 2: 1 Work + 1 Judge | `--work-gpu 0 --judge-gpu 1` | 12 h | 18496 |
| `eagle3-spec-decode/` | `eagle3-spec-decode-engine-qwen3-8b-h100-v1-seed` | 2: 1 Work + 1 Judge | `--work-gpu 0 --judge-gpu 1` | 24 h | 18386 |
| `llada-h100/` | `llada-h100-decoder-optimization-v2-vanilla-seed` | 2: 1 Work + 1 Judge | `--work-gpu 0 --judge-gpu 1` | 24 h | 18384 |
| `minference-v3/` | `minference-sparse-prefill-h100-v3-seed` | 2: 1 Work + 1 Judge | `--work-gpu 0 --judge-gpu 1` | 12 h | 18497 |
| `flux-moe-ag/` | `flux-moe-ag-scatter-grouped-gemm-dual-h100-v1-seed` | 8: shared by Work and the Judge | `--gpus 0,1,2,3,4,5,6,7` | 12 h | 18611 |
| `qwen3-orpo/` | `qwen3-1p7b-orpo-training-optimizer-dual-h100-v4-cross-matrix-seed` | 4: 2 Work + 2 Judge | `--work-gpus 0,1 --judge-gpus 2,3` | 24 h | 18613 |
| `qwen3-gkd/` | `qwen3-4b-to-1p7b-gkd-distillation-optimizer-dual-h100-v1-official-trl-seed` | 4: 2 Work + 2 Judge | `--work-gpus 0,1 --judge-gpus 2,3` | 24 h | 18614 |
| `sdv2-pipeline/` | `streamdiffusionv2-dual-h100-pipeline-runtime-v2-seed` | 4: 2 Work + 2 Judge (NVLink pairs) | `--work-gpus 0,1 --judge-gpus 2,3` | 24 h | 18612 |
| `miles-alfworld/` | `miles-qwen3-alfworld-agentic-rl-2xh100-v1-seed` | 4: 2 Work + 2 Judge (NVLink pairs) | `GPU_INDICES=0,1`, `JUDGE_GPU_INDICES=2,3` in the config | 24 h | 18631 |
| `miles-diffusion-opd/` | `miles-diffusion-sd35m-mopd-2xh100-v2-seed` | 4: 2 Work + 2 Judge (NVLink pairs) | `GPU_INDICES=0,1`, `JUDGE_GPU_INDICES=2,3` in the config | 24 h | 18632 |
| `miles-diffusion-wan/` | `miles-diffusion-wan22-dancegrpo-2xh100-v2-seed` | 4: 2 Work + 2 Judge (NVLink pairs) | `GPU_INDICES=0,1`, `JUDGE_GPU_INDICES=2,3` in the config | 24 h | 18633 |
| `miles-opcd/` | `miles-qwen3-opcd-2xh100-v3-seed` | 4: 2 Work + 2 Judge (NVLink pairs) | `GPU_INDICES=0,1`, `JUDGE_GPU_INDICES=2,3` in the config | 24 h | 18634 |
| `miles-opd/` | `miles-qwen3-opd-scratch-2xh100-v2-seed` | 4: 2 Work + 2 Judge (NVLink pairs) | `GPU_INDICES=0,1`, `JUDGE_GPU_INDICES=2,3` in the config | 24 h | 18635 |
| `concurrent-serving/` | `concurrent-agent-serving-optimizer-qwen36-swe192-v24` | 2 for the host vLLM backend; Work and Judge use no GPU | the backend you start (task README) | 24 h | 18188 |
| `quant-videogen/` | `quant-videogen-cross-model-kv-codec-h100-v4-scratch` | 7: 1 Work + 6 Judge (fixed) | `--work-gpu 0 --judge-gpus 1,...,6` | 24 h | 18623 |
| `kvpress-kv-eviction-scratch/` | `kvpress-kv-eviction-policy-qwen3-8b-h100-v2-scratch` | 8: 1 Work + 7 Judge (fixed) | `--work-gpu 0 --judge-gpus 1,...,7` | 24 h | 18621 |
| `kvzip-kv-scoring/` | `kvzip-query-agnostic-kv-scoring-qwen3-8b-h100-v2-scratch` | 8: 1 Work + 7 Judge (fixed) | `--work-gpu 0 --judge-gpus 1,...,7` | 24 h | 18622 |

The Miles defaults are the values in each directory's `*.env.example` launch config. The
sharded-Judge tasks use a fixed Judge GPU count, the paper's setup: quant-videogen uses 1 Work
GPU and 6 Judge GPUs (7 per run), kvpress and kvzip 1 Work GPU and 7 Judge GPUs (8 per run).
`--judge-gpus` (and `JUDGE_GPUS` for `replay.sh`) chooses which GPUs; `run.sh` and `replay.sh`
refuse any other count, a duplicated GPU, or a Work GPU that is also a Judge GPU.

Each directory has a `run.sh`; all but the Miles and MInference v3 directories
also have a `replay.sh`. The single-GPU launchers delegate to the task's
`scripts/run_fixed_budget.sh` (`--shared-gpu`:
`scripts/run_single_gpu_fixed_budget.sh`); kvpress, kvzip and quant-videogen to
`scripts/run_agent.sh`; flux-moe-ag and sdv2-pipeline to
`scripts/launch_fixed_budget_after_gpu_pair_idle.sh`; qwen3-orpo and qwen3-gkd
to their task's fixed-budget queue script; MInference v3 to
`scripts/run_local_v3.sh`; the Miles launchers to `scripts/run_task.py`.

## Shared options

The single-GPU, dual-GPU and multi-GPU-Judge launchers share these flags, with the
exceptions in the next table (`run.sh --help` lists exactly what each one accepts):

| Flag | Meaning |
| --- | --- |
| `--agent codex\|claude-code` | the agent CLI (default `codex`); every launcher but Miles, whose config sets `AGENT` |
| `--model NAME` | a full model ID (`gpt-*`, `claude-*`), or any ID the `MODEL_API_BASE_URL` endpoint serves |
| `--effort LEVEL` | `max` (default), `ultra`, `xhigh`, `high`, `medium` or `low` |
| `--hours N` / `--timeout SECONDS` | agent budget (`TASK_TIMEOUT`; default 12 h or 24 h, table above) |
| `--run-id ID` | explicit run ID (`RUN_ID` in the environment where the flag is missing) |
| `--auth FILE` | Codex `auth.json` (sets `CODEX_AUTH_JSON_PATH`) |
| `--claude-token FILE` | Claude Code OAuth token file (sets `CLAUDE_OAUTH_TOKEN_FILE`) |
| `--judge-port PORT` | host port of the Judge server (`JUDGE_PORT`); each task has its own default (table above) |
| `--cpuset off\|auto\|CPULIST[@NODES]` | opt-in GPU-local CPU pinning, see [CPU pinning](#cpu-pinning-opt-in) |
| `--dry-run` | resolve and check the plan, then exit; where available |
| `--tmux` | run detached in a new tmux session (not on qwen3-orpo, qwen3-gkd, flux-moe-ag or sdv2-pipeline, whose task scripts manage their own queueing) |
| `--config FILE` | launch config file (below) |
| `-- …` | everything after `--` is forwarded verbatim to the task script |

| launchers | `--run-id` | `--judge-port` | `--dry-run` | `--tmux` | GPUs | run directory |
| --- | --- | --- | --- | --- | --- | --- |
| `attn-qat-train`, `eagle3-spec-decode`, `flash-kmeans`, `llada-h100`, `lmcache-kv-transfer`, `sageattention-qattn`, `standalone-fp8`, `standalone-kgrouped-wgrad`, `standalone-paged-gqa-v2`, `standalone-sparse-mla-v2`, `tilelang-mamba2-ssd-v2` | yes | yes | yes | yes | `--work-gpu`, `--judge-gpu`; `--shared-gpu` with `--gpu` or `--device-uuid` | `outputs/<task>/runs/<run id>/` |
| `minference-v3` | yes | yes | yes | yes | `--work-gpu`, `--judge-gpu` | `outputs/<task>/runs/<run id>/` |
| `kvpress-kv-eviction-scratch` | yes | yes | yes | yes | `--work-gpu`, `--judge-gpus` | `outputs/<task>/runs/<run id>/` |
| `kvzip-kv-scoring`, `quant-videogen` | `RUN_ID` | yes | yes | yes | `--work-gpu`, `--judge-gpus` | `outputs/<task>/runs/<run id>/` |
| `qwen3-gkd` | `RUN_ID` | yes | yes | no | `--work-gpus`, `--judge-gpus`; `--shared-gpus`, `--gpus` | `tasks/latest/<task>/logs/runs/<run id>/` |
| `qwen3-orpo` | `RUN_ID` | yes | no | no | `--work-gpus`, `--judge-gpus`; `--shared-gpus`, `--gpus` | `tasks/latest/<task>/logs/runs/<run id>/` |
| `flux-moe-ag` | `RUN_ID` | yes | no | no | `--gpus` (eight GPUs; `--work-gpus` is an alias) | `tasks/latest/<task>/logs/runs/<run id>/` |
| `sdv2-pipeline` | `RUN_ID` | yes | no | no | `--work-gpus`, `--judge-gpus`; `--shared-gpus`, `--gpus` | `tasks/latest/<task>/logs/runs/<run id>/` |
| `miles-*` | yes | `JUDGE_PORT` in the config | yes | detached unless `--foreground` | `GPU_INDICES` and `JUDGE_GPU_INDICES` in the config; `--judge-gpus`, `--shared-gpus` | `<scratch root>/<task>/runs/<run id>/` (task README) |
| `concurrent-serving` | yes | `JUDGE_PORT` | no | yes | host vLLM backend (task README) | `outputs/<task>/runs/<run id>/` |

Single-GPU launchers put Work on `--work-gpu INDEX` (default 0) and the Judge on
`--judge-gpu INDEX` (default 1), as in the paper's runs. `--shared-gpu` runs both on one GPU,
`--gpu INDEX` (default 0; the launcher looks up its UUID) or `--device-uuid UUID`, and pauses
Work while the Judge measures; `--gpu` or `--device-uuid` alone also selects it. Shared runs
also accept `--allocation-key K` and `--lease-id ID` for a scheduler-issued allocation (or
`SFORGE_ACCELERATOR_DEVICE_UUID`, `SFORGE_ACCELERATOR_ALLOCATION_KEY`,
`SFORGE_TRUSTED_LEASE_ID`). `--legacy-dual-gpu` is an older name for the separate default.
All single-GPU launchers accept `--calibration-epoch ID`. `--calibration-epoch` names a qualified clock/power campaign of the card; without
it a run is recorded as `unqualified_missing_calibration_epoch`, which labels the
run's provenance and does not change how it is scored. Dual-GPU launchers put Work on
`--work-gpus A,B` (default 0,1) and the Judge on `--judge-gpus C,D` (default 2,3), and also take
`--judge-port PORT` and `--label TEXT`; see [Dual-GPU tasks](#dual-gpu-tasks-orpo-gkd-streamdiffusionv2)
for their shared mode (Flux: [eight GPUs](#flux-eight-gpus)). They start without a minimum delay unless
`MIN_DELAY_SECONDS` is exported (`--no-wait` is still accepted). Anything not covered by a flag (for example
`WORK_GPU`, `JUDGE_GPU`, `JUDGE_PORT`) is passed through as environment.

```bash
# Flash K-Means with its defaults (Work GPU 0, Judge GPU 1, 12 h, effort max), detached in tmux:
launch/flash-kmeans/run.sh --model <model> --auth /path/to/auth.json --tmux

# A second single-GPU task next to it, on GPUs 2 and 3:
launch/standalone-fp8/run.sh --model <model> --work-gpu 2 --judge-gpu 3

# The same task on one GPU only:
launch/standalone-fp8/run.sh --model <model> --shared-gpu --gpu 4

# ORPO for 6 h with Work on GPUs 4,5 and the Judge on 6,7, next to a run on 0-3:
launch/qwen3-orpo/run.sh --model <model> --work-gpus 4,5 --judge-gpus 6,7 \
  --judge-port 18713 --hours 6
```

`launch/common.sh` is the single source for the Codex CLI pin
(`CODEX_CLI_VERSION_PIN`) and the Claude Code pin (`CLAUDE_CODE_VERSION_PIN`).

`SFORGE_VERSION_STORE_ROOT` defaults to a marked sibling store
(`../agent_task_version_submit` or `../agent-task-version-submit`) when one exists.
The store is not created for you: make one as `launch/mlsysforge.env.example`
describes (a marker file and one `task.json` per task), or set the variable to
an existing store.

## Launch config

Every host-specific setting can come from a config file instead of being
exported by hand. Discovery order (first hit wins):

1. `--config FILE`
2. `$MLSYSFORGE_CONFIG`
3. `launch/mlsysforge.env`
4. `launch/mlsysforge.json`

A `.env` file is `KEY=value` lines with `#` comments; a `.json` file is a flat
object of the same keys. The keys are the real variable names, and a value is
applied only when that variable is not already set, so precedence is
command-line flag > environment > config file > built-in default. Unknown keys
are rejected, so a typo surfaces at launch. Start from
`launch/mlsysforge.env.example`.

The accepted keys are `LAUNCH_CONFIG_KEYS` in `launch/common.sh`:

| Group | Keys |
| --- | --- |
| Version store | `SFORGE_VERSION_STORE_ROOT`, `SFORGE_VERSION_STORE_OPTIONAL`, `SFORGE_VERSION_STORE_SEARCH_PATH` |
| Agent and model | `SFORGE_AGENT_RUNTIME_DIR`, `CODEX_AUTH_JSON_PATH`, `CLAUDE_OAUTH_TOKEN_FILE`, `CODEX_CLI_VERSION`, `MODEL`, `MODEL_API_BASE_URL`, `MODEL_API_KEY_FILE`, `REASONING_EFFORT`, `TASK_TIMEOUT` |
| Container proxy | `SFORGE_HTTP_PROXY`, `SFORGE_HTTPS_PROXY`, `SFORGE_NO_PROXY` |
| GPUs and ports | `WORK_GPU`, `JUDGE_GPU`, `JUDGE_PORT`, `FLUXMOE_GPU_INDICES`, `FLUXMOE_JUDGE_GPU_INDICES`, `FLUXMOE_PAUSE_WORK_FOR_JUDGE`, `ORPO_GPU_INDICES`, `ORPO_GPU_UUIDS`, `ORPO_JUDGE_GPU_INDICES`, `ORPO_JUDGE_GPU_UUIDS`, `GKD_GPU_INDICES`, `GKD_JUDGE_GPU_INDICES`, `SDV2_GPU_INDICES`, `SDV2_JUDGE_GPU_INDICES` |
| Task inputs | `KVBENCH_VLLM_BASE_URL`, `KVBENCH_HIDDEN_BASELINE_PATH`, `KVBENCH_PUBLIC_BASELINE_PATH`, `SFORGE_KVBENCH_NONCE_SOURCE`, `LLADA_MODEL_REPO`, `QVG_ARTIFACT_ROOT`, `ORPO_MODEL_HOST_PATH`, `ORPO_DATA_HOST_ROOT`, `ORPO_PRIVATE_DATA_HOST_ROOT`, `SDV2_ASSETS_ROOT` |
| CPU pinning | `SFORGE_CPUSET`, `SFORGE_WORK_CPUSET`, `SFORGE_JUDGE_CPUSET`, `SFORGE_CPUSET_MEMS`, `SFORGE_CPUSET_RESERVE`, `SFORGE_CPUSET_UNDERSIZE` |
| Resource profile | `SFORGE_RESOURCE_PROFILE` (`native` uses the task JSON's own CPU/memory sizes) |
| Integrity review | `SFORGE_LLM_JUDGE_*` (see [docs/LLM_JUDGE.md](../docs/LLM_JUDGE.md)) |
| Assets and hardware | `INFRATHON_ASSETS_ROOT` (the asset root, default `~/.cache/infrathon/assets`), `INFRATHON_HARDWARE` (`h100-94gb-400w` or `h100-80gb-700w`; detected from GPU 0 when unset), `HF_HUB_CACHE` (the Hugging Face cache public models are fetched into and mounted from) |
| Miles family | the host locations from `HARNESS_ROOT` on, read by `scripts/miles_common/host_config.py` from the same file; with `INFRATHON_ASSETS_ROOT` set they default to the fetched assets |

The image and asset scripts (`assets/fetch.sh`, `scripts/build_images.sh`,
`scripts/infrathon/pull_images.sh`) do not read this file: they take
`INFRATHON_ASSETS_ROOT`, `INFRATHON_HARDWARE` and `HF_HUB_CACHE` from the environment only.
Export them with the same values first, for example `set -a; . launch/mlsysforge.env; set +a`;
otherwise assets land under the defaults (`~/.cache/...`) and the launcher, which reads the
file, does not find them.

## Agent runtimes

Task images are agent-free. The agent CLI comes from a host runtime directory
that the Docker backend mounts read-only at `/opt/agent-runtime` in Work
containers (and in integrity-review containers), so the agent kind and version
are chosen per launch instead of per image build. Build one with:

```bash
launch/build-agent-runtime.sh --agent codex            # version: CODEX_CLI_VERSION_PIN
launch/build-agent-runtime.sh --agent claude-code --version <CLAUDE_CODE_VERSION_PIN>
```

Options: `--version V`, `--node vX.Y.Z` (Node bundled with the CLI, default
`v20.18.0`), `--root DIR` (default `<repo>/../agent_runtimes`) and `--force`.
The result is `<root>/<agent>-<version>/` with `bin/node`, `bin/npm`, the agent
CLI, `env.sh` and `.sforge-agent-runtime.json` (including a tree SHA-256 for run
manifests). Downloads honour the ambient proxy environment.

Launchers use `../agent_runtimes/codex-<pin>` (or `claude-code-<pin>` with
`--agent claude-code`) unless `SFORGE_AGENT_RUNTIME_DIR` is set. To run another
Codex version, set `CODEX_CLI_VERSION` and `SFORGE_AGENT_RUNTIME_DIR` together.

## Credentials

| Agent route | What to provide |
| --- | --- |
| Codex with a ChatGPT login | an `auth.json` (`--auth FILE` or `CODEX_AUTH_JSON_PATH`). Tokens can refresh inside the container, so use a credential file dedicated to automated runs |
| Claude Code | a mode-0600 file holding one `claude setup-token` OAuth token (`--claude-token FILE` or `CLAUDE_OAUTH_TOKEN_FILE`) and the Claude Code runtime |
| Codex against an OpenAI-compatible endpoint | `MODEL_API_BASE_URL` (the endpoint must serve the Responses API, `/v1/responses`) and `MODEL_API_KEY_FILE` (a file holding the key); `--model` then accepts any ID the endpoint serves |

The integrity review has its own credentials (`SFORGE_LLM_JUDGE_CODEX_AUTH_JSON`
or `SFORGE_LLM_JUDGE_API_BASE_URL` + `SFORGE_LLM_JUDGE_API_KEY_FILE`), and
falls back to the run's Codex runtime and `auth.json`. Runs refuse to start
unless the review is configured (`SFORGE_LLM_JUDGE_MODE=enforce`, `SFORGE_LLM_JUDGE_MODEL`); a
configured but broken reviewer shows up only at the first review, see
[docs/LLM_JUDGE.md](../docs/LLM_JUDGE.md#configuration).

## Model relay

Providers without the Responses API (for example OpenRouter models, or a plain
Chat Completions endpoint), or whose keys should not enter the Work container,
can be reached through an optional host relay:

```bash
python -m sforge.harness.relay.model_proxy --config <private>/relay.json
```

The private JSON config names `key_file` (the upstream key, 0600),
`relay_token_file`, `model`, `effort`, `upstream_base`, `vendor`
(`openrouter` allows pinning one upstream `provider` with no fallback),
`wire` (`responses` to forward native Responses calls, `chat` to convert them
to Chat Completions), `host`, `port` and `log_file`. The relay pins the model
and effort on every request; the Work container receives only the relay token.
Listen on an address the Work container can reach, not loopback. Then select it
in a launch config:

```bash
MODEL_API_BASE_URL=http://<relay-host>:<port>/v1
MODEL_API_KEY_FILE=/path/to/relay.token     # the relay token, never the provider key
MODEL=<model>
REASONING_EFFORT=<effort>
```

How hosted web search behaves on each route is described in
[docs/INTERNET_AND_WEB_SEARCH.md](../docs/INTERNET_AND_WEB_SEARCH.md).

## Outbound network

Work containers sit on a per-run internal Docker network behind a restricted
CONNECT sidecar; the task JSON's `internet` flag decides whether the sidecar
also passes public hosts on ports 80/443 (see
[docs/INTERNET_AND_WEB_SEARCH.md](../docs/INTERNET_AND_WEB_SEARCH.md)).

Docker cannot reach a proxy bound to the host's loopback, and SForge falls
back to the ambient `http_proxy`/`HTTP_PROXY` when `SFORGE_HTTP_PROXY` is
unset — which hands the Work container a `127.0.0.1` proxy pointing at itself.
Codex then fails every request with "connection refused" and the run burns its
budget reconnecting. `run.sh` refuses to start in that state (unless the task's
entry script sets its own sidecar proxy); republish the proxy on the Docker
bridge first:

```bash
DOCKER0_IP="$(ip -4 addr show docker0 | grep -oP 'inet \K[0-9.]+')"
python3 tasks/latest/quant-videogen-cross-model-kv-codec-h100-v4-scratch/scripts/host_proxy_bridge.py \
  --listen-host "$DOCKER0_IP" --listen-port 19081 &

export SFORGE_HTTP_PROXY="http://$DOCKER0_IP:19081"
export SFORGE_HTTPS_PROXY="$SFORGE_HTTP_PROXY"
export SFORGE_NO_PROXY="host.docker.internal,localhost,127.0.0.1,$DOCKER0_IP"
```

Pick a port no other run holds. Verify with a throwaway container before
starting a long run:

```bash
docker run --rm -e https_proxy="$SFORGE_HTTP_PROXY" <work-image> \
  curl -sS -o /dev/null -w '%{http_code}\n' --max-time 25 https://chatgpt.com/backend-api/ps/mcp
```

## CPU pinning (opt-in)

Off by default. `--cpuset auto` (or `SFORGE_CPUSET=auto` in the launch config)
makes SForge pin every Work and Judge container it starts for a run, eval or
replay to the CPUs local to that container's own GPU(s), with memory bound to
the GPU's NUMA node. SForge derives the slice from sysfs between creating and
starting each container, so the launcher only checks the value's grammar and
never touches a GPU.

- Precedence is per variable: `--cpuset` > environment > launch config > off.
  `--cpuset` sets only `SFORGE_CPUSET`, so a
  `SFORGE_WORK_CPUSET`/`SFORGE_JUDGE_CPUSET` from the environment or config
  still wins for its role (the launcher warns); `--cpuset off` turns those off
  too.
- `SFORGE_CPUSET_MEMS`, `SFORGE_CPUSET_RESERVE` and `SFORGE_CPUSET_UNDERSIZE`
  tune the policy (documented in `mlsysforge.env.example`).
- Prefer the per-launch flag. `SFORGE_CPUSET=auto` in the launch config
  reaches every `run.sh` and `replay.sh` that loads it, including
  concurrent-serving (unpinned, but its records and replay identity change);
  opt it out with `export SFORGE_CPUSET=off`.
- When a policy is active, the launcher prints the per-GPU slice table
  (`python3 -m sforge.harness.host_topology table`) before delegating.
- Separate-GPU runs (the default of the single-GPU and dual-GPU launchers) run
  Work and Judge concurrently on their own GPUs, so an explicit global CPU list
  is rejected there; `auto` (each role on its own GPUs' slice) or disjoint
  per-role lists work. A shared run puts both roles on the same slice.
- Running a task's `eval_archive.sh`, `smoke_test.sh` or
  `replay_candidates.sh` directly: `export SFORGE_CPUSET=auto` first.
- concurrent-serving, MInference v3 and the Miles launchers have no `--cpuset`.

See [docs/CPU_AFFINITY.md](../docs/CPU_AFFINITY.md) for how slices are derived,
the tasks that stay unpinned, the records it writes and how to verify a
container.

## Replaying saved candidates

```bash
# Preview which candidates a run saved (no GPU):
launch/standalone-fp8/replay.sh --origin-run RUN_ID \
  --measurement-set hidden-v1 --dry-run --json

# Full serial batch, sampling every 2nd candidate (last always measured):
launch/standalone-fp8/replay.sh --origin-run RUN_ID \
  --measurement-set hidden-v1 --every 2

# Fill in the skipped points later against the same pinned plan:
launch/standalone-fp8/replay.sh --origin-run RUN_ID \
  --measurement-set hidden-v1 --resume
```

`replay.sh` injects the default `--version-store-root` and forwards
everything else to the task's `scripts/replay_candidates.sh` (selection:
`--candidate vN`, `--from-v/--to-v`, `--every N`; control: `--resume`,
`--dry-run`, `--json`, `--timeout`). Its own leading flags are `--config FILE`
and `--cpuset off|auto|CPULIST[@NODES]` (not on concurrent-serving); a pinned
replay is a different measurement set from an unpinned one, so pass the same
`--cpuset` for every replay into one `--measurement-set`.

`llada-h100/replay.sh` is a documented stub: that task saves candidates but has
no batch replay launcher (its Judge needs live host artifact fingerprints); the
stub prints the per-candidate `eval_archive.sh` procedure.

## Tasks with a sharded Judge: kvpress, kvzip, quant-videogen

These launchers give Work one GPU and the Judge one or more Judge GPUs:

```bash
launch/<dir>/run.sh --model M [--work-gpu G --judge-gpus A,B,...] [--judge-port P] \
  [--agent claude-code --claude-token FILE] [--hours N] [--dry-run]
```

Without the GPU flags a run takes GPU 0 for Work and GPUs 1-7 for the Judge (kvpress,
kvzip) or 1-6 (quant-videogen), as in the paper's runs.

Every interactive submission and the independent final Judge (written to
`<run>/final-eval/`, with `final-eval.log` and `final-eval.exit`, on the same
Judge GPUs while the run still holds every GPU lock) run the full hidden suite
sharded over the Judge GPUs, with the timeouts and CPU/memory share that the
task's `scripts/lib.sh` gives for that GPU count. The run refuses unless the
`SFORGE_LLM_JUDGE_*` integrity review is configured with `MODE=enforce`;
`--dry-run` prints the resolved plan and checks without locking or starting
anything. `replay.sh` replays saved candidates through the task's
`scripts/replay_candidates.sh`; a single archive is evaluated with the task's
`scripts/eval_archive.sh`.

- **kvpress** evaluates the frozen 5,000-item suite. The model comes from
  `KV_EVICTION_MODEL_REPO` (or `--model-repo DIR`): a Hugging Face cache repo of
  Qwen/Qwen3-8B at the pinned snapshot, mounted read-only into Work and Judge.
  Both host lock names of every GPU (`/tmp/sforge-gpu-<uuid>.lock`,
  `/tmp/mlsysforge-gpu-<uuid>.lock`) are held from launch to the end of the
  final Judge; containers claiming a GPU (and, with `GPU_RESERVATIONS_FILE` +
  `GPU_RESERVATIONS_HOST_ID`, reserved GPUs) are refused before the first lock.
  The Judge server, `sforge run` and the final Judge die with the launcher; on
  any exit the launcher stops them gracefully (bounded), removes every
  container of the run and checks the GPUs are released before the locks go
  (`completion.json` `gpu_release`). The submission cooldown is
  `EVAL_INTERVAL` (default 1800 s). `--eval-archive FILE [--public-smoke]`
  runs no agent and evaluates FILE through the same final-Judge path.
- **kvzip** runs the full 1,812-row suite over exactly 7 identical Judge GPUs (one
  Phase A worker and one trusted Phase B decode per card), with 8N CPU /
  32+8N GiB. The submission cooldown is `EVAL_INTERVAL` (default 300 s).
  `--dry-run` checks images, model, holdout, review, credentials and runtime
  and reports GPU locks read-only.
- **quant-videogen** runs the full 17-case hidden suite on exactly 6 Judge GPUs,
  with the timeout and CPU/memory share of the task's `scripts/lib.sh`
  `judge_plan`. Evaluate an archive alone with `scripts/eval_archive.sh`
  (`JUDGE_GPUS=1,2,3,4,5,6`).

Run outputs land in `outputs/<task directory>/runs/<run id>/`.

## MInference v3

```bash
launch/minference-v3/run.sh --model M [--work-gpu G --judge-gpu J] [options]
launch/minference-v3/run.sh --judge-gpu J --eval-archive FILE [--skip-integrity-review]
```

Work and every Judge (each submission and the final) get one GPU each (default
Work 0, Judge 1). Before
the first run, build or pull the images (`scripts/build_images.sh` or
`scripts/infrathon/pull_images.sh <task>`) and fetch the assets
(`tasks/latest/<task>/assets/fetch.sh`) into `INFRATHON_ASSETS_ROOT`. Agent runs need
the integrity review configured and a Codex (or, with `--agent claude-code`, Claude
Code) agent runtime. Submissions have an
`EVAL_INTERVAL` cooldown (default 300 s). `--eval-archive` runs no agent and
sends FILE through the final Judge path; `--skip-integrity-review` is for
oracle and qualification archives.

## Dual-GPU tasks: ORPO, GKD, StreamDiffusionV2

By default these tasks run Work on one GPU pair and the Judge on a second pair
(four GPUs, held for the whole run), and Work keeps running while the Judge
measures, as in the paper's runs. An 8-GPU host therefore takes two concurrent
runs; each needs its own pairs and Judge port (`--judge-port`).

| Flag | Meaning |
| --- | --- |
| `--work-gpus A,B` | Work's pair (default 0,1) |
| `--judge-gpus C,D` | the Judge's pair, for every Judge round and the final evaluation (default 2,3); must not overlap the Work pair |
| `--shared-gpus` | run Work and the Judge on the Work pair; the host pauses Work while the Judge measures |
| `--gpus A,B` | a shared run on pair A,B (the same as `--work-gpus A,B --shared-gpus`) |

The same settings come from the environment or the launch config:
`ORPO_GPU_INDICES`, `GKD_GPU_INDICES` and `SDV2_GPU_INDICES`
name the Work pair (the only pair of a shared run), and
`ORPO_JUDGE_GPU_INDICES`, `GKD_JUDGE_GPU_INDICES` and `SDV2_JUDGE_GPU_INDICES` name
the Judge pair, or are `shared` for a shared run. ORPO also takes the pairs' UUIDs
(`ORPO_GPU_UUIDS`, `ORPO_JUDGE_GPU_UUIDS`); without them they are looked up with
`nvidia-smi`.
Flags win over both. A Work pair set without a Judge pair still gets the default
Judge pair 2,3, so give each concurrent run both pairs.

Keep each pair inside one NUMA node (`nvidia-smi topo -m`, CPU affinity column)
so `--cpuset auto` gets GPU-local CPUs. StreamDiffusionV2 needs NVLink-connected
pairs. Each container sees its pair as logical devices 0 and 1.

```bash
# ORPO with its defaults: Work on 0,1, the Judge on 2,3
launch/qwen3-orpo/run.sh --model <model>
# Shared mode: ORPO on pair 4,5 only (Work paused during Judge rounds)
launch/qwen3-orpo/run.sh --model <model> --gpus 4,5
# StreamDiffusionV2 in shared mode on pair 6,7
launch/sdv2-pipeline/run.sh --model <model> --work-gpus 6,7 --shared-gpus
```

Each launcher ends with an independent Judge of the final snapshot on the
Judge pair (the Work pair in shared mode), written next to the run log
(`final-eval/`; `final-eval.json` for GKD); the run's result is the best
valid score across the run's submissions and that final evaluation. `replay.sh` measures saved candidates on one pair, the
Work-pair variable (`ORPO_GPU_INDICES` and so on); it ignores the Judge-pair
settings.

Host requirements beyond [docs/HOST_PREREQUISITES.md](../docs/HOST_PREREQUISITES.md):

* On NVSwitch (SXM) hosts, the fabric manager must have been started together
  with the GPUs. A fabric manager restarted on its own (for example after a
  library upgrade) leaves P2P working but breaks CUDA multicast, and every
  symmetric-memory preflight then fails with `the operation cannot be
  performed in the present state`.
* A CUDA `torch` `python3` first on `PATH` for the host preflight probes, with
  `trl==0.12.1` + `transformers` for ORPO's CPU smoke (a dedicated venv works;
  do not put the repository venv first).
* The agent runtimes ship their own `node`; no system node is needed.

Launch config for these tasks: `SFORGE_VERSION_STORE_ROOT`,
`CODEX_AUTH_JSON_PATH`, the ORPO inputs `ORPO_MODEL_HOST_PATH` /
`ORPO_DATA_HOST_ROOT` / `ORPO_PRIVATE_DATA_HOST_ROOT`, `SDV2_ASSETS_ROOT`, the
`SFORGE_LLM_JUDGE_*` reviewer settings. The default resource profile gives
each role the task JSONs' 16 CPU / 96g.

## Flux: eight GPUs

`launch/flux-moe-ag/run.sh` holds eight GPUs with peer access between every pair for the whole
run: an NVSwitch node (H100 SXM), or an H100 NVL host, whose GPUs are bridged in NVLink pairs
and reach the others over PCIe. Work and the Judge share them: every Judge round, and the final
evaluation, pauses Work and measures on all eight, so a host takes one Flux run at a time.

| Flag | Meaning |
| --- | --- |
| `--gpus A,...,H` | the eight GPUs (default 0-7); `--work-gpus` is an alias, `--shared-gpus` is accepted and changes nothing |
| `--judge-gpus`, `--no-pause-work-for-judge` | refused: the Judge needs all eight GPUs, so Work is always paused while it measures |

`FLUXMOE_GPU_INDICES` in the environment or the launch config names the eight GPUs. The
launcher waits until all eight are idle and their locks are free, takes both lock families
for the run, runs the preflight (peer access between every pair, an eight-rank
symmetric-memory probe; the NVLink/PCIe links are recorded) and gives Work and the Judge the
task JSON's 32 CPU / 128g each,
whatever `SFORGE_RESOURCE_PROFILE` says.

```bash
launch/flux-moe-ag/run.sh --agent codex --model <model> --auth <auth.json>
launch/flux-moe-ag/run.sh --agent claude-code --model <claude-model> --gpus 0,1,2,3,4,5,6,7 \
  --claude-token <token-file>
```

The final Judge of the last snapshot runs on the same eight GPUs and is written next to the
run log (`final-eval/`). `replay.sh` measures saved candidates on the eight GPUs named by
`FLUXMOE_GPU_INDICES`. The host needs what the dual-GPU tasks need (a CUDA `torch` `python3`
first on `PATH` for the preflight, a fabric manager started together with the GPUs).

## Miles tasks

The Miles launchers take everything — agent harness, Codex version, model, API
endpoint, effort, budget and GPU pairs — from a config file (the examples set 24 h,
effort `max`, Work GPUs 0,1, Judge GPUs 2,3 and a Judge port per task):

```bash
launch/miles-<task>/run.sh --config CONFIG [--judge-gpus C,D | --shared-gpus] \
  [--run-id ID] [--dry-run] [--foreground]
```

`GPU_INDICES` in the config is Work's NVLink pair. `JUDGE_GPU_INDICES` is the
Judge's own pair (all four GPUs distinct; Work keeps running while a Judge
measures, and the final evaluation runs there too), or `shared` for the shared
mode, in which the Judge measures on the Work pair and pauses Work.
`--judge-gpus C,D` and `--shared-gpus` override it; with neither in the config
or on the command line the launcher refuses to start. The run holds the GPU
locks of every GPU it uses for its whole duration. `AGENT=claude-code` with
`CLAUDE_OAUTH_TOKEN_FILE` in the config runs Claude Code instead of Codex.

Each Miles directory ships `*.env.example` configs to start from. They use
their own fixed-budget supervisors (`scripts/run_task.py` in the task
directory) and their own host locations from the launch config (the Miles
family keys above). Their supervisors rebuild the child environment from an
allowlist. CPU pinning comes from `SFORGE_CPUSET*` in the launching shell's
environment, else from the host launch config (`launch/mlsysforge.env` or
`$MLSYSFORGE_CONFIG`), never from the Miles `--config` file: the qualifications
were measured with `export SFORGE_CPUSET=auto`, and the launchers refuse a
formal run whose pinning differs from the qualification's. With a separate
Judge pair each role is pinned to its own pair's CPUs
([docs/CPU_AFFINITY.md](../docs/CPU_AFFINITY.md#task-notes)).

Without `--foreground` a Miles launcher detaches into a new tmux session and
hands it only the `SFORGE_CPUSET*` values. A tmux server that is already running
starts that session with its own environment, so `INFRATHON_ASSETS_ROOT`,
`HF_HUB_CACHE` and `INFRATHON_HARDWARE` exported in your shell may not reach
it: the dry run passes and the detached run fails at start. Run with
`--foreground` inside a tmux or screen session started from that shell instead.

## concurrent-serving

`launch/concurrent-serving/run.sh` accepts `--agent`, `--model`, `--effort` (default `max`),
`--hours`/`--timeout` (default 24 h), `--run-id`, `--auth`, `--claude-token`, `--tmux` and
`--config`; the task script validates the agent, model, effort and budget
combination. Its Work and Judge containers are GPU-free; the vLLM backend runs
on the host (`KVBENCH_VLLM_BASE_URL` and the other `KVBENCH_*` keys).

