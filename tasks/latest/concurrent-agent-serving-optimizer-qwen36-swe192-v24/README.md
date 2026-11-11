# Concurrent Coding-Agent Serving Optimizer (Qwen3.6-27B, SWE-bench Lite 192)

192 concurrent, multi-turn coding agents work on SWE-bench Lite instances against one fixed
vLLM server hosting `Qwen/Qwen3.6-27B` (tensor-parallel over two GPUs). The agent under test
builds a transparent OpenAI-compatible scheduling gateway that sits between the coding agents
and the model server, and must raise the aggregate rate of valid executable agent steps per
minute. This is a seed task: the Work image starts from a working, protocol-correct,
program-aware admission gateway that the agent improves on or replaces. The `v24` in the task
id is the task revision (2.4).

## The task

The agent works in `/app` and edits `/app/submission` in place; only that directory is
submitted. The Judge launches the gateway as:

```bash
BACKEND_URL=<Judge-selected OpenAI-compatible /v1 endpoint> PORT=9000 /app/submission/launch.sh
```

`launch.sh` stays in the foreground, listens on `0.0.0.0:${PORT}` and serves
`GET /health`, `GET /v1/models` and `POST /v1/chat/completions` (streaming and
non-streaming). It may also accept the lifecycle notification `POST /programs/release`
(not forwarded to the backend). Requests that share an opaque `X-Agent-Run-ID` belong to
one long-lived agent program; the gateway may use that key only for equality.

The Work image also provides:

- `/app/RUNTIME_CONTRACT.json`: static facts about the model-server boot (model shape, vLLM
  limits, conservative hybrid-cache page geometry). The launcher checks the geometry
  against the live backend before the session.
- `/app/RUN_BUDGET.json`: the agent timeout, submission cooldown and expected hidden
  evaluation time for this run, written by the launcher.
- `/app/submission/ARCHITECTURE.md`: the design and experiment ledger the agent keeps.
- `kvbench` (public evaluation client) and `/workspace/protocol_tests/run_protocol_smoke.sh`
  (loopback protocol checks against a fake backend, no performance score).

Rules enforced by the Judge (a violation makes the submission invalid):

- For every accepted `/v1/models` or `/v1/chat/completions` request, exactly one matching
  request goes to `BACKEND_URL`, and its response is returned unchanged: no changes to
  messages, model names, sampling parameters, token limits, response content, usage,
  status, errors, or streaming chunk content and order.
- No synthesized, cached, replayed, truncated, merged, retried, skipped or speculative
  responses; no extra probe requests or hidden background backend work. Client
  cancellation and disconnect propagate to the matching upstream request.
- The gateway infers workload state only from traffic it proxies. It has no access to GPU
  APIs, vLLM `/metrics`, Docker or model sockets, or private workload data, and must not
  depend on the internet at runtime.

`AGENT_TASK.md` additionally asks the agent not to hard-code the workflow count,
concurrency or window timing, and to record each experiment in `ARCHITECTURE.md`.

The model server, weights, agents, sandboxes and runtime contract are out of scope. The
launcher's default agent budget is 24 hours (86400 s); any minute-aligned budget of at least
one hour is accepted.

## Evaluation

The score is `valid_steps_per_min`: valid executable coding-agent steps per minute,
aggregated over all workflows inside a fixed scored window. Higher is better. Selection is
`valid_then_score`: any valid submission beats any invalid one, and valid submissions rank
by raw rate.

The hidden evaluation runs 192 distinct SWE-bench Lite workflows at concurrency 192, at
most 100 steps each, in a concealed nonce-derived order, with 300 s of unscored warm-up
followed by an exact 1800 s scored window. One hidden evaluation takes about 35-45
minutes. Steps completed during warm-up or after the window do not count.

An evaluation can end as `infrastructure_error` within its first minutes: in the unscored
warm-up the benchmark needs three well-formed agent steps from the model within six
attempts, and Qwen3.6 occasionally misses (about one warm-up in three on our reference
host). Nothing has been measured at that point; submit again. Such an attempt counts as an
ordinary submission: the 300 s cooldown runs from when it was submitted.

A submission is valid only if every hard gate passes:

| Gate | Requirement |
| --- | --- |
| `protocol_identity_fixed_window_lifecycle` | the benchmark runner's evidence is valid (admission, readiness, protocol and streaming correctness, workload identity and bounds, window timing, provenance) |
| `backend_one_to_one_audit` | strict one-to-one request audit, reconciled at the window cutoff |
| `paired_request_success` | request success rate >= max(0.98, Direct - 0.02) |
| `paired_valid_step_ratio` | valid-step ratio >= max(0.90, Direct - 0.06) |
| `candidate_p99_gateway_wait` | p99 time a request waits in the gateway <= 0.4 x scored window (720 s for hidden) |
| `paired_backend_work` | backend tokens per valid step >= 0.90 x Direct |
| `throughput_accounting` | the reported rate is consistent and positive |

"Direct" is a pass-through gateway that forwards every request immediately. Its rate is
measured once per deployment on the same backend boot (see Running) and serves as the
reference for the paired gates and as a diagnostic (`speedup_vs_selected_direct`). It is
not a divisor of the score: a valid submission slower than Direct keeps its raw score.
Judge feedback reports the score, the Direct ratio, gate names with pass/fail and coarse
error classes; it never discloses prompts, task identities, per-request traces or the
order.

Submissions:

- Public: `kvbench run --mode candidate --profile public-swe192-distinct` runs the same
  192-workflow, concurrency-192 workload with 300 s warm-up and a 600 s scored window.
  Public runs are unlimited, one at a time, advisory, and never count toward selection.
- Hidden: `sforge-submit --details`. Manual submissions are unlimited, with a 300 s
  cooldown between successful ones; evaluations are serialized. Automatic background
  evaluation is off.
- Final: at the deadline the launcher freezes `/app/submission` into an immutable archive
  and evaluates it after the session. The run's best is the highest valid hidden result
  among the manual submissions and the final archive.

## Hardware

The Work and Judge containers use no GPU (`NVIDIA_VISIBLE_DEVICES=void`; Work 16 CPUs /
64 GB, Judge 4 CPUs / 8 GB). All GPU work happens in an external, operator-run vLLM backend
on two GPUs that one vLLM engine sees as a tensor-parallel pair. The backend
needs 2 devices with compute capability 8.0 or newer and at least 80,000 MiB each (H100 SXM
80 GB or H100 NVL). The serving profile fixes the FP8 KV cache at 30 GiB per card
(`--kv-cache-memory-bytes 32212254720`), so the logical KV capacity does not depend on the
card.

| Hardware type | Status |
| --- | --- |
| `h100-94gb-400w` (H100 NVL 94 GB) | qualified |
| `h100-80gb-700w` (H100 SXM 80 GB) | qualified |

The serving contract's 95,000 MiB per-device field (also in `RUNTIME_CONTRACT.json`) is
informational. Because the KV cache is fixed at 30 GiB per card, the same serving profile
boots on H100 SXM 80 GB cards, which is how `h100-80gb-700w` runs; the task's execution
profile (`portable.execution_profiles`) uses the 80,000 MiB floor.

Scores are comparable only within one measurement profile: a different card or backend
boot needs fresh Direct baselines.

## Running

Paths in steps 2-5 (`scripts/`, `runtime/`, `oracle/`) are relative to this task directory;
the `launch/` commands in step 6 run from the repository root.

Host prerequisites: Docker with Compose v2, the repository virtual environment, `jq`,
`flock`, `tmux` (for `--tmux`), and disk for about 230 GB of SWE-bench images (measured on a
containerd image store; the images' listed sizes add up to far more because they share
layers) plus the model (55.6 GB) and the vLLM image (about 30 GB).

1. Assets: `tasks/latest/concurrent-agent-serving-optimizer-qwen36-swe192-v24/assets/fetch.sh`
   (files under `INFRATHON_ASSETS_ROOT`). It downloads `Qwen/Qwen3.6-27B` at revision
   `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9` from Hugging Face as plain files (about
   55.6 GB, used by the backend as `KVBENCH_MODEL_PATH`) and the Judge-only workload nonce
   file from the Infrathon dataset (mode 0600; passed as
   `SFORGE_KVBENCH_NONCE_SOURCE`). The model lands in
   `$INFRATHON_ASSETS_ROOT/concurrent-agent-serving-optimizer-qwen36-swe192-v24/public/models/Qwen3.6-27B`,
   which is the directory to use as `KVBENCH_MODEL_PATH`. Keep `HF_HUB_CACHE` on the same
   file system as the asset root: the flat copy is then hard links, not a second 55.6 GB.
2. SWE-bench corpus: `fetch.sh` then runs `scripts/stage_swebench_corpus.py`, which
   re-derives the 192-instance selection from `corpus/`, pulls the 192 public
   `swebench/sweb.eval.x86_64.*` images by digest (about 230 GB on disk; images already
   local are reused) and writes `manifest.json` under `KVBENCH_CORPUS_ROOT`. This needs
   Docker access. `INFRATHON_SKIP_CORPUS=1 assets/fetch.sh` fetches only the files; the
   corpus can be staged later with `python3 scripts/stage_swebench_corpus.py` from the task
   directory (it also accepts `--registry-mirror`). These are public images: if the host's
   Docker login is a token scoped to other repositories, pull them anonymously
   (`DOCKER_CONFIG` pointing at a directory whose `config.json` is `{}`).
3. Images: build them in one of two ways:
   - Pull the published Work and Judge images, then build the rest locally:
     ```bash
     INFRATHON_PULL=1 DOCKER_BUILD_NETWORK=host tasks/latest/concurrent-agent-serving-optimizer-qwen36-swe192-v24/scripts/build_images.sh
     ```
     The Judge holds no Judge-only data (its nonce file is mounted at run time), so the
     published public Judge is the whole Judge.
   - Build everything from source by running the same script without `INFRATHON_PULL`.

   Either way the script builds the nested benchmark images (model API proxy,
   benchmark controller, main stack, sandbox worker) and the backend image from
   `images/backend/Dockerfile` (the digest-pinned official `vllm/vllm-openai:v0.24.0`;
   `SKIP_BACKEND_IMAGE=1` skips it). Proxy variables set in the environment are passed to
   the Judge build. The image IDs and source digest go to `build/image-manifest.json`
   under the task's output directory and are re-checked at launch.
4. Backend: start vLLM on two GPUs with the frozen serving profile and export the origin it
   prints:

   ```bash
   export KVBENCH_MODEL_PATH=<asset root>/concurrent-agent-serving-optimizer-qwen36-swe192-v24/public/models/Qwen3.6-27B
   export KVBENCH_BACKEND_GPUS=0,1
   bash scripts/host_vllm_backend.sh start   # prints: export KVBENCH_VLLM_BASE_URL=http://...
   ```

   The backend must list the `target-model` alias in `/v1/models`. The model directory
   must hold real files, not the Hugging Face cache's symbolic links. The backend container
   is named `kvbench-qwen36-27b-backend`; set `KVBENCH_BACKEND_CONTAINER` to use another name.
   `bash scripts/host_vllm_backend.sh stop` stops it.
5. Direct baselines: measure one hidden and one public Direct baseline on this backend boot
   with the frozen pair runner
   `runtime/frozen/harbor_run_scripts/concurrent-agent-serving-optimizer-qwen36-27b-swe192-v2.1-gpu45/run-gpu67-v2.1-pair.py`
   (Harbor is the agent-evaluation framework the nested benchmark was first built for; the
   version and host tokens in these names are fixed legacy names the scripts use), run with
   `--baseline-only` and `--candidate-dir oracle/reference_gateway` (its tree digest
   must match the one pinned in the runner):
   - hidden: `--suite hidden --profile hidden-pressure-192 --request-timeout-sec 6000`,
     with `--workload-nonce` set to line 1 of the nonce file (300 s warm-up, 1800 s scored);
   - public: `--suite public --profile public-swe192-distinct --request-timeout-sec 3600`,
     with line 2 of the nonce file (300 s warm-up, 600 s scored).

   Each leg writes a new `--output` JSON (existing files are refused); export the two paths
   as `KVBENCH_HIDDEN_BASELINE_PATH` and `KVBENCH_PUBLIC_BASELINE_PATH`. The runner checks
   the nonce against its frozen commitment; keep it out of logs. Also export
   `KV_GPU67_V21_OWNER_TOKEN` (any label for the runner's Docker resources; the variable name
   is a fixed legacy name),
   `KVBENCH_VLLM_BASE_URL`, `KVBENCH_CORPUS_ROOT` and `KVBENCH_MEASUREMENT_LOCK_PATH`, the
   latter a lock file you create first (mode 0600, owned by you) at
   `<XDG runtime dir>/mlsysforge-kvbench-qwen36-swe192-locks/benchmark-measurement.lock`.
   The legs have their own validity checks, including a minimum Direct rate of 74.5 valid
   steps per minute, so a backend that is too slow fails here.
6. Run:

   ```bash
   launch/concurrent-serving/run.sh --model <gpt-* model ID> [--hours N] [--tmux]
   launch/concurrent-serving/run.sh --agent claude-code --claude-token FILE \
     --model claude-opus-5-5 [--hours N]
   ```

   Defaults: 24 hours, effort `max`. Options: `--effort LEVEL`, `--timeout SECONDS`,
   `--run-id ID`, `--auth FILE` (Codex `auth.json`, mode 0600), `--config FILE` (an `.env` or
   `.json` launch config; the four `KVBENCH_*` / `SFORGE_KVBENCH_NONCE_SOURCE` variables may go
   there). The launcher checks the agent runtime (`SFORGE_AGENT_RUNTIME_DIR`, built with
   `launch/build-agent-runtime.sh`), the model credentials, a registered version store
   (`SFORGE_VERSION_STORE_ROOT`), a free Judge port, and a host loopback CONNECT proxy that
   the task bridges into a restricted egress proxy for the Work container. It then runs a
   preflight (backend reachability and serving-contract digest, page geometry, nonce
   commitments, corpus manifest, image IDs) before starting the agent. The LLM integrity review
   must be configured too (`SFORGE_LLM_JUDGE_MODE=enforce` and `SFORGE_LLM_JUDGE_MODEL`; see
   `launch/mlsysforge.env.example`): without a reviewer model the Judge does not start. The task's run script
   expects that host proxy at `127.0.0.1:18080` (a fixed value, not a launcher option) and
   checks that it reaches the model API before starting. A host that reaches the internet
   directly can run the CONNECT-only proxy shipped in the repository, and keep it running
   for the whole run (for example in tmux):

   ```bash
   python3 scripts/infrathon/host_connect_proxy.py   # listens on 127.0.0.1:18080
   ```

   Any other proxy on that address works too if it does not close idle tunnels quickly,
   because the agent client can stay silent for long periods on an open tunnel.

Replay and single-archive evaluation:

```bash
launch/concurrent-serving/replay.sh --origin-run RUN_ID --measurement-set NAME \
  --replay-run-id REPLAY_ID [--candidate vN | --from-v N --to-v M] [--resume] [--dry-run --json]
tasks/latest/concurrent-agent-serving-optimizer-qwen36-swe192-v24/scripts/eval_archive.sh <archive.tar.gz> <run_id>
```

`eval_archive.sh` runs the LLM integrity review first, which needs the `SFORGE_LLM_JUDGE_*`
settings; for a trusted archive (the shipped seed, `oracle/reference_gateway`, a negative
control) set `SFORGE_LLM_JUDGE_MODE=off`. An archive holds the `submission/` directory
(e.g. `tar -C images/work -czf seed.tar.gz submission`). The Judge keeps benchmark state in
`~/.cache/harbor-kv-qwen36-swe192-v2-1-gpu67-validation` (a fixed legacy name), which must
exist; give the baseline runner the same directory if you set `HARBOR_KVBENCH_STATE_ROOT` for
it. The nested stack uses one fixed Docker Compose project name, so one Docker daemon hosts
one deployment of this task at a time. The project keeps its volumes after a run; remove them between runs
that use different `KV_GPU67_V21_OWNER_TOKEN` values, and run the scripts with standard input
not attached to a terminal (`</dev/null`), otherwise Compose stops to ask whether to recreate
them.

A full replay needs `SFORGE_KVBENCH_NONCE_SOURCE` and runs the same preflight before each
evaluation. `scripts/smoke_test.sh` and `scripts/public_api_smoke.sh` check the build
without a performance workload (`--with-oracle` / `--with-live` add a real evaluation).

## Files

| Path | Contents |
| --- | --- |
| `images/work/` | Work image: `AGENT_TASK.md`, starter `submission/`, `RUNTIME_CONTRACT.json`, `RUN_BUDGET.json`, public `kvbench` client, protocol smoke tests |
| `images/judge/`, `judge/` | outer Judge image and adapters (`run_judge.py` hidden, `run_public.py` public); the Judge-only nonce file is not in the repository: it ships in the dataset's `judge/` part and is read from `INFRATHON_ASSETS_ROOT` |
| `images/backend/` | recipe for the external vLLM backend image |
| `runtime/frozen/` | the frozen nested benchmark: serving contract, vLLM model profile, benchmark controller with the vendored mini-SWE-agent, model API proxy, sandbox worker, baseline runner |
| `corpus/` | SWE-bench Lite source rows with image digests and the 192-instance selection |
| `oracle/reference_gateway/` | reference gateway used by the baseline runner and the oracle smoke test |
| `scripts/` | build, backend, corpus staging, preflight, launch, replay, smoke and validation scripts |
| `tasks/` | SForge task spec (Judge command, timeouts, resources, submit paths) |
| `assets/` | asset manifest and `fetch.sh` |
| `tests/` | task unit tests |

## Upstream and license

- Model: `Qwen/Qwen3.6-27B` at revision `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`.
  Apache-2.0 (per its model card).
- Serving: vLLM `v0.24.0` (official `vllm/vllm-openai` image, pinned by digest).
- Workload: SWE-bench Lite (`princeton-nlp/SWE-Bench_Lite` at `6ec7bb89`) and its official
  `swebench/sweb.eval.x86_64.*` evaluation images.
- Agent harness: mini-SWE-agent 1.14.4 core (MIT, Kilian A. Lieret and Carlos E. Jimenez),
  vendored with task-local adaptations from the `examples/inference/mini-swe-agent` tree of
  ThunderAgent at commit `a989f24c2d53096e243aae39775637b8ddd0e380`.
- `oracle/reference_gateway/` uses a scheduler extracted from ThunderAgent (MIT, Copyright
  2026 Hao Kang; `THIRD_PARTY_LICENSE.md`). ThunderAgent:
  [ThunderAgent-org/ThunderAgent](https://github.com/ThunderAgent-org/ThunderAgent).
- The task's own code is under the repository's root `LICENSE` (PolyForm Noncommercial 1.0.0).
