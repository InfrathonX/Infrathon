# Rebuild Paged-GQA Decode v2 on a qualified whole-GPU host

This document is the portable contract of the task: what has to be rebuilt,
and how, to run it on another host or platform. With this file,
`docs/HOST_PREREQUISITES.md` (repository root) and this task directory you can
rebuild both images, qualify the deployment and run the task on a fresh host
with an admissible GPU. The task's `README.md` describes what the task is and
how it is scored. You may use a coding agent to carry out the rebuild, but do
not let it redesign the benchmark.

## Submission evaluation policy

`full-final-shared-v3` (`paged-gqa-shared-final-v3`): every admitted `auto-*`,
`agent-*`, `manual-*`, and `candidate-*` round executes the complete frozen
evaluation: a compute-sanitizer memcheck pass over the private edge cases
flagged `sanitize` (first, before any native call), the native private edge
cases of `edge-holdout-v2` (zero-weight hard gates), and the 10-case scored
grid `final-holdout-v2` (seven measured trials and one warm-up per case).
Timing, thresholds, aggregation, and private input isolation are identical
across these rounds. Interactive rounds retain their original identities and
cannot override `FINAL_VERSION`; host replay retains its candidate override.
Local public tests use the disclosed public grid and the public edge cases of
`edge-public-v2`; they never run a private suite.

## Immutable identities

Preserve these scientific identities byte-for-byte:

- the TaskCore and its resource manifest;
- the 10-case public grid and its seeds, the public edge cases
  (`edge-public-v2`), the scored slots and 63-bit seeds frozen at
  `judge/private/final_cases.json` (`final-holdout-v2`) and the private edge
  cases frozen at `judge/private/edge_cases.json` (`edge-holdout-v2`), whose
  digests `portable/final-suite-manifest.json` records;
- the support domain, the FP32 reference with its BF16-reordered tolerance
  scaling (`<=`, a zero limit demands an exact answer), the guard-band,
  input-check and sanitizer memory-safety checks, the decoy-first timing
  protocol, the causal-pair effective-TFLOPS formula and the
  `sqrt(geomean(q1) * geomean(q16))` metric;
- the agent-facing optimization contract.

Both private instances are **frozen, not drawn per build**. The qualification
draws are the files under `judge/private/` (distributed as Judge-only dataset
assets, see below) and are what `build_images.sh` reads by default. They are
Judge-only: staged mode 0600 into a root-only Judge build context, they must never be copied, mounted or described
inside the Work image (`scripts/validate_layout.py` proves no Work build context
admits them, by path and by SHA sweep). The scored shapes themselves are public
— they are the head configurations of FlashAttention's
`hopper/benchmark_split_kv.py` model table — so only each case's `L_max` slot
and seed are private. **Reproduce the instances, do not redraw them:** both
come from seeded generators (`scripts/generate_final_manifest.py` for the
scored slots and seeds, `scripts/generate_edge_manifest.py` for the private and
the public edge cases), so every host that builds from this tree gets the same
digests and the same suite identities. A redraw or a generator change mints a
**new suite identity**; no earlier score is comparable to it. The Work image
starts from only `images/work` plus the pinned `runtime/` build context. The
Judge is built in two layers: a public Judge image from `images/judge`
(`Dockerfile`) and the same runtime context, and a holdout layer on top of it
(`images/judge/Dockerfile.private`) that adds the private build context carrying
both instances. The holdout layer is always built on the host and never
published.

In the public distribution the two Judge-only instances are dataset assets
rather than repository files: `assets/fetch.sh` downloads them into
`${INFRATHON_ASSETS_ROOT}/<task directory>/judge/` and verifies them against
`assets/manifest.json`. With `INFRATHON_ASSETS_ROOT` set, `scripts/lib.sh` points
`FINAL_CASES_PATH` and `EDGE_CASES_PATH` there; otherwise they default to
`judge/private/final_cases.json` and `judge/private/edge_cases.json`.

## 1. Host prerequisites

Everything generic (Docker with the NVIDIA runtime as **default**, the docker
group, `uv`, CPU torch in the repo venv, `jq flock rsync patch git sha256sum
nvidia-smi`, disk, network) is listed once in `docs/HOST_PREREQUISITES.md`.
Check it first.

Task-specific extras:

| Need | Why |
| --- | --- |
| One **whole** NVIDIA GPU per concurrent run | Work and Judge share one card serially under a single UUID-keyed `flock`. MIG identities are rejected by the launcher until their NVML observation and lock-domain behaviour are separately qualified. |
| `compute-sanitizer` able to attach on the card | The memory-safety gate runs the flagged edge cases under `compute-sanitizer --tool memcheck` from the base image's CUDA 12.9 toolkit before any native call. Nothing is installed, but the tool has to support the architecture and be able to attach to CUDA processes. A GPU virtualization hook preloaded into every process (for example HAMi's `libvgpu.so`) prevents that; such a deployment must set `STANDALONE_PAGED_GQA_SANITIZER_MODE=skip-platform-incompatible` in the Judge environment. |
| No native probes of untrusted kernels on a shared host | A kernel with a wild pointer run natively on odd shapes can wedge the host NVIDIA driver for every GPU until a reboot. Run untrusted kernels only through the Judge (sanitizer first) or under `compute-sanitizer`. |
| Network for the two `docker build` invocations | They install packages. Evaluation itself fetches nothing. |
| Network for the agent during a formal run | The task JSON declares `internet: true`, and the launcher enforces it without host iptables: Work is attached to a per-run `--internal` Docker network whose only exit is the `scripts/restricted_proxy.py` sidecar (CONNECT to `api.openai.com`, `auth.openai.com`, `chatgpt.com` on 443 and, with `--allow-public-egress`, to any globally routable host on 80/443, internal addresses refused; the Judge relayed on port 3130; Codex `web_search="live"`). A preflight proves no direct egress, a tunnel to a public host and a 403 for the Docker bridge gateway before Work starts. The sidecar itself needs outbound 80/443 from the host. |
| Behind a proxy: a Docker-reachable proxy address | `docker build` is invoked with `--network host` and forwards `HTTP_PROXY`/`HTTPS_PROXY`/`NO_PROXY` (and the lowercase spellings) as build args. A loopback-only proxy is not reachable from the build. |
| A version store that registers this `task_id` | Without it the run silently keeps no replayable candidates; the launcher fails closed. See §7. |

No `sudo` is needed beyond the container-toolkit step in the host
prerequisites.

## 2. GPU envelope

The admissible-device contract is the `single-cuda-sm80-32g` execution profile
in `tasks/standalone_paged_gqa_decode_bf16_h100_scratch_v2.json`:

| Field | Value |
| --- | --- |
| `vendor` / `count` | `nvidia` / 1 whole GPU |
| `min_compute_capability` | `8.0` |
| `min_memory_bytes` | `34359738368` (32 GiB) |
| `features` | `cuda`, `bf16`, `fp32`, `triton` |
| `software_abi.protocol_id` | `pytorch280-cu129-triton340-source-sanitizer-v2` |
| `calibration_protocol.protocol_id` | `cuda-event-host-wall-max-median-decoy-v2` |

Enforcement, in order of when you meet it:

- `scripts/lib.sh` — launcher preflight requires `GPU_MIN_FREE_MIB` free on the
  target card before anything starts, and `assert_portable_gpu_idle` compares
  free memory against the profile's `min_memory_bytes` at the final-Judge
  barrier.
- `runtime/paged_gqa_decode_runtime.py` — `MIN_COMPUTE_CAPABILITY` and
  `MIN_DEVICE_MEMORY_BYTES` carry the same 8.0 / 32 GiB pair, plus a
  `device_count() == 1` hard check.
- `images/judge/verify_paged_gqa.py` — the same envelope again as the scored
  `hardware_ok` gate: a card below it produces `valid: false`, not a lower score.
- The runtime also carries `BF16_DENSE_PEAK_BOUND_TFLOPS`, a dense-BF16
  throughput ceiling per compute capability (8.0, 8.6, 8.9, 9.0, 10.0, 10.3,
  12.0). A case above it fails `harness_integrity_ok`; an unlisted capability
  has no ceiling. It is a tamper backstop, not a performance threshold.

**Memory floor.** The declared 32 GiB is a design floor, not a measured
peak; it is never lowered to a measured value. Each trial's tensors are
released after the trial, and the Judge additionally runs guarded edge cases
and a compute-sanitizer process.

## 3. External inputs

**No model, dataset, tokenizer, fixture archive or upstream checkout.** Every
tensor is generated in-process from the case seed by the trusted runtime. The
only inputs outside the source tree are the two small Judge-only case lists.

| What | Source | Size | SHA | Env var / path | Access |
| --- | --- | --- | --- | --- | --- |
| Base image `pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel` | Docker Hub, pulled by both Dockerfiles; also supplies the CUDA 12.9 `compute-sanitizer` | not recorded per task; `docs/HOST_PREREQUISITES.md` budgets 17–20 GB per task image pair, base images shared | pinned by tag only | — | public |
| Upstream `Dao-AILab/flash-attention` @ `0251105a2fb19d2957484b7f023cd8c115286ced` and the upstream test values of vLLM, FlashInfer and SGLang | GitHub | — | commit-pinned | — | public; **semantic provenance only, never fetched by any script and never present in either image** |
| Scored slots and seeds (`final-holdout-v2`, 10 cases) | Judge-only dataset asset `judge/final_cases.json` (`judge/private/final_cases.json` in the source tree), see §4 | < 10 KB | `09545c07b972e5d0cad59f2a555ed59a1586dca2f1c45cbc0b79869d1e932132`, recorded under `frozen_instance` in `final-suite-manifest.json` and per build in `build/image-manifest.json` | `FINAL_CASES_PATH` (default `judge/private/final_cases.json`) | `assets/fetch.sh` (Judge-only) |
| Private edge cases (`edge-holdout-v2`, 160, 12 `sanitize`, 1 of them sanitizer-only) | Judge-only dataset asset `judge/edge_cases.json` (`judge/private/edge_cases.json` in the source tree), see §4 | about 60 KB | `9c9cdddf4f460d932518e82a3550aa086210a3a1feb59a75f60ebc133eedc676` | `EDGE_CASES_PATH` (default `judge/private/edge_cases.json`) | `assets/fetch.sh` (Judge-only) |
| Public edge cases (`edge-public-v2`, 40, 4 `sanitize`) | in-tree at `images/work/benchmark/public_edge_cases.json` | about 16 KB | `6218ec657ad47a1c5d18241e17e103e59c6917b672f5aaa42987ba138c3839ba` (in `public-suite-manifest.json`) | — | in-tree (Work image) |

## 4. Frozen instances (by design)

`scripts/build_images.sh` reads both private instances before it builds the
Judge image, so a plain checkout builds the scored and edge suites with no draw
and no operator input. For each instance the build checks that the source is a
regular non-symlink, that its `suite_id` is `final-holdout-v2` /
`edge-holdout-v2` and that it holds the expected case count, installs it at
mode `600` into the named Judge-only build context, re-checks that the staged
copy is mode `600`, owned by the building user and byte-identical, and
re-verifies the digests after the build.

**Expected digests:** the build manifest's scored-instance and edge-instance
digests must equal the frozen ones in `portable/final-suite-manifest.json`
(scored `09545c07b972e5d0cad59f2a555ed59a1586dca2f1c45cbc0b79869d1e932132`, edge `9c9cdddf4f460d932518e82a3550aa086210a3a1feb59a75f60ebc133eedc676`). A different value means you
are not on the frozen instances and your scores are not comparable with scores
on them.

To mint a **new** scored instance deliberately:

```bash
../../../.venv/bin/python scripts/generate_final_manifest.py \
  --output /some/scratch/final_cases.json --redraw --seed <int>
FINAL_CASES_PATH=/some/scratch/final_cases.json bash scripts/build_images.sh
```

The draw is a seeded `random.Random`, never a system random source, so a redraw
is reproducible from its seed alone. `scripts/generate_edge_manifest.py --check`
regenerates both edge instances and compares them byte-for-byte with the
committed files; rewriting them after a generator change is a new suite
identity. After either kind of change, record the new digests in the suite
manifests, then run `scripts/refresh_portable_manifests.py`; scores from the
old instances do not carry over.

Nothing is captured per deployment.

## 5. Build

Run everything from the task root
(`tasks/latest/standalone-paged-gqa-decode-bf16-h100-v2-scratch`).

Pure CPU/meta checks first — no GPU, no Docker. Run them with the **repository
venv interpreter**, named explicitly — a bare `python3` is the system
interpreter, which typically has neither the `docker` SDK nor the repo's
`sforge` package and then fails preflight cases for purely environmental
reasons:

```bash
../../../.venv/bin/python -B scripts/validate_layout.py .
../../../.venv/bin/python -B -m pytest -p no:cacheprovider tests -q
../../../.venv/bin/python -B scripts/generate_edge_manifest.py --check
../../../.venv/bin/python -B scripts/refresh_portable_manifests.py --check
find scripts -maxdepth 1 -name '*.sh' -print0 | xargs -0 -n1 bash -n
```

(`uv run --frozen python …` from the repository root is the equivalent.)

The test suite recomputes the scientific resource digest, the
delivery/build-resource digest, every portable manifest byte digest and the
real suite case counts from executable sources; it also checks that both
frozen instances are the seeded generators' output, that every edge case lies
inside the declared domain, and that no Work build context admits a private
instance. Every test must pass; one admission test is skipped unless
`PGQA_HISTORICAL_CANDIDATES` points at a directory of kept candidates. Treat any failure as
a hard stop. The task JSON deliberately is not a member of either resource
manifest because it is the root document that binds their digests; each
resource manifest also excludes itself to avoid a recursive hash.

Then build:

```bash
bash scripts/build_images.sh
jq -e '.source_digest and .work.id and .judge.id and .final_cases.sha256 and .edge_cases.sha256' \
  ../../../outputs/standalone-paged-gqa-decode-bf16-h100-v2-scratch/build/image-manifest.json
```

With `INFRATHON_PULL=1`, `build_images.sh` pulls the Work and public Judge images
published for this host's hardware type and refuses unless they were published
from this exact image-driving source (`scripts/infrathon/images.sh`); without it, it
builds both from source. The Judge holdout layer is built locally in either
mode. Environment it reads: `INFRATHON_PULL`, `INFRATHON_ASSETS_ROOT`, `FINAL_CASES_PATH`
and `EDGE_CASES_PATH` (optional, §4), the six proxy variables, and `PYTHON_BIN` /
`SFORGE_BIN` if the repo venv is not at the default location. It
re-execs itself through `sg docker` when your shell is not yet in the docker
group; use `umask 022` in that case.

Keep the resulting source digest, the exact Work/Judge image IDs and both
instance digests with the run. Do not substitute a pre-existing tag without
verifying its recorded image ID — `run_fixed_budget.sh` recomputes them and
refuses a stale identity.

If `REBUILD_PROMPT.md` itself or anything else in
`portable/build-resource-manifest.json` changes, run
`../../../.venv/bin/python scripts/refresh_portable_manifests.py` and confirm
with `--check`: the task JSON's `delivery.build_resource_manifest_sha256`
moves, the scientific `portable.task_core.resource_manifest_sha256` must not.

## 6. Post-build verification

```bash
bash scripts/make_oracle_archive.sh
PHYSICAL_GPU=<index> bash scripts/smoke_test.sh --oracle
```

**This GPU step is the qualification of the deployment — a build that stops at
`build_images.sh` plus the `jq` gate has proved nothing about the card.**
`PHYSICAL_GPU` is the index of the card allocated to you (`nvidia-smi -L` lists
them); `smoke_test.sh` defaults to card **0**, so pass it explicitly on any host
with more than one GPU or you will qualify the wrong card — or collide with
another run.

What `smoke_test.sh --oracle` does, in order: re-runs the static gates, checks
image/source freshness, takes the card's UUID-keyed `flock` and refuses to start
unless the GPU is idle, runs the Oracle candidate through the
participant-visible public runner inside the Work image, packs `oracle/` with
`scripts/make_oracle_archive.sh`, and evaluates that archive through
`scripts/eval_archive.sh` with the trusted Judge: the sanitizer pass, the
native private edge cases and the scored grid.

**Cost:** one whole idle GPU; the design target for one Oracle Judge evaluation,
including the sanitizer pass, is at most 6 minutes on H100.

Expected readings:

| Reading | Expected |
| --- | --- |
| Gate table | all `PASSED`; the terminal shows **9/9** (8 gates plus a `performance` record) |
| Sanitizer pass | runs (mode `required`) and reports no error |
| Private edge cases | all pass, before any scored trial |
| Public runner banner | prints `FINAL_HOLDOUT NOT_RUN` |
| Oracle scores | public full-suite and Judge effective TFLOPS; record both as this deployment's Oracle band |

Oracle TFLOPS values belong to one measurement profile only. On a different
card the numbers simply differ — nothing in the tree gates on a minimum TFLOPS
(the only throughput check is the plausibility ceiling in §2). A failing *gate*
is a real problem; a different *score* is not. Record whatever your card
produces as this deployment's Oracle band before any agent run.

## 7. Formal run

Qualify the card first (above), then launch through the repository launcher
`launch/standalone-paged-gqa-v2/run.sh`. Never hand-assemble the environment block: a
hand-written one silently skips the proxy-reachability guard, the
version-store resolution and the agent-runtime guard.

```bash
launch/standalone-paged-gqa-v2/run.sh \
  --device-uuid GPU-00000000-0000-0000-0000-000000000000 \
  --calibration-epoch campaign-qualified-clock-policy-v1 \
  --judge-port 18096 \
  --model <model> --effort max --hours 2 \
  --auth /path/to/auth.json
```

- `--model` takes a short alias defined in `launch/common.sh` or a full model
  ID; `--effort` takes `max|ultra|xhigh|high|medium|low`.
- `--hours N` and `--timeout SECONDS` are alternatives (the latter wins);
  default 7200 s.
- `--device-uuid` must be a canonical whole-GPU UUID
  (`GPU-` + 8-4-4-4-12 lowercase hex). `--allocation-key` and `--lease-id` are
  optional; the wrapper computes the canonical `allocation-sha256` from the
  device set and refuses a supplied key that does not match it, and generates a
  never-reused lease ID when one is not supplied.
- Single-allocation mode is the default and **rejects** `WORK_GPU`/`JUDGE_GPU`
  with no fallback; a two-card path exists only behind `--legacy-dual-gpu`.
- `--tmux` runs detached; `--run-id ID` overrides the derived `RUN_ID`.

**Budget / interval / cooldown contract.** `scripts/run_fixed_budget.sh` reads
`TASK_TIMEOUT` (default 7200 s) and `EVAL_INTERVAL` (default 300 s) and passes
the *same* `EVAL_INTERVAL` value to both `--eval-interval` and
`--submission-cooldown`, with `--disable-auto-eval` always set. Constraints it
enforces: `TASK_TIMEOUT >= 90`, `TASK_TIMEOUT > EVAL_INTERVAL`, and both
multiples of 60. Frozen, not tunable: `disable_auto_eval` (in-run hidden
evaluations must stay at 0) and the timeout ladder (candidate worker,
compute-sanitizer pass, verifier, Judge adapter, standalone final: 7200 s,
780 s, 8100 s, 8200 s, 8400 s; candidate plus sanitizer plus 60 s fits inside
the verifier budget).

**Ports.** `JUDGE_PORT` (or `--judge-port`) must be 1024–65535 and must not be
3128, 3130 or 3131, which are reserved for the sidecar. The launcher checks the
port is actually free before Work starts and dies otherwise. Default 18096 is
fine for a single run; a scheduler starting concurrent runs on one host must
hand every run its own free port.

**Version store.** `SFORGE_VERSION_STORE_ROOT` must point at a directory holding
`.sforge-version-store.json` that registers this `task_id` (a
`<store>/tasks/<task-directory>/task.json` with
`{"schema_version": 1, "task_directory": "...", "task_id": "standalone_paged_gqa_decode_bf16_h100_scratch_v2"}`).
When unset, `launch/common.sh` searches the conventional store locations beside
the checkout — or `SFORGE_VERSION_STORE_SEARCH_PATH`, a colon-separated list of
absolute paths — and fails closed on ambiguity or absence.
`SFORGE_VERSION_STORE_OPTIONAL=1` downgrades absence to a warning for a run that
deliberately keeps nothing. `RUN_ID` must be unique inside the store: reusing
one raises `VersionStoreCollision` when the final version is persisted after the
deadline, wasting the whole budget.

**Proxy.** The launcher exports its own `SFORGE_HTTP_PROXY`/`SFORGE_HTTPS_PROXY`
(the restricted sidecar on the internal network) and unsets any ambient
`http_proxy`/`HTTP_PROXY` for the host side, so operator proxy variables are
not forwarded to Work. With an ambient loopback-only proxy exported,
`launch/common.sh` still refuses to start: unset `http_proxy`/`HTTP_PROXY` first
(the sidecar does not need it).

**Sanitizer mode.** Leave `STANDALONE_PAGED_GQA_SANITIZER_MODE` unset (that is
`required`) on any host where compute-sanitizer can attach. Set
`skip-platform-incompatible` only on a platform that prevents attach, and plan a
`required`-mode host replay of the final archive.

**Other environment** honoured through the docker-group re-exec:
`CODEX_AUTH_JSON_PATH`, `CODEX_CLI_VERSION` (pinned by `launch/common.sh`),
`FINAL_EVAL_TIMEOUT` (default 8400), `GPU_MIN_FREE_MIB`,
`SFORGE_AGENT_RUNTIME_DIR`.

## 8. Run artifacts and success criteria

```text
outputs/standalone-paged-gqa-decode-bf16-h100-v2-scratch/runs/<RUN_ID>/
  completion.json  experiment-manifest.json  run_config.json
  judge-server.log  sforge-run.log
  <task_id>/  resource_lease.json  formal-precheck.json  formal-postcheck.json
              final_archive.tar.gz  final_result.json  final_snapshot.json
              run_history.json  final-<N>m-eval.log  run_agent.log
              final-<N>m-private-judge-artifacts.tar.gz
              submissions/agent-N/{submission.tar.gz,report.json}
outputs/.../runs/<RUN_ID>-final-<N>m/<task_id>/submissions/manual-1/report.json
```

Success is all of:

- `resource_lease.json` has `state` = `CLOSED`;
- `formal-precheck.json` and `formal-postcheck.json` both PASS (the postcheck
  also pins the evaluated archive digest and the private Judge artifact's suite
  identities, manifest digests and case counts);
- `completion.json` has `status = passed`, `sforge_run_exit = 0`,
  `final_eval_exit = 0`, `hidden_evaluations_during_work = 0` and
  `portable_contract.execution_profile_id = single-cuda-sm80-32g`;
- the standalone final report `submissions/manual-1/report.json` has
  `valid = true` with 9/9 records green.

`completion.json` also carries `raw_score_comparison`: the ordered pair of the
evaluation-contract SHA-256 and the measurement-profile comparability key, with
`policy = same_evaluation_contract_and_measurement_profile_only`. Raw effective
TFLOPS may be compared only when that identity is identical and
`qualification_status` is not `unqualified_missing_calibration_epoch`. Scores
from any other evaluation contract, including earlier versions of this task,
are never comparable. Record the stable measurement profile with the run
(model, compute capability, memory, driver, both image IDs, software ABI,
calibration protocol and the scheduler-issued calibration epoch); record UUID
and observed current clocks only in the deployment attestation.

## 9. When something fails

In this order:

1. `sforge-run.log` — agent/Work lifecycle and the SForge run exit code.
2. `judge-server.log` — Judge registration and per-submission evaluation; a
   `403 Forbidden` here usually means two runs shared a Judge port.
3. `run_agent.log` — what the agent actually did inside Work.
4. `final-<N>m-eval.log` — the standalone final evaluation.
5. `formal-precheck.json` / `formal-postcheck.json` — they print the specific
   rejection reason (archive digest, hidden-evaluation count, manifest digest,
   case count, evaluation mode).
6. The private Judge artifact copied next to the run holds the per-case detail
   the agent never sees.

Other things that bite:

- A run that dies in preflight with exit 78 failed the harness-contract or
  portable-digest check before touching Docker or the GPU; the message names the
  mismatching item.
- An invalid Judge report names the first failing gate; gates after it were not
  evaluated. The agent sees only the number of failed private edge cases, never
  which ones.
- A `memory_safety_ok` failure in `required` mode on a new platform may mean
  compute-sanitizer cannot attach there; set the deployment's sanitizer mode as
  in §7 rather than editing the Judge.
- If `nvidia-smi` hangs, stop all GPU work on the host at once: a wedged driver
  blocks every card and every session until it is reset.
- Serialising runs on one card: wait for the previous run's containers to be
  really gone, not just for the script to return.
- `scripts/replay_candidates.sh` re-evaluates a kept candidate series without a
  new agent run; `scripts/status.sh` and `scripts/validate_formal_artifacts.py`
  need an explicit log root when the outputs directory is not the default.

## 10. Moving to another GPU

Re-measure on the new card: the Oracle public and Judge TFLOPS bands (§6), the
NVML memory peaks, and the timeout head-room if the card is materially slower
than an H100. Re-check `GPU_MIN_FREE_MIB` against the new card's total memory,
confirm that the sanitizer pass completes there, and look up its compute
capability in the plausibility table (§2): an unlisted capability runs without
that ceiling, and adding a row is a TaskCore change.

Nothing else changes: the ABI, the support domain, the five-cell/two-family
grid and every `L_max` slot, every public and private edge case, the tolerance
rule and the metric are device-independent, and both references are recomputed
on the target device inside every evaluation. Rebuild both images on the new
host (or transport them with `docker save`/`docker load` — byte-identical
images do **not** by themselves qualify a different GPU or driver), then run §5
and §6 before any agent run. A new GPU model, driver, image ID, software ABI,
calibration protocol or campaign epoch creates a new measurement-profile
comparability key even when the portable TaskCore digest is unchanged.

Candidates cannot query the device (the admitted host subset has no
`torch.cuda.*`), so tile, `num_warps` and `num_stages` choices are hard-coded; a
submission tuned on one architecture still runs on another but is not retuned.
That is the design, and one more reason raw TFLOPS are profile-local.
