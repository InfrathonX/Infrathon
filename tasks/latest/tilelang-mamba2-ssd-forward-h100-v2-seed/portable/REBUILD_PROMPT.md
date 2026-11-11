# Rebuild Mamba-2 SSD Forward v2 on a qualified whole-GPU host

This document is the portable contract of the task: what has to be rebuilt,
and how, to run it on another host or platform. With this file,
`docs/HOST_PREREQUISITES.md` (repository root) and this task directory you can
rebuild both images, qualify the deployment and run the task on a fresh host
with an admissible GPU. The task's `README.md` describes what the task is and
how it is scored. You may use a coding agent to carry out the rebuild, but do
not let it redesign the benchmark.

## Evaluation policy

`tilelang-mamba2-v2-shared-final` (`full-final-shared-v2`): interactive
`sforge-submit` rounds and the sealed deadline re-evaluation both run the same
complete frozen `tilelang-ssd-final-v3` suite: 24 zero-weight hard-gate edge
fixtures and 138 scored cases in four workload families (batch, serving,
long, real), 3 measured trials (1 on a captured layer) and 1 warm-up per case,
with the same precision gates, TileLang-only operator record, timing and
aggregation. Candidate round identities and override permissions remain
separate: interactive rounds cannot override `FINAL_VERSION`. The final is a
fresh same-suite re-evaluation of sealed code. The public suite
(`tilelang-ssd-public-v3`) is separate: 21 edge fixtures and 24 public cases in
the same four families with disclosed seeds, sharing no head geometry or length
with the final suite.

## Immutable identities

Preserve these scientific identities byte-for-byte:

- the TaskCore and its resource manifest;
- the seeded `kernel_v001`, the 24 edge fixtures, the 24 public cases and the
  shared interactive/final suite whose cases, seeds and checked trials are
  frozen at `judge/private/final_cases.json` (digest recorded as
  `frozen_instance` in `portable/final-suite-manifest.json`), and the 24
  captured layers of its real cases (`real_layers.manifest.json`, bound to
  `fixtures/recipe/real_layers/REAL_LAYERS.json`);
- the FP32 chunked SSD reference, its tolerances, the decoy-warmed timing
  protocol, the warm-up/workspace budget gates, the TileLang-only operator
  record, the effective-TFLOPS formula and the family-geomean metric;
- the agent-facing optimization contract.

The scored shapes are Judge-only: 15 deployed head geometries (five of them
per-rank tensor-parallel layouts) at batch 1 to 16 and `L` from under 1,000 to
262,144, with serving lengths drawn from real request traces. The portable final
manifest records each case's family, tier and weight but withholds its shape.

The final instance is **frozen, not drawn per build**. The qualification draw
is the file `judge/private/final_cases.json` and is what `build_images.sh`
reads by default; `final-suite-manifest.json` records its SHA-256 under
`frozen_instance`. It is Judge-only: it is staged mode 0600 into a root-only
Judge build context and must never be copied, mounted or described inside the
Work image (`scripts/validate_layout.py` proves no Work build context admits it,
by path and by SHA sweep). **Reproduce it, do not redraw it:**
`final-suite-manifest.json` freezes the shapes, the draw policy, the seed range
and the two budget gates. `scripts/generate_final_manifest.py --redraw --seed
<int>` draws 18 reproducible 63-bit case seeds from a seeded `random.Random`;
that is a **new suite identity** whose `frozen_instance` record must be updated,
and no earlier score is comparable to it. `FINAL_CASES_PATH` overrides the
default when you deliberately want to build against another instance. Final
scores from two different instances are not scores on the same inputs, so always
compare by `final_cases.sha256`.

The Work image starts from only `images/work` plus the pinned `runtime/` build
context. The Judge is built in two layers: a public Judge image from
`images/judge` (`Dockerfile`) and the same runtime context, and a holdout layer
on top of it (`images/judge/Dockerfile.private`) that adds the private
`final_cases` context. The holdout layer is always built on the host and never
published.

In the public distribution the Judge-only case list is a dataset asset rather
than a repository file: `assets/fetch.sh` downloads it into
`${INFRATHON_ASSETS_ROOT}/<task directory>/judge/final_cases.json` and verifies it
against `assets/manifest.json`. With `INFRATHON_ASSETS_ROOT` set, `scripts/lib.sh`
points `FINAL_CASES_PATH` there; otherwise it defaults to
`judge/private/final_cases.json`.

## 1. Host prerequisites

Everything generic (Docker with the NVIDIA runtime as **default**, the docker
group, `uv`, CPU torch in the repo venv, `jq flock rsync patch git sha256sum
nvidia-smi`, disk, network) is listed once in `docs/HOST_PREREQUISITES.md`.
Check it first.

Task-specific extras:

| Need | Why |
| --- | --- |
| One **whole** NVIDIA GPU per concurrent run with **73 GiB or more** device memory | The floor is the measured peak with only rounding head-room (§2). Work and Judge share the card serially under one UUID-keyed `flock`; the runtime also requires exactly one visible CUDA device. MIG identities are rejected by the launcher. |
| A **`-devel`** CUDA base image — already pinned, do not "slim" it | TileLang shells out to `nvcc --cubin -arch=sm_XXa -std=c++20` on the first call. A driver-only base fails with `fatal error: cuda_runtime.h: No such file or directory`. |
| About 40 GB of Docker disk for this task alone | Each of the two images is about 19 GB. |
| 8 CPUs and 48 GiB host RAM for each of Work and Judge | From the task JSON `work` / `judge` limits. |
| Network for the two `docker build` invocations | They pull the base image and `pip install tilelang==0.1.13`. |
| Network for the agent during a formal run | Work never has a direct route out. The launcher puts Work on an `--internal` Docker network whose only other member is a restricted CONNECT-proxy sidecar (`scripts/restricted_proxy.py`) that relays the Judge port on 3130 and dials its upstreams directly from the default Docker bridge (no host proxy). The task JSON declares `internet: true`, so the sidecar runs with `--allow-public-egress`: CONNECT to any globally routable host on ports 80/443, while loopback, private, Docker-bridge, link-local and other non-global addresses are refused; Codex runs with live web search. The `agent_query` ban on copying external implementations is enforced by the prompt and by source admission. Judge containers run `--network none`. |
| Behind a proxy: a Docker-reachable proxy address | `docker build` runs with `--network host` and forwards `HTTP_PROXY`/`HTTPS_PROXY`/`NO_PROXY` (and the lowercase spellings) as build args; a loopback-only proxy is not reachable from the containers. |
| A version store that registers this `task_id` | Without it the run silently keeps no replayable candidates; the launcher fails closed. See §7. |

No `sudo` is needed beyond the container-toolkit step in the host
prerequisites.

## 2. GPU envelope

The admissible-device contract is the `single-cuda-sm80-73g` execution profile
in `tasks/tilelang_mamba2_ssd_forward_h100_seed_v2.json`:

| Field | Value |
| --- | --- |
| `vendor` / `count` | `nvidia` / 1 whole GPU |
| `min_compute_capability` | `8.0` |
| `min_memory_bytes` | `78383153152` (73 GiB) |
| `features` | `cuda`, `bf16`, `fp32`, `tilelang`, `nvcc` |
| `software_abi.protocol_id` | `pytorch280-cu129-triton340-tilelang0113-nvcc-v1` |
| `calibration_protocol.protocol_id` | `cuda-event-host-wall-max-median-v1` |

Enforcement, in order of when you meet it:

- `scripts/lib.sh` — `assert_portable_gpu_idle` reads the profile's
  `min_memory_bytes` directly and is the authoritative host-side admission.
  The coarser idle check in the same file (`assert_gpu_idle`) defaults
  `GPU_MIN_FREE_MIB` to 74752 MiB, the 73 GiB floor, so an under-sized or busy
  card fails fast there too.
- `runtime/mamba2_ssd_runtime.py` — `MIN_COMPUTE_CAPABILITY = (8, 0)` and
  `MIN_DEVICE_MEMORY_BYTES = 73 * 1024**3`, plus an "exactly one visible device"
  check.
- `images/judge/verify_ssd.py` — the same two constants again as the scored
  `hardware_ok` gate, with a task test asserting the two copies agree. A card
  below the envelope yields `valid: false`, not a lower score.

**Why 73 GiB.** The v2 grid's measured peak device memory (NVML sampling at
1 Hz with per-PID attribution, over complete final evaluations on idle H100
cards) was 74323 MiB (72.58 GiB), driven by an unchunked output comparison.
v3 compares in chunks and caps the candidate workspace at 24 GiB; its largest
case (`x` at about 2^31 elements) holds three tensor sets of 8.9 GiB, an FP32
reference of 7.8 GiB and the workspace. The v3 suite ran end to end on 80 GB
H100 SXM cards with the seed, the oracle and an optimized kernel, but its NVML
peak has **not** been re-measured, so the 73 GiB floor is kept unchanged. The
memory constants are inside the image source digest: changing them requires
regenerated portable digests, regenerated protected hashes and an image rebuild.

## 3. External inputs

**No model, tokenizer or upstream checkout at run time.** Every generated
tensor comes in-process from the case seed by the trusted runtime; the 24 real
cases read layers captured once with `fixtures/recipe/real_layers/` (pinned
Granite-4.0-H-Micro, Bamba-9B-v2 and Granite-4.0-H-350M weights on WikiText-103,
Apache-2.0 models, CC-BY-SA-3.0/GFDL text). `scripts/build_images.sh` fails when
a *tool* is missing (reachable Docker with your user in the `docker` group,
`jq`, `sha256sum`, and an executable `sforge` and python in the repo venv), when
the Judge-only case list is absent, or when either capture directory is missing
or was not produced by the shipped `REAL_LAYERS.json`.

| What | Source | Size | SHA | Env var / path | Access |
| --- | --- | --- | --- | --- | --- |
| Base image `pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel` | Docker Hub, pulled by both Dockerfiles | each of this task's two built images is about 19 GB | pinned by tag only | — | public |
| `tilelang==0.1.13` wheel (bundles CUTLASS headers) | PyPI, `pip install` inside both Dockerfiles | — | version-pinned | — | public |
| Upstream `tile-ai/tilelang` @ `f935b2ad27152c9d970b15c666acaf3ed73e5f2a` | GitHub | — | commit-pinned | — | public; **read-only reference only** — its three submodules are uninitialised, it is not a buildable source bundle, and nothing is built from it |
| Final instance (`tilelang-ssd-final-v3`: 138 scored cases with seeds and checked trials) | Judge-only dataset asset `judge/final_cases.json` (`judge/private/final_cases.json` in the source tree), see §4 | 39 KB | `f7918c71cceab54de5cd72e949005b080d7e58ab767f2f49377094ded5f07379`, recorded in `final-suite-manifest.json` under `frozen_instance` and again per build in `build/image-manifest.json` | `FINAL_CASES_PATH` (default `judge/private/final_cases.json`) | `assets/fetch.sh` (Judge-only) |
| Oracle candidate under `oracle/` | in the repository | small | — | `scripts/make_oracle_archive.sh` | host-only qualification reference, **not a build input** |

The enumeration of the pinned TileLang wheel's capabilities and escape hatches
that source admission rejects is not distributed. It is not needed to build,
qualify or run the task: the rejection list itself lives in
`images/judge/source_admission.py`.

## 4. Frozen instance (by design)

`scripts/build_images.sh` reads the final instance before it builds the Judge
holdout layer. `FINAL_CASES_PATH` defaults to `judge/private/final_cases.json`
(or the asset path under `INFRATHON_ASSETS_ROOT`, §Immutable identities) — the
qualification draw, one 63-bit seed from `[1, 9223372036854775807]` per scored
case — so a plain checkout builds the scored suite with no draw and no operator
input. The build checks the source is a regular non-symlink, that its
`suite_id` is `tilelang-ssd-final-v3` and that it holds exactly 138 cases,
installs it at mode `600` into a named Judge-only build context, re-checks that
the staged copy is mode `600`, owned by the building user and byte-identical,
and re-verifies both the image-driving source digest and the instance digest
after the build — any drift is a hard `die`.

**Expected digest:** `final_cases.sha256` in the build manifest must equal
`f7918c71cceab54de5cd72e949005b080d7e58ab767f2f49377094ded5f07379`, the
`frozen_instance.sha256` in `portable/final-suite-manifest.json`. A different
value means you are not on the frozen instance and your scores are not
comparable with scores on it.

To mint a **new** suite identity deliberately:

```bash
../../../.venv/bin/python scripts/generate_final_manifest.py \
  --output /some/scratch/final_cases.json --redraw --seed <int>
FINAL_CASES_PATH=/some/scratch/final_cases.json bash scripts/build_images.sh
```

The draw is a seeded `random.Random` (default seed:
`int.from_bytes(sha256("<suite_id>|<task_id>").digest()[:8], "big")`), never a
system random source, so a redraw is reproducible from its seed alone. Record
the new instance under `frozen_instance` before comparing any score against it;
scores from the old instance do not carry over.

Nothing is captured per deployment.

## 5. Build

Run everything from the task root
(`tasks/latest/tilelang-mamba2-ssd-forward-h100-v2-seed`).

Pure CPU/meta checks first — no GPU, no Docker. Run them with the **repository
venv interpreter**, named explicitly — a bare `python3` is the system
interpreter, which typically has neither the `docker` SDK nor the repo's
`sforge` package and then fails preflight cases for purely environmental
reasons:

```bash
../../../.venv/bin/python -B scripts/validate_layout.py .
../../../.venv/bin/python -B -m pytest -p no:cacheprovider tests -q
../../../.venv/bin/python -B scripts/refresh_portable_manifests.py --check
find scripts -maxdepth 1 -name '*.sh' -print0 | xargs -0 -n1 bash -n
```

(`uv run --frozen python …` from the repository root is the equivalent.)

The test suite recomputes the scientific resource digest, the
delivery/build-resource digest, every portable manifest byte digest and the
real suite case counts (36 / 36 / 36, each 18 zero-weight edge gates plus 18
scored rows) from the runtime, verifier and generator. It also pins the Judge
adapter's edge-case count to the trusted runtime — a stale copy there renders
every gate as ERROR instead of PASS, which reads like an infrastructure fault.
Treat any failure as a hard stop.

The task JSON intentionally is not a member of either resource manifest because
it is the root document that binds their digests; each resource manifest also
excludes itself to avoid a recursive hash.

Then build:

```bash
bash scripts/build_images.sh
bash scripts/make_oracle_archive.sh
jq -e '.source_digest and .work.id and .judge.id and .final_cases.sha256' \
  ../../../outputs/tilelang-mamba2-ssd-forward-h100-v2-seed/build/image-manifest.json
```

With `INFRATHON_PULL=1`, `build_images.sh` pulls the Work and public Judge images
published for this host's hardware type and refuses unless they were published
from this exact image-driving source (`scripts/infrathon/images.sh`); without it, it
builds both from source. The Judge holdout layer is built locally in either
mode. Environment it reads: `INFRATHON_PULL`, `INFRATHON_ASSETS_ROOT`, `FINAL_CASES_PATH`
(optional, §4), the six proxy variables, and `PYTHON_BIN` / `SFORGE_BIN` if the
repo venv is not at the default location. It re-execs itself through
`sg docker` when your shell is not yet in the docker group; use `umask 022` in
that case.

Products: the Work image `tilelang_mamba2_ssd.work.<task_id>:local`, the public
Judge image `tilelang_mamba2_ssd.judge-public.<task_id>:local`, the Judge image `tilelang_mamba2_ssd.judge.<task_id>:local` (public
Judge plus holdout layer), the build manifest (`schema_version: 2`, with
`source_digest`, the Work, Judge and public-Judge image IDs and
`final_cases.{sha256,suite_id,case_count}`) and the build logs under the same
outputs directory. The source digest and the instance digest are recomputed
before and after the build and any change is a `die`. Keep the source digest,
the exact image IDs and the instance digest with the run; do not substitute a
pre-existing tag without verifying its recorded image ID.

If `REBUILD_PROMPT.md` itself or anything else in
`portable/build-resource-manifest.json` changes, run
`../../../.venv/bin/python scripts/refresh_portable_manifests.py` and confirm
with `--check`: the task JSON's `delivery.build_resource_manifest_sha256`
moves, the scientific `portable.task_core.resource_manifest_sha256` must not.

## 6. Post-build verification

```bash
PHYSICAL_GPU=<index> bash scripts/smoke_test.sh --oracle
bash scripts/status.sh
```

`smoke_test.sh --oracle` re-runs the static gates, checks image/source and
final-instance freshness, runs the Oracle candidate through the
participant-visible public runner inside the Work image, then packs `oracle/`
and evaluates it through `scripts/eval_archive.sh` on the final suite, with the
GPU pin and GPU release barriers around it. `PHYSICAL_GPU` is the index of the
card allocated to you; pass it explicitly on any host with more than one GPU.

Expected readings:

| Reading | Expected |
| --- | --- |
| Final line | `Focused task smoke: PASS` |
| Gate table | all ten gates `PASSED` — `profile_ok`, `suite_ok`, `source_ok`, `hardware_ok`, `correctness_ok`, `edge_ok`, `fresh_input_ok`, `fresh_output_ok`, `budget_ok`, `tilelang_only_ok` — plus the `performance` record |
| Edge fixtures | 18/18 pass, before any scored trial |
| Relative L2 | within the `6e-3` limit on every trial; a value far above the honest-implementation range means a bug, not tolerance |
| Cold-start first-call compile | well inside the frozen 300 s warm-up cap |
| Oracle scores | the public aggregate and the final-suite aggregate in effective TFLOPS; record both as this deployment's Oracle band |

TFLOPS values belong to one measurement profile only. On a different card the
numbers simply differ — nothing gates on a TFLOPS threshold. A failing *gate*
is a real problem; a different *score* is not. **Before any agent run you must
complete the source/image/profile qualification on the target card** —
`make_oracle_archive.sh` + `eval_archive.sh` with every gate passing — and
record the Oracle band.

## 7. Formal run

Qualify the card first (above), then launch through the repository launcher.
Never hand-assemble the environment block: a hand-written one silently skips the
proxy-reachability guard, the version-store resolution and the agent-runtime
guard.

```bash
launch/tilelang-mamba2-ssd-v2/run.sh \
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
- `--tmux` runs detached; `--run-id ID` overrides the derived `RUN_ID`;
  `--config FILE` loads a launch config.

**Budget / interval / cooldown contract.** `scripts/run_fixed_budget.sh` reads
`TASK_TIMEOUT` (default 7200 s; must be ≥ 90, a multiple of 60, and greater than
`EVAL_INTERVAL`), `EVAL_INTERVAL` (default **300 s**, used as both
`--eval-interval` and `--submission-cooldown`) and `FINAL_EVAL_TIMEOUT`
(default **5400 s**), with `--disable-auto-eval` always set. The qualified
combination is `timeout: 7200`, `eval_interval: 300`,
`submission_cooldown: 300`, `disable_auto_eval: true`, with 8 CPUs and 48 GiB
for each of Work and Judge.

**Frozen, not tunable:** `disable_auto_eval` (in-run hidden evaluations must stay
at 0); the timeout ladder (candidate worker 4500 s → verifier 4900 s → Judge
adapter 5100 s → `judge.eval_timeout` 5400 s); and the equality
`FINAL_EVAL_TIMEOUT == judge.eval_timeout == 5400`. A smaller
`FINAL_EVAL_TIMEOUT` sits below the Judge's own 5100 s ceiling, and the harness
would kill the standalone final evaluation first.

**Ports.** `JUDGE_PORT` (or `--judge-port`) must be 1024–65535 and must not be
3128, 3130 or 3131, which are reserved for the restricted proxy sidecar
(CONNECT proxy, Judge relay, optional model-API relay). The launcher checks the
port is actually free before Work starts and dies otherwise. Default 18096 is
fine for a single run; a scheduler starting concurrent runs on one host must
hand every run its own free port.

**Version store.** `SFORGE_VERSION_STORE_ROOT` must point at a directory holding
`.sforge-version-store.json` that registers this `task_id` (a
`<store>/tasks/<task-directory>/task.json` with
`{"schema_version": 1, "task_directory": "...", "task_id": "tilelang_mamba2_ssd_forward_h100_seed_v2"}`).
When unset, `launch/common.sh` searches the conventional store locations beside
the checkout — or `SFORGE_VERSION_STORE_SEARCH_PATH`, a colon-separated list of
absolute paths — and fails closed on ambiguity or absence.
`SFORGE_VERSION_STORE_OPTIONAL=1` downgrades absence to a warning. `RUN_ID` must
be unique inside the store. The store keeps intermediate implementations that
may not survive in the end-of-run snapshot; count saves from the store
manifest, not from the agent transcript.

**Proxy.** The Work container never receives an operator proxy: the launcher
exports `SFORGE_HTTP_PROXY` / `SFORGE_HTTPS_PROXY` itself, pointing at the
restricted sidecar on the internal network, and `scripts/sforge_isolated_entry.py`
attaches Work to that network. The sidecar (`scripts/restricted_proxy.py`, the
same file as flash-kmeans' `restricted_connect_proxy.py`) enforces its policy
itself and dials directly from the default Docker bridge, so no host proxy is
involved; the host only needs ordinary outbound TCP on 80/443. Before Work
exists the launcher proves from inside the internal network that the sidecar
answers, the Judge relay on 3130 reaches the coordinator, `https://example.com`
is not reachable directly, `CONNECT chatgpt.com:443` answers 200, and — with
public egress enabled — `CONNECT github.com:443` answers 200 while a CONNECT to
the Docker bridge gateway (a non-global address) answers 403 (in allowlist-only
mode `github.com` answers 403). The sidecar's and entry's SHA-256 and the
network mode (`docker_internal_restricted_connect_proxy`,
`direct_public_egress: false`) are recorded in `experiment-manifest.json`.

**Single-allocation environment** exported by the runner and checked by
`scripts/single_gpu_runtime_preflight.py` before Docker or GPU state is touched:
`SFORGE_SINGLE_GPU_HANDOFF=1`, `SFORGE_ACCELERATOR_DEVICE_UUID`,
`SFORGE_ACCELERATOR_ALLOCATION_KEY`, `SFORGE_ACCELERATOR_OBSERVER=nvml`,
`SFORGE_TRUSTED_LEASE_ID`, plus `SFORGE_PRE_SUBMIT_HOOK` pointing at the
root-only hook inside the lab directory. One UUID-keyed `flock` spans Work, every
interactive Judge and the standalone final. A mismatch returns **78**.
Other environment honoured through the docker-group re-exec:
`CODEX_AUTH_JSON_PATH` (set by `--auth`), `CODEX_CLI_VERSION` (pinned by
`launch/common.sh`), `FINAL_EVAL_TIMEOUT`, `GPU_MIN_FREE_MIB`,
`SFORGE_AGENT_RUNTIME_DIR`.

## 8. Run artifacts and success criteria

```text
outputs/tilelang-mamba2-ssd-forward-h100-v2-seed/runs/<RUN_ID>/
  completion.json            # written only when the whole chain finishes
  experiment-manifest.json   # scheduling, measurement_profile, raw_score_comparison
  run_config.json  judge-server.log  sforge-run.log
  <task_id>/  resource_lease.json  formal-precheck.json  formal-postcheck.json
              final_archive.tar.gz  final_result.json
              final-<N>m-eval.log  run_agent.log
              submissions/agent-N/{submission.tar.gz,report.json}
outputs/.../runs/<RUN_ID>-final-<N>m/<task_id>/submissions/manual-1/report.json
```

Success is: the lease state is **CLOSED** and `completion.json` exists.
`scripts/validate_formal_artifacts.py` asserts `lease.state == "CLOSED"` and
binds the final run id, the archive digest and the report digest; a launcher
exit code of 0 and a valid final `sforge eval` are preconditions.

**Important distinction:** a candidate that *fails* on the final suite is a
candidate-level result, not a protocol fault. In that case the lease still
closes normally and the postcheck refuses by design, so no `completion.json` is
written. Do not read that as a broken deployment.

`completion.json` and `experiment-manifest.json` carry `raw_score_comparison`:
the ordered pair of the evaluation-contract SHA-256 and the measurement-profile
comparability key. Raw effective TFLOPS may be compared only when that identity
is identical and `qualification_status` is not
`unqualified_missing_calibration_epoch`. Merely satisfying the broad
`single-cuda-sm80-73g` envelope does not make two runs comparable, and neither
does a matching image digest across builds against different instances. Record
the stable measurement profile with the run (model, compute capability, memory,
driver, both image IDs, software ABI, calibration protocol and the
scheduler-issued calibration epoch); record UUID and observed current clocks
only in the deployment attestation.

## 9. When something fails

In this order:

1. The tail of `sforge-run.log`.
2. `<task_id>/final-<N>m-eval.log` — the standalone final evaluation.
3. `judge-server.log` — Judge registration and per-submission evaluation.
4. `formal-precheck.json` / `formal-postcheck.json` — the specific rejection.
5. The tmux session log under `launch/tilelang-mamba2-ssd-v2/logs/`, then the
   build logs.

Other things that bite:

- Exit code **78** is printed straight into the launcher output and means a
  contract, port or UUID problem; Docker and the GPU were never touched.
- A Judge result where **every** gate reads ERROR (rather than one gate failing)
  usually means a manifest-identity mismatch, not an infrastructure fault — for
  example a stale edge-case count in the Judge adapter, which a task test pins.
- `fatal error: cuda_runtime.h: No such file or directory` during the first
  TileLang call means the base image is not `-devel`.
- The Judge suite's wall time is set by the reference, which is recomputed for
  every measured trial — a faster candidate does not shorten the evaluation
  proportionally.
- `scripts/replay_candidates.sh` re-evaluates a kept candidate series. Use
  `--per-version`: the default mode evaluates the whole submission tree as it
  stood, so a run whose `FINAL_VERSION` took only a few values would measure
  only those few implementations however many evaluations it ran.

## 10. Moving to another GPU

Re-measure on the new card: the Oracle public and final-suite bands (§6), the
NVML memory peaks (the floor has only rounding head-room, §2), the real warm-up
head-room under the frozen 300 s cap (it tracks the host CPU's `nvcc` speed),
the margin under the timeout ladder, and the per-case coefficient of variation
across repeated paired evaluations.

Nothing else changes: the ABI, the mathematics, the three tolerance constants,
the `4.0` workspace multiple (it is a ratio against each case's own IO, so it is
portable by construction), the scored shapes, the 18 edge fixtures and the
metric formula are all device-independent.

Order for a different model, driver, image, software ABI, calibration protocol
or epoch — any of which creates a new measurement-profile comparability key even
when the portable TaskCore digest is unchanged: rebuild both images on the new
host (which reuses the frozen instance unless you pass `FINAL_CASES_PATH`), run
§5's CPU/meta checks, then redo the full source/image/profile qualification in
§6 before any agent run. `docker save`/`docker load` preserves the contract, but
byte-identical images do **not** by themselves qualify a different GPU or
driver. Pass a calibration epoch only for a campaign whose operator has fixed
and documented its clock/power policy.
