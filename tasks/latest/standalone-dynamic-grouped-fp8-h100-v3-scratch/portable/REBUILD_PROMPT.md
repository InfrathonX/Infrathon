# Rebuild Dynamic Grouped FP8 GEMM v3 on a qualified whole-GPU host

Evaluation policy: `full-final-shared-v3` / `dynamic-grouped-fp8-shared-final-v3`. Every admitted `auto-*`, `agent-*`, `manual-*`, and `candidate-*` round executes the complete frozen evaluation: the 158 private edge cases of `edge-holdout-v3` (zero-weight hard gates), the 32-case scored grid `final-holdout-v3`, and a compute-sanitizer memcheck pass over the 6 edge cases flagged `sanitize`. Timing, thresholds, aggregation, and private input isolation are identical across these rounds. Interactive rounds retain their original identities and cannot override `FINAL_VERSION`; host replay retains its candidate override. Local public tests use the disclosed public grid (the same 32 shapes with public seeds) and the 40 public edge cases of `edge-public-v3`; they never run a private suite.

Before formal precheck, the launcher stages the exact run-level `experiment-manifest.json` bytes at `<task_run_root>/experiment-manifest.json`, bound to the SHA-256 it recorded when creating the run manifest. Staging rejects a changed source, symlinks, and an existing destination, and creates the task copy with mode `0600`. The precheck reads this task-local copy and never falls back to a parent directory.

Operational handoff. With this file, `docs/HOST_PREREQUISITES.md` and this task
directory you can rebuild both images, qualify the deployment and run the task
on a fresh host with an admissible GPU. What the task is and how it is scored
is described in the agent brief `images/work/AGENT_TASK.md`, the Judge
(`judge/`, `images/judge/`) and the trusted runtime (`runtime/`). You may use a
build agent, but do not let the build agent redesign the benchmark.

## Immutable identities

Preserve these scientific identities byte-for-byte:

- the TaskCore and its resource manifest;
- the 6 public cases and their seeds, the 22 public edge cases
  (`edge-public-v3`), the 32 scored-grid seeds frozen at
  `judge/private/final_cases.json` (`final-holdout-v3`) and the 158 private
  edge cases frozen at `judge/private/edge_cases.json` (`edge-holdout-v3`),
  whose digests `portable/final-suite-manifest.json` records under
  `frozen_instance` and `edge_suite.frozen_instance`;
- the two references and their limits (the FP64 product of the dequantized FP8
  operands, `5e-5`; DeepGEMM's `calc_diff` against the BF16 source GEMM,
  `1e-3` on slices of at least 16,384 outputs), the memory-safety and
  harness-integrity checks, the timing protocol with its decoy pre-call, the
  effective-TFLOPS formula and the two-family geometric-mean metric;
- the agent-facing optimization contract.

Both private instances are **frozen, not drawn per build**. They are committed
inside the task under `judge/private/` and are what `build_images.sh` reads by
default; every v3 score is measured on them. They are Judge-only: staged mode
0600 into a root-only Judge build context, they must never be copied, mounted
or described inside the Work image (`scripts/validate_layout.py` proves no Work
build context admits them, by path and by SHA sweep). The scored shapes
themselves are public - they are DeepGEMM's own `enumerate_m_grouped_masked`
grid at `559d79f` - so only their seeds are private. **Reproduce the
instances, do not redraw them:** both come from seeded generators
(`scripts/generate_final_manifest.py` for the scored seeds,
`scripts/generate_edge_manifest.py` for the private and the public edge cases)
whose default seed is derived from the suite identity, so every host that
builds from this tree gets the same digests and the same suite identities.
`scripts/generate_final_manifest.py --redraw --seed <int>`, or
`scripts/generate_edge_manifest.py --write` after a generator change, mints a
**new suite identity**; no earlier score is comparable to it.
`FINAL_CASES_PATH` / `EDGE_CASES_PATH` still override the defaults when you
deliberately want to build against another instance. The Work image starts
from only `images/work` plus the pinned `runtime/` build context; the Judge
image is built independently from `images/judge`, the same runtime context and
the private `final_cases` context, which carries both instances.

## 1. Host prerequisites

Everything generic (Docker with the NVIDIA runtime as **default**, the docker
group, `uv`, CPU torch in the repo venv, `jq flock rsync patch git sha256sum
nvidia-smi`, disk, network) is listed once in `docs/HOST_PREREQUISITES.md`.
Check it first.

Task-specific extras:

| Need | Why |
| --- | --- |
| One **whole** NVIDIA GPU per concurrent run, **compute capability 8.9 or newer** | FP8 E4M3 tensor cores are the contract, so an sm_80 card is rejected by design. Work and Judge share the card serially under a single UUID-keyed `flock`, and the runtime additionally rejects a container that sees anything other than exactly one CUDA device. MIG identities are rejected by the launcher. |
| `compute-sanitizer` able to run on the card | The memory-safety gate runs the 6 flagged edge cases under `compute-sanitizer --tool memcheck` from the base image's CUDA 12.9 toolkit; the Judge build asserts the binary is present, so nothing is installed, but the tool has to support the card's architecture and be able to attach to CUDA processes. A GPU virtualization hook preloaded into every process prevents that; such a deployment must set `STANDALONE_GROUPED_FP8_SANITIZER_MODE=skip-platform-incompatible`, and the skipped pass is recorded as `sanitizer_ran = 0`. |
| Network for the two `docker build` invocations | They install packages; nothing else is fetched. |
| Network for the agent during a formal run | The task JSON declares `internet: true`, and the launcher enforces it without host iptables: Work is attached to a per-run `--internal` Docker network whose only exit is the `scripts/restricted_proxy.py` sidecar (CONNECT to `api.openai.com`, `auth.openai.com`, `chatgpt.com` on 443 and, with `--allow-public-egress`, to any globally routable host on 80/443, internal addresses refused; the Judge relayed on port 3130; Codex `web_search="live"`). A preflight proves no direct egress, a tunnel to a public host and a 403 for the Docker bridge gateway before Work starts. The `--enable-internet` flag still passed to `sforge run` only bypasses SForge's iptables path; the manifest records `network.mode = docker_internal_restricted_connect_proxy`. The sidecar itself needs outbound 80/443 from the host. Evaluation itself fetches nothing. |
| The LLM integrity review | The Judge server reviews every agent submission on CPU before the GPU Judge (`docs/LLM_JUDGE.md`, task policy `llm_judge/POLICY.md`); with no `SFORGE_LLM_JUDGE_*` settings it uses the run's own Codex runtime and credentials. `scripts/eval_archive.sh` reviews its archive too unless `TRUSTED_CONTROL=1`, which is for the trusted Oracle and deliberate control archives only (`scripts/smoke_test.sh --oracle` sets it); an `sforge eval` whose review cannot run stops instead of admitting the archive. |
| Behind a proxy: a Docker-reachable proxy address | `docker build` runs with `--network host` and forwards `HTTP_PROXY`/`HTTPS_PROXY`/`NO_PROXY` (and the lowercase spellings) as build args; a loopback-only proxy is not reachable from the containers. |
| A version store that registers this `task_id` | Without it the run silently keeps no replayable candidates; the launcher fails closed. See §7. |

No `sudo` is needed beyond the container-toolkit step in the host
prerequisites.

## 2. GPU envelope

The admissible-device contract is the `single-cuda-sm89-16g` execution profile
in `tasks/standalone_dynamic_grouped_fp8_h100_scratch_v3.json`:

| Field | Value |
| --- | --- |
| `vendor` / `count` | `nvidia` / 1 whole GPU |
| `min_compute_capability` | `8.9` |
| `min_memory_bytes` | `17179869184` (16 GiB) |
| `features` | `cuda`, `fp8-e4m3`, `bf16`, `fp32`, `triton` |
| `software_abi.protocol_id` | `pytorch280-cu129-triton340-fp8-source-sanitizer-v2` |
| `calibration_protocol.protocol_id` | `cuda-event-host-wall-max-median-decoy-v2` |

Enforcement, in order of when you meet it:

- `scripts/lib.sh` — `assert_gpu_idle` (used by `smoke_test.sh`,
  `eval_archive.sh` and the dual-GPU path) requires `GPU_MIN_FREE_MIB`
  free (default 14336 MiB) and no compute process on the card; in
  single-allocation mode `assert_portable_gpu_idle` requires the profile's
  compute capability and `min_memory_bytes` both total and free, before Work
  is created and again at the final-Judge barrier.
- `runtime/grouped_fp8_runtime.py` — `MIN_COMPUTE_CAPABILITY` and
  `MIN_DEVICE_MEMORY_BYTES` carry the same 8.9 / 16 GiB pair, plus a
  visible-device-count check.
- `images/judge/verify_grouped.py` — the same envelope again as the scored
  `hardware_ok` gate. A card below it produces `valid: false`, not a lower
  score.
- The runtime and the verifier also carry `FP8_DENSE_PEAK_BOUND_TFLOPS`, a
  dense-FP8 throughput ceiling per compute capability (8.9, 9.0, 10.0, 10.3,
  12.0). A case above it fails `harness_integrity_ok`; an unlisted capability
  has no ceiling. It is a tamper backstop, not a performance threshold.

**Memory floor.** The declared 16 GiB is a contract, not a measurement, and is
never lowered to a measured Work or Judge peak: lowering it would admit cards
this task was never qualified on.

## 3. External inputs

**There are none.** Every tensor is generated in-process from the case seed by
the trusted runtime, so no model, dataset, tokenizer, fixture archive or
upstream checkout has to be fetched.

| What | Source | Size | SHA | Env var / path | Access |
| --- | --- | --- | --- | --- | --- |
| Base image `pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel` | Docker Hub, pulled by both Dockerfiles; also supplies the CUDA 12.9 `compute-sanitizer` | not recorded per task; `docs/HOST_PREREQUISITES.md` budgets 17–20 GB per task image pair, base images shared | pinned by tag only | — | public |
| Upstream `deepseek-ai/DeepGEMM` @ `559d79fb6994a58b8a15b4b93bf13ccc16edf247` | GitHub | — | commit-pinned | — | public; **semantic provenance only** — already ported into `runtime/` behind the marker `deepgemm-559d79f-derived-v3-judge-port`, never fetched by any script and never present in either image |
| Scored-grid seeds (`final-holdout-v3`, 32 cases) | in-tree at `judge/private/final_cases.json`, see §4 | < 10 KB | `31c8be8693164dfe596aa4bbe8c5a0de79ec661e3b9c0acba9556c99ae98441b`, recorded in `final-suite-manifest.json` under `frozen_instance` and again per build in `build/image-manifest.json` | `FINAL_CASES_PATH` (default `judge/private/final_cases.json`) | in-tree (Judge-only path) |
| Private edge cases (`edge-holdout-v3`, 158 cases, 6 `sanitize`) | in-tree at `judge/private/edge_cases.json`, see §4 | < 50 KB | `059a14336b64b2eb504504b15840f6491d14ea28b4f406a7a907d8087637a08d`, recorded under `edge_suite.frozen_instance` and again per build in `build/image-manifest.json` | `EDGE_CASES_PATH` (default `judge/private/edge_cases.json`) | in-tree (Judge-only path) |
| Public edge cases (`edge-public-v3`, 40 cases, 4 `sanitize`) | in-tree at `images/work/benchmark/public_edge_cases.json` | < 15 KB | recorded in `public-suite-manifest.json` under `edge_suite.sha256` | — | in-tree (Work image) |
| Oracle archive under `oracle/` | in the repository | small | — | `scripts/make_oracle_archive.sh` | host-only qualification reference, **deliberately not a build input** |

## 4. Frozen holdout instance (by design)

`scripts/build_images.sh` reads both private instances before it builds the
Judge image. `FINAL_CASES_PATH` defaults to the committed
`judge/private/final_cases.json` - one seed per scored-grid case, drawn from
`[10000, 2147483646]` - and `EDGE_CASES_PATH` to the committed
`judge/private/edge_cases.json`, so a plain checkout builds the scored and edge
suites with no draw and no operator input. For each instance the build checks
that the source is a regular non-symlink, that its `suite_id` is
`final-holdout-v3` / `edge-holdout-v3` and that it holds exactly 32 / 158
cases, installs it at mode `600` into the named Judge-only build context, and
re-checks that the staged copy is mode `600`, owned by the building user and
byte-identical; after the build it re-verifies the image-driving source digest
and both instance digests — any drift is a hard `die`.

**Expected digests:** in the build manifest, `final_cases.sha256` must equal
`31c8be8693164dfe596aa4bbe8c5a0de79ec661e3b9c0acba9556c99ae98441b`
(`frozen_instance.sha256`) and `edge_cases.sha256` must equal
`059a14336b64b2eb504504b15840f6491d14ea28b4f406a7a907d8087637a08d`
(`edge_suite.frozen_instance.sha256`) in `portable/final-suite-manifest.json`.
A different value means you are not on the frozen instances and your scores
are not comparable with scores on the frozen instances.

To mint a **new** suite identity deliberately:

```bash
../../../.venv/bin/python scripts/generate_final_manifest.py \
  --output /some/scratch/final_cases.json --redraw --seed 20261005
FINAL_CASES_PATH=/some/scratch/final_cases.json bash scripts/build_images.sh
```

The draw is a seeded `random.Random` (default seed:
`int.from_bytes(sha256("<suite_id>|<task_id>").digest()[:8], "big")`), never a
system random source, so a redraw is reproducible from its seed alone. The
edge generator derives its seeds the same way from `edge-holdout-v3` and
`edge-public-v3`: `scripts/generate_edge_manifest.py --check` regenerates both
edge instances and compares them byte-for-byte with the committed files, and
`--write` rewrites them (a new suite identity whenever the bytes change). After
either kind of change, record the new digests by hand (`frozen_instance` and
`edge_suite.frozen_instance` in the final and interactive suite manifests,
`edge_suite.sha256` in the public one), then run
`scripts/refresh_portable_manifests.py`; scores from the old instances do not
carry over.

Nothing is captured per deployment.

## 5. Build

Run everything from the task root
(`tasks/latest/standalone-dynamic-grouped-fp8-h100-v3-scratch`).

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
delivery/build-resource digest and every portable manifest byte digest (it
fails if `refresh_portable_manifests.py --check` reports drift), and the real
suite case counts (6 public / 32 interactive / 32 final scored cases, 158
private and 22 public edge cases) from the runtime, verifier, public runner and
generators. It also checks that both frozen instances are the seeded
generators' output, that public and private seeds and edge draws are disjoint,
and that no Work build context admits a private instance. The two expected
skips are the below-floor capability fixtures in
`tests/test_capability_envelope.py`. Treat any failure as a hard stop. Run them
with an interpreter that has `torch` (a CPU build is enough), `yaml`, `docker`
and the repo's `sforge` package importable: the trusted runtime imports torch
when it loads, so without it the runtime-backed tests error rather than skip.

The task JSON intentionally is not a member of either resource manifest because
it is the root document that binds their digests; each resource manifest also
excludes itself to avoid a recursive hash.

Then build:

```bash
bash scripts/build_images.sh
bash scripts/make_oracle_archive.sh
jq -e '.source_digest and .work.id and .judge.id and .final_cases.sha256 and .edge_cases.sha256' \
  ../../../outputs/standalone-dynamic-grouped-fp8-h100-v3-scratch/build/image-manifest.json
```

Environment it reads: `FINAL_CASES_PATH` / `EDGE_CASES_PATH` (optional, §4),
the six proxy variables, and `PYTHON_BIN` / `SFORGE_BIN` if the repo venv is
not at the default location. It re-execs itself through `sg docker` when your
shell is not yet in the docker group; use `umask 022` in that case.

Products: the Work image
`standalone_grouped_fp8.work.standalone_dynamic_grouped_fp8_h100_scratch_v3:local`,
the Judge image
`standalone_grouped_fp8.judge.standalone_dynamic_grouped_fp8_h100_scratch_v3:local`,
the build manifest (`schema_version: 3`, with `source_digest`, both image IDs,
`final_cases.{sha256,suite_id,case_count}` and
`edge_cases.{sha256,suite_id,case_count}`), the two build logs under the same
outputs directory, and the host-only Oracle archive `artifacts/oracle.tar.gz`.
Keep the source digest, the exact image IDs and both instance digests with the
run; `run_fixed_budget.sh` recomputes them and refuses a stale identity.

If `REBUILD_PROMPT.md` itself or anything else in
`portable/build-resource-manifest.json` changes, run
`../../../.venv/bin/python scripts/refresh_portable_manifests.py` and confirm with `--check`: the
task JSON's `delivery.build_resource_manifest_sha256` moves, the scientific
`portable.task_core.resource_manifest_sha256` must not.

## 6. Post-build verification

```bash
PHYSICAL_GPU=<index> bash scripts/smoke_test.sh --oracle
```

It re-runs the static gates (layout validation, the CPU test suite,
`sforge list`), checks image, source and both instance digests for freshness,
runs the Oracle through the participant-visible public runner (the full
32-shape grid) inside the Work image, then packs `oracle/` and evaluates it
through `scripts/eval_archive.sh` with `TRUSTED_CONTROL=1`: the complete Judge
evaluation (158 edge cases, the 32-case scored grid, the sanitizer pass)
without the LLM integrity review, which the trusted Oracle does not need. It
takes the card's `flock` and refuses to start if the GPU is not idle.

Expected bands:

| Reading | Expected |
| --- | --- |
| Gate table | all `PASSED`; the terminal shows **9/9** (8 gates plus a `performance` record) |
| Public runner banner | prints `FINAL_HOLDOUT NOT_RUN` and `CANDIDATE_ONLY YES` |
| Public full-suite Oracle | a `PUBLIC_PRIMARY_TFLOPS` reading for this card; record it |
| Judge Oracle | valid, with all 158 private edge cases passing and a clean sanitizer pass; record the score and family split for this card |
| Judge suite wall | must stay well inside the 3300 s verifier ceiling |

**The public and Judge Oracle scores measure different shapes.** The public full
suite times 6 small cases with uneven row counts; the Judge scores 32 larger
shapes, about 40 times the work per trial, so public scores do not predict Judge
scores. The task's "public pass ≠ Judge
pass" signal comes from the private edge cases and the sanitizer pass: a
candidate that passes the public grid can still fail `edge_correctness_ok` or
`memory_safety_ok`, which is why the public edge and sanitizer suites exist.

Oracle TFLOPS values are provenance for one measurement profile only. On a
different card the numbers simply differ — nothing in the tree gates on a
minimum TFLOPS (the only throughput check is the plausibility ceiling in §2). A
failing *gate* is a real problem; a different *score* is not. Record whatever
your card produces as the Oracle band before any agent run.

## 7. Formal run

Qualify the card first (above), then launch through the repository launcher.
Never hand-assemble the environment block: a hand-written one silently skips the
proxy-reachability guard, the version-store resolution and the agent-runtime
guard.

```bash
launch/standalone-fp8/run.sh \
  --device-uuid GPU-00000000-0000-0000-0000-000000000000 \
  --calibration-epoch campaign-qualified-clock-policy-v1 \
  --judge-port 18096 \
  --model <model> --effort ultra --hours 2 \
  --auth /path/to/auth.json
```

- `launch/standalone-fp8/` launches v3.
- `--model` takes a short alias defined in `launch/common.sh` or a full model
  ID, including any ID served by the OpenAI-compatible endpoint named in
  `MODEL_API_BASE_URL` (key in `MODEL_API_KEY_FILE`). `--effort` is parsed as
  `max|ultra|xhigh|high|medium|low`, and the task runner accepts
  `high|max|xhigh|ultra`.
- `--hours N` and `--timeout SECONDS` are alternatives (the latter wins);
  default 7200 s.
- `--device-uuid` must be a canonical whole-GPU UUID
  (`GPU-` + 8-4-4-4-12 lowercase hex). `--allocation-key` and `--lease-id` are
  optional; the wrapper computes the canonical `allocation-sha256` from the
  device set and refuses a supplied key that does not match it, and generates a
  never-reused lease ID when one is not supplied.
- Single-allocation mode is the default and **rejects** `WORK_GPU`/`JUDGE_GPU`
  with no fallback; a two-card path exists only behind `--legacy-dual-gpu`.
- `--cpuset auto|off|CPULIST[@NODES]` pins Work and Judge to GPU-local CPUs
  (default off; `docs/CPU_AFFINITY.md`).
- `--tmux` runs detached; `--run-id ID` overrides the derived `RUN_ID`
  (`standalone-grouped-fp8-v3-<profile>-<model>-<effort>-<N>m-<stamp>` in
  single-allocation mode, `standalone-grouped-fp8-v3-h100-…` in the two-card
  mode).

**Budget / interval / cooldown contract.** `scripts/run_fixed_budget.sh` reads
`TASK_TIMEOUT` (default 7200 s), `EVAL_INTERVAL` (default 300 s) and
`FINAL_EVAL_TIMEOUT` (default 3600 s), and passes the *same* `EVAL_INTERVAL`
value to both `--eval-interval` and `--submission-cooldown`, with
`--disable-auto-eval` always set. Constraints it enforces: `TASK_TIMEOUT >= 90`,
`TASK_TIMEOUT > EVAL_INTERVAL`, and both multiples of 60. The qualified
combination is `timeout: 7200`, `eval_interval: 300`,
`submission_cooldown: 300`, `disable_auto_eval: true`, `internet: true`.
Frozen, not tunable: `disable_auto_eval` (in-run hidden evaluations must stay
at 0) and the timeout ladder (candidate worker 2400 s and compute-sanitizer
pass 780 s → verifier 3300 s → Judge adapter 3400 s, inside the SForge-side
`judge.eval_timeout` of 3600 s).

**Ports.** `JUDGE_PORT` (or `--judge-port`) must be 1024–65535 and must not be
3128, 3130 or 3131, which are reserved for the sidecar. The launcher checks the port is actually
free before Work starts and dies otherwise. Default 18096 is fine for a single
run; a scheduler starting concurrent runs on one host must hand every run its
own free port.

**Version store.** `SFORGE_VERSION_STORE_ROOT` must point at a directory holding
`.sforge-version-store.json` that registers this `task_id` (a
`<store>/tasks/<task-directory>/task.json` with
`{"schema_version": 1, "task_directory": "...", "task_id": "standalone_dynamic_grouped_fp8_h100_scratch_v3"}`).
When unset, `launch/common.sh` searches the conventional store locations beside
the checkout — or `SFORGE_VERSION_STORE_SEARCH_PATH`, a colon-separated list of
absolute paths — and fails closed on ambiguity or absence.
`SFORGE_VERSION_STORE_OPTIONAL=1` downgrades absence to a warning.
`RUN_ID` must be unique inside the store: reusing one raises
`VersionStoreCollision` when the final version is persisted after the deadline,
wasting the whole budget.

**Proxy.** The launcher exports its own `SFORGE_HTTP_PROXY`/`SFORGE_HTTPS_PROXY`
(the restricted sidecar on the internal network) and unsets any ambient
`http_proxy`/`HTTP_PROXY` for the host side, so operator proxy variables are
not forwarded to Work. With an ambient loopback-only proxy exported,
`launch/common.sh` still refuses to start in the default single-allocation
mode: unset `http_proxy`/`HTTP_PROXY` first (the sidecar does not need it).
`scripts/restricted_proxy.py` is the task-local restricted egress sidecar and
`scripts/sforge_isolated_entry.py` the SForge entry that pins Work to the
internal network; the launcher records both digests in the run manifest.

**Other environment** honoured through the docker-group re-exec:
`CODEX_AUTH_JSON_PATH` (set by `--auth`), `CODEX_CLI_VERSION` (pinned by
`launch/common.sh`), `FINAL_EVAL_TIMEOUT`, `GPU_MIN_FREE_MIB`,
`SFORGE_AGENT_RUNTIME_DIR`, `SFORGE_CALIBRATION_EPOCH` (set by
`--calibration-epoch`), `FINAL_CASES_PATH`, `EDGE_CASES_PATH`,
`MODEL_API_BASE_URL`, `MODEL_API_KEY_FILE`. The single-allocation contract
itself (`SFORGE_SINGLE_GPU_HANDOFF=1`, `SFORGE_ACCELERATOR_DEVICE_UUID`,
`SFORGE_ACCELERATOR_ALLOCATION_KEY`, `SFORGE_ACCELERATOR_OBSERVER=nvml`,
`SFORGE_TRUSTED_LEASE_ID`) is exported by the wrapper and survives the re-exec.

## 8. Run artifacts and success criteria

```text
outputs/standalone-dynamic-grouped-fp8-h100-v3-scratch/runs/<RUN_ID>/
  completion.json  experiment-manifest.json  run_config.json
  judge-server.log  sforge-run.log
  <task_id>/  resource_lease.json  formal-precheck.json  formal-postcheck.json
              final_archive.tar.gz  final_result.json  final_snapshot.json
              run_history.json  final-<N>m-eval.log  run_agent.log
              final-<N>m-private-judge-artifacts.tar.gz   # mode 0600
              submissions/agent-N/{submission.tar.gz,report.json}
outputs/.../runs/<RUN_ID>-final-<N>m/<task_id>/submissions/manual-1/report.json
```

Success is all of:

- `resource_lease.json` has `state` = `CLOSED`;
- `formal-precheck.json` and `formal-postcheck.json` both PASS;
- `completion.json` has `status = passed`, `sforge_run_exit = 0`,
  `final_eval_exit = 0`, `hidden_evaluations_during_work = 0` and
  `portable_contract.execution_profile_id = single-cuda-sm89-16g`;
- the standalone final report `submissions/manual-1/report.json` has
  `valid = true` with 9/9 records green.

`completion.json` also carries `raw_score_comparison`: the ordered pair of the
evaluation-contract SHA-256 and the measurement-profile comparability key, with
`policy = same_evaluation_contract_and_measurement_profile_only`. Raw effective
TFLOPS may be compared only when that identity is identical and
`qualification_status` is not `unqualified_missing_calibration_epoch`. Merely
satisfying the broad `single-cuda-sm89-16g` envelope does not make two runs
comparable. Scores from any other task version are never comparable with this
v3 contract. Record the stable measurement profile with the
run (model, compute capability, memory, driver, both image IDs, software ABI,
calibration protocol and the scheduler-issued calibration epoch); record UUID
and observed current clocks only in the deployment attestation.

## 9. When something fails

In this order:

1. `sforge-run.log` — agent/Work lifecycle and the SForge run exit code.
2. `judge-server.log` — Judge registration, the LLM integrity review and
   per-submission evaluation.
3. `<task_id>/run_agent.log` — what the agent actually did inside Work.
4. `final-<N>m-eval.log` and `final_result.json` — the standalone final
   evaluation.
5. `formal-precheck.json` / `formal-postcheck.json` — they print the specific
   rejection reason.

Other things that bite:

- Exit code **78** from preflight means a harness-contract or portable-digest
  mismatch, refused before Docker or the GPU was touched; the message names the
  failing item.
- `sforge eval` (for example inside `eval_archive.sh`) exits **2** with "the
  LLM integrity review cannot run" when the review is enabled but cannot start;
  configure it as in `docs/LLM_JUDGE.md`, or use `TRUSTED_CONTROL=1` for a
  trusted control archive.
- An invalid Judge report names the first failing gate (`failed_gate`); gates
  after it were not evaluated. `edge_cases_total` / `edge_cases_failed` give the
  number of failed private edge cases, never which ones.
- A card with compute capability below 8.9 fails `hardware_ok` by design, not by
  accident.
- Serialising runs on one card: wait for the previous run's containers to be
  really gone, not just for the script to return.
- `scripts/replay_candidates.sh` (or `launch/standalone-fp8/replay.sh`)
  re-evaluates a kept candidate series without a new agent run;
  `scripts/status.sh [RUN_ID]` summarises a run; without an argument it picks the
  newest formal run directory of either mode.

## 10. Moving to another GPU

Re-measure on the new card: the Oracle public and Judge TFLOPS bands (§6), the
NVML Work and Judge memory peaks, and the head-room under the timeout ladder
(2400 s worker, 780 s sanitizer, 3300 s verifier) if the card is materially
slower than an H100. Re-check `GPU_MIN_FREE_MIB` against the new card's total
memory, confirm that the sanitizer pass completes there, and look up its
compute capability in the plausibility table (§2): an unlisted capability runs
without that ceiling, and adding a row is a TaskCore change.

Nothing else changes: the two ABIs, the support domain, every public grid shape
and seed, every public and private edge case, the `5e-5` exact and `1e-3`
source limits and the metric are device-independent, and both references are
recomputed on the target device inside every evaluation.

Order for a different model, driver or image: run §5's CPU/meta checks, rebuild
both images (which reuses the frozen instances unless you pass
`FINAL_CASES_PATH` / `EDGE_CASES_PATH`), then run §6 on the new card before any
agent run. Scores after that are **not** directly comparable with the old
profile's. `docker save`/`docker load` preserves the contract, but
byte-identical images do **not** by themselves qualify a different GPU or
driver. Pass a calibration epoch only for a campaign whose operator has fixed
and documented its clock/power policy.
