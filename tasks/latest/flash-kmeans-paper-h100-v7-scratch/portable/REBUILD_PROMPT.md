# Rebuild Flash K-Means v7 on a qualified whole-GPU host

Use this task bundle to reproduce the Work/Judge images and qualify a new
single-CUDA deployment. You may use Codex, Claude, or another build agent, but
do not let the build agent redesign the benchmark. What the task *is* — the
metric, the gates, the suites — is defined by the TaskCore listed in
`portable/resource-manifest.json`, the agent brief and the task README; this
file is only how to build, verify and run it.

Preserve these scientific identities byte-for-byte:

- the TaskCore and its resource manifest;
- all 11 public cases and all 22 host-private final cases: 14 positive-weight
  performance rows plus 8 zero-weight correctness hard gates;
- the oracle, selection and six-group family-first latency metric;
- the agent-facing optimization contract.

The final suite manifest and all hidden Judge resources are host-private. They
must never be copied, mounted, or described inside the Work image. The Work
image starts from only `images/work`; the Judge image extends it from the
separate `images/judge` context.

## 1. Host prerequisites

`<repo>/docs/HOST_PREREQUISITES.md` lists everything that is not
task-specific: Docker >= 24 with the **NVIDIA runtime as default**, your user in the `docker`
group, `uv` with Python 3.10-3.13, `jq flock rsync patch git sha256sum
nvidia-smi`, CPU `torch` + `numpy` in the repo venv, >= 150 GB free on Docker's
data root, and outbound access to Docker Hub, Hugging Face and GitHub. Check
them first.

Task-specific extras:

| Extra | Why |
| --- | --- |
| `tar` on the host | the archive builders and `eval_archive*.sh` pack and unpack candidate archives |
| **an upstream `flash-kmeans` checkout** pinned at `12ed49b2eddc207dfc5ed27dba244af2365bd9c6`, exported as `FLASH_KMEANS_UPSTREAM` | this repository deliberately does not vendor upstream source; `scripts/make_host_calibration_archives.sh` needs it, and `scripts/smoke_test.sh` needs the archives it produces. `scripts/build_images.sh` does **not** need it |
| **two** distinct physical GPUs, if you want to run `scripts/smoke_test.sh` or `scripts/eval_archive_smoke.sh` | both use a dual-GPU layout and read `WORK_GPU` / `JUDGE_GPU` (defaults 1 and 2). The formal run is single-allocation and *rejects* those variables; do not mix the two |
| 32 GiB of host RAM per container | Work and Judge each take `cpu_limit: 8`, `mem_limit: 32g` |
| `ss` (iproute2) | the launcher refuses to start when the coordinator port is already listening |

The task JSON sets `internet: false`, so the containers have no egress at run
time. Only the base-image pull and the upstream checkout need the network.

## 2. GPU envelope

The task JSON's `portable.execution_profiles` carries exactly one profile,
`single-cuda-sm80-80g`: `count = 1`, `min_compute_capability = "8.0"`,
`min_memory_bytes = 84825604096` (79 GiB), features `cuda / fp16 / fp32 /
triton`. Both total and free memory must satisfy it. MIG is not qualified.

Who enforces it:

- `scripts/lib.sh`'s `assert_portable_gpu_idle`, fed the numbers that
  `scripts/run_fixed_budget.sh` reads out of the task JSON;
- before the allocation, an NVML idle attestation through
  `sforge.harness.accelerator_observer.NvmlAcceleratorObserver.attest_idle`;
- `scripts/lib.sh`'s `assert_container_accelerator`, which pins the container to
  the exact device with `--gpus device=<UUID>`.

There is **no** sm_90 / FP8 / wgmma / TMA / NVLink dependency anywhere in the
code; the verifier's only device predicate is `torch.cuda.is_available()`, and
the preflight does not match on GPU model name ("GPU model is provenance only").
The `h100` in the task id and directory name is part of the task identity, not
a gate.

The Judge's full-suite device-memory peak, measured by NVML on an H100 SXM
80 GB card with the `official_oracle_fa3epi_h100` archive, is about 53 GiB
(54,263 MiB); the Work container's peak is well under 8 GiB.

**Why the floor is 79 GiB.** The card size is set entirely by the trusted
Judge. The Judge's peak is set by the two kernel-shaped `update` cases
(`hidden_massive_update`, B=1 N=33M D=128, and `hidden_batched_update`, B=32
N=1M D=128), whose steady-state protocol keeps several 8 GiB input sets and
their guard clones resident at once; the largest deployment case holds one
16 GiB input and passes with ample headroom. `images/judge/benchmark/kernel_bench.py`'s
`make_inputs` used to build `x_sq` for those cases through a full-shape FP32
copy of x plus its square, a 33 GiB transient on top of that resident set,
which put the peak at about 83.2 GiB and made an 80 GB card OOM inside trusted
input generation (reported as a candidate memory-gate failure). `x_sq` is not
an `update` input, so `make_inputs` now builds it row-chunked for `update`
only; assign/full keep the unchunked expression, so every scored input is
bit-identical to before. 79 GiB is the floor shared with the other 80 GB-class
tasks; it leaves about 26 GiB of spare device memory on an H100 SXM 80 GB card.
Anything that moves the peak (a new driver, a different PyTorch minor, an added
case) should be re-measured rather than assumed.

flash-kmeans carries no `MIN_DEVICE_MEMORY_BYTES`-style constant in its runtime
or verifier, so the profile value lives only in the task JSON.

### Pre-Hopper hosts and the compatibility oracle

The GPU model is not a gate, but the **host-side external comparison** hits two
upstream limits on anything that is not Hopper-class. The official reference
adapter assumes the fixed H100 opt-in shared-memory budget (232448 bytes) and
routes fp16/bf16 tuning tables around SMEM fitting, and the upstream heuristics
only know a handful of GPUs by name — every other card is forced through the
split-D kernel, which then clamps `BLOCK_D` to `D` and fails `tl.arange` for a
non-power-of-two `D`.

The repository therefore ships a minimal upstream patch and its notes in
`host_reference/official_oracle_ampere_compat/`, built by
`scripts/make_ampere_compat_oracle_archive.sh` into
`artifacts/official-oracle-ampere-compat.tar.gz`. The patch teaches the arch
pickers two more Ampere parts and routes them to the A100 tables, keeps
`BLOCK_D` a power of two, and passes the fp16/bf16 tables through SMEM fitting;
with the builder's `--device-smem-budget` flag the fixed H100 budget becomes
`min(props.shared_memory_per_block_optin, 232448)`, still without dynamic
`getattr`. Every one of those changes is a no-op on H100/H200/A100.

This archive is a **different candidate identity** from
`official-oracle.tar.gz` and must be reported as such; it is not a replacement
for the H100-accepted archive, and `official_oracle/` stays byte-exact. If the
original archive fails only inside the candidate (a Triton compile error or
`OutOfResources`), that is the signal to use the compatibility archive — and,
if the card's name is new, to add it to the patch's arch mapping pointing at
the closest same-generation table.

## 3. External inputs

| What | Source and pin | Size | SHA if pinned | Env var / path | Availability |
| --- | --- | --- | --- | --- | --- |
| Base image for both images | `pytorch271_cu128` (task JSON `base_image`), software ABI `pytorch271-cu128-triton-source-v1` | ~8 GB | by tag | pulled by `scripts/build_images.sh` | public |
| Upstream flash-kmeans checkout | `svg-project/flash-kmeans` @ `12ed49b2eddc207dfc5ed27dba244af2365bd9c6` | small | pinned by commit | `FLASH_KMEANS_UPSTREAM=<clone path>` | public; **required only for the host calibration archives**, not for the image build. Only `flash_kmeans/assign_euclid_triton.py` and `flash_kmeans/centroid_update_triton.py` are read |
| `fast_pytorch_kmeans` checkout | `DeMoriarty/fast_pytorch_kmeans` @ `aec7cd96a10f7cfe49f8cc4668b3729e3534e1de` | small | pinned by commit | vendored under `host_reference/`; the pinned commit ships as `host_reference/fast_pytorch_kmeans/UPSTREAM_COMMIT`, which the archive builder reads (no `.git` needed) | public |
| `fastkmeans` checkout | `AnswerDotAI/fastkmeans` @ `596bff0a2f91df998d30bd3a5e607c237dfa251c` | small | pinned by commit | vendored under `host_reference/` | public |
| `artifacts/*.tar.gz` calibration archives | produced here by `scripts/make_host_calibration_archives.sh` (and `make_ampere_compat_oracle_archive.sh`) | small | recorded per archive | `artifacts/` (gitignored; the repository ships no `artifacts/`) | built locally |
| Private corpus / fixture / final manifest / model snapshot | **none** | — | — | — | not applicable |

There is no owner-provided input. This is a scratch kernel task: **the hidden
cases are generated inside the Judge image from frozen seeds at evaluation
time**, by `images/judge/hidden_cases.py` and `verify_flash_kmeans.py`. There is
nothing to place, fetch or capture, and `scripts/build_images.sh` reads no
environment variables at all — the single biggest difference from the other
single-GPU tasks.

### Why there is no per-deployment capture and no per-build draw here

Two deployments with the same image source digest run a byte-identical 22-case
suite. What is withheld is not "which instance was drawn" but **the identity of
the whole case set** — ids, shapes, seeds, score-group membership and per-case
weights exist only inside the Judge image and its root-only artifacts, never in
the Work image. The public contract is the six score-group names and weights,
the two coarse family names, and the fresh-process protocol; nothing else.

## 4. CPU checks before any image or GPU operation

From the task root, with the repository venv's interpreter (or any Python that
can import the task's test dependencies):

```bash
python3 -B -m pytest -p no:cacheprovider tests -q
python3 -B scripts/validate_layout.py .
```

`scripts/smoke_test.sh` and `scripts/build_images.sh` run the same tests
internally through `python3 -m unittest discover -s tests -v`.

Those tests recompute the scientific resource digest, delivery/build-resource
digest, manifest byte digests and real suite case counts. Treat any mismatch as
a hard stop. The task JSON intentionally is not a member of either resource
manifest because it is the root document that binds their digests; each
resource manifest also excludes itself to avoid a recursive hash.

One test is skipped without a GPU. The tests in
`tests/test_single_gpu_migration.py` fork
`scripts/single_gpu_runtime_preflight.py` and need the SForge orchestration
dependencies (including the `docker` module) in the same interpreter; install
Torch and those dependencies in one interpreter to get a clean full run.

## 5. Build

```bash
bash scripts/build_images.sh
jq -e '.source_digest and .work.id and .judge.id' \
  ../../../outputs/flash-kmeans-paper-h100-v7-scratch/build/image-manifest.json
```

`build_images.sh` needs only `jq` and `python3`. It copies `judge/run_judge.py`
into `images/judge/`, runs `scripts/validate_layout.py`, builds
`flash_kmeans.work.<task_id>:local` and `flash_kmeans.judge.<task_id>:local`
(the Judge takes the Work image through `--build-arg WORK_IMAGE`), re-checks
that the image source digest is unchanged across the build, and writes
`created_at`, `source_digest` and both image IDs to
`outputs/flash-kmeans-paper-h100-v7-scratch/build/image-manifest.json`.

The image source digest covers `images/work`, `images/judge` and
`judge/run_judge.py`; `assert_built_images_current` compares it before every
smoke and every run, so any edit to those three requires a rebuild. Keep the
source digest and the exact image IDs with the run, and do not substitute a
pre-existing tag without verifying its recorded image ID.

To build the host-side comparison archives (needed by the smoke below):

```bash
FLASH_KMEANS_UPSTREAM=<upstream checkout> bash scripts/make_host_calibration_archives.sh
# pre-Hopper hosts additionally:
FLASH_KMEANS_UPSTREAM=<upstream checkout> bash scripts/make_ampere_compat_oracle_archive.sh
```

These write `official-oracle.tar.gz`, `torch-fallback-baseline.tar.gz`,
`fast-pytorch-kmeans-baseline.tar.gz`, `fastkmeans-baseline.tar.gz`,
`standard-pytorch-kmeans-baseline.tar.gz` (and the Ampere-compat variant) under
`artifacts/`. Nothing under `host_reference/` or `artifacts/` is a Docker build
input; the archives go to the Judge one at a time as ordinary candidate
submissions, and no host implementation ever becomes a timing denominator.

## 6. Post-build verification

```bash
FLASH_KMEANS_UPSTREAM=<upstream checkout> \
  WORK_GPU=<index> JUDGE_GPU=<other index> bash scripts/smoke_test.sh
# single archive re-check:
JUDGE_GPU=<index> bash scripts/eval_archive_smoke.sh <archive.tar.gz> [RUN_ID]
```

The smoke does: layout plus unit tests -> image boundary checks (the Work image
cannot see the Judge or benchmark trees; the Judge's candidate user cannot sudo
and cannot read the verifier source) -> a GPU numerical regression -> running
the official-oracle archive as an ordinary candidate through a full public suite
-> one `sforge eval` of that archive on the smoke suite, asserting `valid: true`.

Expected values:

- the public run asserts `authoritative: false`, no `score`, 3 golden rows,
  3 probes, 5 fresh-process rows each with a distinct pid, 2 usable scale
  ladders, and no leak of `baseline_ms` / `speedup` / `denominator`;
- the smoke `sforge eval` must report `valid: true`;
- **public rows carry no absolute expected band at all.** They print raw
  candidate-only effective assignment TFLOP/s and relative retention, which are
  only self-comparable inside one measurement profile. Do not treat any number
  here as a pass threshold.

`scripts/smoke_test.sh` and `scripts/eval_archive_smoke.sh` use a dual-GPU
layout and need two distinct physical cards, while the formal run is
single-allocation and rejects `WORK_GPU` / `JUDGE_GPU`.

## 7. Formal run

```bash
launch/flash-kmeans/run.sh \
  --device-uuid GPU-00000000-0000-0000-0000-000000000000 \
  --calibration-epoch campaign-qualified-clock-policy-v1 \
  --judge-port 18284 \
  --model <model> --effort <effort> --hours 2 \
  [--allocation-key allocation-sha256:<64hex>] [--lease-id ID] \
  [--run-id ID] [--auth FILE] [--config FILE] [--tmux]
```

Qualify one scheduler-allocated whole NVIDIA GPU against the profile above
before this. Record the stable measurement profile (model, compute capability,
memory, driver, image IDs, software ABI, calibration protocol, and a
scheduler-issued calibration epoch tied to a documented clock/power policy).
Record UUID and observed current clocks only in deployment attestation. Without
a calibration epoch, the run is unqualified for a cross-run latency
leaderboard.

`--device-uuid` must be a scheduler-issued canonical whole-GPU UUID: no index,
alias or MIG identity. `--allocation-key` is recomputed by
`sforge.harness.resource_handoff.canonical_allocation_key` when omitted and
must equal the canonical value when supplied; `--lease-id` is generated once,
never reused, when omitted. **Single mode rejects `WORK_GPU` / `JUDGE_GPU`** in
both the launcher and the task runner and never falls back; the dual-GPU
path is reachable only through an explicit `--legacy-dual-gpu`, which in
turn refuses to inherit any single-allocation variable.

**Budget / interval / cooldown contract.**

| Quantity | Default | Constraint | Frozen? |
| --- | --- | --- | --- |
| `TASK_TIMEOUT` | 7200 s | positive, a multiple of 60, strictly greater than `EVAL_INTERVAL` | tunable — no contract-locked budget |
| `EVAL_INTERVAL` | 1800 s | positive, a multiple of 60 | tunable; since auto-eval is always off it only sets the scheduling tick and produces no automatic Judge run |
| `SUBMISSION_COOLDOWN` | 300 s | positive | **keep at 300** |
| auto-eval | always off (`--disable-auto-eval` is hard-coded in the runner) | — | **frozen** |
| `JUDGE_PORT` | **18284** | 1024-65535 and not 3128 | tunable |

Formal runs use `eval_interval: 1800`, `submission_cooldown: 300`, auto-eval
disabled, `internet: false`, and Work and Judge each 8 CPU + 32 g.

**Ports.** The scheduler must allocate a different free `--judge-port` for every
concurrent run on the same host; the runner binds it in preflight and refuses a
conflict.

**Version store.** `SFORGE_VERSION_STORE_ROOT` must point at a store carrying a
`.sforge-version-store.json` marker that registers this task id (resolved by
`launch/common.sh`'s `export_version_store_root`);
`SFORGE_AGENT_RUNTIME_DIR` at an agent runtime built by
`launch/build-agent-runtime.sh`; `CODEX_AUTH_JSON_PATH` at a regular
non-symlink file, mode 0600, owned by the launching user. `RUN_ID` must be
globally unique inside one store. **This task's runner has no preflight guard
for it**: a duplicate only surfaces after the deadline, as a
`VersionStoreCollision` while persisting the final version, and the whole agent
budget is wasted. Pick a fresh id by hand.

**Proxy.** A container cannot reach a proxy bound to host loopback, and the
launcher refuses to hand one to a container. Republish the host's egress proxy
on the Docker bridge and set `SFORGE_HTTP_PROXY` / `SFORGE_HTTPS_PROXY` /
`SFORGE_NO_PROXY` to that bridge address in `launch/mlsysforge.env`, then
`env -u` the shell's own `http_proxy` / `HTTP_PROXY` / … before launching.
Leave all three unset on a host with direct internet access. The run itself is
offline, so the proxy matters only for the build and the upstream clone.

## 8. Run artifacts and success criteria

With `<out> = outputs/flash-kmeans-paper-h100-v7-scratch`:

```text
<out>/runs/<RUN_ID>/
├── sforge-run.log
├── completion.json
└── <task_id>/
    ├── resource_lease.json
    ├── final_archive.tar.gz / final_result.json / final_snapshot.json / run_history.json
    ├── formal-precheck.json / formal-postcheck.json
    ├── final-<N>m-eval.log / final-<N>m-report.json
    ├── final-<N>m-private-judge-artifacts.tar.gz   (mode 0600)
    └── curve/curve.json, curve/curve.md
<out>/runs/<RUN_ID>-final-<N>m/<task_id>/submissions/manual-1/report.json
```

The last path is the **authoritative** standalone-final report.

Success means all of: `resource_lease.json` in state **CLOSED**;
`formal-precheck.json` and `formal-postcheck.json` both PASS; the standalone
final's `report.json` with `valid: true`; both files under `curve/` non-empty;
and `completion.json` written with `status: "passed"`. `completion.json` also
records `measurement_profile.comparability_key_sha256` and
`raw_latency_comparison.identity_sha256`, which decide whether the score can be
compared with any other run. `final_result.json` is SForge's best-so-far
summary and is **deliberately not** treated as the final point by
`collect_curve.py`; the authoritative score is the standalone final's
`report.json`.

**On failure, look at these first**, in order: `sforge-run.log` (agent and
harness) -> `formal-precheck.json` (the frozen boundary; single mode requires
`docker_pause_never_resume_final` plus frozen plus paused) -> `final-<N>m-eval.log`
(the standalone final) -> `formal-postcheck.json` (the rejection reason;
"independent Judge did not accept the frozen submission" is a candidate-level
failure, not a protocol failure) -> `curve/curve.json` (its `schedule_anchor`
field says whether the curve is anchored at seal or at snapshot).

A fenced run needs manual reconciliation — kill that run's `sforge serve`,
`docker rm -f` the paused Work container, move the run directory out of
`runs/` — or the coordinator refuses to attach any non-CLOSED old lease on the
same allocation. When serialising runs on one card, wait for the previous run's
containers to disappear from `docker ps` and then about 30 s more, or the next
run hits "claimed by a running container".

## 9. What to re-measure on another GPU

This task needs **no control measurement** on the target card: the Judge has no
baseline, no oracle timing and no reference cache, so every quantity is measured
inside each run. Re-measure and re-record only: the raw millisecond scores
(under a new comparability key); the NVML device-memory peaks, and with them
whether the 79 GiB floor is still right; and the headroom left by the timeout
ladder (per-deployment case 180 s, verifier 3300 s, final 3600 s) on a card
that is markedly slower than an H100.

Do **not** re-calibrate: the execution-profile key (it is a capability envelope,
not a model), the case counts (11 public / 22 final), the six score groups and
their weights, the extra-memory bound formulas, or the suite
`manifest_sha256` values. The 22 cases themselves are fixed by the Judge image's
bytes and do not move with the card.

Moving images with `docker save` / `docker load` keeps the contract identical,
but identical image bytes do not qualify a different GPU or driver by
themselves. On a pre-Hopper card, expect to need the Ampere-compat archive for
the host-side comparison (section 2) and to confirm that the card actually has
the device memory the final suite needs.

Raw latency may be compared only when both runs are qualified and their
recorded comparison identity is identical. That identity is the ordered pair
of the evaluation-contract SHA-256 and the measurement-profile comparability
key; the contract digest is deliberately not folded into the pure machine
profile hash. Merely satisfying the broad `single-cuda-sm80-80g` capability
envelope does not make two runs comparable.
