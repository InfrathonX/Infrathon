# Rebuild and run the LLaDA Decoder Optimization v2 task on a qualified host

This file is the **operational layer**. With this file, `docs/HOST_PREREQUISITES.md`
(repository root `docs/`) and this task directory, an operator or a coding agent can
rebuild the Work/Judge images, verify them, and run the task on a fresh host with an
admissible GPU. The agent-facing description of the task (metric, suites, Judge
decision) is in the task `README.md`.

You may use Codex, Claude, or another build agent, but **do not let the build agent
redesign the benchmark.**

Preserve these scientific identities byte-for-byte:

- the TaskCore and its resource manifest;
- the twelve rows, the committed 520-generation stratified workload
  (`images/judge/private_data/hidden_cases.json.gz`, bound by the workload
  policy's `hidden_cases_sha256`) and the complete frozen final workload
  for every interactive submission;
- the frozen `GSAI-ML/LLaDA-8B-Instruct` checkpoint binding (revision
  `6059b30c9531767ce47e0ca6a8161a6bb0c67693`), the deliberately slow
  `decoder_v001` seed, the shipping trusted runtime, the frozen workload,
  memory and quality-control policies, the protected runtime hashes, the
  trusted controller, worker, cohort sampling, row lifecycle, NVML sampler and
  quality modules, and the fail-closed run-wide gates;
- the agent-facing optimization contract.

The Judge image builds `FROM` the Work image and copies the verifier and the private
data on top; the Work image never carries `private_data`, `reference_quality.json` or
`host_reference`. The checkpoint is a host mount bound read-only by
`scripts/sforge_h100_entry.py` and validated by `scripts/validate_model_artifact.py`.
The task JSON is deliberately not a member of either resource manifest: it is the root
document that binds their digests, and each resource manifest also excludes itself to
avoid a recursive hash.

All relative paths below are relative to this task directory. `<repo>` is the
repository root. Every host-private value (GPU index, UUID, directory, port, proxy
address) is written as a placeholder — substitute your own.

---

## 1. Host prerequisites

Do `docs/HOST_PREREQUISITES.md` first: Docker >= 24 with the NVIDIA runtime as the
**default** runtime, your user in the `docker` group, `uv` plus the repository venv,
`jq flock rsync patch git sha256sum nvidia-smi`, and >= 150 GB free on the Docker data
root. On top of that this task needs:

| Extra need | Why | Checked by |
| --- | --- | --- |
| `ss` (`iproute2`) | the runner refuses to start when the Judge port is already listening | `scripts/run_fixed_budget.sh` preflight |
| `gzip`, `realpath`, `python3` on the host | workload commitment hashing, policy realpaths and layout validation | each script declares its own list through the `require_command` helper in `scripts/lib.sh` |
| An executable SForge checkout | the single-allocation preflight and the standalone final go through the repository virtualenv's `sforge` and the interpreter next to it; override the location with `SFORGE_BIN` if it is not the default one inside the checkout | `scripts/lib.sh` `require_sforge` |
| ~16 GiB free for the local Hugging Face repository | the checkpoint is mounted, never copied into an image | `scripts/validate_model_artifact.py` |
| Outbound network **at build time** | base image and pip packages. At run time the task declares `internet: true`: Work stays on an `--internal` Docker network and reaches the outside only through the run's CONNECT sidecar, started with `--allow-public-egress` (public hosts on 443/80 plus the model API hosts; private, loopback and link-local addresses refused), and the agent runs with live web search. `internet: false` restores the allowlist-only sidecar and an offline agent | task JSON |
| Two distinct GPUs **if you want the image/submission-plane smokes** | those two scripts require `WORK_GPU` and `JUDGE_GPU` to resolve to different UUIDs (see §6) | `scripts/smoke_test.sh`, `scripts/smoke_submission_plane.sh` |
| Agent CLI runtime + auth file | formal agent runs only, not for rebuilding and qualifying | `scripts/run_fixed_budget.sh` |

`sudo` is needed only for the container-toolkit step in the host prerequisites.

---

## 2. GPU envelope and who enforces each part

The envelope comes from the task JSON `portable.execution_profiles`. Two profiles are
declared with an identical capability envelope; they differ only in device count and in
how the run is driven.

| Item | Value | Enforced by |
| --- | --- | --- |
| Profile id (default) | `single-cuda-sm80-34g`, `count = 1` | `portable.defaults.execution_profile_id`; `scripts/run_fixed_budget.sh` asserts in its single-allocation branch that the default profile is this id and that its count is 1 |
| Profile id (legacy) | `dual-cuda-sm80-34g`, `count = 2` | selected only by `launch/llada-h100/run.sh --legacy-dual-gpu`, which hard-codes the id |
| Compute capability | `min_compute_capability = "8.0"` | `MIN_COMPUTE_CAPABILITY` in `images/judge/verifier/verify_decoder.py`, raised from `hardware_inventory()`; the host side re-checks it through `GPU_MIN_COMPUTE_CAPABILITY` in `scripts/lib.sh` |
| Device memory | `min_memory_bytes = 36507222016` (34 GiB) | `MIN_DEVICE_MEMORY_BYTES` in the same verifier; `GPU_MIN_TOTAL_MEMORY_MIB = 34816` in `scripts/lib.sh`, used by `assert_gpu_available` (`GPU_MIN_FREE_MIB` has no literal default and falls back to that same value) |
| Frozen memory policy floor | `min_device_memory_mib = 34816` | `memory_policy.json`, loaded by `load_frozen_memory_policy` and mirrored into the Work image's public `memory_contract.json` |
| Features | `cuda`, `bf16`, `nvml`, `read-only-model-mount` | `images/judge/verifier/nvml_sampler.py` for NVML; the read-only mount is bound by the task Docker adapter |
| Software ABI | `pytorch271-cu128-transformers4490-multiarch-v1` (PyTorch 2.7.1 / CUDA 12.8 / cuDNN 9 base, transformers 4.49.0) | both Dockerfiles; `portable/software-abi.json` carries the pinned spec digest |
| Container resources | Work and Judge: 8 CPU, 48 GiB memory each | task JSON `work` / `judge`; the launcher passes `--judge-cpu-limit 8 --judge-mem-limit 48g` explicitly for the final Judge |
| Host RAM / pinned memory | no separate requirement beyond the two 48 GiB container limits | — |

There is no MIG qualification: whole GPUs only. GPU model name, UUID and observed
clocks are recorded as provenance and are **not** admission criteria — but the model
name, compute capability and driver version do enter the measurement-profile key (§4).

The declared 34 GiB floor is a contract value and sits above the measured peak device
memory of both a full Judge final and a full public suite. Plan the card around 34 GiB.

---

## 3. External inputs

| Input | What it is / source with pin | Size | SHA | Where it is expected | Availability |
| --- | --- | --- | --- | --- | --- |
| LLaDA-8B-Instruct local repository | Hugging Face `GSAI-ML/LLaDA-8B-Instruct`, revision `6059b30c9531767ce47e0ca6a8161a6bb0c67693`, in the standard HF-cache layout (`models--GSAI-ML--LLaDA-8B-Instruct/` with `blobs/` and `snapshots/<revision>/`) | `logical_size_bytes = 16037449215`; `parameter_count = 8015581184` | 16 individually pinned files listed in `docs/model_manifest.json` | env var **`LLADA_MODEL_REPO`**, pointing at the repository root, mounted read-only | **public** — re-download by repo + revision |
| GSM8K | `openai/gsm8k`, config `main`, revision `e53f048856ff4f594e959d75785d2c2d37b678ee` | fetched at prepare time | revision pin | fetched by `scripts/prepare_benchmark_data.py` into the Judge-only workload | **public** |
| MATH | `EleutherAI/hendrycks_math`, revision `21a5633873b6a120296cce3e2df9d5550074f4a3` | fetched at prepare time | revision pin | same | **public** |
| HumanEval | `openai/openai_humaneval`, config `openai_humaneval`, revision `7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544` | fetched at prepare time | revision pin | same | **public** |
| MBPP | `google-research-datasets/mbpp`, config `full`, revision `4bb6404fdc6cacfda99d4ac4205087b89d32030c` | fetched at prepare time | revision pin | same | **public** |
| Prompt-construction reference | lm-evaluation-harness v0.4.8 @ `6d2abda4fd171e68a8789330c4149e37c1ca0bda`; Fast-dLLM @ `a9b81e4caa240c8cad4f7dc1889ff4852a0fca5b` | — | commit pins in `docs/dataset_manifest.json` | consulted by the generator, not vendored | **public** |
| Committed hidden workload | `images/judge/private_data/hidden_cases.json.gz` — the 520-case stratified final suite, produced deterministically by the generator from the pins above | ships with the task | `hidden_cases_sha256 = d0c4084b570d27ba0318fed1d66732f27c4a60a434095506015e2a8029fda64e` bound in the workload policy, the control manifest and `portable/final-suite-manifest.json` | already in the task directory | **ships with the repository**; regenerating it from the pins reproduces the same digest byte-for-byte, gzip container included (§4) |
| Committed public fixtures | `images/work/public_tests/public_cases.json` — the 44-generation visible suite | ships with the task | `public_sha256 = cde46815812b860ac4a4732bb800f40fce49c9caec53d5e4e76a3eb4899b6163` recorded in `docs/dataset_manifest.json` | already in the task directory | **in-tree frozen instance**: the generator is provenance, not a recipe — eleven synthetic HumanEval entries carry a prompt-token attestation no generator run emits |
| Producer quality reference | `images/judge/private_data/reference_quality.json` plus one `reference_cache/<reference profile key>.json` per GPU model + compute capability | ships with the task | bound by `portable/resource-manifest.json` | baked into the Judge image with the whole cache; the verifier selects the entry for the card it observes | **ships with the repository for the qualified profiles**; a new profile must produce its own (§4) |
| Calibration / control archives | `artifacts/vanilla-reference.tar.gz`, `corrected-decoder-reference.tar.gz`, `positive-near-threshold-control.tar.gz`, `negative-quality-control.tar.gz`, `near-cap-stress-reference.tar.gz` | small | producer, positive and negative control digests recorded in `quality_controls.json` | `artifacts/` (gitignored) | **regenerated locally, deterministically** by `scripts/make_reference_archives.py` from `host_reference/corrected_decoder/candidate.py` and the shipped seed; no external download |
| Agent CLI runtime and auth file | agent runtime tree marked with `.sforge-agent-runtime.json`, version pinned to `CODEX_CLI_VERSION_PIN` in `launch/common.sh`; a mode-600 `auth.json` owned by the launching user | — | tree SHA recorded in the runtime marker | `SFORGE_AGENT_RUNTIME_DIR`, `CODEX_AUTH_JSON_PATH` | **provided by the operator** (build the runtime with `<repo>/launch/build-agent-runtime.sh`, supply your own credential file) |

Nothing in this task is owner-provided *data*: every scientific input is either a public
pinned artifact or regenerated in place. The only operator-provided items are the agent
credential and the agent runtime, and neither touches the evaluation.

---

## 4. What is frozen, what each deployment captures

This is design, not a defect.

**Frozen once, identical on every host.** The twelve rows and their shapes, the 520
committed generations and 216,320 requested slots, the per-dataset quota of 130 targets,
the dataset revisions, the generator's own SHA-256 and the task-owned ordering seed, the
context-exclusion rule (`exact_prompt_tokens + gen_length > 4096` removes the whole
target) together with the one exclusion it produces, and the digest
`hidden_cases_sha256` that binds all of it.

**Re-running `scripts/prepare_benchmark_data.py` on a released tree is refused
cleanly.** The generator authenticates the frozen ordering nonce, resolves the pinned
dataset revisions and checks the workload policy **before** it writes anything, so a
released tree comes out of a refused run byte-identical:

```
RuntimeError: data regeneration may update only an UNQUALIFIED/PENDING workload policy
```

The shipped policies are `qualification_status: QUALIFIED` / `release_status: FROZEN`;
regeneration proper belongs to the qualification flow, which starts from an
`UNQUALIFIED/PENDING` policy and ends in a reviewed re-freeze.

To verify the frozen bytes instead, use the generator's two non-destructive paths —
`--dry-run`, which computes every digest and writes no file anywhere, and
`--output-root <scratch tree>`, which writes the generated files into a copy and leaves
`--task-root` alone. The hidden suite reproduces byte-for-byte; the public fixtures and
the dataset manifest are in-tree frozen instances and do not. Cheaper still, and
sufficient for a rebuild: `python3 scripts/validate_workload_profile.py --task-root .`
plus the §6.1 suite, which check every digest the runner checks.

**The `ORDERING_SEED` literal in the generator is frozen data, not a label.**
Every stratified sort, tie-break and row assignment is keyed on those exact
bytes, so editing the literal — renaming it, reformatting it — silently draws a
different 520-case suite that still passes every shape and count check. The
generator compares `sha256(ORDERING_SEED)` against the frozen
`ordering_seed_sha256` at start-up and dies if they differ.

**The pinned dataset revisions are not enforced by the endpoint that serves the
rows.** The Hugging Face datasets-server `/rows` endpoint ignores a `revision`
query parameter and answers from the repository's current default branch. The
generator therefore resolves each pinned revision against that branch itself
and refuses to generate when the data files differ, rather than freezing
whatever upstream published last.

The two digests in `docs/dataset_manifest.json` that describe the *tool* rather
than the data — `generator_sha256`, and the `dataset_manifest_sha256` pointers
that follow it — are provenance, not case bytes. When a portability fix edits
the generator without changing a single generated byte, refresh those and
rebind the pointers in `workload_policy.json`,
`images/judge/private_data/workload_policy.json` and `portable/` — including
`portable/final-suite-manifest.json`, which carries its own
`dataset_manifest_sha256`. `scripts/refresh_portable_manifests.py` rebinds all
of the `portable/` side, that pointer included, and then re-derives the task
JSON digests; run it and then `--check`. Leave `hidden_sha256`,
`public_sha256`, `ordering_seed_sha256` and the counts untouched.
`quality_policy.control_build_identity.dataset_manifest_sha256` records what the
quality-control build saw and keeps that value.

**Frozen membership for every scored round.** Interactive and terminal evaluation
both execute all 520 committed cases; there is no round-dependent subset. Trusted
`agent-N` and submission identity are audit fields, not workload selectors. Neither
phase redraws the suite. Public diagnostics are a distinct workload.

**Captured per deployment.** The producer quality reference and its per-profile cache
entry. The Judge compares a candidate's answers pairwise against a *producer reference
measured on the same GPU model under the same software ABI*. The selection and refusal key
is

```text
reference_profile_key = SHA-256( canonical_json{ gpu_name, compute_capability } )
```

plus the software ABI the references were measured under (`cuda_runtime`, `torch`,
`transformers`, pinned as `REFERENCE_SOFTWARE_ABI` in the verifier and checked against the
running Judge). Field lists are in `portable/calibration-protocol.json`. The cache file is
`reference_cache/<reference_profile_key>.json`; the Judge re-derives the key from the live
device and from the reference's own recorded profile and **refuses a reference of another
GPU model / compute capability, or a Judge whose ABI differs**, failing closed rather than
scoring against a foreign baseline. The fuller five-field `measurement_profile_key`
(`gpu_name, compute_capability, driver_version, work_image_id, judge_image_id`) is
recorded in every reference and report as provenance only: a host driver update or an
image rebuild with unchanged software keeps the reference. A new card model, a new
compute capability, or a new CUDA / torch / transformers version requires its own
producer run (roughly 3 h of GPU time).

**What this means for comparing scores.** Raw tokens/s are comparable only when both runs
are qualified and their recorded comparison identity is identical. That identity is the
ordered pair of the evaluation-contract SHA-256 and the measurement-profile comparability
key, recorded as `raw_score_comparison` in `experiment-manifest.json` and
`completion.json`; the contract digest is deliberately not folded into the pure machine
profile hash. Merely satisfying the broad `single-cuda-sm80-34g` capability envelope does
not make two runs comparable: the memory cap is device-relative and the quality reference
belongs to one measurement profile. Pass `--calibration-epoch` only for a campaign
whose operator has fixed and documented a clock/power policy — without one the manifest
records `unqualified_missing_calibration_epoch` and the run must not enter a cross-run
tokens/s leaderboard.

---

## 5. Build

Run everything from this task directory. Do the pure CPU/meta checks **before** any
image or GPU operation:

```bash
cd <repo>/tasks/latest/llada-h100-decoder-optimization-v2-vanilla-seed
python3 -B -m pytest -p no:cacheprovider tests -q
python3 -B scripts/validate_layout.py .
python3 scripts/refresh_portable_manifests.py --check
```

Those tests recompute the scientific resource digest, the delivery/build-resource digest,
the manifest byte digests and the real public-case, interactive-row and final-row counts
from the shipped workload, policies and verifier. Treat any mismatch as a hard stop.

If `artifacts/` is empty (it is gitignored), regenerate the deterministic archives first,
using the same near-cap stress target the frozen policy was qualified with (recorded as
`stress_extra_mib` in `memory_qualification_fixtures.json`):

```bash
python3 scripts/make_reference_archives.py --task-root . --stress-extra-mib 61440
sha256sum artifacts/corrected-decoder-reference.tar.gz   # must equal quality_controls.json
```

Then build both images:

```bash
JUDGE_GPU=<index of the card the Judge will use> \
LLADA_MODEL_REPO=<path to the local LLaDA-8B-Instruct repository> \
  bash scripts/build_images.sh
```

`scripts/build_images.sh` in order: copies `judge/run_judge.py` and the source-admission
module into the build contexts (the Work copy becomes `runtime/source_policy.py`); when
`JUDGE_GPU` is set, runs `scripts/select_reference_profile.py --judge-gpu <n>
--check --allow-pending`, which refuses a card whose GPU model has no reference (the
Judge image carries every `reference_cache/` entry and the verifier selects the one for
the card it observes); runs `scripts/validate_layout.py`; builds Work
then Judge (`--build-arg WORK_IMAGE=...`); and finally compares the source digest taken
before and after the build — its only `die` is *"image sources changed during the build;
discard and rebuild"*. It writes `<repo>/outputs/<task directory name>/build/image-manifest.json`
with both image IDs and the source digest.

To narrow the CUDA architectures compiled into the Work image, pass a `--build-arg` for
`TORCH_CUDA_ARCH_LIST` (the Dockerfile declares it as an `ARG` with a multi-architecture
default). Keep the resulting source digest and the exact Work/Judge image IDs with the
run: the reference cache, the frozen policies and the calibration provenance pin the
image and source identities they were produced on.

**`scripts/` and `tasks/` are inside the image source digest.** Editing any script —
including `lib.sh` and the validators — obliges a rebuild; otherwise the runner's
image/source freshness check refuses to start.

---

## 6. Post-build verification

### 6.1 No GPU needed

```bash
python3 scripts/validate_layout.py .
python3 -B -m pytest -p no:cacheprovider tests -q
find scripts -maxdepth 1 -name '*.sh' -print0 | xargs -0 -n1 bash -n
PYTHONPYCACHEPREFIX="$(mktemp -d)" python3 -m compileall -q .
```

**Never run `python3 -m compileall -q .` without `PYTHONPYCACHEPREFIX`.** Plain
`compileall` writes `__pycache__/` trees into the task directory, one of
them inside the seed solution
(`images/work/workdir/decoder_lab/solutions/decoder_v001/`), and the **next**
`scripts/build_images.sh` then fails closed with

```
ERROR: provided starter fails Judge source admission: unsupported submission file: solutions/decoder_v001/__pycache__/__init__.cpython-3xx.pyc
```

The `PYTHONPYCACHEPREFIX` form above compiles exactly the same files and puts
every `.pyc` under a throwaway directory instead, so the check is safe to run
in any order. `-B` on the pytest line does the same job for the test run. If a
build ever reports that error, the tree already has caches in it; clear them
with `find . -type d -name __pycache__ -prune -exec rm -rf {} +` and rebuild.

Expected results:

| Command | Expected |
| --- | --- |
| `validate_layout.py .` | `layout validation PASS: <task dir>` |
| `pytest tests -q` | run with the repository virtualenv interpreter, which is the authoritative one; a system `python3` without the harness package importable produces many spurious failures. Environment-dependent tests in `tests/test_eval_artifacts.py` may fail or skip depending on the host |
| `refresh_portable_manifests.py --check` | exit 0, no output |
| `compileall` with `PYTHONPYCACHEPREFIX` | exit 0, and `find . -type d -name __pycache__` stays empty |

### 6.2 Built-policy boundary (images only, no GPU)

```bash
python3 scripts/validate_built_memory_policy.py \
  --judge-image llada_decoder.judge.llada_h100_decoder_optimization_v2:local \
  --work-image  llada_decoder.work.llada_h100_decoder_optimization_v2:local \
  --require-frozen
python3 scripts/validate_built_workload_policy.py \
  --judge-image llada_decoder.judge.llada_h100_decoder_optimization_v2:local \
  --work-image  llada_decoder.work.llada_h100_decoder_optimization_v2:local \
  --require-frozen
```

Both validators inspect the built images over the Docker socket and re-exec themselves
through `sg docker` when it is unreachable, the same way `ensure_docker_group` in
`scripts/lib.sh` handles the shell entry points — so they work in a login session that
predates your `docker` group membership. The re-exec is skipped when the socket already
answers, when your user is not listed in the `docker` group, or when `sg` is absent; in
those cases the real inspection error is reported.

Both must pass under `--require-frozen` (both shipped policies are
`release_status: FROZEN`). They make a stronger claim than checking the source JSON: the
tracked policy bytes must equal the Judge build-context copy **and** the bytes inside the
built Judge image, and the Work image must contain neither private policy — only the
sanitized `/workspace/public_tests/memory_contract.json`, whose bytes are also checked.

### 6.3 GPU smokes

```bash
# Two DISTINCT GPUs required by these two scripts.
WORK_GPU=<index A> JUDGE_GPU=<index B> LLADA_MODEL_REPO=<repo path> \
  bash scripts/smoke_test.sh
WORK_GPU=<index A> JUDGE_GPU=<index B> LLADA_MODEL_REPO=<repo path> \
  bash scripts/smoke_submission_plane.sh artifacts/corrected-decoder-reference.tar.gz
```

Expected: `LLaDA dual-H100 decoder smoke PASS (frozen-release-policy)` and
`Dual-H100 submission-plane smoke PASS`. `smoke_test.sh` runs the public suite
`--cpu-only` and `--quick` inside the Work container, proves the Work image carries no
Judge assets, and proves an unprivileged user inside the Judge image cannot read the
private data. `smoke_submission_plane.sh` additionally asserts, from inside the Work
container, `device_count == 1`, capability `>= (8, 0)` and total memory `>= 34 GiB`.

**On a single-GPU host these two scripts cannot run** (they refuse equal indices). Use the
archive path in §6.4 as the GPU-side acceptance test instead.

### 6.4 One-card acceptance: reference replay

`scripts/eval_archive.sh` needs only `JUDGE_GPU`, so it works on a one-card host. The
mandatory release gate is the ordinary 520-case replay of the byte-identical corrected
archive:

```bash
LLADA_REFERENCE_REPLAY=1 LLADA_MODEL_REPO=<repo path> JUDGE_GPU=<index> \
  bash scripts/eval_archive.sh artifacts/corrected-decoder-reference.tar.gz <replay run id>

jq -e '
  .status == "PASS" and .mode == "final" and .release_evidence == true
  and .reference_replay == true and .generation_count == 520
  and .requested_slots == 216320
' <repo>/outputs/<task directory name>/runs/<replay run id>/evaluation-attestation.json
```

Run IDs are write-once; `LLADA_RESUME=1` is admissible only with an identical archive,
source, image, model, GPU, runtime, policy and run id. The vanilla seed archive on the
same suite is *expected* to die with `candidate worker protocol timed out` — it is
deliberately the slow baseline, not a regression.

If you are standing up a **new** measurement profile, the replay is only the last step;
run the full producer → positive/negative control → freeze → replay calibration chain
first. Two ordering rules there are not optional: materializing `reference_quality.json`
changes the Judge source identity, so rebuild the images and re-run the two built-policy
validators before any control; and the control aggregator derives the threshold itself
from exactly one canonical 520-case report per precommitted control — there is no
hand-tuned quality margin, and the finalizer accepts no timeout flags.

---

## 7. Formal run

```bash
LLADA_MODEL_REPO=<path to the local LLaDA-8B-Instruct repository> \
EVAL_INTERVAL=1800 SUBMISSION_COOLDOWN=300 \
SFORGE_VERSION_STORE_ROOT=<version store that registers this task id> \
SFORGE_AGENT_RUNTIME_DIR=<agent runtime tree> \
<repo>/launch/llada-h100/run.sh \
  --model <model id> --effort <level> --hours 1 \
  --run-id <globally unique RUN_ID> \
  --device-uuid GPU-00000000-0000-0000-0000-000000000000 \
  --judge-port 18384 \
  [--allocation-key allocation-sha256:<64 lowercase hex>] \
  [--lease-id <never-reused label>] \
  [--calibration-epoch <qualified campaign epoch>] \
  [--auth <path to auth.json>] [--tmux]
```

This is the **default single-allocation profile**. The legacy two-GPU profile is
retained for comparison only and is reached with an explicit flag:

```bash
WORK_GPU=<index A> JUDGE_GPU=<index B> \
  <repo>/launch/llada-h100/run.sh --legacy-dual-gpu --model <model id> --effort <level> --hours 2
```

Single mode **rejects** `WORK_GPU` / `JUDGE_GPU` and never falls back to the legacy path;
`--legacy-dual-gpu` symmetrically rejects inherited single-allocation identity variables.

### 7.1 What the launcher checks and what it forwards

- `--device-uuid` must be a canonical whole-GPU UUID (`GPU-` plus 8-4-4-4-12 lowercase
  hex) — never an index, alias or MIG identity. The single wrapper computes the canonical
  `allocation-sha256` from that device set; a scheduler-supplied `--allocation-key` must
  match it exactly. `--lease-id` is optional: when omitted a never-reused
  `lease-llada-<hex>` label is generated once. All three survive the docker-group re-exec.
- `--judge-port` defaults to **18384**, must be 1024–65535 and must not be 3128 (the
  restricted CONNECT proxy). It must also be genuinely free — the runner checks the
  listener list and dies before Work starts. Concurrent runs on one host need distinct
  ports.
- `--hours N` / `--timeout SECONDS` set `TASK_TIMEOUT`; the launcher default is 7200 s.
- `--model` takes a full model ID served by the configured model API (or a launcher
  short name). The runner accepts `REASONING_EFFORT` in `high|max|xhigh|ultra`
  (Claude Code agents: `high|xhigh|max`). `CODEX_CLI_VERSION` must equal the pin in
  `launch/common.sh`, and the agent runtime marker's version must equal it too.
- `SFORGE_VERSION_STORE_ROOT` must point at a store containing
  `.sforge-version-store.json` **and registering this task id** at
  `<store>/tasks/<task directory>/task.json`; otherwise the launcher dies rather than
  silently keeping no replayable candidates. The registration is keyed by task
  **directory**: a store entry written under another directory name must be *renamed*,
  not duplicated. Two directories carrying the same `task_id` make the coordinator
  refuse to start with `VersionStoreError: duplicate task_id in version store:
  llada_h100_decoder_optimization_v2`, which the launcher surfaces only as
  `ERROR: service did not become healthy` — the real reason is in the run
  directory's `judge-server.log`.
- Proxy: the fixed-budget body starts its own restricted CONNECT sidecar on the internal
  network and exports `SFORGE_HTTP_PROXY` / `SFORGE_HTTPS_PROXY` at that
  container-reachable address. The launcher's loopback-proxy guard only inspects the entry
  script it execs, and the single-allocation entry is a thin wrapper, so in single mode an
  ambient `http_proxy` pointing at host loopback will still abort the launch. Either unset
  the ambient proxy variables or export `SFORGE_HTTP_PROXY=http://<docker-bridge-ip>:<port>`
  before launching. `127.0.0.1` and `localhost` are forced into `NO_PROXY` for host-local
  control traffic.
- `LLADA_MODEL_REPO`, `EVAL_INTERVAL` and `SUBMISSION_COOLDOWN` are **not** set by the
  launcher; they are passed down as environment variables and survive the re-exec.

### 7.2 Budget / interval / cooldown contract

| Quantity | Value | Constraint |
| --- | --- | --- |
| `TASK_TIMEOUT` | 7200 s default | set by `--hours` / `--timeout`; must be at least 90 s |
| `EVAL_INTERVAL` | 1800 s default | — |
| `SUBMISSION_COOLDOWN` | **frozen at 300 s** | the runner dies with *"SUBMISSION_COOLDOWN must match frozen workload policy (300)"* on any other value; the number comes from `workload_policy.json` |
| `FINAL_EVAL_TIMEOUT` | **frozen at 3900 s** | same treatment; it is the frozen final outer deadline |
| Final inner budget | **3600 s**, exported to the Judge container as `LLADA_FINAL_JUDGE_TIMEOUT` | from the same frozen policy; see §8 for what happens when a candidate exceeds it |
| Standalone-final orchestration ceiling | 7200 s | task JSON `judge.eval_timeout`; the runner refuses a frozen outer deadline above it |
| Auto-evaluation | off | only a manual `sforge-submit` triggers the interactive Judge |
| `RUN_ID` | must be globally unique | the runner dies in preflight if the run directory or `<version store>/tasks/<task directory>/runs/<RUN_ID>` already exists, so a collision costs nothing instead of the whole budget |

`submission_cooldown` is never taken from run configuration; the runner injects it from
the frozen policy.

---

## 8. Run artifacts, success criteria, triage

### 8.1 Where things land

Under `<repo>/outputs/<task directory name>/runs/<RUN_ID>/`:

- run root: `resource_lease.json`, `completion.json`, `experiment-manifest.json`,
  `formal-precheck.json`, `full-model-validation.json`, `runtime-provenance.json`,
  `sforge-run.log`, `<task_id>/final-<N>m-eval.log`;
- `<task_id>/`: `submissions/agent-N/submission.tar.gz`, `final_archive.tar.gz`,
  `final_result.json`, `final_snapshot.json`, `run_history.json`.

Build evidence stays at `<repo>/outputs/<task directory name>/build/image-manifest.json`
and the two image build logs. The private Judge artifact
(`/logs/verifier/decoder_private.json`) never enters the Work container.

### 8.2 Success

- `scripts/validate_formal_artifacts.py pre` and `post` both PASS (the runner runs them
  and writes `formal-precheck.json`);
- `resource_lease.json` state is **CLOSED** (single-allocation profile only — the legacy
  profile has no lease state machine);
- the launcher exits 0 and `completion.json` is written;
- the 520-case standalone final reports `valid=true`.

A candidate rejected on the quality hard gate still closes the lease normally, and the
post-check reports *"independent Judge did not accept the frozen submission"* by design —
that is a candidate-level result, not an infrastructure failure.

**A final fenced on the inner budget is different.** The frozen final
`inner_timeout_seconds = 3600` is enforced inside the Judge through
`LLADA_FINAL_JUDGE_TIMEOUT` (`judge/run_judge.py` reads that variable for every round id
that is not `agent-N`). When a candidate exceeds it the Judge kills the verifier before it
can write `decoder_private.json`; the harness marks the evaluation
`evaluation_status: infrastructure_error` with the summary *"Evaluation infrastructure did
not complete authoritatively"*; the coordinator's final-judge path raises, fails the lease
closed into state **FENCED**, returns 409 to `sforge eval`, and the runner exits without a
post-check. The consequence: "candidate too slow" and "infrastructure broke" are
indistinguishable in the result, and that run produces no usable final score. This is a
known design limit, not a misconfiguration.

### 8.3 What to look at first on failure

| Symptom | Look at |
| --- | --- |
| Never started | `sforge-run.log` and the preflight output: UUID flock, NVML idle attestation, Judge port already in use, `RUN_ID` already in the version store, image/source freshness, agent runtime version, auth file mode |
| Build refused | `logs/build/work-image.log`, `logs/build/judge-image.log`, and the `image sources changed during the build` message |
| Judge refuses the reference | the verifier's fail-closed message about a reference from another measurement profile (GPU model / compute capability, or software ABI) — run `scripts/select_reference_profile.py --judge-gpu <n> --check` and, if there is no entry for this GPU model, calibrate (§4) |
| Interactive round invalid | the sanitized aggregate in the round report: which gate is false. Quality uses the same qualified hard gate as terminal evaluation |
| Final invalid | the quality hard gate; compare the reported producer-relative retention against the frozen threshold |
| Final fenced | `resource_lease.json` state and its `fail_closed` reason string, plus `<task_id>/final-<N>m-eval.log`; see §8.2 |
| Launch refused for GPU occupancy | the conservative container claim check in `scripts/lib.sh` treats a long-lived container with `NVIDIA_VISIBLE_DEVICES=all` as holding the card even when it computes nothing; clear those containers or pick another card |

### 8.4 What to re-measure on another GPU

1. **Same model, same driver, same two image IDs:** nothing. The device profile key is
   unchanged; just pass the new `--device-uuid`.
2. **Different GPU model, different driver, or any image rebuild:** the whole calibration
   chain (producer → controls → freeze → replay), then add the new
   `reference_cache/<device profile key>.json` to `portable/resource-manifest.json` and
   `portable/build-resource-manifest.json` and run
   `python3 scripts/refresh_portable_manifests.py`.
3. **Re-measure the device-memory envelope** (1 Hz NVML sampling across one full
   trusted-Judge final and one full public suite) and compare against the §2 floor. Anything
   that moves a peak upward — a new driver, a different PyTorch minor, an added case —
   must be re-measured rather than assumed absorbed by the floor.
4. **Do not retune** the timeout ladder, the row-sentinel growth gate, the source policy or
   the 300 s cooldown to fit a slower card: they are qualification ceilings and workload
   constants, and a post-freeze replay may validate variance but may never be used to tune
   thresholds, caps or deadlines.
5. **A different software ABI** (CUDA, torch or transformers version) is a new evaluation
   identity, not a tuning knob: it changes the profile's `software_abi` digest and requires
   full re-qualification.

## Complete scored evaluation (selection protocol v3)

Every interactive Judge submission and terminal evaluation runs the same frozen
520 generations, dataset records, seeds, twelve row shapes, row worker lifecycle,
quality/memory gates and final case/row/inner/outer deadlines. Public evaluators
remain diagnostics. The trusted `agent-N` identity and `FINAL_VERSION` selection
are preserved; interactive candidate-version overrides are forbidden. Each
interactive `(round_id, submission_id)` has a separate root-only checkpoint scope.
The host binds the submitted archive digest and runtime/image identity through
the private context, never candidate-visible environment. Scores from any earlier
protocol with a smaller per-round subset are a different workload and are not
comparable. Qualifying and calibrating the Judge for a measurement profile is
required before release scoring on it.
