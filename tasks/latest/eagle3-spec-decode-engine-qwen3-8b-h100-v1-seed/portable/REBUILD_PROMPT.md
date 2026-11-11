# Rebuild the EAGLE-3 Speculative Decode Engine v1 on a qualified whole-GPU host

Use this task bundle to reproduce the Work/Judge images and qualify a new
single-CUDA deployment. You may use Codex, Claude, or another build agent, but
do not let the build agent redesign the benchmark. What the task *is* — the
metric, the gates, the suites — is specified in `images/work/AGENT_TASK.md`;
this file is only how to build, verify and run it.

Preserve these scientific identities byte-for-byte:

- the TaskCore and its resource manifest;
- the six cells (chat, math, code, long context, mixed batch 8, thinking-on
  math) with their public prompt pools, the committed hidden anchor / final /
  warm-up prompt slices (`images/judge/private_data`, bound by
  `commitments.json` and `protected_hashes.txt`), and the per-run HMAC anchor
  sampling;
- the frozen Qwen3-8B target and Tengyunw EAGLE-3 head bindings, the complete
  seed engine (also the Judge's root-only seed reference), the qualified
  policy (`eps_hard`, `p_soft`, the eight margin-CDF tiers, the seed-relative
  cap multiples), the trusted controller, worker, sampling and scoring, and
  the eleven run-wide fail-closed gates;
- the agent-facing optimization contract.

The Judge image builds `FROM` the Work image and copies the verifier, the
private data and the seed reference on top; the Work image never carries
`private_data` or `host_reference`. The frozen target and head are host
mounts bound read-only by `scripts/sforge_h100_entry.py` and identified by
`scripts/model_artifact_identity.py`.

## 1. Host prerequisites

`docs/HOST_PREREQUISITES.md` lists everything that is not task-specific:
Docker >= 24 with the **NVIDIA runtime as default**, your user in the `docker`
group, `uv` with Python 3.10-3.13, `jq flock rsync patch git sha256sum
nvidia-smi`, CPU `torch` + `numpy` in the repo venv, **20-25 GB free on
Docker's data root for one task** (17-20 GB of that is this task's Work + Judge
pair), and outbound access to Docker Hub, Hugging Face and GitHub. Check them
first. The larger disk figure in that document is the budget for all tasks
together, not for this one.

Task-specific extras:

| Extra | Why |
| --- | --- |
| ~18 GB more free in the Hugging Face cache, plus ~2 GB beside it | the pinned Qwen3-8B BF16 snapshot (~16 GB) and the converted draft head (~800 MB written twice: conversion output and frozen copy) |
| a pinned **EAGLE upstream checkout** | `scripts/convert_head.py` imports `eagle.model.cnets` and `eagle.model.configs` to convert the draft head. The checkout is host-only and never enters an image |
| 64 GiB of host RAM per container | Work and Judge each take `cpu_limit: 8`, `mem_limit: 64g` |
| `ss` (iproute2) | the runner checks the coordinator port in preflight and dies if it is busy |
| **no host torch needed** | the head conversion runs *inside the Work image* (section 3) |

At run time the task is offline (`internet: false`): the Work container reaches
only a restricted proxy sidecar (`scripts/restricted_proxy.py`, the
flash-kmeans `restricted_connect_proxy.py` byte for byte) that admits CONNECT
solely to `api.openai.com` / `auth.openai.com` / `chatgpt.com:443`, refuses
everything else with 403, and relays origin-form HTTP on 3130 to the fixed
Judge server; the **Judge container runs `--network none`**,
and both images force Hugging Face offline mode. Only the image build and the
one-time model/head fetch need the network.

## 2. GPU envelope

The task JSON's `portable.execution_profiles` carries exactly one profile,
`single-cuda-sm80-34g`: `count = 1`, `min_compute_capability = "8.0"`,
`min_memory_bytes = 36507222016` (34 GiB), features `cuda / bf16 / fp32 /
eager-attention / nvcc / read-only-model-mount`. MIG is not qualified.

Who enforces it:

- `scripts/preflight.py` carries `MIN_COMPUTE_CAPABILITY = (8, 0)` and
  `MIN_DEVICE_MEMORY_MIB = 34 * 1024`;
- `scripts/lib.sh` defaults `GPU_MIN_FREE_MIB` to 34816 and
  `GPU_MIN_COMPUTE_CAPABILITY` to 8.0, enforced by `assert_gpu_available`;
- `scripts/smoke_submission_plane.sh` re-checks `get_device_capability(0) >= (8, 0)`;
- `scripts/sforge_h100_entry.py` validates only the execution-profile id.

There is **no GPU model or compute-capability assertion anywhere in the trusted
runtime or verifier**, and no GPU model-name gate. Exclusivity is a
separate check: the target card must have no compute processes *and* must not
be claimed by any running container through `NVIDIA_VISIBLE_DEVICES` or an
`EAGLE3_*_GPU_UUID` variable (`running_container_claims_gpu` ->
`assert_uuid_unassigned`).

Measured NVML peaks on an idle H100 are about 21.8 GiB for the Judge (final
suite) and 21.7 GiB for Work (public suite), so the measured floor is 22 GiB.
The declared floor is 34 GiB: a floor is a contract, and lowering it would
admit cards the task was never qualified on.
The real resident set is the frozen Qwen3-8B target plus
the EAGLE-3 head plus one tree KV cache; 34 GiB is that need plus the
launcher's own margin.

`TORCH_CUDA_ARCH_LIST` is a Work-image build arg defaulting to
`8.0;8.6;8.9;9.0;10.0`, so a candidate's self-compiled `cpp_extension` covers
the whole envelope. Narrow it only when the fleet is known to be
single-architecture; widen it if your target architecture is outside that list.

## 3. External inputs

| What | Source and pin | Size | SHA if pinned | Env var / path | Availability |
| --- | --- | --- | --- | --- | --- |
| Base image for both images | `pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel` (torch 2.8.0+cu129, CUDA 12.9 nvcc, Python 3.11, Transformers **4.53.1**, Triton 3.4.0) | ~9 GB | by tag | pulled by `scripts/build_images.sh` | public |
| Qwen3-8B target snapshot | `Qwen/Qwen3-8B` @ `b968826d9c46dd6066d109eabc6255188de91218` (Apache-2.0, BF16) | ~16 GB | snapshot `config.json` SHA-256 `f7c4eadf…`, checked by `scripts/model_artifact_identity.py` | `EAGLE3_TARGET_REPO`, default the HF-cache repo root `models--Qwen--Qwen3-8B`; mounted `/opt/models/qwen3-8b-hf:ro` | public, ungated |
| EAGLE upstream checkout | `SafeAILab/EAGLE` @ `cb7e0841fe0c206c6ed74a197ad5e2a1f13f5a2b` (Apache-2.0) | small | commit pin; the six depended-on files and their SHA-256 are listed below | placed on `PYTHONPATH` for the conversion; host-only, **never copied into either image** | public |
| EAGLE-3 draft head source | `Tengyunw/qwen3_8b_eagle3` @ `2a1059d51f622b8cad7d7d72840153ffea5488a0` (MIT) | ~2 GB | revision pinned; the converted output is digest-checked | downloaded by `scripts/convert_head.py`, which audits both candidate head repositories but only the Tengyunw one is frozen and shipped | public, ungated |
| Converted + frozen draft head | produced here by conversion, then `scripts/freeze_head_artifact.sh` | ~2 GB | `model.safetensors` SHA-256 `596f0d62006793a8fa70cbd0ea5e7d907504a1b30ec349c841bf357af529b4b0`, `config.json` SHA-256 `d5816b5d7c70e68e07a493942f4a2905a14e403bd2fce3cf92750478cf101685`; also needs `MANIFEST.json` (its `revision` field must equal the pinned revision) and `SOURCE_README.md` | `EAGLE3_HEAD_DIR`; mounted `/opt/models/eagle3-head:ro` | **reproducible from the public sources above** — the conversion reproduces the manifest digest exactly |
| Hidden prompt slices, policy, seed reference | already in the repository, inside `images/judge/private_data` | small | bound by `commitments.json` and the 46 entries of `protected_hashes.txt` | baked into the Judge image | in-repo |

There is no owner-provided input: every external input is public and pinned,
and the frozen head is reproduced locally from them.

The EAGLE files the seed depends on, at the pinned commit:

| File | SHA-256 |
| --- | --- |
| `eagle/model/ea_model.py` | `f834fc37d350b9109ad91ac27a29eb2be59153dc0aac48ab753c6e43aa91d506` |
| `eagle/model/cnets.py` | `b0e2512db4529c9689ffd768e9a2812059a5715d786686a131a751c56fbf3292` |
| `eagle/model/utils.py` | `68dadf1977dd76624f87d0dc5ce9f840783ccaeb7554861f1206daa664367fe9` |
| `eagle/model/modeling_qwen3_kv.py` | `a3cc5ce2f78239da5d26c0dc19a5a6293a1ba330f0e43675fc857bb6a7957a46` |
| `eagle/model/kv_cache.py` | `f41d85e0210583a25b2714fea78efceed1efcf68052bae7c156b1a477b0c337b` |
| `eagle/model/configs.py` | `14ae62877cd9ab12c9076fc70fe940714b9ca980865b00ba974de340df2d1ac5` |

Transformers stays at 4.53.1: later versions remove
`transformers.utils.LossKwargs`, which the pinned EAGLE model imports.

### The head conversion is a required rebuild step

`build_images.sh` itself does not read the model, but `scripts/preflight.py`,
`smoke_test.sh --docker` and **every formal run** verify the identity of both
read-only mounts first and die if either is missing. So a fresh host must
produce them before anything else.

Step 1 — fetch the pinned target snapshot into the HF cache and point
`EAGLE3_TARGET_REPO` at the repository root:

```bash
hf download Qwen/Qwen3-8B --revision b968826d9c46dd6066d109eabc6255188de91218
```

Step 2 — **convert the draft head inside the Work image.**
`scripts/convert_head.py --out <heads.json> [--data-root <dir>]` downloads the
head repositories, audits every source file, loads `pytorch_model.bin` with
`weights_only=True`, and writes safetensors. It imports
`eagle.model.cnets.Model` and `eagle.model.configs.EConfig`, so the pinned
EAGLE checkout must be on `PYTHONPATH`, and it needs torch — which the Work
image has and the host does not. Neither `scripts/` nor the EAGLE checkout is
baked into the image, so both are bind-mounted read-only. The exact command,
with `<work-dir>` any writable host directory with ~2 GB free and
`<eagle-checkout>` the pinned `SafeAILab/EAGLE` clone:

```bash
mkdir -p <work-dir>/out <work-dir>/convert-data
docker run --rm \
  --network host \
  --user "$(id -u):$(id -g)" \
  --env HOME=/tmp \
  --env PYTHONPATH=/opt/eagle \
  --env HF_HUB_OFFLINE=0 \
  --env HTTP_PROXY --env HTTPS_PROXY --env NO_PROXY \
  --env http_proxy --env https_proxy --env no_proxy \
  --volume "$PWD/scripts:/opt/task-scripts:ro" \
  --volume "<eagle-checkout>:/opt/eagle:ro" \
  --volume "<work-dir>:/work" \
  --workdir /work \
  eagle3_spec_decode.work.eagle3_spec_decode_engine_qwen3_8b_h100_v1:local \
  python3 /opt/task-scripts/convert_head.py \
    --out /work/out/heads.json \
    --data-root /work/convert-data
```

Every flag earns its place, and leaving one out fails in a way that does not
name the cause:

| Flag | Why it is required |
| --- | --- |
| `--env HF_HUB_OFFLINE=0` | **both images force Hugging Face offline mode** (see §1), so without this the very first call dies with `OfflineModeIsEnabled: Cannot reach https://huggingface.co/api/models/…` |
| `--network host` and the six proxy variables | the download is the only networked step; on a host whose proxy listens on loopback, a bridge-network container cannot reach it |
| `--user "$(id -u):$(id -g)"` | the image runs as root by default, which would leave root-owned files in `<work-dir>` that the later freeze and `chmod` cannot touch |
| `--env HOME=/tmp` | with a non-root uid the image's `$HOME` is not writable, and `huggingface_hub` wants a writable home |
| `--volume "<work-dir>:/work"` | `--data-root` holds both the ~1.5 GB head cache and the converted output; put it on the host so a re-run does not re-download |

No GPU is requested: the conversion is `map_location="cpu"` throughout. Its
outputs are `<work-dir>/out/heads.json` plus
`<work-dir>/convert-data/heads/eagle3-head/`, and on a warm head cache it takes
about 1.5 minutes. Both candidate repositories are audited; only the Tengyunw
head is frozen and shipped.

Step 3 — freeze and fix the permissions:

```bash
EAGLE3_HEAD_SOURCE_DIR=<data-root>/heads/eagle3-head \
EAGLE3_HEADS_JSON=<conversion output>/heads.json \
  bash scripts/freeze_head_artifact.sh
chmod 0444 "$EAGLE3_HEAD_DIR"/model.safetensors
```

`freeze_head_artifact.sh` copies only `model.safetensors`, `config.json`,
`MANIFEST.json` and `SOURCE_README.md` into `EAGLE3_FROZEN_HEAD_DIR`, checks
the converted digest and revision against the conversion's `heads.json`, then
removes the write bits. The task **never loads the original pickle
checkpoint**.

**The frozen head must end up world-readable (`0444`).** The conversion writes
`0400`; the Judge's UID-dropped worker then reports
`FileNotFoundError: /opt/models/eagle3-head/model.safetensors`, which the
controller surfaces as a corrupted worker protocol frame — a badly misleading
symptom for a permission bug.

Both mount paths must be **real directories, not symlinks**, readable and
traversable by uid 1000 (`require_real_directory` /
`require_deprivileged_reachable` in `scripts/preflight.py`); an HF blob symlink
is accepted only when it points inside the repository. `EAGLE3_TARGET_REPO`
must be the HF **repository root** (containing `snapshots/<rev>/`), not an
arbitrary snapshot directory.

### Nothing here is drawn per build or captured per deployment

No part of this suite is re-drawn at
build time. The public, anchor, final and warm-up prompt slices are committed
before the build, mutually exclusive, globally de-duplicated by rendered token
sequence, and shipped inside the Judge image. The only per-run randomness is
the interactive **sampling** (per-run HMAC-SHA256 over prompt ids). What each
deployment does produce for itself is the two *model* artifacts above, and
their bytes must match the frozen digests exactly.

## 4. CPU checks before any image or GPU operation

From the task root:

```bash
python3 -B -m pytest -p no:cacheprovider tests -q
python3 -B scripts/validate_layout.py .
python3 -B scripts/generate_protected_hashes.py --check
bash -n scripts/*.sh
```

Those tests recompute the scientific resource digest, delivery/build-resource
digest, manifest byte digests and real suite cell counts. Treat any mismatch
as a hard stop. The task JSON intentionally is not a member of either resource
manifest because it is the root document that binds their digests; each
resource manifest also excludes itself to avoid a recursive hash.

**Use an interpreter that has Torch, `safetensors` and `sforge`.** Without
Torch several test files fail to import; with the CPU Torch that
`docs/HOST_PREREQUISITES.md` installs but no `safetensors`,
`tests/test_f_timing.py` stops collection with `ModuleNotFoundError: No module
named 'safetensors'`. The `tests/test_single_gpu_migration.py` cases fork
`scripts/single_gpu_runtime_preflight.py`, which imports the real coordinator,
so they also need `sforge` (which brings fastapi/starlette/uvicorn and the
`docker` module with it); installing the `docker` module alone only moves the
error to `ModuleNotFoundError: No module named 'fastapi'`. With Torch,
`safetensors` and `sforge` present the full suite passes.
`scripts/validate_layout.py` reports `status: PASS` and
`generate_protected_hashes.py --check` succeeds silently.

## 5. Build

```bash
bash scripts/build_images.sh        # takes no arguments; any argument is fatal
jq -e '.source_digest and .work.id and .judge.id' \
  ../../../outputs/eagle3-spec-decode-engine-qwen3-8b-h100-v1-seed/build/image-manifest.json
```

It first runs `validate_layout.py` and `generate_protected_hashes.py --check`,
then builds the Work image, then the Judge image with
`--build-arg WORK_IMAGE=<work>` (the Judge is `FROM` the Work image). Products:
images `eagle3_spec_decode.{work,judge}.<task_id>:local`, plus
`build/image-manifest.json` (mode 0444, recording `source_digest`, both
immutable image IDs and the SHA-256 of `protected_hashes.txt` and
`policy.json`) and `image-inspect.json`.

**`scripts/` is itself inside the image source digest**: after editing any
script — including `lib.sh` and the validators — you must rebuild, or the
runner's image/source freshness check refuses to start.

Keep the source digest and the exact Work/Judge image IDs with the run.

**After any rebuild the Judge reports `PENDING_QUALIFICATION`.** The qualified
policy (`images/judge/private_data/policy.json`) pins the image, source and
runtime-provenance identities it was qualified on — including the cooperative
pre-submit hook layer of the single-allocation profile. The owner
re-qualifies (build, submission-plane smoke, one anchor pass and one final
pass) and updates only the policy's status and identity fields. A seed score
taken before that is provenance, not a leaderboard entry.

## 6. Post-build verification

```bash
JUDGE_GPU=<index> bash scripts/smoke_test.sh --docker
```

`smoke_test.sh --docker` runs `assert_built_images_current`, takes the
whole-card flock, runs `preflight.py` (including `--image-manifest`), then
starts the Judge container with `--network none` and runs a full
`EAGLE3_ALL_ANCHOR=1` anchor pass.

**Where to read the result.** The script prints exactly one line to stdout
(`Judge GPU: physical <n>, …`) and redirects the structured verdict — every
number below — into

```text
outputs/<task directory>/smokes/C2_smoke_docker_structured.log
```

(`${OUTPUT_ROOT}/smokes/`, which defaults to
`<repo>/outputs/eagle3-spec-decode-engine-qwen3-8b-h100-v1-seed/`). The file
begins with a `>>>>> Start Structured Result` marker followed by one JSON
object; read it with, for example,
`sed -n '/^{/p' <that file> | jq '.details[] | {name, status, score}'`.
Comparing against the values below without opening that file is not possible.

Expected values:

- the `details[]` array carries **twelve** entries, all `"status": "PASSED"` —
  the **11 run-wide gates** `protected_ok`, `source_ok`, `worker_ok`, `o1_ok`,
  `o2_ok`, `tiers_ok`, `o3_o4_ok`, `o5_ok`, `wall_ok`, `memory_ok`, `idle_ok`,
  each with `"score": 1.0`, plus the `performance` entry whose `score` is the
  throughput itself. `metrics` repeats the same eleven `*_ok` values as `1.0`
  and carries `score_tokens_per_s`, `verifier_completed: true` and
  `verifier_returncode: 0`; `valid` is `true` and `pass_rate` is `1.0`;
- **the gates are the contract, not the raw number**: the seed's tokens/s
  moves with the card and with host contention (a few per cent between
  rebuilds on the same card model is expected); record your own card's value;
- a seed archive sent through the final holdout with `scripts/eval_archive.sh`
  also passes **12/12**.

**`smoke_submission_plane.sh` needs two distinct physical GPUs and therefore
cannot run on the allocation this task is qualified for.**

```bash
WORK_GPU=<a> JUDGE_GPU=<b> bash scripts/smoke_submission_plane.sh /path/to/known-valid-archive.tar.gz
```

It scores one real interactive `agent-1` round from a known-valid archive, and
fails if the Judge container did not receive its role environment
(`NVIDIA_VISIBLE_DEVICES`, the Judge role, `cuda:0` and the two deadlines).
But it requires **both** `WORK_GPU` and `JUDGE_GPU` and refuses to start when
they are equal (`Work and Judge GPU indices must differ`, plus
`assert_distinct_gpu_pair`), while §2's execution profile
`single-cuda-sm80-34g` has `count = 1`: Work and Judge share one card serially.
So on the single-GPU allocation this task is qualified for, **skip this step**
— it is a two-card lab check of the submission plane, not part of single-GPU
acceptance. `smoke_test.sh --docker` above is the acceptance test; run the
submission-plane smoke only on a host where you can spare a second idle,
non-reserved card.

On a slower card check the `wall_ok` and `memory_ok` rows first; the caps are
derived per run from the seed on the same device.

`smoke_test.sh --no-docker` needs an external interpreter
(`EAGLE3_HOST_PYTHON`) and has a physical card index hard-coded; it is only for
host-side lab reproduction.

A hand-made seed archive for `scripts/eval_archive.sh` is the content of
`images/work/workdir/spec_lab` restricted to `solutions/`, `FINAL_VERSION` and
`notes.md`, packed **without** `./` prefixes (the archive validator rejects a
leading `./` member), and the run id must start with `manual-` (or
`candidate-`, or end in `-final`) to count as a host-side evaluation.

## 7. Formal run

```bash
launch/eagle3-spec-decode/run.sh \
  --device-uuid GPU-00000000-0000-0000-0000-000000000000 \
  --calibration-epoch campaign-qualified-clock-policy-v1 \
  --judge-port 18386 \
  --model <model> --effort max --hours 2 \
  [--allocation-key allocation-sha256:<64hex>] [--lease-id ID] \
  [--run-id ID] [--auth FILE] [--config FILE] [--tmux]
```

Qualify one scheduler-allocated whole NVIDIA GPU against the profile above
before this. Record the stable measurement profile (model, compute capability,
memory, driver, image IDs, software ABI, calibration protocol, and a
scheduler-issued calibration epoch tied to a documented clock/power policy).
Record UUID and observed current clocks only in deployment attestation.
Without a calibration epoch, the run is unqualified for a cross-run tokens/s
leaderboard.

**Budget / interval / cooldown contract.**

| Quantity | Default | Frozen? |
| --- | --- | --- |
| `--model` / `--effort` | allowlisted by the launcher (`launch/eagle3-spec-decode/run.sh --help` lists the accepted names and efforts) | frozen |
| `TASK_TIMEOUT` (agent budget) | 7200 s, set by `--hours` / `--timeout`, floor 90 s | tunable — not a contract value |
| `EVAL_INTERVAL` | 1800 s, must be `< TASK_TIMEOUT` | tunable |
| submission cooldown | **equals `EVAL_INTERVAL`**, passed straight through; no independent knob | derived |
| auto-eval | always off (`--disable-auto-eval`) | **frozen** |
| `FINAL_EVAL_TIMEOUT` | 5400 s, aligned with the task JSON's `judge.eval_timeout` | keep aligned |
| `RUN_ID` | `eagle3-spec-<model>-<effort>-<N>m-<UTC timestamp>`; the runner dies immediately if the run directory exists | must be globally unique inside one version store |

The internal ladder is build cap 600 s -> worker 4800 s -> Judge outer /
standalone-final orchestration 5400 s.

**Ports — read this before scripting concurrency.** `--judge-port` must be
1024-65535 and must be neither 3128 nor 3130, and the runner checks with
`ss -ltn` in preflight and dies if it is occupied. The default is 18386
(the runner's `JUDGE_PORT:-18386`). **Always pass `--judge-port`
explicitly** with the port the scheduler reserved for this run; every
concurrent run on the same host needs its own free port.

**Version store.** `SFORGE_VERSION_STORE_ROOT` must explicitly point at a store
carrying a `.sforge-version-store.json` marker that registers this task id, or
preflight reports "version store … does not register";
`SFORGE_AGENT_RUNTIME_DIR` must point at an agent runtime built by
`launch/build-agent-runtime.sh`, and `CODEX_AUTH_JSON_PATH` at a regular
non-symlink file, mode 0600, owned by the launching user. Reusing a `RUN_ID`
inside one store collides when the final version is persisted after the
deadline and wastes the whole agent budget.

**Proxy.** No host proxy is involved: the Work container's restricted sidecar
enforces the model-auth allowlist itself and dials those three hosts directly
from the default Docker bridge, so the host only needs ordinary outbound TCP to
them. Before Work exists the launcher proves from inside the internal network
that the sidecar answers, the Judge relay on 3130 reaches the coordinator,
`https://example.com` is not reachable directly, `CONNECT chatgpt.com:443`
answers 200 and `CONNECT github.com:443` answers 403; the sidecar's SHA-256 and
the network mode (`docker_internal_restricted_connect_proxy`,
`direct_public_egress: false`, `connect_allowlist`) are recorded in
`experiment-manifest.json`. `env -u` the shell's own `http_proxy` /
`https_proxy` / `HTTP_PROXY` / `HTTPS_PROXY` before launching.

## 8. Run artifacts and success criteria

```text
outputs/<task dir>/runs/<RUN_ID>/
  experiment-manifest.json  image-build-manifest.json  oracle-policy.json
  judge-server.log  sforge-run.log  runtime-provenance.json
  <task_id>/
    resource_lease.json
    formal-precheck.json  formal-postcheck.json  completion.json   (0444)
    final_archive.tar.gz  final_result.json  run_history.json
    submissions/agent-N/report.json
outputs/<task dir>/runs/<RUN_ID>-final/<task_id>/submissions/manual-1/report.json
```

Success means all of: `resource_lease.json` in state **CLOSED**;
`formal-postcheck.json` PASS; `completion.json` with `status = passed`; and the
standalone final's `manual-1/report.json` with `valid = true`. The lease's
`attestations` record, in order, `after_work_pause_before_judge` ->
`after_judge_destroy_before_work_resume` (one pair per interactive round) ->
`after_final_work_destroy` -> `before_final_judge` ->
`after_final_judge_destroy`.

When the candidate is invalid on the hidden final, the lease still closes
normally but the postcheck reports "independent Judge did not accept the frozen
submission" **by design** and `completion.json` is not written. That is a
candidate-level result, not a protocol failure.

**On failure, look at these first**, in order: `sforge-run.log` ->
`judge-server.log` -> `runs/<RUN_ID>-final/eval.log` -> the two
`formal-*check.json` files (which check failed) -> the `metrics` gate vector in
`submissions/*/report.json` (which of the eleven gates is 0).

Operational traps: the "claimed by a running container" check
is conservative — it reads only the `NVIDIA_VISIBLE_DEVICES` /
`EAGLE3_*_GPU_UUID` environment of running containers, not their actual memory
use, so one stale foreign pinned container with zero memory in use is enough to
fail preflight on that card; when serialising runs, wait for the previous run's
containers to disappear from `docker ps` and then about 30 s more. Single-GPU
and dual-GPU tasks also use different GPU-lock namespaces and cannot see each
other's locks, so never schedule one of each on the same card.

## 9. What to re-measure on another GPU

**Same model, different card: re-measure nothing.** Every card-dependent
quantity is measured inside the evaluation itself — the wall and memory caps
and the oracle tier floors are all ratios the Judge derives by first running
the trusted seed reference **on its own card inside the same evaluation**, and
`policy.json` contains no absolute per-cell seconds or bytes (`preflight.py`
actively rejects `wall_caps_s` / `memory_caps_bytes` if they reappear). Just
change the device UUID.

Different model or driver, re-confirm:

| Quantity | Card-dependent | Action |
| --- | --- | --- |
| the capability envelope | no | just confirm the card satisfies it |
| wall / memory caps, tier floors | yes, but **automatically** | none |
| `TORCH_CUDA_ARCH_LIST` | yes | pass the build arg and rebuild if the target architecture is outside the default list |
| `measurement_profile` comparability key | yes | a new card's raw tok/s must **not** be compared with the old card's; compare only ratios against the seed |
| the timeout ladder | no (they are effort budgets) | revisit only if the target card is markedly slower |
| the images | no, unless `scripts/` changed | a card swap alone needs no rebuild |
| the policy status | — | **any rebuild sets it to `PENDING_QUALIFICATION`** until the owner re-qualifies |
| the Work and Judge NVML memory peaks (§2) | yes | re-measure |

`deployment_attestation` (physical UUID plus observed clocks, for audit only)
and `measurement_profile` (the comparability key) are two **separate**
identities; do not mix them.

Raw tokens/s may be compared only when both runs are qualified and their
recorded comparison identity is identical. That identity is the ordered pair
of the evaluation-contract SHA-256 and the measurement-profile comparability
key; the contract digest is deliberately not folded into the pure machine
profile hash. Merely satisfying the broad `single-cuda-sm80-34g` capability
envelope does not make two runs comparable; the wall and memory caps are
multiples of the seed measured on the same device inside the same evaluation.
