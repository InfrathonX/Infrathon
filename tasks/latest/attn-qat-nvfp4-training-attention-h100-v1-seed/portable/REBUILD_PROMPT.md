# Rebuild Attn-QAT NVFP4 Training Attention v1 on a qualified whole-GPU host

**Evaluation policy:** `attn-qat-shared-final-v3` (`full-final-shared-v2`).
Interactive `sforge-submit` and the sealed deadline re-evaluation both use the
same complete frozen `final-holdout-v3` suite: 15 hard-gate fixtures and 287
scored cases in two domains, video and LLM (209 synthetic at 3 trials and 1 warm-up, 78 real
layers at one step). Candidate round identities and override
permissions remain separate. The final is a fresh same-suite re-evaluation of
sealed code, not an unseen holdout.

Use this task bundle to reproduce the Work/Judge images and qualify a new
single-CUDA deployment. You may use Codex, Claude, or another build agent, but
do not let the build agent redesign the benchmark. What the task *is* — the
metric, the gates, the suites — is specified in `images/work/AGENT_TASK.md`;
this file is only how to build, verify and run it.

Preserve these scientific identities byte-for-byte:

- the TaskCore and its resource manifest;
- the 15 disclosed edge fixtures (the e7 Wan gradient capture bound by its
  recipe, `fixtures/recipe/RECIPE.json`, and captured per deployment by
  `fixtures/recipe/build_fixture.sh`), the 29 public cases and the shared
  interactive/final 287-case root-only frozen suite drawn by
  `scripts/generate_final_manifest.py`, with its 78 real layers captured from
  `fixtures/recipe/real_layers/REAL_LAYERS.json` (Wan) and `REAL_LAYERS_V2.json`
  (CogVideoX-2B, Qwen3-14B, Qwen3.5-4B);
- the fp32 golden, the per-tensor relative-L2 gates and per-fixture rows, the
  saved-bundle bound, the timing protocol, the effective-work formula and the
  two-domain, per-family, per-tier geometric-mean metric;
- the agent-facing optimization contract.

## 1. Host prerequisites

`docs/HOST_PREREQUISITES.md` lists everything that is not task-specific:
Docker >= 24 with the **NVIDIA runtime as default**, your user in the `docker`
group, `uv` with Python 3.10-3.13, `jq flock rsync patch git sha256sum
nvidia-smi`, CPU `torch` + `numpy` in the repo venv, >= 150 GB free on Docker's
data root, and outbound access to Docker Hub, Hugging Face and GitHub. Check
them first.

Task-specific extras:

| Extra | Why |
| --- | --- |
| ~18 GB more free disk and a Hugging Face cache (`HF_HOME`) | the e7 capture downloads the pinned `Wan-AI/Wan2.1-T2V-1.3B` snapshot (~16 GB on disk) once |
| `hf`, the Hugging Face CLI | `fixtures/recipe/build_fixture.sh` calls `require_command hf` and dies without it |
| one whole idle GPU with **~12 GiB free**, for about three minutes | `fixtures/recipe/build_fixture.sh` runs a real Wan2.1 QAD stage-1 training step on `FIXTURE_GPU` |
| 48 GiB of host RAM per container | Work and Judge each take `cpu_limit: 8`, `mem_limit: 48g` |
| `ss` (iproute2) | the launcher refuses to start when the coordinator port is already listening |

The task JSON sets `internet: false`, so the Work container has no general
egress. Only image builds and the one capture need the network. The launcher
enforces this itself: Work runs on a per-run `--internal` Docker network whose
only peer is a task-owned CONNECT sidecar (`scripts/restricted_proxy.py`, run
from the Work image) that allows TLS tunnels solely to `api.openai.com`,
`auth.openai.com` and `chatgpt.com` and relays the Judge port;
`scripts/sforge_isolated_entry.py` places Work on that network and disables
Codex web search. No host iptables and no host proxy are needed; a preflight
from inside the network proves direct egress fails and the allowlist answers
200 / 403 before the agent budget starts.

## 2. GPU envelope

The task JSON's `portable.execution_profiles` carries exactly one profile,
`single-cuda-sm89-40g`: `count = 1`, `min_compute_capability = "8.9"`,
`min_memory_bytes = 42949672960` (40 GiB), features `cuda / bf16 / fp8-e4m3 /
fp32 / triton`. MIG is rejected — the launcher only accepts a canonical
whole-GPU UUID matching `^GPU-[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}$`.

Who enforces it:

- `runtime/attn_qat_runtime.py` carries `MIN_COMPUTE_CAPABILITY = (8, 9)` and
  `MIN_DEVICE_MEMORY_BYTES = 16 * 1024**3`, asserted in `hardware_inventory`;
- `images/judge/verify_attn_qat.py` carries the same pair, and
  `tests/test_capability_envelope.py` asserts the two copies are equal;
- `scripts/single_gpu_runtime_preflight.py` checks the harness contract and the
  portable digests before anything touches Docker (exit 78 on failure), and
  `scripts/lib.sh` requires `GPU_MIN_FREE_MIB` free (default 14336) on a card
  with no compute processes.

The GPU *name* is not a gate: name and UUID are recorded only in
`deployment_attestation`. `cc >= 8.9` is a contract requirement, not a
preference — the e4m3 block scale must come from a native e4m3 conversion, so
Ampere cannot run this task at all. The `h100` segment in the task id and the
directory name is a stable identifier, not a profile gate.

The Judge's largest case is computed at about 26 GiB resident before the
candidate's own scratch, and the public suite at about 11.3 GiB. The declared
floor is 40 GiB: a floor is a contract, and lowering it would admit cards the
task was never qualified on. Measure the NVML peaks during qualification.

## 3. External inputs

| What | Source and pin | Size | SHA if pinned | Env var / path | Availability |
| --- | --- | --- | --- | --- | --- |
| Base image for both images | `pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel` (torch 2.8.0 / Triton 3.4.0) | ~9 GB | by tag | pulled by `scripts/build_images.sh` | public |
| e7 model snapshot | `Wan-AI/Wan2.1-T2V-1.3B` @ `37ec512624d61f7aa208f7ea8140a131f93afc9a` | ~17.6 GB | pinned by revision **and per-file SHA-256** in `fixtures/recipe/RECIPE.json` | host HF cache (`HF_HOME`), fetched by `build_fixture.sh` | public, ungated |
| e7 upstream pipeline | `Wan-Video/Wan2.1` @ `9737cba9c1c3c4d04b33fcad41c111989865d315` (modules `wan.configs`, `wan.modules.model`, `wan.modules.t5`, `wan.utils.fm_solvers_unipc`) | small | pinned by commit in `RECIPE.json` | cloned into the throwaway builder image at capture time | public |
| e7 capture | produced here, per deployment | ~120 MB | **none by design** | `fixtures/e7_wan13b_grad_b29_t1_s8192.pt` + its capture manifest; `FIXTURE_PATH` + `FIXTURE_MANIFEST_PATH` to supply an existing pair | per-deployment capture |
| Final holdout instance | in-tree at `judge/private/final_cases.json` | < 64 KB | `a9ba63f04dd1c1f1ea5d389231bfc0127a7a3120655860151c0e473d043cf4dd`, recorded in `final-suite-manifest.json` under `frozen_instance` and again per build in the build manifest | `FINAL_CASES_PATH`, default `judge/private/final_cases.json` | in-tree, Judge-only path |
| Upstream FastVideo checkout | not needed | — | — | — | `fastvideo` is a forbidden package inside both images |

There is no owner-provided input: everything above is either public and pinned
or produced by this deployment.

### The e7 capture is per deployment; the holdout instance is frozen

**The e7 edge input** is a real Wan2.1-T2V-1.3B QAD stage-1 training step:
`q, k, v, o, do` for the first 8192 tokens of block 29's self-attention, as
`[1, 8192, 12, 128]` bf16. It is **recipe-bound, not byte-bound**.
`fixtures/recipe/RECIPE.json` pins the model snapshot (with per-file digests),
the upstream commit and modules, the prompt and negative prompt, the geometry
(81 frames at 832x480, latent 21x60x104, `seq_len` 32760), how the latent is
produced, the training step, the hook point (`do` comes from a tensor hook on
`o` during the first graph-carrying forward of non-reentrant checkpointing) and
**per-tensor sanity bands**. The runtime enforces those bands on every load, so
a wrong capture fails closed even though the bytes differ across GPUs. The
bands are centred on an H100 reference capture and span x/÷2 on RMS and x/÷4
on the absolute maximum; `do` sits around 1e-8 because it is the gradient of a
mean-reduced MSE loss, a loss-normalisation constant rather than an operator
property. `RECIPE.json`'s `reference_capture` block (sha256 `58a7773f…`) is
**provenance only**: no deployment is required to reproduce those bytes, and
the capture's own SHA-256 must never be pinned anywhere in the task.

**The final holdout instance** is **frozen, not drawn per build**. The
qualification draw is committed inside the task at
`judge/private/final_cases.json` and is what `build_images.sh` reads by
default; `final-suite-manifest.json` records its SHA-256 under
`frozen_instance`. It is Judge-only: it is staged mode 0600 into a root-only
Judge build context and must never be copied, mounted or described inside the
Work image (`scripts/validate_layout.py` proves no Work build context admits
it, by path and by SHA sweep). `portable/final-suite-manifest.json` freezes the
the case grid, the weights and the drawing strategy, and
pins which slot and which seed the frozen instance uses. Rebuilding does not
replace the holdout: every host that builds from this tree gets the same
`final_cases.sha256`, so scores stay comparable.

**Expected digest:** `final_cases.sha256` in the build manifest must equal
`a9ba63f04dd1c1f1ea5d389231bfc0127a7a3120655860151c0e473d043cf4dd`.

`scripts/generate_final_manifest.py --redraw --seed <int>` draws a
reproducible *new* instance from a seeded `random.Random`; that is a **new suite
identity** whose `frozen_instance` record must be updated, and no earlier score
is comparable to it. `FINAL_CASES_PATH` overrides the default when you
deliberately want to build against another instance.

The Work image starts from only `images/work` plus the pinned `runtime/` build
context; the Judge image is built independently from `images/judge`, the same
runtime context and the private `final_cases` context.

## 4. CPU checks before any image or GPU operation

From the task root, with the **repository venv interpreter named explicitly** —
a bare `python3` is the system interpreter, which normally has neither the
`docker` SDK nor the repo's `sforge` package and turns real coverage into
environmental failures:

```bash
../../../.venv/bin/python -B -m pytest -p no:cacheprovider tests -q
../../../.venv/bin/python -B scripts/validate_layout.py .
```

(`uv run --frozen python …` from the repository root is the equivalent.)

Those tests recompute the scientific resource digest, delivery/build-resource
digest, manifest byte digests and real suite case counts. Treat any mismatch
as a hard stop. The task JSON intentionally is not a member of either resource
manifest because it is the root document that binds their digests; each
resource manifest also excludes itself to avoid a recursive hash.

With CPU `torch` in the venv (`docs/HOST_PREREQUISITES.md` installs it) every
test passes apart from a few skips. Without `torch` the Torch-dependent cases
are skipped or never collected; treat that as a warning, not a pass, and
install CPU `torch`. The venv also supplies the `docker` SDK that the
`tests/test_single_gpu_migration.py` handoff cases fork into; without it they
fail on `No module named 'docker'`, which is a missing dependency rather than a
task defect.

`validate_layout.py` passes on a checkout that carries no capture at all: an
absent e7 is not an error, because `build_images.sh` takes one before it
builds. A capture that *is* present must be accompanied by the manifest binding
it to the shipped `RECIPE.json`, and an explicitly supplied pair is asserted to
exist. The validator also rejects stray `__pycache__` inside the task tree, so
run tests with `PYTHONDONTWRITEBYTECODE=1` or
`PYTHONPYCACHEPREFIX=<scratch>`.

## 5. Capture e7, then build

The capture is recipe-bound, not byte-bound: its bytes are expected to differ
from any other deployment's, and the runtime enforces the recipe's per-tensor
sanity bands instead of a digest. `build_images.sh` takes the capture itself
when `fixtures/` is empty; pass `FIXTURE_PATH` and `FIXTURE_MANIFEST_PATH`
instead to reuse one this deployment already holds.

`FIXTURE_GPU` must name the whole idle card **allocated to you** — never a
hardcoded `nvidia-smi --id=<fixed index>`, which on a shared host is somebody
else's card.
Read the UUID of your own index with
`nvidia-smi --query-gpu=uuid --format=csv,noheader --id=<your index>`, or pass
that index directly (the script accepts either form).

```bash
export FIXTURE_GPU=GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx   # your allocated card
bash fixtures/recipe/build_fixture.sh    # optional; build_images.sh does it too
[FINAL_CASES_PATH=<an existing holdout>] bash scripts/build_images.sh
bash scripts/make_oracle_archive.sh
jq -e '.source_digest and .work.id and .judge.id and .final_cases.sha256
       and .edge_fixture.sha256 and .edge_fixture.recipe_sha256' \
  ../../../outputs/attn-qat-nvfp4-training-attention-h100-v1-seed/build/image-manifest.json
```

A supplied pair is validated where it lies before anything is copied into the
task tree, so a capture from another recipe is rejected without disturbing the
tree or the recorded build manifest; run the same check by hand with
`../../../.venv/bin/python scripts/validate_layout.py . --fixture <capture>
--fixture-manifest <manifest>`.

**`build_fixture.sh` is the one script that does *not* re-exec itself through
`sg docker`.** Everything under `scripts/` does (`ensure_docker_group` in
`scripts/lib.sh`); this one calls `docker build` directly, so on a host where
your login shell is not yet in the `docker` group it dies on the socket. Wrap it
yourself and keep the umask inside the wrapper, so the staged capture and its
manifest stay readable by the build:

```bash
sg docker -c 'umask 022; FIXTURE_GPU='"$FIXTURE_GPU"' bash fixtures/recipe/build_fixture.sh'
```

`build_fixture.sh` needs egress for the pinned model snapshot
(`hf download Wan-AI/Wan2.1-T2V-1.3B --revision <sha>`, honouring `HF_HOME`, a
no-op when the revision is already cached) and for the pinned upstream
checkout, and about 12 GiB of device memory and three minutes on the
accelerator named by `FIXTURE_GPU` (it also accepts an index). It forwards
`HTTP_PROXY`/`HTTPS_PROXY` to the builder image build and to `hf download`.
Record the capture SHA-256 and the recipe SHA-256 it prints with the run: the
capture digest identifies this deployment's bytes, the recipe digest is the
scientific identity that must match every other deployment.

`build_images.sh` validates the holdout's mode (0600), owner, `suite_id`
(`final-holdout-v1`) and case count (exactly 4), installs it into a separate
build context, and builds the two images independently. It computes the image
source digest before and after and dies if either the source or the holdout
changed mid-build. Products:

- Work `attn_qat_training_attention.work.attn_qat_nvfp4_training_attention_h100_seed_v1:local`
- Judge `attn_qat_training_attention.judge.attn_qat_nvfp4_training_attention_h100_seed_v1:local`
- `outputs/attn-qat-nvfp4-training-attention-h100-v1-seed/build/image-manifest.json`
  (`schema_version: 2`, with `source_digest`, both image IDs, the holdout's
  sha256/suite_id/case_count, and this deployment's capture and recipe digests).

`run_fixed_budget.sh` recomputes those and refuses a stale source or image
identity. Keep the source digest, the exact image IDs, the holdout digest and
the capture/recipe digests with the run; do not substitute a pre-existing tag
without verifying its recorded image ID.

## 6. Post-build verification

```bash
bash scripts/smoke_test.sh                               # layout + CPU tests, no GPU
PHYSICAL_GPU=<index> bash scripts/smoke_test.sh --oracle # needs the GPU
```

**The `--oracle` line is what qualifies the deployment.** A build that stops at
`build_images.sh` plus the `jq` gate has proved only that the images exist.
`PHYSICAL_GPU` is the index of the card allocated to you (`nvidia-smi -L` lists
them); `smoke_test.sh` defaults to card **0**, so pass it explicitly on any
multi-GPU host or you will qualify — or disturb — the wrong card.

`--oracle` takes the card's UUID flock, asserts the card is idle, runs the seed
through the participant-visible `public_runner.py --suite full` inside the Work
image, then packs `oracle/` with `make_oracle_archive.sh` and sends it through
`eval_archive.sh` for a full trusted-Judge evaluation. **On a new card the
oracle must pass every gate before an agent touches it.**

**Cost:** one whole idle GPU with ~30 GiB free for roughly **25 minutes** (the
full public suite plus a 302-case Judge pass dominated by the fp32 golden). No
network — the capture and images already exist by this point.

Expected result:

- every gate PASSED, all 15 edge fixtures then the 287 scored cases;
- the Judge's `score_tflops` on the frozen instance `a9ba63f0…`; repeat runs of
  one build on one card agree to about 0.1 %;
- the full 302-case wall time on the order of 1200 s, dominated by the fp32
  golden.

The public-suite primary is reported separately and is not the score.
Absolute TFLOPS move with the card; the gate limits do not. Record your own
card's oracle score, worst `error / limit` and positive-control band before
any agent run.

## 7. Formal run

```bash
launch/attn-qat-train/run.sh \
  --device-uuid GPU-00000000-0000-0000-0000-000000000000 \
  --calibration-epoch campaign-qualified-clock-policy-v1 \
  --judge-port 18096 \
  --model <model> --effort max --hours 2 \
  [--auth <codex auth.json>] [--run-id <unique id>] [--tmux]
```

Qualify one scheduler-allocated whole NVIDIA GPU against the profile above
before this. Record the stable measurement profile (model, compute capability,
memory, driver, image IDs, software ABI, calibration protocol, and a
scheduler-issued calibration epoch tied to a documented clock/power policy).
Record UUID and observed current clocks only in deployment attestation.
Without a calibration epoch, the run is unqualified for a cross-run
effective-TFLOPS leaderboard.

`--model` takes a model ID (see `launch/attn-qat-train/run.sh --help`).
`--hours N` is equivalent to `--timeout $((N*3600))`; `--timeout` wins. Other
flags: `--allocation-key`, `--lease-id`, `--config`, and `--legacy-dual-gpu`
for the dual-GPU `WORK_GPU`/`JUDGE_GPU` path (mutually exclusive with the
single-GPU flags). Single mode **rejects** `WORK_GPU` / `JUDGE_GPU` and never
falls back.

**Budget / interval / cooldown contract.**

| Quantity | Default | Frozen? |
| --- | --- | --- |
| `TASK_TIMEOUT` (agent budget) | 7200 s | tunable; must be >= 90 s, a multiple of 60, and strictly greater than `EVAL_INTERVAL` |
| `EVAL_INTERVAL` | 300 s | tunable; must be a multiple of 60 |
| submission cooldown | **equals `EVAL_INTERVAL`** — not an independent knob | follows `EVAL_INTERVAL` |
| auto eval | permanently off (`--disable-auto-eval`) | **frozen** |
| `internet` | false | **frozen** (task JSON) |
| `JUDGE_PORT` | 18096 | tunable; 1024-65535, not 3128 or 3131 (sidecar ports), and actually free — the runner checks the listening sockets in preflight |

Work and Judge each run with 8 CPUs and 48 GiB of memory.

**Ports.** The scheduler must allocate a different free `--judge-port` for every
concurrent run on the same host; the runner binds it in preflight and reports
"Address already in use" otherwise.

**Version store.** `SFORGE_VERSION_STORE_ROOT` must point at a store carrying a
`.sforge-version-store.json` marker that registers this task id;
`SFORGE_AGENT_RUNTIME_DIR` at an agent runtime built by
`launch/build-agent-runtime.sh`; `CODEX_AUTH_JSON_PATH` at a regular
non-symlink file, mode 0600, owned by the launching user. `RUN_ID` must be
globally unique inside one store — reusing one fails only after the deadline,
when the final version is persisted, with `VersionStoreCollision`, and wastes
the whole agent budget.

**Proxy.** A container cannot reach a proxy bound to host loopback, and the
launcher refuses to hand one to a container. Republish the host's egress proxy
on the Docker bridge and set `SFORGE_HTTP_PROXY` / `SFORGE_HTTPS_PROXY` /
`SFORGE_NO_PROXY` to that bridge address in `launch/mlsysforge.env`, then
`env -u` the shell's own `http_proxy` / `HTTPS_PROXY` / … before launching.
Leave all three unset on a host with direct internet access. For this task the
proxy matters only for image builds and the capture; the run itself is offline
and `run_fixed_budget.sh` overrides `SFORGE_HTTP(S)_PROXY` with its own sidecar
address before Work starts (the experiment manifest records
`network.mode: docker_internal_restricted_connect_proxy`,
`direct_public_egress: false` and the sidecar's `restricted_proxy_sha256`).

Single mode exports these automatically:
`SFORGE_SINGLE_GPU_HANDOFF=1`, `SFORGE_ACCELERATOR_DEVICE_UUID`,
`SFORGE_ACCELERATOR_ALLOCATION_KEY` (a canonical device-set digest; when
supplied it must match byte for byte), `SFORGE_ACCELERATOR_OBSERVER=nvml`,
`SFORGE_TRUSTED_LEASE_ID` (a never-reused label-safe id is generated when
omitted), and the agent-side
`SFORGE_PRE_SUBMIT_HOOK=/app/workdir/attn_qat_lab/.sforge/prepare_submit.sh`.

## 8. Run artifacts and success criteria

```text
outputs/attn-qat-nvfp4-training-attention-h100-v1-seed/
├── build/image-manifest.json
└── runs/<RUN_ID>/
    ├── completion.json
    └── <task_id>/
        ├── experiment-manifest.json
        ├── resource_lease.json
        ├── formal-postcheck.json
        ├── final-<N>m-private-judge-artifacts.tar.gz   (0600)
        └── submissions/agent-*/report.json
outputs/.../runs/<final_eval_id>/<task_id>/submissions/manual-1/report.json
```

Success means all of: `resource_lease.json` in state **CLOSED**;
`formal-postcheck.json` PASS; the standalone final's `manual-1/report.json`
with `valid = true`; and `completion.json` present with `status = passed`.

When the candidate is invalid on the hidden final, the lease still closes
normally but the postcheck reports "independent Judge did not accept the frozen
submission" **by design** and `completion.json` is not written. That is a
candidate-level result, not a protocol failure.

**On failure, look at these first**, in order: the launcher's `run.log` ->
`runs/<RUN_ID>/<task_id>/formal-postcheck.json` -> the standalone final's
`submissions/manual-1/report.json` -> `attn_qat_v1_results.json` inside
`final-<N>m-private-judge-artifacts.tar.gz`. Two operational traps:
serialising runs on one card requires waiting for the previous run's containers
to disappear from `docker ps` and then about 30 s more (otherwise "claimed by a
running container"), and a duplicate `RUN_ID` fails only at the very end.

## 9. What to re-measure on another GPU

Re-measure and re-record: the absolute TFLOPS of the seed, the positive control
and any candidate; the expected bands of `scripts/host_smoke.py`; the margin by
which the oracle archive clears its gates; the headroom left by the timeout
ladder; the Work and Judge NVML memory peaks; and the whole
measurement-profile comparability key. Re-capture e7 (or deliberately reuse a
capture) and decide explicitly whether to re-draw the final holdout.

Do **not** re-calibrate: any accuracy limit (the `GATE_CALIBRATION` constants
are a numerical property, not a card property), the case counts (15/15/15), the
the case grid, the timeout ladder (a qualification ceiling),
e7's recipe and sanity bands, or the suite `manifest_sha256` values.

Changing card model, driver, images, software ABI or calibration epoch each
produces a **new** measurement-profile comparability key even when the portable
TaskCore digest is unchanged; raw TFLOPS across that boundary are not
comparable. Moving images with `docker save` / `docker load` keeps the contract
identical, but identical image bytes do not mean the new card is qualified.

On an sm_89-class card the contract is satisfied (cc >= 8.9), but the seed sets
`warp_specialize=True`, which only compiles on sm_90+; turn it off or the seed
itself fails to build. That is seed tuning, not contract.

Raw effective TFLOPS may be compared only when both runs are qualified and
their recorded comparison identity is identical. That identity is the ordered
pair of the evaluation-contract SHA-256 and the measurement-profile
comparability key; the contract digest is deliberately not folded into the
pure machine profile hash. Merely satisfying the broad `single-cuda-sm89-40g`
capability envelope does not make two runs comparable.
