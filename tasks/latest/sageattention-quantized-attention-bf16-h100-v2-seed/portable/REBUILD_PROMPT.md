# Rebuild Quantized Attention v2 on a qualified whole-GPU host

This document is the portable contract of the task: what has to be rebuilt,
and how, to run it on another host or platform. With this file,
`docs/HOST_PREREQUISITES.md` (repository root) and this task directory you can
rebuild the images, qualify the deployment and run the task on a fresh host
with an admissible GPU. The task's `README.md` describes what the task is and
how it is scored. You may use a coding agent to carry out the rebuild, but do
not let it redesign the benchmark.

## Submission evaluation policy

`full-final-shared-v3` (`sageattention-v3-shared-final-v1`): every admitted
`auto-*`, `agent-*`, `manual-*`, and `candidate-*` round executes the complete
frozen `final-holdout-v3` suite: 8 edge gates and 774 scored cases (635
synthetic and 139 captured real layers, one warm-up and three measured trials
each). Timing,
thresholds, aggregation, and private input isolation are identical across these
rounds. Interactive rounds retain their original identities and cannot override
`FINAL_VERSION`; host replay retains its candidate override. Local public tests
use the disclosed public suite (19 public cases, 8 edge gates and 2 real layers
of its own); they never run the private suite.

Before formal precheck, the launcher stages the exact run-level
`experiment-manifest.json` bytes at `<task_run_root>/experiment-manifest.json`,
bound to the SHA-256 it recorded when creating the run manifest. Staging rejects
a changed source, symlinks, and an existing destination, and creates the task
copy with mode `0600`. The precheck reads this task-local copy and never falls
back to a parent directory.

## Immutable identities

Preserve these scientific identities byte-for-byte:

- the TaskCore and its resource manifest;
- the 8 disclosed edge fixtures (the e8 Wan2.1 capture is bound by
  `fixtures/recipe/RECIPE.json`, never by the SHA-256 of one host's bytes), the
  19 public cases and the 774-case final suite frozen as the Judge-only asset
  `judge/final_cases.json` (`final-holdout-v3`, read through
  `judge/private/final_cases.json`), whose digest
  `portable/final-suite-manifest.json` records;
- the 139 scored real layers of ten recipes in `fixtures/recipe/real_layers/`
  (CogVideoX-2B steps 0/10/40, Wan2.1 1.3B and 14B, SD3.5 Medium, Llama-3.1-8B
  and Llama-3-8B-262k prefills of 16K to 262K tokens), with the row
  relative-L2 p99.9 limit at `max(2.5 * reference, 0.02)`;
- the SageAttention-1 fake-quantization reference and its accuracy gates, the
  timing protocol, the effective-TFLOPS formula and the two-family
  geometric-mean metric;
- the agent-facing optimization contract.

## 1. Host prerequisites

`docs/HOST_PREREQUISITES.md` lists everything that is not task-specific:
Docker with the **NVIDIA runtime as default**, your user in the `docker`
group, `uv`, `jq flock rsync patch git sha256sum nvidia-smi`, CPU `torch` in
the repository venv, disk, and network for the image builds. Check it first.

Task-specific extras:

| Need | Why |
| --- | --- |
| About 62 GB under `INFRATHON_ASSETS_ROOT` | the reference activation captures (`assets/fetch.sh`, §3); the 139 scored real layers (about 58 GB) are mounted read-only into the Judge at run time, never copied into an image |
| `ss` (iproute2) | the launcher refuses to start when the coordinator port is already listening |
| Only to re-capture (optional): `hf`, a Hugging Face cache and one idle GPU | `fixtures/recipe/build_fixture.sh` and `fixtures/recipe/real_layers/build_real_layers.sh` download the pinned public snapshots (Wan2.1-T2V-1.3B ~17 GB, CogVideoX-2B ~13 GB, Llama-3.1-8B ~15 GB, the WikiText-2 test parquet) and run them to the capture point |

Network policy follows the task JSON's `internet` field (`true` for this
task). The launcher enforces it without SForge's host-iptables mode: Work lives
on an `--internal` Docker network whose only other member is a CONNECT-proxy
sidecar (`scripts/restricted_proxy.py`), which relays the Judge port on 3130
and admits CONNECT to `api.openai.com`, `auth.openai.com` and
`chatgpt.com:443`; with `internet: true` it also runs `--allow-public-egress`
(CONNECT to any globally routable host on ports 80/443, internal addresses
refused) and Codex runs with live web search. With `internet: false` everything
outside the allowlist gets 403 and web search is disabled. Work has no other
route, and Judge containers run `--network none`.

## 2. GPU envelope

The task JSON's `portable.execution_profiles` carries exactly one profile,
`single-cuda-sm80-64g`: `count = 1`, `min_compute_capability = "8.0"`,
`min_memory_bytes = 68719476736` (64 GiB), features `cuda / bf16 / int8 / fp32
/ triton`. MIG is not qualified; the card must be a whole GPU.

Who enforces it:

- `runtime/quant_attention_runtime.py` carries `MIN_COMPUTE_CAPABILITY` and
  `MIN_DEVICE_MEMORY_BYTES` and refuses to run below them;
- the Judge verifier `images/judge/verify_quant_attention.py` re-checks the
  same pair and reports it as the scored `hardware_ok` gate;
- the launcher preflight in `scripts/lib.sh` requires `GPU_MIN_FREE_MIB` free
  (default 14336) on an otherwise idle card before it touches Docker.

The Judge's device-memory peak over the full final suite is about 44 GiB (the
262K-token prefill cases, whose three trial inputs stay allocated for the
fresh-pointer check); the declared 64 GiB leaves room for candidate scratch.

FP8 e4m3 is an *optional* implementation surface that needs cc >= 8.9. The
admission gate only asks for cc >= 8.0 (INT8 `tl.dot` + BF16), so an Ampere
card runs the task with one design option fewer, not with a different contract.

## 3. External inputs

| What | Source and pin | Size | Where the build reads it | Visibility |
| --- | --- | --- | --- | --- |
| Base image for both images | `pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel` (torch 2.8.0 / CUDA 12.9 / Triton 3.4.0, asserted at build time) | ~9 GB | pulled by `scripts/build_images.sh` | public |
| Final holdout instance | dataset asset `judge/final_cases.json` (a development checkout keeps it at `judge/private/final_cases.json`); SHA-256 in `assets/manifest.json` and `portable/final-suite-manifest.json` (`frozen_instance`) | < 10 KB | `FINAL_CASES_PATH`; `scripts/lib.sh` points it at the asset root when `INFRATHON_ASSETS_ROOT` is set or the repository copy is absent | Judge-only |
| e8 capture | dataset asset `public/e8_wan13b_p0_b15_s00_cond_8192.pt` + `.manifest.json` (recipe `fixtures/recipe/RECIPE.json`) | ~75 MB | asset root, or `fixtures/` when a capture made on this host is there, or `FIXTURE_PATH` / `FIXTURE_MANIFEST_PATH` | public (both images) |
| Real-layer captures, Judge | dataset assets `judge/real_layers/cogvideox2b-p0-s00/` (30 layers, 12.3 GB) and `judge/real_layers/llama31-8b-wikitext2-16k/` (32 layers, 6.4 GB), each with `capture.manifest.json` | 18.7 GB | asset root, or `fixtures/real_layers/` when captured on this host, or `REAL_LAYERS_DIR` | Judge-only |
| Real-layer captures, Work | dataset asset `public/public_layers/` (`public-video/`, `public-lm/`, captured by `fixtures/recipe/public_layers/build_public_layers.sh`; not copies of the Judge's) | 0.2 GB | asset root (same rules) | public (agent-visible) |
| Upstream SageAttention checkout | not needed | — | — | asserted *absent* from both images |

`assets/fetch.sh` downloads every dataset asset into
`${INFRATHON_ASSETS_ROOT}/<task directory>/` (default
`~/.cache/infrathon/assets`) and verifies it against
`assets/manifest.json`. No model or dataset download is needed to build or run
the task.

### Captures are recipe-bound; the holdout instance is frozen

**The captures** (e8 and the 139 scored real layers) are real activations of public
models at a pinned capture point. Each is bound by its recipe, which pins the
model snapshot and its file hashes, the upstream commit, the builder dependency
set, the exact capture point (prompt, seed, denoising step, layers, text
window) and the sanity bands the runtime enforces. The runtime checks every
file against its capture manifest, the manifest's recipe digest against the
recipe shipped next to the trusted runtime, and then geometry, dtype, layout,
finiteness and the recipe's per-tensor sanity bands (RMS within x/÷2 and
abs-max within x/÷4 of the reference capture). The dataset assets are one
reference capture that satisfies these checks. A deployment may capture its
own copy instead; cross-GPU numeric drift in the captured bytes is expected and
accepted, while a wrong capture point fails closed. **Never re-pin a capture's
SHA-256 in the recipes or the portable manifests**: the asset manifest pins the
reference capture as data, the recipes stay the contract.

Optional re-capture (needs one idle GPU and the downloads of §1):

```bash
export FIXTURE_GPU=GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx      # your allocated card
sg docker -c 'umask 022; FIXTURE_GPU='"$FIXTURE_GPU"' bash fixtures/recipe/build_fixture.sh'
export REAL_LAYER_GPU="$FIXTURE_GPU"
sg docker -c 'REAL_LAYER_GPU='"$REAL_LAYER_GPU"' bash fixtures/recipe/real_layers/build_real_layers.sh'
```

The captures land in `fixtures/` and `fixtures/real_layers/` (gitignored) and
take precedence over the assets. `build_real_layers.sh` downloads the Llama
weights from an ungated mirror (`NousResearch/Meta-Llama-3.1-8B` at a revision
whose files are byte-identical to the recipe's pinned `meta-llama/Llama-3.1-8B`
snapshot) and verifies every pinned SHA-256 before it loads anything;
`REAL_LAYER_NO_MIRROR=1` downloads the gated original instead.

**The final holdout instance** is frozen, not drawn per build. It is Judge-only:
staged mode 0600 into a root-only Judge build context, it must never be copied,
mounted or described inside the Work image (`scripts/validate_layout.py` proves
no Work build context admits it, by path and by SHA sweep).
`portable/final-suite-manifest.json` lists every case's id, family, tier,
protocol and weight (shapes, lengths, models and seeds are withheld) and pins
the frozen instance's SHA-256, so every host that builds from this tree gets the
same `final_cases.sha256` and raw effective TFLOPS are comparable across hosts
within one measurement profile.
`scripts/generate_final_manifest.py --redraw --seed <int>` draws a reproducible
*new* instance; that is a **new suite identity** whose `frozen_instance` record
must be updated, and no earlier score is comparable to it.

### Image layout

Four images, two of them publishable:

| Image | Built from | Carries | Published |
| --- | --- | --- | --- |
| Work (published) `…work-public…:local` | `images/work/Dockerfile` | seed lab, public controller, Work's own public runtime | yes |
| Work (run) `…work…:local` | `images/work/Dockerfile.local` over the published Work | + the two public real layers at `/workspace/benchmark/fixtures/real_layers/` | no, host-built |
| public Judge `…judge-public…:local` | `images/judge/Dockerfile`, the same runtime and fixtures contexts | Judge, worker, runtime, recipes, the e8 capture | yes |
| Judge (run) `…judge…:local` | `images/judge/Dockerfile.private` over the public Judge | + the final suite (the 139 real layers are mounted read-only at run time by `scripts/sforge_judge_entry.py`) | no, host-built |

The host layers reproduce the paths, owners and modes of a single-file build,
so the Judge and the agent see the same files either way. No published image
carries a real-layer capture or the holdout.

## 4. CPU checks before any image or GPU operation

From the task root, with the **repository venv interpreter named explicitly** —
a bare `python3` is the system interpreter and lacks the repository's `sforge`
package:

```bash
../../../.venv/bin/python -B -m pytest -p no:cacheprovider tests -q
../../../.venv/bin/python -B scripts/validate_layout.py .
../../../.venv/bin/python -B scripts/refresh_portable_manifests.py --check
```

The tests recompute the scientific resource digest, delivery/build-resource
digest, manifest byte digests and suite case counts. Treat any mismatch as a
hard stop. The task JSON intentionally is not a member of either resource
manifest because it is the root document that binds their digests; each
resource manifest also excludes itself to avoid a recursive hash. Tests that
need the Judge-only holdout read it from the repository copy, else from the
asset root, else skip. Install CPU `torch` in the venv: without it the
Torch-dependent cases are skipped, which is lost coverage, not a pass.

Use `PYTHONDONTWRITEBYTECODE=1` or `PYTHONPYCACHEPREFIX=<scratch>`: the layout
validator rejects stray `__pycache__` inside the task tree.

## 5. Fetch, then build

```bash
tasks/latest/sageattention-quantized-attention-bf16-h100-v2-seed/assets/fetch.sh
cd tasks/latest/sageattention-quantized-attention-bf16-h100-v2-seed
bash scripts/build_images.sh                # both publishable images from source
# or: INFRATHON_PULL=1 bash scripts/build_images.sh   pull the published Work and public Judge
jq -e '.source_digest and .work.id and .work_local.id and .judge_public.id and .judge.id
       and .final_cases.sha256 and .fixture.sha256 and .fixture.recipe_sha256
       and (.real_layers | length == 2)' \
  ../../../outputs/sageattention-quantized-attention-bf16-h100-v2-seed/build/image-manifest.json
```

`build_images.sh` needs `docker`, `jq` and `sha256sum`; it re-execs itself
through `sg docker` when your shell has no docker group yet (use `umask 022` in
such a shell). It stages the final holdout (a regular non-symlink file with
`suite_id == final-holdout-v3` and exactly 774 cases, copied mode 0600 and
owned by the building user), locates the e8 capture and the real-layer captures
(§3), checks each real-layer file against its capture manifest and each
manifest against its recipe, and checks the two Work layers against their
capture manifests. Layout validation runs after that and is
handed the exact e8 pair being baked in (`validate_layout.py . --fixture …
--fixture-manifest …`). It then builds (or, with `INFRATHON_PULL=1`, pulls by digest
from `images/published.json`, refusing images published from different
image-driving source) the published Work and public Judge, and always builds
the two host layers.

Products: the four local images and, at the repository root,
`outputs/sageattention-quantized-attention-bf16-h100-v2-seed/build/image-manifest.json`
(source digest, all four image IDs, the holdout's digest/suite/case count, the
e8 capture and recipe digests, and per real-layer capture its recipe and
manifest digests) and, after `scripts/make_oracle_archive.sh`, a host-only
oracle archive. Every evaluation entry checks the recorded image IDs and digests
are still current (`assert_built_images_current`); do not substitute a
pre-existing tag without rebuilding.

## 6. Post-build verification

```bash
PHYSICAL_GPU=<index> bash scripts/smoke_test.sh --oracle
```

**This is the step that qualifies the deployment.** `PHYSICAL_GPU` is the index
of the card allocated to you (`nvidia-smi -L` lists them); `smoke_test.sh`
defaults to card **0**, so pass it explicitly on any multi-GPU host.

`--oracle` re-runs the layout validator and the task tests, then runs the
packaged seed through the participant-visible public runner inside the Work
image on the locked card (4 public cases, 8 edge gates, 2 real layers), packs
`oracle/` with `scripts/make_oracle_archive.sh`, and evaluates it through
`scripts/eval_archive.sh` (with `TRUSTED_CONTROL=1`, which skips the LLM
integrity review for this trusted control only) — a full trusted-Judge pass on
the final suite with the card's UUID flock held and an idle assertion on both
sides. Further control archives go through
`TRUSTED_CONTROL=1 PHYSICAL_GPU=<index> scripts/eval_archive.sh <archive> [RUN_ID]`.

Pass criterion: `valid: true` with all nine gates at 1 (`profile_ok`,
`suite_ok`, `source_ok`, `hardware_ok`, `correctness_ok`, `edge_ok`,
`real_activation_ok`, `fresh_input_ok`, `fresh_output_ok`), **not** a TFLOPS
threshold. Record the score as this card's reference: raw TFLOPS are a
property of the card (and its power limit) and are compared only within one
measurement profile. The Judge's private report also records an unscored BF16
cuDNN SDPA reference on the same card. A Judge pass over the 75-case final
suite takes on the order of 2 minutes.

## 7. Formal run

```bash
launch/sageattention-qattn/run.sh \
  --device-uuid GPU-00000000-0000-0000-0000-000000000000 \
  --judge-port 18096 \
  --model <model id> --effort max --hours 2 \
  --auth /path/to/auth.json
```

Qualify one scheduler-allocated whole NVIDIA GPU against the profile above
before this. Record the stable measurement profile (model, compute capability,
memory, driver, image IDs, software ABI, calibration protocol, and, when the
scheduler issues one, a calibration epoch tied to a documented clock/power
policy, `--calibration-epoch`). Record UUID and observed current clocks only in
deployment attestation. Without a calibration epoch, the run is unqualified for
a cross-run effective-TFLOPS leaderboard.

Flags: `--model` takes a full model ID (any Codex model, a `claude-*` model with
`--agent claude-code`, or any ID the `MODEL_API_BASE_URL` endpoint serves);
`--effort` takes `max|ultra|xhigh|high|medium|low`; `--hours N` and
`--timeout SECONDS` are alternatives (the latter wins, default 7200 s);
`--run-id` sets an explicit RUN_ID. Single-GPU is the default path:
`--device-uuid` must be a canonical whole-GPU UUID, `--allocation-key` and
`--lease-id` are optional (a never-reused lease ID is generated when omitted),
and single mode *rejects* the `WORK_GPU`/`JUDGE_GPU` variables rather than
falling back. A two-GPU path is available behind `--legacy-dual-gpu`.

**Budget / interval / cooldown contract.** The default run is `timeout: 7200`,
`eval_interval: 300`, `submission_cooldown: 300`, `disable_auto_eval: true`. The runner's `EVAL_INTERVAL` defaults to 1800 s and the same
value is passed to both `--eval-interval` and `--submission-cooldown`, so the
agent's manual submissions are rate-limited to one per interval. Tunable:
`TASK_TIMEOUT` (via `--hours`/`--timeout`; must be >= 90 s, a multiple of 60,
and strictly greater than `EVAL_INTERVAL`, which must also be a multiple of
60), `EVAL_INTERVAL`, `JUDGE_PORT`, `RUN_ID`. Frozen: `disable_auto_eval`
(in-run hidden evaluations are always 0) and the three Judge timeouts
(candidate worker 3000 s -> verifier 3300 s -> adapter 3400 s, with the task
JSON's `judge.eval_timeout = 3600` as the outer bound).

**Integrity review.** The Judge server runs the mandatory LLM integrity review
(`docs/LLM_JUDGE.md`): the launcher refuses to start unless
`SFORGE_LLM_JUDGE_MODE=enforce` and the reviewer settings
(`SFORGE_LLM_JUDGE_*`, including `SFORGE_LLM_JUDGE_MODEL`) are ready.

**Ports.** `--judge-port` must be 1024-65535, must not be 3128, 3130 or 3131
(reserved for the restricted proxy sidecar: CONNECT proxy, Judge relay,
optional model-API relay), and must be free when the launcher starts (it checks
the listening sockets and refuses otherwise). Default 18096. The scheduler must
allocate a different free port for every concurrent run on the same host.

**Version store.** Set `SFORGE_VERSION_STORE_ROOT` to a store that carries a
`.sforge-version-store.json` marker and registers this task under
`<store>/tasks/sageattention-quantized-attention-bf16-h100-v2-seed/task.json`;
launches fail closed otherwise. `SFORGE_AGENT_RUNTIME_DIR` must point at an
agent runtime built by `launch/build-agent-runtime.sh`, and
`CODEX_AUTH_JSON_PATH` at a regular non-symlink file, mode 0600, owned by the
launching user. `RUN_ID` must be unique inside one version store — reusing one
makes the deadline-time persistence of the final version fail with
`VersionStoreCollision` and wastes the whole budget.

**Proxy.** The Work container never receives an operator proxy: the launcher
exports `SFORGE_HTTP_PROXY` / `SFORGE_HTTPS_PROXY` itself, pointing at the
restricted sidecar on the internal network, and `scripts/sforge_isolated_entry.py`
attaches Work to that network. Before Work exists the launcher proves from
inside the internal network that the sidecar answers, the Judge relay on 3130
reaches the coordinator, `https://example.com` is not reachable directly,
`CONNECT chatgpt.com:443` answers 200, and `CONNECT github.com:443` answers 200
with public egress (403 in allowlist mode).
The sidecar's and entry's SHA-256 and the network mode
(`docker_internal_restricted_connect_proxy`, `direct_public_egress: false`,
`connect_allowlist`) are recorded in `experiment-manifest.json`.
`SFORGE_HTTP_PROXY` / `SFORGE_HTTPS_PROXY` / `SFORGE_NO_PROXY` set by the
operator matter only for image builds and re-captures; see
`launch/mlsysforge.env.example`.

## 8. Run artifacts and success criteria

```text
outputs/sageattention-quantized-attention-bf16-h100-v2-seed/runs/<RUN_ID>/
  completion.json
  experiment-manifest.json  run_config.json  judge-server.log  sforge-run.log
  <task_id>/
    resource_lease.json
    formal-precheck.json  formal-postcheck.json
    final_archive.tar.gz  final_result.json  final-<N>m-eval.log
    submissions/agent-N/{submission.tar.gz,report.json}
outputs/.../runs/<RUN_ID>-final-<N>m/<task_id>/submissions/manual-1/report.json
```

Success means all of: `resource_lease.json` in state **CLOSED**;
`formal-postcheck.json` PASS; `completion.json` with `status = passed`,
`sforge_run_exit = 0`, `final_eval_exit = 0`,
`hidden_evaluations_during_work = 0` and
`execution_profile_id = single-cuda-sm80-64g`; and the standalone final's
`manual-1/report.json` with `valid = true`.

**On failure, look at these first**, in order: `sforge-run.log` (agent side),
`judge-server.log` (coordinator and lease state), `<task_id>/run_agent.log`,
`final-<N>m-eval.log` (standalone final), and the rejection reasons inside
`formal-precheck.json` / `formal-postcheck.json`. Two recurring operational
traps: serialising runs on one card requires waiting for the previous run's
containers to actually exit (starting the next one within ~30 s hits "claimed
by a running container"), and a duplicate `RUN_ID` fails only at the very end.

## 9. What to re-measure on another GPU

Re-measure and re-record: every TFLOPS reading (the seed reference, anchor and
final scores); the headroom left by the three Judge timeouts on a
slower card; `GPU_MIN_FREE_MIB` if the card is smaller; the device-memory peak;
and the whole measurement-profile comparability key. The reference captures
may be reused on any card, or re-captured there (§3).

Do **not** re-calibrate: the ABI, the frozen final suite and the public cases, the
eight edge fixtures and the capture recipes, the accuracy gates (they are
differential against the trusted Sage1 simulation and contain no absolute
performance term), the `score_tflops` formula, the capability constants, or
the suite `manifest_sha256` values.

Raw effective TFLOPS may be compared only when both runs are qualified and
their recorded comparison identity is identical. That identity is the ordered
pair of the evaluation-contract SHA-256 and the measurement-profile
comparability key; the contract digest is deliberately not folded into the
pure machine profile hash. Merely satisfying the broad `single-cuda-sm80-64g`
capability envelope does not make two runs comparable.
