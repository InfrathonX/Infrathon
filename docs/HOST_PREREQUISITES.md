# Host prerequisites for rebuilding any task on a new GPU machine

Every task's rebuild instructions (`tasks/latest/<task>/portable/REBUILD_PROMPT.md`
where a task has one) assume the host below. Nothing here depends on the GPU
model; the task's execution profile (in its task JSON) decides whether the card
is admissible.

## Once per host

| Need | Check | If missing |
| --- | --- | --- |
| Docker ≥ 24 with the **NVIDIA runtime as default** | `docker info --format '{{.DefaultRuntime}}'` prints `nvidia`; `docker run --rm -e NVIDIA_VISIBLE_DEVICES=all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi -L` lists the GPU | install `nvidia-container-toolkit`, then `sudo nvidia-ctk runtime configure --runtime=docker --set-as-default && sudo systemctl restart docker`. The harness passes GPUs with `-e NVIDIA_VISIBLE_DEVICES=<uuid>`, not `--gpus`, so a non-default runtime silently yields containers without `nvidia-smi`. |
| Your user in the `docker` group | `id -nG \| grep -qw docker` | `sudo usermod -aG docker $USER`. **That does not take effect in a shell that is already running**, so either open a new login session or use `sg docker -c 'umask 022; …'`. Most task entry points re-exec themselves through `sg docker` when the socket is unreachable, but **not all of them** — see the exception list below. The `umask 022` is not optional: captures, staged manifests and build contexts are written into the task tree and must stay readable by the build. |
| `uv`, Python 3.12 (3.10–3.13 accepted) | `uv --version` | `curl -LsSf https://astral.sh/uv/install.sh \| sh`; it lands in `~/.local/bin` |
| `jq flock rsync patch git sha256sum nvidia-smi` | `command -v …` | distro packages; `nvidia-smi` comes with the driver |
| `hf`, the Hugging Face CLI | `hf --version` prints **1.23.0** or newer | `uv tool install 'huggingface_hub[cli]==1.23.0'`. **Hard requirement, not a convenience**: attn-qat's `fixtures/recipe/build_fixture.sh` calls `require_command hf` and dies without it, and the Qwen3-8B snapshot kvzip and kvpress bind is fetched with it. 1.23.0 is the version the recipes here are exercised with; see the note below on why the version matters. |
| Repo venv | `uv sync --frozen --dev --python 3.12 && uv run --frozen python -m pytest -q` | — |
| **CPU `torch` + `numpy` in the venv** | `.venv/bin/python -c 'import torch, numpy'` | `uv pip install --python .venv/bin/python numpy==2.5.2`, then `uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cpu torch==2.14.1`. Several tasks' host-side tests, asset preparers and layout checks import them (kvpress, kvzip, the standalone kernels' edge-case generators, which `build_images.sh` runs); without torch the tests are *skipped*, which hides failures. Not in `uv.lock` on purpose. |
| Disk, images | **One task: 20–25 GB free.** **All non-Miles tasks: ~400 GB**, and that is images plus the per-deployment captures and staged assets they write beside them. The Miles tasks add their own base images (see [scripts/infrathon/README.md](../scripts/infrathon/README.md)). **Find the data root** — it is not always `/var/lib/docker` — with `docker info --format '{{.DockerRootDir}}'`, then `df -h "$(docker info --format '{{.DockerRootDir}}')"` before you start. | see the cold-build budget below |
| Disk, assets and model cache | **The sum of `size` in the task's `assets/manifest.json`** (median task ~15 GiB; quant-videogen ~200 GiB, Wan2.2 ~66 GiB, SageAttention ~64 GiB). All 25 tasks for one hardware type: up to ~600 GiB, less where tasks share a model. Assets go to `INFRATHON_ASSETS_ROOT`, models to the Hugging Face cache (`HF_HUB_CACHE`, default `~/.cache/huggingface/hub`); the cache also keeps a copy of the Judge-only dataset files | see the cold-build budget below |
| Network | `docker pull`, Hugging Face, GitHub | models and pinned upstream checkouts are fetched, never vendored |

### Which entry points re-exec through `sg docker`, and which do not

`ensure_docker_group` in a task's `scripts/lib.sh` (or, for llada's two Python
validators, in `scripts/built_image_policy.py`) probes `docker info`, and when
it fails re-execs the script once under `sg docker` with `umask 022`, forwarding
only that task's own non-secret controls. It dies with a named error if Docker
is still unreachable afterwards, or if your user is not listed in the `docker`
group at all.

**Covered** — run these directly, in any shell: every `scripts/build_images.sh`
(all tasks except qwen3-orpo and MInference), eagle3's `smoke_test.sh`,
`smoke_submission_plane.sh`, `eval_archive.sh`, `replay_candidates.sh` and
`run_fixed_budget.sh`, llada's `smoke_test.sh`, `eval_archive.sh`,
`smoke_submission_plane.sh`, `run_fixed_budget.sh` and both
`validate_built_*_policy.py`, minference's `run_local_v3.sh`, and qwen3-gkd's
`managed_seed_smoke.sh`, `eval_archive.sh` and `replay_candidates.sh`.

**Not covered** — wrap these yourself as
`sg docker -c "umask 022; cd <task dir>; <command>"`:

| Entry point | Task(s) |
| --- | --- |
| `fixtures/recipe/build_fixture.sh` | attn-qat |
| `scripts/build_images.sh` | qwen3-orpo |
| `scripts/managed_seed_smoke.sh` | qwen3-orpo |
| the `launch_*_after_*_idle.sh` waiters | qwen3-orpo, qwen3-gkd |

Python helpers that a covered shell script invokes (`scripts/preflight.py`,
`scripts/runtime_identity.py`, `scripts/sforge_dual_h100_entry.py`, the
StreamDiffusionV2 `F*_*.py` batch tools, …) inherit the re-exec'd process's
groups and are not separate exceptions; they only need a wrapper if you run them
standalone from an un-refreshed shell.

## Base images are not pulled for you

`docker build` uses whatever bytes a tag already names locally and fetches the
base only when it is absent; eagle3's `scripts/build_images.sh` passes
`--pull=false` explicitly to say so. **Neither form refreshes a tag you already
have**, and every base below is a mutable tag, so two hosts can build the "same"
image from different bytes without any message saying so. Pull them yourself,
once per host, before the first build — it also keeps a 9–18 GB download out of
the middle of a timed build:

```bash
docker pull pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel   # most tasks
docker pull pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel   # llada, flash-kmeans
docker pull ubuntu:24.04                                  # concurrent-serving (digest-pinned)
docker pull pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel   # streamdiffusionv2 (digest-pinned)
```

The last two are pinned by digest in their Dockerfiles, so those two builds are
reproducible regardless; the two `pytorch/pytorch` tags above are not, and are
the ones worth pulling deliberately. Record the image ID you pulled next to the
build — it is part of the measurement profile.

## Task-specific host extras

The table above is what *every* task needs. A few tasks need one more thing in
the interpreter their host-side checks run under. The `torch` row applies to every
run of those tasks; the `trl` row only when you rebuild them, and each task's rebuild
layer repeats it next to the command that needs it.

| Task(s) | Extra | Why, and how |
| --- | --- | --- |
| flux-moe-ag, streamdiffusionv2 | a **CUDA build of `torch`** in the `python3` first on `PATH` | Once it has locked the GPUs, the launcher runs `scripts/preflight.py` under the host `python3`: streamdiffusionv2 on the Work pair and, in the default separate mode, again on the Judge pair, checking NVLink peer access and symmetric memory between the two GPUs of a pair; flux-moe-ag once on its eight GPUs, checking peer access between every pair (NVLink or PCIe) and an eight-rank symmetric-memory rendezvous. A system `python3` without torch fails this preflight. A dedicated venv first on `PATH` works: `uv venv ~/.venvs/host-torch --python 3.12 && uv pip install --python ~/.venvs/host-torch/bin/python --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0`, then `export PATH=~/.venvs/host-torch/bin:$PATH`. Do not put the repository `.venv` first: its torch is the CPU build. |
| quant-videogen | **`git-lfs`** on `PATH`, and about 200 GiB free under `INFRATHON_ASSETS_ROOT` | `assets/fetch.sh` checks out Glyph-SDXL-v2 from ModelScope, whose weights are Git LFS files; without `git-lfs` the fetch stops and names it. The release binary works without root: unpack it from https://github.com/git-lfs/git-lfs/releases and put the `git-lfs` executable on `PATH`. The models, checkouts and evaluation inputs are about 200 GiB, and the tree the launcher mounts is assembled as hard links into the asset root, so it takes no more. |
| qwen3-orpo, qwen3-gkd | **`trl==0.12.1`** (plus `transformers`) in the system `python3` | `scripts/smoke_test.sh` runs `images/work/public_tests/run_public_tests.sh --cpu-only`, which builds the official-TRL reference. Without the wheel the eval prints `PackageNotFoundError: No package metadata was found for trl` and exits 1, and `set -euo pipefail` ends the smoke before pytest. Both smokes check this first and print the command; it is `uv pip install --python "$(command -v python3)" transformers 'trl==0.12.1'`. The version must be **exact**: `runtime/hf_reference.py` binds the baseline to that version *and* to the SHA-256 of its trainer source, the same wheel `images/work/Dockerfile` installs. Install into the **system** `python3`, not by putting the repo venv ahead of it on `PATH` — these tasks' preflight P2P/torch probe runs under the system interpreter. |

## What a cold build costs

Two numbers matter, and they are far apart. Budget for the one you are actually
about to do.

| Item | One task | All non-Miles tasks |
| --- | --- | --- |
| Images | **17–20 GB per image pair** (one Work + one Judge) | **~400 GB.** Sharing helps less than the arithmetic suggests: most pairs sit on the same `pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel` base, which Docker stores once, but several tasks are much larger than the median and a few write large captures or staged assets beside their images. **streamdiffusionv2 is the heaviest pair of the set at ~31 GiB Work + ~42 GiB Judge (~74 GB together)**; concurrent-serving's four images plus the vLLM host image come to **~60 GB**; **minference at ~34 GB is the heaviest of the single-image-pair majority.** Leave **≥ 150 GB free on the Docker data root** before building either of the first two. |
| Captures and staged assets | 0 for most tasks; attn-qat builds a throwaway capture-builder image you can prune afterwards | **streamdiffusionv2 alone stages ~19 GB** of build contexts (an 11.9 GB `weights` tree, 4.5 GB of judge goldens, 1.7 GB of assets, a 188 MB flash-attn wheel) out of a **~15 GiB asset root** that has to be on the host first, and minference writes its activation capture before its build |
| Wall time | **10–30 min** for one image pair on a warm network | proportionally longer for the full set. Almost all of it is `apt` and `pip` inside the builds. |
| Hugging Face cache | **≤ 16 GB** for a single model-bound task | **~70 GB**, with real sharing: `Wan-AI/Wan2.1-T2V-1.3B` **16 GB** for attn-qat; `Qwen/Qwen3-8B` **16 GB** covers kvzip, kvpress and eagle3; `Jang-Hyun/Fast-KVzip` **1.3 GB** is kvzip's host-only oracle asset; minference and llada each pull an 8B model of their own. |

Task data and the Judge-only inputs come from the Hugging Face dataset through each task's
`assets/fetch.sh` (README, Images, data and hardware types).

### Other long steps

These are steps whose cost is easy to underestimate when budgeting a rebuild:

| Step | Typical cost | Note |
| --- | --- | --- |
| minference activation capture | **~7 minutes**, 40 cases, **~32 GB** written | needs the 15 GB model snapshot first; with its image layers, minference is the heaviest single task on the Docker data root |
| qwen3-gkd `managed_seed_smoke.sh` | **~45 minutes** | — |
| llada 520-case reference replay | **~40 minutes** | — |
| eagle3 `smoke_test.sh --docker`, full anchor pass | **~7 minutes** | — |
| eagle3 in-image draft-head conversion | **~1.5 minutes** on a warm head cache | ~2 GB of head files are fetched when cold |
| llada reference producer run | **~3 hours** | needed once on a new gpu_name/compute_capability/driver triple before its replay means anything |

A build that hits Docker's layer cache finishes in seconds and exercises none of
the network path. In particular qwen3-gkd's build requires that **pip can reach
the network inside the build** (`pip download trl==0.12.1` with a
wheel-and-source SHA check). If you are validating a proxy or a network path,
build with `docker build --no-cache` or on a host that has never built the pair.

### On a shared host, check the data root first

Docker's data root is shared by every container on the machine, including ones
you did not start. **When it fills, every container on the host stalls, not just
your build** — and the failure does not always name disk. A build that runs out
of space can die with `ResourceExhausted: … write … metadata_v2.db: no space
left on device`, or with a bare `mkdir: … No space left on device` from a script
that has nothing to do with Docker.

So before each build, and again if anything behaves strangely:

```bash
df -h "$(docker info --format '{{.DockerRootDir}}')"
```

Do not free space with `docker system prune -a` on a shared host: it deletes
other people's images and caches too. Remove only images you built yourself.

## Running more than one run on one host

Concurrency here is per card, and nothing discovers another run for you. Five
things have to be arranged by whoever launches them.

* **One judge port per concurrent run.** The single-GPU runners refuse a port
  that is already in use, so give each run its own `--judge-port`. Each launcher
  has its own default port (launch/README.md), but two runs of the same task, or
  task scripts called directly (they share one default), need a `JUDGE_PORT` of
  their own; without one the second run registers itself against the first
  run's Judge and dies with `403 Invalid admin secret`. The port has to be
  genuinely free at preflight — a runner binds it there, and a listener left
  behind by anything else on the host is enough to stop the launch.
* **Single-GPU and dual-GPU tasks use different GPU lock files**
  (`/tmp/sforge-gpu-<uuid>.lock` versus `/tmp/mlsysforge-gpu-<uuid>.lock`).
  The dual-GPU launchers take both, but a task script called directly may take
  only its own, and then all that separates the two families is an idle probe —
  no compute process, free memory above the floor — so **never schedule a
  single-GPU run and a dual-GPU run on the same card.**
* **A card touched by a foreign process fences the run.** If a process that is
  not the run's own appears on a leased card, the coordinator marks the lease
  `FENCED`, Work is never resumed, and `run_agent` exits with
  `HandoffFencedError` within seconds. Check the card is clear before choosing
  it (`nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv`). Recovering a
  fenced run is manual: kill that run's `sforge serve --port <n>`,
  `docker rm -f` the paused Work container, and move the run directory out of
  `outputs/<task>/runs/` — the coordinator refuses to attach any non-`CLOSED`
  lease on the same allocation.
* **An interrupted single-GPU run blocks the next run of that task on that card.**
  A run stopped by hand (Ctrl-C, a killed tmux session) leaves its lease in
  `WORK_ACTIVE`, and the next launch fails at attach with only
  `409 Client Error: Conflict` in its log (the coordinator's reason, "accelerator
  has an unreconciled durable lease", is not printed). Clean up as for a fenced run:
  stop its `sforge serve`, remove its containers (Work and the restricted proxy) and
  its `-internal` Docker network, and move the run directory out of
  `outputs/<task>/runs/`.
* **Long-lived "inspect" containers claim every GPU.** A container started with
  `NVIDIA_VISIBLE_DEVICES=all` and no device request — the usual shape of a
  `sleep infinity` debugging container — counts as claiming all cards under the
  conservative preflight some tasks use (eagle3, llada), which then refuses
  every card on the host even though the container uses no memory at all. The
  same check trips on the previous run's containers when they have not fully
  exited, so when chaining runs on one card, wait for them to disappear from
  `docker ps` first.
* **`RUN_ID` must be unique within one `SFORGE_VERSION_STORE_ROOT`.** A reused
  id is only detected when the final version is persisted after the deadline
  (`VersionStoreCollision`), which wastes the whole agent budget. Some runners
  check this at preflight; do not rely on it.

## Building behind a proxy

If this host reaches the internet through a proxy, export the six variables
**before** you build anything. Every `scripts/build_images.sh` and every
`fixtures/recipe/build_fixture.sh` runs `docker build --network host` and
forwards the inherited `HTTP_PROXY` / `HTTPS_PROXY` / `NO_PROXY` (and their
lowercase spellings) as build args:

```bash
export http_proxy=http://<proxy-host>:<proxy-port>  https_proxy="$http_proxy"
export HTTP_PROXY="$http_proxy"                     HTTPS_PROXY="$http_proxy"
export no_proxy=localhost,127.0.0.1                 NO_PROXY="$no_proxy"
```

Because builds join the host network, a proxy that only listens on host
loopback is still reachable **during a build**. It is *not* reachable during a
formal run: the launcher hands its own `SFORGE_HTTP_PROXY` /
`SFORGE_HTTPS_PROXY` / `SFORGE_NO_PROXY` to a container on the Docker bridge and
refuses to start when the effective proxy is loopback-only. Keep the two sets of
variables separate — build-time `http_proxy`, run-time `SFORGE_*` at a
bridge-reachable address. With direct internet access leave all of them unset.

## Pin the `hf` CLI, and never parse its stdout

The recipes in this repository are exercised end to end against
**`huggingface_hub` 1.23.0**, which is what `hf --version` reports. Check it
before a rebuild:

```bash
hf --version                       # expect 1.23.0 or newer
python3 -c 'import huggingface_hub; print(huggingface_hub.__version__)'
```

The CLI's **stdout format is not stable across releases**: 1.x prints
`path=/…/snapshots/<revision>` where earlier releases printed the bare
directory. A recipe that treats that line as a path breaks on a CLI upgrade, for
example with `hf download did not report a snapshot directory` before any GPU
work.

minference's `activations/recipe/build_activations.sh` therefore derives the
snapshot directory from the cache layout its own pin implies and treats stdout
as a fallback. **Write any new recipe the same way**:

```text
${HF_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/hub}/models--<org>--<name>/snapshots/<revision>
```

That path is a function of the pin, not of the CLI, and every release writes
it. Verify the per-file digests your recipe pins against whatever directory you
resolved, before spending GPU time on it.

## Hugging Face cache conventions

Tasks bind models by **repository + pinned revision** through the standard
`~/.cache/huggingface/hub/models--<org>--<name>` layout. `assets/fetch.sh` downloads the
revision pinned in the task's `assets/manifest.json` and writes `refs/main`. Without it, fetch
with `hf download <org>/<name> --revision <sha>`; that command does **not** write `refs/main`,
which the kvpress/kvzip validators read, so you then have to create that file yourself.

`<sha>` is the very revision you passed to `hf download` — the 40-hex commit of
the model repository, not a digest of any file. The Qwen3-8B binding that
kvzip, kvpress and eagle3 all share is revision
`b968826d9c46dd6066d109eabc6255188de91218`, so the whole sequence is:

```bash
hf download Qwen/Qwen3-8B --revision b968826d9c46dd6066d109eabc6255188de91218
echo b968826d9c46dd6066d109eabc6255188de91218 \
  > ~/.cache/huggingface/hub/models--Qwen--Qwen3-8B/refs/main
```

`validate_model_repo` in each task's `scripts/lib.sh` reads `refs/main`, then
requires `snapshots/<revision>/config.json` to exist with
`model_type == "qwen3"` and `num_hidden_layers == 36`; it accepts any 40–64 hex
revision satisfying that; the task manifests pin the revision `assets/fetch.sh` uses. Record
the one you used — it is part of the suite identity, and the
kvzip and kvpress qualification records are taken on the revision above.

## Bringing a task up on a new card

The order below is generic; each task's rebuild layer names its own commands and
its own extra artefacts. Step 1 is an automatic predicate. **Steps 2–8 are what
"qualification" means** — they validate the measurement on this card, not the
task; the task itself does not change by one byte anywhere in this list.

1. **Check the capability envelope first.** Read
   `portable.execution_profiles` in the task JSON and confirm what `nvidia-smi`
   reports for compute capability, memory and device count satisfies it. If it
   does not, stop: this card cannot run this task without defining a different,
   smaller suite, which is a different task.
2. **Do everything that needs no GPU**: `python3 scripts/validate_layout.py .`,
   `python3 -m pytest tests -q` (the CPU/meta tests, which check the protected
   hashes against the contract), and `bash scripts/build_images.sh`. Base images
   are generic `pytorch/pytorch:*-devel` and `TORCH_CUDA_ARCH_LIST` already spans
   8.0–10.0, so a failure here means a missing input, not a missing capability.
3. **Run the task's post-build verification on one card** —
   `scripts/smoke_test.sh --oracle` or the equivalent its rebuild layer names —
   and read a failure carefully. A `CompilationError`, an `OutOfResources` or a
   layout error *inside the candidate or reference implementation* is a
   hard-coded tile, shared-memory budget or tuning table, and is fixed by giving
   the reference implementation a new candidate identity. Only a failing **Judge
   gate** (capability envelope, memory, envelope sanity) is a contract problem.
4. **Run the full suite once and watch the memory peak.** The profile's
   `min_memory_bytes` has to cover the **Judge** process — its inputs plus its
   private reference computation — not the candidate's peak.
5. **Look at the timeouts.** If the Judge dies on `eval_timeout` or a worker
   timeout, widen the ladder in the task JSON and the Judge Dockerfile. That is a
   deployment budget and it changes no score.
6. **Measure the noise.** Replay one submission 3–6 times (most tasks ship a
   `replay_*` script) and look at the primary metric's CV. The qualified range
   is 0.3 %–1.3 %; a clearly wider spread on a new card points at clocks and
   boost policy before it points at the task.
7. **Produce the task-specific artefacts**, where a task has them: llada runs
   `scripts/calibrate_reference.py` once per measurement profile and fails closed
   against a reference from any other profile; StreamDiffusionV2 and eagle3
   generate their seed reference inside each Judge run and need no pre-pass.
8. **Register the profile.** Record this card's model, compute capability,
   memory, driver, image IDs and software ABI in `execution_profiles` (a new
   profile id if the envelope moved), together with the seed score and CV.
   From then on this card's raw scores compare only with runs in the same
   profile; against other profiles only relative quantities compare.

## What is *not* a host prerequisite

- The agent CLI (`launch/build-agent-runtime.sh`) and `launch/mlsysforge.env`
  are needed only for formal agent runs, not for rebuilding and qualifying a
  task with its Oracle/seed archive.
- `sudo` is needed only for the toolkit step above.
