# Rebuild the KV Eviction Policy Lab v2 on a qualified host

This is the **operational layer** of the task: how to bring the Work and Judge
images up on a clean host, verify them and evaluate archives. What the task
*is* (metric, suites, gates) is defined by the TaskCore listed in
`portable/resource-manifest.json`, the agent brief `images/work/AGENT_TASK.md`
and the task README. A build agent may run these steps; it must not redesign
the benchmark.

Preserve these scientific identities byte for byte:

- the TaskCore and its resource manifest (`portable/resource-manifest.json`);
- the 85-item public suite `kv-eviction-public-v2` (`corpus/public/items_public.jsonl`)
  and its `one` / `quick` subsets;
- the 5,000-item final holdout `final-holdout-v2`, frozen as
  `judge/private/final_manifest.json` (SHA-256
  `e0bf8572a0a9ca37195af0aa8ac963388a4d389fb20339e4e60115965e08ed65`); the
  rebuilt `final_items.jsonl.gz` must hash to
  `e8230ff7e5d98c742fe94340989d8d2574d2d0f283e8573fa81a58a2771a8476`;
- the frozen Qwen3-8B binding, the context-only Phase A, the bitmask handoff,
  the trusted Phase-B replay, the three ratios (0.5 / 0.75 / 0.875), the
  fidelity metric and the eleven run-wide fail-closed gates;
- the agent-facing contract (`images/work/AGENT_TASK.md`) and the integrity
  policy (`llm_judge/POLICY.md`).

Paths are relative to `tasks/latest/kvpress-kv-eviction-policy-qwen3-8b-h100-v2-scratch/`;
`<repo>` is the repository root. Host-private values are placeholders.

---

## 1. Host prerequisites

Do `<repo>/docs/HOST_PREREQUISITES.md` first: Docker ≥ 24 with the NVIDIA
runtime as the default runtime, your user in the `docker` group, `uv` and the
repository venv (with CPU `torch` for the host tests), `jq flock sha256sum
nvidia-smi`, and ≥ 150 GB free on Docker's data root. In addition:

| Need | Why |
| --- | --- |
| the four pinned Hugging Face datasets in a local hub cache (§3) | `scripts/prepare_items.py materialize` rebuilds the holdout from them at every build; no network is used |
| a data Python with `pyarrow`, `transformers`, `tokenizers`, `jinja2` | `KV_EVICTION_DATA_PYTHON`, or let `scripts/lib.sh` create a uv venv under the logs directory |
| a local Qwen3-8B HF repository with `refs/main` | bind-mounted read-only; `hf download --revision` does not write `refs/main`, create it with the pinned revision |
| ≥ 12 GB free in the Judge container's `/tmp` | the Phase A → Phase B handoff for 5,000 items is ~8 GB; the verifier refuses to start below 1.25 × estimate + 1 GiB |

## 2. GPU envelope and who enforces it

Profile `cuda-sm80-56g-sharded-judge` (task JSON `portable.execution_profiles`):

| Item | Value | Enforced by |
| --- | --- | --- |
| Work | one whole GPU, cc ≥ 8.0, ≥ 56 GiB | `runtime/kv_eviction_runtime.py` (`hardware_inventory`, exactly one visible device) |
| Judge | **1–8 identical whole GPUs**, each cc ≥ 8.0 and ≥ 56 GiB | `images/judge/judge_shards.py` (`validate_fleet`: count, identical name / memory / capability, envelope), then `hardware_inventory` again inside every shard process |
| Judge CPU / memory | `8 × N` CPUs, `32 + 8 × N` GiB | `scripts/eval_archive.sh` (`--judge-cpu-limit`, `--judge-mem-limit`) |
| software ABI | PyTorch 2.8.0 / CUDA 12.9 / Triton 3.4.0 / transformers 4.57.1 / accelerate 1.10.1; `kvpress`, `flash_attn`, `xformers`, `flashinfer` absent | both Dockerfiles assert it at build time; `portable/software-abi.json` |

Measure device memory peaks on the target host and re-measure after any
driver, PyTorch or suite change. GPU name and UUID are provenance only.

## 3. External inputs

| Input | Pin | Notes |
| --- | --- | --- |
| `Qwen/Qwen3-8B` | revision `b968826d9c46dd6066d109eabc6255188de91218` | `KV_EVICTION_MODEL_REPO=<local HF repo>`; ~16 GiB, mounted read-only |
| `simonjegou/ruler` | `24adceac8a0e6532936e8d721cd9e9084d2e4686` | in `KV_EVICTION_HF_HUB` (default `~/.cache/huggingface/hub`) |
| `Xnhyacinth/LongBench` | `2e9ade51ebf45d98942056c0716234f9d5d257d5` | same |
| `simonjegou/loogle` | `6e3eaf28d24ce34933170c85de36be2448d0d2a7` | same |
| `simonjegou/zero_scrolls` | `1994930942afa252f7707cb9c74253b774d7ed8d` | same |
| base image | `pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel` | both Dockerfiles |

Fetch a dataset snapshot with `hf download <repo> --repo-type dataset --revision <pin>`.
There is no owner-only input: everything derives from pinned public sources.
Prove the identities before building:

```bash
"$KV_EVICTION_DATA_PYTHON" scripts/prepare_items.py verify --hf-hub "$KV_EVICTION_HF_HUB" \
  --model-dir "$KV_EVICTION_MODEL_REPO/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
```

A different result is not corruption: a source moved or a rule changed, and
the outcome is a **new suite identity** whose scores are not comparable.

## 4. What is frozen

The repository holds the public items and the holdout **manifest** (dataset,
revision, parquet file, row and SHA-256 per item), never the holdout text.
`scripts/build_images.sh` runs `prepare_items.py materialize`, which rebuilds
the 5,000 items, checks every SHA-256 against the manifest and aborts on any
difference; the file is staged 0600 into a build context that only the Judge
image reads. `scripts/validate_layout.py` proves no Work build context admits
a holdout file and no Work-visible file names a holdout source.

## 5. Build

```bash
export KV_EVICTION_MODEL_REPO=<local Qwen3-8B HF repo>
export KV_EVICTION_HF_HUB=<hub cache with the four datasets>
export KV_EVICTION_DATA_PYTHON=<python with pyarrow + transformers>   # optional
bash scripts/build_images.sh
```

The build writes `<repo>/outputs/kvpress-kv-eviction-policy-qwen3-8b-h100-v2-scratch/build/image-manifest.json`
with the image ids, the image source digest and both holdout digests;
`scripts/eval_archive.sh` refuses to run when the source or the image tags
moved after that build.

## 6. Verification

CPU only:

```bash
.venv/bin/python scripts/validate_layout.py .
.venv/bin/python -m pytest -q tests
.venv/bin/python scripts/refresh_portable_manifests.py --check
```

With GPUs:

1. Inside the Work image: `./tools/run_public_tests.sh --quick --candidate-version 1`;
   the seed must pass every gate.
2. Evaluate the seed archive on the full holdout:
   `JUDGE_GPUS=<i,j,k> scripts/eval_archive.sh <seed.tar.gz>`.

An archive holds the lab root's `FINAL_VERSION`, `notes.md` and
`candidates/policy_vNNN/` (see `scripts/make_oracle_archive.sh`).

## 7. Formal runs

Scored submissions and the final both run the full holdout
(full-final-shared-v2). Formal agent runs go through `scripts/run_agent.sh`
(launcher `<repo>/launch/kvpress-kv-eviction-scratch/run.sh`): one Work GPU
plus exactly 7 Judge GPUs (the paper's setup; the launcher refuses another count), with every interactive Judge and the independent final
Judge running the complete 5,000-item suite sharded over the Judge GPUs.
There is no single-GPU formal path; never route a run through a single-GPU
Judge handoff.

## 8. What to re-measure on another GPU

Device memory peak per Judge GPU, Phase A / Phase B wall time per shard,
suite latency ratio of a reference policy, and scratch usage. Item results do
not depend on the number of Judge GPUs; they may differ slightly across GPU
models, which is why the Judge requires identical devices.
