# Rebuild the KVzip Query-Agnostic KV Scoring Lab v2 on a qualified host

This is the **operational layer** of the task: how to bring the Work and Judge
images up on a clean host, verify them and evaluate archives. What the task
*is* (metric, suites, gates) is defined by the TaskCore listed in
`portable/resource-manifest.json`, the agent brief `images/work/AGENT_TASK.md`
and the task README. A build agent may run these steps; it must not redesign
the benchmark.

Preserve these scientific identities byte for byte:

- the TaskCore and its resource manifest (`portable/resource-manifest.json`);
- the 63-row public suite `kvzip-scoring-public-v2` (`corpus/public/items_public.jsonl`,
  SHA-256 `7ede4c7b26350678e467441e2234ee31adce275c4cf170fcc9bad016762749ce`)
  and its `one` / `quick` subsets;
- the 1,812-row / 6,956-question final holdout `final-holdout-v2`, frozen as
  `judge/private/final_manifest.json` (SHA-256
  `d3e69ab7a4aa17fc30fc089439e6ea8be4ab473e386d9e636e793caa8aca38b9`) plus the
  three private needles `judge/private/needle_private.json` (SHA-256
  `51b496d2477e7c1ac7c4f4a4e607a6a9ae6c6425143a3cd4b326f1ae6b7b472e`); the
  rebuilt `final_items.jsonl.gz` must hash to
  `6ec0115d8daf2b0199643fa067f3ce4f69445bd7825f118809ee71c36e0b0fd0`;
- the frozen Qwen3-8B binding, the context-only Phase A with the fresh-copy
  repeat call, the trusted Phase-B chunked decode, the three retention ratios
  (0.1 / 0.2 / 0.3), the relative-accuracy metric and the eleven run-wide
  fail-closed gates;
- the agent-facing contract (`images/work/AGENT_TASK.md`) and the integrity
  policy (`llm_judge/POLICY.md`).

Paths are relative to `tasks/latest/kvzip-query-agnostic-kv-scoring-qwen3-8b-h100-v2-scratch/`;
`<repo>` is the repository root. Host-private values are placeholders.

---

## 1. Host prerequisites

Do `<repo>/docs/HOST_PREREQUISITES.md` first: Docker ≥ 24 with the NVIDIA
runtime as the default runtime, your user in the `docker` group, `uv` and the
repository venv (with CPU `torch` for the host tests), `jq flock sha256sum
nvidia-smi tar`, and ≥ 150 GB free on Docker's data root. In addition:

| Need | Why |
| --- | --- |
| the pinned datasets in a local hub cache and a KVzip checkout (§3) | `scripts/prepare_items.py materialize` rebuilds the holdout at every build; no network is used |
| a data Python with `pyarrow`, `transformers`, `tokenizers`, `jinja2` | `KV_SCORING_DATA_PYTHON`, or the uv venv `scripts/lib.sh` creates |
| a local Qwen3-8B HF repository with `refs/main` | bind-mounted read-only; create `refs/main` with the pinned revision |
| ≥ 36 GB free in the Judge container's `/tmp` | Phase A hands Phase B FP32 scores, ~27 GB for 1,812 rows; the verifier refuses to start below 1.25 × estimate + 1 GiB |

## 2. GPU envelope and who enforces it

Profile `cuda-sm80-79g-sharded-judge` (task JSON `portable.execution_profiles`):

| Item | Value | Enforced by |
| --- | --- | --- |
| Work | one whole GPU | `runtime/kv_scoring_runtime.py` (`hardware_inventory`, exactly one visible device) |
| Judge | **1–8 identical whole GPUs** within the envelope | `images/judge/judge_shards.py` (`validate_fleet`), then `hardware_inventory` inside every shard process |
| Judge CPU / memory | `8 × N` CPUs, `32 + 8 × N` GiB | `scripts/eval_archive.sh` |
| timeouts | at least candidate 7,200 s, verifier 10,800 s, adapter 10,900 s, `sforge eval` 11,000 s; larger for small `N` (candidate `600 + 6 × 1812 / N` s, verifier `900 + 15 × 1812 / N` s, adapter and eval +100 s / +200 s) | `scripts/lib.sh` (`judge_timeouts`), used by `scripts/eval_archive.sh` and `scripts/run_agent.sh` |
| software ABI | PyTorch 2.8.0 / CUDA 12.9 / Triton 3.4.0 / transformers 4.57.1 / accelerate 1.10.1 / rouge 1.0.1 / tree-sitter-languages 1.10.2 / nltk 3.9.1; `kvzip`, `flash_attn`, `xformers`, `flashinfer` absent | both Dockerfiles assert it; `portable/software-abi.json` |

Measure device memory peaks on the target host. The Phase-B decode chunk size
depends only on token counts (40 GiB of expanded cache, at most 16 pairs),
never on the device, so scores do not depend on the card or the GPU count.

## 3. External inputs

| Input | Pin | Notes |
| --- | --- | --- |
| `Qwen/Qwen3-8B` | revision `b968826d9c46dd6066d109eabc6255188de91218` | `KV_SCORING_MODEL_REPO=<local HF repo>` |
| `Jang-Hyun/SCBench-preprocessed` | `a079be919d3131822c202180ffd4dc322968de45` | in `KV_SCORING_HF_HUB` |
| `rajpurkar/squad` | `7b6d24c440a36b6815f21b70d25016731768db1f` | same |
| `openai/gsm8k` | `740312add88f781978c0658806c59bc2815b9866` | same |
| `simonjegou/ruler` | `24adceac8a0e6532936e8d721cd9e9084d2e4686` | same |
| `snu-mllab/KVzip` checkout | commit `5d8472975d286e3da6b52178af39d18904330c0d` | `KV_SCORING_KVZIP_REPO`; NIAH generator and its essay files |
| Fast KVzip gate (host-only oracle) | `Jang-Hyun/Fast-KVzip` @ `67ce1265f510a946bf22486a30269f99066b3cbb`, `qwen3-8b/q4_dim16_sink16.pt` | only for the qualification oracle; `scripts/prepare_oracle_assets.py` downloads it and asserts three digests (torch 2.8 required) |
| base image | `pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel` | both Dockerfiles |

There is no owner-only input. Prove the identities before building:

```bash
"$KV_SCORING_DATA_PYTHON" scripts/prepare_items.py verify --hf-hub "$KV_SCORING_HF_HUB" \
  --model-dir "$KV_SCORING_MODEL_REPO/snapshots/b968826d9c46dd6066d109eabc6255188de91218" \
  --kvzip-repo "$KV_SCORING_KVZIP_REPO"
```

A different result means a source moved or a rule changed: a **new suite
identity**, not corruption.

## 4. What is frozen

The repository holds the public rows, the holdout **manifest** (source,
revision, row and SHA-256 per row) and the private needles, never the holdout
text. `scripts/build_images.sh` runs `prepare_items.py materialize`, compares
the rebuild with the manifest and stages `final_items.jsonl.gz` 0600 into a
build context that only the Judge image reads. `scripts/validate_layout.py`
proves no Work build context admits the manifest or the needles and that no
Work-visible file contains a needle sentence.

## 5. Build

```bash
export KV_SCORING_MODEL_REPO=<local Qwen3-8B HF repo>
export KV_SCORING_HF_HUB=<hub cache with the four datasets>
export KV_SCORING_KVZIP_REPO=<KVzip checkout at 5d84729>
export KV_SCORING_DATA_PYTHON=<python with pyarrow + transformers>   # optional
bash scripts/build_images.sh
```

The build checks 1,812 rows and writes
`<repo>/outputs/kvzip-query-agnostic-kv-scoring-qwen3-8b-h100-v2-scratch/build/image-manifest.json`;
`scripts/eval_archive.sh` refuses to run when the source or the image tags
moved after that build.

## 6. Verification

CPU only:

```bash
.venv/bin/python scripts/validate_layout.py .
.venv/bin/python -m pytest -q tests
.venv/bin/python scripts/refresh_portable_manifests.py --check
```

With GPUs, the host-only oracle (Fast KVzip gate port):

```bash
PUBLIC_GPU=<i> JUDGE_GPUS=<j,k,l> bash scripts/smoke_test.sh --oracle
```

It prepares the oracle asset if absent (download plus ~10 min CPU), runs the
public quick suite in the Work image on `PUBLIC_GPU`, then evaluates the
oracle archive on the full holdout through `scripts/eval_archive.sh` on
`JUDGE_GPUS`.

## 7. Formal runs

Scored submissions and the final both run the full holdout
(full-final-shared-v2). Formal agent runs go through `scripts/run_agent.sh`
(launcher `<repo>/launch/kvzip-kv-scoring/run.sh`): one Work GPU plus exactly 7
Judge GPUs (the paper's setup; the launcher refuses another count), with every interactive Judge and the independent final Judge
running the complete 1,812-row suite sharded over the Judge GPUs. There is no
single-GPU formal path.

## 8. What to re-measure on another GPU

Device memory peak per Judge GPU, Phase A / Phase B wall time per shard, the
suite overhead ratio of a reference scorer, and scratch usage. Row results do
not depend on the number of Judge GPUs; they may differ slightly across GPU
models, which is why the Judge requires identical devices.
