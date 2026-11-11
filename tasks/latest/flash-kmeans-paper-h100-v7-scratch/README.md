# Flash-KMeans: one Lloyd K-Means iteration on a single GPU (scratch)

The agent writes and optimizes GPU code for one exact Euclidean K-Means (Lloyd)
iteration: a nearest-centroid assignment and a centroid update over batched FP16
data, using PyTorch and/or Triton. The task follows the Flash-KMeans paper
(arXiv:2603.09229), whose point is that the iteration is limited by data movement,
not FLOPs. This is a scratch task: the workspace starts with an empty `kernels/`
directory and no starter, reference, baseline or oracle code. The Judge scores the
latency of the agent's own implementation; lower is better.

## The task

The Work container's working directory is `/app/workdir/flash_kmeans_lab`. The
agent creates versioned implementations and selects one:

```text
kernels/flash_kmeans_v001.py   # first complete version (then v002, v003, ...)
FINAL_VERSION                  # the selected version number, e.g. "1\n"
notes.md                       # experiment notes (submitted, not scored)
```

Only `kernels/`, `FINAL_VERSION` and `notes.md` are submitted. Each version file
exports two functions:

```python
assign(x, centroids, x_sq) -> labels
update(x, labels, old_centroids) -> (counts, new_centroids)
```

| tensor | dtype | shape |
| --- | --- | --- |
| `x` | FP16 | `[B,N,D]` |
| `centroids`, `old_centroids` | FP16 | `[B,K,D]` |
| `x_sq` | FP32, equals `x.float().square().sum(-1)` | `[B,N]` |
| `labels` | CUDA integer | `[B,N]` |
| `counts` | CUDA integer | `[B,K]` |
| `new_centroids` | FP16 (accumulated in FP32) | `[B,K,D]` |

All inputs are contiguous CUDA tensors. Required behavior:

- `assign` returns the nearest centroid under squared Euclidean distance; exact
  ties select the smallest cluster id.
- `update` returns exact counts and the FP32-accumulated mean of the assigned
  points; an empty cluster keeps its old centroid exactly.
- Neither function mutates its inputs, and neither materializes the full
  `[B,N,K]` distance tensor. Extra GPU memory is bounded per stage:

  ```text
  assign: 128 MiB +  8*B*N + 8*B*K*D bytes
  update: 128 MiB + 40*B*N + 8*B*K*D bytes
  full:   128 MiB + 48*B*N + 8*B*K*D bytes
  ```

Rules the Judge enforces:

- Clean room: candidate code may import only `torch`, `triton`, `math`,
  `functools`, `typing` and `__future__`, and may not copy, vendor, install or
  call another K-Means implementation.
- No subprocesses, dynamic code loading, filesystem or network I/O, resetting
  CUDA measurement state, or modifying evaluation state from candidate code.
- Every retained `kernels/flash_kmeans_vNNN.py` is source-checked, not only the
  selected one; incomplete or abandoned versions must be removed.
- Protected workspace files (public tests, the submit hook) must be unchanged.

A required LLM integrity review (`llm_judge/POLICY.md`) runs before the GPU
Judge; it rejects changes of the problem (approximate assignment, subsampling,
lower-precision accumulation), replaying earlier outputs, or detecting public
inputs, and explicitly allows any exact reformulation, tiling, autotuning or
shape-based dispatch.

The Work image includes public tests in `/workspace/public_tests/`
(`run_public_tests.sh --quick`, the full run, and `--candidate-version N` for an
unselected version). They check correctness against literal golden fixtures and
sampled expected values, and run fresh-process probes plus large-K and
large-N/high-D scale ladders that report candidate-only latency, effective
throughput and scaling ratios. These timings are diagnostics only and produce no
score. The public tuples are not the Judge's tuples.

The launcher's default time budget is 12 hours (`TASK_TIMEOUT=43200`).

## Evaluation

The agent submits with `sforge-submit --details`. Manual submissions are
unlimited, with a 300-second cooldown; scheduled automatic evaluation is
disabled. Each Judge run has a 3600-second timeout. When Work ends, the final
archive is captured and judged independently; the run's result is the best valid
score (lowest valid latency) across submissions and this final evaluation.

The Judge evaluates the version named in `FINAL_VERSION` on a private suite of
shapes (generated synthetic inputs, no datasets). A submission is valid only if
all hard gates pass:

1. protected workspace unchanged;
2. clean-room source policy (all retained versions);
3. candidate interface;
4. functional correctness against a private reference: exact labels and ties,
   exact counts, centroids within an FP32-conditioned tolerance
   (`0.003 + 0.003*|mean|` plus FP16 rounding), empty clusters unchanged;
5. input integrity;
6. peak extra memory within the bounds above;
7. synchronized timing protocol;
8. fresh-process deployment protocol (each deployment case runs in a new
   Python process with an empty compilation cache).

For a valid submission the score is a weighted geometric mean of candidate
latency in milliseconds (minimized). Cases are first combined by geometric mean
within six groups, then the public group weights are applied:

```text
score = assign_GM^0.10
      * update_GM^0.10
      * full_GM^0.20
      * control_fresh_process^0.10
      * large_K_fresh_process_GM^0.25
      * large_N_high_D_fresh_process_GM^0.25
```

Steady-state stage cases use median latency, measured in two independent trials
with a third added when they differ by more than 25%. In fresh-process cases the
first call (import, compilation, first synchronized call) is diagnostic with
weight zero; the median warmed call with fresh inputs is scored. No baseline or
oracle is timed: the score is the candidate's own latency, comparable only
within the same measurement profile (same GPU type and clock policy). An invalid
submission gets no score.

Feedback to the agent names failed gates and stage-level diagnostics. For the
`large-K` and `large-N/high-D` families, an invalid submission receives only
`passed`, `failed` or `not_reached`; a valid one may also receive a coarse
latency band and a trend against the best earlier valid submission. Exact hidden
shapes, per-case latencies, weights, seeds and group membership are never
returned.

## Hardware

Two GPUs per run by default: Work runs on GPU 0 and the Judge on GPU 1
(`--work-gpu`, `--judge-gpu`), so the Judge measures on its own GPU while
Work keeps running. With `--shared-gpu` the Work and Judge containers share
one GPU: before each Judge run the Work container's GPU processes are cleaned
up by a root-owned submit hook and the container is paused.

| hardware type | support |
| --- | --- |
| `h100-94gb-400w` (H100 NVL 94 GB) | qualified |
| `h100-80gb-700w` (H100 SXM 80 GB) | qualified |

The execution profile `single-cuda-sm80-80g` requires one NVIDIA GPU with
compute capability 8.0 or newer and at least 79 GiB of memory. On an H100 SXM
80 GB card the full Judge peaks at about 53 GiB of device memory. Work and
Judge containers each get 8 CPUs and 32 GB of host memory. Latencies are
comparable only within one hardware type.

## Running

1. Assets: `tasks/latest/flash-kmeans-paper-h100-v7-scratch/assets/fetch.sh`
   downloads into `INFRATHON_ASSETS_ROOT` (default `~/.cache/infrathon/assets`).
   The task uses no models or datasets; the only asset is the Judge-only hidden
   case definition (`judge/hidden_cases.py`) from the dataset's `judge/` part. It is not in
   the repository; the build reads it from `INFRATHON_ASSETS_ROOT`.
2. Images: `INFRATHON_PULL=1 tasks/latest/flash-kmeans-paper-h100-v7-scratch/scripts/build_images.sh`
   pulls the published Work and public Judge images and builds the holdout layer
   (`images/judge/Dockerfile.private`) locally. Without `INFRATHON_PULL=1` it builds
   all images from source (base `pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel`).
3. Run:

   ```bash
   launch/flash-kmeans/run.sh --model <model-id> [--work-gpu N --judge-gpu M] \
     [--auth ~/.codex/auth.json] [--tmux]
   ```

   Defaults: Work on GPU 0, Judge on GPU 1, 12 hours, effort `max`, Judge port
   18284; `--work-gpu`, `--judge-gpu`, `--hours`, `--effort` and `--judge-port`
   change them. `--shared-gpu` runs Work and Judge on one GPU, chosen with
   `--gpu N` (default 0) or `--device-uuid UUID` (full UUID from `nvidia-smi -L`);
   either flag alone also selects shared mode.
   `--judge-port` must be unique among concurrent runs on a host.
   `run.sh --help` lists all options.

   Replay saved candidates of a finished run:
   `launch/flash-kmeans/replay.sh --origin-run RUN_ID --measurement-set NAME`
   (`--dry-run` previews the selection). Evaluate one submission archive:
   `JUDGE_GPU=<index> tasks/latest/flash-kmeans-paper-h100-v7-scratch/scripts/eval_archive.sh ARCHIVE.tar.gz [RUN_ID]`
   (`JUDGE_GPU` defaults to 2).

Prerequisites the launcher checks: an SForge checkout, Docker access, an agent
runtime (Codex CLI, built with `launch/build-agent-runtime.sh`), and model
credentials: a mode-0600 Codex `auth.json` (`CODEX_AUTH_JSON_PATH` or `--auth`),
or an OpenAI-compatible endpoint (`MODEL_API_BASE_URL` with
`MODEL_API_KEY_FILE`). The required LLM integrity review is configured with
`SFORGE_LLM_JUDGE_*` settings (see `launch/mlsysforge.env.example`). The Work
container reaches the network only through a restricted proxy.

## Files

| path | contents |
| --- | --- |
| `images/work/` | Work image: `AGENT_TASK.md` (the agent's brief), the empty workspace, public tests, submit hook |
| `images/judge/` | Judge image: verifier, benchmark and source-admission code; `Dockerfile.private` adds the hidden cases from the asset root |
| `judge/run_judge.py` | Judge entry point (copied into the Judge image at build time) |
| `tasks/` | SForge task spec (`flash_kmeans_paper_h100_kernel_lab_v7_scratch.json`) |
| `portable/` | portable contract: suites, oracle, calibration, software ABI and resource manifests |
| `assets/` | asset manifest and `fetch.sh` |
| `llm_judge/POLICY.md` | integrity-review policy |
| `host_reference/` | host-only external implementations (fallback, official repository kernels, fast_pytorch_kmeans, fastkmeans) that can be packaged as ordinary candidate archives for comparison; not part of any image. `official_oracle_fa3epi_h100/` is the strongest admissible oracle: the official adapter at upstream `main` commit `b2c1c83b`, plus the epilogue from upstream's unmerged `fa3` branch at commit `39be82a5`, plus an H100 retune (about 30.0 ms vs about 34.6 ms for `official_oracle/` on H100 NVL). `official_oracle_fa3_cutedsl_forced/` is the strongest paper oracle: the same adapter with B=1 assignment on the CuTe DSL SM90 kernel of that `fa3` branch commit. It is not admissible (it imports `flash_kmeans` and `cutlass`); for reference only, it was scored by the full Judge with source admission and the LLM review turned off (`forced_judge/`): 26.4 ms on H100 NVL, with all other gates passing |
| `scripts/` | image build, run, replay and evaluation scripts |
| `tests/` | unit tests for the task contract and Judge adapter |

## Upstream and license

- Paper: Flash-KMeans, [arXiv:2603.09229](https://arxiv.org/abs/2603.09229).
  Official repository [svg-project/flash-kmeans](https://github.com/svg-project/flash-kmeans),
  analyzed at commit `12ed49b2eddc207dfc5ed27dba244af2365bd9c6`. No upstream code
  is in the Work or Judge images. `host_reference/official_oracle/` holds copies
  of two upstream Triton files and `host_reference/baseline/` a copy of its
  PyTorch fallback, used only for host-side comparison.
  The upstream repository's `LICENSE` file is Apache-2.0, while its `pyproject.toml`
  classifier says MIT; see the upstream repository for its license terms.
- [DeMoriarty/fast_pytorch_kmeans](https://github.com/DeMoriarty/fast_pytorch_kmeans)
  at commit `aec7cd96a10f7cfe49f8cc4668b3729e3534e1de` (MIT), host-only.
- [AnswerDotAI/fastkmeans](https://github.com/AnswerDotAI/fastkmeans) at commit
  `596bff0a2f91df998d30bd3a5e607c237dfa251c` (Apache 2.0), host-only.
- Base image: `pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel`.

The task's own code is under the repository's root `LICENSE` (PolyForm Noncommercial 1.0.0).
