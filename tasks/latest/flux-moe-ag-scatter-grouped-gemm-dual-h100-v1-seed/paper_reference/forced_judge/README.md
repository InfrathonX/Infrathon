# Forced-oracle Judge: ByteDance Flux CUTLASS AG+scatter grouped GEMM

LOCAL-ONLY tooling to get the real Judge result of the strongest official
paper implementation, which the agent rules (Triton-only, no compiled
extensions, no `torch.distributed`) do not admit. Agents are always judged by
the unmodified `:local` image; this never retags or pushes it.

## Provenance

- Upstream: https://github.com/bytedance/flux @ `19831ca2d820e3e782ed1d15d8b52d0898b78b26`
  (Apache-2.0), submodules `3rdparty/cutlass` @ `df8a550d3917b0e97f416b2ed8c2d786f7f686a3`,
  `3rdparty/nccl` @ `8c6c5951854a57ba90c4424fa040497f6defac46`.
- Operator: `flux.GemmGroupedV3AGScatter` (`src/moe_ag_scatter/`, the Hopper MoE
  layer-0 op of FLUX arXiv:2406.06858 / Comet arXiv:2502.19811). The MoE ops are
  only compiled with NVSHMEM (`src/CMakeLists.txt`), so the build uses
  `nvidia-nvshmem-cu12==3.3.9` from pip, as upstream `build.sh` does.
- Glue: `oracle/candidates/kernel_v002/submission.py` (heuristic tiles) and
  `kernel_v003/submission.py` (Flux autotuner). Both target the v1 two-GPU ABI; the v2
  eight-GPU suite needs a W-rank port first (see `oracle/notes.md`).

## Files

| file | role |
| --- | --- |
| `Dockerfile` | stage 1 builds Flux (`build.sh --arch 90 --sm-cores 132 --nvshmem --no_test --package`) inside the task Judge image (torch 2.8.0, CUDA 12.9, Python 3.11); stage 2 = private Judge `:local` + NVSHMEM + Flux wheel + admission override |
| `source_admission_forced.py` | wraps the enforced `source_admission.py` (kept as `source_admission_enforced.py`), records its errors as `forced_bypass_errors`, admits the selected source; the rebuilt image rebinds only that file's line in `protected_hashes.txt` |
| `forced_entry.py` | runs the task's own `scripts/sforge_dual_h100_entry.py`, adding `--cpuset-cpus` for the Judge |
| `run_forced_oracle.sh` | same `sforge eval` call as `scripts/eval_archive.sh`, with a temporary task-spec copy whose `judge.image_tag` is `forced-oracle`; `FORCED_TAG=local` runs an admissible control through the normal Judge |
| `make_forced_oracle_archive.sh` | packs one oracle candidate (default `oracle/FINAL_VERSION`) |

## Build

```bash
git clone https://github.com/bytedance/flux.git "$FLUX_SRC" && git -C "$FLUX_SRC" checkout 19831ca2d820e3e782ed1d15d8b52d0898b78b26
git -C "$FLUX_SRC" submodule update --init --depth 1 3rdparty/cutlass 3rdparty/nccl
docker build --build-context flux_src="$FLUX_SRC" --build-arg FLUX_JOBS=40 \
  -t flux_moe_ag_scatter_dual_h100.judge.flux_moe_ag_scatter_grouped_gemm_dual_h100_seed_v1:forced-oracle \
  paper_reference/forced_judge
```

The CUDA build took about 15 minutes with 40 jobs. Image used for the runs:
`sha256:997afaf9b7789ec6d2f0bb16638cdc9584eca11c771cccbd7bfd9fb88ec08763`,
FROM `:local` `sha256:3a311328d681941dc53d1b143cbb4d021ddcadcb310ef753c1d89a31370f229f`.

## Run

```bash
paper_reference/forced_judge/make_forced_oracle_archive.sh /tmp/flux_oracle_v3.tar.gz 3
FLUXMOE_GPU_INDICES=6,7 FORCED_SKIP_PREFLIGHT=1 \
  MLSYSFORGE_ROOT=/path/to/sforge-checkout \
  paper_reference/forced_judge/run_forced_oracle.sh /tmp/flux_oracle_v3.tar.gz forced-flux-v3-gpu67-<stamp>
# admissible control through the normal Judge, same pipeline:
paper_reference/forced_judge/make_forced_oracle_archive.sh /tmp/flux_seed.tar.gz 1
FLUXMOE_GPU_INDICES=6,7 FORCED_SKIP_PREFLIGHT=1 FORCED_TAG=local \
  paper_reference/forced_judge/run_forced_oracle.sh /tmp/flux_seed.tar.gz forced-flux-seedctl-gpu67-<stamp>
```

`FORCED_SKIP_PREFLIGHT=1` was needed because `scripts/preflight.py` needs a CUDA
torch on the host and the dev venv has a CPU-only torch; the wrapper then only
checks that the pair is NV-linked. Each run took about 1 minute; one Flux Judge
at a time per host. Reports land under
`$MLSYSFORGE_ROOT/outputs/<task>/runs/<run id>/.../submissions/manual-1/`
(`report.json`, private `flux_moe_ag_v1_results.json` in
`private_judge_artifacts.tar.gz`).

## Results (H100 NVL pair GPUs 6,7, final holdout `2762d961`, 2026-10-29)

| candidate | runs | mean TFLOPS |
| --- | --- | --- |
| kernel_v001 seed, normal Judge | 171.65, 172.43 | 172.04 |
| kernel_v002 Flux CUTLASS heuristic, forced | 187.63, 185.08, 187.69 | 186.80 |
| kernel_v003 Flux CUTLASS autotuned, forced | 188.58, 190.37, 191.05 | 190.00 |

Run ids and per-cell numbers are in `oracle/notes.md`. Judged bytes:
kernel_v003 `submission.py` sha256 `2681a4e7d3a05efe5e2dce307316219bcac2c609c1e46a372ecc708dc57b64ce`
(archive `696cea23...`), kernel_v002 `57ae0983...` (archive `c6fd1b9b...`).

Gates bypassed: source admission (enforced verdict recorded in the private
report), LLM integrity review (`SFORGE_LLM_JUDGE_MODE=off`). Not bypassed: every
runtime gate (edges, correctness, immutability, fresh input/output, rank
agreement, timing). The task Judge has no compute-sanitizer stage.
