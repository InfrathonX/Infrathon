# Strongest admissible upstream oracle: official adapter + fa3 epilogue + H100 retune

This directory holds the fastest **admissible** host-side oracle measured for this
task: `artifacts/official-oracle-fa3epi-h100.tar.gz`. It is an ordinary candidate
archive. It passes the Judge's source policy and every gate, it runs through the
normal `scripts/eval_archive.sh`, and its score is its own latency. It never
becomes a timing denominator. Nothing here is a Docker build input.

`official_oracle/` (official checkout `12ed49b`, `artifacts/official-oracle.tar.gz`)
stays as the byte-exact **official repository** reference. This variant is a
separate candidate identity and must be reported under its own name.

## What it is

`generated_task_adapter.py` (SHA-256
`849f8a1dc585c13442ddd62f14c5fc50d9ffb3576945569281443add08424dbe`) is built in
three layers:

1. **Official HEAD adapter.** `scripts/build_official_repo_head_archive.py` is run
   on https://github.com/svg-project/flash-kmeans at
   `b2c1c83b49301dda4807990d0a90ae4bf9cdd3b0` (main HEAD as of 2026-10-29, committed
   2026-08-28), so it gets the
   same mechanical vendoring and task-ABI wrappers as `official_oracle/`. Source
   SHA-256: `df209764750a95f5d77bd356abc7b95d7ad949aecc9f35e66e9f147c1663ea54`.
2. **`fa3_epilogue.patch`.** This is upstream-authored code from the **unmerged**
   `fa3` branch, commit `39be82a52e66964aaea816c7740db5526bb9fc90`
   ("[feat] Remove x^2 compute", 2026-05-13). In both Triton Euclidean-assign
   kernels, the distance epilogue drops the `x_sq` load and the `max(.,0)` clamp
   and uses the shifted distance `||c||^2 - 2 x.c`. The argmin is unchanged
   because `x_sq` is constant per row. Only that hunk is taken from the branch.
   The branch's CuTe-DSL path and its other changes are not. The H100 table is
   identical on `fa3` and `main`. Source SHA-256:
   `f1af3b86ed34eed9f69d7545d6fc85a85b48cdb14be6ad16f2b4197f57631394`.
3. **`h100_retune.patch`.** This one is **ours, not upstream's**. It re-tunes two
   rules of upstream's H100 small-D heuristic table on H100 NVL, choosing what
   upstream's own `triton.autotune` over its own `_TUNE_CONFIGS` grid picks:
   - for D == 128 and K <= 65536, it uses BLOCK_K=128, num_warps=4,
     num_stages=2;
   - it no longer applies the `N < 65536 -> BLOCK_N = 64` rule when D >= 512.

   Kernels, epilogue, the update path and the wrappers are untouched.

The task ABI is preserved exactly as in `official_oracle/`: the
`assign(x, centroids, x_sq)` and `update(x, labels, old_centroids)` wrappers
(sorted centroid update, exact int32 counts, retain-old for empty clusters), and
the fixed `232448`-byte H100 shared-memory budget in place of upstream's dynamic
`getattr` helper. The archive still receives `x_sq`; the fa3 epilogue just does
not read it.

## Build

```bash
scripts/make_fa3epi_h100_oracle_archive.sh
# -> artifacts/official-oracle-fa3epi.tar.gz       (layers 1+2: upstream code only)
# -> artifacts/official-oracle-fa3epi-h100.tar.gz  (layers 1+2+3: strongest)
```

Without an upstream checkout, the script packages the committed adapter and
derives the layer-1+2 source by reverse-applying `h100_retune.patch`. Both
SHA-256 values are checked against the measured sources. To re-derive
everything from upstream instead:

```bash
git clone https://github.com/svg-project/flash-kmeans /path/to/fk-head
git -C /path/to/fk-head checkout b2c1c83b49301dda4807990d0a90ae4bf9cdd3b0
FLASH_KMEANS_UPSTREAM_HEAD=/path/to/fk-head scripts/make_fa3epi_h100_oracle_archive.sh
```

This mode fails unless the HEAD adapter, then HEAD + `fa3_epilogue.patch`, then
HEAD + both patches reproduce the recorded SHA-256 values byte for byte. The
archives are deterministic: tar mtime 0 and a gzip header without a timestamp.
`scripts/make_host_calibration_archives.sh` (and so `scripts/make_oracle_archive.sh`)
also builds them.

## Measured (full Judge, H100 NVL 94 GB, score = latency in ms, lower is better)

All runs are from the dev-tree Judge image
`flash_kmeans.judge.flash_kmeans_paper_h100_kernel_lab_v7_scratch:local`
(`3c8e9e6366bb`), with `SFORGE_LLM_JUDGE_MODE=off` and pinned CPUs. Runs on the
same GPU were back-to-back.

| candidate (kernel source SHA-256) | GPU 4, 2026-10-29 19:50-19:59 | GPU 0, 2026-10-29 12:05-13:12 |
|---|---|---|
| official 12ed49b (`70af9175`), `official-oracle.tar.gz` | 34.66 (`oracleup-flash-kmeans-official12ed49b_r1`) | 38.03, 35.29 (`pvsa-flash-kmeans-oracle-20261029-120542`, `pvsa-flash-kmeans-oracle2-20261029-123038`) |
| official HEAD b2c1c83 (`df209764`) | 34.60 (`oracleup-flash-kmeans-head_b2c1c83_r1`) | 35.14, 35.21 (`pvsa-flash-kmeans-paper-20261029-120542`, `pvsa-flash-kmeans-paper2-20261029-120542`) |
| HEAD + H100 retune only (`c7172146`) | not run | 32.12, 32.11 (`pvsa-flash-kmeans-paper_h100tuned-20261029-130324`, `...paper_h100tuned2-20261029-130324`) |
| HEAD + fa3 epilogue (`f1af3b86`), `official-oracle-fa3epi.tar.gz` | 30.53 (`oracleup-flash-kmeans-fa3epi_r1`) | 31.25, 31.34 (`pvsa-flash-kmeans-paper_fa3epi-20261029-120542`, `...paper_fa3epi2-20261029-120542`) |
| **HEAD + fa3 epilogue + H100 retune (`849f8a1d`), `official-oracle-fa3epi-h100.tar.gz`** | **30.02, 29.94** (`oracleup-flash-kmeans-fa3epi_h100_r1`, `oracleup-flash-kmeans-fa3epi_h100_r2`) | 30.68, 30.72 (`pvsa-flash-kmeans-paper_fa3epi_h100tuned-20261029-130324`, `...paper_fa3epi_h100tuned2-20261029-130324`) |

Every run listed is `valid: true`. Upstream's autotune path, used instead of the
H100 table, was INVALID: it failed `memory_ok` and `deployment_protocol_ok`,
because fresh-process rows autotune about 100 configs inside the first call
(`pvsa-flash-kmeans-paper_autotune-20261029-123038`). The GPU 4 rows came from
the archives that `scripts/make_fa3epi_h100_oracle_archive.sh` and
`scripts/build_official_repo_head_archive.py` produce. The GPU 0 rows came from
earlier archives with byte-identical kernel sources; only `notes.md` differs.

Caveats:

- The epilogue is upstream code, but it is not on upstream `main`. If upstream
  merges the `fa3` branch differently, layer 2 should be re-derived.
- The retune is fitted to H100 NVL by us. It is worth about 1.5-2% over layer
  1+2 (30.53 -> 30.0). Report `official-oracle-fa3epi.tar.gz` when an
  upstream-code-only number is wanted.
- Upstream's fastest path, the CuTe-DSL kernels on the `fa3` branch, is not
  admissible: the task's source policy allows only torch/triton imports.
