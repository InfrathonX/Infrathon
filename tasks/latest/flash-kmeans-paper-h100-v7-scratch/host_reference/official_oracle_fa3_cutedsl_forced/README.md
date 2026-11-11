# Forced strongest oracle: upstream fa3-branch CuTe DSL assign (not admissible)

This directory holds the **strongest paper/repo solution measured for this
task**. It is upstream flash-kmeans' unmerged `fa3`-branch CuTe DSL (SM90 TMA +
WGMMA) assignment kernel, scored by the task's **real Judge pipeline** in a
local-only derived Judge image whose source admission is bypassed. It cannot
pass the normal Judge, because it imports `flash_kmeans` and `cutlass`, which
the clean-room policy forbids. Oracles are not bound by that rule.

The best **admissible** oracle is unchanged:
`../official_oracle_fa3epi_h100/` (`artifacts/official-oracle-fa3epi-h100.tar.gz`,
about 30.0 ms). Anything that must pass the normal Judge uses that one, and
`official_oracle/` stays the byte-exact official reference.

| role | archive / candidate | Judge score, ms (lower is better), GPU 4 | gates |
| --- | --- | --- | --- |
| official reference (12ed49b) | `official-oracle.tar.gz` | 34.66 (round 1, `oracleup-flash-kmeans-official12ed49b_r1`) | all pass |
| best admissible | `official-oracle-fa3epi-h100.tar.gz` | 30.01 / 30.01 (this session, `forced-flash-kmeans-control-fa3epi_h100-r1-gpu4-20261029-215115`, `-r2-gpu4-20261029-220419`); 30.02 / 29.94 in round 1 | all pass, normal Judge |
| **forced strongest** (`FINAL_VERSION` = 3) | `kernels/flash_kmeans_v003.py` | **26.42** / 26.55 / 26.46 (`forced-flash-kmeans-v003-r1-gpu4-20261029-213933`, `-v003-r2-gpu4-20261029-215218`, `-v003-r3-gpu4-20261029-220532`) | source admission **bypassed**; every other gate passes |

## Candidates (`kernels/`, byte-identical to what was judged)

| file | sha256 | what | result |
| --- | --- | --- | --- |
| `flash_kmeans_v001.py` | `693bd947...` | pure upstream fa3: `cutedsl_assign_euclid` (upstream default `autotune=True`) + fa3 `triton_lloyd_centroid_step_euclid` update, i.e. the body of upstream's own `cutedsl_kmeans_Euclid` loop | **INVALID** (`forced-flash-kmeans-v001-r1-gpu4-20261029-215431`): the same two failures as v002; its first calls spend 31-51 s autotuning |
| `flash_kmeans_v002.py` | `2febe4c9...` | the same with `autotune=False` (static `_pick_tile`) | **INVALID** (`forced-flash-kmeans-v002-r1-gpu4-20261029-213559`): `correctness` and `deployment_protocol_ok` failed, see below |
| **`flash_kmeans_v003.py`** | `a9493bbf...` | the round-1 strongest admissible adapter (`../official_oracle_fa3epi_h100/generated_task_adapter.py`, sha256 `849f8a1d`), byte for byte, with `assign()` dispatching upstream's CuTe DSL regime (B == 1, fp16/bf16, D % 16 == 0, D <= 512, not D <= 16 and K <= 16; that is upstream's own fallback rule) to `cutedsl_assign_euclid(autotune=False)`. Everything else goes to the adapter's Triton kernels, and `update()` is the adapter's. | **valid, 26.42 ms** |
| `flash_kmeans_v004.py` | `0d5a3c76...` | v003 with `autotune=True` (upstream's default tile selection) | valid, 26.60 ms (`forced-flash-kmeans-v004-r1-gpu4-20261029-214125`). It is slower than v003, and its fresh-process first call takes up to 124 s of autotuning against the 180 s deployment timeout. |

Full sha256 values: v001 `693bd9473c8052e6ec20b9a74ce5fa2baeaee471e6b063b9681752ff4bab0a32`,
v002 `2febe4c9674215417185dd9725fae35d79d400e9b133106b699357ac09d51d4d`,
v003 `a9493bbf47745937407f8fbed1d7743be0c45e3eb1de8b9dda0adba0681826b9`,
v004 `0d5a3c762b73aa31c42779d1af6ddd917c4ef807c9d4c90d807cb348f8decc07`.

### Why pure upstream fa3 (v002) is invalid: two upstream limitations

1. The `fa3` branch forked from `e2fe09a`, before `main` gained
   non-power-of-two and split-D support. Its Triton fallback assign and its
   sorted centroid-update kernel use `tl.arange(0, D)`. That fails to compile
   (`CompilationError` at `offs_d = tl.arange(0, D)`) for the Judge's gate
   cases with D = 33, 127, 7, 129, 31 and 65 (`hidden_exact_tie_tile_boundaries`,
   `hidden_non_power_of_two_full`, `hidden_single_cluster_full`,
   `hidden_hot_empty_update`, `hidden_zipf_update`, `hidden_batched_skew_update`).
2. The fa3 fused finalize (`triton_centroid_finalize`) launches a `(B, K)`
   grid, and CUDA grid dimension 1 is at most 65535. All three K = 65536
   deployment cases therefore fail with `Triton Error [CUDA]: invalid
   argument` (reproduced outside the Judge).

v002's scored cases that ran were correct and fast: large-N/high-D deployment
179.2 / 363.3 ms, against 216.0 / 434.6 ms for the admissible control. v003
keeps exactly that CuTe kernel and routes everything the fa3 branch cannot do
to `main`-HEAD code that is already in the admissible oracle.

### The glue

- Upstream's module-level `_dlpack_cache` in `cutedsl_impl` is keyed by
  `data_ptr` and never evicted, so it keeps every caller tensor alive.
  Measured outside the Judge: memory grows by 3x the input size after 3
  fresh-input calls. Under the Judge's fresh-input protocol that is an
  unbounded leak, so every candidate clears it after each assign call. No
  kernel, tile or launch changes.
- `x_sq` is not used: the CuTe kernel drops the per-row constant `||x||^2`,
  the same epilogue as `../official_oracle_fa3epi_h100/`.

## Provenance

- Upstream: https://github.com/svg-project/flash-kmeans, branch `fa3`, commit
  `39be82a52e66964aaea816c7740db5526bb9fc90` ("[feat] Remove x^2 compute",
  2026-05-13; unmerged). Kernel: `flash_kmeans/_cutedsl_assign_kernel.py`
  (`HopperFlashKmeansAssign`) via `flash_kmeans/cutedsl_impl.py`. It is
  installed from a named build context; upstream is not vendored.
- CuTe DSL: `nvidia-cutlass-dsl==4.5.0` (2026-05-06), the release current
  when the branch was written. Upstream names "cutlass-dsl 4.4.x / 4.5.x". It
  pulls `cuda-python` 13.4.4. The kernels JIT-compile for the visible GPU
  (sm_90a) inside the candidate process. That compile is part of the
  fresh-process first call, which the Judge records with weight 0: 3.5-5.8 s
  for v003.
- The Triton part of v003 is the round-1 adapter: official HEAD `b2c1c83` +
  fa3 epilogue `39be82a` + our H100 retune. See `../official_oracle_fa3epi_h100/README.md`.

## Forced Judge (`forced_judge/`)

| step | in this run |
| --- | --- |
| LLM integrity review | **off** (`SFORGE_LLM_JUDGE_MODE=off`) |
| source admission (`/opt/flash_kmeans_benchmark/source_admission.py::scan_submission`) | **bypassed**. `admission_override.py` keeps the original problems and forces `status: clean` when `FINAL_VERSION` selects an existing kernel. `protected_hashes.txt` is updated for the patched file. The original problem for v003, from the unmodified Judge: `kernels/flash_kmeans_v003.py:1966: prohibited import: flash_kmeans`. Each run stores it as `original_source_admission.json`. |
| everything else: protected files, interface, correctness (exact labels and ties, counts, centroids), input integrity, memory bounds, timing protocol, fresh-process deployment protocol (empty Triton cache, fresh `TMPDIR` per child), aggregation | unchanged |
| docker run | identical to `scripts/eval_archive.sh` (`sforge eval`, 8 CPUs / 32 GB, the resource profile, the per-GPU Judge lock, the idle check, `NVIDIA_VISIBLE_DEVICES` = GPU UUID, `FLASH_KMEANS_TARGET=H100`). Only the image differs: a scratch task JSON selects tag `forced-oracle`. |

The task has no compute-sanitizer stage. The image is
`flash_kmeans.judge.flash_kmeans_paper_h100_kernel_lab_v7_scratch:forced-oracle`,
built `FROM` the private `:local` Judge image. The script checks the layer
lineage, never retags `:local` and never pushes.

```bash
# Prerequisite: the private Judge image (scripts/build_images.sh).
git clone https://github.com/svg-project/flash-kmeans /path/to/flash-kmeans
git -C /path/to/flash-kmeans checkout 39be82a52e66964aaea816c7740db5526bb9fc90
FK_SOURCE=/path/to/flash-kmeans \
  host_reference/official_oracle_fa3_cutedsl_forced/forced_judge/run_forced_oracle.sh \
    --gpu 4 --cpuset 16-19,48-51 --version 3
# Packs FINAL_VERSION=V, kernels/flash_kmeans_vVVV.py and notes.md (this
# directory's) into artifacts/forced_oracle/official-oracle-fa3-cutedsl-vVVV.tar.gz,
# records the original admission, then runs sforge eval.
# Options: --skip-build, --run-id ID; env CUTLASS_DSL_VERSION, LOGS_DIR, SFORGE_BIN.
```

The recorded runs used the dev-tree SForge (`SFORGE_ROOT=<mlsys_forge checkout>`),
on whose Judge build the `:local` images were recorded, and an upstream
`git archive` of the pinned commit as `FK_SOURCE`.

## Measured (H100 NVL 94 GB, GPU 4, Judge cpuset 16-19,48-51, 2026-10-29 UTC)

Group latencies are in ms. "control" is the fresh-process deployment control
row.

| run id | candidate | valid | score | assign | update | full | control | large-K | large-N/high-D |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `forced-flash-kmeans-v003-r1-gpu4-20261029-213933` | v003 | yes | **26.424** | 4.402 | 6.137 | 4.605 | 1.177 | 142.43 | 252.85 |
| `forced-flash-kmeans-v003-r2-gpu4-20261029-215218` | v003 | yes | 26.551 | 4.393 | 6.129 | 4.603 | 1.213 | 142.99 | 254.08 |
| `forced-flash-kmeans-v003-r3-gpu4-20261029-220532` | v003 | yes | 26.459 | 4.439 | 6.140 | 4.595 | 1.173 | 142.76 | 253.46 |
| `forced-flash-kmeans-v004-r1-gpu4-20261029-214125` | v004 (autotune) | yes | 26.598 | 4.423 | 6.152 | 4.597 | 1.147 | 144.35 | 258.32 |
| `forced-flash-kmeans-v002-r1-gpu4-20261029-213559` | v002 (pure fa3, static) | **no** | - | - | - | - | 1.158 | failed | 255.15 |
| `forced-flash-kmeans-v001-r1-gpu4-20261029-215431` | v001 (pure fa3, autotune) | **no** | - | - | - | - | 1.128 | failed | 256.46 |
| `forced-flash-kmeans-control-fa3epi_h100-r1-gpu4-20261029-215115` | admissible control, normal Judge | yes | 30.006 | 4.758 | 6.111 | 4.976 | 1.285 | 172.19 | 306.43 |
| `forced-flash-kmeans-control-fa3epi_h100-r2-gpu4-20261029-220419` | admissible control, normal Judge | yes | 30.009 | 4.752 | 6.095 | 5.001 | 1.285 | 171.72 | 306.63 |

For v001 and v002 the report also shows `input_integrity_ok` and `memory_ok`
as ERROR. Those are consequences of the failed cases, which never produce a
row; every case that ran passed them. `timing_protocol_ok` passed.

The v003 r2 run built the forced image with the committed
`run_forced_oracle.sh` (no `--skip-build`; image id `sha256:7953e7ff...`). r3
judged `artifacts/forced_oracle/official-oracle-fa3-cutedsl-v003.tar.gz`,
sha256 `9e3217d15fb055598783a6ed1a8465d0160a896dcbc4d70417977edf93ce4a2e`,
byte-identical to the run's `submission.tar.gz`. The admissible control runs
used the normal `scripts/eval_archive.sh` and `:local` image on the same GPU
in the same session.

v003 is 12% faster than the best admissible oracle on the same card (26.42-26.55 vs 30.01 ms). The gain
is all in B = 1 assignment: large-K -17%, large-N/high-D -17%, assign -7%,
full -7%. Update is the same code and the same time. Every v003 row passed
correctness, input integrity and memory: the max memory ratio is 0.797, the
same as the control.
