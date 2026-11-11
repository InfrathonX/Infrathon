# Task scripts

`validate_layout.py` and `pytest -q tests` are CPU-only.

Build order:

1. `stage_assets.py` stages weights, clips and embeddings into the asset root
   (`SDV2_ASSETS_ROOT`, default `build/assets`). The Judge-only inputs (final manifest,
   hidden embeddings, final goldens) come from the dataset: `assets/fetch.sh`, then
   `INFRATHON_ASSETS_ROOT` points `build_images.sh` at them.
2. `write_suite_manifests.py` writes the public manifest, the Judge-only final
   manifest (`build/private/final_cases.json`, mode 0600) and every file that binds
   the formal suite's SHA-256 (`runtime/evaluation_protocol.json`, `portable/*.json`).
3. `generate_goldens.py` runs the single-GPU frozen reference under an explicit
   `CUDA_VISIBLE_DEVICES` lock. Goldens are diagnostic; the Judge's parity oracle is the
   seed measured in the same run. Each golden manifest also records the one-GPU
   reference timing.
4. `generate_protected_hashes.py` after any runtime, seed, manifest or golden change.
5. `build_images.sh` builds the Work and Judge images from the task's runtime, final
   manifest and goldens plus the asset-root contexts.

`smoke_test.sh --image` and `eval_archive.sh` need `SDV2_GPU_INDICES=A,B` (an NVLink pair
the operator owns; hold both `/tmp/sforge-gpu-<uuid>.lock` and
`/tmp/mlsysforge-gpu-<uuid>.lock` for the whole run).

Agent runs: `launch_fixed_budget_after_gpu_pair_idle.sh` runs Work on `SDV2_GPU_INDICES` and
the Judge where `SDV2_JUDGE_GPU_INDICES` says: a second NVLink pair `C,D` (separate mode, the
default of `launch/sdv2-pipeline/run.sh`) or `shared` (one pair, Work paused for each Judge
round). It refuses to start when `SDV2_JUDGE_GPU_INDICES` is unset. In separate mode it locks,
probes and preflights all four GPUs for the whole run (both lock names, so a single-GPU
launcher cannot take the Judge pair while it is idle between Judge rounds), starts `serve` and `run` through
`separate_judge_entry.py`, and runs the final `eval_archive.sh` on the Judge pair.

`separate_judge_entry.py` is host-only and is not in either image. It loads
`sforge_dual_h100_entry.py` by path and leaves it byte-identical, so shared-mode runs,
`eval_archive.sh` and candidate replay keep the same evaluator identity. It changes two things:
Judge containers get the Judge pair (Work keeps `SDV2_GPU_INDICES`), and a managed Judge round
takes a lock keyed by the Judge pair and waits for that pair to be idle (same timeout and
`gpu_busy` error) without looking up, pausing or resuming Work. `eval_archive.sh` and `replay_candidates.sh` have no Work
container and always use the one pair in `SDV2_GPU_INDICES`.
