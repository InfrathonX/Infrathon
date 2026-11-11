# Host-only external comparison sources

Nothing under this directory is a Docker build input.

- `baseline/torch_fallback.py` is an exact copy of the repository's
  `flash_kmeans/torch_fallback.py`, SHA-256
  `8a10450f701f1329fe14bb57de7cac57c28fd5bfb502815ad6085ffc92b8d20a`.
- `official_oracle/assign_euclid_triton.py` and
  `official_oracle/centroid_update_triton.py` are exact copies from official
  checkout `12ed49b2eddc207dfc5ed27dba244af2365bd9c6`.
- `fast_pytorch_kmeans/` is a complete upstream checkout of
  `https://github.com/DeMoriarty/fast_pytorch_kmeans` at commit
  `aec7cd96a10f7cfe49f8cc4668b3729e3534e1de`. Its generated task adapter
  preserves the upstream Euclidean assignment and dense membership-matrix
  update, while adding only the batched task ABI, exact counts, and the
  task-required retain-old behavior for empty clusters.
- `fastkmeans/` is a complete upstream checkout of
  `https://github.com/AnswerDotAI/fastkmeans` at commit
  `596bff0a2f91df998d30bd3a5e607c237dfa251c`. Its task adapter uses the
  upstream Triton assignment kernel selected by default on H100 and the
  `kmeans.py` FP32 `index_add_` centroid update, with only the task ABI
  bridges documented in the generated archive.

`scripts/make_host_calibration_archives.sh` writes submission archives under
`artifacts/`. For the fallback, it mechanically extracts the exact upstream
definitions of `euclid_assign_torch_native_chunked` and
`_centroid_update_torch_native`, then appends only the task ABI. The executable
self-test and unrelated full K-means driver are deliberately excluded because
they are outside the two-function task and contain source-policy-forbidden
side effects. The byte-exact upstream file remains unchanged here.

For the official adapter, the script mechanically vendors the upstream
Euclidean assignment and sorted centroid-update kernel/tuning bodies, removes
only executable self-test sections and incompatible unused imports, and
appends the task ABI. Its dynamic device-property compatibility helper is
replaced by the equivalent H100 opt-in shared-memory value (`232448` bytes)
because candidate policy disallows dynamic `getattr`; the Triton kernels,
launch/configuration logic, and H100 behavior are unchanged.

`official_oracle_ampere_compat/` holds the minimal upstream patch and notes for
a pre-Hopper compatibility variant of the same adapter; it is built by
`scripts/make_ampere_compat_oracle_archive.sh` as
`artifacts/official-oracle-ampere-compat.tar.gz`, a separate candidate identity
for GPUs whose names the upstream kernel heuristics do not recognize.

`official_oracle_fa3epi_h100/` holds the strongest admissible upstream-derived
oracle measured for this task: the official adapter built from repository HEAD
`b2c1c83b49301dda4807990d0a90ae4bf9cdd3b0`, plus the upstream-authored
no-`x_sq` distance epilogue from the unmerged `fa3` branch
(`39be82a52e66964aaea816c7740db5526bb9fc90`), plus an H100 NVL retune of two
heuristic-table rules (ours). `scripts/make_fa3epi_h100_oracle_archive.sh`
builds it as `artifacts/official-oracle-fa3epi-h100.tar.gz`, and the
upstream-code-only layer as `artifacts/official-oracle-fa3epi.tar.gz`. On the
same H100 NVL it scores about 30.0 ms, against about 34.6 ms for
`official-oracle.tar.gz`. These are separate candidate identities; see that
directory's README for provenance, run ids and caveats.

`official_oracle_fa3_cutedsl_forced/` holds the **forced strongest** paper
oracle. It is not admissible: it imports upstream's `flash_kmeans` (unmerged
`fa3` branch `39be82a`) and NVIDIA CuTe DSL. `kernels/flash_kmeans_v003.py`
(selected by that directory's `FINAL_VERSION`) is the
`official_oracle_fa3epi_h100` adapter byte for byte, plus B = 1 assignment on
upstream's CuTe DSL SM90 TMA + WGMMA kernel. Its `forced_judge/` scores it
with the full Judge pipeline in a local-only image whose source admission is
bypassed: 26.4 ms, all other gates passing, against 30.0 ms for the
admissible oracle on the same GPU. Pure upstream `fa3` (no `main`-HEAD
fallback) is invalid on this task's gates (non-power-of-two D, K = 65536). See
that directory's README. The admissible oracle above remains the one for
anything that must pass the normal Judge.

The archives are passed to the Judge one at a time as ordinary candidate
submissions; neither source nor generated adapter is copied into the Work or
Judge image. A host-side comparison may be made only after each archive has
received its own candidate-only latency result. No host implementation becomes
a timing denominator or otherwise changes another submission's score.

The `official_oracle/` directory name is retained as artifact provenance; that
implementation is not a performance oracle inside the Judge. The Judge keeps
only its private correctness reference, which supplies expected outputs outside
all timed intervals and supplies no score or timing denominator.
