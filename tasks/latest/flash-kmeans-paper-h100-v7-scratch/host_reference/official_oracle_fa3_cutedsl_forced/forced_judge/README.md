# forced_judge/: local-only forced-oracle Judge image for flash-kmeans

| file | role |
| --- | --- |
| `Dockerfile` | `FROM` the private `:local` Judge image. It adds `nvidia-cutlass-dsl` (default 4.5.0) and upstream flash-kmeans `fa3` @ `39be82a52e66964aaea816c7740db5526bb9fc90` from the named build context `fk_src`, then applies `admission_override.py`. |
| `admission_override.py` | wraps `scan_submission`: it records the problems, forces `status: clean` when FINAL_VERSION selects an existing kernel, and updates `protected_hashes.txt` |
| `run_forced_oracle.sh` | builds the image (tag `:forced-oracle`) and packs `../kernels/flash_kmeans_vVVV.py` + `../notes.md` + FINAL_VERSION. It records the unmodified admission result, then runs `sforge eval` exactly as `scripts/eval_archive.sh` does, with a scratch task JSON that selects the forced tag. |

Provenance, the exact command, the bypassed gates and the measured numbers are
in `../README.md`. This image is never published, never tagged `:local` and
never pushed, and the published Work and Judge images do not use it.
