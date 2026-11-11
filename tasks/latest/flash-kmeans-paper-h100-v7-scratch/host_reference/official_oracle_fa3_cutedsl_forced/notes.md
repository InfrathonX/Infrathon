# Forced oracle: upstream flash-kmeans fa3 branch (CuTe DSL SM90 assign)

Host-side forced oracle, not an agent submission and not admissible: the kernel
imports the upstream `flash_kmeans` package (unmerged `fa3` branch,
39be82a52e66964aaea816c7740db5526bb9fc90) and NVIDIA CuTe DSL
(`nvidia-cutlass-dsl` 4.5.0), which the clean-room source policy forbids. It is
scored only in the local-only forced-oracle Judge image
(`host_reference/official_oracle_fa3_cutedsl_forced/forced_judge/`).
