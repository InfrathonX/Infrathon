# Oracle

FINAL_VERSION=2 selects `candidates/runtime_v002/` (split_by_steps): the seed with the
paper's DiT block balancing as a static per-step split. `candidates/runtime_v001/` is the
lag-ring seed, identical to the Work image seed. The Judge image protects it and
re-measures it in every run. scripts/make_oracle_archive.sh packages this directory
deterministically for Judge evaluation. See notes.md for scores.
