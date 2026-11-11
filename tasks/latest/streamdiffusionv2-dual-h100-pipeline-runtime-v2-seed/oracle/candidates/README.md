# Oracle candidates

- `runtime_v001/`: the source-preserving task seed, identical to the Work image seed. The
  Judge image protects and re-measures it in every run (`seed_runtime`).
- `runtime_v002/`: the seed with the paper's block balancing as a static per-step split
  (`split_by_steps`). `oracle/FINAL_VERSION` selects it.
