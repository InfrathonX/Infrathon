# Host-only GKD oracle: TRL-exact chunked objective (admissible)

`objective_v001` is a Liger-style fused-linear chunked GKD objective with the
exact Hugging Face TRL 0.12.1 `GKDTrainer.generalized_jsd_loss` mathematics
(mixture `beta*student + (1-beta)*teacher`, `beta*KL(teacher||M) +
(1-beta)*KL(student||M)`, temperature scaling, batchmean over scored tokens).
It streams 1,024-row chunks, projects both vocabularies per chunk and
accumulates the student gradients during the forward pass, so no full
`[BT, V]` logits tensor or chunk graph is retained.  It exports only
`build_objective` (no student or teacher patch).

It imports only `torch` and `runtime.api` and passes the participant source
admission policy unchanged; it is the admissible paper/repo-style oracle and is
evaluated by the normal `scripts/eval_archive.sh` Judge.

This archive is calibration-only and is not copied into Work or Judge.
