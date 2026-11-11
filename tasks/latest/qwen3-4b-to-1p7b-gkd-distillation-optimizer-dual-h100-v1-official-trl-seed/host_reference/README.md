# Host-only treatment arms

Nothing under this directory enters the Work or Judge image, and no result
measured here enters participant scoring. These arms show attainable headroom
and exercise the treatment plumbing.

## `gkd_oracle/` — objective-only chunked control

`solutions/objective_v001/candidate.py` is a Liger-style chunked GKD objective
with TRL-exact mathematics: it streams row chunks, projects both vocabularies
per chunk, computes the official mixture/KL weighting (which differs from the
stock `LigerFusedLinearJSDLoss` for `beta != 0.5`), and accumulates student
gradients during the forward pass so no full logits tensor or chunk graph is
ever retained.

It matches the pinned official reference on CPU for the loss, all nine
auxiliary witnesses, and student weight/hidden gradients at
`beta in {0, 0.3, 0.5, 1}` x `temperature in {0.7, 0.9, 1.0, 1.5}`, including
multi-chunk shapes (2,800 rows over a 1,024-row chunk).

## `full_liger_gkd/`: full-Liger forced arm

This arm keeps the same objective, unchanged, and adds the pinned Liger-Kernel
Qwen3 RoPE, RMSNorm-forward and SwiGLU-forward callables
(`780e76b8115bfd7b78592569853b245a33cc7bf0`). They are declared through
`build_model_patch` (student) and `build_teacher_patch` (frozen teacher), and
the task runtime performs every attribute write. Liger CE and fused-linear CE
stay off because the objective owns both projections. The arm imports
`liger_kernel`, so participant admission rejects it. It is measured only in
the local-only forced-oracle Judge (`forced_judge/`).

## Oracle (paper/repo solution) on the formal Judge

`FINAL_ORACLE` names the strongest paper/repo arm. `ADMISSIBLE_ORACLE` names
the strongest arm that passes the normal, admission-enforcing Judge; use it
whenever a check needs an archive the participant Judge accepts. Each arm's
own `FINAL_VERSION` selects the version. Oracles are exempt from the
participant rules: the source-admission allowlist and the LLM integrity
review. Exact commands, image ids and bypassed gates are in
`forced_judge/README.md`.

| Candidate | Admissible | Formal Judge score | Gates | Run id |
| --- | --- | ---: | --- | --- |
| seed `objective_v001` (official TRL 0.12.1) | yes | 99.53 to 100.07 (managed smokes 99.59, 99.92) | all PASS | dev-tree seed runs, `qwen3-gkd-v1-*managed-seed*` |
| `gkd_oracle` (Liger-style chunked objective, TRL-exact), `ADMISSIBLE_ORACLE` | yes | **105.1828** | 10/10 PASS on the normal Judge | `forced-gkd-gkdoracle-admissible-r1` |
| **`full_liger_gkd` (objective + Liger student/teacher kernels), forced, `FINAL_ORACLE`** | no (`liger_kernel` import) | **117.7279 / 117.6430** | 10/10 PASS in both runs, with the source-admission scan and LLM review bypassed | `forced-gkd-fullliger-r1`, `forced-gkd-fullliger-r2` |

Row median ratios (S512-B2, S1024-B2, S2048-B2, S512-B4, S512-B8):

- `gkd_oracle`: 0.952, 1.066, 1.099, 1.049, 1.100
- `full_liger_gkd` (r1): 1.039, 1.136, 1.293, 1.126, 1.317
- `full_liger_gkd` (r2): 1.032, 1.135, 1.292, 1.130, 1.319

The stock `LigerFusedLinearJSDLoss` cannot be scored. Its mixture differs from
TRL whenever `beta != 0.5`, and it returns no auxiliaries, so it fails the
hidden CPU objective cases. Details are in `forced_judge/README.md`.
