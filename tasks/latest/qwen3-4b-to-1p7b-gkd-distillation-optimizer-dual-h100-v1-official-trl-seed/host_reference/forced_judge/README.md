# Forced-oracle Judge (local only)

`host_reference/full_liger_gkd/` is the full-Liger arm. It pairs the TRL-exact
chunked GKD objective from `gkd_oracle` (unchanged) with Liger-Kernel's Qwen3
RoPE, RMSNorm and SwiGLU kernels. Those kernels are declared as a
`Qwen3ModelPatch` for both the student and the frozen teacher. The arm imports
`liger_kernel`, which the participant source policy rejects (`import
'liger_kernel' is outside the allowlist`). Oracles do not have to follow the
participant rules, so this directory measures the arm on the real dual-GPU
Judge in a derived, local-only image. That image adds the pinned Liger package
and switches off only the per-file source scan.

Nothing here enters the Work or Judge images, the image source digest
(`images/work`, `images/judge`) or participant scoring. The derived tag is never
pushed and never overwrites `:local`.

The stock `LigerFusedLinearJSDLoss` was not used as the objective, for three
reasons:

1. Its mixture is `(1-beta)*student + beta*teacher`, while TRL 0.12.1 uses
   `beta*student + (1-beta)*teacher`. The two agree only at `beta=0.5`, and
   no substitution of `beta` reconciles them, because the KL weights stay
   `beta`/`1-beta`.
2. It adds a hard CE term by default.
3. It returns no auxiliaries, and the task needs all nine.

The Judge's hidden CPU objective cases include `beta != 0.5` (the task
requires every `beta` in [0, 1]), so the stock loss would fail
`correctness_ok` before any E2E row was run. That
gives no scoreable number, so that variant was not run.

## Provenance

| Item | Value |
| --- | --- |
| Upstream | `https://github.com/linkedin/Liger-Kernel`, commit `780e76b8115bfd7b78592569853b245a33cc7bf0` (v0.8.1, BSD-2-Clause) |
| Upstream files used | `src/liger_kernel/`, copied verbatim into site-packages (pure Python plus Triton kernels, which JIT-compile for the running sm_90 device on first use) |
| Base image | `qwen3_gkd_dual_h100.judge.qwen3_4b_to_1p7b_gkd_distillation_optimizer_dual_h100_v1:local` (the private Judge, with the holdout), id `sha256:e0a1e39832a7…` |
| Derived image | `…judge.qwen3_4b_to_1p7b_gkd_distillation_optimizer_dual_h100_v1:forced-oracle`, id `sha256:aead1921106a…` |

## Files

- `Dockerfile`: `FROM` the private Judge image. It copies `liger_kernel` from
  the named build context `liger` and applies `admission_override.py`.
- `admission_override.py`: wraps the lab's `runtime/_source_policy_impl.py`
  `_check_python` so that scan errors are recorded (best effort, in
  `/logs/verifier/forced_oracle_admission.json`) and then ignored. The
  structural checks of `admit()` stay in force. It also rewrites the single
  hash line in `/opt/gkd_judge/protected_hashes.txt`, so the verifier's
  protected-file self-check still covers every file.
- `run_forced_oracle.sh`: runs the same flow as `scripts/eval_archive.sh`:
  preflight, `scripts/sforge_dual_h100_entry.py eval`, the same mounts,
  private partition, network `none`, two attested GPU UUIDs, 16 CPUs / 96g
  (the repo resource profile maps this to 32g), and a 12000 s timeout. Only
  two things differ. The Judge image comes from a temporary copy of `tasks/`
  with `judge.image_tag=forced-oracle`; the committed task JSON is
  untouched. And it sets `SFORGE_LLM_JUDGE_MODE=off`.

## Commands

```bash
git clone https://github.com/linkedin/Liger-Kernel "$LIGER_SRC"
git -C "$LIGER_SRC" checkout 780e76b8115bfd7b78592569853b245a33cc7bf0
FORCED_BUILD=1 LIGER_SRC="$LIGER_SRC" GKD_GPU_INDICES=2,3 SFORGE_CPUSET=8-15,40-47 \
  host_reference/forced_judge/run_forced_oracle.sh host_reference/full_liger_gkd forced-gkd-fullliger-r1

# admissible arm: the normal Judge (admission enforced); LLM review off for a trusted oracle
tar -C host_reference/gkd_oracle -czf "$OUT/gkd_oracle.tar.gz" solutions FINAL_VERSION notes.md
GKD_GPU_INDICES=2,3 SFORGE_CPUSET=8-15,40-47 SFORGE_LLM_JUDGE_MODE=off \
  scripts/eval_archive.sh "$OUT/gkd_oracle.tar.gz" forced-gkd-gkdoracle-admissible-r1
```

Reports are written to
`outputs/<task dir>/runs/<RUN_ID>/qwen3_4b_to_1p7b_gkd_distillation_optimizer_dual_h100_v1/submissions/manual-1/report.json`.
Per-row detail is in `private_judge_artifacts.tar.gz` (`gkd_private.json`).

## Gates bypassed (forced arm only)

1. Source admission, per-file scan only (the `liger_kernel` import).
2. The LLM integrity review (host side).

Every other gate ran unmodified and passed. Those gates are the hidden
CPU objective cases, protocol, E2E schema and execution, loss, memory,
correctness evidence, `formal_ready` and `formal_score_enabled`. This task's
Judge has no compute-sanitizer step.

## Measured (H100 NVL 94 GB, physical GPUs 2,3 over NVLink, 2026-10-29)

Row order: S512-B2, S1024-B2, S2048-B2, S512-B4, S512-B8.

| Run id | Arm | Judge | Score | Gates | Row median ratios |
| --- | --- | --- | ---: | --- | --- |
| `forced-gkd-gkdoracle-admissible-r1` | `gkd_oracle` (objective only) | normal `:local`, admission enforced | **105.1828** | 10/10 PASS | 0.9522, 1.0660, 1.0995, 1.0489, 1.0999 |
| `forced-gkd-fullliger-r1` | `full_liger_gkd` (objective + Liger student/teacher kernels) | `:forced-oracle` | **117.7279** | 10/10 PASS (two bypasses above) | 1.0387, 1.1359, 1.2925, 1.1262, 1.3167 |
| `forced-gkd-fullliger-r2` | `full_liger_gkd`, same archive | `:forced-oracle` | **117.6430** | 10/10 PASS (two bypasses above) | 1.0317, 1.1349, 1.2920, 1.1295, 1.3189 |

Run times were 740.0 s, 715.9 s and 700.8 s. Archive sha256: `gkd_oracle`
`c651e214…b28f`, `full_liger_gkd` `eb283590…6bb1`. The candidate files under
`host_reference/<arm>/solutions/objective_v001/` are byte-identical to the
judged files.
