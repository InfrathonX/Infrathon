# Forced-oracle Judge (local only)

The repo's own optimized ORPO solution, `host_reference/full_liger_qwen3/`
(stock Liger-Kernel `LigerFusedLinearORPOLoss` plus Liger's Qwen3 RoPE,
RMSNorm and SwiGLU kernels), imports `liger_kernel`. The participant source
policy (`runtime/_source_policy_impl.py`) rejects that import:
`import 'liger_kernel' is outside the allowlist`. Oracles do not have to follow
the participant rules, so this directory runs the arm through the real
dual-GPU Judge in a derived, local-only image. That image adds the pinned Liger
package and turns off only the per-file source scan.

Nothing here enters the Work or Judge images, the image source digest
(`images/work`, `images/judge`) or participant scoring. The derived tag is never
pushed and never overwrites `:local`.

## Provenance

| Item | Value |
| --- | --- |
| Upstream | `https://github.com/linkedin/Liger-Kernel`, commit `780e76b8115bfd7b78592569853b245a33cc7bf0` (v0.8.1, BSD-2-Clause) |
| Upstream files used | `src/liger_kernel/` copied verbatim into site-packages (pure Python plus Triton kernels, which JIT-compile for the running sm_90 device on first use; nothing is prebuilt) |
| Source hashes | match `full_liger_qwen3/run_distributed_probe.py` `PINNED_SOURCE_SHA256` (for example `chunked_loss/orpo_loss.py` `9eef5460…`) |
| Base image | `qwen3_1p7b_orpo_dual_h100.judge.qwen3_1p7b_orpo_training_optimizer_dual_h100_v4:local` (private Judge with holdout), id `sha256:79467cac26c3…` |
| Derived image | `…judge.qwen3_1p7b_orpo_training_optimizer_dual_h100_v4:forced-oracle`, id `sha256:c01814e5d8f2…` |

## Files

- `Dockerfile`: `FROM` the private Judge image. It copies `liger_kernel` from the
  named build context `liger` and applies `admission_override.py`.
- `admission_override.py`: appends a wrapper to the lab's
  `runtime/_source_policy_impl.py`. The wrapper records each `_check_python`
  scan error (in `/logs/verifier/forced_oracle_admission.json`) and then
  ignores it. The structural checks of `admit()` stay in force: contiguous
  `objective_vNNN`, only `candidate.py`/`__init__.py`, `FINAL_VERSION`, and
  `notes.md`. It also rewrites the hash for that one file in
  `/opt/orpo_judge/protected_hashes.txt`, so the verifier's protected-file
  self-check still passes and still covers everything else.
- `run_forced_oracle.sh`: the same steps as `scripts/eval_archive.sh`:
  preflight, `scripts/sforge_dual_h100_entry.py eval`, the same mounts,
  private partition, `network none`, the two attested GPU UUIDs,
  16 CPUs / 96g (the repo resource profile maps the memory to 32g), and a
  12000 s timeout. The only differences are the Judge image (chosen through
  a temporary copy of `tasks/` with `judge.image_tag=forced-oracle`; the
  committed task JSON is not changed) and `SFORGE_LLM_JUDGE_MODE=off`.

## Commands

```bash
git clone https://github.com/linkedin/Liger-Kernel "$LIGER_SRC"
git -C "$LIGER_SRC" checkout 780e76b8115bfd7b78592569853b245a33cc7bf0
# one-time build + run (pair 4,5; CPU slice of that pair)
FORCED_BUILD=1 LIGER_SRC="$LIGER_SRC" ORPO_GPU_INDICES=4,5 SFORGE_CPUSET=16-23,48-55 \
  host_reference/forced_judge/run_forced_oracle.sh host_reference/full_liger_qwen3 forced-orpo-fullliger-v002-r1
```

The arm directory is packed exactly as a Work capture would pack it
(`solutions/`, `FINAL_VERSION`, `notes.md`). The report is written to
`outputs/<task dir>/runs/<RUN_ID>/qwen3_1p7b_orpo_training_optimizer_dual_h100_v4/submissions/manual-1/report.json`,
and the per-row detail is in `private_judge_artifacts.tar.gz` (`orpo_private.json`).

## Gates bypassed

1. Source admission, per-file scan only (the import allowlist). On the host,
   `admit()` rejects the arm with `import 'liger_kernel' is outside the
   allowlist` (both `objective_v001` and `objective_v002` import it). A debug
   replay of the v001 archive in the same image (verifier run directly, with
   the logs on the host) recorded that same error, and only that error, in
   `forced_oracle_admission.json`.
2. The LLM integrity review (host side; `SFORGE_LLM_JUDGE_MODE=off`).

Every other gate is the unmodified Judge and was enforced: hidden CPU objective
cases, protocol, E2E schema/execution, loss, memory, correctness evidence,
`formal_ready` and `formal_score_enabled`. compute-sanitizer is not part of
this task's Judge.

## Measured (H100 NVL 94 GB, NVLink pairs 4,5 and 2,3, 2026-10-29)

| Run id | Version | Score | Gates | Row median ratios (S512-B2, S1024-B2, S2048-B2, S512-B4, S512-B8) |
| --- | --- | ---: | --- | --- |
| `forced-orpo-fullliger-r1` | v001 (stock adapter) | 0 (invalid) | 6/10. Source and correctness passed. The E2E candidate telemetry was rejected with `worker correctness auxiliary structure is invalid`: stock Liger returns `chosen/rejected_logits_mean` in FP32, and the Judge requires the official BF16 | none scored |
| `forced-orpo-fullliger-v002-r1` | v002 (casts aux[2], aux[3] to BF16) | **123.1016** | 10/10 PASS (with the two bypasses above) | 0.9361, 1.2385, 1.3340, 1.3033, 1.4026 |
| `forced-orpo-fullliger-v002-r2` (GPUs 2,3) | v002, same archive | **127.1081** | 10/10 PASS (with the two bypasses above) | 1.0283, 1.3187, 1.3344, 1.3065, 1.4034 |

Run times were 854.7 s and 762.7 s, and the archive sha256 was `1b30004a…a253` both times. Across the two runs the mean is 125.10. Only the short S512-B2 row moved much (0.936 against 1.028); the long rows reproduce to within 0.3%. The v002
`candidate.py` in `full_liger_qwen3/solutions/objective_v002/` is byte-identical
to the judged file.
