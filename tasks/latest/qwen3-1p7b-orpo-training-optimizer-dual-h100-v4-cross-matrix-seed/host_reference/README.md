# Host-only references

Nothing in this directory enters the Work or Judge build contexts. These files
are maintainer positive controls and calibration tools, not participant
dependencies, hidden score caps, or starter code.

## Treatment arms

| Arm | Backbone | Objective |
| --- | --- | --- |
| official TRL reference | stock Qwen3 | installed official TRL 0.12.1 ORPO methods through the task tensor adapter |
| objective-only Liger | stock Qwen3 | pinned `LigerFusedLinearORPOLoss` |
| full Liger | declarative Liger Qwen3 RoPE/RMSNorm/SwiGLU patch | the same pinned fused ORPO objective |

The reference executes official
`trl.trainer.orpo_trainer.ORPOTrainer.get_batch_logps` and
`ORPOTrainer.odds_ratio_loss`. The task-owned adapter owns tensor projection,
next-token ABI alignment, chosen NLL evaluation, and result packaging. The
task-owned harness owns the model, frozen data, FSDP, AdamW, timer, token count,
and validation. This boundary must not be described as the complete upstream
Trainer running end to end.

Formal batches use completion-only OR labels and a separate full-sequence
chosen-NLL target. Padding is ignored with `-100`; prompt tokens remain in the
chosen NLL but not in chosen/rejected OR log probabilities.

`liger_oracle/` is the objective-only adapter. `full_liger_qwen3/` adds a
declarative `Qwen3ModelPatch` with RoPE, RMSNorm-forward, and SwiGLU-forward
callables. The task runtime performs every attribute write and revalidates
topology and tied-weight invariants before FSDP. The full patch covers 113 norm
modules, including attention q/k norms, and 28 MLP modules. Liger CE and
fused-linear-CE remain disabled.

## Static validation

From the task directory:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 host_reference/validate_treatment_arms.py
```

This validates isolation and wiring; it does not substitute for GPU
qualification.

## Oracle (paper/repo solution) on the formal Judge

`FINAL_ORACLE` names the selected arm. That arm's own `FINAL_VERSION` selects
the version. Oracles are exempt from the participant rules (source-admission
allowlist, LLM integrity review). An arm that imports `liger_kernel` is
therefore measured on the real five-row, four-pair dual-GPU Judge in the
local-only `:forced-oracle` image. See `forced_judge/README.md` for the
image, the exact command and the bypassed gates.

| Candidate | Admissible | Formal Judge score | Gates | Run id |
| --- | --- | ---: | --- | --- |
| seed `objective_v001` (official TRL 0.12.1) | yes | 99.56 / 99.73 / 100.06 / 100.09 / 100.29 | all PASS | V4 qualification; `orpo-h100*-seed*` (dev tree) |
| best admissible paper/repo candidate | none: both Liger arms import `liger_kernel` | n/a | n/a | n/a |
| `full_liger_qwen3` v001 (stock adapter), forced | no | 0 (invalid) | E2E telemetry rejected: the stock Liger logits-mean auxiliaries are FP32, the official ones are BF16 | `forced-orpo-fullliger-r1` |
| **`full_liger_qwen3` v002 (aux dtype glue), forced, selected** | no | **123.1016 / 127.1081** (mean 125.10) | 10/10 PASS in both runs, with the source-admission scan and LLM review bypassed | `forced-orpo-fullliger-v002-r1` (GPUs 4,5), `forced-orpo-fullliger-v002-r2` (GPUs 2,3) |

v002 row median ratios, r1 / r2: S512-B2 0.936 / 1.028, S1024-B2
1.238 / 1.319, S2048-B2 1.334 / 1.334, S512-B4 1.303 / 1.307 and S512-B8
1.403 / 1.403. The earlier host-wrapper measurement of 127.15 used one pair
per row and an older aux check. These formal runs supersede it, though r2
reproduces it. `full_liger_qwen3/run_distributed_probe.py`
and `validate_treatment_arms.py` still pin the unmodified v001 adapter.
