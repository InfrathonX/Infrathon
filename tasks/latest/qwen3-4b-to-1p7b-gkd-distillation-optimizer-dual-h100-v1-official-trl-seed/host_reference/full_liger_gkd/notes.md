# Host-only forced arm: full Liger Qwen3 kernels + TRL-exact chunked GKD

`objective_v001` = the `gkd_oracle` objective (unchanged TRL-exact chunked GKD)
plus a declarative `Qwen3ModelPatch` from the pinned Liger-Kernel
(`780e76b8115bfd7b78592569853b245a33cc7bf0`): Liger RoPE, RMSNorm-forward and
SwiGLU-forward, exported for both the student (`build_model_patch`) and the
frozen teacher (`build_teacher_patch`).  Liger CE / fused-linear CE stay off
because the objective owns both projections.

It imports `liger_kernel`, which the participant source policy rejects; it is
evaluated only in the local-only `:forced-oracle` Judge image
(`host_reference/forced_judge/`).  Calibration-only; never copied into Work or
Judge.
