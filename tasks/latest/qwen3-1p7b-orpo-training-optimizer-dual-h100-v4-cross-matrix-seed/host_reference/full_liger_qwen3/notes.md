# Host-only full-Liger candidate archive

`objective_v001` combines the pinned stock Liger fused ORPO loss with a frozen
`Qwen3ModelPatch`. The candidate only declares the RoPE, RMSNorm-forward, and
SwiGLU-forward callables. The task-owned runtime loads the pinned model first,
performs every global/instance attribute write, validates model state
invariants, and only then constructs FSDP.

This archive is calibration-only and is not copied into Work or Judge.
