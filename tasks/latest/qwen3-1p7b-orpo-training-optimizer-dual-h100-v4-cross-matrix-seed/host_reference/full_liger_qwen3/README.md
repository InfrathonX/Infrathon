# Full-Liger Qwen3 host-only qualified positive control

This arm packages the stock fused-ORPO adapter and Liger's Qwen3 backbone
kernels as one host-only candidate archive. It is a maintainer qualification
and Oracle-replay asset; it never enters Work or Judge.

## Versions

- `objective_v001`: the stock adapter. It is pinned by `run_distributed_probe.py`
  (`ORACLE_ADAPTER_SHA256`) and used by the host qualification wrapper.
- `objective_v002`: v001 plus one glue fix. It casts the `chosen_logits_mean`
  and `rejected_logits_mean` auxiliaries to the hidden dtype (BF16), matching
  official TRL; the formal Judge checks auxiliary dtypes. `FINAL_VERSION=2`.
  The formal forced-oracle Judge scored it 123.10 and 127.11, see
  `../forced_judge/README.md`.

## Exact treatment

Each fresh `torchrun` rank enters the task-owned
`runtime.distributed_probe`, which loads Qwen3 and then asks the candidate for
a frozen `Qwen3ModelPatch`. The candidate declares the pinned Liger callables;
the trusted runtime performs the RoPE write and binds all 113 RMSNorm methods
(final + input/post/q/k per layer) and 28 SwiGLU methods before FSDP. This is
equivalent to these treatment options:

```text
rope=true
rms_norm=true
swiglu=true
cross_entropy=false
fused_linear_cross_entropy=false
```

The last two settings are essential. `TrainStepModule` calls `model.model` to
obtain final hidden states, then the stock host-only
`LigerFusedLinearORPOLoss` adapter owns the tied output projection and ORPO
loss. Liger's causal-LM CE forward is therefore neither needed nor allowed in
this arm.

The wrapper applies no model patch. It injects and owns only these driver
arguments:

```text
--kind candidate
--lab-root <task>/host_reference/full_liger_qwen3
--candidate-version 1
```

All other arguments go unchanged to the same task-owned
`runtime.distributed_probe.main`. That driver still owns and validates the
model artifact, frozen/synthetic batch construction, record schedule, token
counts, AdamW configuration, FSDP1 `FULL_SHARD`, BF16 policy, warm-up/timing
boundaries, gradients, and telemetry. The wrapper contains no alternative
training loop.

## Static/CPU validation

From the task directory:

```bash
python3 host_reference/full_liger_qwen3/run_distributed_probe.py --validate-only
```

using the task-relative path to this file. Validation pins the vendor commit
and relevant source hashes, constructs tiny baseline and treatment Qwen3
models on CPU, and exercises the same loader/PatchSpec path. It checks state
invariants, all five tiny-model norm bindings, the SwiGLU and RoPE bindings,
and an unchanged causal-LM forward. It never runs a Triton kernel or initializes
CUDA.

## Two-rank qualification invocation

Invoke the wrapper as the torchrun target and pass the ordinary task-driver
arguments, excluding the three wrapper-owned flags:

```bash
CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.run \
  --standalone --nnodes=1 --nproc-per-node=2 --max-restarts=0 \
  host_reference/full_liger_qwen3/run_distributed_probe.py \
  --model-path /path/to/Qwen3-1.7B \
  --data-path /path/to/frozen/public \
  --expected-data-partition public \
  --expected-manifest-sha256 <sha256> \
  --expected-tensors-sha256 <sha256> \
  --output /tmp/full-liger.json \
  --pair-batch 4 --warmup-steps 5 --timed-steps 10
```

Run this in a fresh process cohort, just like the HF and objective-only arms.
The rank-zero telemetry is annotated with a host-only treatment attestation.
