# Pinned Quant-VideoGen source snapshot

This directory is a copy of the Quant-VideoGen source tree at commit
`0601468f2dbba6a17ac7086faec6d41527cad188` (see `UPSTREAM_COMMIT`;
Apache-2.0 per `pyproject.toml`), together with its experiment integrations
(Self-Forcing, LongCat-Video, HY-WorldPlay, LingBot). Model weights are not
included; they are read-only mounts from the artifact root. The images build
from this task-local copy, so no external checkout is needed.

The task adds adapters to the integrations for device binding and byte
accounting. They are described below.

## HY-WorldPlay 8B cache policy

`experiments/HY-WorldPlay/hyvideo` is this task's adapter for the 8B model,
not released paper code. Its QVG cache follows the paper and the released HY
integration:

- K/V are cached before RoPE and the camera PRoPE transform (paper §5.1,
  pre-RoPE key caching); both are re-applied after `decode`.
- The newest chunk (`chunk_latent_frames` latents) stays BF16 outside the codec,
  as the released `experiments/HY-WorldPlay/wan` path does with `QUANT_FACTOR`;
  only older context passes through `encode`.
- `qvg_cache_stats()` scores exactly the `encode` inputs; the BF16 window is
  reported as `bf16_window_bytes` provenance, like the uncompressed text KV.

With a lossless codec this path reproduces the BF16 golden bit for bit.

## Sharded Judge: device binding and cumulative bytes

The Judge runs up to eight cases at once with every GPU visible (the identity
device mapping, as in `judge/judge_shards.py`). The trusted runners pass each
case's GPU as `QVG_DEVICE_INDEX` (PCI bus order, `CUDA_DEVICE_ORDER=PCI_BUS_ID`)
and its torchrun port as `--master_port`; they never set
`CUDA_VISIBLE_DEVICES`.

- `experiments/LongCat/run_long_t2v.py`: `set_device(QVG_DEVICE_INDEX)`, falling
  back to the upstream `RANK % device_count`. The rank keeps gating the rank-0
  writers, so the device index lives in its own `device_index` variable.
- `experiments/HY-WorldPlay/hyvideo/generate.py`: `set_device(QVG_DEVICE_INDEX)`,
  falling back to `LOCAL_RANK`, bound before the device mesh is built (torch's
  `DeviceMesh` keeps a device chosen before it).
- Without `QVG_DEVICE_INDEX` both entrypoints use the upstream device choice,
  which single-GPU golden staging relies on.

Persistent-cache bytes are summed over the whole rollout, so every compressed
segment counts, not only the last one.

- LongCat: every `quantize_kv_cache` call (one per segment) counts its cache as
  one payload tree over all layers. `dit.kv_stats` holds the sums plus `calls`
  and `last_call`, and the profile reports it as `persistent_kv`.
- HY: every `cache_vision` forward of `worldplay_1_5_transformer.py` closes one
  context pass (`qvg_close_context_pass`). `qvg_cache_stats()` returns the sums
  plus `calls`, `last_call` and the summed `bf16_window_bytes`.
- A non-empty cache whose payload holds zero bytes of CUDA storage fails the run.
- BF16 mode is well defined, with stored = BF16 and a ratio of 1.0. The
  accounting reads storage sizes only and keeps the per-layer rows (and the
  tensors they reference) unchanged, so it does not affect the BF16 goldens.

Both profiles also record `cuda_device_index` and
`other_device_allocated_bytes`: the caching-allocator bytes reserved, now or at
peak, on every visible GPU except the case's own. The controller fails a case
when that number is not 0.
