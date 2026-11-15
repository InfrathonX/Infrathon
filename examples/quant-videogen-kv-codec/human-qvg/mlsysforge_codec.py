"""Task ABI around the unmodified official Quant-VideoGen code.

codec/ is upstream quant_videogen/ at 0601468f2dbba6a17ac7086faec6d41527cad188,
copied byte for byte; only the package directory is renamed. This file is the
only code written for the task. Settings are those of the official run_qvg.sh
scripts: INT2, 256 K/V centroids, block 64, 1 stage, at most 100 k-means
iterations on LongCat and 2 on HY-WorldPlay and Self-Forcing. The ABI has no
model argument, so the head count selects the limit (LongCat has 32 heads).
"""

from dataclasses import asdict, replace

import torch

from codec.compress import compress_kv_cache, get_quantize_fn
from codec.sim.quant.quantize_config import QuantizeConfig
from codec.uncompress import uncompress_kv_cache

_CONFIG = QuantizeConfig(
    quant_type="triton-nstages-kmeans-int2",
    cache_num_k_centroids=256,
    cache_num_v_centroids=256,
    kmeans_max_iters=2,
    quant_block_size=64,
    num_prq_stages=1,
)
_LONGCAT_HEADS = 32
_LONGCAT_KMEANS_MAX_ITERS = 100


def encode(k, v):
    config = _CONFIG
    if k.shape[1] == _LONGCAT_HEADS:
        config = replace(_CONFIG, kmeans_max_iters=_LONGCAT_KMEANS_MAX_ITERS)
    quantize_fn = get_quantize_fn(config.quant_type, config)
    # The model's noise stream stays independent of the codec.
    with torch.random.fork_rng(devices=[k.device.index]):
        packed_k, packed_v = compress_kv_cache(
            k.contiguous(), v.contiguous(), config.quant_type, config, quantize_fn
        )
    # Payloads may hold only tensors and plain containers; decode() restores
    # the objects upstream expects.
    info = {"output_dtype": "bfloat16", "quant_config": asdict(config)}
    packed_k["info"] = info
    packed_v["info"] = dict(info)
    return packed_k, packed_v


def _restored(packed):
    info = packed["info"]
    return {
        **packed,
        "info": {
            "output_dtype": torch.bfloat16,
            "quant_config": QuantizeConfig(**info["quant_config"]),
        },
    }


def decode(packed_k, packed_v):
    return uncompress_kv_cache(_restored(packed_k), _restored(packed_v))
