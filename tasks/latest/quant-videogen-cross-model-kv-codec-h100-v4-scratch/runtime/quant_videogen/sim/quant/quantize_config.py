"""Model-facing labels retained independently of the participant algorithm."""

from dataclasses import dataclass


@dataclass
class QuantizeConfig:
    quant_type: str = "none"
    cache_num_k_centroids: int = 256
    cache_num_v_centroids: int = 256
    kmeans_max_iters: int = 100
    quant_block_size: int = 16
    num_prq_stages: int = 4
    asymmetric: bool = False
    quant_factor: int = 8
    quant_keep_recent_chunks: int = 0
    quant_sink_keep_chunks: int = 0
