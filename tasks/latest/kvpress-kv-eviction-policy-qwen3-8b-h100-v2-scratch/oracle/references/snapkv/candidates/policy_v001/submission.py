import math

import torch
import torch.nn.functional as F


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class EvictionPolicy:
    """Observation-window attention scoring (SnapKV-style) with sinks and a recent window.

    For each layer the last ``window`` context queries are rotated with the
    supplied RoPE tables and scored against every cached key. Attention mass is
    averaged over the query window and over the four query heads that share a
    KV head, smoothed with a small 1-D pooling, and the top positions are kept.
    The first ``n_sink`` positions and the observation window itself are always
    kept. All work is out of place; no input tensor or global RNG is touched.
    """

    n_sink = 4
    window = 64
    pool = 7

    def __init__(self, spec: dict, compression_ratio: float, assets: dict) -> None:
        self.spec = spec
        self.compression_ratio = float(compression_ratio)
        self.assets = assets
        self.n_heads = int(spec["n_heads"])
        self.n_kv = int(spec["n_kv_heads"])
        self.head_dim = int(spec["head_dim"])
        self.n_sink = max(self.n_sink, int(spec.get("n_sink_floor", 4)))

    def layer_budgets(self, seq_len: int, n_kept_total: int) -> list[int] | None:
        return None

    def select(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        weights: dict,
        n_kept: int,
    ) -> torch.Tensor:
        length = int(keys.shape[2])
        n_kv = int(keys.shape[1])
        group = int(queries.shape[1]) // n_kv
        window = min(self.window, max(1, (n_kept - self.n_sink) // 2), length)

        cos, sin = position_embeddings
        cos_w = cos[:, length - window:].unsqueeze(1).float()
        sin_w = sin[:, length - window:].unsqueeze(1).float()
        q = queries[:, :, length - window:].float()
        q = q * cos_w + _rotate_half(q) * sin_w
        q = q.view(1, n_kv, group, window, self.head_dim)

        k = keys.float()
        logits = torch.einsum("bhgwd,bhld->bhgwl", q, k) / math.sqrt(self.head_dim)
        q_pos = torch.arange(length - window, length, device=keys.device).view(window, 1)
        k_pos = torch.arange(length, device=keys.device).view(1, length)
        logits = logits.masked_fill(k_pos > q_pos, float("-inf"))
        probs = torch.softmax(logits, dim=-1)
        scores = probs.mean(dim=(2, 3))  # [1, n_kv, L]
        del logits, probs

        scores = F.avg_pool1d(scores, kernel_size=self.pool, stride=1, padding=self.pool // 2)
        forced = torch.zeros(length, dtype=torch.bool, device=keys.device)
        forced[: self.n_sink] = True
        forced[length - window:] = True
        scores = scores.masked_fill(forced.view(1, 1, length), float("inf"))

        indices = torch.topk(scores, n_kept, dim=-1).indices
        return torch.sort(indices, dim=-1).values.to(torch.int64).contiguous()
