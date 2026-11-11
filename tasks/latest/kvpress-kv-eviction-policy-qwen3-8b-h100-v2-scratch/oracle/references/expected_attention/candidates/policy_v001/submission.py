import math

import torch


class EvictionPolicy:
    """Expected-attention scoring (port of kvpress ExpectedAttentionPress @ 7f359b84).

    Query mean and covariance come from the supplied pre-RoPE queries (sinks
    excluded); both are rotated by the RoPE matrix averaged over the next
    n_future positions.  The expected attention of a future query on each key
    is softmax(mu.k / sqrt(d) + k^T Sigma k / (2d)), averaged over the query
    heads of a KV head and rescaled by (score + epsilon) * ||v||.  Sinks are
    always kept; budgets are uniform.
    """

    n_sink = 4
    n_future = 512
    epsilon = 1e-2

    def __init__(self, spec: dict, compression_ratio: float, assets: dict) -> None:
        self.head_dim = int(spec["head_dim"]); self.theta = float(spec["rope_theta"])
        self.n_sink = max(self.n_sink, int(spec.get("n_sink_floor", 4)))

    def layer_budgets(self, seq_len: int, n_kept_total: int) -> list[int] | None:
        return None

    def _avg_rope(self, length: int, device) -> torch.Tensor:
        d = self.head_dim; half = d // 2
        inv = 1.0 / (self.theta ** (torch.arange(0, half, device=device, dtype=torch.float32) * 2.0 / d))
        pos = torch.arange(length, length + self.n_future, device=device, dtype=torch.float32)
        freqs = torch.outer(pos, inv); emb = torch.cat((freqs, freqs), dim=-1)
        cos, sin = emb.cos(), emb.sin()
        eye = torch.eye(d, device=device, dtype=torch.float32)
        p = torch.zeros((d, d), device=device, dtype=torch.float32)
        p[half:, :half] = torch.eye(half, device=device); p[:half, half:] = -torch.eye(half, device=device)
        return (cos.unsqueeze(1) * eye + sin.unsqueeze(1) * p).mean(dim=0)

    def select(self, layer_idx, hidden_states, queries, keys, values, position_embeddings, weights, n_kept):
        length = int(keys.shape[2]); n_kv = int(keys.shape[1]); n_heads = int(queries.shape[1]); group = n_heads // n_kv; d = self.head_dim
        sink = min(self.n_sink, n_kept, length)
        q = queries[0, :, sink:].float()                       # [H, L', d]
        mu = q.mean(dim=1)                                     # [H, d]
        centered = q - mu.unsqueeze(1)
        cov = torch.einsum("hsi,hsj->hij", centered, centered) / max(1, q.shape[1])
        r = self._avg_rope(length, keys.device)
        mu = mu @ r.T; cov = r @ cov @ r.T
        k = keys[0, :, sink:].float().repeat_interleave(group, dim=0)   # [H, L', d]
        scores = (k @ mu.unsqueeze(-1)).squeeze(-1) / math.sqrt(d)
        scores = scores + torch.einsum("hni,hij,hnj->hn", k, cov, k) / d / 2
        scores = torch.softmax(scores, dim=-1).view(n_kv, group, -1).mean(dim=1)
        scores = (scores + self.epsilon) * values[0, :, sink:].float().norm(dim=-1)
        top = torch.topk(scores, k=n_kept - sink, dim=-1).indices + sink
        sinks = torch.arange(sink, device=keys.device, dtype=torch.int64).view(1, -1).expand(n_kv, sink)
        return torch.cat((sinks, top), dim=-1).unsqueeze(0).contiguous()
