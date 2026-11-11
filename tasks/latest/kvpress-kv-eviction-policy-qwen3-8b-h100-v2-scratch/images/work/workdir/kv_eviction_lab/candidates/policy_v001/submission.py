import torch


class EvictionPolicy:
    """Position-only baseline: keep the first n_sink cached positions and the most recent ones.

    The kept set is exactly ``[0, n_sink) ∪ [L - (n_kept - n_sink), L)``. It
    looks at nothing but positions; every other input of ``select`` is unused.
    """

    n_sink = 4

    def __init__(self, spec: dict, compression_ratio: float, assets: dict) -> None:
        self.spec = spec
        self.compression_ratio = float(compression_ratio)
        self.assets = assets

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
        n_sink = min(self.n_sink, n_kept)
        sinks = torch.arange(n_sink, device=keys.device, dtype=torch.int64)
        recent = torch.arange(length - (n_kept - n_sink), length, device=keys.device, dtype=torch.int64)
        indices = torch.cat((sinks, recent)).view(1, 1, -1)
        return indices.expand(1, int(keys.shape[1]), -1).contiguous()
