import torch

class KVScorer:
    def __init__(self, spec, assets):
        self.spec = spec

    def score(self, layer_idx, hidden_states, queries, keys, values, position_embeddings, weights, ctx_range):
        start, end = ctx_range
        return (-keys[:, :, start:end, :].float().norm(dim=-1)).contiguous()
