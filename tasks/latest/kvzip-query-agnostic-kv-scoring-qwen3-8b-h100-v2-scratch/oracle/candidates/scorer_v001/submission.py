"""Faithful functional port of Fast KVzip's released Weight gate."""
import math

import torch
import torch.nn.functional as F


def _rms_norm(x, weight, eps=1e-6):
    variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
    return (x * torch.rsqrt(variance + eps).to(x.dtype)) * weight


class KVScorer:
    def __init__(self, spec, assets):
        self.spec = spec
        if assets and all(name.startswith("packed/") for name in assets):
            decoded = {}
            for name, packed in assets.items():
                raw_name = name[len("packed/"):]
                dtype = torch.float32 if raw_name.endswith(("q_norm.weight", "k_norm.weight", "k_base")) else torch.bfloat16
                decoded[raw_name] = packed.movedim(0, -1).contiguous().view(dtype).reshape(packed.shape[1:])
            self.assets = decoded
        else:
            self.assets = assets
        self.window_rule = "w=int(0.02*n_ctx) when n_ctx<16000; trailing w scores set to global max"

    def score(self, layer_idx, hidden_states, queries, keys, values,
              position_embeddings, weights, ctx_range):
        p = f"fastkvzip/layer{layer_idx}/"
        x = hidden_states.squeeze(0)
        nseq = x.shape[0]
        nh, ng = self.spec["n_kv_heads"], self.spec["n_heads"] // self.spec["n_kv_heads"]
        od = self.assets[p + "q_norm.weight"].numel()
        q = F.linear(x, self.assets[p + "q_proj.weight"], self.assets[p + "q_proj.bias"])
        k = F.linear(x, self.assets[p + "k_proj.weight"])
        q = _rms_norm(q.view(nseq, nh, ng, od), self.assets[p + "q_norm.weight"])
        k = _rms_norm(k.view(nseq, nh, 1, od), self.assets[p + "k_norm.weight"])
        q = q.transpose(0, 1).transpose(-1, -2)
        k = k.transpose(0, 1)
        logits = torch.matmul(k, q) / math.sqrt(od) + self.assets[p + "b"].unsqueeze(2)
        base = torch.matmul(self.assets[p + "k_base"], q) / math.sqrt(od)
        score = (1 / (1 + torch.exp(base - logits).sum(2, keepdim=True))).mean(-1)
        score = score.squeeze(-1).unsqueeze(0)[:, :, ctx_range[0]:ctx_range[1]].float()
        n_ctx = ctx_range[1] - ctx_range[0]
        window = int(0.02 * n_ctx) if n_ctx < 16000 else 4096
        if window > 0:
            score[..., -window:] = score.max()
        return score
