"""Fast KVzip released gate, upstream-exact numerics, NVIDIA kvpress window rule.

Gate math: Janghyun1230/FastKVzip@e04afaa prefill/attention/gate.py `Weight.forward`
(identical to NVIDIA/kvpress kvpress/presses/fastkvzip_press.py `FastKVzipGate.forward`).
RMSNorm: transformers `Qwen3RMSNorm.forward` (fp32 normalise, cast to input dtype,
multiply by the fp32 weight), as used by both upstream gates.
Window rule: kvpress `FastKVzipPress._score_fast` (non-chunked prefill path listed on
the kvpress leaderboard that the Fast KVzip README points to):
    window = max(1, int(0.02 * ctx_len)) if ctx_len < 32000 else 4096
    scores[..., -window:] = 1.0
The kvpress `n_sink=4` rule protects the first four prefill tokens, which here lie
inside the always-visible system prompt, so it is a no-op and omitted.
Assets: the released `Jang-Hyun/Fast-KVzip` qwen3-8b/q4_dim16_sink16.pt gate in the
task's lossless transport packing (shared from scorer_v001 via assets.ref).
"""
import math

import torch
import torch.nn.functional as F


def _qwen3_rms_norm(x, weight, eps=1e-6):
    input_dtype = x.dtype
    h = x.to(torch.float32)
    variance = h.pow(2).mean(-1, keepdim=True)
    h = h * torch.rsqrt(variance + eps)
    return weight * h.to(input_dtype)


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

    def score(self, layer_idx, hidden_states, queries, keys, values,
              position_embeddings, weights, ctx_range):
        p = f"fastkvzip/layer{layer_idx}/"
        x = hidden_states.squeeze(0)
        nseq = x.shape[0]
        nh, ng = self.spec["n_kv_heads"], self.spec["n_heads"] // self.spec["n_kv_heads"]
        od = self.assets[p + "q_norm.weight"].numel()
        q = F.linear(x, self.assets[p + "q_proj.weight"], self.assets[p + "q_proj.bias"])
        k = F.linear(x, self.assets[p + "k_proj.weight"])
        q = _qwen3_rms_norm(q.view(nseq, nh, ng, od), self.assets[p + "q_norm.weight"])
        k = _qwen3_rms_norm(k.view(nseq, nh, 1, od), self.assets[p + "k_norm.weight"])
        q = q.transpose(0, 1).transpose(-1, -2)
        k = k.transpose(0, 1)
        d = math.sqrt(od)
        logit = torch.matmul(k, q) / d + self.assets[p + "b"].unsqueeze(2)
        logit_base = torch.matmul(self.assets[p + "k_base"], q) / d
        score = 1 / (1 + torch.exp(logit_base - logit).sum(2, keepdim=True))
        score = score.mean(-1).squeeze(-1).unsqueeze(0)
        start, end = int(ctx_range[0]), int(ctx_range[1])
        score = score[:, :, start:end].float().contiguous()
        n_ctx = end - start
        window = max(1, int(0.02 * n_ctx)) if n_ctx < 32000 else 4096
        score[..., -window:] = 1.0
        return score
