"""J2 (and R1) reward-equivalence gate: candidate per-frame HPSv3 μ vs the official-scorer oracle.

Per-frame |Δμ| distribution (mean / p99 / max) plus the exact reward definition (0.1 × mean of the three μ).
Tolerances are provisional until calibrated with honest ports (sdpa/eager, batch sizes, PIL vs GPU resize) versus
planted bugs (M-RoPE port, verl-omni port, logits[:,1], generic prompt, BGR, pixel budget ÷4, frame-0 copy).
S3c honest datapoints: batch noise 0.056 max; faithful port 0.063 max / 0.030 mean.
"""

from __future__ import annotations

import json

import torch

TAU_MEAN, TAU_P99, TAU_MAX = 0.08, 0.2, 0.35  # provisional


def check_scores(candidate: dict, oracle: dict) -> dict:
    c = torch.tensor(candidate["frame_mu"], dtype=torch.float64).reshape(-1, 3)
    o = torch.tensor(oracle["frame_mu"], dtype=torch.float64).reshape(-1, 3)
    if c.shape != o.shape:
        return {"J2_shape": {"ok": False, "candidate": list(c.shape), "oracle": list(o.shape)}}
    d = (c - o).abs().flatten()
    rew = torch.tensor(candidate["reward"], dtype=torch.float64).flatten()
    rew_err = (rew - 0.1 * c.mean(-1)).abs().max()
    return {
        "J2_frame_mu": {"ok": float(d.mean()) <= TAU_MEAN and float(torch.quantile(d, 0.99)) <= TAU_P99
                        and float(d.max()) <= TAU_MAX, "mean": float(d.mean()),
                        "p99": float(torch.quantile(d, 0.99)), "max": float(d.max()), "n_frames": int(d.numel())},
        "J2_reward_definition": {"ok": float(rew_err) <= 1e-9 * max(1.0, float(rew.abs().max())),
                                 "max_err": float(rew_err)},
    }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--oracle", required=True)
    a = ap.parse_args()
    print(json.dumps(check_scores(json.load(open(a.candidate)), json.load(open(a.oracle))), indent=1))
