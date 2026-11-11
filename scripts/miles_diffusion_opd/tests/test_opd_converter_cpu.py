"""CPU flow test: OPD pair converter + MD's DP splitter + microbatch schedule give every global optimizer step
exactly the contract row block (rows [s*R/S, (s+1)*R/S) of the rollout), 3 window pairs per sample."""

import os
import sys
import types

import numpy as np
import pytest
import torch

WORKTREE = os.environ.get("MD_OPD_WORKTREE")
if not WORKTREE:
    pytest.skip("set MD_OPD_WORKTREE to a miles_diffusion 29fde6ee tree with "
                "tasks/latest/miles-diffusion-sd35m-mopd-2xh100-v2-seed/reference/md-opd-reference.patch applied",
                allow_module_level=True)
sys.path.insert(0, WORKTREE)

from miles.ray.data_conversion_hub.opd import convert_samples_to_train_data  # noqa: E402
from miles.utils.train_data_utils import TrainDataDPSplitter, build_microbatch_schedule  # noqa: E402
from miles.utils.types import DenoisingEnv, DiTTrajectory, Sample  # noqa: E402

SIGMAS = torch.tensor([1.0, 0.96, 0.913, 0.858, 0.79, 0.707, 0.602, 0.465, 0.278, 0.0089, 0.0])


def fake_samples(rows: int, n: int, row_base: int) -> list[Sample]:
    out = []
    for g in range(rows):
        row = row_base + g
        seed = 42 + row * n
        start = int(np.random.default_rng(seed).integers(0, 3))
        window = [start, start + 1, start + 2]
        steps = sorted(set(window) | {i + 1 for i in window})
        for j in range(n):
            s = Sample(group_index=row, index=row * n + j, prompt=f"p{row}",
                       metadata={"data_source": "ocr" if row % 2 == 0 else "pickscore", "row_id": row})
            s.dit_trajectory = DiTTrajectory(latents=torch.full((len(steps), 2, 2, 2), float(row * 100 + j)),
                                             timesteps=SIGMAS * 1000, sigmas=SIGMAS,
                                             latent_step_indices=torch.tensor(steps))
            s.denoising_env = DenoisingEnv()
            s.train_metadata = {"sde_step_indices": window}
            out.append(s)
    return out


def test_step_row_blocks():
    args = types.SimpleNamespace(n_samples_per_prompt=8, num_steps_per_rollout=2, actor_num_nodes=1,
                                 actor_num_gpus_per_node=2, sequence_parallel_size=1, teacher_key="data_source")
    rows, n, steps, dp = 8, 8, 2, 2
    data = convert_samples_to_train_data(args, fake_samples(rows, n, row_base=16))
    assert len(data["train_data"]) == rows * n * 3
    shards = TrainDataDPSplitter().split_by_dp(data, dp, mode="contiguous")
    per_step_rows = [set() for _ in range(steps)]
    per_step_pairs = [0] * steps
    for shard in shards:
        pairs = shard["train_data"]
        sched = build_microbatch_schedule(num_pairs_per_optim_step=len(pairs) // steps,
                                          num_optim_steps_per_rollout=steps, micro_batch_size=16)
        for s, ranges in enumerate(sched):
            for lo, hi in ranges:
                for p in pairs[lo:hi]:
                    per_step_rows[s].add(p["row_id"])
                    per_step_pairs[s] += 1
    assert per_step_rows[0] == set(range(16, 20)), per_step_rows
    assert per_step_rows[1] == set(range(20, 24)), per_step_rows
    assert per_step_pairs == [96, 96]
    # each pair's latent is the sample's own x_i at the window step; routes follow metadata
    for p in data["train_data"]:
        assert p["teacher_route"] == ("ocr" if p["row_id"] % 2 == 0 else "pickscore")
        assert float(p["latent"].flatten()[0]) == p["row_id"] * 100 + p["output_idx"]
        assert float(p["timestep"]) == float(SIGMAS[p["step_index"]] * 1000)
        assert float(p["next_timestep"]) == float(SIGMAS[p["step_index"] + 1] * 1000)


if __name__ == "__main__":
    test_step_row_blocks()
    print("ok")
