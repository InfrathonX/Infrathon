"""Placeholder per-sample reward.

On-policy distillation has no task reward: the whole learning signal is the teacher's
per-token feedback. Miles still needs a reward function to build rollout batches, so the
starter returns 0.0 for every sample. With GRPO and a constant reward the advantage is
zero and the student does not move.
"""


async def reward_func(args, sample, **kwargs):
    return 0.0
