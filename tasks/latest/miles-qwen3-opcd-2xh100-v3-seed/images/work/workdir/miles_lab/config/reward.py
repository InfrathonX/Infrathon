"""Diagnostic per-sample reward: 1.0 when the last <answer>X</answer> matches the label.

OPCD does not train on a reward; Miles only needs one to build rollout batches, and
this one makes rollout accuracy visible in the logs.
"""

import re

_ANSWER = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)


def _norm(text):
    return (text or "").strip().strip("()[]").strip().upper()


async def reward_func(args, sample, **kwargs):
    found = _ANSWER.findall(sample.response or "")
    return 1.0 if found and sample.label is not None and _norm(found[-1]) == _norm(str(sample.label)) else 0.0
