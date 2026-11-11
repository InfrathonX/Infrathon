"""Answer grading for the OPD v2 math task: the upstream Miles scorer.

``is_correct`` is ``_is_correct`` of Miles 78527b9
``examples/on_policy_distillation/qwen3_5_35b_selfdistill/rm.py``: a response is correct when EITHER its last
``\\boxed{}`` grades equal to the label (``grade_answer_verl``) OR its last ``Answer:`` line
does (DAPO ``compute_score``). ``math_utils.py`` and ``math_dapo_utils.py`` are byte copies of
``miles/rollout/rm_hub/`` at that commit (Apache-2.0, see README.md).

One addition: every grade runs under a SIGALRM watchdog (``timeout_s``), because
``grade_answer_sympy`` may call ``sympy.simplify`` on model output without a time limit; a
grade that times out counts as incorrect and is reported as such. Call from the main thread.
"""

import signal

from .math_dapo_utils import compute_score as _dapo_score
from .math_utils import grade_answer_verl


class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout()


def _upstream_is_correct(response, label):
    # verbatim logic of rm.py::_is_correct (sample.response / sample.label)
    response = response or ""
    if label is None:
        return False
    try:
        if grade_answer_verl(response, label):
            return True
    except Exception:
        pass
    try:
        result = _dapo_score(response, label)
        if isinstance(result, dict) and result.get("acc"):
            return True
    except Exception:
        pass
    return False


def grade(response, label, timeout_s=10):
    """Returns (correct: bool, timed_out: bool)."""
    old = signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(timeout_s)
    try:
        return _upstream_is_correct(response, label), False
    except _Timeout:
        return False, True
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def is_correct(response, label, timeout_s=10):
    return grade(response, label, timeout_s)[0]
