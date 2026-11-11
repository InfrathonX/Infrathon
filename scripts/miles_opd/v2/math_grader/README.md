# OPD v2 math grader

The answer grader, vendored from upstream Miles.

| File | Origin (radixark/miles @ `78527b9102b3a5f15891128cbec3ac6c2dc4089e`, Apache-2.0) | SHA-256 |
| --- | --- | --- |
| `math_utils.py` | `miles/rollout/rm_hub/math_utils.py`, byte copy | `db83eabd2941fb21253c9f733752d775d80488797582e093f2b94bb29d3d39cc` |
| `math_dapo_utils.py` | `miles/rollout/rm_hub/math_dapo_utils.py`, byte copy (itself adapted from EleutherAI lm-evaluation-harness, Apache-2.0) | `3d6b61357e2ee8f1704ba0eb1726ea2dd65dbf9c76e963f20d5ff6598b8f7b23` |
| `grade.py` | `examples/on_policy_distillation/qwen3_5_35b_selfdistill/rm.py::_is_correct`, plus a per-grade SIGALRM watchdog | — |

Dependencies: `sympy`, `pylatexenc` (both in the Miles base image).
