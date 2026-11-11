"""The HY-WorldPlay rollout must be a pure function of ``--seed``.

`judge/run_hy_worldplay_8b.py` passes the release seed to
`hyvideo/generate.py`, but the pipeline only routes that seed into a *local*
`torch.Generator` for the latent noise.  The autoregressive rollout separately
draws `points_local` from the **global** RNG and uses it to Monte-Carlo the
FOV-overlap ranking that selects which history frames condition every chunk
after the first.  Without a global seed that ranking flips between runs, which
is what made the `hy_worldplay_8b` VBench `image_quality` gate irreproducible
(see docs/tasks/...-v3-scratch.md section 6).  These tests pin the fix at the
source level: the CPU test suite has no GPU and no model weights, so they read
the vendored entrypoint rather than executing a rollout.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

TASK_ROOT = Path(__file__).resolve().parents[1]
HY = TASK_ROOT / "upstream/experiments/HY-WorldPlay/hyvideo"
GENERATE = HY / "generate.py"
PIPELINE = HY / "pipelines/worldplay_video_pipeline.py"
RETRIEVAL = HY / "utils/retrieval_context.py"
LONGCAT = TASK_ROOT / "upstream/experiments/LongCat/run_long_t2v.py"


def _module(path: Path) -> ast.Module:
    return ast.parse(path.read_text())


def _function(path: Path, name: str) -> ast.FunctionDef:
    for node in ast.walk(_module(path)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{path.name} does not define {name}()")


def test_generate_defines_seed_everything() -> None:
    """The HY entrypoint owns a seeding helper, like LongCat's."""
    _function(GENERATE, "seed_everything")


@pytest.mark.parametrize(
    "call",
    ["random.seed", "np.random.seed", "torch.manual_seed", "torch.cuda.manual_seed_all"],
)
def test_seed_everything_seeds_every_global_rng(call: str) -> None:
    """Every global RNG the rollout can reach is seeded from the argument."""
    body = ast.unparse(_function(GENERATE, "seed_everything"))
    assert f"{call}(seed)" in body, f"seed_everything() does not call {call}(seed)"


@pytest.mark.parametrize(
    "assignment",
    ["torch.backends.cudnn.deterministic = True", "torch.backends.cudnn.benchmark = False"],
)
def test_seed_everything_pins_cudnn(assignment: str) -> None:
    """cuDNN is pinned exactly as LongCat's seed_everything pins it."""
    assert assignment in ast.unparse(_function(GENERATE, "seed_everything"))


def test_main_seeds_before_generating() -> None:
    """main() seeds from args.seed *before* it runs the rollout."""
    body = _function(GENERATE, "main").body
    order = []
    for node in ast.walk(ast.Module(body=body, type_ignores=[])):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in ("seed_everything", "generate_video"):
                order.append((node.func.id, ast.unparse(node)))
    names = [n for n, _ in order]
    assert "seed_everything" in names, "main() never calls seed_everything()"
    assert "generate_video" in names, "main() never calls generate_video()"
    assert names.index("seed_everything") < names.index("generate_video"), (
        "main() must seed before generate_video()"
    )
    seeded = dict(order)["seed_everything"]
    assert "args.seed" in seeded, f"seed_everything() must take args.seed, got {seeded}"


def test_longcat_still_seeds_globally() -> None:
    """The sibling workload keeps the behaviour HY is being aligned with."""
    body = ast.unparse(_function(LONGCAT, "seed_everything"))
    for call in ("torch.manual_seed(seed)", "np.random.seed(seed)", "random.seed(seed)"):
        assert call in body
    assert "torch.backends.cudnn.deterministic = True" in body


def test_points_local_still_comes_from_the_global_rng() -> None:
    """Document why the seed matters; fail loudly if upstream changes shape.

    If a future upstream bump gives `generate_points_in_sphere` its own
    generator argument, this assertion is the signal to revisit the fix rather
    than carry a seeding call that no longer does anything.
    """
    assert "torch.rand(n_points)" in RETRIEVAL.read_text(), (
        "generate_points_in_sphere no longer draws from the global RNG"
    )
    assert "generate_points_in_sphere(50000, 8.0)" in PIPELINE.read_text(), (
        "the rollout no longer draws points_local the way the fix assumes"
    )
    assert "points_local=self.points_local" in PIPELINE.read_text(), (
        "points_local no longer feeds select_aligned_memory_frames"
    )


ATTENTION = HY / "models/transformers/modules/attention.py"
LONGCAT_DIT = TASK_ROOT / "upstream/experiments/LongCat/longcat_video/modules/longcat_video_dit.py"


def test_bf16_context_pass_caches_the_exact_tensors() -> None:
    """The BF16 golden (a8ee4657…) stays reproducible: v4's cumulative byte
    accounting only records what the BF16 branch caches, unchanged."""
    function = _function(ATTENTION, "sequence_parallel_attention_vision")
    [switch] = [
        node for node in ast.walk(function)
        if isinstance(node, ast.If) and ast.unparse(node.test) == "_qvg_enabled()"
    ]
    assert [ast.unparse(node) for node in switch.orelse] == [
        "vision_kv['k_vision'] = key",
        "vision_kv['v_vision'] = value",
        "_qvg_record_layer(block_idx, payload_nbytes((key, value)), (key, value), 0)",
    ]


def test_qvg_newest_chunk_stays_bf16() -> None:
    body = ast.unparse(_function(ATTENTION, "_qvg_encode_context"))
    assert "split = max(k.shape[2] - recent_tokens, 0)" in body
    assert "window_k = k[:, :, split:].contiguous()" in body
    assert "payload_k, payload_v = encode(old_k, old_v)" in body


@pytest.mark.parametrize(
    "path, name",
    [
        (ATTENTION, "_qvg_ratio"),
        (ATTENTION, "_qvg_record_layer"),
        (ATTENTION, "_qvg_pass_stats"),
        (ATTENTION, "qvg_close_context_pass"),
        (ATTENTION, "qvg_cache_stats"),
        (LONGCAT_DIT, "_kv_ratio"),
        (LONGCAT_DIT, "kv_call_stats"),
        (LONGCAT_DIT, "accumulate_kv_stats"),
        (LONGCAT_DIT, "_record_kv_stats"),
    ],
)
def test_byte_accounting_runs_no_tensor_operations(path: Path, name: str) -> None:
    """Summing bytes across passes and segments reads storage sizes only."""
    function = _function(path, name)
    names = {node.id for node in ast.walk(function) if isinstance(node, ast.Name)}
    assert "torch" not in names, f"{name}() must not call torch"
