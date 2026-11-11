"""The Work image's public tests stay separate from the Judge's scored cases."""
from __future__ import annotations

import json
import re

from conftest import TASK_ROOT, judge_only_file, _load

public = _load("kv_public_runtime", "images/work/benchmark/kv_transfer_runtime.py")
judge = _load("kv_judge_runtime_split", "runtime/kv_transfer_runtime.py")
runner = _load("kv_public_runner_split", "images/work/benchmark/public_runner.py")
WORK = TASK_ROOT / "images/work"
QUERY = json.loads((TASK_ROOT / "tasks/lmcache_paged_kv_block_transfer_h100_seed_v1.json").read_text())[
    "work"
]["agent_query"]


def _final() -> list[dict]:
    return json.loads(judge_only_file("judge/private/final_cases.json").read_text())["cases"]


def _waves(cases) -> list[dict]:
    return [wave for case in cases for wave in case["waves"]]


def _shape(wave) -> tuple:
    return (wave["layout"], wave["nl"], wave["nh"], wave["hs"])


def test_public_shapes_share_nothing_with_the_scored_holdout() -> None:
    assert not {_shape(w) for w in _waves(public.public_suite())} & {_shape(w) for w in _waves(_final())}


def test_public_and_judge_share_no_case_ids_or_seeds() -> None:
    cases = [*public.public_suite(), *public.edge_fixtures()]
    theirs = [*_final(), *judge.edge_fixtures()]
    assert not {c["case_id"] for c in public.public_suite()} & {c["case_id"] for c in theirs}
    assert not {c["seed"] for c in public.public_suite()} & {c["seed"] for c in _final()}
    assert {c["case_id"] for c in cases} >= {c["case_id"] for c in judge.edge_fixtures()}


def test_public_edges_are_the_judge_edges() -> None:
    """Edges are disclosed hard gates: the public runner runs exactly the Judge's."""
    assert [c["case_id"] for c in public.edge_fixtures()] == [c["case_id"] for c in judge.edge_fixtures()]
    assert [c["waves"] for c in public.edge_fixtures()] == [c["waves"] for c in judge.edge_fixtures()]


def test_public_is_smaller_than_the_judge() -> None:
    size = sum(public.case_logical_bytes(c) for c in public.public_suite())
    assert size < sum(judge.case_logical_bytes(c) for c in _final())


def test_public_covers_every_path() -> None:
    cases = public.public_suite()
    waves = _waves(cases)
    assert {c["family"] for c in cases} == set(public.FAMILIES)
    assert {w["layout"] for w in waves} == {"split", "fused"}
    assert {w["direction"] for w in waves} == {"store", "retrieve"}
    assert {w["hs"] for w in waves} == set(public.HEAD_SIZES)
    assert {w["num_objects"] for w in waves} >= {1, 4}
    assert {c["foreground"] for c in cases if c["arm"] == "overlap"} == {"gemm"}


def test_edges_reach_the_support_domain_limits() -> None:
    waves = _waves(public.edge_fixtures())
    assert max(w["nl"] for w in waves) == public.NL_RANGE[1]
    assert max(w["nh"] for w in waves) == public.NH_RANGE[1]
    assert max(w["nb"] for w in waves) == public.NB_RANGE[1]
    assert {w["hs"] for w in waves} == set(public.HEAD_SIZES)
    assert any(public.wave_object_bytes(w) > public.SCRATCH_BYTES for w in waves)
    assert any(w["skip"] for w in waves)
    assert any(w["chunk_tokens"] % 64 for w in waves)


def test_brief_describes_the_runtime() -> None:
    text = (WORK / "AGENT_TASK.md").read_text()
    for family in public.FAMILIES:
        assert f"`{family}`" in text, family
    assert "1 to 128" in text and "64 or 128" in text
    assert "device^0.35 * host^0.25 * trace^0.10 * overlap^0.30" in text
    assert "seven family scores" in text.replace("\n", " ")
    assert "public_cases.json" in text


def test_runner_uses_public_case_ids() -> None:
    assert set(runner.QUICK_CASES) <= {c["case_id"] for c in public.public_suite()}


def test_agent_visible_files_name_no_private_or_upstream_material() -> None:
    paths = [WORK / "AGENT_TASK.md", *WORK.glob("benchmark/*.py"), *WORK.glob("benchmark/*.h"),
             *(p for p in (WORK / "workdir").rglob("*") if p.is_file())]
    pattern = re.compile(
        r"Judge-owned|holdout|anchor|upstream|f9addd2e|multi_layer|mp_mem|LMCache|vLLM|HAMi|PyroMind|"
        r"H100|NVL|envelope|[Dd]ecoy|sentinel|Trusted|SCORED_|1763|247\.1|headroom"
    )
    for path in paths:
        if path.name == "source_admission.py":
            continue
        for match in pattern.finditer(path.read_text()):
            raise AssertionError((path.name, match.group()))
    assert not pattern.search(QUERY)


def test_work_runtime_has_no_comments_or_function_docstrings() -> None:
    import ast
    import io
    import tokenize

    source = (WORK / "benchmark/kv_transfer_runtime.py").read_text()
    tokens = tokenize.generate_tokens(io.StringIO(source).readline)
    comments = [t.string for t in tokens if t.type == tokenize.COMMENT]
    assert comments == ["#!/usr/bin/env python3"]
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            assert ast.get_docstring(node) is None, node.name
