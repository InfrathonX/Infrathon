"""Invariants of the v2 suite: an eight-GPU world, six families, ~970 hidden cases.

The case schema, token split, effective work and family aggregate live in the
trusted runtime; the Judge verifier re-derives every reported number from the
rows; the public and Judge suites share no (K, N, E_rank, topk) identity, seed or
model; and every constant that records the frozen suite's size agrees with it.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import statistics
from pathlib import Path

import pytest

from conftest import TASK_ROOT, judge_only_file, load_module

runtime = load_module("suite_v2_runtime", "runtime/moe_ag_runtime.py")
verifier = load_module("suite_v2_verifier", "images/judge/verify_moe_ag.py")
verifier.RUNTIME_ROOT = TASK_ROOT / "runtime"
adapter = load_module("suite_v2_adapter", "images/judge/run_judge.py")
generator = load_module("suite_v2_generator", "scripts/generate_final_manifest.py")
layout = load_module("suite_v2_layout", "scripts/validate_layout.py")
public = load_module("suite_v2_public", "images/work/benchmark/public_runner.py")

FROZEN = "judge/private/final_cases.json"


def _judge() -> dict:
    return json.loads(judge_only_file(FROZEN).read_text(encoding="utf-8"))


def _public() -> dict:
    return json.loads((TASK_ROOT / "images/work/benchmark/public_cases.json").read_text(encoding="utf-8"))


def _encode(payload: dict) -> bytes:
    return (json.dumps(payload, indent=1, sort_keys=True, allow_nan=False) + "\n").encode()


def _scored(cases: list[dict]) -> list[dict]:
    return [c for c in cases if c["family"] != "edge"]


# --------------------------------------------------------------------------- frozen suites


def test_generator_reproduces_both_committed_suites() -> None:
    judge_only_file("judge/private/routing/gpt-oss-20b.json")
    final, pub, report = generator.build(generator.DEFAULT_SEED)
    assert hashlib.sha256(_encode(final)).digest() == hashlib.sha256(judge_only_file(FROZEN).read_bytes()).digest()
    assert _encode(pub) == (TASK_ROOT / "images/work/benchmark/public_cases.json").read_bytes()
    assert report["public_judge_identity_overlap"] == 0


def test_every_count_constant_agrees_with_the_frozen_suite() -> None:
    cases = _judge()["cases"]
    scored, edges = len(_scored(cases)), len(cases) - len(_scored(cases))
    assert generator.EXPECTED_CASE_COUNT == len(cases)
    assert verifier.EXPECTED_CASE_COUNT == scored and verifier.EXPECTED_EDGE_CASE_COUNT == edges
    assert adapter.EXPECTED_CASE_COUNT == float(scored) and adapter.EXPECTED_EDGE_CASE_COUNT == float(edges)
    assert layout.EXPECTED_FINAL_CASES == scored
    record = json.loads((TASK_ROOT / "judge/private/MANIFEST.json").read_text(encoding="utf-8"))
    assert record["case_count"] == scored and record["suite_id"] == generator.SUITE_ID == verifier.FINAL_SUITE_ID
    build = (TASK_ROOT / "scripts/build_images.sh").read_text(encoding="utf-8")
    assert f'"${{final_cases_count}}" -eq {len(cases)} ]]' in build
    assert f"exactly {len(cases)} cases ({scored} scored, {edges} edge)" in build
    formal = (TASK_ROOT / "scripts/validate_formal_artifacts.py").read_text(encoding="utf-8")
    assert f'final.get("case_count")!={len(cases)}' in formal


def test_suite_breadth() -> None:
    cases = [runtime.validate_case(c) for c in _judge()["cases"]]
    scored = [c for c in cases if c["scored"]]
    assert {c["family"] for c in scored} == set(runtime.FAMILIES)
    assert {c["tier"] for c in scored} == {"paper", "model", "real"}
    assert sum(1 for c in scored if c["tier"] == "paper") >= 40
    assert {c["group"] for c in cases} == set(runtime.GROUPS)
    # Scored cases use deployable world sizes; non-power-of-two worlds are
    # correctness-only edges, so a kernel cannot hard-code W in {2, 4, 8}.
    assert {c["W"] for c in scored} == {2, 4, 8}
    assert {c["W"] for c in cases if not c["scored"]} >= {3, 6}
    assert {(c["TP"], c["EP"]) for c in scored if c["W"] == 8} >= {(8, 1), (4, 2), (2, 4), (1, 8)}
    assert {"uniform", "prod", "skew", "hot", "segment"} <= {c["routing"]["kind"] for c in scored}
    assert any(c["split"] == "uneven" for c in scored)
    assert all((c["layers"] > 1) == (c["family"] == "sequence") for c in scored)
    assert len({c["model"] for c in scored}) >= 30
    assert len({runtime.case_identity(c) for c in scored}) >= 200
    assert len({c["M"] for c in scored}) >= 300
    assert max(generator.est_peak_gib(c) for c in _judge()["cases"]) <= 60


def test_public_and_judge_share_no_identity_seed_or_model() -> None:
    judge, pub = _judge()["cases"], _public()["cases"]
    assert not {runtime.case_identity(c) for c in _scored(judge)} & {runtime.case_identity(c) for c in _scored(pub)}
    assert not {c["seed"] for c in judge} & {c["seed"] for c in pub}
    assert not {c["case_id"] for c in judge} & {c["case_id"] for c in pub}
    shared = {c["model"] for c in judge} & {c["model"] for c in pub}
    assert shared <= {"edge", "synthetic"}
    assert len(_scored(pub)) < len(_scored(judge)) // 5


def test_verifier_rejects_damaged_manifests() -> None:
    instance = _judge()
    assert len(verifier.validate_final_manifest(instance, runtime)) == len(instance["cases"])
    bad = copy.deepcopy(instance); bad["suite_id"] = "public-v2"
    with pytest.raises(RuntimeError):
        verifier.validate_final_manifest(bad, runtime)
    bad = copy.deepcopy(instance); bad["cases"].pop()
    with pytest.raises(RuntimeError):
        verifier.validate_final_manifest(bad, runtime)
    bad = copy.deepcopy(instance); bad["cases"][1]["seed"] = bad["cases"][0]["seed"]
    with pytest.raises(RuntimeError):
        verifier.validate_final_manifest(bad, runtime)
    bad = copy.deepcopy(instance); bad["cases"][5]["W"] = 9
    with pytest.raises(ValueError):
        verifier.validate_final_manifest(bad, runtime)


# --------------------------------------------------------------------------- case schema


def _base_case() -> dict:
    return copy.deepcopy(next(c for c in _judge()["cases"] if c["family"] == "tp8" and c["layers"] == 1))


def _mutate(**changes) -> dict:
    case = _base_case()
    for key, value in changes.items():
        if value is KeyError:
            case.pop(key)
        else:
            case[key] = value
    return case


@pytest.mark.parametrize(
    "case",
    [
        _mutate(extra=1),
        _mutate(seed=KeyError),
        _mutate(TP=3),
        _mutate(group="w4"),
        _mutate(K=100),
        _mutate(N=12),
        _mutate(layers=2),
        _mutate(split="uneven", split_cv=0.0),
        _mutate(split="even", split_cv=0.5),
        _mutate(routing={"kind": "uniform", "sigma_e": 0.1}),
        _mutate(routing={"kind": "teleport"}),
        _mutate(W=True),
        _mutate(family="edge"),
        _mutate(tier="edge"),
        _mutate(family="prefill"),
        _mutate(source=7),
        _mutate(TP=8, EP=1, E=513, topk=8),
        _mutate(TP=8, EP=1, E=512, topk=8, N=16384, K=16384),
        _mutate(M=1 << 20, K=4096),
        _mutate(topk=17),
    ],
)
def test_validate_case_rejects(case: dict) -> None:
    with pytest.raises(ValueError):
        runtime.validate_case(case)


def test_validate_case_derives_split_and_work() -> None:
    case = runtime.validate_case(_base_case())
    assert case["E_rank"] == case["E"] // case["EP"]
    assert sum(case["split_counts"]) == case["M"] and len(case["split_counts"]) == case["W"]
    assert case["effective_flops"] == 2 * case["M"] * case["topk"] * case["N"] * case["K"] // case["EP"]
    assert case["scored"] is True
    forced = _mutate(routing={"kind": "forced", "expert": 0})
    assert runtime.validate_case(forced)["routing"] == {"kind": "forced", "expert": 0}
    with pytest.raises(ValueError):
        runtime.validate_case(_mutate(routing={"kind": "forced", "expert": _base_case()["E"]}))


CAPTURED = {"kind": "captured", "fixture": "olmoe-1b-7b", "layer": 3, "start": 100, "sha256": "ab" * 32}


@pytest.mark.parametrize(
    "routing",
    [
        {k: v for k, v in CAPTURED.items() if k != "sha256"},
        {**CAPTURED, "sha256": "ab" * 31},
        {**CAPTURED, "sha256": "AB" * 32},
        {**CAPTURED, "fixture": "../olmoe"},
        {**CAPTURED, "fixture": ""},
        {**CAPTURED, "fixture": ".hidden"},
        {**CAPTURED, "fixture": 7},
        {**CAPTURED, "layer": -1},
        {**CAPTURED, "start": "0"},
        {**CAPTURED, "extra": 1},
    ],
)
def test_captured_routing_must_pin_a_named_fixture(routing: dict) -> None:
    with pytest.raises(ValueError):
        runtime.validate_case(_mutate(routing=routing))


def test_captured_routing_is_accepted() -> None:
    assert runtime.validate_case(_mutate(routing=dict(CAPTURED)))["routing"] == CAPTURED


def test_sequence_work_counts_every_layer() -> None:
    case = runtime.validate_case(next(c for c in _judge()["cases"] if c["family"] == "sequence"))
    assert case["layers"] > 1
    assert case["effective_flops"] == runtime.effective_flops(case["M"], case["topk"], case["N"], case["K"], case["EP"]) * case["layers"]


@pytest.mark.parametrize("M,W", [(8, 8), (9, 8), (4096, 8), (777, 6), (3, 3), (1 << 16, 2)])
@pytest.mark.parametrize("split,cv", [("even", 0.0), ("uneven", 0.3), ("uneven", 1.0)])
def test_token_split_partitions_tokens(M: int, W: int, split: str, cv: float) -> None:
    counts = runtime.token_split(M, W, split, cv, 12345)
    assert len(counts) == W and sum(counts) == M and min(counts) >= 1
    assert counts == runtime.token_split(M, W, split, cv, 12345)
    if split == "even":
        assert max(counts) - min(counts) <= 1


def test_uneven_split_depends_on_the_seed() -> None:
    draws = {tuple(runtime.token_split(65536, 8, "uneven", 0.5, seed)) for seed in range(8)}
    assert len(draws) == 8


def test_effective_work_does_not_depend_on_routing() -> None:
    case = _base_case()
    a = runtime.validate_case(case)
    case["routing"] = {"kind": "hot", "sigma_e": 0.5, "hot": 2, "hot_boost": 4.0}
    b = runtime.validate_case(case)
    assert a["effective_flops"] == b["effective_flops"]


# --------------------------------------------------------------------------- scoring


def test_family_and_primary_aggregate() -> None:
    rows = [
        {"scored": True, "family": "tp8", "tier": "paper", "tflops": 100.0, "hidden_comm": 0.2},
        {"scored": True, "family": "tp8", "tier": "paper", "tflops": 400.0, "hidden_comm": 0.6},
        {"scored": True, "family": "tp8", "tier": "model", "tflops": 50.0, "hidden_comm": None},
        {"scored": True, "family": "ep", "tier": "real", "tflops": 30.0, "hidden_comm": None},
        {"scored": True, "family": "ep", "tier": "model", "tflops": 120.0, "hidden_comm": None},
        {"scored": True, "family": "decode", "tier": "model", "tflops": 10.0, "hidden_comm": None},
        {"scored": False, "family": "edge", "tier": "edge", "tflops": 1e9, "hidden_comm": None},
    ]
    got = runtime.aggregate_case_results(rows)
    assert got["family_tflops"] == pytest.approx({"tp8": 100.0, "ep": 60.0, "decode": 10.0})
    assert got["score_tflops"] == got["primary_tflops"] == pytest.approx(60000 ** (1 / 3))
    assert got["hidden_comm"] == pytest.approx(0.4)
    with pytest.raises(RuntimeError):
        runtime.aggregate_case_results([{**rows[0], "tflops": 0.0}])
    with pytest.raises(ValueError):
        runtime.aggregate_case_results(rows[-1:])


def _row(case: dict, trials: int) -> dict:
    times = [1e-3 * (1 + 0.01 * i) for i in range(trials)]
    row = {key: copy.deepcopy(case[key]) for key in verifier.IDENTITY_FIELDS}
    row.update(
        case_id=case["case_id"],
        trial_seconds=times,
        tflops=case["effective_flops"] / statistics.median(times) / 1e12,
        rank_records=[{"rank": r, "rows_per_trial": [1] * trials} for r in range(case["W"])],
        rel_l1=1e-3, cossim=0.99999999, row_rel_l2_p999=2e-3, row_rel_l2_max=3e-3, max_abs=0.01,
        rel_l1_limit=2e-3, row_rel_l2_p999_limit=3e-3, row_rel_l2_max_limit=4e-3,
        freivalds_rel_l2=1e-3, freivalds_worst_row=0.03,
        hidden_comm=0.5 if case["tier"] == "paper" else None,
    )
    return row


def _synthetic() -> tuple[list[dict], dict]:
    cases = _judge()["cases"]
    chosen = [c for c in cases if c["family"] == "edge"][:2]
    for family in runtime.FAMILIES:
        chosen += [c for c in cases if c["family"] == family and c["tier"] == "paper"][:1]
        chosen += [c for c in cases if c["family"] == family and c["tier"] == "model"][:1]
    validated = [runtime.validate_case(c) for c in chosen]
    rows = [_row(c, verifier.EXPECTED_TRIALS) for c in validated if c["scored"]]
    edges = [_row(c, 1) for c in validated if not c["scored"]]
    result = {
        "schema_version": 2, "entrypoints": [runtime.ENTRYPOINT],
        "trials_per_case": verifier.EXPECTED_TRIALS, "warmups": verifier.EXPECTED_WARMUPS,
        "correctness_ok": True, "edge_ok": True, "fresh_input_ok": True, "rank_agreement_ok": True,
        "hardware": {"count": 8, "devices": [
            {"capability": [9, 0], "total_memory_bytes": 80 * 2**30, "peer_access_to_all": True} for _ in range(8)
        ]},
        "cases": rows, "edge_cases": edges, **runtime.aggregate_case_results(rows),
    }
    return chosen, result


def _check(result: dict, cases: list[dict]):
    return verifier.validate_runtime_result(result, cases, expected_trials=verifier.EXPECTED_TRIALS,
                                            expected_warmups=verifier.EXPECTED_WARMUPS)


def test_verifier_accepts_a_consistent_result() -> None:
    cases, result = _synthetic()
    score, families, hidden = _check(result, cases)
    assert set(families) == set(runtime.FAMILIES)
    assert score == pytest.approx(result["score_tflops"]) and hidden == pytest.approx(0.5)


def _first_single_layer(result: dict) -> dict:
    return next(r for r in result["cases"] if r["layers"] == 1)


CORRUPTIONS = {
    "inflated_tflops": lambda r: _first_single_layer(r).update(tflops=_first_single_layer(r)["tflops"] * 1.02),
    "missing_trial": lambda r: r["cases"][0]["trial_seconds"].pop(),
    "missing_rank": lambda r: r["cases"][0]["rank_records"].pop(),
    "identity": lambda r: r["cases"][0].update(M=r["cases"][0]["M"] + 1),
    "accuracy": lambda r: r["cases"][0].update(rel_l1=1.0),
    "cossim": lambda r: r["cases"][0].update(cossim=0.99),
    "projection": lambda r: r["cases"][0].update(freivalds_worst_row=0.5),
    "projection_l2": lambda r: r["cases"][0].update(freivalds_rel_l2=0.5),
    "golden_skipped": lambda r: [_first_single_layer(r).pop(k) for k in verifier.METRICS + verifier.LIMITS],
    "hidden_range": lambda r: r["cases"][0].update(hidden_comm=1.5),
    "family": lambda r: r["family_tflops"].update(tp8=r["family_tflops"]["tp8"] * 1.01),
    "score": lambda r: r.update(score_tflops=r["score_tflops"] * 1.01),
    "suite_hidden": lambda r: r.update(hidden_comm=0.9),
    "hardware_count": lambda r: r["hardware"].update(count=7),
    "hardware_peer": lambda r: r["hardware"]["devices"][3].update(peer_access_to_all=False),
    "hardware_memory": lambda r: r["hardware"]["devices"][0].update(total_memory_bytes=32 * 2**30),
    "gate": lambda r: r.update(edge_ok=False),
    "trials_header": lambda r: r.update(trials_per_case=3),
    "warmups": lambda r: r.update(warmups=0),
    "dropped_case": lambda r: r["cases"].pop(),
    "dropped_edge": lambda r: r["edge_cases"].pop(),
    "renamed_case": lambda r: r["cases"][0].update(case_id="other"),
    "edge_trials": lambda r: r["edge_cases"][0]["trial_seconds"].append(1e-3),
}


@pytest.mark.parametrize("name", sorted(CORRUPTIONS))
def test_verifier_rejects_corrupted_results(name: str) -> None:
    cases, result = _synthetic()
    CORRUPTIONS[name](result)
    with pytest.raises(RuntimeError):
        _check(result, cases)


# --------------------------------------------------------------------------- real routing


def _captured(cases: list[dict]) -> list[dict]:
    return [c for c in cases if c["routing"]["kind"] == "captured"]


def test_real_tier_replays_every_captured_model() -> None:
    judge, pub = _judge()["cases"], _public()["cases"]
    jr, pr = _captured(judge), _captured(pub)
    assert {c["routing"]["fixture"] for c in jr} == set(generator.JUDGE_CAPTURED.values()) == set(layout.JUDGE_ROUTING_FIXTURES)
    assert {c["routing"]["fixture"] for c in pr} == set(generator.PUBLIC_CAPTURED.values()) == set(layout.PUBLIC_ROUTING_FIXTURES)
    assert not set(generator.JUDGE_CAPTURED.values()) & set(generator.PUBLIC_CAPTURED.values())
    for cases, catalog in ((judge, generator.JUDGE_CAPTURED), (pub, generator.PUBLIC_CAPTURED)):
        real = [c for c in cases if c["tier"] == "real"]
        assert real == _captured(cases)
        assert all(catalog[c["model"]] == c["routing"]["fixture"] for c in real)
    assert {c["family"] for c in jr} == set(runtime.FAMILIES)


def test_captured_cases_pin_the_committed_fixtures() -> None:
    record = json.loads((TASK_ROOT / "judge/private/MANIFEST.json").read_text(encoding="utf-8"))
    for case in _captured(_judge()["cases"]):
        name = case["routing"]["fixture"]
        data = judge_only_file(f"judge/private/routing/{name}.npy").read_bytes()
        assert hashlib.sha256(data).hexdigest() == case["routing"]["sha256"] == record["routing_fixtures"][name]
    for case in _captured(_public()["cases"]):
        name = case["routing"]["fixture"]
        path = TASK_ROOT / "images/work/benchmark/routing" / f"{name}.npy"
        meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        assert hashlib.sha256(path.read_bytes()).hexdigest() == case["routing"]["sha256"] == meta["sha256"]
        assert case["M"] <= meta["tokens"] and case["routing"]["layer"] < meta["layers"]
        assert (meta["experts"], meta["topk"]) == (case["E"], case["topk"])


def test_routing_fixture_bytes_must_match_the_pinned_digest(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    import io

    buffer = io.BytesIO()
    np.lib.format.write_array(buffer, np.arange(2 * 16 * 2, dtype=np.uint8).reshape(2, 16, 2) % 8, allow_pickle=False)
    (tmp_path / "toy.npy").write_bytes(buffer.getvalue())
    good = hashlib.sha256(buffer.getvalue()).hexdigest()
    fixtures = runtime.RoutingFixtures(tmp_path)
    assert fixtures.table("toy", good).shape == (2, 16, 2)
    with pytest.raises(RuntimeError, match="digest"):
        runtime.RoutingFixtures(tmp_path).table("toy", "0" * 64)
    with pytest.raises(RuntimeError, match="name"):
        runtime.RoutingFixtures(tmp_path).table("../toy", good)


def test_verifier_stages_only_the_pinned_fixtures_read_only(tmp_path: Path) -> None:
    source = tmp_path / "private"
    source.mkdir()
    blobs = {"a": b"alpha", "b": b"beta", "c": b"gamma"}
    for name, data in blobs.items():
        (source / f"{name}.npy").write_bytes(data)
    pin = lambda name, data: {"routing": {"kind": "captured", "fixture": name, "sha256": hashlib.sha256(data).hexdigest()}}  # noqa: E731
    cases = [pin("a", b"alpha"), pin("a", b"alpha"), pin("b", b"beta"), {"routing": {"kind": "uniform"}}]
    run = tmp_path / "run"
    run.mkdir()
    target = verifier.stage_routing_fixtures(run, cases, source)
    assert sorted(p.name for p in target.iterdir()) == ["a.npy", "b.npy"]
    assert (target / "a.npy").read_bytes() == b"alpha" and (target / "a.npy").stat().st_mode & 0o777 == 0o444
    other = tmp_path / "other"
    other.mkdir()
    assert verifier.stage_routing_fixtures(other, [{"routing": {"kind": "uniform"}}], source) is None
    for bad in ([pin("c", b"alpha")], [pin("a", b"alpha"), pin("a", b"beta")], [pin("missing", b"x")]):
        run = tmp_path / f"run-{len(list(tmp_path.iterdir()))}"
        run.mkdir()
        with pytest.raises((RuntimeError, FileNotFoundError)):
            verifier.stage_routing_fixtures(run, bad, source)


# --------------------------------------------------------------------------- what the agent sees


def test_work_and_judge_trusted_copies_are_identical() -> None:
    pairs = [
        ("runtime/moe_ag_runtime.py", "images/work/benchmark/moe_ag_runtime.py"),
        ("images/judge/source_admission.py", "images/work/benchmark/source_admission.py"),
        ("images/judge/run_judge.py", "judge/run_judge.py"),
    ]
    for a, b in pairs:
        assert (TASK_ROOT / a).read_bytes() == (TASK_ROOT / b).read_bytes(), (a, b)


def _agent_visible_texts() -> dict[str, str]:
    texts: dict[str, str] = {}
    for path in (TASK_ROOT / "images/work").rglob("*"):
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
            try:
                texts[str(path.relative_to(TASK_ROOT))] = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
    task = json.loads(next((TASK_ROOT / "tasks").glob("*.json")).read_text(encoding="utf-8"))
    texts["agent_query"] = json.dumps(task["work"]["agent_query"])
    return texts


def test_no_judge_model_name_reaches_the_agent() -> None:
    judge_only = (set(generator.JUDGE_MOE) | set(generator.JUDGE_DENSE)) - set(generator.PUBLIC_MOE) - set(generator.PUBLIC_DENSE)
    patterns = {name: re.compile(r"(?<![\w.-])" + re.escape(name) + r"(?![\w.-])") for name in judge_only}
    leaks = [(where, name) for where, text in _agent_visible_texts().items()
             for name, pattern in patterns.items() if pattern.search(text)]
    assert leaks == []


def test_agent_is_not_told_the_hidden_suite_composition() -> None:
    # Beyond the support domain the agent learns only that the hidden suite is larger and
    # differs: not its size, its largest shapes, its GPU-count mix or its case tiers.
    judge = _judge()["cases"]
    scored = [c for c in judge if c["family"] != "edge"]
    numbers = {len(scored), len(judge), round(len(scored), -1),
               max(c["M"] for c in scored)}
    public_max_m = max(c["M"] for c in _public()["cases"])
    numbers -= {public_max_m}
    for where, text in _agent_visible_texts().items():
        if where.endswith("public_cases.json"):
            continue
        for n in numbers:
            assert not re.search(rf"(?<![\d.]){n}(?!\d)", text), (where, n)
        for phrase in ("mostly on 8 GPUs", "hidden suite's", "same kinds of workloads as the hidden",
                       "reproducing published experiments", "real-routing replays"):
            assert phrase not in text, (where, phrase)


def test_no_judge_case_identity_reaches_the_agent() -> None:
    judge = _judge()["cases"]
    ids = {c["case_id"] for c in judge}
    seeds = {str(c["seed"]) for c in judge if c["seed"] >= 10**6}
    texts = _agent_visible_texts()
    for where, text in texts.items():
        assert not any(i in text for i in ids), where
        tokens = set(re.findall(r"\d{7,}", text))
        assert not tokens & seeds, where
