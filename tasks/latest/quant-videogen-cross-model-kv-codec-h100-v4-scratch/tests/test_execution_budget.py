"""Latency is bounded by the budgets the Judge enforces, not by an H100 wall."""

from __future__ import annotations

import copy
import importlib.util
import json
import random
import sys
from pathlib import Path

import pytest

from conftest import release_file


ROOT = Path(__file__).resolve().parents[1]
JUDGE = ROOT / "judge"
TASK_JSON = ROOT / "tasks/quant_videogen_cross_model_kv_codec_h100_scratch_v4.json"
sys.path.insert(0, str(JUDGE))

import evaluate_hidden  # noqa: E402
import judge_shards  # noqa: E402


DIMS = evaluate_hidden.VBENCH_DIMENSIONS
SHA = "f" * 64


def load_run_judge():
    spec = importlib.util.spec_from_file_location(
        "qvg_h100_scratch_v4_run_judge_budget", JUDGE / "run_judge.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _gates() -> dict:
    gates = {"psnr_min": 20.0, "ssim_min": 0.5, "lpips_max": 0.3, "vbench_allowed_drop": {d: 1.0 for d in DIMS}}
    return {"mean": gates, "case_floor": copy.deepcopy(gates)}


def _quality_reference() -> dict:
    return {"psnr": 25.0, "ssim": 0.8, "lpips": 0.1, "vbench": {d: 89.5 for d in DIMS}}


def budget_release(*, longcat_cases: int = 6, hy_cases: int = 11, **judge) -> dict:
    """The spec's budgets over a case table of the given size."""
    cases = []
    for i in range(longcat_cases):
        cases.append({
            "case_id": f"lc-{i:03d}", "model": "longcat", "seed": 0, "prompt_index": i + 1, "prompt": f"p{i}",
            "segments": 10, "init_video": f"/eval/hidden/longcat/{i:03d}/init.mp4", "init_video_sha256": SHA,
            "bf16_frames": f"/eval/hidden/longcat/{i:03d}/bf16-10seg.frames.pt", "bf16_frames_sha256": SHA,
            "frame_count": 273, "fidelity_frames": [73, 93], "bf16_vbench": {d: 90.0 for d in DIMS},
            "bf16_kv_bytes": 1000,
        })
    for i in range(hy_cases):
        cases.append({
            "case_id": f"hy-{i:02d}", "model": "hy_worldplay_8b", "seed": 0, "prompt": f"c{i}",
            "image": f"/eval/hidden/hy/{i:02d}/image.png", "image_sha256": SHA, "video_length": 285,
            "pose": "w-8,s-8,a-8,d-8,up-8,down-8,w-8,s-8,a-7",
            "bf16_frames": f"/eval/hidden/hy/{i:02d}/bf16-285.frames.pt", "bf16_frames_sha256": SHA,
            "frame_count": 285, "fidelity_frames": [29, 45], "bf16_vbench": {d: 90.0 for d in DIMS},
            "bf16_kv_bytes": 1000,
        })
    release = {
        "status": "calibrated", "protocol_schema_version": 2, "paper_protocol_sha256": SHA,
        "metric_implementation_sha256": {"fidelity": SHA, "vbench": SHA},
        "prompt_file": "/eval/hidden/prompts/MovieGenVideoBench_extended.txt", "prompt_file_sha256": SHA,
        "judge": {
            "max_gpu_count": 8, "min_gpu_count": 1, "slot_uids": list(range(23001, 23009)),
            "master_port_base": 29600, "threads_per_slot": 4, "start_stagger_seconds": 20,
            "case_overhead_seconds": 120, "controller_overhead_seconds": 300,
            "min_device_memory_bytes": 99_000_000_000, **judge,
        },
        "models": {
            "longcat": {"fps": 15, "stage_seconds": 3000, "fidelity_metric_seconds": 600, "vbench_metric_seconds": 1200,
                        "profile_key": "persistent_kv", "frames_file": "final.frames.pt", "gates": _gates(),
                        "quality_reference": _quality_reference()},
            "hy_worldplay_8b": {"fps": 24, "stage_seconds": 900, "fidelity_metric_seconds": 600, "vbench_metric_seconds": 900,
                                "profile_key": "persistent_vision_kv", "frames_file": "gen.frames.pt", "gates": _gates(),
                                "quality_reference": _quality_reference()},
        },
        "cases": cases,
    }
    return evaluate_hidden.validate_release(release)


def list_schedule_makespan(release: dict, gpu_count: int, durations: dict[str, float]) -> float:
    """Independent re-implementation of the controller's schedule: heaviest
    budget first onto the slot that frees up first (stagger included), then
    every slot runs its list back to back with the given actual durations."""
    stagger = release["judge"]["start_stagger_seconds"]
    units = {c["case_id"]: evaluate_hidden.model_unit_seconds(release, c["model"]) for c in release["cases"]}
    order = {c["case_id"]: i for i, c in enumerate(release["cases"])}
    loads = [slot * stagger for slot in range(gpu_count)]
    lists: list[list[str]] = [[] for _ in range(gpu_count)]
    for case_id in sorted(units, key=lambda x: (-units[x], order[x])):
        target = min(range(gpu_count), key=lambda i: (loads[i], i))
        loads[target] += units[case_id]
        lists[target].append(case_id)
    return max(slot * stagger + sum(durations[x] for x in shard) for slot, shard in enumerate(lists))


def test_case_unit_is_stage_plus_the_slower_concurrent_metric_plus_overhead() -> None:
    release = budget_release()
    assert evaluate_hidden.model_unit_seconds(release, "longcat") == 3000 + 1200 + 120
    assert evaluate_hidden.model_unit_seconds(release, "hy_worldplay_8b") == 900 + 900 + 120


def test_the_bound_is_the_list_schedule_of_full_budgets_plus_controller_overhead() -> None:
    release = budget_release()
    units = {c["case_id"]: evaluate_hidden.model_unit_seconds(release, c["model"]) for c in release["cases"]}
    for count in range(1, 9):
        assert evaluate_hidden.controller_budget_seconds(release, count) == (
            list_schedule_makespan(release, count, units) + 300
        )
    # Spec values: six 4320 s LongCat units and eleven 1920 s HY units on
    # eight staggered slots finish by 6320 s.
    assert evaluate_hidden.controller_budget_seconds(release, 8) == 6320 + 300
    assert evaluate_hidden.controller_budget_seconds(release, 1) == 6 * 4320 + 11 * 1920 + 300


def test_the_bound_covers_any_run_whose_cases_stay_inside_their_budgets() -> None:
    release = budget_release()
    rng = random.Random(4)
    units = {c["case_id"]: evaluate_hidden.model_unit_seconds(release, c["model"]) for c in release["cases"]}
    for _ in range(2000):
        count = rng.randint(1, 8)
        durations = {x: rng.uniform(0.0, 1.0) * unit for x, unit in units.items()}
        overhead = release["judge"]["controller_overhead_seconds"]
        assert list_schedule_makespan(release, count, durations) + overhead <= (
            evaluate_hidden.controller_budget_seconds(release, count)
        )


def test_the_bound_covers_random_suites_and_budgets() -> None:
    rng = random.Random(1006)
    for _ in range(300):
        release = budget_release(
            longcat_cases=rng.randint(1, 8), hy_cases=rng.randint(1, 14),
            start_stagger_seconds=rng.randint(0, 60), case_overhead_seconds=rng.randint(1, 300),
        )
        for name in evaluate_hidden.MODELS:
            for key in ("stage_seconds", "fidelity_metric_seconds", "vbench_metric_seconds"):
                release["models"][name][key] = rng.randint(1, 4000)
        units = {c["case_id"]: evaluate_hidden.model_unit_seconds(release, c["model"]) for c in release["cases"]}
        count = rng.randint(1, 8)
        durations = {x: rng.random() * unit for x, unit in units.items()}
        assert list_schedule_makespan(release, count, durations) + 300 <= (
            evaluate_hidden.controller_budget_seconds(release, count)
        )


def test_the_adapter_derives_the_same_bound_with_the_same_function(tmp_path: Path, monkeypatch) -> None:
    release = budget_release()
    path = tmp_path / "release.json"
    path.write_text(json.dumps(release), encoding="utf-8")
    run_judge = load_run_judge()
    for count in range(1, 9):
        assert run_judge.controller_timeout_seconds(path, count) == (
            evaluate_hidden.controller_budget_seconds(release, count)
        )
    # Not a copy of the formula: the adapter calls the controller's function.
    monkeypatch.setattr(evaluate_hidden, "controller_budget_seconds", lambda _release, count: 1000 + count)
    assert run_judge.controller_timeout_seconds(path, 3) == 1003


def shipped_release_with_placeholder_digests() -> dict:
    """The shipped release, with not-yet-pinned golden digests (null until the
    goldens exist; the controller refuses to run on them) replaced, so its
    budgets and schema can be checked before the goldens are generated."""
    release = json.loads(release_file().read_text(encoding="utf-8"))
    for case in release["cases"]:
        for key in [k for k, v in case.items() if k.endswith("_sha256") and v is None]:
            case[key] = SHA
    return release


def test_the_shipped_release_bound_fits_the_task_eval_timeout(tmp_path: Path) -> None:
    release = evaluate_hidden.validate_release(shipped_release_with_placeholder_digests())
    path = tmp_path / "release.json"
    path.write_text(json.dumps(release), encoding="utf-8")
    judge = release["judge"]
    bound = evaluate_hidden.controller_budget_seconds(release, judge["max_gpu_count"])
    assert load_run_judge().controller_timeout_seconds(path, judge["max_gpu_count"]) == bound
    task = json.loads(TASK_JSON.read_text(encoding="utf-8"))
    assert task["judge"]["eval_timeout"] >= bound


def test_an_unpinned_golden_digest_fails_closed() -> None:
    release = budget_release()
    release["cases"][0]["bf16_frames_sha256"] = None
    with pytest.raises(evaluate_hidden.InfrastructureFailure, match="sha256"):
        evaluate_hidden.validate_release(release)


def test_the_controller_loads_the_shipped_release_exactly_when_every_digest_is_pinned() -> None:
    """The documented qualification order re-binds the golden and init digests
    before the first oracle run; until then the controller refuses the shipped
    release (in any status), and once they are bound it must load it as is."""
    release = json.loads(release_file().read_text(encoding="utf-8"))
    unpinned = [(case["case_id"], key) for case in release["cases"]
                for key in ("bf16_frames_sha256", "init_video_sha256") if key in case and case[key] is None]
    if unpinned:
        case_id, key = unpinned[0]
        with pytest.raises(evaluate_hidden.InfrastructureFailure, match=f"cases.{case_id}.{key}"):
            evaluate_hidden.load_release(release_file())
    else:
        assert evaluate_hidden.load_release(release_file())["status"] == release["status"]


def test_the_execution_budget_summary_lists_every_admitted_gpu_count() -> None:
    release = budget_release(min_gpu_count=2, max_gpu_count=8)
    summary = evaluate_hidden.execution_budget(release)
    assert set(summary["controller_seconds_by_gpu_count"]) == {str(n) for n in range(2, 9)}
    assert summary["models"]["longcat"]["case_unit_seconds"] == 4320
    assert summary["controller_seconds_by_gpu_count"]["8"] == evaluate_hidden.controller_budget_seconds(release, 8)


def _row(wall: float) -> dict:
    return {
        "wall_seconds": wall,
        "fidelity": {"psnr": 30.0, "ssim": 0.9, "lpips": 0.1},
        "vbench": {d: 90.0 for d in DIMS},
    }


@pytest.mark.parametrize("model", ["longcat", "hy_worldplay_8b"])
def test_wall_gate_is_the_models_stage_budget_on_whatever_device_runs_it(model: str) -> None:
    release = budget_release()
    stage = release["models"][model]["stage_seconds"]
    case = next(c for c in release["cases"] if c["model"] == model)
    floor = release["models"][model]["gates"]["case_floor"]
    assert evaluate_hidden.case_checks(_row(stage), case, floor, stage_seconds=stage)["wall_time"] is True
    assert evaluate_hidden.case_checks(_row(stage + 0.001), case, floor, stage_seconds=stage)["wall_time"] is False


def test_no_frozen_wall_constant_survives_in_the_quality_gates() -> None:
    release = budget_release()
    for name, model in release["models"].items():
        for kind, gates in model["gates"].items():
            assert not [key for key in gates if "wall" in key or "second" in key], (name, kind)


@pytest.mark.parametrize(
    "path, value",
    [
        (("models", "longcat", "stage_seconds"), 0),
        (("models", "longcat", "stage_seconds"), "3000"),
        (("models", "hy_worldplay_8b", "vbench_metric_seconds"), True),
        (("judge", "case_overhead_seconds"), -1),
        (("judge", "controller_overhead_seconds"), None),
        (("judge", "start_stagger_seconds"), -20),
        (("judge", "max_gpu_count"), 9),
        (("judge", "min_gpu_count"), 0),
        (("judge", "threads_per_slot"), 0),
        (("judge", "master_port_base"), 65530),
        (("judge", "slot_uids"), [23001, 23001, 23003, 23004, 23005, 23006, 23007, 23008]),
        (("judge", "slot_uids"), [0, 23002, 23003, 23004, 23005, 23006, 23007, 23008]),
        (("judge", "slot_uids"), [23001, 23002]),
    ],
)
def test_an_invalid_budget_fails_closed(path: tuple, value) -> None:
    release = budget_release()
    target = release
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(evaluate_hidden.InfrastructureFailure):
        evaluate_hidden.validate_release(release)


def test_a_missing_budget_fails_closed() -> None:
    release = budget_release()
    del release["models"]["longcat"]["fidelity_metric_seconds"]
    with pytest.raises(evaluate_hidden.InfrastructureFailure):
        evaluate_hidden.validate_release(release)


def test_the_gpu_count_must_stay_inside_the_shard_range() -> None:
    release = budget_release()
    for count in (0, judge_shards.MAX_SHARDS + 1):
        with pytest.raises(evaluate_hidden.InfrastructureFailure):
            evaluate_hidden.controller_budget_seconds(release, count)
