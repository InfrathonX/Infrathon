import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def module(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result


scoring = module(ROOT / 'runtime/scoring.py')
adapter = module(ROOT / 'judge/run_judge.py')
SEED = {'steady_fps': 20, 'push_to_emit_p99_seconds': 2, 'ttff_seconds': 2, 'construction_seconds': 30}


def row(fps=40, p99=4, ttff=3, construction=30):
    return scoring.decorate(
        {'id': 'a', 'steady_fps': fps, 'push_to_emit_p99_seconds': p99, 'ttff_seconds': ttff,
         'construction_seconds': construction},
        SEED,
    )


def timing_module():
    """Execute the production timing functions without importing CUDA/PyTorch."""
    import ast
    tree = ast.parse((ROOT / 'runtime/evaluation.py').read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in {'quantile', 'ledger_timing'}]
    unit = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), *functions], type_ignores=[])
    scope = {'WARMUP_CHUNK': 12, 'Mapping': dict, 'Any': object}
    exec(compile(ast.fix_missing_locations(unit), 'evaluation.py', 'exec'), scope)
    return scope['ledger_timing']


def ledger(emit_s, chunks=16):
    records = [SimpleNamespace(kind='frames', chunk_idx=i, timestamp_ns=int(emit_s[i] * 1e9)) for i in range(chunks)]
    return SimpleNamespace(records=records, frames={i: SimpleNamespace(shape=(4,)) for i in range(chunks)})


def time_it(emit_s, push_s, chunks=16):
    pushes = {i: int(push_s[i] * 1e9) for i in range(chunks)}
    return timing_module()(ledger(emit_s, chunks), pushes, start_ns=pushes[0], construction_seconds=0,
                           execution_seconds=1, construction_to_exit_seconds=1)


def test_formula_depth_penalty_and_aggregate():
    a = row(); b = row(10, 1); b['id'] = 'b'
    assert a['case_score'] == pytest.approx(40 * 0.5 ** .25)
    scores = scoring.suite_metrics([a, b])
    assert scores['score'] == pytest.approx(20)
    assert scores['fps_geomean'] == pytest.approx(20)
    assert scores['p99_geomean_ratio'] == pytest.approx(1)
    assert adapter._derive_metrics([a, b])['score'] == pytest.approx(scores['score'])


def test_latency_ttff_and_construction_boundaries():
    assert scoring.timing_gates(row(p99=6, ttff=6, construction=90)) == {'ttff_ok': True, 'p99_ok': True, 'construction_ok': True}
    assert not scoring.timing_gates(row(p99=6.00001))['p99_ok']
    assert not scoring.timing_gates(row(ttff=6.00001))['ttff_ok']
    assert not scoring.timing_gates(row(construction=90.00001))['construction_ok']


@pytest.mark.parametrize('seed_construction,limit', [(30, 90), (100, 200)])
def test_construction_limit_takes_the_larger_branch(seed_construction, limit):
    seed = {**SEED, 'construction_seconds': seed_construction}
    measured = {'steady_fps': 40, 'push_to_emit_p99_seconds': 2, 'ttff_seconds': 2, 'construction_seconds': limit}
    assert scoring.timing_gates(scoring.decorate(measured, seed))['construction_ok']
    measured['construction_seconds'] = limit + 0.000001
    assert not scoring.timing_gates(scoring.decorate(measured, seed))['construction_ok']


@pytest.mark.parametrize('value', [0, -1, math.nan, math.inf, True])
def test_invalid_metrics_fail_closed(value):
    with pytest.raises(ValueError): row(p99=value)


def test_memory_cap_is_the_same_on_both_ranks():
    # The cap is the seed's larger per-rank peak plus the margin, on every rank,
    # so where the seed happened to place its work does not decide legality.
    seed = {'peak_memory_mib_per_rank': {'0': 10000, '1': 12000}}
    cap = 12000 + scoring.GPU_MEMORY_MARGIN_MIB
    assert scoring.memory_ok({'0': cap, '1': cap}, seed)
    assert not scoring.memory_ok({'0': cap + 1, '1': 10}, seed)
    assert not scoring.memory_ok({'0': 10, '1': cap + 1}, seed)
    # A rank a one-card runtime never touches may read zero.
    assert scoring.memory_ok({'0': 0, '1': cap}, seed)
    assert not scoring.memory_ok({'0': 10000}, seed)
    assert not scoring.memory_ok({'x': 10000, 'y': 10000}, seed)
    assert not scoring.memory_ok({'0': 1, '1': 1}, {'peak_memory_mib_per_gpu': {'0': 10000, '1': 12000}})


def test_adapter_rejects_forged_case_score():
    a = row(); a['case_score'] *= 1.000001
    with pytest.raises(ValueError): adapter._derive_metrics([a])


def test_adapter_rejects_duplicate_rows():
    with pytest.raises(ValueError): adapter._derive_metrics([row(), row()])


def test_steady_window_and_p99_start_at_the_controller_push():
    # Startup chunks wait 50 s, steady chunks 2 s; only chunks >= 12 count.
    emit = [100 + i for i in range(16)]
    push = [emit[i] - (50 if i < 12 else 2) for i in range(16)]
    timing = time_it(emit, push)
    assert timing['push_to_emit_p99_seconds'] == 2
    assert timing['push_to_emit_p50_seconds'] == 2
    # 16 frames from push(12)=110 s to emit(15)=115 s.
    assert timing['steady_fps'] == pytest.approx(16 / 5)


def test_holding_an_output_back_does_not_raise_steady_fps():
    # v1 anchored the window on chunk 11's emission, so holding chunk 11 and
    # bursting the backlog inflated FPS. The push-anchored window does not move.
    chunks = 40
    push = [10 + i for i in range(chunks)]
    honest = [p + 3 for p in push]
    held = list(honest)
    release = honest[11] + 5
    for i in range(11, chunks):
        held[i] = max(honest[i], release + (i - 11) * 1e-3)
    fair, gamed = time_it(honest, push, chunks), time_it(held, push, chunks)
    assert gamed['steady_fps'] == pytest.approx(fair['steady_fps'])
    assert gamed['push_to_emit_p99_seconds'] > fair['push_to_emit_p99_seconds']


def test_outputs_cannot_precede_their_push():
    emit = [100 + i for i in range(16)]
    push = [e - 2 for e in emit]
    push[13] = emit[13] + 1e-6
    push[14], push[15] = push[13] + 1, push[13] + 2
    with pytest.raises(ValueError, match='emitted before it was pushed'):
        time_it(emit, push)


def test_chunk0_push_is_the_start_stamp():
    emit = [100 + i for i in range(16)]
    pushes = {i: int((e - 2) * 1e9) for i, e in enumerate(emit)}
    with pytest.raises(ValueError, match='start'):
        timing_module()(ledger(emit), pushes, start_ns=pushes[0] - 1, construction_seconds=0,
                        execution_seconds=1, construction_to_exit_seconds=1)


@pytest.mark.parametrize('mismatch', ['score', 'fps_geomean', 'p99_geomean_ratio', 'reward', 'gate', 'calibrated'])
def test_adapter_main_rejects_mismatched_trusted_result(tmp_path, monkeypatch, capsys, mismatch):
    import json, sys
    a = row(); metrics = scoring.suite_metrics([a])
    detail = {**metrics, 'valid': True, 'rows': [a], 'calibrated': True, 'gates': {name: True for name in adapter.GATES}}
    reward = {'reward': metrics['score'], 'valid': True, 'gates': dict(detail['gates'])}
    if mismatch == 'reward': reward['reward'] *= 1.00001
    elif mismatch == 'gate': reward['gates']['construction_ok'] = False
    elif mismatch == 'calibrated': detail['calibrated'] = False
    else: detail[mismatch] *= 1.00001
    (tmp_path / 'sdv2_pipeline_v2_results.json').write_text(json.dumps(detail))
    (tmp_path / 'reward.json').write_text(json.dumps(reward))
    monkeypatch.setattr(adapter, '_run', lambda *args: 0)
    monkeypatch.setattr(sys, 'argv', ['run_judge.py', '--log-dir', str(tmp_path)])
    adapter.main()
    output = json.loads(capsys.readouterr().out.splitlines()[1])
    assert output['valid'] is False and output['score'] == 0


def test_adapter_accepts_a_consistent_trusted_result(tmp_path, monkeypatch, capsys):
    import json, sys
    a = row(); metrics = scoring.suite_metrics([a])
    detail = {**metrics, 'valid': True, 'rows': [a], 'calibrated': True, 'gates': {name: True for name in adapter.GATES}}
    (tmp_path / 'sdv2_pipeline_v2_results.json').write_text(json.dumps(detail))
    (tmp_path / 'reward.json').write_text(json.dumps({'reward': metrics['score'], 'valid': True, 'gates': dict(detail['gates'])}))
    monkeypatch.setattr(adapter, '_run', lambda *args: 0)
    monkeypatch.setattr(sys, 'argv', ['run_judge.py', '--log-dir', str(tmp_path)])
    adapter.main()
    output = json.loads(capsys.readouterr().out.splitlines()[1])
    assert output['valid'] is True and output['score'] == pytest.approx(metrics['score'])
    assert 'construction_ok' in output['gates']


def test_final_holdout_covers_the_paper_grid(load_module):
    cases = load_module('runtime/cases.py')
    grid = {((c.height, c.width), c.steps) for c in cases.FINAL_CASES}
    for resolution in ((480, 832), (512, 512)):
        for steps in (1, 2, 3, 4):
            assert (resolution, steps) in grid
    assert len(cases.FINAL_CASES) == 12
    assert {c.steps for c in cases.PUBLIC_CASES} == {2, 4}
    assert {(c.height, c.width) for c in cases.FINAL_CASES} - {(c.height, c.width) for c in cases.PUBLIC_CASES} == {(576, 576), (768, 448), (384, 640)}
    assert all(c.frames >= 257 for c in cases.FINAL_CASES)


def test_work_context_omits_hidden_definitions(tmp_path):
    import subprocess, sys
    source = tmp_path / 'source'; source.mkdir()
    (source / 'cases.py').write_text('private case definitions')
    (source / 'scoring.py').write_text('public shared scoring')
    target = tmp_path / 'target'
    subprocess.run([sys.executable, str(ROOT / 'scripts/prepare_work_runtime.py'), str(source), str(target)], check=True)
    assert not (target / 'cases.py').exists()
    assert (target / 'scoring.py').is_file()


@pytest.mark.parametrize('seed_ttff,limit', [(2, 6), (4, 10)])
def test_ttff_accepts_warmup_and_rejects_above_both_limit_branches(seed_ttff, limit):
    seed = {**SEED, 'ttff_seconds': seed_ttff}
    measured = {'steady_fps': 40, 'push_to_emit_p99_seconds': 2, 'ttff_seconds': limit, 'construction_seconds': 30}
    assert scoring.timing_gates(scoring.decorate(measured, seed))['ttff_ok']
    measured['ttff_seconds'] = limit + 0.000001
    assert not scoring.timing_gates(scoring.decorate(measured, seed))['ttff_ok']
