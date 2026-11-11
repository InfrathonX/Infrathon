"""CPU tests of Wan2.2 DanceGRPO task v2: the 50-step held-out quality gate (judge_v2/quality_eval.py), its
calibration record, the Judge's explicit weights_only loads, the brief and task JSON, and the build wiring."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WAN = ROOT / 'scripts/miles_diffusion_wan'
JUDGE = WAN / 'judge_v2'
TASK = ROOT / 'tasks/latest/miles-diffusion-wan22-dancegrpo-2xh100-v2-seed'
QUAL = json.loads((TASK / 'runtime/qualification.json').read_text())
METRIC = QUAL['metric']
TASK_JSON = json.loads((TASK / 'tasks/miles_diffusion_wan22_dancegrpo_2xh100_v2.json').read_text())
SIBLINGS = ('quality_eval', 'checks_replay', 'checks_probe', 'checks_run', 'checks_score', 'traj_delta',
            'wan_dance_contract', 'oracle_dance_fp32', 'hpsv3_scorer')


def _load(name, path):
    saved = {m: sys.modules.pop(m) for m in SIBLINGS if m in sys.modules}
    saved_path = list(sys.path)
    sys.path[:0] = [str(JUDGE), str(WAN)]
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for m in SIBLINGS:
            sys.modules.pop(m, None)
        sys.modules.update(saved)
        sys.path[:] = saved_path


qe = _load('wan22_v2_quality_eval', JUDGE / 'quality_eval.py')


def test_eval_is_50_steps_with_v1_seeds():
    proto = METRIC['quality_protocol']
    assert qe.SAMPLER['num_inference_steps'] == 50 == proto['num_inference_steps']
    assert qe.EVAL_VERSION == 'wan22-qeval-v2' == proto['eval_version']
    # same (prompt, seed) pairs as v1: only the step count changed
    assert qe.SEED_NAMESPACE == 'wan22-qeval-v1' == proto['namespace']
    assert qe.eval_seed(proto['namespace'], 'hao7_test', 0, 0) == qe.H('wan22-qeval-v1|hao7_test|0|0')


def test_judge_passes_the_step_count_and_a_cap_that_fits_the_eval():
    src = (JUDGE / 'run_judge.py').read_text()
    assert '"--steps", str(proto["num_inference_steps"])' in src
    rj_cap = int(src.split('QUALITY_TIMEOUT_S = ')[1].split()[0])
    assert rj_cap == 2100 and TASK_JSON['judge']['eval_timeout'] == 5000 >= 2627 + 180 + rj_cap


@pytest.mark.parametrize('name', ['checks_run.py', 'checks_probe.py', 'run_judge.py'])
def test_judge_torch_loads_are_weights_only(name):
    src = (JUDGE / name).read_text()
    calls = [line for line in src.splitlines() if 'torch.load(' in line]
    assert calls and all('weights_only=True' in line for line in calls)


def test_gate_and_base_are_the_calibrated_values():
    assert METRIC['quality_gate_min'] == -0.28
    base = METRIC['base_quality_per_repeat']
    assert len(base) == 3 and METRIC['base_quality'] == pytest.approx(sum(base) / 3)
    assert METRIC['base_quality'] == pytest.approx(0.78913, abs=1e-5)
    assert '>= -0.28' in METRIC['formula']


def test_calibration_honest_masters_pass_and_oversized_updates_fail():
    cal = METRIC['quality_calibration_50']
    for lab, row in cal.items():
        q = sum(row['per_repeat_50']) / 3 - METRIC['base_quality']
        assert q == pytest.approx(row['quality_score_50'])
        assert row['gate_50_passed'] == (q >= METRIC['quality_gate_min'])
    honest = ('ref04', 'ref01v2', 'modelav2', 'modelbrec', 'lr_r03')
    assert all(cal[k]['gate_50_passed'] for k in honest)
    assert not cal['ref04_x2']['gate_50_passed'] and not cal['ref04_x3']['gate_50_passed']
    # the cross-check that made v2: the masters v1's 10-step gate rejected are no worse than the references at 50 steps
    assert cal['lr_r12']['quality_score_10'] < -0.2 and cal['lr_r12']['quality_score_50'] > cal['ref04']['quality_score_50']


def test_metric_v2_arithmetic_with_the_v2_gate():
    base = METRIC['base_quality_per_repeat']
    ok = qe.metric_v2([b - 0.17 for b in base], base, [173.0, 172.9, 173.1], -0.28)
    assert ok['quality_gate_passed'] and ok['score'] == pytest.approx(3600 / 173.0)
    bad = qe.metric_v2([b - 0.37 for b in base], base, [100.0, 100.0, 100.0], -0.28)
    assert not bad['quality_gate_passed'] and bad['score'] == 0.0


def test_brief_and_query_state_the_v2_gate_and_the_score():
    brief = (TASK / 'images/work/AGENT_TASK.md').read_text()
    query = TASK_JSON['work']['agent_query']
    assert '≥ −0.28' in brief and '50 steps' in brief and '(0.7891)' in brief
    assert '0.28 below' in query and '3600 / T' in query and 'entry condition, not the score' in query
    for text in (brief, query):
        assert 'no internet' not in text.lower() and 'offline' not in text.lower()


def test_build_packs_the_v2_judge_and_the_steps_into_both_metric_files():
    build = (TASK / 'scripts/build_images.sh').read_text()
    assert '"${WAN}"/judge_v2/{run_judge.py' in build and '"${WAN}"/judge/{' not in build
    assert '"num_inference_steps"' in build
    launch = (TASK / 'scripts/run_task.py').read_text()
    assert '"namespace", "num_inference_steps")' in launch


def test_sxm_qualification_keeps_the_judge_constants():
    """The 80 GB SXM record (runtime/qualification.dual-cuda-sm90-nccl-80g.json) is qualified on its profile and keeps
    what the Judge reads from the image (metric) and from the private root (tau_pi_old, R6) unchanged."""
    sxm = json.loads((TASK / 'runtime/qualification.dual-cuda-sm90-nccl-80g.json').read_text())
    assert sxm['status'].startswith('QUALIFIED') and sxm['execution_profile_id'] == 'dual-cuda-sm90-nccl-80g'
    assert all(sxm[k] == QUAL[k] for k in ('metric', 'tau_pi_old', 'r6', 'r6_public'))
    assert sxm['cpu_pinning']['policy_id'] == QUAL['cpu_pinning']['policy_id']
    runs = sxm['sxm_qualification']['runs']
    refs = [r for name, r in runs.items() if '-ref-' in name]
    assert len(refs) == 3 and all(r['valid'] and r['score'] > 0 and r['quality_score'] >= METRIC['quality_gate_min']
                                  for r in refs)
    assert runs['mdwan-sxm-starter-01']['score'] == 0
    profiles = json.dumps(TASK_JSON)
    assert 'dual-cuda-sm90-nccl-80g' in profiles
