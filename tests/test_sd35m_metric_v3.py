"""CPU tests of the SD3.5-M multi-teacher OPD task v2 (metric v3): per-route closure of the original-to-teacher gap in
task metric (OCR accuracy, PickScore) behind a per-route KL gate and a per-route closure gate; the shard merge of the
task-metric evaluator; the per-seed item seeds; the OCR reward formula; weights_only=True on every candidate record."""
import importlib.util
import json
import math
import re
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
JUDGE = ROOT / 'scripts/miles_diffusion_opd/judge_v2'
TASK = ROOT / 'tasks/latest/miles-diffusion-sd35m-mopd-2xh100-v2-seed'
FROZEN = {'ocr': {'base': 0.2, 'teacher': 0.9}, 'pickscore': {'base': 20.5, 'teacher': 23.0}}


def load(name):
    spec = importlib.util.spec_from_file_location(f'sd35v2_{name}', JUDGE / f'{name}.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


TQE = load('task_quality_eval')


def test_closure_score_and_gates():
    m = TQE.metric_v3({'ocr': 0.83, 'pickscore': 22.75}, FROZEN, True, 100.0, 0.5)
    assert m['closure_per_route'] == pytest.approx({'ocr': 0.9, 'pickscore': 0.9})
    assert m['quality_gate_passed'] and m['score'] == pytest.approx(0.9 * 36)
    # one route below the gate scores 0 whatever the other route does
    m = TQE.metric_v3({'ocr': 0.3, 'pickscore': 25.0}, FROZEN, True, 100.0, 0.5)
    assert not m['quality_gate_passed'] and m['score'] == 0.0
    # the KL gate failing scores 0 even with both closures high
    m = TQE.metric_v3({'ocr': 0.9, 'pickscore': 23.0}, FROZEN, False, 100.0, 0.5)
    assert not m['quality_gate_passed'] and m['score'] == 0.0 and 'KL gate' in m['summary']
    # closure is not capped: better than the teacher counts
    m = TQE.metric_v3({'ocr': 1.04, 'pickscore': 23.5}, FROZEN, True, 50.0, 0.5)
    assert m['quality_score'] > 1 and m['score'] == pytest.approx(m['quality_score'] * 72)


def test_non_finite_student_fails_the_gate_and_reports_json_safe():
    m = TQE.metric_v3({'ocr': float('nan'), 'pickscore': 22.0}, FROZEN, True, 100.0, 0.5)
    assert not m['quality_gate_passed'] and m['score'] == 0.0
    json.dumps(m, allow_nan=False)


def test_unusable_T_and_bad_frozen_values():
    m = TQE.metric_v3({'ocr': 0.83, 'pickscore': 22.75}, FROZEN, True, float('nan'), 0.5)
    assert m['score'] is None
    with pytest.raises(ValueError):
        TQE.metric_v3({'ocr': 0.5, 'pickscore': 21.0}, {'ocr': {'base': 0.5, 'teacher': 0.5}, 'pickscore': FROZEN['pickscore']},
                      True, 100.0, 0.5)
    with pytest.raises(ValueError):
        TQE.metric_v3({'ocr': 0.5}, FROZEN, True, 100.0, 0.5)


def test_item_seed_depends_only_on_case_seed_and_index():
    assert TQE.item_seed(0, 5) == TQE.item_seed(0, 5)
    assert len({TQE.item_seed(c, i) for c in range(3) for i in range(1000)}) == 3000
    assert all(0 <= TQE.item_seed(0, i) < 2 ** 63 for i in range(100))


def _shard(path, k, items):
    path.write_text(json.dumps({'prompts': 'p', 'shard': k, 'num_shards': 2, 'steps': 40, 'res': 512, 'routes': {},
                                'results': [{'name': 'default', 'lora': 'x', 'cfg': 1.0, 'case_seed': 0, 'items': items,
                                             'per_route': TQE.summarize(items), 'seconds': 1.0}]}))


def test_merge_combines_disjoint_shards_and_rejects_overlap(tmp_path):
    items = [{'index': i, 'route': 'ocr' if i < 6 else 'pickscore', 'metric': 'ocr' if i < 6 else 'pickscore',
              'seed': i, 'score': float(i)} for i in range(10)]
    _shard(tmp_path / 'a.json', 0, [it for it in items if it['index'] % 2 == 0])
    _shard(tmp_path / 'b.json', 1, [it for it in items if it['index'] % 2 == 1])
    r = TQE.merge([str(tmp_path / 'a.json'), str(tmp_path / 'b.json')])['results'][0]
    assert r['per_route']['ocr']['n'] == 6 and r['per_route']['ocr']['mean'] == pytest.approx(2.5)
    assert r['per_route']['pickscore']['mean'] == pytest.approx(7.5)
    _shard(tmp_path / 'c.json', 1, [it for it in items if it['index'] % 2 == 0])
    with pytest.raises(ValueError):
        TQE.merge([str(tmp_path / 'a.json'), str(tmp_path / 'c.json')])
    with pytest.raises(ValueError):
        TQE.merge([str(tmp_path / 'a.json')])  # a missing shard


def test_ocr_reward_is_the_flow_grpo_formula(monkeypatch):
    class FakeOCR:
        def __init__(self, text):
            self.text = text

        def ocr(self, img, cls=False):
            return [[[None, (self.text, 0.9)]]] if self.text is not None else [None]

    lev = types.ModuleType('Levenshtein')

    def distance(a, b):
        d = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            prev, d[0] = d[0], i
            for j, cb in enumerate(b, 1):
                prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (ca != cb))
        return d[len(b)]

    lev.distance = distance
    monkeypatch.setitem(sys.modules, 'Levenshtein', lev)
    prompt = 'A sign reading "Open 24 Hours" on a door'
    for text, want in (('OPEN 24 HOURS', 1.0), ('the sign says open24hours!', 1.0), ('Opan 24 Hours', 1 - 1 / 11),
                       ('zzzzzzzzzzzzzzzzzzzzzzzz', 0.0), (None, 0.0)):
        monkeypatch.setattr(TQE, '_OCR', FakeOCR(text))
        assert TQE.ocr_reward((None, prompt)) == pytest.approx(want)


def test_every_candidate_record_is_loaded_with_weights_only():
    for f in JUDGE.glob('*.py'):
        text = f.read_text()
        assert 'weights_only=False' not in text, f
        for line in text.splitlines():
            if 'torch.load(' in line and not line.strip().startswith(('#', 'with torch.load(weights_only')):
                assert 'weights_only=True' in line, (f, line)


def test_judge_imports_no_torch_before_the_last_candidate_launch():
    src = (JUDGE / 'run_judge.py').read_text()
    head = src[: src.index('def main(')]
    assert not re.search(r'^(import torch|from torch)', head, re.M)
    tqe_head = (JUDGE / 'task_quality_eval.py').read_text().split('# ---')[0]
    assert 'import torch' not in tqe_head


def test_task_json_and_brief_identity():
    tj = json.loads((TASK / 'tasks/miles_diffusion_sd35m_mopd_2xh100_v2.json').read_text())
    assert tj['task_id'] == 'miles_diffusion_sd35m_mopd_2xh100_v2'
    assert json.loads((TASK / 'runtime/contract.json').read_text())['task_id'] == tj['task_id']
    brief = (TASK / 'images/work/AGENT_TASK.md').read_text()
    assert not re.search(r'<(CFG|GATE|KLOCR|KLPICK)>', brief), 'unfilled placeholders in AGENT_TASK.md'
