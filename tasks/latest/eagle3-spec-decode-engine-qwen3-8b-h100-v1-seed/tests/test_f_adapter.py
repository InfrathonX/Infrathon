"""E2 arbitrary-cell reproduction plus complete measurement integrity cases."""
import copy
import math
import pytest
from conftest import JUDGE, load_file
adapter = load_file('f_adapter', JUDGE / 'run_judge.py')
NAMES = ('chat_bs1', 'math_bs1', 'code_bs1', 'longctx_bs1', 'mixed_bs8', 'think_bs1')

def records(profile='anchor', all_anchor=True):
    reward = {'schema_version': 1, 'reward': 100., 'score_tokens_per_s': 100., **dict.fromkeys(adapter.GATES, 1.)}
    cells = {}
    for name in NAMES:
        count = (5 if name == 'mixed_bs8' else 20) if profile == 'final' else ((2 if all_anchor else 1) if name == 'mixed_bs8' else (8 if all_anchor else 6))
        batch = 8 if name == 'mixed_bs8' else 1
        cells[name] = {'tokens_per_s': 100., 'generated_tokens': count*batch*100, 'generated_lengths': [100]*(count*batch), 'measurements': [{'wall_s': float(batch), 'generated_lengths': [100]*batch} for _ in range(count)]}
    return reward, {**reward, 'status': 'PASS', 'profile': profile, 'all_anchor': all_anchor, 'cell_tokens_per_s': dict.fromkeys(NAMES, 100.), 'cells': cells}

def valid(reward, detail):
    return adapter.build_structured_result(reward, detail, verifier_completed=True, verifier_returncode=0)['valid']

@pytest.mark.parametrize('profile,all_anchor', [('anchor',True),('anchor',False),('final',False)])
def test_valid_profiles(profile, all_anchor):
    assert valid(*records(profile,all_anchor))

@pytest.mark.parametrize('names', [('made_up',), tuple('abcdefg'), tuple('abcdef'), NAMES])
def test_e2_adapter_incomplete(names):
    reward, detail = records()
    detail['cell_tokens_per_s'] = dict.fromkeys(names, 100.)
    detail.pop('cells')
    assert not valid(reward, detail)

@pytest.mark.parametrize('mutation', ['missing','extra','count','zero','negative','nan','inf','total','lengths','batch','rate','bool_length','profile'])
def test_invalid_measurement(mutation):
    reward, detail = records()
    cell = detail['cells']['chat_bs1']
    if mutation == 'missing': detail['cells'].pop('code_bs1')
    elif mutation == 'extra': detail['cells']['other'] = copy.deepcopy(cell)
    elif mutation == 'count': cell['measurements'].pop()
    elif mutation in ('zero','negative','nan','inf'): cell['measurements'][0]['wall_s'] = {'zero':0.,'negative':-1.,'nan':math.nan,'inf':math.inf}[mutation]
    elif mutation == 'total': cell['generated_tokens'] += 1
    elif mutation == 'lengths': cell['generated_lengths'][0] -= 1
    elif mutation == 'batch': cell['measurements'][0]['generated_lengths'].append(100)
    elif mutation == 'rate': cell['tokens_per_s'] = 101.
    elif mutation == 'bool_length': cell['measurements'][0]['generated_lengths'][0] = True
    elif mutation == 'profile': detail['profile'] = 'unknown'
    assert not valid(reward, detail)
