import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from sforge.harness.agent.codex_tools import direct_tool_catalog, load_direct_tool_catalog


@pytest.fixture
def catalog():
    return {'models': [
        {'slug':'example-model','tool_mode':'code_mode_only','context_window':272000,
         'shell_type':'unified_exec','base_instructions':'Keep the original runtime instructions.'},
        {'slug':'example-model-mini','tool_mode':'code_mode_only','context_window':272000,
         'shell_type':'unified_exec','supported_reasoning_levels':['high','max']},
    ]}


def test_changes_only_selected_model_tool_mode_and_preserves_original(catalog):
    original=copy.deepcopy(catalog)
    actual=json.loads(direct_tool_catalog(catalog,'example-model'))
    expected=copy.deepcopy(original);expected['models'][0]['tool_mode']='direct'
    assert actual==expected
    assert catalog==original


def test_unknown_provider_model_keeps_its_native_fallback(catalog):
    assert json.loads(direct_tool_catalog(catalog,'deepseek-v4-flash'))==catalog


def test_runtime_catalog_is_read_offline_without_provider_credentials(catalog,monkeypatch):
    calls=[]
    monkeypatch.setenv('OPENAI_API_KEY','fixture-secret')
    monkeypatch.setenv('PYROMIND_API_KEY','fixture-platform-secret')
    def run(command,**kwargs):
        calls.append((command,kwargs))
        return SimpleNamespace(stdout=json.dumps(catalog).encode())
    monkeypatch.setattr('sforge.harness.agent.codex_tools.subprocess.run',run)
    value=load_direct_tool_catalog(SimpleNamespace(root=Path('/runtime')),'example-model-mini')
    assert json.loads(value)['models'][1]['tool_mode']=='direct'
    command,kwargs=calls[0]
    assert command==['/runtime/bin/codex','debug','models','--bundled']
    assert 'OPENAI_API_KEY' not in kwargs['env'] and 'PYROMIND_API_KEY' not in kwargs['env']


@pytest.mark.parametrize('document',[{}, {'models':[]}, {'models':[{}]}, {'models':'invalid'}])
def test_invalid_runtime_catalog_is_rejected(document):
    with pytest.raises(ValueError):direct_tool_catalog(document,'example-model')
