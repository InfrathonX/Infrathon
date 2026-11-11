import ast
import io
import json
import os
import tomllib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from sforge.harness.agent_config import (
    API_KEY_PATH, AUTH_CACHE_PATH, AUTH_JOURNAL_PATH, AgentCredentials, CODEX_PROBE_VALIDATION_SOURCE,
    SUPERVISOR_AUTH_SOURCE, auth_secrets, collect_worker_auth_secrets,
    install_and_validate_agent, load_credentials, load_model_config,
    provider_config, read_private_file, redact,
)
from sforge.pyromind.runner import supervisor_source, validate_no_credentials


def cache(access='access-original', refresh='refresh-original', identity='id-original'):
    return {'auth_mode':'chatgpt','OPENAI_API_KEY':None,'tokens':{
        'access_token':access,'refresh_token':refresh,'id_token':identity,'account_id':'account-public'}}


@pytest.fixture
def config_fixture(tmp_path,monkeypatch):
    import sforge.harness.agent_runtime as runtime_module
    runtime=SimpleNamespace(agent='codex',version='0.test',root=tmp_path/'runtime')
    monkeypatch.setattr(runtime_module,'load_agent_runtime',lambda _:runtime)
    auth=tmp_path/'auth.json';auth.write_text(json.dumps(cache()));auth.chmod(0o600)
    key=tmp_path/'key';key.write_text('api-original');key.chmod(0o600)
    base={'MODEL':'example-model-mini','REASONING_EFFORT':'max','SFORGE_AGENT_RUNTIME_DIR':str(runtime.root)}
    def write(values):
        path=tmp_path/'config.env';path.write_text(''.join(f'{k}={v}\n' for k,v in {**base,**values}.items()));return path
    return SimpleNamespace(auth=auth,key=key,runtime=runtime,write=write)


def test_existing_chatgpt_launch_config_is_accepted_without_api_fields(config_fixture):
    fixture=config_fixture
    model=load_model_config(fixture.write({'CODEX_AUTH_JSON_PATH':fixture.auth}))
    credentials=load_credentials(model)
    assert model['AUTH_MODE']=='chatgpt'
    assert credentials.mode=='chatgpt'
    assert credentials.secrets==frozenset({'access-original','refresh-original','id-original'})
    assert 'access-original' not in repr(credentials)
    config=provider_config(model,credentials.mode)
    assert 'model_reasoning_effort = "max"' in config
    assert 'model_provider' not in config and 'base_url' not in config


def test_codex_web_search_follows_task_internet(config_fixture):
    fixture=config_fixture
    model=load_model_config(fixture.write({'MODEL_API_KEY_FILE':fixture.key,'MODEL_API_BASE_URL':'http://model.invalid/v1'}))
    assert tomllib.loads(provider_config(model,'api-key'))['web_search']=='live'
    assert tomllib.loads(provider_config(model,'api-key',web_search=True))['web_search']=='live'
    assert tomllib.loads(provider_config(model,'api-key',web_search=False))['web_search']=='disabled'


def test_api_config_still_requires_url_and_private_key(config_fixture):
    fixture=config_fixture
    model=load_model_config(fixture.write({'MODEL_API_KEY_FILE':fixture.key,'MODEL_API_BASE_URL':'http://model.invalid/v1'}))
    credentials=load_credentials(model)
    assert model['AUTH_MODE']=='api-key'
    assert credentials.secrets==frozenset({'api-original'})
    assert 'model_provider = "sforge-api"' in provider_config(model,credentials.mode)
    with pytest.raises(ValueError,match='requires MODEL_API_KEY_FILE and MODEL_API_BASE_URL'):
        load_model_config(fixture.write({'MODEL_API_KEY_FILE':fixture.key}))


@pytest.mark.parametrize('mode',['api-key','chatgpt'])
def test_provider_installs_submit_wait_hooks_for_both_auth_modes(mode):
    model={'MODEL':'example-model','REASONING_EFFORT':'max','MODEL_API_BASE_URL':'http://model.invalid/v1'}
    config=tomllib.loads(provider_config(model,mode,submit_wait_timeout_seconds=43_320))
    assert config['model']==model['MODEL']
    assert config['features']['hooks'] is True
    assert config['features']['multi_agent'] is False
    assert config['features']['code_mode'] is False
    assert config['features']['code_mode_only'] is False
    assert config['model_catalog_json']=='/home/agent/.codex/sforge-model-catalog.json'
    assert config['hooks']['PreToolUse'][0]['matcher']=='.*'
    assert config['hooks']['PreToolUse'][0]['hooks'][0]['timeout']==43_320
    assert ' post' in config['hooks']['PostToolUse'][0]['hooks'][0]['command']
    if mode=='api-key':
        assert config['model_provider']=='sforge-api'
        assert config['model_providers']['sforge-api']['base_url']==model['MODEL_API_BASE_URL']
    else:
        assert 'model_provider' not in config


@pytest.mark.parametrize('both',[False,True])
def test_zero_or_ambiguous_authentication_is_rejected(config_fixture,both):
    fixture=config_fixture
    values={'CODEX_AUTH_JSON_PATH':fixture.auth,'MODEL_API_KEY_FILE':fixture.key,'MODEL_API_BASE_URL':'http://model.invalid'} if both else {}
    with pytest.raises(ValueError,match='exactly one'):
        load_model_config(fixture.write(values))


def test_credentials_require_regular_exact_0600_file(tmp_path):
    path=tmp_path/'credential';path.write_text('secret')
    for mode in (0o644,0o640,0o400):
        path.chmod(mode)
        with pytest.raises(ValueError,match='0600'):read_private_file(path)
    path.chmod(0o600);assert read_private_file(path)==b'secret'
    link=tmp_path/'symlink';link.symlink_to(path)
    with pytest.raises(OSError):read_private_file(link)
    with pytest.raises(ValueError,match='regular'):read_private_file(tmp_path)


def test_chatgpt_path_cannot_hide_api_auth(config_fixture):
    fixture=config_fixture
    fixture.auth.write_text(json.dumps({'OPENAI_API_KEY':'hidden-api'}))
    with pytest.raises(ValueError,match='ChatGPT access tokens'):
        load_model_config(fixture.write({'CODEX_AUTH_JSON_PATH':fixture.auth}))


@pytest.mark.parametrize('mode',['api-key','chatgpt'])
def test_actual_cli_validation_uses_selected_auth_without_exporting_secret(mode,config_fixture):
    fixture=config_fixture
    values={'CODEX_AUTH_JSON_PATH':fixture.auth} if mode=='chatgpt' else {'MODEL_API_KEY_FILE':fixture.key,'MODEL_API_BASE_URL':'http://model.invalid/v1'}
    model=load_model_config(fixture.write(values));credentials=load_credentials(model)
    backend=Mock();checks=[]
    def check(backend,worker,command,**kwargs):checks.append((command,kwargs));return 'OK'
    result=install_and_validate_agent(backend,object(),model,fixture.runtime,credentials,check,
        tool_catalog='{"models": [{"slug": "fixture", "tool_mode": "direct"}]}')
    written={str(c.args[2]):c.args[1] for c in backend.write_to_container.call_args_list}
    sources=[command[-1] for command,_ in checks if isinstance(command,list) and command[:2]==['python3','-c']]
    assert len(sources)==(1 if mode=='chatgpt' else 2)
    assert result['auth_mode']==mode and result['codex_cli'] is True
    assert result['submission_wait_policy']=='wait-for-result'
    assert '/opt/sforge/submit-wait-hook.py' in written
    assert '/home/agent/.codex/sforge-model-catalog.json' in written
    assert result['responses_api']==(mode=='api-key')
    assert (AUTH_CACHE_PATH in written)==(mode=='chatgpt')
    assert (API_KEY_PATH in written)==(mode=='api-key')
    assert all(secret not in repr(checks) for secret in credentials.secrets)
    assert 'max' in sources[-1]
    assert '--dangerously-bypass-hook-trust' in sources[-1]
    for source in sources:compile(source,'<remote-agent-probe>','exec')


def test_refreshed_and_superseded_tokens_are_protected_before_artifact_export():
    credentials=AgentCredentials('chatgpt',json.dumps(cache()).encode(),frozenset(auth_secrets(cache())))
    current=cache('access-new','refresh-new','id-new')
    responses={AUTH_CACHE_PATH:json.dumps(current).encode(),AUTH_JOURNAL_PATH:json.dumps(['access-between','refresh-between','']).encode()}
    backend=SimpleNamespace(client=SimpleNamespace(read_file=lambda _id,path:responses[path]))
    known=set(credentials.secrets)
    collect_worker_auth_secrets(backend,SimpleNamespace(id='worker'),credentials,known,required=True,journal_required=True)
    assert {'access-original','access-new','refresh-new','id-new','access-between','refresh-between'} <= known
    assert '' not in known
    text=b'hello access-original access-new refresh-between goodbye'
    assert redact(text,known)==b'hello [REDACTED] [REDACTED] [REDACTED] goodbye'
    import tarfile
    payload=io.BytesIO()
    with tarfile.open(fileobj=payload,mode='w:gz') as archive:
        entry=tarfile.TarInfo('candidate.py');entry.size=len(text);archive.addfile(entry,io.BytesIO(text))
    with pytest.raises(ValueError,match='Credential material'):
        validate_no_credentials(payload.getvalue(),known)
    del responses[AUTH_JOURNAL_PATH]
    with pytest.raises(RuntimeError,match='before artifact export'):
        collect_worker_auth_secrets(backend,SimpleNamespace(id='worker'),credentials,known,required=True,journal_required=True)


def test_empty_redaction_values_do_not_corrupt_transcripts():
    assert redact(b'unchanged',{'',None})==b'unchanged'
    assert redact('abc-long abc',{'abc','abc-long',''})=='[REDACTED] [REDACTED]'


@pytest.mark.parametrize('mode',['api-key','chatgpt'])
def test_generated_supervisor_uses_no_api_key_file_for_chatgpt(mode):
    tree=ast.parse(supervisor_source());nodes=[];started=False
    for node in tree.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='env' for t in node.targets):started=True
        if started and isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='fast_failures' for t in node.targets):break
        if started:nodes.append(node)
    reads=[]
    def path(value):
        reads.append(value);return SimpleNamespace(read_text=lambda:'fixture-key')
    namespace={'os':SimpleNamespace(environ={'PATH':'/bin','OPENAI_API_KEY':'inherited','OPENAI_BASE_URL':'wrong','PYROMIND_API_KEY':'platform'}),
               'cfg':{'environment':{},'auth_mode':mode},'pathlib':SimpleNamespace(Path=path), 'deadline':123.0}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<supervisor-env>','exec'),namespace)
    assert reads==([API_KEY_PATH] if mode=='api-key' else [])
    assert ('OPENAI_API_KEY' in namespace['env'])==(mode=='api-key')
    assert 'PYROMIND_API_KEY' not in namespace['env'] and 'OPENAI_BASE_URL' not in namespace['env']
    assert namespace['env']['SFORGE_AGENT_DEADLINE_MONOTONIC']=='123.0'


@pytest.mark.parametrize('document,blocked',[
    ({'schema_version':1,'active':None,'active_tools':{}},False),
    ({'schema_version':1,'active':{'tool_use_id':'pending'},'active_tools':{}},True),
    ({'schema_version':1,'active':None,'active_tools':{'pending-tool':{}}},True),
    ({'schema_version':1,'active':None},True),
])
def test_supervisor_cannot_resume_past_an_undelivered_tool_result(document,blocked):
    tree=ast.parse(supervisor_source())
    loop=next(node for node in ast.walk(tree) if isinstance(node,ast.While)
        and node.body and isinstance(node.body[0],ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='wait_state' for t in node.body[0].targets))
    guard=ast.Module(body=loop.body[:2],type_ignores=[])
    file=SimpleNamespace(exists=lambda:True,read_text=lambda:json.dumps(document))
    namespace={'pathlib':SimpleNamespace(Path=lambda _:file),'json':json,'env':{},
               'recover_ordinary_tools':Mock(side_effect=RuntimeError('No verified writer cancellation boundary')),
               'subprocess':SimpleNamespace(PIPE=-1,run=lambda *args,**kwargs:SimpleNamespace(returncode=0))}
    if blocked:
        with pytest.raises(RuntimeError):exec(compile(guard,'<resume-wait-guard>','exec'),namespace)
    else:
        exec(compile(guard,'<resume-wait-guard>','exec'),namespace)
    assert namespace['recover_ordinary_tools'].call_count == int(bool(document.get('active_tools')) and document.get('active') is None)


def test_supervisor_journal_remembers_rotated_tokens_without_public_artifacts(tmp_path,monkeypatch):
    real_path=Path
    current=tmp_path/'auth.json';current.write_text(json.dumps(cache()))
    def mapped(value):return current if str(value)==AUTH_CACHE_PATH else real_path(value)
    namespace={'cfg':{'auth_mode':'chatgpt'},'json':json,'pathlib':SimpleNamespace(Path=mapped),'ROOT':tmp_path,'os':os}
    exec(SUPERVISOR_AUTH_SOURCE,namespace)
    namespace['remember_auth']()
    current.write_text(json.dumps(cache('access-refreshed','refresh-refreshed','id-refreshed')))
    namespace['remember_auth']()
    journal=tmp_path/'auth-secrets.json'
    assert journal.stat().st_mode & 0o777==0o600
    values=set(json.loads(journal.read_text()))
    assert values==auth_secrets(cache())|auth_secrets(cache('access-refreshed','refresh-refreshed','id-refreshed'))


@pytest.mark.parametrize('events',[
    [{'type':'item.completed','item':{'type':'user_message','text':'Reply exactly READY.'}},{'type':'turn.completed'}],
    [{'type':'item.completed','item':{'type':'agent_message','text':'READY'}},{'type':'turn.failed'}],
    [{'type':'item.completed','item':{'type':'agent_message','text':'NOT READY'}},{'type':'turn.completed'}],
    [{'type':'item.completed','item':{'type':'agent_message','text':'READY'}},{'type':'turn.completed'},{'type':'error'}],
])
def test_cli_probe_rejects_prompt_echo_wrong_answer_or_failed_turn(events):
    result=SimpleNamespace(returncode=0,stdout=('\n'.join(json.dumps(e) for e in events)).encode())
    with pytest.raises(AssertionError):
        exec(CODEX_PROBE_VALIDATION_SOURCE,{'result':result,'json':json})


def test_cli_probe_requires_actual_completed_agent_ready_message():
    events=[{'type':'thread.started'},{'type':'turn.started'},
            {'type':'item.completed','item':{'type':'agent_message','text':' READY\n'}},
            {'type':'turn.completed'}]
    result=SimpleNamespace(returncode=0,stdout=('noise\n'+'\n'.join(json.dumps(e) for e in events)).encode())
    exec(CODEX_PROBE_VALIDATION_SOURCE,{'result':result,'json':json})
