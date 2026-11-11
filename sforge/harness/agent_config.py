"""Codex provider configuration and private credential handling for any task.

Only sanitized model/auth-mode metadata may enter run artifacts. Credential
payloads are controller memory and Worker-private files, never build contexts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path, PurePosixPath
import stat

API_KEY_PATH = '/home/agent/.config/sforge/model.key'
AUTH_CACHE_PATH = '/home/agent/.codex/auth.json'
CLAUDE_AUTH_CACHE_PATH = '/home/agent/.claude/.credentials.json'
AUTH_JOURNAL_PATH = '/run/sforge-agent/auth-secrets.json'
_SECRET_NAMES = frozenset({'access_token','refresh_token','id_token','api_key','openai_api_key',
                           'accesstoken','refreshtoken','idtoken','apikey','openaiapikey',
                           'anthropic_api_key'})


def read_private_file(path: Path) -> bytes:
    """Read one bounded, unchanged 0600 regular file without following symlinks."""
    path = Path(path).expanduser()
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode):
        os.close(descriptor)
        raise ValueError('Agent credential must be a private regular file (0600)')
    with os.fdopen(descriptor, 'rb') as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or stat.S_IMODE(before.st_mode) != 0o600:
            raise ValueError('Agent credential must be a private regular file (0600)')
        if before.st_size > 1024*1024:
            raise ValueError('Agent credential exceeds the supported file size')
        data = stream.read(1024*1024+1)
        after = os.fstat(stream.fileno())
        identity = lambda value:(value.st_dev,value.st_ino,value.st_size,value.st_mtime_ns,value.st_ctime_ns)
        if identity(before) != identity(after) or len(data) != before.st_size:
            raise ValueError('Agent credential changed while being read')
    return data


def auth_secrets(document) -> set[str]:
    """Collect only credential fields; never treat empty values as secrets."""
    result = set()
    def visit(value):
        if isinstance(value,dict):
            for name,item in value.items():
                if str(name).lower() in _SECRET_NAMES and isinstance(item,str) and item:
                    result.add(item)
                elif isinstance(item,(dict,list)):
                    visit(item)
        elif isinstance(value,list):
            for item in value: visit(item)
    visit(document)
    return result


def parse_chatgpt_cache(data: bytes) -> dict:
    try:
        value=json.loads(data)
    except (ValueError,UnicodeDecodeError):
        raise ValueError('Codex auth cache is not valid JSON') from None
    if (not isinstance(value,dict) or value.get('auth_mode') not in (None,'chatgpt')
            or value.get('OPENAI_API_KEY') or not isinstance(value.get('tokens'),dict)
            or not isinstance(value['tokens'].get('access_token'),str)
            or not value['tokens']['access_token']):
        raise ValueError('Codex auth cache must contain ChatGPT access tokens, not an API key')
    return value


def load_model_config(path: Path) -> dict[str,str]:
    """Read existing launch env as data; require exactly one authentication mode."""
    from dotenv import dotenv_values
    values={key:value for key,value in dotenv_values(path,interpolate=False).items() if value is not None}
    if values.get('AGENT') == 'claude-code':
        from sforge.harness.agent.claude_agent import load_model_config as load_claude_config
        return load_claude_config(path)
    if values.get('AGENT', 'codex') != 'codex':
        raise ValueError('Supported Agent runtimes are codex and claude-code')
    if values.get('CLAUDE_AUTH_JSON_PATH'):
        raise ValueError('Codex cannot use a Claude OAuth credential cache')
    if not values.get('MODEL') or not values.get('SFORGE_AGENT_RUNTIME_DIR'):
        raise ValueError('Model config requires MODEL and SFORGE_AGENT_RUNTIME_DIR')
    cache=bool(values.get('CODEX_AUTH_JSON_PATH'))
    api=bool(values.get('MODEL_API_KEY_FILE') or values.get('MODEL_API_BASE_URL'))
    if cache == api:
        raise ValueError('Choose exactly one Agent authentication mode: CODEX_AUTH_JSON_PATH or MODEL_API_KEY_FILE + MODEL_API_BASE_URL')
    if cache:
        mode='chatgpt'
        name='CODEX_AUTH_JSON_PATH'
        parse_chatgpt_cache(read_private_file(Path(values[name])))
    else:
        mode='api-key'
        if not values.get('MODEL_API_KEY_FILE') or not values.get('MODEL_API_BASE_URL'):
            raise ValueError('API authentication requires MODEL_API_KEY_FILE and MODEL_API_BASE_URL')
        name='MODEL_API_KEY_FILE'
        if not read_private_file(Path(values[name])).strip():
            raise ValueError('Model key file is empty')
    values[name]=str(Path(values[name]).expanduser().absolute())
    values['SFORGE_AGENT_RUNTIME_DIR']=str(Path(values['SFORGE_AGENT_RUNTIME_DIR']).expanduser().absolute())
    from sforge.harness.agent_runtime import load_agent_runtime
    runtime=load_agent_runtime(Path(values['SFORGE_AGENT_RUNTIME_DIR']))
    if runtime.agent != 'codex':
        raise ValueError('This launcher requires a Codex runtime')
    values.update(AGENT='codex',AUTH_MODE=mode,runtime_version=runtime.version)
    return values


@dataclass(frozen=True)
class AgentCredentials:
    mode: str
    payload: bytes = field(repr=False)
    secrets: frozenset[str] = field(repr=False)


def load_credentials(model: dict) -> AgentCredentials:
    mode=model['AUTH_MODE']
    if mode in {'anthropic-api-key', 'claude-oauth'}:
        from sforge.harness.agent.claude_agent import load_credentials as load_claude_credentials
        return load_claude_credentials(model)
    if mode=='chatgpt':
        payload=read_private_file(Path(model['CODEX_AUTH_JSON_PATH']))
        secrets=auth_secrets(parse_chatgpt_cache(payload))
    elif mode=='api-key':
        payload=read_private_file(Path(model['MODEL_API_KEY_FILE'])).strip()
        if not payload: raise ValueError('Model key file is empty')
        try: secrets={payload.decode('utf-8')}
        except UnicodeDecodeError: raise ValueError('Model API key must be UTF-8') from None
    else: raise ValueError('Unsupported Agent authentication mode')
    return AgentCredentials(mode,payload,frozenset(secrets))


def redact(data: str | bytes, secrets) -> str | bytes:
    """Longest first avoids partial-token leakage; empty values cannot corrupt text."""
    for secret in sorted({v for v in secrets if isinstance(v,str) and v},key=len,reverse=True):
        if isinstance(data,bytes): data=data.replace(secret.encode(),b'[REDACTED]')
        else: data=data.replace(secret,'[REDACTED]')
    return data


def collect_worker_auth_secrets(backend,worker,credentials,known_secrets,*,required=False,journal_required=False):
    """Harvest refreshed auth plus the root-owned rotation journal before export."""
    if credentials is None or credentials.mode not in {'chatgpt', 'claude-oauth'}: return
    if credentials.mode == 'claude-oauth':
        from sforge.harness.agent.claude_agent import parse_claude_oauth_cache
        cache_path, parse_cache = CLAUDE_AUTH_CACHE_PATH, parse_claude_oauth_cache
    else:
        cache_path, parse_cache = AUTH_CACHE_PATH, parse_chatgpt_cache
    try:
        document=parse_cache(backend.client.read_file(worker.id,cache_path))
        known_secrets.update(auth_secrets(document))
        try: journal=json.loads(backend.client.read_file(worker.id,AUTH_JOURNAL_PATH))
        except Exception:
            if journal_required: raise
            journal=[]  # Before supervisor start there is no journal.
        if not isinstance(journal,list) or any(not isinstance(v,str) for v in journal):
            raise ValueError('Agent credential rotation journal is invalid')
        known_secrets.update(v for v in journal if v)
    except Exception:
        if required:
            raise RuntimeError('Cannot collect refreshed Agent credentials before artifact export') from None


def provider_config(model: dict, mode: str, *, submit_wait_timeout_seconds: int = 21_720,
                    web_search: bool = True) -> str:
    from sforge.harness.agent.codex_tools import CODEX_TOOL_CATALOG_PATH
    from sforge.harness.submit_wait import submit_wait_hooks_config_fragment

    # Live web search follows the task's `internet` flag; every task allows it
    # since 2026-10-07 (docs/INTERNET_AND_WEB_SEARCH.md).
    config='web_search = '+json.dumps('live' if web_search else 'disabled')+'\nmodel = '+json.dumps(model['MODEL'])+'\n'
    config+='model_reasoning_effort = '+json.dumps(model.get('REASONING_EFFORT','high'))+'\n'
    config+='model_catalog_json = '+json.dumps(CODEX_TOOL_CATALOG_PATH)+'\n'
    if mode=='api-key':
        config+='model_provider = "sforge-api"\n'
    config+='[features]\nhooks = true\nmulti_agent = false\ncode_mode = false\ncode_mode_only = false\n'
    config+=submit_wait_hooks_config_fragment(timeout_seconds=submit_wait_timeout_seconds)
    if mode=='api-key':
        config+='\n[model_providers.sforge-api]\nname = "sforge-api"\n'
        config+='base_url = '+json.dumps(model['MODEL_API_BASE_URL'].rstrip('/'))+'\nenv_key = "OPENAI_API_KEY"\nwire_api = "responses"\n'
    return config



CODEX_PROBE_VALIDATION_SOURCE = r'''
events=[]
for line in result.stdout.splitlines():
    try: event=json.loads(line)
    except (ValueError,UnicodeDecodeError): continue
    if isinstance(event,dict): events.append(event)
assert result.returncode==0, 'Codex model preflight failed'
assert any(event.get('type')=='turn.completed' for event in events), 'Codex probe did not complete a successful turn'
assert not any(event.get('type') in ('turn.failed','error') for event in events), 'Codex probe reported an error'
messages=[event['item'].get('text','') for event in events
          if event.get('type')=='item.completed' and isinstance(event.get('item'),dict)
          and event['item'].get('type')=='agent_message']
assert messages and messages[-1].strip()=='READY', 'Codex did not return the exact READY reply'
'''


def install_and_validate_agent(backend,worker,model,runtime,credentials,check,
                               *, submit_wait_timeout_seconds: int = 21_720,
                               tool_catalog: str | None = None, web_search: bool = True) -> dict:
    """Install only this run's auth and exercise its actual Codex transport."""
    if runtime.agent == 'claude-code':
        from sforge.harness.agent.claude_agent import install_and_validate_agent as install_claude
        return install_claude(backend,worker,model,runtime,credentials,check,
            submit_wait_timeout_seconds=submit_wait_timeout_seconds)
    from sforge.harness.submit_wait import (
        SUBMIT_WAIT_HOOK_PATH, SUBMIT_WAIT_STATE_DIR, generate_submit_wait_hook_script,
    )
    from sforge.harness.agent.codex_tools import CODEX_TOOL_CATALOG_PATH, load_direct_tool_catalog
    if tool_catalog is None:
        tool_catalog=load_direct_tool_catalog(runtime,model['MODEL'])
    print('Uploading pinned Codex runtime '+runtime.version,flush=True)
    backend.copy_to_container(worker,runtime.root,PurePosixPath('/opt/agent-runtime'))
    check(backend,worker,'chmod -R a+rX /opt/agent-runtime',timeout=30)
    check(backend,worker,['mkdir','-p',str(PurePosixPath(SUBMIT_WAIT_HOOK_PATH).parent),SUBMIT_WAIT_STATE_DIR],timeout=15)
    backend.write_to_container(worker,generate_submit_wait_hook_script(),PurePosixPath(SUBMIT_WAIT_HOOK_PATH))
    check(backend,worker,'chown root:root '+SUBMIT_WAIT_HOOK_PATH+'; chmod 644 '+SUBMIT_WAIT_HOOK_PATH+
          '; chown agent:agent '+SUBMIT_WAIT_STATE_DIR+'; chmod 700 '+SUBMIT_WAIT_STATE_DIR,timeout=15)
    destination=API_KEY_PATH if credentials.mode=='api-key' else AUTH_CACHE_PATH
    other=AUTH_CACHE_PATH if credentials.mode=='api-key' else API_KEY_PATH
    check(backend,worker,['rm','-f',other],timeout=15)
    # Precreate privately before the SDK writes any bytes. Existing task images
    # contain neither credential file; install preserves private permissions.
    check(backend,worker,['install','-o','agent','-g','agent','-m','600','/dev/null',destination],timeout=15)
    backend.write_to_container(worker,credentials.payload.decode(),PurePosixPath(destination))
    backend.write_to_container(worker,provider_config(model,credentials.mode,
        submit_wait_timeout_seconds=submit_wait_timeout_seconds,web_search=web_search),PurePosixPath('/home/agent/.codex/config.toml'))
    backend.write_to_container(worker,tool_catalog,PurePosixPath(CODEX_TOOL_CATALOG_PATH))
    check(backend,worker,'chown -R agent:agent /home/agent/.codex /home/agent/.config; chmod 700 /home/agent/.codex /home/agent/.config/sforge; chmod 600 /home/agent/.codex/config.toml '+destination,timeout=30)
    if credentials.mode=='api-key':
        api_probe='''import json,urllib.request,pathlib
key=pathlib.Path(KEY_PATH).read_text().strip()
body={'model':MODEL,'input':'Reply with the single word READY.','max_output_tokens':32}
req=urllib.request.Request(URL+'/responses',data=json.dumps(body).encode(),headers={'Content-Type':'application/json','Authorization':'Bearer '+key})
o=urllib.request.build_opener(urllib.request.ProxyHandler({}))
with o.open(req,timeout=90) as response: data=json.load(response)
assert data.get('object')=='response' or 'output' in data, 'Endpoint did not return Responses API format'
print('Worker model Responses API verified')
'''.replace('KEY_PATH',repr(API_KEY_PATH)).replace('MODEL',repr(model['MODEL'])).replace('URL',repr(model['MODEL_API_BASE_URL'].rstrip('/')))
        print(check(backend,worker,['python3','-c',api_probe],timeout=110),flush=True)
    cli_probe='''import os,pathlib,subprocess,json
env=dict(os.environ);env.update(HOME='/home/agent',USER='agent',LOGNAME='agent')
env['PATH']='/opt/agent-runtime/bin:'+env['PATH']
for name in ('OPENAI_API_KEY','OPENAI_BASE_URL','OPENAI_API_BASE','CODEX_API_KEY'): env.pop(name,None)
if AUTH_MODE=='api-key': env['OPENAI_API_KEY']=pathlib.Path(KEY_PATH).read_text().strip()
cmd=['runuser','-u','agent','--','codex','exec','--json','--skip-git-repo-check','--dangerously-bypass-approvals-and-sandbox','--dangerously-bypass-hook-trust','--enable','hooks','--disable','multi_agent','--disable','code_mode','--disable','code_mode_only','-m',MODEL,'-c','model_reasoning_effort='+json.dumps(EFFORT),'Reply with exactly READY. Do not use tools or change files.']
result=subprocess.run(cmd,cwd='/tmp',env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=150)
# CODEX_JSON_VALIDATION
print('Pinned Codex CLI and configured model preflight passed')
'''.replace('# CODEX_JSON_VALIDATION',CODEX_PROBE_VALIDATION_SOURCE).replace('AUTH_MODE',repr(credentials.mode)).replace('KEY_PATH',repr(API_KEY_PATH)).replace('MODEL',repr(model['MODEL'])).replace('EFFORT',repr(model.get('REASONING_EFFORT','high')))
    print(check(backend,worker,['python3','-c',cli_probe],timeout=170),flush=True)
    return {'status':'passed','auth_mode':credentials.mode,'runtime_uploaded':True,
            'credentials_injected':True,'responses_api':credentials.mode=='api-key','codex_cli':True,
            'submission_wait_policy':'wait-for-result',
            **({'responses_api_reason':'not-applicable-chatgpt-auth'} if credentials.mode=='chatgpt' else {})}


# Embedded in the root supervisor. Only credential-field values are journaled;
# this private file is never part of any candidate or exported artifact.
SUPERVISOR_AUTH_SOURCE = r'''
auth_secret_history=set()
def remember_auth(strict=True):
    mode=cfg.get('auth_mode','api-key')
    if mode not in ('chatgpt','claude-oauth'): return
    auth_path='/home/agent/.claude/.credentials.json' if mode=='claude-oauth' else '/home/agent/.codex/auth.json'
    try: document=json.loads(pathlib.Path(auth_path).read_text())
    except (OSError,ValueError):
        if strict: raise RuntimeError('Cannot collect current Agent auth credentials') from None
        return
    prior=len(auth_secret_history)
    def visit(value):
        if isinstance(value,dict):
            for name,item in value.items():
                if str(name).lower() in ('access_token','refresh_token','id_token','api_key','openai_api_key','accesstoken','refreshtoken','idtoken','apikey','openaiapikey','anthropic_api_key') and isinstance(item,str) and item:
                    auth_secret_history.add(item)
                elif isinstance(item,(dict,list)): visit(item)
        elif isinstance(value,list):
            for item in value: visit(item)
    visit(document)
    if not auth_secret_history: raise RuntimeError('Agent auth contains no credentials to protect')
    if len(auth_secret_history)==prior and (ROOT/'auth-secrets.json').exists(): return
    fd=os.open(ROOT/'auth-secrets.json',os.O_WRONLY|os.O_CREAT|os.O_TRUNC,0o600)
    with os.fdopen(fd,'w') as stream: json.dump(sorted(auth_secret_history),stream)
'''
