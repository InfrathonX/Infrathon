"""Optional, CPU-only process/UID tests; no model credentials or GPU requests.

RUN_SFORGE_SUPERVISOR_DOCKER_TESTS=1 SFORGE_SUPERVISOR_TEST_IMAGE=sha256:... pytest ...
The image must already exist locally; these tests never pull or access a network.
"""
import hashlib
import io
import json
import os
import re
import tarfile
import uuid

import pytest

from sforge.pyromind.runner import supervisor_source
from sforge.harness.submit_wait import generate_submit_wait_hook_script


pytestmark = pytest.mark.skipif(
    os.environ.get('RUN_SFORGE_SUPERVISOR_DOCKER_TESTS') != '1',
    reason='Explicit opt-in required for an isolated, CPU-only Docker test')


FAKE_CLI = r'''
import json,os,pathlib,subprocess,sys
cwd=pathlib.Path.cwd()
attempt=int((cwd/'attempt').read_text())+1 if (cwd/'attempt').exists() else 1
(cwd/'attempt').write_text(str(attempt))
print(json.dumps({'type':'item.completed','item':{'type':'command_execution','exit_code':0}}),flush=True)
hook='/opt/sforge/submit-wait-hook.py'
event={'session_id':'test-session','tool_use_id':'ordinary-1','tool_name':'Bash',
       'cwd':str(cwd),'tool_input':{'command':'sleep 90'}}
if attempt==1:
 if sys.argv[1]=='submission':
  event['tool_use_id']='submit-1';event['tool_input']['command']='sforge-submit'
 result=subprocess.run(['python3',hook,'pre'],input=json.dumps(event),text=True,
                       capture_output=True,check=True)
 assert json.loads(result.stdout).get('hookSpecificOutput',{}).get('permissionDecision')!='deny',result.stdout
 if sys.argv[1]=='ordinary':
  child=subprocess.Popen(['python3','-c','import time; time.sleep(90)'],
                         start_new_session=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
  (cwd/'child.pid').write_text(str(child.pid))
else:
 # A failed ordinary command still has a terminal Post callback. This next
 # invocation also proves the root cancellation did not strand a root-owned gate.
 event['tool_use_id']='ordinary-2';event['tool_input']['command']='false'
 pre=subprocess.run(['python3',hook,'pre'],input=json.dumps(event),text=True,capture_output=True,check=True)
 assert json.loads(pre.stdout).get('hookSpecificOutput',{}).get('permissionDecision')!='deny',pre.stdout
 event['tool_response']={'exit_code':1,'output':'expected failed experiment'}
 subprocess.run(['python3',hook,'post'],input=json.dumps(event),text=True,capture_output=True,check=True)
 (cwd/'resumed.json').write_text(json.dumps({'notice':os.environ.get('SFORGE_CANCELLED_TOOLS_NOTICE'),
                                         'remaining':int(os.environ['SFORGE_REMAINING_SECONDS'])}))
print(json.dumps({'type':'turn.completed'}))
'''


DRIVER = r'''
import hashlib,json,os,pathlib,pwd,subprocess,sys,time
try: account=pwd.getpwnam('agent')
except KeyError:
 subprocess.run(['useradd','-m','-s','/bin/bash','agent'],check=True)
 account=pwd.getpwnam('agent')
assert account.pw_uid>0
root=pathlib.Path('/run/sforge-agent');root.mkdir(mode=0o700)
gate=pathlib.Path('/run/sforge-submit-wait');gate.mkdir(mode=0o700)
os.chown(gate,account.pw_uid,account.pw_gid)
cwd=pathlib.Path('/tmp/candidate');cwd.mkdir()
os.chown(cwd,account.pw_uid,account.pw_gid)
(cwd/'candidate.py').write_text('answer = 42\n')
os.chown(cwd/'candidate.py',account.pw_uid,account.pw_gid)
key=pathlib.Path('/home/agent/.config/sforge/model.key');key.parent.mkdir(parents=True,exist_ok=True)
key.write_text('synthetic-test-key');key.chmod(0o600);os.chown(key,account.pw_uid,account.pw_gid)
target=pathlib.Path('/opt/sforge');target.mkdir(parents=True,exist_ok=True)
(target/'submit-wait-hook.py').write_bytes(pathlib.Path('/verification/hook.py').read_bytes())
submit=pathlib.Path('/usr/local/bin/sforge-submit')
submit.write_text('#!/bin/sh\nprintf "synthetic score\\n"\n');submit.chmod(0o755)
scenario=sys.argv[1]
command='python3 /verification/fake_cli.py '+scenario
cfg={'seconds':13,'deadline_at':time.time()+13,'cwd':str(cwd),'environment':{},
     'auth_mode':'api-key','first_command':command,'resume_command':command,
     'submit_paths':['candidate.py'],'submit_exclude':[]}
(root/'supervisor.json').write_text(json.dumps(cfg))
run=subprocess.run(['python3','/verification/supervisor.py'],capture_output=True,text=True,timeout=40)
assert run.returncode==0,run.stderr
status=json.loads((root/'status.json').read_text())
state=json.loads((gate/'state.json').read_text())
archive=(root/'final.tar.gz').read_bytes()
assert status['candidate_archive_sha256']==hashlib.sha256(archive).hexdigest()
assert status['candidate_archive_bytes']==len(archive)
assert status['no_agent_processes'] is True
assert status['deadline_at']==cfg['deadline_at']
if scenario=='ordinary':
 assert status['status']=='completed',status
 assert status['ordinary_tool_recoveries']==1,status
 assert status['last_tool_recovery']['outcome']=='cancelled_at_writer_boundary'
 assert int((cwd/'attempt').read_text())>=2
 resumed=json.loads((cwd/'resumed.json').read_text())
 assert 'unknown' in resumed['notice'] and 0<resumed['remaining']<13
 assert state['active'] is None and not state['active_tools'],state
 assert (gate/'state.json').stat().st_uid==account.pw_uid
 # The real detached child must have exited, not just been removed from ledger.
 child=pathlib.Path('/proc')/(cwd/'child.pid').read_text()/'status'
 if child.exists(): assert '\nState:\tZ' in child.read_text()
else:
 assert status['status']=='failed',status
 assert 'unfinished submission' in status['error']
 assert not status.get('ordinary_tool_recoveries')
 assert int((cwd/'attempt').read_text())==1
 assert state['active']['result_ready'] is True
 assert (gate/('result-'+state['active']['result_id']+'.json')).exists()
print(json.dumps({'scenario':scenario,'status':status['status'],
                  'ordinary_tool_recoveries':status.get('ordinary_tool_recoveries',0),
                  'archive_verified':True,'no_agent_processes':True,
                  'original_deadline_preserved':True,'real_model_called':False}))
'''


@pytest.mark.parametrize('scenario', ['ordinary', 'submission'])
def test_real_root_uid_boundary_and_failure_snapshot(scenario, tmp_path):
    import docker
    image = os.environ.get('SFORGE_SUPERVISOR_TEST_IMAGE', '')
    assert re.fullmatch(r'sha256:[a-f0-9]{64}', image), 'Use a cached immutable image ID'
    client = docker.from_env()
    client.images.get(image)  # Local lookup only; no pull.
    files = {'driver.py': DRIVER, 'fake_cli.py': FAKE_CLI,
             'supervisor.py': supervisor_source(), 'hook.py': generate_submit_wait_hook_script()}
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode='w') as bundle:
        for name, text in files.items():
            data = text.encode()
            info = tarfile.TarInfo('verification/' + name)
            info.size, info.mode = len(data), 0o644
            bundle.addfile(info, io.BytesIO(data))
    container = client.containers.create(
        image, name='sforge-supervisor-test-' + uuid.uuid4().hex[:12], user='root',
        entrypoint=['python3', '-I', '/verification/driver.py', scenario],
        network_mode='none', runtime='runc', device_requests=[],
        environment={'NVIDIA_VISIBLE_DEVICES':'void','CUDA_VISIBLE_DEVICES':''},
        nano_cpus=500_000_000, mem_limit='512m', labels={'sforge.test':'supervisor-recovery'})
    try:
        assert container.put_archive('/', payload.getvalue())
        container.start()
        result = container.wait(timeout=55)
        log = container.logs().decode()
        (tmp_path / (scenario + '.log')).write_text(log)
        assert result['StatusCode'] == 0, log
        receipt = json.loads(log.splitlines()[-1])
        assert receipt['archive_verified'] and receipt['original_deadline_preserved']
    finally:
        container.remove(force=True)
        client.close()
