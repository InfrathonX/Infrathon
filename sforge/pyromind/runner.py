"""Persistent Worker, native SForge submission service, ephemeral Judges.

The local service only orchestrates. Every evaluation, including final holdout,
runs in a new remote Sandbox with its role's resources. The Worker supervisor enforces a process/UID writer
boundary; it does not claim Docker's whole-container frozen-snapshot contract.
"""
from __future__ import annotations

import hashlib
from contextlib import contextmanager
import importlib.util
import io
import json
import math
import os
import re
from pathlib import Path, PurePosixPath
import shlex
import signal
import socket
import subprocess
import tarfile
import threading
import time

from sforge.pyromind.cli import checked, write_json
from sforge.harness.agent_config import (
    SUPERVISOR_AUTH_SOURCE, collect_worker_auth_secrets, install_and_validate_agent,
    load_credentials, load_model_config, redact,
)


def duration_seconds(hours: float) -> int:
    if isinstance(hours, bool) or not math.isfinite(hours) or hours <= 0 or hours > 24:
        raise ValueError("hours must be finite and in (0, 24]")
    return max(1, int(hours * 3600))


@contextmanager
def private_runtime_diagnostics(output: Path, sandbox_id: str):
    try:
        yield
    except Exception as error:
        detail = getattr(error, 'runtime_diagnostics', None)
        if detail:
            write_json(output/'runtime-diagnostics'/f'{sandbox_id}.json',
                       {'error': str(error), 'diagnostics': detail})
        raise


def initialize_task_store(root: Path, run_id: str, task_id: str) -> None:
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / ".sforge-version-store.json", {
        "schema_version": 1, "kind": "sforge-version-store", "store_id": run_id})
    task_root = root / "tasks" / task_id
    (task_root / "runs").mkdir(parents=True)
    write_json(task_root / "task.json", {"schema_version": 1, "task_id": task_id,
                                         "task_directory": task_id})


def public_artifact_paths(output: Path):
    """Never publish private verifier files, service credentials, or transcripts."""
    for path in sorted((output / "version-store").rglob("*")):
        if (path.is_file() and not path.is_symlink()
                and not any(part.startswith(".tmp-") for part in path.parts)
                and not path.name.endswith(".lock")):
            yield path
    for name in ("run.json", "history.json", "final.json", "final-snapshot.json", "failure-snapshot.json",
                 "submit-wait-diagnostics.json", "recovery-verification.json"):
        path = output / name
        if path.is_file() and not path.is_symlink():
            yield path


class ArtifactMirror:
    """Controller-only Storage credentials; Worker receives a read-only mount."""
    def __init__(self, output: Path, prefix: str, *, client=None, bucket=None):
        if client is None:
            from sforge.pyromind.storage import storage_client
            client, selected_bucket = storage_client()
            bucket = bucket or selected_bucket
        self.client = client
        self.bucket = bucket
        if not bucket:
            raise ValueError('Artifact bucket is required with an injected client')
        self.output, self.prefix = output, prefix
        self.seen = {}
        self.lock = threading.Lock()

    def sync(self):
        with self.lock:
            self._sync()

    def _sync(self):
        for path in public_artifact_paths(self.output):
            data = path.read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            relative = path.relative_to(self.output).as_posix()
            if self.seen.get(relative) == digest:
                continue
            try:
                self.client.put_object(self.bucket, self.prefix + "/" + relative,
                                       io.BytesIO(data), len(data))
            except Exception as exc:
                raise RuntimeError('Artifact Storage upload failed ('+type(exc).__name__+')') from None
            self.seen[relative] = digest

    def verify_versions(self, versions):
        for version in versions:
            key = self.prefix + "/" + version.archive_path.relative_to(self.output).as_posix()
            response = self.client.get_object(self.bucket, key)
            try:
                digest = hashlib.sha256(response.read()).hexdigest()
            finally:
                response.close()
                response.release_conn()
            if digest != version.archive_sha256:
                raise RuntimeError("Persisted version digest mismatch")


def controller_url(port: int) -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        probe.connect(("8.8.8.8", 53))
        return f"http://{probe.getsockname()[0]}:{port}"


def verify_shared_version(backend, worker, version, output, mirror):
    """Compare the immutable bytes through all three independently read paths."""
    mirror.sync()
    mirror.verify_versions([version])
    remote_path = '/shared/' + version.archive_path.relative_to(output).as_posix()
    probe = """import hashlib,json,os,time
from pathlib import Path
path=Path(ARCHIVE_PATH)
deadline=time.monotonic()+30
while True:
    try:
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if digest==EXPECTED_DIGEST: break
    except FileNotFoundError: pass
    if time.monotonic()>=deadline: raise RuntimeError('Worker shared archive is absent or has a different digest')
    time.sleep(1)
readonly=bool(os.statvfs('/shared').f_flag & os.ST_RDONLY)
assert readonly, 'Worker artifact mount must be read-only'
print(json.dumps({'archive_sha256':digest,'mount_read_only':readonly}))
""".replace('ARCHIVE_PATH',repr(remote_path)).replace('EXPECTED_DIGEST',repr(version.archive_sha256))
    visible=json.loads(checked(backend,worker,['runuser','-u','agent','--','python3','-c',probe],timeout=45))
    if visible.get('archive_sha256') != version.archive_sha256 or visible.get('mount_read_only') is not True:
        raise RuntimeError('Worker shared archive verification failed')
    evidence={'worker_id':worker.id,'worker_shared_path':remote_path,
              'local_archive_sha256':version.archive_sha256,'storage_readback_verified':True,
              'worker_shared_archive_sha256':visible['archive_sha256'],
              'worker_mount_read_only':visible['mount_read_only']}
    write_json(output/'shared-version-verification.json',evidence)
    return evidence


WORKER_INTERNET_PROBE = r"""import json,urllib.request
result={}
for url in ('https://pypi.org/simple/pip/','https://github.com/'):
    try:
        with urllib.request.urlopen(urllib.request.Request(url,method='HEAD'),timeout=15) as response:
            result[url]=response.status
    except Exception as error:
        result[url]=type(error).__name__
print(json.dumps(result))
"""


def probe_worker_internet(backend, worker):
    """Record whether the Worker reaches the public internet; never fails the run.

    Every task allows internet since 2026-10-07. PyroMind egress is the platform
    default, so this is observed evidence, not an enforced guarantee.
    """
    try:
        output = checked(backend, worker, ['python3', '-c', WORKER_INTERNET_PROBE], timeout=60)
        results = json.loads(output.strip().splitlines()[-1])
    except Exception as error:
        results = {'probe_error': type(error).__name__}
    reachable = any(value == 200 for value in results.values())
    print('Worker internet ' + ('reachable' if reachable else 'NOT reachable') + ': ' + json.dumps(results), flush=True)
    return {'reachable': reachable, 'results': results}


def supervisor_source() -> str:
    """Root supervisor; all agent processes use one nonprivileged UID."""
    return (r'''import hashlib, json, os, pathlib, pwd, signal, subprocess, time
ROOT = pathlib.Path('/run/sforge-agent')
cfg = json.loads((ROOT/'supervisor.json').read_text())
uid = pwd.getpwnam('agent').pw_uid
assert uid != 0
started = time.time()
deadline_at = cfg.get('deadline_at', started + cfg['seconds'])
remaining = max(0, min(cfg['seconds'], deadline_at-started))
deadline = time.monotonic() + remaining
status = {'status':'running','started_at':started,'deadline_at':deadline_at,
          'budget_seconds':cfg['seconds'],'attempts':0,'writer_boundary':'agent_uid_and_process_group'}
if 'deadline_at' in cfg: status['remaining_seconds_at_start'] = remaining
# AUTH_JOURNAL_HELPER
def publish():
    p=ROOT/'status.tmp'; p.write_text(json.dumps(status)); p.replace(ROOT/'status.json')
def live():
    found=[]
    for p in pathlib.Path('/proc').glob('[0-9]*'):
        try:
            s=(p/'status').read_text().splitlines()
            owners=next(x for x in s if x.startswith('Uid:')).split()[1:]
            state=next(x for x in s if x.startswith('State:')).split()[1]
            if str(uid) in owners and state != 'Z': found.append(int(p.name))
        except (FileNotFoundError, ProcessLookupError): pass
    return found
def stop_writers():
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for pid in live():
            try: os.kill(pid,sig)
            except ProcessLookupError: pass
        until=time.monotonic()+5
        while live() and time.monotonic()<until: time.sleep(.1)
    if live(): raise RuntimeError('Agent writer processes remain; refusing final archive')
def recover_ordinary_tools():
    # This runs only between CLI invocations. Killing detached work establishes
    # cancellation, never a successful tool result or delivery of a submission.
    stop_writers()
    recovered=subprocess.run(['python3','/opt/sforge/submit-wait-hook.py',
                              'retire-ordinary-at-boundary',str(uid)],
                             env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=30)
    if recovered.returncode != 0:
        raise RuntimeError('Cannot establish an ordinary-tool cancellation boundary')
    record=json.loads(recovered.stdout)
    if record.get('outcome') != 'cancelled_at_writer_boundary' or not isinstance(record.get('retired_tools'),list):
        raise RuntimeError('Invalid ordinary-tool cancellation receipt')
    status['ordinary_tool_recoveries']=status.get('ordinary_tool_recoveries',0)+1
    status['last_tool_recovery']=record
    env['SFORGE_CANCELLED_TOOLS_NOTICE']='Previous unfinished ordinary tools were cancelled by the supervisor. Their results are unknown; re-run any experiment you still need.'
    publish()
def capture_candidate():
    stop_writers()
    remember_auth()
    status['writers_stopped_at']=time.time()
    subprocess.run(['chown','-hR','root:root',cfg['cwd']],check=True,timeout=30)
    temporary=ROOT/'final.tmp.tar.gz'
    args=['tar','czf',str(temporary),'--exclude=.git']
    args += ['--exclude='+p for p in cfg['submit_exclude']]
    args += [p for p in cfg['submit_paths'] if pathlib.Path(cfg['cwd'],p).exists()]
    if len(args)==4+len(cfg['submit_exclude']): args += ['--files-from','/dev/null']
    subprocess.run(args,cwd=cfg['cwd'],check=True,timeout=180)
    if live(): raise RuntimeError('Agent process reappeared after writer boundary')
    digest=hashlib.sha256()
    with temporary.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024),b''): digest.update(chunk)
    size=temporary.stat().st_size
    temporary.replace(ROOT/'final.tar.gz')
    status.update(no_agent_processes=True,candidate_archive_available=True,
                  candidate_archive_sha256=digest.hexdigest(),candidate_archive_bytes=size)
publish()
env=dict(os.environ)
env.update(cfg['environment'])
env['SFORGE_AGENT_DEADLINE_MONOTONIC']=str(deadline)
env.update(HOME='/home/agent',USER='agent',LOGNAME='agent')
env['PATH']='/opt/agent-runtime/bin:'+env.get('PATH','')
for name in tuple(env):
    if name.startswith(('ANTHROPIC','CLAUDE')) or name in ('OPENAI_API_KEY','OPENAI_BASE_URL','OPENAI_API_BASE','CODEX_API_KEY'): env.pop(name,None)
if cfg.get('auth_mode','api-key')=='api-key':
    env['OPENAI_API_KEY']=pathlib.Path('/home/agent/.config/sforge/model.key').read_text().strip()
elif cfg.get('auth_mode')=='anthropic-api-key':
    env['ANTHROPIC_API_KEY']=pathlib.Path('/home/agent/.config/sforge/model.key').read_text().strip()
    env['ANTHROPIC_BASE_URL']=cfg['model_api_base_url']
    env['CLAUDE_CODE_EFFORT_LEVEL']=cfg['reasoning_effort']
    env['CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC']='1'
elif cfg.get('auth_mode')=='claude-oauth':
    # Native ~/.claude/.credentials.json retains refreshToken so the CLI can
    # renew access during a long run. Never reinterpret it as an API key or
    # route subscription OAuth to a custom model endpoint.
    env['CLAUDE_CODE_EFFORT_LEVEL']=cfg['reasoning_effort']
    env['CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC']='1'
env.pop('PYROMIND_API_KEY',None)
env.pop('SFORGE_ADMIN_SECRET',None)
fast_failures=0
empty_turns=0
try:
    remember_auth()
    with (ROOT/'agent.jsonl').open('ab',buffering=0) as logs:
        while time.monotonic()<deadline:
            wait_state=pathlib.Path('/run/sforge-submit-wait/state.json')
            if wait_state.exists():
                proof=subprocess.run(['runuser','-u','agent','--','python3','/opt/sforge/submit-wait-hook.py','reconcile'],
                                     env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=30)
                if proof.returncode != 0:
                    raise RuntimeError('Invalid submission wait state or proof; refusing to resume Agent')
                pending=json.loads(wait_state.read_text())
                if (not isinstance(pending,dict) or pending.get('schema_version')!=1 or 'active' not in pending
                        or not isinstance(pending.get('active_tools'),dict)):
                    raise RuntimeError('Invalid submission wait state; refusing to resume Agent')
                if pending['active'] is not None:
                    raise RuntimeError('Previous Codex invocation stopped with an unfinished submission tool; refusing to resume before its result is delivered')
                if pending['active_tools']:
                    recover_ordinary_tools()
            # Recovery time belongs to the original budget. Never admit a new
            # CLI after cancellation or a boundary operation crossed deadline.
            if time.monotonic() >= deadline or (ROOT/'stop').exists(): break
            status['attempts']+=1; publish()
            command=cfg['first_command'] if status['attempts']==1 else cfg['resume_command']
            segment=time.monotonic()
            log_start=logs.tell()
            env['SFORGE_REMAINING_SECONDS']=str(max(0,int(deadline-time.monotonic())))
            proc=subprocess.Popen(['runuser','-u','agent','--','/bin/bash','-c',command],
                                  cwd=cfg['cwd'],env=env,stdout=logs,stderr=subprocess.STDOUT,start_new_session=True)
            while proc.poll() is None and time.monotonic()<deadline:
                if (ROOT/'stop').exists(): deadline=time.monotonic(); break
                remember_auth(strict=False)
                time.sleep(.5)
            if proc.poll() is None:
                try: os.killpg(proc.pid,signal.SIGTERM)
                except ProcessLookupError: pass
                try: proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try: os.killpg(proc.pid,signal.SIGKILL)
                    except ProcessLookupError: pass
                    proc.wait(timeout=5)
            remember_auth()
            status['last_exit_code']=proc.returncode
            used_tools=False
            with (ROOT/'agent.jsonl').open('rb') as transcript:
                transcript.seek(log_start)
                for line in transcript:
                    try: event=json.loads(line)
                    except (ValueError,UnicodeDecodeError): continue
                    # stdout and stderr share this transcript. A diagnostic's
                    # patch line (e.g. -46) is valid JSON, but not a CLI event.
                    if not isinstance(event,dict): continue
                    item=event.get('item')
                    if isinstance(item,dict) and item.get('type') in ('command_execution','file_change','mcp_tool_call','web_search'):
                        used_tools=True
                    message=event.get('message')
                    if event.get('type')=='assistant' and isinstance(message,dict):
                        blocks=message.get('content')
                        if isinstance(blocks,list) and any(isinstance(block,dict) and block.get('type')=='tool_use'
                                                          for block in blocks):
                            used_tools=True
            # A failed experiment is Agent work. Only repeated CLI failures
            # before any tool work count as inability to launch the runtime.
            if time.monotonic()-segment<20 and proc.returncode != 0 and not used_tools:
                fast_failures+=1
                if fast_failures>=3: raise RuntimeError('Agent CLI repeatedly failed before doing work')
            else: fast_failures=0
            empty_turns=0 if used_tools else empty_turns+1
            status['consecutive_no_tool_turns']=empty_turns
            status['tool_turns']=status.get('tool_turns',0)+int(used_tools)
            status['no_tool_turns']=status.get('no_tool_turns',0)+int(not used_tools)
            publish()
            # Keep the original absolute budget; avoid hundreds of empty resume calls.
            delay=5 if used_tools else min(300,15*2**min(empty_turns,5))
            until=min(deadline,time.monotonic()+delay)
            while time.monotonic()<until:
                if (ROOT/'stop').exists(): deadline=time.monotonic(); break
                time.sleep(min(.5,max(0,until-time.monotonic())))
    capture_candidate()
    status.update(status='completed',finished_at=time.time(),no_agent_processes=True)
except BaseException as exc:
    status.update(status='failed',error=type(exc).__name__+': '+str(exc),finished_at=time.time())
    try:
        capture_candidate()
    except BaseException as preservation_error:
        status['candidate_preservation_error']=type(preservation_error).__name__
finally:
    publish()
'''.replace('# AUTH_JOURNAL_HELPER', SUPERVISOR_AUTH_SOURCE))


def _remote_json(backend, worker, path):
    return json.loads(backend.client.read_file(worker.id, path))


def preserve_failed_worker_candidate(backend, worker, output, task, store, mirror,
                                    credentials, known_secrets, run_id, progress):
    """Persist an interrupted snapshot without marking the run's budget complete.

    The Supervisor's stopped-UID boundary and archive digest must both hold.
    An interrupted checkpoint does not close the VersionStore's final tombstone.
    """
    if (progress.get('status') != 'failed' or progress.get('no_agent_processes') is not True
            or progress.get('candidate_archive_available') is not True):
        raise RuntimeError('Failed Worker has no verified candidate writer boundary')
    collect_worker_auth_secrets(backend, worker, credentials, known_secrets,
                               required=True, journal_required=True)
    archive = backend.client.read_file(worker.id, '/run/sforge-agent/final.tar.gz')
    digest = hashlib.sha256(archive).hexdigest()
    if (digest != progress.get('candidate_archive_sha256')
            or len(archive) != progress.get('candidate_archive_bytes')):
        raise RuntimeError('Interrupted Worker archive differs from its boundary receipt')
    store.validate_archive(task_id=task.task_id, archive=archive,
                           submit_paths=task.submit_paths, submit_exclude=task.submit_exclude)
    validate_no_credentials(archive, known_secrets)
    version = store.record_archive(
        task_id=task.task_id, run_id=run_id, source='checkpoint', round_id='supervisor-failure',
        label='supervisor-failure', archive=archive, submit_paths=task.submit_paths,
        submit_exclude=task.submit_exclude, content_identity_exclude=task.candidate_identity_exclude,
        metadata={'writer_boundary': 'agent_uid_and_process_group', 'worker_id': worker.id,
                  'budget_completed': False, 'termination_reason': 'supervisor-failure'})
    receipt = {'version_id': version.version_id, 'archive_sha256': digest,
               'archive_bytes': len(archive), 'worker_id': worker.id,
               'writer_boundary': 'agent_uid_and_process_group', 'budget_completed': False,
               'local_archive_saved': True, 'storage_readback_verified': False}
    write_json(output / 'failure-snapshot.json', receipt)
    mirror.sync()
    mirror.verify_versions([version])
    receipt['storage_readback_verified'] = True
    write_json(output / 'failure-snapshot.json', receipt)
    mirror.sync()
    return receipt


def submit_wait_diagnostics(document):
    """Export scheduling metadata only, never commands or cached tool output."""
    if (not isinstance(document,dict) or type(document.get('schema_version')) is not int
            or document.get('schema_version') != 1
            or 'active' not in document or not isinstance(document.get('active_tools'),dict)
            or len(document['active_tools']) > 32):
        raise ValueError('Invalid submission wait diagnostic state')
    def metadata(record):
        if not isinstance(record,dict):
            raise ValueError('Invalid submission wait diagnostic entry')
        result={}
        for key in ('session_id','tool_use_id','tool_name','outcome'):
            value=record.get(key)
            if value is not None:
                if not isinstance(value,str) or not re.fullmatch(r'[A-Za-z0-9_.:-]{1,240}',value):
                    raise ValueError('Invalid submission wait diagnostic identity')
                result[key]=value
        for key in ('started_at','finished_at'):
            value=record.get(key)
            if value is not None:
                if isinstance(value,str):
                    from datetime import datetime
                    if not re.fullmatch(r'[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{3}Z',value):
                        raise ValueError('Invalid submission wait diagnostic timestamp')
                    try:
                        datetime.strptime(value,'%Y-%m-%dT%H:%M:%S.%fZ')
                    except ValueError:
                        raise ValueError('Invalid submission wait diagnostic timestamp') from None
                elif isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value <= 0:
                    raise ValueError('Invalid submission wait diagnostic timestamp')
                result[key]=value
        if 'timeout_ms' in record:
            value=record['timeout_ms']
            if type(value) is not int or not 0 < value <= 172800000:
                raise ValueError('Invalid submission wait diagnostic timeout')
            result['timeout_ms']=value
        if 'result_ready' in record:
            if not isinstance(record['result_ready'],bool):
                raise ValueError('Invalid submission wait diagnostic result status')
            result['result_ready']=record['result_ready']
        return result
    return {'schema_version':1,
            'active_submission':None if document['active'] is None else metadata(document['active']),
            'pending_tools':[metadata(record) for record in document['active_tools'].values()],
            'last_completion':None if document.get('last_completion') is None else metadata(document['last_completion'])}


def asset_bindings_for_role(bindings, role):
    """Select already deployment-validated rows without changing literal identities."""
    if role not in ('worker', 'judge'):
        raise ValueError('Unknown asset consumer role')
    selected=[]
    for item in bindings:
        private=item.get('visibility', 'public')=='private'
        consumers=item.get('consumers', ['worker', 'judge'])
        allowed=({frozenset({'judge'})} if private else
                 {frozenset({'worker', 'judge'}), frozenset({'worker'})})
        if (item.get('visibility', 'public') not in ('public', 'private')
                or not isinstance(consumers, list) or not all(isinstance(c,str) for c in consumers)
                or len(consumers)!=len(set(consumers)) or frozenset(consumers) not in allowed):
            raise ValueError('Asset consumers are not deployment-validated roles')
        if role in consumers:selected.append(item)
    return selected


def asset_mounts_for_role(bindings, role, mount_type):
    return [mount_type(host_path='/workspace/'+item['object_prefix'],
            mount_path=item['mount_path'],read_only=True)
            for item in asset_bindings_for_role(bindings, role)]


def verify_asset_mounts(backend, handle, bindings, evidence_path, *, private=False):
    """Read public inputs as Agent; private Judge inputs as its trusted root."""
    if not bindings:
        return []
    probe = r'''import hashlib,json,os,pathlib,stat,sys
bindings=json.loads(sys.argv[1]); results=[]
for binding in bindings:
    root=pathlib.Path(binding['mount_path'])
    assert os.statvfs(root).f_flag & os.ST_RDONLY, 'Asset mount must be read-only'
    expected={record['path'] for record in binding['files']}; observed=set()
    assert len(expected)==len(binding['files']), 'Duplicate asset file'
    assert not root.is_symlink(), 'Asset root must not be a symlink'
    for directory, dirs, files in os.walk(root,followlinks=False):
        for name in dirs:
            assert not (pathlib.Path(directory)/name).is_symlink(), 'Asset directory symlink'
        for name in files:
            path=pathlib.Path(directory)/name
            assert stat.S_ISREG(path.lstat().st_mode), 'Asset entry must be a regular file'
            relative=path.relative_to(root).as_posix()
            assert relative in expected, 'Unlisted file in mounted asset tree'
            observed.add(relative)
    assert observed==expected, 'Mounted asset tree is incomplete'
    verified=[]
    for record in binding['files']:
        path=root/record['path']; assert path.is_file() and not path.is_symlink()
        h=hashlib.sha256(); size=0
        with path.open('rb') as stream:
            while True:
                block=stream.read(8*1024*1024)
                if not block: break
                h.update(block); size+=len(block)
        assert size==record['size'] and h.hexdigest()==record['sha256'], 'Mounted asset integrity mismatch'
        verified.append(record['path'])
    results.append({'mount_path':str(root),'read_only':True,'uid':os.getuid(),
                    'manifest_sha256':binding['manifest_sha256'],'verified_files':verified})
print('SFORGE_ASSETS='+json.dumps(results))
'''
    command = ['python3','-c',probe,json.dumps(bindings)]
    if not private:
        command = ['runuser','-u','agent','--',*command]
    result = backend.exec_run_with_timeout(handle, command, timeout=1800)
    if result.exit_code or result.timed_out:
        raise RuntimeError('Shared read-only model/data verification failed: '+result.output[-1500:])
    records = [line.partition('=')[2] for line in result.output.splitlines()
               if line.startswith('SFORGE_ASSETS=')]
    if len(records) != 1:
        raise RuntimeError('Shared asset verification returned no unique record')
    evidence = json.loads(records[0])
    write_json(evidence_path,evidence)
    return evidence


def capture_host_memory(backend, handle, output):
    """Record cgroup limits/peak/OOM/reclaim counters and CPU throttling;
    never read candidate/private files.  memory.stat is limited to the
    composition (anon/file/...) and the cumulative reclaim counters."""
    probe = r'''import json,pathlib
root=pathlib.Path('/sys/fs/cgroup');result={}
for line in pathlib.Path('/proc/self/cgroup').read_text().splitlines():
    if line.startswith('0::'):
        relative=line[3:].lstrip('/')
        candidate=root/relative
        if '..' not in pathlib.PurePosixPath(relative).parts and (candidate/'memory.max').is_file():
            root=candidate
for name in ['memory.max','memory.current','memory.peak','memory.events','cpu.max','cpu.stat']:
    p=root/name
    if p.is_file(): result[name]=p.read_text().strip()
keep={'anon','file','kernel','shmem','file_mapped','sock','pgscan','pgsteal','pgmajfault',
      'workingset_refault_anon','workingset_refault_file'}
p=root/'memory.stat'
if p.is_file():
    result['memory.stat']='\n'.join(line for line in p.read_text().splitlines()
                                    if line.split(' ',1)[0] in keep)
print(json.dumps(result))
'''
    result=backend.exec_run_with_timeout(handle,['python3','-c',probe],timeout=20)
    if result.exit_code or result.timed_out:
        raise RuntimeError('Cannot capture Sandbox cgroup resource evidence')
    metrics=json.loads(result.output.strip())
    write_json(output,metrics)
    return metrics


def verify_qualification_memory(output):
    paths=list((output/'resource-usage').glob('*.json'))
    if not paths:
        raise RuntimeError('Qualification lacks cgroup memory evidence')
    for path in paths:
        metrics=json.loads(path.read_text())
        if not {'memory.max','memory.peak','memory.events'} <= metrics.keys():
            raise RuntimeError('Qualification cgroup memory evidence is incomplete')
        if metrics['memory.max']=='max' or int(metrics['memory.peak'])<=0:
            raise RuntimeError('Qualification requires a finite observed memory allocation')
        counters=dict(line.split() for line in metrics['memory.events'].splitlines())
        if any(int(counters.get(name,'0')) for name in ('oom','oom_kill','oom_group_kill')):
            raise RuntimeError('Qualification encountered a host-memory OOM event')


def _private_module(repo):
    path = repo / "deploy/pyromind/kernel/bootstrap.py"
    spec = importlib.util.spec_from_file_location("sforge_pyromind_kernel_bootstrap", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _private_installer(repo):
    """Compatibility helper also used by the separate oracle qualifier."""
    return _private_module(repo).install_private_judge


def _drain(state, seconds):
    deadline = time.monotonic() + seconds
    while any(item['status'] in {'queued', 'running'} for item in state.submissions.values()):
        if time.monotonic() >= deadline:
            raise TimeoutError("Judge queue did not finish before the drain deadline")
        time.sleep(2)


def validate_no_credentials(archive: bytes, secrets) -> None:
    with tarfile.open(fileobj=io.BytesIO(archive), mode='r:gz') as bundle:
        for member in bundle:
            if not member.isfile():
                continue
            stream = bundle.extractfile(member)
            content = stream.read()
            if any(secret and secret.encode() in content for secret in secrets):
                raise ValueError('Credential material found in candidate archive')


def write_history(output: Path, history: dict) -> None:
    """Keep the full host history under logs/ (never mirrored); the mirrored,
    Worker-readable history.json omits the integrity-review audit."""
    from sforge.pyromind.integrity_review import worker_visible_history
    write_json(output/'logs'/'history-host.json', history)
    write_json(output/'history.json', worker_visible_history(history))


def evaluate_reviewed_final(*, judge_state, evaluate_submission, task_spec,
                            archive: bytes, stored_version, run_id: str,
                            round_id: str, log_dir: Path, **judge_kwargs):
    """Apply the configured agent review to the exact sealed final version.

    The final is agent-authored, even though its independent GPU Judge uses
    a separate run ID. Review against the original run's starting checkpoint
    and reuse the normal review cache, shadow mode, and error policy.
    """
    from sforge.harness.judge_server import SubmissionStatus
    from sforge.harness.llm_judge.reviewer import (
        IntegrityReview, STATUS_ERROR, _write_json,
    )

    if (stored_version.task_id != task_spec.task_id
            or stored_version.run_id != run_id
            or not isinstance(archive, bytes)
            or len(archive) != stored_version.archive_size
            or hashlib.sha256(archive).hexdigest() != stored_version.archive_sha256):
        raise ValueError('Final review bytes differ from the sealed task/run version')
    if task_spec.task_id not in judge_state.tasks:
        raise ValueError('Final review task is not loaded in the Judge service')
    submission_id = run_id + '-final'
    if submission_id in judge_state.submissions:
        raise ValueError('Final submission already exists')
    entry = {'status': SubmissionStatus.RUNNING, 'task_id': task_spec.task_id,
             'run_id': run_id, 'version_id': stored_version.version_id}
    judge_state.submissions[submission_id] = entry
    try:
        # Keep the original run identity for its baseline. The agent prefix
        # deliberately gives final archives the same configured review scope
        # as interactive submissions, without exposing a new bypass kind.
        judge_state._run_integrity_review(
            submission_id, task_spec.task_id, archive, run_id, 'agent-final')
        review = entry.get('integrity_review')
        review_config = judge_state.config.llm_judge
        if review_config.covers('agent') and not isinstance(review, IntegrityReview):
            # A covered review must return a result. An unexpected missing or
            # invalid result follows the error policy, never an implicit pass.
            # Off and deliberately uncovered kinds retain their normal path.
            review = IntegrityReview(
                review_id=submission_id, task_id=task_spec.task_id,
                mode=review_config.mode, status=STATUS_ERROR,
                blocking=(review_config.enforcing and review_config.on_error == 'block'),
                reason='the integrity review did not produce a valid result',
            )
            entry['integrity_review'] = review
        if isinstance(review, IntegrityReview):
            # Controller-only audit: bind the full verdict to the immutable
            # archive, independently of the review bundle/cache identity.
            _write_json(log_dir/'integrity_review'/'final-binding.json', {
                'schema_version': 1, 'task_id': task_spec.task_id,
                'run_id': run_id, 'submission_id': submission_id,
                'version_id': stored_version.version_id,
                'archive_sha256': stored_version.archive_sha256,
                'archive_bytes': len(archive), 'review': review.to_record()})
        if isinstance(review, IntegrityReview) and review.blocking:
            report = judge_state._integrity_terminal_report(
                task_spec, submission_id, archive, review, log_dir)
        else:
            judge_kwargs["evaluation_phase"] = "final"
            report = evaluate_submission(
                task_spec=task_spec, archive=archive,
                submission_id=submission_id, run_id=run_id+'-final',
                round_id=round_id, log_dir=log_dir,
                stored_version=stored_version, **judge_kwargs)
        entry.update(status=SubmissionStatus.COMPLETED, report=report.to_dict())
        return report
    except BaseException:
        entry['status'] = SubmissionStatus.ERROR
        raise


def guard_judge_backend(backend, cancelled, name_prefix=None, environment=None):
    """Cancel queued and running work; late create responses clean their exact ID."""
    create = backend.create_container
    execute = backend.exec_run_with_timeout
    def guarded_create(*args, **kwargs):
        if cancelled.is_set():
            raise RuntimeError('Controller is stopping; new Judge admission closed')
        if environment:
            kwargs['environment'] = {**(kwargs.get('environment') or {}), **environment}
        if name_prefix:
            if len(args) >= 2:
                args = (args[0], name_prefix + '-' + args[1], *args[2:])
            else:
                kwargs['name'] = name_prefix + '-' + kwargs['name']
        handle = create(*args, **kwargs)
        if cancelled.is_set():
            backend.cleanup_container(handle)
            raise RuntimeError('Controller stopped while Judge was being created')
        return handle
    def guarded_execute(*args, **kwargs):
        kwargs['shutdown_event'] = cancelled
        return execute(*args, **kwargs)
    backend.create_container = guarded_create
    backend.exec_run_with_timeout = guarded_execute
    backend.startup_cancelled = cancelled.is_set


def reconcile_run_sandboxes(backend, name_prefix):
    """Clean only this run's short, collision-resistant platform name prefix."""
    from sforge.harness.backend.pyromind_backend import PyroMindHandle
    if re.fullmatch(r'sf-[0-9a-f]{20}', name_prefix) is None:
        raise ValueError('Invalid run Sandbox name prefix')
    for sandbox in backend.client.list():
        if sandbox.name.startswith(name_prefix + '-'):
            backend.cleanup_container(PyroMindHandle(sandbox.id, sandbox.name))
    return [sandbox.id for sandbox in backend.client.list()
            if sandbox.name.startswith(name_prefix + '-')]


def reconcile_creation_ownership(backends):
    """Unknown accepted creates retain resource admission after LIST absence."""
    remaining, unresolved, errors = set(), set(), []
    for backend in backends:
        reconcile = getattr(backend, 'reconcile_creation_intents', None)
        if not callable(reconcile):
            continue
        try:
            receipt = reconcile()
            if (not isinstance(receipt, dict)
                    or not isinstance(receipt.get('remaining_sandbox_ids'), list)
                    or not isinstance(receipt.get('unresolved_creation_names'), list)
                    or not isinstance(receipt.get('cleanup_errors'), list)):
                raise RuntimeError('Sandbox creation cleanup receipt is invalid')
            remaining.update(receipt['remaining_sandbox_ids'])
            unresolved.update(receipt['unresolved_creation_names'])
            errors.extend(receipt['cleanup_errors'])
        except Exception as exc:
            errors.append('CreationOwnership:' + type(exc).__name__)
            names = getattr(backend, 'unresolved_creation_names', None)
            if callable(names):
                try:
                    unresolved.update(names())
                except Exception:
                    pass  # A failed journal read still retains cleanup_errors.
    errors.extend('unresolved-create:' + name for name in sorted(unresolved))
    return {'remaining_sandbox_ids': sorted(remaining),
            'unresolved_creation_names': sorted(unresolved),
            'cleanup_errors': sorted(set(errors))}


@contextmanager
def termination_cleanup():
    """Turn SIGTERM into normal owned-resource cleanup, restoring the caller."""
    installed = threading.current_thread() is threading.main_thread()
    previous = None
    if installed:
        previous = signal.getsignal(signal.SIGTERM)
        def terminate(signum, frame):
            # A second SIGTERM must not interrupt the bounded cleanup section.
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            raise KeyboardInterrupt('Controller received SIGTERM')
        signal.signal(signal.SIGTERM, terminate)
    try:
        yield
    finally:
        if installed:
            signal.signal(signal.SIGTERM, previous)


def run_task(args):
    with termination_cleanup():
        return _run_task(args)


def configure_runtime_cleanup(backend, private_module, task_root, output, *, role):
    """Capture bounded per-role diagnostics on failure as well as normal shutdown."""
    if role not in ('worker', 'judge'):
        raise ValueError('Unknown runtime cleanup role')
    collector_name = 'collect_worker_diagnostics' if role == 'worker' else 'collect_private_diagnostics'
    original_cleanup=backend.cleanup_container
    def cleanup_with_resource_evidence(handle,logger=None):
        try:
            if handle is not None:
                collector = getattr(private_module, collector_name, None)
                if collector:
                    private_root = output/'.private'
                    private_root.mkdir(parents=True, mode=0o700, exist_ok=True)
                    private_root.chmod(0o700)
                    directory = private_root/'diagnostics'/handle.id
                    directory.mkdir(parents=True, mode=0o700, exist_ok=True)
                    directory.chmod(0o700)
                    try:
                        collector(backend,handle,task_root,directory)
                    except Exception as error:
                        # Optional diagnostics must never prevent resource deletion.
                        write_json(directory/'capture-unavailable.json',{'error':type(error).__name__})
                capture_host_memory(backend,handle,output/'resource-usage'/f'{handle.id}.json')
        except Exception as error:
            if handle is not None:
                write_json(output/'resource-usage'/f'{handle.id}-unavailable.json',{'error':type(error).__name__})
        finally:
            original_cleanup(handle,logger)
    backend.cleanup_container=cleanup_with_resource_evidence


def configure_resource_capture(backend, directory):
    """Record a CPU-only (review) Sandbox's cgroup usage before deletion.

    ``directory`` is a subdirectory of ``resource-usage`` so qualification
    memory checks, which read only the Worker/Judge files, ignore it."""
    original_cleanup = backend.cleanup_container
    def cleanup_with_resource_evidence(handle, logger=None):
        try:
            if handle is not None:
                capture_host_memory(backend, handle, directory/f'{handle.id}.json')
        except Exception as error:
            if handle is not None:
                write_json(directory/f'{handle.id}-unavailable.json', {'error': type(error).__name__})
        finally:
            original_cleanup(handle, logger)
    backend.cleanup_container = cleanup_with_resource_evidence


def configure_judge_runtime(judge_backend, deployment, private_module, task_root, output,
                            bindings, private_metadata, private_bindings=()):
    """Same role runtime, asset and private-payload checks for each evaluation."""
    from sforge.pyromind.deployment import verify_gpu_inventory
    from sforge.pyromind.gpu_runtime import (capture_gpu_topology, prepare_hami_runtime,
                                            prepare_private_tmp, verify_distributed_runtime)
    resources = deployment.resources_for_role('judge')
    bindings = asset_bindings_for_role(bindings, 'judge')
    private_bindings = asset_bindings_for_role(private_bindings, 'judge')
    install_private = private_module.install_private_judge
    configure_runtime_cleanup(judge_backend, private_module, task_root, output, role='judge')
    def prepare_judge(handle):
        platform_runtime = None
        gpu = None
        with private_runtime_diagnostics(output,handle.id):
            if deployment.execution_kind_for_role('judge') == 'gpu':
                platform_runtime=prepare_hami_runtime(judge_backend,handle,gpu_count=resources['gpu'])
            private_tmp=prepare_private_tmp(judge_backend,handle) if deployment.config.get('private_tmp_acl') else None
        if deployment.execution_kind_for_role('judge') == 'gpu':
            gpu=judge_backend._checked(handle,['nvidia-smi','--query-gpu=name,uuid,memory.total','--format=csv,noheader']).strip()
            verify_gpu_inventory(gpu,resources)
            write_json(output/'gpu-topology'/f'{handle.id}.json',capture_gpu_topology(judge_backend,handle))
            if resources['gpu'] > 1:
                with private_runtime_diagnostics(output,handle.id):
                    write_json(output/'distributed-runtime'/f'{handle.id}.json',
                        verify_distributed_runtime(judge_backend,handle,gpu_count=resources['gpu'],
                            min_memory_bytes=resources['min_gpu_memory_bytes']))
        verify_asset_mounts(judge_backend,handle,bindings,output/'asset-checks'/f'{handle.id}.json')
        verify_asset_mounts(judge_backend,handle,private_bindings,output/'private-asset-checks'/f'{handle.id}.json',private=True)
        private=install_private(judge_backend,handle,task_root)
        if private != private_metadata:
            raise RuntimeError('Private Judge payload changed after this run started')
        if hasattr(private_module, 'validate_judge'):
            write_json(output/'judge-task-preflight'/f'{handle.id}.json',
                       private_module.validate_judge(judge_backend,handle,task_root))
        runtime_probe = checked(judge_backend,handle,deployment.runtime_probe_command('judge'),timeout=90)
        write_json(output/'judge-bootstrap'/f'{handle.id}.json',{'sandbox_id':handle.id,'gpu':gpu,
            'runtime_probe':runtime_probe,
            'platform_gpu_runtime':platform_runtime,
            'private_temporary_permissions':private_tmp,
            'private_payload_manifest':private,
            'private_payload_manifest_sha256':hashlib.sha256(json.dumps(private,sort_keys=True).encode()).hexdigest()})
        judge_backend._event('cpu_verified' if resources['gpu'] == 0 else 'gpu_verified',
                             handle,gpu=gpu)
    judge_backend.on_started = prepare_judge


def private_context_environment(private_module, task_root, run_id, output):
    hook = getattr(private_module, 'private_evaluation_context', None)
    context = hook(task_root, run_id, output) if hook else {}
    return encode_private_context(context)


def encode_private_context(context):
    if (not isinstance(context, dict) or any(
            not isinstance(key, str) or not key or not all(c.isupper() or c.isdigit() or c == '_' for c in key)
            or not isinstance(value, str) or '\x00' in value for key, value in context.items())):
        raise ValueError('Invalid controller-owned private evaluation context')
    return {'SFORGE_PRIVATE_CONTEXT_' + key: value for key, value in context.items()}


def private_context_secrets(environment):
    return {value for key,value in environment.items()
            if value and re.search(r'(SECRET|TOKEN|PASSWORD|KEY)(_|$)',key)}


def task_submission_evaluator(private_module, task_root, evaluate, **controller_context):
    factory = getattr(private_module, 'submission_evaluator', None)
    if factory is not None:
        # Task-specific trusted Controller adapters may own preparation before
        # their native Judge clock starts. The existing persisted-version
        # wrapper below still gates acceptance and supplies the exact mount.
        selected = factory(task_root, evaluate, **controller_context)
        if not callable(selected):
            raise ValueError('Private submission evaluator factory must return a callable')
        projection = getattr(selected, 'assert_private_projection', None)
        if projection is not None:
            try:
                mirror = controller_context.get('mirror')
                if not callable(projection) or mirror is None or not hasattr(mirror, 'output'):
                    raise ValueError('Native evaluator requires the actual Worker artifact projection')
                projection(mirror.output)
            except BaseException:
                # Constructors may already hold private owner locks. Fail
                # before submission acceptance, and explicitly close those
                # owners rather than losing their lifecycle object.
                cancel_task_evaluator(selected)
                close_task_evaluator(selected)
                raise
        return selected
    hook = getattr(private_module, 'evaluation_command', None)
    if not hook:
        return evaluate
    def wrapped(*args, **kwargs):
        # Explicit public evaluation commands retain their own original entrypoint.
        if kwargs.get('eval_cmd') is None:
            task = kwargs.get('task_spec') or args[0]
            kwargs['eval_cmd'] = hook(task_root, task.judge.eval_cmd)
        return evaluate(*args, **kwargs)
    return wrapped


def cancel_task_evaluator(evaluator):
    """Stop native admission promptly; this hook must not wait on remote work."""
    cancel = getattr(evaluator, 'cancel', None)
    if callable(cancel):
        cancel()


def close_task_evaluator(evaluator, *, timeout=300):
    """Require exact-owned cleanup before closing a native Controller service."""
    close = getattr(evaluator, 'close', None)
    if not callable(close):
        return {'cleanup_hook_used': False}
    receipt = close(timeout=timeout)
    if not isinstance(receipt, dict) or receipt.get('cleaned') is not True:
        raise RuntimeError('Task evaluator cleanup was not independently confirmed')
    # Per-leg capabilities, raw evidence and endpoints cannot enter run.json.
    return {'cleanup_hook_used': True, 'cleaned': True}


def _run_task(args):
    """Block through preflight or the full wall budget, final judge and cleanup."""
    import requests
    import uvicorn
    from pyromind_sdk.client.models import VolumeMount
    from sforge.harness.backend.pyromind_backend import PyroMindBackend, PyroMindHandle
    from sforge.harness.config import SForgeConfig
    from sforge.harness.constants import ADMIN_SECRET
    from sforge.harness.evolve_scripts import generate_evolve_prompt, generate_save_version_script, generate_submit_script
    from sforge.harness.judge_server import create_app
    from sforge.harness.run_evaluation import judge_submission
    from sforge.harness.version_store import VersionStore
    from sforge.harness.agent_runtime import load_agent_runtime

    seconds = duration_seconds(args.hours)
    parent_run = getattr(args, 'resume_from_run', None)
    requested_version = getattr(args, 'resume_version', None)
    if requested_version and not parent_run:
        raise ValueError('--resume-version requires --resume-from-run')
    if parent_run and args.preflight:
        raise ValueError('Task-only preflight cannot recover an Agent run')
    from sforge.pyromind.gpu_runtime import (capture_gpu_topology, prepare_hami_runtime,
                                            prepare_private_tmp, verify_distributed_runtime)
    repo = Path(__file__).resolve().parents[2]
    from sforge.pyromind.deployment import (Deployment, qualification_contract_digest,
                                           verify_gpu_inventory, validate_mount_separation)
    deployment = Deployment.load(Path(args.deployment), repo)
    deployment_source_sha256 = hashlib.sha256(deployment.path.read_bytes()).hexdigest()
    deployment.assert_launch_ready(preflight=args.preflight)
    source_commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    source_dirty = bool(subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True))
    worker_resources = deployment.resources_for_role('worker')
    judge_resources = deployment.resources_for_role('judge')
    bindings = deployment.asset_bindings(getattr(args, 'assets_file', None))
    private_bindings = (deployment.judge_asset_bindings(getattr(args, 'judge_assets_file', None))
                        if deployment.config.get('private_asset_requirements') or getattr(args, 'judge_assets_file', None) else [])
    validate_mount_separation(bindings, private_bindings)
    task_root = Path(args.task_dir or deployment.local_path('task_directory')).resolve()
    tasks_dir = task_root / "tasks"
    images = json.loads(Path(args.images_file or deployment.local_path('images_file')).read_text())
    worker_image, judge_image = images['worker'], images['judge']
    if '@sha256:' not in worker_image or '@sha256:' not in judge_image:
        raise ValueError("Task launch requires published immutable image digests")
    # Task qualification has no dependency on an Agent provider or runtime.
    model, credentials, runtime, tool_catalog = {}, None, None, None
    known_secrets = {value for value in (os.getenv('PYROMIND_API_KEY',''), ADMIN_SECRET) if value}
    if not args.preflight:
        model = load_model_config(Path(args.model_config))
        credentials = load_credentials(model)
        known_secrets.update(credentials.secrets)
        runtime = load_agent_runtime(Path(model['SFORGE_AGENT_RUNTIME_DIR']))
        if runtime.agent == 'codex':
            from sforge.harness.agent.codex_tools import load_direct_tool_catalog
            tool_catalog=load_direct_tool_catalog(runtime,model['MODEL'])
    from sforge.pyromind import integrity_review
    review_resources = integrity_review.review_resources(deployment)
    review_model_path = getattr(args, 'review_model_config', None)
    review_mode = getattr(args, 'llm_judge_mode', None) or 'enforce'
    if review_model_path:
        review_model = load_model_config(Path(review_model_path))
        known_secrets.update(load_credentials(review_model).secrets)
    elif not args.preflight:
        review_model = model
    else:
        # Task-only preflight has no model service; review only on request.
        review_model, review_mode = None, 'off'
    review_settings = integrity_review.review_config(
        review_model or {}, mode=review_mode, explicit=bool(review_model_path),
        resources=review_resources,
        on_error=deployment.config.get('review_on_error', 'admit'),
        implicit_model=deployment.config.get('review_model'),
        implicit_reasoning_effort=deployment.config.get('review_reasoning_effort'))
    from sforge.harness.benchmark import load_benchmark
    from sforge.harness.task_spec import load_all_tasks
    task = load_all_tasks(tasks_dir, load_benchmark(tasks_dir))[0]
    if task.task_id != deployment.config['task_id']:
        raise ValueError('Task identity differs from deployment profile')
    recovery = None
    if parent_run:
        from sforge.pyromind.recovery import load_recovery
        recovery = load_recovery(parent_run, version_id=requested_version, task=task,
                                 model=model, runtime=runtime,
                                 worker_image=worker_image, judge_image=judge_image)
        # --hours never grants a new budget when recovering a failed run.
        seconds = recovery.budget_seconds
    from sforge.pyromind.run_identity import make_run_identity
    identity = make_run_identity(deployment.config['name'], preflight=args.preflight,
                                 model=model, budget_seconds=seconds)
    run_id = identity['run_id']
    sandbox_name_prefix = identity['sandbox_name_prefix']
    output = Path(args.artifacts).resolve() / run_id
    output.mkdir(parents=True, exist_ok=False)
    # Bind the loaded inputs before remote work or later profile status edits.
    write_json(output/'deployment.json',deployment.config)
    write_json(output/'images.json',images)
    prefix = 'sforge-pyromind/' + run_id
    initialize_task_store(output/'version-store', run_id, task.task_id)
    store = VersionStore.open(output/'version-store')
    pinned_storages = [item['storage_identity'] for item in bindings+private_bindings if 'storage_identity' in item]
    if pinned_storages:
        from sforge.pyromind.storage import storage_connection
        storage_api, storage_bucket, storage_endpoint = storage_connection()
        actual_storage = {'endpoint': storage_endpoint, 'bucket': storage_bucket}
        if any(identity != actual_storage for identity in pinned_storages):
            raise ValueError('Shared assets belong to a different storage endpoint or bucket')
        mirror = ArtifactMirror(output, prefix, client=storage_api, bucket=storage_bucket)
    else:
        mirror = ArtifactMirror(output, prefix)
    mirror.sync()
    private_module = deployment.bootstrap()
    install_private = private_module.install_private_judge
    private_metadata = private_module.private_payload_metadata(task_root)
    private_manifest_sha256 = hashlib.sha256(json.dumps(private_metadata,sort_keys=True).encode()).hexdigest()
    write_json(output/'private-payload-manifest.json',private_metadata)
    image_map = {task.judge_image_key: judge_image}
    worker_bindings = asset_bindings_for_role(bindings, 'worker')
    judge_bindings = asset_bindings_for_role(bindings, 'judge')
    worker_mounts = asset_mounts_for_role(worker_bindings, 'worker', VolumeMount)
    judge_mounts = asset_mounts_for_role(judge_bindings, 'judge', VolumeMount)
    worker_backend = PyroMindBackend(event_path=output/'sandbox-events.jsonl',
        gpu_count=worker_resources['gpu'],gpu_card=worker_resources.get('gpu_card'),
        volume_mounts=worker_mounts+[VolumeMount(host_path='/workspace/'+prefix,mount_path='/shared',read_only=True)])
    private_mounts = [VolumeMount(host_path='/workspace/'+item['object_prefix'],
                                 mount_path=item['mount_path'],read_only=True) for item in private_bindings]
    judge_backend = PyroMindBackend(image_map=image_map,event_path=output/'sandbox-events.jsonl',
        gpu_count=judge_resources['gpu'],gpu_card=judge_resources.get('gpu_card'),volume_mounts=judge_mounts+private_mounts)

    worker_backend.ready_timeout=judge_backend.ready_timeout=deployment.sandbox_ready_timeout_seconds
    evaluation_budget = task.judge.eval_timeout + judge_backend.ready_timeout + deployment.config.get('judge_bootstrap_timeout',240) + 600
    cancelled=threading.Event()
    guard_judge_backend(judge_backend,cancelled,sandbox_name_prefix,
                        environment=deployment.config.get('judge_environment'))
    configure_judge_runtime(judge_backend,deployment,private_module,task_root,output,
                            judge_bindings,private_metadata,private_bindings)
    config = SForgeConfig(backend='pyromind',tasks_dir=tasks_dir,version_store_root=output/'version-store',
                         log_dir=output/'logs',judge_cpu_limit=judge_resources['cpu'],judge_mem_limit=judge_resources['memory'],
                         judge_extra_env=private_context_environment(private_module,task_root,run_id,output),
                         llm_judge=review_settings)
    known_secrets.update(private_context_secrets(config.judge_extra_env))
    # Persist before the server queues a Judge, not in response middleware:
    # middleware failures after queueing can cancel its BackgroundTasks and
    # strand an accepted submission with agent_inflight set forever.
    from sforge.pyromind.version_transport import PersistedCandidateTransport
    candidate_transport = PersistedCandidateTransport(mirror, store, task.task_id, run_id)
    app = create_app(config, version_persister=candidate_transport.persist)
    judge_state = app.state.judge_state
    judge_state.backend = judge_backend
    review_backend = None
    staged_review_runtime = None
    if review_settings.enabled:
        staged_review_runtime = integrity_review.stage_runtime(
            load_agent_runtime(review_settings.agent_runtime_dir), mirror.client, mirror.bucket,
            Path.home() / '.cache' / 'sforge-pyromind' / 'review-runtime')
        review_backend = PyroMindBackend(event_path=output/'sandbox-events.jsonl', gpu_count=0,
            volume_mounts=[integrity_review.runtime_mount(staged_review_runtime)])
        review_backend.ready_timeout = deployment.sandbox_ready_timeout_seconds
        configure_resource_capture(review_backend, output/'resource-usage'/'review')
        guard_judge_backend(review_backend, cancelled, sandbox_name_prefix)
        judge_state.configure_integrity_review(integrity_review.build_review_engine(
            review_settings, review_backend, judge_image=judge_image, name_prefix='review',
            staged_runtime=staged_review_runtime))
    review_allowance = integrity_review.review_allowance_seconds(
        review_settings, deployment.sandbox_ready_timeout_seconds)
    listener = socket.socket()
    listener.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
    listener.bind((getattr(args,'listen_host','0.0.0.0'),getattr(args,'listen_port',8080)))
    port = listener.getsockname()[1]
    external_url = getattr(args,'judge_url',None) or controller_url(port)
    local_url = 'http://127.0.0.1:'+str(port)
    try:
        selected_evaluator = task_submission_evaluator(private_module,task_root,judge_submission,
                task_spec=task,config=config,judge_backend=judge_backend,run_id=run_id,
                output=output,deployment=deployment,controller_origin=external_url,
                judge_app=app,cancelled=cancelled,store=store,mirror=mirror)
        evaluate_submission = candidate_transport.evaluate(selected_evaluator,
            lambda submission_id: judge_state.submissions.get(submission_id))
    except BaseException:
        listener.close()
        raise
    judge_state.submission_evaluator = evaluate_submission
    server = uvicorn.Server(uvicorn.Config(app,access_log=False,log_level='warning'))
    thread = threading.Thread(target=server.run,kwargs={'sockets':[listener]},daemon=True)
    thread.start()
    session = requests.Session(); session.trust_env=False
    state = {**identity,'task_id':task.task_id,'status':'starting',
             'budget_seconds':None if args.preflight else seconds,
             'budget_scope':'not-applicable' if args.preflight else 'agent-wall-clock',
             'agent_started':False,
             'source_commit':source_commit,'source_tree_dirty':source_dirty,
             'deployment_sha256':deployment_source_sha256,
             'deployment_snapshot_sha256':hashlib.sha256((output/'deployment.json').read_bytes()).hexdigest(),
             'images_snapshot_sha256':hashlib.sha256((output/'images.json').read_bytes()).hexdigest(),
             'runtime_contract_sha256':qualification_contract_digest(deployment.config),
             'worker_image':worker_image,'judge_image':judge_image,'model':model.get('MODEL'),
             'agent_auth_mode':credentials.mode if credentials else None,
             'reasoning_effort':model.get('REASONING_EFFORT','high') if credentials else None,
             'qualification_scope':'task-only' if args.preflight else 'task-and-agent',
             'agent_model_validation':{'status':'not-run','reason':'task-only-preflight'} if args.preflight else {'status':'pending'},
             'private_payload_manifest_sha256':private_manifest_sha256,
             'runtime_version':runtime.version if runtime else None,'judge_url':external_url,'controller':'local-cpu-orchestration-only',
             'agent':runtime.agent if runtime else None,
             'agent_submission_policy':None if args.preflight else 'wait-for-result',
             'submit_wait_enforcement':None if args.preflight else 'synchronous-'+runtime.agent+'-pre-tool-hook',
             'codex_tool_mode':'direct' if runtime and runtime.agent=='codex' else None,
             'codex_tool_catalog_sha256':hashlib.sha256(tool_catalog.encode()).hexdigest() if tool_catalog else None,
             'judge_lifecycle':('one-new-sandbox-per-submission-and-final' if judge_resources['gpu'] == 0
                                else 'one-new-H100-sandbox-per-submission-and-final'),'scheduled_evaluation':False,
             'shared_object_prefix':prefix,'shared_mount':'/shared','platform_key_remote':False,
             'shared_persistence':'exact-version-storage-readback-before-ack',
             'candidate_transport':'readonly-version-mount',
             'single_gpu_handoff':False,'allocation_mode':'independent_sandboxes',
             'worker_resources':worker_resources,'judge_resources':judge_resources,
             'deployment':deployment.config,'asset_bindings':bindings,
             'egress_policy':'platform-default-not-attested',
             'worker_internet':task.internet,'agent_web_search':'live' if task.internet else 'disabled',
             'integrity_review':integrity_review.describe(review_settings,resources=review_resources,
                 judge_image=judge_image,staged_runtime=staged_review_runtime,
                 explicit_model_config=bool(review_model_path))}
    if recovery:
        state['recovery'] = recovery.metadata()
        state['run_label'] += ' · recovered candidate · original deadline retained'
    write_json(output/'run.json',state)
    print(f"RUN_DIRECTORY={output}",flush=True)
    configure_runtime_cleanup(worker_backend, private_module, task_root, output, role='worker')
    worker = None
    registration = None
    cleanup_errors=[]
    try:
        for _ in range(100):
            if server.started: break
            time.sleep(.1)
        if not server.started: raise RuntimeError('Controller HTTP service failed to start')
        response=session.post(local_url+'/api/v1/register',json={'task_id':task.task_id,'run_id':run_id,
            'admin_secret':ADMIN_SECRET,'max_agent_submissions':deployment.config['max_submissions'],
            'submission_cooldown':deployment.config['submission_cooldown'],
            'require_version_saving':True,'required_version_store_id':run_id},timeout=10)
        response.raise_for_status();registration=response.json()
        known_secrets.add(registration['token'])
        environment={'SFORGE_JUDGE_URL':external_url,'SFORGE_TOKEN':registration['token'],
            'SFORGE_PATCH_DIR':task.cwd,'SFORGE_SUBMIT_PATHS':' '.join(task.submit_paths),
            'SFORGE_SUBMIT_EXCLUDE_FLAGS':' '.join('--exclude='+p for p in task.submit_exclude),
            'SFORGE_SUBMIT_POST_TIMEOUT':'120',
            'SFORGE_SUBMIT_POLL_ATTEMPTS':str((evaluation_budget+review_allowance+9)//10),
            **deployment.config.get('worker_environment',{}),
            'SFORGE_SUBMIT_TOOL_TIMEOUT_MS':str((seconds+60)*1000),
            'SFORGE_SUBMIT_WAIT_STATE_DIR':'/run/sforge-submit-wait',
            'NO_PROXY':'localhost,127.0.0.1,'+external_url.split('/')[2].split(':')[0]}
        if recovery:
            from sforge.pyromind.recovery import remaining_budget
            remaining_budget(recovery.original_deadline_at)
        worker=worker_backend.create_container(worker_image,sandbox_name_prefix+'-worker',environment=environment,cpu_limit=worker_resources['cpu'],mem_limit=worker_resources['memory'])
        state.update(worker_id=worker.id,worker_name=worker.name);write_json(output/'run.json',state)
        print('Starting persistent '+(worker_resources.get('gpu_card') if worker_resources['gpu'] else 'CPU')+' Worker '+worker.id,flush=True)
        def prepare_worker(handle):
            # Admission occurs before initial save, model calls or Agent code.
            # A typed placement rejection can then replace this allocation safely.
            state.update(worker_id=handle.id, worker_name=handle.name)
            if deployment.execution_kind_for_role('worker') == 'gpu':
                with private_runtime_diagnostics(output,handle.id):
                    state['platform_gpu_runtime']=prepare_hami_runtime(worker_backend,handle,gpu_count=worker_resources['gpu'])
            checked(worker_backend,handle,"id agent >/dev/null 2>&1 || useradd -m -s /bin/bash agent; mkdir -p /run/sforge-agent /home/agent/.codex /home/agent/.config/sforge; chmod 700 /run/sforge-agent; chown -R agent:agent /home/agent",timeout=30)
            checked(worker_backend,handle,"gpasswd -d agent sudo >/dev/null 2>&1 || true; rm -f /etc/sudoers.d/agent /etc/sudoers.d/mlsysforge-agent",timeout=60)
            state['worker_runtime_probe'] = checked(worker_backend,handle,
                deployment.runtime_probe_command('worker'),timeout=90)
            if task.task_id == 'minference_sparse_prefill_h100_seed_v3':
                hook = getattr(private_module, 'install_worker_bindings', None)
                if not callable(hook):
                    raise RuntimeError('MInference Worker requires its task-specific two-role asset bootstrap')
                write_json(output/'worker-asset-bootstrap.json',
                    hook(worker_backend, handle, task_root, worker_bindings))
            if hasattr(private_module,'validate_worker'):
                evidence=private_module.validate_worker(worker_backend,handle,task_root)
                write_json(output/'worker-task-preflight.json',evidence)
            write_json(output/'run.json',state)
        worker_backend.on_started = prepare_worker
        worker_backend.start_container(worker)
        probe="import os,urllib.request; o=urllib.request.build_opener(urllib.request.ProxyHandler({})); assert o.open(os.environ['SFORGE_JUDGE_URL']+'/openapi.json',timeout=20).status==200; print('Worker reaches CPU submission service')"
        print(checked(worker_backend,worker,['python3','-c',probe],timeout=30),flush=True)
        if task.internet:
            state['worker_internet_probe']=probe_worker_internet(worker_backend,worker)
            write_json(output/'run.json',state)
        gpu = None
        if deployment.execution_kind_for_role('worker') == 'gpu':
            gpu=checked(worker_backend,worker,['nvidia-smi','--query-gpu=name,uuid,memory.total','--format=csv,noheader'],timeout=30)
            verify_gpu_inventory(gpu,worker_resources)
            write_json(output/'gpu-topology'/f'{worker.id}.json',capture_gpu_topology(worker_backend,worker))
            if worker_resources['gpu'] > 1:
                with private_runtime_diagnostics(output,worker.id):
                    write_json(output/'distributed-runtime'/f'{worker.id}.json',
                        verify_distributed_runtime(worker_backend,worker,gpu_count=worker_resources['gpu'],
                            min_memory_bytes=worker_resources['min_gpu_memory_bytes']))
        verify_asset_mounts(worker_backend,worker,worker_bindings,output/'asset-checks'/f'{worker.id}.json')
        state['worker_gpu']=gpu.strip() if gpu is not None else None;write_json(output/'run.json',state)
        if deployment.config.get('require_host_memory_evidence'):
            capture_host_memory(worker_backend,worker,output/'resource-usage'/f'{worker.id}.json')
            verify_qualification_memory(output)
        for name,content in [('sforge-submit',generate_submit_script()),('sforge-save-version',generate_save_version_script())]:
            worker_backend.write_to_container(worker,content,PurePosixPath('/usr/local/bin/'+name))
        checked(worker_backend,worker,'chmod 755 /usr/local/bin/sforge-submit /usr/local/bin/sforge-save-version',timeout=30)
        worker_backend.write_to_container(worker,json.dumps(environment),PurePosixPath('/run/sforge-agent/environment.json'))
        worker_env=''.join('export '+name+'='+shlex.quote(value)+'\n' for name,value in environment.items())
        worker_backend.write_to_container(worker,worker_env,PurePosixPath('/etc/sforge/worker.env'))
        worker_backend.write_to_container(worker,'. /etc/sforge/worker.env\n',PurePosixPath('/etc/profile.d/sforge.sh'))
        checked(worker_backend,worker,'chown root:agent /etc/sforge /etc/sforge/worker.env; chmod 750 /etc/sforge; chmod 640 /etc/sforge/worker.env; chmod 644 /etc/profile.d/sforge.sh',timeout=15)
        if recovery:
            from sforge.pyromind.recovery import restore_worker
            restored = restore_worker(recovery, worker_backend, worker, task, checked)
            write_json(output/'recovery-verification.json', restored)
        saved=checked(worker_backend,worker,['sforge-save-version','--label','pyromind-initial'],timeout=120)
        (output/'initial-save.txt').write_text(saved)
        initial=next(version for version in store.list_versions(source='checkpoint')
                     if version.label=='pyromind-initial')
        verify_shared_version(worker_backend,worker,initial,output,mirror)
        reviewer = getattr(judge_state, 'integrity_reviewer', None)
        if reviewer is not None:
            judge_state.set_run_baseline(run_id, integrity_review.capture_baseline(
                task, reviewer.policy_for(task), store, initial,
                lambda path: worker_backend.client.read_file(worker.id, path)))
        print('Checkpoint SHA-256 matches local archive, platform Storage and Worker read-only /shared',flush=True)
        if args.preflight:
            print('Running native Worker sforge-submit against a new ephemeral H100 Judge',flush=True)
            submitted=checked(worker_backend,worker,['sforge-submit','--details'],timeout=evaluation_budget+60)
            (output/'initial-submit.txt').write_text(submitted)
            result=list(judge_state.submissions.values())[-1]
            report=result.get('report') or {}
            if result['status']!='completed' or report.get('valid') is not deployment.config['starter_valid']:
                raise RuntimeError('Starter qualification did not match deployment expectation')
            band=deployment.config.get('starter_score_band')
            if band and (report.get('score') is None or not band[0] <= report['score'] <= band[1]):
                raise RuntimeError('Seed self-comparison score outside qualification band; inspect infrastructure before Agent run')
            if report.get('metrics',{}).get('evaluation_status')=='infrastructure_error':
                raise RuntimeError('Starter failed due to infrastructure, not the candidate')
            if deployment.config.get('require_host_memory_evidence'):
                capture_host_memory(worker_backend,worker,output/'resource-usage'/f'{worker.id}.json')
                verify_qualification_memory(output)
            write_history(output,judge_state.get_run_history(run_id,task.task_id))
            mirror.sync()
            state.update(status='preflight-completed',finished_at=time.time())
            write_json(output/'run.json',state)
            return output
        state['agent_model_validation']={'status':'running'};write_json(output/'run.json',state)
        state['agent_model_validation']=install_and_validate_agent(
            worker_backend,worker,model,runtime,credentials,checked,
            submit_wait_timeout_seconds=seconds+120,tool_catalog=tool_catalog,web_search=task.internet)
        collect_worker_auth_secrets(worker_backend,worker,credentials,known_secrets,required=True)
        write_json(output/'run.json',state)
        query = task.work.agent_query.split('Work and Judge share the single CUDA GPU')[0]
        if worker_resources['gpu'] > 0 and judge_resources['gpu'] > 0:
            query += '\n\nPyroMind deployment: Worker and each ephemeral Judge have independent GPU allocations. '
        else:
            query += '\n\nPyroMind deployment: Worker and each ephemeral Judge have independent Sandbox allocations. '
            query += ('Worker execution: ' + deployment.execution_kind_for_role('worker')
                      + '; Judge execution: ' + deployment.execution_kind_for_role('judge') + '. ')
        query += 'Only manual sforge-submit triggers interactive judging; no scheduled evaluations. '
        query += ('Run sforge-submit as a standalone foreground command in its own tool call. '
                  'Wait for its final score or explicit failure before editing code, running tests, '
                  'saving versions, or calling any other tools. Never background, wrap, pipe, '
                  'or combine a submission with other commands. Stop your existing experiments '
                  'before submitting. The submission tool returns after evaluation, not after queueing. ')
        query += deployment.config['evaluation_description']
        if recovery:
            from sforge.pyromind.recovery import remaining_budget
            query += (f" About {int(remaining_budget(recovery.original_deadline_at))} wall-clock seconds remain "
                      "until the original deadline, including waits for feedback. ")
        else:
            query += f" Your wall-clock budget is {seconds} seconds, including waits for feedback. "
        query += 'Save retained versions. Do not read credentials or copy external implementations into your submission. '
        query += deployment.config.get('agent_guidance','')
        if recovery:
            query += recovery.guidance()
        announce=bool((registration.get('integrity_review') or {}).get('announce'))
        prompt=generate_evolve_prompt(query,submit_paths=task.submit_paths,internet=task.internet,max_submissions=deployment.config['max_submissions'],submission_cooldown=deployment.config['submission_cooldown'],version_saving_enabled=True,integrity_review=announce)
        worker_backend.write_to_container(worker,prompt,PurePosixPath('/home/agent/task-prompt.md'))
        checked(worker_backend,worker,'chown agent:agent /home/agent/task-prompt.md',timeout=15)
        effort=model.get('REASONING_EFFORT','high')
        prompt_argument = ' "$(cat /home/agent/task-prompt.md)\n${SFORGE_CANCELLED_TOOLS_NOTICE:-}"'
        if recovery:
            prompt_argument = ' "$(cat /home/agent/task-prompt.md)\nActual remaining wall-clock seconds at this invocation: $SFORGE_REMAINING_SECONDS\n${SFORGE_CANCELLED_TOOLS_NOTICE:-}"'
        first='codex exec --json --skip-git-repo-check --dangerously-bypass-approvals-and-sandbox --dangerously-bypass-hook-trust --enable hooks --disable multi_agent --disable code_mode --disable code_mode_only -m '+shlex.quote(model['MODEL'])+' -c '+shlex.quote('model_reasoning_effort='+json.dumps(effort))+prompt_argument
        resume='codex exec resume --last --json --skip-git-repo-check --dangerously-bypass-approvals-and-sandbox --dangerously-bypass-hook-trust --enable hooks --disable multi_agent --disable code_mode --disable code_mode_only -m '+shlex.quote(model['MODEL'])+' -c '+shlex.quote('model_reasoning_effort='+json.dumps(effort))+' '+shlex.quote('Continue this task using the deployment instructions in /home/agent/task-prompt.md. Choose a concrete useful experiment, use tools to measure it, preserve the seed and best candidate, save checkpoints and update FINAL_VERSION. Do not repeatedly report completion while budget remains. Remaining wall-clock seconds: ')+'"$SFORGE_REMAINING_SECONDS\n${SFORGE_CANCELLED_TOOLS_NOTICE:-}"'
        if runtime.agent == 'claude-code':
            from sforge.harness.agent.claude_agent import claude_command
            # Quote fixed CLI arguments. The prompt and remaining budget expand
            # inside the Worker only; no credential enters a shell command.
            first=shlex.join(claude_command(model,'',web_search=task.internet)[:-1])+prompt_argument
            continuation='Continue the task in /home/agent/task-prompt.md. Run a concrete useful experiment, preserve the seed and best candidate, save checkpoints and update FINAL_VERSION. Do not stop while budget remains. Remaining wall-clock seconds: '
            resume=shlex.join(claude_command(model,'',resume='continue',web_search=task.internet)[:-1])+' '+shlex.quote(continuation)+'"$SFORGE_REMAINING_SECONDS\n${SFORGE_CANCELLED_TOOLS_NOTICE:-}"'
        supervisor={'seconds':seconds,'cwd':task.cwd,'environment':environment,'auth_mode':credentials.mode,'first_command':first,
                    'resume_command':resume,'submit_paths':task.submit_paths,'submit_exclude':task.submit_exclude,
                    'agent':runtime.agent,'model_api_base_url':model.get('MODEL_API_BASE_URL'),
                    'reasoning_effort':effort}
        if recovery:
            from sforge.pyromind.recovery import remaining_budget
            remaining_budget(recovery.original_deadline_at)
            supervisor['deadline_at'] = recovery.original_deadline_at
        worker_backend.write_to_container(worker,json.dumps(supervisor),PurePosixPath('/run/sforge-agent/supervisor.json'))
        worker_backend.write_to_container(worker,supervisor_source(),PurePosixPath('/run/sforge-agent/supervisor.py'))
        worker_backend._checked(worker,'chmod 600 /run/sforge-agent/*; nohup python3 /run/sforge-agent/supervisor.py >/run/sforge-agent/supervisor.log 2>&1 </dev/null & echo $! >/run/sforge-agent/supervisor.pid')
        state.update(status='agent-running',agent_started=True,agent_started_at=time.time());write_json(output/'run.json',state)
        status_failures=0
        remaining = max(0, recovery.original_deadline_at-time.time()) if recovery else seconds
        supervision_deadline=time.monotonic()+remaining+300
        while True:
            time.sleep(15)
            try:
                progress=_remote_json(worker_backend,worker,'/run/sforge-agent/status.json')
                status_failures=0
            except Exception:
                status_failures+=1
                if status_failures>=8 or time.monotonic()>supervision_deadline: raise RuntimeError('Worker supervisor status unavailable')
                continue
            state['agent_progress']=progress;write_json(output/'run.json',state)
            write_history(output,judge_state.get_run_history(run_id,task.task_id));mirror.sync()
            if progress['status']=='failed':
                state['budget_completed']=False
                try:
                    state['failure_preservation']=preserve_failed_worker_candidate(
                        worker_backend, worker, output, task, store, mirror,
                        credentials, known_secrets, run_id, progress)
                except Exception as preservation_error:
                    state['failure_preservation_error']=type(preservation_error).__name__
                raise RuntimeError('Worker supervisor failed: '+progress.get('error','unknown'))
            if progress['status']=='completed': break
            if time.monotonic()>supervision_deadline: raise TimeoutError('Worker exceeded bounded supervision deadline')
        if not progress.get('no_agent_processes'): raise RuntimeError('No verified final writer boundary')
        # Seal admission by revoking the task token only after every writer exited.
        with judge_state._tokens_lock:
            judge_state.tokens.pop(registration['token'],None)
        _drain(judge_state,evaluation_budget+review_allowance)
        write_history(output,judge_state.get_run_history(run_id,task.task_id))
        collect_worker_auth_secrets(worker_backend,worker,credentials,known_secrets,required=True,journal_required=True)
        archive=worker_backend.client.read_file(worker.id,'/run/sforge-agent/final.tar.gz')
        store.validate_archive(task_id=task.task_id,archive=archive,submit_paths=task.submit_paths,submit_exclude=task.submit_exclude)
        validate_no_credentials(archive,known_secrets)
        final_version=store.record_archive(task_id=task.task_id,run_id=run_id,source='final',round_id='final',
            label='budget-final',archive=archive,submit_paths=task.submit_paths,submit_exclude=task.submit_exclude,
            content_identity_exclude=task.candidate_identity_exclude,metadata={'writer_boundary':'agent_uid_and_process_group',
                'budget_seconds':seconds,'worker_id':worker.id,'worker_image':worker_image,'judge_image':judge_image,
                'private_payload_manifest_sha256':private_manifest_sha256})
        write_json(output/'final-snapshot.json',{'archive_sha256':final_version.archive_sha256,
            'archive_bytes':len(archive),'writer_boundary':'agent_uid_and_process_group','docker_frozen_snapshot':False,
            'agent':progress,'worker_id':worker.id})
        transcript=worker_backend.client.read_file(worker.id,'/run/sforge-agent/agent.jsonl')
        transcript=redact(transcript,known_secrets)
        (output/'agent.jsonl').write_bytes(transcript)
        mirror.sync()
        mirror.verify_versions([final_version])
        # Work is destroyed before the independent final GPU is allocated.
        worker_backend.cleanup_container(worker);worker=None
        state.update(status='final-evaluation-running');write_json(output/'run.json',state)
        print('Agent budget completed; Worker removed; reviewing sealed final before independent Judge',flush=True)
        report=evaluate_reviewed_final(judge_state=judge_state,evaluate_submission=evaluate_submission,
            task_spec=task,archive=archive,config=config,backend=judge_backend,run_id=run_id,
            round_id=deployment.evaluation_round_id('final'),log_dir=output/'logs/final',
            judge_image_ref=judge_image,stored_version=final_version)
        write_json(output/'final.json',report.to_dict())
        if report.to_dict().get('metrics',{}).get('evaluation_status')=='infrastructure_error':
            raise RuntimeError('Final Judge encountered an infrastructure error; see local final report')
        state.update(status='completed',final_valid=report.to_dict().get('valid'),final_score=report.to_dict().get('score'),finished_at=time.time())
        write_json(output/'run.json',state)
        mirror.sync();mirror.verify_versions(store.list_versions())
        print('Final evaluation complete; report: '+str(output/'final.json'),flush=True)
        return output
    except BaseException as exc:
        if worker is not None:
            collect_worker_auth_secrets(worker_backend,worker,credentials,known_secrets)
        message=redact(str(exc),known_secrets)
        if state.get('agent_model_validation',{}).get('status')=='running':
            state['agent_model_validation']={'status':'failed'}
        state.update(status='failed',error=message,finished_at=time.time());write_json(output/'run.json',state)
        raise RuntimeError(message) from None
    finally:
        cancelled.set()
        try:
            cancel_task_evaluator(selected_evaluator)
        except Exception as exc:
            cleanup_errors.append('TaskEvaluatorCancel:' + type(exc).__name__)
        if worker is not None:
            if state.get('status') == 'failed' and state.get('agent_started'):
                try:
                    diagnostics=worker_backend.client.read_file(worker.id,'/run/sforge-submit-wait/state.json')
                    if len(diagnostics)>16*1024:
                        raise ValueError('Submission wait diagnostic state exceeds its bound')
                    sanitized=submit_wait_diagnostics(json.loads(diagnostics))
                    sanitized['worker_id']=worker.id
                    sanitized=json.loads(redact(json.dumps(sanitized),known_secrets))
                    write_json(output/'submit-wait-diagnostics.json',sanitized)
                except Exception:
                    pass  # Diagnostic availability must never prevent GPU cleanup.
            try:
                worker_backend.write_to_container(worker,'stop',PurePosixPath('/run/sforge-agent/stop'))
            except Exception: pass
            if not args.preflight and not (output/'agent.jsonl').exists():
                try:
                    collect_worker_auth_secrets(worker_backend,worker,credentials,known_secrets,required=True,
                                                journal_required='agent_started_at' in state)
                    transcript=worker_backend.client.read_file(worker.id,'/run/sforge-agent/agent.jsonl')
                    # The stop marker does not prove writers have exited on a
                    # failure path. Include rotations through the transcript
                    # read; an unavailable/partial journal skips diagnostics.
                    collect_worker_auth_secrets(worker_backend,worker,credentials,known_secrets,required=True,
                                                journal_required='agent_started_at' in state)
                    transcript=redact(transcript,known_secrets)
                    (output/'agent.jsonl').write_bytes(transcript)
                except Exception: pass  # This is best-effort diagnostics, never a cleanup prerequisite.
        if registration is not None:
            with judge_state._tokens_lock:
                judge_state.tokens.pop(registration['token'],None)
        try:
            # Native ownership uses its own exact-ID ledger, independent of
            # the generic sf-* naming scheme. Keep its bridge alive while the
            # owner interrupts/drains measurement and deletes every owned ID.
            state['task_evaluator_cleanup'] = close_task_evaluator(selected_evaluator,timeout=300)
        except Exception as exc:
            state['task_evaluator_cleanup'] = {'cleaned': False, 'error': type(exc).__name__}
            cleanup_errors.append('TaskEvaluatorClose:' + type(exc).__name__)
        server.should_exit=True
        judge_state._reaper_stop.set()
        if worker is not None:
            try: worker_backend.cleanup_container(worker)
            except Exception as exc: cleanup_errors.append(type(exc).__name__)
        events_path=output/'sandbox-events.jsonl'
        if events_path.exists():
            events=[json.loads(line) for line in events_path.read_text().splitlines()]
            deleted={item['id'] for item in events if item['event']=='deleted'}
            for item in events:
                if item['event']=='created' and item['id'] not in deleted:
                    try: judge_backend.cleanup_container(PyroMindHandle(item['id'],item['name']))
                    except Exception as exc: cleanup_errors.append(type(exc).__name__)
        thread.join(timeout=20);listener.close()
        try: _drain(judge_state,60)
        except Exception as exc: cleanup_errors.append(type(exc).__name__)
        # A timed-out create request can leave a resource before its ID is known.
        # The unique per-run name prefix restricts this reconciliation to our run.
        try:
            remaining=reconcile_run_sandboxes(judge_backend,sandbox_name_prefix)
            state['remaining_sandbox_ids']=remaining
            if remaining: cleanup_errors.append('SandboxesStillActive')
        except Exception as exc: cleanup_errors.append(type(exc).__name__)
        ownership = reconcile_creation_ownership(
            (worker_backend, judge_backend) + ((review_backend,) if review_backend is not None else ()))
        state['unresolved_creation_names'] = ownership['unresolved_creation_names']
        cleanup_errors.extend(ownership['cleanup_errors'])
        if state.get('remaining_sandbox_ids') is not None:
            state['remaining_sandbox_ids'] = sorted(set(state['remaining_sandbox_ids'])
                                                   | set(ownership['remaining_sandbox_ids']))
        if ownership['unresolved_creation_names'] or ownership['cleanup_errors']:
            state['remaining_sandbox_ids'] = None
        state['cleanup_errors']=cleanup_errors;write_json(output/'run.json',state)
        try:
            mirror.sync();mirror.verify_versions(store.list_versions())
        except Exception as exc:
            state['artifact_sync_error']=type(exc).__name__;write_json(output/'run.json',state)
            if not cleanup_errors: cleanup_errors.append(type(exc).__name__)
        session.close()
        if cleanup_errors:
            if state['status']!='failed':
                state['status']='cleanup-failed';write_json(output/'run.json',state)
                raise RuntimeError('Cleanup/storage checks failed: '+','.join(cleanup_errors))
            print('Cleanup/storage checks need attention: '+','.join(cleanup_errors),flush=True)
