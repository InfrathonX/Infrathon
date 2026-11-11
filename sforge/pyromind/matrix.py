"""Persist a task/model experiment matrix with conservative GPU reservations.

Every admitted run reserves Worker plus one Judge for its entire lifecycle.
This bounds peak allocation without requiring independently running Controllers
to race for the last GPU. Queued entries consume neither GPUs nor run budget.
The manifest contains local credential-file paths, never credential values.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import uuid
import hashlib
import re
from zoneinfo import ZoneInfo


def atomic_json(path: Path, document):
    temporary = path.with_suffix('.tmp')
    descriptor=os.open(temporary,os.O_WRONLY|os.O_CREAT|os.O_TRUNC,0o600)
    os.fchmod(descriptor,0o600)
    with os.fdopen(descriptor,'w') as stream:
        json.dump(document, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def process_identity(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()
        return None if fields[0] == 'Z' else fields[19]
    except (FileNotFoundError, ProcessLookupError):
        return None


def latest_run(root: Path):
    paths = sorted(root.glob('*/run.json'))
    if not paths:
        return None
    if len(paths) != 1:
        raise ValueError('A matrix slot must identify exactly one run')
    return json.loads(paths[0].read_text())


def reservation(entry, deployment):
    return sum(deployment.resources_for_role(role)['gpu'] for role in ('worker', 'judge'))


def reserved_gpu(entries):
    # A crashed Controller is fenced until its own lifecycle confirms no remote
    # resources remain. It must never free GPU budget solely because PID exited.
    return sum(entry['reserved_gpu'] for entry in entries
               if entry.get('status') in ('launching', 'starting', 'running', 'finalizing', 'cleanup-unconfirmed'))


def confirmed_cleanup(run):
    return bool(run and run.get('remaining_sandbox_ids') == []
                and run.get('cleanup_errors') == []
                and run.get('status') in ('completed','failed','cleanup-failed'))


def startup_retry_allowed(run, failures):
    """Retry platform startup only; never reset an Agent's running12h budget."""
    return (failures < 3 and confirmed_cleanup(run) and run.get('status')=='failed'
            and run.get('agent_started') is False
            and bool(re.fullmatch(r'Sandbox sb-[a-z0-9]+ (?:entered (?:failed|error)|not ready within [0-9]+s)',
                                  run.get('error',''))))


def provider_probe_matches(probe, model_path, model):
    from sforge.harness.agent_runtime import load_agent_runtime
    runtime=load_agent_runtime(Path(model['SFORGE_AGENT_RUNTIME_DIR']))
    return (probe.get('status')=='passed' and probe.get('model')==model['MODEL']
            and probe.get('reasoning_effort')==model.get('REASONING_EFFORT','high')
            and probe.get('model_config_sha256')==hashlib.sha256(model_path.read_bytes()).hexdigest()
            and probe.get('runtime_version')==runtime.version
            and probe.get('runtime_tree_sha256')==runtime.tree_sha256)


def choose_port():
    # Keep the matrix admission lock until the child has bound its listener.
    # Port zero also lets runner bind atomically; it computes its public URL from
    # the actual chosen port, so no hard-coded Controller port is shared.
    return 0


def validate_manifest(manifest):
    if (manifest.get('schema_version') != 1 or type(manifest.get('max_gpu')) is not int
            or not 1 <= manifest['max_gpu'] <= 28 or manifest.get('hours_each') != 12
            or not isinstance(manifest.get('slots'),list) or not manifest['slots']):
        raise ValueError('Matrix requires explicit12h runs and total GPU ceiling at most28')
    seen=set()
    limits=manifest.get('max_parallel_by_deployment',{})
    if not isinstance(limits,dict):raise ValueError('Deployment concurrency limits must be a dictionary')
    for path,limit in limits.items():
        if Path(path).is_absolute() or '..' in Path(path).parts or type(limit) is not int or limit<1:
            raise ValueError('Deployment concurrency limits require relative paths and positive counts')
    for entry in manifest['slots']:
        if (not isinstance(entry,dict) or not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,79}',entry.get('slot',''))
                or entry['slot'] in seen):
            raise ValueError('Matrix slot IDs must be unique safe directory names')
        seen.add(entry['slot'])
        if entry.get('execution','sandbox') not in ('sandbox','external-local'):
            raise ValueError('Unknown matrix execution mode')
        for key in ('deployment','assets_file','judge_assets_file'):
            if key in entry:
                path=Path(entry[key])
                if path.is_absolute() or '..' in path.parts:
                    raise ValueError('Matrix deployment/assets paths must stay inside source checkout')
        review=entry.get('review_model_config',manifest.get('review_model_config'))
        if review is not None and (not isinstance(review,str) or not Path(review).is_absolute()):
            raise ValueError('The integrity reviewer model config must be an absolute local path')


def append_manifest_slots(state, manifest):
    """Add experiments on restart while retaining every existing run's ownership."""
    known = {entry['slot'] for entry in state['slots']}
    if len(known) != len(state['slots']):
        raise ValueError('Persisted matrix contains duplicate slot ownership')
    for selected in manifest['slots']:
        if selected['slot'] not in known:
            state['slots'].append({**selected, 'status': 'queued'})
            known.add(selected['slot'])


def health_record(entry, run):
    progress = (run or {}).get('agent_progress') or {}
    deadline = progress.get('deadline_at')
    submissions=None
    if entry.get('artifact_root') and (run or {}).get('run_id'):
        path=Path(entry['artifact_root'])/run['run_id']/'history.json'
        if path.exists():
            history=json.loads(path.read_text())
            if isinstance(history,list):submissions=len(history)
            elif isinstance(history,dict) and type(history.get('agent_submissions')) is int:
                submissions=history['agent_submissions']
    return {'slot': entry['slot'], 'status': entry['status'],
            'run_id': (run or {}).get('run_id'),
            'agent_status': progress.get('status'),
            'remaining_seconds': max(0, int(deadline-time.time())) if isinstance(deadline, (int, float)) else None,
            'submissions': submissions,
            'last_exit_code': progress.get('last_exit_code'),
            'attempts': progress.get('attempts')}


def live_external_gpu(client, entries):
    """Count unrelated/Pending allocations; list errors prevent new admission."""
    prefixes = {entry.get('sandbox_name_prefix') for entry in entries
                if entry.get('status') in ('launching', 'starting', 'running', 'finalizing', 'cleanup-unconfirmed')}
    prefixes.discard(None)
    total = 0
    for sandbox in client.list():
        if any(sandbox.name.startswith(prefix + '-') for prefix in prefixes):
            continue
        data = sandbox.model_dump() if hasattr(sandbox, 'model_dump') else vars(sandbox)
        resources = data.get('resources') or {}
        gpu = resources.get('gpu', 0)
        total += int(gpu or 0)
    return total


def external_local_health(entry):
    path=entry.get('local_state_path')
    record={'slot':entry['slot'],'status':'external-local','execution':'local-docker',
            'model':entry.get('requested_model'),'local_state_path':path}
    if not path or not Path(path).exists():
        record['local_status']='preparing'
        return record
    try:
        state=json.loads(Path(path).read_text())
        for field in ('status','run_id','pid','agent_started_at','deadline_at','submissions','remaining_seconds'):
            if field in state:record['local_'+field]=state[field]
        deadline=state.get('deadline_at')
        if isinstance(deadline,(int,float)):
            record['remaining_seconds']=max(0,int(deadline-time.time()))
    except (OSError,ValueError):record['local_status']='state-read-failed'
    return record


def deployment_at_capacity(entry, entries, limits):
    limit=limits.get(entry['deployment'])
    if limit is None:return False
    active={'launching','starting','running','finalizing','cleanup-unconfirmed'}
    return sum(item.get('deployment')==entry['deployment'] and item.get('status') in active
               for item in entries)>=limit


def controller_command(repo: Path, deployment_path: Path, model_path: Path, artifacts: Path,
                       entry: dict, manifest: dict) -> list[str]:
    """The ``run`` command for one admitted slot (12-hour Agent budget)."""
    command = [str(repo/'launch/pyromind.sh'), 'run', '--deployment', str(deployment_path),
               '--model-config', str(model_path), '--hours', '12', '--listen-port', str(choose_port()),
               '--artifacts', str(artifacts)]
    for field, option in (('assets_file','--assets-file'), ('judge_assets_file','--judge-assets-file')):
        if entry.get(field):
            command += [option, str(repo/entry[field])]
    # The mandatory integrity review uses one calibrated reviewer for every
    # slot unless a slot names its own (manifest-level default).
    review_config = entry.get('review_model_config', manifest.get('review_model_config'))
    if review_config:
        command += ['--review-model-config', review_config]
    return command


def run_matrix(manifest_path: Path, output: Path, *, monitor_seconds=1800, poll_seconds=15):
    from sforge.harness.agent_config import load_model_config
    from sforge.pyromind.deployment import Deployment
    from pyromind_sdk import PyroMindAPIClient

    repo = Path(__file__).resolve().parents[2]
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock = (output/'matrix.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = json.loads(manifest_path.read_text())
    validate_manifest(manifest)
    state_path = output/'matrix.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        'schema_version': 1, 'experiment': manifest['experiment'], 'max_gpu': manifest['max_gpu'],
        'budget_hours_each': 12, 'reservation_policy': 'Worker-plus-one-Judge-per-run',
        'monitor_seconds': monitor_seconds, 'timezone': 'Pacific/Auckland',
        'slots': [{**entry, 'status': 'queued'} for entry in manifest['slots']]}
    append_manifest_slots(state, manifest)
    state.pop('scheduler_stopped_at', None)
    manifest_slots={entry['slot']:entry for entry in manifest['slots']}
    for entry in state['slots']:
        selected=manifest_slots.get(entry['slot'],{})
        if selected.get('execution')=='external-local':
            if entry.get('status') in ('launching','starting','running','finalizing','cleanup-unconfirmed'):
                raise ValueError('Cannot move an allocated Sandbox run to external-local')
            entry.update(execution='external-local',status='external-local',
                         local_state_path=selected.get('local_state_path'))
            entry.pop('blocker',None)
    stopping = False
    def stop(*_):
        nonlocal stopping
        stopping = True
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop)
    client = PyroMindAPIClient(timeout=30, max_retries=0).sandboxes
    next_monitor = 0
    while not stopping:
        for entry in state['slots']:
            if entry['status']=='failed':
                run=latest_run(Path(entry['artifact_root']))
                failures=entry.setdefault('startup_failures',[])
                if startup_retry_allowed(run,len(failures)+1):
                    failures.append({'run_id':run['run_id'],'artifact_root':entry['artifact_root'],
                                     'worker_id':run.get('worker_id'),'reason':'platform-startup-failed-before-Agent',
                                     'remaining_sandbox_ids':[],'cleanup_errors':[]})
                    entry.update(status='queued',not_before=time.time()+60)
                    entry.pop('sandbox_name_prefix',None)
            if entry['status'] not in ('launching', 'starting', 'running', 'finalizing', 'cleanup-unconfirmed'):
                continue
            run = latest_run(Path(entry['artifact_root']))
            if run:
                entry.update(run_id=run['run_id'], sandbox_name_prefix=run['sandbox_name_prefix'],
                             run_status=run['status'], agent_started_at=run.get('agent_started_at'))
            alive = bool(entry.get('process_start_ticks')) and process_identity(entry['pid']) == entry['process_start_ticks']
            if alive:
                entry['status'] = ('running' if run and run.get('agent_started')
                                   else 'starting')
            elif (entry.get('gate_ack') and isinstance(entry.get('gate_expires_at'),(int,float))
                    and not Path(entry['gate_ack']).exists() and time.time()>entry['gate_expires_at']):
                # No acknowledgement means even an unrecorded gate child could
                # never exec the Controller. Its admission timeout has elapsed.
                entry.update(status='queued', blocker='Unacknowledged launch gate expired')
            elif confirmed_cleanup(run):
                entry['status'] = 'completed' if run.get('status') == 'completed' else 'failed'
                entry['finished_at'] = time.time()
            else:
                entry['status'] = 'cleanup-unconfirmed'
        try:
            external = live_external_gpu(client, state['slots'])
            state['external_gpu'] = external
            state.pop('admission_error', None)
        except Exception as exc:
            external = state['max_gpu']
            state['admission_error'] = type(exc).__name__
        for entry in state['slots']:
            if entry['status'] not in ('queued', 'blocked'):
                continue
            if time.time()<entry.get('not_before',0):
                continue
            try:
                model_path = Path(entry['model_config'])
                probe = json.loads(Path(entry['provider_probe']).read_text())
                model = load_model_config(model_path)
                if model['MODEL'] != entry.get('requested_model'):
                    raise ValueError('Configured provider model differs from the requested experiment')
                if not provider_probe_matches(probe,model_path,model):
                    raise ValueError('Provider tool probe is not qualified')
                deployment = Deployment.load(repo/entry['deployment'], repo)
                deployment.assert_launch_ready()
            except Exception as exc:
                entry.update(status='blocked', blocker=type(exc).__name__)
                continue
            count = reservation(entry, deployment)
            if deployment_at_capacity(entry,state['slots'],manifest.get('max_parallel_by_deployment',{})):
                entry.update(status='queued',blocker='Deployment concurrency ceiling')
                continue
            if external + reserved_gpu(state['slots']) + count > state['max_gpu']:
                entry.update(status='queued', blocker='GPU reservation ceiling')
                continue
            root = output/entry['slot']
            root.mkdir(exist_ok=True)
            attempt=len(entry.get('startup_failures',[]))+1
            attempt_root=root/('attempt-'+str(attempt).zfill(3))
            attempt_root.mkdir(exist_ok=True)
            log = attempt_root/'controller.log'
            command = controller_command(repo, deployment.path, model_path, attempt_root/'runs',
                                         entry, manifest)
            nonce=uuid.uuid4().hex
            gate_path=root/(nonce+'.launch.json')
            ack_path=root/(nonce+'.ack.json')
            expires_at=time.time()+60
            atomic_json(gate_path,{'nonce':nonce,'command':command,'ack_path':str(ack_path),'expires_at':expires_at})
            entry.update(status='launching',reserved_gpu=count,artifact_root=str(attempt_root/'runs'),
                         gate_ack=str(ack_path),gate_expires_at=expires_at)
            atomic_json(state_path,state)
            with log.open('ab') as stream:
                proc = subprocess.Popen([sys.executable,'-m','sforge.pyromind.launch_gate',str(gate_path)],
                                        cwd=repo, stdin=subprocess.DEVNULL, stdout=stream,
                                        stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
            entry.update(status='starting', reserved_gpu=count, pid=proc.pid,
                         process_start_ticks=process_identity(proc.pid), artifact_root=str(attempt_root/'runs'),
                         controller_log=str(log), launched_at=time.time(), model=model['MODEL'],
                         agent=model['AGENT'], reasoning_effort=model.get('REASONING_EFFORT'))
            entry.pop('blocker', None)
            entry.pop('finished_at', None)
            atomic_json(state_path, state)
            atomic_json(ack_path,{'nonce':nonce})
        state['reserved_gpu'] = reserved_gpu(state['slots'])
        state['updated_at'] = datetime.now(ZoneInfo('Pacific/Auckland')).isoformat()
        atomic_json(state_path, state)
        if time.monotonic() >= next_monitor:
            records = [external_local_health(entry) if entry.get('execution')=='external-local'
                       else health_record(entry, latest_run(Path(entry['artifact_root'])))
                       if entry.get('artifact_root') else health_record(entry, None)
                       for entry in state['slots']]
            receipt = {'checked_at': state['updated_at'], 'reserved_gpu': state['reserved_gpu'],
                       'external_gpu': external, 'slots': records}
            with (output/'monitor.jsonl').open('a') as stream:
                stream.write(json.dumps(receipt) + '\n')
            print(json.dumps(receipt), flush=True)
            next_monitor = time.monotonic() + monitor_seconds
        # Missing provider/topology configuration remains queued for correction.
        if all(entry['status'] in ('completed', 'failed') for entry in state['slots']):
            break
        time.sleep(poll_seconds)
    state['scheduler_stopped_at'] = time.time()
    atomic_json(state_path, state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--env-file', type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    from dotenv import load_dotenv
    load_dotenv(args.env_file, interpolate=False)
    run_matrix(args.manifest, args.output)


if __name__ == '__main__':
    main()
