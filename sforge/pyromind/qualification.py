"""Run original Judges on a trusted positive control, without exposing it to a Worker."""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import threading
import tarfile
import time
import uuid

from sforge.pyromind.cli import write_json
from sforge.pyromind.deployment import Deployment, validate_mount_separation, qualification_contract_digest
from sforge.pyromind.runner import (
    configure_judge_runtime, asset_bindings_for_role, asset_mounts_for_role, guard_judge_backend, private_context_environment, encode_private_context, private_context_secrets,
    task_submission_evaluator, termination_cleanup,
    reconcile_creation_ownership,
)


def positive_result(report):
    """A rejected empty starter or an infrastructure error is not a positive control."""
    score = report.get('score')
    return (report.get('valid') is True and type(score) in (int, float)
            and math.isfinite(score) and score > 0
            and report.get('metrics', {}).get('evaluation_status') != 'infrastructure_error')


def native_requalification_spec(deployment, module, args):
    if not getattr(args, 'native_requalification', False):
        deployment.assert_launch_ready(preflight=True)
        return None
    if not hasattr(module, 'qualification_runtime_spec') or not hasattr(module, 'qualification_evaluation_context'):
        raise ValueError('This task has no native requalification adapter')
    spec = module.qualification_runtime_spec()
    blockers = set(deployment.config.get('execution_blockers', []))
    allowed = set(spec.get('allowed_execution_blockers', []))
    if (spec.get('mode') != 'non-release-qualification'
            or spec.get('release_qualification_permitted') is not False
            or not allowed or not blockers <= allowed
            or list(args.rounds) != ['final'] or not getattr(args, 'model_validation', None)):
        raise ValueError('Native non-release qualification requires its exact gates, final-only round and full model validation')
    return spec


def install_checkpoint_capture(backend, output, spec):
    """Persist the native checkpoint bytes before deletion; never silently retry/reset deadlines."""
    path = PurePosixPath(spec['checkpoint_mount'])
    if path != PurePosixPath('/logs/verifier/checkpoints'):
        raise ValueError('Unsupported native checkpoint location')
    started, cleanup = backend.on_started, backend.cleanup_container
    prepared = set()
    def prepare(handle):
        started(handle)
        backend._checked(handle,['mkdir','-p',str(path)])
        backend._checked(handle,['chown','root:root',str(path)])
        backend._checked(handle,['chmod','700',str(path)])
        prepared.add(handle.id)
    def persist_then_cleanup(handle, logger=None):
        error = None
        try:
            if handle is not None and handle.id in prepared:
                payload = backend.copy_from_container(handle,path)
                checkpoint_files = validate_checkpoint_archive(payload)
                directory = output/'.private/checkpoints';directory.mkdir(parents=True,mode=0o700,exist_ok=True)
                archive = directory/(handle.id+'.tar')
                with archive.open('wb') as stream:
                    stream.write(payload);stream.flush();os.fsync(stream.fileno())
                archive.chmod(0o600)
                write_json(directory/(handle.id+'.json'),{'sandbox_id':handle.id,'source':str(path),
                    'sha256':hashlib.sha256(payload).hexdigest(),'bytes':len(payload),'copied_before_delete':True,
                    'verified_archive_members':checkpoint_files,
                    'automatic_retry':False,'deadline_reset':False})
        except Exception as exc:
            error = type(exc).__name__
            write_json(output/'.private/checkpoint-capture-error.json',{'error':error})
        finally:
            cleanup(handle,logger)
        if error:
            raise RuntimeError('Private checkpoint capture failed before cleanup: '+error)
    backend.on_started=prepare
    backend.cleanup_container=persist_then_cleanup


def validate_checkpoint_archive(payload):
    """Check complete regular-file payloads without extracting any remote paths."""
    with tarfile.open(fileobj=io.BytesIO(payload),mode='r:*') as bundle:
        members=bundle.getmembers()
        if not members or not any(m.name.rstrip('/')=='checkpoints' and m.isdir() for m in members):
            raise ValueError('Checkpoint archive has no root directory')
        seen=set()
        for member in members:
            path=PurePosixPath(member.name)
            if (path.is_absolute() or '..' in path.parts or path.parts[0]!='checkpoints'
                    or member.name in seen or not (member.isfile() or member.isdir())):
                raise ValueError('Unsafe checkpoint archive member')
            seen.add(member.name)
            if member.isfile():
                stream=bundle.extractfile(member);size=0
                for block in iter(lambda:stream.read(1024*1024),b''):size+=len(block)
                if size!=member.size:raise ValueError('Checkpoint archive is truncated')
        return sorted(seen)


def check_native_report(path, spec):
    """Check completion, while preserving the native distinction from release evidence."""
    with tarfile.open(path,'r:gz') as bundle:
        members=bundle.getmembers()
        if len(members)!=1 or not members[0].isfile():
            raise ValueError('Unexpected native private report archive')
        detail=json.load(bundle.extractfile(members[0]))
    contract=spec['success_contract']
    if (any(detail.get(key) is not expected for key,expected in contract['private_flags'].items())
            or detail.get('generation_count') != contract['generation_count']
            or len(detail.get('records',[])) != contract['record_count']
            or len(detail.get('rows',{})) != contract['row_count']
            or len(detail.get('row_commitments',[])) != contract['row_commitment_count']):
        raise RuntimeError('Native non-release qualification did not complete its original full workload')
    return {'qualification':True,'release_evidence':False,'generation_count':detail['generation_count'],
            'rows':len(detail['rows']),'records':len(detail['records'])}


def qualify_task(args):
    with termination_cleanup():
        return _qualify_task(args)


def _qualify_task(args):
    from pyromind_sdk.client.models import VolumeMount
    from sforge.harness.backend.pyromind_backend import PyroMindBackend, PyroMindHandle
    from sforge.harness.benchmark import load_benchmark
    from sforge.harness.config import SForgeConfig
    from sforge.harness.run_evaluation import judge_submission, _validate_submission_archive, PRIVATE_JUDGE_ARTIFACT_NAME
    from sforge.harness.task_spec import load_all_tasks

    repo = Path(__file__).resolve().parents[2]
    deployment = Deployment.load(Path(args.deployment), repo)
    deployment_source_sha256 = hashlib.sha256(deployment.path.read_bytes()).hexdigest()
    private_module = deployment.bootstrap()
    native_spec = native_requalification_spec(deployment, private_module, args)
    task_root = deployment.local_path('task_directory')
    tasks_dir = task_root / 'tasks'
    task = load_all_tasks(tasks_dir, load_benchmark(tasks_dir))[0]
    if task.task_id != deployment.config['task_id']:
        raise ValueError('Task identity differs from deployment')
    images = json.loads(deployment.local_path('images_file').read_text())
    image = images['judge']
    if '@sha256:' not in image:
        raise ValueError('Qualification requires an immutable published image')
    candidate = Path(args.candidate_archive).resolve()
    if not candidate.is_file() or candidate.is_symlink():
        raise ValueError('Qualification requires a controller-local regular candidate archive')
    archive = candidate.read_bytes()
    _validate_submission_archive(archive)
    bindings = deployment.asset_bindings(getattr(args, 'assets_file', None))
    private_bindings = deployment.judge_asset_bindings(getattr(args, 'judge_assets_file', None))
    validate_mount_separation(bindings, private_bindings)
    if bindings or private_bindings:
        from sforge.pyromind.storage import storage_connection
        _, bucket, endpoint = storage_connection()
        identity = {'endpoint': endpoint, 'bucket': bucket}
        if any(item.get('storage_identity', identity) != identity for item in bindings+private_bindings):
            raise ValueError('Qualification assets belong to a different Storage account')
    resources = deployment.resources_for_role('judge')
    private_metadata = private_module.private_payload_metadata(task_root)
    run_id = 'q-' + deployment.config['name'][:12] + '-' + time.strftime('%Y%m%d-%H%M%S', time.gmtime()) + '-' + uuid.uuid4().hex[:6]
    output = Path(args.artifacts).resolve() / run_id
    output.mkdir(parents=True, mode=0o700, exist_ok=False)
    output.chmod(0o700)
    (output/'candidate.tar.gz').write_bytes(archive)
    (output/'candidate.tar.gz').chmod(0o600)
    write_json(output/'deployment.json',deployment.config)
    write_json(output/'images.json',images)
    write_json(output/'private-payload-manifest.json', private_metadata)
    events = output/'sandbox-events.jsonl'
    judge_bindings = asset_bindings_for_role(bindings, 'judge')
    mounts = asset_mounts_for_role(judge_bindings+private_bindings, 'judge', VolumeMount)
    backend = PyroMindBackend(event_path=events, gpu_count=resources['gpu'],
                             gpu_card=resources.get('gpu_card'), volume_mounts=mounts)
    backend.ready_timeout = deployment.sandbox_ready_timeout_seconds
    cancelled = threading.Event()
    judge_environment = {**deployment.config.get('judge_environment',{}),
                         **(native_spec or {}).get('judge_environment',{})}
    guard_judge_backend(backend, cancelled, run_id, environment=judge_environment)
    configure_judge_runtime(backend,deployment,private_module,task_root,output,
                            judge_bindings,private_metadata,private_bindings)
    if native_spec:
        context = private_module.qualification_evaluation_context(task_root,run_id,output,
            archive=candidate,work_image_id=images['worker'].split('@',1)[1],
            judge_image_id=image.split('@',1)[1],model_validation=args.model_validation,
            runtime_entry=Path(__file__))
        private_env = encode_private_context(context)
        install_checkpoint_capture(backend,output,native_spec)
    else:
        private_env = private_context_environment(private_module,task_root,run_id,output)
    config = SForgeConfig(backend='pyromind',tasks_dir=tasks_dir,log_dir=output/'logs',
                         judge_cpu_limit=resources['cpu'],judge_mem_limit=resources['memory'],
                         judge_extra_env=private_env)
    evaluate = task_submission_evaluator(private_module,task_root,judge_submission)
    state = {'schema_version':1,'run_id':run_id,'task_id':task.task_id,
             'status':'running','scope':'trusted-positive-original-judge-protocol',
             'worker_created':False,'agent_started':False,'shared_artifacts':False,
             'judge_image':image,'resources':resources,'rounds':list(args.rounds),
             'deployment_sha256':deployment_source_sha256,
             'runtime_contract_sha256':qualification_contract_digest(deployment.config),
             'deployment_snapshot_sha256':hashlib.sha256((output/'deployment.json').read_bytes()).hexdigest(),
             'images_snapshot_sha256':hashlib.sha256((output/'images.json').read_bytes()).hexdigest(),
             'candidate_source':str(candidate),'candidate_sha256':hashlib.sha256(archive).hexdigest(),
             'private_manifest_sha256':hashlib.sha256(json.dumps(private_metadata,sort_keys=True).encode()).hexdigest(),
             'started_at':time.time(),'results':{}}
    if native_spec:
        state.update(scope='native-non-release-requalification',native_protocol=native_spec,
                     release_qualification_permitted=False)
    write_json(output/'run.json',state)
    print(f'QUALIFICATION_DIRECTORY={output}',flush=True)
    try:
        for mode in args.rounds:
            round_id = deployment.evaluation_round_id(mode)
            state.update(current_round=mode);write_json(output/'run.json',state)
            print(f'{deployment.config["name"]}: starting original {mode} evaluation',flush=True)
            report = evaluate(task_spec=task,archive=archive,config=config,backend=backend,
                submission_id=run_id+'-'+mode,run_id=run_id,round_id=round_id,
                log_dir=output/mode,judge_image_ref=image)
            record = report.to_dict()
            write_json(output/(mode+'.json'),record)
            passed = positive_result(record)
            if passed and native_spec:
                state['native_workload_verification'] = check_native_report(
                    output/mode/PRIVATE_JUDGE_ARTIFACT_NAME,native_spec)
            state['results'][mode] = {'passed':passed,'valid':record.get('valid'),
                                      'score':record.get('score'),'report':mode+'.json','round_id':round_id}
            write_json(output/'run.json',state)
            if not passed:
                raise RuntimeError(f'Original {mode} positive-control evaluation failed; inspect {output/(mode+".json")}')
            print(f'{deployment.config["name"]}: {mode} valid=true score={record["score"]}',flush=True)
        state.update(status='non-release-passed' if native_spec else 'passed')
    except BaseException as exc:
        message = str(exc)
        for value in (os.getenv('PYROMIND_API_KEY'),os.getenv('PYROMIND_STORAGE_SECRET_KEY'),*private_context_secrets(config.judge_extra_env)):
            if value:
                message = message.replace(value,'[REDACTED]')
        state.update(status='failed',error=message)
        raise RuntimeError(message) from None
    finally:
        cancelled.set()
        cleanup_errors = []
        owned = {}
        if events.exists():
            for line in events.read_text().splitlines():
                entry = json.loads(line)
                if entry.get('event') == 'created':owned[entry['id']] = entry['name']
        try:
            # Reconcile exact created IDs and late creates with this unique run prefix.
            current = backend.client.list()
            for sandbox in current:
                if sandbox.id in owned or sandbox.name.startswith(run_id+'-'):
                    try:backend.cleanup_container(PyroMindHandle(sandbox.id,sandbox.name))
                    except Exception as exc:cleanup_errors.append(type(exc).__name__+':'+sandbox.id)
            remaining = [s.id for s in backend.client.list() if s.id in owned or s.name.startswith(run_id+'-')]
        except Exception as exc:
            cleanup_errors.append(type(exc).__name__);remaining=None
        ownership = reconcile_creation_ownership((backend,))
        cleanup_errors.extend(ownership['cleanup_errors'])
        if remaining is not None:
            remaining = sorted(set(remaining) | set(ownership['remaining_sandbox_ids']))
        if ownership['unresolved_creation_names'] or ownership['cleanup_errors']:
            remaining = None
        state['unresolved_creation_names'] = ownership['unresolved_creation_names']
        state.update(finished_at=time.time(),cleanup_errors=cleanup_errors,remaining_sandbox_ids=remaining)
        if cleanup_errors or remaining != []:
            state['status']='cleanup-failed'
        write_json(output/'run.json',state)
    if state['status'] not in {'passed','non-release-passed'}:
        raise RuntimeError('Qualification cleanup is incomplete: '+str(output))
    print(f'PASS: original evaluations completed; all Judges deleted: {output}',flush=True)
    return output
