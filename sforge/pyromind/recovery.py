"""Recover saved candidate code into a new Worker without resetting its budget.

The parent run and its immutable VersionStore remain untouched. Codex session
files are not archived by the launcher, so this is a code/history continuation,
not a restoration of the original model conversation.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import tempfile
import time

from sforge.harness.version_store import VersionStore


def _timestamp(value, field):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f'Recovery {field} must be a finite timestamp')
    return float(value)


def remaining_budget(deadline_at, *, now=None):
    """Fail before allocation/startup if the original absolute budget expired."""
    remaining = _timestamp(deadline_at, 'deadline_at') - (time.time() if now is None else now)
    if remaining <= 0:
        raise ValueError('Original Agent deadline has expired; recovery cannot start a new budget')
    return remaining


@dataclass(frozen=True)
class RecoveryPlan:
    parent_directory: Path
    parent_run_id: str
    parent_version_id: str
    archive: bytes
    archive_sha256: str
    members: tuple[str, ...]
    budget_seconds: int
    original_agent_started_at: float
    original_deadline_at: float
    recovery_count: int
    public_history: dict

    def metadata(self, *, now=None):
        return {'parent_run_id': self.parent_run_id,
                'parent_version_id': self.parent_version_id,
                'parent_archive_sha256': self.archive_sha256,
                'original_agent_started_at': self.original_agent_started_at,
                'original_deadline_at': self.original_deadline_at,
                'original_budget_seconds': self.budget_seconds,
                'remaining_seconds_at_recovery': max(0, self.original_deadline_at -
                                                    (time.time() if now is None else now)),
                'recovery_count': self.recovery_count,
                'conversation_restored': False,
                'mode': 'saved-candidate-and-public-history'}

    def guidance(self):
        deadline = datetime.fromtimestamp(self.original_deadline_at, timezone.utc).isoformat()
        return (' This is a recovery of your earlier interrupted run, not a new scratch attempt. '
                'The saved candidate files and notes.md are already restored. Do not reinitialize or '
                'overwrite retained candidates. Read notes.md and /home/agent/recovery-history.json '
                'for the earlier agent-visible results. Keep FINAL_VERSION on the best previously '
                'Judge-passing candidate until a new submission improves it, and continue unfinished '
                'experiments in new or retained candidate files. The original Codex conversation '
                'was not restored. Startup, downtime and feedback waits count toward the original '
                f'absolute deadline {deadline}; no new budget begins. ')


def load_recovery(parent_directory, *, version_id=None, task, model, runtime,
                  worker_image, judge_image, now=None):
    """Bind an immutable, submit-eligible snapshot to the original run deadline."""
    parent_directory = Path(parent_directory).expanduser().resolve()
    parent = json.loads((parent_directory / 'run.json').read_text())
    if not isinstance(parent, dict) or parent.get('status') != 'failed':
        raise ValueError('Recovery requires a failed parent run; a running run must not be duplicated')
    parent_run_id = parent.get('run_id')
    if not isinstance(parent_run_id, str) or parent_run_id != parent_directory.name:
        raise ValueError('Recovery parent run identity does not match its directory')
    expected = {'task_id': task.task_id, 'model': model['MODEL'],
                'reasoning_effort': model.get('REASONING_EFFORT', 'high'),
                'runtime_version': runtime.version,
                'worker_image': worker_image, 'judge_image': judge_image}
    for field, value in expected.items():
        if parent.get(field) != value:
            raise ValueError(f'Recovery {field} differs from the parent run')
    for field in ('cleanup_errors', 'remaining_sandbox_ids'):
        completed = parent.get(field)
        if not isinstance(completed, list) or completed:
            raise ValueError('Recovery requires confirmed cleanup of the failed parent run')
    budget = parent.get('budget_seconds')
    if isinstance(budget, bool) or not isinstance(budget, int) or not 1 <= budget <= 86400:
        raise ValueError('Recovery parent Agent budget is invalid')
    previous = parent.get('recovery') or {}
    progress = parent.get('agent_progress') or {}
    started = _timestamp(previous.get('original_agent_started_at',
                                      progress.get('started_at', parent.get('agent_started_at'))),
                         'agent_started_at')
    deadline = _timestamp(previous.get('original_deadline_at', progress.get('deadline_at')),
                          'deadline_at')
    if abs(deadline - started - budget) > 1:
        raise ValueError('Recovery original deadline does not match the original Agent budget')
    remaining_budget(deadline, now=now)
    count = previous.get('recovery_count', 0)
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError('Recovery parent recovery_count is invalid')
    store = VersionStore.open(parent_directory / 'version-store')
    if store.store_id != parent_run_id:
        raise ValueError('Recovery parent VersionStore identity mismatch')
    versions = store.list_versions(task=task.task_id, run_id=parent_run_id,
                                   source=None if version_id else 'checkpoint')
    if version_id:
        versions = [version for version in versions if version.version_id == version_id]
    if not versions:
        raise ValueError('Recovery saved version was not found in the parent run')
    version = versions[-1]
    archive = version.archive_path.read_bytes()
    if len(archive) != version.archive_size or hashlib.sha256(archive).hexdigest() != version.archive_sha256:
        raise ValueError('Recovery archive size or SHA-256 differs from its immutable manifest')
    members = store.validate_archive(task_id=task.task_id, archive=archive,
        submit_paths=task.submit_paths, submit_exclude=task.submit_exclude)
    history_path = parent_directory / 'history.json'
    history = json.loads(history_path.read_text()) if history_path.exists() else {}
    # history.json is already the controller's agent-visible view. Copy only
    # its documented summary fields, never private evaluator reports/logs.
    history = {key: history[key] for key in ('run_id', 'best_score', 'best_pass_rate', 'best_round',
               'agent_submissions', 'entries') if isinstance(history, dict) and key in history}
    if 'entries' in history:
        visible_fields = {'round', 'status', 'valid', 'score', 'pass_rate', 'passed', 'failed',
                          'summary', 'version_id', 'candidate_id', 'submitted_at'}
        if not isinstance(history['entries'], list):
            raise ValueError('Recovery parent public history entries are invalid')
        history['entries'] = [{key: value for key, value in entry.items() if key in visible_fields}
                              for entry in history['entries'] if isinstance(entry, dict)]
    return RecoveryPlan(parent_directory, parent_run_id, version.version_id, archive,
                        version.archive_sha256, tuple(members), budget, started, deadline,
                        count + 1, history)


_RESTORE_SOURCE = r'''import hashlib,json,os,pathlib,pwd,stat,tarfile,time
cfg=json.loads(__import__('sys').argv[1])
assert time.time()<cfg['deadline_at'], 'Original Agent deadline expired during Worker startup'
archive=pathlib.Path(cfg['archive_path'])
assert hashlib.sha256(archive.read_bytes()).hexdigest()==cfg['archive_sha256'], 'Recovery archive SHA mismatch'
root=pathlib.Path(cfg['cwd'])
assert root.is_absolute() and root.is_dir() and not root.is_symlink(), 'Invalid recovery workspace'
agent=pwd.getpwnam('agent')
verified=[]
with tarfile.open(archive) as bundle:
    members={member.name:member for member in bundle.getmembers() if member.isfile()}
    assert set(members)==set(cfg['members']), 'Recovery archive file contract changed'
    for name in cfg['members']:
        components=pathlib.PurePosixPath(name).parts
        assert components and not name.startswith('/') and all(p not in ('','..','.') for p in components)
        directory=root
        for part in components[:-1]:
            directory=directory/part
            if directory.exists() or directory.is_symlink():
                assert directory.is_dir() and not directory.is_symlink(), 'Recovery parent is not a real directory'
            else:
                directory.mkdir(mode=0o755); os.chown(directory,agent.pw_uid,agent.pw_gid)
        target=directory/components[-1]
        if target.exists() or target.is_symlink():
            details=target.lstat()
            assert stat.S_ISREG(details.st_mode) and details.st_nlink==1, 'Recovery target must be a regular non-linked file'
        member=members[name]
        contents=bundle.extractfile(member).read()
        descriptor=os.open(target,os.O_WRONLY|os.O_CREAT|os.O_TRUNC|os.O_NOFOLLOW,0o600)
        with os.fdopen(descriptor,'wb') as output: output.write(contents)
        os.chmod(target,0o755 if member.mode&0o111 else 0o644)
        os.chown(target,agent.pw_uid,agent.pw_gid)
        expected=hashlib.sha256(contents).hexdigest()
        assert hashlib.sha256(target.read_bytes()).hexdigest()==expected, 'Restored candidate file SHA mismatch'
        verified.append({'path':name,'sha256':expected,'bytes':len(contents)})
print(json.dumps({'archive_sha256':cfg['archive_sha256'],'files':verified,'conversation_restored':False}))
'''


def restore_worker(plan, backend, worker, task, checked):
    """Restore ordinary candidate files before any Agent or initial save starts."""
    remaining_budget(plan.original_deadline_at)
    remote = '/run/sforge-agent/recovery.tar.gz'
    with tempfile.TemporaryDirectory(prefix='sforge-recovery-') as directory:
        archive_path = Path(directory) / 'submission.tar.gz'
        archive_path.write_bytes(plan.archive)
        archive_path.chmod(0o600)
        backend.copy_to_container(worker, archive_path, PurePosixPath(remote))
    cfg = {'archive_path': remote, 'archive_sha256': plan.archive_sha256,
           'members': plan.members, 'cwd': task.cwd, 'deadline_at': plan.original_deadline_at}
    try:
        evidence = json.loads(checked(backend, worker, ['python3', '-c', _RESTORE_SOURCE,
                                json.dumps(cfg)], timeout=120))
    finally:
        backend.client.delete_file(worker.id, remote)
    backend.write_to_container(worker, json.dumps(plan.public_history, indent=2),
                               PurePosixPath('/home/agent/recovery-history.json'))
    checked(backend, worker, ['chown', 'agent:agent', '/home/agent/recovery-history.json'], timeout=15)
    return evidence
