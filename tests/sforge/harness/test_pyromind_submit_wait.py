"""Execute the installed, package-independent hooks, including process races."""
import fcntl
import datetime
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import tomllib
from types import SimpleNamespace

import pytest

from sforge.harness.submit_wait import (
    SUBMIT_WAIT_DEFAULT_TIMEOUT_MS, SUBMIT_WAIT_HOOK_PATH, SUBMIT_WAIT_STATE_DIR,
    generate_submit_wait_hook_script, submit_wait_hooks_config_fragment,
)


@pytest.fixture
def hook(tmp_path):
    script = tmp_path / 'standalone-hook.py'
    script.write_text(generate_submit_wait_hook_script())
    state = tmp_path / 'gate'
    binary_directory = tmp_path / 'bin'
    binary_directory.mkdir()
    submit = binary_directory / 'sforge-submit'
    submit.write_text('#!' + sys.executable + '''
import os, pathlib, time
marker = os.environ.get('FAKE_SUBMIT_MARKER')
if marker:
    pathlib.Path(marker).write_text('started')
time.sleep(float(os.environ.get('FAKE_SUBMIT_SLEEP', '0')))
print('Score: 180.0')
raise SystemExit(int(os.environ.get('FAKE_SUBMIT_EXIT', '0')))
''')
    submit.chmod(0o755)
    env = dict(os.environ, SFORGE_SUBMIT_WAIT_STATE_DIR=str(state))
    env['PATH'] = str(binary_directory) + os.pathsep + env['PATH']
    env.pop('SFORGE_SUBMIT_TOOL_TIMEOUT_MS', None)
    env.pop('PYTHONPATH', None)

    def run(phase, event, **changes):
        result = subprocess.run([sys.executable, '-I', str(script), phase],
                                input=json.dumps(event), text=True, capture_output=True,
                                env={**env, **changes}, timeout=10)
        assert result.returncode == 0, result.stderr
        assert not result.stderr
        return json.loads(result.stdout)

    run.script, run.state, run.env, run.binary_directory = script, state, env, binary_directory
    return run


def pre(command='sforge-submit', identifier='submit-1', session='session-1', tool='Bash'):
    return {'hook_event_name': 'PreToolUse', 'session_id': session, 'tool_use_id': identifier,
            'cwd': str(Path.cwd()),
            'tool_name': tool, 'tool_input': {'command': command, 'timeout_ms': 1000}}


def post(identifier='submit-1', session='session-1', response=None):
    return {'hook_event_name': 'PostToolUse', 'session_id': session, 'tool_use_id': identifier,
            'tool_name': 'Bash', 'tool_input': {'command': 'sforge-submit'},
            'tool_response': {'exit_code': 0, 'output': 'Score: 180.0'} if response is None else response}


def decision(result):
    return result.get('hookSpecificOutput', {}).get('permissionDecision')


def state(hook):
    return json.loads((hook.state / 'state.json').read_text())


@pytest.fixture
def rollout(tmp_path):
    session = '01a0f8de-b002-74b1-bb63-0fb2b78d9a6d'
    home = tmp_path / 'codex-home'
    stamp = int(session.replace('-', '')[:12], 16) / 1000
    date = datetime.datetime.fromtimestamp(stamp, datetime.timezone.utc).strftime('%Y/%m/%d')
    directory = home / 'sessions' / date
    directory.mkdir(parents=True)
    path = directory / ('rollout-2026-10-01T19-08-58-' + session + '.jsonl')
    metadata = {'type': 'session_meta', 'payload': {'id': session, 'session_id': session}}
    path.write_text(json.dumps(metadata) + '\n')

    def append(item):
        with path.open('a') as stream:
            stream.write(json.dumps({'type': 'response_item', 'payload': item}) + '\n')

    def event(identifier='failed-patch', tool='apply_patch', command='a patch'):
        return {**pre(command, identifier, session, tool), 'transcript_path': str(path)}

    def call(identifier='failed-patch', name='apply_patch', arguments=None, custom=True):
        append({'type': 'custom_tool_call' if custom else 'function_call', 'call_id': identifier,
                'name': name, 'input': 'a patch' if custom else None,
                'arguments': json.dumps(arguments or {})})

    def output(identifier='failed-patch', text='apply_patch verification failed: missing context', custom=True):
        append({'type': 'custom_tool_call_output' if custom else 'function_call_output',
                'call_id': identifier, 'output': text})

    return {'home': home, 'path': path, 'session': session, 'metadata': metadata,
            'event': event, 'call': call, 'output': output, 'append': append,
            'env': {'CODEX_HOME': str(home)}}


def reconcile(hook, **environment):
    result = subprocess.run([sys.executable, '-I', str(hook.script), 'reconcile'],
                            capture_output=True, text=True,
                            env={**hook.env, **environment}, timeout=10)
    assert not result.stderr
    return result.returncode, json.loads(result.stdout)


def test_failed_native_patch_without_post_is_reconciled_before_submit(hook, rollout):
    hook('pre', rollout['event'](), **rollout['env'])
    rollout['call']()
    rollout['output']()
    assert len(state(hook)['active_tools']) == 1
    assert decision(hook('pre', pre(session=rollout['session']), **rollout['env'])) == 'allow'
    assert state(hook)['active_tools'] == {}
    assert state(hook)['active']['tool_use_id'] == 'submit-1'


@pytest.mark.parametrize('output', [
    'Chunk ID: abc\nProcess exited with code 1\nOutput:\nordinary experiment failed',
    'exec_command failed: command could not be executed',
    {'exit_code': 1, 'status': 'error', 'output': 'ordinary tool failure'},
])
def test_ordinary_shell_failure_without_post_is_terminal_and_does_not_block_submit(hook, rollout, output):
    event = rollout['event']('failed-experiment', 'Bash', 'exit 1')
    hook('pre', event, **rollout['env'])
    rollout['call']('failed-experiment', 'exec_command', {'cmd': 'exit 1'}, custom=False)
    rollout['output']('failed-experiment', output, custom=False)
    assert state(hook)['last_completion'] is None
    code, summary = reconcile(hook, **rollout['env'])
    assert code == 0 and summary['reconciled_tools'] == 1
    assert state(hook)['active_tools'] == {} and state(hook)['last_completion'] is None
    assert decision(hook('pre', pre(session=rollout['session']), **rollout['env'])) == 'allow'


def test_completed_log_printing_running_marker_is_proven_terminal(hook, rollout):
    hook('pre', rollout['event']('cat-log', 'Bash', 'cat logfile'), **rollout['env'])
    # Codex Post carries raw stdout, so this ambiguous message conservatively
    # remains pending until the exact rollout supplies its runtime header.
    raw = 'Process running with session ID 111\n'
    hook('post', post('cat-log', rollout['session'], response=raw), **rollout['env'])
    rollout['call']('cat-log', 'exec_command', {'cmd': 'cat logfile'}, custom=False)
    rollout['output']('cat-log', 'Chunk ID: abc\nProcess exited with code 0\nOutput:\n' + raw, custom=False)
    assert len(state(hook)['active_tools']) == 1
    assert decision(hook('pre', pre(session=rollout['session']), **rollout['env'])) == 'allow'
    assert state(hook)['active_tools'] == {}


@pytest.mark.parametrize('stdout', [
    'Process running with session ID 111\n',
    'An old experiment timed out with exit code 1\n',
    'Chunk ID: historical\nProcess running with session ID 111\nOutput:\nold log',
    'Warning: truncated output\nProcess running with session ID 111\n' + 'feedback ' * 100,
])
def test_owned_replay_terminal_post_uses_cache_exit_not_arbitrary_stdout(hook, stdout):
    reply = hook('pre', pre())
    event = post(response=stdout)
    event['tool_input'] = reply['hookSpecificOutput']['updatedInput']
    assert hook('post', event) == {}
    assert state(hook)['active'] is None
    assert state(hook)['last_completion']['outcome'] == 'completed'


def test_owned_replay_structured_running_response_still_blocks(hook):
    reply = hook('pre', pre())
    event = post(response={'session_id': 123, 'output': 'Score: 180.0'})
    event['tool_input'] = reply['hookSpecificOutput']['updatedInput']
    assert hook('post', event)['decision'] == 'block'
    assert state(hook)['active'] is not None


def test_structured_terminal_metadata_outranks_printed_running_marker(hook):
    hook('pre', pre('cat logfile', 'cat-log'))
    event = post('cat-log', response={'exit_code': 0, 'output': 'Process running with session ID 111'})
    assert hook('post', event) == {}
    assert state(hook)['active_tools'] == {}


def test_submit_structured_exit_zero_with_printed_running_marker_releases(hook):
    hook('pre', pre())
    event = post(response={'exit_code': 0, 'output': 'Process running with session ID 111'})
    assert hook('post', event) == {}
    assert state(hook)['active'] is None
    assert state(hook)['last_completion']['outcome'] == 'completed'


def test_different_replay_command_cannot_approve_a_running_submit_post(hook):
    hook('pre', pre())
    event = post(response='Process running with session ID 123')
    event['tool_input'] = {'command': 'python3 ' + str(hook.script) + ' replay ' + '0' * 32}
    assert hook('post', event)['decision'] == 'block'
    assert state(hook)['active'] is not None


def test_reconcile_only_retires_proven_matching_call_and_leaves_other_work(hook, rollout):
    hook('pre', rollout['event'](), **rollout['env'])
    hook('pre', rollout['event']('other-work', 'Bash', 'sleep 500'), **rollout['env'])
    rollout['call']()
    rollout['output']()
    code, summary = reconcile(hook, **rollout['env'])
    assert code == 0 and summary['reconciled_tools'] == 1
    assert [t['tool_use_id'] for t in summary['pending_tools']] == ['other-work']
    assert decision(hook('pre', pre(session=rollout['session']), **rollout['env'])) == 'deny'


def test_reconciliation_summary_never_includes_output_or_transcript_contents(hook, rollout):
    hook('pre', rollout['event'](), **rollout['env'])
    rollout['call']()
    rollout['output'](text='SECRET OUTPUT MUST NEVER ENTER DIAGNOSTICS')
    code, summary = reconcile(hook, **rollout['env'])
    assert code == 0
    assert summary == {'schema_version': 1, 'active_submission': False,
                       'pending_tools': [], 'reconciled_tools': 1}
    assert 'SECRET' not in json.dumps(summary)


@pytest.mark.parametrize('damage', [
    'missing', 'malformed', 'wrong-session', 'wrong-call', 'partial-output',
    'duplicate-call', 'wrong-output-kind', 'oversized-line',
])
def test_missing_invalid_or_ambiguous_rollout_never_releases_pending_call(hook, rollout, damage):
    hook('pre', rollout['event'](), **rollout['env'])
    rollout['call']()
    if damage == 'wrong-call':
        rollout['output']('a-different-call')
    elif damage == 'duplicate-call':
        rollout['call']()
        rollout['output']()
    elif damage == 'wrong-output-kind':
        rollout['output'](custom=False)
    elif damage == 'partial-output':
        rollout['output']()
        payload = rollout['path'].read_bytes()
        rollout['path'].write_bytes(payload[:-1])
    else:
        rollout['output']()
    if damage == 'missing':
        rollout['path'].unlink()
    elif damage == 'malformed':
        with rollout['path'].open('a') as stream:
            stream.write('not-json\n')
    elif damage == 'wrong-session':
        content = rollout['path'].read_text().splitlines()
        content[0] = json.dumps({'type': 'session_meta', 'payload': {'id': 'different-session'}})
        rollout['path'].write_text('\n'.join(content) + '\n')
    elif damage == 'oversized-line':
        with rollout['path'].open('a') as stream:
            stream.write(json.dumps({'type': 'noise', 'payload': 'x' * (1024 * 1024)}) + '\n')
    assert reconcile(hook, **rollout['env'])[1]['reconciled_tools'] == 0
    assert decision(hook('pre', pre(session=rollout['session']), **rollout['env'])) == 'deny'
    assert len(state(hook)['active_tools']) == 1


@pytest.mark.parametrize('location', ['auth-directory', 'outside-home', 'symlink-file', 'symlink-directory'])
def test_proof_paths_are_confined_to_real_codex_session_files(hook, rollout, tmp_path, location):
    original = rollout['path']
    if location == 'auth-directory':
        path = rollout['home'] / '.config' / original.name
    elif location == 'outside-home':
        path = tmp_path / original.name
    elif location == 'symlink-file':
        path = original.parent / ('rollout-linked-' + rollout['session'] + '.jsonl')
    else:
        path = rollout['home'] / 'sessions' / 'linked' / original.name
    path.parent.mkdir(parents=True, exist_ok=True)
    if location == 'symlink-file':
        path.symlink_to(original)
    elif location == 'symlink-directory':
        path.parent.rmdir()
        path.parent.symlink_to(original.parent, target_is_directory=True)
    else:
        path.write_bytes(original.read_bytes())
    event = {**rollout['event'](), 'transcript_path': str(path)}
    hook('pre', event, **rollout['env'])
    # Force an invalid recorded path to verify it cannot become proof even when
    # the file body contains a matching terminal response.
    value = state(hook)
    next(iter(value['active_tools'].values()))['transcript_path'] = str(path)
    (hook.state / 'state.json').write_text(json.dumps(value))
    if location != 'symlink-file':
        with path.open('a') as stream:
            stream.write(json.dumps({'type': 'response_item', 'payload': {
                'type': 'custom_tool_call', 'call_id': 'failed-patch', 'name': 'apply_patch'}}) + '\n')
            stream.write(json.dumps({'type': 'response_item', 'payload': {
                'type': 'custom_tool_call_output', 'call_id': 'failed-patch', 'output': 'failure'}}) + '\n')
    assert reconcile(hook, **rollout['env'])[1]['reconciled_tools'] == 0
    assert decision(hook('pre', pre(session=rollout['session']), **rollout['env'])) == 'deny'


def test_old_ledger_without_transcript_path_uses_exact_uuid_session_file(hook, rollout):
    event = rollout['event']()
    del event['transcript_path']
    hook('pre', event, **rollout['env'])
    assert 'transcript_path' not in next(iter(state(hook)['active_tools'].values()))
    rollout['call']()
    rollout['output']()
    assert reconcile(hook, **rollout['env'])[1]['reconciled_tools'] == 1


def test_old_ledger_duplicate_session_files_are_ambiguous(hook, rollout):
    event = rollout['event']()
    del event['transcript_path']
    hook('pre', event, **rollout['env'])
    rollout['call']()
    rollout['output']()
    archive = rollout['home'] / 'archived_sessions'
    archive.mkdir()
    (archive / rollout['path'].name).write_bytes(rollout['path'].read_bytes())
    assert reconcile(hook, **rollout['env'])[1]['reconciled_tools'] == 0


@pytest.mark.parametrize('completion', ['none', 'running-again', 'wrong-session', 'unknown-process', 'stdout-spoof'])
def test_async_original_call_remains_pending_without_correlated_process_exit(hook, rollout, completion, tmp_path):
    # `sleep 500` is still alive as the Agent user, so even an unknown handle
    # cannot release the gate.
    rollout['env']['SFORGE_SUBMIT_WAIT_PROC_ROOT'] = fake_proc(tmp_path, {500: (os.getuid(), 'S')})
    hook('pre', rollout['event']('async-exec', 'Bash', 'sleep 500'), **rollout['env'])
    rollout['call']('async-exec', 'exec_command', {'cmd': 'sleep 500'}, custom=False)
    rollout['output']('async-exec', 'Chunk ID: abc\nProcess running with session ID 123\nOutput:\n', custom=False)
    if completion != 'none':
        rollout['call']('poll', 'write_stdin', {'session_id': 456 if completion == 'wrong-session' else 123}, custom=False)
        text = ('Chunk ID: xyz\nProcess running with session ID 123\nOutput:\n' if completion == 'running-again'
                else 'write_stdin failed: Unknown process id 123' if completion == 'unknown-process'
                else 'write_stdin failed\nOutput:\nProcess exited with code 0\n' if completion == 'stdout-spoof'
                else 'Chunk ID: xyz\nProcess exited with code 0\nOutput:\n')
        rollout['output']('poll', text, custom=False)
    assert reconcile(hook, **rollout['env'])[1]['reconciled_tools'] == 0
    assert decision(hook('pre', pre(session=rollout['session']), **rollout['env'])) == 'deny'


def fake_proc(tmp_path, processes):
    """A /proc stand-in: {pid: (uid, state)}; the hook scans it for live Agent processes."""
    root = tmp_path / 'fake-proc'
    root.mkdir(exist_ok=True)
    for pid, (uid, process_state) in processes.items():
        (root / str(pid)).mkdir()
        (root / str(pid) / 'status').write_text(
            f'Name:\tpython3\nState:\t{process_state} (x)\nUid:\t{uid}\t{uid}\t{uid}\t{uid}\n')
    return str(root)


def vanished_exec(rollout, handle=123, poll_handle=123):
    rollout['call']('async-exec', 'exec_command', {'cmd': 'python3 bench.py'}, custom=False)
    rollout['output']('async-exec', f'Chunk ID: abc\nProcess running with session ID {handle}\nOutput:\n', custom=False)
    rollout['call']('poll', 'write_stdin', {'session_id': poll_handle}, custom=False)
    rollout['output']('poll', f'write_stdin failed: Unknown process id {poll_handle}', custom=False)


@pytest.mark.parametrize('processes', [{}, {41: (os.getuid(), 'Z')}, {42: (os.getuid() + 1, 'S')}])
def test_unknown_handle_with_no_live_agent_process_releases_submit(hook, rollout, tmp_path, processes):
    # The Agent killed its own background job; Codex then reports the handle as
    # unknown. With no live process of the Agent's uid left (zombies and other
    # users do not count), the submit is admitted and only that exec is retired.
    hook('pre', rollout['event']('async-exec', 'Bash', 'python3 bench.py'), **rollout['env'])
    vanished_exec(rollout)
    proc = fake_proc(tmp_path, processes)
    assert reconcile(hook, **rollout['env'], SFORGE_SUBMIT_WAIT_PROC_ROOT=proc)[1]['reconciled_tools'] == 0
    result = hook('pre', pre(session=rollout['session']), **rollout['env'], SFORGE_SUBMIT_WAIT_PROC_ROOT=proc)
    assert decision(result) == 'allow'
    assert state(hook)['active_tools'] == {}
    assert state(hook)['active']['tool_use_id'] == 'submit-1'


@pytest.mark.parametrize('case', ['live-agent-process', 'other-handle', 'no-poll', 'unreadable-status'])
def test_unknown_handle_without_exit_proof_keeps_submit_denied(hook, rollout, tmp_path, case):
    hook('pre', rollout['event']('async-exec', 'Bash', 'python3 bench.py'), **rollout['env'])
    if case == 'no-poll':
        rollout['call']('async-exec', 'exec_command', {'cmd': 'python3 bench.py'}, custom=False)
        rollout['output']('async-exec', 'Chunk ID: abc\nProcess running with session ID 123\nOutput:\n', custom=False)
    else:
        vanished_exec(rollout, poll_handle=456 if case == 'other-handle' else 123)
    processes = {77: (os.getuid(), 'S')} if case == 'live-agent-process' else {}
    proc = fake_proc(tmp_path, processes)
    if case == 'unreadable-status':
        (Path(proc) / '78').mkdir()
        (Path(proc) / '78' / 'status').write_text('garbage\n')
    result = hook('pre', pre(session=rollout['session']), **rollout['env'], SFORGE_SUBMIT_WAIT_PROC_ROOT=proc)
    assert decision(result) == 'deny'
    assert len(state(hook)['active_tools']) == 1 and state(hook)['active'] is None


def test_unknown_handle_proof_never_retires_unrelated_pending_tool(hook, rollout, tmp_path):
    hook('pre', rollout['event']('async-exec', 'Bash', 'python3 bench.py'), **rollout['env'])
    hook('pre', rollout['event']('pending-patch', 'apply_patch'), **rollout['env'])
    vanished_exec(rollout)
    proc = fake_proc(tmp_path, {})
    result = hook('pre', pre(session=rollout['session']), **rollout['env'], SFORGE_SUBMIT_WAIT_PROC_ROOT=proc)
    assert decision(result) == 'deny'
    assert [tool['tool_use_id'] for tool in state(hook)['active_tools'].values()] == ['pending-patch']


def age_pending_tools(hook, seconds):
    value = state(hook)
    started = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=seconds))
    for tool in value['active_tools'].values():
        tool['started_at'] = started.strftime('%Y-%m-%dT%H:%M:%S.') + f'{started.microsecond // 1000:03d}Z'
    (hook.state / 'state.json').write_text(json.dumps(value))


def test_stale_backgrounded_tool_with_no_live_agent_process_releases_submit(hook, tmp_path):
    # Claude moved a 600 s Bash call to the background; its only post says
    # "running" and the job later ended without another callback.
    assert hook('pre', pre('ncu python prof.py', 'profile')) == {}
    assert hook('post', post('profile', response={'backgroundTaskId': 'bif2io7ti', 'status': 'running'})) == {}
    hook('pre', pre('pytest', 'patch'))
    age_pending_tools(hook, 8 * 3600)
    result = hook('pre', pre(), SFORGE_SUBMIT_WAIT_PROC_ROOT=fake_proc(tmp_path, {41: (os.getuid(), 'Z')}))
    assert decision(result) == 'allow'
    assert state(hook)['active_tools'] == {}
    assert state(hook)['active']['tool_use_id'] == 'submit-1'


@pytest.mark.parametrize('case', ['live-agent-process', 'recent', 'unreadable-status'])
def test_pending_tool_without_stale_and_process_proof_keeps_submit_denied(hook, tmp_path, case):
    assert hook('pre', pre('ncu python prof.py', 'profile')) == {}
    age_pending_tools(hook, 60 if case == 'recent' else 8 * 3600)
    proc = fake_proc(tmp_path, {77: (os.getuid(), 'S')} if case == 'live-agent-process' else {})
    if case == 'unreadable-status':
        (Path(proc) / '78').mkdir()
        (Path(proc) / '78' / 'status').write_text('garbage\n')
    result = hook('pre', pre(), SFORGE_SUBMIT_WAIT_PROC_ROOT=proc)
    assert decision(result) == 'deny'
    assert len(state(hook)['active_tools']) == 1 and state(hook)['active'] is None


def test_stale_rule_never_releases_an_old_active_submission(hook, tmp_path):
    hook('pre', pre())
    value = state(hook)
    value['active']['started_at'] = '2000-01-01T00:00:00.000Z'
    (hook.state / 'state.json').write_text(json.dumps(value))
    result = hook('pre', pre(identifier='later'), SFORGE_SUBMIT_WAIT_PROC_ROOT=fake_proc(tmp_path, {}))
    assert decision(result) == 'deny'
    assert state(hook)['active'] == value['active']


def test_async_original_exact_write_stdin_completion_is_terminal_proof(hook, rollout):
    hook('pre', rollout['event']('async-exec', 'Bash', 'sleep 500'), **rollout['env'])
    rollout['call']('async-exec', 'exec_command', {'cmd': 'sleep 500'}, custom=False)
    rollout['output']('async-exec', 'Chunk ID: abc\nProcess running with session ID 123\nOutput:\n', custom=False)
    rollout['call']('poll', 'write_stdin', {'session_id': 123}, custom=False)
    rollout['output']('poll', 'Chunk ID: xyz\nProcess exited with code 1\nOutput:\nfailure', custom=False)
    assert reconcile(hook, **rollout['env'])[1]['reconciled_tools'] == 1
    assert decision(hook('pre', pre(session=rollout['session']), **rollout['env'])) == 'allow'


def test_unknown_process_retires_only_its_own_failed_tool_not_prior_exec(hook, rollout):
    hook('pre', rollout['event']('unknown-poll', 'write_stdin'), **rollout['env'])
    rollout['call']('unknown-poll', 'write_stdin', {'session_id': 123}, custom=False)
    rollout['output']('unknown-poll', 'write_stdin failed: Unknown process id 123', custom=False)
    assert reconcile(hook, **rollout['env'])[1]['reconciled_tools'] == 1


def test_reconciliation_cannot_release_actual_submission_even_with_matching_output(hook, rollout):
    event = rollout['event']('real-submit', 'Bash', 'sforge-submit')
    assert decision(hook('pre', event, **rollout['env'])) == 'allow'
    rollout['call']('real-submit', 'exec_command', {'cmd': 'sforge-submit'}, custom=False)
    rollout['output']('real-submit', 'Chunk ID: xyz\nProcess exited with code 0\nOutput:\nScore: 180', custom=False)
    before = state(hook)
    code, summary = reconcile(hook, **rollout['env'])
    assert code == 0 and summary['active_submission'] is True
    assert summary['reconciled_tools'] == 0
    assert state(hook) == before
    assert decision(hook('pre', rollout['event']('edit', 'apply_patch'), **rollout['env'])) == 'deny'


def test_invalid_state_reconcile_is_nonzero_and_preserves_original(hook):
    hook.state.mkdir()
    (hook.state / 'state.json').write_text('corrupted')
    code, summary = reconcile(hook)
    assert code == 1 and 'error' in summary
    assert (hook.state / 'state.json').read_text() == 'corrupted'


def test_config_covers_all_pre_and_post_tools_as_valid_inline_toml():
    value = tomllib.loads(submit_wait_hooks_config_fragment())
    assert SUBMIT_WAIT_HOOK_PATH == '/opt/sforge/submit-wait-hook.py'
    assert SUBMIT_WAIT_STATE_DIR == '/run/sforge-submit-wait'
    for event, phase in (('PreToolUse', 'pre'), ('PostToolUse', 'post')):
        registration = value['hooks'][event]
        assert registration[0]['matcher'] == '.*'
        assert registration[0]['hooks'] == [{'type': 'command', 'command':
            'python3 /opt/sforge/submit-wait-hook.py ' + phase, 'timeout': 21720 if phase == 'pre' else 15}]
    configured = tomllib.loads(submit_wait_hooks_config_fragment(timeout_seconds=43320))
    assert configured['hooks']['PreToolUse'][0]['hooks'][0]['timeout'] == 43320


def test_submit_is_blocking_and_unrelated_completion_cannot_release_it(hook):
    event = pre('sforge-submit --version v001')
    reply = hook('pre', event)
    assert decision(reply) == 'allow'
    updated = reply['hookSpecificOutput']['updatedInput']
    assert list(updated) == ['command']
    replay = subprocess.run(shlex.split(updated['command']), capture_output=True, text=True, env=hook.env)
    assert replay.returncode == 0
    assert replay.stdout == 'Score: 180.0\n'
    active = state(hook)['active']
    assert active['session_id'] == 'session-1'
    assert active['tool_use_id'] == 'submit-1'
    assert active['started_at'].endswith('Z')
    assert active['timeout_ms'] == SUBMIT_WAIT_DEFAULT_TIMEOUT_MS
    assert active['result_ready'] is True
    for event in (pre('touch candidate.py', 'edit-1'), pre('pytest', 'tests-1'),
                  pre('sforge-save-version', 'save-1'), pre('sforge-submit', 'submit-2'),
                  pre('sforge-submit --list', 'lookup-1'), pre('sforge-submit'),
                  pre('anything', 'editor-1', tool='apply_patch')):
        reply = hook('pre', event)
        assert decision(reply) == 'deny'
        assert 'Wait for that original submission' in reply['hookSpecificOutput']['permissionDecisionReason']
    assert hook('post', post('unrelated')) == {}
    assert hook('post', post(session='other-session')) == {}
    assert state(hook)['active'] == active
    assert hook('post', post()) == {}
    assert state(hook)['active'] is None
    assert state(hook)['last_completion']['outcome'] == 'completed'
    assert hook('pre', pre('pytest', 'tests-after')) == {}


@pytest.mark.parametrize('response', [
    {'exit_code': 1, 'output': 'Submit failed'},
    {'timed_out': True, 'output': 'Command timed out'},
    {'status': 'cancelled'},
    'Process exited with code 1',
])
def test_failed_or_timeout_shell_is_explicitly_released(hook, response):
    assert decision(hook('pre', pre())) == 'allow'
    assert hook('post', post(response=response)) == {}
    assert state(hook)['active'] is None
    assert state(hook)['last_completion']['outcome'] == 'failed'
    assert decision(hook('pre', pre(identifier='retry'))) == 'allow'


@pytest.mark.parametrize('response', [
    {'session_id': 123, 'output': 'Waiting'},
    {'status': 'running'},
    'Process running with session ID 123',
])
def test_yielded_process_never_counts_as_final_result(hook, response):
    hook('pre', pre())
    assert hook('post', post(response=response))['decision'] == 'block'
    assert state(hook)['active'] is not None
    assert decision(hook('pre', pre('touch later', 'other'))) == 'deny'
    # A terminal callback for the same original call, if supplied by a runtime,
    # is the only event that can release this gate.
    assert hook('post', post()) == {}
    assert state(hook)['active'] is None


@pytest.mark.parametrize('command', [
    'sforge-submit &', 'nohup sforge-submit', 'sforge-submit | cat',
    'cd /workspace && sforge-submit', 'sforge-submit; touch candidate.py',
    'sforge-submit\ntouch candidate.py', 'sforge-submit > /tmp/result',
    'sforge-submit --version $(cat id)', 'timeout 5 sforge-submit',
    'env A=1 sforge-submit', 'sforge-submit --help --version v001',
])
def test_submission_cannot_background_or_bundle_an_operation(hook, command):
    assert decision(hook('pre', pre(command))) == 'deny'
    assert not (hook.state / 'state.json').exists()


@pytest.mark.parametrize('command', ['sforge-submit --help', 'sforge-submit -h',
                                     'sforge-submit --list', 'sforge-submit -l',
                                     'pytest -q', 'sforge-save-version --json'])
def test_non_submission_commands_are_tracked_until_their_final_post(hook, command):
    assert hook('pre', pre(command)) == {}
    assert state(hook)['active'] is None
    assert len(state(hook)['active_tools']) == 1
    assert hook('post', post()) == {}
    assert state(hook)['active_tools'] == {}


def test_custom_runtime_timeout_is_internal_and_tool_only_replays_the_result(hook):
    event = pre()
    event['tool_input'].update(workdir='/workspace/project', login=False)
    result = hook('pre', event, SFORGE_SUBMIT_TOOL_TIMEOUT_MS='43260000')
    assert list(result['hookSpecificOutput']['updatedInput']) == ['command']
    assert 'replay' in result['hookSpecificOutput']['updatedInput']['command']
    assert state(hook)['active']['timeout_ms'] == 43260000


@pytest.mark.parametrize('value', ['', '0', '-10', 'one-minute', '172800001'])
def test_invalid_runtime_timeout_is_a_valid_denial(hook, value):
    assert decision(hook('pre', pre(), SFORGE_SUBMIT_TOOL_TIMEOUT_MS=value)) == 'deny'
    assert not (hook.state / 'state.json').exists()


@pytest.mark.parametrize('payload', ['not-json', '{}', '{"schema_version":1,"active":{}}',
                                     '{"schema_version":1,"active":null,"last_completion":"bad"}'])
def test_corrupted_state_denies_instead_of_returning_a_hook_error(hook, payload):
    hook.state.mkdir()
    (hook.state / 'state.json').write_text(payload)
    assert decision(hook('pre', pre('pytest'))) == 'deny'
    assert hook('post', post())['decision'] == 'block'
    assert (hook.state / 'state.json').read_text() == payload


def test_invalid_input_returns_valid_denial_with_successful_hook_exit(hook):
    for payload in ('not-json', '[]', 'null'):
        result = subprocess.run([sys.executable, '-I', str(hook.script), 'pre'], input=payload,
                                text=True, capture_output=True, env=hook.env, timeout=10)
        assert result.returncode == 0
        assert decision(json.loads(result.stdout)) == 'deny'


def test_missing_post_response_keeps_gate_until_confirmed_completion(hook):
    hook('pre', pre())
    event = post()
    del event['tool_response']
    assert hook('post', event)['decision'] == 'block'
    assert state(hook)['active'] is not None


def test_eight_real_hook_processes_contend_for_one_persistent_gate(hook):
    hook.state.mkdir()
    processes = []
    # Hold the same lock from this independent process. This verifies that the
    # generated hooks actually lock, rather than relying on sequential timing.
    with (hook.state / 'lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            for number in range(8):
                process = subprocess.Popen([sys.executable, '-I', str(hook.script), 'pre'],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    text=True, env=hook.env)
                process.stdin.write(json.dumps(pre(identifier='racer-' + str(number))))
                process.stdin.close()
                process.stdin = None
                processes.append(process)
            time.sleep(0.15)
            assert all(process.poll() is None for process in processes)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
        results = []
        for process in processes:
            stdout, stderr = process.communicate(timeout=10)
            assert process.returncode == 0 and not stderr
            results.append(json.loads(stdout))
    assert [decision(value) for value in results].count('allow') == 1
    assert [decision(value) for value in results].count('deny') == 7
    winner = state(hook)['active']['tool_use_id']
    assert hook('post', post(identifier=winner)) == {}
    assert state(hook)['active'] is None
    assert decision(hook('pre', pre(identifier='next-round'))) == 'allow'


def test_old_pending_state_is_never_unlocked_by_elapsed_time(hook):
    hook('pre', pre())
    value = state(hook)
    value['active']['started_at'] = '2000-01-01T00:00:00.000Z'
    (hook.state / 'state.json').write_text(json.dumps(value))
    assert decision(hook('pre', pre('pytest', 'later'))) == 'deny'
    assert state(hook)['active'] == value['active']


def test_pre_hook_does_not_return_until_submit_finishes_and_other_tools_are_denied(hook, tmp_path):
    marker = tmp_path / 'submit-started'
    process = subprocess.Popen([sys.executable, '-I', str(hook.script), 'pre'],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        env={**hook.env, 'FAKE_SUBMIT_MARKER': str(marker), 'FAKE_SUBMIT_SLEEP': '0.8'})
    started = time.monotonic()
    process.stdin.write(json.dumps(pre()))
    process.stdin.close()
    process.stdin = None
    while not marker.exists() and time.monotonic() - started < 3:
        time.sleep(0.01)
    assert marker.exists()
    assert process.poll() is None
    assert state(hook)['active']['result_ready'] is False
    # This must promptly deny, rather than block on a lock held for the whole
    # evaluation (which could let a runtime hook timeout fail open).
    deny_started = time.monotonic()
    assert decision(hook('pre', pre('touch candidate.py', 'edit'))) == 'deny'
    assert time.monotonic() - deny_started < 0.5
    assert process.poll() is None
    stdout, stderr = process.communicate(timeout=3)
    assert process.returncode == 0 and not stderr
    assert time.monotonic() - started >= 0.8
    reply = json.loads(stdout)
    assert decision(reply) == 'allow'
    assert state(hook)['active']['result_ready'] is True
    result = subprocess.run(shlex.split(reply['hookSpecificOutput']['updatedInput']['command']),
                            env=hook.env, text=True, capture_output=True, timeout=3)
    assert result.returncode == 0 and result.stdout == 'Score: 180.0\n'
    assert hook('post', post(response={'exit_code': result.returncode, 'output': result.stdout})) == {}
    assert state(hook)['active'] is None


def test_internal_timeout_returns_explicit_failure_then_post_releases(hook):
    started = time.monotonic()
    reply = hook('pre', pre(), SFORGE_SUBMIT_TOOL_TIMEOUT_MS='100', FAKE_SUBMIT_SLEEP='5')
    assert decision(reply) == 'allow'
    assert time.monotonic() - started < 2
    assert state(hook)['active']['result_ready'] is True
    result = subprocess.run(shlex.split(reply['hookSpecificOutput']['updatedInput']['command']),
                            env=hook.env, text=True, capture_output=True, timeout=3)
    assert result.returncode == 1
    assert 'submission wait timed out' in result.stdout
    assert hook('post', post(response={'exit_code': result.returncode, 'output': result.stdout})) == {}
    assert state(hook)['last_completion']['outcome'] == 'failed'
    assert hook('pre', pre('pytest', 'after-timeout')) == {}


def test_native_submit_nonzero_code_survives_replay(hook):
    reply = hook('pre', pre(), FAKE_SUBMIT_EXIT='3')
    result = subprocess.run(shlex.split(reply['hookSpecificOutput']['updatedInput']['command']),
                            env=hook.env, text=True, capture_output=True, timeout=3)
    assert result.returncode == 3
    assert result.stdout == 'Score: 180.0\n'
    assert hook('post', post(response={'exit_code': result.returncode, 'output': result.stdout})) == {}
    assert state(hook)['last_completion']['outcome'] == 'failed'


def test_remaining_agent_budget_bounds_the_internal_wait(hook):
    reply = hook('pre', pre(), SFORGE_REMAINING_SECONDS='0.1', FAKE_SUBMIT_SLEEP='5')
    assert decision(reply) == 'allow'
    assert state(hook)['active']['timeout_ms'] == 100
    result = subprocess.run(shlex.split(reply['hookSpecificOutput']['updatedInput']['command']),
                            env=hook.env, text=True, capture_output=True, timeout=3)
    assert result.returncode == 1


def test_missing_native_command_is_a_replayable_explicit_failure(hook):
    command = str(hook.binary_directory / 'missing' / 'sforge-submit')
    reply = hook('pre', pre(command))
    result = subprocess.run(shlex.split(reply['hookSpecificOutput']['updatedInput']['command']),
                            env=hook.env, text=True, capture_output=True, timeout=3)
    assert result.returncode == 1
    assert 'could not complete' in result.stdout


@pytest.mark.parametrize('arguments', ['--details', '-d', '--details --version v001',
                                      '--version v001 -d'])
def test_details_flag_is_a_real_submission_not_a_read_only_lookup(hook, arguments, tmp_path):
    marker = tmp_path / 'submit-started'
    result = hook('pre', pre('sforge-submit ' + arguments), FAKE_SUBMIT_MARKER=str(marker))
    assert decision(result) == 'allow'
    assert marker.read_text() == 'started'
    assert state(hook)['active']['result_ready'] is True


def test_absolute_agent_deadline_bounds_wait_at_time_of_submission(hook):
    reply = hook('pre', pre(), SFORGE_AGENT_DEADLINE_AT=str(time.time() + 0.15),
                 FAKE_SUBMIT_SLEEP='5')
    assert decision(reply) == 'allow'
    assert 0 < state(hook)['active']['timeout_ms'] <= 150
    result = subprocess.run(shlex.split(reply['hookSpecificOutput']['updatedInput']['command']),
                            env=hook.env, text=True, capture_output=True, timeout=3)
    assert result.returncode == 1


def test_expired_absolute_agent_deadline_denies_without_launching_submit(hook, tmp_path):
    marker = tmp_path / 'submit-started'
    result = hook('pre', pre(), SFORGE_AGENT_DEADLINE_AT=str(time.time() - 1),
                  FAKE_SUBMIT_MARKER=str(marker))
    assert decision(result) == 'deny'
    assert not marker.exists()
    assert not (hook.state / 'state.json').exists()


def test_cached_native_exit_code_handles_stdout_only_post_hook(hook):
    hook('pre', pre(), FAKE_SUBMIT_EXIT='3')
    assert hook('post', post(response='Score: 180.0\n')) == {}
    assert state(hook)['active'] is None
    assert state(hook)['last_completion']['outcome'] == 'failed'


@pytest.mark.parametrize('change', ['missing', 'tool', 'session', 'result-id', 'exit-code'])
def test_post_cannot_release_without_an_owned_valid_result_cache(hook, change):
    hook('pre', pre())
    active = state(hook)['active']
    path = hook.state / ('result-' + active['result_id'] + '.json')
    if change == 'missing':
        path.unlink()
    else:
        value = json.loads(path.read_text())
        field = {'tool': 'tool_use_id', 'session': 'session_id', 'result-id': 'result_id',
                 'exit-code': 'exit_code'}[change]
        value[field] = 'invalid'
        path.write_text(json.dumps(value))
    assert hook('post', post())['decision'] == 'block'
    assert state(hook)['active'] == active
    assert decision(hook('pre', pre('pytest', 'unrelated'))) == 'deny'


def test_monotonic_deadline_limits_real_wait_without_wall_clock_dependency(hook):
    reply = hook('pre', pre(), SFORGE_AGENT_DEADLINE_MONOTONIC=str(time.monotonic() + 0.15),
                 FAKE_SUBMIT_SLEEP='5')
    assert decision(reply) == 'allow'
    assert 0 < state(hook)['active']['timeout_ms'] <= 150
    result = subprocess.run(shlex.split(reply['hookSpecificOutput']['updatedInput']['command']),
                            env=hook.env, text=True, capture_output=True, timeout=3)
    assert result.returncode == 1


@pytest.mark.parametrize('deadline', ['nan', 'inf', '-inf', '0', 'not-a-time'])
def test_invalid_or_expired_monotonic_deadline_denies_without_a_gate(hook, deadline):
    assert decision(hook('pre', pre(), SFORGE_AGENT_DEADLINE_MONOTONIC=deadline)) == 'deny'
    assert not (hook.state / 'state.json').exists()


def test_existing_tool_must_finish_before_submit_is_allowed(hook):
    assert hook('pre', pre('pytest', 'tests')) == {}
    pending = state(hook)['active_tools']
    assert decision(hook('pre', pre())) == 'deny'
    assert state(hook)['active'] is None
    # A different tool or a different session cannot acknowledge this tool.
    assert hook('post', post('other')) == {}
    assert hook('post', post('tests', session='different')) == {}
    assert state(hook)['active_tools'] == pending
    assert hook('post', post('tests', response={'session_id': 91, 'output': 'Waiting'})) == {}
    assert state(hook)['active_tools'] == pending
    assert hook('post', post('tests', response={'exit_code': 1, 'output': 'Test failed'})) == {}
    assert state(hook)['active_tools'] == {}
    assert decision(hook('pre', pre())) == 'allow'


@pytest.mark.parametrize('name', ['Agent', 'spawn_agent', 'spawn_agents', 'followup_agent',
                                  'followup_task', 'resume_agent', 'send_message',
                                  'functions.spawn_agent', 'collaboration.send_message'])
def test_agent_delegation_tools_are_explicitly_disabled(hook, name):
    reply = hook('pre', pre('ignored', tool=name))
    assert decision(reply) == 'deny'
    assert 'one Agent' in reply['hookSpecificOutput']['permissionDecisionReason']
    assert not (hook.state / 'state.json').exists()


def test_regular_tool_missing_identifiers_is_denied(hook):
    event = pre('pytest')
    del event['tool_use_id']
    assert decision(hook('pre', event)) == 'deny'
    assert not (hook.state / 'state.json').exists()


def test_race_between_a_submit_and_a_regular_tool_cannot_allow_both(hook):
    hook.state.mkdir()
    processes = []
    with (hook.state / 'lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for event in (pre(identifier='submit'), pre('pytest', 'tests')):
            process = subprocess.Popen([sys.executable, '-I', str(hook.script), 'pre'],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, env=hook.env)
            process.stdin.write(json.dumps(event))
            process.stdin.close()
            process.stdin = None
            processes.append(process)
        time.sleep(0.1)
        assert all(process.poll() is None for process in processes)
        fcntl.flock(lock, fcntl.LOCK_UN)
        results = []
        for process in processes:
            stdout, stderr = process.communicate(timeout=3)
            assert process.returncode == 0 and not stderr
            results.append(json.loads(stdout))
    submit, regular = results
    assert (decision(submit) == 'allow' and decision(regular) == 'deny') or (
        decision(submit) == 'deny' and regular == {})
    assert not (state(hook)['active'] and state(hook)['active_tools'])


def test_tracking_map_is_bounded_and_contains_no_shell_commands(hook):
    for number in range(32):
        assert hook('pre', pre('echo private-input-' + str(number), 'tool-' + str(number))) == {}
    assert len(state(hook)['active_tools']) == 32
    assert 'private-input' not in (hook.state / 'state.json').read_text()
    assert decision(hook('pre', pre('echo overflow', 'overflow'))) == 'deny'
    assert len(state(hook)['active_tools']) == 32


@pytest.fixture
def boundary_module():
    namespace = {'__name__': 'installed_submit_wait_boundary_test'}
    exec(compile(generate_submit_wait_hook_script(), 'standalone-hook.py', 'exec'), namespace)
    return namespace


@pytest.fixture
def authorized_boundary(boundary_module, monkeypatch):
    uid = os.getuid()
    if uid == 0:
        pytest.skip('Identity-preserving unit fixture requires an actual non-root file owner')
    monkeypatch.setattr(boundary_module['os'], 'geteuid', lambda: 0)
    monkeypatch.setattr(boundary_module['pwd'], 'getpwnam',
                        lambda name: SimpleNamespace(pw_uid=uid, pw_gid=os.getgid()))
    boundary_module['agent_writer_pids'] = lambda value: []
    return boundary_module, uid


@pytest.mark.parametrize('tool', ['Bash', 'apply_patch', 'write_stdin'])
def test_root_boundary_cancels_ordinary_tool_without_claiming_success(hook, authorized_boundary, tool):
    module, uid = authorized_boundary
    hook('pre', pre('opaque ordinary operation', identifier='unfinished', tool=tool))
    before = state(hook)
    result = module['retire_ordinary_at_boundary'](hook.state, uid)
    assert result['outcome'] == 'cancelled_at_writer_boundary'
    assert result['retired_tool_count'] == 1 and result['agent_uid'] == uid
    assert result['retired_tools'] == [{
        **{key: next(iter(before['active_tools'].values()))[key]
           for key in ('session_id', 'tool_use_id', 'tool_name', 'started_at')},
        'outcome': 'cancelled_at_writer_boundary'}]
    assert state(hook) == {**before, 'active_tools': {}}
    assert 'opaque ordinary operation' not in json.dumps(result)
    assert 'transcript_path' not in json.dumps(result)


def test_boundary_preserves_actual_state_identity_and_new_lock_access(hook, authorized_boundary, monkeypatch):
    module, uid = authorized_boundary
    hook('pre', pre('opaque edit', identifier='unfinished', tool='apply_patch'))
    target = hook.state / 'state.json'
    before = target.stat()
    (hook.state / 'lock').unlink()
    ownership = []
    original_fchown = os.fchown

    def record_owner(descriptor, owner_uid, owner_gid):
        ownership.append((owner_uid, owner_gid))
        return original_fchown(descriptor, owner_uid, owner_gid)

    monkeypatch.setattr(module['os'], 'fchown', record_owner)
    module['retire_ordinary_at_boundary'](hook.state, uid)
    for path in (target, hook.state / 'lock'):
        actual = path.stat()
        assert (actual.st_uid, actual.st_gid) == (before.st_uid, before.st_gid)
        assert actual.st_mode & 0o777 == 0o600
    assert ownership == [(before.st_uid, before.st_gid)] * 2
    assert hook.state.stat().st_uid == uid
    assert hook.state.stat().st_mode & 0o777 == 0o700
    assert decision(hook('pre', pre(identifier='next-submit'))) == 'allow'


@pytest.mark.parametrize('missing', ['directory', 'state'])
def test_boundary_does_not_create_root_private_state(hook, authorized_boundary, missing):
    module, uid = authorized_boundary
    if missing == 'state':
        hook.state.mkdir(mode=0o700)
    with pytest.raises(FileNotFoundError):
        module['retire_ordinary_at_boundary'](hook.state, uid)
    assert not (hook.state / 'state.json').exists()
    assert not (hook.state / 'lock').exists()
    assert hook.state.exists() == (missing == 'state')


@pytest.mark.parametrize('damage', ['public-directory', 'public-state', 'state-symlink', 'directory-symlink'])
def test_boundary_requires_private_regular_agent_state(hook, authorized_boundary, tmp_path, damage):
    module, uid = authorized_boundary
    hook('pre', pre('opaque edit', identifier='unfinished', tool='apply_patch'))
    target = hook.state / 'state.json'
    before = target.read_bytes()
    directory = hook.state
    if damage == 'public-directory':
        directory.chmod(0o755)
    elif damage == 'public-state':
        target.chmod(0o644)
    elif damage == 'state-symlink':
        original = directory / 'original-state'
        target.rename(original)
        target.symlink_to(original)
    else:
        directory = tmp_path / 'alias'
        directory.symlink_to(hook.state, target_is_directory=True)
    with pytest.raises(ValueError, match='Agent-owned private state'):
        module['retire_ordinary_at_boundary'](directory, uid)
    assert target.read_bytes() == before


@pytest.mark.parametrize('ready', [False, True])
def test_boundary_never_retires_active_submit_even_with_ready_cache(hook, authorized_boundary, ready):
    module, uid = authorized_boundary
    hook('pre', pre())
    value = state(hook)
    result_path = hook.state / ('result-' + value['active']['result_id'] + '.json')
    value['active']['result_ready'] = ready
    state_path = hook.state / 'state.json'
    state_path.write_text(json.dumps(value))
    before_state, before_result = state_path.read_bytes(), result_path.read_bytes()
    module['agent_writer_pids'] = lambda value: pytest.fail('Submit refusal must precede any cancellation proof')
    with pytest.raises(RuntimeError, match='active submission'):
        module['retire_ordinary_at_boundary'](hook.state, uid)
    assert state_path.read_bytes() == before_state and result_path.read_bytes() == before_result


@pytest.mark.parametrize('failure', ['writer-alive', 'unreadable', 'unknown-io'])
def test_boundary_refuses_incomplete_uid_process_proof_without_state_change(hook, authorized_boundary, failure):
    module, uid = authorized_boundary
    hook('pre', pre('sleep 900', identifier='unfinished'))
    before = (hook.state / 'state.json').read_bytes()
    if failure == 'writer-alive':
        module['agent_writer_pids'] = lambda value: [123]
        error = RuntimeError
    else:
        error = PermissionError if failure == 'unreadable' else OSError
        def reject(value):
            raise error('Process proof unavailable')
        module['agent_writer_pids'] = reject
    with pytest.raises(error):
        module['retire_ordinary_at_boundary'](hook.state, uid)
    assert (hook.state / 'state.json').read_bytes() == before


def test_boundary_requires_actual_root_before_lookup_or_process_scan(hook, authorized_boundary, monkeypatch):
    module, uid = authorized_boundary
    hook('pre', pre('sleep 900', identifier='unfinished'))
    before = (hook.state / 'state.json').read_bytes()
    monkeypatch.setattr(module['os'], 'geteuid', lambda: uid)
    monkeypatch.setattr(module['pwd'], 'getpwnam', lambda name: pytest.fail('Non-root must refuse immediately'))
    with pytest.raises(PermissionError, match='Only root'):
        module['retire_ordinary_at_boundary'](hook.state, uid)
    assert (hook.state / 'state.json').read_bytes() == before


@pytest.mark.parametrize('uid', [0, -1, 1, 2**32, '42420', True])
def test_boundary_requires_exact_non_root_agent_account(hook, authorized_boundary, uid):
    module, _ = authorized_boundary
    with pytest.raises(ValueError, match='non-root agent account'):
        module['retire_ordinary_at_boundary'](hook.state, uid)
    assert not hook.state.exists()


def test_boundary_retirement_preserves_submit_completion_and_result_files_then_allows_submit(hook, authorized_boundary):
    module, uid = authorized_boundary
    hook('pre', pre())
    hook('post', post())
    completion = state(hook)['last_completion']
    orphan = hook.state / ('result-' + 'a' * 32 + '.json')
    orphan.write_bytes(b'preserved submit result')
    hook('pre', pre('opaque patch', identifier='unfinished', tool='apply_patch'))
    module['retire_ordinary_at_boundary'](hook.state, uid)
    assert state(hook)['last_completion'] == completion
    assert orphan.read_bytes() == b'preserved submit result'
    assert decision(hook('pre', pre(identifier='next-submit'))) == 'allow'


def test_boundary_cli_refuses_non_root_and_leaves_state_bytes(hook):
    if os.geteuid() == 0:
        pytest.skip('This CLI test requires an actual non-root pytest process')
    hook('pre', pre('sleep 900', identifier='unfinished'))
    before = (hook.state / 'state.json').read_bytes()
    result = subprocess.run([sys.executable, '-I', str(hook.script), 'retire-ordinary-at-boundary', '42420'],
                            env=hook.env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 1 and not result.stderr
    assert json.loads(result.stdout)['error_type'] == 'PermissionError'
    assert (hook.state / 'state.json').read_bytes() == before


def test_boundary_cli_returns_safe_cancellation_record(hook, authorized_boundary, monkeypatch, capsys):
    module, uid = authorized_boundary
    hook('pre', pre('opaque edit', identifier='unfinished', tool='apply_patch'))
    monkeypatch.setattr(module['sys'], 'argv', [str(hook.script), 'retire-ordinary-at-boundary', str(uid)])
    monkeypatch.setenv('SFORGE_SUBMIT_WAIT_STATE_DIR', str(hook.state))
    assert module['main']() == 0
    record = json.loads(capsys.readouterr().out)
    assert record['outcome'] == 'cancelled_at_writer_boundary'
    assert record['retired_tools'][0]['tool_use_id'] == 'unfinished'
    assert state(hook)['active_tools'] == {}


def test_boundary_invalid_state_remains_byte_identical(hook, authorized_boundary):
    module, uid = authorized_boundary
    hook.state.mkdir()
    target = hook.state / 'state.json'
    target.write_bytes(b'{"schema_version":1,"active":null,"active_tools":"corrupted"}')
    before = target.read_bytes()
    with pytest.raises(ValueError):
        module['retire_ordinary_at_boundary'](hook.state, uid)
    assert target.read_bytes() == before


def test_proc_boundary_checks_all_uid_columns_and_ignores_only_zombies(boundary_module, tmp_path):
    root = tmp_path / 'proc'
    root.mkdir()
    for pid, uids, process_state in [(10, '0 0 0 0', 'S'), (11, '0 42420 0 0', 'S'),
                                     (12, '42420 42420 42420 42420', 'Z')]:
        directory = root / str(pid)
        directory.mkdir()
        (directory / 'status').write_text('State:\t' + process_state + '\nUid:\t' + uids + '\n')
    boundary_module['Path'] = lambda value: root if str(value) == '/proc' else Path(value)
    assert boundary_module['agent_writer_pids'](42420) == [11]


@pytest.mark.parametrize('damage', ['permission', 'unknown-io', 'missing-status', 'malformed'])
def test_proc_boundary_cannot_ignore_unreadable_or_unknown_process(boundary_module, tmp_path, monkeypatch, damage):
    root = tmp_path / 'proc'
    directory = root / '11'
    directory.mkdir(parents=True)
    status_path = directory / 'status'
    if damage != 'missing-status':
        status_path.write_text('State:\tS\nUid:\t0 0 0 0\n' if damage != 'malformed' else 'State:\tS\n')
    boundary_module['Path'] = lambda value: root if str(value) == '/proc' else Path(value)
    if damage in ('permission', 'unknown-io'):
        error = PermissionError if damage == 'permission' else OSError
        original = Path.read_text
        def unreadable(path, *args, **kwargs):
            if path == status_path:
                raise error('Process status is unreadable')
            return original(path, *args, **kwargs)
        monkeypatch.setattr(Path, 'read_text', unreadable)
    else:
        error = RuntimeError
    with pytest.raises(error):
        boundary_module['agent_writer_pids'](42420)


def test_proc_boundary_can_ignore_a_process_that_disappeared(boundary_module, tmp_path):
    missing = tmp_path / '11'
    root = SimpleNamespace(iterdir=lambda: iter([missing]))
    boundary_module['Path'] = lambda value: root if str(value) == '/proc' else Path(value)
    assert boundary_module['agent_writer_pids'](42420) == []


def test_proc_directory_permission_failure_cannot_prove_a_boundary(boundary_module):
    def unreadable():
        raise PermissionError('Cannot list proc')
    root = SimpleNamespace(iterdir=unreadable)
    boundary_module['Path'] = lambda value: root if str(value) == '/proc' else Path(value)
    with pytest.raises(PermissionError):
        boundary_module['agent_writer_pids'](42420)


def test_idle_submit_parse_error_explains_input_without_claiming_active_submit(hook):
    result = hook('pre', pre('sforge-submit && echo malformed'))
    reason = result['hookSpecificOutput']['permissionDecisionReason']
    assert decision(result) == 'deny' and 'Run sforge-submit alone' in reason
    assert 'previous sforge-submit' not in reason and not (hook.state / 'state.json').exists()


def test_active_submit_still_returns_strict_wait_before_parsing_other_input(hook):
    hook('pre', pre())
    before = (hook.state / 'state.json').read_bytes()
    result = hook('pre', pre('sforge-submit && echo malformed', identifier='other'))
    assert 'previous sforge-submit' in result['hookSpecificOutput']['permissionDecisionReason']
    assert (hook.state / 'state.json').read_bytes() == before


def test_bad_hook_json_reports_parse_reason_only_when_no_active_submit(hook):
    def invalid():
        reply = subprocess.run([sys.executable, '-I', str(hook.script), 'pre'], input='not-json',
                               text=True, capture_output=True, env=hook.env, timeout=10)
        assert reply.returncode == 0 and not reply.stderr
        return json.loads(reply.stdout)['hookSpecificOutput']['permissionDecisionReason']
    assert 'invalid JSON' in invalid() and 'previous sforge-submit' not in invalid()
    hook('pre', pre())
    before = (hook.state / 'state.json').read_bytes()
    assert 'previous sforge-submit' in invalid()
    assert (hook.state / 'state.json').read_bytes() == before


def test_untracked_ledger_lets_a_submit_follow_a_tool_that_never_reported_completion(hook):
    # Local two-GPU Claude Code runs: a Bash call Claude moved past 600 s to the
    # background never sends its PostToolUse. Without the ordinary-tool ledger the
    # next submit still runs to its final result instead of being refused forever.
    hook.script.write_text(generate_submit_wait_hook_script(track_ordinary_tools=False))
    for index in range(40):   # more than the 32-entry ledger bound
        assert hook('pre', pre('python3 bench.py', 'bench-%d' % index)) == {}
    assert not (hook.state / 'state.json').exists() or state(hook)['active_tools'] == {}
    reply = hook('pre', pre('sforge-submit --details'))
    assert decision(reply) == 'allow'
    replay = subprocess.run(shlex.split(reply['hookSpecificOutput']['updatedInput']['command']),
                            capture_output=True, text=True, env=hook.env)
    assert replay.stdout == 'Score: 180.0\n'
    # The submit itself is still exclusive until its own result is confirmed.
    for event in (pre('pytest', 'tests-1'), pre('sforge-submit', 'submit-2')):
        reply = hook('pre', event)
        assert decision(reply) == 'deny'
        assert 'Wait for that original submission' in reply['hookSpecificOutput']['permissionDecisionReason']
    assert decision(hook('pre', pre('anything', 'agent-1', tool='Agent'))) == 'deny'
    assert hook('post', post()) == {}
    assert state(hook)['active'] is None
    assert hook('pre', pre('pytest', 'tests-after')) == {}
    assert decision(hook('pre', pre('anything', 'agent-2', tool='Agent'))) == 'deny'


def test_default_hook_keeps_the_ordinary_tool_ledger():
    assert 'TRACK_ORDINARY_TOOLS = True\n' in generate_submit_wait_hook_script()
    assert 'TRACK_ORDINARY_TOOLS = False\n' in generate_submit_wait_hook_script(track_ordinary_tools=False)


# Claude Code 2.1.287's actual result for a Bash call auto-backgrounded at 600 s
# (local SDV2 run, 2026-10-10). No later PostToolUse reports the job's exit.
CLAUDE_AUTO_BACKGROUNDED = ('Command did not complete within its 600s timeout and was moved to the background '
                            '(ID: b2tvk0p4s). Output is being written to: /tmp/claude-1000/x/tasks/b2tvk0p4s.output. '
                            'You will be notified when it completes.')


@pytest.mark.parametrize('track', [True, False])
def test_claude_auto_backgrounded_bash_then_submit(hook, track):
    hook.script.write_text(generate_submit_wait_hook_script(track_ordinary_tools=track))
    assert hook('pre', pre('timeout 1200 python xcheck.py', 'bench')) == {}
    # Through the real Claude adapter, which keeps a runtime background ID pending.
    from sforge.harness.agent.claude_agent import normalize_hook_event
    event = normalize_hook_event(post('bench', response={'stdout': CLAUDE_AUTO_BACKGROUNDED, 'stderr': '',
                                                         'interrupted': False, 'backgroundTaskId': 'b2tvk0p4s'}),
                                 'post')
    assert event['tool_response']['status'] == 'running'
    assert hook('post', event) == {}
    reply = hook('pre', pre('sforge-submit --details'))
    if track:   # the ledger keeps the backgrounded call pending: every later submit is refused
        assert decision(reply) == 'deny'
        assert 'An earlier tool is still running' in reply['hookSpecificOutput']['permissionDecisionReason']
    else:       # local two-GPU Claude runs: the submit runs to its result
        assert decision(reply) == 'allow'
