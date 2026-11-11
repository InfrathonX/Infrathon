"""Codex hook policy that serializes a submit and its final tool response.

The generated hook is standalone Python for installation in a Worker. This is a
tool scheduling policy, not a filesystem or operating-system security boundary.
The PreToolUse hook waits for the native submit itself and then replaces the
shell command with a fast replay of the captured result. The model therefore
cannot receive a yielded shell session before the evaluation has completed.
"""
from __future__ import annotations

import json
import shlex
import textwrap

SUBMIT_WAIT_HOOK_PATH = '/opt/sforge/submit-wait-hook.py'
SUBMIT_WAIT_STATE_DIR = '/run/sforge-submit-wait'
SUBMIT_WAIT_DEFAULT_TIMEOUT_MS = 21_660_000


def submit_wait_hooks_config_fragment(hook_path: str = SUBMIT_WAIT_HOOK_PATH,
                                     *, timeout_seconds: int = 21_720) -> str:
    """Inline Codex TOML; match every tool so edits cannot bypass a pending submit."""
    if not hook_path.startswith('/') or '\n' in hook_path or '\x00' in hook_path:
        raise ValueError('Submit wait hook path must be an absolute single-line path')
    if type(timeout_seconds) is not int or not 120 <= timeout_seconds <= 172_920:
        raise ValueError('Submit wait hook timeout must include an execution safety margin')
    sections = []
    for event, phase in (('PreToolUse', 'pre'), ('PostToolUse', 'post')):
        command = 'python3 ' + shlex.quote(hook_path) + ' ' + phase
        sections.append(
            '[[hooks.' + event + ']]\nmatcher = ".*"\n'
            '[[hooks.' + event + '.hooks]]\ntype = "command"\n'
            'command = ' + json.dumps(command) + '\ntimeout = '
            + str(timeout_seconds if phase == 'pre' else 15) + '\n'
        )
    return '\n'.join(sections)


def generate_submit_wait_hook_script(*, track_ordinary_tools: bool = True) -> str:
    """Return a script needing only the Worker's Python standard library.

    With track_ordinary_tools=False the hook still runs sforge-submit to its
    final result and refuses a second submit meanwhile, but keeps no ledger of
    ordinary tools and so never refuses a submit because an earlier tool looks
    pending. Local two-GPU Claude Code runs use this: Claude moves a Bash call
    past 600 s to the background without a final PostToolUse, which would leave
    the ledger pending forever, and the managed Judge's own idle-GPU check
    (gpu_busy) already rejects an evaluation that would overlap Agent GPU work,
    exactly as for local Codex runs, which have no such ledger.
    """
    source = textwrap.dedent(_HOOK_SOURCE).lstrip()
    if not track_ordinary_tools:
        marker = 'TRACK_ORDINARY_TOOLS = True\n'
        if source.count(marker) != 1:
            raise ValueError('Submit wait hook source has no tool ledger switch')
        source = source.replace(marker, 'TRACK_ORDINARY_TOOLS = False\n')
    return source


_HOOK_SOURCE = r'''
#!/usr/bin/env python3
"""SForge submit tool gate. No SForge package or platform credentials required."""
import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import pwd
import re
import shlex
import signal
import stat
import subprocess
import sys
import tempfile
import time
import uuid

STATE_DIR = '/run/sforge-submit-wait'
DEFAULT_TIMEOUT_MS = 21660000
MAX_STATE_BYTES = 16384
MAX_INPUT_BYTES = 1024 * 1024
MAX_RESULT_BYTES = 4 * 1024 * 1024
MAX_ACTIVE_TOOLS = 32
TRACK_ORDINARY_TOOLS = True
MAX_ROLLOUT_TAIL_BYTES = 8 * 1024 * 1024
MAX_ROLLOUT_LINE_BYTES = 1024 * 1024
MAX_ROLLOUT_FILES = 8
MAX_PROOF_SESSIONS = 4
STALE_TOOL_SECONDS = 900
UUID_PATTERN = r'[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}'
AGENT_TOOLS = {'agent', 'spawn_agent', 'spawn_agents', 'followup_agent', 'followup_task',
               'resume_agent', 'send_message', 'wait_agent', 'interrupt_agent', 'close_agent'}
WAIT_REASON = ('A previous sforge-submit is still waiting for its final result. '
               'Wait for that original submission tool to finish; do not edit files, '
               'run tests, save versions, or start other tools while it is pending.')


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='milliseconds').replace('+00:00', 'Z')


def deny(phase, reason):
    if phase == 'post':
        return {'decision': 'block', 'reason': reason}
    return {'hookSpecificOutput': {'hookEventName': 'PreToolUse',
            'permissionDecision': 'deny', 'permissionDecisionReason': reason}}


def nonempty_string(value, maximum=256):
    return isinstance(value, str) and 0 < len(value) <= maximum and not any(ord(c) < 32 for c in value)


def valid_active(value):
    return (isinstance(value, dict)
            and set(value) == {'tool_use_id', 'session_id', 'tool_name', 'started_at', 'timeout_ms', 'result_id', 'result_ready'}
            and all(nonempty_string(value.get(k)) for k in ('tool_use_id', 'session_id', 'tool_name'))
            and nonempty_string(value.get('started_at'), 40)
            and re.fullmatch(r'\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z', value['started_at']) is not None
            and type(value.get('timeout_ms')) is int and 0 < value['timeout_ms'] <= 172800000
            and isinstance(value.get('result_id'), str)
            and re.fullmatch(r'[a-f0-9]{32}', value['result_id']) is not None
            and type(value.get('result_ready')) is bool)


def tool_key(session_id, tool_use_id):
    return hashlib.sha256((session_id + '\0' + tool_use_id).encode()).hexdigest()


def valid_tools(tools):
    if not isinstance(tools, dict) or len(tools) > MAX_ACTIVE_TOOLS:
        return False
    for key, value in tools.items():
        if (not isinstance(value, dict)
                or set(value) not in ({'tool_use_id', 'session_id', 'tool_name', 'started_at'},
                    {'tool_use_id', 'session_id', 'tool_name', 'started_at', 'transcript_path'})
                or not all(nonempty_string(value.get(k)) for k in ('tool_use_id', 'session_id', 'tool_name'))
                or not nonempty_string(value.get('started_at'), 40)
                or re.fullmatch(r'\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z', value['started_at']) is None
                or key != tool_key(value['session_id'], value['tool_use_id'])):
            return False
        if ('transcript_path' in value and (not nonempty_string(value['transcript_path'], 4096)
                or not Path(value['transcript_path']).is_absolute())):
            return False
    return True


def read_state(path):
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except FileNotFoundError:
        return {'schema_version': 1, 'active': None, 'active_tools': {}, 'last_completion': None}
    with os.fdopen(fd, 'rb') as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_STATE_BYTES:
            raise ValueError('Invalid submit wait state file')
        data = stream.read(MAX_STATE_BYTES + 1)
    value = json.loads(data)
    if (not isinstance(value, dict) or set(value) != {'schema_version', 'active', 'active_tools', 'last_completion'}
            or type(value.get('schema_version')) is not int or value.get('schema_version') != 1
            or not valid_tools(value.get('active_tools'))
            or (value.get('active') is not None and not valid_active(value['active']))):
        raise ValueError('Invalid submit wait state')
    last = value.get('last_completion')
    if last is not None and (not isinstance(last, dict)
            or set(last) != {'tool_use_id', 'session_id', 'finished_at', 'outcome'}
            or not all(nonempty_string(last.get(k)) for k in ('tool_use_id', 'session_id'))
            or not nonempty_string(last.get('finished_at'), 40)
            or last.get('outcome') not in ('completed', 'failed')):
        raise ValueError('Invalid submit completion state')
    return value


def write_state(path, value, *, owner=None):
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()
    limit = MAX_STATE_BYTES if path.name == 'state.json' else MAX_RESULT_BYTES + MAX_STATE_BYTES
    if len(payload) > limit:
        raise ValueError('Submit wait state exceeds size limit')
    fd, temporary = tempfile.mkstemp(prefix='.state-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            if owner is not None:
                # A root recovery must publish an Agent-readable inode, rather
                # than replace its private state with a root-owned tempfile.
                os.fchown(stream.fileno(), *owner)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def submit_command(tool_name, tool_input):
    """Return submit/readonly/other, rejecting submit wrappers and shell composition.

    The policy requires the native CLI as a standalone command. It does not try
    to discover malicious indirect execution through arbitrary programs.
    """
    if tool_name.lower() not in ('bash', 'shell', 'shell_command', 'exec_command', 'local_shell'):
        return 'other', None
    if not isinstance(tool_input, dict):
        raise ValueError('Shell input must be an object')
    command = tool_input.get('command', tool_input.get('cmd'))
    if isinstance(command, list):
        if len(command) == 3 and command[0] in ('bash', '/bin/bash', 'sh', '/bin/sh') and command[1] in ('-c', '-lc'):
            command = command[2]
        else:
            command = shlex.join(command) if all(isinstance(v, str) for v in command) else ''
    if not isinstance(command, str):
        raise ValueError('Shell command must be a string or argument list')
    lexer = shlex.shlex(command, posix=True, punctuation_chars=';&|<>()')
    lexer.whitespace_split = True
    lexer.commenters = '#'
    try:
        words = list(lexer)
    except ValueError:
        if 'sforge-submit' in command:
            raise ValueError('Use sforge-submit as a standalone command with valid arguments') from None
        return 'other', None
    includes_submit = any(Path(word).name == 'sforge-submit' for word in words)
    if not includes_submit:
        return 'other', None
    if (not words or Path(words[0]).name != 'sforge-submit'
            or '\n' in command.strip() or '\r' in command.strip()
            or any(any(c in word for c in ';&|<>()\n\r`$') for word in words)):
        raise ValueError('Run sforge-submit alone: no background &, nohup, pipes, substitutions, or compound shell commands')
    arguments = words[1:]
    if arguments in (['--help'], ['-h'], ['--list'], ['-l']):
        return 'readonly', None
    # Preserve CLI validation of version IDs and future native flags, while
    # refusing a read-only flag combined with a submit argument.
    if any(arg in ('--help', '-h', '--list', '-l') for arg in arguments):
        raise ValueError('Read-only sforge-submit flags must be used as a standalone lookup')
    return 'submit', words


def response_running(response):
    if isinstance(response, dict):
        if any(response.get(k) is not None for k in ('session_id', 'process_id')) and response.get('exit_code') is None:
            return True
        if response.get('status') in ('running', 'in_progress', 'pending'):
            return True
        if (type(response.get('exit_code')) is int
                or response.get('status') in ('completed', 'failed', 'error', 'timeout', 'cancelled')
                or response.get('timed_out') is True):
            return False
        response = response.get('output', response.get('stdout', ''))
    # Rollouts include a runtime header followed by arbitrary command stdout.
    # A finished `cat logfile` may print a previous running-process message.
    # Such stdout is not evidence that this invocation is still running.
    header = response.partition('\nOutput:')[0] if isinstance(response, str) else ''
    return re.search(r'(?:Process|Script) running with (?:session|cell) ID', header, re.I) is not None


def response_failed(response):
    if isinstance(response, dict):
        code = response.get('exit_code')
        return (response.get('timed_out') is True or response.get('is_error') is True
                or (type(code) is int and code != 0)
                or response.get('status') in ('failed', 'error', 'timeout', 'cancelled'))
    header = response.partition('\nOutput:')[0] if isinstance(response, str) else ''
    return bool(re.search(r'(?:exit(?:ed)?(?:\s+with)?\s+(?:code\s*)[1-9]\d*|timed out)', header, re.I))


def owned_replay_post(event, active):
    """Pinned Codex Bash Post carries raw, possibly truncated command stdout.

    In Codex 0.159.3 this callback exists only after the process exits. Require
    the exact replay command and original ownership before using that terminal
    callback contract; stdout itself can contain arbitrary process/error text.
    Structured responses still go through the normal runtime-status checks.
    """
    if (event.get('hook_event_name') != 'PostToolUse'
            or event.get('tool_name') != 'Bash'
            or not isinstance(event.get('tool_response'), str)
            or not isinstance(event.get('tool_input'), dict)):
        return False
    command = event['tool_input'].get('command')
    if not isinstance(command, str):
        return False
    try:
        arguments = shlex.split(command)
    except ValueError:
        return False
    return arguments == ['python3', str(Path(sys.argv[0]).absolute()), 'replay', active['result_id']]


def codex_home():
    # Use the real Codex configuration location. Never inspect auth/config files.
    return Path(os.environ.get('CODEX_HOME', str(Path.home() / '.codex'))).resolve()


def allowed_rollout_path(path, session_id):
    if re.fullmatch(UUID_PATTERN, session_id) is None:
        return None
    candidate = Path(path)
    if not candidate.is_absolute() or not candidate.name.startswith('rollout-'):
        return None
    if not candidate.name.endswith('-' + session_id + '.jsonl'):
        return None
    home = codex_home()
    try:
        relative = candidate.relative_to(home)
        if relative.parts[0] not in ('sessions', 'archived_sessions') or '..' in relative.parts:
            return None
        # Reject symlinks at every component below Codex home, including sessions.
        prefix = home
        for part in relative.parts:
            prefix = prefix / part
            if prefix.is_symlink():
                return None
        if candidate.resolve(strict=True) != candidate:
            return None
    except (ValueError, OSError, IndexError):
        return None
    return candidate


def rollout_candidates(tool):
    path = tool.get('transcript_path')
    if path is not None:
        candidate = allowed_rollout_path(path, tool['session_id'])
        return [candidate] if candidate is not None else []
    # Backward-compatible ledgers did not record transcript_path. UUIDv7 embeds
    # the session creation date: look only in those days and archived_sessions.
    session_id = tool['session_id']
    if re.fullmatch(UUID_PATTERN, session_id) is None or session_id[14] != '7':
        return []
    try:
        stamp = int(session_id.replace('-', '')[:12], 16) / 1000
        created = datetime.datetime.fromtimestamp(stamp, datetime.timezone.utc)
    except (ValueError, OverflowError, OSError):
        return []
    home = codex_home()
    directories = [home / 'archived_sessions']
    directories += [home / 'sessions' / (created + datetime.timedelta(days=d)).strftime('%Y/%m/%d')
                    for d in (-1, 0, 1)]
    candidates = []
    for directory in directories:
        if directory.is_symlink() or not directory.is_dir():
            continue
        for index, item in enumerate(directory.iterdir()):
            if index >= 16384:
                break
            if item.name.endswith('-' + session_id + '.jsonl'):
                candidate = allowed_rollout_path(str(item), session_id)
                if candidate is not None:
                    candidates.append(candidate)
            if len(candidates) > MAX_ROLLOUT_FILES:
                return []
    return candidates


def rollout_items(path, session_id):
    """Read a bounded snapshot, verifying identity before parsing any tool body."""
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, 'rb') as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode):
            raise ValueError('Rollout must be a regular file')
        first = stream.readline(MAX_ROLLOUT_LINE_BYTES + 1)
        if len(first) > MAX_ROLLOUT_LINE_BYTES or not first.endswith(b'\n'):
            raise ValueError('Invalid rollout identity record')
        metadata = json.loads(first)
        payload = metadata.get('payload')
        if (metadata.get('type') != 'session_meta' or not isinstance(payload, dict)
                or payload.get('id') != session_id
                or payload.get('session_id', session_id) != session_id):
            raise ValueError('Rollout identity does not match pending tool')
        offset = max(0, info.st_size - MAX_ROLLOUT_TAIL_BYTES)
        stream.seek(offset)
        data = stream.read(min(info.st_size - offset, MAX_ROLLOUT_TAIL_BYTES))
    if offset:
        _, separator, data = data.partition(b'\n')
        if not separator:
            return []
    # An append may be in progress. A partial final record proves nothing.
    if not data.endswith(b'\n'):
        data = data.rsplit(b'\n', 1)[0] + b'\n' if b'\n' in data else b''
    items = []
    for line in data.splitlines():
        if len(line) > MAX_ROLLOUT_LINE_BYTES:
            raise ValueError('Rollout record exceeds proof limit')
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError('Invalid rollout record')
        if record.get('type') == 'response_item':
            item = record.get('payload')
            if not isinstance(item, dict):
                raise ValueError('Invalid rollout response item')
            items.append(item)
    return items


def terminal_rollout_calls(items):
    return rollout_call_proofs(items)[0]


def rollout_call_proofs(items):
    """Exact response output proves completion, except yielded process handles.

    A failed native tool can skip PostToolUse in Codex 0.159.3. Its rollout still
    records the model-facing call output. Async exec calls instead retain a
    running handle until a matching write_stdin response explicitly exits.

    Returns (terminal, vanished). `vanished` holds yielded exec calls whose own
    handle a later exact write_stdin reported as unknown (for example after the
    Agent killed the process). That alone does not prove exit; the submit gate
    pairs it with an empty live-process check (`live_agent_processes`).
    """
    calls, terminal, running, vanished = {}, set(), {}, set()
    duplicated = set()
    for item in items:
        kind, call_id = item.get('type'), item.get('call_id')
        if not nonempty_string(call_id):
            continue
        if kind in ('function_call', 'custom_tool_call'):
            if call_id in calls:
                duplicated.add(call_id)
            calls[call_id] = item
            continue
        if kind not in ('function_call_output', 'custom_tool_call_output') or call_id not in calls:
            continue
        output = item.get('output')
        if not isinstance(output, (str, dict)):
            continue
        call = calls[call_id]
        expected = 'custom_tool_call_output' if call['type'] == 'custom_tool_call' else 'function_call_output'
        if kind != expected:
            continue
        if response_running(output):
            if isinstance(output, str):
                header = output.partition('\nOutput:')[0]
                match = re.search(r'^Process running with session ID (\d+)\s*$', header, re.M)
                if match and call.get('name') == 'exec_command':
                    running[call_id] = int(match.group(1))
            continue
        # Once an original exec yielded, only an explicit process-end response
        # can retire it. An error for a different invocation is insufficient.
        if call_id not in running:
            terminal.add(call_id)
        elif isinstance(output, str) and re.search(r'^Process exited with code -?\d+\s*$', output.partition('\nOutput:')[0], re.M):
            terminal.add(call_id)
        if call.get('name') == 'write_stdin' and isinstance(output, str):
            try:
                arguments = json.loads(call.get('arguments', ''))
                process_id = arguments.get('session_id')
            except (ValueError, AttributeError, TypeError):
                continue
            if (type(process_id) is int and
                    re.search(r'^Process exited with code -?\d+\s*$', output.partition('\nOutput:')[0], re.M)):
                terminal.update(original for original, handle in running.items() if handle == process_id)
            elif (type(process_id) is int
                    and re.fullmatch(r'write_stdin failed: Unknown process id (\d+)\s*', output) is not None
                    and int(re.fullmatch(r'write_stdin failed: Unknown process id (\d+)\s*', output).group(1)) == process_id):
                vanished.update(original for original, handle in running.items() if handle == process_id)
    # A reused call ID cannot establish which execution produced the output.
    terminal.difference_update(duplicated)
    vanished.difference_update(duplicated)
    vanished.difference_update(terminal)
    return terminal, vanished


def reconcile_ordinary_tools(state):
    """Remove only exact, terminal ordinary calls; never release a submission."""
    if state['active'] is not None or not state['active_tools']:
        return 0
    proofs = {}
    retired = []
    for key, tool in state['active_tools'].items():
        paths = rollout_candidates(tool)
        proof_key = (tool['session_id'], tuple(str(p) for p in paths))
        if proof_key not in proofs:
            terminal = set()
            try:
                # Duplicate files for one session are ambiguous. Do not merge
                # proofs across them or allow an unbounded multi-session scan.
                if len(paths) == 1 and len(proofs) < MAX_PROOF_SESSIONS:
                    terminal = terminal_rollout_calls(rollout_items(paths[0], tool['session_id']))
            except (OSError, ValueError, TypeError):
                terminal = set()
            proofs[proof_key] = terminal
        if tool['tool_use_id'] in proofs[proof_key]:
            retired.append(key)
    for key in retired:
        del state['active_tools'][key]
    return len(retired)


def process_ancestors():
    """PIDs from this hook up to init: Codex and its wrappers run the hook."""
    chain, pid = set(), os.getpid()
    while pid > 1 and pid not in chain:
        chain.add(pid)
        try:
            fields = Path('/proc', str(pid), 'stat').read_text().rpartition(')')[2].split()
            pid = int(fields[1])
        except (OSError, ValueError, IndexError):
            return None
    return chain


def live_agent_processes():
    """Live (non-zombie) processes of this uid outside the hook's ancestry.

    The Worker starts Codex as the Agent user, and every command the Agent runs,
    including one detached with nohup/setsid, keeps that uid. An empty result is
    therefore positive evidence that no earlier Agent command is still running.
    Returns None when the process table cannot be read completely.
    """
    ancestors = process_ancestors()
    if ancestors is None:
        return None
    root, uid, live = Path(os.environ.get('SFORGE_SUBMIT_WAIT_PROC_ROOT', '/proc')), os.getuid(), []
    try:
        entries = [entry for entry in os.listdir(root) if entry.isdigit()]
    except OSError:
        return None
    for entry in entries:
        pid = int(entry)
        if pid in ancestors:
            continue
        try:
            status = (root / entry / 'status').read_text()
        except FileNotFoundError:
            continue
        except OSError:
            return None
        uid_line = re.search(r'^Uid:\s+(\d+)', status, re.M)
        state_line = re.search(r'^State:\s+(\S)', status, re.M)
        if uid_line is None or state_line is None:
            return None
        if int(uid_line.group(1)) == uid and state_line.group(1) not in ('Z', 'X'):
            live.append(pid)
    return live


def retire_vanished_exec_tools(state):
    """Before a submit only: retire yielded execs whose own handle is unknown
    to Codex, and only when no process of the Agent's uid is still alive.
    Ordinary tools without that proof, and any active submission, are untouched.
    """
    if state['active'] is not None or not state['active_tools']:
        return 0
    candidates = []
    for key, tool in state['active_tools'].items():
        paths = rollout_candidates(tool)
        if len(paths) != 1:
            continue
        try:
            vanished = rollout_call_proofs(rollout_items(paths[0], tool['session_id']))[1]
        except (OSError, ValueError, TypeError):
            continue
        if tool['tool_use_id'] in vanished:
            candidates.append(key)
    if not candidates or live_agent_processes() != []:
        return 0
    for key in candidates:
        del state['active_tools'][key]
    return len(candidates)


def retire_stale_tools_without_agent_processes(state):
    """Before a submit only: retire tools pending longer than STALE_TOOL_SECONDS
    when no process of the Agent's uid is still alive.

    Claude moves a Bash call past its 600 s limit to the background and never
    sends a final PostToolUse for it, so it stays pending after the job ends.
    A Codex exec whose rollout cannot be proven (for example a record above the
    proof limit) stays pending the same way. Every command keeps the Agent's
    uid, so an empty process table proves those jobs ended; in-process tools
    (edits, reads) never stay pending this long. Elapsed time alone never
    retires a tool, and an active submission is untouched.
    """
    if state['active'] is not None or not state['active_tools']:
        return 0
    now = datetime.datetime.now(datetime.timezone.utc)
    stale = []
    for key, tool in state['active_tools'].items():
        started = datetime.datetime.strptime(tool['started_at'], '%Y-%m-%dT%H:%M:%S.%fZ')
        if (now - started.replace(tzinfo=datetime.timezone.utc)).total_seconds() > STALE_TOOL_SECONDS:
            stale.append(key)
    if not stale or live_agent_processes() != []:
        return 0
    for key in stale:
        del state['active_tools'][key]
    return len(stale)


def reconciliation_summary(state, retired):
    return {'schema_version': 1, 'active_submission': state['active'] is not None,
            'pending_tools': [{k: tool[k] for k in ('session_id', 'tool_use_id', 'tool_name', 'started_at')}
                              for tool in state['active_tools'].values()],
            'reconciled_tools': retired}


def agent_writer_pids(agent_uid):
    """Fail closed unless every visible process has an authoritative UID/state."""
    writers = []
    for entry in Path('/proc').iterdir():
        if re.fullmatch(r'[0-9]+', entry.name) is None:
            continue
        try:
            data = (entry / 'status').read_text(encoding='utf-8')
        except FileNotFoundError:
            # A process disappearing during the scan is harmless. A remaining
            # process with missing status cannot establish the writer boundary.
            try:
                entry.stat()
            except FileNotFoundError:
                continue
            raise RuntimeError('Cannot verify a remaining process status') from None
        if len(data) > 65536:
            raise RuntimeError('Process status exceeds the verification limit')
        uid_lines = [line.split()[1:] for line in data.splitlines() if line.startswith('Uid:')]
        state_lines = [line.split()[1:] for line in data.splitlines() if line.startswith('State:')]
        if (len(uid_lines) != 1 or len(uid_lines[0]) != 4
                or any(re.fullmatch(r'[0-9]+', value) is None for value in uid_lines[0])
                or len(state_lines) != 1 or not state_lines[0]
                or len(state_lines[0][0]) != 1):
            raise RuntimeError('Process UID/state cannot be verified')
        if agent_uid in map(int, uid_lines[0]) and state_lines[0][0] != 'Z':
            writers.append(int(entry.name))
    return sorted(writers)


def retire_ordinary_at_boundary(directory, agent_uid):
    """Root may cancel ordinary ledgers only after every Agent writer is gone.

    This proves cancellation at a writer boundary, never command success. An
    active submission, including a cached result, cannot use this recovery.
    """
    if os.geteuid() != 0:
        raise PermissionError('Only root may retire ordinary tools at a writer boundary')
    if (type(agent_uid) is not int or not 0 < agent_uid < 2**32
            or pwd.getpwnam('agent').pw_uid != agent_uid):
        raise ValueError('Writer boundary UID must equal the non-root agent account')
    directory = Path(directory)
    if not directory.is_absolute():
        raise ValueError('State directory must be absolute')
    # Recovery only accepts an established Agent-owned state. In particular,
    # root must not create a private root-owned directory on an idle run.
    directory_info = directory.lstat()
    if (not stat.S_ISDIR(directory_info.st_mode) or directory_info.st_uid != agent_uid
            or stat.S_IMODE(directory_info.st_mode) != 0o700):
        raise ValueError('Writer boundary requires an Agent-owned private state directory')
    state_path = directory / 'state.json'
    metadata = state_path.lstat()
    if (not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != agent_uid
            or stat.S_IMODE(metadata.st_mode) != 0o600):
        raise ValueError('Writer boundary requires an Agent-owned private state file')
    owner = (metadata.st_uid, metadata.st_gid)
    with state_lock(directory, owner=owner):
        # Confirm ownership again while holding the same lock used by hooks.
        locked_metadata = state_path.lstat()
        if (not stat.S_ISREG(locked_metadata.st_mode)
                or (locked_metadata.st_uid, locked_metadata.st_gid) != owner
                or stat.S_IMODE(locked_metadata.st_mode) != 0o600):
            raise ValueError('Writer boundary state ownership changed before locking')
        state = read_state(state_path)
        if state['active'] is not None:
            raise RuntimeError('An active submission cannot be retired at an ordinary writer boundary')
        if agent_writer_pids(agent_uid):
            raise RuntimeError('Agent writer processes remain; ordinary tools cannot be retired')
        retired = [{**{key: tool[key] for key in ('session_id', 'tool_use_id', 'tool_name', 'started_at')},
                    'outcome': 'cancelled_at_writer_boundary'}
                   for tool in state['active_tools'].values()]
        if retired:
            state['active_tools'] = {}
            write_state(state_path, state, owner=owner)
        return {'schema_version': 1, 'agent_uid': agent_uid, 'active_submission': False,
                'outcome': 'cancelled_at_writer_boundary', 'retired_tools': retired,
                'retired_tool_count': len(retired), 'pending_tools': []}


def evaluate(event, phase, state_path):
    if not isinstance(event, dict):
        raise ValueError('Hook input must be an object')
    state = read_state(state_path)
    active = state['active']
    if not all(nonempty_string(event.get(k)) for k in ('tool_use_id', 'session_id')):
        raise ValueError('Every tool hook requires a valid tool_use_id and session_id')
    key = tool_key(event['session_id'], event['tool_use_id'])
    if phase == 'post':
        if active is None:
            if key in state['active_tools']:
                if 'tool_response' not in event:
                    return deny(phase, 'Cannot confirm completion of an earlier tool before a submission.')
                if not response_running(event['tool_response']):
                    del state['active_tools'][key]
                    write_state(state_path, state)
            return {}
        if (event.get('tool_use_id') != active['tool_use_id']
                or event.get('session_id') != active['session_id']):
            return {}
        if not active['result_ready'] or 'tool_response' not in event:
            return deny(phase, 'Cannot confirm completion of the original sforge-submit tool; ' + WAIT_REASON)
        response = event['tool_response']
        result = load_result(state_path.parent, active)
        replay_callback = owned_replay_post(event, active)
        if not replay_callback and response_running(response):
            return deny(phase, 'The submit tool returned a running session rather than a final result. ' + WAIT_REASON)
        state['last_completion'] = {'tool_use_id': active['tool_use_id'], 'session_id': active['session_id'],
                'finished_at': utc_now(), 'outcome': 'failed'
                    if result['exit_code'] != 0 or (not replay_callback and response_failed(response)) else 'completed'}
        state['active'] = None
        write_state(state_path, state)
        try:
            (state_path.parent / ('result-' + active['result_id'] + '.json')).unlink()
        except FileNotFoundError:
            pass
        return {}
    if active is not None:
        return deny(phase, WAIT_REASON)
    if reconcile_ordinary_tools(state):
        write_state(state_path, state)
    tool_name = event.get('tool_name')
    if not nonempty_string(tool_name):
        raise ValueError('Hook input must identify the tool')
    if tool_name.lower().split('.')[-1] in AGENT_TOOLS:
        return deny(phase, 'This SForge run uses one Agent. Delegating to another Agent is disabled so submission waiting cannot be bypassed.')
    try:
        kind, arguments = submit_command(tool_name, event.get('tool_input'))
    except ValueError as error:
        # No submit is active here. Explain malformed/compound input instead
        # of claiming the Agent must wait for a nonexistent earlier submit.
        return deny(phase, str(error))
    if kind != 'submit':
        if not TRACK_ORDINARY_TOOLS:
            return {}
        if key in state['active_tools']:
            return deny(phase, 'This tool call is already pending; wait for its original result.')
        if len(state['active_tools']) >= MAX_ACTIVE_TOOLS:
            return deny(phase, 'Too many tools are still pending; wait for their results before starting another operation.')
        state['active_tools'][key] = {'tool_use_id': event['tool_use_id'],
            'session_id': event['session_id'], 'tool_name': tool_name, 'started_at': utc_now()}
        transcript = event.get('transcript_path')
        if isinstance(transcript, str):
            allowed = allowed_rollout_path(transcript, event['session_id'])
            if allowed is not None:
                state['active_tools'][key]['transcript_path'] = str(allowed)
        write_state(state_path, state)
        return {}
    if state['active_tools'] and retire_vanished_exec_tools(state):
        write_state(state_path, state)
    if state['active_tools'] and retire_stale_tools_without_agent_processes(state):
        write_state(state_path, state)
    if state['active_tools']:
        return deny(phase, 'An earlier tool is still running. Wait for its original session/result or stop that job before running sforge-submit. Evaluation cannot overlap another Agent operation. A tool pending over 15 minutes is released automatically once none of your processes remain.')
    timeout_value = os.environ.get('SFORGE_SUBMIT_TOOL_TIMEOUT_MS', str(DEFAULT_TIMEOUT_MS))
    if not re.fullmatch(r'[1-9]\d*', timeout_value):
        raise ValueError('Invalid SFORGE_SUBMIT_TOOL_TIMEOUT_MS')
    timeout_ms = int(timeout_value)
    if timeout_ms > 172800000:
        raise ValueError('Submit tool timeout exceeds supported limit')
    remaining = os.environ.get('SFORGE_REMAINING_SECONDS')
    if remaining is not None:
        if not re.fullmatch(r'[0-9]+(?:\.[0-9]+)?', remaining) or float(remaining) <= 0:
            raise ValueError('Invalid remaining Agent budget')
        timeout_ms = min(timeout_ms, max(1, int(float(remaining) * 1000)))
    deadline_value = os.environ.get('SFORGE_AGENT_DEADLINE_AT')
    if deadline_value is not None:
        if not re.fullmatch(r'[0-9]+(?:\.[0-9]+)?', deadline_value):
            raise ValueError('Invalid absolute Agent deadline')
        remaining = float(deadline_value) - time.time()
        if remaining <= 0:
            raise ValueError('Agent budget has expired')
        timeout_ms = min(timeout_ms, max(1, int(remaining * 1000)))
    monotonic_deadline = os.environ.get('SFORGE_AGENT_DEADLINE_MONOTONIC')
    if monotonic_deadline is not None:
        remaining = float(monotonic_deadline) - time.monotonic()
        if not math.isfinite(remaining) or remaining <= 0:
            raise ValueError('Invalid or expired monotonic Agent deadline')
        timeout_ms = min(timeout_ms, max(1, int(remaining * 1000)))
    working_directory = event.get('cwd')
    if not isinstance(working_directory, str) or not Path(working_directory).is_absolute():
        raise ValueError('Submit requires the hook working directory')
    result_id = uuid.uuid4().hex
    state['active'] = {'tool_use_id': event['tool_use_id'], 'session_id': event['session_id'],
            'tool_name': tool_name, 'started_at': utc_now(), 'timeout_ms': timeout_ms,
            'result_id': result_id, 'result_ready': False}
    write_state(state_path, state)
    return {'execute': arguments, 'cwd': working_directory, 'active': dict(state['active'])}


def run_submission(action, directory):
    """Run outside the state lock, so other tool hooks can promptly deny."""
    active = action['active']
    log_path = directory / ('output-' + active['result_id'] + '.log')
    descriptor = os.open(log_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
    process = None
    failure = ''
    code = 1
    with os.fdopen(descriptor, 'wb') as output:
        try:
            process = subprocess.Popen(action['execute'], cwd=action['cwd'], env=os.environ,
                stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
            code = process.wait(timeout=active['timeout_ms'] / 1000)
        except subprocess.TimeoutExpired:
            failure = ('\nSForge submission wait timed out before a final result. '
                       'The Controller evaluation may still continue; this tool has failed.\n')
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
        except Exception:
            failure = '\nSForge submission could not complete; this tool has failed.\n'
            if process is not None and process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
    with log_path.open('rb') as stream:
        payload = stream.read(MAX_RESULT_BYTES + 1)
    if len(payload) > MAX_RESULT_BYTES:
        payload = payload[:MAX_RESULT_BYTES] + b'\n[SForge submit output truncated]\n'
    output_text = payload.decode('utf-8', errors='replace')[:MAX_RESULT_BYTES // 4] + failure
    result = {'schema_version': 1, 'result_id': active['result_id'],
              'tool_use_id': active['tool_use_id'], 'session_id': active['session_id'],
              'exit_code': code if code >= 0 else 1, 'output': output_text}
    state_path = directory / 'state.json'
    with state_lock(directory):
        state = read_state(state_path)
        if state['active'] != active:
            raise ValueError('Submit ownership changed before result persistence')
        write_state(directory / ('result-' + active['result_id'] + '.json'), result)
        state['active']['result_ready'] = True
        write_state(state_path, state)
    log_path.unlink()
    replay = 'python3 ' + shlex.quote(str(Path(sys.argv[0]).absolute())) + ' replay ' + active['result_id']
    return {'hookSpecificOutput': {'hookEventName': 'PreToolUse', 'permissionDecision': 'allow',
                                  'updatedInput': {'command': replay}}}


class state_lock:
    def __init__(self, directory, *, owner=None):
        self.directory = directory
        self.owner = owner

    def __enter__(self):
        descriptor = os.open(self.directory / 'lock', os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        self.stream = os.fdopen(descriptor, 'a+b')
        try:
            if not stat.S_ISREG(os.fstat(self.stream.fileno()).st_mode):
                raise ValueError('Submit wait lock must be a regular file')
            fcntl.flock(self.stream, fcntl.LOCK_EX)
            if self.owner is not None:
                os.fchown(self.stream.fileno(), *self.owner)
                os.fchmod(self.stream.fileno(), 0o600)
        except BaseException:
            self.stream.close()
            raise
        return self

    def __exit__(self, *ignored):
        self.stream.close()


def load_result(directory, active):
    descriptor = os.open(directory / ('result-' + active['result_id'] + '.json'), os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, 'rb') as stream:
        metadata = os.fstat(stream.fileno())
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > MAX_RESULT_BYTES + MAX_STATE_BYTES:
            raise ValueError('Invalid result cache')
        result = json.load(stream)
    if (not isinstance(result, dict)
            or set(result) != {'schema_version', 'result_id', 'tool_use_id', 'session_id', 'exit_code', 'output'}
            or type(result.get('schema_version')) is not int or result.get('schema_version') != 1
            or result.get('result_id') != active['result_id']
            or result.get('tool_use_id') != active['tool_use_id']
            or result.get('session_id') != active['session_id']
            or not isinstance(result.get('output'), str) or type(result.get('exit_code')) is not int
            or not 0 <= result['exit_code'] <= 255):
        raise ValueError('Result cache does not match submit ownership')
    return result


def replay_result(directory, result_id):
    if re.fullmatch(r'[a-f0-9]{32}', result_id) is None:
        raise ValueError('Invalid result identity')
    with state_lock(directory):
        state = read_state(directory / 'state.json')
        active = state['active']
        if active is None or active['result_id'] != result_id or not active['result_ready']:
            raise ValueError('Submit result is not available for this call')
        result = load_result(directory, active)
    sys.stdout.write(result['output'])
    return result['exit_code']


def main():
    if sys.argv[1:2] == ['retire-ordinary-at-boundary']:
        try:
            if len(sys.argv) != 3 or re.fullmatch(r'[1-9][0-9]{0,9}', sys.argv[2]) is None:
                raise ValueError('Writer boundary requires one positive agent UID')
            directory = Path(os.environ.get('SFORGE_SUBMIT_WAIT_STATE_DIR', STATE_DIR))
            result = retire_ordinary_at_boundary(directory, int(sys.argv[2]))
            print(json.dumps(result, separators=(',', ':')))
            return 0
        except Exception as error:
            print(json.dumps({'schema_version': 1, 'error': 'Cannot retire ordinary tools at writer boundary',
                              'error_type': type(error).__name__}, separators=(',', ':')))
            return 1
    if len(sys.argv) == 2 and sys.argv[1] == 'reconcile':
        try:
            directory = Path(os.environ.get('SFORGE_SUBMIT_WAIT_STATE_DIR', STATE_DIR))
            if not directory.is_absolute():
                raise ValueError('State directory must be absolute')
            directory.mkdir(mode=0o700, parents=True, exist_ok=True)
            with state_lock(directory):
                state_path = directory / 'state.json'
                state = read_state(state_path)
                retired = reconcile_ordinary_tools(state)
                if retired:
                    write_state(state_path, state)
                result = reconciliation_summary(state, retired)
            print(json.dumps(result, separators=(',', ':')))
            return 0
        except Exception:
            print(json.dumps({'schema_version': 1, 'error': 'Cannot validate pending tool completion proofs'}))
            return 1
    if len(sys.argv) == 3 and sys.argv[1] == 'replay':
        try:
            directory = Path(os.environ.get('SFORGE_SUBMIT_WAIT_STATE_DIR', STATE_DIR))
            return replay_result(directory, sys.argv[2])
        except Exception:
            print('SForge submit result replay failed; no valid result is available.', file=sys.stderr)
            return 1
    phase = sys.argv[1] if len(sys.argv) == 2 and sys.argv[1] in ('pre', 'post') else 'pre'
    try:
        payload = sys.stdin.buffer.read(MAX_INPUT_BYTES + 1)
        if len(payload) > MAX_INPUT_BYTES:
            raise ValueError('Hook input exceeds size limit')
        event = json.loads(payload)
        directory = Path(os.environ.get('SFORGE_SUBMIT_WAIT_STATE_DIR', STATE_DIR))
        if not directory.is_absolute():
            raise ValueError('Submit wait state directory must be absolute')
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        with state_lock(directory):
            result = evaluate(event, phase, directory / 'state.json')
        if 'execute' in result:
            result = run_submission(result, directory)
    except Exception as error:
        # Hook crashes are commonly treated as non-blocking by agent CLIs. Emit
        # a valid policy denial with exit zero, including for corrupted state.
        reason = 'SForge submit wait policy could not validate its state or input; refusing this tool.'
        try:
            directory = Path(os.environ.get('SFORGE_SUBMIT_WAIT_STATE_DIR', STATE_DIR))
            if not directory.is_absolute():
                raise ValueError('State directory must be absolute')
            if directory.exists():
                with state_lock(directory):
                    current = read_state(directory / 'state.json')
            else:
                current = read_state(directory / 'state.json')
            if current['active'] is not None:
                reason += ' ' + WAIT_REASON
            elif phase == 'pre' and isinstance(error, ValueError):
                reason = ('Hook input is invalid JSON: ' + error.msg if isinstance(error, json.JSONDecodeError)
                          else 'Invalid numeric Agent deadline' if str(error).startswith('could not convert string to float:')
                          else str(error))
        except Exception:
            pass
        result = deny(phase, reason)
    print(json.dumps(result, separators=(',', ':')))
    return 0


if __name__ == '__main__':
    sys.exit(main())
'''
