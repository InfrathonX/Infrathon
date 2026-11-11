"""Small, read-only run metadata catalog and separately authorized publication.

This indexes historical run.json snapshots, never live Sandbox state. Publication
adds navigation objects only; it cannot move or rewrite a run or its archives.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import html
import io
import json
import math
import re
from urllib.parse import quote
from zoneinfo import ZoneInfo


MAX_METADATA_BYTES = 1024 * 1024
RUN_KINDS = ('agent', 'preflight', 'judge-check', 'transport-check', 'unknown')
_ID = re.compile(r'[A-Za-z0-9][A-Za-z0-9._-]{0,239}\Z')
_LABELS = {
    'agent': 'Agent 实验', 'preflight': 'Worker 流程预检',
    'judge-check': 'Judge 验证', 'transport-check': '保存与挂载验证',
    'unknown': '尚未分类（元数据不足或损坏）',
}
_EXECUTION_LABELS = {'started': '已启动', 'not-started': '未启动',
                     'not-applicable': '不适用', 'unknown': '无启动记录'}


def _prefix(value):
    if (not isinstance(value, str) or not value or value.startswith('/')
            or any(not _ID.fullmatch(p) or p in {'.', '..'} for p in value.split('/'))):
        raise ValueError('Invalid catalog object prefix')
    return value


def _text(value):
    if not isinstance(value, str):
        return None
    # Bound display fields and strip control/newline/bidi characters.
    return ''.join(c if c.isprintable() else ' ' for c in value[:512]).strip() or None


def _seconds(value):
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            if math.isfinite(value) and value >= 0:
                return value
        except OverflowError:
            pass
    return None


def _started(value):
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return _seconds(value) is not None and value > 0
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace('Z', '+00:00')).year >= 1970
        except ValueError:
            pass
    return False


def _classify(data):
    explicit = data.get('run_kind')
    scope = data.get('qualification_scope')
    progress = data.get('agent_progress')
    progress = progress if isinstance(progress, dict) else {}
    started = _started(data.get('agent_started_at')) or _started(progress.get('started_at'))
    if explicit in ('agent', 'preflight'):
        kind, evidence = explicit, 'explicit-run-kind'
    elif scope == 'task-only' or data.get('status') == 'preflight-completed':
        kind, evidence = 'preflight', 'task-only-or-preflight-status'
    elif scope == 'task-and-agent':
        kind, evidence = 'agent', 'task-and-agent-scope'
    elif (isinstance(data.get('scope'), str)
          and data['scope'].startswith(('trusted-positive', 'native-non-release'))
          or data.get('kind') == 'trusted-oracle-qualification'):
        kind, evidence = 'judge-check', 'judge-qualification-scope'
    elif data.get('agent_started') is False and data.get('mode') in ('current', 'saved'):
        kind, evidence = 'transport-check', 'transport-mode-without-agent'
    elif started:
        kind, evidence = 'agent', 'agent-start-timestamp'
    else:
        kind, evidence = 'unknown', 'insufficient-metadata'
    execution = ('started' if started else 'not-started' if data.get('agent_started') is False
                 else 'unknown') if kind == 'agent' else (
        'unknown' if kind == 'unknown' else 'not-applicable')
    return kind, evidence, execution


def _record(run_id, data):
    kind, evidence, execution = _classify(data)
    deployment = data.get('deployment')
    profile = data.get('profile')
    if profile is None and isinstance(deployment, dict):
        profile = deployment.get('name')
    budget = _seconds(data.get('budget_seconds'))
    return {
        'run_id': run_id, 'task_id': _text(data.get('task_id')),
        'profile': _text(profile), 'run_kind': kind,
        'classification_evidence': evidence,
        'status': _text(data.get('status')) or 'unknown',
        'agent_execution': execution,
        'model': _text(data.get('model')) if kind in ('agent', 'unknown') else None,
        'reasoning_effort': _text(data.get('reasoning_effort')) if kind in ('agent', 'unknown') else None,
        'budget_seconds': budget,
        'applied_agent_budget_seconds': budget if execution == 'started' else None,
        'run_path': run_id + '/', 'metadata_path': run_id + '/run.json',
    }


def _read_limited(client, bucket, key, maximum):
    response = client.get_object(bucket, key)
    try:
        data = response.read(maximum + 1)
    finally:
        response.close()
        response.release_conn()
    if len(data) > maximum:
        raise ValueError('Catalog metadata exceeds size limit')
    return data


def _reject_constant(value):
    raise ValueError('Nonfinite JSON number')


def build_run_catalog(client, bucket, prefix='sforge-pyromind'):
    """GET at most 1 MiB per run.json, with one nonrecursive directory listing.

    Bad/missing run metadata remains visible as an unknown row plus an issue.
    Exceptions and arbitrary metadata are never copied into public output.
    """
    prefix = _prefix(prefix)
    rows, issues, seen = [], [], set()
    for item in client.list_objects(bucket, prefix=prefix + '/', recursive=False):
        key = item.object_name
        if not isinstance(key, str) or not key.startswith(prefix + '/'):
            issues.append({'code': 'invalid-directory-entry'})
            continue
        if not getattr(item, 'is_dir', key.endswith('/')):
            continue
        relative = key[len(prefix) + 1:]
        if relative == '_index/':
            continue
        run_id = relative[:-1] if relative.endswith('/') else ''
        if not _ID.fullmatch(run_id) or '/' in run_id:
            issues.append({'code': 'unsafe-run-directory'})
            continue
        if run_id in seen:
            continue
        seen.add(run_id)
        try:
            content = _read_limited(client, bucket, key + 'run.json', MAX_METADATA_BYTES)
        except Exception:
            issues.append({'code': 'run-metadata-unreadable', 'run_id': run_id})
            rows.append(_record(run_id, {'status': 'metadata-unavailable'}))
            continue
        try:
            data = json.loads(content, parse_constant=_reject_constant)
            if not isinstance(data, dict):
                raise ValueError('Run metadata must be an object')
            if data.get('run_id') != run_id:
                raise ValueError('Run identity does not match directory')
            row = _record(run_id, data)
        except (ValueError, TypeError, UnicodeDecodeError):
            issues.append({'code': 'invalid-run-metadata', 'run_id': run_id})
            rows.append(_record(run_id, {'status': 'metadata-invalid'}))
            continue
        rows.append(row)
        if row['run_kind'] == 'unknown':
            issues.append({'code': 'run-kind-unknown', 'run_id': run_id})
    return {'schema_version': 1,
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'snapshot_notice': 'Historical run.json metadata; not live monitoring.',
            'runs': sorted(rows, key=lambda row: row['run_id']), 'issues': issues}


def _markdown(value):
    value = _text(value) or '—'
    return html.escape(value, quote=True).replace('\\', '&#92;').replace('|', '&#124;').replace(
        '[', '&#91;').replace(']', '&#93;').replace('`', '&#96;').replace('*', '&#42;').replace('_', '&#95;')


def _json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()


def _safe_row(row):
    # render_catalog_files is also a public boundary: never serialize unknown keys.
    run_id = row.get('run_id')
    if not isinstance(run_id, str) or not _ID.fullmatch(run_id):
        raise ValueError('Invalid run ID in catalog')
    kind = row.get('run_kind')
    if kind not in RUN_KINDS:
        raise ValueError('Invalid run kind in catalog')
    execution = row.get('agent_execution')
    if execution not in ('started', 'not-started', 'not-applicable', 'unknown'):
        raise ValueError('Invalid agent execution in catalog')
    result = {key: _text(row.get(key)) for key in (
        'task_id', 'profile', 'classification_evidence', 'status', 'model', 'reasoning_effort')}
    result.update(run_id=run_id, run_kind=kind, agent_execution=execution,
                  budget_seconds=_seconds(row.get('budget_seconds')),
                  applied_agent_budget_seconds=_seconds(row.get('applied_agent_budget_seconds')),
                  run_path=run_id + '/', metadata_path=run_id + '/run.json')
    if kind not in ('agent', 'unknown'):
        result.update(model=None, reasoning_effort=None)
    if kind != 'agent' or execution != 'started':
        result['applied_agent_budget_seconds'] = None
    return result


def _pointer_name(row):
    seconds = row['budget_seconds'] if row['run_kind'] == 'agent' else None
    hours = format(seconds / 3600, 'g') + 'h' if seconds is not None else 'no-agent-budget'
    label = '--'.join(str(v or 'na') for v in (
        row['profile'] or row['task_id'], row['model'], row['reasoning_effort'], hours))
    label = re.sub(r'[^A-Za-z0-9._-]+', '-', label).strip('-.') or 'run'
    name = label + '--' + row['run_id'] + '.json'
    if len(name) > 220:
        digest = hashlib.sha256(name.encode()).hexdigest()[:16]
        name = label[:75] + '--' + row['run_id'][:115] + '--' + digest + '.json'
    return name


def render_catalog_files(catalog):
    """Return small relative navigation-object keys and their UTF-8 bytes."""
    rows = sorted((_safe_row(row) for row in catalog['runs']), key=lambda row: row['run_id'])
    generated_at = _text(catalog.get('generated_at')) or 'unknown'
    try:
        instant = datetime.fromisoformat(generated_at.replace('Z', '+00:00'))
        if instant.tzinfo is None:
            raise ValueError('Catalog timestamp must include timezone')
        displayed_time = instant.astimezone(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S %z')
        time_label = '索引生成时间（北京时间）：'
    except ValueError:
        displayed_time, time_label = generated_at, '索引生成时间（原始记录）：'
    issues = []
    for issue in catalog.get('issues', []):
        value = {'code': _text(issue.get('code')) or 'unknown'}
        if isinstance(issue.get('run_id'), str) and _ID.fullmatch(issue['run_id']):
            value['run_id'] = issue['run_id']
        issues.append(value)
    clean = {'schema_version': 1, 'generated_at': generated_at,
             'snapshot_notice': 'Historical run.json metadata; not live monitoring.',
             'runs': rows, 'issues': issues}
    lines = ['# Run 目录导航', '', time_label + _markdown(displayed_time), '',
             '这是历史 run.json 的分类快照，不是实时监控。原始归档未搬迁，run.json 未改写。',
             '正式 Agent 实验也可能尚未启动或失败；请同时检查 Agent 状态和记录状态。',
             'Agent 设置预算不表示实际已运行时长；预检不使用历史默认六小时值。',
             '路径相对于本目录；详细字段见 [run-index.json](run-index.json)。', '']
    files = {'run-index.json': _json_bytes(clean)}
    for kind in RUN_KINDS:
        selected = [row for row in rows if row['run_kind'] == kind]
        lines.extend(['## ' + _LABELS[kind] + ' (' + str(len(selected)) + ')', '',
                      '| 原始目录 | 任务 | 模型 | Effort | 设置 Agent 预算 | Agent 状态 | 记录状态 |',
                      '| --- | --- | --- | --- | --- | --- | --- |'])
        for row in selected:
            seconds = row['budget_seconds'] if kind == 'agent' else None
            hours = format(seconds / 3600, 'g') + ' 小时' if seconds is not None else (
                '不适用' if kind not in ('agent', 'unknown') else '未知')
            link = '[' + _markdown(row['run_id']) + '](' + quote(row['run_id']) + '/)'
            cells = [link] + [_markdown(v) for v in (
                row['profile'] or row['task_id'], row['model'], row['reasoning_effort'], hours,
                _EXECUTION_LABELS[row['agent_execution']], row['status'])]
            lines.append('| ' + ' | '.join(cells) + ' |')
            files['_index/' + kind + '/' + _pointer_name(row)] = _json_bytes({
                'schema_version': 1, 'generated_at': generated_at,
                'snapshot_notice': clean['snapshot_notice'],
                'path_base': 'catalog root; not this pointer directory',
                'run_path_from_pointer_directory': '../../' + row['run_path'],
                'run': row})
        lines.append('')
    lines.extend(['## 元数据问题 (' + str(len(issues)) + ')', '',
                  '缺失、损坏或无法分类的目录不会被当成成功实验；请检查原始 run.json。', ''])
    lines.extend('- ' + _markdown(issue.get('run_id')) + ': ' + _markdown(issue['code'])
                 for issue in issues)
    files['RUNS.md'] = ('\n'.join(lines) + '\n').encode()
    return files


def _allowed_publication_key(key):
    if key in ('RUNS.md', 'run-index.json'):
        return True
    if not isinstance(key, str):
        return False
    parts = key.split('/')
    return (len(parts) == 3 and parts[0] == '_index' and parts[1] in RUN_KINDS
            and len(parts[2]) <= 220 and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*\.json', parts[2])
            is not None)


def publish_catalog_files(client, bucket, files, prefix='sforge-pyromind'):
    """PUT navigation objects only, then verify their exact bytes by GET.

    No listing, deleting, moving, or overwriting of original run paths is allowed.
    Validate the complete batch before making any remote mutation.
    """
    prefix = _prefix(prefix)
    for key, data in files.items():
        if not _allowed_publication_key(key) or not isinstance(data, bytes):
            raise ValueError('Publication is restricted to run catalog navigation objects')
    published = []
    # Root index files are written after their small pointers.
    for relative in sorted(files, key=lambda key: (not key.startswith('_index/'), key)):
        key, data = prefix + '/' + relative, files[relative]
        client.put_object(bucket, key, io.BytesIO(data), len(data))
        actual = _read_limited(client, bucket, key, len(data))
        if actual != data:
            raise RuntimeError('Catalog publication readback mismatch')
        published.append(relative)
    return published
