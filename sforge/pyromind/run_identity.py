"""Readable Storage run IDs and short, stable Sandbox ownership prefixes.

Run IDs are never parsed to recover configuration: run.json retains the full
values. Sandbox names have a separate prefix because the backend truncates
names before appending its own random suffix.
"""
from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import re
import uuid


def _text(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{field} must be a nonempty string')
    return value


def _slug(value, limit):
    cleaned = re.sub(r'[^a-z0-9]+', '-', value.lower()).strip('-') or 'value'
    # Hash transformed values as well as truncated ones: a/b and a-b must not
    # silently receive the same displayed identifier.
    if cleaned != value or len(cleaned) > limit:
        digest = hashlib.sha256(value.encode()).hexdigest()[:8]
        cleaned = cleaned[:limit - 9].rstrip('-') + '-' + digest
    return cleaned


def _timestamp(value):
    if value is None:
        value = datetime.now(timezone.utc)
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError('timestamp must include a timezone')
        return value.astimezone(timezone.utc).strftime('%Y%m%d-%H%M%S')
    if isinstance(value, str) and re.fullmatch(r'[0-9]{8}-[0-9]{6}', value):
        datetime.strptime(value, '%Y%m%d-%H%M%S')
        return value
    raise ValueError('timestamp must be an aware datetime or YYYYmmdd-HHMMSS in UTC')


def make_run_identity(profile, *, preflight, model, budget_seconds,
                      timestamp=None, suffix=None):
    """Return public identity fields without copying the model environment.

    ``model`` is the loaded model configuration mapping. Only MODEL and
    REASONING_EFFORT are read; provider URLs and credentials are never included.
    A preflight has no Agent identity or active Agent time budget.
    """
    profile = _text(profile, 'profile')
    if not isinstance(preflight, bool):
        raise ValueError('preflight must be a boolean')
    if not isinstance(model, Mapping):
        raise ValueError('model must be a configuration mapping')
    stamp = _timestamp(timestamp)
    suffix = uuid.uuid4().hex[:6] if suffix is None else suffix
    if not isinstance(suffix, str) or re.fullmatch(r'[0-9a-f]{6}', suffix) is None:
        raise ValueError('suffix must be six lowercase hexadecimal characters')
    kind = 'preflight' if preflight else 'agent'
    parts = [kind, _slug(profile, 32)]
    if preflight:
        label = f'Preflight · {profile} · no Agent'
    else:
        name = _text(model.get('MODEL'), 'MODEL')
        effort = _text(model.get('REASONING_EFFORT', 'high'), 'REASONING_EFFORT')
        if (isinstance(budget_seconds, bool) or not isinstance(budget_seconds, int)
                or not 1 <= budget_seconds <= 86400):
            raise ValueError('Agent budget_seconds must be an integer in [1, 86400]')
        budget = (f'{budget_seconds // 3600}h' if budget_seconds % 3600 == 0
                  else f'{budget_seconds}s')
        parts.extend([_slug(name, 40), _slug(effort, 12), budget])
        label = f'Agent · {profile} · {name} · effort={effort} · budget={budget}'
    parts.extend([stamp, suffix])
    run_id = '--'.join(parts)
    if len(run_id) > 180:
        raise ValueError('Run identity exceeds the safe VersionStore limit')
    return {'run_id': run_id, 'run_kind': kind,
            'run_label': ' '.join(label.split()),
            'sandbox_name_prefix': 'sf-' + hashlib.sha256(run_id.encode()).hexdigest()[:20]}
