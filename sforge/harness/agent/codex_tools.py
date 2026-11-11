"""Select direct Codex tools without changing a pinned model's other metadata."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

CODEX_TOOL_CATALOG_PATH = '/home/agent/.codex/sforge-model-catalog.json'
MAX_CATALOG_BYTES = 4 * 1024 * 1024


def direct_tool_catalog(document: dict, model: str) -> str:
    """Retain runtime defaults, changing only the selected model's tool mode.

    Unknown provider models retain Codex's own fallback metadata. In particular,
    this must not borrow an OpenAI model's context limit or prompt for DeepSeek.
    """
    if not isinstance(document, dict) or not isinstance(document.get('models'), list):
        raise ValueError('Pinned Codex runtime did not provide a bundled model catalog')
    if not document['models'] or not isinstance(model, str) or not model:
        raise ValueError('A model and nonempty bundled Codex catalog are required')
    catalog = json.loads(json.dumps(document))
    for entry in catalog['models']:
        if not isinstance(entry, dict) or not isinstance(entry.get('slug'), str):
            raise ValueError('Invalid bundled Codex model metadata')
        if entry['slug'] == model:
            entry['tool_mode'] = 'direct'
    serialized = json.dumps(catalog, ensure_ascii=False)
    if len(serialized.encode()) > MAX_CATALOG_BYTES:
        raise ValueError('Bundled Codex model catalog exceeds size limit')
    return serialized


def load_direct_tool_catalog(runtime, model: str) -> str:
    """Read this runtime's embedded catalog offline, before allocating a Worker."""
    executable = Path(runtime.root) / 'bin' / 'codex'
    # The bundled debug command does not query a provider or need credentials.
    environment = {k: v for k, v in os.environ.items()
                   if k in ('PATH', 'LANG', 'LC_ALL', 'USER', 'LOGNAME', 'SHELL')}
    # bin/codex is a node script; the runtime ships its own pinned node, so a
    # host without a system node can still read the catalog.
    environment['PATH'] = str(Path(runtime.root) / 'bin') + os.pathsep + environment.get('PATH', os.defpath)
    try:
        result = subprocess.run(
            [str(executable), 'debug', 'models', '--bundled'],
            env=environment, capture_output=True, timeout=30, check=True,
        )
        if len(result.stdout) > MAX_CATALOG_BYTES:
            raise ValueError('Bundled Codex model catalog exceeds size limit')
        return direct_tool_catalog(json.loads(result.stdout), model)
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        raise ValueError('Pinned Codex runtime cannot configure direct submission tools; '
                         'use the verified Codex 0.156.1 runtime') from exc
