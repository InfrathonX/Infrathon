#!/usr/bin/env python3
"""Reuse the existing SForge relay with arbitrary CONNECT destinations disabled."""
import importlib.util
from pathlib import Path

source = Path('/opt/existing/restricted_connect_proxy.py')
spec = importlib.util.spec_from_file_location('sforge_existing_relay', source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.ALLOWED_CONNECT_HOSTS = frozenset()
raise SystemExit(module.main())
