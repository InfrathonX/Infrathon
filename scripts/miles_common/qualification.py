#!/usr/bin/env python3
"""The qualification record of a Miles task for an execution profile.

A task's runtime/qualification.json records the qualification on its default execution profile. A task
qualified on another profile too (e.g. the 80 GB H100 SXM pair, `dual-cuda-sm90-nccl-80g`) keeps that
qualification next to it in runtime/qualification.<profile_id>.json: the host-side record (status, reference
scores and T, CPU pinning, memory peaks) of that hardware. The Judge image carries only
runtime/qualification.json, so the sections the Judge reads (its scoring constants and gates) must be the same
in both records; see `differing_sections`.
"""
from __future__ import annotations

import json
from pathlib import Path


def record_path(default: Path, profile_id: str) -> Path:
    """``default`` (runtime/qualification.json) if it was measured on ``profile_id``, else
    qualification.<profile_id>.json next to it when that file exists, else ``default`` (whose profile mismatch
    the caller then reports)."""
    default = Path(default)
    if json.loads(default.read_text()).get("execution_profile_id") == profile_id:
        return default
    other = default.with_name(f"qualification.{profile_id}.json")
    return other if other.is_file() else default


def load(default: Path, profile_id: str) -> tuple[str, dict]:
    """(a printable name of the record, the record) for ``profile_id``."""
    path = record_path(default, profile_id)
    return f"{path.parent.name}/{path.name}", json.loads(path.read_text())


def differing_sections(default: Path, profile_id: str, sections: list[str]) -> list[str]:
    """The ``sections`` whose value differs between ``default`` (the record in the Judge image) and the record
    of ``profile_id``."""
    base = json.loads(Path(default).read_text())
    other = json.loads(record_path(default, profile_id).read_text())
    return [s for s in sections if base.get(s) != other.get(s)]
