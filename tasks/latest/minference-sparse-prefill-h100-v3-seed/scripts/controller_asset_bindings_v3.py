"""Trusted host metadata loader for the on-demand asset contract.

This supplies the explicit expected descriptor to the controller; the installed
Judge independently uses its protected root400 contract before bootstrap.
No Docker, upload, model reads, or GPU API is called here.
"""
from __future__ import annotations
import hashlib
import os
from pathlib import Path
import re
import stat
import external_assets_on_demand_v3 as A


def read_controller_json(path, expected_sha256=None):
    path=Path(path)
    if (not path.is_absolute() or any(p.is_symlink() for p in (path,*path.parents))
            or '..' in path.parts):raise ValueError('exact absolute controller input required')
    before=path.lstat()
    if (not stat.S_ISREG(before.st_mode) or before.st_nlink!=1 or before.st_uid!=os.geteuid()
            or stat.S_IMODE(before.st_mode) not in (0o400,0o600)
            or not 0<before.st_size<=A.MAX_INPUT_BYTES):raise ValueError('controller-owned private bounded input required')
    raw=A._read_file(path,limit=A.MAX_INPUT_BYTES)
    if expected_sha256 is not None and (not re.fullmatch('[0-9a-f]{64}',expected_sha256)
            or A.digest(raw)!=expected_sha256):raise ValueError('expected protected contract digest differs')
    return raw,A.strict_json(raw)


def load_asset_bindings(path, *, expected_contract_path, expected_contract_sha256):
    raw,contract=read_controller_json(expected_contract_path,expected_contract_sha256)
    A.validate_contract(contract)
    _,bindings=read_controller_json(path)
    return A.validate_bindings(bindings,contract)


def validate_asset_inspection(inspection,bindings,image_id,*,contract):
    # Exact four scopes/image/root/no-network checks retain their original code.
    return A.validate_inspection(inspection,bindings,image_id,contract=contract)
