"""Prepared controller hook: project old two roles, keep QA78 mounted separately.

Accept only already Deployment-validated standard records. No SDK or asset
upload is performed here. Caller supplies the image-owned public descriptor.
"""
from __future__ import annotations
import copy

SMOKE_ID = 'd28499bd345f06abbad189941a5c5971a9b700ae01af283c9f4681c1349b480f'
SMOKE_MOUNT = '/workspace/benchmark/qa-smoke'
SMOKE_FILES = [
    {'path':'manifest.json','size':68370,'sha256':'1a23433aa7bdd20e831bfa93096f369c53814e10a8f06810f69a6909764f2788'},
    {'path':'prompts.jsonl','size':13152557,'sha256':'ed1ca52f26fb6bfdf1f0402ae66b8ed9664f39ab986ed8b268d83beff97d1d66'}]


def project_worker_assets(rows, *, contract, validator):
    """Return native two-role JSON; optional smoke remains a standard mount."""
    descriptors = {r['role']:r for r in validator.validate_contract(contract, 'public-diagnostic')['assets']}
    selected, seen = [], set()
    for original in rows:
        row = copy.deepcopy(original);role=row.get('role')
        if role in seen:raise ValueError('duplicate worker asset role')
        seen.add(role)
        if row.get('visibility','public')!='public' or 'worker' not in row.get('consumers',[]) or row.get('read_only') is not True:
            raise ValueError('only Deployment-validated public worker RO rows are accepted')
        if role not in descriptors:raise ValueError('unexpected worker asset role')
        descriptor=descriptors[role]
        if (row.get('asset_id')!=descriptor['asset_id'] or row.get('manifest_sha256')!=descriptor['asset_id']
            or row.get('files')!=descriptor['files']):
            raise ValueError('standard row differs from frozen image descriptor')
        selected.append({**descriptor,'mount_path':row['mount_path'],
            'source_directory':'/workspace/'+row['object_prefix'],'read_only':True,
            'visibility':'public','consumers':['worker'],'storage_identity':row['storage_identity']})
    return validator.validate_bindings({'schema_version':1,'protocol':validator.PROTOCOL,
        'assets':selected},contract,purpose='public-diagnostic')
