"""Prepare shared model/data bindings from an explicit controller-local spec."""
import json
from pathlib import Path

from sforge.pyromind.assets import prepare_asset
from sforge.pyromind.cli import write_json
from sforge.pyromind.storage import storage_client, storage_connection


def prepare_assets(args):
    spec_path = args.spec.resolve()
    spec = json.loads(spec_path.read_text())
    schema = spec.get('schema_version')
    if schema not in (1, 2) or not spec.get('assets'):
        raise ValueError('Asset preparation spec requires schema_version 1 or 2 and assets')
    catalog = None
    if schema == 2:
        from sforge.pyromind.asset_catalog import AssetCatalog
        client, bucket, endpoint = storage_connection()
        catalog = AssetCatalog(client, bucket, endpoint, scope=spec.get('scope', 'shared'))
    else:
        client, bucket = storage_client()
    bindings = []
    mounts = set()
    for item in spec['assets']:
        visibility = item.get('visibility', 'public')
        consumers = item.get('consumers', ['worker', 'judge'])
        if visibility == 'private':
            if schema != 2 or spec.get('scope') != 'judge-private' or consumers != ['judge']:
                raise ValueError('Private assets require schema 2, judge-private scope and Judge-only consumers')
        elif (visibility != 'public' or consumers not in (['worker'], ['worker', 'judge'], ['judge', 'worker'])
              or spec.get('scope') == 'judge-private'):
            raise ValueError('Public assets require shared scope and Worker or Worker/Judge consumers')
        if item['mount_path'] in mounts:
            raise ValueError('Duplicate shared asset mount')
        mounts.add(item['mount_path'])
        source = Path(item['source']).expanduser() if item.get('source') else None
        manifest_path = Path(item['manifest']).expanduser()
        if source is not None and not source.is_absolute():
            source = spec_path.parent / source
        if not manifest_path.is_absolute():
            manifest_path = spec_path.parent / manifest_path
        manifest = json.loads(manifest_path.read_text())
        if catalog is not None:
            result = catalog.ensure(manifest, local_root=source, provenance=item['provenance'],
                                    kind=item.get('kind', 'models'), legacy_binding=item.get('legacy_binding'))
            bindings.append(catalog.binding(result, item['mount_path'], item.get('role')))
            print(f"Verified {item['mount_path']}: asset {result['asset_id']}, "
                  f"uploaded {result['uploaded_files']}, reused {result['reused_files']} files", flush=True)
        else:
            if source is None:
                raise ValueError('Schema 1 requires a local source')
            result = prepare_asset(client,bucket,source,manifest,item['object_prefix'])
            bindings.append({'mount_path':item['mount_path'],'object_prefix':result['data_prefix'],
                'manifest_sha256':result['manifest_sha256'],'read_only':True,'files':manifest['files']})
            print(f"Verified {item['mount_path']}: {result['file_count']} files, {result['total_bytes']} bytes",flush=True)
        bindings[-1].update(visibility=visibility, consumers=list(consumers))
    write_json(args.output.resolve(),{'schema_version':1,'assets':bindings})
    print(f'Read-only asset bindings: {args.output.resolve()}',flush=True)
