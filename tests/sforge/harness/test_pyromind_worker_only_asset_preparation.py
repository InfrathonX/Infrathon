"""Public QA diagnostics preserve Worker-only scope through standard prepare-assets."""
import json
from types import SimpleNamespace

import pytest

from sforge.pyromind import asset_commands, asset_catalog


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    manifest=tmp_path/'manifest.json'
    manifest.write_text(json.dumps({'schema_version':1,'files':[
        {'path':'manifest.json','sha256':'a'*64,'size':1}]}))
    monkeypatch.setattr(asset_commands,'storage_connection',lambda: (object(),'bucket','endpoint'))
    calls=[]
    class Catalog:
        def __init__(self,*args,scope):self.scope=scope
        def ensure(self,*args,**kwargs):
            calls.append(kwargs)
            return {'asset_id':'a'*64,'uploaded_files':0,'reused_files':1}
        def binding(self,result,mount_path,role):
            return {'mount_path':mount_path,'role':role,'read_only':True}
    monkeypatch.setattr(asset_catalog,'AssetCatalog',Catalog)
    def invoke(visibility='public',consumers=None,scope='shared',explicit=True):
        row={'source':None,'manifest':str(manifest),'mount_path':'/workspace/benchmark/qa-smoke',
            'kind':'datasets','role':'public-quality-smoke','provenance':{},'visibility':visibility}
        if explicit:row['consumers']=consumers
        spec=tmp_path/'spec.json';spec.write_text(json.dumps({'schema_version':2,'scope':scope,'assets':[row]}))
        output=tmp_path/'bindings.json'
        asset_commands.prepare_assets(SimpleNamespace(spec=spec,output=output))
        return json.loads(output.read_text())['assets'][0]
    return invoke,calls


def test_public_smoke_remains_worker_only(prepared):
    invoke,calls=prepared
    row=invoke(consumers=['worker'])
    assert row['visibility']=='public' and row['consumers']==['worker'] and row['read_only'] is True
    assert len(calls)==1 and calls[0]['local_root'] is None


def test_existing_default_public_scope_remains_both(prepared):
    invoke,_=prepared
    assert invoke(explicit=False)['consumers']==['worker','judge']


def test_private_scope_remains_judge_only(prepared):
    invoke,_=prepared
    assert invoke('private',['judge'],'judge-private')['consumers']==['judge']


@pytest.mark.parametrize('visibility,consumers,scope',[
    ('public',['judge'],'shared'),('public',[],'shared'),('public',['worker','worker'],'shared'),
    ('public',None,'shared'),('public',['worker'],'judge-private'),
    ('private',['worker'],'judge-private'),('private',['worker','judge'],'judge-private'),
    ('private',['judge'],'shared'),('unknown',['worker'],'shared')])
def test_invalid_or_private_worker_scope_rejected_before_catalog_payload(prepared,visibility,consumers,scope):
    invoke,calls=prepared
    with pytest.raises(ValueError):invoke(visibility,consumers,scope)
    assert calls==[]
