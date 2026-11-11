#!/usr/bin/env python3
import argparse
from pathlib import Path
from v3_contract import TASK_ID,PROTOCOL,read_json,resource_document,serialize,digest,grid,public_grid
ROOT=Path(__file__).resolve().parents[1]
def refreshed(root):
    root=Path(root);writes={}
    for role in ('public','interactive','final'):
        p=root/'portable'/(role+'-suite-manifest.json');v=read_json(p)
        v.update(contract_protocol=PROTOCOL,evaluation_kind='kernel-only-v3',quality_scope=None,gpu_count=1)
        v['cases']=public_grid() if role=='public' else grid('final');v['selection_semantics']='numerical fidelity/integrity then single-GPU throughput; identical submit/final'
        writes[p]=serialize(v)
    for name,delivery in [('resource-manifest.json',False),('build-resource-manifest.json',True)]:
        document=resource_document(root,delivery=delivery)
        import hashlib
        for row in document['resources']:
            path=root/row['path']
            if path in writes:row['sha256']='sha256:'+hashlib.sha256(writes[path].encode()).hexdigest()
        writes[root/'portable'/name]=serialize(document)
    task_path=root/'tasks'/(TASK_ID+'.json');task=read_json(task_path)
    import hashlib
    task['portable']['task_core']['resource_manifest_sha256']='sha256:'+hashlib.sha256(writes[root/'portable/resource-manifest.json'].encode()).hexdigest()
    task['delivery']['build_resource_manifest_sha256']='sha256:'+hashlib.sha256(writes[root/'portable/build-resource-manifest.json'].encode()).hexdigest()
    task['scientific_artifacts']={'schema_version':1,'contract_protocol':PROTOCOL,'manifest_sha256':'sha256:'+digest(root/'portable/scientific-artifacts.json')}
    for suite in task['portable']['evaluation_suites'].values():
        for role in ('public','interactive','final'):suite['roles'][role]={'manifest_sha256':'sha256:'+hashlib.sha256(writes[root/'portable'/(role+'-suite-manifest.json')].encode()).hexdigest(),'case_count':48}
        for key in ('oracle','selection'):suite[key]={'protocol_id':'kernel-fidelity-then-throughput-v3','spec_sha256':'sha256:'+digest(root/'portable'/(key+'-protocol.json'))}
    writes[task_path]=serialize(task)
    return writes
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');a=p.parse_args();w=refreshed(ROOT)
    drift=[str(k.relative_to(ROOT)) for k,v in w.items() if k.read_text()!=v]
    if a.check:
        print('drift:',drift) if drift else None
        raise SystemExit(bool(drift))
    for k,v in w.items():k.write_text(v)
    print('kernel-only resource/task bindings refreshed; no qualification claim')
