#!/usr/bin/env python3
import argparse,importlib.util,json,sys
from pathlib import Path
from v3_contract import TASK_ID,PROTOCOL,resource_document,read_json
ROOT=Path(__file__).resolve().parents[1]
def validate_layout(root=ROOT,*,structural_only=False,**kwargs):
    root=Path(root);errors=[]
    try:
        task=read_json(root/'tasks'/(TASK_ID+'.json'))
        assert task['judge']['private_bootstrap']=='minference-kernel-prefill-v3'
        assert task['portable']['task_core']['definition']['evaluation_kind']=='kernel-only-v3'
        assert 'downstream_quality_scope' not in task['portable']['task_core']['definition']
        for name,delivery in [('resource-manifest.json',False),('build-resource-manifest.json',True)]:assert read_json(root/'portable'/name)==resource_document(root,delivery=delivery)
        assert (root/'judge/run_judge.py').read_bytes()==(root/'images/judge/run_judge.py').read_bytes()
        assert (root/'oracle/candidates/kernel_v001/submission.py').read_bytes()==(root/'images/work/workdir/minference_prefill_lab/candidates/kernel_v001/submission.py').read_bytes()
        sys.path.insert(0,str(root/'scripts'));from generate_protected_hashes import generate
        assert (root/'images/judge/protected_hashes.txt').read_text()==generate(root)
        assert len(read_json(root/'qualification/external-assets-contract.json')['assets'])==2
        # Measured local qualification evidence is optional: it is not part of a
        # source export, and only a host that holds its records can check it.
        evidence=root/'qualification/qualification_evidence.json'
        if not structural_only and evidence.is_file():
            from validate_formal_artifacts import validate_evidence
            errors+=validate_evidence(read_json(evidence),root)
    except (AssertionError,OSError,ValueError,KeyError) as e:errors.append('kernel layout/source binding failed: '+str(e))
    return errors
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--structural-only',action='store_true');a=p.parse_args();errors=validate_layout(structural_only=a.structural_only)
    for e in errors:print(e,file=sys.stderr)
    if not errors:print('kernel-only structure checked; no qualification claim')
    raise SystemExit(bool(errors))
