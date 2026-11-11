#!/usr/bin/env python3
"""Validate measured kernel-only evidence; retired full-QA schema is rejected."""
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
REQUIRED={'public','interactive','final','numeric_calibration','dense_baseline','gpu_controls','llm_review','cpu_checks'}
def validate_evidence(v,root=ROOT,**kwargs):
    errors=[]
    try:
        assert v['protocol']=='minference-kernel-only-qualification-v1' and v['status']=='qualified-local-unpublished' and v['published'] is False
        assert v['source_resources_sha256']==hashlib.sha256((Path(root)/'portable/resource-manifest.json').read_bytes()).hexdigest()
        assert set(v['records'])==REQUIRED
        data={}
        for name,r in v['records'].items():
            p=Path(r['path']);assert p.is_absolute() and not any(x.is_symlink() for x in (p,*p.parents));raw=p.read_bytes()
            assert len(raw)==r['size'] and hashlib.sha256(raw).hexdigest()==r['sha256'];data[name]=json.loads(raw)
        starter=hashlib.sha256((Path(root)/'oracle/candidates/kernel_v001/submission.py').read_bytes()).hexdigest()
        assert data['public']['completed'] is True and data['public']['valid'] is True
        assert data['public']['image_id']==v['images']['work'] and data['public']['gpu_uuid']==v['gpu_uuid']
        assert data['public']['candidate_sha256']==starter
        for phase in ('interactive','final'):
            assert data[phase]['valid'] is True and data[phase]['phase']==phase
            assert data[phase]['archive_sha256']==v['archive_sha256'][phase]
            assert data[phase]['gpu_uuid']==v['gpu_uuid'] and data[phase]['image_id']==v['images']['judge']
            report=data[phase]['report'];assert report['valid'] is True and report['score']>0
            assert report['metrics']['evaluation_kind']=='kernel-only-v3'
        assert data['numeric_calibration']['status']=='completed' and len(data['numeric_calibration']['measurements'])==48
        assert data['dense_baseline']['status']=='completed' and data['dense_baseline']['same_qkv'] is True
        assert data['numeric_calibration']['candidate_sha256']==starter and data['dense_baseline']['candidate_sha256']==starter
        assert len(data['dense_baseline']['measurements'])==48 and all(r['same_qkv'] is True for r in data['dense_baseline']['measurements'])
        assert data['gpu_controls']['status']=='passed' and data['gpu_controls']['seed_sha256']==starter
        assert data['gpu_controls']['image_id']==v['images']['judge'] and data['gpu_controls']['gpu_uuid']==v['gpu_uuid']
        assert data['gpu_controls']['coverage']['complete_control_matrix'] is True and all(c['matches_expected_runtime'] for c in data['gpu_controls']['controls'])
        assert len(data['llm_review']['reviews'])==9 and all(r['matches_expected'] is True for r in data['llm_review']['reviews'])
        assert next(r for r in data['llm_review']['reviews'] if r['label']=='honest-seed')['candidate_source_sha256']==starter
        assert data['llm_review']['policy_sha256']==hashlib.sha256((Path(root)/'llm_judge/POLICY.md').read_bytes()).hexdigest()
        assert data['llm_review']['brief_sha256']==hashlib.sha256((Path(root)/'images/work/workdir/minference_prefill_lab/README_AGENT.md').read_bytes()).hexdigest()
        assert data['cpu_checks']['status']=='passed'
        policy=json.loads((Path(root)/'qualification/numerical_policy.json').read_bytes());assert policy['status']=='frozen'
        sys.path.insert(0,str(Path(root)/'runtime'))
        from kernel_fidelity_v3 import passes
        assert all(passes(r['seed_uniform'],r['seed_tail'],policy['limits']['default']) for r in data['numeric_calibration']['measurements'])
        assert v['numeric_policy_sha256']==hashlib.sha256((Path(root)/'qualification/numerical_policy.json').read_bytes()).hexdigest()
    except (AssertionError,KeyError,OSError,ValueError,TypeError) as e:errors.append('literal current kernel qualification incomplete: '+str(e))
    return errors
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,default=ROOT/'qualification/qualification_evidence.json');a=p.parse_args()
    try:errors=validate_evidence(json.loads(a.evidence.read_bytes()))
    except (OSError,ValueError) as e:errors=[str(e)]
    for x in errors:print(x,file=sys.stderr)
    raise SystemExit(bool(errors))
