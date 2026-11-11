#!/usr/bin/env python3
"""Check actual thin image source/software and exact clean software ancestry on CPU."""
import argparse
import json
import os
from pathlib import Path
from built_image_policy import validate_work_image,validate_judge_image,validate_image_ancestry,validate_software_parent_evidence,validate_rebuilt_software_parent

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work-image');p.add_argument('--judge-image');p.add_argument('--base-image',required=True)
    p.add_argument('--software-parent-receipt',type=Path);p.add_argument('--software-equivalence-receipt',type=Path)
    p.add_argument('--software-parent-verification',type=Path,help='a rebuilt parent: scripts/verify_software_parent.py --receipt')
    p.add_argument('--source-root',type=Path,required=True);p.add_argument('--stage-receipt',type=Path,required=True)
    p.add_argument('--container-engine',default='docker');p.add_argument('--image-only',action='store_true',required=True)
    p.add_argument('--prequalification',action='store_true');p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(argv)
    if not (a.work_image or a.judge_image):p.error('provide at least one immutable Work/Judge image')
    if bool(a.software_parent_verification)==bool(a.software_parent_receipt or a.software_equivalence_receipt) or (a.software_parent_receipt is None)!=(a.software_equivalence_receipt is None):
        p.error('give either both clean603d receipts or one rebuilt-parent verification')
    def provenance():
        if a.software_parent_verification:return validate_rebuilt_software_parent(a.software_parent_verification,engine=a.container_engine)
        return validate_software_parent_evidence(a.software_parent_receipt,a.software_equivalence_receipt)
    if not a.output.is_absolute() or '..' in a.output.parts or a.output.exists() or any(x.is_symlink() for x in a.output.parents):p.error('new canonical absolute receipt output required')
    record={'schema_version':2,'protocol':'minference-full-ruler-thin-image-policy-v3','status':'failed','qualified':False,'gpu_execution':False,'external_assets_authenticated':False,'roles':{}}
    try:
        record['software_parent_provenance']=provenance()
        rebuilt=record['software_parent_provenance']['clean_image_id'] if a.software_parent_verification else None
        for role,image,check in [('work',a.work_image,validate_work_image),('judge',a.judge_image,validate_judge_image)]:
            if image:
                ancestry=validate_image_ancestry(engine=a.container_engine,image_id=image,base_image_id=a.base_image,rebuilt_parent_id=rebuilt)
                if ancestry['base_layers']!=record['software_parent_provenance']['exact_equal_rootfs_layers']:raise ValueError('actual clean parent RootFS differs from retained derivative proof')
                result=check(engine=a.container_engine,image=image,image_only=True,source_root=a.source_root,stage_receipt=a.stage_receipt,prequalification=a.prequalification)
                after=validate_image_ancestry(engine=a.container_engine,image_id=image,base_image_id=a.base_image,rebuilt_parent_id=rebuilt)
                if after!=ancestry:raise ValueError('immutable image/base ancestry changed during check')
                record['roles'][role]={'policy':result,'ancestry':ancestry}
        if provenance()!=record['software_parent_provenance']:raise ValueError('software parent proof changed')
        record['status']='passed';rc=0
    except Exception as exc:record['error']={'type':type(exc).__name__,'message':str(exc)[:1024]};rc=1
    fd=os.open(a.output,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'w') as stream:json.dump(record,stream,indent=2,sort_keys=True);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    print(json.dumps({'status':record['status'],'qualified':False,'external_assets_authenticated':False,'gpu_execution':False}));return rc
if __name__=='__main__':raise SystemExit(main())
