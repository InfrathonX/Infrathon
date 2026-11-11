#!/usr/bin/env python3
"""Root author entry: fresh final32K bundle, then current12 finite controls.

Caller owns and monitors the actual Judge container/GPU; this script neither
creates containers nor schedules/cleans GPUs. Installed runtime/recipe bytes
are used unchanged. A failure retains at most one bounded scratch bundle.
"""
from __future__ import annotations
import argparse,hashlib,importlib.util,json,os,stat,sys,time
from pathlib import Path

def require(test,message):
    if not test:raise ValueError(message)

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module

def validated_args(args):
    require(args.output.is_absolute() and '..' not in args.output.parts and not args.output.exists()
        and not any(p.is_symlink() for p in (args.output,*args.output.parents)),'fresh absolute non-symlink output required')
    require(args.output.parent.is_dir(),'existing private output parent required')
    info=args.output.parent.stat()
    require(info.st_uid==0 and stat.S_IMODE(info.st_mode)==0o700,'root private0700 output parent required')
    require(os.geteuid()==0,'owned Judge root execution required')
    require(args.trusted_root.is_absolute() and not any(p.is_symlink() for p in (args.trusted_root,*args.trusted_root.parents)),'installed trusted root must be canonical')

def execute(args):
    validated_args(args)
    require(hashlib.sha256(args.runner.read_bytes()).hexdigest()==args.runner_sha256,'current literal battery runner differs')
    runner=load('current32_author_battery',args.runner)
    modules=runner.trusted_modules(args.trusted_root)
    inspect_raw=runner.read(args.inspection,private=True)
    cid,pool=runner.validate_inspection(runner.parse(inspect_raw),image_id=args.image_id,gpu_uuid=args.gpu_uuid)
    policy,policy_sha=modules['runtime'].load_numerical_policy(args.numerical_policy)
    require(policy.get('status')=='frozen','actual frozen policy required before capture')
    seed_raw=runner.read(args.seed_source);builder_raw=runner.read(args.builder_source)
    battery=runner.parse(runner.read(args.battery/'cases.json'))
    require(battery.get('builder_sha256')==runner.digest(builder_raw),'actual builder identity differs')
    builder=runner.builder_from_raw(args.builder_source,builder_raw)
    expected={label:source.encode() for label,source,*_ in builder.sources(seed_raw.decode())}
    runner.validate_battery(battery,args.battery,modules['admission'],runner.digest(seed_raw),expected_sources=expected)
    sys.path[:0]=[str(args.trusted_root/'runtime'),str(args.trusted_root/'activations/recipe')]
    import ruler_v3, capture_activations as capture
    from on_demand_suite_v3 import load_fixed_inputs
    from prepare_single_prompt_v3 import TrustedPromptPreparer,write_private_json
    from stream_parent_v3 import release_bundle
    single=modules['single'];recipe_path=args.trusted_root/'activations/recipe/RULER_V3_RECIPE.json'
    recipe=runner.parse(runner.read(recipe_path));fixed=runner.parse(runner.read(args.fixed_plan,private=True))
    single.validate_plan(fixed)
    require(fixed['partition']=='final','current genuine final fixed-input plan required')
    prompts,manifest=load_fixed_inputs(args.prompts,fixed,recipe_path,ruler_v3)
    selected=sorted([p for p in prompts if p['length']==32768],key=lambda p:p['id'])[0]
    args.output.mkdir(mode=0o700);args.output.chmod(0o700)
    receipt={'schema_version':1,'protocol':'minference-kernel-author-control-capture-v1',
        'status':'preparing','qualified':False,'candidate_execution':False,'capture_then_controls':True,
        'image_id':args.image_id,'gpu_uuid':args.gpu_uuid,'actual_container_id':cid,'pool_gpu_uuids':pool,
        'inspection_sha256':runner.digest(inspect_raw),'source_resources_sha256':runner.digest(runner.read(args.source_resources)),
        'author_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'runner_sha256':args.runner_sha256,'seed_sha256':runner.digest(seed_raw),'numerical_policy_sha256':policy_sha,
        'fixed_input_plan_sha256':runner.digest(runner.read(args.fixed_plan,private=True)),
        'ownership_scope':'Caller independently attests actual container identity/source/mounts and monitors physical GPU; this script does not lease or clean a container.'}
    def persist():
        raw=json.dumps(receipt,sort_keys=True,indent=2).encode()+b'\n'
        path=args.output/'capture-execution.json';temporary=args.output/'capture-execution.tmp'
        fd=os.open(temporary,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
        with os.fdopen(fd,'wb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,path)
    persist();started=time.monotonic()
    try:
        import torch,transformers
        require(torch.__version__=='2.8.0+cu129' and transformers.__version__=='4.57.1','current software epoch differs')
        bindings=ruler_v3.capture_bindings(recipe,manifest,args.prompts,image_id=args.image_id,
            software={'torch':torch.__version__,'cuda':torch.version.cuda,'transformers':transformers.__version__})
        bindings.update(attention_backend='flash_attention_2',flash_attention_version='2.8.3',
            capture_helper_sha256=modules['sources']['activations/recipe/capture_activations.py'],
            on_demand_sources={name:modules['sources']['runtime/'+name] for name in runner.RUNTIME_FILES},
            on_demand_suite_sha256=modules['sources']['runtime/on_demand_suite_v3.py'])
        plan=single.on_demand_plan(fixed,bindings)
        plan_path=args.output/'execution-plan.json';write_private_json(plan_path,plan)
        preparer=TrustedPromptPreparer(args.model,recipe,manifest,plan,image_id=args.image_id,
            expected_gpu_uuid=args.gpu_uuid,capture_helper=capture,torch=torch,transformers=transformers,purpose='private-judge')
        scratch=args.output/'fresh-bundle';bundle=preparer.capture(selected,scratch)
        runner.validate_capture(bundle,plan,single,modules['sources'],image_id=args.image_id,gpu_uuid=args.gpu_uuid,recipe=recipe)
        receipt.update(status='captured',prompt_id=selected['id'],actual_n=selected['actual_input_length'],
            scratch_bytes=bundle['scratch_bytes'],execution_plan_sha256=single.canonical_sha(plan))
        persist()
        argv=['--trusted-root',str(args.trusted_root),'--battery',str(args.battery),
            '--seed-source',str(args.seed_source),'--builder-source',str(args.builder_source),
            '--bundle-root',str(scratch),'--plan',str(plan_path),'--numerical-policy',str(args.numerical_policy),
            '--inspection',str(args.inspection),'--source-resources',str(args.source_resources),
            '--image-id',args.image_id,'--gpu-uuid',args.gpu_uuid,'--output',str(args.output/'controls')]
        rc=runner.main(argv)
        receipt.update(returncode=rc,status='passed' if rc==0 else 'failed',controls_summary_sha256=runner.digest(runner.read(args.output/'controls/summary.json',private=True)))
        if rc==0:
            # Authenticate current payloads/identities before releasing exactly
            # this newly generated bundle. Original metadata stays in controls.
            handles,identities=single.open_bundle(scratch,bundle,plan,modules['archive'])
            for handle in handles.values():handle.close()
            release_bundle(scratch,bundle,identities);receipt['scratch_released']=True
        return rc
    except BaseException as error:
        receipt.update(status='failed',error_type=type(error).__name__,private_message=str(error));raise
    finally:
        receipt['wall_seconds']=time.monotonic()-started;persist()

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['trusted-root','runner','battery','seed-source','builder-source','source-resources','fixed-plan','prompts','model','numerical-policy','inspection','output']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--image-id',required=True);p.add_argument('--gpu-uuid',required=True);p.add_argument('--runner-sha256',required=True)
    a=p.parse_args(argv)
    try:return execute(a)
    except (OSError,ValueError,KeyError,TypeError,RuntimeError) as error:
        print('current12 control capture failed closed ('+type(error).__name__+')',file=sys.stderr);return 1
if __name__=='__main__':raise SystemExit(main())
