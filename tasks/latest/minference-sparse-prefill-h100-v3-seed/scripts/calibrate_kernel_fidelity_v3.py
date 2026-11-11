#!/usr/bin/env python3
"""Author public-only12 prompt fidelity/resource diagnostic; never QA generation.

Uses original dense FA3 and independently chunked FP32 references. It writes
measurements only, never freezes a policy or claims production qualification.
"""
import argparse,hashlib,importlib.util,json,os,resource,sys,time
from pathlib import Path

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(name,p):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True);p.add_argument('--image',required=True);p.add_argument('--gpu',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--candidate',type=Path);p.add_argument('--performance',action='store_true');a=p.parse_args()
    assert os.geteuid()==1000 and not a.output.exists();a.output.mkdir(mode=0o700)
    sys.path[:0]=[str(a.root/'runtime'),str(a.root/'activations/recipe')]
    import torch,transformers
    import activation_archive_v3 as archive
    import minference_prefill_runtime as runtime
    import single_prompt_archive_v3 as single
    from prepare_single_prompt_v3 import TrustedPromptPreparer
    from stream_parent_v3 import release_bundle
    from kernel_fidelity_v3 import energy_stats
    import capture_activations as capture
    import ruler_v3 as recipes
    from flash_attn_interface import flash_attn_func
    recipe=json.loads((a.root/'activations/recipe/RULER_V3_RECIPE.json').read_bytes())
    prompts,manifest=recipes.load_prompts(a.inputs,expected_partition='public')
    fixed=json.loads((a.root/'qualification/public-performance-plan.json').read_bytes())
    bindings=recipes.capture_bindings(recipe,manifest,a.inputs,image_id=a.image,
        software={'torch':torch.__version__,'cuda':torch.version.cuda,'transformers':transformers.__version__})
    bindings.update(attention_backend='flash_attention_2',flash_attention_version='2.8.3',capture_helper_sha256=sha(capture.__file__))
    plan=single.on_demand_plan(fixed,bindings)
    chosen=[]
    for task in ('niah_single_2','niah_multikey_1','vt','fwe'):
        for budget in (32768,65536,131072):
            chosen.append(sorted((x for x in prompts if x['task']==task and x['length']==budget),key=lambda x:x['id'])[0])
    source=a.candidate or a.root/'oracle/candidates/kernel_v001/submission.py';candidate=load('public_calibration_seed',source)
    out={'protocol':'minference-public-kernel-fidelity-calibration-v1','qualified':False,'gpu_uuid':a.gpu,'image_id':a.image,
        'candidate_sha256':sha(source),'selection':'first lexical prompt in each4task x3budget cell, before results',
        'measurements':[],'status':'running','host_memory_limit_bytes':48*1024**3,'scratch_bytes_max':0}
    def save():
        (a.output/'report.tmp').write_text(json.dumps(out,allow_nan=False));(a.output/'report.tmp').chmod(0o600)
        (a.output/'report.tmp').replace(a.output/'report.json')
    began=time.monotonic();save()
    preparer=TrustedPromptPreparer(a.model,recipe,manifest,plan,image_id=a.image,expected_gpu_uuid=a.gpu,
        capture_helper=capture,torch=torch,transformers=transformers,purpose='public-diagnostic')
    scratch=a.output/'scratch';scratch.mkdir(mode=0o700)
    for i,prompt in enumerate(chosen):
        print('calibration',i+1,'/12',prompt['id'],flush=True)
        root=scratch/f'prompt-{i:02d}';bundle=preparer.capture(prompt,root)
        handles,identities=single.open_bundle(root,bundle,plan,archive)
        try:
            out['scratch_bytes_max']=max(out['scratch_bytes_max'],bundle['scratch_bytes'])
            for layer in (1,4,15,27):
                fid=prompt['id']+f'-L{layer}';values=single.read_checked(torch,handles[fid],identities[fid],archive)
                q,k,v=(values[n] for n in ('q','k','v'));n=q.shape[2]
                uniform,tail,rows=runtime._probe_rows(torch,n,20261007,fid,q.device)
                ref=runtime.probe_dense_reference(torch,q,k,v,128**-.5,rows)
                pred=torch.empty_like(q);candidate.minference_sparse_prefill_bf16(q,k,v,128**-.5,pred);torch.cuda.synchronize()
                dense=flash_attn_func(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),causal=True,softmax_scale=128**-.5).transpose(1,2)
                split=uniform.numel();sample=pred.index_select(2,rows);dense_sample=dense.index_select(2,rows)
                row={'prompt_id':prompt['id'],'task':prompt['task'],'budget':prompt['length'],'layer':layer,'n':n,
                    'seed_uniform':energy_stats(torch,sample[:,:,:split],ref[:,:,:split]),'seed_tail':energy_stats(torch,sample[:,:,split:],ref[:,:,split:]),
                    'dense_uniform':energy_stats(torch,dense_sample[:,:,:split],ref[:,:,:split]),'dense_tail':energy_stats(torch,dense_sample[:,:,split:],ref[:,:,split:]),
                    'scaled_dense_uniform':energy_stats(torch,0.6*dense_sample[:,:,:split],ref[:,:,:split]),
                    'peak_gpu_allocated_bytes':int(torch.cuda.max_memory_allocated()),'host_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
                if a.performance:
                    candidate.minference_sparse_prefill_bf16(q,k,v,128**-.5,pred);flash_attn_func(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),causal=True,softmax_scale=128**-.5);torch.cuda.synchronize()
                    clock=runtime.TrustedClock.capture(torch);seed_times=[];dense_times=[]
                    def dense_entry(q,k,v,scale,out):
                        torch.Tensor.copy_(out,flash_attn_func(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),causal=True,softmax_scale=scale).transpose(1,2))
                    for trial in range(3):
                        snapshot,timing=runtime._checked_call(torch,values,candidate.minference_sparse_prefill_bf16,clock,measured=True);seed_times.append(timing.seconds);del snapshot
                        snapshot,timing=runtime._checked_call(torch,values,dense_entry,clock,measured=True);dense_times.append(timing.seconds);del snapshot
                    row.update(seed_trial_seconds=seed_times,dense_trial_seconds=dense_times,same_qkv=True)
                out['measurements'].append(row);save()
                del q,k,v,values,ref,pred,dense,sample,dense_sample,uniform,tail,rows;torch.cuda.empty_cache()
        finally:
            for h in handles.values():h.close()
            release_bundle(root,bundle,identities)
    out.update(same_qkv=bool(a.performance),status='completed',wall_seconds=time.monotonic()-began,scratch_files_remaining=sum(1 for p in scratch.rglob('*') if p.is_file()))
    save();print('public calibration complete',flush=True)
if __name__=='__main__':main()
