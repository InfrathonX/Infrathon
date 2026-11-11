#!/usr/bin/env python3
"""Formal single-GPU kernel adapter; no model-answer quality dependency."""
import argparse,json,math,os,statistics,subprocess,sys
from pathlib import Path
PROTOCOL='minference-real-ruler-prefill-v3'
GATES=('suite_ok','source_ok','hardware_ok','correctness_ok','fresh_input_ok','fresh_output_ok','probe_fresh_ok','manifest_ok')
def build_structured_result(reward,detail,*,verifier_completed,verifier_returncode):
    valid=False;score=0.
    try:
        assert verifier_completed is True and verifier_returncode==0
        for v in (reward,detail):
            assert v['protocol']==PROTOCOL and v['evaluation_kind']=='kernel-only-v3' and v['case_count']==48 and v['boundary_case_count']==11
            assert v['score_enabled']==1 and v['provisional'] is False and all(v[g]==1 for g in GATES)
        rows=detail['cases'];assert len(rows)==48
        expected={(t,n,l) for t in ('niah_single_2','niah_multikey_1','vt','fwe') for n in (32768,65536,131072) for l in (1,4,15,27)}
        assert {(r['task'],r['context_budget'],r['layer']) for r in rows}==expected
        med=[]
        for r in rows:
            assert len(r['trial_ns'])==len(r['trial_seconds'])==3
            values=[]
            for n,s in zip(r['trial_ns'],r['trial_seconds']):
                assert type(n) is int and 64<=n<r['context_budget'] and type(s) in (int,float) and math.isfinite(s) and s>0
                values.append(8192*n*(n+1)/s/1e12)
            m=statistics.median(values);assert math.isclose(m,r['median_tflops'],rel_tol=1e-9);med.append(m)
        score=math.exp(statistics.mean(math.log(x) for x in med))
        assert all(math.isclose(v['raw_primary_tflops'],score,rel_tol=1e-9) and math.isclose(v['reward'],score,rel_tol=1e-9) for v in (reward,detail))
        valid=True
    except (KeyError,TypeError,ValueError,AssertionError,OverflowError):score=0.
    return {'valid':valid,'score':score,'pass_rate':float(valid),'summary':'Kernel fidelity/integrity and performance passed.' if valid else 'Kernel evaluation failed closed.',
        'details':[],'metrics':{'evaluation_kind':'kernel-only-v3','case_count':48 if valid else 0,'boundary_case_count':11 if valid else 0,'raw_primary_tflops':score,
            'verifier_completed':verifier_completed,'verifier_returncode':verifier_returncode,'score_provisional':False,'candidate_output_suppressed':True}}
def evaluate(verifier,app_dir,log_dir,timeout):
    log_dir.mkdir(parents=True,exist_ok=True);log_dir.chmod(0o700)
    for name in ('reward.json','minference_prefill_h100_seed_v3_results.json'):(log_dir/name).unlink(missing_ok=True)
    process=subprocess.Popen([sys.executable,str(verifier),'--app-dir',str(app_dir),'--log-dir',str(log_dir)],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
    try:code=process.wait(timeout=timeout);complete=True
    except subprocess.TimeoutExpired:
        import signal
        os.killpg(process.pid,signal.SIGKILL);process.wait();code=process.returncode;complete=False
    def read(name):
        try:return json.loads((log_dir/name).read_bytes())
        except (OSError,ValueError):return {}
    return build_structured_result(read('reward.json'),read('minference_prefill_h100_seed_v3_results.json'),verifier_completed=complete,verifier_returncode=code)
def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--verifier',type=Path,default=Path('/opt/minference_prefill_judge/verify_minference_prefill.py'));p.add_argument('--app-dir',type=Path,default=Path('/app'));p.add_argument('--log-dir',type=Path,default=Path('/logs/verifier'));p.add_argument('--timeout',type=int,default=7200);a=p.parse_args(argv)
    print('>>>>> Start Structured Result');print(json.dumps(evaluate(a.verifier,a.app_dir,a.log_dir,a.timeout),allow_nan=False));print('>>>>> End Structured Result')
if __name__=='__main__':main()
