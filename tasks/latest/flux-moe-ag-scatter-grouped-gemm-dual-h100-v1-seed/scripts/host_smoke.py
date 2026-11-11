#!/usr/bin/env python3
"""Host eight-GPU smoke: the public edges plus one public case per family, two trials."""
from __future__ import annotations
import argparse,json,os,subprocess,sys,tempfile,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PUBLIC_CASES=ROOT/"images/work/benchmark/public_cases.json"
def smoke_cases():
    cases=json.loads(PUBLIC_CASES.read_text())["cases"]; picked={}
    for case in cases:
        if case["family"]!="edge":picked.setdefault(case["family"],case)
    return [c for c in cases if c["family"]=="edge"]+list(picked.values())
def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument("--output",type=Path); p.add_argument("--candidate",type=Path,default=ROOT/"oracle/candidates/kernel_v001/submission.py"); p.add_argument("--timeout",type=int,default=1800); a=p.parse_args(argv)
    cases=smoke_cases()
    with tempfile.TemporaryDirectory(prefix="moe-ag-host-smoke-") as raw:
        work=Path(raw); output=work/"result.json"; manifest={"schema_version":2,"suite_id":"host-smoke-v2","cases":cases}; env=dict(os.environ)
        for key in tuple(env):
            if key.lower() in {"http_proxy","https_proxy","all_proxy","no_proxy"}:env.pop(key,None)
        # Host smoke is a Work-side diagnostic identity, not a Judge suite.
        command=[sys.executable,str(ROOT/"images/work/benchmark/world_worker.py"),"--candidate",str(a.candidate),"--output",str(output),"--runtime-root",str(ROOT/"runtime"),"--cache",str(work/"cache"),"--trials","2","--warmups","1","--timeout",str(a.timeout),"--expected-count",str(len(cases))]+(["--fixtures",str(ROOT/"images/work/benchmark/routing")] if (ROOT/"images/work/benchmark/routing").is_dir() else [])
        started=time.perf_counter(); completed=subprocess.run(command,input=json.dumps(manifest),text=True,env=env,timeout=a.timeout+30,check=False)
        if completed.returncode:raise SystemExit(f"host smoke worker failed: {completed.returncode}")
        result=json.loads(output.read_text()); result["wall_seconds"]=time.perf_counter()-started
    encoded=json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n"
    if a.output:a.output.write_text(encoded)
    print(json.dumps({"score_tflops":result["score_tflops"],"family_tflops":result["family_tflops"],"wall_seconds":result["wall_seconds"]},sort_keys=True)); return 0
if __name__=="__main__":raise SystemExit(main())
