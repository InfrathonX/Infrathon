#!/usr/bin/env python3
"""Authorized Task-5 host smoke: eight edges plus five anchors, 2/1."""
from __future__ import annotations
import argparse,importlib.util,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def load():
    path=ROOT/"runtime/quant_attention_runtime.py"; spec=importlib.util.spec_from_file_location("qattn_host_smoke_runtime",path); module=importlib.util.module_from_spec(spec); sys.modules[spec.name]=module; spec.loader.exec_module(module); return module
def main():
    p=argparse.ArgumentParser(); p.add_argument("--output",type=Path); a=p.parse_args(); r=load()
    cases=[]; seeds={"D1":17011,"D2":27017,"D3":37021,"L1":47023,"L2":57037}
    for cell in r.CELLS: cases.append({"case_id":f"anchor-{cell.lower()}","family":r.CELL_GEOMETRY[cell][0],"cell":cell,"l":r.CELL_L_SLOTS[cell][0],"seed":seeds[cell]})
    os.environ["SAGE_QATTN_FIXTURE_DIR"]=str(ROOT/"fixtures"); candidate=ROOT/"oracle/candidates/kernel_v001/submission.py"; started=time.perf_counter()
    result=r.run_suite(candidate,cases,trials=2,warmups=1,require_hardware=True); result["wall_seconds"]=time.perf_counter()-started
    for group in (result["edge_cases"],result["cases"]):
        for row in group:
            row["rel_l1_ratio"]=row["rel_l1"]/row["reference_rel_l1"]
            row["head_ratio"]=row["worst_head_rel_l1"]/row["reference_worst_head_rel_l1"]
            row["p999_ratio"]=row["row_rel_l2_p999"]/row["reference_row_rel_l2_p999"]
            row["rowmax_ratio"]=row["row_rel_l2_max"]/row["reference_row_rel_l2_max"]
    encoded=json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n"
    if a.output:a.output.write_text(encoded)
    print(encoded,end=""); return 0
if __name__=="__main__": raise SystemExit(main())
