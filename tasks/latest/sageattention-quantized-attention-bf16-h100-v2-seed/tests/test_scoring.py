from __future__ import annotations

import math
import pytest
from conftest import load_module

runtime = load_module("qattn_runtime_scoring", "runtime/quant_attention_runtime.py")

def test_flop_accounting() -> None:
    assert runtime.pairs_for_length(4, False) == 16
    assert runtime.pairs_for_length(4, True) == 10
    assert runtime.effective_tflops(2, 3, 64, 10, 1e-6) == pytest.approx(0.01536)

def _metrics(**changes: float) -> dict[str, float]:
    row = {"rel_l1":0.01,"cossim":0.999,"rmse":0.001,"worst_head_rel_l1":0.01,"row_rel_l2_p999":0.005,"row_rel_l2_max":0.01}
    row.update(changes); return row

def test_realised_limits_use_floors_and_multipliers() -> None:
    assert runtime.realised_limits(_metrics()) == {
        "rel_l1_limit":0.03,"worst_head_rel_l1_limit":0.04,
        "row_rel_l2_p999_limit":0.02,"row_rel_l2_max_limit":0.05,
    }
    limits=runtime.realised_limits(_metrics(rel_l1=0.02,worst_head_rel_l1=0.02,row_rel_l2_p999=0.02,row_rel_l2_max=0.03))
    assert limits == pytest.approx({"rel_l1_limit":0.06,"worst_head_rel_l1_limit":0.06,"row_rel_l2_p999_limit":0.04,"row_rel_l2_max_limit":0.075})

@pytest.mark.parametrize(("field","value","message"),(("rel_l1",0.031,"RelL1"),("cossim",0.997,"CosSim"),("worst_head_rel_l1",0.041,"head"),("row_rel_l2_p999",0.021,"p99.9"),("row_rel_l2_max",0.051,"max")))
def test_each_gate_fails_only_on_its_inequality(field: str,value: float,message: str) -> None:
    with pytest.raises(RuntimeError,match=message): runtime.check_metric_bundle(_metrics(**{field:value}),_metrics())
    runtime.check_metric_bundle(_metrics(),_metrics())

def test_aggregate_balances_tiers_sources_and_families() -> None:
    row=lambda family,tier,model,tflops:{"scored":True,"family":family,"tier":tier,"model":model,"tflops":tflops}
    rows=[row("video","recipe","a",100.0),row("video","sweep","a",400.0),row("video","sweep","b",400.0),
          row("image","sweep","c",50.0),
          row("llm","recipe","d",25.0),row("llm","sweep","d",100.0),
          row("real","real","s1",10.0),row("real","real","s1",1000.0),row("real","real","s2",400.0)]
    result=runtime.aggregate_case_results(rows)
    # Tiers count half each however many cases they hold; real sources count equally.
    assert result["family_tflops"] == pytest.approx({"video":200.0,"image":50.0,"llm":50.0,"real":200.0})
    assert result["primary_tflops"] == pytest.approx((200.0*50.0*50.0*200.0)**0.25)
    verifier=load_module("qattn_verifier_scoring","images/judge/verify_quant_attention.py")
    for family,value in result["family_tflops"].items():
        assert verifier.family_score(rows,family) == pytest.approx(value)


def test_row_sampling_keeps_the_first_and_last_chunk_and_an_eighth() -> None:
    assert runtime.checked_chunks(65535,1)==tuple(range(0,65535,1024))
    for l in (65536,131071,262144):
        chunks=runtime.checked_chunks(l,123)
        assert chunks[0]==0 and chunks[-1]==(l-1)//1024*1024
        assert len(chunks)<=(l+1023)//1024//runtime.ROW_SAMPLE_STRIDE+3
        assert chunks!=runtime.checked_chunks(l,124)  # hidden: depends on the case seed
