"""Deferred output materialisation and mutable clock regressions."""
import time
import torch
import pytest

pytest.importorskip("safetensors", reason="the timing tests load the Work runtime, which needs safetensors")
from runtime import evaluate  # noqa: E402
from runtime.api import EngineContract, GenerationSpec, GenerationResult

@pytest.mark.parametrize('scaled_clock', [False, True])
def test_materialisation_is_timed(monkeypatch, scaled_clock):
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    monkeypatch.setattr(torch.cuda, 'reset_peak_memory_stats', lambda: None)
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda: 0)
    original_clock = time.perf_counter
    if scaled_clock:
        monkeypatch.setattr(time, 'perf_counter', lambda: original_clock() * .01)
    class Lazy(GenerationResult):
        def __init__(self):
            self.actual = None
            self.trace = {}
        @property
        def token_ids(self):
            if self.actual is None:
                time.sleep(.025)
                self.actual = [torch.tensor([1,2],dtype=torch.int64)]
            return self.actual
    class Engine:
        def generate(self, prompts, spec):
            return Lazy()
    prompt = torch.tensor([1],dtype=torch.int64)
    contract = EngineContract('t','r','d','r',device='cpu')
    result = evaluate.run_cell(Engine(), [prompt], GenerationSpec(1), contract, prompt)
    assert result.measurements[0].wall_s >= .025
