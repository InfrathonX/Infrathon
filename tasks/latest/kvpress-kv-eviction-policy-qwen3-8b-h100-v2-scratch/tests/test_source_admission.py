from __future__ import annotations
from pathlib import Path
import pytest
from source_admission import validate_submission

BASE = '''import torch
class EvictionPolicy:
 def __init__(self, spec, compression_ratio, assets): self.spec=spec
 def layer_budgets(self, seq_len, n_kept_total): return None
 def select(self, layer_idx, hidden_states, queries, keys, values, position_embeddings, weights, n_kept):
  return torch.topk(keys.float().square().sum(dim=-1), n_kept, dim=-1).indices
'''

def admit(tmp_path: Path, source: str = BASE, asset: bytes | None = None, *, workspace_mode: bool = False):
    root = tmp_path / "lab"; d = root / "candidates" / "policy_v001"; d.mkdir(parents=True)
    (root / "FINAL_VERSION").write_text("1\n"); (root / "notes.md").write_text("notes\n"); (d / "submission.py").write_text(source)
    if asset is not None: (d / "assets.pt").write_bytes(asset)
    return validate_submission(root, workspace_mode=workspace_mode)

@pytest.mark.parametrize("name,fragment", [
 ("open", "open('x')"), ("exec", "exec('x')"), ("eval", "eval('1')"),
 ("compile", "compile('x','x','exec')"), ("dynamic_import", "__import__('os')"),
 ("getattr", "getattr(keys,'shape')"), ("setattr", "setattr(keys,'x',1)"),
 ("globals", "globals()"), ("locals", "locals()"), ("vars", "vars(keys)"),
 ("dunder", "keys.__class__"), ("file_literal", "'/opt/models/x'"),
 ("runtime_literal", "'kv_eviction_runtime'"), ("worker_literal", "'candidate_worker'"),
 ("logs_literal", "'/logs/verifier/x'"), ("torch_load", "torch.load('x')"),
 ("torch_save", "torch.save(keys,'x')"), ("compile_torch", "torch.compile(lambda x:x)"),
 ("manual_seed", "torch.manual_seed(1)"), ("cuda_seed", "torch.cuda.manual_seed(1)"),
 ("cuda_stream", "torch.cuda.Stream()"), ("distributed", "torch.distributed.init_process_group()"),
 ("hub", "torch.hub.load('x','y')"), ("cpp", "torch.utils.cpp_extension.load('x')"),
 ("ops_library", "torch.ops.load_library('x')"), ("input_zero", "keys.zero_()"),
 ("torch_from_file", "torch.from_file('x')"),
 ("set_device", "torch.cuda.set_device(1)"), ("device_context", "torch.cuda.device(1)"),
 ("reset_peak", "torch.cuda.reset_peak_memory_stats(1)"),
 ("reset_accumulated", "torch.cuda.reset_accumulated_memory_stats()"),
 ("input_copy", "values.copy_(keys)"), ("global", "global BAD"),
 ("nonlocal", "nonlocal bad"), ("module_assign", "torch.foo = 1"),
])
def test_rejected_routes(tmp_path, name, fragment):
    if fragment.startswith("nonlocal"):
        fragment = "def outer():\n   bad=1\n   def inner():\n    nonlocal bad"
    source = BASE.replace("  return torch.topk", f"  {fragment}\n  return torch.topk")
    assert not admit(tmp_path, source)["ok"], name

@pytest.mark.parametrize("source", [
 BASE,
 BASE.replace("keys.float().square().sum(dim=-1)", "values.float().norm(dim=-1)"),
 BASE.replace("return None", "return [n_kept_total // 36] * 36"),
 BASE.replace("import torch", "import torch\nimport torch.nn.functional as F").replace("keys.float().square().sum(dim=-1)", "F.avg_pool1d(keys.float().norm(dim=-1), 5, 1, 2)"),
 BASE.replace("self.spec=spec", "self.spec=spec; self.assets=assets"),
 BASE.replace("import torch", "import torch\nimport triton\nimport triton.language as tl"),
])
def test_accepted_variants(tmp_path, source):
    result = admit(tmp_path, source)
    assert result["ok"], result["errors"]

def test_layout_rejects_extra_file(tmp_path):
    result = admit(tmp_path); p = Path(result["selected_path"]).parent; (p / "payload.bin").write_bytes(b"x")
    assert not validate_submission(p.parents[1])["ok"]

def test_workspace_mode_ignores_generated_python_caches(tmp_path):
    result = admit(tmp_path)
    root = Path(result["selected_path"]).parents[2]
    policy = Path(result["selected_path"]).parent
    (root / "__pycache__").mkdir(); (root / "__pycache__/root.cpython-311.pyc").write_bytes(b"cache")
    (root / "root.cpython-311.pyc").write_bytes(b"cache")
    (policy / "__pycache__").mkdir(); (policy / "__pycache__/submission.cpython-311.pyc").write_bytes(b"cache")
    (policy / "submission.cpython-311.pyc").write_bytes(b"cache")
    workspace = validate_submission(root, workspace_mode=True)
    assert workspace["ok"], workspace["errors"]

def test_judge_mode_rejects_generated_python_caches(tmp_path):
    result = admit(tmp_path)
    root = Path(result["selected_path"]).parents[2]
    policy = Path(result["selected_path"]).parent
    (root / "__pycache__").mkdir(); (root / "root.cpython-311.pyc").write_bytes(b"cache")
    (policy / "__pycache__").mkdir(); (policy / "submission.cpython-311.pyc").write_bytes(b"cache")
    judge = validate_submission(root)
    assert not judge["ok"]
    assert any("unexpected root entry" in error for error in judge["errors"])
    assert any("unexpected candidate entry" in error for error in judge["errors"])

def test_candidate_override(tmp_path):
    result = admit(tmp_path)
    assert validate_submission(Path(result["selected_path"]).parents[2], 1)["ok"]

@pytest.mark.parametrize("header,call", [("import torch as t", "t.from_file('x')"), ("from torch import load", "load('x')"), ("from torch import manual_seed as seed", "seed(1)")])
def test_rejects_aliased_torch_escape(tmp_path, header, call):
    source = BASE.replace("import torch", header).replace("  return torch.topk", f"  {call}\n  return torch.topk")
    assert not admit(tmp_path, source)["ok"]

@pytest.mark.parametrize("fragment", ["f = torch.load\n  f('x')", "f = torch.from_file\n  [f][0]('x')", "f = torch\n  f.save(keys, 'x')"])
def test_rejects_assignment_alias_propagation(tmp_path, fragment):
    source = BASE.replace("  return torch.topk", f"  {fragment}\n  return torch.topk")
    assert not admit(tmp_path, source)["ok"]

def test_selected_oracle_archive_is_admitted(tmp_path: Path) -> None:
    """The Oracle archive (FINAL_VERSION, notes.md, the selected policy) passes Judge admission."""
    oracle = Path(__file__).resolve().parents[1] / "oracle"
    version = int((oracle / "FINAL_VERSION").read_text().strip())
    selected = f"candidates/policy_v{version:03d}/submission.py"
    root = tmp_path / "oracle"; (root / selected).parent.mkdir(parents=True)
    for name in ("FINAL_VERSION", "notes.md", selected): (root / name).write_bytes((oracle / name).read_bytes())
    result = validate_submission(root)
    assert result["ok"], result["errors"]
    assert result["final_version"] == version and not result["warnings"]
