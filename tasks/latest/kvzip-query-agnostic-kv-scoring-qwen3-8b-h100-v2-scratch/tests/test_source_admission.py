from __future__ import annotations
from pathlib import Path
import pytest
from source_admission import validate_submission

BASE = '''import torch
class KVScorer:
 def __init__(self, spec, assets): self.spec=spec; self.assets=assets
 def score(self, layer_idx, hidden_states, queries, keys, values, position_embeddings, weights, ctx_range):
  start,end=ctx_range
  return keys[:,:,start:end].float().square().sum(dim=-1)
'''


def make_root(tmp_path: Path, source: str = BASE, asset: bytes | None = None):
    root = tmp_path / "lab"; d = root / "candidates" / "scorer_v001"; d.mkdir(parents=True)
    (root / "FINAL_VERSION").write_text("1\n"); (root / "notes.md").write_text("notes\n")
    (d / "submission.py").write_text(source)
    if asset is not None: (d / "assets.pt").write_bytes(asset)
    return root


def admit(tmp_path, source=BASE, asset=None, workspace_mode=False):
    return validate_submission(make_root(tmp_path, source, asset), workspace_mode=workspace_mode)


@pytest.mark.parametrize("name,fragment", [
 ("open", "open('x')"), ("exec", "exec('x')"), ("eval", "eval('1')"),
 ("compile", "compile('x','x','exec')"), ("dynamic_import", "__import__('os')"),
 ("getattr", "getattr(keys,'shape')"), ("setattr", "setattr(keys,'x',1)"),
 ("globals", "globals()"), ("locals", "locals()"), ("vars", "vars(keys)"),
 ("dunder", "keys.__class__"), ("file_literal", "'/opt/models/x'"),
 ("runtime_literal", "'kv_scoring_runtime'"), ("worker_literal", "'candidate_worker'"),
 ("logs_literal", "'/logs/verifier/x'"), ("torch_load", "torch.load('x')"),
 ("torch_save", "torch.save(keys,'x')"), ("torch_compile", "torch.compile(lambda x:x)"),
 ("manual_seed", "torch.manual_seed(1)"), ("cuda_seed", "torch.cuda.manual_seed(1)"),
 ("cuda_stream", "torch.cuda.Stream()"), ("distributed", "torch.distributed.init_process_group()"),
 ("hub", "torch.hub.load('x','y')"), ("cpp", "torch.utils.cpp_extension.load('x')"),
 ("ops_library", "torch.ops.load_library('x')"), ("input_zero", "keys.zero_()"),
 ("torch_from_file", "torch.from_file('x')"), ("input_copy", "values.copy_(keys)"),
 ("set_device", "torch.cuda.set_device(1)"), ("device_context", "torch.cuda.device(1)"),
 ("reset_peak", "torch.cuda.reset_peak_memory_stats(1)"),
 ("reset_accumulated", "torch.cuda.reset_accumulated_memory_stats()"),
 ("global", "global BAD"), ("module_assign", "torch.foo = 1"),
])
def test_rejected_routes(tmp_path, name, fragment):
    source = BASE.replace("  start,end", f"  {fragment}\n  start,end")
    assert not admit(tmp_path, source)["ok"], name


@pytest.mark.parametrize("source", [
 BASE,
 BASE.replace("keys[:,:,start:end].float().square().sum(dim=-1)", "values[:,:,start:end].float().norm(dim=-1)"),
 BASE.replace("import torch", "import torch\nimport torch.nn.functional as F").replace("keys[:,:,start:end].float().square().sum(dim=-1)", "F.avg_pool1d(keys[:,:,start:end].float().norm(dim=-1), 5, 1, 2)"),
 BASE.replace("self.assets=assets", "self.assets=assets; self.scale=1.0"),
 BASE.replace("import torch", "import torch\nimport triton\nimport triton.language as tl"),
 BASE.replace("square().sum(dim=-1)", "norm(dim=-1)"),
])
def test_accepted_variants(tmp_path, source):
    result = admit(tmp_path, source)
    assert result["ok"], result["errors"]


def test_assets_ref_resolves_one_hop(tmp_path):
    root = make_root(tmp_path, asset=b"asset")
    d = root / "candidates" / "scorer_v002"; d.mkdir()
    (d / "submission.py").write_text(BASE); (d / "assets.ref").write_text("scorer_v001\n")
    (root / "FINAL_VERSION").write_text("2\n")
    result = validate_submission(root)
    assert result["ok"], result["errors"]
    assert result["selected_assets_path"].endswith("scorer_v001/assets.pt")


@pytest.mark.parametrize("kind", ["self", "missing", "chain", "both"])
def test_assets_ref_rejects_unsafe_forms(tmp_path, kind):
    root = make_root(tmp_path, asset=b"asset")
    d = root / "candidates" / "scorer_v002"; d.mkdir(); (d / "submission.py").write_text(BASE)
    (root / "FINAL_VERSION").write_text("2\n")
    if kind == "self": (d / "assets.ref").write_text("scorer_v002\n")
    elif kind == "missing": (d / "assets.ref").write_text("scorer_v003\n")
    elif kind == "both":
        (d / "assets.ref").write_text("scorer_v001\n"); (d / "assets.pt").write_bytes(b"x")
    else:
        (d / "assets.ref").write_text("scorer_v003\n")
        e = root / "candidates" / "scorer_v003"; e.mkdir(); (e / "submission.py").write_text(BASE); (e / "assets.ref").write_text("scorer_v001\n")
    assert not validate_submission(root)["ok"]


def test_layout_rejects_extra_file(tmp_path):
    root = make_root(tmp_path); (root / "candidates/scorer_v001/payload.bin").write_bytes(b"x")
    assert not validate_submission(root)["ok"]


def test_empty_selected_source_rejected(tmp_path):
    assert not admit(tmp_path, "")["ok"]


def test_workspace_mode_ignores_generated_python_caches(tmp_path):
    root = make_root(tmp_path); d = root / "candidates/scorer_v001"
    (root / "__pycache__").mkdir(); (root / "root.pyc").write_bytes(b"x")
    (d / "__pycache__").mkdir(); (d / "submission.pyc").write_bytes(b"x")
    assert validate_submission(root, workspace_mode=True)["ok"]


def test_asset_size_cap_accepts_100_mib_and_rejects_129_mib(tmp_path):
    accepted = make_root(tmp_path / "accepted")
    with (accepted / "candidates/scorer_v001/assets.pt").open("wb") as handle:
        handle.truncate(100 * 1024 * 1024)
    assert validate_submission(accepted)["ok"]

    rejected = make_root(tmp_path / "rejected")
    with (rejected / "candidates/scorer_v001/assets.pt").open("wb") as handle:
        handle.truncate(129 * 1024 * 1024)
    result = validate_submission(rejected)
    assert not result["ok"]
    assert any("file too large" in error for error in result["errors"])


@pytest.mark.parametrize("header,call", [("import torch as t", "t.from_file('x')"), ("from torch import load", "load('x')"), ("from torch import manual_seed as seed", "seed(1)")])
def test_rejects_aliased_torch_escape(tmp_path, header, call):
    source = BASE.replace("import torch", header).replace("  start,end", f"  {call}\n  start,end")
    assert not admit(tmp_path, source)["ok"]


@pytest.mark.parametrize("fragment", ["f = torch.load\n  f('x')", "f = torch.from_file\n  [f][0]('x')", "f = torch\n  f.save(keys, 'x')"])
def test_rejects_assignment_alias_propagation(tmp_path, fragment):
    source = BASE.replace("  start,end", f"  {fragment}\n  start,end")
    assert not admit(tmp_path, source)["ok"]
