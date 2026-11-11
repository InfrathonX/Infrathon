from __future__ import annotations
import pytest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def test_admits_minimal_runtime(load_module,tmp_path):
    admission=load_module("images/judge/source_admission.py"); root=tmp_path/"runtime_v001"; root.mkdir(); (root/"__init__.py").write_text(""); (root/"runtime.py").write_text("class Runtime:\n    pass\n")
    result=admission.admit_candidate(root); assert result["version"]==1 and result["python_files"]==2
def test_rejects_wrong_name_link_and_bad_syntax(load_module,tmp_path):
    admission=load_module("images/judge/source_admission.py")
    wrong=tmp_path/"candidate"; wrong.mkdir(); (wrong/"__init__.py").write_text(""); (wrong/"runtime.py").write_text("class Runtime: pass\n")
    with pytest.raises(admission.AdmissionError): admission.admit_candidate(wrong)
    root=tmp_path/"runtime_v002"; root.mkdir(); (root/"__init__.py").write_text(""); (root/"runtime.py").write_text("class Runtime(:\n")
    with pytest.raises(admission.AdmissionError): admission.admit_candidate(root)
def test_selected_oracle_candidate_is_admitted(load_module):
    admission=load_module("images/judge/source_admission.py"); oracle=ROOT/"oracle"
    version=int((oracle/"FINAL_VERSION").read_text().strip()); selected=oracle/"candidates"/f"runtime_v{version:03d}"
    assert (oracle/"candidates/runtime_v001").is_dir() and admission.admit_candidate(selected)["version"]==version
