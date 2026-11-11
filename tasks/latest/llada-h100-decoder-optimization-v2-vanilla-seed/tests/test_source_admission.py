from __future__ import annotations

from conftest import load_module


def admission(task_root):
    return load_module(
        "admission_under_test",
        task_root / "images/judge/verifier/source_admission.py",
    )


def create_lab(tmp_path, source="import torch\nfrom runtime.api import GenerationResult\n"):
    lab = tmp_path / "lab"
    candidate = lab / "solutions/decoder_v001/candidate.py"
    candidate.parent.mkdir(parents=True)
    candidate.write_text(source)
    (lab / "FINAL_VERSION").write_text("1\n")
    (lab / "notes.md").write_text("notes\n")
    return lab


def test_accepts_narrow_python_source(task_root, tmp_path):
    module = admission(task_root)
    result = module.admit(create_lab(tmp_path))
    assert result["final_version"] == 1


def test_accepts_starter_feature_probe(task_root, tmp_path):
    module = admission(task_root)
    starter = (
        task_root
        / "images/work/workdir/decoder_lab/solutions/decoder_v001/candidate.py"
    ).read_text()
    result = module.admit(create_lab(tmp_path / "starter", starter))
    assert result["final_version"] == 1

    probe_source = """\
def probe(model):
    allow_cpu = getattr(model, "mlsysforge_allow_cpu", False)
    return allow_cpu
"""
    result = module.admit(create_lab(tmp_path / "probe", probe_source))
    assert result["final_version"] == 1


def test_rejects_filesystem_import_and_symlink(task_root, tmp_path):
    module = admission(task_root)
    lab = create_lab(tmp_path, "import os\n")
    try:
        module.admit(lab)
    except module.AdmissionError:
        pass
    else:
        raise AssertionError("forbidden import was accepted")
    (lab / "solutions/decoder_v001/candidate.py").unlink()
    (lab / "solutions/decoder_v001/candidate.py").symlink_to(lab / "notes.md")
    try:
        module.admit(lab)
    except module.AdmissionError:
        pass
    else:
        raise AssertionError("symlink was accepted")


def test_rejects_introspection_and_untrusted_runtime_modules(task_root, tmp_path):
    module = admission(task_root)
    bad_sources = (
        "import time\n",
        "import operator\n",
        "from runtime.loader import sampled_weight_fingerprint\n",
        "def build(model, tokenizer):\n    return model.forward.__globals__\n",
        "def build(model, tokenizer):\n    model.forward = tokenizer\n",
        "g = getattr\nimp = __import__\nreader = open\n",
        "def build(model, tokenizer):\n    return getattr(model, 'forward')\n",
        "def build(model, tokenizer):\n    return getattr(model, 'past_' + 'key_values')\n",
        "def build(model, tokenizer):\n    return getattr(model, 'past_key_values', None)\n",
        "def build(model, tokenizer):\n    return getattr(model, 'mlsysforge_disable_compile', False)\n",
        (
            "import operator\n"
            "def build(model, tokenizer):\n"
            "    scope = operator.attrgetter('__globals__')(lambda: 0)\n"
            "    return operator.getitem(scope, '__builtins__')\n"
        ),
    )
    for index, source in enumerate(bad_sources):
        lab = create_lab(tmp_path / str(index), source)
        try:
            module.admit(lab)
        except module.AdmissionError:
            continue
        raise AssertionError(f"unsafe source {index} was accepted")


def test_unselected_compile_experiment_does_not_poison_final(task_root, tmp_path):
    module = admission(task_root)
    lab = create_lab(tmp_path)
    experiment = lab / "solutions/decoder_v002/candidate.py"
    experiment.parent.mkdir(parents=True)
    experiment.write_text(
        "def build(model, tokenizer):\n"
        "    compiled_forward = model.forward\n"
        "    model.forward = compiled_forward\n"
        "    return model\n"
    )
    result = module.admit(lab)
    assert result["final_version"] == 1
    assert result["versions"] == [1, 2]


def test_rejects_payload_files_and_oversized_notes(task_root, tmp_path):
    module = admission(task_root)
    lab = create_lab(tmp_path)
    (lab / "solutions/decoder_v001/payload.md").write_text("hidden payload")
    try:
        module.admit(lab)
    except module.AdmissionError:
        pass
    else:
        raise AssertionError("solution payload file was accepted")
    (lab / "solutions/decoder_v001/payload.md").unlink()
    (lab / "notes.md").write_text("x" * (module.MAX_NOTES_BYTES + 1))
    try:
        module.admit(lab)
    except module.AdmissionError:
        pass
    else:
        raise AssertionError("oversized notes were accepted")
