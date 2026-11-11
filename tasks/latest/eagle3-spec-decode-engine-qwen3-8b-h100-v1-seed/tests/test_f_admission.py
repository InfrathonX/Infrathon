"""E2 minimal policy cases, copied without executing candidate source."""
import json
from pathlib import Path
import pytest
from source_admission import scan_python, validate_submission

FIXTURES = Path(__file__).parent / "fixtures/admission"
CASES = json.loads((FIXTURES / "expected.json").read_text())

@pytest.mark.parametrize("name", sorted(CASES))
def test_e2_case(name, tmp_path):
    if name == "unselected_sibling":
        (tmp_path / "solutions/engine_v001").mkdir(parents=True)
        (tmp_path / "solutions/engine_v002").mkdir()
        (tmp_path / "FINAL_VERSION").write_text("1\n")
        (tmp_path / "notes.md").write_text("fixture\n")
        (tmp_path / "solutions/engine_v001/candidate.py").write_text((FIXTURES / (name + ".py")).read_text())
        (tmp_path / "solutions/engine_v002/candidate.py").write_text("import socket\nmarker = True\n")
        rejected = not validate_submission(tmp_path)["ok"]
    else:
        rejected = bool(scan_python(FIXTURES / (name + ".py"), name + ".py"))
    assert rejected == (CASES[name]["expected"] == "REJECT")

@pytest.mark.parametrize('source, rejected', [
    ('import gc\ngc.collect()\n', False),
    ("import torch\nname='bfloat16'\ndtype=getattr(torch,name)\n", False),
    ('import sys\nframe=sys._getframe()\n', True),
    ('from runtime.api import validate_result\nx=validate_result.__globals__\n', True),
    ('import runtime.api as api\napi.__dict__.update(value=1)\n', True),
])
def test_semantic_alias_boundaries(source, rejected, tmp_path):
    path = tmp_path / 'case.py'
    path.write_text(source)
    assert bool(scan_python(path, 'case.py')) == rejected


@pytest.mark.parametrize('source, rejected', [
    ("import gc\ngetattr(gc, 'get_objects')()\n", True),
    ("import gc as collector\nname='get_referrers'\ngetattr(collector,name)([])\n", True),
    ("import gc\ngc.__dict__['get_objects']()\n", True),
    ("import gc\nvars(gc)['get_referents']([])\n", True),
    ("import time\nvars(time)['perf_counter'] = lambda: 0\n", True),
    ("import gc\ngetattr(gc, 'collect')()\n", False),
])
def test_gc_lookup_routes(source, rejected, tmp_path):
    path = tmp_path / 'case.py'
    path.write_text(source)
    assert bool(scan_python(path, 'case.py')) == rejected


@pytest.mark.parametrize('source, rejected', [
    ("import torch\nsync=getattr(torch.cuda,'synchronize')\n", False),
    ("import torch\nk=getattr(torch.nn.attention,'sdpa_kernel',None)\n", False),
    ("import triton\nlanguage=getattr(triton,'language')\n", False),
    ("import torch\nloader=getattr(torch,'load')\n", True),
    ("import torch\nmodule=getattr(torch,'serialization')\n", True),
    ("import torch\ntable=getattr(torch,'__dict__')\n", True),
    ("import torch\ndef f(name):\n return getattr(torch.backends.cuda,name)\n", True),
    ("import torch\ndef f(bits):\n return getattr(torch,f'float{bits}')\n", True),
    ("import torch\nsetattr(torch.backends.cuda.matmul,'allow_tf32',True)\n", True),
    ("import sys\nframe=getattr(sys,'_getframe')\n", True),
    ("import os\ncwd=getattr(os,'getcwd')\n", True),
    ("import runtime.api as api\nfn=getattr(api,'validate_result')\n", True),
])
def test_feature_getattr_routes(source, rejected, tmp_path):
    path = tmp_path / 'case.py'
    path.write_text(source)
    assert bool(scan_python(path, 'case.py')) == rejected


@pytest.mark.parametrize('source, rejected', [
    ("def f(x):\n open(f'/tmp/{x}.log','w')\n", False),
    ("def f(x):\n open('/tmp/cache_'+x,'wb')\n", False),
    ("def f(x):\n open('/tmp/%s.log'%x,'w')\n", False),
    ("def f(x):\n open('/tmp/{}.log'.format(x),'w')\n", False),
    ("from pathlib import Path\ndef f(x):\n p=Path('/tmp')/x\n p.write_text('ok')\n", False),
    ("def f(x):\n open(f'/app/workdir/spec_lab/{x}')\n", False),
    ("import os\ndef f(x):\n open(os.path.join(os.path.dirname(__file__),x))\n", False),
    ("open('/tmp/torch_extensions/x/kernel.so','rb')\n", False),
    ("def f(x):\n open(f'/tmp{x}','w')\n", True),
    ("def f(x):\n open('/tmp/../'+x,'w')\n", True),
    ("def f(x):\n open(f'/app/workdir/spec_lab/{x}','w')\n", True),
    ("def f(x):\n open(f'/opt/models/{x}','w')\n", True),
    ("def f(x):\n open(f'/etc/{x}')\n", True),
    ("def f(x):\n open(f'{x}/log','w')\n", True),
    ("def f(x):\n open('{}/log'.format(x),'w')\n", True),
    ("def f(x):\n open(x)\n", True),
    ("import os\ndef f(x):\n open(os.path.join(os.path.dirname(__file__),x),'w')\n", True),
    ("from pathlib import Path\ndef f(x):\n (Path(__file__).parent/'..'/'..'/'..'/x).read_text()\n", True),
    ("from pathlib import Path\ndef f(x):\n p=Path('/etc')/x\n p.read_text()\n", True),
    ("open('/tmp/../etc/passwd')\n", True),
])
def test_static_prefix_paths(source, rejected, tmp_path):
    path = tmp_path / 'case.py'
    path.write_text(source)
    assert bool(scan_python(path, 'case.py')) == rejected
