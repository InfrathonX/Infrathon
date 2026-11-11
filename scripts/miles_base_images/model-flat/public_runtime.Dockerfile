# Runtime filesystem for public smoke; dependency qualification is separate.
# The parent is the locally built NumPy/SciPy compatibility image. Verify this
# local tag resolves to image ID sha256:65f202abaf079dc2184bc9d6e2fe34f5368338004fa11d5d71a2e4ad99c0ea22
# before building; Docker FROM does not accept a local config image ID directly.
# Build with an empty context. Exact IDs/commands are recorded in the report.
ARG BASE_IMAGE=mlsysforge/miles-validation-parent:65f202abaf079dc2184bc9d6e2fe34f5368338004fa11d5d71a2e4ad99c0ea22
FROM ${BASE_IMAGE}

LABEL mlsys-authoring="miles-model-public-runtime"

# Remove the bundled modern Miles checkout, including its future git history,
# editable-install metadata/finder, and finder bytecode. Retain all model support
# in Megatron-LM, Megatron Bridge, mbridge, Transformers and SGLang. The one
# Megatron compatibility edit below only makes a Miles helper import optional.
RUN python - <<'PY'
from pathlib import Path
import shutil

site = Path('/usr/local/lib/python3.12/dist-packages')
targets = [
    Path('/root/miles'),
    site / 'miles-0.1.0.dist-info',
    site / '__editable__.miles-0.1.0.pth',
    site / '__editable___miles_0_1_0_finder.py',
    *sorted((site / '__pycache__').glob('__editable___miles_0_1_0_finder.*.pyc')),
]
for path in targets:
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)
PY

# This newer Megatron-LM requires a Miles helper absent in the historical base.
# Make only that missing optional module compatible. Do not supply the helper
# implementation or suppress dependency/import/execution errors from a helper
# that is present. Both the historical control and the reference use this image.
RUN python - <<'PY'
from pathlib import Path
import hashlib

path = Path('/root/Megatron-LM/megatron/training/training.py')
original = path.read_text()
assert hashlib.sha256(path.read_bytes()).hexdigest() == '29f5246bf7ded29c20848d0648270afc0b4d007be490541ecdec331ee4026e66'
old = '''    from miles.backends.megatron_utils.fp32_param_utils import enforce_marked_param_dtypes

    enforce_marked_param_dtypes(model)
'''
new = '''    try:
        from miles.backends.megatron_utils.fp32_param_utils import enforce_marked_param_dtypes
    except ModuleNotFoundError as exc:
        if exc.name != "miles.backends.megatron_utils.fp32_param_utils":
            raise
    else:
        enforce_marked_param_dtypes(model)
'''
assert original.count(old) == 1
path.write_text(original.replace(old, new))
PY

# A fresh interpreter must have no fallback import of the modern answer.
RUN python - <<'PY'
import importlib.metadata as metadata
import importlib.util
from pathlib import Path

assert not Path('/root/miles').exists()
for name in ['miles', 'miles_plugins']:
    assert importlib.util.find_spec(name) is None, name
try:
    metadata.distribution('miles')
except metadata.PackageNotFoundError:
    pass
else:
    raise RuntimeError('bundled Miles installation remains')
for name in ['torch', 'transformers', 'sglang', 'ray', 'fla',
             'transformer_engine', 'mbridge', 'megatron']:
    assert importlib.util.find_spec(name) is not None, name
PY

WORKDIR /workspace
