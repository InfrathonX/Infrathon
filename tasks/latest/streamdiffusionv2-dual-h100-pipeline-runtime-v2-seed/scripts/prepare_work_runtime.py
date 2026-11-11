"""Copy public runtime code into the Work build context without private case definitions."""
import shutil, sys
from pathlib import Path
source, target = map(Path, sys.argv[1:])
if target.exists(): shutil.rmtree(target)
shutil.copytree(source, target, ignore=shutil.ignore_patterns('__pycache__','*.pyc','cases.py','README.md','evaluation_protocol.json'))
