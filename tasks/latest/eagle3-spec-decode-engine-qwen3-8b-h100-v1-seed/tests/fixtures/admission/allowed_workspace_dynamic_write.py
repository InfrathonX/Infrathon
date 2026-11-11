import os
from pathlib import Path
BUILD='/app/workdir/spec_lab/build'
def dump(name):
 open(f'/app/workdir/spec_lab/build/{name}.bin','wb').write(b'ok')
 open(os.path.join(BUILD,name),'w').write('ok')
 (Path(BUILD)/name).write_bytes(b'ok')
