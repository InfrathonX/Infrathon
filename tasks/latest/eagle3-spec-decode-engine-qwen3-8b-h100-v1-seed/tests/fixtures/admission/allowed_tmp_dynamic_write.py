import os
from pathlib import Path
def dump(name):
 open(f'/tmp/{name}.log','w').write('ok')
 open(os.path.join('/tmp',name),'w').write('ok')
 (Path('/tmp')/name).write_text('ok')
