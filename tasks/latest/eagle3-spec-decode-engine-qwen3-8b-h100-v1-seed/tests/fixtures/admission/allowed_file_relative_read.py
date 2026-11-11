import os
from pathlib import Path
open(os.path.join(os.path.dirname(__file__),'k.cu')).read()
(Path(__file__).parent/'k.cuh').read_text()
