"""Exercise one-at-a-time request delivery using a real CPU subprocess."""
import os
import sys
import pytest
from conftest import load_file, VERIFIER
controller = load_file('f_controller', VERIFIER / 'controller.py')

def test_no_future_requests_during_build_or_generate(tmp_path):
    worker = tmp_path / 'worker.py'
    worker.write_text('''import sys, socket, json, time
s = socket.socket(fileno=int(sys.argv[-1]))
f = s.makefile('rwb')
def read(): return json.loads(f.readline())
def write(x): f.write((json.dumps(x)+'\\n').encode()); f.flush()
header = read()
assert set(header) == {'policy'}
s.setblocking(False)
try:
    assert not s.recv(1, socket.MSG_PEEK), 'future requests sent before build'
except BlockingIOError: pass
s.setblocking(True)
write({'kind':'built','build_s':0.001})
while True:
    frame = read()
    if frame['kind'] == 'done': break
    assert 'cells' not in frame and 'rows' in frame
    write({'kind':'ready'})
    assert read() == {'kind':'go'}
    time.sleep(0.02)
    write({'kind':'result', 'outputs':[[1,2] for _ in frame['rows']], 'prompt_lengths':[1]*len(frame['rows']), 'measurement':{'wall_s':0.000001,'generated_lengths':[1]*len(frame['rows']),'peak_memory_bytes':1,'device_idle':True}})
''')
    cells = [{'cell':'chat_bs1','batch_size':1,'max_new_tokens':512,'enable_thinking':False,'warmup':[{'messages':['warm']}],'rows':[{'messages':['first']},{'messages':['future']}]}]
    completed = controller.run_worker([sys.executable,str(worker)], {'policy':{},'cells':cells}, 10., dict(os.environ))
    assert completed.returncode == 0, completed.stderr
    result = completed.worker_result['cells']['chat_bs1']
    assert len(result['measurements']) == 2
    assert all(row['wall_s'] >= 0.02 for row in result['measurements'])
    assert result['rerun_expected'] == result['rerun_outputs']


@pytest.mark.parametrize("probe", ["heap", "clock", "lazy", "queued"])
def test_real_worker_heap_and_deferred_result_boundary(tmp_path, probe):
    """Run the actual worker with CPU loaders, a GC probe and a deferred result."""
    pytest.importorskip("safetensors", reason="the worker imports runtime.loader, which needs safetensors")
    import json
    from pathlib import Path
    from conftest import SPEC
    before = os.environ.get('F_TEST_BEFORE_IMAGES')
    verifier = Path(before) / 'judge/verifier' if before else VERIFIER
    supervisor = load_file('f_real_supervisor', verifier / 'controller.py')
    version = tmp_path / 'solutions/engine_v001'
    version.mkdir(parents=True)
    (version / '__init__.py').write_text('')
    (version / 'candidate.py').write_text('''import gc, time, torch
from runtime.api import GenerationResult

def build(target, draft, tokenizer, contract):
    future = [x for x in gc.get_objects() if type(x) is dict and 'policy' in x and 'cells' in x]
    if PROBE == 'heap':
        assert not future, 'future hidden request manifest visible during build'
    original_clock = time.perf_counter
    class Lazy(GenerationResult):
        def __init__(self, prompts):
            self.prompts = prompts
            self.actual = None
            self.trace = {'future_requests_visible': False}
        @property
        def token_ids(self):
            if self.actual is None:
                time.sleep(.025)
                self.actual = [torch.cat([p, p.new_tensor([2])]) for p in self.prompts]
            return self.actual
    class Engine:
        def generate(self,prompts,spec):
            if PROBE == 'clock':
                time.perf_counter = lambda: original_clock() * .01
                time.sleep(.025)
            if PROBE == 'lazy':
                return Lazy(prompts)
            if PROBE == 'queued':
                torch.cuda._sleep(1)
            return GenerationResult([torch.cat([p, p.new_tensor([2])]) for p in prompts], {'future_requests_visible': bool(future)})
    return Engine()
'''.replace('PROBE', repr(probe)))
    (tmp_path / 'MANIFEST.json').write_text(json.dumps({'revision':'r','repo_id':'d'}))
    bootstrap = tmp_path / 'bootstrap.py'
    bootstrap.write_text(f'''import sys, importlib.util, torch
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,{str(SPEC)!r})
import runtime.loader as loader
import runtime.prompts as prompts
from runtime.api import EngineContract
loader.default_head_dir = lambda: Path({str(tmp_path)!r})
loader.build_contract = lambda *a,**k: EngineContract('t','r','d','r',device='cpu')
loader.load_target = lambda *a: None
loader.load_tokenizer = lambda *a: None
loader.load_draft_head = lambda *a: None
prompts.render = lambda messages,*a: torch.tensor([messages[0]['value']],dtype=torch.int64)
pending = [False]
def synchronize():
    if pending[0]:
        import time
        time.sleep(.025)
        pending[0] = False
def enqueue(cycles): pending[0] = True
torch.cuda.synchronize = synchronize
torch.cuda._sleep = enqueue
torch.cuda.reset_peak_memory_stats = lambda: None
torch.cuda.max_memory_allocated = lambda: 0
torch.cuda.current_stream = lambda: SimpleNamespace(query=lambda: not pending[0])
spec = importlib.util.spec_from_file_location('actual_worker',{str(verifier / 'candidate_worker.py')!r})
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)
worker.SPEC_ROOT = Path({str(SPEC)!r})
worker.drop_privileges = lambda: None
raise SystemExit(worker.main())
''')
    rows = [{'messages':[{'value':i}]} for i in [1,3]]
    request = {'policy':{'build_cap_s':30,'wall_caps_s':{'chat_bs1':5}},'cells':[{'cell':'chat_bs1','batch_size':1,'max_new_tokens':1,'enable_thinking':False,'warmup':[{'messages':[{'value':4}]}],'rows':rows}]}
    output = tmp_path / 'output.json'
    completed = supervisor.run_worker([sys.executable,str(bootstrap),'--candidate-root',str(tmp_path),'--candidate-version','1','--output',str(output)],request,30,dict(os.environ))
    assert completed.returncode == 0, completed.stderr.decode()
    worker_result = completed.worker_result if hasattr(completed, 'worker_result') else json.loads(output.read_text())
    result = worker_result['cells']['chat_bs1']
    if probe == 'heap':
        assert all(row['trace']['future_requests_visible'] is False for row in result['measurements'])
    else:
        assert all(row['wall_s'] >= .025 for row in result['measurements'])
    if probe == 'queued':
        assert all(row.get('pending_current_stream_at_return') is True for row in result['measurements'])
        assert all(row['return_to_sync_s'] >= .025 for row in result['measurements'])
