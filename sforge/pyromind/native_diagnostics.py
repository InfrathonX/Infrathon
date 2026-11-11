"""Bounded private logging for unchanged native torchrun admission probes."""
from pathlib import Path
import json
import re


def torchrun_log_directory(handle, namespace):
    if not isinstance(namespace, str) or not re.fullmatch(r"[a-z][a-z0-9-]{0,40}", namespace):
        raise ValueError("Invalid native diagnostic namespace")
    identity = getattr(handle, 'id', handle)
    if not isinstance(identity, str) or not re.fullmatch(r'[a-z0-9-]+', identity):
        raise ValueError('Invalid native diagnostic Sandbox identity')
    return '/tmp/sforge-' + namespace + '-native-preflight-' + identity


def collect_torchrun_logs(backend, handle, directory, *, namespace):
    """Bounded no-symlink collection of torchrun's native per-rank log files."""
    source = r'''import json,os,stat,sys
root=sys.argv[1];records=[];total=0
try:rootfd=os.open(root,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
except FileNotFoundError:
 print(json.dumps({'source':root,'files':[],'absent':True}));raise SystemExit(0)
try:
 for base,dirs,files,dirfd in os.fwalk('.',dir_fd=rootfd,follow_symlinks=False):
  for name in files:
   if name not in {'stdout.log','stderr.log','error.json'}:continue
   fd=os.open(name,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=dirfd)
   try:
    meta=os.fstat(fd)
    if not stat.S_ISREG(meta.st_mode) or meta.st_nlink!=1 or meta.st_size>1024*1024:raise RuntimeError('Unsafe or oversized diagnostic log')
    data=b''
    while len(data)<=1024*1024:
     block=os.read(fd,min(65536,1024*1024+1-len(data)))
     if not block:break
     data+=block
    total+=len(data)
    if len(data)>1024*1024 or total>4*1024*1024 or len(records)>=64:raise RuntimeError('Diagnostic log budget exceeded')
    records.append({'path':base+'/'+name,'bytes':len(data),'uid':meta.st_uid,'content':data.decode('utf-8',errors='replace')})
   finally:os.close(fd)
finally:os.close(rootfd)
print(json.dumps({'source':root,'files':records,'bytes':total}))
'''
    result = backend.exec_run_with_timeout(handle,
        ['python3', '-c', source, torchrun_log_directory(handle, namespace)], timeout=30)
    if result.exit_code or result.timed_out:
        raise RuntimeError('Native torchrun diagnostic log collection failed')
    record = json.loads(result.output)
    destination = Path(directory)/'native-torchrun-logs.json'
    destination.write_text(json.dumps(record, indent=2)+'\n')
    destination.chmod(0o600)
