"""Prototype Controller-owned namespaces for SWE tool commands.

This is a fail-closed runtime hook for ``SWEFleet``. It is not qualification:
the prototype uses four CPUs/eight GiB, and this hook does not impose
the original Docker CPU quota, pids cgroup or seccomp profile. System files and
the image's /root remain read-only; only /testbed and private tmp mounts are
writable. That filesystem difference needs independent task validation.
"""
from __future__ import annotations

import hashlib
import json
import re
import shlex
import uuid


RUNTIME_SETUP_ID = 'swe-tool-namespaces-v1'
_MARKER = 'SFORGE_SWE_TOOL_RUNTIME='
_FAILURE_MARKER = 'SFORGE_SWE_TOOL_FAILURE='
_FAILURE_PHASES = {
    'holder_start', 'initial_pid_user_checks', 'holder_ptrace_protection',
    'mount_propagation_private', 'testbed_bind_writable', 'inherited_mounts_readonly',
    'private_run_mount', 'private_tmp_mount', 'private_var_tmp_mount',
    'private_shm_mount', 'private_mqueue_mount', 'private_pid_proc_mount',
    'filesystem_proof', 'namespace_holder_start', 'holder_identity_and_owner_record',
}
_FAILURE_TYPES = {
    'AssertionError', 'RuntimeError', 'CalledProcessError', 'OSError',
    'PermissionError', 'FileNotFoundError', 'TimeoutError', 'ValueError', 'KeyError',
}
_NAMESPACES = ('user', 'mnt', 'net', 'pid', 'ipc')
_GAPS = (
    'cpu_cfs_quota_2_not_enforced',
    'pids_cgroup_limit_1024_not_enforced',
    'original_seccomp_profile_not_attested',
    'readonly_system_and_home_semantics_require_validation',
    'platform_control_plane_filesystem_allowlist_not_attested',
    'platform_device_allowlist_not_attested',
)


class ToolRuntimeError(RuntimeError):
    """Setup or a subsequent integrity check failed; ordinary exec is forbidden."""


# The SDK executes these programs in a fresh Sandbox. Isolated Python (-I)
# prevents /testbed or inherited PYTHONPATH from supplying Controller imports.
_COMMON = r'''
import hashlib,json,os,re,stat,sys
NAMESPACES=('user','mnt','net','pid','ipc')
def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':')).encode()
def private_directory(path):
    value=os.lstat(path)
    assert stat.S_ISDIR(value.st_mode) and value.st_uid==0 and value.st_mode & 0o777==0o700, 'unsafe Controller directory'
def secure_file(path):
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_CLOEXEC)
    value=os.fstat(fd)
    assert stat.S_ISREG(value.st_mode) and value.st_uid==0 and value.st_mode & 0o777==0o600 and value.st_nlink==1, 'unsafe Controller file'
    return fd
def starttime(pid):
    data=open('/proc/'+str(pid)+'/stat').read()
    return int(data[data.rfind(')')+2:].split()[19])
def trusted_binary(path):
    path=os.path.realpath(path)
    assert os.path.isabs(path) and os.path.isfile(path) and os.access(path,os.X_OK), 'missing runtime binary'
    current=path
    while True:
        value=os.lstat(current)
        assert value.st_uid==0 and not value.st_mode & 0o022, 'writable runtime binary path'
        if current=='/': break
        current=os.path.dirname(current)
    return path
def namespace_ids(pid):
    return {name:os.stat('/proc/'+str(pid)+'/ns/'+name).st_ino for name in NAMESPACES}
def private_mount_propagation(lines):
    seen=False
    for line in lines:
        seen=True
        fields=line.split()
        try: end=fields.index('-')
        except ValueError: return False
        if end<6 or any(value.startswith(('shared:','master:','propagate_from:')) for value in fields[6:end]):
            return False
    return seen
'''

_HOLDER = _COMMON + r'''
import ctypes,signal,subprocess,time
cfg=json.loads(sys.argv[1]); ready=int(sys.argv[2])
phase='initial_pid_user_checks'
assert os.getpid()==1 and os.geteuid()==0, 'not a new PID namespace init'
status=open('/proc/self/status').read()
outer_pid=int(re.search(r'^NSpid:\s+(\d+)',status,re.M).group(1))
libc=ctypes.CDLL(None,use_errno=True)
phase='holder_ptrace_protection'
assert libc.prctl(4,0,0,0,0)==0 and libc.prctl(3,0,0,0,0)==0, 'cannot protect holder from ptrace'
assert libc.prctl(15,b'sforge-tool-init',0,0,0)==0, 'cannot name holder'
def mount(*args):
    subprocess.run([cfg['binaries']['mount'],*args],stdin=subprocess.DEVNULL,check=True,
                   stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
def mount_path(field):
    return re.sub(r'\\([0-7]{3})',lambda value:chr(int(value[1],8)),field)
phase='mount_propagation_private'; mount('--make-rprivate','/')
assert private_mount_propagation(open('/proc/self/mountinfo')), 'mount propagation is not private'
phase='testbed_bind_writable'
mount('--bind','/testbed','/testbed')
mount('-o','remount,bind,rw,nosuid,nodev','/testbed')
# Tighten every inherited mount, including injected /etc and /sys submounts.
# A remount modifies only this private mount namespace, never the SDK's view.
paths={mount_path(line.split()[4]) for line in open('/proc/self/mountinfo')}
phase='inherited_mounts_readonly'
for path in sorted(paths,key=lambda value:(value.count('/'),len(value)),reverse=True):
    if path=='/testbed' or path.startswith('/testbed/'):
        continue
    mount('-o','remount,bind,ro',path)
for path,mode in (('/run','0755'),('/tmp','1777'),('/var/tmp','1777'),('/dev/shm','1777')):
    phase={'/run':'private_run_mount','/tmp':'private_tmp_mount',
           '/var/tmp':'private_var_tmp_mount','/dev/shm':'private_shm_mount'}[path]
    assert os.path.isdir(path) and not os.path.islink(path), 'missing private mount point'
    mount('-t','tmpfs','-o','nosuid,nodev,mode='+mode,'sforge-tool-private',path)
if os.path.isdir('/dev/mqueue'):
    phase='private_mqueue_mount'
    mount('-t','tmpfs','-o','nosuid,nodev,noexec,mode=1777','sforge-tool-private','/dev/mqueue')
phase='private_pid_proc_mount'; mount('-t','proc','-o','ro,nosuid,nodev,noexec','proc','/proc')
phase='filesystem_proof'
assert os.statvfs('/').f_flag & os.ST_RDONLY, 'root is not read-only'
assert os.statvfs('/root').f_flag & os.ST_RDONLY, 'image home is not read-only'
assert not os.statvfs('/testbed').f_flag & os.ST_RDONLY, 'testbed is not writable'
assert os.statvfs('/sys/fs/cgroup').f_flag & os.ST_RDONLY, 'cgroup is writable'
assert any(' - cgroup2 ' in line and mount_path(line.split()[4])=='/sys/fs/cgroup'
           for line in open('/proc/self/mountinfo')), 'cgroup2 is required'
assert not os.path.exists(cfg['job_directory']), 'Controller state is visible inside tool mount namespace'
os.mkdir('/tmp/sforge-cache',0o700)
if cfg['affinity'] is not None:
    os.sched_setaffinity(0,cfg['affinity'])
signal.signal(signal.SIGCHLD,signal.SIG_IGN)
message={'pid':outer_pid,'namespaces':namespace_ids('self'),'holder_dumpable':False,
         'mount_propagation_private':True,
         'holder_affinity':sorted(os.sched_getaffinity(0))}
data=canonical(message)+b'\n'
assert len(data)<4096
os.write(ready,data); os.close(ready)
while True:
    time.sleep(3600)
'''
_HOLDER = (_COMMON + "phase='holder_start'\ntry:\n"
           + '\n'.join('    ' + line for line in _HOLDER[len(_COMMON):].splitlines())
           + r'''
except BaseException as error:
    try:
        os.write(ready,canonical({'failure_phase':phase,'error_type':type(error).__name__})+b'\n')
    except BaseException: pass
    sys.exit(126)
''')

_BOOTSTRAP = _COMMON + r'''
import selectors,shutil,subprocess,time
cfg=json.loads(sys.argv[1]); assert os.geteuid()==0, 'Controller bootstrap must run as root'
assert re.fullmatch(r'[0-9a-f]{32}',cfg['identity'])
assert os.path.isdir('/testbed') and not os.path.islink('/testbed'), 'unsafe testbed'
assert os.path.isdir('/root') and not os.path.islink('/root'), 'unsafe image home'
root='/run/sforge-swe-tool'
try: os.mkdir(root,0o700)
except FileExistsError: pass
private_directory(root)
job=root+'/'+cfg['identity']; os.mkdir(job,0o700); private_directory(job)
binaries={name:trusted_binary(shutil.which(name) or '') for name in ('unshare','mount','nsenter','setpriv','env')}
binaries['python']=trusted_binary(sys.executable)
allowed=sorted(os.sched_getaffinity(0)); count=cfg['affinity_count']
assert count is None or (type(count) is int and 1<=count<=len(allowed)), 'invalid tool affinity'
affinity=None if count is None else allowed[:count]
environment={'PATH':os.environ.get('PATH','/usr/local/bin:/usr/bin:/bin'),
             'HOME':'/root','LANG':os.environ.get('LANG','C.UTF-8'),
             'SHELL':'/bin/bash','USER':'root','LOGNAME':'root','TMPDIR':'/tmp',
             'XDG_CACHE_HOME':'/tmp/sforge-cache'}
assert all(isinstance(value,str) and '\x00' not in value and len(value)<8192 for value in environment.values())
holder={'binaries':binaries,'job_directory':job,'affinity':affinity}
read_fd,write_fd=os.pipe()
log_fd=os.open(job+'/holder.log',os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
process=subprocess.Popen([binaries['unshare'],'--user','--map-root-user','--mount','--net',
                          '--pid','--ipc','--propagation=unchanged','--fork','--kill-child=KILL',
                          binaries['python'],'-I','-c',cfg['holder_source'],
                          json.dumps(holder),str(write_fd)],
                         stdin=subprocess.DEVNULL,stdout=log_fd,stderr=log_fd,
                         pass_fds=(write_fd,),start_new_session=True,env=environment,cwd='/')
os.close(write_fd); os.close(log_fd)
selector=selectors.DefaultSelector(); selector.register(read_fd,selectors.EVENT_READ)
failure_phase='namespace_holder_start'
try:
    assert selector.select(cfg['wait_seconds']), 'namespace holder readiness timed out'
    data=os.read(read_fd,4097)
    assert 0<len(data)<=4096 and data.endswith(b'\n'), 'namespace holder failed'
    message=json.loads(data)
    if 'failure_phase' in message:
        failure_phase=message['failure_phase']
        raise RuntimeError('namespace holder initialization failed')
    failure_phase='holder_identity_and_owner_record'
    pid=message['pid']; assert type(pid) is int and pid>1 and process.poll() is None
    namespaces=namespace_ids(pid); assert namespaces==message['namespaces'], 'holder namespace identity changed'
    outer=namespace_ids('self')
    assert all(namespaces[name]!=outer[name] for name in NAMESPACES), 'namespace separation failed'
    assert message['holder_dumpable'] is False and message['mount_propagation_private'] is True
    record={'schema_version':1,'identity':cfg['identity'],'job_directory':job,
            'pid':pid,'pid_starttime':starttime(pid),'namespaces':namespaces,
            'outer_namespaces':outer,'bootstrap_pid':os.getpid(),'unshare_pid':process.pid,
            'binaries':binaries,'environment':environment,'affinity':affinity,
            'holder_affinity':message['holder_affinity'],'holder_dumpable':False}
    record['mount_propagation_private']=True
    payload=canonical(record)
    temporary=job+'/owner.tmp'
    fd=os.open(temporary,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as stream:
        stream.write(payload); stream.flush(); os.fsync(stream.fileno())
    owner=job+'/owner.json'
    assert not os.path.lexists(owner)
    os.rename(temporary,owner)
    directory_fd=os.open(job,os.O_RDONLY|os.O_DIRECTORY)
    os.fsync(directory_fd); os.close(directory_fd)
    print('SFORGE_SWE_TOOL_RUNTIME='+json.dumps({'owner':record,'owner_path':owner,
                                              'owner_sha256':hashlib.sha256(payload).hexdigest()},sort_keys=True),flush=True)
except BaseException as error:
    # No command has been accepted yet. Fleet also owns the Sandbox and will
    # delete it on failure, including an ambiguous SDK bootstrap response.
    try: os.killpg(process.pid,9)
    except ProcessLookupError: pass
    print('SFORGE_SWE_TOOL_FAILURE='+json.dumps({'phase':failure_phase,'error_type':type(error).__name__}),flush=True)
    raise
finally:
    selector.close(); os.close(read_fd)
'''

_ENTRY = r'''
import json,os,re,sys
expected=json.loads(sys.argv[1]); command=sys.argv[2:]
status=open('/proc/self/status').read()
assert all(int(re.search(r'^'+name+r':\s+(\S+)',status,re.M).group(1),16)==0
           for name in ('CapInh','CapPrm','CapEff','CapBnd','CapAmb')), 'tool capabilities not empty'
assert re.search(r'^NoNewPrivs:\s+1$',status,re.M), 'no_new_privs not applied'
assert {name:os.stat('/proc/self/ns/'+name).st_ino for name in expected['namespaces']}==expected['namespaces'], 'tool namespace entry failed'
assert os.statvfs('/').f_flag & os.ST_RDONLY, 'tool retained the Controller filesystem root'
assert not os.path.exists(expected['job_directory']), 'tool can see Controller state'
mounts=list(open('/proc/self/mountinfo')); assert mounts, 'tool mount table is unavailable'
for line in mounts:
    fields=line.split(); end=fields.index('-')
    assert not any(value.startswith(('shared:','master:','propagate_from:')) for value in fields[6:end]), 'tool mount propagation is not private'
# nsenter needed these descriptors to enter the pinned namespaces/root. Do not
# give tools even a reference to Controller-opened descriptors.
for value in os.listdir('/proc/self/fd'):
    if value.isdigit() and int(value)>=3:
        try: os.close(int(value))
        except OSError: pass
os.chdir('/testbed')
os.execvpe(command[0],command,os.environ)
'''

_LAUNCHER = _COMMON + r'''
assert os.geteuid()==0, 'tool launcher is Controller-only'
owner_path,expected=sys.argv[1:3]; command=sys.argv[3:]
assert command and os.path.isabs(owner_path) and re.fullmatch(r'[0-9a-f]{64}',expected)
job=os.path.dirname(owner_path); private_directory(os.path.dirname(job)); private_directory(job)
fd=secure_file(owner_path)
with os.fdopen(fd,'rb') as stream:
    payload=stream.read(32769)
assert len(payload)<=32768 and hashlib.sha256(payload).hexdigest()==expected, 'Controller owner record changed'
owner=json.loads(payload)
assert owner['schema_version']==1 and owner['job_directory']==job
pid=owner['pid']; assert type(pid) is int and pid>1
assert starttime(pid)==owner['pid_starttime'], 'namespace holder PID was reused'
handles={}
for name in NAMESPACES:
    descriptor=os.open('/proc/'+str(pid)+'/ns/'+name,os.O_RDONLY|os.O_CLOEXEC)
    assert os.fstat(descriptor).st_ino==owner['namespaces'][name], 'namespace holder identity changed'
    os.set_inheritable(descriptor,True); handles[name]=descriptor
root_fd=os.open('/proc/'+str(pid)+'/root',os.O_PATH|os.O_DIRECTORY|os.O_CLOEXEC)
assert stat.S_ISDIR(os.fstat(root_fd).st_mode), 'holder filesystem root is invalid'
os.set_inheritable(root_fd,True)
# Open FDs pin exactly the checked namespaces, even if the holder exits or its
# numeric PID is reused between this validation and nsenter's setns calls.
assert starttime(pid)==owner['pid_starttime'], 'namespace holder disappeared'
if owner['affinity'] is not None:
    os.sched_setaffinity(0,owner['affinity'])
options={'user':'--user','mnt':'--mount','net':'--net','pid':'--pid','ipc':'--ipc'}
argv=[owner['binaries']['nsenter']]
argv += [options[name]+'=/proc/self/fd/'+str(handles[name]) for name in NAMESPACES]
argv += ['--root=/proc/self/fd/'+str(root_fd),'--wd=/proc/self/fd/'+str(root_fd)]
argv += ['--preserve-credentials','--',owner['binaries']['setpriv'],
         '--bounding-set=-all','--inh-caps=-all','--ambient-caps=-all','--no-new-privs',
         owner['binaries']['env'],'-i']
argv += [key+'='+value for key,value in sorted(owner['environment'].items())]
guard={'namespaces':owner['namespaces'],'job_directory':owner['job_directory']}
argv += [owner['binaries']['python'],'-I','-c',ENTRY_SOURCE,json.dumps(guard),*command]
os.chdir('/')
os.execv(argv[0],argv)
'''
_LAUNCHER = _LAUNCHER.replace('ENTRY_SOURCE', repr(_ENTRY))

_PROBE = _COMMON + r'''
import errno,json,os,re,socket,sys
cfg=json.loads(sys.argv[1]); phase=sys.argv[2]
status=open('/proc/self/status').read()
caps={name:int(re.search(r'^'+name+r':\s+(\S+)',status,re.M).group(1),16)
      for name in ('CapInh','CapPrm','CapEff','CapBnd','CapAmb')}
assert all(value==0 for value in caps.values()), 'tool still has capabilities'
assert re.search(r'^NoNewPrivs:\s+1$',status,re.M), 'tool can gain privileges'
assert {name:os.stat('/proc/self/ns/'+name).st_ino for name in cfg['namespaces']}==cfg['namespaces'], 'tool namespaces differ'
assert open('/proc/1/comm').read().strip()=='sforge-tool-init', 'tool proc exposes another PID namespace'
try:
    with open('/proc/1/environ','rb') as stream: stream.read(1)
except PermissionError: pass
else: raise AssertionError('holder environment is ptrace-readable')
assert not os.path.exists(cfg['job_directory']), 'Controller owner state is visible'
assert private_mount_propagation(open('/proc/self/mountinfo')), 'tool mount propagation is not private'
assert os.statvfs('/').f_flag & os.ST_RDONLY and os.statvfs('/root').f_flag & os.ST_RDONLY
assert os.statvfs('/sys/fs/cgroup').f_flag & os.ST_RDONLY
assert not os.statvfs('/tmp').f_flag & os.ST_RDONLY and not os.statvfs('/testbed').f_flag & os.ST_RDONLY
assert set(name for _,name in socket.if_nameindex())=={'lo'}, 'network namespace contains external interfaces'
with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as connection:
    connection.settimeout(1)
    try: connection.connect(('192.0.2.1',9))
    except OSError as error: assert error.errno in (errno.ENETUNREACH,errno.EHOSTUNREACH), 'egress isolation not proved'
    else: raise AssertionError('network egress succeeded')
assert os.environ['HOME']=='/root' and os.environ['PATH']==cfg['environment']['PATH']
assert os.environ['LANG']==cfg['environment']['LANG']
assert set(os.environ)<=set(cfg['environment']), 'Controller environment leaked'
affinity=sorted(os.sched_getaffinity(0))
if cfg['affinity'] is not None: assert affinity==cfg['affinity'], 'tool affinity differs'
sentinel='/tmp/.sforge-runtime-'+cfg['identity']
if phase=='write':
    fd=os.open(sentinel,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'w') as stream: stream.write(cfg['identity'])
    writable='/testbed/.sforge-runtime-'+cfg['identity']
    fd=os.open(writable,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600); os.close(fd); os.unlink(writable)
else:
    assert open(sentinel).read()==cfg['identity'], 'tool tmp did not persist between SDK calls'
    os.unlink(sentinel)
print('SFORGE_SWE_TOOL_RUNTIME='+json.dumps({'phase':phase,'namespaces':cfg['namespaces'],
      'capabilities':caps,'no_new_privs':True,'controller_state_hidden':True,
      'holder_ptrace_denied':True,'proc_namespace_verified':True,
      'mount_propagation_private':True,
      'network_route_unreachable':True,'cgroup2_readonly':True,
      'image_home_preserved_readonly':True,'testbed_writable':True,'affinity':affinity},sort_keys=True))
'''

_MOUNT_DIAGNOSTIC_HELPERS = r'''
import ctypes,errno,fcntl
MS_RDONLY=1; MS_NOSUID=2; MS_NODEV=4; MS_NOEXEC=8
MS_REMOUNT=32; MS_BIND=4096; MS_REC=16384; MS_PRIVATE=1<<18
def diagnostic_mount(libc,label,source,target,fstype,flags,data=None):
    encode=lambda value: None if value is None else os.fsencode(value)
    ctypes.set_errno(0)
    returncode=libc.mount(encode(source),encode(target),encode(fstype),flags,encode(data))
    number=ctypes.get_errno() if returncode else 0
    return {'operation':label,'ok':returncode==0,'returncode':returncode,
            'errno':number,'errno_name':errno.errorcode.get(number,'UNKNOWN') if number else None,
            'flags':flags,'filesystem':fstype}
def mount_rows(lines):
    rows=[]
    for line in lines:
        fields=line.split(); end=fields.index('-')
        assert end>=6 and len(fields)>=end+4, 'malformed mountinfo'
        path=re.sub(r'\\([0-7]{3})',lambda match:chr(int(match[1],8)),fields[4])
        rows.append({'path':path,'mount_id':int(fields[0]),'parent_id':int(fields[1]),
                     'mount_options':fields[5].split(','),'propagation':fields[6:end],
                     'filesystem':fields[end+1]})
    assert rows, 'mountinfo unavailable'
    return rows
def safe_independent_mounts(rows):
    # A slave can receive parent events but does not send them out. This is
    # adequate only for this fixed, short-lived Controller diagnostic. The
    # untrusted tool runtime separately requires *every* mount to be private.
    return bool(rows) and not any(field.startswith('shared:')
                                 for row in rows for field in row['propagation'])
def namespace_user_owners(namespaces):
    owners={}
    for name in namespaces:
        if name=='user': continue
        descriptor=os.open('/proc/self/ns/'+name,os.O_RDONLY|os.O_CLOEXEC)
        try:
            try: owner=fcntl.ioctl(descriptor,0xb701) # NS_GET_USERNS, linux/nsfs.h.
            except OSError as error:
                owners[name]={'errno':error.errno,'errno_name':errno.errorcode.get(error.errno,'UNKNOWN')}
            else:
                try: owners[name]={'user_namespace_inode':os.fstat(owner).st_ino,
                                  'owned_by_current_user_namespace':os.fstat(owner).st_ino==namespaces['user']}
                finally: os.close(owner)
        finally: os.close(descriptor)
    return owners
def safe_lsm_label():
    try:
        with open('/proc/self/attr/current','rb') as stream: value=stream.read(4097)
    except OSError as error:
        return {'readable':False,'errno':error.errno,'errno_name':errno.errorcode.get(error.errno,'UNKNOWN')}
    if len(value)>4096: return {'readable':True,'oversized':True}
    value=value.rstrip(b'\x00\n')
    # Arbitrary policy names are not public diagnostic output.
    category='unconfined' if value==b'unconfined' else 'confined_or_other'
    return {'readable':True,'category':category,'sha256':hashlib.sha256(value).hexdigest()}
'''

_MOUNT_PROBE = _COMMON + _MOUNT_DIAGNOSTIC_HELPERS + r'''
import tempfile
cfg=json.loads(sys.argv[1]); phase='child_identity'
result={'qualification_passed':False,'namespace_order':cfg['namespace_order'],'operations':[]}
try:
    assert os.getpid()==1 and os.geteuid()==0, 'missing new PID/user namespace'
    status=open('/proc/self/status').read()
    result['cap_eff']=re.search(r'^CapEff:\s+(\S+)',status,re.M).group(1)
    for key in ('Seccomp','Seccomp_filters','NoNewPrivs'):
        value=re.search(r'^'+key+r':\s+(\d+)',status,re.M)
        if value: result[key]=int(value.group(1))
    result['kernel_release']=os.uname().release; result['machine']=os.uname().machine
    result['namespaces']=namespace_ids('self')
    result['namespace_user_owners']=namespace_user_owners(result['namespaces'])
    result['lsm_label']=safe_lsm_label()
    rows=mount_rows(open('/proc/self/mountinfo'))
    result['selected_inherited_mounts']=[row for row in rows if row['path'] in
                                       ('/','/tmp','/run','/proc','/sys/fs/cgroup','/dev','/testbed')]
    result['inherited_mounts_private']=private_mount_propagation(open('/proc/self/mountinfo'))
    assert int(result['cap_eff'],16) & (1<<21), 'child lacks namespace CAP_SYS_ADMIN'
    libc=ctypes.CDLL(None,use_errno=True)
    libc.mount.argtypes=[ctypes.c_char_p,ctypes.c_char_p,ctypes.c_char_p,ctypes.c_ulong,ctypes.c_char_p]
    libc.mount.restype=ctypes.c_int
    def mount(label,source,target,fstype,flags,data=None):
        operation=diagnostic_mount(libc,label,source,target,fstype,flags,data)
        result['operations'].append(operation)
        return operation
    phase='mount_propagation_private'
    mount('recursive_private',None,'/',None,MS_REC|MS_PRIVATE)
    rows=mount_rows(open('/proc/self/mountinfo'))
    result['mounts_private_after_syscall']=private_mount_propagation(open('/proc/self/mountinfo'))
    phase='independent_mount_safety'
    result['independent_mounts_attempted']=safe_independent_mounts(rows)
    if result['independent_mounts_attempted']:
        # No candidate is executed. Distinct private targets let each syscall
        # report independently after rprivate fails; no ordinary-exec fallback.
        assert os.path.isdir('/tmp') and not os.path.islink('/tmp'), 'unsafe diagnostic tmp'
        base=tempfile.mkdtemp(prefix='sforge-mount-probe-',dir='/tmp'); private_directory(base)
        for name in ('tmpfs','proc','testbed'): os.mkdir(base+'/'+name,0o700)
        phase='independent_mount_calls'
        temporary=mount('new_tmpfs','sforge-mount-probe',base+'/tmpfs','tmpfs',
                        MS_NOSUID|MS_NODEV|MS_NOEXEC,'mode=0700,size=16m')
        proc=mount('new_pid_proc','proc',base+'/proc','proc',
                   MS_RDONLY|MS_NOSUID|MS_NODEV|MS_NOEXEC)
        if proc['ok']:
            text=open(base+'/proc/1/stat').read()
            init_start=int(text[text.rfind(')')+2:].split()[19])
            proc['pid1_matches_probe']=(init_start==starttime('self') and
                os.stat(base+'/proc/1/ns/pid').st_ino==result['namespaces']['pid'])
        bound=mount('new_testbed_bind','/testbed',base+'/testbed',None,MS_BIND)
        if bound['ok']:
            mount('testbed_bind_rw',None,base+'/testbed',None,MS_REMOUNT|MS_BIND|MS_NOSUID|MS_NODEV)
            readonly=mount('testbed_bind_ro',None,base+'/testbed',None,
                           MS_REMOUNT|MS_BIND|MS_RDONLY|MS_NOSUID|MS_NODEV)
            if readonly['ok']: readonly['readonly_verified']=bool(os.statvfs(base+'/testbed').f_flag & os.ST_RDONLY)
        result['overlay_requested']=cfg['overlay']
        if cfg['overlay'] and temporary['ok']:
            for lower,tag in (('/testbed','testbed'),('/','image_root')):
                location=base+'/tmpfs/'+tag; os.mkdir(location,0o700)
                for name in ('upper','work','merged'): os.mkdir(location+'/'+name,0o700)
                overlay=mount('new_overlay_'+tag,'overlay',location+'/merged','overlay',
                              MS_NOSUID|MS_NODEV,'userxattr,index=off,lowerdir='+lower+
                              ',upperdir='+location+'/upper,workdir='+location+'/work')
                if overlay['ok']:
                    filename='/.sforge-private-overlay-'+os.path.basename(base)
                    assert not os.path.lexists(lower+filename), 'preexisting probe marker'
                    fd=os.open(location+'/merged'+filename,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
                    os.close(fd)
                    overlay['private_write_verified']=not os.path.lexists(lower+filename)
                    os.unlink(location+'/merged'+filename)
        elif cfg['overlay']:
            result['overlay_skipped_reason']='private_tmpfs_unavailable'
        # Bind remount is per-mount; it does not change the underlying root FS.
        readonly=mount('root_bind_readonly',None,'/',None,MS_REMOUNT|MS_BIND|MS_RDONLY)
        if readonly['ok']: readonly['readonly_verified']=bool(os.statvfs('/').f_flag & os.ST_RDONLY)
    else:
        result['independent_mounts_blocked_reason']='inherited_shared_mounts_can_propagate_outward'
    result['cgroup2_readonly']=bool(os.statvfs('/sys/fs/cgroup').f_flag & os.ST_RDONLY)
    failed=[operation['operation'] for operation in result['operations'] if not operation['ok']]
    result.update(status='failed' if failed or not result['mounts_private_after_syscall'] else 'passed',
                  phase='complete',diagnostic_completed=True,failed_operations=failed)
except BaseException as error:
    result.update(status='failed',phase=phase,error_type=type(error).__name__,diagnostic_completed=False)
print('SFORGE_SWE_MOUNT_PROBE='+json.dumps(result,sort_keys=True),flush=True)
sys.exit(0 if result['status']=='passed' else 1)
'''


def mount_namespace_probe_command(*, overlay=True, namespace_order='combined'):
    """Return one short-lived kernel diagnostic command; never execute it here.

    A Controller may pass the string directly to SDK ``exec_command`` in an
    already-owned SWE Sandbox and persist its result before cleanup. Direct
    libc mount calls report errno without libmount's read-only retry.
    Independent mount calls continue after a propagation failure only when no
    inherited mount is shared. Slave mounts still fail the actual runtime's
    private propagation check. ``user_first`` isolates user namespace creation
    from creation of the other four namespaces; it changes this diagnostic
    only. Neither sequence executes a tool or qualifies a task.
    """
    if type(overlay) is not bool:
        raise ValueError('Overlay probe choice must be boolean')
    if namespace_order not in ('combined', 'user_first'):
        raise ValueError('Namespace order must be combined or user_first')
    prefix = ['unshare', '--user', '--map-root-user']
    if namespace_order == 'user_first':
        prefix += ['--propagation=unchanged', '--', 'unshare']
    prefix += ['--mount', '--net', '--pid', '--ipc', '--propagation=unchanged',
               '--fork', '--kill-child=KILL']
    return shlex.join([*prefix, 'python3', '-I', '-c', _MOUNT_PROBE,
                       json.dumps({'overlay': overlay, 'namespace_order': namespace_order})])


_PLATFORM_CAPABILITY_PROBE = _COMMON + r'''
import ctypes,errno
mode=sys.argv[1]
result={'qualification_passed':False,'namespace_mode':mode,'observations':{}}
def read_number_or_max(path):
    try:
        value=open(path).read(128).strip()
        assert re.fullmatch(r'(?:max|[0-9]+)(?: [0-9]+)?',value)
        return {'readable':True,'value':value}
    except OSError as error:
        return {'readable':False,'errno':error.errno,'errno_name':errno.errorcode.get(error.errno,'UNKNOWN')}
    except AssertionError: return {'readable':True,'format_unrecognized':True}
def proc_pid1_namespace(current_inode):
    try:
        return {'readable':True,'matches_current_namespace':os.stat('/proc/1/ns/pid').st_ino==current_inode}
    except OSError as error:
        return {'readable':False,'matches_current_namespace':None,
                'errno':error.errno,'errno_name':errno.errorcode.get(error.errno,'UNKNOWN')}
try:
    result['kernel_release']=os.uname().release; result['machine']=os.uname().machine
    result['pid_in_current_namespace']=os.getpid()
    result['namespaces']=namespace_ids('self')
    result['proc_pid1_namespace']=proc_pid1_namespace(result['namespaces']['pid'])
    status=open('/proc/self/status').read()
    for key in ('CapEff','Seccomp','Seccomp_filters','NoNewPrivs'):
        value=re.search(r'^'+key+r':\s+(\S+)',status,re.M)
        if value: result['observations'][key]=value.group(1)
    try:
        with open('/proc/self/attr/current','rb') as stream: label=stream.read(4097)
        assert len(label)<=4096
        result['lsm_label_sha256']=hashlib.sha256(label.rstrip(b'\x00\n')).hexdigest()
    except (OSError,AssertionError): result['lsm_label_unavailable']=True
    if result['machine'] in ('x86_64','aarch64'):
        # These Linux architectures use __NR_landlock_create_ruleset=444.
        # VERSION only queries support. No ruleset/policy is installed.
        libc=ctypes.CDLL(None,use_errno=True); libc.syscall.restype=ctypes.c_long
        ctypes.set_errno(0)
        version=libc.syscall(ctypes.c_long(444),ctypes.c_void_p(0),ctypes.c_size_t(0),ctypes.c_uint(1))
        number=ctypes.get_errno() if version<0 else 0
        result['landlock_abi']={'version':version if version>=0 else None,'errno':number,
                               'errno_name':errno.errorcode.get(number,'UNKNOWN') if number else None,
                               'policy_installed':False}
    else: result['landlock_abi']={'unsupported_probe_architecture':True,'policy_installed':False}
    cgroup_path=next(line.split(':',2)[2].strip() for line in open('/proc/self/cgroup') if line.startswith('0::'))
    assert cgroup_path.startswith('/') and '..' not in cgroup_path.split('/'), 'ambiguous cgroup namespace path'
    current='/sys/fs/cgroup'+cgroup_path.rstrip('/')
    result['cgroup_path_resolves_to_mounted_tree']=os.path.isfile(current+'/cgroup.procs')
    if not result['cgroup_path_resolves_to_mounted_tree']: current='/sys/fs/cgroup'
    # If the process path does not resolve, these are mount-root observations,
    # not attestation of the process's effective quota.
    result['cgroup_observation_scope']='current' if result['cgroup_path_resolves_to_mounted_tree'] else 'mount_root_only'
    result['cgroup_mount_readonly']=bool(os.statvfs('/sys/fs/cgroup').f_flag & os.ST_RDONLY)
    result['cgroup_values']={name:read_number_or_max(current+'/'+name) for name in
                            ('pids.max','pids.current','cpu.max','memory.max','memory.swap.max')}
    result['status']='observed'
except BaseException as error:
    result.update(status='failed',error_type=type(error).__name__)
print('SFORGE_SWE_PLATFORM_PROBE='+json.dumps(result,sort_keys=True),flush=True)
sys.exit(0 if result['status']=='observed' else 1)
'''


def platform_isolation_probe_command(*, namespace_mode='outer'):
    """Return a bounded, fixed capability query without changing platform policy.

    Landlock VERSION and cgroup reads do not install isolation. The optional
    mount-free namespace mode only checks the inherited proc view; it never
    executes a tool or returns an execution prefix. Results cannot qualify a
    task. Run only in an owned diagnostic Sandbox and persist exact-ID cleanup.
    """
    if namespace_mode not in ('outer', 'user_net_pid_ipc_without_mount'):
        raise ValueError('Unsupported platform diagnostic namespace mode')
    argv = ['python3', '-I', '-c', _PLATFORM_CAPABILITY_PROBE, namespace_mode]
    if namespace_mode != 'outer':
        argv = ['unshare', '--user', '--map-root-user', '--net', '--pid', '--ipc',
                '--propagation=unchanged', '--fork', '--kill-child=KILL', *argv]
    return shlex.join(argv)


def _checked_json(sdk, sandbox_id, argv, timeout):
    try:
        result = sdk.exec_command(sandbox_id, shlex.join(argv), timeout=timeout)
    except Exception as error:
        raise ToolRuntimeError('SWE tool runtime transport failed: ' + type(error).__name__) from None
    if type(getattr(result, 'returncode', None)) is not int or result.returncode != 0:
        output = getattr(result, 'output', '')
        if isinstance(output, str) and len(output.encode()) <= 32 * 1024:
            failures = [line[len(_FAILURE_MARKER):] for line in output.splitlines() if line.startswith(_FAILURE_MARKER)]
            if len(failures) == 1:
                try:
                    failure = json.loads(failures[0])
                    if (set(failure) == {'phase', 'error_type'}
                            and isinstance(failure['phase'], str) and failure['phase'] in _FAILURE_PHASES
                            and isinstance(failure['error_type'], str) and failure['error_type'] in _FAILURE_TYPES):
                        raise ToolRuntimeError('SWE tool runtime failed at ' + failure['phase'] + ' (' + failure['error_type'] + ')')
                except (TypeError, ValueError, KeyError):
                    pass
        raise ToolRuntimeError('SWE tool runtime namespace or integrity operation failed')
    output = getattr(result, 'output', '')
    if not isinstance(output, str) or len(output.encode()) > 32 * 1024:
        raise ToolRuntimeError('SWE tool runtime produced invalid evidence')
    lines = [line[len(_MARKER):] for line in output.splitlines() if line.startswith(_MARKER)]
    try:
        if len(lines) != 1:
            raise ValueError('unique evidence required')
        value = json.loads(lines[0])
        if not isinstance(value, dict):
            raise ValueError('mapping required')
        return value
    except (TypeError, ValueError):
        raise ToolRuntimeError('SWE tool runtime produced invalid evidence') from None


def prepare_swe_tool_runtime(sdk, sandbox_id, *, setup_timeout=90, affinity_count=None):
    """Start a holder, then prove isolation and tmp persistence across SDK calls.

    Pass this function as Fleet's Controller-only ``runtime_setup`` hook and pin
    ``runtime_setup_id=RUNTIME_SETUP_ID``. An optional affinity limits the number
    of logical CPUs, not CFS CPU quota. All failures raise; there is no ordinary
    exec fallback. Fleet must delete the owned Sandbox if this hook fails.
    """
    if not isinstance(sandbox_id, str) or not sandbox_id or '\x00' in sandbox_id:
        raise ValueError('Invalid Sandbox identity')
    if type(setup_timeout) is not int or not 10 <= setup_timeout <= 300:
        raise ValueError('Setup timeout must be 10..300 seconds')
    if affinity_count is not None and (type(affinity_count) is not int or not 1 <= affinity_count <= 256):
        raise ValueError('Affinity must be a positive logical CPU count')
    identity = uuid.uuid4().hex
    config = {'identity': identity, 'affinity_count': affinity_count,
              'wait_seconds': setup_timeout - 5, 'holder_source': _HOLDER}
    setup = _checked_json(sdk, sandbox_id, ['python3', '-I', '-c', _BOOTSTRAP, json.dumps(config)], setup_timeout)
    try:
        owner = setup['owner']
        expected_path = '/run/sforge-swe-tool/' + identity + '/owner.json'
        if (setup['owner_path'] != expected_path or owner['identity'] != identity
                or owner['schema_version'] != 1 or owner['holder_dumpable'] is not False
                or owner['mount_propagation_private'] is not True
                or owner['job_directory'] != expected_path.rsplit('/', 1)[0]
                or type(owner['pid']) is not int or owner['pid'] <= 1
                or type(owner['pid_starttime']) is not int or owner['pid_starttime'] < 1
                or type(owner['bootstrap_pid']) is not int or owner['bootstrap_pid'] <= 1
                or type(owner['unshare_pid']) is not int or owner['unshare_pid'] <= 1
                or set(owner['namespaces']) != set(_NAMESPACES)
                or set(owner['outer_namespaces']) != set(_NAMESPACES)
                or any(type(owner['namespaces'][name]) is not int or owner['namespaces'][name] < 1
                       or type(owner['outer_namespaces'][name]) is not int or owner['outer_namespaces'][name] < 1
                       or owner['namespaces'][name] == owner['outer_namespaces'][name] for name in _NAMESPACES)):
            raise ValueError('owner identity differs')
        if (set(owner['binaries']) != {'unshare', 'mount', 'nsenter', 'setpriv', 'env', 'python'}
                or any(not isinstance(value, str) or not value.startswith('/') or '\x00' in value
                       for value in owner['binaries'].values())
                or set(owner['environment']) != {'PATH', 'HOME', 'LANG', 'SHELL', 'USER', 'LOGNAME', 'TMPDIR', 'XDG_CACHE_HOME'}
                or any(not isinstance(value, str) or '\x00' in value or len(value) > 8192
                       for value in owner['environment'].values())
                or owner['environment']['HOME'] != '/root' or owner['environment']['TMPDIR'] != '/tmp'
                or not isinstance(owner['holder_affinity'], list) or not owner['holder_affinity']
                or any(type(cpu) is not int or cpu < 0 for cpu in owner['holder_affinity'])
                or (affinity_count is None and owner['affinity'] is not None)
                or (affinity_count is not None and (owner['affinity'] != owner['holder_affinity']
                    or len(owner['holder_affinity']) != affinity_count))):
            raise ValueError('owner execution policy differs')
        payload = json.dumps(owner, sort_keys=True, separators=(',', ':')).encode()
        if setup['owner_sha256'] != hashlib.sha256(payload).hexdigest():
            raise ValueError('owner digest differs')
        python = owner['binaries']['python']
        if not isinstance(python, str) or not python.startswith('/') or '\x00' in python:
            raise ValueError('absolute Controller Python required')
    except (KeyError, TypeError, ValueError):
        raise ToolRuntimeError('SWE tool runtime owner evidence is invalid') from None
    prefix = [python, '-I', '-c', _LAUNCHER, expected_path, setup['owner_sha256']]
    probes = []
    for phase in ('write', 'read'):
        probe = _checked_json(sdk, sandbox_id, [*prefix, python, '-I', '-c', _PROBE,
                                               json.dumps(owner), phase], setup_timeout)
        if (probe.get('phase') != phase or probe.get('namespaces') != owner['namespaces']
                or probe.get('capabilities') != {name: 0 for name in ('CapInh', 'CapPrm', 'CapEff', 'CapBnd', 'CapAmb')}
                or any(type(value) is not int for value in probe.get('capabilities', {}).values())
                or not isinstance(probe.get('affinity'), list) or not probe['affinity']
                or any(type(cpu) is not int or cpu < 0 for cpu in probe['affinity'])
                or (affinity_count is not None and probe['affinity'] != owner['affinity'])
                or any(probe.get(name) is not True for name in (
                    'no_new_privs', 'controller_state_hidden', 'holder_ptrace_denied',
                    'proc_namespace_verified', 'mount_propagation_private', 'network_route_unreachable', 'cgroup2_readonly',
                    'image_home_preserved_readonly', 'testbed_writable'))):
            raise ToolRuntimeError('SWE tool runtime isolation evidence is invalid')
        probes.append(probe)
    return {'exec_prefix': prefix, 'evidence': {
        'schema_version': 1, 'runtime_setup_id': RUNTIME_SETUP_ID,
        'prototype_only': True, 'formal_task_ready': False,
        'owner_path': expected_path, 'owner_sha256': setup['owner_sha256'],
        'holder_pid': owner['pid'], 'holder_pid_starttime': owner['pid_starttime'],
        'bootstrap_pid': owner['bootstrap_pid'], 'unshare_pid': owner['unshare_pid'],
        'namespaces': owner['namespaces'], 'outer_namespaces': owner['outer_namespaces'],
        'controller_state_hidden': True, 'holder_ptrace_denied': True,
        'mount_propagation_private': True,
        'persistent_namespace_and_tmp_verified': True, 'capabilities': probes[-1]['capabilities'],
        'no_new_privs': True, 'network_route_unreachable': True, 'cgroup2_readonly': True,
        'root_policy': 'readonly-system-and-image-home; writable-testbed-and-private-tmp',
        'home': '/root', 'lang': owner['environment']['LANG'],
        'tool_affinity': probes[-1]['affinity'], 'affinity_limit_requested': affinity_count,
        'cpu_cfs_quota_enforced': False, 'pids_cgroup_limit_enforced': False,
        'qualification_gaps': list(_GAPS),
    }}
