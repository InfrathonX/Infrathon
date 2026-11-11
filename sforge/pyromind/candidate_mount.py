"""Controller-owned references to one immutable, persisted candidate directory.

These references are never accepted as Worker-supplied paths. The controller
resolves an authenticated version ID and verifies its Storage objects before
constructing one. A Judge only receives that exact version directory.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import PurePosixPath


CANDIDATE_MOUNT_PATH = PurePosixPath('/mnt/sforge-candidate')
_ID = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,199}')
_VERSION_ID = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,254}')
_SHA256 = re.compile(r'[0-9a-f]{64}')


@dataclass(frozen=True)
class MountedCandidateArchive:
    task_id: str
    run_id: str
    version_id: str
    host_path: str
    archive_sha256: str
    archive_size: int
    manifest_sha256: str

    def __post_init__(self):
        for name, pattern in (('task_id', _ID), ('run_id', _ID),
                              ('version_id', _VERSION_ID)):
            value = getattr(self, name)
            if not isinstance(value, str) or pattern.fullmatch(value) is None:
                raise ValueError('Invalid mounted candidate ' + name)
        expected = (
            '/workspace/sforge-pyromind/' + self.run_id
            + '/version-store/tasks/' + self.task_id + '/runs/' + self.run_id
            + '/versions/' + self.version_id
        )
        if self.host_path != expected:
            raise ValueError('Candidate mount must select the exact owned version directory')
        for name in ('archive_sha256', 'manifest_sha256'):
            value = getattr(self, name)
            if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
                raise ValueError('Invalid mounted candidate ' + name)
        if (isinstance(self.archive_size, bool)
                or not isinstance(self.archive_size, int)
                or not 1 <= self.archive_size <= 128 * 1024 * 1024):
            raise ValueError('Invalid mounted candidate archive size')

    @property
    def archive_path(self) -> PurePosixPath:
        return CANDIDATE_MOUNT_PATH / 'submission.tar.gz'

    def validate_archive(self, archive: bytes, *, task_id: str) -> None:
        if self.task_id != task_id:
            raise ValueError('Mounted candidate belongs to a different task')
        if (len(archive) != self.archive_size
                or hashlib.sha256(archive).hexdigest() != self.archive_sha256):
            raise ValueError('Mounted candidate does not match the submitted archive')

    def verification_command(self) -> list[str]:
        return ['python3', '-c', MOUNT_VERIFICATION_SOURCE,
                json.dumps(asdict(self), sort_keys=True)]


# Kept self-contained so the same program runs in pre-existing task images;
# installing a new Python package or exposing Storage credentials is unnecessary.
# Missing newly mounted objects may become visible shortly. Other integrity
# failures must stop immediately, never trigger a transport fallback.
MOUNT_VERIFICATION_SOURCE = r'''import hashlib,json,os,pathlib,stat,sys,time
reference=json.loads(sys.argv[1])
mount=pathlib.Path('/mnt/sforge-candidate')
deadline=time.monotonic()+30
def read_regular(directory,name,expected_size=None,limit=None):
    fd=os.open(name,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=directory)
    try:
        info=os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise RuntimeError('Mounted candidate input is not a regular file')
        if expected_size is not None and info.st_size!=expected_size:
            raise RuntimeError('Mounted candidate archive size mismatch')
        if limit is not None and info.st_size>limit:
            raise RuntimeError('Mounted candidate manifest exceeds size limit')
        digest=hashlib.sha256()
        parts=[]
        size=0
        while True:
            chunk=os.read(fd,1024*1024)
            if not chunk: break
            size+=len(chunk)
            if (expected_size is not None and size>expected_size) or (limit is not None and size>limit):
                raise RuntimeError('Mounted candidate input changed size')
            digest.update(chunk)
            if limit is not None: parts.append(chunk)
        if expected_size is not None and size!=expected_size:
            raise RuntimeError('Mounted candidate archive read was incomplete')
        return digest.hexdigest(),b''.join(parts)
    finally:
        os.close(fd)
while True:
    directory=None
    try:
        if mount.resolve(strict=True)!=mount:
            raise RuntimeError('Mounted candidate directory traverses a symlink')
        directory=os.open(str(mount),os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
        if not os.fstatvfs(directory).f_flag & os.ST_RDONLY:
            raise RuntimeError('Mounted candidate directory is not read-only')
        manifest_digest,manifest_bytes=read_regular(directory,'manifest.json',limit=4*1024*1024)
        if manifest_digest!=reference['manifest_sha256']:
            raise RuntimeError('Mounted candidate manifest digest mismatch')
        manifest=json.loads(manifest_bytes)
        for key in ('task_id','run_id','version_id','archive_sha256','archive_size'):
            if manifest.get(key)!=reference[key]:
                raise RuntimeError('Mounted candidate manifest identity mismatch')
        if manifest.get('task_directory')!=reference['task_id']:
            raise RuntimeError('Mounted candidate task directory mismatch')
        archive_digest,_=read_regular(directory,'submission.tar.gz',expected_size=reference['archive_size'])
        if archive_digest!=reference['archive_sha256']:
            raise RuntimeError('Mounted candidate archive digest mismatch')
        print(json.dumps({'archive_sha256':archive_digest,'manifest_sha256':manifest_digest,
                          'archive_size':reference['archive_size'],'read_only':True,
                          'version_id':reference['version_id']}))
        break
    except FileNotFoundError:
        if time.monotonic()>=deadline:
            raise RuntimeError('Mounted candidate did not become visible before deadline') from None
        time.sleep(1)
    finally:
        if directory is not None: os.close(directory)
'''
