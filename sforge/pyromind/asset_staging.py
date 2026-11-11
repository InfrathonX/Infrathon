"""Materialize a pinned file selection without copying cached weights on one FS."""
from __future__ import annotations

import errno
import hashlib
import os
from pathlib import Path
import shutil
import stat
import tempfile

from sforge.pyromind.assets import _manifest_bytes, _relative_path, CHUNK_SIZE
from sforge.pyromind.asset_catalog import asset_id


def stage_files(source_root, files, staging_root, *, hf_repo_root=None):
    """Hash selected local bytes and link them into a controller-only staging tree.

    HF snapshot links must resolve inside that repository's blobs directory.
    No chmod is applied to hardlinks (which would mutate the cache). Cross-device
    copies are reported explicitly. The uploader independently revalidates every
    byte and detects concurrent source modification before publishing ready.
    """
    source_root = Path(source_root).expanduser().absolute()
    hf_repo_root = Path(hf_repo_root).expanduser().resolve() if hf_repo_root else None
    resolved, entries = {}, []
    for row in files:
        name = _relative_path(row['path'])
        requested = source_root / name
        source = requested.resolve(strict=True)
        if requested.is_symlink():
            if hf_repo_root is None or not source.is_relative_to(hf_repo_root / 'blobs'):
                raise ValueError('Asset link escapes its pinned model repository')
        elif any(p.is_symlink() for p in requested.parents):
            raise ValueError('Asset source parent is a symlink')
        info = source.stat()
        if not stat.S_ISREG(info.st_mode):
            raise ValueError('Asset source is not a regular file')
        digest = hashlib.sha256()
        with source.open('rb') as stream:
            for chunk in iter(lambda: stream.read(CHUNK_SIZE), b''):
                digest.update(chunk)
        after = source.stat()
        if (info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns) != (
                after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise ValueError('Asset source changed while hashing')
        sha = digest.hexdigest()
        if ((row.get('size') is not None and row['size'] != info.st_size)
                or (row.get('sha256') is not None and row['sha256'] != sha)):
            raise ValueError('Pinned asset bytes do not match: ' + name)
        entries.append({'path': name, 'size': info.st_size, 'sha256': sha})
        resolved[name] = source
    manifest, _ = _manifest_bytes({'schema_version': 1, 'files': entries})
    destination = Path(staging_root).expanduser().absolute() / asset_id(manifest)
    destination.mkdir(parents=True, mode=0o700, exist_ok=True)
    if destination.is_symlink() or any(p.is_symlink() for p in destination.parents):
        raise ValueError('Staging root must not contain symlinks')
    linked_bytes = copied_bytes = 0
    for entry in manifest['files']:
        target = destination / entry['path']
        target.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
        if target.is_symlink() or any(p.is_symlink() for p in target.parents):
            raise ValueError('Unsafe staged asset path')
        if target.exists():
            # A later upload rehashes all bytes; never overwrite an existing view.
            continue
        try:
            os.link(resolved[entry['path']], target, follow_symlinks=False)
            linked_bytes += entry['size']
        except OSError as error:
            if error.errno != errno.EXDEV:
                raise
            temporary = None
            try:
                with resolved[entry['path']].open('rb') as reader, tempfile.NamedTemporaryFile(
                        dir=target.parent, prefix='.partial-', delete=False) as writer:
                    temporary = Path(writer.name)
                    shutil.copyfileobj(reader, writer, length=CHUNK_SIZE)
                    writer.flush()
                    os.fsync(writer.fileno())
                # Same-directory link publishes complete bytes without overwriting
                # another staging writer. The normal uploader verifies the winner.
                try:
                    os.link(temporary, target, follow_symlinks=False)
                except FileExistsError:
                    pass
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
            copied_bytes += entry['size']
    return destination, manifest, {'hardlinked_bytes': linked_bytes, 'copied_bytes': copied_bytes}
