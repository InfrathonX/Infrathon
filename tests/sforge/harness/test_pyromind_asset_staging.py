import hashlib
import errno
import os
from pathlib import Path

import pytest

from sforge.pyromind.asset_staging import stage_files


def test_hf_snapshot_stages_without_copying_or_changing_cache_permissions(tmp_path):
    repo = tmp_path / 'models--test--model'
    (repo / 'blobs').mkdir(parents=True)
    snapshot = repo / 'snapshots' / ('a' * 40)
    snapshot.mkdir(parents=True)
    blob = repo / 'blobs' / 'weights'; blob.write_bytes(b'weights'); blob.chmod(0o444)
    (snapshot / 'model.bin').symlink_to('../../blobs/weights')
    files = [{'path': 'model.bin', 'size': 7, 'sha256': hashlib.sha256(b'weights').hexdigest()}]
    stage, manifest, stats = stage_files(snapshot, files, tmp_path / 'stage', hf_repo_root=repo)
    assert os.stat(stage / 'model.bin').st_ino == os.stat(blob).st_ino
    assert blob.stat().st_mode & 0o777 == 0o444
    assert not (stage / 'model.bin').is_symlink()
    assert stats == {'hardlinked_bytes': 7, 'copied_bytes': 0}
    assert manifest['files'] == files


def test_snapshot_link_cannot_escape_repository(tmp_path):
    repo = tmp_path / 'repo'; repo.mkdir()
    outside = tmp_path / 'secret'; outside.write_text('never upload')
    (repo / 'model.bin').symlink_to(outside)
    with pytest.raises(ValueError, match='escapes'):
        stage_files(repo, [{'path': 'model.bin'}], tmp_path / 'stage', hf_repo_root=repo)
    assert not (tmp_path / 'stage').exists()


def test_wrong_pinned_digest_prevents_staging(tmp_path):
    source = tmp_path / 'source'; source.mkdir()
    (source / 'model.bin').write_bytes(b'wrong')
    with pytest.raises(ValueError, match='do not match'):
        stage_files(source, [{'path': 'model.bin', 'sha256': 'a' * 64}], tmp_path / 'stage')
    assert not (tmp_path / 'stage').exists()


def test_interrupted_cross_device_copy_can_be_retried(tmp_path, monkeypatch):
    import sforge.pyromind.asset_staging as staging
    source = tmp_path / 'source'; source.mkdir()
    (source / 'weights').write_bytes(b'complete weights')
    real_link, real_copy = staging.os.link, staging.shutil.copyfileobj
    def cross_device(src, dst, **kwargs):
        if Path(src).parent == source:
            raise OSError(errno.EXDEV, 'cross device')
        return real_link(src, dst, **kwargs)
    def interrupted(reader, writer, **kwargs):
        writer.write(b'partial')
        raise OSError(errno.ENOSPC, 'disk full')
    monkeypatch.setattr(staging.os, 'link', cross_device)
    monkeypatch.setattr(staging.shutil, 'copyfileobj', interrupted)
    with pytest.raises(OSError):
        stage_files(source, [{'path': 'weights'}], tmp_path / 'stage')
    assert not list((tmp_path / 'stage').rglob('weights'))
    monkeypatch.setattr(staging.shutil, 'copyfileobj', real_copy)
    dest, _, stats = stage_files(source, [{'path': 'weights'}], tmp_path / 'stage')
    assert (dest / 'weights').read_bytes() == b'complete weights'
    assert stats['copied_bytes'] == len(b'complete weights')
