import hashlib
import io
import json
import tarfile

import pytest

from sforge.pyromind.registry_publish import docker_hub_tag, oci_archive_identity, publish_audited_image


def make_archive(tmp_path, *, wrong_config=False, other_platform=False,
                 extra_reference=False, symlink=False):
    files = {}

    def blob(value, kind):
        raw = json.dumps(value).encode()
        digest = hashlib.sha256(raw).hexdigest()
        files['blobs/sha256/' + digest] = raw
        return {'digest': 'sha256:' + digest, 'size': len(raw), 'mediaType': kind}

    config = blob({'os': 'linux', 'architecture': 'amd64',
                   'rootfs': {'diff_ids': ['sha256:' + 'a' * 64]}},
                  'application/vnd.oci.image.config.v1+json')
    manifest = blob({'schemaVersion': 2, 'config': config,
                     'layers': [{'digest': 'sha256:' + 'b' * 64}]},
                    'application/vnd.oci.image.manifest.v1+json')
    manifest['platform'] = {'os': 'linux', 'architecture': 'arm64' if other_platform else 'amd64'}
    index = blob({'schemaVersion': 2, 'manifests': [manifest]},
                 'application/vnd.oci.image.index.v1+json')
    files['index.json'] = json.dumps({'manifests': [index] * (2 if extra_reference else 1)}).encode()
    files['oci-layout'] = b'{"imageLayoutVersion":"1.0.0"}'
    if wrong_config:
        files['blobs/sha256/' + config['digest'].split(':')[1]] = b'{}'
    path = tmp_path / 'image.tar'
    with tarfile.open(path, 'w') as archive:
        for name, value in files.items():
            member = tarfile.TarInfo(name)
            if symlink and name == 'oci-layout':
                member.type, member.linkname = tarfile.SYMTYPE, '/etc/passwd'
                archive.addfile(member)
            else:
                member.size = len(value)
                archive.addfile(member, io.BytesIO(value))
    return path, index, manifest, config


def test_index_identity_is_distinct_from_runtime_config(tmp_path):
    path, index, manifest, config = make_archive(tmp_path)
    identity = oci_archive_identity(path, index['digest'])
    assert identity['audited_image_id'] == index['digest']
    assert identity['runtime_manifest_digest'] == manifest['digest']
    assert identity['config_digest'] == config['digest']
    assert identity['audited_image_id'] != identity['config_digest']
    assert identity['rootfs_diff_ids'] == ['sha256:' + 'a' * 64]
    assert identity['all_index_members_preserved']


def test_export_of_different_image_is_rejected(tmp_path):
    path, *_ = make_archive(tmp_path)
    with pytest.raises(ValueError, match='only the audited image'):
        oci_archive_identity(path, 'sha256:' + 'c' * 64)


def test_changed_config_cannot_use_original_index(tmp_path):
    path, index, *_ = make_archive(tmp_path, wrong_config=True)
    with pytest.raises(ValueError, match='metadata digest mismatch'):
        oci_archive_identity(path, index['digest'])


def test_unreviewed_second_reference_is_rejected(tmp_path):
    path, index, *_ = make_archive(tmp_path, extra_reference=True)
    with pytest.raises(ValueError, match='only the audited image'):
        oci_archive_identity(path, index['digest'])


def test_missing_runtime_platform_is_rejected(tmp_path):
    path, index, *_ = make_archive(tmp_path, other_platform=True)
    with pytest.raises(ValueError, match='linux/amd64'):
        oci_archive_identity(path, index['digest'])


def test_archive_metadata_must_not_be_a_symlink(tmp_path):
    path, index, *_ = make_archive(tmp_path, symlink=True)
    with pytest.raises(ValueError, match='metadata member'):
        oci_archive_identity(path, index['digest'])


def test_explicit_hub_host_has_one_canonical_destination():
    assert docker_hub_tag('docker.io/pyrominddynamics/sforge:worker') == 'pyrominddynamics/sforge:worker'
    assert docker_hub_tag('pyrominddynamics/sforge:worker') == 'pyrominddynamics/sforge:worker'


@pytest.mark.parametrize('tag', ['ghcr.io/org/repo:worker', 'ghcr.io/repo:worker',
                               'localhost:5000/org/repo:worker', 'org/nested/repo:worker'])
def test_other_registry_or_ambiguous_path_is_rejected_before_docker(tmp_path, monkeypatch, tag):
    def unexpected(*args, **kwargs):
        pytest.fail('Invalid destination reached Docker or registry')
    monkeypatch.setattr('sforge.pyromind.registry_publish.subprocess.check_output', unexpected)
    monkeypatch.setattr('sforge.pyromind.registry_publish.subprocess.run', unexpected)
    with pytest.raises(ValueError, match='Docker Hub namespace'):
        publish_audited_image('sha256:' + 'a' * 64, tag, evidence_dir=tmp_path)
