"""Publish an audited local image while preserving its OCI content identity.

Task builders retain responsibility for checking public inputs and CPU audits.
This module only transports the approved local image to its selected registry.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import tempfile


DOCKER = ("docker", "--context", "default")
DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
INDEX_TYPES = {"application/vnd.oci.image.index.v1+json",
               "application/vnd.docker.distribution.manifest.list.v2+json"}
MANIFEST_TYPES = {"application/vnd.oci.image.manifest.v1+json",
                  "application/vnd.docker.distribution.manifest.v2+json"}


def docker_hub_tag(tag):
    """Use one destination identity for Docker, skopeo and registry readback."""
    if tag.startswith("docker.io/"):
        tag = tag.removeprefix("docker.io/")
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*/[a-z0-9][a-z0-9._-]*:[A-Za-z0-9_][A-Za-z0-9_.-]*", tag):
        raise ValueError("Expected a Docker Hub namespace/repository:tag")
    return tag


def _write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def skopeo_executable():
    configured = os.environ.get("SFORGE_SKOPEO") or shutil.which("skopeo")
    bundled = Path(__file__).resolve().parents[3] / ".runtime/skopeo-tool/root/usr/bin/skopeo"
    if configured:
        return configured
    if bundled.is_file():
        return str(bundled)
    raise RuntimeError("Install skopeo or set SFORGE_SKOPEO to publish registry-scoped images")


def oci_archive_identity(archive, image_id):
    """Check exported metadata against the approved index or manifest digest.

    Docker with a containerd store reports an OCI index as .Id, rather than a
    config digest. Preserve that index, including attestations, in the transfer.
    Layer integrity is checked by skopeo during the copy, not by this metadata
    reader. No archive members are extracted to the filesystem.
    """
    if not DIGEST.fullmatch(image_id):
        raise ValueError("Invalid audited image digest")
    with tarfile.open(archive, "r:") as tar:
        def read(name):
            member = tar.getmember(name)
            if not member.isfile() or member.size > 16 * 1024 * 1024:
                raise ValueError("Invalid OCI metadata member")
            return tar.extractfile(member).read()

        def blob(descriptor):
            digest = descriptor["digest"]
            if not DIGEST.fullmatch(digest):
                raise ValueError("Unsupported OCI digest")
            raw = read("blobs/sha256/" + digest.split(":", 1)[1])
            if len(raw) != descriptor["size"] or "sha256:" + hashlib.sha256(raw).hexdigest() != digest:
                raise ValueError("Exported OCI metadata digest mismatch")
            return json.loads(raw)

        layout = json.loads(read("oci-layout"))
        if layout.get("imageLayoutVersion") != "1.0.0":
            raise ValueError("Unsupported OCI layout")
        references = json.loads(read("index.json"))["manifests"]
        if len(references) != 1 or references[0]["digest"] != image_id:
            raise ValueError("Export does not contain only the audited image")
        root = references[0]
        document = blob(root)
        if root["mediaType"] in INDEX_TYPES:
            platforms = [entry for entry in document["manifests"]
                         if entry.get("platform", {}).get("os") == "linux"
                         and entry["platform"].get("architecture") == "amd64"]
            if len(platforms) != 1:
                raise ValueError("Expected exactly one linux/amd64 runtime manifest")
            selected = platforms[0]
            manifest = blob(selected)
        elif root["mediaType"] in MANIFEST_TYPES:
            selected, manifest = root, document
        else:
            raise ValueError("Audited image is not an OCI index or manifest")
        config = blob(manifest["config"])
        if config.get("os") != "linux" or config.get("architecture") != "amd64":
            raise ValueError("Exported image has an unexpected runtime platform")
        return {
            "audited_image_id": image_id, "media_type": root["mediaType"],
            "runtime_manifest_digest": selected["digest"],
            "config_digest": manifest["config"]["digest"],
            "layer_digests": [entry["digest"] for entry in manifest["layers"]],
            "rootfs_diff_ids": config["rootfs"]["diff_ids"],
            "os": config["os"], "architecture": config["architecture"],
            "all_index_members_preserved": True,
        }


def publish_audited_image(image_id, tag, *, evidence_dir, transport="auto",
                          source_archive=None, docker=DOCKER):
    """Publish an OCI image; return a verified reference and transport receipt.

    `auto` first uses Docker. A repository-scope failure switches to an OCI
    archive from the immutable local ID, which needs only destination scopes.
    `oci` selects that path directly. Credentials stay in the local auth file.
    """
    if not DIGEST.fullmatch(image_id):
        raise ValueError("Invalid audited image digest")
    tag = docker_hub_tag(tag)
    if transport not in {"auto", "oci"}:
        raise ValueError("Unknown publication transport")
    repository = tag.rsplit(":", 1)[0]
    evidence = Path(evidence_dir)
    evidence.mkdir(parents=True, exist_ok=True)
    inspection = json.loads(subprocess.check_output([*docker, "image", "inspect", image_id]))[0]
    if inspection["Id"] != image_id:
        raise ValueError("Audited local image identity mismatch")
    descriptor = inspection.get("Descriptor", {})
    if descriptor.get("digest") != image_id or descriptor.get("mediaType") not in INDEX_TYPES | MANIFEST_TYPES:
        raise ValueError("Exact publication requires a Docker OCI index/manifest image ID")
    executable = skopeo_executable()
    authfile = Path.home() / ".docker/config.json"
    method, digest, source = None, None, None
    if transport == "auto":
        subprocess.run([*docker, "tag", image_id, tag], check=True)
        response = subprocess.run([*docker, "push", tag], text=True,
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (evidence / "docker-push.log").write_text(response.stdout)
        if response.returncode == 0:
            matches = re.findall(r"digest: (sha256:[0-9a-f]{64})", response.stdout)
            if not matches:
                raise RuntimeError("Docker push did not report an immutable digest")
            digest, method = matches[-1], "docker-push"
        elif "insufficient scopes" not in response.stdout:
            raise RuntimeError("Docker push failed; see publication evidence")
    if digest is None:
        with tempfile.TemporaryDirectory(prefix="sforge-oci-publish-") as temporary:
            directory = Path(temporary)
            archive = Path(source_archive).resolve() if source_archive else directory / "image.tar"
            if source_archive is None:
                subprocess.run([*docker, "save", "--output", str(archive), image_id], check=True)
                archive.chmod(0o600)
            source = oci_archive_identity(archive, image_id)
            _write(evidence / "source-identity.json", source)
            policy = directory / "policy.json"
            _write(policy, {"default": [{"type": "reject"}], "transports": {
                "oci-archive": {str(archive): [{"type": "insecureAcceptAnything"}]}}})
            digestfile = directory / "digest"
            with (evidence / "oci-copy.log").open("w") as output:
                subprocess.run([executable, "--policy", str(policy), "copy", "--all",
                                "--preserve-digests", "--retry-times", "2", "--dest-authfile",
                                str(authfile), "--digestfile", str(digestfile),
                                "oci-archive:" + str(archive), "docker://docker.io/" + tag],
                               stdout=output, stderr=subprocess.STDOUT, check=True)
            digest = digestfile.read_text().strip()
            method = "oci-archive-preserve-digests"
    if digest != image_id:
        raise RuntimeError("Published digest differs from the approved OCI identity")
    reference = repository + "@" + digest
    raw = subprocess.check_output([executable, "inspect", "--raw", "--authfile", str(authfile),
                                   "docker://docker.io/" + reference])
    if "sha256:" + hashlib.sha256(raw).hexdigest() != digest:
        raise RuntimeError("Registry returned a different manifest/index")
    (evidence / "registry-manifest.json").write_bytes(raw)
    receipt = {"schema_version": 1, "audited_image_id": image_id, "tag": tag,
               "reference": reference, "transport": method,
               "registry_digest_verified": True, "rebuilt": False,
               "source_identity": source}
    _write(evidence / "receipt.json", receipt)
    return receipt
