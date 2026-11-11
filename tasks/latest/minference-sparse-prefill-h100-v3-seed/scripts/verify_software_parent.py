#!/usr/bin/env python3
"""Check a software parent image against the recorded clean parent's package files.

    scripts/verify_software_parent.py IMAGE [--receipt FILE]

references/software-parent/clean-software-files.json is the file manifest the
2026-10-05 clean parent (sha256:603d5163...) was measured with; its SHA-256 is
bound by original-file-equivalence-receipt.json (clean_file_manifest_sha256).
An image built from images/parent/Dockerfile must hold the same files under
site-packages with the same SHA-256.  Bytecode and the installer's own records
(RECORD, INSTALLER, REQUESTED, direct_url.json) depend on how a package was
installed, not on what it is, and are not compared.  The image must also start
with the official Torch base's layers (OFFICIAL_TORCH_IMAGE_ID, present locally).

With --receipt, a passing check writes the receipt scripts/build_images.sh takes
as SOFTWARE_PARENT_VERIFICATION for a parent other than the recorded 603d5163:
the image ID and layers, the Dockerfile and manifest digests, and the counts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "references/software-parent"
SITE = "/opt/conda/lib/python3.11/site-packages"
INSTALLER_RECORDS = {"RECORD", "INSTALLER", "REQUESTED", "direct_url.json"}
OFFICIAL_TORCH_IMAGE_ID = "sha256:cf5aa3f7045a68c10d80f546746591c5ccae6a33729e5e32625ff76bd2c036fe"
RECEIPT_PROTOCOL = "minference-rebuilt-software-parent-v1"

LISTING = r"""
import hashlib, json, os, sys
site, roots = sys.argv[1], json.loads(sys.argv[2])
out = {}
for root in roots:
    path = os.path.join(site, root)
    files = [path] if os.path.isfile(path) else [os.path.join(d, f) for d, _, fs in os.walk(path) for f in fs]
    for name in files:
        with open(name, "rb") as handle:
            out[os.path.relpath(name, site)] = hashlib.sha256(handle.read()).hexdigest()
print(json.dumps(out))
"""


def compared(path: str) -> bool:
    return not path.endswith(".pyc") and path.rsplit("/", 1)[-1] not in INSTALLER_RECORDS


def _inspect(image: str) -> dict:
    return json.loads(subprocess.run(["docker", "image", "inspect", image], check=True,
                                     capture_output=True, text=True).stdout)[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image")
    parser.add_argument("--receipt", type=Path, help="write the verification receipt here (a new file)")
    args = parser.parse_args()
    if args.receipt is not None and (args.receipt.exists() or not args.receipt.is_absolute()):
        parser.error("--receipt must be a new absolute path")
    image = args.image
    inspected = _inspect(image)
    layers = inspected["RootFS"]["Layers"]
    official = _inspect(OFFICIAL_TORCH_IMAGE_ID)["RootFS"]["Layers"]
    if layers[:len(official)] != official:
        print(f"{image} does not start with the official Torch base's {len(official)} layers", file=sys.stderr)
        return 1
    manifest_raw = (REFERENCE / "clean-software-files.json").read_bytes()
    receipt = json.loads((REFERENCE / "original-file-equivalence-receipt.json").read_text(encoding="utf-8"))
    if hashlib.sha256(manifest_raw).hexdigest() != receipt["clean_file_manifest_sha256"]:
        print("clean-software-files.json is not the manifest the receipt binds", file=sys.stderr)
        return 1
    expected = {f["path"]: f["sha256"] for f in json.loads(manifest_raw)["files"] if compared(f["path"])}
    roots = sorted({path.split("/", 1)[0] for path in expected})
    listing = subprocess.run(
        ["docker", "run", "--rm", "--pull=never", "--network", "none", "--entrypoint", "python3",
         image, "-c", LISTING, SITE, json.dumps(roots)],
        check=True, capture_output=True, text=True,
    ).stdout
    actual = {path: digest for path, digest in json.loads(listing).items() if compared(path)}
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    changed = sorted(path for path in set(expected) & set(actual) if expected[path] != actual[path])
    if missing or extra or changed:
        print(json.dumps({"missing": missing[:20], "extra": extra[:20], "changed": changed[:20],
                          "counts": [len(missing), len(extra), len(changed)]}, indent=1))
        return 1
    print(f"software parent matches the recorded clean parent: {len(expected)} package files identical")
    if args.receipt is not None:
        receipt_body = {
            "protocol": RECEIPT_PROTOCOL,
            "status": "identical",
            "image_id": inspected["Id"],
            "rootfs_layers": layers,
            "official_base_image_id": OFFICIAL_TORCH_IMAGE_ID,
            "official_base_layer_count": len(official),
            "dockerfile_sha256": hashlib.sha256((ROOT / "images/parent/Dockerfile").read_bytes()).hexdigest(),
            "clean_file_manifest_sha256": receipt["clean_file_manifest_sha256"],
            "compared_package_files": len(expected),
            "not_compared": ["*.pyc", *sorted(INSTALLER_RECORDS)],
            "qualified": False,
        }
        fd = os.open(args.receipt, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "w") as stream:
            json.dump(receipt_body, stream, indent=2, sort_keys=True)
            stream.write("\n")
        print(f"wrote {args.receipt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
