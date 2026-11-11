#!/usr/bin/env python3
"""Author prototype worker; root-created transport and five inherited FDs only.

The parent must admit/seal candidate bytes before this process is launched.
No pathname to the private prompt dataset or model is needed by this worker.
The process is intentionally one prompt long; shared agent Triton cache allows
shape compilation reuse, while L2 warm-up always runs for each target layer.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from single_prompt_archive_v3 import canonical_bytes, stable_stat, validate_bundle
from measure_single_prompt_v3 import measure_prompt


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trusted-root", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--purpose", choices=("private-judge", "public-diagnostic"), required=True)
    args = p.parse_args(argv)
    required_uid = 0 if args.purpose == "private-judge" else 1000
    if os.geteuid() != required_uid:
        raise ValueError("worker setup UID differs from purpose")
    raw = sys.stdin.buffer.read(1024**2 + 1)
    sys.stdin.close()
    if not 0 < len(raw) <= 1024**2:
        raise ValueError("private transport is empty or oversized")
    sys.path.insert(0, str(args.trusted_root / "runtime"))
    sys.path.insert(0, str(args.trusted_root))
    import activation_archive_v3 as archive
    import minference_prefill_runtime as runtime
    import candidate_worker
    value = archive.strict_json(raw)
    bundle, plan = value["bundle"], value["plan"]
    validate_bundle(bundle, plan)
    # Authenticate the exact candidate before importing any candidate code.
    if hashlib.sha256(args.candidate.read_bytes()).hexdigest() != value["candidate_sha256"]:
        raise ValueError("candidate source identity differs")
    handles = {}
    try:
        if set(value["fds"]) != set(bundle["fixtures"]):
            raise ValueError("inherited FD membership differs")
        for fid, descriptor in value["fds"].items():
            fd = descriptor["fd"]
            if type(fd) is not int or fd < 3 or stable_stat(os.fstat(fd)) != descriptor["identity"]:
                raise ValueError("inherited FD identity differs")
            if archive.digest_fd(fd, bundle["fixtures"][fid]["size"]) != bundle["fixtures"][fid]["sha256"]:
                raise ValueError("inherited FD digest differs")
            handles[fid] = archive.ActivationHandle(fid, bundle["fixtures"][fid], fd,
                descriptor["header"], descriptor["payload_offset"])
        import torch
        if torch.__version__ != "2.8.0+cu129":
            raise ValueError("prototype requires the declared Torch 2.8.0+cu129 epoch")
        inventory = runtime.hardware_inventory(torch)
        if str(inventory["uuid"]).lower().removeprefix("gpu-") != bundle["gpu_uuid"].lower().removeprefix("gpu-"):
            raise ValueError("candidate worker GPU differs from preparation GPU")
        result = measure_prompt(args.candidate, bundle=bundle, plan=plan, handles=handles,
            identities={fid: d["identity"] for fid,d in value["fds"].items()},
            policy=value["policy"], policy_sha256=value["policy_sha256"],
            expected_candidate_sha256=value["candidate_sha256"], nonce=value["nonce"],
            runtime=runtime, archive=archive, torch=torch,
            before_candidate_import=(candidate_worker._drop_to_agent if args.purpose == "private-judge" else None),
            purpose=args.purpose)
        # The private nonce has no public/result field and never enters argv.
        sys.stdout.buffer.write(canonical_bytes(result)); sys.stdout.flush()
        return 0
    finally:
        for h in handles.values(): h.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        # Candidate exceptions must not leak private transport in stderr.
        print("single-prompt worker failed closed", file=sys.stderr)
        raise SystemExit(1)
