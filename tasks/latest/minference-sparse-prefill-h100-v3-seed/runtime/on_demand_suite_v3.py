#!/usr/bin/env python3
"""Common on-demand kernel entrypoint; public diagnostic or protected Judge.

Model/preparation runs in this process; candidate executes in a separate child.
This executable owns no GPU allocation API: trusted controller must provide an
actual image/device binding and monitor the whole owned container's PID set.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import uuid

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from single_prompt_archive_v3 import (PROTOCOL, on_demand_plan, canonical_sha,
    validate_plan, validate_purpose)
from prepare_single_prompt_v3 import TrustedPromptPreparer, write_private_json
from stream_parent_v3 import run_stream


def load_fixed_inputs(root, plan, recipe_path, recipe_loader):
    """Exact token/text hashes, 36 distinct prompts and frozen input bindings."""
    prompts, manifest = recipe_loader.load_prompts(root, expected_partition=plan["partition"])
    if {p["id"] for p in prompts} != {f["prompt_id"] for f in plan["fixtures"].values()}:
        raise ValueError("fixed prompt identities differ")
    bindings = plan["input_bindings"]
    actual = {"prompt_manifest_sha256": hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest(),
        "prompts_sha256": manifest["prompts_sha256"],
        "recipe_sha256": hashlib.sha256(recipe_path.read_bytes()).hexdigest()}
    if bindings != actual:
        raise ValueError("fixed prompt/model recipe binding differs")
    from prepare_single_prompt_v3 import verify_prompt
    for p in prompts: verify_prompt(plan, p)
    return prompts, manifest


def envelope(result, *, plan, manifest_sha256, policy_sha256, hardware, bundles, purpose):
    if len(bundles) != 36 or len(result["cases"]) != 48:
        raise ValueError("on-demand suite was not complete")
    return {"schema_version": 3, "protocol": "minference-real-ruler-prefill-v3",
        "suite_type": "final" if purpose == "private-judge" else "public",
        "manifest_sha256": manifest_sha256, "numerical_policy_sha256": policy_sha256,
        "candidate_sha256": result["candidate_sha256"], "hardware": hardware,
        "ok": result["ok"], "score_enabled": purpose == "private-judge",
        "score": result["raw_primary_tflops"], "raw_primary_tflops": result["raw_primary_tflops"],
        "cases": result["cases"], "boundary_checks":result["boundary_checks"], "on_demand": {"protocol": PROTOCOL,
            "purpose": purpose, "parent_execution_uid": 0 if purpose == "private-judge" else 1000,
            "candidate_execution_uid": 1000,
            "candidate_privilege_drop": purpose == "private-judge",
            "execution_plan_sha256": canonical_sha(plan),
            "scratch_bytes_max": max(b["scratch_bytes"] for b in bundles),
            "preparation_seconds_total": sum(b["preparation_seconds"] for b in bundles),
            "preparation_timed_in_score": False, "persistent_qkv_files": 0,
            "preparation_bundles": bundles},
        "evaluation_kind": "kernel-only-v3",
        "qualified": False}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--purpose", choices=("private-judge", "public-diagnostic"), required=True)
    p.add_argument("--trusted-root", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--prompts", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--numerical-policy", type=Path, required=True)
    p.add_argument("--context", type=Path, required=True)
    p.add_argument("--scratch-parent", type=Path, required=True)
    p.add_argument("--run-output", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--worker-timeout", type=float, default=600)
    args = p.parse_args(argv)
    global FAILURE_OUTPUT, FAILURE_RUN_OUTPUT
    FAILURE_OUTPUT, FAILURE_RUN_OUTPUT = args.output, args.run_output
    sys.path.insert(0, str(args.trusted_root / "activations/recipe"))
    import activation_archive_v3 as archive
    import minference_prefill_runtime as runtime
    import ruler_v3 as recipes
    import capture_activations as capture
    from quality_v3 import file_sha256
    raw = sys.stdin.buffer.read(1024**2 + 1); sys.stdin.close()
    transport = archive.strict_json(raw)
    if len(raw) > 1024**2 or set(transport) != {"candidate_sha256", "manifest_sha256", "numerical_policy_sha256", "evaluation_nonce"}:
        raise ValueError("kernel transport schema differs")
    fixed_plan, manifest_sha = archive.read_json(args.plan)
    validate_plan(fixed_plan)
    validate_purpose(args.purpose, fixed_plan["partition"], os.geteuid())
    if manifest_sha != transport["manifest_sha256"]:
        raise ValueError("fixed plan digest differs")
    policy, policy_sha = runtime.load_numerical_policy(args.numerical_policy)
    if policy_sha != transport["numerical_policy_sha256"]:
        raise ValueError("numeric policy digest differs")
    if file_sha256(args.candidate) != transport["candidate_sha256"]:
        raise ValueError("approved candidate differs")
    context, _ = archive.read_json(args.context)
    if set(context) != {"image_id", "gpu_uuid", "purpose"} or context["purpose"] != args.purpose:
        raise ValueError("actual kernel context differs")
    if args.purpose == "private-judge":
        info = args.context.lstat()
        if info.st_uid != 0 or info.st_nlink != 1 or stat.S_IMODE(info.st_mode) != 0o400:
            raise ValueError("private kernel context is not root-protected")
    recipe_path = args.trusted_root / "activations/recipe/RULER_V3_RECIPE.json"
    recipe, _ = archive.read_json(recipe_path)
    prompts, prompt_manifest = load_fixed_inputs(args.prompts, fixed_plan, recipe_path, recipes)
    import torch
    import transformers
    if torch.__version__ != "2.8.0+cu129" or transformers.__version__ != "4.57.1":
        raise ValueError("HF/Torch on-demand runtime epoch differs")
    hardware = runtime.hardware_inventory(torch)
    bindings = recipes.capture_bindings(recipe, prompt_manifest, args.prompts,
        image_id=context["image_id"], software={"torch": torch.__version__, "cuda": torch.version.cuda,
        "transformers": transformers.__version__})
    bindings.update(attention_backend="flash_attention_2", flash_attention_version="2.8.3",
        capture_helper_sha256=file_sha256(Path(capture.__file__)),
        on_demand_sources={name:file_sha256(HERE / name) for name in ("single_prompt_archive_v3.py", "prepare_single_prompt_v3.py", "measure_single_prompt_v3.py", "run_single_prompt_worker_v3.py", "stream_parent_v3.py")},
        on_demand_suite_sha256=file_sha256(Path(__file__)))
    plan = on_demand_plan(fixed_plan, bindings)
    preparer = TrustedPromptPreparer(args.model, recipe, prompt_manifest, plan,
        image_id=context["image_id"], expected_gpu_uuid=context["gpu_uuid"],
        capture_helper=capture, torch=torch, transformers=transformers, purpose=args.purpose)
    # Seal an exact snapshot with agent-readable traversal, root read-only for
    # private Judge. Never ask the child to traverse a private 0700 ancestor.
    stage = args.run_output.parent / ("approved-on-demand-" + uuid.uuid4().hex)
    stage.mkdir(mode=0o711)
    if args.purpose == "private-judge" and any(parent.stat().st_mode & 0o001 == 0 for parent in stage.parents):
        raise ValueError("sealed source ancestor is not traversable by candidate UID")
    candidate_snapshot = stage / "submission.py"
    source = args.candidate.read_bytes()
    if hashlib.sha256(source).hexdigest() != transport["candidate_sha256"]:
        raise ValueError("candidate changed before byte snapshot")
    fd = os.open(candidate_snapshot, os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW, 0o444)
    with os.fdopen(fd,"wb") as stream:
        stream.write(source); stream.flush(); os.fsync(stream.fileno())
    os.chmod(candidate_snapshot,0o444)
    result = run_stream(preparer, prompts, plan=plan, candidate=candidate_snapshot,
        candidate_sha256=transport["candidate_sha256"], policy=policy, policy_sha256=policy_sha,
        archive=archive, trusted_root=args.trusted_root, scratch_parent=args.scratch_parent,
        output=args.run_output, worker_timeout=args.worker_timeout, purpose=args.purpose,
        nonce=transport["evaluation_nonce"])
    bundles = [archive.read_json(args.run_output / f"prompt-{i:02d}-bundle.json")[0] for i in range(36)]
    report = envelope(result, plan=plan, manifest_sha256=manifest_sha,
        policy_sha256=policy_sha, hardware=hardware, bundles=bundles, purpose=args.purpose)
    write_private_json(args.output, report)
    return 0


if __name__ == "__main__":
    try: raise SystemExit(main())
    except Exception as error:
        output=globals().get('FAILURE_OUTPUT');root=globals().get('FAILURE_RUN_OUTPUT')
        reports=[]
        if root is not None and root.is_dir():
            for path in sorted(root.glob('prompt-*-report.json')):
                if path.is_file() and path.stat().st_size<=1024**2:reports.append(json.loads(path.read_bytes()))
        if output is not None and not output.exists():
            write_private_json(output,{'ok':False,'failure_category':'correctness','error_type':type(error).__name__,
                'private_message':str(error)[:1024],'completed_prompt_count':len(reports),
                'prompt_reports':reports,'qualified':False})
        print("on-demand kernel preparation/measurement failed closed", file=sys.stderr)
        raise SystemExit(1)
