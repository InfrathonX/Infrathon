#!/usr/bin/env python3
"""CPU-only static validator for the SDV2 dual-GPU task shell (v2)."""
from __future__ import annotations
import argparse, hashlib, json, os, re
from pathlib import Path
import yaml

TASK_DIRECTORY="streamdiffusionv2-dual-h100-pipeline-runtime-v2-seed"
TASK_ID="streamdiffusionv2_dual_h100_pipeline_runtime_seed_v2"
PROTOCOL_ID="sdv2-pipeline-v2-full-final"
FORMAL_CASE_COUNT=12
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def prefix(p):
    lines=p.read_text().splitlines(keepends=True)
    for i,line in enumerate(lines):
        if line.startswith("# ---- divergence marker:"): return "".join(lines[:i])
    return ""
def judge_asset(root: Path, build_relative: str, asset_relative: str):
    """A Judge-only input: the build/ copy first, else the dataset copy at
    $INFRATHON_ASSETS_ROOT/<task>/<asset path> (assets/fetch.sh), else None."""
    local=root/build_relative
    if local.exists(): return local
    assets=Path(os.environ.get("INFRATHON_ASSETS_ROOT") or Path.home()/".cache/infrathon/assets")
    fetched=assets/root.name/asset_relative
    return fetched if fetched.exists() else None
def validate(root: Path):
    errors=[]
    required=("README.md",".gitignore","tasks/BENCHMARK.yaml",f"tasks/{TASK_ID}.json","runtime/__init__.py","runtime/README.md","runtime/media_gen.py","runtime/cases.py","runtime/evaluation.py","runtime/hardware.py","runtime/manifests/public_cases.json","runtime/sdv2_core/UPSTREAM_PROVENANCE.md","images/work/Dockerfile","images/work/.dockerignore","images/work/AGENT_TASK.md","images/work/codex-config.toml","images/work/benchmark/public_runner.py","images/work/benchmark/candidate_worker.py","images/work/benchmark/source_admission.py","images/work/benchmark/run_public_tests.sh","images/judge/Dockerfile","images/judge/.dockerignore","images/judge/run_judge.py","images/judge/verify_sdv2_pipeline.py","images/judge/candidate_worker.py","images/judge/source_admission.py","images/judge/protected_hashes/h100-94gb-400w.txt","images/judge/protected_hashes/h100-80gb-700w.txt","judge/run_judge.py","oracle/README.md","oracle/FINAL_VERSION","oracle/candidates/runtime_v001/runtime.py","scripts/lib.sh","scripts/build_images.sh","scripts/generate_goldens.py","scripts/write_suite_manifests.py","scripts/generate_protected_hashes.py","scripts/validate_layout.py","scripts/stage_assets.py","scripts/fetch_clips.py","assets/prompts/public_prompts.json","assets/clips/clips_manifest.json","assets/clips/THIRD_PARTY_NOTICES.md","assets/weights/checkpoint_manifest.sha256","assets/upstream/smoke_load.py","tests/test_contract.py","tests/test_capability_envelope.py","tests/test_seed_reference.py")
    required += ("runtime/evaluation_protocol.json", "portable/selection-protocol.json", "portable/formal-suite-manifest.json")
    if root.name != TASK_DIRECTORY: errors.append("wrong task directory name")
    for rel in required:
        p=root/rel
        if not p.is_file() or p.is_symlink(): errors.append(f"missing regular file: {rel}")
    try:
        spec=json.loads((root/"tasks"/f"{TASK_ID}.json").read_text())
        expected={"task_id":TASK_ID,"cwd":"/app/workdir/sdv2_pipeline_lab","submit_paths":["candidates","FINAL_VERSION","notes.md"]}
        for key,value in expected.items():
            if spec.get(key)!=value: errors.append(f"task spec {key} mismatch")
        if spec.get("version_enumeration",{}).get("directory_pattern") != "runtime_v([0-9]{3})": errors.append("candidate directory pattern mismatch")
        profile=spec.get("portable",{}).get("execution_profiles",{}).get(spec.get("portable",{}).get("defaults",{}).get("execution_profile_id",""))
        if not isinstance(profile,dict) or profile.get("count")!=2 or not profile.get("min_memory_bytes") or not profile.get("min_compute_capability"):
            errors.append("task spec is missing a portable capability envelope")
        portable=spec["portable"]
        # Envelope form, like the other dual-GPU tasks: the harness's strict
        # full-contract parser runs only when task_core/evaluation_suites appear.
        if "evaluation_suites" in portable or "task_core" in portable: errors.append("portable block must stay in envelope form")
        formal=json.loads((root/"portable/formal-suite-manifest.json").read_text())
        selection=json.loads((root/"portable/selection-protocol.json").read_text())
        policy=json.loads((root/"runtime/evaluation_protocol.json").read_text())
        # The formal suite manifest and goldens are generated (build/) or fetched from the
        # dataset; the suite identity is checked whenever the manifest is at hand.
        final_cases=judge_asset(root,"build/private/final_cases.json","judge/final_cases.json")
        private=digest(final_cases) if final_cases else formal["selected_manifest_sha256"]
        if not (formal["selected_manifest_sha256"] == policy["selected_manifest_sha256"] == private == selection["final"]["selected_manifest_sha256"] == selection["interactive"]["selected_manifest_sha256"]): errors.append("selected private suite identity mismatch")
        if policy["case_count"] != FORMAL_CASE_COUNT or policy["protocol_id"] != PROTOCOL_ID or formal["evaluation_protocol_id"] != PROTOCOL_ID or selection["protocol_id"] != PROTOCOL_ID: errors.append("formal protocol version/count mismatch")
        if not (policy["timeouts"] == formal["timeouts"] == selection["timeouts"]): errors.append("formal timeouts differ between protocol files")
        if policy["timeouts"]["outer_judge_seconds"] != spec["judge"]["eval_timeout"]: errors.append("formal Judge timeout mismatch")
    except Exception as exc: errors.append(f"invalid task JSON: {exc}")
    for rel in ("build/goldens_judge/final/manifest.json",):
        # A host-generated build (scripts/write_suite_manifests.py) must be complete.
        if (root/"build/private/final_cases.json").is_file() and not (root/rel).is_file(): errors.append(f"missing regular file: {rel}")
    # Internal run configs are optional; when present they must keep auto eval off.
    for experiment in sorted(root.glob("experiment-*.yaml")):
        try:
            exp=yaml.safe_load(experiment.read_text())
            if exp.get("defaults",{}).get("disable_auto_eval") is not True: errors.append(f"auto eval must be disabled: {experiment.name}")
        except Exception as exc: errors.append(f"invalid experiment YAML {experiment.name}: {exc}")
    readme=(root/"README.md").read_text()
    if "## Evaluation submission policy" in readme:
        for text in ("final_archive.tar.gz","submission.tar.gz"):
            if text not in readme: errors.append(f"README missing {text}")
    # In-tree evaluation inputs: 6 public and 12 hidden prompt embeddings; the
    # hidden ones live on the Judge-only path and must never be copied by the
    # Work Dockerfile (its ``assets`` context is the public work-context/ subset).
    if len([p for p in (root/"assets/embeddings/public").glob("*.pt") if p.is_file() and not p.is_symlink()]) != 6: errors.append("expected 6 public prompt embeddings under assets/embeddings/public")
    hidden=judge_asset(root,"judge/private/embeddings/hidden","judge/embeddings/hidden")
    if hidden is not None and len([p for p in hidden.glob("*.pt") if p.is_file() and not p.is_symlink()]) != 12: errors.append(f"expected 12 hidden prompt embeddings under {hidden}")
    work=root/"images/work/Dockerfile"; judge=root/"images/judge/Dockerfile"
    for token in ("embeddings/hidden","sintel_square_513_cut","tears_square576_257","tears_portrait_513_cut","final_cases"):
        if token in work.read_text(): errors.append(f"Work Dockerfile references a Judge-only input: {token}")
    if prefix(work)=="" or prefix(work)!=prefix(judge): errors.append("Dockerfile shared prefixes differ")
    if digest(root/"judge/run_judge.py") != digest(root/"images/judge/run_judge.py"): errors.append("judge adapter ports differ")
    if digest(root/"images/work/benchmark/candidate_worker.py") != digest(root/"images/judge/candidate_worker.py"): errors.append("candidate worker ports differ")
    verifier=(root/"images/judge/verify_sdv2_pipeline.py").read_text()
    for gate in ("profile_ok","hashes_ok","admission_ok","seed_reference_ok","worker_ok","completeness_ok","parity_ok","liveness_ok","resources_ok","ttff_ok","p99_ok","construction_ok"):
        if gate not in verifier: errors.append(f"verifier missing {gate}")
    for token in ("SEED_RUNTIME","_seed_reference","_validate_hardware_record"):
        if token not in verifier: errors.append(f"verifier missing the in-run seed reference: {token}")
    if 'calibration["cases"]' in verifier: errors.append("verifier still gates on the frozen calibration table")
    if "failure_detail" not in verifier or '"valid": False' not in verifier or '"score": 0.0' not in verifier: errors.append("verifier must remain fail-closed")
    if "TODO(seam)" in verifier: errors.append("verifier still contains an open seam marker")
    # v2 timing contract: the controller stamps pushes; the runtime gets no frames up front.
    sink=(root/"runtime/sdv2_core/sink.py").read_text()
    worker=(root/"images/judge/candidate_worker.py").read_text()
    if "def mark_push" in sink or "push_timestamps" in sink: errors.append("the candidate-visible sink must not record push times")
    if "plan.meta()" not in worker or "pushed_ns[chunk.index] = time.monotonic_ns()" not in worker: errors.append("the controller must stamp pushes and pass the frames-free plan")
    for seed in (root/"oracle/candidates/runtime_v001", root/"images/work/workdir/sdv2_pipeline_lab/candidates/runtime_v001"):
        if "mark_push" in (seed/"runtime.py").read_text(): errors.append(f"seed still stamps pushes: {seed}")
    if sorted(p.name for p in (root/"oracle/candidates/runtime_v001").iterdir()) != sorted(p.name for p in (root/"images/work/workdir/sdv2_pipeline_lab/candidates/runtime_v001").iterdir()) or any(digest(root/"oracle/candidates/runtime_v001"/name) != digest(root/"images/work/workdir/sdv2_pipeline_lab/candidates/runtime_v001"/name) for name in ("runtime.py","__init__.py","README.md")):
        errors.append("Work seed copy differs from the oracle seed")
    return errors
def main():
    p=argparse.ArgumentParser(); p.add_argument("root",nargs="?",type=Path,default=Path(__file__).resolve().parents[1]); a=p.parse_args(); errors=validate(a.root.resolve())
    if errors:
        print(f"layout validation failed ({len(errors)} errors)"); [print(" -",e) for e in errors]; return 1
    print("layout validation passed:",a.root.resolve()); return 0
if __name__=="__main__": raise SystemExit(main())
