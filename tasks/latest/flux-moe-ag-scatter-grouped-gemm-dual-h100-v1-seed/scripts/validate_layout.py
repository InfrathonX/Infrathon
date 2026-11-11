#!/usr/bin/env python3
"""Static fail-closed layout validator for the Flux MoE AG task."""
from __future__ import annotations
import hashlib,json,os,py_compile,sys
from pathlib import Path

TASK_JSON="tasks/flux_moe_ag_scatter_grouped_gemm_dual_h100_seed_v1.json"
EXPECTED_FINAL_CASES=967
# Flux has no portable/*.json layer, and the task JSON's `portable` block is an
# exact-keyed capability envelope that rejects extra fields, so the task's
# final-suite record - the role portable/final-suite-manifest.json plays for the
# nine portable tasks - lives here instead.
FINAL_SUITE_RECORD="judge/private/MANIFEST.json"
# Judge-only evaluation inputs: committed with the task, admitted by no Work
# build context.  The frozen qualification draw of the final holdout is the
# default build input (the holdout is not drawn per build);
# ``validate_private_inputs`` proves the Work build can never see it.
# The routing fixtures the hidden suite's captured-routing cases pin (by sha256)
# are Judge-only inputs too.
JUDGE_ROUTING_FIXTURES=("gpt-oss-20b","phi-3.5-moe","mixtral-8x7b","granite-3.1-3b-a800m")
PUBLIC_ROUTING_FIXTURES=("olmoe-1b-7b","qwen3-30b-a3b")
SENSITIVE_JUDGE_INPUTS=("judge/private/final_cases.json",)+tuple(f"judge/private/routing/{name}.npy" for name in JUDGE_ROUTING_FIXTURES)
# A tree without the repository copy (the public tree) reads them as dataset
# assets: assets/fetch.sh places them under <INFRATHON_ASSETS_ROOT>/<task directory>/.
JUDGE_INPUT_ASSETS={"judge/private/final_cases.json":"judge/final_cases.json",**{f"judge/private/routing/{name}.npy":f"judge/routing/{name}.npy" for name in JUDGE_ROUTING_FIXTURES}}
def judge_input_path(root,relative):
    """The repository copy of a Judge-only input, else its fetched dataset asset, else None."""
    path=Path(root)/relative
    if path.exists() or path.is_symlink():return path
    assets_root=os.environ.get("INFRATHON_ASSETS_ROOT") or "~/.cache/infrathon/assets"
    fetched=Path(assets_root).expanduser()/Path(root).name/JUDGE_INPUT_ASSETS[relative]
    return fetched if fetched.is_file() else None
# Everything the Work image build can read: its own context directory plus the
# named build contexts scripts/build_images.sh passes to `docker build`.
WORK_BUILD_CONTEXTS=("images/work",)
REQUIRED=("README.md",".gitignore","runtime/moe_ag_runtime.py","images/work/benchmark/moe_ag_runtime.py","images/judge/Dockerfile","images/judge/Dockerfile.private","images/judge/candidate_worker.py","images/judge/run_judge.py","images/judge/source_admission.py","images/judge/verify_moe_ag.py","images/judge/protected_hashes.txt","images/work/Dockerfile","images/work/AGENT_TASK.md","images/work/benchmark/public_runner.py","images/work/benchmark/world_worker.py","images/work/benchmark/public_cases.json",*(f"images/work/benchmark/routing/{name}.{ext}" for name in PUBLIC_ROUTING_FIXTURES for ext in ("npy","json")),"scripts/capture_routing.py","scripts/build_routing_fixtures.py","images/work/benchmark/source_admission.py","images/work/workdir/moe_ag_lab/FINAL_VERSION","images/work/workdir/moe_ag_lab/candidates/kernel_v001/submission.py","judge/run_judge.py","oracle/FINAL_VERSION","oracle/candidates/kernel_v001/submission.py","scripts/build_images.sh","scripts/eval_archive.sh","scripts/generate_final_manifest.py","scripts/host_smoke.py","scripts/launch_fixed_budget_after_gpu_pair_idle.sh","scripts/lib.sh","scripts/make_oracle_archive.sh","scripts/preflight.py","scripts/replay_candidates.sh","scripts/sforge_dual_h100_entry.py","scripts/smoke_test.sh","scripts/status.sh","scripts/validate_formal_artifacts.py","tasks/BENCHMARK.yaml","tasks/flux_moe_ag_scatter_grouped_gemm_dual_h100_seed_v1.json","tests/test_source_admission.py","tests/test_private_inputs.py",FINAL_SUITE_RECORD,)
FORBIDDEN=("q"+"attn","sage"+"attention","quantized"+"_attention","SAGE"+"_"+"Q"+"ATTN","fl"+"ce")
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def _context_files(context):
    return [p for p in sorted(context.rglob("*")) if p.is_file() and not p.is_symlink() and "__pycache__" not in p.parts and p.suffix!=".pyc"]
def validate_private_inputs(root):
    """Prove the frozen holdout is present, recorded and outside every Work build context.

    Three independent proofs: the instance exists on the Judge-only path (or, in
    a tree without the repository copy, as a fetched dataset asset; with neither,
    the content checks are skipped); that
    path is not inside any Work build context (by path); and no file any Work
    build context admits has the same SHA-256 (by content).
    """
    root=Path(root).resolve(); errors=[]; sensitive={}
    for relative in SENSITIVE_JUDGE_INPUTS:
        path=judge_input_path(root,relative)
        if path is None:continue
        if not path.is_file() or path.is_symlink() or path.stat().st_size==0:
            errors.append(f"missing Judge-only evaluation input: {relative}"); continue
        sensitive[digest(path)]=relative
    contexts=[(name,(root/name).resolve()) for name in WORK_BUILD_CONTEXTS]
    for relative in SENSITIVE_JUDGE_INPUTS:
        resolved=(root/relative).resolve()
        for name,context in contexts:
            if resolved.is_relative_to(context):
                errors.append(f"{relative} lives inside the Work build context {name}")
    for name,context in contexts:
        if not context.is_dir():
            errors.append(f"missing Work build context: {name}"); continue
        for path in _context_files(context):
            if digest(path) in sensitive:
                errors.append(f"Work build context {name} admits {path.relative_to(root)}, byte-identical to the Judge-only {sensitive[digest(path)]}")
    work_dockerfile=root/"images/work/Dockerfile"
    if work_dockerfile.is_file():
        work=work_dockerfile.read_text()
        for token in ("judge/private","final_cases","task_runtime"):
            if token in work:errors.append(f"Work Dockerfile must not reference {token!r}")
    record_path=root/FINAL_SUITE_RECORD
    if not record_path.is_file() or record_path.is_symlink():
        errors.append(f"missing final-suite record: {FINAL_SUITE_RECORD}")
    else:
        try:record=json.loads(record_path.read_text())
        except Exception as exc:record=None;errors.append(f"{FINAL_SUITE_RECORD} invalid: {exc}")
        if not isinstance(record,dict):
            errors.append(f"{FINAL_SUITE_RECORD} must contain a JSON object")
        else:
            if record.get("suite_id")!="final-holdout-v2":errors.append(f"{FINAL_SUITE_RECORD}: suite_id must be 'final-holdout-v2'")
            if record.get("case_count")!=EXPECTED_FINAL_CASES:errors.append(f"{FINAL_SUITE_RECORD}: case_count must be {EXPECTED_FINAL_CASES}")
            frozen=record.get("frozen_instance")
            if not isinstance(frozen,dict):
                errors.append(f"{FINAL_SUITE_RECORD}: missing frozen_instance")
            else:
                if frozen.get("path")!=SENSITIVE_JUDGE_INPUTS[0]:
                    errors.append(f"{FINAL_SUITE_RECORD}: frozen_instance.path must be {SENSITIVE_JUDGE_INPUTS[0]!r}")
                instance=judge_input_path(root,SENSITIVE_JUDGE_INPUTS[0])
                if instance is not None and instance.is_file() and frozen.get("sha256")!=digest(instance):
                    errors.append(f"{FINAL_SUITE_RECORD}: frozen_instance.sha256 does not match the committed instance")
                if not str(frozen.get("provenance","")).strip():
                    errors.append(f"{FINAL_SUITE_RECORD}: frozen_instance.provenance must record which draw this is")
            pinned=record.get("routing_fixtures")
            if not isinstance(pinned,dict) or set(pinned)!=set(JUDGE_ROUTING_FIXTURES):
                errors.append(f"{FINAL_SUITE_RECORD}: routing_fixtures must record the sha256 of {sorted(JUDGE_ROUTING_FIXTURES)}")
            else:
                for name in JUDGE_ROUTING_FIXTURES:
                    path=judge_input_path(root,f"judge/private/routing/{name}.npy")
                    if path is not None and path.is_file() and pinned.get(name)!=digest(path):
                        errors.append(f"{FINAL_SUITE_RECORD}: routing_fixtures.{name} does not match the committed fixture")
            policy=record.get("draw_policy")
            if not isinstance(policy,dict) or "frozen-instance-by-default" not in str(policy.get("default","")):
                errors.append(f"{FINAL_SUITE_RECORD}: draw_policy.default must state frozen-instance-by-default")
            elif "NEW suite identity" not in str(policy.get("redraw","")):
                errors.append(f"{FINAL_SUITE_RECORD}: draw_policy.redraw must state that a redraw is a NEW suite identity")
    # The final-suite record must never be mistaken for holdout content.
    if record_path.is_file() and (root/SENSITIVE_JUDGE_INPUTS[0]).is_file():
        if "cases" in json.loads(record_path.read_text() or "{}"):
            errors.append(f"{FINAL_SUITE_RECORD} must record the digest, never the drawn cases")
    generator_path=root/"scripts/generate_final_manifest.py"
    if generator_path.is_file():
        generator=generator_path.read_text()
        for fragment in ("--seed","--redraw","DEFAULT_SEED","def default_seed("):
            if fragment not in generator:errors.append(f"scripts/generate_final_manifest.py: missing {fragment!r}")
        for forbidden in ("secrets.SystemRandom","import secrets"):
            if forbidden in generator:errors.append(f"scripts/generate_final_manifest.py must not use {forbidden!r}: the draw is seeded")
    build_path=root/"scripts/build_images.sh"
    if build_path.is_file():
        build=build_path.read_text()
        for fragment in ('final_cases_source="${FINAL_CASES_PATH:-${DEFAULT_FINAL_CASES_PATH}}"','install -m 0600 "${final_cases_source}" "${final_cases_context}/final_cases.json"',"\"$(stat -c '%a' \"${final_cases_context}/final_cases.json\")\" == \"600\"","\"$(stat -c '%u' \"${final_cases_context}/final_cases.json\")\" == \"$(id -u)\""):
            if fragment not in build:errors.append(f"scripts/build_images.sh: missing {fragment!r}")
        if '--output "${final_cases_source}"' in build:
            errors.append("scripts/build_images.sh must not draw the holdout per build: the frozen instance is the default input")
    lib_path=root/"scripts/lib.sh"
    if lib_path.is_file():
        frozen_default='DEFAULT_FINAL_CASES_PATH="${MIGRATION_ROOT}/'+SENSITIVE_JUDGE_INPUTS[0]+'"'
        if frozen_default not in lib_path.read_text():errors.append(f"scripts/lib.sh: missing {frozen_default!r}")
    return errors
def main(argv=None):
    root=Path((argv or sys.argv[1:])[0] if (argv or sys.argv[1:]) else Path(__file__).resolve().parents[1]).resolve(); errors=[]
    for relative in REQUIRED:
        path=root/relative
        if not path.is_file() or path.is_symlink():errors.append(f"missing/unsafe required file: {relative}")
    try:
        task=json.loads((root/TASK_JSON).read_text())
        if task.get("task_id")!="flux_moe_ag_scatter_grouped_gemm_dual_h100_seed_v1":errors.append("task id differs")
    except Exception as exc:errors.append(f"task JSON invalid: {exc}")
    # The protected Judge worker intentionally accepts only Judge-owned suite
    # identities.  The Work-side pair worker additionally accepts public and
    # host-smoke identities, so those two files must not be byte-paired.
    pairs=(("images/judge/run_judge.py","judge/run_judge.py"),("runtime/moe_ag_runtime.py","images/work/benchmark/moe_ag_runtime.py"),("images/judge/source_admission.py","images/work/benchmark/source_admission.py"),("images/work/workdir/moe_ag_lab/candidates/kernel_v001/submission.py","oracle/candidates/kernel_v001/submission.py"))
    for left,right in pairs:
        if (root/left).read_bytes()!=(root/right).read_bytes():errors.append(f"paired files differ: {left} {right}")
    # Generated, git-ignored trees (run logs, build context, host artifacts) are evidence,
    # not task source; SForge writes non-executable helper scripts under logs/runs.
    GENERATED_DIRS={"logs","artifacts","build","__pycache__",".pytest_cache"}
    def generated(path):return bool(set(path.relative_to(root).parts[:-1])&GENERATED_DIRS)
    for path in root.rglob("*"):
        if generated(path) or not path.is_file() or path.suffix in {".pyc"}:continue
        if path.as_posix().endswith("docs/SOURCE_PROVENANCE.md"):continue
        try:text=path.read_text().lower()
        except UnicodeDecodeError:continue
        for marker in FORBIDDEN:
            if marker.lower() in text:errors.append(f"legacy marker {marker}: {path.relative_to(root)}")
    for path in root.rglob("*.py"):
        if generated(path):continue
        try:py_compile.compile(str(path),doraise=True)
        except py_compile.PyCompileError as exc:errors.append(str(exc))
    for path in root.rglob("*.sh"):
        if generated(path):continue
        if not os.access(path,os.X_OK):errors.append(f"shell script not executable: {path.relative_to(root)}")
    errors.extend(validate_private_inputs(root))
    if errors:
        print("\n".join(errors),file=sys.stderr); return 1
    print(f"layout validation: PASS ({len(REQUIRED)} required files)"); return 0
if __name__=="__main__":raise SystemExit(main())
