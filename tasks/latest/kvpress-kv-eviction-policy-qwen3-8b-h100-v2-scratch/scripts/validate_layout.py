#!/usr/bin/env python3
"""Static layout checks for the kvpress v2 task, run before every image build."""
from __future__ import annotations
import hashlib, json, os, sys
from pathlib import Path

TASK_ID = "kvpress_kv_eviction_policy_qwen3_8b_h100_scratch_v2"
TASK_DIRECTORY = "kvpress-kv-eviction-policy-qwen3-8b-h100-v2-scratch"
# Judge-only evaluation input: the frozen holdout manifest (dataset/revision/
# file/row/sha256 per item, plus task names).  It must never reach a Work build
# context, by path or by content.  The in-repository copy is used when present,
# else the Infrathon asset (assets/fetch.sh) at
# <INFRATHON_ASSETS_ROOT>/<task directory>/<asset path>.
SENSITIVE_JUDGE_INPUTS = ("judge/private/final_manifest.json",)
SENSITIVE_JUDGE_ASSETS = {"judge/private/final_manifest.json": "judge/final_manifest.json"}
# Everything the Work image build can read: its own context directory plus the
# named build contexts scripts/build_images.sh passes to `docker build`.
WORK_BUILD_CONTEXTS = ("images/work", "runtime", "corpus/public")
# Holdout task names that are also ordinary English words are not searched for.
COMMON_WORDS = {"quality"}
REQUIRED = (
    ".gitignore", "README.md", "llm_judge/POLICY.md",
    "images/work/Dockerfile", "images/work/.dockerignore", "images/work/codex-config.toml", "images/work/AGENT_TASK.md",
    "images/work/benchmark/public_runner.py", "images/work/benchmark/run_public_tests.sh", "images/work/benchmark/source_admission.py",
    "images/work/workdir/kv_eviction_lab/README_AGENT.md", "images/work/workdir/kv_eviction_lab/FINAL_VERSION", "images/work/workdir/kv_eviction_lab/notes.md",
    "images/work/workdir/kv_eviction_lab/candidates/policy_v001/submission.py", "images/work/workdir/kv_eviction_lab/tools/run_one.sh", "images/work/workdir/kv_eviction_lab/tools/run_public_tests.sh",
    "images/judge/Dockerfile", "images/judge/.dockerignore", "images/judge/run_judge.py", "images/judge/verify_kv_eviction.py", "images/judge/candidate_worker.py",
    "images/judge/trusted_replay.py", "images/judge/judge_shards.py", "images/judge/source_admission.py", "images/judge/protected_hashes.txt",
    "runtime/kv_eviction_runtime.py", "corpus/public/items_public.jsonl", "corpus/public/MANIFEST.json",
    "oracle/FINAL_VERSION", "oracle/notes.md", "oracle/candidates/policy_v001/submission.py",
    "scripts/lib.sh", "scripts/build_images.sh", "scripts/eval_archive.sh", "scripts/prepare_items.py", "scripts/sforge_entry.py",
    "scripts/update_protected_hashes.py", "scripts/validate_layout.py",
    "tasks/BENCHMARK.yaml", f"tasks/{TASK_ID}.json",
)


def judge_input(root: Path, relative: str) -> Path | None:
    """The Judge-only input: in the repository, else in the asset root; None when neither."""
    path = root / relative
    if path.exists() or path.is_symlink():
        return path
    assets = Path(os.environ.get("INFRATHON_ASSETS_ROOT") or Path.home() / ".cache/infrathon/assets")
    path = assets / TASK_DIRECTORY / SENSITIVE_JUDGE_ASSETS[relative]
    return path if path.exists() else None


def _context_files(base: Path):
    for path in sorted(base.rglob("*")):
        if path.is_file() and not path.is_symlink() and "__pycache__" not in path.parts and path.suffix != ".pyc":
            yield path


def validate_private_inputs(root: Path) -> list[str]:
    errors: list[str] = []; sensitive: dict[str, str] = {}; inputs: dict[str, Path] = {}
    for relative in SENSITIVE_JUDGE_INPUTS:
        path = judge_input(root, relative)
        # Not fetched (public checkout without assets/fetch.sh): nothing to leak, and
        # scripts/build_images.sh refuses to build without it.
        if path is None: continue
        if not path.is_file() or path.is_symlink() or path.stat().st_size == 0:
            errors.append(f"missing Judge-only evaluation input: {relative}"); continue
        sensitive[hashlib.sha256(path.read_bytes()).hexdigest()] = relative; inputs[relative] = path
    contexts = [(name, (root / name).resolve()) for name in WORK_BUILD_CONTEXTS]
    for relative in SENSITIVE_JUDGE_INPUTS:
        resolved = (root / relative).resolve()
        for name, context in contexts:
            if resolved.is_relative_to(context): errors.append(f"{relative} lives inside the Work build context {name}")
    for name, context in contexts:
        if not context.is_dir(): errors.append(f"missing Work build context: {name}"); continue
        for path in _context_files(context):
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest in sensitive: errors.append(f"Work build context {name} admits {path.relative_to(root)}, byte-identical to the Judge-only {sensitive[digest]}")
    work = (root / "images/work/Dockerfile").read_text(encoding="utf-8")
    for token in ("judge/private", "final_items", "final_manifest", "corpus_private"):
        if token in work: errors.append(f"Work Dockerfile must not reference {token!r}")
    judge = (root / "images/judge/Dockerfile").read_text(encoding="utf-8")
    # The public Judge is publishable; only its host-built holdout layer
    # (Dockerfile.private, FROM the public Judge) receives the final_items context.
    if "final_items" in judge: errors.append("public Judge Dockerfile must not receive the final_items context")
    holdout_layer = (root / "images/judge/Dockerfile.private").read_text(encoding="utf-8")
    if "FROM ${JUDGE_PUBLIC_IMAGE}" not in holdout_layer: errors.append("Judge holdout layer must be built FROM the public Judge image")
    if "--from=final_items final_manifest.json final_items.jsonl.gz" not in holdout_layer: errors.append("Judge holdout layer must copy the holdout from the final_items build context")
    build = (root / "scripts/build_images.sh").read_text(encoding="utf-8")
    for fragment in ('prepare_items.py" materialize', '--build-context "final_items=${FINAL_ITEMS_CONTEXT}"', 'install -m 0600 "${FINAL_MANIFEST_PATH}"'):
        if fragment not in build: errors.append(f"scripts/build_images.sh: missing {fragment!r}")
    # The public items never name a holdout task or carry a holdout item.
    if SENSITIVE_JUDGE_INPUTS[0] not in inputs: return errors
    manifest = json.loads(inputs[SENSITIVE_JUDGE_INPUTS[0]].read_text(encoding="utf-8"))
    holdout_tasks = {e["task"] for e in manifest["items"] if not e["task"].startswith("ruler.")}
    holdout_sha = {e["sha256"] for e in manifest["items"]}
    sys.path.insert(0, str(root / "runtime")); import kv_eviction_runtime as rt  # noqa: E402
    public = rt.load_items_jsonl(root / "corpus/public/items_public.jsonl")
    for item in public:
        if item["task"] in holdout_tasks: errors.append(f"public item {item['item_id']} uses holdout task {item['task']}")
        if rt.item_sha256(item) in holdout_sha: errors.append(f"public item {item['item_id']} is a holdout item")
    import re
    names = {task.split(".", 1)[1] for task in holdout_tasks} - COMMON_WORDS
    for path in _context_files((root / "images/work").resolve()):
        if path.suffix in (".md", ".py", ".sh", ".txt", ".toml"):
            text = path.read_text(encoding="utf-8", errors="replace").lower()
            for name in sorted(names):
                if re.search(rf"\b{re.escape(name.lower())}\b", text): errors.append(f"{path.relative_to(root)} names holdout task {name!r}")
    return errors


def validate_multi_gpu_judge(root: Path) -> list[str]:
    errors: list[str] = []
    adapter = (root / "scripts/sforge_entry.py").read_text(encoding="utf-8")
    for fragment in ("KV_EVICTION_JUDGE_GPU_UUIDS", "DockerBackend.create_container = create_container", "_judge_device_request"):
        if fragment not in adapter: errors.append(f"scripts/sforge_entry.py: missing {fragment!r}")
    verifier = (root / "images/judge/verify_kv_eviction.py").read_text(encoding="utf-8")
    for fragment in ("judge_shards.visible_gpus()", "judge_shards.validate_fleet(", "judge_shards.cpu_slices(", "judge_shards.plan_shards(", "shard rows do not cover the suite exactly once"):
        if fragment not in verifier: errors.append(f"images/judge/verify_kv_eviction.py: missing {fragment!r}")
    return errors


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parents[1]).resolve()
    missing = [name for name in REQUIRED if not (root / name).is_file()]
    if missing: raise SystemExit("missing required files: " + ", ".join(missing))
    task = json.loads((root / f"tasks/{TASK_ID}.json").read_text())
    assert task["task_id"] == TASK_ID
    assert task["version_enumeration"]["directory_pattern"] == "policy_v([0-9]{3})"
    assert task["submit_paths"] == ["candidates", "FINAL_VERSION", "notes.md"]
    manifest = json.loads((root / "corpus/public/MANIFEST.json").read_text())
    assert hashlib.sha256((root / "corpus/public/items_public.jsonl").read_bytes()).hexdigest() == manifest["files"]["items_public.jsonl"]
    assert (root / "images/work/benchmark/source_admission.py").read_bytes() == (root / "images/judge/source_admission.py").read_bytes()
    assert (root / "oracle/candidates/policy_v001/submission.py").read_bytes() == (root / "images/work/workdir/kv_eviction_lab/candidates/policy_v001/submission.py").read_bytes()
    errors = validate_private_inputs(root) + validate_multi_gpu_judge(root)
    if errors: raise SystemExit("layout errors: " + "; ".join(errors))
    print(f"layout OK: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
