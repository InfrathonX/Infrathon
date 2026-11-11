#!/usr/bin/env python3
"""Static layout checks for the KVzip v2 task, run before every image build."""
from __future__ import annotations
import hashlib, json, os, sys
from pathlib import Path

TASK_ID = "kvzip_query_agnostic_kv_scoring_qwen3_8b_h100_scratch_v2"
# Judge-only evaluation inputs committed with the task: the frozen holdout
# manifest and the three private needles.  Neither may reach a Work build
# context, by path or by content, and no Work file may contain a needle.
SENSITIVE_JUDGE_INPUTS = ("judge/private/final_manifest.json", "judge/private/needle_private.json")
# A public checkout does not carry them: they ship as Infrathon dataset
# assets (assets/manifest.json, fetched by assets/fetch.sh into the asset root
# as <task directory>/judge/<name>, the same paths scripts/lib.sh uses).
JUDGE_INPUT_ASSETS = {name: "judge/" + name.rsplit("/", 1)[1] for name in SENSITIVE_JUDGE_INPUTS}
WORK_BUILD_CONTEXTS = ("images/work", "runtime", "corpus/public")
REQUIRED = (
    ".gitignore", "README.md", "llm_judge/POLICY.md",
    "images/work/Dockerfile", "images/work/.dockerignore", "images/work/codex-config.toml", "images/work/AGENT_TASK.md",
    "images/work/benchmark/public_runner.py", "images/work/benchmark/run_public_tests.sh", "images/work/benchmark/source_admission.py",
    "images/work/workdir/kv_scoring_lab/README_AGENT.md", "images/work/workdir/kv_scoring_lab/FINAL_VERSION", "images/work/workdir/kv_scoring_lab/notes.md",
    "images/work/workdir/kv_scoring_lab/candidates/scorer_v001/submission.py", "images/work/workdir/kv_scoring_lab/tools/run_one.sh", "images/work/workdir/kv_scoring_lab/tools/run_public_tests.sh",
    "images/judge/Dockerfile", "images/judge/.dockerignore", "images/judge/run_judge.py", "images/judge/verify_kv_scoring.py", "images/judge/candidate_worker.py",
    "images/judge/trusted_replay.py", "images/judge/judge_shards.py", "images/judge/source_admission.py", "images/judge/protected_hashes.txt",
    "runtime/kv_scoring_runtime.py", "runtime/kv_scoring_metric.py", "corpus/public/items_public.jsonl", "corpus/public/MANIFEST.json",
    "oracle/FINAL_VERSION", "oracle/notes.md", "oracle/candidates/scorer_v001/submission.py",
    "scripts/lib.sh", "scripts/build_images.sh", "scripts/eval_archive.sh", "scripts/prepare_items.py", "scripts/sforge_entry.py",
    "scripts/update_protected_hashes.py", "scripts/validate_layout.py", "scripts/prepare_released_weights_fingerprint.py",
    "images/judge/released_weights_gate.py",
    "tasks/BENCHMARK.yaml", f"tasks/{TASK_ID}.json",
)


def assets_root() -> Path:
    value = os.environ.get("INFRATHON_ASSETS_ROOT")
    if value:
        return Path(value).expanduser()
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "infrathon" / "assets"


def judge_input(root: Path, relative: str) -> Path | None:
    """The Judge-only input ``relative`` (its in-repository path): the in-repository
    copy, else the fetched dataset asset it was staged from, else None."""
    path = root / relative
    if path.exists() or path.is_symlink():
        return path
    if relative not in JUDGE_INPUT_ASSETS:
        return None
    task_directory = json.loads((root / "assets/manifest.json").read_text(encoding="utf-8"))["task_directory"]
    fetched = assets_root() / task_directory / JUDGE_INPUT_ASSETS[relative]
    return fetched if fetched.is_file() else None


def _context_files(base: Path):
    for path in sorted(base.rglob("*")):
        if path.is_file() and not path.is_symlink() and "__pycache__" not in path.parts and path.suffix != ".pyc":
            yield path


def validate_private_inputs(root: Path) -> list[str]:
    """Checks that need the Judge-only inputs run only where they are available
    (in the repository or fetched into the asset root)."""
    errors: list[str] = []; sensitive: dict[str, str] = {}; found: dict[str, Path] = {}
    for relative in SENSITIVE_JUDGE_INPUTS:
        path = judge_input(root, relative)
        if path is None: continue
        if not path.is_file() or path.is_symlink() or path.stat().st_size == 0:
            errors.append(f"missing Judge-only evaluation input: {relative}"); continue
        found[relative] = path
        sensitive[hashlib.sha256(path.read_bytes()).hexdigest()] = relative
    contexts = [(name, (root / name).resolve()) for name in WORK_BUILD_CONTEXTS]
    for relative in SENSITIVE_JUDGE_INPUTS:
        resolved = (root / relative).resolve()
        for name, context in contexts:
            if resolved.is_relative_to(context): errors.append(f"{relative} lives inside the Work build context {name}")
    needles_path = found.get("judge/private/needle_private.json")
    needles = json.loads(needles_path.read_text(encoding="utf-8")) if needles_path else []
    secrets = [x["needle"].strip() for x in needles] + [x["answer"].strip() for x in needles]
    for name, context in contexts:
        if not context.is_dir(): errors.append(f"missing Work build context: {name}"); continue
        for path in _context_files(context):
            data = path.read_bytes(); digest = hashlib.sha256(data).hexdigest()
            if digest in sensitive: errors.append(f"Work build context {name} admits {path.relative_to(root)}, byte-identical to the Judge-only {sensitive[digest]}")
            text = data.decode("utf-8", errors="replace")
            for secret in secrets:
                if secret and secret in text: errors.append(f"{path.relative_to(root)} contains a private needle")
    work = (root / "images/work/Dockerfile").read_text(encoding="utf-8")
    for token in ("judge/private", "final_items", "final_manifest", "needle_private"):
        if token in work: errors.append(f"Work Dockerfile must not reference {token!r}")
    judge = (root / "images/judge/Dockerfile").read_text(encoding="utf-8")
    # The public Judge is publishable; only its host-built holdout layer
    # (Dockerfile.private, FROM the public Judge) receives the final_items context.
    if "final_items" in judge: errors.append("public Judge Dockerfile must not receive the final_items context")
    holdout_layer = (root / "images/judge/Dockerfile.private").read_text(encoding="utf-8")
    if "FROM ${JUDGE_PUBLIC_IMAGE}" not in holdout_layer: errors.append("Judge holdout layer must be built FROM the public Judge image")
    if "--from=final_items final_manifest.json final_items.jsonl.gz released_weights_fingerprint.pt" not in holdout_layer: errors.append("Judge holdout layer must copy the holdout from the final_items build context")
    build = (root / "scripts/build_images.sh").read_text(encoding="utf-8")
    for fragment in ('prepare_items.py" materialize', 'prepare_released_weights_fingerprint.py" \\', '--build-context "final_items=${FINAL_ITEMS_CONTEXT}"', 'install -m 0600 "${FINAL_MANIFEST_PATH}"'):
        if fragment not in build: errors.append(f"scripts/build_images.sh: missing {fragment!r}")
    manifest_path = found.get(SENSITIVE_JUDGE_INPUTS[0])
    if manifest_path is not None:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        holdout_sha = {e["sha256"] for e in manifest["items"]}
        sys.path.insert(0, str(root / "runtime")); import kv_scoring_runtime as rt  # noqa: E402
        for item in rt.load_items_jsonl(root / "corpus/public/items_public.jsonl"):
            if rt.item_sha256(item) in holdout_sha: errors.append(f"public row {item['item_id']} is a holdout row")
    return errors


def validate_multi_gpu_judge(root: Path) -> list[str]:
    errors: list[str] = []
    adapter = (root / "scripts/sforge_entry.py").read_text(encoding="utf-8")
    for fragment in ("KV_SCORING_JUDGE_GPU_UUIDS", "DockerBackend.create_container = create_container", "_judge_device_request"):
        if fragment not in adapter: errors.append(f"scripts/sforge_entry.py: missing {fragment!r}")
    verifier = (root / "images/judge/verify_kv_scoring.py").read_text(encoding="utf-8")
    for fragment in ("released_weights_check(asset, run_root, sessions)", 'detail["assets_ok"] = 1.0', "judge_shards.visible_gpus()", "judge_shards.validate_fleet(", "judge_shards.cpu_slices(", "judge_shards.plan_shards(", "shard rows do not cover the suite exactly once"):
        if fragment not in verifier: errors.append(f"images/judge/verify_kv_scoring.py: missing {fragment!r}")
    return errors


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parents[1]).resolve()
    missing = [name for name in REQUIRED if not (root / name).is_file()]
    if missing: raise SystemExit("missing required files: " + ", ".join(missing))
    task = json.loads((root / f"tasks/{TASK_ID}.json").read_text())
    assert task["task_id"] == TASK_ID
    assert task["version_enumeration"]["directory_pattern"] == "scorer_v([0-9]{3})"
    assert task["submit_paths"] == ["candidates", "FINAL_VERSION", "notes.md"]
    manifest = json.loads((root / "corpus/public/MANIFEST.json").read_text())
    assert hashlib.sha256((root / "corpus/public/items_public.jsonl").read_bytes()).hexdigest() == manifest["files"]["items_public.jsonl"]
    assert (root / "images/work/benchmark/source_admission.py").read_bytes() == (root / "images/judge/source_admission.py").read_bytes()
    if (root / "images/work/workdir/kv_scoring_lab/candidates/scorer_v001/submission.py").stat().st_size != 0: raise SystemExit("the Work seed scorer must stay an empty file")
    errors = validate_private_inputs(root) + validate_multi_gpu_judge(root)
    if errors: raise SystemExit("layout errors: " + "; ".join(errors))
    absent = [name for name in SENSITIVE_JUDGE_INPUTS if judge_input(root, name) is None]
    if absent: print("Judge-only inputs not present (run assets/fetch.sh); their checks were skipped: " + ", ".join(absent))
    print(f"layout OK: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
