#!/usr/bin/env python3
"""Write the v2 portable contract documents under portable/.

The resource lists below are the reviewed contract; this tool only renders
them with current digests (one resource or case per line, so
scripts/refresh_portable_manifests.py can keep the digests current) and then
the refresh tool binds every document digest into the task JSON.

    scripts/write_portable_manifests.py && scripts/refresh_portable_manifests.py
"""
from __future__ import annotations

import collections
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runtime"))
import kv_scoring_runtime as rt  # noqa: E402

LAB = "images/work/workdir/kv_scoring_lab"
TASK_CORE = [
    "runtime/kv_scoring_runtime.py", "runtime/kv_scoring_metric.py", "images/judge/verify_kv_scoring.py", "images/judge/candidate_worker.py", "images/judge/trusted_replay.py",
    "images/judge/judge_shards.py", "images/judge/released_weights_gate.py", "images/judge/run_judge.py", "images/judge/source_admission.py", "images/work/benchmark/source_admission.py",
    "images/work/benchmark/public_runner.py", "images/work/AGENT_TASK.md", f"{LAB}/README_AGENT.md", "corpus/public/items_public.jsonl",
    "judge/private/final_manifest.json", "judge/private/needle_private.json", "scripts/prepare_items.py", "llm_judge/POLICY.md",
]
DELIVERY = [
    "tasks/BENCHMARK.yaml", "images/judge/Dockerfile", "images/judge/.dockerignore", "images/judge/protected_hashes.txt", "images/work/Dockerfile",
    "images/work/.dockerignore", "images/work/codex-config.toml", "images/work/benchmark/run_public_tests.sh", "images/work/benchmark/prepare_submit.sh",
    f"{LAB}/FINAL_VERSION", f"{LAB}/notes.md", f"{LAB}/candidates/scorer_v001/submission.py", f"{LAB}/tools/run_one.sh", f"{LAB}/tools/run_public_tests.sh", f"{LAB}/tools/check_submission_size.sh",
    "corpus/public/MANIFEST.json", "scripts/lib.sh", "scripts/build_images.sh", "scripts/eval_archive.sh", "scripts/sforge_entry.py",
    "scripts/prepare_released_weights_fingerprint.py",
    "scripts/validate_layout.py", "scripts/update_protected_hashes.py", "scripts/write_portable_manifests.py", "scripts/refresh_portable_manifests.py",
]
WEIGHT = "rows averaged per task; tasks averaged per group; relative to full cache; groups averaged per ratio; ratios averaged"
MODE = "phase_a_context_only_scores_then_trusted_phase_b_masked_decode"


def sha(path: str) -> str:
    return "sha256:" + hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def render(doc: dict, list_key: str | None = None, items: list | None = None) -> str:
    """Two-space JSON with one list entry per line (the refresh tool edits those lines)."""
    if list_key is None:
        return json.dumps(doc, indent=2, sort_keys=True) + "\n"
    head = json.dumps({k: v for k, v in doc.items()}, indent=2, sort_keys=True)
    marker = f'"{list_key}": "__LIST__"'
    body = ",\n".join("    " + json.dumps(x, sort_keys=True) for x in items)
    return head.replace(marker, f'"{list_key}": [\n{body}\n  ]') + "\n"


def write(name: str, text: str) -> None:
    (ROOT / "portable" / name).write_text(text, encoding="utf-8")


def resources(kind: str, includes: str, excludes, paths: list[str]) -> str:
    doc = {"schema_version": 1, "boundary": {"kind": kind, "includes": includes, "excludes": excludes}, "resources": "__LIST__"}
    return render(doc, "resources", [{"path": p, "sha256": sha(p)} for p in paths])


def main() -> int:
    write("resource-manifest.json", resources("task_core", "scientific contract: runtime, trusted Judge code, briefs, public items, frozen holdout manifest, item generator, integrity policy",
                                              "launch and deployment plumbing (build-resource-manifest.json)", TASK_CORE))
    write("build-resource-manifest.json", resources("delivery", "image build inputs, Work lab files, launch adapters and contract tooling",
                                                    ["the frozen holdout items (rebuilt at image build from judge/private/final_manifest.json)"], DELIVERY))
    public = rt.PUBLIC_ITEMS; quick = {x["item_id"] for x in rt.suite_items("quick")}; one = {x["item_id"] for x in rt.suite_items("one")}
    per_task = collections.Counter(x["task"] for x in public)
    cases = [{"item_id": x["item_id"], "task": x["task"], "group": rt.TASKS[x["task"]].group, "questions": len(x["questions"]), "context_tokens": x["meta"]["context_tokens"],
              "in_one": x["item_id"] in one, "in_quick": x["item_id"] in quick, "ratios_per_mille": [100, 200, 300], "role": "performance", "measurement_mode": MODE,
              "weight": {"numerator": 1, "denominator": per_task[x["task"]]}, "weight_semantics": WEIGHT} for x in public]
    write("public-suite-manifest.json", render({"schema_version": 1, "suite_id": rt.PUBLIC_SUITE_ID, "case_counts": {"total": len(cases), "scored_performance": len(cases), "zero_weight_hard_gate": 0},
                                                "subsets": {"one": sorted(one), "quick": [x["item_id"] for x in rt.suite_items("quick")]}, "cases": "__LIST__"}, "cases", cases))
    manifest = json.loads((ROOT / "judge/private/final_manifest.json").read_text())
    counts = manifest["counts_per_task"]
    final_cases = [{"item_id": e["item_id"], "task": e["task"], "group": rt.TASKS[e["task"]].group, "context_tokens": e["context_tokens"], "ratios_per_mille": [100, 200, 300], "role": "performance",
                    "measurement_mode": MODE, "weight": {"numerator": 1, "denominator": counts[e["task"]]}} for e in manifest["items"]]
    frozen = {"path": "judge/private/final_manifest.json", "sha256": hashlib.sha256((ROOT / "judge/private/final_manifest.json").read_bytes()).hexdigest(),
              "provenance": "scripts/prepare_items.py build (seed kvzip-scoring-v2) following the KVzip eval protocol; rows rebuilt from pinned sources and compared at image build"}
    for name, role in (("final-suite-manifest.json", "final"), ("interactive-suite-manifest.json", "interactive")):
        doc = {"schema_version": 1, "suite_id": rt.FINAL_SUITE_ID, "role": role, "case_counts": {"total": len(final_cases), "scored_performance": len(final_cases), "zero_weight_hard_gate": 0},
               "counts_per_task": counts, "frozen_instance": frozen, "same_suite_for_interactive_and_final": True, "cases": "__LIST__"}
        write(name, render(doc, "cases", final_cases))
    write("oracle-protocol.json", render({
        "schema_version": 1, "protocol_id": "trusted-masked-decode-relative-accuracy-v2",
        "acceptance": "every run-wide gate passes; otherwise the score is 0 and the result is a trusted gate failure",
        "metric": {"task": "mean over rows of the mean over questions of the task metric", "relative": "acc(task, r) / acc(task, full)",
                   "group": "mean over tasks", "ratio": "mean over retrieval, contextual_qa, redundancy", "primary": "100 x mean over 100, 200, 300 per mille retention"},
        "gates": {"names": ["profile_ok", "suite_ok", "source_ok", "assets_ok", "hardware_ok", "scores_ok", "immutable_ok", "rng_ok", "overhead_ok", "memory_ok", "metric_ok", "reference_ok"],
                  "overhead_suite_fraction_percent": 25, "overhead_item_cap_percent": 50, "overhead_item_floor": "10 x the shortest trusted prefill in the suite",
                  "memory_limit_bytes": rt.MEMORY_LIMIT_BYTES},
        "integrity": ["source_admission", "released_weights_fingerprint", "fresh_scorer_per_row", "three_pass_scoring_only_the_row_pass_timed", "context_only_phase_a", "immutable_inputs", "rng_unchanged", "llm_integrity_review"],
        "reference": {"model": rt.MODEL_ID, "decode": "greedy, (condition, question) pairs in chunks of min(16, 40 GiB / per-sequence KV bytes)", "attention": "per-layer per-KV-head context masks, system and appended positions visible",
                      "golden": "full-cache answers recomputed every run"},
    }))
    write("selection-protocol.json", render({
        "schema_version": 1, "protocol_id": "kvzip-scoring-full-final-shared-v2-sharded", "evaluation_policy": "full-final-shared-v2", "same_suite_for_interactive_and_final": True,
        "groups": list(rt.GROUPS), "tasks": list(rt.TASKS), "ratios_per_mille": [100, 200, 300],
        "public": {"suite_id": rt.PUBLIC_SUITE_ID, "full": {"case_count": len(public)}, "quick": {"case_count": len(quick), "rule": "first two rows of every task"}, "one": {"case_count": 1, "rule": "first squad row"}},
        "interactive": {"suite_id": rt.FINAL_SUITE_ID, "case_count": len(final_cases), "routing": "round_id prefix auto- or agent-", "rounds": "scored agent submissions; archive FINAL_VERSION only", "run_gate_policy": "all_11_run_wide_gates_must_pass"},
        "final": {"suite_id": rt.FINAL_SUITE_ID, "case_count": len(final_cases), "routing": "round_id prefix manual- or candidate-", "rounds": "deadline final and trusted host replays", "run_gate_policy": "all_11_run_wide_gates_must_pass"},
        "sharding": {"gpus": "1-8 identical devices visible to the Judge", "partition": "longest-processing-time by estimated cost, deterministic", "per_shard": "Phase A then Phase B on one GPU, NUMA-local whole-core CPU slice", "score_dependence": "none"},
    }))
    write("software-abi.json", render({
        "schema_version": 1, "protocol_id": "pytorch280-cu129-transformers4571-sdpa-v2", "language": "python3", "source_runtime": ["pytorch-2.8.0", "cuda-12.9", "triton-3.4.0", "transformers-4.57.1", "accelerate-1.10.1", "rouge-1.0.1", "tree-sitter-languages-1.10.2", "nltk-3.9.1"],
        "accelerator_runtime": "cuda-12.9-compatible", "attention_implementation": "sdpa", "required_dtypes": ["bf16", "fp32"], "forbidden_packages": ["kvzip", "flash_attn", "xformers", "flashinfer"],
        "model_binding": {"model_id": rt.MODEL_ID, "mount": "/opt/models/qwen3-8b (read-only)", "adapter": "scripts/sforge_entry.py", "offline": True},
        "policy_assets": {"file": "candidates/scorer_vNNN/assets.pt or a single-hop assets.ref", "loader": "torch.load(weights_only=True) string-to-tensor dict, gzip transport allowed", "max_bytes": rt.MAX_ASSET_BYTES, "max_expanded_bytes": rt.MAX_ASSET_EXPANDED_BYTES},
    }))
    write("calibration-protocol.json", render({
        "schema_version": 1, "protocol_id": "host-wall-cuda-event-three-pass-v2", "clock": "max_of_synchronized_host_wall_and_cuda_event per score call; only the pass on the row's own tensors is timed, after an untimed pass on random tensors of the same shapes and before an untimed determinism pass on fresh copies",
        "aggregation": "sum_over_layers_per_row", "warmup": "untimed fully gated passes over all layers on 2048- and 128-token random decoys",
        "fresh_process": "per shard: Phase A and Phase B are separate processes on the same GPU", "overhead_reference": "trusted plain prefill of the same row re-measured in Phase B on the same device",
        "comparability": "same_measurement_profile",
    }))
    print("wrote portable/*.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
