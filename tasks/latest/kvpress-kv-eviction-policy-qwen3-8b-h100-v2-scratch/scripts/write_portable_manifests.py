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
import kv_eviction_runtime as rt  # noqa: E402

LAB = "images/work/workdir/kv_eviction_lab"
TASK_CORE = [
    "runtime/kv_eviction_runtime.py", "images/judge/verify_kv_eviction.py", "images/judge/candidate_worker.py", "images/judge/trusted_replay.py",
    "images/judge/judge_shards.py", "images/judge/run_judge.py", "images/judge/source_admission.py", "images/work/benchmark/source_admission.py",
    "images/work/benchmark/public_runner.py", "images/work/AGENT_TASK.md", f"{LAB}/README_AGENT.md", "corpus/public/items_public.jsonl",
    "judge/private/final_manifest.json", "scripts/prepare_items.py", "llm_judge/POLICY.md",
]
DELIVERY = [
    "tasks/BENCHMARK.yaml", "images/judge/Dockerfile", "images/judge/.dockerignore", "images/judge/protected_hashes.txt", "images/work/Dockerfile",
    "images/work/.dockerignore", "images/work/codex-config.toml", "images/work/benchmark/run_public_tests.sh", "images/work/benchmark/prepare_submit.sh",
    f"{LAB}/FINAL_VERSION", f"{LAB}/notes.md", f"{LAB}/candidates/policy_v001/submission.py", f"{LAB}/tools/run_one.sh", f"{LAB}/tools/run_public_tests.sh",
    "corpus/public/MANIFEST.json", "scripts/lib.sh", "scripts/build_images.sh", "scripts/eval_archive.sh", "scripts/sforge_entry.py",
    "scripts/validate_layout.py", "scripts/update_protected_hashes.py", "scripts/write_portable_manifests.py", "scripts/refresh_portable_manifests.py",
]
WEIGHT = "equal weight inside its family mean; families geomean per ratio; ratios mean"
MODE = "phase_a_context_only_indices_then_trusted_phase_b_replay"


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
    fam = collections.Counter(x["family"] for x in public)
    cases = [{"item_id": x["item_id"], "family": x["family"], "task": x["task"], "context_tokens": x["meta"]["context_tokens"], "in_one": x["item_id"] in one,
              "in_quick": x["item_id"] in quick, "ratios_per_mille": [500, 750, 875], "role": "performance", "measurement_mode": MODE,
              "weight": {"numerator": 1, "denominator": fam[x["family"]]}, "weight_semantics": WEIGHT} for x in public]
    write("public-suite-manifest.json", render({"schema_version": 1, "suite_id": rt.PUBLIC_SUITE_ID, "case_counts": {"total": len(cases), "scored_performance": len(cases), "zero_weight_hard_gate": 0},
                                                "subsets": {"one": sorted(one), "quick": [x["item_id"] for x in rt.suite_items("quick")]}, "cases": "__LIST__"}, "cases", cases))
    manifest = json.loads((ROOT / "judge/private/final_manifest.json").read_text())
    counts = manifest["counts_per_family"]
    final_cases = [{"item_id": e["item_id"], "family": e["family"], "context_tokens": e["context_tokens"], "ratios_per_mille": [500, 750, 875], "role": "performance",
                    "measurement_mode": MODE, "weight": {"numerator": 1, "denominator": counts[e["family"]]}} for e in manifest["items"]]
    frozen = {"path": "judge/private/final_manifest.json", "sha256": hashlib.sha256((ROOT / "judge/private/final_manifest.json").read_bytes()).hexdigest(),
              "provenance": "scripts/prepare_items.py build (seed kvpress-kv-eviction-v2) from the pinned kvpress registry datasets; items rebuilt and SHA-256-checked at image build"}
    for name, role in (("final-suite-manifest.json", "final"), ("interactive-suite-manifest.json", "interactive")):
        doc = {"schema_version": 1, "suite_id": rt.FINAL_SUITE_ID, "role": role, "case_counts": {"total": len(final_cases), "scored_performance": len(final_cases), "zero_weight_hard_gate": 0},
               "counts_per_family": counts, "frozen_instance": frozen, "same_suite_for_interactive_and_final": True, "cases": "__LIST__"}
        write(name, render(doc, "cases", final_cases))
    write("oracle-protocol.json", render({
        "schema_version": 1, "protocol_id": "trusted-replay-fidelity-kl-v2",
        "acceptance": "every run-wide gate passes; otherwise the score is 0 and the result is a trusted gate failure",
        "fidelity": {"per_item_ratio": "exp(-mean_token KL(full || compressed))", "family": "arithmetic mean over the family's items",
                     "ratio": "geometric mean over retrieval, aggregation, qa, generation", "primary": "100 x mean over 500, 750, 875 per mille eviction"},
        "gates": {"names": ["profile_ok", "suite_ok", "source_ok", "hardware_ok", "indices_ok", "budgets_ok", "immutable_ok", "rng_ok", "latency_ok", "memory_ok", "fidelity_ok"],
                  "latency_suite_fraction_percent": 30, "latency_item_cap_percent": 60, "latency_denominator": "trusted plain prefill of the same context, same GPU, Phase B",
                  "memory_limit_bytes": rt.MEMORY_LIMIT_BYTES, "min_keep": rt.MIN_KEEP, "n_sink_floor_advisory": 4},
        "integrity": ["source_admission", "fresh_policy_per_item_and_ratio", "fresh_input_copies_per_select_slowest_ratio_charged", "context_only_phase_a", "immutable_inputs", "rng_unchanged", "llm_integrity_review"],
        "reference": {"model": rt.MODEL_ID, "phase_a": "unprivileged worker: context-only prefill, candidate select, validated selections stored as per-(layer, head) bitmasks", "phase_b": "trusted replay: plain prefill, reference continuation, bitmask re-validation, teacher-forced full and compressed replay with kept pairs in position order",
                      "reference_tokens_min": rt.R_MIN, "reference_tokens_max": rt.R_MAX, "eos_suppressed_for_first_tokens": rt.R_MIN, "golden": "full-cache distribution recomputed every run"},
    }))
    write("selection-protocol.json", render({
        "schema_version": 1, "protocol_id": "kv-eviction-full-final-shared-v2-sharded", "evaluation_policy": "full-final-shared-v2", "same_suite_for_interactive_and_final": True,
        "families": list(rt.FAMILIES), "ratios_per_mille": [500, 750, 875],
        "public": {"suite_id": rt.PUBLIC_SUITE_ID, "full": {"case_count": len(public)}, "quick": {"case_count": len(quick), "rule": "first item of every (family, task)"}, "one": {"case_count": 1, "rule": "first public item"}},
        "interactive": {"suite_id": rt.FINAL_SUITE_ID, "case_count": len(final_cases), "routing": "round_id prefix auto- or agent-", "rounds": "scored agent submissions; archive FINAL_VERSION only", "run_gate_policy": "all_11_run_wide_gates_must_pass"},
        "final": {"suite_id": rt.FINAL_SUITE_ID, "case_count": len(final_cases), "routing": "round_id prefix manual- or candidate-", "rounds": "deadline final and trusted host replays", "run_gate_policy": "all_11_run_wide_gates_must_pass"},
        "sharding": {"gpus": "1-8 identical devices visible to the Judge", "partition": "longest-processing-time by estimated cost, deterministic", "per_shard": "Phase A then Phase B on one GPU, NUMA-local whole-core CPU slice", "score_dependence": "none"},
    }))
    write("software-abi.json", render({
        "schema_version": 1, "protocol_id": "pytorch280-cu129-transformers4571-sdpa-v2", "language": "python3", "source_runtime": ["pytorch-2.8.0", "cuda-12.9", "triton-3.4.0", "transformers-4.57.1", "accelerate-1.10.1"],
        "accelerator_runtime": "cuda-12.9-compatible", "attention_implementation": "sdpa", "required_dtypes": ["bf16", "fp32"], "forbidden_packages": ["kvpress", "flash_attn", "xformers", "flashinfer"],
        "model_binding": {"model_id": rt.MODEL_ID, "mount": "/opt/models/qwen3-8b (read-only)", "adapter": "scripts/sforge_entry.py", "offline": True},
        "policy_assets": {"file": "candidates/policy_vNNN/assets.pt", "loader": "torch.load(weights_only=True) string-to-tensor dict", "max_bytes": rt.MAX_ASSET_BYTES},
    }))
    write("calibration-protocol.json", render({
        "schema_version": 1, "protocol_id": "host-wall-cuda-event-slowest-ratio-v2", "clock": "max_of_synchronized_host_wall_and_cuda_event_per_layer_select",
        "aggregation": "per layer the slowest of the item's three ratio calls, summed over layers", "inputs": "fresh copies of the per-item tensors for every timed select call, after one untimed select pass per ratio on random tensors with the item's shapes; fresh policy instance per item and ratio",
        "warmup": "one untimed fully gated select pass over all layers per ratio on a 2048-token random decoy", "fresh_process": "per shard: Phase A and Phase B are separate processes on the same GPU",
        "latency_reference": "trusted plain prefill of the same context re-measured in Phase B on the same device", "comparability": "same_measurement_profile",
    }))
    print("wrote portable/*.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
