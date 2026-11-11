from __future__ import annotations

import copy
import gzip
import hashlib
import importlib.util
import io
import json
import tarfile

import pytest


def load_validator(task_root):
    path = task_root / "scripts/validate_formal_artifacts.py"
    spec = importlib.util.spec_from_file_location("formal_provenance_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def valid_private(module, hidden):
    records = [
        {
            "row_id": case["row_id"],
            "case_sha256": hashlib.sha256(case["case_id"].encode()).hexdigest(),
            "correct": False,
            "parsed": True,
            "useful_tokens": 1,
            "seconds": 1.0,
        }
        for case in hidden
    ]
    commitment = hashlib.sha256(
        "\n".join(sorted(record["case_sha256"] for record in records)).encode()
    ).hexdigest()
    return {
        "evaluation_mode": "final",
        "workload_profile": module.WORKLOAD_PROFILE,
        "generation_count": module.GENERATION_COUNT,
        "requested_slots": module.REQUESTED_SLOTS,
        "cohort_case_count": module.GENERATION_COUNT,
        "cohort_row_counts": module.FINAL_ROW_COUNTS,
        "cohort_category_definition": module.CATEGORY_DEFINITION,
        "cohort_category_counts": module.CATEGORY_COUNTS,
        "cohort_bonus_categories": [],
        "submission_number": None,
        "cohort_cycle": None,
        "cohort_slot": None,
        "cohort_case_set_sha256": commitment,
        "records": records,
        "rows": {
            row_id: {
                "cases": count,
                "correct": 0,
                "parsed": count,
                "useful_tokens": count,
                "seconds": float(count),
                "throughput_tok_s": 1.0,
            }
            for row_id, count in module.FINAL_ROW_COUNTS.items()
        },
    }


def test_formal_workload_recomputes_rows_score_and_canonical_mapping(task_root, private_data):
    module = load_validator(task_root)
    hidden = json.loads(
        gzip.decompress(
            (private_data / "hidden_cases.json.gz").read_bytes()
        )
    )
    private = valid_private(module, hidden)
    assert module.validate_final_workload(private, hidden) == pytest.approx(1.0)

    permuted = copy.deepcopy(private)
    first = 0
    second = next(
        index
        for index, record in enumerate(permuted["records"])
        if record["row_id"] != permuted["records"][first]["row_id"]
    )
    permuted["records"][first]["case_sha256"], permuted["records"][second][
        "case_sha256"
    ] = (
        permuted["records"][second]["case_sha256"],
        permuted["records"][first]["case_sha256"],
    )
    with pytest.raises(ValueError, match="wrong canonical row"):
        module.validate_final_workload(permuted, hidden)

    forged_score = copy.deepcopy(private)
    forged_score["rows"][forged_score["records"][0]["row_id"]][
        "throughput_tok_s"
    ] = 99.0
    with pytest.raises(ValueError, match="row throughput mismatch"):
        module.validate_final_workload(forged_score, hidden)


def test_formal_private_artifact_rejects_noncanonical_tar_names(task_root, tmp_path):
    module = load_validator(task_root)
    artifact = tmp_path / "private.tar.gz"
    payload = b"{}"
    with tarfile.open(artifact, "w:gz") as bundle:
        member = tarfile.TarInfo("../decoder_private.json")
        member.size = len(payload)
        bundle.addfile(member, io.BytesIO(payload))
    with pytest.raises(ValueError, match="unexpected private artifact"):
        module.exact_private_payload(artifact)


def test_release_preflights_require_host_tool_provenance(task_root):
    finalizer = (task_root / "scripts/finalize_h100_release_policy.py").read_text(encoding="utf-8")
    evaluator = (task_root / "scripts/validate_eval_artifacts.py").read_text(encoding="utf-8")
    for field in (
        "archive_sha256",
        "workload_policy_sha256",
        "memory_policy_sha256",
        "model_validation_sha256",
        "runtime_entry_sha256",
        "source_digest",
        "work_image_id",
        "judge_image_id",
    ):
        assert field in finalizer
        assert field in evaluator


def test_manual_eval_closes_source_and_image_toctou_after_model_recheck(task_root):
    source = (task_root / "scripts/eval_archive.sh").read_text(encoding="utf-8")
    for literal in (
        'source_digest_start="$(jq -er',
        'work_image_id_start="$(jq -er',
        'judge_image_id_start="$(jq -er',
        'source_digest_end="$(image_source_digest)"',
        'work_image_id_end="$(docker image inspect',
        'judge_image_id_end="$(docker image inspect',
        '"${source_digest_end}" == "${source_digest_start}"',
        '"${work_image_id_end}" == "${work_image_id_start}"',
        '"${judge_image_id_end}" == "${judge_image_id_start}"',
        'judge_runtime_tree_sha256_end',
        '"${judge_runtime_tree_sha256_end}" == "${judge_runtime_tree_sha256_start}"',
    ):
        assert literal in source

    post_model = source.index('--output "${model_validation_end_snapshot}"')
    model_accept = source.index(
        'cp -- "${model_validation_end_snapshot}" "${run_root}/full-model-validation.json"'
    )
    end_barrier = source.index("Evaluation source/image end barrier: PASS")
    manifest_accept = source.index(
        'cp -- "${evaluation_manifest_snapshot}" "${run_root}/evaluation-image-manifest.json"'
    )
    private_accept = source.index(
        'python3 "${MIGRATION_ROOT}/scripts/validate_eval_artifacts.py"'
    )
    assert post_model < model_accept < end_barrier < manifest_accept < private_accept
