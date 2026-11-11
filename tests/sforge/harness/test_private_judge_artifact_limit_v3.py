"""Private evidence count expansion retains the existing path restrictions."""
import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest

REPO = Path(os.environ.get("SFORGE_LIMIT_TEST_REPOSITORY", str(Path(__file__).resolve().parents[3])))
SOURCE = Path(os.environ.get("SFORGE_LIMIT_TEST_SOURCE", str(REPO / "sforge/harness/task_spec.py")))

@pytest.fixture(scope="module")
def task_spec():
    spec = importlib.util.spec_from_file_location("artifact_limit16_task_spec_under_test", SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module

def test_current_minference_private_files_load_without_rewriting_metadata(task_spec):
    document = json.loads((REPO / "tasks/latest/minference-sparse-prefill-h100-v3-seed/tasks/minference_sparse_prefill_h100_seed_v3.json").read_bytes())
    paths = document["judge"]["private_artifact_paths"]
    assert len(paths) == 3
    assert document["judge"]["private_artifact_path"] in paths
    assert task_spec.validate_private_artifact_paths(paths) == paths

def test_exact_sixteen_is_accepted_and_seventeen_is_rejected(task_spec):
    assert task_spec.PRIVATE_JUDGE_ARTIFACT_MAX_FILES == 16
    paths = [f"/logs/verifier/private-{index:02d}.json" for index in range(17)]
    assert task_spec.validate_private_artifact_paths(paths[:16]) == paths[:16]
    with pytest.raises(ValueError, match="at most 16"):
        task_spec.validate_private_artifact_paths(paths)

@pytest.mark.parametrize("paths", [
    ["/logs/verifier/a.json", "/logs/verifier/a.json"],
    ["/logs/verifier/../private.json"],
    ["/logs/verifier/nested/a.json"],
    ["/logs//verifier/a.json"],
    ["/tmp/a.json"],
    ["/logs/verifier/a\n.json"],
    ["/logs/verifier/a\x00.json"],
    [123],
    "/logs/verifier/a.json",
])
def test_count_change_does_not_relax_canonical_unique_root_paths(task_spec, paths):
    with pytest.raises(ValueError):
        task_spec.validate_private_artifact_paths(paths)

def test_empty_and_legacy_single_file_behavior_is_preserved(task_spec):
    assert task_spec.validate_private_artifact_paths([]) == []
    path = "/logs/verifier/legacy.json"
    assert task_spec.validate_private_artifact_paths([path]) == [path]
