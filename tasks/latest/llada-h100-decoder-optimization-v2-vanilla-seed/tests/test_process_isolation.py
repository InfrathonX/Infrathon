from __future__ import annotations

import ast


def test_candidate_worker_is_gold_free(task_root):
    worker = (
        task_root / "images/judge/verifier/candidate_worker.py"
    ).read_text(encoding="utf-8")
    for forbidden in (
        "HIDDEN_PATH",
        "REFERENCE_PATH",
        "score_completion",
        "quality_diagnostic",
        'request["request_text"]',
        'request["gold"]',
        'request["tests"]',
        "SFORGE_PRIVATE_CONTEXT",
        "SFORGE_EVAL_ROUND_ID",
        "COHORT_SECRET",
    ):
        assert forbidden not in worker
    assert 'request["prompt_ids"]' in worker
    assert "PR_SET_NO_NEW_PRIVS" in worker


def test_controller_never_imports_candidate(task_root):
    controller = (
        task_root / "images/judge/verifier/verify_decoder.py"
    ).read_text(encoding="utf-8")
    assert "import_candidate" not in controller
    assert "load_model_and_candidate" not in controller
    assert "CandidateWorker" in controller
    assert '"prompt_ids": prompt_ids' in controller
    assert controller.index("worker = CandidateWorker") < controller.index(
        "hidden = json.loads"
    )
    assert controller.index("worker = CandidateWorker") < controller.index(
        "trusted_metadata, cohort_secret = read_evaluation_context()"
    )
    assert '"gold"' not in controller.split("class CandidateWorker", 1)[1].split(
        "def encode_prompt", 1
    )[0]


def test_candidate_worker_execs_into_a_clean_process_image(task_root):
    controller = (
        task_root / "images/judge/verifier/verify_decoder.py"
    ).read_text(encoding="utf-8")
    worker = (
        task_root / "images/judge/verifier/candidate_worker.py"
    ).read_text(encoding="utf-8")
    candidate_worker = controller.split("class CandidateWorker", 1)[1].split(
        "def _receive", 1
    )[0]
    assert "os.execve(" in candidate_worker
    assert "from candidate_worker import serve" not in controller
    assert '"COHORT_SECRET"' not in candidate_worker
    assert '"SFORGE_EVAL_RUN_ID"' not in candidate_worker
    assert "verify_decoder" not in worker
    assert "from cohort" not in worker


def test_quality_diagnostic_does_not_gate_throughput(task_root):
    controller = (
        task_root / "images/judge/verifier/verify_decoder.py"
    ).read_text(encoding="utf-8")
    assert "valid = performance_ok" in controller
    assert "valid = quality_ok and performance_ok" not in controller
    assert "geomean if valid else 0.0" in controller


def test_pilot_does_not_require_release_quality_or_memory_evidence(task_root):
    controller = (
        task_root / "images/judge/verifier/verify_decoder.py"
    ).read_text(encoding="utf-8")
    runner = (task_root / "scripts/run_fixed_budget.sh").read_text(
        encoding="utf-8"
    )
    formal = (task_root / "scripts/validate_formal_artifacts.py").read_text(
        encoding="utf-8"
    )

    assert "not calibration and not qualification and not pilot" in controller
    assert "if not pilot and (" in controller
    assert "enforce_memory_stability=not pilot" in controller
    assert "enforce_stability=not pilot" in controller
    assert "and (memory_leak_free or pilot)" in controller
    assert '_positive_env("LLADA_PILOT_FINAL_OUTER_TIMEOUT")' in controller
    assert 'if [[ "${PILOT_MODE}" == "1" ]]; then' in runner
    assert "PILOT_FINAL_VERIFIER_TIMEOUT=7000" in runner
    assert "PILOT_FINAL_JUDGE_TIMEOUT=7200" in runner
    pilot_gate_block = formal.split("required_gates =", 1)[1].split(
        "for gate in required_gates", 1
    )[0]
    assert '"memory_leak_free"' in pilot_gate_block.split(
        "if not args.pilot:", 1
    )[1]


def test_code_grader_has_a_distinct_uid(task_root):
    quality = (
        task_root / "images/judge/verifier/quality.py"
    ).read_text(encoding="utf-8")
    dockerfile = (task_root / "images/judge/Dockerfile").read_text(encoding="utf-8")
    assert 'pwd.getpwnam("mlsysforge-grader")' in quality
    assert "groupadd --gid 60001 mlsysforge-grader" in dockerfile
    assert "useradd --uid 60001 --gid 60001 --no-create-home" in dockerfile
    assert "useradd --system" not in dockerfile
    assert quality.index("os.setresuid(") < quality.index(
        "resource.setrlimit(resource.RLIMIT_NPROC"
    )


def test_public_code_grader_does_not_apply_host_wide_nproc_to_agent_uid(task_root):
    quality = (
        task_root / "images/work/public_tests/public_quality.py"
    ).read_text(encoding="utf-8")
    root_branch = quality.split("if os.geteuid() == 0:", 1)[1].split(
        "def run_restricted_python", 1
    )[0]
    assert "resource.setrlimit(resource.RLIMIT_NPROC" in root_branch
    assert "RLIMIT_NPROC is therefore a private-Judge boundary only" in quality


def test_protected_manifest_paths_exactly_match_verifier_mapping(task_root):
    verifier_path = task_root / "images/judge/verifier/verify_decoder.py"
    module = ast.parse(verifier_path.read_text(encoding="utf-8"))
    verify_function = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "verify_protected"
    )
    mapping_assignment = next(
        node
        for node in verify_function.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "mapping" for target in node.targets)
    )
    assert isinstance(mapping_assignment.value, ast.Dict)
    mapping_paths = {
        ast.literal_eval(key)
        for key in mapping_assignment.value.keys
        if key is not None
    }
    manifest_paths = {
        line.split(None, 1)[1]
        for line in (
            task_root / "images/judge/protected_hashes.txt"
        ).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    assert mapping_paths == manifest_paths
