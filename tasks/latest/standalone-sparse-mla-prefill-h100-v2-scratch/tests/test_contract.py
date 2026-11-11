from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import subprocess
import tarfile
from pathlib import Path

import yaml

from conftest import TASK_ROOT, load_module


TASK_ID = "standalone_sparse_mla_prefill_h100_scratch_v2"
layout = load_module("sparse_mla_layout_contract", "scripts/validate_layout.py")
formal = load_module("sparse_mla_formal_contract", "scripts/validate_formal_artifacts.py")
public_runner = load_module(
    "sparse_mla_public_controller_contract", "images/work/benchmark/public_runner.py"
)


def shell_commands_from_markdown(markdown: str) -> list[list[str]]:
    fenced_shell = re.compile(r"```(?:bash|sh)\s*\n(.*?)```", re.DOTALL)
    commands: list[list[str]] = []
    for block in fenced_shell.findall(markdown):
        for line in block.splitlines():
            if line.strip() and not line.lstrip().startswith("#"):
                commands.append(shlex.split(line))
    return commands


def capture_wrapper_exec(
    script: Path, arguments: list[str], tmp_path: Path, label: str
) -> list[str]:
    capture = tmp_path / f"{label}.argv"
    harness = (
        'selected="$1"; capture="$2"; shift 2; '
        'exec() { printf "%s\\n" "$@" > "$capture"; }; '
        'source "$selected"'
    )
    subprocess.run(
        [
            "bash",
            "-c",
            harness,
            "wrapper-harness",
            str(script),
            str(capture),
            *arguments,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return capture.read_text(encoding="utf-8").splitlines()


def test_required_contract_files_exist() -> None:
    required = (
        "runtime/sparse_mla_prefill_runtime.py",
        "images/judge/candidate_worker.py",
        "images/judge/protected_hashes.txt",
        "images/judge/run_judge.py",
        "images/judge/source_admission.py",
        "images/judge/verify_sparse_mla.py",
        "images/work/benchmark/public_edge_cases.json",
        "images/work/benchmark/public_runner.py",
        "images/work/benchmark/run_public_tests.sh",
        "images/work/benchmark/source_admission.py",
        "images/work/AGENT_TASK.md",
        "images/work/workdir/sparse_mla_prefill_lab/README_AGENT.md",
        "llm_judge/POLICY.md",
        "oracle/candidates/kernel_v001/submission.py",
        "images/work/workdir/sparse_mla_prefill_lab/candidates/kernel_v001/submission.py",
        "judge/run_judge.py",
        "scripts/generate_final_manifest.py",
        "scripts/generate_edge_manifest.py",
        "scripts/make_oracle_archive.sh",
        "scripts/validate_formal_artifacts.py",
        "scripts/validate_layout.py",
        "tasks/BENCHMARK.yaml",
        f"tasks/{TASK_ID}.json",
    )
    missing = [relative for relative in required if not (TASK_ROOT / relative).is_file()]
    assert missing == []
    assert sorted(path.name for path in (TASK_ROOT / "tasks").glob("*.json")) == [f"{TASK_ID}.json"]


def test_mirrored_admission_and_adapter_are_byte_identical() -> None:
    assert (TASK_ROOT / "images/work/benchmark/source_admission.py").read_bytes() == (
        TASK_ROOT / "images/judge/source_admission.py"
    ).read_bytes()
    assert (TASK_ROOT / "judge/run_judge.py").read_bytes() == (
        TASK_ROOT / "images/judge/run_judge.py"
    ).read_bytes()


def test_protected_hash_manifest_matches_image_driving_sources() -> None:
    mapping = {
        "judge:candidate_worker.py": TASK_ROOT / "images/judge/candidate_worker.py",
        "judge:run_judge.py": TASK_ROOT / "images/judge/run_judge.py",
        "judge:source_admission.py": TASK_ROOT / "images/judge/source_admission.py",
        "judge:verify_sparse_mla.py": TASK_ROOT / "images/judge/verify_sparse_mla.py",
        "worker:sparse_mla_prefill_runtime.py": TASK_ROOT / "runtime/sparse_mla_prefill_runtime.py",
    }
    observed: dict[str, str] = {}
    for raw in (TASK_ROOT / "images/judge/protected_hashes.txt").read_text(
        encoding="utf-8"
    ).splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        digest, label = line.split(None, 1)
        assert label not in observed
        observed[label] = digest
    assert set(observed) == set(mapping)
    for label, path in mapping.items():
        assert observed[label] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_layout_and_formal_validators_consume_the_v3_identity() -> None:
    assert layout.TASK_ID == TASK_ID
    assert layout.BENCHMARK == "standalone_sparse_mla_prefill"
    assert layout.TASK_DIRECTORY == "standalone-sparse-mla-prefill-h100-v2-scratch"
    assert layout.WORKSPACE == "/app/workdir/sparse_mla_prefill_lab"
    assert layout.PRIVATE_ARTIFACT == (
        "/logs/verifier/standalone_sparse_mla_prefill_v2_results.json"
    )
    assert (layout.FINAL_SUITE_ID, layout.EDGE_SUITE_ID, layout.PUBLIC_EDGE_SUITE_ID) == (
        "final-holdout-v4", "edge-holdout-v3", "edge-public-v3",
    )
    assert layout.FINAL_MANIFEST_SCHEMA_VERSION == 5
    assert formal.PRIVATE_MEMBER == "standalone_sparse_mla_prefill_v2_results.json"
    assert formal.FINAL_SUITE_ID == "final-holdout-v4"
    assert formal.EDGE_SUITE_ID == "edge-holdout-v3"
    assert formal.EDGE_CASE_COUNT == layout.EXPECTED_EDGE_CASES == 470
    assert layout.EXPECTED_FINAL_CASES == 997 and layout.EXPECTED_PUBLIC_EDGE_CASES == 73
    assert formal.EVALUATION_POLICY == "full-final-shared-v3"
    assert formal.SELECTION_PROTOCOL_ID == "sparse-mla-shared-final-v3"
    assert formal.ADAPTER_SOURCE == "standalone_sparse_mla_prefill_v2_verifier_adapter"
    assert formal.FAMILY_KEYS == ("v32_tflops", "glm5_tflops", "v4_flash_tflops", "v4_pro_tflops")
    assert not hasattr(formal, "AUTO_SUITE_ID")
    assert layout.validate_layout(TASK_ROOT) == []


def test_lib_and_build_carry_the_same_suite_identities_and_counts() -> None:
    lib = (TASK_ROOT / "scripts/lib.sh").read_text(encoding="utf-8")
    for fragment in (
        f'TASK_ID="{TASK_ID}"',
        'FINAL_SUITE_ID="final-holdout-v4"',
        'EDGE_SUITE_ID="edge-holdout-v3"',
        'PUBLIC_EDGE_SUITE_ID="edge-public-v3"',
        "FINAL_CASE_COUNT=997",
        "EDGE_CASE_COUNT=470",
        "PUBLIC_EDGE_CASE_COUNT=73",
        "EDGE_CASES_PATH",
        "Edge holdout freshness: PASS",
    ):
        assert fragment in lib, fragment
    build = (TASK_ROOT / "scripts/build_images.sh").read_text(encoding="utf-8")
    work_docker = (TASK_ROOT / "images/work/Dockerfile").read_text(encoding="utf-8")
    judge_docker = (TASK_ROOT / "images/judge/Dockerfile").read_text(encoding="utf-8")
    holdout_layer = (TASK_ROOT / "images/judge/Dockerfile.private").read_text(encoding="utf-8")
    assert build.count('--build-context "final_cases=${final_cases_context}"') == 1
    assert '"${final_cases_context}/edge_cases.json"' in build
    assert '"${PUBLIC_EDGE_CASE_COUNT}"' in build
    assert "COPY --from=final_cases" not in work_docker
    assert "COPY benchmark/ /workspace/benchmark/" in work_docker
    assert "final_cases" not in judge_docker
    assert "FROM ${JUDGE_PUBLIC_IMAGE}" in holdout_layer
    assert "COPY --from=final_cases final_cases.json edge_cases.json" in holdout_layer
    assert "compute-sanitizer" in judge_docker
    run = (TASK_ROOT / "scripts/run_fixed_budget.sh").read_text(encoding="utf-8")
    for fragment in (
        'manual_evaluation_suite: "final-holdout-v4"',
        'evaluation_policy: "full-final-shared-v3"',
        'selection_protocol_id: "sparse-mla-shared-final-v3"',
        "edge_holdout: {",
        'validate_formal_artifacts.py" stage',
    ):
        assert fragment in run, fragment
    assert run.index('validate_formal_artifacts.py" stage') < run.index('validate_formal_artifacts.py" pre')


def test_no_v1_identity_remains_in_task_owned_metadata_and_scripts() -> None:
    stale = (
        "final-holdout-v1",
        "full-final-shared-v2",
        "sparse-mla-shared-final-v2",
        "standalone_sparse_mla_prefill_h100_scratch_v1",
        "standalone-sparse-mla-prefill-h100-v1-scratch",
        "standalone_sparse_mla_prefill_v1_",
        "standalone-sparse-mla-v1",
        "flashmla-sparse-prefill-v1",
        "m1_64",
        "m1_128",
    )
    paths = [
        *sorted((TASK_ROOT / "scripts").glob("*.sh")),
        *sorted((TASK_ROOT / "scripts").glob("*.py")),
        *sorted((TASK_ROOT / "tasks").iterdir()),
        *sorted(TASK_ROOT.glob("experiment-*.yaml")),
        *(path for path in sorted((TASK_ROOT / "portable").glob("*.json"))),
    ]
    for path in paths:
        if not path.is_file():
            continue
        payload = path.read_text(encoding="utf-8")
        if path.name == "selection-protocol.json":
            # The one deliberate historical reference.
            document = json.loads(payload)
            assert "final-holdout-v1" in document.pop("historical_interactive_policy")
            payload = json.dumps(document)
        for token in stale:
            assert token not in payload, (path.relative_to(TASK_ROOT), token)


def test_task_metadata_selects_controller_covering_the_public_cases(
    tmp_path: Path,
) -> None:
    task = json.loads(
        (TASK_ROOT / "tasks" / f"{TASK_ID}.json").read_text(encoding="utf-8")
    )
    # The run configuration is internal and optional (the public tree has none).
    experiment_path = next(
        iter(sorted(TASK_ROOT.glob("experiment-*-max-2h.yaml"))), TASK_ROOT / "experiment.yaml"
    )
    experiment = (
        yaml.safe_load(experiment_path.read_text(encoding="utf-8"))
        if experiment_path.is_file()
        else None
    )
    benchmark = yaml.safe_load(
        (TASK_ROOT / "tasks/BENCHMARK.yaml").read_text(encoding="utf-8")
    )
    assert task["task_id"] == TASK_ID
    assert benchmark["name"] == "standalone_sparse_mla_prefill"
    assert task["base_image"] in benchmark["base_images"]
    assert task["cwd"] == "/app/workdir/sparse_mla_prefill_lab"
    assert task["internet"] is True
    if experiment is not None:
        assert experiment["defaults"]["internet"] is True
    assert task["work"]["image_tag"] == task["judge"]["image_tag"] == "local"
    assert task["judge"]["parser"] == "structured_json"
    assert task["judge"]["score_direction"] == "maximize"
    assert task["judge"]["private_artifact_path"] == (
        "/logs/verifier/standalone_sparse_mla_prefill_v2_results.json"
    )
    if experiment is not None:
        assert set(experiment["tasks"]) == {TASK_ID}
        assert isinstance(experiment["defaults"]["model"], str) and experiment["defaults"]["model"]
        assert experiment["env"]["SFORGE_CODEX_REASONING_EFFORT"] == "max"
    assert task["submit_paths"] == ["candidates", "FINAL_VERSION", "notes.md"]

    query = task["work"]["agent_query"]
    for required in ("d_qk of 512 or 576", "multiple of 16 from 16 to 128", "s_q on grid axis 0", "64-bit",
                     "1048576", "stale indices past topk_length", "streams of consecutive calls",
                     "FP32 accumulation", "no FP8, INT8 or FP4", "--edge", "--sanitize", "hidden test cases"):
        assert required in query, required
    commands = shell_commands_from_markdown(query)
    public_commands = [
        command
        for command in commands
        if command and Path(command[0]).name == "run_public_tests.sh"
    ]
    assert len(public_commands) == 3
    command = ["1" if token == "N" else token for token in public_commands[0]]
    assert Path(command[0]) == Path("tools/run_public_tests.sh")
    assert command.count("--candidate-version") == 1
    version_index = command.index("--candidate-version")
    assert int(command[version_index + 1]) in range(1, 1000)
    assert not {"--quick", "--edge", "--sanitize"} & set(command)

    workspace = TASK_ROOT / "images/work/workdir/sparse_mla_prefill_lab"
    selected_controller = workspace / command[0]
    assert selected_controller.is_file()
    workspace_dispatch = capture_wrapper_exec(
        selected_controller, command[1:], tmp_path, "workspace-controller"
    )
    assert workspace_dispatch == [
        "/workspace/benchmark/run_public_tests.sh",
        "--candidate-version",
        command[version_index + 1],
    ]

    benchmark_controller = (
        TASK_ROOT / "images/work/benchmark" / Path(workspace_dispatch[0]).name
    )
    benchmark_dispatch = capture_wrapper_exec(
        benchmark_controller, workspace_dispatch[1:], tmp_path, "benchmark-controller"
    )
    assert benchmark_dispatch == [
        "python3",
        "/workspace/benchmark/public_runner.py",
        "--lab-root",
        task["cwd"],
        "--suite",
        "full",
        "--candidate-version",
        command[version_index + 1],
    ]

    selected_suite = benchmark_dispatch[benchmark_dispatch.index("--suite") + 1]
    selected_runner = TASK_ROOT / "images/work/benchmark" / Path(benchmark_dispatch[1]).name
    assert Path(public_runner.__file__).resolve() == selected_runner.resolve()
    full_cases = public_runner.cases_for_suite(selected_suite)
    assert full_cases == public_runner.public_cases() and len(full_cases) == 108
    base = {"case_id", "group", "family", "h_q", "topk", "s_q", "s_kv", "sm_scale", "sink_mode", "seed"}
    extra = {"basic": {"inputs"}, "length": {"length_divisor", "ragged"}, "multi": {"request_lengths"},
             "trace": {"calls"}, "replay": {"calls"}, "pattern": {"index_pattern"}}
    assert all(set(case) == base | extra.get(case["group"], set()) for case in full_cases)
    assert all(case["seed"] < 2**32 for case in full_cases)


def test_work_starter_is_exactly_empty_and_selected() -> None:
    workspace = TASK_ROOT / "images/work/workdir/sparse_mla_prefill_lab"
    starter = workspace / "candidates/kernel_v001/submission.py"
    assert starter.is_file()
    assert starter.stat().st_size == 0
    assert starter.read_bytes() == b""
    assert (workspace / "FINAL_VERSION").read_text(encoding="utf-8").strip() == "1"


def _oracle_selector(name: str) -> int:
    text = (TASK_ROOT / "oracle" / name).read_text(encoding="utf-8")
    assert re.fullmatch(r"[1-9][0-9]{0,2}\n?", text), name
    return int(text)


def test_oracle_selectors_name_existing_candidates() -> None:
    # kernel_v001 stays the in-task Triton control; FINAL_VERSION selects the
    # strongest oracle (the forced FlashMLA paper candidate), ADMISSIBLE_VERSION
    # the candidate the normal Judge admits.
    assert (TASK_ROOT / "oracle/candidates/kernel_v001/submission.py").is_file()
    for name in ("FINAL_VERSION", "ADMISSIBLE_VERSION"):
        version = _oracle_selector(name)
        assert (TASK_ROOT / f"oracle/candidates/kernel_v{version:03d}/submission.py").is_file()


def test_oracle_archive_contains_only_explicit_submission_members(
    tmp_path: Path,
) -> None:
    script = TASK_ROOT / "scripts/make_oracle_archive.sh"
    for args, selector in (([], "FINAL_VERSION"), (["--admissible"], "ADMISSIBLE_VERSION")):
        out = tmp_path / selector
        env = dict(os.environ)
        env["ARTIFACTS_DIR"] = str(out)
        completed = subprocess.run(
            ["bash", str(script), *args],
            cwd=TASK_ROOT,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        archive = Path(completed.stdout.splitlines()[0])
        assert archive == out / "oracle.tar.gz"
        version = _oracle_selector(selector)
        selected = f"candidates/kernel_v{version:03d}/submission.py"
        with tarfile.open(archive, mode="r:gz") as bundle:
            members = bundle.getmembers()
            assert {member.name for member in members} == {"FINAL_VERSION", "notes.md", selected}
            assert all(member.isfile() for member in members)
            assert int(bundle.extractfile("FINAL_VERSION").read()) == version
            assert (
                bundle.extractfile(selected).read()
                == (TASK_ROOT / "oracle" / selected).read_bytes()
            )
        formal._validate_archive(
            archive,
            root=out.resolve(),
            expected_size=archive.stat().st_size,
        )


def test_oracle_smoke_skips_only_the_llm_review_of_the_trusted_control() -> None:
    smoke = (TASK_ROOT / "scripts/smoke_test.sh").read_text(encoding="utf-8")
    evaluate = (TASK_ROOT / "scripts/eval_archive.sh").read_text(encoding="utf-8")
    assert 'PHYSICAL_GPU="${physical_gpu}" TRUSTED_CONTROL=1 \\' in smoke
    assert smoke.count("TRUSTED_CONTROL=1") == 1
    assert 'if [[ "${TRUSTED_CONTROL:-0}" == "1" ]]; then' in evaluate
    assert "review_args=(--skip-integrity-review)" in evaluate
    assert 'run_id="${2:-standalone-sparse-mla-v2-final-eval-' in evaluate
