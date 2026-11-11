"""Contract pins: the duplicated constants can never drift apart."""

from __future__ import annotations

import ctypes
import json

from conftest import TASK_ROOT, _load

runtime = _load("kv_runtime", "runtime/kv_transfer_runtime.py")
verifier = _load("kv_verifier", "images/judge/verify_kv_transfer.py")
admission = _load("kv_admission", "images/judge/source_admission.py")
adapter = _load("kv_adapter", "judge/run_judge.py")


def test_abi_struct_matches_header() -> None:
    header = (TASK_ROOT / "runtime/transfer_abi.h").read_text()
    # ctypes mirror must be exactly 128 bytes and field-for-field aligned.
    assert ctypes.sizeof(runtime.TransferParams) == 128
    for field, _kind in runtime.TransferParams._fields_:
        if field == "_pad0":
            continue
        assert field in header, field


def test_verifier_constants_match_runtime() -> None:
    assert verifier.BS == runtime.BS
    assert verifier.HEAD_SIZES == runtime.HEAD_SIZES
    assert verifier.ESIZE == runtime.ESIZE
    assert verifier.NL_RANGE == runtime.NL_RANGE
    assert verifier.NH_RANGE == runtime.NH_RANGE
    assert verifier.NB_RANGE == runtime.NB_RANGE
    assert verifier.OBJECTS_RANGE == runtime.OBJECTS_RANGE
    assert verifier.ARMS == runtime.ARMS
    assert verifier.FAMILIES == runtime.FAMILIES
    assert verifier.GROUP_FAMILIES == runtime.GROUP_FAMILIES
    assert verifier.GROUP_WEIGHTS == runtime.GROUP_WEIGHTS
    assert verifier.SMALL_CALL_BYTES == runtime.SMALL_CALL_BYTES
    assert verifier.FOREGROUNDS == runtime.FOREGROUNDS
    assert verifier.MAX_DIRECT_WAVES == runtime.MAX_DIRECT_WAVES
    assert verifier.MAX_TRACE_WAVES == runtime.MAX_TRACE_WAVES
    assert verifier.OVERLAP_COVER == runtime.OVERLAP_COVER
    assert verifier.MAX_OVERLAP_LOSS == runtime.MAX_OVERLAP_LOSS
    assert verifier.WAVE_KEYS == runtime.WAVE_KEYS
    assert verifier.CASE_KEYS == runtime.CASE_KEYS
    assert verifier.POOL_KEYS == runtime.POOL_KEYS
    assert verifier.DIRECT_KEYS == runtime.DIRECT_KEYS
    assert verifier.DUPLEX_KEYS == runtime.DUPLEX_KEYS
    assert verifier.TIERS == runtime.TIERS
    assert verifier.STRATUM_RE.pattern == runtime.STRATUM_RE.pattern
    assert verifier.ENVELOPE_MARGIN == runtime.ENVELOPE_MARGIN
    assert verifier.ENVELOPE_PROTOCOL == runtime.ENVELOPE_PROTOCOL
    assert verifier.ENVELOPE_PROBE_BYTES == runtime.ENVELOPE_PROBE_BYTES
    assert verifier.ENVELOPE_PROBE_REPEATS == runtime.ENVELOPE_PROBE_REPEATS
    assert verifier.ENVELOPE_MIN_DEVICE_GBPS == runtime.ENVELOPE_MIN_DEVICE_GBPS
    assert verifier.ENVELOPE_MIN_HOST_GBPS == runtime.ENVELOPE_MIN_HOST_GBPS
    assert verifier.MIN_COMPUTE_CAPABILITY == runtime.MIN_COMPUTE_CAPABILITY
    assert verifier.MIN_DEVICE_MEMORY_BYTES == runtime.MIN_DEVICE_MEMORY_BYTES
    assert verifier.EXPECTED_CONSTANTS == runtime.RECORD_CONSTANTS
    assert verifier.PUBLIC_SUITE_ID == runtime.PUBLIC_SUITE_ID
    assert verifier.FINAL_SUITE_ID == runtime.FINAL_SUITE_ID
    assert verifier.ENTRYPOINT == runtime.ENTRYPOINT


def test_edge_ids_match_runtime() -> None:
    assert tuple(case["case_id"] for case in runtime.edge_fixtures()) == (
        verifier.EDGE_CASE_IDS
    )


def _public_cases():
    return runtime.load_case_table(TASK_ROOT / "images/judge/public_cases.json")["cases"]


def test_wave_arithmetic_matches_runtime() -> None:
    for case in (*_public_cases(), *runtime.edge_fixtures()):
        for wave in case["waves"]:
            assert verifier.wave_logical_bytes(wave) == runtime.wave_logical_bytes(wave)
        assert verifier.case_logical_bytes(case) == runtime.case_logical_bytes(case)


def test_committed_tables_match_runtime() -> None:
    judge = (TASK_ROOT / "images/judge/public_cases.json").read_bytes()
    work = (TASK_ROOT / "images/work/benchmark/public_cases.json").read_bytes()
    assert judge == work
    table = runtime.load_case_table(TASK_ROOT / "images/judge/public_cases.json")
    assert table["suite_id"] == runtime.PUBLIC_SUITE_ID
    assert {case["family"] for case in table["cases"]} == set(runtime.FAMILIES)
    for case in json.loads(judge)["cases"]:
        verifier.validate_scored_case(case)


def test_work_runtime_is_derived_from_the_trusted_runtime(tmp_path) -> None:
    import subprocess
    import sys

    out = tmp_path / "kv_transfer_runtime.py"
    subprocess.run(
        [sys.executable, str(TASK_ROOT / "scripts/make_public_runtime.py"),
         str(TASK_ROOT / "runtime/kv_transfer_runtime.py"), str(out)],
        check=True,
    )
    assert out.read_bytes() == (TASK_ROOT / "images/work/benchmark/kv_transfer_runtime.py").read_bytes()


def test_admission_cuda_allowlist_matches_binary_allowlist() -> None:
    # Every callable CUDA API the source scan admits must be loadable by the
    # binary audit, and vice versa (types/enums excluded).
    callable_tokens = {
        token
        for token in admission.ALLOWED_CUDA_TOKENS
        if token in runtime.ALLOWED_CUDA_SYMBOLS
    }
    assert callable_tokens == set(runtime.ALLOWED_CUDA_SYMBOLS)


def test_adapter_group_keys_and_weights() -> None:
    assert set(adapter.GROUP_KEYS) == {"device_gbps", "host_gbps", "trace_gbps", "overlap_gbps"}
    assert adapter.FAMILY_KEYS == verifier.FAMILY_KEYS
    assert {group: tuple(keys) for group, keys in adapter.GROUP_FAMILY_KEYS.items()} == {
        f"{group}_gbps": tuple(f"family_{name.replace('-', '_')}_gbps" for name in families)
        for group, families in runtime.GROUP_FAMILIES.items()
    }
    assert adapter.GROUP_WEIGHTS == {
        f"{name}_gbps": weight for name, weight in runtime.GROUP_WEIGHTS.items()
    }
    assert 1.0 not in adapter.PROFILE_IDENTITIES
    assert adapter.PROFILE_IDENTITIES[2.0] == (
        "final", runtime.FINAL_SUITE_ID, float(verifier.EXPECTED_FINAL_COUNT)
    )
    assert adapter.EXPECTED_EDGE_CASE_COUNT == float(len(verifier.EDGE_CASE_IDS))


def test_judge_adapter_copies_are_identical() -> None:
    a = (TASK_ROOT / "judge/run_judge.py").read_bytes()
    b = (TASK_ROOT / "images/judge/run_judge.py").read_bytes()
    assert a == b


def test_work_and_judge_admission_copies_are_identical() -> None:
    a = (TASK_ROOT / "images/judge/source_admission.py").read_bytes()
    b = (TASK_ROOT / "images/work/benchmark/source_admission.py").read_bytes()
    assert a == b


def test_seed_copies_are_identical() -> None:
    a = (TASK_ROOT / "oracle/candidates/kernel_v001/submission.cu").read_bytes()
    b = (
        TASK_ROOT
        / "images/work/workdir/kv_transfer_lab/candidates/kernel_v001/submission.cu"
    ).read_bytes()
    assert a == b


def test_abi_header_copies_share_the_same_code() -> None:
    def code(path: str) -> list[str]:
        text = (TASK_ROOT / path).read_text()
        lines = [line.split("//")[0].strip() for line in text.splitlines()]
        return [line for line in lines if line]

    assert code("runtime/transfer_abi.h") == code("images/work/benchmark/transfer_abi.h")
    assert (TASK_ROOT / "images/work/benchmark/transfer_abi.h").read_bytes() == (
        TASK_ROOT / "images/work/workdir/kv_transfer_lab/transfer_abi.h"
    ).read_bytes()


def test_protected_hashes_are_current() -> None:
    lines = (TASK_ROOT / "images/judge/protected_hashes.txt").read_text().splitlines()
    records = {}
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        digest, label = line.split(None, 1)
        records[label] = digest
    import hashlib

    def sha(relative: str) -> str:
        return hashlib.sha256((TASK_ROOT / relative).read_bytes()).hexdigest()

    assert records == {
        "judge:candidate_worker.py": sha("images/judge/candidate_worker.py"),
        "judge:public_cases.json": sha("images/judge/public_cases.json"),
        "judge:run_judge.py": sha("images/judge/run_judge.py"),
        "judge:source_admission.py": sha("images/judge/source_admission.py"),
        "judge:verify_kv_transfer.py": sha("images/judge/verify_kv_transfer.py"),
        "worker:kv_transfer_runtime.py": sha("runtime/kv_transfer_runtime.py"),
        "worker:transfer_abi.h": sha("runtime/transfer_abi.h"),
    }
