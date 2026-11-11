from __future__ import annotations

import hashlib
import re
import shlex

from conftest import TASK_ROOT, load_module

public = load_module("attn_qat_public_contract", "images/work/benchmark/public_runner.py")
admission = load_module("attn_qat_admission_contract", "images/judge/source_admission.py")
verifier = load_module("attn_qat_verifier_contract", "images/judge/verify_attn_qat.py")
work_runtime = load_module("attn_qat_work_runtime_contract", "images/work/benchmark/attn_qat_runtime.py")
judge_runtime = load_module("attn_qat_judge_runtime_contract", "runtime/attn_qat_runtime.py")
adapter = load_module("attn_qat_adapter_contract", "images/judge/run_judge.py")
validator = load_module("attn_qat_validator_contract", "scripts/validate_formal_artifacts.py")


def commands(text: str):
    blocks = re.findall(r"```(?:bash|sh)\s*\n(.*?)```", text, re.S)
    return [
        shlex.split(line)
        for block in blocks
        for line in block.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def test_seed_is_byte_identical_and_selected() -> None:
    work = TASK_ROOT / "images/work/workdir/attn_qat_lab"
    oracle = TASK_ROOT / "oracle"
    assert (work / "candidates/kernel_v001/submission.py").read_bytes() == (
        oracle / "candidates/kernel_v001/submission.py"
    ).read_bytes()
    assert (work / "FINAL_VERSION").read_text().strip() == "1"
    # kernel_v001 stays the seed control; the oracle may select a stronger later candidate.
    oracle_version = int((oracle / "FINAL_VERSION").read_text().strip())
    assert oracle_version >= 1
    assert (oracle / f"candidates/kernel_v{oracle_version:03d}/submission.py").is_file()
    result = admission.validate_submission(work, 1, workspace_mode=True)
    assert result["ok"] is True, result["errors"]
    assert result["selected_sha256"] == hashlib.sha256(
        (oracle / "candidates/kernel_v001/submission.py").read_bytes()
    ).hexdigest()


def test_admission_expects_the_two_entrypoint_abi() -> None:
    assert admission.FORWARD_ENTRYPOINT == "attn_qat_fwd"
    assert admission.BACKWARD_ENTRYPOINT == "attn_qat_bwd"
    assert admission.EXPECTED_PARAMETERS == {
        "attn_qat_fwd": ("q", "k", "v", "softmax_scale", "causal", "o"),
        "attn_qat_bwd": (
            "q",
            "k",
            "v",
            "softmax_scale",
            "causal",
            "o",
            "saved",
            "do",
            "dq",
            "dk",
            "dv",
        ),
    }
    assert admission.SCALAR_PARAMETERS == {"softmax_scale", "causal"}
    assert admission.CONTAINER_PARAMETERS == {"saved"}


def test_admission_rejects_the_named_launch_time_routes() -> None:
    assert "triton.autotune" not in admission.ALLOWED_KERNEL_DECORATORS
    assert set(admission.REJECTED_KERNEL_DECORATORS) == {
        "triton.autotune",
        "triton.heuristics",
    }
    assert set(admission.REJECTED_CALLS) == {"triton.language.inline_asm_elementwise"}
    assert set(admission.REJECTED_CALL_PREFIXES) == {"triton.language.target_info"}
    assert set(admission.FORBIDDEN_ATTRIBUTE_PREFIXES) == {"torch.autograd"}
    assert admission.ALLOCATOR_SETTER == "triton.set_allocator"
    assert admission.ALLOCATOR_PARAMETER_COUNT == 3


def test_family_keys_are_identical_across_every_trusted_file() -> None:
    # The rCM run found a stale family key set in the formal-artifact validator only after
    # its first model run; the three lists are pinned against each other here instead.
    assert tuple(key.removesuffix("_tflops") for key in validator.FAMILY_KEYS) == verifier.FAMILIES
    assert tuple(key.removesuffix("_tflops") for key in validator.VIDEO_FAMILY_KEYS) == (
        verifier.VIDEO_FAMILIES
    )
    assert tuple(key.removesuffix("_tflops") for key in adapter.FAMILY_KEYS) == verifier.DOMAINS
    assert public.FAMILY_NAMES == work_runtime.FAMILIES == (
        "d128", "d64", "cross", "real", "llm", "llm_d256", "real_llm"
    )
    assert public.DOMAIN_NAMES == work_runtime.DOMAINS == verifier.DOMAINS


def test_public_partition_and_commands() -> None:
    assert len(public.cases_for_suite("one")) == 1
    assert [x["case_id"] for x in public.cases_for_suite("quick")] == list(public.QUICK_CASE_IDS)
    assert len(public.cases_for_suite("full")) == 29
    assert public.GEOMETRY == work_runtime.PUBLIC_GRID
    assert {cell: public.SEEDS[cell] for cell in ("P1", "P2", "P3", "P4")} == {
        "P1": 1103, "P2": 2207, "P3": 3301, "P4": 4409,
    }
    text = (TASK_ROOT / "images/work/AGENT_TASK.md").read_text()
    parsed = commands(text)
    assert ["./tools/run_one.sh", "--version", "1"] in parsed
    assert ["./tools/run_public_tests.sh", "--quick", "--candidate-version", "1"] in parsed
    assert ["./tools/run_public_tests.sh", "--candidate-version", "1"] in parsed


def test_agent_contract_discloses_the_exact_contract() -> None:
    text = (TASK_ROOT / "images/work/AGENT_TASK.md").read_text()
    for fragment in (
        # gates
        "`<= 1.0e-2`",
        "`<= 8.0e-2`",
        "`<= 3.0e-2`",
        "`<= 1.2e-1`",
        "public edge cases have fixed inputs, so each one carries its own limits",
        "`max |candidate| <= 1e-4`",
        "quantizes the *normalized* `P`",
        # score
        "work      = 2 * work_fwd + work_bwd = 26 * base",
        "time      = 2 * t_fwd + t_bwd",
        # the two facts that make the contract what it is
        "the **FINAL** row maximum".replace("**", ""),
        "6 * 2^-10 = 5.86e-3",
        "round-to-nearest-EVEN",
        "round-trips through a NaN",
        # saved cap and immutability
        "bytes(q) + bytes(k) + bytes(v) + bytes(o) + 4 * B * H * S_q",
        "`o` is byte-identical between the end of the forward and the end of the backward",
        # admission
        "`@triton.autotune` and `@triton.heuristics` are **not** permitted",
        "`tl.inline_asm_elementwise` is not permitted",
        "`triton.language.target_info` is not permitted",
        "return torch.empty(<first parameter>, dtype=torch.int8, device=...)",
        "from triton.tools.tensor_descriptor import TensorDescriptor",
    ):
        assert fragment in text, fragment


def test_agent_contract_does_not_leak_lab_material() -> None:
    text = (TASK_ROOT / "images/work/AGENT_TASK.md").read_text().lower()
    for leak in (
        "positive control",
        "headroom ladder",
        "planted",
        "upstream tflops",
        "h08b",
        "h09exactp",
        "39.18 ms",
    ):
        assert leak not in text, leak


def test_work_image_has_fixture_and_absence_asserts() -> None:
    text = (TASK_ROOT / "images/work/Dockerfile").read_text().lower()
    for name in (
        "fastvideo",
        "fastvideo_kernel",
        "flash_attn",
        "flash_attn_interface",
        "sageattention",
        "flashinfer",
        "xformers",
    ):
        assert name in text, name
    assert "fixtures" not in text and "task_runtime" not in text and "recipe" not in text


def test_judge_image_has_absence_asserts_and_private_manifest() -> None:
    text = (TASK_ROOT / "images/judge/Dockerfile").read_text().lower()
    for name in ("fastvideo", "flash_attn", "sageattention", "flashinfer", "xformers"):
        assert name in text, name
    # The holdout reaches only the host-built layer FROM the public Judge.
    assert "final_cases" not in text
    holdout_layer = (TASK_ROOT / "images/judge/Dockerfile.private").read_text().lower()
    assert "from ${judge_public_image}" in holdout_layer and "--from=final_cases" in holdout_layer
    assert "torch.__version__" in text and "3.4.0" in text
    assert "attn_qat_verify_trials=3" in text
    assert "attn_qat_fixture_dir=/opt/attn_qat_judge/fixtures" in text


def test_admission_copies_are_byte_identical() -> None:
    assert (TASK_ROOT / "images/judge/source_admission.py").read_bytes() == (
        TASK_ROOT / "images/work/benchmark/source_admission.py"
    ).read_bytes()


def test_protected_hashes_match() -> None:
    mapping = {
        "judge:candidate_worker.py": "images/judge/candidate_worker.py",
        "judge:run_judge.py": "images/judge/run_judge.py",
        "judge:source_admission.py": "images/judge/source_admission.py",
        "judge:verify_attn_qat.py": "images/judge/verify_attn_qat.py",
        "worker:attn_qat_runtime.py": "runtime/attn_qat_runtime.py",
        "worker:RECIPE.json": "fixtures/recipe/RECIPE.json",
        "worker:REAL_LAYERS.json": "fixtures/recipe/real_layers/REAL_LAYERS.json",
        "worker:REAL_LAYERS_V2.json": "fixtures/recipe/real_layers/REAL_LAYERS_V2.json",
    }
    observed = {
        line.split()[1]: line.split()[0]
        for line in (TASK_ROOT / "images/judge/protected_hashes.txt").read_text().splitlines()
        if line.strip()
    }
    assert set(observed) == set(mapping)
    for label, relative in mapping.items():
        assert observed[label] == hashlib.sha256((TASK_ROOT / relative).read_bytes()).hexdigest()


def test_verifier_and_runtime_edge_sets_agree() -> None:
    runtime = load_module("attn_qat_runtime_contract", "runtime/attn_qat_runtime.py")
    assert runtime.EDGE_CASE_IDS == verifier.EDGE_CASE_IDS
    assert runtime.EXPECTED_EDGE_CASE_COUNT == 15
    assert verifier.EXPECTED_CASE_COUNT == 287
    assert verifier.FAMILIES == runtime.FAMILIES
    assert verifier.PUBLIC_GEOMETRIES == runtime.PUBLIC_GEOMETRIES
    assert verifier.PUBLIC_LENGTHS == runtime.PUBLIC_LENGTHS
    assert verifier.FAMILY_LATTICES == runtime.FAMILY_LATTICES
    assert {k: set(v) for k, v in verifier.FAMILY_HEAD_DIMS.items()} == {
        k: set(v) for k, v in runtime.FAMILY_HEAD_DIMS.items()
    }
    assert verifier.LLM_LENGTH_QUANTUM == runtime.LLM_LENGTH_QUANTUM
    assert verifier.REAL_LAYER_PATTERN.pattern == runtime.REAL_LAYER_PATTERN.pattern
    assert verifier.KANDINSKY5_TEXT_TOKENS == runtime.KANDINSKY5_TEXT_TOKENS
    assert list(runtime.ENTRYPOINTS) == list(verifier.ENTRYPOINTS)


def test_public_cases_share_nothing_with_the_scored_or_edge_cases() -> None:
    generator = load_module("attn_qat_contract_generator", "scripts/generate_final_manifest.py")
    rows = generator.grid_cases()
    grid = list(work_runtime.PUBLIC_GRID.values())
    geometry = {(h, hk, d) for _f, _b, h, hk, d, _q, _k, _c in grid}
    assert not geometry & {(r["h"], r["h_kv"], r["d"]) for r in rows}
    assert not {q for *_x, q, _k, _c in grid} & {r["s_q"] for r in rows}
    assert geometry == judge_runtime.PUBLIC_GEOMETRIES
    assert {q for *_x, q, _k, _c in grid} == judge_runtime.PUBLIC_LENGTHS
    def identity(case):
        return tuple(case.get(key, case["h"] if key == "h_kv" else False)
                     for key in ("b", "s_q", "s_kv", "h", "h_kv", "d", "causal"))
    assert not {identity(c) for c in work_runtime.EDGE_CASES} & {identity(c) for c in judge_runtime.EDGE_CASES}
    assert not {c["seed"] for c in work_runtime.EDGE_CASES} & {c["seed"] for c in judge_runtime.EDGE_CASES}
    assert set(work_runtime.EDGE_GATE_LIMITS) == set(work_runtime.EDGE_CASE_IDS)
    pairs = judge_runtime.attention_pairs
    public_work = sum(b * h * pairs(q, k, c) * d for _f, b, h, _hk, d, q, k, c in grid)
    hidden_work = sum(r["b"] * r["h"] * pairs(r["s_q"], r["s_kv"], r["causal"]) * r["d"] for r in rows)
    assert hidden_work > 5 * public_work


def test_agent_visible_files_carry_no_hidden_cases_upstream_or_internal_notes() -> None:
    lab = TASK_ROOT / "images/work/workdir/attn_qat_lab"
    paths = [
        TASK_ROOT / "images/work/AGENT_TASK.md",
        TASK_ROOT / "images/work/benchmark/attn_qat_runtime.py",
        TASK_ROOT / "images/work/benchmark/public_runner.py",
        TASK_ROOT / "images/work/benchmark/run_public_tests.sh",
        lab / "README_AGENT.md",
        lab / "notes.md",
        lab / "tools/run_one.sh",
        lab / "tools/run_public_tests.sh",
        lab / "candidates/kernel_v001/submission.py",
    ]
    query = __import__("json").loads(next((TASK_ROOT / "tasks").glob("*.json")).read_text())["work"]["agent_query"]
    tokens = ("FastVideo", "fastvideo", "upstream", "Upstream", "hao-ai", "7f83164", "MLSys", "Forge",
              "DESIGN", "RUN_RESULTS", "holdout", "anchor", "wan13b", "ltx2", "kandinsky", "Kandinsky",
              "decoy", "Decoy", "FlashAttention", "e7_wan", "RECIPE", "Derived from", "Apache",
              "SM90", "SM100", "PyroMind", "HAMi", "single-accelerator", "prepare_submit",
              "24960", "32760", "5070", "33792", "47616", "31200", "75600", "wan14b", "REAL_LAYERS",
              "cogvideox", "CogVideoX", "hunyuan", "Hunyuan", "Qwen3-14B", "qwen3-14b", "Qwen3.5",
              "qwen35", "Llama", "LongBench", "17776", "67576", "65536")
    for name, text in [(p.name, p.read_text(encoding="utf-8")) for p in paths] + [("agent_query", query)]:
        for token in tokens:
            assert token not in text, (name, token)
