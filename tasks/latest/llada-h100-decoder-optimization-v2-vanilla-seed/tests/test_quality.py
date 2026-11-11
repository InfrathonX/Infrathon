from __future__ import annotations

from conftest import load_module


def quality(task_root):
    return load_module(
        "quality_under_test",
        task_root / "images/judge/verifier/quality.py",
    )


def test_gsm_parser_uses_last_or_strict_answer(task_root):
    module = quality(task_root)
    assert module.extract_gsm_answer("work 12\n#### 1,234") == "1234"
    assert module.extract_gsm_answer("first 12 then -7") == "-7"
    assert module.extract_gsm_answer("none") is None


def test_math_normalization(task_root):
    module = quality(task_root)
    assert module.normalize_math("1/2") == module.normalize_math("\\frac{1}{2}")
    assert module.normalize_math("\\left( 3 \\right)") == "(3)"
    assert module.extract_math_answer(r"work \boxed{\frac{3}{16}}") == r"\frac{3}{16}"
    assert module.extract_math_answer(r"The answer is $\frac{3}{16}$.") == r"\frac{3}{16}"


def test_generated_code_policy_rejects_io(task_root):
    module = quality(task_root)
    assert module.code_is_safe("def f(x): return x + 1")[0]
    assert not module.code_is_safe("import os\ndef f(): return os.listdir('/')")[0]
    assert not module.code_is_safe("def f(): return open('/tmp/x').read()")[0]
    assert not module.code_is_safe(
        "(lambda: 0).__globals__['__builtins__']['__import__']('os')._exit(0)"
    )[0]
    assert not module.code_is_safe("raise SystemExit(0)")[0]
    assert not module.code_is_safe("quit()")[0]
    assert not module.code_is_safe(
        "import operator\ng = operator.attrgetter\ng('_' + '_globals' + '_' + '_')(lambda: 0)"
    )[0]
    assert not module.code_is_safe(
        "import string\nf = string.Formatter()\nf.get_field('0.' + '_' + '_globals' + '_' + '_', (lambda: 0,), {})"
    )[0]


def test_code_runner_requires_trusted_terminal_signal(task_root):
    module = quality(task_root)
    assert module.run_restricted_python("assert 2 + 2 == 4") == (True, "passed")
    passed, reason = module.run_restricted_python("assert False")
    assert passed is False
    assert reason == "exit:1"


def test_humaneval_code_postprocessing(task_root):
    module = quality(task_root)
    case = {
        "benchmark": "humaneval",
        "entry_point": "increment",
        "prompt": 'def increment(value):\n    """Return value plus one."""\n',
        "tests": "assert increment(4) == 5\n",
    }
    completion = (
        "Here is the implementation:\n```python\n"
        "def increment(value):\n    return value + 1\n```\nTrailing prose"
    )
    result = module.score_completion(case, completion)
    assert result == {"correct": True, "parsed": True, "code_reason": "passed"}


def test_producer_relative_quality_measurement_is_exact_and_tamper_evident(task_root):
    module = load_module(
        "quality_threshold_under_test",
        task_root / "images/judge/verifier/quality_threshold.py",
    )
    producer = {
        "row_a": {"correct": 80, "parsed": 100, "useful_tokens": 1_000},
        "row_b": {"correct": 40, "parsed": 50, "useful_tokens": 500},
    }
    candidate = {
        "row_a": {"correct": 72, "parsed": 95, "useful_tokens": 900},
        "row_b": {"correct": 38, "parsed": 45, "useful_tokens": 475},
    }
    measured = module.measure_quality_retention(candidate, producer)
    assert measured["metric"] == "producer_relative_min_retention_ppm_v1"
    assert measured["score_ppm"] == 900_000
    module.validate_measurement(measured)
    forged = dict(measured)
    forged["score_ppm"] += 1
    with __import__("pytest").raises(ValueError, match="commitment"):
        module.validate_measurement(forged)


def test_useful_token_inflation_above_ceiling_is_reported(task_root):
    module = load_module(
        "quality_threshold_ceiling_under_test",
        task_root / "images/judge/verifier/quality_threshold.py",
    )
    producer = {
        "row_a": {"correct": 10, "parsed": 10, "useful_tokens": 1_000},
        "row_b": {"correct": 10, "parsed": 10, "useful_tokens": 1_000},
    }
    honest = {
        "row_a": {"correct": 10, "parsed": 10, "useful_tokens": 1_150},
        "row_b": {"correct": 10, "parsed": 10, "useful_tokens": 1_000},
    }
    assert module.useful_tokens_above_ceiling(
        module.measure_quality_retention(honest, producer)
    ) == []
    # Rewriting EOS slots on one row (e.g. the fixed-buffer HumanEval count)
    # keeps every quality count but multiplies that row's numerator.
    inflated = {
        "row_a": {"correct": 10, "parsed": 10, "useful_tokens": 4_000},
        "row_b": {"correct": 10, "parsed": 10, "useful_tokens": 1_000},
    }
    measured = module.measure_quality_retention(inflated, producer)
    assert measured["score_ppm"] == 1_000_000
    assert module.useful_tokens_above_ceiling(measured) == [
        "aggregate:useful_tokens",
        "row:row_a:useful_tokens",
    ]


def _twelve_rows(correct_by_benchmark):
    rows = {}
    for row_id in (
        "table1_gsm8k_5shot_g256",
        "table1_gsm8k_5shot_g512",
        "table4_gsm8k_5shot_g1024",
        "table5_gsm8k_8shot_g256",
        "table5_gsm8k_8shot_g512",
        "table5_gsm8k_8shot_g1024",
        "table1_math_4shot_g256",
        "table1_math_4shot_g512",
        "table1_humaneval_0shot_g256",
        "table1_humaneval_0shot_g512",
        "table1_mbpp_3shot_g256",
        "table1_mbpp_3shot_g512",
    ):
        benchmark = row_id.split("_")[1]
        rows[row_id] = {
            "correct": correct_by_benchmark[benchmark],
            "parsed": 20,
            "useful_tokens": 1_000,
        }
    return rows


def test_fixed_floors_apply_to_aggregate_and_benchmark(task_root):
    module = load_module(
        "quality_threshold_floors_under_test",
        task_root / "images/judge/verifier/quality_threshold.py",
    )
    producer = _twelve_rows({"gsm8k": 10, "math": 10, "humaneval": 10, "mbpp": 10})

    def failures(candidate):
        measured = module.measure_quality_retention(candidate, producer)
        return module.fixed_gate_failures(
            measured, module.benchmark_retention(candidate, producer)
        )

    # One answer lost in every row: 90% everywhere, passes both floors.
    assert failures(_twelve_rows({"gsm8k": 9, "math": 9, "humaneval": 9, "mbpp": 9})) == []
    # MATH alone loses 30%: aggregate stays at 92.5% but the benchmark floor fails.
    assert failures(
        _twelve_rows({"gsm8k": 10, "math": 7, "humaneval": 10, "mbpp": 10})
    ) == ["producer_relative_benchmark_below_floor"]
    # Every benchmark loses 20%: each benchmark passes, the aggregate floor fails.
    assert failures(
        _twelve_rows({"gsm8k": 8, "math": 8, "humaneval": 8, "mbpp": 8})
    ) == ["producer_relative_aggregate_below_floor"]
    with __import__("pytest").raises(ValueError, match="benchmark"):
        module.benchmark_of_row("table9_unknown_g256")


def test_quality_diagnostic_applies_the_fixed_gate(task_root):
    controller = (
        task_root / "images/judge/verifier/verify_decoder.py"
    ).read_text(encoding="utf-8")
    diagnostic = controller.split("def quality_diagnostic(", 1)[1].split("\ndef ", 1)[0]
    assert "failures.extend(fixed_gate_failures(measurement, benchmark_ppm))" in diagnostic
    assert '"quality_gate": quality_gate,' in controller
    validator = (task_root / "scripts/validate_eval_artifacts.py").read_text(
        encoding="utf-8"
    )
    assert "private quality verdict ignores a fixed-gate failure" in validator
