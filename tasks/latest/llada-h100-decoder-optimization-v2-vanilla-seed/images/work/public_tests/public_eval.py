#!/usr/bin/env python3
"""Candidate-visible ABI and real-model development tests."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

LAB_ROOT = Path("/app/workdir/decoder_lab")
MODEL_PATH = Path(os.environ.get("LLADA_MODEL_PATH", "/opt/models/llada-hf/snapshots/6059b30c9531767ce47e0ca6a8161a6bb0c67693"))
PUBLIC_CASES = Path("/workspace/public_tests/public_cases.json")
BENCHMARKS = ("gsm8k", "math", "humaneval", "mbpp")
PROFILE_COUNTS = {"quick": 4, "full": 40}
# The public suite runs shorter generation lengths than the Judge.  Each public
# case lists a generation length; the length that actually runs is looked up
# here (listed length -> length run).  The quick cases keep their length.  A
# length inside a case or row name is only a label: the "gen_length" of each
# reported row is the length that ran.
PUBLIC_GEN_LENGTH = {128: 128, 256: 192, 512: 384, 1024: 768}


def _runtime():
    import sys

    sys.path.insert(0, str(LAB_ROOT))
    from runtime.api import GenerationSpec, validate_result
    from runtime.loader import import_candidate, load_model_and_candidate
    from runtime.source_policy import admit
    from public_quality import score_completion

    return GenerationSpec, validate_result, import_candidate, load_model_and_candidate, score_completion, admit


class ToyModel:
    mlsysforge_allow_cpu = True

    def __init__(self, vocab_size: int = 8) -> None:
        self.vocab_size = vocab_size
        self.calls = 0

    def __call__(self, tokens: torch.Tensor, **_: Any) -> SimpleNamespace:
        self.calls += 1
        batch, length = tokens.shape
        logits = torch.full((batch, length, self.vocab_size), -6.0)
        positions = torch.arange(length)
        predictions = positions.remainder(self.vocab_size - 2) + 1
        logits[0, positions, predictions] = 6.0 + positions.float() / 100
        return SimpleNamespace(logits=logits)


def cpu_tests(version: int | None) -> dict[str, Any]:
    GenerationSpec, validate_result, import_candidate, _, _, _ = _runtime()
    selected, module = import_candidate(LAB_ROOT, version)
    model = ToyModel()
    engine = module.build(model, SimpleNamespace())
    prompt = torch.tensor([[1, 2, 3]], dtype=torch.long)
    spec = GenerationSpec(gen_length=8, block_length=4, temperature=0.0, mask_id=7)
    first = engine.generate(prompt.clone(), spec)
    validate_result(first, prompt, spec)
    calls_first = model.calls
    second = engine.generate(prompt.clone(), spec)
    validate_result(second, prompt, spec)
    if not torch.equal(first.token_ids, second.token_ids):
        raise AssertionError("repeat request is nondeterministic or cache state leaked")
    if calls_first <= 0 or model.calls <= calls_first:
        raise AssertionError("candidate did not execute the supplied model on each request")
    if not torch.equal(first.token_ids[:, :3], prompt):
        raise AssertionError("prompt changed")
    return {
        "mode": "cpu-only",
        "version": selected,
        "model_calls_first": calls_first,
        "status": "PASS",
    }


def decode_completion(
    tokenizer,
    token_ids: torch.Tensor,
    prompt_length: int,
    stop: list[str],
    *,
    humaneval: bool = False,
) -> tuple[str, int]:
    generated = token_ids[0, prompt_length:].tolist()
    eos = getattr(tokenizer, "eos_token_id", 126081)
    if humaneval:
        # The visible code task checks the complete fixed output buffer.
        return (
            tokenizer.decode(generated, skip_special_tokens=True),
            sum(token != eos for token in generated),
        )
    if eos in generated:
        generated = generated[: generated.index(eos)]
    text = tokenizer.decode(generated, skip_special_tokens=False)
    for marker in stop:
        if marker in text:
            text = text.split(marker, 1)[0]
    clean = tokenizer.decode(tokenizer(text, add_special_tokens=False)["input_ids"], skip_special_tokens=True)
    useful = len(tokenizer(clean, add_special_tokens=False)["input_ids"])
    return clean, useful


def select_public_cases(cases: list[dict[str, Any]], quick: bool) -> list[dict[str, Any]]:
    profile = "quick" if quick else "full"
    selected = [case for case in cases if case.get("public_profile") == profile]
    expected = PROFILE_COUNTS[profile]
    if len(selected) != expected:
        raise RuntimeError(
            f"public {profile} profile has {len(selected)} cases, expected {expected}"
        )
    counts = {
        benchmark: sum(case["benchmark"] == benchmark for case in selected)
        for benchmark in BENCHMARKS
    }
    expected_per_dataset = 1 if quick else 10
    if counts != {benchmark: expected_per_dataset for benchmark in BENCHMARKS}:
        raise RuntimeError(f"public {profile} dataset coverage drift: {counts}")
    return selected


def public_gen_length(case: dict[str, Any]) -> int:
    """Generation length the public suite runs for this case."""
    listed = case["gen_length"]
    if listed not in PUBLIC_GEN_LENGTH:
        raise RuntimeError(
            f"public case {case.get('case_id')} lists generation length {listed}, "
            "which has no public length"
        )
    return PUBLIC_GEN_LENGTH[listed]


def summarize_datasets(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for benchmark in BENCHMARKS:
        selected = [row for row in rows if row["benchmark"] == benchmark]
        evaluated = [row for row in selected if row["quality"].get("evaluated")]
        seconds = sum(float(row["seconds"]) for row in selected)
        useful_tokens = sum(int(row["useful_tokens"]) for row in selected)
        correct = sum(bool(row["quality"].get("correct")) for row in evaluated)
        parsed = sum(bool(row["quality"].get("parsed")) for row in evaluated)
        summary[benchmark] = {
            "cases": len(selected),
            "evaluated": len(evaluated),
            "correct": correct,
            "parsed": parsed,
            "accuracy": round(correct / len(evaluated), 6) if evaluated else None,
            "parse_rate": round(parsed / len(evaluated), 6) if evaluated else None,
            "seconds": round(seconds, 6),
            "useful_tokens": useful_tokens,
            "throughput_tok_s": round(useful_tokens / seconds, 4) if seconds else 0.0,
            "nfe": sum(int(row["nfe"]) for row in selected),
        }
    return summary


def real_tests(version: int | None, quick: bool) -> dict[str, Any]:
    GenerationSpec, validate_result, _, load_model_and_candidate, score_completion, _ = _runtime()
    if not torch.cuda.is_available():
        raise RuntimeError("real-model tests require CUDA")
    if not MODEL_PATH.is_dir():
        raise RuntimeError(f"read-only model mount is missing: {MODEL_PATH}")
    loaded = load_model_and_candidate(LAB_ROOT, MODEL_PATH, torch.device("cuda"), version)
    cases = json.loads(PUBLIC_CASES.read_text(encoding="utf-8"))
    cases = select_public_cases(cases, quick)

    original_forward = loaded.model.forward
    counter = {"calls": 0}

    def counted_forward(*args, **kwargs):
        counter["calls"] += 1
        return original_forward(*args, **kwargs)

    loaded.model.forward = counted_forward
    rows = []
    for case in cases:
        chat = [{"role": "user", "content": case["request_text"]}]
        rendered = loaded.tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
        prompt = loaded.tokenizer(rendered, return_tensors="pt", add_special_tokens=False)["input_ids"].to("cuda")
        gen_length = public_gen_length(case)
        spec = GenerationSpec(gen_length, case["block_length"])
        before = counter["calls"]
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = loaded.engine.generate(prompt, spec)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        validate_result(result, prompt, spec)
        completion, useful = decode_completion(
            loaded.tokenizer,
            result.token_ids,
            int(prompt.shape[1]),
            case["stop"],
            humaneval=case["benchmark"] == "humaneval",
        )
        quality = {"evaluated": True, **score_completion(case, completion)}
        nfe = counter["calls"] - before
        rows.append(
            {
                "case": case["case_id"],
                "benchmark": case["benchmark"],
                "row_id": case["row_id"],
                "gen_length": gen_length,
                "quality": quality,
                "seconds": round(elapsed, 6),
                "useful_tokens": useful,
                "throughput_tok_s": round(useful / elapsed, 4),
                "nfe": nfe,
                "slots_per_nfe": round(gen_length / max(1, nfe), 4),
                "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 4),
                "completion_preview": completion[:240],
            }
        )
    return {
        "mode": "quick" if quick else "full",
        "version": loaded.version,
        "status": "PASS",
        "case_count": len(rows),
        "dataset_summary": summarize_datasets(rows),
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--cpu-only", action="store_true")
    modes.add_argument("--quick", action="store_true")
    parser.add_argument("--candidate-version", type=int)
    args = parser.parse_args()
    try:
        *_, admit = _runtime()
        source_report = admit(LAB_ROOT, args.candidate_version)
        report = cpu_tests(args.candidate_version)
        if not args.cpu_only:
            report = real_tests(args.candidate_version, args.quick)
        report["source_policy"] = {
            "selected_version": source_report["final_version"],
            "retained_versions": len(source_report["versions"]),
            "status": "PASS",
        }
    except Exception as exc:
        report = {"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
