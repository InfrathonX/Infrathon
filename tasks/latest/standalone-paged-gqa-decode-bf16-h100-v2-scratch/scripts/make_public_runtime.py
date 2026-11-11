#!/usr/bin/env python3
"""Derive the public Work runtime from the trusted runtime.

The public copy drops the integrity sentinel, the throughput bound and the
Judge-side vocabulary; everything that generates inputs, calls the candidate,
times it and checks it is the same code.

    scripts/make_public_runtime.py            # rewrite images/work/benchmark/paged_gqa_decode_runtime.py
    scripts/make_public_runtime.py --check    # fail if the shipped copy is stale
"""
import re
import sys
from pathlib import Path

TASK_ROOT = Path(__file__).resolve().parents[1]
SOURCE = TASK_ROOT / "runtime/paged_gqa_decode_runtime.py"
TARGET = TASK_ROOT / "images/work/benchmark/paged_gqa_decode_runtime.py"
BANNED = ("decoy", "Decoy", "upstream", "Upstream", "FlashAttention", "FlashInfer", "PagedAttention",
          "vLLM", "SGLang", "hopper", "0251105a", "test_flash_attn", "benchmark_split_kv",
          "HarnessSentinel", "plausibility_bound", "final-holdout", "edge-holdout",
          "Llama", "Mistral", "Qwen")


def cut(text: str, start: str, end: str) -> str:
    i = text.index(start)
    j = text.index(end, i)
    return text[:i] + text[j:]


def derive(s: str) -> str:
    a = s.index('"""Trusted runtime')
    b = s.index('"""', a + 3) + 3
    s = s[:a] + '''"""Public test runtime for the paged-GQA decode task.

Generates the public cases, calls the candidate, times it and checks every
output against an FP32 reference over the exact BF16 inputs.  The public
controller provides the case manifests.
"""''' + s[b:]
    s = s.replace("import builtins\n", "")
    s = re.sub(r'GENERATOR_PROVENANCE = "[^"]*"', 'GENERATOR_PROVENANCE = "paged-gqa-decode-v4-public"', s)
    s = cut(s, "# Physical plausibility:", "\n\n# ------")
    s = cut(s, "def plausibility_bound(", "def use_exact_fp32()")
    s = cut(s, "# --------------------------------------------------------------------------\n# Harness integrity sentinel.",
            "def _sha256(")
    s = s.replace("def _sha256(", '''class _NoChecks:
    """The public runner does not run the Judge's integrity checks."""

    checks = 0

    def check(self, where: str) -> None:
        return None


def _sha256(''', 1)
    s = s.replace("sentinel: HarnessSentinel,", "sentinel: Any,")
    s = s.replace("    sentinel = HarnessSentinel()\n", "    sentinel = _NoChecks()\n")
    s = s.replace("    bound = plausibility_bound(hardware.get(\"capability\")) if hardware else None\n",
                  "    bound = None\n")
    s = s.replace('            "plausibility_bound_tflops": bound,\n', "")
    s = s.replace("    except HarnessIntegrityFailure as exc:\n        return fail(exc.gate, str(exc), exc.detail)\n",
                  "    except GateFailure as exc:\n        return fail(exc.gate, str(exc), exc.detail)\n")
    # The public runner reports no harness-integrity gate.
    s = s.replace('GATE_ORDER = (\n    "edge_correctness_ok",\n    "memory_safety_ok",\n    "harness_integrity_ok",\n    "correctness_ok",\n)',
                  'GATE_ORDER = (\n    "edge_correctness_ok",\n    "memory_safety_ok",\n    "correctness_ok",\n)')
    s = s.replace('    gates["harness_integrity_ok"] = True\n', "")
    # Vocabulary.
    s = s.replace("# outlier-feature distribution of the FlashAttention-3 numerics study.",
                  "# distribution of outlier features in large models.")
    s = s.replace("(upstream test_flash_attn_kvcache rule: max <= 2x + floor,", "(max <= 2x + floor,")
    s = s.replace("""        # FlashInfer's extreme-negative-logit regression: every visible logit is
        # far below any finite masking sentinel (here about -4.6e4).""",
                  """        # Extreme negative logits: every visible logit is far below any finite
        # masking value (here about -4.6e4).""")
    s = s.replace("(PagedAttention section 4.4: one physical block, several\n        # logical blocks)",
                  "(one physical block, several logical blocks)")
    s = re.sub(r"\(upstream ``attention_ref\(upcast=False,\s*reorder_ops=True\)`` role\)", "", s)
    s = s.replace("upstream", "source").replace("Upstream", "Source")
    s = re.sub(r"\bdecoy_lengths\b", "precall_lengths", s)
    s = s.replace('"decoy-lengths"', '"precall-lengths"').replace('"decoy call"', '"pre-call"')
    s = s.replace('"decoy"', '"precall"')
    s = s.replace('"decoy_precalls": trials,', '"untimed_precalls": trials,')
    s = s.replace('"decoy_precall": True,', '"untimed_precall": True,')
    s = re.sub(r"\bdecoy\b", "precall", s)
    s = s.replace("Decoy pre-call", "Untimed pre-call").replace("a decoy and a measured call",
                                                                 "an untimed and a measured call")
    s = s.replace("decoy", "precall").replace("Decoy", "Precall")
    return s


def main() -> int:
    text = derive(SOURCE.read_text(encoding="utf-8"))
    leaks = [word for word in BANNED if word in text]
    if leaks:
        raise SystemExit(f"public runtime still contains: {leaks}")
    compile(text, str(TARGET), "exec")
    if "--check" in sys.argv[1:]:
        if TARGET.read_text(encoding="utf-8") != text:
            print(f"{TARGET.relative_to(TASK_ROOT)} is stale; run scripts/make_public_runtime.py")
            return 1
        return 0
    TARGET.write_text(text, encoding="utf-8")
    print(f"wrote {TARGET.relative_to(TASK_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
