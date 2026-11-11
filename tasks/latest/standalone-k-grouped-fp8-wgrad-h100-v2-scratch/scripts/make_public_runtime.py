"""Derive the public Work runtime from the trusted runtime (no integrity sentinel,
no throughput bound, no Judge-side vocabulary).

    scripts/make_public_runtime.py runtime/k_grouped_wgrad_runtime.py \
        images/work/benchmark/k_grouped_wgrad_runtime.py
"""
import re
import sys

src, dst = sys.argv[1], sys.argv[2]
s = open(src).read()
a = s.index('"""Trusted runtime')
b = s.index('"""', a + 3) + 3
s = s[:a] + '"""Public test runtime for the standalone K-grouped FP8 wgrad GEMM task, v3."""' + s[b:]
s = s.replace("import builtins\n", "")
s = s.replace('GENERATOR_PROVENANCE = "moe-training-synthetic-v3"\n', "")


def cut(text, start_marker, end_marker):
    i = text.index(start_marker)
    j = text.index(end_marker, i)
    return text[:i] + text[j:]


s = cut(s, "# Physical plausibility:", "\n\n# ------")
s = cut(s, "# Correctness.  E is the exact FP64 product", "ELEMENT_RATIO_LIMIT = ")
s = s.replace("ELEMENT_RATIO_LIMIT = ", """# Correctness.  E is the exact FP64 product of the dequantized FP8 operands
# plus C; R[g, m, n] = sqrt(sum_k (a s_a)^2 (b s_b)^2) is the element-wise
# product scale.  The source metric compares against the unquantized BF16
# source GEMM and is only applied to slices with enough elements.
ELEMENT_RATIO_LIMIT = """, 1)
s = cut(s, "def plausibility_bound(", "# --------------------------------------------------------------------------\n# Harness integrity sentinel.")
s = cut(s, "# --------------------------------------------------------------------------\n# Harness integrity sentinel.",
        "def _sha256(path: Path) -> str:")
s = s.replace("def _sha256(path: Path) -> str:", '''# --------------------------------------------------------------------------
# Candidate loading.
# --------------------------------------------------------------------------
class _NoChecks:
    """The public runner does not run the Judge's integrity checks."""

    checks = 0

    def check(self, where: str) -> None:
        return None


def _sha256(path: Path) -> str:''', 1)
s = s.replace("sentinel: HarnessSentinel,", "sentinel: Any,")
s = s.replace("    sentinel = HarnessSentinel()\n", "    sentinel = _NoChecks()\n")
s = s.replace("    bound = plausibility_bound(hardware.get(\"capability\")) if hardware else None\n", "    bound = None\n")
s = s.replace('        "generator_provenance": GENERATOR_PROVENANCE,\n', "")
s = s.replace('            "plausibility_bound_tflops": bound,\n', "")
s = s.replace("""        # Decoy pre-call: same shapes and hint, different group sizes, data and
        # tensors.  It absorbs compilation keyed by shape or hint, and it is
        # released before the measured tensors exist.
""", """        # Untimed pre-call: same shapes and hint, different group sizes, data and
        # tensors, released before the measured tensors exist.
""")
s = re.sub(r"\bdecoy\b", "precall", s)
s = s.replace('"decoy-ks"', '"precall-ks"')
s = s.replace('"precall call"', '"pre-call"')
s = s.replace('"decoy_precall": True,', '"untimed_precall": True,')
s = s.replace('"decoy_ks": "redrawn",', '"precall_ks": "redrawn",')
s = s.replace('"decoy_precalls": trials,', '"untimed_precalls": trials,')
s = s.replace("upstream metric", "source metric").replace("The upstream metric", "The source metric")
s = s.replace("recorded as provenance only", "recorded for information only")
for banned in ("decoy", "Decoy", "upstream", "Upstream", "DeepGEMM", "calc_diff", "provenance",
               "HarnessSentinel", "plausibility_bound", "559d79f"):
    assert banned not in s, banned
open(dst, "w").write(s)
