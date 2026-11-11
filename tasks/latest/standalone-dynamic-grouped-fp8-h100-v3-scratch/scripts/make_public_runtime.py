"""Derive the public Work runtime from the trusted runtime (no integrity sentinel,
no throughput bound, no Judge-side vocabulary)."""
import re, sys
src, dst = sys.argv[1], sys.argv[2]
s = open(src).read()
a = s.index('"""Trusted runtime'); b = s.index('"""', a + 3) + 3
s = s[:a] + '"""Public test runtime for the dynamic-grouped FP8 GEMM task."""' + s[b:]
s = s.replace("import builtins\n", "")
s = s.replace('GENERATOR_PROVENANCE = "moe-serving-synthetic-v4"\n', "")
def cut(text, start_marker, end_marker):
    i = text.index(start_marker); j = text.index(end_marker, i)
    return text[:i] + text[j:]
s = cut(s, "# Physical plausibility:", "\n\n# ------")
s = cut(s, "def plausibility_bound(", "# --------------------------------------------------------------------------\n# Harness integrity sentinel.")
s = cut(s, "# --------------------------------------------------------------------------\n# Harness integrity sentinel.", "# --------------------------------------------------------------------------\n# Candidate loading.")
s = s.replace("# --------------------------------------------------------------------------\n# Candidate loading.", '''class _NoChecks:
    """The public runner does not run the Judge's integrity checks."""

    checks = 0

    def check(self, where: str) -> None:
        return None


# --------------------------------------------------------------------------
# Candidate loading.''', 1)
s = s.replace("sentinel: HarnessSentinel,", "sentinel: Any,")
s = s.replace("    sentinel = HarnessSentinel()\n", "    sentinel = _NoChecks()\n")
s = s.replace("    bound = plausibility_bound(hardware.get(\"capability\")) if hardware else None\n", "    bound = None\n")
s = s.replace('            "generator_provenance": GENERATOR_PROVENANCE,\n', "")
s = s.replace('        "generator_provenance": GENERATOR_PROVENANCE,\n', "")
s = s.replace('            "plausibility_bound_tflops": bound,\n', "")
s = s.replace("DECOY_SEED_SALT", "PRECALL_SEED_SALT")
s = s.replace("""        # Decoy pre-call: same shapes and hint, different row counts, data and
        # tensors.  It absorbs compilation keyed by shape or hint, and it is
        # released before the measured tensors exist.
""", """        # Untimed pre-call: same shapes and hint, different row counts, data and
        # tensors, released before the measured tensors exist.
""")
s = re.sub(r"\bdecoy_seed\b", "precall_seed", s)
s = re.sub(r"\bdecoy\b", "precall", s)
s = s.replace('"decoy call"', '"pre-call"').replace('"precall call"', '"pre-call"')
s = s.replace('"decoy_precall": True,', '"untimed_precall": True,')
s = s.replace('"decoy_rows": "redrawn",', '"precall_rows": "redrawn",')
s = s.replace('"decoy_precalls": trials,', '"untimed_precalls": trials,')
s = s.replace("upstream metric", "source metric")
s = s.replace("recorded as provenance only", "recorded for information only")
for banned in ("decoy", "Decoy", "upstream", "Upstream", "DeepGEMM", "calc_diff", "provenance",
               "HarnessSentinel", "plausibility_bound", "559d79f"):
    assert banned not in s, banned
open(dst, "w").write(s)
