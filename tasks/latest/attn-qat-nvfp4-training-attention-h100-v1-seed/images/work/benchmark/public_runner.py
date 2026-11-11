#!/usr/bin/env python3
"""Candidate-only public feedback for the Attn-QAT training attention.

Runs the public edge cases first, then one public case per cell.  It reports geometry,
the drawn regime, the four accuracy statistics, the limits that applied, the saved-bundle
size, the separated forward and backward medians and the TFLOPS.  Hidden cases are never
run here.
"""
from __future__ import annotations
import argparse, json, math, sys, traceback
from pathlib import Path
from typing import Any, Callable
from source_admission import validate_submission

ROOT = Path(__file__).resolve().parent


def _load() -> Callable[..., dict[str, Any]]:
    """Import the trusted runtime, then drop it from the module and path namespaces."""
    from attn_qat_runtime import run_suite

    sys.path = [x for x in sys.path if Path(x or ".").resolve() != ROOT]
    sys.modules.pop("attn_qat_runtime", None)
    return run_suite


# (family, b, h, h_kv, d, s_q, s_kv, causal).  P1 to P4 are for bring-up; the rest cover the
# regimes the hidden evaluation covers: long video sequences, D = 64, cross attention, causal
# grouped-query LLM attention at D = 128 and D = 256, and real captured layers.
GEOMETRY = {
    "P1": ("d128", 1, 4, 4, 128, 9984, 9984, False),
    "P2": ("d128", 1, 8, 8, 128, 6120, 6120, False),
    "P3": ("d128", 1, 24, 24, 128, 2728, 2728, False),
    "P4": ("d64", 1, 16, 16, 64, 9216, 9216, False),
    "P5": ("d128", 1, 4, 4, 128, 2784, 2784, False),
    "P6": ("d128", 1, 4, 4, 128, 26208, 26208, False),
    "P7": ("d128", 1, 4, 4, 128, 57440, 57440, False),
    "P8": ("d128", 1, 8, 8, 128, 12384, 12384, False),
    "P9": ("d128", 1, 8, 8, 128, 26208, 26208, False),
    "P10": ("d128", 1, 8, 8, 128, 57440, 57440, False),
    "P11": ("d128", 1, 24, 24, 128, 9696, 9696, False),
    "P12": ("d128", 1, 24, 24, 128, 18016, 18016, False),
    "P13": ("d128", 1, 24, 24, 128, 26208, 26208, False),
    "P14": ("d64", 1, 16, 16, 64, 20480, 20480, False),
    "P15": ("d64", 1, 16, 16, 64, 40960, 40960, False),
    "P16": ("d64", 1, 24, 24, 64, 20480, 20480, False),
    "P17": ("d64", 1, 24, 24, 64, 40960, 40960, False),
    "P18": ("cross", 1, 16, 16, 64, 9216, 320, False),
    "P19": ("cross", 1, 16, 16, 64, 20480, 320, False),
    "P20": ("cross", 1, 16, 16, 64, 40960, 320, False),
    "PR": ("real", 1, 24, 24, 128, 27280, 27280, False),
    "P21": ("llm", 4, 28, 4, 128, 3584, 3584, True),
    "P22": ("llm", 2, 28, 4, 128, 7168, 7168, True),
    "P23": ("llm", 1, 28, 4, 128, 14336, 14336, True),
    "P24": ("llm", 2, 14, 2, 128, 7168, 7168, True),
    "P25": ("llm", 1, 14, 2, 128, 28672, 28672, True),
    "P26": ("llm_d256", 4, 20, 4, 256, 3584, 3584, True),
    "P27": ("llm_d256", 1, 20, 4, 256, 14336, 14336, True),
    "PRL": ("real_llm", 1, 16, 8, 128, 7168, 7168, True),
}
SEEDS = {"P1": 1103, "P2": 2207, "P3": 3301, "P4": 4409}
SEEDS.update({cell: 5000 + 113 * index for index, cell in enumerate(GEOMETRY) if cell not in SEEDS})
PUBLIC_LAYERS = {"PR": "public-video-layer", "PRL": "public-llm-layer"}
REAL_FAMILIES = ("real", "real_llm")


def _case(cell: str, *, perf_only: bool = False) -> dict[str, Any]:
    """One public case in the runtime's schema."""
    return {
        "case_id": f"public-{cell.lower()}",
        "cell": cell,
        "seed": SEEDS[cell],
        "gated": not perf_only,
        "perf_only": perf_only,
        "layer": PUBLIC_LAYERS.get(cell),
    }


PUBLIC_CASES = tuple(_case(cell) for cell in GEOMETRY)
# The quick suite covers both video head dimensions (one with an unaligned length), one causal
# grouped-query case, and one long sequence that only reports its speed.
QUICK_CASE_IDS = ("public-p2", "public-p4", "public-p24", "public-p10")
QUICK_PERF_ONLY = "P10"
FAMILY_NAMES = ("d128", "d64", "cross", "real", "llm", "llm_d256", "real_llm")
DOMAIN_NAMES = ("video", "llm")
GATE_TENSORS = ("o", "dq", "dk", "dv")
METRICS = tuple(
    f"{name}_{statistic}"
    for name in GATE_TENSORS
    for statistic in ("rel_l2", "head_rel_l2", "max_abs", "rel_l2_limit")
)
IDENTITY = (
    "case_id",
    "family",
    "tier",
    "b",
    "s_q",
    "s_kv",
    "h",
    "h_kv",
    "d",
    "causal",
    "softmax_scale",
    "effective_flops",
    "forward_flops",
    "backward_flops",
)
SAVED = ("saved_bytes", "saved_bytes_limit", "saved_tensors", "gate_group")
EDGE_FIELDS = IDENTITY + SAVED + METRICS
CASE_FIELDS = (
    IDENTITY
    + SAVED[:-1]
    + (
        "seed",
        "gated",
        "regime",
        "median_us",
        "median_forward_us",
        "median_backward_us",
        "tflops",
        "fwd_tflops",
        "bwd_tflops",
    )
)


def cases_for_suite(name: str, *, only: str | None = None, perf_only: bool = False):
    """Public case list for `one`, `quick` or `full`, optionally one case or timing only."""
    if only is not None:
        cell = only.upper()
        if cell not in GEOMETRY:
            raise ValueError("unknown case")
        return [_case(cell, perf_only=perf_only and GEOMETRY[cell][0] not in REAL_FAMILIES)]
    if name == "one":
        return [_case("P1", perf_only=perf_only)]
    if name == "quick":
        return [_case("P2"), _case("P4"), _case("P24"), _case(QUICK_PERF_ONLY, perf_only=True)]
    if name == "full":
        return [
            _case(cell, perf_only=perf_only and GEOMETRY[cell][0] not in REAL_FAMILIES)
            for cell in GEOMETRY
        ]
    raise ValueError("unknown suite")


def _fields(row: dict[str, Any], fields):
    """Project a trusted result row onto the disclosed allowlist, failing closed."""
    missing = [x for x in fields if x not in row]
    if missing:
        raise RuntimeError(f"trusted result missing {missing}")
    return {x: row[x] for x in fields}


def _aggregate(value: Any, names=FAMILY_NAMES):
    """Validate and copy a family or domain aggregate."""
    if not isinstance(value, dict) or not value or not set(value) <= set(names):
        raise RuntimeError("bad aggregate")
    result = {str(k): float(v) for k, v in value.items()}
    if not all(math.isfinite(v) and v > 0 for v in result.values()):
        raise RuntimeError("bad aggregate value")
    return result


def public_payload(result: dict[str, Any], suite: str, version: int, count: int):
    """Build the disclosed payload from a trusted runtime result."""
    return {
        "schema_version": 1,
        "suite": suite,
        "candidate_version": version,
        "num_candidate_versions": count,
        "candidate_only": True,
        "hidden_cases_run": False,
        "source_admission_ok": True,
        "correctness_ok": bool(result["correctness_ok"]),
        "edge_ok": bool(result["edge_ok"]),
        "fresh_input_ok": bool(result["fresh_input_ok"]),
        "fresh_output_ok": bool(result["fresh_output_ok"]),
        "saved_cap_ok": bool(result["saved_cap_ok"]),
        "trials_per_case": int(result["trials_per_case"]),
        "warmups": int(result["warmups"]),
        "gate_calibration": str(result["gate_calibration"]),
        "edge_cases": [_fields(dict(x), EDGE_FIELDS) for x in result["edge_cases"]],
        "cases": [
            _fields(dict(x), CASE_FIELDS + (("gate_group",) + METRICS if x["gated"] else ()))
            for x in result["cases"]
        ],
        "family_tflops": _aggregate(result["family_tflops"]),
        "domain_tflops": _aggregate(result["domain_tflops"], DOMAIN_NAMES),
        "primary_tflops": float(result["primary_tflops"]),
    }


def _print(payload):
    """Human-readable summary, one line per case."""
    print(
        "SOURCE_ADMISSION          PASS\n"
        "EDGE_CASES                PASS\n"
        "CORRECTNESS               PASS\n"
        "SAVED_ACTIVATION_CAP      PASS\n"
        "FRESH_INPUTS_AND_OUTPUT   PASS\n"
        "CANDIDATE_ONLY            YES\n"
        "HIDDEN_CASES              NOT_RUN"
    )
    for row in payload["cases"]:
        print(
            f"{row['case_id']:<12} {row['tflops']:9.3f} TFLOPS  "
            f"fwd={row['median_forward_us'] / 1000:8.3f} ms ({row['fwd_tflops']:7.2f})  "
            f"bwd={row['median_backward_us'] / 1000:8.3f} ms ({row['bwd_tflops']:7.2f})"
        )
        if not row["gated"]:
            print(f"{'':<12} timing only: no accuracy check")
            continue
        print(
            f"{'':<12} o={row['o_rel_l2']:.5f}/{row['o_rel_l2_limit']:.5f}  "
            f"dq={row['dq_rel_l2']:.5f}/{row['dq_rel_l2_limit']:.5f}  "
            f"dk={row['dk_rel_l2']:.5f}/{row['dk_rel_l2_limit']:.5f}  "
            f"dv={row['dv_rel_l2']:.5f}/{row['dv_rel_l2_limit']:.5f}  "
            f"saved={row['saved_bytes']}/{row['saved_bytes_limit']}"
        )
    for family, value in sorted(payload["family_tflops"].items()):
        print(f"FAMILY_{family.upper()}_TFLOPS {value:.6f}")
    for domain, value in sorted(payload["domain_tflops"].items()):
        print(f"DOMAIN_{domain.upper()}_TFLOPS {value:.6f}")
    print(f"PUBLIC_PRIMARY_TFLOPS {payload['primary_tflops']:.6f}")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--lab-root", type=Path, default=Path("/app/workdir/attn_qat_lab"))
    p.add_argument("--suite", choices=("one", "quick", "full"), default="full")
    p.add_argument("--candidate-version", type=int, required=True)
    p.add_argument("--case", help="run one public case, for example P10 or PR")
    p.add_argument("--perf-only", action="store_true", help="report TFLOPS without any golden")
    p.add_argument("--json-output", type=Path)
    a = p.parse_args(argv)
    try:
        admission = validate_submission(a.lab_root, a.candidate_version, workspace_mode=True)
        if not admission["ok"]:
            raise RuntimeError(
                "public source admission failed: " + "; ".join(admission["errors"])
            )
        run = _load()
        globals().pop("_load", None)
        result = run(
            Path(admission["selected_path"]),
            cases_for_suite(a.suite, only=a.case, perf_only=a.perf_only),
            trials=3 if a.suite == "full" else 1,
            warmups=1,
        )
        payload = public_payload(
            result, a.suite, a.candidate_version, int(admission["num_candidate_versions"])
        )
        if a.json_output:
            a.json_output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        _print(payload)
        print(json.dumps(payload, sort_keys=True))
        return 0
    except Exception as exc:
        print(
            "CORRECTNESS               FAIL\n"
            "CANDIDATE_ONLY            YES\n"
            "HIDDEN_CASES              NOT_RUN",
            file=sys.stderr,
        )
        print(f"PUBLIC_FAILURE: {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
