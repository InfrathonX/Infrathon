#!/usr/bin/env python3
"""Candidate-only public feedback for the quantized attention BF16 task."""
from __future__ import annotations
import argparse,json,math,os,sys,traceback
from pathlib import Path
from typing import Any,Callable
from source_admission import validate_submission

ROOT = Path(__file__).resolve().parent


def _load() -> tuple[Callable[...,dict[str,Any]],tuple[dict[str,Any],...]]:
    from quant_attention_runtime import PUBLIC_REAL_LAYER_CASES, run_suite

    sys.path = [x for x in sys.path if Path(x or ".").resolve() != ROOT]
    sys.modules.pop("quant_attention_runtime", None)
    return run_suite, tuple(dict(x) for x in PUBLIC_REAL_LAYER_CASES)


# (family, B, H_q, H_kv, D, causal, profile, lengths).  None of these geometries or lengths
# is used by a hidden case; they cover the same kinds of call.
GROUPS = (
    ("video", 1, 20, 20, 128, False, "drift", (5184, 24192, 61441, 96001)),
    ("video", 2, 16, 16, 64, False, "spiky", (6697, 20276, 44017)),
    ("video", 1, 16, 16, 128, False, "drift", (96001,)),
    ("image", 2, 16, 16, 64, False, "spiky", (2050, 5001)),
    ("image", 1, 20, 20, 128, False, "drift", (4700,)),
    ("llm", 1, 24, 6, 128, True, "spiky", (4160, 14336, 65001, 190001)),
    ("llm", 2, 12, 4, 64, True, "drift", (3001, 7168)),
    ("llm", 1, 16, 4, 128, True, "spiky", (32001, 190001)),
)
# The four cases public tests had before; they keep their ids and seeds.
LEGACY = {("video", 20, 5184): ("public-p1", 1103), ("video", 16, 6697): ("public-p2", 2207),
          ("llm", 24, 4160): ("public-p3", 3301), ("llm", 12, 3001): ("public-p4", 4409)}


def _public_cases() -> tuple[dict[str, Any], ...]:
    cases = []
    for index, (family, b, hq, hkv, d, causal, profile, lengths) in enumerate(GROUPS):
        for j, l in enumerate(lengths):
            case_id, seed = LEGACY.get((family, hq, l), (f"public-{family}-h{hq}-d{d}-l{l}", 5000 + 100 * index + j))
            cases.append({"case_id": case_id, "family": family, "b": b, "h_q": hq, "h_kv": hkv, "d": d,
                          "l": l, "is_causal": causal, "regime": profile, "seed": seed, "gated": True})
    return tuple(cases)


PUBLIC_CASES = _public_cases()
QUICK_CASE_IDS = ("public-p1", "public-p4")
QUICK_PERF_ONLY_ID = "public-video-h16-d128-l96001"
METRICS = (
    "rel_l1",
    "cossim",
    "rmse",
    "worst_head_rel_l1",
    "row_rel_l2_p999",
    "row_rel_l2_max",
    "reference_rel_l1",
    "reference_cossim",
    "reference_rmse",
    "reference_worst_head_rel_l1",
    "reference_row_rel_l2_p999",
    "reference_row_rel_l2_max",
    "rel_l1_limit",
    "worst_head_rel_l1_limit",
    "row_rel_l2_p999_limit",
    "row_rel_l2_max_limit",
)
IDENTITY = (
    "case_id",
    "family",
    "b",
    "l",
    "h_q",
    "h_kv",
    "d",
    "is_causal",
    "regime",
    "softmax_scale",
)
EDGE_FIELDS = IDENTITY + METRICS
CASE_FIELDS = IDENTITY + ("gated", "warmup_seconds", "median_us", "tflops")


def cases_for_suite(name: str, case_id: str | None = None, perf_only: bool = False):
    if case_id:
        chosen = [dict(x) for x in PUBLIC_CASES if x["case_id"] == case_id]
        if not chosen:
            raise ValueError(f"unknown public case {case_id}; known: {', '.join(x['case_id'] for x in PUBLIC_CASES)}")
    elif name == "one":
        chosen = [dict(PUBLIC_CASES[0])]
    elif name == "quick":
        chosen = [dict(x) for x in PUBLIC_CASES if x["case_id"] in QUICK_CASE_IDS]
        chosen += [dict(x, gated=False) for x in PUBLIC_CASES if x["case_id"] == QUICK_PERF_ONLY_ID]
    elif name == "full":
        chosen = [dict(x) for x in PUBLIC_CASES]
    else:
        raise ValueError("unknown suite")
    if perf_only:
        chosen = [dict(x, gated=False) for x in chosen]
    return chosen


def _fields(row: dict[str,Any],fields):
    missing = [x for x in fields if x not in row]
    if missing:
        raise RuntimeError(f"trusted result missing {missing}")
    out = {x: row[x] for x in fields}
    if row.get("gated", True):
        out.update({x: row[x] for x in METRICS})
    return out


def _aggregate(value: Any):
    if not isinstance(value, dict) or not value or not set(value) <= {"video", "image", "llm", "real"}:
        raise RuntimeError("bad family aggregate")
    result = {str(k): float(v) for k, v in value.items()}
    if not all(math.isfinite(v) and v > 0 for v in result.values()):
        raise RuntimeError("bad aggregate value")
    return result


def public_payload(result: dict[str,Any],suite: str,version: int,count: int):
    return {
        "schema_version": 2,
        "suite": suite,
        "candidate_version": version,
        "num_candidate_versions": count,
        "candidate_only": True,
        "hidden_cases_run": False,
        "source_admission_ok": True,
        "correctness_ok": bool(result["correctness_ok"]),
        "edge_ok": bool(result["edge_ok"]),
        "real_activation_ok": bool(result["real_activation_ok"]),
        "trials_per_case": int(result["trials_per_case"]),
        "warmups": int(result["warmups"]),
        "edge_cases": [{x: row[x] for x in EDGE_FIELDS} for row in result["edge_cases"]],
        "cases": [_fields(dict(x), CASE_FIELDS) for x in result["cases"]],
        "family_tflops": _aggregate(result["family_tflops"]),
        "primary_tflops": float(result["primary_tflops"]),
    }


def _print(payload):
    print("SOURCE_ADMISSION          PASS\nEDGE_FIXTURES             PASS\nCORRECTNESS               PASS\nCANDIDATE_ONLY            YES\nHIDDEN_CASES              NOT_RUN")
    for row in payload["cases"]:
        accuracy = (f"RelL1={row['rel_l1']:.5f}/{row['rel_l1_limit']:.5f} p99.9={row['row_rel_l2_p999']:.5f}/{row['row_rel_l2_p999_limit']:.5f} Cos={row['cossim']:.6f}"
                    if "rel_l1" in row else "perf-only (no golden check)")
        print(f"{row['case_id']:<32} {row['family']:<5} L={row['l']:<7} {row['tflops']:9.3f} TFLOPS  warm-up {row['warmup_seconds']:.1f}s  {accuracy}")
    for family,value in sorted(payload["family_tflops"].items()):
        print(f"FAMILY_{family.upper()}_TFLOPS {value:.6f}")
    print(f"PUBLIC_PRIMARY_TFLOPS {payload['primary_tflops']:.6f}")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument(
        "--lab-root",
        type=Path,
        default=Path("/app/workdir/quant_attention_lab"),
    )
    p.add_argument("--suite", choices=("one", "quick", "full"), default="full")
    p.add_argument("--case", help="run one public case by id (real layers excluded)")
    p.add_argument("--perf-only", action="store_true", help="skip the golden comparison; time only")
    p.add_argument("--candidate-version", type=int, required=True)
    p.add_argument("--json-output", type=Path)
    p.add_argument(
        "--fixture-dir",
        type=Path,
        default=Path("/workspace/benchmark/fixtures"),
    )
    a = p.parse_args(argv)
    try:
        admission = validate_submission(
            a.lab_root,
            a.candidate_version,
            workspace_mode=True,
        )
        if not admission["ok"]:
            raise RuntimeError(
                "public source admission failed: " + "; ".join(admission["errors"])
            )
        os.environ["QATTN_FIXTURE_DIR"] = str(a.fixture_dir)
        run, public_real_layers = _load()
        globals().pop("_load", None)
        real = () if a.case or a.suite == "one" else tuple(dict(x, gated=not a.perf_only) for x in public_real_layers)
        result = run(
            Path(admission["selected_path"]),
            cases_for_suite(a.suite, a.case, a.perf_only),
            trials=3 if a.suite == "full" or a.case else 1,
            warmups=1,
            real_layer_cases=real,
            progress=lambda line: print(line, file=sys.stderr, flush=True),
        )
        suite = f"case:{a.case}" if a.case else a.suite
        payload = public_payload(
            result,
            suite,
            a.candidate_version,
            int(admission["num_candidate_versions"]),
        )
        if a.json_output:
            a.json_output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        _print(payload)
        print(json.dumps(payload, sort_keys=True))
        return 0
    except Exception as exc:
        print(
            "CORRECTNESS               FAIL\nCANDIDATE_ONLY            YES\nHIDDEN_CASES              NOT_RUN",
            file=sys.stderr,
        )
        print(f"PUBLIC_FAILURE: {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
