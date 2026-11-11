#!/usr/bin/env python3
"""Candidate-only public feedback for MoE all-gather, scatter and grouped GEMM on eight GPUs."""

from __future__ import annotations
import argparse, json, math, os, subprocess, sys, tempfile, traceback
from pathlib import Path
from typing import Any
from source_admission import validate_submission

ROOT = Path(__file__).resolve().parent
PUBLIC_CASES_PATH = ROOT / "public_cases.json"
FAMILIES = ("tp8", "tp4", "tp2", "ep", "decode", "sequence")
SUITE_IDS = {"one": "public-one-v2", "quick": "public-quick-v2", "full": "public-v2", "select": "public-select-v2"}
METRICS = (
    "rel_l1", "cossim", "row_rel_l2_p999", "row_rel_l2_max", "max_abs",
    "rel_l1_limit", "row_rel_l2_p999_limit", "row_rel_l2_max_limit",
    "freivalds_rel_l2", "freivalds_worst_row",
)
IDENTITY = ("case_id", "family", "tier", "group", "W", "TP", "EP", "K", "N", "E", "E_rank", "topk", "M",
            "layers", "split_counts", "effective_flops")


def load_public_cases() -> list[dict[str, Any]]:
    payload = json.loads(PUBLIC_CASES_PATH.read_text())
    if payload.get("schema_version") != 2 or not isinstance(payload.get("cases"), list):
        raise RuntimeError("public case file differs")
    return [dict(x) for x in payload["cases"]]


def cases_for_suite(name: str, cases: list[dict[str, Any]], *, family: str | None = None,
                    case_ids: list[str] | None = None) -> list[dict[str, Any]]:
    edges = [x for x in cases if x["family"] == "edge"]
    scored = [x for x in cases if x["family"] != "edge"]
    if case_ids:
        wanted = set(case_ids)
        chosen = [x for x in cases if x["case_id"] in wanted]
        if len(chosen) != len(wanted):
            raise RuntimeError(f"unknown public case ids: {sorted(wanted - {x['case_id'] for x in chosen})}")
        if not any(x["family"] != "edge" for x in chosen):
            raise RuntimeError("select at least one scored case")
        return chosen
    if family is not None:
        chosen = [x for x in scored if x["family"] == family]
        if not chosen:
            raise RuntimeError(f"no public cases in family {family}")
        return chosen
    if name == "one":
        return [scored[0]]
    if name == "quick":
        picked: dict[str, dict[str, Any]] = {}
        for case in scored:
            picked.setdefault(case["family"], case)
        return edges[:4] + list(picked.values())
    if name == "full":
        return edges + scored
    raise ValueError("unknown suite")


def _fields(row: dict[str, Any], fields: tuple[str, ...], optional: tuple[str, ...] = ()) -> dict[str, Any]:
    missing = [x for x in fields if x not in row]
    if missing:
        raise RuntimeError(f"result missing {missing}")
    clean = {x: row[x] for x in fields}
    clean.update({x: row[x] for x in optional if x in row})
    return clean


def public_payload(result: dict[str, Any], suite: str, version: int, count: int) -> dict[str, Any]:
    cases = []
    for row in result["cases"]:
        clean = _fields(row, IDENTITY + ("median_ms", "tflops", "regime"), METRICS + ("hidden_comm",))
        trial_ms = [float(x) * 1e3 for x in row["trial_seconds"]]
        if not trial_ms or not all(math.isfinite(x) and x > 0 for x in trial_ms):
            raise RuntimeError("bad trial spread")
        clean["trial_min_ms"] = min(trial_ms)
        clean["trial_max_ms"] = max(trial_ms)
        clean["per_rank_rows"] = [x["rows_per_trial"][-1] for x in row["rank_records"]]
        cases.append(clean)
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
        "rank_agreement_ok": bool(result["rank_agreement_ok"]),
        "trials_per_case": int(result["trials_per_case"]),
        "edge_cases": [_fields(x, IDENTITY, METRICS) for x in result["edge_cases"]],
        "cases": cases,
        "family_tflops": {k: float(v) for k, v in result["family_tflops"].items()},
        "score_tflops": float(result["score_tflops"]),
        "hidden_comm": result.get("hidden_comm"),
    }


def run_world(candidate: Path, cases: list[dict[str, Any]], suite_id: str, trials: int, timeout: int,
              progress: bool) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="moe-ag-public-") as raw:
        root = Path(raw)
        output = root / "result.json"
        manifest = {"schema_version": 2, "suite_id": suite_id, "cases": cases}
        env = dict(os.environ)
        for key in tuple(env):
            if key.lower() in {"http_proxy", "https_proxy", "all_proxy", "no_proxy"}:
                env.pop(key, None)
        command = [
            sys.executable, str(ROOT / "world_worker.py"),
            "--candidate", str(candidate),
            "--output", str(output),
            "--runtime-root", str(ROOT),
            "--cache", str(root / "cache"),
            "--trials", str(trials),
            "--warmups", "1",
            "--timeout", str(timeout),
            "--expected-count", str(len(cases)),
        ]
        if (ROOT / "routing").is_dir():
            command += ["--fixtures", str(ROOT / "routing")]
        if progress:
            command.append("--progress")
        completed = subprocess.run(command, input=json.dumps(manifest), text=True, env=env,
                                   timeout=timeout + 30, check=False)
        if completed.returncode:
            raise RuntimeError(f"eight-rank public worker failed: {completed.returncode}")
        return json.loads(output.read_text())


def _print(payload: dict[str, Any]) -> None:
    print(
        "SOURCE_ADMISSION          PASS\nEDGE_FIXTURES             PASS\nCORRECTNESS               PASS\n"
        "RANK_AGREEMENT            PASS\nCANDIDATE_ONLY            YES\nHIDDEN_CASES              NOT_RUN"
    )
    for row in payload["cases"]:
        line = (
            f"{row['case_id']:<28} {row['family']:<8} W={row['W']} TP={row['TP']} EP={row['EP']} "
            f"M={row['M']:<7} {row['tflops']:9.3f} TFLOPS "
            f"min/median/max={row['trial_min_ms']:.3f}/{row['median_ms']:.3f}/{row['trial_max_ms']:.3f} ms"
        )
        if "rel_l1" in row:
            line += f" RelL1={row['rel_l1']:.2e}/{row['rel_l1_limit']:.2e}"
        if row.get("hidden_comm") is not None:
            line += f" hidden_comm={row['hidden_comm']:.3f}"
        print(line)
    families = " ".join(f"{k}={v:.3f}" for k, v in payload["family_tflops"].items())
    print(f"PUBLIC_FAMILY_TFLOPS {families}")
    print(f"PUBLIC_PRIMARY_TFLOPS {payload['score_tflops']:.6f}")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--lab-root", type=Path, default=Path("/app/workdir/moe_ag_lab"))
    p.add_argument("--suite", choices=("one", "quick", "full"), default="full")
    p.add_argument("--family", choices=FAMILIES)
    p.add_argument("--case", action="append", dest="cases")
    p.add_argument("--trials", type=int)
    p.add_argument("--candidate-version", type=int, required=True)
    p.add_argument("--json-output", type=Path)
    p.add_argument("--timeout", type=int, default=5400)
    p.add_argument("--quiet", action="store_true")
    a = p.parse_args(argv)
    try:
        admission = validate_submission(a.lab_root, a.candidate_version, workspace_mode=True)
        if not admission["ok"]:
            raise RuntimeError("public source admission failed: " + "; ".join(admission["errors"]))
        cases = cases_for_suite(a.suite, load_public_cases(), family=a.family, case_ids=a.cases)
        suite = "select" if (a.family or a.cases) else a.suite
        trials = a.trials if a.trials else (3 if suite in ("full", "select") else 1)
        if not 1 <= trials <= 7:
            raise RuntimeError("trials must be 1-7")
        result = run_world(Path(admission["selected_path"]), cases, SUITE_IDS[suite], trials, a.timeout,
                           progress=not a.quiet)
        payload = public_payload(result, suite, a.candidate_version, int(admission["num_candidate_versions"]))
        if a.json_output:
            a.json_output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        _print(payload)
        return 0
    except Exception as exc:
        print(f"PUBLIC_FAILURE: {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
