"""Check pipelines shared by the public runner and the Judge of the Miles ALFWorld task.

Both take a finished `task_entry.py` output directory, its audit directory and the
contract it ran with, and return {check_name: {"passed": bool, ...}}. No candidate code
is imported; every tool runs as a separate process.
"""

import json
import shutil
from pathlib import Path

import alfworld_checks as oc


def provenance(audit, mode, steps, contract_path, final_hf, work, lr=None):
    out = Path(work) / f"provenance_{mode}.json"
    cmd = [oc.PY, oc.tool("audit/alfworld_provenance_check.py"), "--audit-dir", str(audit), "--mode", mode,
           "--steps", str(steps), "--contract", str(contract_path), "--initial-hf", oc.POLICY,
           "--final-hf", str(final_hf), "--out", str(out)]
    if lr is not None:
        cmd += ["--lr", repr(float(lr))]
    code, res = oc.run_check(cmd, Path(work) / f"provenance_{mode}.log")
    return res or {"passed": False, "error": f"provenance checker produced no report (exit {code})"}


def oracle_cmd(dump, dest, contract, snapshot=None, grads=None, grad_norm=None):
    cmd = [oc.PY, oc.tool("alfworld_oracle.py"), "--dump", dump, "--policy", oc.POLICY,
           "--eps-clip", str(contract["loss"]["eps_clip"]), "--eps-clip-high", str(contract["loss"]["eps_clip_high"]),
           "--out", str(dest)]
    if snapshot:
        cmd += ["--policy-snapshot", snapshot]
    if grads:
        cmd += ["--grad-records", ",".join(grads)]
        if grad_norm is not None:
            cmd += ["--grad-norm", repr(float(grad_norm))]
    return cmd


def strip(r):
    return {k: v for k, v in (r or {}).items() if k not in ("per_sample",)}


def check_probe(out_dir, batch_path, contract_path, audit, work, qual=None):
    """P1-P4 (structure), fp32 oracle (loss and trainer log-probs), provenance (one step)."""
    qual = qual or {}
    work = Path(work)
    contract = json.loads(Path(contract_path).read_text())
    checks = {}
    code, res = oc.run_check([oc.PY, oc.tool("alfworld_probe_check.py"), "--batch", str(batch_path),
                              "--out-dir", str(out_dir), "--contract", str(contract_path)], work / "probe_check.log")
    checks["probe_dump"] = res or {"passed": False, "error": f"probe checker produced no report (exit {code})"}
    lr = json.loads(Path(batch_path).read_text())["lr"]
    prov = provenance(audit, "probe", 1, contract_path, Path(out_dir) / "final_hf", work, lr=lr)
    checks["probe_provenance"] = prov
    jobs = [("probe", oracle_cmd(oc.step_glob(out_dir, 0), work / "oracle_probe.json", contract), work / "oracle_probe.log")]
    out = oc.run_parallel(jobs)
    code, r = out["probe"]
    ok, why = oc.oracle_verdict(r, qual.get("oracle_limits"))
    checks["probe_oracle"] = {"passed": ok and code == 0, "reasons": why, **strip(r)}
    return checks, prov


def check_run(out_dir, contract_path, audit, work, rng, qual=None, probe_prov=None, n_served=1, keep_snapshots=False):
    """Provenance, audit E1-E6, per-step loss, same path as the probe, fp32 oracle (loss and trainer
    log-probs) at step 0 and at `n_served` random steps whose served weights were copied. Returns
    (checks, audit_report). The served-weight copies (audit/snapshots, one full policy each) are
    deleted when the checks end, whatever the outcome, unless `keep_snapshots`."""
    try:
        return _check_run(out_dir, contract_path, audit, work, rng, qual, probe_prov, n_served)
    finally:
        if not keep_snapshots:
            shutil.rmtree(Path(audit) / "snapshots", ignore_errors=True)


def _check_run(out_dir, contract_path, audit, work, rng, qual, probe_prov, n_served):
    qual = qual or {}
    work = Path(work)
    contract = json.loads(Path(contract_path).read_text())
    R = contract["rounds"]
    checks = {}
    prov = provenance(audit, "run", R, contract_path, Path(out_dir) / "final_hf", work)
    checks["provenance"] = prov
    excl = work / "exclude_intervals.json"
    oc.write_json(excl, prov.get("exclude_intervals") or [])
    code, res = oc.run_check([oc.PY, oc.tool("audit/alfworld_audit_check.py"), "--audit-dir", str(audit),
                              "--dumps", oc.step_files(out_dir), "--contract", str(contract_path),
                              "--tokenizer", oc.POLICY, "--exclude-intervals", str(excl),
                              "--out", str(work / "audit_check.json")], work / "audit_check.log")
    checks["audit"] = res or {"passed": False, "error": f"audit checker produced no report (exit {code})"}
    checks["steps"] = oc.steps_self_consistency(out_dir, R, contract["loss"]["eps_clip"], contract["loss"]["eps_clip_high"])
    if probe_prov is not None:
        want, got = probe_prov.get("model_config") or {}, prov.get("model_config") or {}
        diff = {k: [want.get(k), got.get(k)] for k in sorted(set(want) | set(got)) if want.get(k) != got.get(k)}
        checks["same_path"] = {"passed": bool(want) and not diff, "differences_probe_vs_run": diff}
    if not all(v.get("passed") for v in checks.values()):
        return checks, checks["audit"]

    by_step = {}
    for s in prov.get("snapshots") or []:
        if "step" in s:
            by_step.setdefault(s["step"], s["path"])
    chosen = sorted(rng.sample(sorted(by_step), min(n_served, len(by_step))))
    jobs = [("step0", oracle_cmd(oc.step_glob(out_dir, 0), work / "oracle_step0.json", contract), work / "oracle_step0.log")]
    for k in chosen:
        jobs.append((f"served{k}", oracle_cmd(oc.step_glob(out_dir, k), work / f"oracle_step{k}.json", contract,
                                              snapshot=by_step[k]), work / f"oracle_step{k}.log"))
    out = oc.run_parallel(jobs)
    code, r = out["step0"]
    ok, why = oc.oracle_verdict(r, qual.get("oracle_limits"))
    checks["oracle_step0"] = {"passed": ok and code == 0, "reasons": why, **strip(r)}
    served = {}
    for k in chosen:
        code, r = out[f"served{k}"]
        ok, why = oc.oracle_verdict(r, qual.get("oracle_limits"))
        served[str(k)] = {"passed": ok and code == 0, "reasons": why, **strip(r)}
    checks["oracle_served_steps"] = {"passed": bool(chosen) and all(v["passed"] for v in served.values()),
                                     "steps": chosen, "detail": served}
    return checks, checks["audit"]
