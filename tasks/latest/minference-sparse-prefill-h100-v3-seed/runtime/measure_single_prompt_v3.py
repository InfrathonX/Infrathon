"""Candidate-side prototype; consume five preopened FDs after permanent drop.

The trusted caller opens/authenticates files and policy, creates CUDA timing
primitives and binds helpers BEFORE importing candidate. No model execution or
disk preparation enters the measured window. Not GPU qualified yet.
"""
from __future__ import annotations
from kernel_fidelity_v3 import energy_stats, BOUNDARY_LENGTHS, passes, validate_boundary_checks
from pathlib import Path
import os
import math
from single_prompt_archive_v3 import TARGET_LAYERS, canonical_sha, read_checked, validate_bundle



def measure_boundaries(torch,entrypoint,checked,clock,full_reference,limits,nonce):
    """Fresh synthetic ABI checks, outside throughput, full FP32 reference."""
    checks=[]
    generator=torch.Generator(device='cuda').manual_seed(nonce % (2**63-1))
    for n in BOUNDARY_LENGTHS:
        def inputs():
            return {name:torch.randn((1,heads,n,128),device='cuda',dtype=torch.bfloat16,generator=generator)
                    for name,heads in (('q',32),('k',8),('v',8))}
        warm=inputs();discarded,_=checked(torch,warm,entrypoint,clock,measured=False)
        del discarded,warm
        values=inputs();actual,_=checked(torch,values,entrypoint,clock,measured=False)
        reference=full_reference(torch,values['q'],values['k'],values['v'],128**-.5)
        uniform=energy_stats(torch,actual,reference)
        tail=energy_stats(torch,actual[:,:,-min(n,64):],reference[:,:,-min(n,64):])
        checks.append({'n':n,'uniform_stats':uniform,'tail_stats':tail,'numeric_passed':passes(uniform,tail,limits),
                       'full_write_ok':True,'input_immutable_ok':True})
        del values,actual,reference
    return validate_boundary_checks(checks,limits)


def measure_prompt(candidate, *, bundle, plan, handles, identities, policy,
                   policy_sha256, expected_candidate_sha256, nonce, runtime, archive,
                   torch, before_candidate_import, purpose="private-judge"):
    validate_bundle(bundle, plan)
    if type(nonce) is not int or not 0 < nonce < 2**128:
        raise ValueError("a private nonzero 128-bit row nonce is required")
    if set(handles) != set(bundle["fixtures"]) or set(identities) != set(handles):
        raise ValueError("preopened FD set differs")
    from single_prompt_archive_v3 import validate_purpose
    validate_purpose(purpose, plan["partition"], os.geteuid())
    if bundle.get("purpose") != purpose:
        raise ValueError("bundle purpose differs")
    if purpose == "private-judge" and before_candidate_import is None:
        raise ValueError("trusted root setup and permanent candidate drop required")
    candidate = Path(candidate)
    # Bind trusted callable objects and CUDA primitives before arbitrary code.
    checked, probe, stats = runtime._checked_call, runtime.probe_dense_reference, runtime.row_error_stats
    select_rows, limits_fn, passes_fn = runtime._probe_rows, runtime.case_limits, runtime.numeric_passes
    reader, scale = read_checked, runtime.SOFTMAX_SCALE
    runtime._use_exact_fp32(torch)
    clock = runtime.TrustedClock.capture(torch)
    boundary_check = measure_boundaries
    full_reference = runtime.trusted_dense_reference
    if before_candidate_import is not None:
        before_candidate_import()
    if os.getresuid() != (1000, 1000, 1000) or os.getresgid() != (1000, 1000, 1000):
        raise ValueError("candidate must run after permanent agent UID/GID drop")
    module, entrypoint, actual_sha = runtime.load_candidate(candidate)
    if actual_sha != expected_candidate_sha256:
        raise ValueError("candidate sealed identity differs")
    pid = bundle["prompt_id"]
    rows = []
    boundary_checks = boundary_check(torch,entrypoint,checked,clock,full_reference,policy['limits']['default'],nonce) if pid==plan['case_groups'][0]['trials'][0].rsplit('-L',1)[0] else None
    for layer in TARGET_LAYERS:
        wid, tid = pid + "-L2", pid + f"-L{layer}"
        warm = reader(torch, handles[wid], identities[wid], archive)
        discarded, _ = checked(torch, warm, entrypoint, clock, measured=False)
        del discarded, warm
        values = reader(torch, handles[tid], identities[tid], archive)
        f = bundle["fixtures"][tid]
        uniform_rows, tail_rows, all_rows = select_rows(torch, f["actual_n"], nonce, tid, values["q"].device)
        reference = probe(torch, values["q"], values["k"], values["v"], scale, all_rows)
        out, measurement = checked(torch, values, entrypoint, clock, measured=True)
        selected = out.index_select(2, all_rows)
        split = int(uniform_rows.numel())
        uniform = {**stats(torch, selected[:, :, :split], reference[:, :, :split]).as_dict(), **energy_stats(torch, selected[:, :, :split], reference[:, :, :split])}
        tail = {**stats(torch, selected[:, :, split:], reference[:, :, split:]).as_dict(), **energy_stats(torch, selected[:, :, split:], reference[:, :, split:])}
        group = next(g for g in plan["case_groups"] if tid in g["trials"])
        limits = limits_fn(policy, group)
        rows.append({"fixture_id": tid, "prompt_id": pid, "input_token_sha256": f["input_token_sha256"],
            "actual_n": f["actual_n"], "seconds": measurement.seconds,
            "host_seconds": measurement.host_seconds, "event_seconds": measurement.event_seconds,
            "uniform_stats": uniform, "tail_stats": tail, "limits": limits,
            "numeric_passed": passes_fn(uniform, tail, limits),
            "full_write_ok": True, "input_immutable_ok": True})
        del values, reference, selected, out, all_rows, uniform_rows, tail_rows
    return {"schema_version": 1, "protocol": bundle["protocol"], "prompt_id": pid,
        "plan_sha256": canonical_sha(plan), "bundle_sha256": canonical_sha(bundle),
        "candidate_sha256": actual_sha, "numerical_policy_sha256": policy_sha256,
        "trial_randomness_protocol": runtime.TRIAL_RANDOMNESS_PROTOCOL,
        "purpose": purpose, "execution_uid": 1000,
        "candidate_privilege_drop": purpose == "private-judge", "trials": rows, "boundary_checks":boundary_checks,
        "qualified": False}
