"""Energy-normalized dense-reference fidelity for approximate attention.

Computed outside the kernel timer. Uniform and tail sets are checked separately;
each head is also constrained so low-energy heads cannot hide in a global mean.
No seed-relative multiplier or downstream score participates.
"""
import math

PROTOCOL = 'minference-kernel-energy-fidelity-v1'
REFERENCE_ENERGY_FLOOR = 1e-12
LIMIT_KEYS = ('nrmse','head_max_nrmse','tail_nrmse','tail_head_max_nrmse')

def energy_stats(torch, actual, reference):
    if actual.shape != reference.shape or actual.ndim != 4:
        raise ValueError('matching [B,H,rows,D] samples required')
    a=actual.float();r=reference.float()
    if not bool(torch.isfinite(a).all()) or not bool(torch.isfinite(r).all()):
        return {'finite':False,'nrmse':float('inf'),'head_max_nrmse':float('inf')}
    err=(a-r).square().sum(dim=(0,2,3));ref=r.square().sum(dim=(0,2,3))
    ratios=(err/ref.clamp_min(REFERENCE_ENERGY_FLOOR)).sqrt()
    return {'finite':True,'nrmse':float((err.sum()/ref.sum().clamp_min(REFERENCE_ENERGY_FLOOR)).sqrt()),
        'head_max_nrmse':float(ratios.max()),'head_nrmse':[float(x) for x in ratios],
        'reference_energy':float(ref.sum()),'reference_energy_floor_heads':int((ref<REFERENCE_ENERGY_FLOOR).sum()),
        'rows':actual.shape[0]*actual.shape[1]*actual.shape[2]}

def validate_limits(limits):
    if set(limits)!=set(LIMIT_KEYS) or any(type(v) not in (int,float) or not math.isfinite(v) or not 0<v<1 for v in limits.values()):
        raise ValueError('independent finite energy-error budgets strictly below1 required')
    return limits

def passes(uniform,tail,limits):
    validate_limits(limits)
    return bool(uniform.get('finite') and tail.get('finite') and all(
        type(s.get(k)) in (int,float) and math.isfinite(s[k]) and s[k]<=limits[p+k]
        for s,p in ((uniform,''),(tail,'tail_')) for k in ('nrmse','head_max_nrmse')))


BOUNDARY_LENGTHS = (64,65,511,1889,2047,2048,2049,4095,4096,4097,8193)

def validate_boundary_checks(checks,limits):
    if not isinstance(checks,list) or [c.get('n') for c in checks]!=list(BOUNDARY_LENGTHS):
        raise ValueError('complete kernel ABI boundary checks required')
    for c in checks:
        n=c['n']
        if any(c.get(k) is not True for k in ('numeric_passed','full_write_ok','input_immutable_ok')) or not passes(c['uniform_stats'],c['tail_stats'],limits):
            raise ValueError('kernel ABI boundary fidelity/integrity failed')
        for side,count in (('uniform_stats',n),('tail_stats',min(n,64))):
            stats=c[side];heads=stats.get('head_nrmse')
            if stats.get('rows')!=32*count or not isinstance(heads,list) or len(heads)!=32 or any(type(x) not in (int,float) or not math.isfinite(x) or x<0 for x in heads) or max(heads)!=stats['head_max_nrmse']:
                raise ValueError('kernel boundary reference coverage differs')
    return checks
