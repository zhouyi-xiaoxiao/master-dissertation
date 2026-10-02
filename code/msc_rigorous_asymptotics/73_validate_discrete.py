"""Validation (not part of the proof) of Theorem 7.8: exact discrete modes (time stepping here; floating-point study for larger N)
versus the certified discrete window  t_cf - c_minus/(X mu) < t* < t_cf + c_plus/(X mu) + 2,  q <= 1/2."""
import json, math
import numpy as np
from r4lib import scalars, exact_modes_D, mu_real, DATA


def mode_stepper(d, N, q):
    shape = (N,) * d
    p = np.zeros(shape); p[(0,) * d] = 1.0
    tgt = (N - 1,) * d
    mu = mu_real(d, N, q)
    tmax = int(12.0 / mu)
    best, tbest = 0.0, None
    surv_prev = 1.0
    for t in range(1, tmax + 1):
        new = (1 - q) * p
        for ax in range(d):
            mv = np.moveaxis(p, ax, 0)
            nw = np.moveaxis(new, ax, 0)
            nw[1:] += (q / (2 * d)) * mv[:-1]
            nw[:-1] += (q / (2 * d)) * mv[1:]
            nw[0] += (q / (2 * d)) * mv[0]
            nw[-1] += (q / (2 * d)) * mv[-1]
        f = new[tgt]
        # f(t) = mass arriving at the target at step t (target mass before the step is 0)
        new[tgt] = 0.0
        if f > best:
            best, tbest = f, t
        p = new
        if tbest is not None and t > 3 * tbest + 50:
            break
    return tbest


cert = {}
for d in (2, 3):
    cert[d] = [json.loads(l) for l in open(DATA + '/72_cert_discrete_d%d.jsonl' % d)]


def block(d, N):
    for r in cert[d]:
        if r['Na'] <= N and (r['Nb'] is None or N <= r['Nb']):
            return r


cases = [(2, N, q) for N in (12, 16, 20, 30, 45) for q in (0.5, 0.25)] + [(3, N, q) for N in (10, 13, 17) for q in (0.5, 0.2)]
out = []
mD = exact_modes_D()
ok_all = True
for d, N, q in cases:
    tstar = mode_stepper(d, N, q)
    s = scalars(d, N)
    mu = mu_real(d, N, q)
    r = block(d, N)
    lo = (s['tau_cf'] - r['c_minus'] / s['X']) / mu
    hi = (s['tau_cf'] + r['c_plus'] / s['X']) / mu + 2
    rec = dict(d=d, N=N, q=q, t_star=tstar, t_lo=lo, t_hi=hi, inside=bool(lo < tstar < hi), theta=float(s['X'] * (mu * tstar - s['tau_cf'])),
               c_minus=r['c_minus'], c_plus=r['c_plus'])
    ok_all &= rec['inside']; out.append(rec); print(json.dumps(rec), flush=True)
for (d, N, q), (mode, mfpt) in sorted(mD.items()):
    if d == 1 or q > 0.5 or N < 12:
        continue
    if (d == 2 and N > 1800) or (d == 3 and N > 160):
        continue
    s = scalars(d, N)
    mu = mu_real(d, N, q)
    r = block(d, N)
    lo = (s['tau_cf'] - r['c_minus'] / s['X']) / mu
    hi = (s['tau_cf'] + r['c_plus'] / s['X']) / mu + 2
    rec = dict(d=d, N=N, q=q, t_star=mode, t_lo=lo, t_hi=hi, inside=bool(lo < mode < hi), theta=float(s['X'] * (mu * mode - s['tau_cf'])),
               c_minus=r['c_minus'], c_plus=r['c_plus'], source='float_study')
    ok_all &= rec['inside']; out.append(rec); print(json.dumps(rec), flush=True)
print('ALL INSIDE' if ok_all else 'SOME OUTSIDE')
json.dump(out, open(DATA + '/73_validate_discrete.json', 'w'), indent=1)
