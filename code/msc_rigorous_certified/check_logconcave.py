"""check_logconcave.py -- sanity check for Theorem E (1D, q <= 1/2: the PMF is log-concave) and
exploration of log-concavity in other cases.  Exact rational arithmetic (fractions.Fraction).
Checks f(t)^2 >= f(t-1) f(t+1) for t0 < t <= horizon, horizon = 4 * (tail time).
Output: certs/logconcavity_check.json
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import json, os, sys
from fractions import Fraction
from itertools import product
HERE = os.path.dirname(os.path.abspath(__file__))


def pmf(d, N, q, horizon_factor=4):
    a, x0 = (N - 1,) * d, (0,) * d
    out = {}
    for y in product(range(N), repeat=d):
        if y == a:
            continue
        row = {y: 1 - q}
        for ax in range(d):
            for sgn in (-1, 1):
                z = list(y); z[ax] += sgn
                z = tuple(z) if 0 <= z[ax] < N else y
                row[z] = row.get(z, 0) + q / (2 * d)
        out[y] = {z: p for z, p in row.items() if p != 0}
    v = {x0: Fraction(1)}
    f = [Fraction(0)]
    T = None
    while True:
        f.append(sum(p * out[y].get(a, 0) for y, p in v.items()))
        w = {}
        for y, p in v.items():
            for z, pz in out[y].items():
                if z != a:
                    w[z] = w.get(z, 0) + p * pz
        if T is None and all(w.get(y, 0) < v.get(y, 0) for y in out):
            T = len(f) - 1
        if T is not None and len(f) - 1 >= horizon_factor * T + 5:
            return f, T
        v = w


res = []
cases = [(1, N, q) for q in (Fraction(1, 2), Fraction(1, 3), Fraction(1, 10), Fraction(4, 5), Fraction(3, 5), Fraction(1)) for N in (2, 3, 4, 5, 8, 12, 20)]
cases += [(2, N, q) for q in (Fraction(1, 2), Fraction(4, 5), Fraction(1)) for N in (2, 3, 4, 5, 6, 7)]
cases += [(3, N, q) for q in (Fraction(1, 2), Fraction(4, 5)) for N in (2, 3, 4)]
for d, N, q in cases:
    f, T = pmf(d, N, q)
    t0 = next(t for t in range(1, len(f)) if f[t] > 0)
    viol = [t for t in range(t0 + 1, len(f) - 1) if f[t] ** 2 < f[t - 1] * f[t + 1]]
    res.append(dict(d=d, N=N, q=str(q), t0=t0, horizon=len(f) - 1, tail_time=T, logconcave_on_horizon=not viol,
                    first_violations=viol[:5], n_violations=len(viol)))
    print(res[-1], flush=True)
json.dump(res, open(_os.path.join(_R, 'data', 'msc_rigorous_certified', 'logconcavity_check.json'), "w"), indent=1)
