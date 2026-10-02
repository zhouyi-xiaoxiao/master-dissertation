"""
recheck_headline.py -- from-scratch re-check of headline claims of note R4, written separately from the main code.
Written without importing unilib.py.  Exact arithmetic (fractions) where stated.
"""
from fractions import Fraction as Fr
import itertools, json, os, sys
import numpy as np

def box_P(N, d, q):
    """transition matrix (Fraction entries as dict rows) of lazy walk, activity q, move-cancel walls"""
    sites = list(itertools.product(range(N), repeat=d))
    idx = {s: i for i, s in enumerate(sites)}
    rows = []
    for s in sites:
        row = {}
        stay = 1 - q
        for ax in range(d):
            for sg in (-1, 1):
                t = list(s); t[ax] += sg
                if 0 <= t[ax] < N:
                    j = idx[tuple(t)]
                    row[j] = row.get(j, 0) + q / (2 * d)
                else:
                    stay += q / (2 * d)
        row[idx[s]] = row.get(idx[s], 0) + stay
        rows.append(row)
    return sites, idx, rows

def fpt_pmf(N, d, q, x0, a, tmax):
    sites, idx, rows = box_P(N, d, q)
    ia, i0 = idx[a], idx[x0]
    rho = {i0: Fr(1) if isinstance(q, Fr) else 1.0}
    f = [0]
    for t in range(1, tmax + 1):
        new = {}
        hit = 0
        for i, w in rho.items():
            for j, p in rows[i].items():
                if j == ia:
                    hit += w * p
                else:
                    new[j] = new.get(j, 0) + w * p
        f.append(hit)
        rho = new
    return f

def sign_changes(seq):
    s = [x for x in seq if x != 0]
    return sum(1 for i in range(len(s) - 1) if (s[i] > 0) != (s[i + 1] > 0))

out = {}
# (1) d=1, N=4, q=1: f(3)=1/8, f(4)=1/16, f(5)=3/32
f = fpt_pmf(4, 1, Fr(1), (0,), (3,), 6)
out['1D N4 q1'] = [str(x) for x in f]
assert f[3] == Fr(1, 8) and f[4] == Fr(1, 16) and f[5] == Fr(3, 32)
# (2) d=1 N=4: unimodal at q=4/5 on a window; not at q=81/100
for q, expect in ((Fr(4, 5), True), (Fr(81, 100), False)):
    f = fpt_pmf(4, 1, q, (0,), (3,), 80)
    df = [f[t + 1] - f[t] for t in range(len(f) - 1)]
    assert (sign_changes(df) <= 1) == expect, (q, sign_changes(df))
# (3) d=2 N=9 q=1 corner-corner: dip at t*=109
f = fpt_pmf(9, 2, Fr(1), (0, 0), (8, 8), 112)
assert f[108] > f[109] < f[110], (f[108], f[109], f[110])
out['2D N9 q1 dip at 109'] = True
# (4) Thm 8.1(a): d=3,N=3, (3,2,2)->(1,2,2) (1-based) at q=1/2: f(2)=1/144 > f(6)=6577/995328 < f(25)
f = fpt_pmf(3, 3, Fr(1, 2), (2, 1, 1), (0, 1, 1), 26)
assert f[2] == Fr(1, 144) and f[6] == Fr(6577, 995328) and f[6] < f[25], (f[2], f[6], float(f[25]))
out['3D N3 face-centres q=1/2'] = [str(f[2]), str(f[6]), float(f[25])]
# (5) Prop 8.4: d=2,N=3 centre->corner q=1/2
f = fpt_pmf(3, 2, Fr(1, 2), (1, 1), (2, 2), 7)
assert (f[4], f[5], f[6]) == (Fr(67, 2048), Fr(481, 16384), Fr(3457, 131072)) and f[5] ** 2 < f[4] * f[6]
# (6) Theorem 4.12(ii) & Lemma 4.7(a): exact check, d=2, N=4,5 q=1/2 and q=q_N-ish (<= 1/(1+cos(pi/N)))
def free_prop(N, d, q, x0, a, tmax):
    sites, idx, rows = box_P(N, d, q)
    v = {idx[x0]: Fr(1)}
    u = [v.get(idx[a], Fr(0))]
    for t in range(tmax):
        new = {}
        for i, w in v.items():
            for j, p in rows[i].items():
                new[j] = new.get(j, 0) + w * p
        v = new
        u.append(v.get(idx[a], Fr(0)))
    return u
def noreturn(N, d, q, a, tmax):
    """E(k)=P_a(X_1..X_k != a)"""
    sites, idx, rows = box_P(N, d, q)
    ia = idx[a]
    v = {ia: Fr(1)}
    E = [Fr(1)]
    for t in range(tmax):
        new = {}
        for i, w in v.items():
            for j, p in rows[i].items():
                if j != ia:
                    new[j] = new.get(j, 0) + w * p
        v = new
        E.append(sum(v.values()))
    return E
for (N, d, q, x0, a) in [(4, 2, Fr(1, 2), (0, 0), (3, 3)), (5, 2, Fr(1, 2), (0, 0), (2, 2)), (5, 2, Fr(1, 2), (2, 2), (4, 4)),
                          (3, 3, Fr(3, 5), (0, 0, 0), (2, 2, 2)), (4, 2, Fr(58, 100), (0, 0), (3, 3))]:
    tmax = 90
    u = free_prop(N, d, q, x0, a, tmax)
    E = noreturn(N, d, q, a, tmax)
    f = fpt_pmf(N, d, q, x0, a, tmax)
    phi = [u[0]] + [u[t] - u[t - 1] for t in range(1, tmax + 1)]
    # last exit identity
    for t in range(tmax + 1):
        assert f[t] == sum(E[k] * phi[t - k] for k in range(t + 1)), (N, d, q, t)
    # phi log-concave, support interval
    assert all(p >= 0 for p in phi)
    pos = [t for t in range(tmax + 1) if phi[t] > 0]
    assert pos == list(range(pos[0], tmax + 1))
    assert all(phi[t] ** 2 >= phi[t - 1] * phi[t + 1] for t in range(1, tmax)), (N, d, q)
    df = [f[t + 1] - f[t] for t in range(tmax)]
    assert sign_changes(df) <= 1
out['Thm 4.12(ii)/Lemma 4.7(a) exact windows'] = True
# q_N check: N=4: q_N = 1/(1+cos(pi/4)) = 0.5857..; 58/100 < q_N ok (used above).
print(json.dumps(out, indent=1))
print("ALL CHECKS PASS")
