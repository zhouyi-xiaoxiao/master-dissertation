"""check_lemmas.py -- numerical sanity checks of every lemma of note R5 (exact arithmetic where possible).
These checks are NOT part of any proof; they guard against mis-stated lemmas and coding errors.
Output: certs/lemma_checks.json
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import json, os, random, sys
from fractions import Fraction
from itertools import product
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fpcore, fplimb
from mfpt_enclosure import enclosure

res = {}
random.seed(20261001)


def build(d, N, q):
    """Explicit Q (dict of dicts of Fractions) and the vector P(y, a), straight from the model rules."""
    a = (N - 1,) * d
    sites = [y for y in product(range(N), repeat=d) if y != a]
    Q, r = {}, {}
    for y in sites:
        row = {y: 1 - q}
        for ax in range(d):
            for sgn in (-1, 1):
                z = list(y); z[ax] += sgn
                z = tuple(z) if 0 <= z[ax] < N else y
                row[z] = row.get(z, 0) + q / (2 * d)
        r[y] = row.pop(a, Fraction(0))
        Q[y] = {z: p for z, p in row.items() if p != 0}
    return sites, Q, r


def apply(Q, v):          # (Q v)(z) = sum_y v(y) Q(y, z)   (Q symmetric)
    w = {}
    for y, p in v.items():
        for z, pz in Q[y].items():
            w[z] = w.get(z, 0) + p * pz
    return w


CASES = [(1, 2, Fraction(4, 5)), (1, 6, Fraction(4, 5)), (1, 9, Fraction(1, 2)), (1, 7, Fraction(1)), (2, 2, Fraction(1)),
         (2, 4, Fraction(4, 5)), (2, 5, Fraction(1, 2)), (2, 5, Fraction(1)), (3, 2, Fraction(1)), (3, 3, Fraction(4, 5)), (3, 3, Fraction(1, 2))]

# ---- Lemma (basic identities), Lemma (absorption), Lemma (positivity), Lemma (tail)
ok_basic = ok_sym = ok_pos = ok_rho = ok_tail = ok_geo = True
for d, N, q in CASES:
    sites, Q, r = build(d, N, q)
    x0 = (0,) * d
    ok_sym &= all(Q[y].get(z, 0) == Q[z].get(y, 0) for y in sites for z in Q[y])
    ok_pos &= all(len(Q[y]) > 0 for y in sites)
    n0 = d * (N - 1)
    one = {y: Fraction(1) for y in sites}
    w = dict(one)
    for _ in range(n0):
        w = apply(Q, w)
    ok_rho &= max(w.values()) <= 1 - (q / (2 * d)) ** n0
    v = {x0: Fraction(1)}
    S_prev, f, vs = Fraction(1), [None], [None]
    T = None
    for t in range(1, 400):
        ft = sum(p * r[y] for y, p in v.items())                     # (b): f(t) = (q/2d) sigma_t
        vs.append(v)
        vn = apply(Q, v)
        S_t = sum(vn.values())                                        # S(t) = 1^T Q^t e_x0
        ok_basic &= (ft == S_prev - S_t)                              # (c)
        # (b) in the form used by the programs: neighbours of the target
        nb = fpcore.neighbours_of_target(d, N)
        ok_basic &= (ft == q / (2 * d) * sum(v.get(y, 0) for y in nb))
        f.append(ft)
        if T is None and all(vn.get(y, 0) < v.get(y, 0) for y in sites):
            T = t
            gamma = max(vn.get(y, 0) / v[y] for y in sites)
        if T is not None:
            ok_tail &= all(vn.get(y, 0) < v.get(y, 0) for y in sites)   # v keeps decreasing everywhere
            if t > T:
                ok_tail &= f[t] < f[t - 1]
                ok_geo &= f[t] <= gamma ** (t - T) * f[T]
        S_prev, v = S_t, vn
        if T is not None and t > 4 * T + 20:
            break
res["Lemma basic: f = (q/2d) sigma = S(t-1)-S(t) (exact)"] = bool(ok_basic)
res["Q symmetric (exact)"] = bool(ok_sym)
res["Lemma positivity: every row of Q has a positive entry"] = bool(ok_pos)
res["Lemma absorption: ||Q^n0||_inf <= 1 - (q/2d)^n0 (exact)"] = bool(ok_rho)
res["Lemma tail: after T, v and f keep decreasing (exact, horizon 4T)"] = bool(ok_tail)
res["Lemma tail: geometric bound f(T+s) <= gamma^s f(T) (exact)"] = bool(ok_geo)

# ---- Lemma (enclosure): L_t <= 2^P v_t <= U_t <= 2^P, with a deliberately small P to stress the rounding
ok_encl = True
for d, N, q in CASES:
    cm, cs, D = fpcore.coeffs(d, q.numerator, q.denominator)
    for P in (8, 20, 40):
        X = np.zeros((N,) * d, dtype=object); X[...] = 0; X[(0,) * d] = 1
        ts = random.randint(1, 4)
        for _ in range(ts - 1):
            X = fpcore.apply_M(X, d, N, cm, cs)
        L, U = fpcore.to_fixed_point(X, ts, D, P)
        Xe, t = X, ts
        for _ in range(60):
            for idx in np.ndindex(X.shape):
                v = Fraction(int(Xe[idx]), D ** (t - 1)) * 2 ** P
                ok_encl &= (0 <= L[idx] <= v <= U[idx] <= 2 ** P)
            L = fpcore.apply_M(L, d, N, cm, cs) // D
            U = -((-fpcore.apply_M(U, d, N, cm, cs)) // D)
            Xe = fpcore.apply_M(Xe, d, N, cm, cs)
            t += 1
res["Lemma enclosure: L <= 2^P v <= U <= 2^P (exact, P = 8, 20, 40)"] = bool(ok_encl)

# ---- Lemma (limb engine): one step on random data equals the Python-integer step
ok_limb = True
for d, N, qn, qd in [(1, 7, 4, 5), (2, 5, 4, 5), (3, 4, 4, 5), (1, 6, 1, 2), (2, 6, 1, 2), (3, 3, 1, 2), (1, 5, 1, 1), (2, 4, 1, 1), (3, 3, 1, 1)]:
    cm, cs, D = fpcore.coeffs(d, qn, qd)
    for trial in range(40):
        Z = np.zeros((N,) * d, dtype=object)
        for idx in np.ndindex(Z.shape):
            Z[idx] = random.choice([0, 1, 2 ** 112, 2 ** 112 - 1, random.getrandbits(112), random.getrandbits(58), (1 << 58) - 1, 1 << 58])
        Z[(N - 1,) * d] = 0
        A, Bv = fplimb.Limbs(d, N, 2), fplimb.Limbs(d, N, 2)
        A.load(Z)
        work = tuple(np.zeros((N,) * d, dtype=np.int64) for _ in range(3))
        fplimb.step(A, Bv, cm, cs, D, False, work)
        ok_limb &= bool((Bv.dump() == fpcore.apply_M(Z, d, N, cm, cs) // D).all())
        fplimb.step(A, Bv, cm, cs, D, True, work)
        ok_limb &= bool((Bv.dump() == -((-fpcore.apply_M(Z, d, N, cm, cs)) // D)).all())
        ok_limb &= bool((A.dump() == Z).all())
res["Lemma limb: floor/ceil step identical to Python integers (360 random vectors incl. extreme limbs)"] = bool(ok_limb)

# ---- Lemma (MFPT enclosure) against an exact rational solve
def exact_tau(d, N):
    sites, Q, r = build(d, N, Fraction(1))
    idx = {y: i for i, y in enumerate(sites)}
    n = len(sites)
    A = [[Fraction(0)] * (n + 1) for _ in range(n)]
    for y in sites:
        i = idx[y]
        A[i][i] += 1
        for z, p in Q[y].items():
            A[i][idx[z]] -= p
        A[i][n] = Fraction(1)
    for c in range(n):
        piv = next(rw for rw in range(c, n) if A[rw][c] != 0)
        A[c], A[piv] = A[piv], A[c]
        pv = A[c][c]
        A[c] = [x / pv for x in A[c]]
        for rw in range(n):
            if rw != c and A[rw][c] != 0:
                fct = A[rw][c]
                A[rw] = [x - fct * y for x, y in zip(A[rw], A[c])]
    return A[idx[(0,) * d]][n]
ok_mfpt, vals = True, {}
for d, N in [(1, 2), (1, 9), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (3, 2), (3, 3)]:
    ex = exact_tau(d, N)
    lo, hi = enclosure(d, N)
    ok_mfpt &= (lo <= ex <= hi)
    vals[f"d={d},N={N}"] = str(ex)
    if d == 1:
        ok_mfpt &= (ex == N * (N - 1))
res["Lemma MFPT: enclosure contains the exact rational q*MFPT; 1D equals N(N-1)"] = bool(ok_mfpt)
res["exact q*MFPT values"] = vals

# ---- Theorem (log-concavity): product formula (6.1) in 1D against the exact PMF (floating eigenvalues)
ok_prod = True
for N, q in [(4, 0.5), (7, 0.5), (7, 0.3), (6, 0.8)]:
    n = N - 1
    Qm = np.zeros((n, n))
    for i in range(n):
        Qm[i, i] = 1 - q + (q / 2 if i == 0 else 0)
        if i + 1 < n:
            Qm[i, i + 1] = Qm[i + 1, i] = q / 2
    th = np.linalg.eigvalsh(Qm)
    ok_prod &= abs(np.prod(1 - th) - (q / 2) ** n) < 1e-12
    # f = coefficients of prod (1-th) z / (1 - th z)
    coef = np.zeros(200); coef[0] = 1.0
    for x in th:
        new = np.zeros(200)
        acc = 0.0
        for k in range(1, 200):
            acc = acc * x + coef[k - 1] * (1 - x)       # new[k] = sum_j (1-x) x^(j-1) coef[k-j]
            new[k] = acc
        coef = new
    v = np.zeros(n); v[0] = 1.0
    for t in range(1, 150):
        ok_prod &= abs(q / 2 * v[n - 1] - coef[t]) < 1e-12
        v = Qm @ v
res["Theorem log-concavity: product formula (6.1) and (q/2)^n = prod(1-theta_j) (float, 1e-12)"] = bool(ok_prod)

json.dump(res, open(_os.path.join(_R, 'data', 'msc_rigorous_certified', 'lemma_checks.json'), "w"), indent=1)
print(json.dumps(res, indent=1))
assert all(v for v in res.values() if isinstance(v, bool))
