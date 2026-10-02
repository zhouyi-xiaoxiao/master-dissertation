"""check_packed.py -- numerical sanity checks of the lemmas used by verify_packed.py
(none of this is part of a proof; it guards against a mis-stated lemma or a mis-coded mask).

 1. Lemma 'packed integer arithmetic' (a): Box.apply_M against a direct site-by-site application of the
    model rules, on random vectors, d = 1, 2, 3, several N, field widths and all nine (d, q).
 2. Lemma (b): Box.less_everywhere against a direct comparison, including equal and adjacent entries.
 3. Lemma (c): packed reciprocal division against entrywise floor(z R / 2^K).
 4. Lemma 'one vector and a scalar error bound': C_t <= 2^P v_t <= C_t + b_t against exact rationals,
    for P = 12, 24, 40 and K = P + 8, over 3 tail times.
 5. Proposition 'a tail time always exists': lambda_1 simple and dominant, v_{t+1}/v_t -> lambda_1 (float).
 6. The check in exact mode against verify_exact.py (two separately written exact programs), small N,
    five values of q.
Output: certs/packed_checks.json
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import json, os, random, sys
from fractions import Fraction
from itertools import product
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import verify_packed as vp
import verify_exact as ve

random.seed(20261001)
res = {}
QS = ((4, 5), (1, 2), (1, 1))


def direct_M(z, d, N, cm, cs):
    """M z from the model rules; z: dict site -> int with z[target] = 0."""
    a = (N - 1,) * d
    out = {y: 0 for y in z}
    for y, val in z.items():
        if y == a or val == 0:
            continue
        out[y] += cs * val
        for ax in range(d):
            for sgn in (-1, 1):
                yy = list(y)
                yy[ax] += sgn
                dest = tuple(yy) if 0 <= yy[ax] < N else y        # cancelled move
                out[dest] += cm * val
    out[a] = 0
    return out


def sites(d, N):        # in the order of the packed index i(y) = sum y_k N^(k-1)
    return [tuple((i // N ** ax) % N for ax in range(d)) for i in range(N ** d)]


# ---- 1. stencil
ok, n = True, 0
for d, Ns in ((1, (2, 3, 4, 7, 12)), (2, (2, 3, 4, 6)), (3, (2, 3, 4))):
    for N in Ns:
        st = sites(d, N)
        for qn, qd in QS:
            cm, cs, D = vp.coeffs(d, qn, qd)
            for wb in (2, 5, 30):
                box = vp.Box(d, N, wb)
                top = ((1 << box.w) - 1) // D
                for trial in range(6):
                    vals = [random.choice((0, 1, top, random.randrange(top + 1))) for _ in st]
                    vals[-1] = 0
                    got = box.unpack(box.apply_M(box.pack(vals), cm, cs))
                    want = direct_M(dict(zip(st, vals)), d, N, cm, cs)
                    ok = ok and got == [want[y] for y in st]
                    n += 1
res["Lemma packed (a): stencil equals the model rules"] = dict(ok=bool(ok), cases=n)

# ---- 2. comparison
ok, n = True, 0
for d, N in ((1, 5), (2, 3), (2, 4), (3, 3)):
    for wb in (2, 4, 30):
        box = vp.Box(d, N, wb)
        for trial in range(400):
            slack = random.choice((0, 1, 2, 7, 100))
            bound = (1 << (box.w - 1)) - slack - 2
            base = [random.randrange(bound + 1) for _ in range(box.n)]
            A = [max(0, min(bound, x + random.choice((-slack - 2, -slack - 1, -slack, 0, 1, -bound)))) for x in base]
            A[-1] = base[-1] = 0
            got = box.less_everywhere(box.pack(A), box.pack(base), slack, bound)
            want = all(a + slack < b for a, b in zip(A[:-1], base[:-1]))
            ok = ok and got == want
            n += 1
res["Lemma packed (b): comparison at every site"] = dict(ok=bool(ok), cases=n)

# ---- 3. reciprocal division
ok, n = True, 0
for wb, K in ((4, 12), (30, 120), (9, 40)):
    for D in (2, 4, 5, 6, 8, 12, 15):
        R = (1 << K) // D
        box = vp.Box(2, 4, wb)
        low = ((1 << (box.w * box.n)) - 1) // ((1 << box.w) - 1) * ((1 << (box.w - K)) - 1)
        zmax = ((1 << box.w) - 1) // R
        for trial in range(100):
            z = [random.choice((0, 1, zmax, random.randrange(zmax + 1))) for _ in range(box.n)]
            got = box.unpack(((box.pack(z) * R) >> K) & low)
            ok = ok and got == [(x * R) >> K for x in z]
            n += 1
res["Lemma packed (c): division by reciprocal multiplication"] = dict(ok=bool(ok), cases=n)

# ---- 4. one vector and a scalar error bound, against exact rationals
ok, n, worst = True, 0, 0
for d, N in ((1, 6), (1, 11), (2, 4), (2, 6), (3, 3)):
    st = sites(d, N)
    for qn, qd in QS:
        cm, cs, D = vp.coeffs(d, qn, qd)
        T = ve.exact_pmf(d, N, Fraction(qn, qd))[1]
        for P in (12, 24, 40):
            K = P + 8
            R = (1 << K) // D
            box = vp.Box(d, N, (P + K) // 8 + 1)
            low = ((1 << (box.w * box.n)) - 1) // ((1 << box.w) - 1) * ((1 << (box.w - K)) - 1)
            v = {y: Fraction(0) for y in st}
            v[st[0]] = Fraction(1)
            for ts in (1, 3):
                vv = dict(v)
                for _ in range(ts - 1):
                    w_ = direct_M({y: x for y, x in vv.items()}, d, N, cm, cs)
                    vv = {y: Fraction(x, D) for y, x in w_.items()}
                C = box.pack([(vv[y] * 2 ** P).__floor__() for y in st])
                b = 1
                for t in range(ts, 3 * T + 5):
                    cv = box.unpack(C)
                    for y, c in zip(st[:-1], cv[:-1]):
                        ok = ok and 0 <= c <= vv[y] * 2 ** P <= c + b and c <= 2 ** P
                        worst = max(worst, float(vv[y] * 2 ** P - c) / b)
                    C, b = ((box.apply_M(C, cm, cs) * R) >> K) & low, b + 2
                    w_ = direct_M(vv, d, N, cm, cs)
                    vv = {y: Fraction(x, D) for y, x in w_.items()}
                    n += 1
res["Lemma single: C_t <= 2^P v_t <= C_t + b_t (exact rationals)"] = dict(ok=bool(ok), steps=n, largest_fraction_of_b_used=worst)

# ---- 5. Proposition termination (float)
import numpy as np
ok, rows = True, []
for d, N, (qn, qd) in ((1, 8, (4, 5)), (1, 8, (1, 1)), (2, 5, (1, 1)), (2, 5, (4, 5)), (3, 3, (1, 1)), (3, 4, (1, 2))):
    st = sites(d, N)[:-1]
    idx = {y: i for i, y in enumerate(st)}
    cm, cs, D = vp.coeffs(d, qn, qd)
    Qm = np.zeros((len(st), len(st)))
    for y in st:
        e = {yy: 0 for yy in sites(d, N)}
        e[y] = 1
        for yy, val in direct_M(e, d, N, cm, cs).items():
            if val:
                Qm[idx[yy], idx[y]] = val / D
    ev, vec = np.linalg.eigh(Qm)
    lam1, lam2 = ev[-1], max(abs(ev[0]), abs(ev[-2]))
    phi = vec[:, -1] * np.sign(vec[0, -1])
    v = np.zeros(len(st))
    v[0] = 1.0
    for _ in range(int(40 / np.log(lam1 / lam2)) + 50):
        v = Qm @ v
        v /= v.sum()
    ratio = (Qm @ v) / v
    good = bool(np.allclose(Qm, Qm.T) and lam2 < lam1 < 1 and phi.min() > 0 and np.allclose(ratio, lam1, rtol=1e-9))
    ok = ok and good
    rows.append(dict(d=d, N=N, q=f"{qn}/{qd}", lambda1=float(lam1), second_modulus=float(lam2), ok=good))
res["Proposition termination: lambda_1 simple and dominant, ratio -> lambda_1 (float)"] = dict(ok=bool(ok), cases=rows)

# ---- 6. two separately written exact programs
ok, n = True, 0
for d, Nmax in ((1, 12), (2, 6), (3, 4)):
    for qn, qd in ((4, 5), (1, 2), (1, 1), (2, 3), (9, 10)):
        for N in range(2, Nmax + 1):
            f, T = ve.exact_pmf(d, N, Fraction(qn, qd))
            t0, argmax, uni, nmax = ve.analyse(f, T)
            c = vp.certify(d, N, qn, qd, exact_only=True)
            ok = ok and (c["t0"], c["maximisers_of_lower_bound"], c["T_tail"], c["unimodal_certified"],
                         c["certified_local_maxima_upto_T"]) == (t0, argmax, T, uni, nmax)
            r = vp.certify(d, N, qn, qd)
            ok = ok and r["t0"] == t0 and r["mode"] in argmax and r["T_tail"] >= T
            ok = ok and (not r["mode_certified"] or argmax == [r["mode"]]) and (not r["unimodal_certified"] or uni)
            n += 1
res["verify_packed (exact and rounded) against verify_exact.py"] = dict(ok=bool(ok), cases=n)

# ---- 7. Theorem 'the constant kappa_1': numerical sanity of each step of the proof (mpmath, 40 digits)
from mpmath import mp, mpf, exp, pi, nsum, inf, diff, findroot, log
mp.dps = 40
phi = lambda tau: nsum(lambda m: (-1) ** (int(m) + 1) * (2 * m - 1) * exp(-(2 * m - 1) ** 2 * pi ** 2 * tau / 4), [1, inf])
dphi = lambda tau: nsum(lambda m: (-1) ** int(m) * (2 * m - 1) ** 3 * pi ** 2 / 4 * exp(-(2 * m - 1) ** 2 * pi ** 2 * tau / 4), [1, inf])
bn = lambda n, tau: (2 * n + 1) * (mpf((2 * n + 1) ** 2) / 4 - 3 * tau / 2) * exp(-mpf(n * (n + 1)) / tau)
Bf = lambda tau: mpf(1) / 4 - 3 * tau / 2 - sum((-1) ** (n + 1) * bn(n, tau) for n in range(1, 12))
ok = True
for tau in (mpf(1) / 12, mpf(1) / 8, mpf(1) / 6, mpf(1) / 5, mpf(1) / 4):            # identity of Step 3
    ok = ok and abs(pi ** mpf(1.5) * tau ** mpf(3.5) * exp(1 / (4 * tau)) * dphi(tau) - Bf(tau)) < mpf(10) ** -30
for n in (1, 2, 3, 4):                                                              # bound on |b_n'| on (0, 1/4]
    for j in range(1, 51):
        tau = mpf(j) / 200
        ok = ok and abs(diff(lambda x: bn(n, x), tau)) <= mpf(441) / 2 * n ** 5 * exp(-4 * n * (n + 1)) * (1 + mpf(10) ** -20)
grid = [mpf(j) / 400 for j in range(8, 1200)]                                        # exactly one sign change of phi'
signs = [dphi(t) > 0 for t in grid]
changes = sum(1 for a, b in zip(signs, signs[1:]) if a != b)
root = findroot(dphi, mpf(1) / 6)
cj = json.load(open(_os.path.join(_R, 'data', 'msc_rigorous_certified', 'constants.json')))
lo, hi = (Fraction(int(x.split("/")[0]), int(x.split("/")[1])) for x in (cj["tau_star"]["lo"], cj["tau_star"]["hi"]))
inside = mpf(lo.numerator) / lo.denominator - mpf(10) ** -30 < root < mpf(hi.numerator) / hi.denominator + mpf(10) ** -30
tau1 = log(27) / (2 * pi ** 2)
ok = ok and changes == 1 and signs[0] and not signs[-1] and bool(inside) and tau1 < mpf(1) / 4
res["Theorem tau: Step-3 identity, bound on |b_n'|, single sign change of phi', root inside the certified enclosure (mpmath)"] = dict(
    ok=bool(ok), sign_changes_on_grid=changes, root=str(root), two_tau_star=str(2 * root))

res["ALL OK"] = all(v["ok"] for v in res.values() if isinstance(v, dict))
json.dump(res, open(_os.path.join(_R, 'data', 'msc_rigorous_certified', 'packed_checks.json'), "w"), indent=1)
print(json.dumps(res, indent=1))
sys.exit(0 if res["ALL OK"] else 1)
