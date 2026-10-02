"""Sanity checks of the exact formulas (Theorems 2.2, 2.3 and Corollary 2.5 of note R1).

(1) spectral formula f(t) = (2q/(2N-1)) sum_m sin(th_m) sin(d th_m) lam_m^(t-1)
    against exact rational matrix powers (all starts);
(2) generating-function product F(z) = prod (1-lam_m) z / (1 - lam_m z)  (x0 = 1),
    i.e. f(n+k) = (q/2)^n h_k(lam);  prod_m (1-lam_m) = (q/2)^(N-1);
(3) MFPT = (N(N-1) - x0(x0-1))/q, variance formula for x0=1;
(4) f(t) = 0 for t < d, f(d) = (q/2)^d.
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import sys, json, os
from fractions import Fraction
import numpy as np
import mpmath as mp
sys.path.insert(0, os.path.dirname(__file__))
from lib1d import pmf_exact, spectral

mp.mp.dps = 40
out = {}
worst = 0.0
for N in [2, 3, 4, 5, 8, 13]:
    for q in [Fraction(3, 10), Fraction(1, 2), Fraction(4, 5), Fraction(1)]:
        for x0 in range(1, N):
            tmax = 60
            fe = pmf_exact(N, q, x0, tmax)
            d = N - x0
            assert all(fe[t] == 0 for t in range(1, d)), "support"
            assert fe[d] == (q / 2) ** d, "first value"
            # spectral in mp
            qq = mp.mpf(q.numerator) / q.denominator
            err = mp.mpf(0)
            for t in range(1, tmax + 1):
                s = mp.mpf(0)
                for m in range(1, N):
                    th = (2 * m - 1) * mp.pi / (2 * N - 1)
                    lam = 1 - qq * (1 - mp.cos(th))
                    s += mp.sin(th) * mp.sin(d * th) * lam ** (t - 1)
                s *= 2 * qq / (2 * N - 1)
                err = max(err, abs(s - mp.mpf(fe[t].numerator) / fe[t].denominator))
            worst = max(worst, float(err))
out["spectral_vs_exact_max_abs_err"] = worst
print("spectral vs exact rational: max abs err", worst)

# (2) product formula / complete homogeneous symmetric polynomials
worst2 = 0.0
for N in [2, 3, 5, 9, 14]:
    for q in [Fraction(3, 10), Fraction(1, 2), Fraction(4, 5), Fraction(1)]:
        n = N - 1
        fe = pmf_exact(N, q, 1, n + 40)
        qq = mp.mpf(q.numerator) / q.denominator
        lam = [1 - qq * (1 - mp.cos((2 * m - 1) * mp.pi / (2 * N - 1))) for m in range(1, N)]
        prod = mp.mpf(1)
        for l in lam:
            prod *= (1 - l)
        e1 = abs(prod - (qq / 2) ** n)
        # h_k via recursion: coefficients of prod 1/(1 - lam z)
        h = [mp.mpf(1)] + [mp.mpf(0)] * 40
        for l in lam:
            for k in range(1, 41):
                h[k] = h[k] + l * h[k - 1]
        e2 = max(abs((qq / 2) ** n * h[k] - mp.mpf(fe[n + k].numerator) / fe[n + k].denominator) for k in range(41))
        worst2 = max(worst2, float(e1), float(e2))
out["product_formula_max_abs_err"] = worst2
print("product formula: max abs err", worst2)

# (3) moments (exact rationals, by solving linear systems exactly with sympy)
import sympy as sp
ok = True
for N in [2, 3, 4, 7, 11]:
    for q in [sp.Rational(3, 10), sp.Rational(4, 5), sp.Integer(1)]:
        n = N - 1
        Q = sp.zeros(n, n)
        for j in range(n):
            Q[j, j] = 1 - q
            if j > 0:
                Q[j, j - 1] = q / 2
            if j < n - 1:
                Q[j, j + 1] = q / 2
        Q[0, 0] = 1 - q / 2
        I = sp.eye(n)
        m1 = (I - Q).LUsolve(sp.ones(n, 1))
        for x0 in range(1, N):
            ok &= sp.simplify(m1[x0 - 1] - (N * (N - 1) - x0 * (x0 - 1)) / q) == 0
        # second factorial moment: E[T(T-1)] = 2 Q (I-Q)^{-2} 1
        m2 = 2 * Q * (I - Q).LUsolve(m1)
        var = sp.simplify(m2[0] + m1[0] - m1[0] ** 2)
        # sum of geometrics: var = sum lam/(1-lam)^2 ; closed form check vs numeric
        lam = [1 - float(q) * (1 - np.cos((2 * m - 1) * np.pi / (2 * N - 1))) for m in range(1, N)]
        vnum = sum(l / (1 - l) ** 2 for l in lam)
        ok &= abs(float(var) - vnum) < 1e-9 * max(1, vnum)
out["mfpt_formula_ok"] = bool(ok)
print("MFPT (N(N-1)-x0(x0-1))/q and variance = sum lam/(1-lam)^2 :", ok)

json.dump(out, open(_os.path.join(_R, 'data', 'msc_rigorous_1d', '01_sanity.json'), "w"), indent=1)
