"""Numerical sanity checks (not proofs) for the general-start statements of Section 8 (Prop. 8.11, Thm 8.12):
 (1) the lattice identities  (L^2/q) f_{x0}(t) = S^st_0(t),  (L^4/q^2)(f_{x0}(t+1)-f_{x0}(t)) = S^st_1(t)  against exact rational PMFs;
 (2) the elementary bounds for Ast_j = S^(2j+1) cos(sqrt zeta):  -alpha1 <= Ast_j' <= 0,  0 <= Ast_j'' <= alpha2  on [0, pi^2/4];
 (3) Proposition 8.11: the true remainder R = L^4 (S^st_1 - G_xi' - H/L^2) (50-digit evaluation) lies in the certified ball
     of start.R_enclosure_start, for several (N, x0, v, q), continuous (q = 0) and discrete time;
 (4) Lemma 8.2 (zero counts): number of sign changes of g_{x0}^{(i)} on a fine grid is <= i (i = 1, 2, 3) for sample (N, x0);
 (5) Theorem 8.3(2),(3): G_xi' has one sign change, G_xi'' two, G_xi''' three on a grid, G_xi''(c(xi)) < 0  (sample xi);
     c''(0) = -2 (finite differences).
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import os, sys, json, random
from fractions import Fraction as Fr
sys.path.insert(0, os.path.dirname(__file__))
import mpmath as mp
import numpy as np
import core, start
from core import arb, interval
from lib1d import pmf_exact

core.set_prec(160)
mp.mp.dps = 50
random.seed(7)
res = {}


def sig(k):
    return 1 if ((k - 1) // 2) % 2 == 0 else -1


def lattice_sum(j, N, x0, v, q, t=None):
    """S^st_j: q = 0 -> continuous time with scaled time v; q > 0, t integer -> discrete (v ignored)."""
    L = mp.mpf(N) - mp.mpf(1) / 2
    xi = (x0 - mp.mpf(1) / 2) / L
    s = mp.mpf(0)
    for m in range(1, N):
        k = 2 * m - 1
        y = k * mp.pi / 4; x = y / L
        S = mp.sin(x) / x
        A = S ** (2 * j + 1) * mp.cos(x)
        if t is None:
            if q == 0:
                e = mp.e ** (-2 * v * y * y * S * S)
            else:
                w = 2 * q * mp.sin(x) ** 2
                e = mp.e ** (-2 * v * y * y * S * S * (-mp.log(1 - w) / w))
        else:
            e = (1 - 2 * q * mp.sin(x) ** 2) ** (t - 1)
        s += sig(k) * mp.cos(2 * y * xi) * 2 * y * (-2 * y * y) ** j * A * e
    return s


def Gx(j, u, xi):
    s = mp.mpf(0)
    for k in range(1, 800, 2):
        y = k * mp.pi / 4
        e = mp.e ** (-2 * u * y * y)
        s += sig(k) * 2 * y * mp.cos(2 * y * xi) * (-2 * y * y) ** j * e
        if k > 30 and e * y ** (2 * j + 1) < mp.mpf(10) ** -60:
            break
    return s


# (1)
worst = mp.mpf(0)
for N, x0, q in [(5, 2, Fr(1, 2)), (7, 3, Fr(1, 3)), (9, 1, Fr(4, 5)), (8, 5, Fr(1, 4)), (6, 4, Fr(1))]:
    L = mp.mpf(N) - mp.mpf(1) / 2
    f = pmf_exact(N, q, x0, 60)
    qq = mp.mpf(q.numerator) / q.denominator
    for t in range(1, 59):
        ft = mp.mpf(f[t].numerator) / f[t].denominator
        ft1 = mp.mpf(f[t + 1].numerator) / f[t + 1].denominator
        worst = max(worst, abs(L ** 2 / qq * ft - lattice_sum(0, N, x0, None, qq, t)),
                    abs(L ** 4 / qq ** 2 * (ft1 - ft) - lattice_sum(1, N, x0, None, qq, t)))
res["identities_max_abs_err"] = float(worst)
print("(1) lattice identities, max abs err:", mp.nstr(worst, 3))

# (2)
viol = mp.mpf("-1")
S_ = lambda z: mp.sin(mp.sqrt(z)) / mp.sqrt(z)
for j, (a1, a2) in {0: (mp.mpf(2) / 3, mp.mpf(4) / 15), 1: (mp.mpf(1), mp.mpf(4) / 5)}.items():
    A = lambda z: S_(z) ** (2 * j + 1) * mp.cos(mp.sqrt(z))
    for i in range(1, 201):
        z = mp.mpf(i) / 200 * mp.pi ** 2 / 4 * mp.mpf("0.9999")
        d1 = mp.diff(A, z); d2 = mp.diff(A, z, 2)
        viol = max(viol, d1, -d1 - a1, -d2, d2 - a2, A(z) - 1, -A(z))
    viol = max(viol, abs(mp.diff(A, mp.mpf("1e-12")) + mp.mpf(j + 2) / 3) - mp.mpf("1e-9"))
res["Ast_bounds_max_violation"] = float(viol)
print("(2) bounds for Ast_j, max violation (<= ~1e-20 required):", mp.nstr(viol, 3))

# (3)
ok3 = True
rows = []
for N0, k0 in [(61, 27), (101, 31)]:
    L0 = arb(2 * N0 - 1) / 2
    for N in [N0, N0 + 37, 3 * N0]:
        L = mp.mpf(N) - mp.mpf(1) / 2
        for frac in [0.0, 0.13, 0.31, 0.5]:
            x0 = max(1, min(int((2 * N + 1) // 4), int(round(frac * float(L) + 0.5))))
            xi = (x0 - mp.mpf(1) / 2) / L
            cx = mp.mpf(start.c_float(float(xi)))
            for q in [mp.mpf(0), mp.mpf("0.3"), mp.mpf("0.5")]:
                for v in [cx * mp.mpf("0.97"), cx, cx * mp.mpf("1.04")]:
                    S1 = lattice_sum(1, N, x0, v, q)
                    H = mp.mpf(1) / 2 * Gx(2, v, xi) + v / 6 * (1 - 3 * q) * Gx(3, v, xi)
                    Rt = L ** 4 * (S1 - Gx(1, v, xi) - H / L ** 2)
                    Xi = arb(2 * x0 - 1) / (2 * N - 1)
                    qa = arb(mp.nstr(q, 20))
                    Renc, Bnd, _ = start.R_enclosure_start(1, arb(mp.nstr(v, 40)), Xi, L0, k0, q=qa, qbar=qa if q > 0 else 0)
                    inside = bool(Renc.contains(arb(mp.nstr(Rt, 45))))
                    ok3 &= inside
                    rows.append(dict(N0=N0, N=N, x0=x0, q=float(q), v=float(v), R_true=float(Rt), encl=Renc.str(8), inside=inside))
res["prop86_all_inside"] = ok3; res["prop86_rows"] = rows
print("(3) Proposition 8.11: true remainder inside the certified ball in all", len(rows), "cases:", ok3)
print("    e.g.", rows[5], rows[-1])


# (4)
def gder(i, N, x0, us):
    """G_{N,x0}^{(i)} on a numpy grid (float64 is enough for counting sign changes away from underflow)."""
    L = N - 0.5; xi = (x0 - 0.5) / L
    m = np.arange(1, N); k = 2 * m - 1; y = k * np.pi / 4; x = y / L
    sg = np.where(((k - 1) // 2) % 2 == 0, 1.0, -1.0)
    S = np.sin(x) / x
    out = np.zeros_like(us)
    for kk in range(len(k)):
        out += sg[kk] * 2 * y[kk] * S[kk] * np.cos(x[kk]) * np.cos(2 * y[kk] * xi) * (-2 * y[kk] ** 2 * S[kk] ** 2) ** i * np.exp(-2 * us * y[kk] ** 2 * S[kk] ** 2)
    return out


def nsign(a, tol):
    s = np.sign(np.where(np.abs(a) < tol, 0, a)); s = s[s != 0]
    return int(np.sum(s[1:] != s[:-1]))


ok4 = True
for N, x0 in [(12, 1), (12, 5), (30, 8), (30, 20), (60, 15), (60, 45)]:
    L = N - 0.5; xi = (x0 - 0.5) / L
    cx = start.c_float(xi)
    us = np.linspace(cx * 0.25, cx * 6, 20001)
    for i in (1, 2, 3):
        a = gder(i, N, x0, us)
        ok4 &= nsign(a, 1e-9 * np.max(np.abs(a))) <= i
res["zero_counts_ok"] = bool(ok4)
print("(4) Lemma 8.2 (sign changes of g^(i) <= i on a grid):", ok4)

# (5)
ok5 = True
for xif in [0.0, 0.2, 0.37, 0.5, 0.8]:
    cx = start.c_float(xif)
    us = np.geomspace(cx * 0.05, cx * 40, 40001)
    cnt = []
    for i in (1, 2, 3):
        a = np.array([start._Gf(i, u, xif) for u in us])
        cnt.append(nsign(a, 1e-12 * np.max(np.abs(a))))
    ok5 &= (cnt == [1, 2, 3]) and start._Gf(2, cx, xif) < 0
h = 1e-3
c0, c1, c2 = start.c_float(0.0), start.c_float(h), start.c_float(2 * h)
cpp = (2 * c0 - 5 * c1 + 4 * c2 - start.c_float(3 * h)) / h ** 2
res["Gxi_sign_patterns_ok"] = bool(ok5); res["c_second_derivative_at_0"] = cpp
print("(5) sign changes of G_xi', G_xi'', G_xi''' = 1, 2, 3 and G_xi''(c) < 0:", ok5, ";  c''(0) ~", round(cpp, 4), "(exact: -2)")
json.dump(res, open(_os.path.join(_R, 'data', 'msc_rigorous_1d', '27_start_sanity.json'), "w"), indent=1)
