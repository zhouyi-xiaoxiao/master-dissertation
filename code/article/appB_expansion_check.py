#!/usr/bin/env python
"""appB_expansion_check.py -- numerical and symbolic check of every displayed statement of
Supplementary Section S4 ("Proof of the large-N expansion"), in the notation printed there.

  q T_N = 4N A_N - 8N E_N,   A_N = sum_{k=1}^{N-1} w(y_k),  E_N = sum_{k=1}^{N-1} w(y_k) U(v_k),
  w(y) = cos^2 y sqrt(1+sin^2 y)/sin y,   y_k = pi k/(2N),   U(v) = v/(1+v),
  v_k = (-1)^k exp(-2N arsinh(sin y_k)).

What is checked (all recorded in ../data/appB_expansion_check.json):
  A. Taylor data: r(y) = w(y) - 1/y, Acal(Y) = y w(y), Ecal(Y) = arsinh(sin y)/y - 1 (Y = y^2),
     the numbers e_{n,m}, the coefficients a_m (exact, sympy).
  B. I_r and its two pieces (50 digits).
  C. The analytic bounds used in Step 2: |v_k| <= exp(-kappa0 k), w(y_k) <= sqrt(2) N/k,
     |Ecal| <= 1/2 and the value of max|Acal| on |Y| <= delta^2 (delta = 1/2), the Cauchy
     bound on the Taylor coefficients g_{k,m} and the Taylor remainder bound.
  D. Classical inputs (50 digits): E2, E4, E6 at tau = i and tau0 = (1+i)/2; eta(i)^8 = E4(i)/12;
     prod(1 - p^j)^24 = 64 e^pi eta(i)^24; J = ln(2 pi/varpi) - pi/4; the theta_3 route.
  E. The Lambert series of Step 5 (direct summation against the closed forms).
  F. Coefficients C_2, C_0, ..., C_{-8}: (i) exact evaluation of the general formula with the
     quasi-modular rules (sympy), (ii) direct numerical evaluation of the general formula with
     numerically summed Lambert series (no modular input), (iii) the values stored by the
     main implementation (data/msc_proof/asymptotic_series.json).
  G. Remainders: N^{2M} |q T_N - expansion to order M| for N = 16 ... 4096, M = 1 ... 5, and the
     same for the two parts 4N A_N and -8N E_N separately (Steps 1 and 2).

Usage: python appB_expansion_check.py      (about one minute; < 200 MB)
"""
from __future__ import annotations
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository

import json
import os
import time

import mpmath as mp
import sympy as sp

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = _os.path.join(_R, 'data', 'article', 'appB_expansion_check.json')
ROOT = _R            # 
STORED = _os.path.join(_R, 'data', 'msc_proof', 'asymptotic_series.json')

mp.mp.dps = 60
T0 = time.time()
CHECKS = []
MMAX = 5


def record(label, statement, err, tol, extra=None):
    err = float(err)
    row = {"label": label, "statement": statement, "max_deviation": err,
           "tolerance": float(tol), "passed": bool(err <= float(tol))}
    if extra:
        row.update(extra)
    CHECKS.append(row)
    print(("PASS " if row["passed"] else "FAIL ") + f"{label:28s} dev={err:.2e}  {statement}")


# ------------------------------------------------------------------ A. Taylor data (exact)
y, Y = sp.symbols("y Y")
ORDER = 2 * MMAX + 4
w_series = sp.series(sp.cos(y) ** 2 * sp.sqrt(1 + sp.sin(y) ** 2) / sp.sin(y), y, 0, ORDER).removeO()
r_series = sp.expand(w_series - 1 / y)
A_series = sp.expand(w_series * y)
E_series = sp.expand(sp.series(sp.asinh(sp.sin(y)) / y, y, 0, ORDER).removeO() - 1)
Acal = sum(A_series.coeff(y, 2 * i) * Y ** i for i in range(0, MMAX + 1))
Ecal = sum(E_series.coeff(y, 2 * i) * Y ** i for i in range(1, MMAX + 1))
r_coeffs = [r_series.coeff(y, 2 * j - 1) for j in range(1, MMAX + 1)]
e_nm = {(n, m): sp.expand(Acal * (-Ecal) ** n / sp.factorial(n)).coeff(Y, m)
        for m in range(0, MMAX + 1) for n in range(0, m + 1)}
taylor = {
    "r(y)": [str(c) for c in r_coeffs[:3]],
    "Acal(Y)": [str(Acal.coeff(Y, i)) for i in range(0, 3)],
    "Ecal(Y)": [str(Ecal.coeff(Y, i)) for i in range(1, 3)],
    "e_nm": {f"e_{n},{m}": str(v) for (n, m), v in e_nm.items() if m <= 2},
}
expected = {"r": [sp.Rational(-1, 3), sp.Rational(-47, 90), sp.Rational(941, 1890)],
            "e": {(0, 0): 1, (0, 1): sp.Rational(-1, 3), (1, 1): sp.Rational(1, 3),
                  (0, 2): sp.Rational(-47, 90), (1, 2): sp.Rational(-5, 18), (2, 2): sp.Rational(1, 18)}}
ok = all(r_coeffs[i] == expected["r"][i] for i in range(3)) and all(e_nm[k] == v for k, v in expected["e"].items())
record("A1 Taylor data", "r = -y/3 - 47y^3/90 + 941y^5/1890; e_{0,1}=-1/3, e_{1,1}=1/3, e_{0,2}=-47/90, e_{1,2}=-5/18, e_{2,2}=1/18",
       0.0 if ok else 1.0, 0.0)

PI = sp.Symbol("pi", positive=True)
a_sym = {m: -4 * sp.bernoulli(2 * m) / sp.factorial(2 * m) * (PI / 2) ** (2 * m - 1)
         * r_coeffs[m - 1] * sp.factorial(2 * m - 1) for m in range(1, MMAX + 1)}
ok = sp.simplify(a_sym[1] - PI / 18) == 0 and sp.simplify(a_sym[2] + 47 * PI ** 3 / 21600) == 0
record("A2 a_1, a_2", "a_1 = pi/18, a_2 = -47 pi^3/21600", 0.0 if ok else 1.0, 0.0)


# ------------------------------------------------------------------ numerical functions
def w(yv):
    s = mp.sin(yv)
    return mp.cos(yv) ** 2 * mp.sqrt(1 + s * s) / s


def rfun(yv):
    return w(yv) - 1 / yv


def Acal_num(yv):          # y w(y), also for complex y
    s = mp.sin(yv)
    return yv * mp.cos(yv) ** 2 * mp.sqrt(1 + s * s) / s


def Ecal_num(yv):
    return mp.asinh(mp.sin(yv)) / yv - 1


def U(v):
    return v / (1 + v)


# ------------------------------------------------------------------ B. the integral I_r
I_r_closed = mp.mpf(3) / 2 * mp.log(2) - mp.log(mp.pi) - mp.mpf(1) / 2
I_r_quad = mp.quad(rfun, mp.linspace(0, mp.pi / 2, 9))
record("B1 I_r", "int_0^{pi/2} (w(y) - 1/y) dy = (3/2) ln 2 - ln pi - 1/2", abs(I_r_quad - I_r_closed), 1e-45)
p1 = mp.quad(lambda t: 1 / mp.sin(t) - 1 / t, mp.linspace(0, mp.pi / 2, 9))
record("B2 piece 1", "int_0^{pi/2} (1/sin y - 1/y) dy = ln(4/pi)", abs(p1 - mp.log(4 / mp.pi)), 1e-45)
p2 = mp.quad(lambda t: (mp.cos(t) ** 2 * mp.sqrt(1 + mp.sin(t) ** 2) - 1) / mp.sin(t), mp.linspace(0, mp.pi / 2, 9))
record("B3 piece 2", "int_0^{pi/2} (cos^2 y sqrt(1+sin^2 y) - 1)/sin y dy = -1/2 - (1/2) ln 2",
       abs(p2 - (-mp.mpf(1) / 2 - mp.log(2) / 2)), 1e-45)
psi32 = mp.digamma(mp.mpf(3) / 2) + mp.euler
psi12 = mp.digamma(mp.mpf(1) / 2) + mp.euler
record("B4 digamma values", "psi(3/2)+gamma = 2 - 2 ln 2, psi(1/2)+gamma = -2 ln 2",
       max(abs(psi32 - (2 - 2 * mp.log(2))), abs(psi12 + 2 * mp.log(2))), 1e-50)

# ------------------------------------------------------------------ C. bounds of Step 2
kappa0 = 2 * mp.asinh(1)
worst_v, worst_w = mp.mpf(0), mp.mpf(0)
for N in (2, 3, 5, 8, 13, 50, 200, 1001):
    for k in range(1, N):
        yk = mp.pi * k / (2 * N)
        absv = mp.exp(-2 * N * mp.asinh(mp.sin(yk)))
        worst_v = max(worst_v, absv / mp.exp(-kappa0 * k))          # must be <= 1
        worst_w = max(worst_w, w(yk) / (mp.sqrt(2) * N / k))        # must be <= 1
record("C1 |v_k| bound", "max |v_k| exp(kappa0 k) <= 1, kappa0 = 2 arsinh 1 (N = 2..1001)", max(worst_v - 1, 0), 0.0,
       {"max_ratio": float(worst_v)})
record("C2 w bound", "max w(y_k) k/(sqrt2 N) <= 1 (N = 2..1001)", max(worst_w - 1, 0), 0.0, {"max_ratio": float(worst_w)})

delta = mp.mpf(1) / 2
with mp.workdps(30):
    npts = 4000
    maxE = mp.mpf(0)
    maxA = mp.mpf(0)
    for j in range(npts):
        yy = delta * mp.expjpi(mp.mpf(2 * j) / npts)        # |y| = delta  <=>  |Y| = delta^2
        maxE = max(maxE, abs(Ecal_num(yy)))
        maxA = max(maxA, abs(Acal_num(yy)))
    analytic_E_bound = (mp.asin(mp.sinh(delta)) - delta) / delta
record("C3 |Ecal| on |Y| = 1/4", "max |arsinh(sin y)/y - 1| on |y| = 1/2 is below 1/2 (and below (arcsin(sinh(1/2)) - 1/2)/(1/2))",
       max(maxE - mp.mpf(1) / 2, 0), 0.0,
       {"max_abs_Ecal": float(maxE), "analytic_bound": float(analytic_E_bound), "max_abs_Acal": float(maxA),
        "delta": 0.5, "radius_of_analyticity_in_y": float(mp.asinh(1))})
A_delta = maxA

# derivatives D^n U, D = v d/dv
v = sp.Symbol("v")
DU = [v / (1 + v)]
for n in range(1, MMAX + 1):
    DU.append(sp.simplify(v * sp.diff(DU[-1], v)))
DU_num = [sp.lambdify(v, e, "mpmath") for e in DU]
pfrak = -mp.exp(-mp.pi)
e_num = {k: mp.mpf(sp.Rational(val).p) / mp.mpf(sp.Rational(val).q) for k, val in e_nm.items()}


def g_km(k, m):
    """Taylor coefficient [Y^m] of G_k(Y) = Acal(Y) U(p^k exp(-pi k Ecal(Y)))."""
    return mp.fsum(e_num[(n, m)] * (mp.pi * k) ** n * DU_num[n](pfrak ** k) for n in range(0, m + 1))


def G_k(k, Yv):
    yy = mp.sqrt(Yv)
    return Acal_num(yy) * U(pfrak ** k * mp.exp(-mp.pi * k * Ecal_num(yy)))


# g_{k,m} against a contour-integral (Cauchy) evaluation, and the two bounds
worst_cauchy, worst_coeff_bound, worst_rem_bound = mp.mpf(0), mp.mpf(0), mp.mpf(0)
with mp.workdps(30):
    npts = 256
    rad = delta ** 2
    for k in (1, 2, 3, 5, 8):
        vals = [G_k(k, rad * mp.expjpi(mp.mpf(2 * j) / npts)) for j in range(npts)]
        for m in range(0, MMAX + 1):
            cm = mp.fsum(vals[j] * mp.expjpi(-mp.mpf(2 * j * m) / npts) for j in range(npts)) / npts / rad ** m
            worst_cauchy = max(worst_cauchy, abs(cm - g_km(k, m)) / max(abs(g_km(k, m)), mp.mpf(10) ** (-40)))
            bound = mp.mpf(4) / 3 * A_delta * mp.exp(-mp.pi * k / 2) / rad ** m
            worst_coeff_bound = max(worst_coeff_bound, abs(g_km(k, m)) / bound)
        for M in (1, 2, 3, 4, 5):
            for Yv in (rad / 2, rad / 5, rad / 50):
                rem = abs(G_k(k, Yv) - mp.fsum(g_km(k, m) * Yv ** m for m in range(0, M + 1)))
                bound = mp.mpf(8) / 3 * A_delta * mp.exp(-mp.pi * k / 2) * (Yv / rad) ** (M + 1)
                worst_rem_bound = max(worst_rem_bound, rem / bound)
record("C4 g_{k,m} formula", "sum_n e_{n,m} (pi k)^n (D^n U)(p^k) equals the Cauchy integral of G_k (k = 1,2,3,5,8; m <= 5), relative",
       worst_cauchy, 1e-20)
record("C5 Cauchy bound", "|g_{k,m}| <= (4/3) A_delta exp(-pi k/2) delta^{-2m}", max(worst_coeff_bound - 1, 0), 0.0,
       {"max_ratio": float(worst_coeff_bound)})
record("C6 Taylor remainder bound", "|G_k(Y) - sum_{m<=M} g_{k,m} Y^m| <= (8/3) A_delta exp(-pi k/2) (Y/delta^2)^{M+1} for 0 <= Y <= delta^2/2",
       max(worst_rem_bound - 1, 0), 0.0, {"max_ratio": float(worst_rem_bound)})

# ------------------------------------------------------------------ D. classical inputs
varpi = mp.gamma(mp.mpf(1) / 4) ** 2 / (2 * mp.sqrt(2 * mp.pi))
Xi = varpi ** 4 / mp.pi ** 2
e4i = 3 * Xi / mp.pi ** 2


def eis(wt, z, terms=200):
    """E_wt as a power series in z = exp(2 pi i tau): 1 - (2 wt/B_wt) sum_n n^{wt-1} z^n/(1-z^n)."""
    B = mp.bernoulli(wt)
    return 1 - 2 * wt / B * mp.fsum(mp.mpf(n) ** (wt - 1) * z ** n / (1 - z ** n) for n in range(1, terms))


zi = mp.exp(-2 * mp.pi)          # tau = i
z0 = pfrak                       # tau0 = (1+i)/2
record("D1 E2(i)", "E_2(i) = 3/pi", abs(eis(2, zi) - 3 / mp.pi), 1e-50)
record("D2 E4(i)", "E_4(i) = 3 Gamma(1/4)^8/(2 pi)^6 = 3 Xi/pi^2",
       max(abs(eis(4, zi) - e4i), abs(e4i - 3 * mp.gamma(mp.mpf(1) / 4) ** 8 / (2 * mp.pi) ** 6)), 1e-50)
record("D3 E6(i)", "E_6(i) = 0", abs(eis(6, zi)), 1e-50)
record("D4 E2(tau0)", "E_2((1+i)/2) = 6/pi", abs(eis(2, z0) - 6 / mp.pi), 1e-50)
record("D5 E4(tau0)", "E_4((1+i)/2) = -4 E_4(i)", abs(eis(4, z0) + 4 * e4i), 1e-50)
record("D6 E6(tau0)", "E_6((1+i)/2) = 0", abs(eis(6, z0)), 1e-48)
eta_i = mp.exp(-mp.pi / 12) * mp.nprod(lambda n: 1 - mp.exp(-2 * mp.pi * n), [1, mp.inf])
record("D7 eta(i)^24", "eta(i)^24 = E_4(i)^3/1728 = Xi^3/(64 pi^6)",
       max(abs(eta_i ** 24 - e4i ** 3 / 1728), abs(eta_i ** 24 - Xi ** 3 / (64 * mp.pi ** 6))), 1e-50)
prod_p = mp.nprod(lambda j: 1 - pfrak ** j, [1, mp.inf])
record("D8 product at tau0", "prod_j (1 - p^j)^24 = 64 e^pi eta(i)^24,  p = -e^{-pi}",
       abs(prod_p ** 24 / (64 * mp.exp(mp.pi) * eta_i ** 24) - 1), 1e-50)
J_series = -2 * mp.fsum(U(pfrak ** k) / k for k in range(1, 80))
J_closed = mp.log(2 * mp.pi / varpi) - mp.pi / 4
record("D9 J", "J = -2 sum_k U(p^k)/k = ln(2 pi/varpi) - pi/4", abs(J_series - J_closed), 1e-50)
J_prod = 2 * mp.log(prod_p) - 4 * mp.log(mp.nprod(lambda j: 1 - pfrak ** (2 * j), [1, mp.inf]))
record("D10 J as products", "J = 2 ln prod(1-p^j) - 4 ln prod(1-p^{2j})", abs(J_prod - J_closed), 1e-50)
theta3 = mp.jtheta(3, 0, mp.exp(-mp.pi))
J_theta = mp.log(theta3) - 3 * mp.log(mp.nprod(lambda n: 1 - mp.exp(-2 * mp.pi * n), [1, mp.inf]))
record("D11 J via theta_3", "J = ln theta_3(e^{-pi}) - 3 ln prod(1 - e^{-2 pi n}),  theta_3(e^{-pi}) = pi^{1/4}/Gamma(3/4)",
       max(abs(J_theta - J_closed), abs(theta3 - mp.pi ** mp.mpf("0.25") / mp.gamma(mp.mpf(3) / 4))), 1e-50)

# ------------------------------------------------------------------ E. Lambert series of Step 5
KMAX = 90


def DnLambda(n, wexp):
    """(D^n Lambda_w)(p) = sum_k k^{w+n} (D^n U)(p^k)."""
    return mp.fsum(mp.mpf(k) ** (wexp + n) * DU_num[n](pfrak ** k) for k in range(1, KMAX))


lam = {"Lambda1": (DnLambda(0, 1), -mp.mpf(1) / 24),
       "DLambda1": (DnLambda(1, 1), -e4i / 36),
       "Lambda3": (DnLambda(0, 3), (1 - 6 * e4i) / 240),
       "DLambda3": (DnLambda(1, 3), -e4i / (20 * mp.pi)),
       "D2Lambda3": (DnLambda(2, 3), -e4i / (8 * mp.pi ** 2) + e4i ** 2 / 216)}
for name, (num, closed) in lam.items():
    record("E " + name, f"{name}(p) equals its closed form", abs(num - closed), 1e-50)

# ------------------------------------------------------------------ F. coefficients
E2s, E4s, E6s, eps = sp.symbols("E2 E4 E6 epsilon")
XI = sp.Symbol("Xi", positive=True)


def Dq(expr):
    return sp.expand(sp.diff(expr, E2s) * (E2s ** 2 - E4s) / 12 + sp.diff(expr, E4s) * (E2s * E4s - E6s) / 3
                     + sp.diff(expr, E6s) * (E2s * E6s - E4s ** 2) / 2)


EIS = {2: E2s, 4: E4s, 6: E6s, 8: E4s ** 2, 10: E4s * E6s}
at_tau0 = {E2s: 6 / PI, E4s: -4 * eps, E6s: 0}
at_i = {E2s: 3 / PI, E4s: eps, E6s: 0}
C_exact, b_exact = {}, {}
for m in range(1, MMAX + 1):
    Ln = sp.bernoulli(2 * m) / (4 * m) * (1 - EIS[2 * m])
    tot = 0
    for n in range(0, m + 1):
        tot += e_nm[(n, m)] * PI ** (2 * m - 1 + n) * (Ln.subs(at_tau0) - 2 ** (n + 1) * Ln.subs(at_i))
        Ln = Dq(Ln)
    b_exact[m] = sp.expand(-16 * sp.Rational(1, 4 ** m) * tot)
    C_exact[m] = sp.factor(sp.expand((a_sym[m] + b_exact[m]).subs(eps, 3 * XI / PI ** 2)))

closed_printed = {
    1: XI / 9,
    2: -PI * XI * (25 * XI + 648) / 10800,
    3: PI ** 2 * XI ** 2 * (1225 * XI + 12879) / 1905120,
    4: -PI ** 3 * XI ** 2 * (79625 * XI ** 2 + 1215000 * XI + 960498) / 435456000,
    5: PI ** 4 * XI ** 3 * (7703465 * XI ** 2 + 92129400 * XI + 118943883) / 63228211200,
}
ok = all(sp.simplify(C_exact[m] - closed_printed[m]) == 0 for m in closed_printed)
record("F1 closed forms (exact)", "general formula + quasi-modular rules reproduce the printed closed forms of C_0 ... C_{-8}",
       0.0 if ok else 1.0, 0.0, {"closed_forms": {f"C_{2 - 2 * m}": str(C_exact[m]) for m in C_exact}})
ok = sp.simplify(b_exact[1] - (-PI / 18 + PI ** 2 * eps / 27)) == 0
record("F1b b_1", "b_1 = -pi/18 + pi^2 E_4(i)/27", 0.0 if ok else 1.0, 0.0)
ok = sp.simplify(a_sym[2] + b_exact[2] - (-PI ** 3 * eps / 50 - PI ** 5 * eps ** 2 / 3888)) == 0
record("F1c C_{-2}", "C_{-2} = -pi^3 E_4(i)/50 - pi^5 E_4(i)^2/3888", 0.0 if ok else 1.0, 0.0)


def to_mp(expr):
    return mp.mpf(str(sp.N(expr.subs({PI: sp.pi, XI: sp.Float(mp.nstr(Xi, 58), 58)}), 55)))


C2_closed = 8 / mp.pi * (mp.euler + mp.log(4 * mp.sqrt(2) / varpi) - mp.mpf(1) / 2 - mp.pi / 4)
C_num_closed = {0: C2_closed}
C_num_closed.update({m: to_mp(closed_printed[m]) for m in closed_printed})

# (ii) direct numerical evaluation of the general formula, no modular input
a_num = {m: to_mp(a_sym[m]) for m in a_sym}
b_num = {m: -16 / mp.mpf(4) ** m * mp.fsum(e_num[(n, m)] * mp.pi ** (2 * m - 1 + n) * DnLambda(n, 2 * m - 1)
                                           for n in range(0, m + 1)) for m in range(0, MMAX + 1)}
C_num_formula = {0: 8 / mp.pi * (mp.euler + I_r_closed) + b_num[0]}
C_num_formula.update({m: a_num[m] + b_num[m] for m in range(1, MMAX + 1)})
dev = {m: abs(C_num_formula[m] - C_num_closed[m]) for m in C_num_closed}
record("F2 general formula (numeric)", "C_{2-2m} = a_m + b_m with numerically summed Lambert series equals the closed forms, m = 0..5",
       max(dev.values()), 1e-45, {"abs_dev": {f"C_{2 - 2 * m}": mp.nstr(dev[m], 3) for m in dev}})
record("F2b b_0", "b_0 = (8/pi) J", abs(b_num[0] - 8 / mp.pi * J_closed), 1e-50)

# (iii) stored values of the main implementation
with open(STORED) as fh:
    stored = json.load(fh)["coefficients"]
dev_st = max(abs(mp.mpf(stored[str(m)]["closed_value"]) - C_num_closed[m]) for m in range(1, MMAX + 1))
record("F3 stored values", "closed-form values agree with data/msc_proof/asymptotic_series.json (30 digits stored)",
       dev_st, 1e-27)

# ------------------------------------------------------------------ G. remainders
mp.mp.dps = 50


def parts(N):
    """A_N, E_N and q T_N from the overflow-free single sum."""
    A, E = mp.mpf(0), mp.mpf(0)
    for k in range(1, N):
        yk = mp.pi * k / (2 * N)
        wk = w(yk)
        vk = (-1) ** k * mp.exp(-2 * N * mp.asinh(mp.sin(yk)))
        A += wk
        E += wk * U(vk)
    return A, E, 4 * N * A - 8 * N * E


NS = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
rem_tot, rem_alg, rem_exp = {}, {}, {}
exact_small = {}
for N in NS:
    A, E, qT = parts(N)
    if N == 16:
        # cross-check of the split against the coth/tanh form (b) of the single sum
        tot = mp.mpf(0)
        for k in range(1, N):
            yk = mp.pi * k / (2 * N)
            t = mp.tanh(N * mp.asinh(mp.sin(yk)))
            tot += w(yk) * (1 / t if k % 2 else t)
        record("G0 split", "4N A_N - 8N E_N equals form (b) of the single sum (N = 16)", abs(4 * N * tot - qT) / qT, 1e-45)
    Nf = mp.mpf(N)
    alg = 4 * N * A - 8 * Nf ** 2 / mp.pi * (mp.log(Nf) + mp.euler + I_r_closed)
    ex = -8 * N * E - b_num[0] * Nf ** 2
    tot_res = qT - 8 * Nf ** 2 / mp.pi * mp.log(Nf) - C_num_closed[0] * Nf ** 2
    for M in range(1, MMAX + 1):
        alg -= a_num[M] * Nf ** (2 - 2 * M)
        ex -= b_num[M] * Nf ** (2 - 2 * M)
        tot_res -= C_num_closed[M] * Nf ** (2 - 2 * M)
        rem_alg.setdefault(M, {})[N] = mp.nstr(abs(alg) * Nf ** (2 * M), 6)
        rem_exp.setdefault(M, {})[N] = mp.nstr(abs(ex) * Nf ** (2 * M), 6)
        rem_tot.setdefault(M, {})[N] = mp.nstr(abs(tot_res) * Nf ** (2 * M), 6)
# Steps 1 and 2 separately and the total: N^{2M} |R_M| must converge to the modulus of the next coefficient
# (|a_{M+1}|, |b_{M+1}|, |C_{-2M}|) for M = 1..4, and must have settled for M = 5 (no further coefficient is computed here).
lim_dev = mp.mpf(0)
for table, nxt in ((rem_alg, a_num), (rem_exp, b_num), (rem_tot, C_num_closed)):
    for M in range(1, MMAX):
        lim_dev = max(lim_dev, abs(mp.mpf(table[M][4096]) - abs(nxt[M + 1])) / abs(nxt[M + 1]))
record("G1 next coefficient", "N^{2M} |R_M| at N = 4096 equals |a_{M+1}|, |b_{M+1}|, |C_{-2M}| for the algebraic part, the exponentially small part and the total (M = 1..4), relative",
       lim_dev, 1e-4, {"N": NS})
settle = max(abs(mp.mpf(t[MMAX][4096]) / mp.mpf(t[MMAX][2048]) - 1) for t in (rem_alg, rem_exp, rem_tot))
record("G2 order M = 5 bounded", "N^{10} |R_5| changes by less than the reported relative amount between N = 2048 and N = 4096 (all three)",
       settle, 1e-4, {"N^10 |R_5| total at 4096": rem_tot[MMAX][4096]})
bounded = max(max(float(t[M][N]) for N in NS) / float(t[M][4096]) for t in (rem_alg, rem_exp, rem_tot) for M in t)
record("G3 uniform bound", "sup over N = 16..4096 of N^{2M} |R_M| divided by its value at N = 4096 stays below 1.05 (M = 1..5, all three)",
       max(bounded - 1.05, 0.0), 0.0, {"max_ratio": bounded})

out = {
    "script": "code/article/appB_expansion_check.py",
    "dps": 60,
    "n_checks": len(CHECKS),
    "n_failed": sum(1 for c in CHECKS if not c["passed"]),
    "taylor": taylor,
    "constants": {"kappa0": mp.nstr(kappa0, 20), "arsinh1": mp.nstr(mp.asinh(1), 20), "varpi": mp.nstr(varpi, 40),
                  "Xi": mp.nstr(Xi, 40), "E4(i)": mp.nstr(e4i, 40), "I_r": mp.nstr(I_r_closed, 40),
                  "J": mp.nstr(J_closed, 40), "exp(-pi/2)": mp.nstr(mp.exp(-mp.pi / 2), 10)},
    "a_m": {str(m): {"exact": str(sp.nsimplify(a_sym[m])), "value": mp.nstr(a_num[m], 30)} for m in a_sym},
    "b_m": {str(m): mp.nstr(b_num[m], 30) for m in b_num},
    "coefficients": {f"C_{2 - 2 * m}": {"closed_form_value": mp.nstr(C_num_closed[m], 40),
                                        "general_formula_numeric": mp.nstr(C_num_formula[m], 40)} for m in C_num_closed},
    "scaled_remainders_total": rem_tot,
    "scaled_remainders_algebraic_part": rem_alg,
    "scaled_remainders_exponential_part": rem_exp,
    "checks": CHECKS,
    "seconds": round(time.time() - T0, 1),
}
with open(OUT, "w") as fh:
    json.dump(out, fh, indent=1)
print(f"{out['n_checks']} checks, {out['n_failed']} failed, {out['seconds']} s -> {OUT}")
