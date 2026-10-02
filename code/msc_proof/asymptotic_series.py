"""
Closed-form coefficients of the large-N expansion to arbitrary order (Theorem 4.5 of the article, Supplementary Section S4).

  q T_N = (8/pi) N^2 ln N + C_2 N^2 + sum_{m>=1} C_{2-2m} N^{2-2m},

  C_{2-2m} = a_m + b_m,
  a_m = -4 B_{2m}/(2m)! (pi/2)^{2m-1} r^{(2m-1)}(0)                 (Euler-Maclaurin, lower endpoint)
  b_m = -16 4^{-m} sum_{n=0}^{m} kappa_{n,m} pi^{2m-1+n} (theta^n Lambda_{2m-1})(p),   p = -e^{-pi},
  Lambda_{2m-1}(p) = sum_k k^{2m-1} p^k/(1+p^k) = S_{2m-1}(p) - 2 S_{2m-1}(p^2),
  S_{2m-1} = (B_{2m}/(4m)) (1 - E_{2m})   (Eisenstein series), theta = p d/dp,
  kappa_{n,m} = [Y^m] phi(Y) (-Omega(Y))^n / n!,
  phi(y^2) = y cos^2 y sqrt(1+sin^2 y)/sin y,  Omega(y^2) = asinh(sin y)/y - 1.

The quasi-modular evaluation uses E_2(i)=3/pi, E_6(i)=0, E_4(i)=e:=3 varpi^4/pi^4 and, at
tau_0=(1+i)/2:  E_2=6/pi, E_4=-4e, E_6=0, together with Ramanujan's differential system.
Everything is exact (sympy rationals); the results are polynomials in pi and
X = varpi^4/pi^2. They are then compared with (i) the direct numerical series and
(ii) Richardson-extrapolated residuals of the exact single sum.

Output: results/asymptotic_series.json
"""
from __future__ import annotations
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository

import json
import os

import mpmath as mp
import sympy as sp

import mfpt2d as M

HERE = os.path.dirname(os.path.abspath(__file__))
RES = _os.path.join(_R, 'data', 'msc_proof')
MMAX = 5

y, Y = sp.symbols("y Y")
E2, E4, E6 = sp.symbols("E2 E4 E6")
PI, e, X = sp.symbols("pi e X", positive=True)

ORDER = 2 * MMAX + 4
f_series = sp.series(sp.cos(y) ** 2 * sp.sqrt(1 + sp.sin(y) ** 2) / sp.sin(y), y, 0, ORDER).removeO()
r_series = sp.expand(f_series - 1 / y)
phi_series = sp.expand(f_series * y)
omega_series = sp.expand(sp.series(sp.asinh(sp.sin(y)) / y, y, 0, ORDER).removeO() - 1)
phiY = sum(phi_series.coeff(y, 2 * i) * Y ** i for i in range(0, MMAX + 1))
OmY = sum(omega_series.coeff(y, 2 * i) * Y ** i for i in range(1, MMAX + 1))


def theta(expr):
    """Ramanujan: theta E2 = (E2^2-E4)/12, theta E4 = (E2 E4 - E6)/3, theta E6 = (E2 E6 - E4^2)/2."""
    return sp.expand(sp.diff(expr, E2) * (E2 ** 2 - E4) / 12 + sp.diff(expr, E4) * (E2 * E4 - E6) / 3
                     + sp.diff(expr, E6) * (E2 * E6 - E4 ** 2) / 2)


def eisenstein(w):
    """E_w as a polynomial in E4, E6 (E2 itself for w=2)."""
    if w == 2:
        return E2
    table = {4: E4, 6: E6, 8: E4 ** 2, 10: E4 * E6,
             12: (441 * E4 ** 3 + 250 * E6 ** 2) / sp.Integer(691),
             14: E4 ** 2 * E6}
    return table[w]


at_tau0 = {E2: 6 / PI, E4: -4 * e, E6: 0}
at_i = {E2: 3 / PI, E4: e, E6: 0}
coeffs = {}
for m in range(1, MMAX + 1):
    B2m = sp.bernoulli(2 * m)
    r_der = r_series.coeff(y, 2 * m - 1) * sp.factorial(2 * m - 1)
    a_m = -4 * B2m / sp.factorial(2 * m) * (PI / 2) ** (2 * m - 1) * r_der
    S = B2m / (4 * m) * (1 - eisenstein(2 * m))
    b_m = 0
    Sn = S
    for n in range(0, m + 1):
        kappa = sp.expand(phiY * (-OmY) ** n / sp.factorial(n)).coeff(Y, m)
        lam = Sn.subs(at_tau0) - 2 ** (n + 1) * Sn.subs(at_i)
        b_m += kappa * PI ** (2 * m - 1 + n) * lam
        Sn = theta(Sn)
    b_m = -16 * sp.Rational(1, 4 ** m) * b_m
    C = sp.expand((a_m + b_m).subs(e, 3 * X / PI ** 2))
    coeffs[m] = sp.collect(sp.factor_terms(C), X)
    print(f"C_{{{2 - 2 * m}}} =", sp.simplify(C))

# ---------------------------------------------------------------- numerical comparison
mp.mp.dps = 60
g14 = mp.gamma(mp.mpf(1) / 4)
varpi = g14 ** 2 / (2 * mp.sqrt(2 * mp.pi))
Xn = varpi ** 4 / mp.pi ** 2
num = {m: mp.mpf(sp.N(coeffs[m].subs({PI: sp.pi, X: sp.Float(str(Xn), 60)}), 55)) for m in coeffs}
C2 = 8 / mp.pi * (mp.euler + mp.log(4 * mp.sqrt(2) / varpi) - mp.mpf(1) / 2 - mp.pi / 4)

# exact values on N = 2^j and successive Richardson extrapolation of the next unknown coefficient
Ns = [2 ** j for j in range(6, 15)]
Tv = {N: M.clean_single_mp(N, 1, 60) for N in Ns}
report = {}
for mtop in range(1, MMAX + 1):
    # residual after subtracting everything up to (but excluding) order m = mtop
    seq = []
    for N in Ns:
        res = Tv[N] - 8 / mp.pi * N ** 2 * mp.log(N) - C2 * N ** 2
        for m in range(1, mtop):
            res -= num[m] * mp.mpf(N) ** (2 - 2 * m)
        seq.append(res * mp.mpf(N) ** (2 * mtop - 2))
    # Richardson table in powers of N^-2 (ratio 4)
    tab = seq[:]
    lvl = 1
    while len(tab) > 1:
        tab = [(4 ** lvl * tab[i + 1] - tab[i]) / (4 ** lvl - 1) for i in range(len(tab) - 1)]
        lvl += 1
    est = tab[0]
    report[mtop] = dict(order=f"N^{2 - 2 * mtop}", closed_form=str(coeffs[mtop]),
                        closed_value=mp.nstr(num[mtop], 30), richardson_estimate=mp.nstr(est, 30),
                        abs_diff=mp.nstr(abs(est - num[mtop]), 3))
    print(f"m={mtop}: closed {mp.nstr(num[mtop], 25)}  Richardson {mp.nstr(est, 25)}  diff {mp.nstr(abs(est - num[mtop]), 3)}")

with open(_os.path.join(_R, 'data', 'msc_proof', 'asymptotic_series.json'), "w") as f:
    json.dump(dict(X="varpi^4/pi^2", varpi=mp.nstr(varpi, 40), coefficients=report,
                   resistance_coefficients={f"c_{2 * m}": mp.nstr(num[m] / 2, 25) for m in num}), f, indent=1)
