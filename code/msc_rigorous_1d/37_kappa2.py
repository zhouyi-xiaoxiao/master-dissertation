"""Theorem 5.7: the limit kappa_2 = lim_N L^4 (u_N - c + kappa/L^2) in closed form, and its enclosure.

Proof of Theorem 5.7 (existence of the limit):  kappa_2 = -Gamma_0 / G''(c) with
   Gamma_0 = (1/2) G'''(c) kappa^2 - kappa H_1'(c) + R_0,   H_1'(c) = (11/12) G'''(c) + (c/6) G''''(c),
   R_0 = sum_{k odd} sigma_k (-2) y_k^7 F_1''(0; y_k, c)
       = (113/480) G'''(c) + (49 c/360) G''''(c) + (c^2/72) G^(5)(c),
because F_1(zeta) = A_1(zeta) exp(-2 c y^2 beta(zeta)) has F_1''(0) = [113/60 - (49/45) a + a^2/9] e^{-a}, a = 2 c y^2
(A_1(0) = 1, A_1'(0) = -3/2, A_1''(0) = 113/60, beta'(0) = -1/3, beta''(0) = 4/45).

(1) Ball enclosure of kappa_2 (Arb, 256 bits; c from data/03_constants.json with radius 1e-70; series tails by
    Lemma 4.1) and comparison with the enclosure [-1.5309330, -1.5308464] of data/12_cont_theorem.json (row 2001).
(2) Symbolic check (sympy) of the Taylor coefficients A_1(0), A_1'(0), A_1''(0), beta'(0), beta''(0) and of F_1''(0).
(3) Numerical sanity (mpmath, 40 digits): s_N = L^4 (u_N - c + kappa/L^2) for N = 501, 1001, 2001, 4001, 8001, where
    u_N is the zero of G_N' (finite sum), and L^2 (s_N - kappa_2).
Output: data/37_kappa2.json; log: logs/37_kappa2.log.
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import os, sys, json
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
import core
from core import arb, G, interval
import sympy as sp
import mpmath as mp

core.set_prec(256)
const = json.load(open(_os.path.join(_R, 'data', 'msc_rigorous_1d', '03_constants.json')))
cm = arb(const["c_mid_80_digits"])
c = interval(cm - arb("1e-70"), cm + arb("1e-70"))
G2, G3, G4, G5 = G(2, c), G(3, c), G(4, c), G(5, c)
kappa = arb(3) / 4 + c / 6 * G3 / G2
H1p = arb(11) / 12 * G3 + c / 6 * G4
R0 = arb(113) / 480 * G3 + 49 * c / 360 * G4 + c * c / 72 * G5
Gamma0 = G3 * kappa * kappa / 2 - kappa * H1p + R0
kappa2 = -Gamma0 / G2
d12 = json.load(open(_os.path.join(_R, 'data', 'msc_rigorous_1d', '12_cont_theorem.json')))["2001"]
inside = bool(kappa2 > arb(d12["kappa2_lower"]) and kappa2 < arb(d12["kappa2_upper"]))
print("G^(5)(c) =", G5.str(30))
print("R_0      =", R0.str(30))
print("Gamma_0  =", Gamma0.str(30))
print("kappa_2  =", kappa2.str(30))
print("inside the enclosure of data/12_cont_theorem.json (N0 = 2001):", inside)

# (2) symbolic Taylor coefficients
z, a = sp.symbols("zeta a", positive=True)
x = sp.sqrt(z)
S = sp.sin(x) / x
A1 = S ** 3 * sp.cos(x) ** 2
beta = S ** 2
ser = lambda e: sp.series(e, z, 0, 3).removeO()
A1s, bs = sp.expand(ser(A1)), sp.expand(ser(beta))
coef = lambda e, i: sp.nsimplify(e.coeff(z, i))
chk = dict(A1_0=coef(A1s, 0), A1p_0=coef(A1s, 1), A1pp_0=2 * coef(A1s, 2), betap_0=coef(bs, 1), betapp_0=2 * coef(bs, 2))
F = sp.expand(ser(A1s * sp.exp(-a * (bs - 1))))           # F_1 e^{a}, with a = 2 c y^2
Fpp0 = sp.expand(2 * F.coeff(z, 2))
sym_ok = (chk == dict(A1_0=1, A1p_0=sp.Rational(-3, 2), A1pp_0=sp.Rational(113, 60), betap_0=sp.Rational(-1, 3),
                      betapp_0=sp.Rational(4, 45))
          and sp.simplify(Fpp0 - (sp.Rational(113, 60) - sp.Rational(49, 45) * a + a ** 2 / 9)) == 0)
print("(2) Taylor coefficients", chk, " F_1''(0) e^a =", Fpp0, " ok:", sym_ok)

# (3) numerical convergence
mp.mp.dps = 40
cmp_, k2 = mp.mpf(const["c_mid_80_digits"]), mp.mpf(kappa2.mid().str(30, radius=False))
kap = mp.mpf(kappa.mid().str(30, radius=False))


def GNp(u, N):
    L = mp.mpf(2 * N - 1) / 2
    s = mp.mpf(0)
    for m in range(1, N):
        th = (2 * m - 1) * mp.pi / (2 * N - 1)
        mu = 1 - mp.cos(th)
        s += (1 if m % 2 else -1) * L * mp.sin(th) * mp.cos(th / 2) * (-L * L * mu) * mp.exp(-L * L * mu * u)
    return s


rows = []
for N in [501, 1001, 2001, 4001, 8001]:
    L = mp.mpf(2 * N - 1) / 2
    u0 = cmp_ - kap / L ** 2
    uN = mp.findroot(lambda u: GNp(u, N), u0)
    sN = L ** 4 * (uN - cmp_ + kap / L ** 2)
    rows.append(dict(N=N, s_N=mp.nstr(sN, 12), L2_times_diff=mp.nstr(L ** 2 * (sN - k2), 8)))
    print("(3)", rows[-1], flush=True)
json.dump(dict(kappa2=kappa2.str(30), Gamma0=Gamma0.str(30), R0=R0.str(30), G5c=G5.str(30),
               inside_12_enclosure=inside, symbolic_ok=bool(sym_ok), convergence=rows),
          open(_os.path.join(_R, 'data', 'msc_rigorous_1d', '37_kappa2.json'), "w"), indent=1)
print("ALL OK:", inside and bool(sym_ok))
