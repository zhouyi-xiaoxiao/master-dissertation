"""Certified (ball arithmetic, python-flint / Arb) enclosures of the constants used in note R2.

Every number printed as 'ENCLOSURE' is a rigorous enclosure: Arb balls are inclusion-monotone, and the
truncation errors of the series / integrals are bounded analytically in the comments below (and in the text).
Output: ../data/06_certified_constants.json
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import os, json, itertools, math, sys
from flint import arb, acb, ctx
ctx.prec = 200
OUT = {}
def enc(name, x, note=""):
    lo, hi = x.lower(), x.upper()
    OUT[name] = {"mid": x.mid().str(40, radius=False), "rad": x.rad().str(5, radius=False), "str": x.str(30), "note": note}
    print(f"ENCLOSURE {name:28s} = {x.str(30)}   {note}")

pi = arb.pi(); ln2 = arb(2).log(); gam = arb.const_euler()

# ---- A. sup |Psi''| on [0, pi/2],  Psi(x) = Phi(x) - 1/x = c(x) + Q(sin x) --------------------------------
#  c(x) = csc x - 1/x has a power series with positive coefficients (A&S 4.3.68), so c'' is increasing and >= 0.
#  Q(s) = -s(1+s^2-s^4)/D(s),  D(s) = 1 + (1-s^2) sqrt(1+s^2)  (cancellation-free form).
def Q12(s):
    """Q'(s), Q''(s) for a ball s in [0,1]; quotient rule on Q = -Nn/D."""
    r = (1 + s * s).sqrt()
    Nn = s * (1 + s * s - s ** 4); N1 = 1 + 3 * s * s - 5 * s ** 4; N2 = 6 * s - 20 * s ** 3
    D = 1 + (1 - s * s) * r
    D1 = -s * (1 + 3 * s * s) / r
    D2 = -(1 + 9 * s * s + 6 * s ** 4) / r ** 3
    Q1 = -(N1 * D - Nn * D1) / D ** 2
    Q2 = -(N2 * D * D - Nn * D * D2 - 2 * N1 * D * D1 + 2 * Nn * D1 * D1) / D ** 3
    return Q1, Q2
def cpp(x):          # c''(x) at a point (high precision absorbs the cancellation)
    return x.csc() * (x.csc() ** 2 + x.cot() ** 2) - 2 / x ** 3
M = 4000
half_pi = pi / 2
sup_hi = arb(0); inf_lo = arb(0)
c_prev = arb(0)                     # c''(0+) = 0  (c(x) = x/6 + 7x^3/360 + ..., c''(0) = 0)
for i in range(M):
    a = half_pi * i / M; b = half_pi * (i + 1) / M
    xb = a.union(b)                 # ball containing [a, b]
    c_next = cpp(b)
    s = xb.sin(); co = xb.cos()
    Q1, Q2 = Q12(s)
    g2 = Q2 * co * co - Q1 * s                      # (Q o sin)''
    hi = arb(c_next.upper()) + arb(g2.upper())      # c'' increasing: c''(x) <= c''(b)
    lo = arb(c_prev.lower()) + arb(g2.lower())      #                 c''(x) >= c''(a)
    sup_hi = sup_hi.max(hi) if i else hi
    inf_lo = inf_lo.min(lo) if i else lo
    c_prev = c_next
print("Psi'' certified range on [0,pi/2]: [", inf_lo.str(8), ",", sup_hi.str(8), "]")
OUT["Psi2_range"] = [float(inf_lo.lower()), float(sup_hi.upper())]
assert sup_hi < arb("2.35") and inf_lo > arb("-2.35")
print("CERTIFIED: sup |Psi''| <= 2.35 = M_2")
# the range quoted in Lemma 9.5(d) is asserted as well
assert sup_hi < arb("2.3125") and inf_lo > arb("-0.7382")
print("CERTIFIED: -0.7382 <= Psi'' <= 2.3125 on [0, pi/2]  (Lemma 9.5(d))")

# ---- B. constants of the two-dimensional theorem -----------------------------------------------------------
G14 = (arb(1) / 4).gamma()
I_Psi = -arb(1) / 2 + arb(3) / 2 * ln2 - pi.log()
Lam = -pi / 12 - G14.log() + ln2 + arb(3) / 4 * pi.log()                   # = sum 1/(k (e^{2 pi k}-1))
Lam_series = sum(1 / (arb(k) * ((2 * pi * k).exp() - 1)) for k in range(1, 40))
tailL = arb(0, 2 * math.exp(-2 * math.pi * 40))                             # sum_{k>=40} ... <= 2 e^{-80 pi}
assert (Lam_series + tailL).overlaps(Lam)
Sig = ln2 / 2 - pi / 12                                                     # = sum (-1)^{k+1}/(k sinh(pi k))
Sig_series = sum((-1) ** (k + 1) / (arb(k) * (pi * k).sinh()) for k in range(1, 60))
assert (Sig_series + arb(0, 4 * math.exp(-math.pi * 60))).overlaps(Sig)
c_tau = 8 / pi * (gam + I_Psi + 2 * Lam) - arb(2) / 3
c_mt = 4 * ln2 / pi
enc("Lambda_series", Lam, "sum_{k>=1} 1/(k(e^{2 pi k}-1)); closed form -pi/12 - ln Gamma(1/4) + ln 2 + (3/4) ln pi overlaps series")
enc("Sigma_alt", Sig, "sum_{k>=1} (-1)^{k+1}/(k sinh(pi k)) = (ln 2)/2 - pi/12 (overlaps series)")
enc("I_Psi", I_Psi, "int_0^{pi/2} (Phi(x) - 1/x) dx")
enc("c_tau_2D", c_tau, "tau_2 = (8/pi) N^2 ln N + c_tau N^2 + R")
enc("c_m_minus_tau_2D", c_mt, "4 ln 2 / pi")
enc("c_m_2D", c_tau + c_mt, "MFPT constant C_2 of the 2D corner-to-corner walk")
kap = 2 * arb(1).asinh()
enc("kappa", kap, "2 asinh(1) = ln(3 + 2 sqrt 2)")
S1 = sum(arb(k) ** 2 / (kap * k).sinh() ** 2 for k in range(1, 80))
S2 = sum(2 * arb(k) / ((2 * kap * k).exp() - 1) for k in range(1, 80))
S3 = sum(arb(k) ** 2 * (kap * k).cosh() / (kap * k).sinh() ** 2 for k in range(1, 120))
S4 = sum(arb(k) / (kap * k).sinh() for k in range(1, 120))
for nm, v in (("S1", S1), ("S2", S2), ("S3", S3), ("S4", S4)):
    enc(nm, v + arb(0, 1e-30), "auxiliary sum (tail beyond the truncation < 1e-30)")

# ---- C. lattice sums Z_d = sum_{k in Z^d \ 0} |k|^{-4}: rigorous upper bounds ----------------------------
def Z_upper(d, R):
    tot = arb(0)
    for k in itertools.combinations_with_replacement(range(R + 1), d):   # 0 <= k1 <= ... <= kd <= R
        if not any(k):
            continue
        # multiplicity: permutations * sign choices
        from collections import Counter
        cnt = Counter(k); perms = math.factorial(d)
        for v in cnt.values(): perms //= math.factorial(v)
        signs = 2 ** sum(1 for v in k if v)
        r2 = sum(v * v for v in k)
        tot += arb(perms * signs) / arb(r2) ** 2
    rho = arb(R) + arb(1) / 2
    tail = (1 + arb(d).sqrt() / (2 * (R + 1))) ** 4 * (pi / rho ** 2 if d == 2 else 4 * pi / rho)
    return tot, tot + tail
z2lo, z2hi = Z_upper(2, 400)
enc("Z_2_lower", z2lo, "partial sum |k|_inf <= 400"); enc("Z_2_upper", z2hi, "partial sum + tail bound")
Z2_closed = 4 * arb(2).zeta() * arb.const_catalan()
enc("Z_2_closed", Z2_closed, "4 zeta(2) G (Lorenz-Hardy), for comparison")
assert z2lo < Z2_closed and Z2_closed < z2hi
z3lo, z3hi = Z_upper(3, 150)
enc("Z_3_lower", z3lo, "partial sum |k|_inf <= 150"); enc("Z_3_upper", z3hi, "partial sum + tail bound")

# ---- D. three-dimensional constants ---------------------------------------------------------------------
# C_3 = int_0^inf e^{-t} (I_0(t/3)+I_1(t/3))^3 dt = 3 int_0^inf a(s)^3 ds,  a(s) = e^{-s}(I_0(s)+I_1(s)).
def a3(x, analytic):
    v = (acb(0).bessel_i(x, scaled=True) if False else None)
    return None
def integrand(x, analytic):
    i0 = x.bessel_i(0, scaled=True); i1 = x.bessel_i(1, scaled=True)
    return (i0 + i1) ** 3
try:
    test = acb(1).bessel_i(0, scaled=True)
except TypeError:
    # older signature: acb.bessel_i(self, n)  -> I_n(self); scale by hand
    def integrand(x, analytic):
        e = (-x).exp()
        return ((x.bessel_i(0) + x.bessel_i(1)) * e) ** 3
S = 10 ** 6            # split point (the tail bounds of Lemma 10.7 require S >= 200)
ctx.prec = 160
main = acb.integral(integrand, 0, S, rel_tol=arb(2) ** -100, abs_tol=arb(2) ** -100)
ctx.prec = 200
main = main.real
b = (2 / pi) ** (arb(3) / 2)
Sa = arb(S)
tail_hi = 3 * b * (2 * Sa ** (-arb(1) / 2) * (1 + arb(4) * arb(-150).exp()) - arb(1) / 4 * Sa ** (-arb(3) / 2) + arb(3) / 160 * Sa ** (-arb(5) / 2))
tail_lo = 3 * b * (2 * Sa ** (-arb(1) / 2) - arb(1) / 4 * Sa ** (-arb(3) / 2) - arb(9) / 80 * Sa ** (-arb(5) / 2))
C3 = 3 * main + tail_lo.union(tail_hi)
enc("C_3", C3, "Watson-type constant sum_{y in {0,1}^3} u(y) = lim tau_3/N^3; integral on [0,10^6] by acb.integral + analytic tail bounds")
# cross-check with a second split point, S = 200
ctx.prec = 160
main200 = acb.integral(integrand, 0, 200, rel_tol=arb(2) ** -100, abs_tol=arb(2) ** -100).real
ctx.prec = 200
S2 = arb(200)
C3_200 = 3 * main200 + (3 * b * (2 * S2 ** (-arb(1) / 2) - arb(1) / 4 * S2 ** (-arb(3) / 2) - arb(9) / 80 * S2 ** (-arb(5) / 2))).union(
    3 * b * (2 * S2 ** (-arb(1) / 2) * (1 + arb(4) * arb(-150).exp()) - arb(1) / 4 * S2 ** (-arb(3) / 2) + arb(3) / 160 * S2 ** (-arb(5) / 2)))
assert C3_200.overlaps(C3) and arb("4.3204598") < C3_200 < arb("4.3204603")
assert arb("4.320460169373") < C3 < arb("4.320460169375")
u0 = arb(6).sqrt() / (32 * pi ** 3) * (arb(1) / 24).gamma() * (arb(5) / 24).gamma() * (arb(7) / 24).gamma() * (arb(11) / 24).gamma()
enc("u0_Watson", u0, "Watson's integral u(0) (Glasser-Zucker closed form), for the remark on C_3")
# alternating lattice sum for m_3 - tau_3
Rr = 24
alt = arb(0); pos = arb(0)
for k1 in range(-Rr, Rr + 1):
    for k2 in range(-Rr, Rr + 1):
        if k1 == 0 and k2 == 0: continue
        r = arb(k1 * k1 + k2 * k2).sqrt()
        alt += (-1) ** (k1 + k2) / (r * (pi * r).sinh())
        pos += 1 / (r * ((2 * pi * r).exp() - 1))
alt += arb(0, 17 * math.exp(-math.pi * (Rr + 1)) * 1.01)
pos += arb(0, 17 * math.exp(-2 * math.pi * (Rr + 1)) * 1.01)
c_mt3 = 1 - 6 / pi * alt
enc("c_m_minus_tau_3D", c_mt3, "1 - (6/pi) sum_{k in Z^2\\0} (-1)^{k1+k2}/(|k| sinh(pi|k|))")
# E_infinity = (4/pi) sum_{Z^2\0} F(|k|) + (16/pi) sum_{k>=1} int_0^inf F(sqrt(k^2+v^2)) dv,  F(r) = 1/(r (e^{2 pi r}-1))
def Fint(k, V=12):
    f = lambda v, analytic: 1 / ((v * v + k * k).sqrt(analytic=analytic) * ((2 * acb.pi() * (v * v + k * k).sqrt(analytic=analytic)).exp() - 1))
    ctx.prec = 140
    val = acb.integral(f, 0, V, rel_tol=arb(2) ** -90, abs_tol=arb(2) ** -110).real
    ctx.prec = 200
    tail = arb(0, math.exp(-2 * math.pi * V) / (2 * math.pi * V) * 1.01)      # int_V^inf F <= e^{-2 pi V}/(2 pi V (1-e^{-2 pi V}))
    return val + tail
E2 = sum(Fint(k) for k in range(1, 9)) + arb(0, 8 * math.exp(-math.sqrt(2) * math.pi * 9))   # sum_{k>=9} <= sum e^{-sqrt2 pi k}/(k sqrt2 pi (1-e^{-2pi}))
E_inf = 4 / pi * pos + 16 / pi * E2
enc("E_inf", E_inf, "limit of (sum_k e(k) + D_N)/N")
c_tau3 = 3 / pi * (4 * gam - 4 * pi.log() + pi - 8 * ln2) - 1 + 3 * E_inf
enc("c_tau_3D", c_tau3, "lim (tau_3 - C_3 N^3)/N^2")
enc("c_m_3D", c_tau3 + c_mt3, "lim (m_3 - C_3 N^3)/N^2 (numerical estimate of the article: C_3' = -3.887)")
enc("Kr0", -2 * (1 - ln2), "K_r(0)")
enc("eps_b_limit", pi + 2 - 6 * ln2, "-J_b(0)/2")
json.dump(OUT, open(_os.path.join(_R, 'data', 'msc_rigorous_spectral', '06_certified_constants.json'), "w"), indent=1)
print("done")
