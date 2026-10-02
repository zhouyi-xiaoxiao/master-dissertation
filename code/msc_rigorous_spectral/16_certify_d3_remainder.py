"""Certified (Arb) constants for the explicit O(N) remainder in d = 3 (Theorem 10.1(c),(d), Lemmas 10.8-10.11).

 (1) K(1) from the closed form K(s) = (4+2s^2) K(k) - (2+2s^2) E(k), k = 1/(1+s^2)  (Lemma 10.8), compared with rigorous quadrature.
 (2) V >= int_0^pi |J_b''(theta)| d theta, with
        4 J_b''(theta) = -4(1-2s^2)(M-pi) - 40(1-s^2) B - 8(1-s^2)^2 B/s^2 + 16(1-s^2)^2 (K-E) + 16(1-2s^2) ln s + 24 - 32 s^2,
     s = sin(theta/2), M = K(s) + 4 ln s (increasing), B = 1 - (1+s^2) E + s^2(s^2+2)/(1+s^2) K (increasing), K - E decreasing in s.
     On each cell [theta_i, theta_{i+1}] the three monotone functions are enclosed by the hull of their end-point values
     (thin balls), s by the hull of the end points, and the formula is evaluated in ball arithmetic.
     On [0, theta_0] the analytic bound |4 J_b''| <= 47 + 41 ln(1/s), s >= theta/pi, is used (cross-check only; not needed for the theorem).
 (3) c_delta:  |D_N + sum_k e_N(theta_k) - E_inf N| <= c_delta / N   (Lemma 10.11),
     c_alt  :  |Alt_N - N A_inf| <= c_alt / N                         (Lemma 10.11).
 (2b) V_g >= int |g'| (Lemma 10.9(d)); (3b) delta_inf, alpha_inf (Theorem 10.1(d));
 (4) the rounded constants quoted in Theorem 10.1(c) and Corollary 11.1.
Output: ../data/16_certified_d3_remainder.json
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import os, json, math
from flint import arb, acb, ctx
ctx.prec = 256
pi = arb.pi()
HERE = os.path.dirname(__file__)
OUT = {}
def enc(name, x, note=""):
    OUT[name] = {"str": x.str(25), "lower": float(x.lower()), "upper": float(x.upper()), "note": note}
    print(f"ENCLOSURE {name:26s} = {x.str(25)}   {note}", flush=True)

def ellK(s):
    m = (1 / (1 + s * s)) ** 2
    return acb(m).elliptic_k().real
def ellE(s):
    m = (1 / (1 + s * s)) ** 2
    return acb(m).elliptic_e().real
def Kfun(s):
    return (4 + 2 * s * s) * ellK(s) - (2 + 2 * s * s) * ellE(s)
def Mfun(s):
    return Kfun(s) + 4 * s.log()
def Bfun(s):
    return 1 - (1 + s * s) * ellE(s) + s * s * (s * s + 2) / (1 + s * s) * ellK(s)
def KEfun(s):
    return ellK(s) - ellE(s)

# ---- (1) K(1), M(0+) ---------------------------------------------------------------------------------------
K1 = Kfun(arb(1))
enc("K(1) closed form", K1, "6 K(1/2) - 4 E(1/2)")
# second value of K(1) by rigorous quadrature of 4 cos^2(p) sqrt((2+sin^2 p)/(1+sin^2 p)) on [0, pi/2]  (definition (10.1) with t = sin p)
def fK1(p_, analytic):
    s2 = p_.sin() ** 2
    return 4 * p_.cos() ** 2 * ((2 + s2) / (1 + s2)).sqrt(analytic=analytic)
ctx.prec = 160
K1_quad = acb.integral(fK1, 0, acb.pi() / 2, rel_tol=arb(2) ** -110, abs_tol=arb(2) ** -110).real
ctx.prec = 256
assert K1.overlaps(K1_quad) and abs(K1 - K1_quad) < arb("1e-25")       # closed form (Lemma 10.8) agrees with the integral definition
M0 = 6 * arb(2).log() - 2
assert abs(Mfun(arb("1e-10")) - M0) < arb("1e-17")          # M(0+) = 6 ln 2 - 2 (numerical confirmation at s = 1e-10)
assert M0 - pi > arb("-0.983") and K1 - pi < arb("1.104")   # M - pi in [-0.983, 1.104] on (0,1]

# ---- (2) V ---------------------------------------------------------------------------------------------------
def four_Jb2_cell(th_lo, th_hi, cache):
    s_lo = (th_lo / 2).sin(); s_hi = (th_hi / 2).sin()
    if "lo" in cache:
        Ml, Bl, KEl = cache["lo"]
    else:
        Ml, Bl, KEl = Mfun(s_lo), Bfun(s_lo), KEfun(s_lo)
    Mh, Bh, KEh = Mfun(s_hi), Bfun(s_hi), KEfun(s_hi)
    cache["lo"] = (Mh, Bh, KEh)
    assert Ml < Mh and Bl < Bh and KEh < KEl                 # consistent with the proven monotonicity
    S = s_lo.union(s_hi); Mb = Ml.union(Mh); Bb = Bl.union(Bh); KEb = KEl.union(KEh)
    S2 = S * S
    return (-4 * (1 - 2 * S2) * (Mb - pi) - 40 * (1 - S2) * Bb - 8 * (1 - S2) ** 2 * Bb / S2 + 16 * (1 - S2) ** 2 * KEb
            + 16 * (1 - 2 * S2) * S.log() + 24 - 32 * S2)

theta0 = 1e-6
grid = []
t = theta0
while t < 0.05:
    grid.append(t); t *= 1.0004
nu = int(math.ceil((math.pi - 0.05) / 5e-5))
grid += [0.05 + (i / nu) * (3.1415 - 0.05) for i in range(nu)] + [3.1415]
cells = [(arb(grid[i]), arb(grid[i + 1])) for i in range(len(grid) - 1)]   # floats are exact dyadic numbers
cells.append((arb(3.1415), pi))                                             # last cell up to pi (ball)
V_hi = arb(0); V_lo = arb(0); cache = {}
worst_width = 0.0
for (a, b) in cells:
    e = four_Jb2_cell(a, b, cache)
    w = b - a
    V_hi += w * arb(abs(e).upper()) / 4
    V_lo += w * arb(e.abs_lower()) / 4
    worst_width = max(worst_width, float(e.rad()))
tail0 = arb(theta0) / 4 * (47 + 41 * (1 + (pi / arb(theta0)).log()))       # int_0^theta0 (47 + 41 ln(pi/theta))/4
enc("V lower (from theta_0)", V_lo, "sum of cell-wise lower bounds of |J_b''| on [theta_0, pi]")
enc("tail [0,theta_0]", tail0, "analytic bound, theta_0 = 1e-6")
V = arb((V_hi + tail0).upper())
enc("V upper", V, f"int_0^pi |J_b''| <= V  ({len(cells)} cells, max ball radius of 4J_b'' = {worst_width:.3f})")
assert V < arb("2.651")
cV = V / 4
enc("V/4", cV, "coefficient of N in the bound of |R_3|; |eps_b(N) - eps_b(inf)| <= pi V/(12 N)")
enc("pi V/12", pi * V / 12)

# ---- (2b) V_g >= int_0^pi |g'(theta)| d theta = (1/4) int_0^1 |G'(s)| ds,  g = J_b'' - 2 ln sin(theta/2) = G(s)/4 -------------
#   G'(s) = s * { -16(1-2s^2) B/s^2 + 16 (M-pi) - 40[(2-4s^2) B/s^2 - 2(1-s^2) b] + 32(1-s^2) b - 32(1-s^2) ell
#                 - (32+32s^2) ln s - 80 - 8 s^2 }  +  8 (1-s^2)^2 w(s)/s,
#   b = B/s^2 - (K-E),  e = E/((1+s^2)(2+s^2)),  ell = K - E + ln s,  w = 1 + 2b - 4e,
#   w' = (4/s)(e - b) - 4 e',   e' = 2 s (K-E)/((1+s^2)^2 (2+s^2)) - E (6s+4s^3)/((1+s^2)(2+s^2))^2.
#   Monotone in s: M, B, E increasing, K-E decreasing; w is enclosed on a cell by w(s_lo) +- (s_hi-s_lo) sup|w'|.
def point_data(sv):
    Kv = ellK(sv); Ev = ellE(sv)
    Mv = (4 + 2 * sv * sv) * Kv - (2 + 2 * sv * sv) * Ev + 4 * sv.log()
    Bv = 1 - (1 + sv * sv) * Ev + sv * sv * (sv * sv + 2) / (1 + sv * sv) * Kv
    KEv = Kv - Ev
    wv = 1 + 2 * (Bv / (sv * sv) - KEv) - 4 * Ev / ((1 + sv * sv) * (2 + sv * sv))
    return Mv, Bv, KEv, Ev, wv
s1 = 0.002
sgrid = [s1]
while sgrid[-1] < 0.3:
    sgrid.append(sgrid[-1] * 1.0001)
ns = 40000
sgrid = sgrid[:-1] + [sgrid[-2] + (1 - sgrid[-2]) * i / ns for i in range(1, ns + 1)]
Vg_hi = arb(0); Vg_lo = arb(0); prev = point_data(arb(sgrid[0])); maxG = arb(0); neg = True
for i in range(len(sgrid) - 1):
    slo = arb(sgrid[i]); shi = arb(sgrid[i + 1])
    cur = point_data(shi)
    (Ml, Bl, KEl, El, wl), (Mh, Bh, KEh, Eh, wh) = prev, cur
    assert Ml < Mh and Bl < Bh and KEh < KEl and El < Eh
    S = slo.union(shi); S2 = S * S
    Mb = Ml.union(Mh); Bb = Bl.union(Bh); KEb = KEl.union(KEh); Eb = El.union(Eh)
    Bk = Bb / S2; bb = Bk - KEb; eb = Eb / ((1 + S2) * (2 + S2)); ell = KEb + S.log()
    ep = 2 * S * KEb / ((1 + S2) ** 2 * (2 + S2)) - Eb * (6 * S + 4 * S * S2) / ((1 + S2) * (2 + S2)) ** 2
    W1 = arb(abs(4 / S * (eb - bb) - 4 * ep).upper())
    wb = wl + arb(0, 1) * (shi - slo) * W1
    assert wb.contains(wh) or wb.overlaps(wh)
    Gp = S * (-16 * (1 - 2 * S2) * Bk + 16 * (Mb - pi) - 40 * ((2 - 4 * S2) * Bk - 2 * (1 - S2) * bb) + 32 * (1 - S2) * bb - 32 * (1 - S2) * ell
              - (32 + 32 * S2) * S.log() - 80 - 8 * S2) + 8 * (1 - S2) ** 2 * wb / S
    Vg_hi += (shi - slo) * arb(abs(Gp).upper()); Vg_lo += (shi - slo) * arb(Gp.abs_lower())
    neg = neg and bool(Gp < 0)
    prev = cur
L1 = (1 / arb(s1)).log()
tailG = arb(s1) ** 2 * (arb("416.4") / 2 + arb("168.01") * (L1 / 2 + arb(1) / 4))      # int_0^{s1} s (416.4 + 168.01 ln(1/s)) ds
Vg = arb(((Vg_hi + tailG) / 4).upper())
enc("V_g lower (from s_1)", Vg_lo / 4, "cell-wise lower bound of (1/4) int_{s_1}^1 |G'|")
enc("tail [0,s_1] of int|G'|", tailG, "analytic bound, s_1 = 0.002")
enc("V_g upper", Vg, f"int_0^pi |g'| <= V_g ({len(sgrid)-1} cells); G' < 0 certified on [s_1,1]: {neg}")
G1 = -4 * (-1) * (K1 - pi) + 24 - 32            # G(1) = 4(M(1)-pi) - 8 = 4 J_b''(pi)
G0 = -4 * (M0 - pi) - 4 + 8 * (arb(3) / 2 * arb(2).log() - 1) + 24      # G(0+) = -4(M(0)-pi) - 8 b(0) + 8 ell(0) + 24
enc("G(0+)", G0, "-4(M(0)-pi) - 4 + 8(3/2 ln 2 - 1) + 24"); enc("G(1)", G1, "4(K(1)-pi) - 8")
enc("(G(0+)-G(1))/4", (G0 - G1) / 4, "= variation of g if G is monotone (consistency: must be <= V_g)")
assert (G0 - G1) / 4 < Vg and Vg < arb("6.97")
coef_g = arb(3).sqrt() * pi / 72 * Vg
enc("sqrt3 pi V_g/72", coef_g, "|(3/pi) N^2 r_b(N)| <= this"); enc("3 zeta(3)/(4 pi)", 3 * arb(3).zeta() / (4 * pi))

# ---- (3) c_delta, c_alt -----------------------------------------------------------------------------------------
kap3 = arb(2).sqrt() * arb(2).sqrt().asinh()
q = (-2 * kap3).exp()
def Q(r):        # r >= 1
    z = kap3 * r
    return 2 * pi * r * 2 / ((2 * z).exp() - 1) + 2 * pi ** 2 / 3 * r * r / z.sinh() ** 2
def Qa(r):
    z = kap3 * r
    return 2 * pi * r / z.sinh() + 2 * pi ** 2 / 3 * r * r * z.cosh() / z.sinh() ** 2
a1 = 4 * pi / (1 - q); a2 = 8 * pi ** 2 / 3 / (1 - q) ** 2                     # Q(r)  <= e^{-2 kap3 r}(a1 r + a2 r^2),  r >= 1
a3 = 4 * pi / (1 - q); a4 = 4 * pi ** 2 / 3 * (1 + q) / (1 - q) ** 2            # Qa(r) <= e^{-kap3 r}(a3 r + a4 r^2),   r >= 1
for r in (arb("1.01"), arb(2).sqrt(), arb(3), arb(7)):   # sanity only (equality at r = 1); the inequalities are proved in the text
    assert Q(r) < (-2 * kap3 * r).exp() * (a1 * r + a2 * r * r) and Qa(r) < (-kap3 * r).exp() * (a3 * r + a4 * r * r)
R = 30
SQ = arb(0); SQa = arb(0)
for i in range(0, R + 1):
    for j in range(0, R + 1):
        if i or j:
            mult = (1 if i == 0 else 2) * (1 if j == 0 else 2)
            r = arb(i * i + j * j).sqrt()
            SQ += mult * Q(r); SQa += mult * Qa(r)
# tails: sum_{|k|_inf = j > R} <= 8 j * bound(j) (bounds decreasing in r >= 1, |k| >= j); j = R+1..400 summed, j > 400: j^3 e^{-kap3 j} <= e^{-j}
tailQ = sum(8 * j * (-2 * kap3 * j).exp() * (a1 * j + a2 * j * j) for j in range(R + 1, 401)) + 8 * (a1 + a2) * arb(-400).exp() * 2
tailQa = sum(8 * j * (-kap3 * j).exp() * (a3 * j + a4 * j * j) for j in range(R + 1, 401)) + 8 * (a3 + a4) * arb(-400).exp() * 2
SQ = SQ + tailQ; SQa = SQa + tailQa
enc("sum_{Z^2\\0} Q", SQ); enc("sum_{Z^2\\0} Q_a", SQa)
# I_Q = sum_{k>=1} int_0^inf Q(sqrt(k^2+v^2)) dv <= sum_k [a1 k^2 (K_0+K_2)(z)/2 + a2 k^3 (3K_1+K_3)(z)/4], z = 2 kap3 k
IQ = arb(0)
for k in range(1, 61):
    z = 2 * kap3 * k
    IQ += a1 * k * k * (z.bessel_k(0) + z.bessel_k(2)) / 2 + a2 * k ** 3 * (3 * z.bessel_k(1) + z.bessel_k(3)) / 4
# k > 60: K_nu(z) <= e^{-z} sqrt(2 pi/z) e^{nu^2/(2z)} <= e^{-z} (z >= 197, nu <= 3)
IQ += sum((a1 * k * k + a2 * k ** 3) * (-2 * kap3 * k).exp() for k in range(61, 401)) + (a1 + a2) * arb(-400).exp() * 2
enc("I_Q", IQ, "bound for sum_k int_0^inf Q(sqrt(k^2+v^2)) dv (Bessel K)")
e2 = (-2 * pi).exp(); e4 = (-4 * pi).exp()
tail_delta = 4 * e4 * (32 / pi / ((1 - e2) * (1 - e4)) + 16 / pi * (1 / (2 * arb(2).sqrt() * (1 - e2) * (1 - e4)) + 1 / (2 * pi * (1 - e4))))
c_delta = arb((SQ / 4 + IQ + tail_delta).upper())
enc("tail_delta", tail_delta, "sup_{N>=2} N (T_1 + T_2)")
assert tail_delta < arb("1.8e-4")            # the rounded statement "N (T_1 + T_2) <= 1.8e-4" in the proof of Lemma 10.11 (line-by-line check of the note, point 3.5)
enc("c_delta", c_delta, "|D_N + sum_k e_N(theta_k) - E_inf N| <= c_delta/N")
e1 = (-pi).exp()
tail_alt = 32 / pi * 4 * e2 / ((1 - e1) * (1 - e4))
c_alt = arb((SQa / 4 + tail_alt).upper())
enc("tail_alt", tail_alt); enc("c_alt", c_alt, "|Alt_N - N A_inf| <= c_alt/N")

# ---- (3b) the limit constants delta_inf, alpha_inf of Theorem 10.1(d) ------------------------------------------------
def chi(k1, k2):
    r2 = k1 * k1 + k2 * k2; k4 = k1 ** 4 + k2 ** 4
    return r2, k4 / (6 * r2) - r2 / 2, k4 / (6 * r2) + r2 / 6
def p_fun(k1, k2):
    r2, cm, cp = chi(k1, k2); r = r2.sqrt()
    return 2 * pi / r * cm * 2 / ((2 * pi * r).exp() - 1) + 2 * pi ** 2 * cp / (pi * r).sinh() ** 2
def pa_fun(k1, k2):
    r2, cm, cp = chi(k1, k2); r = r2.sqrt()
    return 2 * pi / r * cm / (pi * r).sinh() + 2 * pi ** 2 * cp * (pi * r).cosh() / (pi * r).sinh() ** 2
qp = (-2 * pi).exp()
b1 = 2 * pi / (1 - qp); b2 = 8 * pi ** 2 / 3 / (1 - qp) ** 2          # |p(kappa)|   <= e^{-2 pi r}(b1 r + b2 r^2), r >= 1
b3 = 2 * pi / (1 - qp); b4 = 4 * pi ** 2 / 3 * (1 + qp) / (1 - qp) ** 2   # |p_a(kappa)| <= e^{-pi r}(b3 r + b4 r^2),  r >= 1
Rp = 12
dl = arb(0); al = arb(0)
for i in range(0, Rp + 1):
    for j in range(0, Rp + 1):
        if i or j:
            mult = arb((1 if i == 0 else 2) * (1 if j == 0 else 2)) / 4
            dl += mult * p_fun(arb(i), arb(j)); al += mult * (-1) ** (i + j) * pa_fun(arb(i), arb(j))
tp = sum(2 * j * (-2 * pi * j).exp() * (b1 * j + b2 * j * j) for j in range(Rp + 1, 200)) + 4 * (b1 + b2) * arb(-400).exp()
tpa = sum(2 * j * (-pi * j).exp() * (b3 * j + b4 * j * j) for j in range(Rp + 1, 200)) + 4 * (b3 + b4) * arb(-200).exp()
dl += arb(0, 1) * tp; al += arb(0, 1) * tpa          # (1/4) sum_{|k|_inf = j} <= 2 j * bound(j)
def pint(j, Vmax=14):
    f = lambda v, analytic: (lambda r2: (2 * acb.pi() / r2.sqrt(analytic=analytic) * ((j ** 4 + v ** 4) / (6 * r2) - r2 / 2) * 2 / ((2 * acb.pi() * r2.sqrt(analytic=analytic)).exp() - 1)
                                         + 2 * acb.pi() ** 2 * ((j ** 4 + v ** 4) / (6 * r2) + r2 / 6) / (acb.pi() * r2.sqrt(analytic=analytic)).sinh() ** 2))(v * v + j * j)
    ctx.prec = 160
    val = acb.integral(f, 0, Vmax, rel_tol=arb(2) ** -90, abs_tol=arb(2) ** -100).real
    ctx.prec = 256
    c = 2 * pi; Vm = arb(Vmax)          # tail: int_V^inf e^{-c v}(b1 r + b2 r^2) dv with r <= v + j, e^{-2 pi r} <= e^{-2 pi v}
    tail = (-c * Vm).exp() * (b1 * ((Vm + j) / c + 1 / c ** 2) + b2 * ((Vm + j) ** 2 / c + 2 * (Vm + j) / c ** 2 + 2 / c ** 3))
    return val + arb(0, 1) * tail
di = sum(pint(j) for j in range(1, Rp + 1))
# j > Rp: int_0^inf |p(j,v)| dv <= b1 j^2 (K_0+K_2)(2 pi j)/2 + b2 j^3 (3K_1+K_3)(2 pi j)/4 <= (b1 j^2 + b2 j^3) e^{-2 pi j}  (z >= 81)
di += arb(0, 1) * (sum((b1 * j * j + b2 * j ** 3) * (-2 * pi * j).exp() for j in range(Rp + 1, 200)) + (b1 + b2) * arb(-400).exp())
delta_inf = dl + di; alpha_inf = al
enc("delta_inf", delta_inf, "lim N delta_N = (1/4) sum p(k) + sum_j int_0^inf p(j,v) dv")
enc("alpha_inf", alpha_inf, "lim N (Alt_N - A_inf N) = (1/4) sum (-1)^{k1+k2} p_a(k)")
c0_tau3 = 1 + pi / 12 + 3 * arb(3).zeta() / (4 * pi) + 3 * delta_inf
c0_mt3 = -1 - 3 * alpha_inf
enc("c0_tau3", c0_tau3, "lim R_3(N) = 1 + pi/12 + 3 zeta(3)/(4 pi) + 3 delta_inf")
enc("c0_mtau3", c0_mt3, "lim R_3'(N) = -1 - 3 alpha_inf")
enc("C_3''", c0_tau3 + c0_mt3, "constant term of m_3")
assert arb("0.0695430676") < delta_inf < arb("0.0695430677") and arb("-0.2336942395") < alpha_inf < arb("-0.2336942394")
assert arb("1.7573985") < c0_tau3 < arb("1.7573986") and arb("-0.2989173") < c0_mt3 < arb("-0.2989172")
assert arb("1.4584812") < c0_tau3 + c0_mt3 < arb("1.4584813")
assert abs(delta_inf) < c_delta and abs(alpha_inf) < c_alt

# ---- (4) rounded constants of Theorem 10.1(c) ---------------------------------------------------------------------
M2 = arb("2.35")                                   # sup |Psi''| <= 2.35 (Lemma 9.5(d), certified in script 06)
rho_lo = 1 - 2 / pi - pi ** 2 * M2 / 8; rho_hi = 1 + pi ** 2 * M2 / 8     # rho_N in (rho_lo, rho_hi), proof of Theorem 10.1(b)
assert rho_lo > arb("-2.536") and rho_hi < arb("3.900")
z3 = 3 * arb(3).zeta() / (4 * pi)
R3_lo = z3 - coef_g - 3 * c_delta + rho_lo; R3_hi = z3 + coef_g + 3 * c_delta + rho_hi
enc("R_3 lower", R3_lo, "tau_3 - C_3 N^3 - c_tau3 N^2 >= this, all N >= 2"); enc("R_3 upper", R3_hi)
enc("3 c_alt", 3 * c_alt, "|m_3 - tau_3 - c_mtau3 N^2 + 1| <= 3 c_alt")
# weaker first-order bound (cross-check only): |eps_b(N) - eps_b(inf)| <= pi V/(12 N)
OUT["rounded"] = {}
def rounded(name, val, x, upper=True):
    assert (x < arb(val)) if upper else (x > arb(val)), name
    OUT["rounded"][name] = val
rounded("V", "2.651", V); rounded("V_g", "6.97", Vg); rounded("sqrt3piVg/72", "0.527", coef_g)
rounded("c_delta", "4.864", c_delta); rounded("c_alt", "27.13", c_alt)
assert coef_g < arb("0.527") and 3 * c_delta < arb("14.60") and 3 * c_alt < arb("81.4")
rounded("R3_lo", "-17.4", R3_lo, upper=False); rounded("R3_hi", "19.4", R3_hi)
rounded("3c_alt", "81.4", 3 * c_alt)
# constants of Corollary 11.1 (d = 3, sharp form):  Xbar = (pi^2/6)(C_3 N + c_tau3) + O(1/N)
C06 = json.load(open(_os.path.join(_R, 'data', 'msc_rigorous_spectral', '06_certified_constants.json')))
def from06(name):
    mid = arb(C06[name]["mid"]); rad = arb(C06[name]["rad"])
    return mid + arb(0, 1) * (rad * arb("1.001") + arb("1e-38") * (abs(mid) + 1))
C3 = from06("C_3"); c_tau3 = from06("c_tau_3D"); c_mt3 = from06("c_m_minus_tau_3D")
p26 = pi ** 2 / 6
enc("pi^2 C_3/6", p26 * C3); enc("pi^2 c_tau3/6", p26 * c_tau3); enc("pi^2 c_mtau3/6", p26 * c_mt3)
assert arb("7.10687211") < p26 * C3 < arb("7.10687212") and arb("-8.913634") < p26 * c_tau3 < arb("-8.913632")
assert p26 * arb("19.4") < arb("31.92") and p26 * arb("17.4") < arb("28.63") and pi ** 2 / 12 * p26 * C3 < arb("5.8452")
assert pi ** 2 / 12 * (-p26 * c_tau3) > arb("7.331") and arb("28.63") - arb("7.331") < arb("21.3")
assert arb("2.5193561") < p26 * c_mt3 < arb("2.5193562")
assert p26 * arb("82.4") < arb("135.55") and p26 * arb("82.4") + pi ** 2 / 12 * p26 * c_mt3 < arb("137.62") and p26 * arb("80.4") < arb("132.26")
p24 = pi ** 2 / 4; ln2 = arb(2).log()
assert p24 * arb("9.06") < arb("22.36") and p24 * arb("9.06") + pi ** 2 / 12 * pi * ln2 < arb("24.15") and p24 * arb("7.72") < arb("19.05")
assert abs(p24 * 4 * ln2 / pi - pi * ln2) < arb("1e-60")           # (pi^2/4) c_mtau = pi ln 2
assert arb("0.3544") < p26 * c_mt3 / (p26 * C3) < arb("0.3545")
json.dump(OUT, open(_os.path.join(_R, 'data', 'msc_rigorous_spectral', '16_certified_d3_remainder.json'), "w"), indent=1)
print("ALL CERTIFIED CHECKS OK")
