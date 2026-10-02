"""start.py -- rigorous (Arb) building blocks for Section 8 (general start), on top of core.py.

Notation (note R1, Section 8):  xi = (x0 - 1/2)/L in [0, 1),  y_k = k pi/4 (k odd), sigma_k = (-1)^((k-1)/2),
   G_xi^(j)(u) = sum_{k odd} sigma_k 2 y_k cos(2 y_k xi) (-2 y_k^2)^j exp(-2 u y_k^2),
   lattice sum   S^st_j = sum_{k odd <= 2N-3} sigma_k cos(2 y_k xi) 2 y_k (-2 y_k^2)^j Ast_j(zeta_k) exp(-2 v y_k^2 B(zeta_k; q)),
                 Ast_j(zeta) = S(zeta)^(2j+1) cos(sqrt zeta),   B = beta Lam(2 q zeta beta)   (q = 0: continuous time),
   expansion (Proposition 8.11):  S^st_j = G_xi^(j)(v) + H_j(v)/L^2 + R/L^4,
                 H_j = (j+2)/6 G_xi^(j+1) + (v/6)(1-3q) G_xi^(j+2),
   R in  sum_{k<=k0} sigma_k cos(2 y_k xi) (-2)^j y_k^(2j+5) Fst_j''(Z_k) + [-Bnd, Bnd],  Z_k = [0, y_k^2/L0^2],
   Bnd = B_low + B_high + B_tail with the constants  alpha1 = (j+2)/3,  alpha2 = 4/15 (j=0), 4/5 (j=1).
"""
import math
from flint import arb, ctx
import core
from core import ub, interval, sigma, pos_tail, pos_sum, J2, S_jet, Lam_derivs, B_consts, BETA_STAR, chi_rate

ALPHA1 = {0: arb(2) / 3, 1: arb(1)}
ALPHA2 = {0: arb(4) / 15, 1: arb(4) / 5}


def Gxi(j, u, xi, K=None):
    """ball for G_xi^(j)(u); u, xi balls, u.lower() > 0."""
    PI = arb.pi()
    u = arb(u); xi = arb(xi)
    ulo = arb(u.lower())
    assert ulo > 0
    if K is None:
        K = int(4 / math.pi * math.sqrt(ctx.prec * 0.6931 / (2 * float(ulo)))) + 9
        K = max(K, int(2 * 0.9 / (float(ulo) * math.pi ** 2)) + 11)      # makes the tail ratio of Lemma 4.1 < 1/2
        K += (K + 1) % 2
    s = arb(0)
    for k in range(1, K, 2):
        y = k * PI / 4
        s += sigma(k) * 2 * y * (2 * y * xi).cos() * (-2 * y * y) ** j * (-2 * u * y * y).exp()
    tail = 2 ** (j + 1) * pos_tail(2 * j + 1, ulo, K)
    return s + arb(0, tail.upper())


def cos_jet(Z, M=40):
    """C1(zeta) = cos(sqrt zeta) = sum (-1)^n zeta^n/(2n)!  and its first two zeta-derivatives over the ball Z, |Z| < 10."""
    Z = arb(Z)
    zu = ub(Z)
    assert zu < 10
    s0 = arb(0); s1 = arb(0); s2 = arb(0)
    fact = arb(1)
    for n in range(0, M + 1):
        if n > 0:
            fact = fact * (2 * n - 1) * (2 * n)
        sg = 1 if n % 2 == 0 else -1
        s0 += sg * Z ** n / fact
        if n >= 1:
            s1 += sg * n * Z ** (n - 1) / fact
        if n >= 2:
            s2 += sg * n * (n - 1) * Z ** (n - 2) / fact
    n = M + 1
    fact = fact * (2 * n - 1) * (2 * n)
    # for n > M >= 40 consecutive terms (also of the derivative series) decrease by a factor < 1/2
    t0 = 2 * zu ** n / fact
    t1 = 2 * n * zu ** (n - 1) / fact
    t2 = 2 * n * (n - 1) * zu ** (n - 2) / fact
    return J2(s0 + arb(0, t0.upper()), s1 + arb(0, t1.upper()), s2 + arb(0, t2.upper()))


def F_jet_start(j, Z, y, v, q=None):
    """jet (F, F', F'') in zeta of Fst_j(zeta; y, v, q) = S^(2j+1) cos(sqrt zeta) exp(-2 v y^2 B(zeta; q)) over the ball Z."""
    S = S_jet(Z)
    zeta = J2(Z, 1, 0)
    beta = S * S
    A = S
    for _ in range(2 * j):
        A = A * S
    A = A * cos_jet(Z)
    if q is None:
        B = beta
    else:
        w = zeta * beta * (2 * arb(q))
        l0, l1, l2 = Lam_derivs(w.a)
        B = beta * w.compose(l0, l1, l2)
    E = (B * (-2 * arb(v) * arb(y) * arb(y))).exp()
    return A * E


def low_tail_bound(j, vlo, vhi, k0, qbar=0):
    b0, b1, b2 = B_consts(qbar)
    vhi = ub(vhi)
    al1, al2 = ALPHA1[j], ALPHA2[j]
    poly = [al2, 2 * vhi * (2 * al1 * b1 + b2), 4 * vhi * vhi * b1 * b1]
    return ub(2 ** j * pos_sum(2 * j + 5, arb(vlo) * BETA_STAR, k0 + 2, poly))


def high_bound(j, vlo, L0, qbar=0):
    PI = arb.pi()
    vlo = arb(arb(vlo).lower()); L0 = arb(L0)
    chi = chi_rate(qbar)
    assert L0 * L0 > (2 * j + 6) / (2 * chi * vlo), "L0 too small for the monotone high-mode bound"
    return ub(2 ** (j + 1) * (PI / 2) ** (2 * j + 1) * L0 ** (2 * j + 6) * (-chi * vlo * L0 * L0).exp())


def cont_tail_bound(j, vlo, vhi, L0, qbar=0):
    PI = arb.pi()
    b0, b1, b2 = B_consts(qbar)
    Kst = int(math.floor(2 * float(arb(L0).lower()) / 3)) + 1
    while arb(Kst) > 2 * arb(L0) / 3 and Kst > 1:
        Kst -= 1
    Kst = max(1, Kst)
    c4 = (6 / PI) ** 4
    c2 = (6 / PI) ** 2
    poly = [c4 + c2 * ALPHA1[j], c2 * 2 * ub(vhi) * b0]
    return ub(2 ** (j + 1) * pos_sum(2 * j + 5, vlo, Kst, poly))


def R_enclosure_start(j, V, Xi, L0, k0, q=None, qbar=0):
    """Ball containing R = L^4 [S^st_j - G_xi^(j)(v) - H_j(v)/L^2] for every L >= L0, v in V, xi in Xi
    (q None / qbar 0: continuous time; else q in the ball q, q <= qbar <= 4/5)."""
    PI = arb.pi()
    V = arb(V); L0 = arb(L0); Xi = arb(Xi)
    assert k0 % 2 == 1 and arb(3 * k0) <= 2 * L0, "need k0 <= 2 L0/3"
    R = arb(0)
    for k in range(1, k0 + 1, 2):
        y = k * PI / 4
        Z = interval(arb(0), y * y / (L0 * L0))
        F = F_jet_start(j, Z, y, V, q)
        R += sigma(k) * (2 * y * Xi).cos() * (-2) ** j * y ** (2 * j + 5) * F.c
    vlo, vhi = arb(V.lower()), arb(V.upper())
    bl = low_tail_bound(j, vlo, vhi, k0, qbar)
    bh = high_bound(j, vlo, L0, qbar)
    bt = cont_tail_bound(j, vlo, vhi, L0, qbar)
    Bnd = ub(bl + bh + bt)
    return R + arb(0, Bnd.upper()), Bnd, (bl, bh, bt)


# ---------------------------------------------------------------------------------------------
# enclosure of c(xi) over a ball Xi (uses Theorem 8.3(2): G_xi' has exactly one zero, + before, - after)
# ---------------------------------------------------------------------------------------------
import numpy as np
_K = np.arange(1, 600, 2)
_Y = _K * np.pi / 4
_SG = np.where(((_K - 1) // 2) % 2 == 0, 1.0, -1.0)


def _Gf(j, u, xi):
    return float(np.sum(_SG * 2 * _Y * np.cos(2 * _Y * xi) * (-2 * _Y * _Y) ** j * np.exp(-2 * u * _Y * _Y)))


def c_float(xi, guess=None):
    """float approximation of c(xi) (no rigour)."""
    from scipy.optimize import brentq
    if guess is not None:
        u = guess
        for _ in range(50):
            du = _Gf(1, u, xi) / _Gf(2, u, xi)
            u -= du
            if abs(du) < 1e-15 * u:
                return u
    u = (1 - xi) ** 2 / 3 * 0.6
    prev = _Gf(1, u, xi)
    while True:
        u2 = u * 1.05
        cur = _Gf(1, u2, xi)
        if prev > 0 and cur <= 0:
            break
        u = u2; prev = cur
    return brentq(lambda w_: _Gf(1, w_, xi), u, u2, xtol=1e-16, rtol=1e-15)


def c_enclosure(Xi, guess=None):
    """ball C with c(xi) in C for every xi in the ball Xi: certified sign change G_xi'(C_lo) > 0 > G_xi'(C_hi) for all
    xi in Xi (then c(xi) in (C_lo, C_hi) by Theorem 8.3(2)).  For an exact xi (zero radius) the midpoint is first refined
    by Newton steps in ball arithmetic, so that the enclosure has radius about 2^(-prec/2)."""
    Xi = arb(Xi)
    xm = float(Xi.mid())
    rad = float(Xi.rad())
    cm = c_float(abs(xm), guess)
    cmid = arb(repr(cm))
    if rad < 1e-25:                       # exact (rational) xi: refine the midpoint by Newton steps in ball arithmetic
        for _ in range(8):
            cmid = arb((cmid - Gxi(1, cmid, Xi) / Gxi(2, cmid, Xi)).mid())
        r = 2.0 ** (-ctx.prec // 2)
    else:
        h = 1e-6
        slope = abs(c_float(abs(xm) + h, cm) - cm) / h           # |c'(xi)| (float estimate) -> first guess of the radius
        r = 1.3 * slope * rad + 4 * rad * rad + 1e-13
    for _ in range(80):
        rr = arb(repr(r))
        if (Gxi(1, cmid - rr, Xi) > 0) and (Gxi(1, cmid + rr, Xi) < 0):
            return interval(cmid - rr, cmid + rr), cm
        r *= 1.6
    raise RuntimeError("c enclosure failed for xi ~ %g" % xm)
