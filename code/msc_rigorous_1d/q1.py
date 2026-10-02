"""q1.py -- rigorous building blocks for Section 7 (the simple random walk, q = 1).

Notation of Section 7:  yh_k = k*pi/2 (all integers k >= 1), x_k = yh_k/L, zeta_k = x_k^2, tau_k = (-1)^(k/2) (k even),
   cal E^(j)(v) = sum_{k even >= 2} tau_k (yh_k^2/2) (-yh_k^2/2)^j exp(-v yh_k^2/2),
   M(zeta) = -2 ln cos(sqrt zeta)/zeta,  low range: zeta <= pi^2/9 (cos x >= 1/2).
   L^2 f(t)            = P(t)  -+ E(t)/L       (t = D mod 2;  upper sign: D = N-1 "near", lower: D = N "far"),
   L^4 (f(t+2)-f(t))   = P1(t) -+ E1(t)/L,
   P  = G(v)   + R_P0/L^2,   E  = calE(v)  + R_E0/L^2,
   P1 = 2 G'(v) + [(5/2) G''(v) - (2v/3) G'''(v)]/L^2 + R_P/L^4,   E1 = 2 calE'(v) + R_E/L^2,   v = (t-1)/L^2.
Remainder bounds (Lemma 7.3):  |R_*| <= rho_* = low + high + tail  (functions below).
"""
import math
from flint import arb, ctx
import core
from core import ub, interval

M1 = None
M2 = None


def consts():
    """m1 = M'(pi^2/9), m2 = M''(pi^2/9) (closed forms)."""
    PI = arb.pi()
    x = PI / 3
    m1 = (x * x.tan() + 2 * x.cos().log()) / x ** 4
    m2 = (x * x / x.cos() ** 2 - 5 * x * x.tan() - 8 * x.cos().log()) / (2 * x ** 6)
    return m1, m2


def calE(j, v, K=None):
    """ball for cal E^(j)(v), v ball with positive lower bound."""
    PI = arb.pi()
    v = arb(v)
    vlo = arb(v.lower())
    assert vlo > 0
    if K is None:
        K = int(2 / math.pi * math.sqrt(2 * ctx.prec * 0.6931 / float(vlo))) + 10
        K += K % 2
    s = arb(0)
    for k in range(2, K, 2):
        yh = k * PI / 2
        e = yh * yh / 2
        s += (1 if (k // 2) % 2 == 0 else -1) * e * (-e) ** j * (-v * e).exp()
    tail = psum(2 * j + 2, vlo, K, [arb(1)]) / 2 ** (j + 1)
    return s + arb(0, tail.upper())


def psum(p, a, kstart, poly):
    """certified upper bound of  sum_{k = kstart, kstart+2, ...} yh_k^p * (sum_i poly[i] yh_k^(2i)) * exp(-a yh_k^2/2),
    yh_k = k pi/2, a > 0, poly[i] >= 0."""
    PI = arb.pi()
    a = arb(arb(a).lower())
    assert a > 0
    pmax = p + 2 * (len(poly) - 1)
    K = kstart
    while True:
        r = (1 + arb(2) / K) ** pmax * (-a * PI * PI * (K + 1) / 2).exp()
        if r < arb("0.5"):
            break
        K += 2
    s = arb(0)
    for k in range(kstart, K, 2):
        yh = k * PI / 2
        pol = arb(0)
        for i, ci in enumerate(poly):
            pol += ub(ci) * yh ** (2 * i)
        s += yh ** p * pol * (-a * yh * yh / 2).exp()
    for i, ci in enumerate(poly):
        yK = K * PI / 2
        tK = yK ** (p + 2 * i) * (-a * yK * yK / 2).exp()
        r = (1 + arb(2) / K) ** (p + 2 * i) * (-a * PI * PI * (K + 1) / 2).exp()
        s += ub(ci) * ub(tK / (1 - r))
    return ub(s)


def kstart_tail(L0, parity):
    """smallest k of the given parity that can exceed 2L/3 for some L >= L0 (a superset start)."""
    K = int(math.floor(2 * float(arb(L0).lower()) / 3))
    K = max(K, 1)
    if K % 2 != parity:
        K += 1
    if parity == 0:
        K = max(K, 2)
    return K


def high(p, cnt_pow, vlo, L0, extra):
    """L^extra * L^cnt_pow * (pi L/2)^p * 2^(-vlo L^2) at L = L0 (decreasing in L beyond the checked threshold)."""
    PI = arb.pi()
    L0 = arb(L0); vlo = arb(arb(vlo).lower())
    tot = extra + cnt_pow + p
    assert L0 * L0 > tot / (2 * vlo * arb(2).log()), "L0 too small for monotone high-mode bound"
    return ub((PI / 2) ** p * L0 ** tot * (-vlo * L0 * L0 * arb(2).log()).exp())


def remainders(vlo, vhi, L0):
    """(rho_P, rho_E, rho_P0, rho_E0): certified bounds of |R_P|, |R_E|, |R_P0|, |R_E0| for all L >= L0, v in [vlo, vhi]."""
    PI = arb.pi()
    m1, m2 = consts()
    vlo = arb(arb(vlo).lower()); vhi = ub(vhi)
    h = vhi / 2                       # a^+ = h * yh^2
    c2 = (3 / PI) ** 2
    c4 = (3 / PI) ** 4
    A2 = arb(111) / 320
    # P1, second order:  low: (yh^7/2)[A2 + 2 a (5/8) m1 + a^2 m1^2 + a m2] e^{-a^-}
    low = psum(7, vlo, 1, [A2, h * (arb(5) / 4 * m1 + m2), h * h * m1 * m1]) / 2
    tail = psum(7, vlo, kstart_tail(L0, 1), [c4 + c2 * arb(5) / 8, c2 * h / 6])
    rho_P = ub(low + high(3, 1, vlo, L0, 4) + tail)
    # E1, first order: low: (yh^6/2)(13/24 + a m1) e^{-a^-};  tail: (3/pi)^2 (yh^6/2);  high: L^2 * L * (pi L/2)^4/2
    low = psum(6, vlo, 2, [arb(13) / 24, h * m1]) / 2
    tail = psum(6, vlo, kstart_tail(L0, 0), [c2]) / 2
    rho_E = ub(low + high(4, 1, vlo, L0, 2) / 2 + tail)
    # P, first order: low: yh^3 (7/24 + a m1) e^{-a^-}; tail: (3/pi)^2 yh^3; high: L^2 * L * (pi L/2)
    low = psum(3, vlo, 1, [arb(7) / 24, h * m1])
    tail = psum(3, vlo, kstart_tail(L0, 1), [c2])
    rho_P0 = ub(low + high(1, 1, vlo, L0, 2) + tail)
    # E, first order: low: (yh^4/2)(5/24 + a m1) e^{-a^-}; tail: (3/pi)^2 yh^4/2; high: L^2 * L * (pi L/2)^2/2
    low = psum(4, vlo, 2, [arb(5) / 24, h * m1]) / 2
    tail = psum(4, vlo, kstart_tail(L0, 0), [c2]) / 2
    rho_E0 = ub(low + high(2, 1, vlo, L0, 2) / 2 + tail)
    return rho_P, rho_E, rho_P0, rho_E0
