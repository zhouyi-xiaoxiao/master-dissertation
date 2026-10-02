"""Rigorous (ball-arithmetic, Arb via python-flint) building blocks.

All functions return `arb` balls that are guaranteed to contain the exact value.

Notation (note R1):  y_k = k*pi/4 (k odd),  sigma_k = (-1)^((k-1)/2),
  G^{(j)}(u) = sum_{k odd} sigma_k * 2*y_k * (-2*y_k^2)^j * exp(-2*u*y_k^2)
  (so G(u) = (pi/2) sum sigma_k k exp(-k^2 pi^2 u/8)).
  G_N^{(j)}(u) = L * sum_{m=1}^{N-1} (-1)^{m+1} sin(th_m) cos(th_m/2) (-L^2 (1-cos th_m))^j
                 * exp(-L^2 (1-cos th_m) u),   th_m = (2m-1) pi/(2N-1), L = N-1/2.
"""
from flint import arb, ctx, fmpq

ctx.prec = 256
PI = arb.pi()


def set_prec(p):
    ctx.prec = p
    global PI
    PI = arb.pi()


def _pos_tail(p, a_lower, K):
    """Upper bound (arb, nonneg) for sum_{k >= K, k odd} y_k^p exp(-2 a y_k^2) with a >= a_lower > 0,
    by a geometric majorant; requires the ratio bound r < 1 (checked)."""
    yK = K * PI / 4
    tK = yK ** p * (-2 * a_lower * yK * yK).exp()
    # t_{k+2}/t_k = (1+2/k)^p exp(-2a (y_{k+2}^2 - y_k^2)) <= (1+2/K)^p exp(-2a pi^2 (4K+4)/16)
    r = (1 + arb(2) / K) ** p * (-2 * a_lower * PI * PI * (4 * K + 4) / 16).exp()
    assert r < arb("0.5"), "tail ratio not < 1/2; increase K"
    return tK / (1 - r)


def G(j, u, K=None):
    """Enclosure of G^{(j)}(u), u an arb with u.lower() > 0."""
    u = arb(u)
    ulo = arb(u.lower())
    assert ulo > 0
    if K is None:
        # choose K so that exp(-2 u y_K^2) is far below precision
        import math
        K = int(4 / math.pi * math.sqrt(ctx.prec * 0.6931 / (2 * float(ulo)))) + 9
        K += (K + 1) % 2  # make odd
    s = arb(0)
    for k in range(1, K, 2):
        y = k * PI / 4
        sg = 1 if ((k - 1) // 2) % 2 == 0 else -1
        s += sg * 2 * y * (-2 * y * y) ** j * (-2 * u * y * y).exp()
    tail = 2 ** (j + 1) * _pos_tail(2 * j + 1, ulo, K)
    return s + arb(0, tail.upper())


def GN(j, u, N):
    """Enclosure of G_N^{(j)}(u) (finite lattice sum)."""
    u = arb(u)
    L = arb(2 * N - 1) / 2
    s = arb(0)
    for m in range(1, N):
        th = (2 * m - 1) * PI / (2 * N - 1)
        sg = 1 if (m - 1) % 2 == 0 else -1
        mu = 1 - th.cos()
        s += sg * L * th.sin() * (th / 2).cos() * (-L * L * mu) ** j * (-L * L * mu * u).exp()
    return s


def hull(a, b):
    return arb(a).union(arb(b))


def interval(lo, hi):
    """Ball containing [lo, hi]."""
    lo = arb(lo); hi = arb(hi)
    return lo.union(hi)


def up(x):
    """Upper bound of |x| as an arb (exact point)."""
    return arb(arb(x).abs_upper())
