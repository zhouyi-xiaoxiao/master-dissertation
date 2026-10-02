"""core.py -- rigorous (Arb ball arithmetic, python-flint) building blocks for Sections 4-7 of note R1.

Every function returns balls that contain the exact value, or certified upper bounds.
The formulas implemented here are EXACTLY those of Lemmas 5.1-5.3 and Proposition 5.4 ("lattice expansion") of note R1:

  y_k = k*pi/4 (k odd), sigma_k = (-1)^((k-1)/2), x_k = y_k/L, zeta_k = x_k^2, L = N - 1/2.
  S(zeta) = sin(sqrt zeta)/sqrt zeta,  C(zeta) = cos^2(sqrt zeta) = 1 - zeta S^2,  beta = S^2,
  A_j = S^(2j+1) C  (j = 0, 1, 2),
  Lam(w) = -ln(1-w)/w,   B(zeta; q) = beta * Lam(2 q zeta beta)   (q = None: continuous time, B = beta),
  F_j(zeta; y, v, q) = A_j(zeta) exp(-2 v y^2 B(zeta; q)).

  G^(j)(u) = sum_{k odd} sigma_k 2 y_k (-2 y_k^2)^j exp(-2 u y_k^2).

  lattice sum  S_j = sum_{k odd <= 2N-3} sigma_k 2 y_k (-2 y_k^2)^j A_j(zeta_k) lambda_k^(t-1)
               (continuous time: exp(-2 v y_k^2 beta(zeta_k)) instead of lambda_k^(t-1)),
  expansion    S_j = G^(j)(v) + H_j(v)/L^2 + R/L^4,
               H_j = (7+2j)/12 G^(j+1) + (v/6)(1-3q) G^(j+2)     (q = 0 in continuous time),
  R in  sum_{k<=k0} sigma_k (-2)^j y_k^(2j+5) F_j''(Z_k) + [-Bnd, Bnd],   Z_k = [0, y_k^2/L0^2],
  Bnd = B_low + B_high + B_tail   (Proposition 5.4).
"""
import math
from flint import arb, ctx

ctx.prec = 200
PI = arb.pi()


def set_prec(p):
    global PI
    ctx.prec = p
    PI = arb.pi()


def ub(x):
    """exact upper bound of |x| as a point ball"""
    return arb(arb(x).abs_upper())


def interval(lo, hi):
    return arb(lo).union(arb(hi))


def sigma(k):
    return 1 if ((k - 1) // 2) % 2 == 0 else -1


# ---------------------------------------------------------------------------------------------
# positive tails:  sum_{k >= K, k odd} y_k^p exp(-2 a y_k^2)  <=  t_K/(1-r)      (Lemma 4.1)
# ---------------------------------------------------------------------------------------------
def pos_tail(p, a, K):
    a = arb(arb(a).lower())
    assert a > 0 and K % 2 == 1
    yK = K * PI / 4
    tK = yK ** p * (-2 * a * yK * yK).exp()
    r = (1 + arb(2) / K) ** p * (-a * PI * PI * (K + 1) / 2).exp()
    assert r < arb("0.5"), "tail ratio not < 1/2; increase K"
    return ub(tK / (1 - r))


def pos_sum(p, a, Kstart, poly=None):
    """certified upper bound of sum_{k >= Kstart, k odd} y_k^p exp(-2 a y_k^2) * poly(y_k^2),
    poly = list of nonnegative coefficients c_i of (y^2)^i  (default [1])."""
    if poly is None:
        poly = [arb(1)]
    a = arb(arb(a).lower())
    if Kstart % 2 == 0:
        Kstart += 1
    K = Kstart
    # find K where the ratio test holds for the highest power
    pmax = p + 2 * (len(poly) - 1)
    while True:
        r = (1 + arb(2) / K) ** pmax * (-a * PI * PI * (K + 1) / 2).exp()
        if r < arb("0.5"):
            break
        K += 2
    s = arb(0)
    for k in range(Kstart, K, 2):
        y = k * PI / 4
        pol = arb(0)
        for i, ci in enumerate(poly):
            pol += ub(ci) * y ** (2 * i)
        s += y ** p * (-2 * a * y * y).exp() * pol
    for i, ci in enumerate(poly):
        s += ub(ci) * pos_tail(p + 2 * i, a, K)
    return ub(s)


# ---------------------------------------------------------------------------------------------
# the limiting functions
# ---------------------------------------------------------------------------------------------
def G(j, u, K=None):
    """ball for G^(j)(u), u a ball with positive lower bound."""
    u = arb(u)
    ulo = arb(u.lower())
    assert ulo > 0
    if K is None:
        K = int(4 / math.pi * math.sqrt(ctx.prec * 0.6931 / (2 * float(ulo)))) + 9
        K += (K + 1) % 2
    s = arb(0)
    for k in range(1, K, 2):
        y = k * PI / 4
        s += sigma(k) * 2 * y * (-2 * y * y) ** j * (-2 * u * y * y).exp()
    tail = 2 ** (j + 1) * pos_tail(2 * j + 1, ulo, K)
    return s + arb(0, tail.upper())


def GN(j, u, N):
    """ball for the continuous-time lattice function G_N^(j)(u) (finite sum, eq. (10) of note R1)."""
    u = arb(u)
    L = arb(2 * N - 1) / 2
    s = arb(0)
    for m in range(1, N):
        th = (2 * m - 1) * PI / (2 * N - 1)
        mu = 1 - th.cos()
        s += (1 if m % 2 == 1 else -1) * L * th.sin() * (th / 2).cos() * (-L * L * mu) ** j * (-L * L * mu * u).exp()
    return s


# ---------------------------------------------------------------------------------------------
# second-order jets in zeta (value, d/dzeta, d^2/dzeta^2), enclosures over a whole ball
# ---------------------------------------------------------------------------------------------
class J2:
    __slots__ = ("a", "b", "c")

    def __init__(self, a, b=0, c=0):
        self.a = arb(a); self.b = arb(b); self.c = arb(c)

    def __add__(self, o):
        o = o if isinstance(o, J2) else J2(o)
        return J2(self.a + o.a, self.b + o.b, self.c + o.c)
    __radd__ = __add__

    def __neg__(self):
        return J2(-self.a, -self.b, -self.c)

    def __sub__(self, o):
        o = o if isinstance(o, J2) else J2(o)
        return J2(self.a - o.a, self.b - o.b, self.c - o.c)

    def __rsub__(self, o):
        return (-self) + o

    def __mul__(self, o):
        if not isinstance(o, J2):
            o = arb(o)
            return J2(self.a * o, self.b * o, self.c * o)
        return J2(self.a * o.a, self.a * o.b + self.b * o.a, self.a * o.c + 2 * self.b * o.b + self.c * o.a)
    __rmul__ = __mul__

    def exp(self):
        e = self.a.exp()
        return J2(e, e * self.b, e * (self.c + self.b * self.b))

    def compose(self, f0, f1, f2):
        """jet of f(self), given enclosures of f, f', f'' on the ball self.a"""
        return J2(f0, f1 * self.b, f2 * self.b * self.b + f1 * self.c)


def S_jet(Z, M=40):
    """S, S', S'' over the ball Z, 0 <= Z <= 10 (power series sum (-1)^n Z^n/(2n+1)!, tail <= 2 * first omitted term)."""
    Z = arb(Z)
    zu = ub(Z)
    assert zu < 10
    s0 = arb(0); s1 = arb(0); s2 = arb(0)
    fact = arb(1)
    for n in range(0, M + 1):
        if n > 0:
            fact = fact * (2 * n) * (2 * n + 1)
        sg = 1 if n % 2 == 0 else -1
        s0 += sg * Z ** n / fact
        if n >= 1:
            s1 += sg * n * Z ** (n - 1) / fact
        if n >= 2:
            s2 += sg * n * (n - 1) * Z ** (n - 2) / fact
    n = M + 1
    fact = fact * (2 * n) * (2 * n + 1)
    # for n > M >= 40 consecutive terms decrease by a factor <= 2*10/((2n+2)(2n+3)) < 1/2
    t0 = 2 * zu ** n / fact
    t1 = 2 * n * zu ** (n - 1) / fact
    t2 = 2 * n * (n - 1) * zu ** (n - 2) / fact
    return J2(s0 + arb(0, t0.upper()), s1 + arb(0, t1.upper()), s2 + arb(0, t2.upper()))


def Lam_derivs(w):
    """Lam, Lam', Lam'' over a ball w, 0 <= w <= 0.5; Lam(w) = sum_{i>=0} w^i/(i+1)."""
    w = arb(w)
    wu = ub(w)
    assert wu < arb("0.5000001")
    wf = max(float(wu), 1e-30)
    M = min(600, int(ctx.prec * 0.6932 / max(-math.log(wf), 0.6931)) + 8)
    l0 = arb(0); l1 = arb(0); l2 = arb(0)
    for i in range(0, M + 1):
        l0 += w ** i / (i + 1)
        if i >= 1:
            l1 += i * w ** (i - 1) / (i + 1)
        if i >= 2:
            l2 += i * (i - 1) * w ** (i - 2) / (i + 1)
    t0 = wu ** (M + 1) / (1 - wu)
    t1 = wu ** M / (1 - wu)
    t2 = wu ** (M - 1) * ((M + 1) / (1 - wu) + wu / (1 - wu) ** 2)
    return l0 + arb(0, t0.upper()), l1 + arb(0, t1.upper()), l2 + arb(0, t2.upper())


def F_jet(j, Z, y, v, q=None):
    """jet (F, F', F'') in zeta of F_j(zeta; y, v, q) over the ball Z."""
    S = S_jet(Z)
    zeta = J2(Z, 1, 0)
    beta = S * S
    A = S
    for _ in range(2 * j):
        A = A * S
    A = A * (1 - zeta * beta)
    if q is None:
        B = beta
    else:
        w = zeta * beta * (2 * arb(q))
        l0, l1, l2 = Lam_derivs(w.a)
        B = beta * w.compose(l0, l1, l2)
    E = (B * (-2 * arb(v) * arb(y) * arb(y))).exp()
    return A * E


# ---------------------------------------------------------------------------------------------
# the analytic constants of Lemmas 5.1 / 5.2
# ---------------------------------------------------------------------------------------------
ALPHA1 = {0: arb(7) / 6, 1: arb(9) / 6, 2: arb(11) / 6}
ALPHA2 = {0: arb(61) / 60, 1: arb(113) / 60, 2: arb(107) / 36}
BETA_STAR = 9 / (PI * PI)          # inf of beta on the low range zeta <= pi^2/36


def lam_consts(wbar):
    """Lam(wbar), Lam'(wbar), Lam''(wbar) from the closed forms (wbar in (0, 1/2])."""
    w = arb(wbar)
    l0 = -(1 - w).log() / w
    l1 = (w / (1 - w) + (1 - w).log()) / (w * w)
    l2 = 1 / (w * (1 - w) ** 2) - 2 / (w * w * (1 - w)) - 2 * (1 - w).log() / w ** 3
    return l0, l1, l2


def B_consts(qbar):
    """(b0, b1, b2):  |B'(0)| <= b0,  |B'| <= b1,  |B''| <= b2 on the low range, for all q in (0, qbar].
    qbar = 0 means continuous time (B = beta)."""
    if qbar == 0:
        return arb(1) / 3, arb(1) / 3, arb(4) / 45
    qb = arb(qbar)
    l0, l1, l2 = lam_consts(qb / 2)
    b0 = max_ball(arb(1) / 3, qb - arb(1) / 3)
    b1 = max_ball(l0 / 3, 2 * qb * l1)
    b2 = max_ball(arb(4) / 45 * l0 + 4 * qb * qb * l2, arb(8) / 3 * qb * l1)
    return ub(b0), ub(b1), ub(b2)


def max_ball(a, b):
    au, bu = ub(a), ub(b)
    return au if au >= bu else bu


def low_tail_bound(j, vlo, vhi, k0, qbar=0):
    """B_low of Proposition 5.4: sum_{k odd > k0} 2^j y^(2j+5) e^{-2 vlo y^2 beta*} [al2 + 2 a al1 b1 + a^2 b1^2 + a b2],
    a = 2 vhi y^2."""
    b0, b1, b2 = B_consts(qbar)
    vhi = ub(vhi)
    al1, al2 = ALPHA1[j], ALPHA2[j]
    poly = [al2, 2 * vhi * (2 * al1 * b1 + b2), 4 * vhi * vhi * b1 * b1]
    return ub(2 ** j * pos_sum(2 * j + 5, arb(vlo) * BETA_STAR, k0 + 2, poly))


def chi_rate(qbar):
    """chi(qbar): every high lattice mode satisfies |lambda|^(t-1) <= exp(-chi v L^2), v = q(t-1)/L^2, for all q <= qbar < 1.
    chi = 1/2 if qbar <= 4/5 (|lambda| <= 1 - q/2), else min(1/2, -ln(2 qbar - 1)/qbar)   (|lambda| <= max(1-q/2, 2q-1))."""
    qb = arb(qbar)
    if qb <= arb(4) / 5:
        return arb(1) / 2
    assert qb < 1
    r = -(2 * qb - 1).log() / qb
    return arb(1) / 2 if r > arb(1) / 2 else arb(r.lower())


HIGH_TOL = None      # "tolerance mode": if set (a string like "1e-12"), B_high is replaced by this number, i.e. the
                     # certificates are valid for every L >= L0 for which 2^(j+1)(pi/2)^(2j+1) L^(2j+6) exp(-chi v_- L^2) <= HIGH_TOL


def high_bound(j, vlo, L0, qbar=0):
    """B_high of Prop. 5.4: 2^(j+1) (pi/2)^(2j+1) L^(2j+6) exp(-chi vlo L^2), decreasing for L >= L0."""
    if HIGH_TOL is not None:
        return arb(HIGH_TOL)
    vlo = arb(arb(vlo).lower()); L0 = arb(L0)
    chi = chi_rate(qbar)
    assert L0 * L0 > (2 * j + 6) / (2 * chi * vlo), "L0 too small for the monotone high-mode bound"
    return ub(2 ** (j + 1) * (PI / 2) ** (2 * j + 1) * L0 ** (2 * j + 6) * (-chi * vlo * L0 * L0).exp())


def cont_tail_bound(j, vlo, vhi, L0, qbar=0):
    """B_tail of Proposition 5.4: sum_{k odd > 2 L0/3} 2^(j+1) y^(2j+5) e^{-2 vlo y^2} [(6/pi)^4 + (6/pi)^2 (al1 + 2 vhi b0 y^2)]."""
    b0, b1, b2 = B_consts(qbar)
    Kst = int(math.floor(2 * float(arb(L0).lower()) / 3)) + 1      # smallest integer > 2 L0/3 (or equal: superset)
    while arb(Kst) > 2 * arb(L0) / 3 and Kst > 1:
        Kst -= 1                                                    # step down to be safely a superset
    Kst = max(1, Kst)
    c4 = (6 / PI) ** 4
    c2 = (6 / PI) ** 2
    poly = [c4 + c2 * ALPHA1[j], c2 * 2 * ub(vhi) * b0]
    return ub(2 ** (j + 1) * pos_sum(2 * j + 5, vlo, Kst, poly))


def R_enclosure(j, V, L0, k0, q=None, qbar=0):
    """Ball containing R = L^4 [S_j - G^(j)(v) - H_j(v)/L^2] for every L = N - 1/2 >= L0, every v in V
    (continuous time: q None, qbar 0; discrete time: integer t with v = q(t-1)/L^2 in V, q in the ball q <= qbar < 1).
    Returns (R, Bnd, parts)."""
    V = arb(V); L0 = arb(L0)
    assert k0 % 2 == 1 and arb(3 * k0) <= 2 * L0, "need k0 <= 2 L0/3"
    R = arb(0)
    for k in range(1, k0 + 1, 2):
        y = k * PI / 4
        Z = interval(arb(0), y * y / (L0 * L0))
        F = F_jet(j, Z, y, V, q)
        R += sigma(k) * (-2) ** j * y ** (2 * j + 5) * F.c
    vlo, vhi = arb(V.lower()), arb(V.upper())
    bl = low_tail_bound(j, vlo, vhi, k0, qbar)
    bh = high_bound(j, vlo, L0, qbar)
    bt = cont_tail_bound(j, vlo, vhi, L0, qbar)
    Bnd = ub(bl + bh + bt)
    return R + arb(0, Bnd.upper()), Bnd, (bl, bh, bt)


def H(j, v, q=0):
    """H_j^{(q)}(v) = (7+2j)/12 G^(j+1)(v) + (v/6)(1-3q) G^(j+2)(v)."""
    v = arb(v)
    return arb(7 + 2 * j) / 12 * G(j + 1, v) + v / 6 * (1 - 3 * arb(q)) * G(j + 2, v)
