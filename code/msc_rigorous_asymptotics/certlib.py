"""Rigorous (Arb ball arithmetic) evaluation of the enclosures of Sections 3-5 of note R3.

Everything here is a direct transcription of formulas proved in note R3:
  Lemma 3.1 (nu0, b0), Lemma 3.2 (EU, m0), Lemma 3.3 (theta, e, a1), Lemma 3.4 (near poles), Lemma 3.5 (beta_hat, pi_a, R_tot),
  Lemma 4.1 (sigma <= Z_d), Prop. 4.5/4.6 (X_s bounds), Lemma 4.8 (T_1 enclosures), Theorem 2.3 (sign conditions),
  Section 5 (the scaled sign function Psi), Section 7 (discrete time: Lemma 7.6, Lemma 7.11, Theorem 7.13, Lemma 7.16).
A "ball" is a flint.arb; intervals [lo, hi] are represented by balls containing them.
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import json, os, math
from flint import arb, ctx

ctx.prec = 90
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = _os.path.join(_R, 'data', 'msc_rigorous_asymptotics')
PI = arb.pi()
CONST = json.load(open(_os.path.join(_R, 'data', 'msc_rigorous_asymptotics', '50_cert_constants.json')))
Z_HI = {2: arb(CONST['Z2_check_upper'][1]), 3: arb(CONST['Z3'][1])}      # self-contained: partial sum + tail (Lemma 4.2)
C3_LO, C3_HI = arb(CONST['c3'][0]), arb(CONST['c3'][1])
CK_LO, CK_HI = arb(CONST['ckappa_lo_10']), arb(CONST['ckappa_hi_10'])          # valid for Z >= 10
Z1 = 10
CBOX = {2: arb(CONST['cbox2'][1]), 3: arb(CONST['cbox3'][1])}
T_S0 = {2: arb(CONST['T2_s0_1'][1]), 3: arb(CONST['T3_s0_1'][1])}                # T_d(s_0), s_0 = 1  (Prop. 4.5/4.6)
J_HI = arb(CONST['J'][1])
NMIN = 10                                                                       # all X_s bounds are for N >= NMIN


# ----------------------------------------------------------------------------- ball helpers
def L(a):
    return arb(a).lower()


def U(a):
    return arb(a).upper()


def iv(lo, hi):
    return arb(lo).union(arb(hi))


def fl(a):
    return float(arb(a).mid())


def ylog(y):
    """ball for y ln(1/y), y in [ya, yb] subset [0, 1/e] (increasing there)."""
    ya, yb = L(y), U(y)
    assert yb < arb('0.36')
    lo = arb(0) if not (ya > 0) else ya * (1 / ya).log()
    return iv(L(lo), U(yb * (1 / yb).log()))


def log1p_over(a, y):
    """ball for ln(1 + a y)/y, a ball, y in [ya, yb], ya >= 0, 1 + a y > 0.  Increasing in a, decreasing in y."""
    ya, yb = L(y), U(y)
    alo, ahi = L(a), U(a)
    assert (1 + alo * yb > 0) and (1 + alo * ya > 0)
    lo = (1 + alo * yb).log() / yb
    hi = (1 + ahi * ya).log() / ya if (ya > 0) else ahi
    return iv(L(lo), U(hi))


# ----------------------------------------------------------------------------- X_s bounds (Prop. 4.5 / 4.6)
def eps_iota(d):
    """upper bound for the image corrections eps_d of Prop. 4.5/4.6, valid for all N >= NMIN (decreasing in N)."""
    N = arb(NMIN)
    x = PI / (2 * N)
    c0 = (arb(2).cosh() - 1) / 2
    beta = (2 * N - 1) ** 2 * x.sin() ** 2 / c0
    GA = 1 / (1 - (-(4 / c0)).exp())
    cA = GA * PI * PI.sqrt()                       # iota_A(s) <= cA s^{-1/2} e^{-beta/s}
    iA1 = cA * (-beta).exp()                       # iota_A(1)  (iota_A increasing on (0,1] as beta >= 1/2)
    iB = 4 * N * (-(2 * N - 1)).exp() / (1 - (-2 * N).exp())
    assert beta > 1
    if d == 2:
        return U(2 * cA * PI.sqrt() * (-beta).exp() / beta + 4 * iB * PI.sqrt() + (iA1 + iB) ** 2)
    return U(3 * cA * PI * (-beta).exp() / beta + 3 * iB * PI * (1 + (N * N / PI).log()) + 6 * (iA1 + iB) ** 2 * PI.sqrt()
             + (iA1 + iB) ** 3)


EPS_IOTA = {2: eps_iota(2), 3: eps_iota(3)}


def Xs_upper(d, N):
    """rigorous upper bound for X_s(N), N >= NMIN (Prop. 4.5/4.6 with s_0 = 1); increasing in N."""
    N = arb(N)
    assert L(N) >= NMIN
    x = PI / (2 * N)
    sx2 = x.sin() ** 2
    lam = 1 / (2 * sx2)
    assert L(lam) >= Z1
    if d == 2:
        K2 = 2 / PI * lam.log() + CK_HI
        return U(2 * sx2 * N ** 2 * K2 - 1 + T_S0[2] + EPS_IOTA[2])
    tail = 2 * (N * x.sin()) ** 3 * (4 / PI) ** arb('1.5') * (1 - 1 / (3 * lam))
    return U(2 * sx2 * N ** 3 * J_HI - tail - 1 + T_S0[3] + EPS_IOTA[3])


def Xs_lower(d, N):
    N = arb(N)
    x = PI / (2 * N)
    if d == 2:
        Z0 = 2 * N * N / PI
        assert L(Z0) >= Z1 and L(N) >= NMIN
        K2 = 2 / PI * Z0.log() + CK_LO
        return L(2 * x.sin() ** 2 * (N * N * K2 - Z0))
    return L(2 * x.sin() ** 2 * N ** 3 * 2 * C3_LO / PI ** 2 - 2 * x.sin() ** 2 * 6 * N * N / PI)


def N_range(d, ya, yb, N0):
    """(N_lo, N_hi): every N >= N0 with 1/X_s(N) in [ya, yb] satisfies N_lo <= N <= N_hi  (N_hi = None if ya = 0)."""
    # N_lo: largest value such that Xs_upper(N) < 1/yb for all N < N_lo (Xs_upper increasing in N)
    target = float(U(1 / arb(yb)))
    f = lambda n: float(Xs_upper(d, n))
    lo, hi_ = float(N0), float(N0)
    if f(lo) >= target:
        N_lo = float(N0)
    else:
        while f(hi_) < target and hi_ < 1e300:
            hi_ *= 2
        if hi_ >= 1e300:
            N_lo = 1e300
        else:
            for _ in range(200):
                mid = 0.5 * (lo + hi_)
                if f(mid) < target:
                    lo = mid
                else:
                    hi_ = mid
            N_lo = lo * (1 - 1e-9)
            assert Xs_upper(d, N_lo) < 1 / arb(yb)
    N_lo = max(float(N0), N_lo)
    N_hi = None
    if arb(ya) > 0:
        target = float(L(1 / arb(ya)))
        g = lambda n: float(Xs_lower(d, n))
        lo, hi_ = max(N_lo, 6.0), max(N_lo, 6.0)
        while g(hi_) <= target and hi_ < 1e300:
            hi_ *= 2
        if hi_ < 1e300:
            for _ in range(200):
                mid = 0.5 * (lo + hi_)
                if g(mid) <= target:
                    lo = mid
                else:
                    hi_ = mid
            N_hi = hi_ * (1 + 1e-9)
            assert Xs_lower(d, N_hi) > 1 / arb(ya)
    return N_lo, N_hi


# ----------------------------------------------------------------------------- lattice constant T_1 (Lemma 4.8)
def eps_k(k, x):
    return (k * x).sin() ** 2 / x.sin() ** 2 if k > 0 else arb(0)


_T1_cache = {}


NFULL = {2: 64, 3: 64}


def T1_bounds(d, N_lo, N_hi=None, Kmax=24):
    """[T1_lo, T1_hi] for T_1 = sum_{mu_k >= eps_2} w_k/(mu_k(mu_k-1)), valid for all integers N in [N_lo, N_hi]."""
    # The certificates pass the block ends as Python integers, which are used exactly (no float round trip, which would be
    # inexact for N > 2^53).  Real-valued ranges (used only by the sanity check 51_validate_enclosures.py) are rounded inwards
    # to the integers they contain; they must be below 2^52 so that the float rounding is exact.
    def _ilo(v):
        if isinstance(v, int):
            return v
        assert abs(v) < 2.0 ** 52
        return int(math.ceil(v - 1e-9))

    def _ihi(v):
        if v is None or isinstance(v, int):
            return v
        assert abs(v) < 2.0 ** 52
        return int(math.floor(v + 1e-9))
    Nl = _ilo(N_lo)
    Nh = _ihi(N_hi)
    key = (d, Nl, Nh, Kmax)
    if key in _T1_cache:
        return _T1_cache[key]
    if Nh is not None and Nh <= NFULL[d]:
        # exact finite sums for each integer N in the range (Lemma 4.8, case K = N-1)
        lo, hi_ = None, None
        for N in range(Nl, Nh + 1):
            l1, h1 = _T1_trunc(d, N, N, N - 1, tail=False)
            lo = l1 if lo is None else min(lo, l1)
            hi_ = h1 if hi_ is None else max(hi_, h1)
        res = (lo, hi_)
    else:
        res = _T1_trunc(d, Nl, Nh, min(Kmax, Nl - 1), tail=True)
    _T1_cache[key] = res
    return res


def _T1_trunc(d, Nl, Nh, K, tail=True):
    """explicit sum over k in {0..K}^d plus (if tail) the tail bound of Lemma 4.8; integers N in [Nl, Nh] (Nh None = inf)."""
    xb = PI / (2 * arb(Nl))                                   # largest x
    xa = arb(0) if Nh is None else PI / (2 * arb(Nh))         # smallest x
    N_hi = Nh
    # per-coordinate quantities
    eps_min = [eps_k(k, xb) for k in range(K + 1)]            # eps_k decreasing in x -> minimal at xb
    eps_max = [arb(k * k) if N_hi is None else eps_k(k, xa) for k in range(K + 1)]
    gam_min = [arb(1)] + [2 * (k * xb).cos() ** 2 for k in range(1, K + 1)]
    gam_max = [arb(1)] + [arb(2) if N_hi is None else 2 * (k * xa).cos() ** 2 for k in range(1, K + 1)]
    lo, hi_ = arb(0), arb(0)
    import itertools
    for k in itertools.combinations_with_replacement(range(K + 1), d):
        if max(k) < 2:
            continue
        # multiplicity of the sorted tuple
        mult = math.factorial(d)
        for c in set(k):
            mult //= math.factorial(k.count(c))
        mu_min = sum(eps_min[i] for i in k); mu_max = sum(eps_max[i] for i in k)
        wmin = arb(1); wmax = arb(1)
        for i in k:
            wmin *= gam_min[i]; wmax *= gam_max[i]
        lo += mult * wmin / (mu_max * (mu_max - 1))
        hi_ += mult * wmax / (mu_min * (mu_min - 1))
    # tail |k|_inf > K :  <= (M/(M-1)) * tail_Z(K),  M = eps_{K+1}(xb)
    if tail:
        M = eps_k(K + 1, xb)
        Kq = arb(K)
        if d == 2:
            tailZ = CBOX[2] / Kq ** 2 + arb(4) / (3 * Kq ** 3)
        else:
            tailZ = CBOX[3] / Kq + 3 * CBOX[2] / Kq ** 2 + arb(2) / Kq ** 3
        hi_ += M / (M - 1) * tailZ
    return (L(lo), U(hi_))


# ----------------------------------------------------------------------------- pole-data enclosures (Section 3)
def h_d(d, nu):
    if d == 2:
        return 2 / (1 - nu) - 1 / (2 - nu)
    return 3 / (1 - nu) - 3 / (2 - nu) + 1 / (3 - nu)


def enclosures(d, y, N_lo, N_hi=None, Kmax=24):
    """all enclosures as balls, valid for every integer N in [N_lo, N_hi] with 1/X_s(N) in the ball y."""
    y = arb(y)
    xb = PI / (2 * arb(N_lo))
    xa = arb(0) if N_hi is None else PI / (2 * arb(N_hi))
    x = iv(L(xa), U(xb))
    cos2 = iv(L(xb.cos() ** 2), U(xa.cos() ** 2))            # cos^2 x decreasing in x
    c2x = iv(L((2 * xb).cos() ** 2), U((2 * xa).cos() ** 2))
    A = 2 * cos2; W = d * A; e2 = 4 * cos2; g2 = 2 * c2x
    C1 = d * (d - 1) * A * A; C2 = d * g2 * e2; W2 = C1 / 2; W3 = A ** 3 if d == 3 else arb(0)
    E = dict(d=d, y=y, x=x, A=A, W=W, e2=e2, g2=g2, C1=C1, C2=C2, W2=W2, W3=W3)
    sig_hi = Z_HI[d]
    # --- Lemma 3.1
    yb = U(y)
    s_hi = y * y * sig_hi / (1 - y)                           # nu0 = y (1 - s),  W nu0^2 <= s <= s_hi
    r0lo = 1 - U(s_hi)
    assert r0lo > 0
    r0hi = 1 - L(W * (y * r0lo) ** 2)
    r0 = iv(r0lo, r0hi)
    nu0 = y * r0
    E.update(r0=r0, nu0=nu0, s_over_y=iv(L(W * y * r0lo ** 2), U(y * sig_hi / (1 - y))))
    inv_b0 = iv(L(1 + nu0 ** 2 * W), U(1 + nu0 ** 2 * sig_hi / (1 - nu0) ** 2))
    E['inv_b0_hi'] = U(inv_b0)
    E['one_minus_b0_over_y_hi'] = U(r0 * nu0 * sig_hi / (1 - nu0) ** 2)      # (1-b0)/y <= (nu0/y) nu0 sigma/(1-nu0)^2
    E['ln_inv_b0_over_y'] = iv(arb(0), U(r0 * nu0 * sig_hi / (1 - nu0) ** 2))
    # --- Lemma 3.2
    Ainv_e2 = g2 * (-(e2 * A.log())).exp()                    # gamma_2 A^{-eps_2}
    EU_lo = (arb(3) / 2 if d == 2 else arb(11) / 6) - d * Ainv_e2 / e2 + A.log() * (1 - Ainv_e2 ** d)
    EU_hi = A.log() + (arb(3) / 2 if d == 2 else arb(11) / 6)
    EU = iv(L(EU_lo), U(EU_hi))
    E['EU'] = EU
    E['lnm0_over_nu0_hi'] = U(A.log() + h_d(d, nu0))          # ln m0 <= nu0 (ln A + h_d(nu0)),  ln m0 >= nu0 EU
    # --- Lemma 3.3
    T1lo, T1hi = T1_bounds(d, N_lo, N_hi, Kmax)
    # monotone iteration:  e >= e_lo  =>  theta <= thbar(e_lo)  =>  e <= e_hi(thbar)  =>  theta >= thlo(e_hi)  =>  e >= e_lo'(thlo) ...
    e_lo = L(W2 / 2 + W3 / 6 + T1lo)                             # theta >= 0
    for _ in range(6):
        den = 1 - (W + 1 - e_lo) * y
        assert L(den) > 0
        thbar = U(W * y / den)
        assert thbar < 1
        Snu_hi = W2 / (2 * (1 - thbar)) + W3 / (3 * (2 - thbar)) + T1hi * (e2 - 1) / (e2 - 1 - thbar)
        e_hi = U(thbar / (1 + thbar) + (1 + thbar) * Snu_hi)
        den_lo = 1 - (W + 1 - e_hi) * y
        assert L(den_lo) > 0
        thlo = L(W * y / den_lo)
        thlo = thlo if thlo > 0 else arb(0)
        Snu_lo = W2 / (2 * (1 - thlo)) + W3 / (3 * (2 - thlo)) + T1lo
        e_lo_new = L(thlo / (1 + thlo) + (1 + thlo) * Snu_lo)
        if e_lo_new > e_lo:
            e_lo = e_lo_new
    e = iv(e_lo, e_hi)
    a_ = W + 1 - e
    assert L(1 - a_ * y) > 0
    th_over_y = W / (1 - a_ * y)
    theta = y * th_over_y
    assert U(theta) < 1
    S2nu = iv(L(W2), U(W2 / (1 - theta) ** 2 + W3 / (2 - theta) ** 2
                        + T1hi * (e2 / (e2 - 1)) * ((e2 - 1) / (e2 - 1 - theta)) ** 2))
    zeta = 1 / (W * (1 + theta) ** 2) + S2nu / W
    a1_over_y = W / ((1 + (e - 1) * y) * (1 + theta ** 2 * zeta))
    E.update(T1=iv(T1lo, T1hi), e=e, a_=a_, th_over_y=th_over_y, theta=theta, zeta=zeta, a1_over_y=a1_over_y)
    # --- Lemma 3.4 (near poles): monotone iteration for theta_i/y in [lo, hi]
    def near(i):
        """enclosure of theta_i / y for the near pole above level m_i (i = 2: m = 2; i = 3: m = 3, d = 3)."""
        if i == 2:
            base = W + W2 / 2                                     # sum_{l<=2} W_l/m_l
            low = lambda th: W / (1 + th) + 1 / (2 + th)          # sum_{l<2} W_l/(nu-m_l) + 1/nu,  nu = 2 + th
            up_lo = lambda th: (W3 / (3 * (1 - th)) if d == 3 else arb(0)) + T1lo
            up_hi = lambda th: (W3 / (3 * (1 - th)) if d == 3 else arb(0)) + T1hi * (e2 - 1) / (e2 - 2 - th)
            m, Wi = 2, W2
        else:
            base = W + W2 / 2 + W3 / 3
            low = lambda th: W / (2 + th) + W2 / (1 + th) + 1 / (3 + th)
            up_lo = lambda th: T1lo + arb(0) * th
            up_hi = lambda th: T1hi * (e2 - 1) / (e2 - 3 - th)
            m, Wi = 3, W3
        # W_i/theta_i = X_s - base + nu * SUM_{l>i} W_l/(m_l(m_l-nu)) - low(theta_i),  nu = m + theta_i
        thlo = arb(0)
        hi_y = None
        for _ in range(4):
            den = 1 - (base + low(thlo) - (m + thlo) * up_lo(thlo)) * y           # lower bound of y * W_i/theta_i
            assert L(den) > 0
            hi_y = U(Wi / den)
            thhi = U(hi_y * yb)
            assert L(e2 - m - thhi) > 0 and thhi < 1
            den2 = 1 - (base + low(thhi) - (m + thhi) * up_hi(thhi)) * y          # upper bound of y * W_i/theta_i
            assert L(den2) > 0
            lo_y = L(Wi / den2)
            thlo = L(lo_y * L(y))
            thlo = thlo if thlo > 0 else arb(0)
        return lo_y, hi_y, thlo, thhi
    lo2, hi2, th2lo, th2 = near(2)
    rho2_over_y = U(hi2 * th2 / (2 * W2))                     # rho_2 <= theta_2^2/(2 W_2);  rho_2/y <= (theta_2/y)^2 y/(2W_2)
    E.update(th2_over_y=iv(lo2, hi2), th2=th2, th2lo=th2lo, rho2_over_y=rho2_over_y)
    if d == 3:
        lo3, hi3, th3lo, th3 = near(3)
        rho3_over_y = U(hi3 * th3 / (3 * W3))
        E.update(th3_over_y=iv(lo3, hi3), th3=th3, th3lo=th3lo, rho3_over_y=rho3_over_y)
    # --- Lemma 3.5
    b1_over_y_lo = L(theta * a1_over_y / (W * (1 + theta)))
    E['betahat_over_y'] = iv(arb(0), U(E['one_minus_b0_over_y_hi'] - b1_over_y_lo))
    Nl = arb(N_lo)
    E['pia_hi'] = U(Nl ** (-d)); E['Rtot_hi'] = U(arb(d) / 4 * Nl ** (2 - d))
    # --- coupling xi = sin^2 x / y = sin^2(x) X_s
    if N_hi is not None:
        # sin^2 x <= sin^2(pi/(2 N_lo)),  X_s <= Xs_upper(N_hi)  (Xs_upper increasing)
        E['xi'] = iv(arb(0), U(xb.sin() ** 2 * Xs_upper(d, N_hi)))
        E['pia_over_y_hi'] = U(Nl ** (-d) * Xs_upper(d, N_hi))    # pi_a / y = N^{-d} X_s
    else:
        # N >= N_lo:  sin^2 x X_s <= (pi/2N)^2 Xbar(N),  Xbar(N) <= 2 pi ln N + 1.3 (d=2),  <= c3 N (d=3)   [Cor. 4.7]
        maj = (2 * PI * Nl.log() + arb('1.3')) if d == 2 else C3_HI * Nl
        E['xi'] = iv(arb(0), U((PI / (2 * Nl)) ** 2 * maj))
        # xi * ln(1/y) <= (pi/2N)^2 maj(N) ln(maj(N)), decreasing in N >= N_lo  (used only in discrete time)
        E['xi_ln'] = U((PI / (2 * Nl)) ** 2 * maj * maj.log())
        E['pia_over_y_hi'] = U(Nl ** (-d) * maj)                  # N^{-d} X_s <= N^{-d} maj(N), decreasing in N
    return E


# ----------------------------------------------------------------------------- Lemma 7.16: q-uniform constants
def Dfun(N, a, d):
    """N * D_a(x), x = pi/(2N), with D_a(x) = 2 sin 2x - (2a/d) sin^2 x (a = 1, 2) and
    D_e2(x) = sin 2x (2 - (2/d) sin 2x) (the case a = eps_2 = 4 cos^2 x).  Increasing in N >= 10 (Lemma 7.16(c))."""
    N = arb(N)
    s2 = (PI / N).sin()                       # sin 2x
    s1 = (PI / (2 * N)).sin()                 # sin x
    if a == 'e2':
        return N * s2 * (2 - arb(2) / d * s2)
    return N * (2 * s2 - arb(2 * a) / d * s1 ** 2)


def quni_bounds(E, d, N_lo, N_hi):
    """Lemma 7.16: bounds, uniform in q, for the q-dependent constants of Lemma 7.11 / Theorem 7.13.
    Valid for every integer N in [N_lo, N_hi] (N_hi = None: all N >= N_lo) and every q in (0,1) with sin(pi/N) <= (1-q)/q:
      qu_pB_over_y[a] >= pi_a X_s / D_a(x)      >= (pi_a / y) * q / (2(1-q) - a mu)     (a = 1, 2, 'e2' = eps_2)
      qu_pB['e2']     >= pi_a / D_e2(x)          >= pi_a * q / (2(1-q) - eps_2 mu)
      qu_pS_over_y    >= pi_a X_s / sin 2x       >= (pi_a / y) * q / (1-q)."""
    assert N_lo >= NMIN
    Nl = arb(N_lo)
    out = {}
    if N_hi is not None:
        # pi_a <= N_lo^{-d};  X_s <= Xs_upper(N_hi);  D_a(x) increasing in x  =>  D_a(x) >= D_a(pi/(2 N_hi))
        Nh = arb(N_hi)
        Xh = arb(Xs_upper(d, N_hi))
        for a in (1, 2, 'e2'):
            Da = Dfun(N_hi, a, d) / Nh
            assert L(Da) > 0
            out[a] = U(Nl ** (-d) * Xh / Da)
        E['qu_pB'] = {'e2': U(Nl ** (-d) / (Dfun(N_hi, 'e2', d) / Nh))}
        E['qu_pS_over_y'] = U(Nl ** (-d) * Xh / (PI / Nh).sin())
    else:
        # N >= N_lo:  pi_a X_s / D_a = N^{1-d} X_s / (N D_a);  N^{1-d} maj(N) decreasing, N D_a(x_N) and N sin(pi/N) increasing
        maj = (2 * PI * Nl.log() + arb('1.3')) if d == 2 else C3_HI * Nl
        for a in (1, 2, 'e2'):
            Da = Dfun(N_lo, a, d)
            assert L(Da) > 0
            out[a] = U(Nl ** (1 - d) * maj / Da)
        E['qu_pB'] = {'e2': U(Nl ** (1 - d) / Dfun(N_lo, 'e2', d))}
        E['qu_pS_over_y'] = U(Nl ** (1 - d) * maj / (Nl * (PI / Nl).sin()))
    E['qu_pB_over_y'] = out
    E['quni'] = True
    return E


# ----------------------------------------------------------------------------- the scaled sign function (Section 5)
def Psi(E, c, side, disc=False, qmax=0.5, nounimodal=False, shift=None, quni=False):
    """side = '+': ball containing an UPPER bound function for S(t)/y at t = tau_cf + c*y'  (c = c_plus);
       side = '-': LOWER bound function for S(t)/y at t = tau_cf + c*y'  (c = -c_minus).
       Here S(t) = ln(a1/a0) - (nu1-nu0) t + [correction logs] has the sign of the bound Phi_+ resp. Phi_- of Thm 2.3.
       Returns the ball of  S/y  INCLUDING the term T5 = -c (1+theta-nu0)/(1+EU y).
       quni=True (with qmax=1, nounimodal=True, E from quni_bounds): the q-dependent constants are replaced by the
       q-uniform bounds of Lemma 7.16, valid for every q in (0,1) with sin(pi/N) <= (1-q)/q."""
    d, y = E['d'], E['y']
    W, A, e2, g2, C1, C2, W2, W3 = E['W'], E['A'], E['e2'], E['g2'], E['C1'], E['C2'], E['W2'], E['W3']
    r0, nu0, EU, e, theta, zeta = E['r0'], E['nu0'], E['EU'], E['e'], E['theta'], E['zeta']
    thy, a_, xi = E['th_over_y'], E['a_'], E['xi']
    # a float constant c (a decimal number with 3 digits after the point) is converted through its decimal string,
    # so that the ball contains the decimal value printed in the tables (not only its binary approximation)
    c = arb(repr(c)) if isinstance(c, float) else arb(c)
    if quni:
        assert disc and nounimodal and float(qmax) == 1.0 and E.get('quni', False)
    cosx2 = A / 2
    if disc:
        # discrete time, q <= 1/2:  mu = (2q/d) sin^2 x <= sin^2 x/d = xi*y/d.  The integer time t_int with
        # mu*t_int in [tau_cf + c y', tau_cf + c y' + mu)  (side '+')  resp. (tau_cf + c y' - mu, tau_cf + c y'] (side '-')
        # corresponds to c in c + [0, mu X] resp. c - [0, mu X],  mu X <= xi (1 + EU y)/d.
        fq = 2 * arb(str(qmax))                                   # mu = (2q/d) sin^2 x <= fq * xi * y / d
        muX = iv(arb(0), U(fq * xi * (1 + EU * y) / d))
        c = c + muX if side == '+' else c - muX
    # y * tau_cf and related
    yp_over_y = 1 / (1 + EU * y)                                  # y'/y
    ytcf = (y * arb(2 * d).log() + y * (1 + EU * y).log() + ylog(y)) / (1 + (2 * d - 1) * y * yp_over_y)
    ytcf = iv(max(L(ytcf), arb(0)), U(ytcf))                      # y*tau_cf >= 0
    b = arb(0) if shift is None else arb(shift)                   # t = tau_cf + c y' - b,  b >= 0
    yt = ytcf + c * y * y * yp_over_y - b * y                     # y * t
    # t > 0 (checked for every ball c, also when c straddles 0):  tau_cf is increasing in X and X >= 1/yb,
    # so tau_cf >= tau_cf(X = 1/yb);  max(-c, 0) y' <= max(-c, 0) yb
    Xl = 1 / U(y)
    cneg = U(-c) if U(-c) > 0 else arb(0)
    assert L(Xl * (2 * d * Xl).log() / (Xl + 2 * d - 1)) > U(cneg * U(y) + U(b))
    yt = iv(max(L(yt), arb(0)), U(yt))                            # y*t >= 0
    lncos = iv(L(-xi * y / cosx2), arb(0))                        # ln cos^2 x  in [-sin^2x/cos^2x, 0],  sin^2 x = xi*y
    delta = -lncos + (1 + EU * y).log() - (2 * d - 1) * ytcf * yp_over_y + c * y * yp_over_y - b     # t - ln(W/y)
    th_t = thy * yt                                               # theta * t
    nu0_t = r0 * yt                                               # nu0 * t
    em_d = (-delta).exp()
    eps_t = (y / W) ** 2 * (-2 * delta).exp() * (4 * xi * yt).exp()        # e^{-(eps2-2) t}
    TD = arb(0)
    if disc:
        mu = iv(arb(0), U(fq * xi * y / d))                       # mu
        mut = iv(arb(0), U(fq * xi * yt / d))                     # mu * tau
        if L(y) > 0:
            xit = U(xi * yt / y)                                  # xi * tau
        else:
            cpos = U(c) if U(c) > 0 else arb(0)
            xit = U(E['xi_ln'] + xi * ((2 * d * (1 + EU * y)).log() + cpos * y * yp_over_y))
        nu1 = 1 + theta
        assert U(mu * e2 * (1 + e2)) < 1 and U(mu * nu1) < 1
        # main term:  -(nu_hat_D - nu_hat) tau / y,  nu_hat_D - nu_hat in mu*[nu1^2/2 - nu0^2/(2(1-mu nu0)), nu1^2/(2(1-mu nu1)) - nu0^2/2]
        TD = iv(L(-(fq * xit / d) * (nu1 ** 2 / (2 * (1 - mu * nu1)) - nu0 ** 2 / 2)), arb(0))
    nqP = arb(0)
    if nounimodal:
        # negative poles (q > 1/2):  |P_neg|/(a1 E_{nu1} y) <= nqP   (Lemma 7.11);  for q <= 1/2 this is just extra slack
        assert disc
        if quni:
            # Lemma 7.16:  (q/2) W/(1-mu) <= W/(2(1-mu));  q/(2(1-q) - a mu) <= 1/D_a(x);  (H4) and q + mu <= 1 hold by Lemma 7.16(b)
            nqB2_y = E['pia_over_y_hi'] * W / (2 * (1 - mu)) + E['qu_pB_over_y'][2] * C1
            nqBe_y = (-2 * delta).exp() * (4 * xi * yt).exp() * C2 * e2 * (y * E['qu_pB']['e2']) / 2 / W ** 2
        else:
            qm = arb(str(qmax))
            assert L(2 * (1 - qm) - e2 * mu) > 0 and U(2 * qm - 1) < L(1 - 2 * mu)
            nqB2_y = E['pia_over_y_hi'] * (qm / 2) * (W / (1 - mu) + 2 * C1 / (2 * (1 - qm) - 2 * mu))
            nqBe_y = (-2 * delta).exp() * (4 * xi * yt).exp() * C2 * e2 * (y * E['pia_hi']) * (qm / 2) / (2 * (1 - qm) - e2 * mu) / W ** 2
        nqP = iv(arb(0), U(em_d * th_t.exp() * (nqB2_y + nqBe_y) / (W * E['a1_over_y'])))
    # ---- T1
    T1 = -log1p_over(e - 1, y)
    z = theta ** 2 * zeta
    T1 += -(thy * theta * zeta) * iv(L(1 - z / 2), arb(1))        # -ln(1+z)/y in -(z/y)[1 - z/2, 1]
    T1 += E['ln_inv_b0_over_y']
    s_y = E['s_over_y']                                           # s/y,  nu0/y = 1 - s
    s_hi = U(s_y) * U(y)
    T1 += iv(L(2 * s_y), U(2 * s_y / (1 - s_hi)))                 # -2 ln(1-s)/y in [2s/y, 2s/(y(1-s))]
    # (v) + T3:  -[ln m0 + ln(1+EU y)]/y ;  ln m0/y in [r0*EU, r0*lnm0_over_nu0_hi]
    lnm0_y = iv(L(r0 * EU), U(r0 * E['lnm0_over_nu0_hi']))
    m0 = (y * lnm0_y).exp()
    T13 = -lnm0_y - log1p_over(EU, y)
    # ---- T2
    T2 = iv(L(-xi / cosx2), arb(0))
    # ---- T4
    q4 = -(2 * d - 1) * EU / (1 + EU * y) - W * a_ / (1 - a_ * y) - s_y
    T4 = ytcf * (2 * d * xi + q4)
    # ---- T5
    T5 = -c * (1 + theta - nu0) * yp_over_y
    # ---- T6
    if side == '+':
        sx = ((d - 1) * A).log()
        # (e^{u}-1)/u in [1, (1+e^{u})/2] for u >= 0 (mean of the convex function e^{v} over [0,u]);  u = theta*s_x
        umax = U(theta) * U(sx)
        Iplus = W * sx * iv(arb(1), U((1 + umax.exp()) / 2)) + C1 * (-(1 - theta) * sx).exp() / (1 - theta) \
            + C2 * (-(e2 - 1 - theta) * sx).exp() / (e2 - 1 - theta)
        kap_hi = W + (1 + theta) * Iplus
        if disc:
            sxD = sx + mu                                         # s_x = mu*S in [ln((d-1)A), ln((d-1)A) + mu]
            umaxD = U(theta) * U(sxD) / (1 - U(mu * nu1))
            IplusD = W * sxD / (1 - mu * nu1) * iv(arb(1), U((1 + umaxD.exp()) / 2)) + C1 * (-(1 - theta) * sx).exp() / (1 - theta) \
                + C2 * (-(e2 - 1 - theta) * sx).exp() / (e2 - 1 - theta)
            kap_hi = W / (1 - mu * nu1) + (1 + theta) * IplusD
        # P_rest e^{nu1 t}/(a1 y)
        cbeta = 2 * W + 2 * C1 * e2 / (e2 - 2)
        B2_y = cbeta * E['betahat_over_y']
        # near poles:  rho_i N_i(t)/y  with  N_2 <= 2C1 min(t, 1/theta_2) + C2 eps2/(eps2-2-th2bar),
        #                                   N_3 <= 2C1/(1+th3lo) + C2 eps2/(eps2-3-th3bar)
        th2 = E['th2']; hi2 = U(E['th2_over_y'])
        D2 = 2 * W2
        B2_y += E['rho2_over_y'] * C2 * e2 / (e2 - 2 - th2)
        cand = U(hi2 * hi2 * yt / D2)                              # (rho2/y) * t  <= (th2bar/y)^2 (y t)/D2
        if disc:
            cand = U(cand / (1 - 2 * mu))                          # tau -> tau/(1-2mu)
        if E['th2lo'] > 0:
            c1 = U(E['rho2_over_y'] / E['th2lo'])                  # (rho2/y) / theta_2
            if c1 < cand:
                cand = c1
        B2_y += 2 * C1 * cand
        if d == 3:
            th3 = E['th3']
            B2_y += E['rho3_over_y'] * (2 * C1 / (1 + E['th3lo']) + C2 * e2 / (e2 - 3 - th3))
        Be_y = C2 * e2 * (y * E['pia_hi'] + yt * E['Rtot_hi']) * (-2 * delta).exp() * (4 * xi * yt).exp() / W ** 2
        if disc:
            Be_y = Be_y / (1 - e2 * mu)                            # tau R_tot -> tau R_tot/(1 - eps2 mu)
        Prp = em_d * th_t.exp() * (B2_y + Be_y) / (W * E['a1_over_y'])
        Prp = iv(arb(0), U(Prp))
        # E_-^lo / y
        l_over_y = em_d * (A - g2 * iv(arb(0), U(y / W * em_d * eps_t))) / W
        G = arb(1) if d == 2 else (2 - y * l_over_y)
        Em_lo_y = em_d * nu0_t.exp() * l_over_y * G / (r0 * m0)
        if disc:
            # positive terms of W g(t)/E_{nu0}:  (coef) A E_2/E_{nu0} >= (coef) A e^{-(2-nu0)tau} chi2,  1-chi2 <= 2(2-nu0) mu tau/(1-2mu);
            # the term +2 A gam2 E_{2+eps2}/E_{nu0} (d=3) is dropped.
            coef = 1 if d == 2 else 2
            base = A * (-2 * delta).exp() * nu0_t.exp() / (W * r0 * m0)        # W A e^{-(2-nu0)tau}/(nu0 m0 y)
            loss = coef * base * (2 * (2 - nu0) * mut / (1 - 2 * mu))
            if d == 3:
                loss += 2 * g2 * base * (y / W) ** 2 * (-2 * delta).exp() * eps_t
            Em_lo_y = Em_lo_y - iv(arb(0), U(loss))
        Em_lo_y = iv(arb(0), U(Em_lo_y)) if not L(Em_lo_y) > 0 else Em_lo_y
        if nounimodal:
            Em_lo_y = arb(0)                                       # region A argument uses A(t) < 1
            Prp = Prp + nqP
        T6 = log1p_over(kap_hi * thy / W + Prp, y) - log1p_over(iv(L(Em_lo_y), L(Em_lo_y)), y)
        aux = dict(kap_hi=kap_hi, Prp=Prp, Em_lo_y=Em_lo_y)
    else:
        # kappa_lo(t)
        small = y / W * em_d * th_t.exp()                         # e^{-(1-theta) t}
        rbar_e = small * (C1 + C2 * eps_t)                        # rbar(t) e^{(1+theta) t}

        def F(a, exp_at):                                         # (1 - e^{-a t})/a  with e^{-a t} given as ball
            return (1 - exp_at) / a
        e1 = small                                                # e^{-(1-th)t}
        tiny = iv(arb(0), U(small * eps_t))                       # e^{-(eps2-th)t} = e^{-(1-th)t} e^{-(eps2-1)t} <= small*eps_t
        dl = (mu / (1 - mu * (1 + theta))) if disc else arb(0)       # discrete loss in each positive sum (Lemma 7.6)
        if d == 2:
            integ = A * (F(1 - theta, e1) - dl) - g2 * F(e2 - theta, iv(arb(0), U(tiny)))
        else:
            e2t = iv(arb(0), U(y / W * em_d * small))            # e^{-(2-th) t} = e^{-t} e^{-(1-th) t}
            integ = (2 * A * (F(1 - theta, e1) - dl) - 2 * g2 * F(e2 - theta, iv(arb(0), U(tiny))) - A * A * F(2 - theta, e2t)
                     + 2 * A * g2 * (F(1 + e2 - theta, iv(arb(0), U(tiny))) - dl) - g2 * g2 * F(2 * e2 - theta, iv(arb(0), U(tiny))))
        kap_lo = W - rbar_e + (1 + theta) * W * integ
        kl = L(kap_lo)
        kap_lo = iv(kl if kl > 0 else arb(0), max(U(kap_lo), arb(0)))
        Em_hi_y = (-2 * delta).exp() * nu0_t.exp() * (2 * C1 / (2 - nu0) + C2 * e2 * eps_t / (e2 - nu0)) / (W ** 2 * r0 * m0)
        inner = L(kap_lo * thy / W - nqP)
        assert L(1 + inner * U(y)) > 0
        T6 = log1p_over(iv(inner, inner), y) - log1p_over(iv(U(Em_hi_y), U(Em_hi_y)), y)
        aux = dict(kap_lo=kap_lo, Em_hi_y=Em_hi_y)
    total = T1 + T13 + T2 + T4 + T5 + T6 + TD
    bl = L(b) if L(b) > 0 else arb(0)
    aux['shift_lo'] = L(bl * L(1 + theta - nu0) / U(y))           # the term + b (1+theta-nu0)/y  of S/y  (lower bound)
    aux['yt'] = yt
    aux.update(T1=T1, T13=T13, T2=T2, T4=T4, T5=T5, T6=T6, TD=TD, c_ball=c, ytcf=ytcf, delta=delta,
               factor=(1 + theta - nu0) * yp_over_y)
    return total, aux


# ----------------------------------------------------------------------------- search for the constants c_plus, c_minus
def find_c(E, side, margin=0.0, c_start=-8.0, c_max=40.0, step=0.5, **kw):
    """smallest c (on a grid of 1e-3, found by scanning + bisection) for which the sign condition is VERIFIED in ball arithmetic:
       side '+':  sup S^+(tau_cf + c y')/y < 0;     side '-':  inf S^-(tau_cf - c y')/y > 0.
    Returns (c + margin rounded up to 1e-3, True) after a final verification at the returned value, or (None, False)."""
    def ok(c):
        try:
            tot, aux = Psi(E, c if side == '+' else -c, side, **kw)
        except AssertionError:
            return False
        return bool(U(tot) < 0) if side == '+' else bool(L(tot) > 0)
    c = c_start
    while c <= c_max and not ok(c):
        c += step
    if c > c_max:
        return None, False
    lo_, hi_ = c - step, c
    for _ in range(12):
        mid = 0.5 * (lo_ + hi_)
        if ok(mid):
            hi_ = mid
        else:
            lo_ = mid
    cfin = math.ceil((hi_ + margin) * 1000 - 1e-9) / 1000.0
    return cfin, ok(cfin)


# ----------------------------------------------------------------------------- Section 7.5: the four regions (no unimodality)
TAU_E = {2: '1.2', 3: '1.8'}
B_B = {2: '2.4', 3: '3.0'}


def regions(E, c_minus, qmax, mesh_ratio=1.25, max_step=0.06, quni=False, c_plus=None):
    """checks of Theorem 7.13 for one block: regions B, C, D (region A = the '+' sign check with nounimodal=True).
    Returns dict of booleans and margins."""
    d, y = E['d'], E['y']
    W, A, e2, g2, C1, C2 = E['W'], E['A'], E['e2'], E['g2'], E['C1'], E['C2']
    r0, nu0, theta, thy, xi, EU = E['r0'], E['nu0'], E['theta'], E['th_over_y'], E['xi'], E['EU']
    qm = arb(str(qmax)); fq = 2 * qm
    if quni:
        assert float(qmax) == 1.0 and E.get('quni', False)
    bB = arb(B_B[d]); tauE = arb(TAU_E[d])
    out = {}
    kw = dict(disc=True, qmax=qmax, nounimodal=True, quni=quni)
    Xl = 1 / U(y)
    tau_cf_lo = L(Xl * (2 * d * Xl).log() / (Xl + 2 * d - 1))       # tau_cf >= tau_cf(X = 1/yb)  (increasing in X)
    if c_plus is not None:
        # hypotheses of Theorem 7.13 on t_- < t_+ and mu t_+ >= 1:
        #   t_- <= (tau_cf - c_- y')/mu < (tau_cf + c_+ y')/mu <= t_+  iff  c_+ + c_- > 0;
        #   mu t_+ >= tau_cf + c_+ y' >= tau_cf_lo - max(-c_+, 0) yb >= 1
        cpl = arb(repr(c_plus)); cmi = arb(repr(c_minus))
        assert L(cpl + cmi) > 0
        cpneg = U(-cpl) if U(-cpl) > 0 else arb(0)
        assert L(tau_cf_lo - cpneg * U(y)) >= 1
        out['tplus_ok'] = True
    # ---- region B: shifts b in [0, bB]
    okB = True; b0 = arb(0); step = arb('0.01'); worst = None
    while b0 < bB:
        b1 = b0 + step
        if b1 > bB:
            b1 = bB
        tot, aux = Psi(E, -c_minus, '-', shift=iv(b0, b1), **kw)
        val = L(tot) + aux['shift_lo']
        worst = val if worst is None or val < worst else worst
        if not (val > 0):
            okB = False; break
        b0 = b1; step = step * arb(str(mesh_ratio))
        if step > arb(str(max_step)):
            step = arb(str(max_step))
    out['B'] = okB; out['B_margin'] = float(worst)
    # ---- quantities at t_B = tau_cf - c_minus y' - bB  and at t_minus
    totB, auxB = Psi(E, -c_minus, '-', shift=bB, **kw)
    ytB, deltaB = auxB['yt'], auxB['delta']
    mu = iv(arb(0), U(fq * xi * y / d))
    mutB = iv(arb(0), U(fq * xi * ytB / d))
    nu1 = 1 + theta
    # region C needs tau_E <= tau_B  (i.e. y*tau_B >= tau_E * y)
    # tau_E + mu <= tau_B = tau_cf - c_minus y' - bB:  tau_cf >= tau_cf(X = 1/yb)  (increasing in X),  c_minus y' <= c_minus yb
    cpos = arb(repr(c_minus)) if c_minus > 0 else arb(0)
    okC = bool(tau_cf_lo - U(cpos * U(y)) - bB > tauE + U(mu))
    # B_1(t)/y >= a1/y * nu1/W * E_{nu1}/E_1 - (rho0 nu0 / y) (E_{nu0}/E_1)/(1-nu0),  evaluated at t_B (B_1 is decreasing in t)
    B1_y = E['a1_over_y'] * nu1 / W * (-(thy * ytB) / (1 - nu1 * mu)).exp() \
        - r0 ** 2 * W * (deltaB - r0 * ytB + mutB * (1 - nu0) / (1 - mu)).exp() / (1 - nu0)
    cE = W - C1 * (-tauE).exp() - C2 * (-(e2 - 1) * tauE).exp()
    if quni:
        # Lemma 7.16: (pi_a/y) q/(2(1-q) - a mu) <= qu_pB_over_y[a]
        om_y = E['qu_pB_over_y'][1] / 2
        rhsC = (E['qu_pB_over_y'][2] * C1 * (-tauE).exp() + E['qu_pB_over_y']['e2'] * C2 * (e2 - 1) * (-(e2 - 1) * tauE).exp()) / 2
    else:
        om_y = E['pia_over_y_hi'] * (qm / 2) / (2 * (1 - qm) - mu)
        rhsC = E['pia_over_y_hi'] * (qm / 2) * (C1 * (-tauE).exp() / (2 * (1 - qm) - 2 * mu)
                                                 + C2 * (e2 - 1) * (-(e2 - 1) * tauE).exp() / (2 * (1 - qm) - e2 * mu))
    okC = okC and bool(L(cE) > 0) and bool(L(B1_y - om_y) > 0) and bool(L(cE) * L(B1_y - om_y) > U(rhsC))
    out['C'] = okC; out['C_margin'] = float(L(cE) * L(B1_y - om_y) - U(rhsC)); out['cE'] = float(L(cE)); out['B1_y'] = float(L(B1_y))
    # ---- region D
    tauEp = tauE + mu                                             # mu t_E in [tau_E, tau_E + mu)
    Qlo = d * (d - 1) * (A * A * (-2 * tauEp / (1 - 2 * mu)).exp() / 2 - 2 * A * g2 * e2 * (-(1 + e2) * tauE).exp() / (1 + e2)
                         - (d - 2) * A ** 3 * (-3 * tauE).exp() / 3)
    if quni:
        negv_rho0 = E['qu_pS_over_y'] / 4 * E['inv_b0_hi'] / L(r0)          # Lemma 7.16: (pi_a/y) q/(1-q) <= qu_pS_over_y
    else:
        negv_rho0 = E['pia_over_y_hi'] * qm / (4 * (1 - qm)) * E['inv_b0_hi'] / L(r0)
    lhs = 1 - L(Qlo) + U(nu0 / (1 - nu0)) + 2 * U(negv_rho0)
    tot0, aux0 = Psi(E, -c_minus, '-', **kw)
    yt0, delta0 = aux0['yt'], aux0['delta']
    # t_ref = t_minus:  E_{nu0}(t_ref) >= exp(-nu0 tau/(1-nu0 mu)),  W E_1(t_ref) <= W e^{-(tau - mu)} = y e^{-delta} e^{mu}
    rhs = (-(r0 * yt0) / (1 - nu0 * mu)).exp() * (1 - y * (-delta0).exp() * mu.exp())
    okD = bool(U(lhs) < L(rhs))
    out['D'] = okD; out['D_margin'] = float(L(rhs) - U(lhs))
    return out
