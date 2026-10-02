#!/usr/bin/env python
"""appA_proofs_check.py -- numerical check of every displayed identity of Supplementary Section S3
("Proofs of the mean first-passage time formulas") in the ZERO-BASED form printed there.

Model (built here site by site from the rules, not from any formula):
  lazy nearest-neighbour walk on {0,...,N-1}^d; with probability 1-q the walker rests,
  otherwise one of the 2d directions is chosen uniformly and a move that would leave the
  lattice is cancelled.

Reference values
  * exact rational solves of (I - Q) h = 1 (fractions.Fraction, Gaussian elimination),
  * float64 sparse direct solves for larger N.

Formulas are evaluated with mpmath at 40 significant digits.  Every check is recorded with
the label of the equation it tests, the number of cases, the largest deviation and a pass flag.

Output: ../data/appA_proofs_check.json
Usage : python appA_proofs_check.py            (about one minute)
"""
from __future__ import annotations
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository

import itertools
import json
import math
import os
import time
from fractions import Fraction

import mpmath as mp
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = _os.path.join(_R, 'data', 'article', 'appA_proofs_check.json')
ROOT = _R          # 

mp.mp.dps = 40
TOL_MP = mp.mpf(10) ** (-30)      # formula (40 digits) against an exact rational value
TOL_FL = 1e-9                     # formula against a float64 direct solve
T0 = time.time()
CHECKS = []
rng = np.random.default_rng(20261001)


def record(label, statement, cases, err, tol, extra=None):
    err = float(err)
    row = {"label": label, "statement": statement, "cases": int(cases),
           "max_deviation": err, "tolerance": float(tol), "passed": bool(err <= float(tol))}
    if extra:
        row.update(extra)
    CHECKS.append(row)
    print(("PASS " if row["passed"] else "FAIL ") + f"{label:34s} cases={cases:6d} max dev={err:.2e}  {statement}")


# ---------------------------------------------------------------------------
# model, built from the rules
# ---------------------------------------------------------------------------
def sites(N, d):
    return list(itertools.product(range(N), repeat=d))


def build_P_rules(N, d, q):
    """Dense transition matrix from the rules (q may be a float or a Fraction)."""
    S = sites(N, d)
    idx = {x: i for i, x in enumerate(S)}
    n = len(S)
    zero = q * 0
    P = [[zero for _ in range(n)] for _ in range(n)]
    for x in S:
        i = idx[x]
        P[i][i] += 1 - q
        for ax in range(d):
            for sgn in (-1, 1):
                y = list(x)
                y[ax] += sgn
                if 0 <= y[ax] < N:
                    P[i][idx[tuple(y)]] += q / (2 * d)
                else:
                    P[i][i] += q / (2 * d)         # cancelled move
    return P, S, idx


def solve_fraction(A, b):
    """Gaussian elimination with exact rationals."""
    n = len(A)
    M = [list(map(Fraction, A[i])) + [Fraction(b[i])] for i in range(n)]
    for c in range(n):
        p = next(r for r in range(c, n) if M[r][c] != 0)
        M[c], M[p] = M[p], M[c]
        inv = 1 / M[c][c]
        M[c] = [v * inv for v in M[c]]
        for r in range(n):
            if r != c and M[r][c] != 0:
                f = M[r][c]
                Mr, Mc = M[r], M[c]
                M[r] = [Mr[j] - f * Mc[j] if Mc[j] != 0 else Mr[j] for j in range(n + 1)]
    return [M[i][n] for i in range(n)]


_exact_cache = {}


def exact_hit(N, d, a, q=Fraction(1)):
    """Exact mean hitting times of a from every start: dict start -> Fraction (T, not q*T)."""
    key = (N, d, tuple(a), q)
    if key in _exact_cache:
        return _exact_cache[key]
    P, S, idx = build_P_rules(N, d, q)
    ia = idx[tuple(a)]
    keep = [i for i in range(len(S)) if i != ia]
    A = [[(1 if i == j else 0) - P[i][j] for j in keep] for i in keep]
    h = solve_fraction(A, [1] * len(keep))
    res = {S[i]: h[r] for r, i in enumerate(keep)}
    res[tuple(a)] = Fraction(0)
    _exact_cache[key] = res
    return res


def build_P_sparse(N, d, q):
    n = N ** d
    idx = np.arange(n).reshape((N,) * d)
    rows_, cols_, vals_ = [], [], []
    diag = np.full(n, 1.0 - q)
    for ax in range(d):
        for sgn in (-1, 1):
            src = [slice(None)] * d
            dst = [slice(None)] * d
            if sgn == 1:
                src[ax] = slice(0, N - 1); dst[ax] = slice(1, N)
            else:
                src[ax] = slice(1, N); dst[ax] = slice(0, N - 1)
            s_ = idx[tuple(src)].ravel(); t_ = idx[tuple(dst)].ravel()
            rows_.append(s_); cols_.append(t_); vals_.append(np.full(s_.size, q / (2 * d)))
            wall = [slice(None)] * d
            wall[ax] = N - 1 if sgn == 1 else 0
            diag[idx[tuple(wall)].ravel()] += q / (2 * d)
    P = sp.coo_matrix((np.concatenate(vals_), (np.concatenate(rows_), np.concatenate(cols_))), shape=(n, n)).tocsr()
    return P + sp.diags(diag), idx


def float_hit(N, d, a, q):
    """float64 sparse direct solve: array of hitting times of a, indexed like idx."""
    P, idx = build_P_sparse(N, d, q)
    n = N ** d
    ia = int(idx[tuple(a)])
    keep = np.array([i for i in range(n) if i != ia])
    Q = P[keep][:, keep]
    m = spla.spsolve((sp.identity(n - 1) - Q).tocsc(), np.ones(n - 1))
    full = np.zeros(n)
    full[keep] = m
    return full, idx


# ---------------------------------------------------------------------------
# notation of the article (zero-based)
# ---------------------------------------------------------------------------
def al(k):
    return 1 if k == 0 else 2


def vth(k, N):                       # \vartheta_k = pi k / N
    return mp.pi * k / N


def th(k, m, N):                     # \theta_k(m) = pi k (2m+1) / (2N)
    return mp.pi * k * (2 * m + 1) / (2 * N)


def c(k, N):
    return mp.cos(mp.pi * k / (2 * N))


def s(k, N):
    return mp.sin(mp.pi * k / (2 * N))


def sig(k, N):
    return 2 - mp.cos(vth(k, N))


def phi(k, N):
    return 2 * mp.asinh(s(k, N))


def h1(m, m2, N):                    # eq. (s4_mfpt:eq-1d)
    return (m2 - m) * (m2 + m + 1) if m <= m2 else (m - m2) * (2 * N - 1 - m - m2)


def cosine_sum(N, d, o, a):          # eq. (s4_mfpt:eq-cosine): q T_{o->a}
    tot = mp.mpf(0)
    for k in itertools.product(range(N), repeat=d):
        if not any(k):
            continue
        A = 1
        ca = mp.mpf(1); co = mp.mpf(1); den = mp.mpf(0)
        for ki, oi, ai in zip(k, o, a):
            A *= al(ki)
            ca *= mp.cos(th(ki, ai, N)) ** 2
            co *= mp.cos(th(ki, oi, N)) * mp.cos(th(ki, ai, N))
            den += s(ki, N) ** 2
        tot += A * (ca - co) / den
    return mp.mpf(d) / 2 * tot


def cosine_cc(N, d):                 # eq. (s4_mfpt:eq-cosine-cc): q T_N^{(d)}
    tot = mp.mpf(0)
    for k in itertools.product(range(N), repeat=d):
        if sum(k) % 2 == 0:
            continue
        num = mp.mpf(1); den = mp.mpf(0)
        for ki in k:
            num *= al(ki) * c(ki, N) ** 2
            den += s(ki, N) ** 2
        tot += num / den
    return d * tot


def G_direct(N, k):                  # definition in eq. (appA_proofs:eq-split)
    return mp.fsum(al(j) * c(j, N) ** 2 * (1 - (-1) ** (k + j)) / (sig(k, N) - mp.cos(vth(j, N))) for j in range(N))


def G_closed(N, k):                  # eq. (appA_proofs:eq-Gk)
    f = phi(k, N)
    return N * (1 + sig(k, N)) * (mp.cosh(N * f) - (-1) ** k) / (mp.sinh(f) * mp.sinh(N * f)) - N


def Bk(N, k):
    f = phi(k, N)
    return mp.cosh(N * f) + mp.cosh((N - 1) * f) - (-1) ** k * (1 + mp.cosh(f))


def form_a(N):                       # eq. (s4_mfpt:eq-single-a)
    return 2 * N * (N - 1) + 4 * N * mp.fsum(
        c(k, N) ** 2 * Bk(N, k) / (mp.sinh(phi(k, N)) * mp.sinh(N * phi(k, N))) for k in range(1, N))


def gk(N, k):
    x = N * mp.asinh(s(k, N))
    return mp.coth(x) if k % 2 else mp.tanh(x)


def form_b(N):                       # eq. (s4_mfpt:eq-single-S)
    return 4 * N * mp.fsum(c(k, N) ** 2 * mp.sqrt(1 + s(k, N) ** 2) / s(k, N) * gk(N, k) for k in range(1, N))


def S_odd(N):                        # eq. (s4_mfpt:eq-half-def)
    return mp.fsum(c(k, N) ** 2 * mp.coth(phi(k, N) / 2) * mp.coth(N * phi(k, N) / 2) for k in range(1, N, 2))


def S_even(N):
    return mp.fsum(c(k, N) ** 2 * mp.coth(phi(k, N) / 2) * mp.tanh(N * phi(k, N) / 2) for k in range(2, N, 2))


def Psi(k, m, m2, N):                # eq. (s4_mfpt:eq-Psi)
    f = phi(k, N)
    return (mp.cosh((N - abs(m - m2)) * f) + mp.cosh((N - 1 - m - m2) * f)) / (mp.sinh(f) * mp.sinh(N * f))


def Ck(k, m, m2, N):
    return mp.cos(th(k, m, N)) * mp.cos(th(k, m2, N))


def pair(N, o, a):                   # eq. (s4_mfpt:eq-pair): q T_{o->a}
    return 2 * h1(o[1], a[1], N) + 4 * N * mp.fsum(
        Ck(k, a[0], a[0], N) * Psi(k, a[1], a[1], N) - Ck(k, o[0], a[0], N) * Psi(k, o[1], a[1], N)
        for k in range(1, N))


def phi2(k, j, N):                   # \varphi_{kj} of Proposition s4_mfpt:prop-3d
    return 2 * mp.asinh(mp.sqrt(s(k, N) ** 2 + s(j, N) ** 2))


def form_3d(N):                      # eq. (s4_mfpt:eq-3d)
    tot = mp.mpf(0)
    for k in range(N):
        for j in range(N):
            if k == 0 and j == 0:
                continue
            f = phi2(k, j, N)
            g = mp.coth(N * f / 2) if (k + j) % 2 else mp.tanh(N * f / 2)
            tot += al(k) * al(j) * c(k, N) ** 2 * c(j, N) ** 2 * mp.coth(f / 2) * g
    return 3 * N * (tot - N * (N - 1))


def relerr(x, ref):
    ref = mp.mpf(ref.numerator) / ref.denominator if isinstance(ref, Fraction) else mp.mpf(ref)
    return abs(mp.mpf(x) - ref) / max(abs(ref), mp.mpf(1))


def fr(x):
    return mp.mpf(x.numerator) / x.denominator


# ---------------------------------------------------------------------------
# 1. Lemma appA_proofs:lem-spectrum
# ---------------------------------------------------------------------------
def T_chain(N):
    T = np.zeros((N, N))
    for m in range(N):
        for dm in (-1, 1):
            if 0 <= m + dm < N:
                T[m, m + dm] = 0.5
            else:
                T[m, m] += 0.5
    return T


def psi_matrix(N):
    m = np.arange(N)
    Psi_ = np.zeros((N, N))
    for k in range(N):
        Psi_[:, k] = math.sqrt((1.0 if k == 0 else 2.0) / N) * np.cos(math.pi * k * (2 * m + 1) / (2 * N))
    return Psi_


NS_SMALL = [2, 3, 4, 5, 6, 7, 8, 9, 16, 35, 36]
e_eig = e_orth = e_bdry = e_corner = 0.0
ncase = 0
for N in NS_SMALL:
    T = T_chain(N)
    Ps = psi_matrix(N)
    lam = np.cos(np.pi * np.arange(N) / N)
    e_eig = max(e_eig, np.abs(T @ Ps - Ps * lam[None, :]).max())
    e_orth = max(e_orth, np.abs(Ps.T @ Ps - np.eye(N)).max())
    for k in range(N):
        u = lambda m: math.cos(math.pi * k * (2 * m + 1) / (2 * N))
        e_bdry = max(e_bdry, abs(u(-1) - u(0)), abs(u(N) - u(N - 1)))
        ck_ = math.cos(math.pi * k / (2 * N))
        e_corner = max(e_corner, abs(u(0) - ck_), abs(u(N - 1) - (-1) ** k * ck_))
        ncase += 1
record("appA_proofs:lem-spectrum", "T psi_k = cos(pi k/N) psi_k (float64, absolute)", ncase, e_eig, 1e-13)
record("appA_proofs:lem-spectrum", "psi_k orthonormal", len(NS_SMALL), e_orth, 1e-13)
record("appA_proofs:lem-spectrum", "u_k(-1)=u_k(0), u_k(N)=u_k(N-1)", ncase, e_bdry, 1e-13)
record("appA_proofs:eq-corner-cos", "cos theta_k(0)=c_k, cos theta_k(N-1)=(-1)^k c_k", ncase, e_corner, 1e-13)

e_kron = e_gap = e_pinv = 0.0
ncase = 0
for d, Ns in ((1, (2, 3, 6, 9)), (2, (2, 3, 4, 5)), (3, (2, 3, 4))):
    for N in Ns:
        for q in (0.8, 0.3, 1.0):
            P, S, idx = build_P_rules(N, d, q)
            P = np.array(P, dtype=float)
            T = T_chain(N)
            K = (1 - q) * np.eye(N ** d)
            for i in range(d):
                mats = [np.eye(N)] * d
                mats[i] = T
                Ti = mats[0]
                for M_ in mats[1:]:
                    Ti = np.kron(Ti, M_)
                K = K + (q / d) * Ti
            e_kron = max(e_kron, np.abs(P - K).max())
            Ps1 = psi_matrix(N)
            V = Ps1
            for _ in range(d - 1):
                V = np.kron(V, Ps1)
            ks = sites(N, d)
            gap = np.array([(2 * q / d) * sum(math.sin(math.pi * ki / (2 * N)) ** 2 for ki in k) for k in ks])
            e_gap = max(e_gap, np.abs((np.eye(N ** d) - P) @ V - V * gap[None, :]).max())
            inv = np.where(gap > 1e-14, 1.0 / np.where(gap > 1e-14, gap, 1.0), 0.0)
            Lp = (V * inv[None, :]) @ V.T
            ref = np.linalg.pinv(np.eye(N ** d) - P)
            e_pinv = max(e_pinv, np.abs(Lp - ref).max() / np.abs(ref).max())
            ncase += 1
record("appA_proofs:lem-spectrum", "P = (1-q) I + (q/d) sum_i T^(i), P built from the rules, d = 1,2,3", ncase, e_kron, 1e-14)
record("appA_proofs:lem-spectrum", "(I-P) Psi_k = (2q/d) sum_i s_{k_i}^2 Psi_k, d = 1,2,3", ncase, e_gap, 1e-13)
record("appA_proofs:eq-Lplus", "spectral form of L^+ against numpy pinv (relative)", ncase, e_pinv, 1e-10)

# ---------------------------------------------------------------------------
# 2. Theorem s4_mfpt:thm-cosine (eq-cosine, eq-cosine-cc, eq-1d) against exact rational solves
# ---------------------------------------------------------------------------
err = mp.mpf(0); ncase = 0
for N in range(2, 8):                                   # d = 1, all ordered pairs, two activities
    for q in (Fraction(1), Fraction(4, 5)):
        for a in range(N):
            ex = exact_hit(N, 1, (a,), q)
            for o in range(N):
                if o == a:
                    continue
                err = max(err, relerr(cosine_sum(N, 1, (o,), (a,)), q * ex[(o,)]))
                err = max(err, relerr(h1(o, a, N), q * ex[(o,)]))
                ncase += 1
record("s4_mfpt:eq-cosine / eq-1d", "d=1: cosine sum = h_1(m,m') = q T (exact), all ordered pairs, N=2..7, q=1 and 4/5", ncase, err, TOL_MP)

err = mp.mpf(0); ncase = 0
for N in (2, 3, 4):                                     # d = 2, all ordered pairs
    for q in (Fraction(1), Fraction(3, 7)):
        for a in sites(N, 2):
            ex = exact_hit(N, 2, a, q)
            for o in sites(N, 2):
                if o == a:
                    continue
                err = max(err, relerr(cosine_sum(N, 2, o, a), q * ex[o]))
                ncase += 1
for N in (5, 6, 7, 8):                                  # d = 2, random targets, six starts each
    for _ in range(3):
        a = tuple(int(v) for v in rng.integers(0, N, 2))
        ex = exact_hit(N, 2, a)
        for _ in range(6):
            o = tuple(int(v) for v in rng.integers(0, N, 2))
            if o == a:
                continue
            err = max(err, relerr(cosine_sum(N, 2, o, a), ex[o]))
            ncase += 1
record("s4_mfpt:eq-cosine", "d=2: cosine double sum = q T (exact): all pairs N=2..4 (q=1, 3/7), random pairs N=5..8", ncase, err, TOL_MP)

err = mp.mpf(0); ncase = 0
for N in (2, 3):                                        # d = 3
    targets = sites(N, 3) if N == 2 else [tuple(int(v) for v in rng.integers(0, N, 3)) for _ in range(4)]
    for a in targets:
        ex = exact_hit(N, 3, a, Fraction(4, 5))
        for o in sites(N, 3):
            if o == a:
                continue
            err = max(err, relerr(cosine_sum(N, 3, o, a), Fraction(4, 5) * ex[o]))
            ncase += 1
record("s4_mfpt:eq-cosine", "d=3: cosine triple sum = q T (exact), N=2 all pairs, N=3 four targets, q=4/5", ncase, err, TOL_MP)

EXACT_CC = {}
err = mp.mpf(0); ncase = 0
for d, Ns in ((1, range(2, 13)), (2, range(2, 10)), (3, (2, 3, 4))):
    for N in Ns:
        ex = exact_hit(N, d, (N - 1,) * d)[(0,) * d]
        EXACT_CC[(d, N)] = ex
        err = max(err, relerr(cosine_cc(N, d), ex))
        if d == 1:
            err = max(err, relerr(N * (N - 1), ex))
        ncase += 1
record("s4_mfpt:eq-cosine-cc", "corner-to-corner odd-sector sum = q T_N^(d) (exact): d=1 N=2..12, d=2 N=2..9, d=3 N=2..4", ncase, err, TOL_MP)

# ---------------------------------------------------------------------------
# 3. Lemma appA_proofs:lem-ring
# ---------------------------------------------------------------------------
err = mp.mpf(0); ncase = 0
with mp.workdps(100):      # the left side cancels down to ~1e-25 for sigma = 10, M = 40, m = M/2: use 100 digits here
    for M in list(range(1, 13)) + [25, 40]:
        for sg in (mp.mpf("1.001"), mp.mpf("1.5"), mp.mpf(3), mp.mpf(10)):
            f = mp.acosh(sg)
            for m in range(M + 1):
                lhs = mp.fsum(mp.cos(2 * mp.pi * l * m / M) / (sg - mp.cos(2 * mp.pi * l / M)) for l in range(M))
                rhs = M * mp.cosh((mp.mpf(M) / 2 - m) * f) / (mp.sinh(f) * mp.sinh(M * f / 2))
                err = max(err, abs(lhs - rhs) / abs(rhs))
                ncase += 1
record("appA_proofs:eq-ring", "ring Green's function, M=1..12,25,40, 0<=m<=M, sigma=1.001,1.5,3,10 (relative, 100 digits)", ncase, err, mp.mpf(10) ** (-28))


def ring_half_lhs(N, m, sg):
    return mp.fsum(al(j) * mp.cos(m * vth(j, N)) / (sg - mp.cos(vth(j, N))) for j in range(N))


def ring_half_rhs(N, m, sg):
    f = mp.acosh(sg)
    return 2 * N * mp.cosh((N - m) * f) / (mp.sinh(f) * mp.sinh(N * f)) - (-1) ** m / (sg + 1)


err = mp.mpf(0); errE1 = mp.mpf(0); ncase = 0
viol = {}
for N in (2, 3, 4, 5, 6, 7, 8, 9, 16, 35):
    for sg in (mp.mpf("1.001"), mp.mpf("1.5"), mp.mpf(3), mp.mpf(10)):
        f = mp.acosh(sg)
        for m in range(2 * N + 1):
            lhs = ring_half_lhs(N, m, sg)
            err = max(err, abs(lhs - ring_half_rhs(N, m, sg)) / max(abs(lhs), 1))
            # form (E1): j >= 1 only, divided by 2
            lhsE = mp.fsum(mp.cos(m * vth(j, N)) / (sg - mp.cos(vth(j, N))) for j in range(1, N))
            rhsE = N * mp.cosh((N - m) * f) / (mp.sinh(f) * mp.sinh(N * f)) - (1 / (sg - 1) + (-1) ** m / (sg + 1)) / 2
            errE1 = max(errE1, abs(lhsE - rhsE) / max(abs(lhsE), 1))
            ncase += 1
    viol[str(N)] = {"m=2N+1": float(abs(ring_half_lhs(N, 2 * N + 1, mp.mpf(3)) - ring_half_rhs(N, 2 * N + 1, mp.mpf(3)))),
                    "m=-1": float(abs(ring_half_lhs(N, -1, mp.mpf(3)) - ring_half_rhs(N, -1, mp.mpf(3))))}
record("appA_proofs:eq-ring-half", "half-range form, N=2..9,16,35, 0<=m<=2N, four values of sigma", ncase, err, mp.mpf(10) ** (-28))
record("appA_proofs:eq-E1", "form with j>=1 only (Eq. (E1) of Giuggioli 2020), same cases", ncase, errE1, mp.mpf(10) ** (-28))
minviol = min(min(v.values()) for v in viol.values())
record("appA_proofs:eq-ring-half (range)", "identity is VIOLATED at m=2N+1 and m=-1 (sigma=3): smallest violation must exceed 1e-3",
       2 * len(viol), 0.0 if minviol > 1e-3 else 1.0, 0.5, {"violations_sigma_3": viol})

# ---------------------------------------------------------------------------
# 4. Lemma appA_proofs:lem-1d
# ---------------------------------------------------------------------------
err = mp.mpf(0); err2 = mp.mpf(0); ncase = 0
for N in range(2, 13):
    for a in range(N):
        ex = exact_hit(N, 1, (a,))            # q = 1 is the chain T itself
        for o in range(N):
            val = mp.fsum(al(j) * (mp.cos(th(j, a, N)) ** 2 - mp.cos(th(j, o, N)) * mp.cos(th(j, a, N)))
                          / (1 - mp.cos(vth(j, N))) for j in range(1, N))
            err = max(err, abs(h1(o, a, N) - fr(ex[(o,)])))
            err2 = max(err2, abs(val - h1(o, a, N)))
            ncase += 1
record("appA_proofs:lem-1d", "h_1(m,m') = exact hitting time of the chain T, all pairs, N=2..12 (absolute)", ncase, err, 0.0)
record("appA_proofs:eq-1d-sum", "spectral sum = h_1(m,m'), all pairs, N=2..12 (absolute)", ncase, err2, mp.mpf(10) ** (-30))
err = mp.mpf(0)
for N in range(2, 61):
    v = mp.fsum(mp.cot(mp.pi * j / (2 * N)) ** 2 for j in range(1, N, 2))
    err = max(err, abs(v - mp.mpf(N * (N - 1)) / 2))
record("appA_proofs:eq-cot", "sum over odd j of cot^2(pi j/2N) = N(N-1)/2, N=2..60 (absolute)", 59, err, mp.mpf(10) ** (-30))

# ---------------------------------------------------------------------------
# 5. Theorem s4_mfpt:thm-single
# ---------------------------------------------------------------------------
NS_MP = list(range(2, 13)) + [35, 36, 59, 60]
err_split = err_G = err_B = err_sum = err_alt = mp.mpf(0)
nG = 0
for N in NS_MP:
    split = N * (N - 1) + mp.fsum(2 * c(k, N) ** 2 * G_direct(N, k) for k in range(1, N))
    ref = fr(EXACT_CC[(2, N)]) if (2, N) in EXACT_CC else cosine_cc(N, 2)
    err_split = max(err_split, abs(split - ref / 2) / ref)
    err_sum = max(err_sum, abs(sum(al(j) for j in range(N)) - (2 * N - 1)),
                  abs(sum(al(j) * (-1) ** j for j in range(N)) + (-1) ** N))
    for k in range(1, N):
        f = phi(k, N)
        err_G = max(err_G, abs(G_direct(N, k) - G_closed(N, k)) / abs(G_closed(N, k)))
        b2 = (1 + sig(k, N)) * (mp.cosh(N * f) - (-1) ** k) - mp.sinh(f) * mp.sinh(N * f)
        err_B = max(err_B, abs(Bk(N, k) - b2) / abs(b2))
        err_B = max(err_B, abs(G_closed(N, k) - N * Bk(N, k) / (mp.sinh(f) * mp.sinh(N * f))) / abs(G_closed(N, k)))
        err_alt = max(err_alt, abs(G_closed(N, k) - N * (mp.coth(f / 2) * gk(N, k) - 1)) / abs(G_closed(N, k)),
                      abs(mp.coth(f / 2) - mp.sqrt(1 + s(k, N) ** 2) / s(k, N)), abs(mp.sinh(f / 2) - s(k, N)),
                      abs(mp.cosh(f) - sig(k, N)))
        nG += 1
record("appA_proofs:eq-split", "q T_N/2 = N(N-1) + sum_k 2 c_k^2 G_k (G_k summed directly), N=2..12,35,36,59,60", len(NS_MP), err_split, TOL_MP)
record("appA_proofs:eq-alpha-sums", "sum alpha_j = 2N-1, sum alpha_j (-1)^j = -(-1)^N (exact integers)", len(NS_MP), err_sum, 0.0)
record("appA_proofs:eq-Gk", "closed form of G_k against the direct sum over j, all k", nG, err_G, mp.mpf(10) ** (-28))
record("appA_proofs:eq-Bk", "B_k = (1+sigma_k)[cosh N phi_k - (-1)^k] - sinh phi_k sinh N phi_k and G_k = N B_k/(sinh sinh)", nG, err_B, mp.mpf(10) ** (-28))
record("appA_proofs:eq-Gk-coth", "G_k = N[coth(phi_k/2) g_k - 1]; sinh(phi_k/2)=s_k; coth(phi_k/2)=sqrt(1+s_k^2)/s_k; cosh phi_k = sigma_k", nG, err_alt, mp.mpf(10) ** (-28))

err_c = err_codd = err_cosodd = mp.mpf(0)
for N in range(2, 61):
    err_c = max(err_c, abs(mp.fsum(c(k, N) ** 2 for k in range(1, N)) - mp.mpf(N - 1) / 2))
    err_codd = max(err_codd, abs(mp.fsum(c(k, N) ** 2 for k in range(1, N, 2)) - mp.mpf(N) / 4))
    so = mp.fsum(mp.cos(vth(k, N)) for k in range(1, N, 2))
    err_cosodd = max(err_cosodd, abs(so - (mp.mpf(1) / 2 if N % 2 else 0)))
record("appA_proofs:eq-csum", "sum_{k=1}^{N-1} c_k^2 = (N-1)/2, N=2..60 (absolute)", 59, err_c, mp.mpf(10) ** (-30))
record("appA_proofs:eq-csum-odd", "sum over odd k of c_k^2 = N/4, N=2..60, both parities (absolute)", 59, err_codd, mp.mpf(10) ** (-30))
record("appA_proofs:eq-csum-odd", "sum over odd k of cos(pi k/N) = 0 (N even), 1/2 (N odd), N=2..60", 59, err_cosodd, mp.mpf(10) ** (-30))

err_a = err_b = err_c1 = err_c2 = err_diff = err_H = err_16 = mp.mpf(0)
table_exact = []
for N in range(2, 10):
    ex = EXACT_CC[(2, N)]
    err_a = max(err_a, relerr(form_a(N), ex))
    err_b = max(err_b, relerr(form_b(N), ex))
    err_c1 = max(err_c1, relerr(8 * N * S_odd(N) - 2 * N * N, ex))
    err_c2 = max(err_c2, relerr(2 * N * N + 8 * N * S_even(N), ex))
    table_exact.append({"N": N, "qT_N_exact": f"{ex.numerator}/{ex.denominator}" if ex.denominator != 1 else str(ex.numerator),
                        "qT_N_float": float(ex)})
record("s4_mfpt:eq-single-a", "form (a) = exact rational q T_N, N=2..9", 8, err_a, TOL_MP)
record("s4_mfpt:eq-single-S", "form (b) = exact rational q T_N, N=2..9", 8, err_b, TOL_MP)
record("s4_mfpt:eq-half", "8N S_odd - 2N^2 = exact rational q T_N, N=2..9", 8, err_c1, TOL_MP)
record("s4_mfpt:eq-half", "2N^2 + 8N S_even = exact rational q T_N, N=2..9", 8, err_c2, TOL_MP)

nH = 0
err_ab = mp.mpf(0)
for N in NS_MP:
    ref = cosine_cc(N, 2)
    b = form_b(N)
    err_ab = max(err_ab, abs(form_a(N) - ref) / ref, abs(b - ref) / ref,
                 abs(8 * N * S_odd(N) - 2 * N * N - ref) / ref, abs(2 * N * N + 8 * N * S_even(N) - ref) / ref)
    err_diff = max(err_diff, abs(S_odd(N) - S_even(N) - mp.mpf(N) / 2))
    tot16 = mp.mpf(0)
    for k in range(1, N, 2):
        Hd = mp.fsum(al(j) * c(j, N) ** 2 / (sig(k, N) - mp.cos(vth(j, N))) for j in range(0, N, 2))
        Hring = mp.fsum(mp.cos(mp.pi * l / N) ** 2 / (sig(k, N) - mp.cos(2 * mp.pi * l / N)) for l in range(N))
        Hc = mp.mpf(N) / 2 * (mp.coth(phi(k, N) / 2) * mp.coth(N * phi(k, N) / 2) - 1)
        err_H = max(err_H, abs(Hd - Hc) / abs(Hc), abs(Hring - Hc) / abs(Hc))
        tot16 += c(k, N) ** 2 * Hd
        nH += 1
    err_16 = max(err_16, abs(16 * tot16 - ref) / ref)
record("s4_mfpt:eq-single-a/S/half", "forms (a), (b), (c) = cosine double sum (40 digits), N=2..12,35,36,59,60", len(NS_MP), err_ab, mp.mpf(10) ** (-28))
record("s4_mfpt:eq-half", "S_odd - S_even = N/2 (absolute), same N", len(NS_MP), err_diff, mp.mpf(10) ** (-28))
record("appA_proofs:eq-Hk", "H_k: sum over even j = sum over Z_N = (N/2)[coth(phi_k/2) coth(N phi_k/2) - 1], odd k", nH, err_H, mp.mpf(10) ** (-28))
record("appA_proofs:eq-odd-even", "q T_N = 16 sum over odd k of c_k^2 H_k, same N", len(NS_MP), err_16, mp.mpf(10) ** (-28))

# float64 direct solves, both parities, larger N, q = 0.8
err = 0.0
rows_float = []
for N in (16, 35, 36, 100, 101):
    full, idx = float_hit(N, 2, (N - 1, N - 1), 0.8)
    ref = 0.8 * full[idx[0, 0]]
    vals = {"a": float(form_a(N)), "b": float(form_b(N)), "c_odd": float(8 * N * S_odd(N) - 2 * N * N),
            "c_even": float(2 * N * N + 8 * N * S_even(N))}
    e = max(abs(v / ref - 1) for v in vals.values())
    err = max(err, e)
    rows_float.append({"N": N, "qT_solve": ref, "max_rel_dev_forms": e})
record("s4_mfpt:thm-single", "forms (a), (b), (c) against float64 sparse solves, N=16,35,36,100,101, q=0.8", 5, err, TOL_FL)

# ---------------------------------------------------------------------------
# 6. Theorem s4_mfpt:thm-pair
# ---------------------------------------------------------------------------
err_prod = err_inner = mp.mpf(0); n1 = 0
for N in (2, 3, 4, 5, 8, 9):
    for m in range(N):
        for m2 in range(N):
            for k in range(1, N):
                for j in range(N):
                    lhs = mp.cos(th(j, m, N)) * mp.cos(th(j, m2, N))
                    rhs = (mp.cos(abs(m - m2) * vth(j, N)) + mp.cos((m + m2 + 1) * vth(j, N))) / 2
                    err_prod = max(err_prod, abs(lhs - rhs))
                inner = mp.fsum(al(j) * mp.cos(th(j, m, N)) * mp.cos(th(j, m2, N)) / (sig(k, N) - mp.cos(vth(j, N)))
                                for j in range(N))
                err_inner = max(err_inner, abs(inner - N * Psi(k, m, m2, N)) / abs(inner))
                n1 += 1
record("appA_proofs:eq-prod", "cos theta_j(m) cos theta_j(m') = [cos(|m-m'| vartheta_j) + cos((m+m'+1) vartheta_j)]/2", n1, err_prod, mp.mpf(10) ** (-35))
record("appA_proofs:eq-inner-pair", "sum_j alpha_j cos cos/(sigma_k - cos vartheta_j) = N Psi_k(m,m'), all m,m',k, N=2..5,8,9", n1, err_inner, mp.mpf(10) ** (-28))

err = mp.mpf(0); ncase = 0
for N in (2, 3, 4, 5):
    for q in (Fraction(1), Fraction(3, 7)):
        for a in sites(N, 2):
            ex = exact_hit(N, 2, a, q)
            for o in sites(N, 2):
                if o == a:
                    continue
                err = max(err, relerr(pair(N, o, a), q * ex[o]))
                ncase += 1
n_all = ncase
for N in (6, 7, 8, 9):
    for _ in range(3):
        a = tuple(int(v) for v in rng.integers(0, N, 2))
        ex = exact_hit(N, 2, a)
        for o in sites(N, 2):
            if o == a:
                continue
            err = max(err, relerr(pair(N, o, a), ex[o]))
            ncase += 1
record("s4_mfpt:eq-pair", f"pair formula = exact rational q T: all {n_all} ordered pairs N=2..5 (q=1, 3/7); every start for 12 random targets N=6..9",
       ncase, err, TOL_MP)

err = 0.0; ncase = 0
for N in (35, 36):
    for a in ((N - 1, N - 1), (8, 22), (0, 17)):
        full, idx = float_hit(N, 2, a, 0.8)
        for _ in range(12):
            o = tuple(int(v) for v in rng.integers(0, N, 2))
            if o == a:
                continue
            ref = 0.8 * full[idx[o]]
            err = max(err, abs(float(pair(N, o, a)) / ref - 1))
            ncase += 1
record("s4_mfpt:eq-pair", "pair formula against float64 sparse solves, N=35,36, three targets, random starts, q=0.8", ncase, err, TOL_FL)

err = mp.mpf(0)
for N in NS_MP:
    for k in range(1, N):
        f = phi(k, N); den = mp.sinh(f) * mp.sinh(N * f)
        err = max(err, abs(Psi(k, N - 1, N - 1, N) - (mp.cosh(N * f) + mp.cosh((N - 1) * f)) / den) * den,
                  abs(Psi(k, 0, N - 1, N) - (mp.cosh(f) + 1) / den) * den)
    err = max(err, abs(pair(N, (0, 0), (N - 1, N - 1)) - form_a(N)) / form_a(N))
record("appA_proofs:eq-pair-corner", "pair formula at opposite corners reduces to form (a)", len(NS_MP), err, mp.mpf(10) ** (-28))

err = mp.mpf(0); ncase = 0
for N in (2, 3, 8, 35):
    for k in range(1, N):
        f = phi(k, N)
        for m in range(2 * N + 1):
            lhs = mp.cosh((N - m) * f) / mp.sinh(N * f)
            rhs = (mp.exp(-m * f) + mp.exp(-(2 * N - m) * f)) / (1 - mp.exp(-2 * N * f))
            err = max(err, abs(lhs - rhs) / abs(lhs))
            ncase += 1
record("appA_proofs:eq-stable", "cosh((N-m) phi)/sinh(N phi) = (e^{-m phi} + e^{-(2N-m) phi})/(1 - e^{-2N phi}), 0<=m<=2N", ncase, err, mp.mpf(10) ** (-28))

# ---------------------------------------------------------------------------
# 7. Corollary s4_mfpt:cor-resistance
# ---------------------------------------------------------------------------
def grid_laplacian(N, d=2):
    S = sites(N, d)
    idx = {x: i for i, x in enumerate(S)}
    L = np.zeros((len(S), len(S)))
    for x in S:
        for ax in range(d):
            y = list(x); y[ax] += 1
            if y[ax] < N:
                i, j = idx[x], idx[tuple(y)]
                L[i, i] += 1; L[j, j] += 1; L[i, j] -= 1; L[j, i] -= 1
    return L, idx


err_lap = err_comm = err_R = 0.0; ncase = 0
for N in range(2, 13):
    Lg, idx = grid_laplacian(N)
    G = np.linalg.pinv(Lg)
    for q in (0.8, 0.37):
        P, S, idx2 = build_P_rules(N, 2, q)
        err_lap = max(err_lap, np.abs(np.eye(N * N) - np.array(P, dtype=float) - (q / 4) * Lg).max())
    R = lambda x, y: G[idx[x], idx[x]] + G[idx[y], idx[y]] - 2 * G[idx[x], idx[y]]
    RN = R((0, 0), (N - 1, N - 1))
    qT = float(form_b(N))
    err_R = max(err_R, abs(qT / (2 * N * N * RN) - 1), abs((1 + 4 / N * float(S_even(N))) / RN - 1),
                abs((4 / N * float(S_odd(N)) - 1) / RN - 1))
    for _ in range(4):
        o = tuple(int(v) for v in rng.integers(0, N, 2)); a = tuple(int(v) for v in rng.integers(0, N, 2))
        if o == a:
            continue
        comm = float(pair(N, o, a) + pair(N, a, o))          # q (T_{o->a} + T_{a->o})
        err_comm = max(err_comm, abs(comm / (4 * N * N * R(o, a)) - 1))
        ncase += 1
record("appA_proofs:eq-laplacian", "I - P = (q/4)(D - A), P built from the rules, N=2..12, q=0.8, 0.37", 22, err_lap, 1e-14)
record("s4_mfpt:eq-resistance", "q (T_{o->a}+T_{a->o}) = 4 N^2 R(o,a), R from the Laplacian pseudo-inverse, random pairs", ncase, err_comm, 1e-10)
record("s4_mfpt:eq-resistance", "q T_N = 2 N^2 R_N; R_N = 1 + (4/N) S_even = (4/N) S_odd - 1, N=2..12", 11, err_R, 1e-10)
R4 = EXACT_CC[(2, 4)] / (2 * 16)
record("s4_mfpt:eq-resistance", "exact: q T_4/(2*16) = 13/7 (Wu 2004, Example 4: corner-to-corner resistance of the 4x4 grid)", 1,
       0.0 if R4 == Fraction(13, 7) else 1.0, 0.0, {"R_4": f"{R4.numerator}/{R4.denominator}"})

# ---------------------------------------------------------------------------
# 8. Proposition s4_mfpt:prop-3d
# ---------------------------------------------------------------------------
err_G3 = mp.mpf(0); err_al = mp.mpf(0); nG3 = 0
for N in (2, 3, 4, 5, 6, 7, 8):
    err_al = max(err_al, abs(mp.fsum(al(k) * c(k, N) ** 2 for k in range(N)) - N))
    for k in range(N):
        for j in range(N):
            if k == 0 and j == 0:
                Gd = mp.fsum(al(l) * c(l, N) ** 2 * (1 - (-1) ** l) / (1 - mp.cos(vth(l, N))) for l in range(1, N))
                err_G3 = max(err_G3, abs(Gd - N * (N - 1)) / (N * (N - 1)))
                continue
            sg = 3 - mp.cos(vth(k, N)) - mp.cos(vth(j, N))
            f = phi2(k, j, N)
            Gd = mp.fsum(al(l) * c(l, N) ** 2 * (1 - (-1) ** (k + j + l)) / (sg - mp.cos(vth(l, N))) for l in range(N))
            Gc = N * (1 + sg) * (mp.cosh(N * f) - (-1) ** (k + j)) / (mp.sinh(f) * mp.sinh(N * f)) - N
            g = mp.coth(N * f / 2) if (k + j) % 2 else mp.tanh(N * f / 2)
            err_G3 = max(err_G3, abs(Gd - Gc) / abs(Gc), abs(Gc - N * (mp.coth(f / 2) * g - 1)) / abs(Gc), abs(mp.cosh(f) - sg))
            nG3 += 1
record("appA_proofs:eq-Gkj", "d=3 inner sum G_kj: direct sum = closed form = N[coth(phi_kj/2) g_kj - 1]; G_00 = N(N-1); N=2..8", nG3, err_G3, mp.mpf(10) ** (-28))
record("appA_proofs:eq-alpha-c", "sum_{k=0}^{N-1} alpha_k c_k^2 = N, N=2..8 (absolute)", 7, err_al, mp.mpf(10) ** (-30))

err = mp.mpf(0)
ex3 = []
for N in (2, 3, 4):
    ex = EXACT_CC[(3, N)]
    err = max(err, relerr(form_3d(N), ex), relerr(cosine_cc(N, 3), ex))
    ex3.append({"N": N, "qT_exact": f"{ex.numerator}/{ex.denominator}" if ex.denominator != 1 else str(ex.numerator), "qT_float": float(ex)})
record("s4_mfpt:eq-3d", "double sum = triple sum = exact rational q T_N^(3), N=2,3,4", 3, err, TOL_MP)
err = 0.0
rows3 = []
for N in (5, 6, 7, 8, 9, 10, 11, 12):
    full, idx = float_hit(N, 3, (N - 1,) * 3, 0.8)
    ref = 0.8 * full[idx[0, 0, 0]]
    v = float(form_3d(N))
    err = max(err, abs(v / ref - 1))
    rows3.append({"N": N, "qT_solve": ref, "qT_double_sum": v})
record("s4_mfpt:eq-3d", "double sum against float64 sparse solves, N=5..12 (both parities), q=0.8", len(rows3), err, TOL_FL)

# ---------------------------------------------------------------------------
# 9. The finite sum F_m (Eq. (E2) of Giuggioli 2020)
# ---------------------------------------------------------------------------
err = mp.mpf(0); ncase = 0
for N in (2, 3, 5, 8, 13):
    for m in range(0, 2 * N + 1):
        Fm = mp.fsum(mp.cos(m * mp.pi * k / N) / (1 - mp.cos(mp.pi * k / N)) for k in range(1, N))
        rhs = (mp.mpf(N) ** 2 + mp.mpf(1) / 2) / 3 + m * (mp.mpf(m) / 2 - N) + mp.mpf((-1) ** (m + 1) - 1) / 4
        err = max(err, abs(Fm - rhs))
        ncase += 1
record("appA_proofs:lem-1d", "finite sum F_m = Eq. (E2) of Giuggioli 2020, for 0<=m<=2N, N=2,3,5,8,13 (absolute)", ncase, err, mp.mpf(10) ** (-28))

# float64 overflow of form (a) as printed
def form_a_float(N):
    k = np.arange(1, N)
    ph = np.arccosh(2 - np.cos(np.pi * k / N))
    with np.errstate(over="ignore", invalid="ignore"):
        B = np.cosh(N * ph) + np.cosh((N - 1) * ph) - (-1.0) ** k * (1 + np.cosh(ph))
        return float(2 * N * (N - 1) + 4 * N * np.sum(np.cos(np.pi * k / (2 * N)) ** 2 * B / (np.sinh(ph) * np.sinh(N * ph))))


def form_b_float(N):
    k = np.arange(1, N)
    sk_ = np.sin(np.pi * k / (2 * N))
    t = np.tanh(N * np.arcsinh(sk_))
    g = np.where(k % 2 == 1, 1 / t, t)
    return float(4 * N * np.sum(np.cos(np.pi * k / (2 * N)) ** 2 * np.sqrt(1 + sk_ ** 2) / sk_ * g))


a402, a403 = form_a_float(402), form_a_float(403)
b402, b403, b5000 = form_b_float(402), form_b_float(403), form_b_float(5000)
a_small_ok = all(np.isfinite(form_a_float(N)) and abs(form_a_float(N) / form_b_float(N) - 1) < 1e-10 for N in range(2, 403))
a_large_bad = not any(np.isfinite(form_a_float(N)) for N in list(range(403, 451)) + [500, 1000, 5000])
overflow = {"form_a_float64_finite_and_equal_to_form_b_for_all_N_2..402": bool(a_small_ok),
            "form_a_float64_not_finite_for_N_403..450_500_1000_5000": bool(a_large_bad),
            "largest_double": float(np.finfo(float).max), "ln_largest_double": float(np.log(np.finfo(float).max)),
            "arcosh_3": float(np.arccosh(3.0)),
            "N_phi_{N-1}_at_N402": float(402 * np.arccosh(2 - np.cos(np.pi * 401 / 402))),
            "N_phi_{N-1}_at_N403": float(403 * np.arccosh(2 - np.cos(np.pi * 402 / 403))),
            "form_a_float64_N402": a402, "form_a_float64_N403_is_finite": bool(np.isfinite(a403)),
            "form_b_float64_N402": b402, "rel_dev_a_vs_b_N402": abs(a402 / b402 - 1),
            "form_b_float64_N403": b403, "form_b_float64_N5000": b5000,
            "rel_dev_b_float_vs_40_digits_N403": abs(b403 / float(form_b(403)) - 1)}
record("s4_mfpt:eq-single-a (float64)", "form (a) is finite and equal to form (b) for N=2..402; it is not finite for N=403..450, 500, 1000, 5000; form (b) stays finite",
       3, 0.0 if (np.isfinite(a402) and not np.isfinite(a403) and np.isfinite(b5000) and overflow["rel_dev_a_vs_b_N402"] < 1e-10
                  and a_small_ok and a_large_bad) else 1.0, 0.5)

# ---------------------------------------------------------------------------
mp_checks = [r for r in CHECKS if r["tolerance"] < 1e-20 and r["tolerance"] > 0]
fl_checks = [r for r in CHECKS if "float64 sparse" in r["statement"]]
summary = {
    "max_deviation_multiprecision_checks": max(r["max_deviation"] for r in mp_checks),
    "n_multiprecision_checks": len(mp_checks),
    "max_rel_deviation_vs_float64_sparse_solves": max(r["max_deviation"] for r in fl_checks),
    "n_float64_sparse_solve_checks": len(fl_checks),
    "total_cases": sum(r["cases"] for r in CHECKS),
}
out = {
    "script": "code/article/appA_proofs_check.py",
    "summary": summary,
    "mp_dps": mp.mp.dps,
    "n_checks": len(CHECKS),
    "n_failed": sum(not r["passed"] for r in CHECKS),
    "runtime_s": round(time.time() - T0, 1),
    "checks": CHECKS,
    "exact_corner_values_d2": table_exact,
    "exact_corner_values_d3": ex3,
    "float_solves_d2": rows_float,
    "float_solves_d3": rows3,
    "float64_overflow": overflow,
}
with open(OUT, "w") as f:
    json.dump(out, f, indent=1)
print(f"\n{len(CHECKS)} checks, {out['n_failed']} failed, {out['runtime_s']} s -> {os.path.relpath(OUT, ROOT)}")
