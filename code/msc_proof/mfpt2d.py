"""
Core library: 2D reflecting-square lazy walk, corner-to-corner (and general) MFPT.

Model (Section 2 of the article):
  sites (x, y) in {1..N}^2; at each step, with prob. 1-q the walker rests;
  with prob. q it picks one of the 4 lattice directions uniformly (q/4 each);
  a move that would leave the square is cancelled (walker rests).
  Target a is absorbing; MFPT T_{s->a} = E_s[min{t >= 0 : X_t = a}].

Everything here is deterministic; no random numbers except in general_pairs.py
(seeded there).

Formulas implemented (labels local to this module; the results are Section 4 of the article):
  direct_*         : solve (I - Q) m = 1 on transient states (Eq. (2.1))
  double_sum_*     : canonical cosine double sum (Theorem 1, Eq. (D))
  single_sum_form_a_*: form (a) of the single sum, evaluated as written
  clean_single_*   : Theorem 2 overflow-free form (Eq. (S))
  general_single_* : Theorem 3 (arbitrary start/target single sum)
"""
from __future__ import annotations

from fractions import Fraction
import math

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import mpmath as mp


# ----------------------------------------------------------------------------
# 1. Direct construction from the model rules (independent of any spectral fact)
# ----------------------------------------------------------------------------

def transition_matrix(N: int, q: float) -> sp.csr_matrix:
    """Full N^2 x N^2 transition matrix, built only from the stated rules.
    Site (x, y) (1-based) has index (x-1)*N + (y-1)."""
    rows, cols, vals = [], [], []
    for x in range(1, N + 1):
        for y in range(1, N + 1):
            i = (x - 1) * N + (y - 1)
            stay = 1.0 - q
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                xx, yy = x + dx, y + dy
                if 1 <= xx <= N and 1 <= yy <= N:
                    rows.append(i); cols.append((xx - 1) * N + (yy - 1)); vals.append(q / 4)
                else:
                    stay += q / 4          # cancelled attempt
            rows.append(i); cols.append(i); vals.append(stay)
    return sp.csr_matrix((vals, (rows, cols)), shape=(N * N, N * N))


def direct_mfpt(N: int, q: float, start=(1, 1), target=None) -> float:
    """MFPT by sparse direct solve of (I - Q) m = 1 (float64)."""
    if target is None:
        target = (N, N)
    P = transition_matrix(N, q)
    ia = (target[0] - 1) * N + (target[1] - 1)
    keep = np.array([i for i in range(N * N) if i != ia])
    Q = P[keep][:, keep]
    A = (sp.identity(len(keep), format="csc") - Q).tocsc()
    m = spla.spsolve(A, np.ones(len(keep)))
    i_s = (start[0] - 1) * N + (start[1] - 1)
    if i_s == ia:
        return 0.0
    pos = int(np.searchsorted(keep, i_s))
    return float(m[pos])


def direct_mfpt_cg(N: int, q: float, rtol: float = 1e-14) -> float:
    """Corner-to-corner MFPT by preconditioned CG (for large N, float64).
    I - Q is symmetric positive definite (P is symmetric)."""
    P = transition_matrix(N, q)
    ia = N * N - 1
    Q = P[:ia, :ia]
    A = (sp.identity(ia, format="csr") - Q).tocsr()
    d = A.diagonal()
    M = spla.LinearOperator(A.shape, matvec=lambda v: v / d)
    b = np.ones(ia)
    m, info = spla.cg(A, b, rtol=rtol, atol=0.0, maxiter=200000, M=M)
    if info != 0:
        raise RuntimeError(f"CG did not converge (info={info})")
    return float(m[0])


def exact_mfpt_rational(N: int, q: Fraction = Fraction(1), start=(1, 1), target=None) -> Fraction:
    """Exact rational MFPT by banded Gaussian elimination in Fractions.
    I - Q is symmetric positive definite, so no pivoting is needed; with the
    lexicographic ordering its half-bandwidth is N."""
    if target is None:
        target = (N, N)
    q = Fraction(q)
    idx = {}
    order = []
    for x in range(1, N + 1):
        for y in range(1, N + 1):
            if (x, y) != tuple(target):
                idx[(x, y)] = len(order)
                order.append((x, y))
    n = len(order)
    bw = N  # half-bandwidth
    # store rows as dicts {col: value} restricted to band
    A = [dict() for _ in range(n)]
    b = [Fraction(1)] * n
    for (x, y), i in idx.items():
        diag = Fraction(1)  # I
        # subtract Q
        stay = 1 - q
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            xx, yy = x + dx, y + dy
            if 1 <= xx <= N and 1 <= yy <= N:
                if (xx, yy) == tuple(target):
                    continue  # absorbed: no entry in Q
                j = idx[(xx, yy)]
                A[i][j] = A[i].get(j, Fraction(0)) - q / 4
            else:
                stay += q / 4
        A[i][i] = A[i].get(i, Fraction(0)) + diag - stay
    # forward elimination
    for k in range(n):
        piv = A[k][k]
        rowk = A[k]
        for i in range(k + 1, min(n, k + bw + 1)):
            aik = A[i].get(k)
            if not aik:
                continue
            f = aik / piv
            Ai = A[i]
            for j, v in rowk.items():
                if j < k:
                    continue
                Ai[j] = Ai.get(j, Fraction(0)) - f * v
            Ai.pop(k, None)
            b[i] -= f * b[k]
    # back substitution
    m = [Fraction(0)] * n
    for k in range(n - 1, -1, -1):
        s = b[k]
        for j, v in A[k].items():
            if j > k:
                s -= v * m[j]
        m[k] = s / A[k][k]
    if tuple(start) == tuple(target):
        return Fraction(0)
    return m[idx[tuple(start)]]


# ----------------------------------------------------------------------------
# 2. Spectral double sum (Theorem 1)
# ----------------------------------------------------------------------------

def double_sum_np(N: int, q: float) -> float:
    """Canonical cosine double sum, float64, vectorised by rows:
    T = (2/q) sum'_{k,j} a_k a_j c_k^2 c_j^2 [1-(-1)^{k+j}] / (2 - cos x_k - cos x_j)."""
    k = np.arange(N)
    x = np.pi * k / N
    c2 = np.cos(np.pi * k / (2 * N)) ** 2
    al = np.where(k == 0, 1.0, 2.0)
    w = al * c2
    par = (-1.0) ** k
    cosx = np.cos(x)
    tot = []
    for kk in range(N):
        num = 1.0 - par[kk] * par
        den = 2.0 - cosx[kk] - cosx
        if kk == 0:
            num = num.copy(); num[0] = 0.0; den = den.copy(); den[0] = 1.0
        tot.append(math.fsum(w[kk] * w * num / den))
    return 2.0 / q * math.fsum(tot)


def double_sum_mp(N: int, q, dps: int = 50):
    with mp.workdps(dps):
        q = mp.mpf(q)
        xs = [mp.pi * k / N for k in range(N)]
        cosx = [mp.cos(v) for v in xs]
        w = [(1 if k == 0 else 2) * mp.cos(mp.pi * k / (2 * N)) ** 2 for k in range(N)]
        s = mp.mpf(0)
        for k in range(N):
            for j in range(N):
                if (k + j) % 2 == 0:
                    continue  # numerator 1-(-1)^{k+j} = 0 (includes (0,0))
                s += w[k] * w[j] * 2 / (2 - cosx[k] - cosx[j])
        return 2 / q * s


def general_double_sum_mp(N: int, q, s, a, dps: int = 40):
    """T_{s->a} = (2/q) sum' a_k a_j [C_k(a1,a1) C_j(a2,a2) - C_k(s1,a1) C_j(s2,a2)]/(2-cos x_k-cos x_j)."""
    with mp.workdps(dps):
        q = mp.mpf(q)
        def th(k, n):
            return mp.pi * k * (2 * n - 1) / (2 * N)
        tot = mp.mpf(0)
        cosx = [mp.cos(mp.pi * k / N) for k in range(N)]
        for k in range(N):
            ak = 1 if k == 0 else 2
            Caa1 = mp.cos(th(k, a[0])) ** 2
            Csa1 = mp.cos(th(k, s[0])) * mp.cos(th(k, a[0]))
            for j in range(N):
                if k == 0 and j == 0:
                    continue
                aj = 1 if j == 0 else 2
                Caa2 = mp.cos(th(j, a[1])) ** 2
                Csa2 = mp.cos(th(j, s[1])) * mp.cos(th(j, a[1]))
                tot += ak * aj * (Caa1 * Caa2 - Csa1 * Csa2) / (2 - cosx[k] - cosx[j])
        return 2 / q * tot


# ----------------------------------------------------------------------------
# 3. Single sums
# ----------------------------------------------------------------------------

def single_sum_form_a_mp(N: int, q, dps: int = 50):
    """Form (a) of the single sum, evaluated as written (overflows in float64 for N >= 403)."""
    with mp.workdps(dps):
        q = mp.mpf(q)
        s = mp.mpf(0)
        for k in range(1, N):
            phi = mp.acosh(2 - mp.cos(mp.pi * k / N))
            Bk = mp.cosh(N * phi) + mp.cosh((N - 1) * phi) - (-1) ** k * (1 + mp.cosh(phi))
            s += mp.cos(mp.pi * k / (2 * N)) ** 2 * Bk / (mp.sinh(phi) * mp.sinh(N * phi))
        return 2 * N * (N - 1) / q + 4 * N / q * s


def single_sum_form_a_np(N: int, q: float) -> float:
    """Form (a) evaluated literally in float64 (to exhibit the overflow at N >= 403)."""
    k = np.arange(1, N)
    with np.errstate(over="ignore", invalid="ignore"):
        phi = np.arccosh(2 - np.cos(np.pi * k / N))
        Bk = np.cosh(N * phi) + np.cosh((N - 1) * phi) - (-1.0) ** k * (1 + np.cosh(phi))
        terms = np.cos(np.pi * k / (2 * N)) ** 2 * Bk / (np.sinh(phi) * np.sinh(N * phi))
        return 2 * N * (N - 1) / q + 4 * N / q * float(np.sum(terms))


def clean_single_mp(N: int, q, dps: int = 50):
    """Theorem 2 (clean, overflow-free form):
    T = (4N/q) sum_{k=1}^{N-1} cos^2(pi k/2N) * sqrt(1+s_k^2)/s_k * g_k,
    s_k = sin(pi k/2N), g_k = coth(N asinh s_k) (k odd) or tanh(N asinh s_k) (k even)."""
    with mp.workdps(dps):
        q = mp.mpf(q)
        tot = mp.mpf(0)
        for k in range(1, N):
            y = mp.pi * k / (2 * N)
            sk = mp.sin(y)
            t = mp.tanh(N * mp.asinh(sk))
            g = 1 / t if k % 2 else t
            tot += mp.cos(y) ** 2 * mp.sqrt(1 + sk ** 2) / sk * g
        return 4 * N / q * tot


def clean_single_np(N: int, q: float) -> float:
    k = np.arange(1, N)
    y = np.pi * k / (2 * N)
    sk = np.sin(y)
    t = np.tanh(N * np.arcsinh(sk))
    g = np.where(k % 2 == 1, 1.0 / t, t)
    terms = np.cos(y) ** 2 * np.sqrt(1 + sk ** 2) / sk * g
    return 4 * N / q * math.fsum(terms)


def F_m(N: int, m: int):
    """Closed form of F_m = sum_{j=1}^{N-1} cos(m pi j/N)/(1-cos(pi j/N)), 0 <= m <= 2N (Lemma 5)."""
    return Fraction((N - m) ** 2, 2) - Fraction(N * N, 6) - Fraction(1, 12) - Fraction((-1) ** m, 4)


def tau_1d(N: int, s: int, a: int) -> int:
    """Hitting time s -> a of the 1D cancelled-move chain on {1..N} with q = 1 (Lemma 4)."""
    if s <= a:
        return (a - s) * (a + s - 1)
    return (s - a) * (2 * N + 1 - a - s)


def general_single_mp(N: int, q, s, a, dps: int = 40):
    """Theorem 3: single-sum MFPT for arbitrary start s=(s1,s2) and target a=(a1,a2):
    q T = 2 tau_1d(s2,a2) + 4N sum_k [cos^2 th_k(a1) Psi_k(a2,a2) - cos th_k(s1) cos th_k(a1) Psi_k(s2,a2)]."""
    with mp.workdps(dps):
        q = mp.mpf(q)
        s1, s2 = s
        a1, a2 = a
        dG0 = F_m(N, 0) + F_m(N, 2 * a2 - 1) - F_m(N, abs(s2 - a2)) - F_m(N, s2 + a2 - 1)
        assert dG0 == tau_1d(N, s2, a2)      # k=0 sector = 1D hitting time (Lemma 4)
        dG0 = mp.mpf(dG0.numerator) / dG0.denominator

        def C(k, u, v):
            return mp.cos(mp.pi * k * (2 * u - 1) / (2 * N)) * mp.cos(mp.pi * k * (2 * v - 1) / (2 * N))

        def Psi(k, u, v, phi):
            m1 = abs(u - v)
            m2 = u + v - 1
            return (mp.cosh((N - m1) * phi) + mp.cosh((N - m2) * phi)) / (mp.sinh(phi) * mp.sinh(N * phi))

        tot = mp.mpf(0)
        for k in range(1, N):
            phi = 2 * mp.asinh(mp.sin(mp.pi * k / (2 * N)))
            tot += C(k, a1, a1) * Psi(k, a2, a2, phi) - C(k, s1, a1) * Psi(k, s2, a2, phi)
        return 2 / q * (dG0 + 2 * N * tot)


# ----------------------------------------------------------------------------
# 4. Asymptotic constants (Theorem 3)
# ----------------------------------------------------------------------------

def asymptotic_constants(dps: int = 50):
    with mp.workdps(dps):
        g14 = mp.gamma(mp.mpf(1) / 4)
        C2 = 8 / mp.pi * (mp.euler + 4 * mp.log(2) + mp.log(mp.pi) / 2 - 2 * mp.log(g14)
                          - mp.mpf(1) / 2 - mp.pi / 4)
        C0 = g14 ** 8 / (576 * mp.pi ** 4)
        return C2, C0


if __name__ == "__main__":
    for N in (2, 3, 4, 10):
        print(N, direct_mfpt(N, 0.8), double_sum_np(N, 0.8), single_sum_form_a_np(N, 0.8),
              clean_single_np(N, 0.8), float(exact_mfpt_rational(N, Fraction(4, 5))))
    print(asymptotic_constants())
