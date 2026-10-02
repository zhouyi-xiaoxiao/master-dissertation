"""Second implementation, written separately for the check, of the 2D reflecting lazy walk and of
the closed forms of the proofs (Supplementary Sections S3 and S4).  Written from their *statements*
and from the model rules (Section 2 of the article), without
reusing the main implementation's scripts.

Model: walker on {0..N-1}^2.  Each step: with prob 1-q rest; else choose one of the
4 directions uniformly; if the move leaves the lattice it is cancelled (walker stays).
Target site absorbing.
"""
from fractions import Fraction
import mpmath as mp


# ----------------------------------------------------------------------------
# model, built from the rules only
# ----------------------------------------------------------------------------
def transition_rows(N, q):
    """Return dict-of-dicts P[x][y] with exact Fractions, x=(i,j)."""
    q = Fraction(q)
    P = {}
    for i in range(N):
        for j in range(N):
            row = {}
            row[(i, j)] = 1 - q
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                a, b = i + di, j + dj
                if 0 <= a < N and 0 <= b < N:
                    row[(a, b)] = row.get((a, b), 0) + q / 4
                else:
                    row[(i, j)] = row[(i, j)] + q / 4   # cancelled move
            P[(i, j)] = row
    return P


def exact_hitting_times(N, q, target):
    """Solve (I-Q) m = 1 exactly (Fractions) for all starts; banded Gaussian
    elimination in row-major order (bandwidth <= N).  Returns dict start->Fraction."""
    P = transition_rows(N, q)
    sites = [(i, j) for i in range(N) for j in range(N) if (i, j) != target]
    idx = {s: n for n, s in enumerate(sites)}
    n = len(sites)
    # sparse rows as dicts col->Fraction, augmented rhs
    A = []
    for s in sites:
        row = {}
        for y, p in P[s].items():
            if y == target:
                continue
            row[idx[y]] = row.get(idx[y], 0) - p
        row[idx[s]] = row.get(idx[s], 0) + 1
        A.append(row)
    b = [Fraction(1)] * n
    # forward elimination (matrix is an irreducibly diagonally dominant M-matrix;
    # pivots are nonzero without pivoting)
    # column -> rows below that have an entry: maintain by scanning band
    band = N + 1
    for c in range(n):
        piv = A[c][c]
        assert piv != 0
        for r in range(c + 1, min(n, c + band + 1)):
            arc = A[r].get(c)
            if arc:
                f = arc / piv
                rowc = A[c]
                rowr = A[r]
                for cc, v in rowc.items():
                    if cc > c:
                        nv = rowr.get(cc, 0) - f * v
                        rowr[cc] = nv
                del rowr[c]
                b[r] -= f * b[c]
    x = [None] * n
    for r in range(n - 1, -1, -1):
        s = b[r]
        for cc, v in A[r].items():
            if cc > r:
                s -= v * x[cc]
        x[r] = s / A[r][r]
    out = {s: x[idx[s]] for s in sites}
    out[target] = Fraction(0)
    return out


# ----------------------------------------------------------------------------
# closed forms (mpmath).  All return q*T (q-independent quantity).
# ----------------------------------------------------------------------------
def _alpha(k):
    return 1 if k == 0 else 2


def qT_double_corner(N):
    """(D): 2 * sum_{k+j odd} a_k a_j c_k^2 c_j^2/(s_k^2+s_j^2)."""
    c2 = [mp.cos(mp.pi * k / (2 * N)) ** 2 for k in range(N)]
    s2 = [mp.sin(mp.pi * k / (2 * N)) ** 2 for k in range(N)]
    tot = mp.mpf(0)
    for k in range(N):
        for j in range(N):
            if (k + j) % 2 == 1:
                tot += _alpha(k) * _alpha(j) * c2[k] * c2[j] / (s2[k] + s2[j])
    return 2 * tot


def qT_double_general(N, s, a):
    """(G) with 1-based coordinates s=(s1,s2), a=(a1,a2)."""
    def th(k, m):
        return mp.pi * k * (2 * m - 1) / (2 * N)
    tot = mp.mpf(0)
    cosx = [mp.cos(mp.pi * k / N) for k in range(N)]
    Ca1 = [mp.cos(th(k, a[0])) ** 2 for k in range(N)]
    Ca2 = [mp.cos(th(k, a[1])) ** 2 for k in range(N)]
    Cs1 = [mp.cos(th(k, s[0])) * mp.cos(th(k, a[0])) for k in range(N)]
    Cs2 = [mp.cos(th(k, s[1])) * mp.cos(th(k, a[1])) for k in range(N)]
    for k in range(N):
        for j in range(N):
            if k == 0 and j == 0:
                continue
            tot += _alpha(k) * _alpha(j) * (Ca1[k] * Ca2[j] - Cs1[k] * Cs2[j]) / (2 - cosx[k] - cosx[j])
    return 2 * tot


def qT_form_a(N):
    """form (a) of the single sum, times q, evaluated as written."""
    tot = mp.mpf(0)
    for k in range(1, N):
        phi = mp.acosh(2 - mp.cos(mp.pi * k / N))
        B = mp.cosh(N * phi) + mp.cosh((N - 1) * phi) - (-1) ** k * (1 + mp.cosh(phi))
        tot += mp.cos(mp.pi * k / (2 * N)) ** 2 * B / (mp.sinh(phi) * mp.sinh(N * phi))
    return 2 * N * (N - 1) + 4 * N * tot


def half_sums(N):
    """S_odd, S_even of Theorem 2(c)."""
    So = mp.mpf(0)
    Se = mp.mpf(0)
    for k in range(1, N):
        s = mp.sin(mp.pi * k / (2 * N))
        c2 = 1 - s * s
        w = c2 * mp.sqrt(1 + s * s) / s
        u = N * mp.asinh(s)
        if k % 2:
            So += w * mp.coth(u)
        else:
            Se += w * mp.tanh(u)
    return So, Se


def qT_S(N):
    So, Se = half_sums(N)
    return 4 * N * (So + Se)


def tau1(N, u, v):
    if u <= v:
        return (v - u) * (v + u - 1)
    return (u - v) * (2 * N + 1 - u - v)


def qT_thm3(N, s, a):
    """Theorem 3, 1-based coordinates."""
    def th(k, m):
        return mp.pi * k * (2 * m - 1) / (2 * N)

    def Psi(phi, u, v):
        return (mp.cosh((N - abs(u - v)) * phi) + mp.cosh((N + 1 - u - v) * phi)) / (mp.sinh(phi) * mp.sinh(N * phi))
    tot = mp.mpf(0)
    for k in range(1, N):
        phi = mp.acosh(2 - mp.cos(mp.pi * k / N))
        tot += (mp.cos(th(k, a[0])) ** 2 * Psi(phi, a[1], a[1])
                - mp.cos(th(k, s[0])) * mp.cos(th(k, a[0])) * Psi(phi, s[1], a[1]))
    return 2 * tau1(N, s[1], a[1]) + 4 * N * tot
