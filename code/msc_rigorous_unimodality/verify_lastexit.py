"""
verify_lastexit.py -- sanity checks for Section 4.4 of note R4
("last-exit decomposition + strong unimodality": Lemmas 4.7-4.11, Theorem 4.12, Corollary 4.13).

Exact checks use Python integers / fractions.Fraction; spectral checks use mpmath (50 digits) or numpy
(float64) as stated in each check name.  Results -> data/verify_lastexit.json.

Run:  python verify_lastexit.py
"""
from __future__ import annotations
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository

import itertools
import json
import math
import os
import random
import sys
from fractions import Fraction

import mpmath as mp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from unilib import laplacian_q1, killed, site_index  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = _os.path.join(_R, 'data', 'msc_rigorous_unimodality')
RESULTS = []


def record(name, ok, detail=""):
    RESULTS.append(dict(check=name, ok=bool(ok), detail=str(detail)))
    print(("PASS " if ok else "FAIL ") + name + (" :: " + str(detail) if detail else ""), flush=True)
    with open(_os.path.join(_R, 'data', 'msc_rigorous_unimodality', 'verify_lastexit.json'), "w") as fh:
        json.dump(RESULTS, fh, indent=1)


# ---------------------------------------------------------------------------
# exact integer machinery for the box (unkilled and killed chain)
# ---------------------------------------------------------------------------
class Box:
    """Lazy walk on {0..N-1}^d, activity q = qn/qd, one-step denominator D = 2 d qd (no reduction)."""

    def __init__(self, N, d, qn, qd):
        self.N, self.d, self.qn, self.qd = N, d, qn, qd
        self.shape = (N,) * d
        self.D = 2 * d * qd
        nblk = np.zeros(self.shape, dtype=object)
        for ax in range(d):
            i0 = [slice(None)] * d; i0[ax] = 0; nblk[tuple(i0)] += 1
            i1 = [slice(None)] * d; i1[ax] = N - 1; nblk[tuple(i1)] += 1
        self.stay = np.empty(self.shape, dtype=object)
        for idx in np.ndindex(*self.shape):
            self.stay[idx] = 2 * d * (qd - qn) + qn * int(nblk[idx])
        self.sl = []
        for ax in range(d):
            lo = [slice(None)] * d; hi = [slice(None)] * d
            lo[ax] = slice(0, N - 1); hi[ax] = slice(1, N)
            self.sl.append((tuple(lo), tuple(hi)))

    def zeros(self):
        R = np.empty(self.shape, dtype=object)
        for idx in np.ndindex(*self.shape):
            R[idx] = 0
        return R

    def step(self, R):
        """(D * P_q) R, exact integers (P_q symmetric, so this is also R^T P_q)."""
        new = self.stay * R
        mR = R * self.qn
        for lo, hi in self.sl:
            new[hi] += mR[lo]
            new[lo] += mR[hi]
        return new


def exact_sequences(N, d, qn, qd, start, target, T):
    """Integers U[t], V[t], F[t], E[t] (t = 0..T) with
       u(t) = P_q^t(x0,a) = U[t]/D^t,  v(t) = P_q^t(a,a) = V[t]/D^t,
       f(t) = P_{x0}(T=t) = F[t]/D^t,  E(t) = P_a(no return to a in steps 1..t) = E[t]/D^t."""
    bx = Box(N, d, qn, qd)
    D = bx.D
    start, target = tuple(start), tuple(target)
    # unkilled from x0 and from a
    R = bx.zeros(); R[start] = 1
    Ra = bx.zeros(); Ra[target] = 1
    U, V = [R[target]], [Ra[target]]
    # killed from x0
    K = bx.zeros(); K[start] = 1
    F = [0]
    # escape: start at a, leave at the first step, then killed evolution
    Wesc = None
    E = [1]
    for t in range(1, T + 1):
        R = bx.step(R); U.append(R[target])
        Ra = bx.step(Ra); V.append(Ra[target])
        K = bx.step(K); F.append(K[target]); K[target] = 0
        if Wesc is None:
            e = bx.zeros(); e[target] = 1
            Wesc = bx.step(e); Wesc[target] = 0
        else:
            Wesc = bx.step(Wesc); Wesc[target] = 0
        E.append(int(sum(Wesc.flat)))
    return U, V, F, E, D


def geom(geo, N, d):
    c = (N - 1) // 2
    if geo == "CC":
        return (0,) * d, (N - 1,) * d
    if geo == "C2M":
        return (0,) * d, (c,) * d
    if geo == "M2C":
        return (c,) * d, (N - 1,) * d
    raise ValueError


# ---------------------------------------------------------------------------
# W1  discrete last-exit identities (Lemma 4.7), exact, arbitrary pairs and q
# ---------------------------------------------------------------------------
def w1():
    ok = True
    cases = [(1, 5, (0,), (4,)), (1, 6, (1,), (4,)), (2, 3, (0, 0), (2, 2)), (2, 4, (0, 1), (3, 2)), (2, 5, (0, 0), (2, 2)),
             (2, 5, (2, 2), (4, 4)), (2, 4, (1, 3), (1, 1)), (3, 3, (0, 0, 0), (2, 2, 2)), (3, 3, (2, 1, 1), (0, 1, 1))]
    ncheck = 0
    for (d, N, s, a) in cases:
        for (qn, qd) in ((1, 1), (4, 5), (1, 2), (1, 3)):
            T = 36
            U, V, F, E, D = exact_sequences(N, d, qn, qd, s, a, T)
            # E nonincreasing, E(0)=1:  E[t]/D^t >= E[t+1]/D^{t+1}
            for t in range(T):
                ok &= (E[t] * D >= E[t + 1] >= 0)
            Phi = [U[0]] + [U[t] - D * U[t - 1] for t in range(1, T + 1)]       # phi(t) D^t
            cumF = 0
            for t in range(T + 1):
                # (i) P(T<=t) = sum_s u(s) E(t-s)
                cumF = cumF * D + F[t] if t > 0 else F[0]
                ok &= (cumF == sum(U[s] * E[t - s] for s in range(t + 1)))
                # (ii) f(t) = sum_k E(k) phi(t-k)
                ok &= (F[t] == sum(E[k] * Phi[t - k] for k in range(t + 1)))
                # (iii) last exit for v: sum_s v(s) E(t-s) = 1
                ok &= (sum(V[s] * E[t - s] for s in range(t + 1)) == D ** t)
                ncheck += 3
            # (iv) Delta f(t) = phi(t+1) - sum_k nu(k) phi(t-k), nu(k) = E(k)-E(k+1)
            for t in range(T):
                lhs = F[t + 1] - D * F[t]
                rhs = Phi[t + 1] - sum((E[k] * D - E[k + 1]) * Phi[t - k] for k in range(t + 1))
                ok &= (lhs == rhs)
                ncheck += 1
    record("W1 Lemma 4.7 (discrete): P(T<=t)=sum u(s)E(t-s); f=E*phi; sum v(s)E(t-s)=1; Delta f = phi(t+1)-sum nu(k)phi(t-k); "
           "E nonincreasing (exact integers, 9 pairs x 4 activities x 36 steps)", ok, "%d identities" % ncheck)


# ---------------------------------------------------------------------------
# W2  continuous last-exit identity in the Laplace domain, exact rationals
# ---------------------------------------------------------------------------
def frac_solve(Mat, rhs):
    n = len(Mat)
    A = [row[:] + [rhs[i]] for i, row in enumerate(Mat)]
    for c in range(n):
        p = next(i for i in range(c, n) if A[i][c] != 0)
        A[c], A[p] = A[p], A[c]
        inv = 1 / A[c][c]
        A[c] = [v * inv for v in A[c]]
        for i in range(n):
            if i != c and A[i][c] != 0:
                fac = A[i][c]
                A[i] = [vi - fac * vc for vi, vc in zip(A[i], A[c])]
    return [A[i][n] for i in range(n)]


def exact_L(N, d):
    sites, idx = site_index(N, d)
    n = len(sites)
    w = Fraction(1, 2 * d)
    L = [[Fraction(0)] * n for _ in range(n)]
    for s in sites:
        i = idx[s]
        for ax in range(d):
            for dx in (-1, 1):
                y = list(s); y[ax] += dx
                if 0 <= y[ax] < N:
                    j = idx[tuple(y)]
                    L[i][j] -= w
                    L[i][i] += w
    return L, sites, idx


def w2():
    ok = True
    cases = [(1, 5, (0,), (4,)), (1, 6, (1,), (3,)), (2, 3, (0, 0), (2, 2)), (2, 4, (0, 1), (3, 2)), (2, 5, (0, 0), (2, 2)),
             (3, 3, (0, 0, 0), (2, 2, 2)), (3, 3, (2, 1, 1), (0, 1, 1))]
    for (d, N, s, a) in cases:
        L, sites, idx = exact_L(N, d)
        n = len(sites)
        ia, ix = idx[tuple(a)], idx[tuple(s)]
        keep = [i for i in range(n) if i != ia]
        pos = keep.index(ix)
        rvec = [-L[i][ia] for i in keep]
        for q in (Fraction(1), Fraction(1, 2)):
            for sv in (Fraction(1, 3), Fraction(2), Fraction(7, 50)):
                # resolvents
                Mfull = [[(sv if i == j else 0) + q * L[i][j] for j in range(n)] for i in range(n)]
                ea = [Fraction(int(i == ia)) for i in range(n)]
                col = frac_solve(Mfull, ea)          # (s+qL)^{-1} e_a
                uhat, vhat = col[ix], col[ia]
                Mk = [[(sv if i == j else 0) + q * L[keep[i]][keep[j]] for j in range(n - 1)] for i in range(n - 1)]
                y = frac_solve(Mk, rvec)             # (s+qA)^{-1} r
                ghat = q * y[pos]
                z = frac_solve(Mk, [Fraction(1)] * (n - 1))
                rhat = q * sum(rv * zv for rv, zv in zip(rvec, z))
                ok &= (uhat == ghat * vhat)                       # first passage
                ok &= (sv * vhat * (1 + rhat) == 1)               # last exit for v
                ok &= (ghat == sv * uhat * (1 + rhat))            # g = u' + r*u'
    record("W2 Lemma 4.7 (continuous), Laplace domain: uhat=ghat*vhat, s*vhat*(1+rhat)=1, ghat=s*uhat*(1+rhat) "
           "(exact rationals, 7 pairs x 2 rates x 3 values of s)", ok)


# ---------------------------------------------------------------------------
# spectral helpers (mpmath) for the box
# ---------------------------------------------------------------------------
def h1d_mp(N, c, j, k):
    """coefficients/rates of p_t(j,k) = sum_m coef_m exp(-rate_m t) for the 1D reflecting walk on {1..N} with
    jump rate c/2 per direction (1-based j,k)."""
    coefs, rates = [], []
    for m in range(N):
        cm = (mp.mpf(1) / N) if m == 0 else (mp.mpf(2) / N)
        coefs.append(cm * mp.cos(m * mp.pi * (j - mp.mpf(1) / 2) / N) * mp.cos(m * mp.pi * (k - mp.mpf(1) / 2) / N))
        rates.append(c * (1 - mp.cos(m * mp.pi / N)))
    return coefs, rates


def expsum(coefs, rates, t, k=0):
    return mp.fsum(cf * (-rt) ** k * mp.exp(-rt * t) for cf, rt in zip(coefs, rates))


def u_derivs_product(N, d, j, k, t, q=1):
    """u = p^d, p = 1D propagator j->k with rate c=q/d.  Returns u, u', u'', u''' at t (mp)."""
    coefs, rates = h1d_mp(N, mp.mpf(q) / d, j, k)
    p0, p1, p2, p3 = (expsum(coefs, rates, t, i) for i in range(4))
    u0 = p0 ** d
    u1 = d * p0 ** (d - 1) * p1
    u2 = d * (d - 1) * p0 ** (d - 2) * p1 ** 2 + d * p0 ** (d - 1) * p2 if d >= 2 else p2
    if d == 1:
        u3 = p3
    else:
        u3 = (d * (d - 1) * (d - 2) * p0 ** (d - 3) * p1 ** 3 if d >= 3 else 0) \
             + 3 * d * (d - 1) * p0 ** (d - 2) * p1 * p2 + d * p0 ** (d - 1) * p3
    return u0, u1, u2, u3


def killed_spectral_float(N, d, start, target):
    L, sites, idx = laplacian_q1(N, d)
    Lk, keep = killed(L, idx, target)
    ia = idx[tuple(target)]
    rvec = -L[keep, ia]
    al, Phi = np.linalg.eigh(Lk)
    pos = keep.index(idx[tuple(start)])
    proj_r = Phi.T @ rvec
    cg = Phi[pos, :] * proj_r               # g(t) = sum cg e^{-al t}   (unit rate)
    cr = proj_r * Phi.sum(axis=0)           # r(t) = sum cr e^{-al t}
    cnu = proj_r ** 2                       # nu(t) = -r'(t) = sum cnu e^{-al t}
    return al, cg, cr, cnu, Lk


# ---------------------------------------------------------------------------
# W3  g' = u'' + r(0)u' - nu*u'  and  g'(t) = u'(t) B(t)   (mp quadrature)
# ---------------------------------------------------------------------------
def w3():
    mp.mp.dps = 30
    ok = True
    det = []
    for (d, N, s, a) in [(2, 4, (0, 0), (3, 3)), (2, 5, (0, 0), (2, 2)), (3, 3, (0, 0, 0), (2, 2, 2)), (2, 4, (0, 1), (3, 2))]:
        al, cg, cr, cnu, _ = killed_spectral_float(N, d, s, a)
        L, sites, idx = laplacian_q1(N, d)
        th, Psi = np.linalg.eigh(L)
        cu = Psi[idx[tuple(s)], :] * Psi[idx[tuple(a)], :]
        up = lambda t: float(np.sum(-th * cu * np.exp(-th * t)))
        upp = lambda t: float(np.sum(th ** 2 * cu * np.exp(-th * t)))
        nu = lambda t: float(np.sum(cnu * np.exp(-al * t)))
        gp = lambda t: float(np.sum(-al * cg * np.exp(-al * t)))
        r0 = float(np.sum(cr))
        worst = 0.0
        for t in (1.5, 4.0, 9.0, 20.0, 45.0):
            conv = mp.quad(lambda tau: nu(float(tau)) * up(t - float(tau)), mp.linspace(0, t, 9))
            rhs = upp(t) + r0 * up(t) - float(conv)
            worst = max(worst, abs(rhs - gp(t)) / max(abs(gp(t)), 1e-300, 1e-3 * abs(r0 * up(t))))
        ok &= worst < 1e-8
        det.append("d%d N%d %s->%s: max rel err %.1e" % (d, N, s, a, worst))
    record("W3 identity g'(t) = u''(t) + r(0)u'(t) - int_0^t nu(tau)u'(t-tau)dtau, nu=-r' (float spectral sums + mp quadrature)",
           ok, " | ".join(det))


# ---------------------------------------------------------------------------
# W4  the three placements of the article, continuous time: u'>0, (log u')''<=0, B=g'/u' nonincreasing, g' one sign change
# ---------------------------------------------------------------------------
def w4():
    mp.mp.dps = 70
    ok = True
    det = []
    cases = [("CC", 1, 6), ("CC", 1, 12), ("CC", 2, 3), ("CC", 2, 8), ("CC", 2, 16), ("CC", 2, 24), ("CC", 3, 4), ("CC", 3, 8),
             ("C2M", 2, 5), ("C2M", 2, 9), ("C2M", 2, 15), ("M2C", 2, 5), ("M2C", 2, 9), ("M2C", 2, 15),
             ("C2M", 3, 5), ("M2C", 3, 5), ("C2M", 3, 7), ("M2C", 3, 7), ("C2M", 1, 9), ("M2C", 1, 9)]
    for geo, d, N in cases:
        s, a = geom(geo, N, d)
        al, cg, cr, cnu, _ = killed_spectral_float(N, d, s, a)
        j, k = s[0] + 1, a[0] + 1
        tmax = 8.0 * d * N * N
        ts = np.linspace(tmax / 1600, tmax, 1600)
        u1s, lcs, gps = [], [], []
        for t in ts:
            u0, u1, u2, u3 = u_derivs_product(N, d, j, k, mp.mpf(float(t)))
            u1s.append(u1)
            lcs.append((u3 * u1 - u2 ** 2) / u1 ** 2)          # (log u')''
            gps.append(float(np.sum(-al * cg * np.exp(-al * t))))
        u1max = max(u1s)
        # the spectral sum for u' is accurate to ~1e-68 absolute; use only times with u' > 1e-30 * max
        rel = [i for i in range(len(ts)) if u1s[i] > mp.mpf(10) ** (-30) * u1max]
        upos = all(u1s[i] > 0 for i in range(len(ts)))
        lcmax = max(float(lcs[i]) for i in rel)
        g = np.array([float(np.sum(cg * np.exp(-al * t))) for t in ts])
        good = [i for i in rel if g[i] > 1e-7 * g.max()]      # float g' is unreliable where g is ~0
        Bg = [mp.mpf(gps[i]) / u1s[i] for i in good]
        maxinc = max(float((Bg[i + 1] - Bg[i]) / (abs(Bg[i]) + abs(Bg[i + 1]))) for i in range(len(Bg) - 1))
        sg = np.sign(np.array([gps[i] for i in good])); sg = sg[sg != 0]
        nchg = int(np.sum(sg[1:] != sg[:-1]))
        case_ok = upos and lcmax <= 1e-25 and maxinc <= 1e-7 and nchg == 1
        ok &= case_ok
        det.append("%s d%d N%d: max(log u')''=%.2e, max rel increase of B=%.1e, sign changes of g'=%d, grid pts used %d"
                   % (geo, d, N, lcmax, maxinc, nchg, len(good)))
    record("W4 Theorem 4.12 (continuous): u'>0, (log u')''<=0 (mp 70 digits), B=g'/u' nonincreasing, g' changes sign once "
           "(float killed spectrum), 20 cases, grid of 1600 times up to 8dN^2 (restricted to u' > 1e-30 max u')", ok, " | ".join(det))


# ---------------------------------------------------------------------------
# W5  1D birth-death end-to-end propagator (Lemma 4.9): discrete and continuous
# ---------------------------------------------------------------------------
def bd_matrix_full(N, q):
    """1D lazy reflecting walk on N sites (Fractions)."""
    K = [[Fraction(0)] * N for _ in range(N)]
    for i in range(N):
        for dx in (-1, 1):
            jn = i + dx
            if 0 <= jn < N:
                K[i][jn] += q / 2
            else:
                K[i][i] += q / 2
        K[i][i] += 1 - q
    return K


def bd_matrix_folded(N, q):
    """Y = |X - c| on {0..M}, M=(N-1)/2 (N odd)."""
    M = (N - 1) // 2
    K = [[Fraction(0)] * (M + 1) for _ in range(M + 1)]
    for y in range(M + 1):
        if y == 0:
            K[0][1] = q; K[0][0] = 1 - q
        elif y == M:
            K[M][M - 1] = q / 2; K[M][M] = 1 - q / 2
        else:
            K[y][y - 1] = q / 2; K[y][y + 1] = q / 2; K[y][y] = 1 - q
    return K


def mat_pow_entry(K, i, j, T):
    n = len(K)
    row = [Fraction(int(c == i)) for c in range(n)]
    out = [row[j]]
    for _ in range(T):
        row = [sum(row[a] * K[a][b] for a in range(n) if row[a] != 0 and K[a][b] != 0) for b in range(n)]
        out.append(row[j])
    return out


def w5():
    mp.mp.dps = 50
    ok = True
    det = []
    for N in (2, 3, 5, 8, 11):
        for q in (Fraction(1, 2), Fraction(1, 3), Fraction(1, 10)):
            T = 6 * N + 30
            h = mat_pow_entry(bd_matrix_full(N, q), 0, N - 1, T)
            lam = [1 - mp.mpf(q.numerator) / q.denominator * (1 - mp.cos(kk * mp.pi / N)) for kk in range(1, N)]
            ok &= all(l > -mp.mpf(10) ** (-40) for l in lam)
            # pmf of the sum of independent geometrics on {1,2,..} with success prob 1-lam_k
            pmf = [mp.mpf(1)]
            for l in lam:
                new = [mp.mpf(0)] * (T + 1)
                for n0, pv in enumerate(pmf):
                    if pv == 0:
                        continue
                    for jj in range(1, T + 1 - n0):
                        new[n0 + jj] += pv * (1 - l) * l ** (jj - 1)
                pmf = new + [mp.mpf(0)] * (T + 1 - len(new))
            err = max(abs(mp.mpf(N) * (mp.mpf(h[n].numerator) / h[n].denominator
                                          - (mp.mpf(h[n - 1].numerator) / h[n - 1].denominator if n else 0)) - pmf[n])
                      for n in range(T + 1))
            ok &= err < mp.mpf(10) ** (-40)
            # exact log-concavity of dh and of h on the window
            dh = [h[0]] + [h[n] - h[n - 1] for n in range(1, T + 1)]
            ok &= all(v >= 0 for v in dh)
            ok &= all(dh[n] ** 2 >= dh[n - 1] * dh[n + 1] for n in range(1, T))
            ok &= all(h[n] ** 2 >= h[n - 1] * h[n + 1] for n in range(1, T))
    det.append("end-to-end N in {2,3,5,8,11}, q in {1/2,1/3,1/10}: N*dh = pmf of sum of geometrics (err<1e-40), dh and h log-concave (exact)")
    # folded chain = centre geometry
    for N in (3, 5, 7, 11):
        M = (N - 1) // 2
        for q in (Fraction(1, 2), Fraction(1, 3), Fraction(4, 5)):
            T = 5 * N + 20
            full = mat_pow_entry(bd_matrix_full(N, q), 0, M, T)       # 1 -> centre
            fold = mat_pow_entry(bd_matrix_folded(N, q), M, 0, T)     # far end -> 0
            ok &= (full == fold)
            if q <= Fraction(1, 2):
                dh = [full[0]] + [full[n] - full[n - 1] for n in range(1, T + 1)]
                ok &= all(v >= 0 for v in dh)
                ok &= all(dh[n] ** 2 >= dh[n - 1] * dh[n + 1] for n in range(1, T))
                ok &= all(full[n] ** 2 >= full[n - 1] * full[n + 1] for n in range(1, T))
    det.append("corner->centre N in {3,5,7,11}: P_q^n(1,c) == folded birth-death chain end-to-end (exact); log-concave for q<=1/2")
    # continuous: Laplace transform of p_t(1,c) equals (1/N)(1/s) prod theta_k/(s+theta_k), theta_k = c(1-cos(2k pi/N))
    for N in (3, 5, 9):
        M = (N - 1) // 2
        cc = mp.mpf(1)
        Lm = mp.zeros(N, N)
        for i in range(N):
            for dx in (-1, 1):
                jn = i + dx
                if 0 <= jn < N:
                    Lm[i, jn] -= cc / 2
                    Lm[i, i] += cc / 2
        for sv in (mp.mpf("0.3"), mp.mpf(2)):
            Minv = mp.inverse(sv * mp.eye(N) + Lm)
            lhs = Minv[0, M]
            rhs = (mp.mpf(1) / N) / sv
            for kk in range(1, M + 1):
                th = cc * (1 - mp.cos(2 * kk * mp.pi / N))
                rhs *= th / (sv + th)
            ok &= abs(lhs - rhs) < mp.mpf(10) ** (-40)
            lhs2 = Minv[0, N - 1]
            rhs2 = (mp.mpf(1) / N) / sv
            for kk in range(1, N):
                th = cc * (1 - mp.cos(kk * mp.pi / N))
                rhs2 *= th / (sv + th)
            ok &= abs(lhs2 - rhs2) < mp.mpf(10) ** (-40)
    det.append("continuous: resolvent entries (1,N) and (1,c) equal (1/N)(1/s)prod theta/(s+theta) (mp, err<1e-40)")
    record("W5 Lemma 4.9: end-to-end propagator of a birth-death chain = pi * CDF of a sum of geometric/exponential variables; "
           "folding for the centre", ok, " | ".join(det))


# ---------------------------------------------------------------------------
# W6  log-concavity lemmas (Lemma 4.10): convolution, binomial convolution, with leading zeros (exact, random)
# ---------------------------------------------------------------------------
def random_lc(rng, length, lead):
    """random log-concave nonnegative rational sequence: lead zeros, then positive with nonincreasing ratios."""
    ratios = sorted([Fraction(rng.randint(1, 40), rng.randint(1, 12)) for _ in range(length - 1)], reverse=True)
    seq = [Fraction(rng.randint(1, 9))]
    for r in ratios:
        seq.append(seq[-1] * r)
    return [Fraction(0)] * lead + seq


def is_lc(seq):
    nzidx = [i for i, v in enumerate(seq) if v != 0]
    if not nzidx:
        return True
    if any(v < 0 for v in seq):
        return False
    if any(seq[i] == 0 for i in range(nzidx[0], nzidx[-1] + 1)):
        return False
    return all(seq[i] ** 2 >= seq[i - 1] * seq[i + 1] for i in range(1, len(seq) - 1))


def w6():
    rng = random.Random(20261001)
    ok = True
    for _ in range(600):
        la, lb = rng.randint(1, 9), rng.randint(1, 9)
        a = random_lc(rng, la, rng.randint(0, 3))
        b = random_lc(rng, lb, rng.randint(0, 3))
        ok &= is_lc(a) and is_lc(b)
        n = len(a) + len(b) - 1
        conv = [sum(a[k] * b[m - k] for k in range(len(a)) if 0 <= m - k < len(b)) for m in range(n)]
        ok &= is_lc(conv)
        # binomial convolution is only "valid" (no truncation effects) for n < min(len) ... use infinite extension by 0
        # -> truncation of a log-concave sequence is log-concave, so the full range is legitimate.
        bc = [sum(math.comb(m, k) * a[k] * b[m - k] for k in range(len(a)) if 0 <= m - k < len(b)) for m in range(n)]
        ok &= is_lc(bc)
    record("W6 Lemma 4.10: ordinary and binomial convolution of nonnegative log-concave sequences without internal zeros "
           "(leading zeros allowed) are log-concave (600 random exact trials)", ok)


# ---------------------------------------------------------------------------
# W7  strong-unimodality lemma (Lemma 4.8), random exact tests (discrete) and a float test (continuous)
# ---------------------------------------------------------------------------
def w7():
    rng = random.Random(7)
    ok = True
    for _ in range(600):
        phi = random_lc(rng, rng.randint(2, 14), rng.randint(0, 4))
        # nonincreasing kappa with kappa_0 = 1
        vals = sorted([Fraction(rng.randint(0, 60), 60) for _ in range(rng.randint(1, 25))], reverse=True)
        kappa = [Fraction(1)] + vals
        n = len(phi) + len(kappa) - 1
        f = [sum(kappa[k] * phi[m - k] for k in range(len(kappa)) if 0 <= m - k < len(phi)) for m in range(n)]
        # NOTE: kappa is truncated (kappa_k = 0 afterwards): still nonincreasing, so the lemma applies on the whole line
        diffs = [f[0]] + [f[i + 1] - f[i] for i in range(n - 1)] + [-f[-1]]
        nz = [1 if v > 0 else -1 for v in diffs if v != 0]
        ok &= sum(1 for i in range(len(nz) - 1) if nz[i] != nz[i + 1]) <= 1
    # continuous float test: phi = gamma-type log-concave, r = random mixture of exponentials
    ok2 = True
    rs = np.random.RandomState(3)
    t = np.linspace(0, 60, 6001)
    dt = t[1] - t[0]
    for _ in range(60):
        k = rs.uniform(1.0, 6.0); th = rs.uniform(0.3, 3.0)
        phi = t ** k * np.exp(-t / th)
        w = rs.uniform(0, 2, size=4); a = rs.uniform(0.01, 5, size=4)
        r = (w[:, None] * np.exp(-np.outer(a, t))).sum(axis=0)
        conv = np.convolve(phi, r)[: len(t)] * dt
        gfun = phi + conv
        dg = np.diff(gfun)
        sg = np.sign(dg[np.abs(dg) > 1e-9 * np.abs(dg).max()])
        ok2 &= int(np.sum(sg[1:] != sg[:-1])) <= 1
    record("W7 Lemma 4.8 (strong unimodality): kappa*phi unimodal for log-concave phi and nonincreasing kappa "
           "(600 random exact discrete trials; 60 random float continuous trials)", ok and ok2)


# ---------------------------------------------------------------------------
# W8  discrete theorem (Theorem 4.12(ii)): phi = Delta u >= 0 log-concave for q <= 1/2, bracket monotone, binomial
#     convolution structure (exact)
# ---------------------------------------------------------------------------
def w8():
    ok = True
    det = []
    cases = [("CC", 1, 7), ("CC", 2, 2), ("CC", 2, 3), ("CC", 2, 5), ("CC", 2, 8), ("CC", 3, 3), ("CC", 3, 4),
             ("C2M", 2, 5), ("C2M", 2, 7), ("M2C", 2, 5), ("M2C", 2, 7), ("C2M", 3, 3), ("M2C", 3, 3), ("C2M", 3, 5), ("M2C", 3, 5)]
    for geo, d, N in cases:
        s, a = geom(geo, N, d)
        Dist = sum(abs(x - y) for x, y in zip(s, a))
        for (qn, qd) in ((1, 2), (1, 3), (1, 8)):
            T = min(700, int(5 * d * N * N * qd / qn) + 30)
            U, V, F, E, D = exact_sequences(N, d, qn, qd, s, a, T)
            Phi = [U[0]] + [U[t] - D * U[t - 1] for t in range(1, T + 1)]        # phi(t) D^t
            c1 = all(Phi[t] == 0 for t in range(Dist)) and all(Phi[t] > 0 for t in range(Dist, T + 1))
            # log-concavity (D-powers cancel)
            c2 = all(Phi[t] ** 2 >= Phi[t - 1] * Phi[t + 1] for t in range(1, T))
            # bracket Brk(t) = Delta f(t)/phi(t+1) nonincreasing for t+1 >= Dist:
            #   Delta f(t) = (F[t+1]-D F[t])/D^{t+1},  phi(t+1) = Phi[t+1]/D^{t+1}
            num = [F[t + 1] - D * F[t] for t in range(T)]
            c3 = all(num[t + 1] * Phi[t + 1] <= num[t] * Phi[t + 2] for t in range(Dist - 1, T - 1))
            sg = [1 if v > 0 else -1 for v in num if v != 0]
            c4 = sum(1 for i in range(len(sg) - 1) if sg[i] != sg[i + 1]) <= 1
            ok &= c1 and c2 and c3 and c4
            if not (c1 and c2 and c3 and c4):
                det.append("FAIL %s d%d N%d q=%d/%d: %s" % (geo, d, N, qn, qd, (c1, c2, c3, c4)))
    det.append("15 geometries x q in {1/2,1/3,1/8}: phi=0 for t<D, phi>0 after, phi log-concave, bracket nonincreasing, "
               "Delta f changes sign at most once (exact, windows up to 700 steps)")
    # binomial convolution identity  d^{t-1} phi(t) = (dh * h * ... * h)_{t-1}
    for geo, d, N in [("CC", 2, 4), ("CC", 3, 3), ("C2M", 2, 5), ("M2C", 3, 3)]:
        s, a = geom(geo, N, d)
        q = Fraction(1, 2)
        T = 40
        U, V, F, E, D = exact_sequences(N, d, 1, 2, s, a, T)
        phi = [Fraction(U[0])] + [Fraction(U[t], D ** t) - Fraction(U[t - 1], D ** (t - 1)) for t in range(1, T + 1)]
        h = mat_pow_entry(bd_matrix_full(N, q), s[0], a[0], T + 1)
        dh1 = [h[n + 1] - h[n] for n in range(T + 1)]          # delta h (n) = h(n+1)-h(n)
        seq = dh1[:T]
        for _ in range(d - 1):
            seq = [sum(math.comb(m, k) * seq[k] * h[m - k] for k in range(m + 1)) for m in range(T)]
        ok &= all(Fraction(d) ** (t - 1) * phi[t] == seq[t - 1] for t in range(1, T + 1))
    det.append("binomial-convolution identity d^{t-1}phi(t) = (delta h * h^{*(d-1)})_{t-1} exact for 4 cases, t<=40")
    record("W8 Theorem 4.12(ii) (discrete, q<=1/2): hypotheses and conclusion on finite windows (exact integers)", ok, " | ".join(det))


# ---------------------------------------------------------------------------
# W9  sharpness / limits: log-concavity of phi fails for q near 1; 1D threshold q_1(N)
# ---------------------------------------------------------------------------
def w9():
    det = []
    ok = True
    # 1D: first inequality of log-concavity of dh holds iff q <= 1/(1+1/sqrt(2(N-1)))  (exact rational test)
    for N in range(3, 14):
        for q in (Fraction(1, 2), Fraction(3, 5), Fraction(7, 10), Fraction(4, 5), Fraction(9, 10)):
            h = mat_pow_entry(bd_matrix_full(N, q), 0, N - 1, N + 2)
            dh = [h[0]] + [h[n] - h[n - 1] for n in range(1, N + 3)]
            first_ok = dh[N] ** 2 >= dh[N - 1] * dh[N + 1]
            # criterion (1-q)^2 * 2(N-1) >= q^2
            crit = (1 - q) ** 2 * 2 * (N - 1) >= q ** 2
            ok &= (first_ok == crit)
    det.append("1D: dh(N)^2 >= dh(N-1)dh(N+1) iff 2(N-1)(1-q)^2 >= q^2 (exact, N=3..13, 5 activities)")
    # d=2 corner pair, q=1: phi not log-concave
    U, V, F, E, D = exact_sequences(5, 2, 1, 1, (0, 0), (4, 4), 60)
    Phi = [U[0]] + [U[t] - D * U[t - 1] for t in range(1, 61)]
    viol = [t for t in range(1, 60) if Phi[t] ** 2 < Phi[t - 1] * Phi[t + 1]]
    ok &= len(viol) > 0
    det.append("d=2 N=5 q=1: phi not log-concave (first violation at t=%d)" % viol[0])
    record("W9 limits of the method: phi = Delta u is not log-concave for q close to 1", ok, " | ".join(det))


# ---------------------------------------------------------------------------
# W10  general theorem on all pairs of small boxes (float): hypothesis => unimodal
# ---------------------------------------------------------------------------
def jets_1d(N, ts, dps=30):
    """jets[(j,k)][i] = Taylor jet (p, p1, p2/2, p3/6) at ts[i] of the 1D propagator p_t(j,k) of the reflecting walk
    on {1..N} with jump rate 1/2 per direction (the caller rescales time)."""
    mp.mp.dps = dps
    out = {}
    for j in range(1, N + 1):
        for k in range(j, N + 1):
            coefs, rates = h1d_mp(N, mp.mpf(1), j, k)
            arr = []
            for t in ts:
                p0, p1, p2, p3 = (expsum(coefs, rates, t, i) for i in range(4))
                arr.append((p0, p1, p2 / 2, p3 / 6))
            out[(j, k)] = arr
            out[(k, j)] = arr
    return out


def jet_mul(a, b):
    return tuple(mp.fsum(a[i] * b[m - i] for i in range(m + 1)) for m in range(4))


def w10():
    ok = True
    det = []
    lists = {}
    for d, N in [(1, 7), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8), (3, 3), (3, 4)]:
        L, sites, idx = laplacian_q1(N, d)
        tmax = 10.0 * d * N * N
        ts = np.concatenate([np.geomspace(tmax * 1e-5, tmax / 1500, 60, endpoint=False), np.linspace(tmax / 1500, tmax, 1500)])
        # d-dim unit-rate walk: each coordinate runs at rate c = 1/d  ->  1D time = t/d (a positive rescaling of time,
        # irrelevant for positivity and log-concavity)
        jets = jets_1d(N, [mp.mpf(float(t)) / d for t in ts])
        n_hyp = n_uni = n_pairs = n_bad = 0
        hyp_list = []
        for a in sites:
            if any(a[i] > (N - 1) - a[i] for i in range(d)) or list(a) != sorted(a):
                continue    # target up to the symmetries of the box
            Lk, keep = killed(L, idx, a)
            al, Phi = np.linalg.eigh(Lk)
            rvec = -L[keep, idx[a]]
            pr = Phi.T @ rvec
            Eal = np.exp(-np.outer(al, ts))
            for s in sites:
                if s == a:
                    continue
                n_pairs += 1
                cg = Phi[keep.index(idx[s]), :] * pr
                g = cg @ Eal
                gp = (-al * cg) @ Eal
                good = g > 1e-9 * g.max()
                sg = np.sign(gp[good]); sg = sg[sg != 0]
                nch = int(np.sum(sg[1:] != sg[:-1]))
                uni = (nch == 0) if (sg.size and sg[0] < 0) else (nch <= 1)
                # hypothesis: u' > 0 and (log u')'' <= 0 on the grid, u = prod_i p_{s_i a_i}; mp jets
                vals = []
                u1max = mp.mpf(0)
                for i in range(len(ts)):
                    jet = jets[(s[0] + 1, a[0] + 1)][i]
                    for c in range(1, d):
                        jet = jet_mul(jet, jets[(s[c] + 1, a[c] + 1)][i])
                    u1, u2, u3 = jet[1], 2 * jet[2], 6 * jet[3]
                    vals.append((u1, u2, u3))
                    u1max = max(u1max, abs(u1))
                hyp = True
                for (u1, u2, u3) in vals:
                    if abs(u1) < mp.mpf(10) ** (-22) * u1max:
                        continue                                      # below the accuracy of the spectral sum
                    if u1 <= 0 or (u3 * u1 - u2 ** 2) > mp.mpf(10) ** (-12) * u1 ** 2:
                        hyp = False
                        break
                if hyp:
                    # necessary consequence of positivity + log-concavity: u' is unimodal, i.e. u'' has the sign
                    # pattern (+..+)(-..-).  (Catches zeros of u' between grid points, e.g. a double zero of u'.)
                    sg2 = [1 if v[1] > 0 else -1 for v in vals if abs(v[1]) > mp.mpf(10) ** (-22) * u1max]
                    if sum(1 for i in range(len(sg2) - 1) if sg2[i] != sg2[i + 1]) > 1 or (sg2 and sg2[-1] > 0):
                        hyp = False
                n_hyp += hyp
                n_uni += uni
                if hyp:
                    hyp_list.append((tuple(x + 1 for x in s), tuple(x + 1 for x in a)))
                if hyp and not uni:
                    n_bad += 1
        ok &= (n_bad == 0)
        lists["d%d_N%d" % (d, N)] = hyp_list
        det.append("d%d N%d: %d pairs, unimodal %d, hypothesis (u'>0, log-concave) %d, hypothesis-but-not-unimodal %d"
                   % (d, N, n_pairs, n_uni, n_hyp, n_bad))
    with open(_os.path.join(_R, 'data', 'msc_rigorous_unimodality', 'pairs_logconcave_hypothesis.json'), "w") as fh:
        json.dump(dict(note="pairs (start,target), 1-based, target reduced by symmetry, for which u'(t)=d/dt P_x0(X_t=a) is "
                            "positive and log-concave on a grid of 1560 times in (0, 10 d N^2] (mp 30 digits; grid test, not a proof)",
                       lists={k: [[list(s), list(a)] for s, a in v] for k, v in lists.items()}), fh, indent=1)
    record("W10 Theorem 4.11 on all pairs of small boxes (grid test: u' from mp jets, g' from float killed spectrum): "
           "no pair with u'>0 log-concave fails to be unimodal", ok, " | ".join(det))


if __name__ == "__main__":
    only = sys.argv[1:]
    for name, fn in [("w1", w1), ("w2", w2), ("w3", w3), ("w4", w4), ("w5", w5), ("w6", w6), ("w7", w7), ("w8", w8),
                     ("w9", w9), ("w10", w10)]:
        if only and name not in only:
            continue
        fn()
    print("ALL PASS" if all(r["ok"] for r in RESULTS) else "SOME FAIL")
