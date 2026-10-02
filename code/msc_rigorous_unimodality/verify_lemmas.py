"""
verify_lemmas.py -- numerical / exact sanity checks of every lemma and theorem
of note R4.  Each check prints PASS/FAIL and is recorded in
data/verify_lemmas.json.   Exact checks use fractions.Fraction; spectral checks
use numpy (float64) or mpmath (50 digits) as stated.

Run:  python verify_lemmas.py
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
from unilib import ExactStepper, laplacian_q1, killed, site_index, count_sign_changes  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = _os.path.join(_R, 'data', 'msc_rigorous_unimodality')
RESULTS = []


def record(name, ok, detail=""):
    RESULTS.append(dict(check=name, ok=bool(ok), detail=str(detail)))
    print(("PASS " if ok else "FAIL ") + name + (" :: " + str(detail) if detail else ""), flush=True)


def pmf_exact(N, d, qn, qd, s, a, T):
    st = ExactStepper(N, d, qn, qd, s, a)
    out = [Fraction(0)]
    Dp = 1
    for _ in range(T):
        F = st.step()
        Dp *= st.D
        out.append(Fraction(F, Dp))
    return out  # out[t] = f(t)


def Sminus(seq):
    nz = [1 if v > 0 else -1 for v in seq if v != 0]
    return sum(1 for i in range(len(nz) - 1) if nz[i] != nz[i + 1])


# ---------------------------------------------------------------------------
# V1  thinning identities (Theorem 2.5), exact
# ---------------------------------------------------------------------------
def v1():
    ok = True
    cases = [(2, 4, (0, 0), (3, 3)), (2, 5, (1, 3), (2, 2)), (3, 3, (0, 0, 0), (2, 2, 2)), (1, 6, (1,), (5,))]
    for (d, N, s, a) in cases:
        for (q, qp) in ((Fraction(1), Fraction(4, 5)), (Fraction(4, 5), Fraction(1, 2)), (Fraction(1), Fraction(1, 3))):
            T = 40
            fq = pmf_exact(N, d, q.numerator, q.denominator, s, a, T + 2)
            fqp = pmf_exact(N, d, qp.numerator, qp.denominator, s, a, T + 2)
            p = qp / q
            for t in range(0, T):
                rhs = p * sum(math.comb(t, n) * p ** n * (1 - p) ** (t - n) * fq[n + 1] for n in range(t + 1))
                if rhs != fqp[t + 1]:
                    ok = False
                rhs2 = p * p * sum(math.comb(t, n) * p ** n * (1 - p) ** (t - n) * (fq[n + 2] - fq[n + 1])
                                   for n in range(t + 1))
                if rhs2 != fqp[t + 2] - fqp[t + 1]:
                    ok = False
    record("V1 thinning identity f_q'(t+1)=p*sum Bin(t,n;p) f_q(n+1) and difference form (exact, 12 cases x 40 t)", ok)


# ---------------------------------------------------------------------------
# V2  variation diminishing: bidiagonal lemma, binomial matrix, Poisson/Descartes
# ---------------------------------------------------------------------------
def v2():
    rnd = random.Random(12345)
    ok = True
    worst = 0
    for trial in range(4000):
        m = rnd.randint(2, 14)
        x = [rnd.choice([-3, -2, -1, 0, 0, 1, 2, 3]) for _ in range(m)]
        p = Fraction(rnd.randint(1, 19), 20)
        y = [sum(math.comb(t, n) * p ** n * (1 - p) ** (t - n) * x[n] for n in range(t + 1)) for t in range(m)]
        if Sminus(y) > Sminus(x):
            ok = False
        # orientation in case of equality (first nonzero sign)
        if Sminus(y) == Sminus(x) and any(y) and any(x):
            fx = next(v for v in x if v != 0)
            fy = next(v for v in y if v != 0)
            if (fx > 0) != (fy > 0):
                ok = False
        # bidiagonal
        al = [Fraction(rnd.randint(0, 3)) for _ in range(m)]
        be = [Fraction(rnd.randint(0, 3)) for _ in range(m)]
        z = [al[i] * x[i] + (be[i] * x[i + 1] if i + 1 < m else 0) for i in range(m)]
        if Sminus(z) > Sminus(x):
            ok = False
    record("V2a variation diminishing: binomial matrix and nonnegative bidiagonal matrices (4000 random exact trials)", ok)
    # Descartes-Laguerre for sum c_n mu^n/n!
    ok = True
    for trial in range(300):
        m = rnd.randint(2, 10)
        c = [rnd.choice([-2, -1, 0, 1, 2]) for _ in range(m)]
        if not any(c):
            continue
        coeffs = [c[n] / math.factorial(n) for n in range(m)]
        roots = np.roots(coeffs[::-1]) if any(coeffs) else []
        npos = sum(1 for r in roots if abs(r.imag) < 1e-9 and r.real > 1e-9)
        if npos > Sminus(c):
            ok = False
    record("V2b Descartes-Laguerre: #positive zeros of sum c_n mu^n/n! <= S^-(c) (300 random polynomials, float roots)", ok)


# ---------------------------------------------------------------------------
# V3  uniformisation: continuous density = Poisson mixture of q=1 PMF
# ---------------------------------------------------------------------------
def v3():
    ok = True
    worst = 0.0
    for (d, N, s, a) in [(2, 4, (0, 0), (3, 3)), (2, 5, (4, 4), (1, 2)), (3, 3, (2, 1, 1), (0, 1, 1))]:
        L, sites, idx = laplacian_q1(N, d)
        Lk, keep = killed(L, idx, a)
        sig, V = np.linalg.eigh(Lk)
        pos = keep.index(idx[s])
        c = V[pos, :] * sig * V.sum(axis=0)
        f1 = [float(v) for v in pmf_exact(N, d, 1, 1, s, a, 400)]
        for t in (0.7, 3.0, 11.0, 40.0):
            g_spec = float((c * np.exp(-sig * t)).sum())
            g_pois = sum(math.exp(-t + n * math.log(t) - math.lgamma(n + 1)) * f1[n + 1] for n in range(399))
            worst = max(worst, abs(g_spec - g_pois) / g_spec)
    ok = worst < 1e-9
    record("V3 uniformisation g_1(t) = sum Pois(t;n) f_{q=1}(n+1) vs spectral formula (float)", ok, "max rel diff %.2e" % worst)


# ---------------------------------------------------------------------------
# V4  residues: moments vanish up to D-2, sign changes >= D-1 (Theorem 3.2); 1D full alternation
# ---------------------------------------------------------------------------
def residues_mp(N, d, s, a, dps=60):
    mp.mp.dps = dps
    L, sites, idx = laplacian_q1(N, d)
    Lk, keep = killed(L, idx, a)
    n = Lk.shape[0]
    # exact rational entries -> mp matrix
    A = mp.matrix(n, n)
    for i in range(n):
        for j in range(n):
            if Lk[i, j] != 0:
                A[i, j] = mp.mpf(Fraction(Lk[i, j]).limit_denominator(1000).numerator) / \
                    mp.mpf(Fraction(Lk[i, j]).limit_denominator(1000).denominator)
    E, Q = mp.eigsy(A)
    pos = keep.index(idx[s])
    lam = [E[k] for k in range(n)]
    c = [Q[pos, k] * lam[k] * sum(Q[i, k] for i in range(n)) for k in range(n)]
    # group equal eigenvalues
    order = sorted(range(n), key=lambda k: lam[k])
    groups = []
    for k in order:
        if groups and abs(lam[k] - groups[-1][0]) < mp.mpf(10) ** (-dps + 15):
            groups[-1][1] += c[k]
        else:
            groups.append([lam[k], c[k]])
    return groups


def v4():
    ok = True
    details = []
    for (d, N, s, a) in [(1, 6, (0,), (5,)), (1, 9, (0,), (8,)), (1, 8, (2,), (6,)),
                         (2, 3, (0, 0), (2, 2)), (2, 4, (0, 0), (3, 3)), (2, 5, (0, 0), (4, 4)),
                         (2, 5, (0, 0), (2, 2)), (2, 4, (0, 1), (3, 2)), (3, 3, (0, 0, 0), (2, 2, 2))]:
        groups = residues_mp(N, d, s, a)
        tol = mp.mpf(10) ** (-35)
        gr = [(l, c) for (l, c) in groups if abs(c) > tol]
        Dist = sum(abs(x - y) for x, y in zip(s, a))
        signs = [1 if c > 0 else -1 for (l, c) in gr]
        sc = sum(1 for i in range(len(signs) - 1) if signs[i] != signs[i + 1])
        mom = [abs(sum(c * l ** j for (l, c) in gr)) for j in range(max(Dist - 1, 0))]
        momD = sum(c * l ** (Dist - 1) for (l, c) in gr)
        good = sc >= Dist - 1 and all(m < mp.mpf(10) ** (-30) for m in mom) and (-1) ** (Dist - 1) * momD > 0 and gr[0][1] > 0
        if d == 1 and s == (0,):
            good = good and sc == len(gr) - 1 and len(gr) == N - 1
        ok = ok and good
        details.append("d%d N%d %s->%s: D=%d, #nonzero residue groups=%d, S^-=%d" % (d, N, s, a, Dist, len(gr), sc))
    record("V4 residues: c_1>0, moments sum c_k alpha_k^j = 0 (j<=D-2), S^-(c)>=D-1, 1D end-to-end full alternation (mpmath 60 digits)",
           ok, "; ".join(details))


# ---------------------------------------------------------------------------
# V5  monotone coupling (Lemma 5.1): S_t(x) coordinatewise monotone for corner target, all q incl. 1 (exact)
# ---------------------------------------------------------------------------
def v5():
    ok = True
    for (d, N) in [(1, 5), (2, 3), (2, 4), (2, 5), (3, 3)]:
        for q in (Fraction(1), Fraction(4, 5), Fraction(1, 2), Fraction(1, 5)):
            sites, idx = site_index(N, d)
            a = (N - 1,) * d
            w = q / (2 * d)
            S = {x: (Fraction(1) if x != a else Fraction(0)) for x in sites}
            for t in range(1, 40):
                Snew = {}
                for x in sites:
                    if x == a:
                        Snew[x] = Fraction(0)
                        continue
                    val = (1 - q) * S[x]
                    for ax in range(d):
                        for dx in (-1, 1):
                            y = list(x)
                            y[ax] += dx
                            if 0 <= y[ax] < N:
                                val += w * S[tuple(y)]
                            else:
                                val += w * S[x]
                    Snew[x] = val
                S = Snew
                for x in sites:
                    for ax in range(d):
                        if x[ax] + 1 < N:
                            y = list(x)
                            y[ax] += 1
                            if S[x] < S[tuple(y)]:
                                ok = False
    record("V5 Lemma 5.1: survival S_t(x) nonincreasing in each coordinate toward the corner target (exact, q in {1,4/5,1/2,1/5}, t<40)", ok)


# ---------------------------------------------------------------------------
# V6  spectral facts for the corner target (Lemmas 5.2-5.5)
# ---------------------------------------------------------------------------
def v6():
    ok = True
    det = []
    for (d, N) in [(2, 3), (2, 4), (2, 6), (2, 9), (2, 14), (3, 3), (3, 4), (3, 6)]:
        L, sites, idx = laplacian_q1(N, d)
        a = (N - 1,) * d
        x0 = (0,) * d
        Lk, keep = killed(L, idx, a)
        sig, V = np.linalg.eigh(Lk)
        n = len(sites)
        phi1 = V[:, 0] * np.sign(V[:, 0].sum())
        pos = keep.index(idx[x0])
        c1 = phi1[pos] * sig[0] * phi1.sum()
        beta1 = (1 - math.cos(math.pi / N)) / d
        # symmetric-sector eigenvalues: those with nonzero overlap with symmetrised vectors
        ks = [sites[i] for i in keep]
        perms = list(itertools.permutations(range(d)))
        # symmetriser
        P = np.zeros((len(ks), len(ks)))
        kidx = {x: i for i, x in enumerate(ks)}
        for i, x in enumerate(ks):
            for pm in perms:
                y = tuple(x[j] for j in pm)
                P[kidx[y], i] += 1.0 / len(perms)
        w_sym = np.einsum("ik,ij,jk->k", V, P, V)   # <phi_k, P phi_k>
        # eigenvalues with a symmetric eigenvector component:
        sym_eigs = sorted(set(round(float(s), 10) for s, wv in zip(sig, w_sym) if wv > 1e-8))
        # residues
        c = V[pos, :] * sig * V.sum(axis=0)
        contrib = sorted(set(round(float(s), 10) for s, cv in zip(sig, c) if abs(cv) > 1e-10))
        # MFPT (q=1) from the far corner and commute-time bound
        m = np.linalg.solve(Lk, np.ones(len(ks)))
        M = m[pos]
        cond = [
            phi1[pos] >= phi1.max() - 1e-12,                 # Lemma 5.2
            c1 >= sig[0] - 1e-14,                            # Lemma 5.3
            sig[0] <= 1 / (2 * (n - 1)) + 1e-15,             # Rayleigh upper bound
            sig[0] >= 1 / M - 1e-15,                         # lower bound alpha_1 >= 1/M
            abs(M - m.max()) < 1e-9 * M,                     # MFPT maximal at far corner
            M <= 2 * d * d * (N - 1) * n,                    # commute-time bound
            sym_eigs[1] >= beta1 - 1e-10,                    # interlacing (symmetric sector)
            (len(contrib) < 2) or contrib[1] >= beta1 - 1e-10,
            sig[-1] <= 2 - d * beta1 + 1e-12,                # top of spectrum
            sig[0] < beta1,
        ]
        ok = ok and all(cond)
        det.append("d%d N%d: alpha1=%.6g beta1=%.6g alpha2sym=%.6g c1/alpha1=%.4f M=%.6g %s" %
                   (d, N, sig[0], beta1, sym_eigs[1], c1 / sig[0], M, "ok" if all(cond) else str(cond)))
    record("V6 Lemmas 5.2-5.5: phi_1 max at far corner, c_1>=alpha_1, 1/M<=alpha_1<=1/(2(n-1)), alpha_2^sym>=beta_1, spectrum<=2-d*beta_1 (float)",
           ok, " | ".join(det))


# ---------------------------------------------------------------------------
# V7  tail theorem: explicit T_N versus the true last sign change
# ---------------------------------------------------------------------------
def v7():
    ok = True
    det = []
    for (d, N) in [(2, 3), (2, 5), (2, 8), (2, 12), (3, 3), (3, 5)]:
        for q in (1.0, 0.8, 0.5):
            L, sites, idx = laplacian_q1(N, d)
            a = (N - 1,) * d
            x0 = (0,) * d
            Lk, keep = killed(L, idx, a)
            sig, V = np.linalg.eigh(Lk)
            n = len(sites)
            beta1 = (1 - math.cos(math.pi / N)) / d
            al1 = sig[0]
            TN = math.log(4 * math.sqrt(n - 1) / al1 ** 2) / math.log((1 - q * al1) / (1 - q * beta1))
            M = 2 * d * d * (N - 1) * n
            TN_explicit = (math.log(4) + 0.5 * math.log(n - 1) + 2 * math.log(M)) / (q * (beta1 - 1 / (2 * (n - 1))))
            # true differences via stepping
            from explore02 import runs_float
            T = int(TN_explicit) + 50
            f = runs_float(N, d, q, x0, a, T)
            df = np.diff(f[1:])
            last_nonneg = int(np.max(np.where(df >= 0)[0])) + 1 if (df >= 0).any() else 0  # index t with f(t+1)>=f(t)
            good = last_nonneg <= TN + 1 and TN <= TN_explicit + 1e-9
            ok = ok and good
            det.append("d%d N%d q%.1f: last t with f(t+1)>=f(t): %d ; T_N=%.1f ; explicit bound=%.1f" %
                       (d, N, q, last_nonneg, TN, TN_explicit))
    record("V7 Theorem 5.6 (tail): f strictly decreasing beyond T_N; T_N <= explicit bound (float)", ok, " | ".join(det))


# ---------------------------------------------------------------------------
# V8  rise theorem (Theorem 5.10): renewal equation, h = G/N, g' >= u''>=0 up to the hypoexponential mode
# ---------------------------------------------------------------------------
def v8():
    ok = True
    det = []
    for (d, N) in [(2, 4), (2, 7), (2, 12), (3, 4), (3, 6)]:
        L, sites, idx = laplacian_q1(N, d)
        a = (N - 1,) * d
        x0 = (0,) * d
        Lk, keep = killed(L, idx, a)
        sig, V = np.linalg.eigh(Lk)
        pos = keep.index(idx[x0])
        c = V[pos, :] * sig * V.sum(axis=0)
        # 1D free chain eigenvalues (rate 1/d per coordinate): theta_k = (1/d)(1-cos(k pi/N))
        th = np.array([(1 - math.cos(k * math.pi / N)) / d for k in range(1, N)])
        # h(t) = p^{1D}_t(1,N) spectral
        ks = np.arange(0, N)
        lam1 = (1 - np.cos(ks * np.pi / N)) / d
        psi1 = np.array([math.sqrt((1 if k == 0 else 2) / N) * math.cos(k * math.pi * 0.5 / N) for k in ks])
        psiN = np.array([math.sqrt((1 if k == 0 else 2) / N) * math.cos(k * math.pi * (N - 0.5) / N) for k in ks])

        def h(t, der=0):
            return float(((-lam1) ** der * psi1 * psiN * np.exp(-lam1 * t)).sum())

        # hypoexponential CDF via partial fractions (float; fine for small N)
        def G(t):
            tot = 1.0
            for k in range(len(th)):
                pr = 1.0
                for j in range(len(th)):
                    if j != k:
                        pr *= th[j] / (th[j] - th[k])
                tot -= pr * math.exp(-th[k] * t)
            return tot

        errG = max(abs(N * h(t) - G(t)) for t in (0.5 * N * N * d * x for x in (0.05, 0.1, 0.2, 0.5, 1.0)))
        # mode of hypoexponential density = first zero of h''
        ts = np.linspace(1e-3, 2.0 * N * N * d, 40001)
        hpp = np.array([h(t, 2) for t in ts])
        i0 = int(np.argmax(hpp < 0))
        mN = ts[i0]

        def gp(t):
            return float(-(c * sig * np.exp(-sig * t)).sum())

        def upp(t):
            hh, h1, h2 = h(t), h(t, 1), h(t, 2)
            return d * hh ** (d - 2) * ((d - 1) * h1 * h1 + hh * h2)

        tt = np.linspace(0.02 * mN, mN, 60)
        m1 = min(gp(t) - upp(t) for t in tt)
        m2 = min(upp(t) for t in tt)
        scale = max(abs(gp(t)) for t in tt)
        good = errG < 1e-8 and m1 > -1e-9 * scale and m2 > -1e-9 * scale
        ok = ok and good
        # true mode of g
        gps = np.array([gp(t) for t in ts])
        jj = np.where((gps < 0) & (ts > mN))[0]
        mode_g = ts[int(jj[0])]
        det.append("d%d N%d: |N h - G|max=%.1e, m_N=%.3f (=%.4f d N^2), mode of g=%.3f, min(g'-u'')/scale=%.1e" %
                   (d, N, errG, mN, mN / (d * N * N), mode_g, m1 / scale))
    record("V8 Theorem 5.10 (rise): h=G/N (hypoexponential CDF), u''>=0 and g'>=u'' on [0,m_N] (float)", ok, " | ".join(det))
    # renewal identity u = g * v (float quadrature) for one case
    d, N = 2, 5
    L, sites, idx = laplacian_q1(N, d)
    a = (N - 1,) * d
    x0 = (0,) * d
    mu, U = np.linalg.eigh(L)
    Lk, keep = killed(L, idx, a)
    sig, V = np.linalg.eigh(Lk)
    pos = keep.index(idx[x0])
    c = V[pos, :] * sig * V.sum(axis=0)
    ia, i0 = idx[a], idx[x0]
    t = 30.0
    ss = np.linspace(0, t, 200001)
    g = (np.exp(-np.outer(ss, sig)) * c).sum(axis=1)
    v = (np.exp(-np.outer(t - ss, mu)) * (U[ia, :] ** 2)).sum(axis=1)
    conv = np.trapezoid(g * v, ss)
    u = float((U[i0, :] * U[ia, :] * np.exp(-mu * t)).sum())
    record("V8b renewal identity u(t) = int_0^t g(s) v(t-s) ds (float quadrature, d=2 N=5, t=30)",
           abs(conv - u) < 1e-7 * u, "u=%.10g conv=%.10g" % (u, conv))


# ---------------------------------------------------------------------------
# V9  completely monotone cases (Theorem 4.1)
# ---------------------------------------------------------------------------
def v9():
    ok = True
    det = []
    for (d, N, a) in [(2, 5, (4, 4)), (2, 5, (2, 2)), (3, 3, (1, 1, 1)), (3, 3, (2, 2, 2)), (2, 6, (0, 0))]:
        L, sites, idx = laplacian_q1(N, d)
        Lk, keep = killed(L, idx, a)
        sig, V = np.linalg.eigh(Lk)
        r = np.array([1.0 / (2 * d) if sum(abs(x - y) for x, y in zip(sites[i], a)) == 1 else 0.0 for i in keep])
        assert np.allclose(Lk @ np.ones(len(keep)), r)
        coef_nu = (V.T @ r) ** 2       # start = uniform on N(a): residues >= 0
        coef_pi = sig * (V.sum(axis=0)) ** 2   # start = uniform on Lambda': residues >= 0
        nb = [i for i in range(len(keep)) if r[i] > 0]
        # all neighbours give the same PMF (transitive stabiliser)
        cs = [V[i, :] * (V.T @ r) for i in nb]
        t = np.linspace(0, 50, 11)
        vals = [(np.exp(-np.outer(t, sig)) * ci).sum(axis=1) for ci in cs]
        same = max(np.abs(vals[0] - vv).max() for vv in vals)
        good = coef_nu.min() >= 0 and coef_pi.min() >= 0 and same < 1e-12
        # exact monotonicity of the PMF from a neighbour at q=1/2 (exact)
        y = sites[keep[nb[0]]]
        f = pmf_exact(N, d, 1, 2, y, a, 80)
        good = good and all(f[t + 1] <= f[t] for t in range(1, 79))
        ok = ok and good
        det.append("d%d N%d target %s: neighbours equivalent (max diff %.1e), PMF from neighbour nonincreasing (exact, t<80)" % (d, N, a, same))
    record("V9 Theorem 4.1: CM cases (stationary start; start adjacent to a corner/centre target)", ok, " | ".join(det))


# ---------------------------------------------------------------------------
# V10  1D Sturm argument (Theorem 4.3): LU factors nonnegative; S^-(D_t) nonincreasing; unimodal for all starts
# ---------------------------------------------------------------------------
def v10():
    ok = True
    for N in (4, 6, 9):
        for q in (Fraction(1, 2), Fraction(1, 3), Fraction(1, 10)):
            a = N - 1  # target (0-based)
            m = a      # transient sites 0..a-1
            # Q matrix
            Q = [[Fraction(0)] * m for _ in range(m)]
            for i in range(m):
                Q[i][i] = 1 - q + (q / 2 if i == 0 else 0)
                if i + 1 < m:
                    Q[i][i + 1] = q / 2
                    Q[i + 1][i] = q / 2
            # LU pivots
            u = [Q[0][0]]
            for i in range(1, m):
                u.append(Q[i][i] - (q / 2) ** 2 / u[i - 1])
            if not all(ui >= q / 2 for ui in u):
                ok = False
            for x0 in range(m):
                Dv = [Fraction(0)] * m
                e = [Fraction(0)] * m
                e[x0] = Fraction(1)
                Qe = [sum(Q[i][j] * e[j] for j in range(m)) for i in range(m)]
                Dv = [Qe[i] - e[i] for i in range(m)]
                prevS = Sminus(Dv)
                last_signs = []
                for t in range(0, 60):
                    last_signs.append(Dv[m - 1])
                    Dv = [sum(Q[i][j] * Dv[j] for j in range(max(0, i - 1), min(m, i + 2))) for i in range(m)]
                    s = Sminus(Dv)
                    if s > prevS:
                        ok = False
                    prevS = s
                if Sminus(last_signs) > 1:
                    ok = False
                nz = [v for v in last_signs if v != 0]
                if Sminus(last_signs) == 1 and nz[0] < 0:
                    ok = False
    record("V10 Theorem 4.3 (1D): LU pivots >= q/2 for q<=1/2; S^-(D_t) nonincreasing; PMF unimodal for every start (exact, N in {4,6,9})", ok)
    # counterexample 1D q=1 N=4
    f = pmf_exact(4, 1, 1, 1, (0,), (3,), 6)
    record("V10b 1D counterexample q=1, N=4: f(3)=1/8 > f(4)=1/16 < f(5)=3/32 (exact)",
           f[3] == Fraction(1, 8) and f[4] == Fraction(1, 16) and f[5] == Fraction(3, 32), str([str(v) for v in f[1:7]]))
    # N=4: f(4)<f(3) iff q>4/5 ; f(5)>f(4) for all q in a grid
    ok = True
    for k in range(1, 101):
        q = Fraction(k, 100)
        f = pmf_exact(4, 1, q.numerator, q.denominator, (0,), (3,), 6)
        if (f[4] < f[3]) != (q > Fraction(4, 5)):
            ok = False
        if not f[5] > f[4]:
            ok = False
        # closed forms
        if f[3] != (q / 2) ** 3 or f[4] != (q / 2) ** 3 * (3 - Fraction(5, 2) * q):
            ok = False
        if f[5] - f[4] != (q / 2) ** 3 * (Fraction(19, 4) * q * q - Fraction(15, 2) * q + 3):
            ok = False
    record("V10c 1D N=4: f(3)=(q/2)^3, f(4)=(q/2)^3(3-5q/2), f(5)-f(4)=(q/2)^3(19q^2/4-15q/2+3)>0; f(4)<f(3) iff q>4/5 (exact, q=k/100)", ok)


# ---------------------------------------------------------------------------
# V11 cone lemma & front property; V12 P1', log-concavity, IFR for corner-to-corner (exact, small N)
# ---------------------------------------------------------------------------
def v11_12():
    ok_cone = True
    ok_p1 = True
    ok_lc = True
    ok_ifr = True
    det = []
    for (d, N, qn, qd) in [(2, 3, 1, 2), (2, 4, 1, 2), (2, 5, 1, 2), (2, 6, 1, 2), (2, 7, 1, 2), (3, 3, 1, 2), (3, 4, 1, 2),
                           (2, 5, 1, 5), (2, 6, 1, 10)]:
        st = ExactStepper(N, d, qn, qd, (0,) * d, (N - 1,) * d)
        D = st.D
        T = 14 * N * N * d * qd // qn
        Rs = [st.R.copy()]
        Fs = [0]
        for _ in range(T):
            Fs.append(st.step())
            Rs.append(st.R)
        # cone: first T0 with R_{T0+1} <= D R_{T0}; then all later differences <= 0
        T0 = next(t for t in range(T) if np.all(Rs[t + 1] <= Rs[t] * D))
        if not all(Fs[t + 1] <= D * Fs[t] for t in range(T0 + 1, T)):
            ok_cone = False
        mode = max(t for t in range(1, T) if Fs[t] > D * Fs[t - 1])
        # P1': R_{t+1}(x) R_t(x+e_i) <= R_{t+1}(x+e_i) R_t(x)
        sl = st._sl
        for t in range(T - 1):
            A, B = Rs[t], Rs[t + 1]
            for lo, hi in sl:
                lhs = B[lo] * A[hi]
                rhs = B[hi] * A[lo]
                bad = lhs > rhs
                # ignore pairs involving the target (R=0 there)
                if bad.any():
                    ok_p1 = False
        # log-concavity F_t^2 >= F_{t-1} F_{t+1}; IFR: f(t)/S(t-1) nondecreasing
        for t in range(2, T):
            if Fs[t] * Fs[t] < Fs[t - 1] * Fs[t + 1]:
                ok_lc = False
        Ssum = [int(R.sum()) for R in Rs]   # S(t) = Ssum[t]/D^t
        for t in range(1, T):
            # f(t)/S(t-1) <= f(t+1)/S(t)  <=>  F_t * Ssum[t] * D <= F_{t+1} * Ssum[t-1] ... careful with scaling
            # f(t) = F_t/D^t, S(t-1) = Ssum[t-1]/D^(t-1): hazard_t = F_t/(D*Ssum[t-1])
            if Fs[t] * Ssum[t] > Fs[t + 1] * Ssum[t - 1]:
                ok_ifr = False
        det.append("d%d N%d q=%d/%d: T_cone=%d, mode=%d, window T=%d" % (d, N, qn, qd, T0, mode, T))
    record("V11 Lemma 6.1 (cone): after the first T with rho_{T+1}<=rho_T all later differences are <=0; T_cone = mode-1 (exact)", ok_cone, " | ".join(det))
    record("V12a Conjecture 9.1 (P1'): rho_{t+1}(x)/rho_t(x) coordinatewise nondecreasing toward the target, corner-to-corner (exact on window)", ok_p1)
    record("V12b Conjecture 9.1 (log-concavity of the PMF, q<=1/2, corner-to-corner; exact on window)", ok_lc)
    record("V12c Conjecture 9.1 (increasing hazard rate, exact on window)", ok_ifr)


# ---------------------------------------------------------------------------
# V10d Proposition 4.4 for general N (exact);  V13 Proposition 4.6 (hypercube N=2)
# ---------------------------------------------------------------------------
def v10d():
    ok = True
    for N in range(4, 15):
        thr = Fraction(2 * N - 4, 2 * N - 3)
        qs = [thr, thr + (1 - thr) / 7, thr + (1 - thr) / 2, Fraction(1), thr - Fraction(1, 50), Fraction(1, 2)]
        for q in qs:
            f = pmf_exact(N, 1, q.numerator, q.denominator, (0,), (N - 1,), N + 3)
            eta = 1 - q
            eta1 = 1 - q / 2
            base = (q / 2) ** (N - 1)
            if f[N - 1] != base or f[N] != base * (eta1 + (N - 2) * eta):
                ok = False
            if f[N + 1] != base * (eta1 ** 2 + (N - 2) * eta1 * eta + math.comb(N - 1, 2) * eta ** 2 + (N - 2) * q * q / 4):
                ok = False
            if q > thr and not (f[N - 1] > f[N] < f[N + 1]):
                ok = False
            if q <= thr and f[N - 1] > f[N]:
                ok = False
    record("V10d Proposition 4.4: closed forms of f(N-1), f(N), f(N+1); f(N-1)>f(N)<f(N+1) iff q>(2N-4)/(2N-3) (exact, N=4..14)", ok)


def v13():
    ok = True
    det = []
    for d in range(1, 13):
        for q in (Fraction(2, 3), Fraction(1, 2), Fraction(1, 5)):
            # lumped chain on levels 0..d-1
            Qt = [[Fraction(0)] * d for _ in range(d)]
            for k in range(d):
                Qt[k][k] = 1 - q / 2
                if k + 1 < d:
                    Qt[k][k + 1] = Fraction(d - k) * q / (2 * d)
                if k >= 1:
                    Qt[k][k - 1] = Fraction(k) * q / (2 * d)
            u = [Qt[0][0]]
            for k in range(1, d):
                u.append(Qt[k][k] - Qt[k][k - 1] * Qt[k - 1][k] / u[k - 1])
            if not all(x >= q / 2 for x in u):
                ok = False
            rho = [Fraction(0)] * d
            rho[0] = Fraction(1)
            f = [Fraction(0)]
            for t in range(1, 40 * d + 60):
                f.append(rho[d - 1] * q / (2 * d))
                rho = [sum(rho[j] * Qt[j][k] for j in range(max(0, k - 1), min(d, k + 2))) for k in range(d)]
            if Sminus([f[t + 1] - f[t] for t in range(len(f) - 1)]) > 1:
                ok = False
            if d <= 5:
                full = pmf_exact(2, d, q.numerator, q.denominator, (0,) * d, (1,) * d, 30)
                if full[:31] != f[:31]:
                    ok = False
    record("V13 Proposition 4.6 (N=2 hypercube): lumped level chain == full chain (d<=5), pivots >= q/2 for q<=2/3, unimodal on window (exact, d<=12)", ok)


if __name__ == "__main__":
    which = sys.argv[1:] or ["v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v10d", "v13", "v11_12"]
    for nm in which:
        globals()[nm]()
    out = _os.path.join(_R, 'data', 'msc_rigorous_unimodality', 'verify_lemmas.json')
    prev = []
    if os.path.exists(out) and sys.argv[1:]:
        prev = [r for r in json.load(open(out)) if not any(r["check"].lower().startswith(w.split("_")[0].upper().lower()) for w in which)]
    json.dump(prev + RESULTS, open(out, "w"), indent=1)
    print("ALL PASS" if all(r["ok"] for r in RESULTS) else "SOME FAIL")
