"""
verify_lc1d.py -- sanity checks for Sections 9.1, 9.2, 9.4 of note R4 (one-dimensional log-concavity certificates).

X1  delta h_q(n) = K_q^{n+1}(1,N) - K_q^n(1,N)  equals (q/2)^{n+1} W_n,  W_n = ((yI + Adj P_m)^n)(1,m), m = N-1, y = 2(1-q)/q
    (exact fractions; direct simulation of the reflecting chain vs the Jacobi-matrix recursion of cert_lc1d.ExactJ).
X2  same for the centre:  K_q^{n+1}(1,c) - K_q^n(1,c) = (q/2)^{n+1} ((yI + Adj P_M - e_M e_M^T)^n)(1,M),  N = 2M+1.
X3  spectral formulas  W_n = sum_k c_k lam_k^n  (Arb ball contains the exact value), both families.
X4  Casorati identity  W_n^2 - W_{n-1}W_{n+1} = - sum_{k<l} c_k c_l (lam_k lam_l)^{n-1} (lam_k-lam_l)^2  and
    sum_{k<l} p_k p_l (lam_k-lam_l)^2 = (sum p)(sum p lam^2) - (sum p lam)^2   (Arb vs exact).
X5  brute-force exact log-concavity scan (integers, long windows n <= 40 m^2) for small m: verdicts agree with
    "log-concave iff first inequality" and with cert_lc1d.
X6  explicit tail time of Proposition 9.6 (y = 1/2, E2E):  n_tail(N) = 1 + (N^2/8)(10 ln N - 8.43) >= certified n0,
    and the elementary bounds used in its proof (Arb), N = 10..400.
X7  Proposition 9.1 in action (exact): d = 2, N = 10 and d = 3, N = 9, q = 4/5: phi = Delta u log-concave on a window
    and the PMF differences change sign once on the window.
X8  float illustration for a dimension not covered by any d-dimensional certificate: d = 5, N = 9, q = 4/5.
X9  the four exceptional centre cases of Proposition 9.12 (phi_d not log-concave), exact integers.
X10 three real numbers a1, a2 > 0 > -b: the Casorati determinant of h_j equals (-a1 a2 b)^j h_j(1/a1,1/a2,-1/b).
X11 the block identity of Lemma 9.3(b) (exact).
"""
from __future__ import annotations
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository

import json
import math
import os
import sys
from fractions import Fraction as Fr

import numpy as np
from flint import arb, ctx

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cert_lc1d as C  # noqa: E402

DATA = _os.path.join(_R, 'data', 'msc_rigorous_unimodality')
RESULTS = []


def record(name, ok, detail=""):
    RESULTS.append(dict(check=name, ok=bool(ok), detail=str(detail)))
    print(("PASS " if ok else "FAIL ") + name + (" :: " + str(detail) if detail else ""), flush=True)
    with open(_os.path.join(_R, 'data', 'msc_rigorous_unimodality', 'verify_lc1d.json'), "w") as fh:
        json.dump(RESULTS, fh, indent=1)


def chain_1d(N, q, nmax, target):
    """h(n) = K_q^n(1, target) for the lazy reflecting walk on {1..N} (exact fractions)"""
    v = [Fr(0)] * N
    v[0] = Fr(1)
    out = [v[target - 1]]
    for _ in range(nmax):
        w = [Fr(0)] * N
        for x in range(N):
            if v[x] == 0:
                continue
            stay = 1 - q
            for dx in (-1, 1):
                z = x + dx
                if 0 <= z < N:
                    w[z] += v[x] * q / 2
                else:
                    stay += q / 2
            w[x] += v[x] * stay
        v = w
        out.append(v[target - 1])
    return out


def exact_W(geom, m, ymode, nmax):
    ex = C.ExactJ(geom, m, ymode)
    out = [ex.W()]
    for _ in range(nmax):
        ex.step()
        out.append(ex.W())
    return out, ex.S, ex.D


# ---------------------------------------------------------------- X1, X2
ok = True
ncase = 0
for N in range(3, 10):
    for q in (Fr(4, 5), Fr(1, 2), Fr(9, 10), Fr(1, 3)):
        m = N - 1
        y = 2 * (1 - q) / q
        nmax = 4 * N + 12
        h = chain_1d(N, q, nmax + 1, N)
        W, S, D = exact_W("E2E", m, (y.numerator, y.denominator), nmax)
        for n in range(nmax + 1):
            lhs = h[n + 1] - h[n]
            rhs = (q / 2) ** (n + 1) * Fr(W[n][0], S ** n)
            ok = ok and lhs == rhs
            ncase += 1
record("X1 delta h_q(n) = (q/2)^(n+1) W_n, E2E (exact, N=3..9, 4 activities)", ok, "%d identities" % ncase)

ok = True
ncase = 0
for N in (5, 7, 9, 11, 13):
    for q in (Fr(4, 5), Fr(1, 2), Fr(9, 10)):
        M = (N - 1) // 2
        y = 2 * (1 - q) / q
        nmax = 4 * N + 12
        h = chain_1d(N, q, nmax + 1, M + 1)
        W, S, D = exact_W("FOLD", M, (y.numerator, y.denominator), nmax)
        for n in range(nmax + 1):
            lhs = h[n + 1] - h[n]
            rhs = (q / 2) ** (n + 1) * Fr(W[n][0], S ** n)
            ok = ok and lhs == rhs
            ncase += 1
record("X2 K^{n+1}(1,c)-K^n(1,c) = (q/2)^(n+1) W~_n, FOLD (exact, N=5..13 odd, 3 activities)", ok, "%d identities" % ncase)

# ---------------------------------------------------------------- X3, X4
ok3 = True
ok4 = True
worst = 0.0
for geom in ("E2E", "FOLD"):
    for m in (3, 4, 7, 12, 25):
        for ymode in ((1, 2), (3, 7), "thr"):
            ctx.prec = 600
            yy, lam, c = C.spectrum(geom, m, ymode)
            nmax = 3 * m + 20
            W, S, D = exact_W(geom, m, ymode, nmax + 1)
            sq = arb(D).sqrt() if D else arb(0)
            Wa = [(arb(a) + arb(b) * sq) / arb(S) ** n for n, (a, b) in enumerate(W)]
            for n in range(1, nmax + 1):
                spec = arb(0)
                for ck, lk in zip(c, lam):
                    spec += ck * C.ipow(lk, n)
                ok3 = ok3 and spec.overlaps(Wa[n]) and float(abs(spec - Wa[n]).upper()) < 1e-100 * max(1.0, float(abs(Wa[n]).upper()))
                # Casorati
                lhs = Wa[n] * Wa[n] - Wa[n - 1] * Wa[n + 1]
                rhs = arb(0)
                for k in range(m):
                    for l in range(k + 1, m):
                        rhs -= c[k] * c[l] * C.ipow(lam[k] * lam[l], n - 1) * (lam[k] - lam[l]) ** 2
                ok4 = ok4 and rhs.overlaps(lhs)
                # variance identity
                p = [abs(ck) * C.ipow(abs(lk), n - 1) for ck, lk in zip(c, lam)]
                s0 = sum(p, arb(0)); s1 = sum((a * b for a, b in zip(p, lam)), arb(0)); s2 = sum((a * b * b for a, b in zip(p, lam)), arb(0))
                tot = arb(0)
                for k in range(m):
                    for l in range(k + 1, m):
                        tot += p[k] * p[l] * (lam[k] - lam[l]) ** 2
                ok4 = ok4 and tot.overlaps(s0 * s2 - s1 * s1)
record("X3 spectral formula W_n = sum c_k lam_k^n (Arb ball contains exact value; E2E/FOLD, m in {3,4,7,12,25}, y in {1/2,3/7,thr})", ok3)
record("X4 Casorati identity and variance identity for Tot_n (Arb vs exact)", ok4)

# ---------------------------------------------------------------- X5 brute force
ok = True
det = []
for geom in ("E2E", "FOLD"):
    for m in range(3, 15):
        for ymode in ((1, 2), (1, 1), (1, 3), "thr"):
            nmax = 40 * m * m
            ex = C.ExactJ(geom, m, ymode)
            for _ in range(m - 1):
                ex.step()
            w0 = ex.W(); ex.step(); w1 = ex.W(); ex.step(); w2 = ex.W()
            first_bad = None
            n_eq = 0
            for n in range(m, nmax):
                sq = C.zs_mul(w1, w1, ex.D); pr = C.zs_mul(w0, w2, ex.D)
                sg = C.zs_sign(sq[0] - pr[0], sq[1] - pr[1], ex.D)
                if sg < 0 and first_bad is None:
                    first_bad = n
                if sg == 0:
                    n_eq += 1
                ex.step(); w0, w1, w2 = w1, w2, ex.W()
            # prediction: log-concave iff first inequality (n = m) holds
            if ymode == "thr":
                pred_lc = True
            else:
                y = Fr(*ymode)
                if geom == "E2E":
                    pred_lc = (m * y * y >= 2)
                else:
                    pred_lc = (m * y * y - 2 * y - 2 >= 0)
            lc = first_bad is None
            if lc != pred_lc:
                ok = False
            if not lc and first_bad != m:
                det.append("%s m=%d y=%s: first violation at n=%d (not n=m)" % (geom, m, ymode, first_bad))
            if ymode == "thr" and m >= 3 and n_eq != 1 and not (geom == "FOLD" and m == 4) and not (geom == "E2E" and m == 2):
                det.append("%s m=%d thr: %d equalities" % (geom, m, n_eq))
record("X5 brute-force exact scan n <= 40 m^2 (m=3..14, y in {1/2,1,1/3,thr}, E2E and FOLD): log-concave iff first inequality", ok,
       "; ".join(det) if det else "violations, when present, start at n = m; one equality at threshold")

# ---------------------------------------------------------------- X6 explicit tail
ok = True
worst_ratio = 0.0
for N in range(10, 401):
    m = N - 1
    ctx.prec = 200
    yy, lam, c = C.spectrum("E2E", m, (1, 2))
    a1, a2 = abs(c[0]), abs(c[1])
    th = arb.pi() / N
    Lam3 = max(float(abs(lam[2]).upper()), float(abs(lam[m - 1]).upper()))
    ok = ok and bool(a1 >= arb(8) / N ** 3) and bool(a2 >= arb(32) / N ** 3)
    ok = ok and bool(lam[0] - lam[1] >= arb(12) / N ** 2)
    ok = ok and bool((lam[1] / Lam3).log() >= arb(8) / N ** 2)
    ntail = 1 + N * N / 8.0 * (10 * math.log(N) - 8.43)
    # find certified n0 as in cert_lc1d (coarse)
    n = 2 * m
    while not C.tail_ok(lam, c, n):
        n = int(n * 1.05) + 1
    ok = ok and (n <= ntail + 1 or N < 12)
    # the crude sufficient condition itself at n = ceil(ntail)
    nn = int(math.ceil(ntail))
    lhs = (lam[1] / Lam3) ** (nn - 1)
    ok = ok and bool(lhs * a1 * a2 * (lam[0] - lam[1]) ** 2 > 8)
    worst_ratio = max(worst_ratio, n / ntail)
record("X6 Proposition 9.6: elementary bounds and explicit tail time n_tail(N) (y=1/2, E2E, N=10..400)", ok,
       "max certified-n0 / n_tail = %.3f" % worst_ratio)


# ---------------------------------------------------------------- X7 exact d-dimensional check
def box_rows(N, d, q):
    import itertools
    sites = list(itertools.product(range(N), repeat=d))
    idx = {s: i for i, s in enumerate(sites)}
    return sites, idx


def exact_phi_f(N, d, qn, qd, tmax):
    """integer arithmetic: u(t)*Dn^t, f(t)*Dn^t with Dn = 2 d qd, corner (0..0) -> corner (N-1..N-1)"""
    shape = (N,) * d
    Dn = 2 * d * qd
    nblk = np.zeros(shape, dtype=object)
    for ax in range(d):
        i0 = [slice(None)] * d; i0[ax] = 0; nblk[tuple(i0)] += 1
        i1 = [slice(None)] * d; i1[ax] = N - 1; nblk[tuple(i1)] += 1
    hold = (qd - qn) * 2 * d + qn * nblk

    def step(v):
        w = hold * v
        for ax in range(d):
            sl_a = [slice(None)] * d; sl_b = [slice(None)] * d
            sl_a[ax] = slice(1, None); sl_b[ax] = slice(None, -1)
            w[tuple(sl_a)] += qn * v[tuple(sl_b)]
            w[tuple(sl_b)] += qn * v[tuple(sl_a)]
        return w

    a = (N - 1,) * d
    v = np.zeros(shape, dtype=object); v[(0,) * d] = 1       # free
    k = np.zeros(shape, dtype=object); k[(0,) * d] = 1       # killed
    U = [0]; F = [0]
    for t in range(1, tmax + 1):
        v = step(v)
        k = step(k)
        F.append(int(k[a])); k[a] = 0
        U.append(int(v[a]))
    return U, F, Dn


ok = True
det = []
for (N, d, tmax) in ((10, 2, 700), (9, 3, 500)):
    U, F, Dn = exact_phi_f(N, d, 4, 5, tmax)
    # phi(t) = u(t) - u(t-1) = (U_t - Dn U_{t-1}) / Dn^t ; work with Phi_t = U_t - Dn U_{t-1}
    Phi = [0] + [U[t] - Dn * U[t - 1] for t in range(1, tmax + 1)]
    lc = all(Phi[t] >= 0 for t in range(tmax + 1)) and all(Phi[t] ** 2 >= Phi[t - 1] * Phi[t + 1] for t in range(1, tmax))
    # f differences: sign of F_{t+1} - Dn F_t
    signs = [(F[t + 1] - Dn * F[t] > 0) - (F[t + 1] - Dn * F[t] < 0) for t in range(tmax)]
    nz = [s for s in signs if s != 0]
    changes = sum(1 for i in range(len(nz) - 1) if nz[i] != nz[i + 1])
    mode = max(range(tmax + 1), key=lambda t: Fr(F[t], Dn ** t))
    ok = ok and lc and changes == 1
    det.append("d=%d N=%d: phi log-concave on t<=%d: %s; sign changes of Delta f: %d; mode %d" % (d, N, tmax, lc, changes, mode))
record("X7 Proposition 9.1 in action (exact integers): q=4/5, corner pair", ok, " | ".join(det))

# ---------------------------------------------------------------- X8 float illustration d=5
N, d, q = 9, 5, 0.8
shape = (N,) * d
nblk = np.zeros(shape)
for ax in range(d):
    i0 = [slice(None)] * d; i0[ax] = 0; nblk[tuple(i0)] += 1
    i1 = [slice(None)] * d; i1[ax] = N - 1; nblk[tuple(i1)] += 1
hold = (1 - q) + q * nblk / (2 * d)
k = np.zeros(shape); k[(0,) * d] = 1.0
a = (N - 1,) * d
f = [0.0]
for t in range(1, 1500):
    w = hold * k
    for ax in range(d):
        sa = [slice(None)] * d; sb = [slice(None)] * d
        sa[ax] = slice(1, None); sb[ax] = slice(None, -1)
        w[tuple(sa)] += (q / (2 * d)) * k[tuple(sb)]
        w[tuple(sb)] += (q / (2 * d)) * k[tuple(sa)]
    f.append(w[a]); w[a] = 0.0
    k = w
f = np.array(f)
df = np.diff(f)
tol = 1e-13 * f.max()
sg = np.sign(np.where(np.abs(df) < tol, 0, df))
nz = sg[sg != 0]
changes = int(np.sum(nz[1:] != nz[:-1]))
record("X8 float illustration: d=5, N=9, q=0.8 corner pair, t<1500: Delta f changes sign once", changes == 1,
       "mode at t=%d, f_max=%.3e" % (int(np.argmax(f)), f.max()))

# ---------------------------------------------------------------- X9 exceptional centre cases (exact)
import cert_blocks as CB  # noqa: E402
ok = True
det = []
for (N, d) in ((3, 2), (5, 2), (7, 2), (3, 3)):
    D = d * (N - 1) // 2
    Phi, Dn = CB.exact_a("CEN", N, d, D + 6)       # phi(t) = Phi[t] / Dn^t
    ok = ok and all(Phi[t] == 0 for t in range(D)) and Phi[D] > 0
    # first inequality phi(D+1)^2 >= phi(D) phi(D+2)  <=>  Phi[D+1]^2 >= Phi[D] Phi[D+2]
    bad = Phi[D + 1] ** 2 < Phi[D] * Phi[D + 2]
    ok = ok and bad
    det.append("N=%d d=%d: Phi(D..D+2)=%s, first inequality %s" % (N, d, Phi[D:D + 3], "FAILS" if bad else "holds"))
record("X9 Proposition 9.12: phi_d is NOT log-concave at q=4/5 for the centre geometry, (N,d) in {(3,2),(5,2),(7,2),(3,3)} (exact)", ok, " | ".join(det))

# ---------------------------------------------------------------- X10 three numbers
import random
random.seed(7)
ok = True
for _ in range(200):
    a1 = Fr(random.randint(1, 40), random.randint(1, 9)); a2 = Fr(random.randint(1, 40), random.randint(1, 9))
    b = Fr(random.randint(1, 40), random.randint(1, 9))
    xs = [a1, a2, -b]
    h = [Fr(1)] + [Fr(0)] * 14
    for x in xs:
        for j in range(1, 15):
            h[j] += x * h[j - 1]
    hi = [Fr(1)] + [Fr(0)] * 14
    for x in (1 / a1, 1 / a2, -1 / b):
        for j in range(1, 15):
            hi[j] += x * hi[j - 1]
    e2 = a1 * a2 - b * (a1 + a2)
    for j in range(1, 14):
        lhs = h[j] ** 2 - h[j - 1] * h[j + 1]
        ok = ok and lhs == (-a1 * a2 * b) ** j * hi[j]
        if e2 >= 0:
            ok = ok and lhs >= 0
    if e2 < 0:
        ok = ok and (h[1] ** 2 - h[0] * h[2] < 0)
record("X10 three numbers a1,a2>0>-b: Casorati = (-a1 a2 b)^j h_j(1/a1,1/a2,-1/b); >= 0 for all j iff e_2 >= 0 (exact, 200 random triples, j<=13)", ok)

# ---------------------------------------------------------------- X11 block identity of Lemma 9.3(b)
from math import comb


def exact_u_phi(geom, N, d, tmax):
    Phi, Dn = CB.exact_a(geom, N, d, tmax)         # phi(t) = Phi[t]/Dn^t, Dn = 10 d
    U = [0] * (tmax + 1)                            # u(t) = Uu[t]/Dn^t ; u(t) = u(t-1) + phi(t)
    for t in range(1, tmax + 1):
        U[t] = Dn * U[t - 1] + Phi[t]
    return Phi, U, Dn


ok = True
ncase = 0
for geom, N in (("CC", 3), ("CC", 4), ("CEN", 5)):
    tmax = 26
    data = {d: exact_u_phi(geom, N, d, tmax) for d in (1, 2, 3)}
    for (d1, d2) in ((1, 1), (2, 1), (1, 2)):
        d = d1 + d2
        Phi_d, U_d, Dn_d = data[d]
        Phi_1, U_1, Dn_1 = data[d1]
        Phi_2, U_2, Dn_2 = data[d2]
        for t in range(1, tmax):
            # d^(t-1) phi_d(t) = sum_k C(t-1,k) d1^k phi_{d1}(k+1) d2^(t-1-k) u_{d2}(t-1-k)
            lhs = Fr(d ** (t - 1) * Phi_d[t], Dn_d ** t)
            rhs = Fr(0)
            for k in range(t):
                rhs += comb(t - 1, k) * Fr(d1 ** k * Phi_1[k + 1], Dn_1 ** (k + 1)) * Fr(d2 ** (t - 1 - k) * U_2[t - 1 - k], Dn_2 ** (t - 1 - k))
            ok = ok and lhs == rhs
            ncase += 1
record("X11 Lemma 9.3(b): d^(t-1) phi_d(t) = sum_k C(t-1,k) d1^k phi_d1(k+1) d2^(t-1-k) u_d2(t-1-k) (exact, 3 geometries, d=2,3)", ok, "%d identities" % ncase)

print("ALL PASS" if all(r["ok"] for r in RESULTS) else "SOME FAILED")
