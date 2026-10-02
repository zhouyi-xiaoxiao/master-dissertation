"""
Numerical check of EVERY intermediate identity used in the proofs of Supplementary Sections S3 and S4.

Each check prints a name, the quantity compared and the maximal absolute (or
relative) discrepancy, and records it in results/identity_checks.json.
Deterministic (no randomness except a fixed-seed choice of sigma values).

Run:  python check_identities.py
"""
from __future__ import annotations
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository

import json
import os
from fractions import Fraction

import numpy as np
import mpmath as mp

import mfpt2d as M

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = _os.path.join(_R, 'data', 'msc_proof', 'identity_checks.json')
mp.mp.dps = 40
results = []


def record(name, statement, max_err, scope, tol, kind="abs"):
    ok = bool(max_err < tol)
    results.append(dict(check=name, statement=statement, max_error=float(max_err),
                        error_kind=kind, scope=scope, tolerance=tol, passed=ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name:34s} max {kind} err = {float(max_err):.3e}   ({scope})")


# ---------------------------------------------------------------- helpers
def T1d(N):
    """1D reflecting (cancelled-move) step operator: 1/2 to each neighbour, cancelled at walls."""
    T = np.zeros((N, N))
    for n in range(N):
        for d in (-1, 1):
            m = n + d
            if 0 <= m < N:
                T[n, m] += 0.5
            else:
                T[n, n] += 0.5
    return T


def psi(N):
    """Orthonormal DCT-II eigenvectors psi[k, n-1] = sqrt(alpha_k/N) cos(pi k (2n-1)/(2N))."""
    k = np.arange(N)[:, None]
    n = np.arange(1, N + 1)[None, :]
    al = np.where(k == 0, 1.0, 2.0)
    return np.sqrt(al / N) * np.cos(np.pi * k * (2 * n - 1) / (2 * N))


# ---------------------------------------------------------------- Lemma 2 (1D spectrum)
err_eig = err_orth = 0.0
for N in range(2, 41):
    T = T1d(N)
    V = psi(N)
    lam = np.cos(np.pi * np.arange(N) / N)
    err_eig = max(err_eig, np.abs(T @ V.T - V.T * lam[None, :]).max())
    err_orth = max(err_orth, np.abs(V @ V.T - np.eye(N)).max())
record("L2a 1D eigenpairs", "T v_k = cos(pi k/N) v_k, v_k(n)=cos(pi k(2n-1)/2N)", err_eig, "N=2..40", 1e-13)
record("L2b 1D orthonormality", "sum_n psi_k(n) psi_l(n) = delta_kl", err_orth, "N=2..40", 1e-13)

# ---------------------------------------------------------------- Eq. (2.2) tensor structure
err_P = err_sym = err_row = 0.0
for N in (2, 3, 4, 7, 12):
    for q in (1.0, 0.8, 0.3):
        P = M.transition_matrix(N, q).toarray()
        T = T1d(N)
        I = np.eye(N)
        P2 = (1 - q) * np.eye(N * N) + q / 2 * (np.kron(T, I) + np.kron(I, T))
        err_P = max(err_P, np.abs(P - P2).max())
        err_sym = max(err_sym, np.abs(P - P.T).max())
        err_row = max(err_row, np.abs(P.sum(1) - 1).max())
record("E2.2 P = (1-q)I + q/2 (TxI + IxT)", "rule-built P equals Kronecker form", err_P, "N in {2,3,4,7,12}, q in {1,.8,.3}", 1e-15)
record("E2.2b P symmetric", "P = P^T", err_sym, "same", 1e-15)
record("E2.2c P stochastic", "row sums = 1", err_row, "same", 1e-14)

# ---------------------------------------------------------------- Lemma 1 (hitting times via pseudo-inverse)
err_pinv = err_hit = err_poisson = 0.0
for N in (2, 3, 5, 8):
    for q in (1.0, 0.8, 0.3):
        P = M.transition_matrix(N, q).toarray()
        L = np.eye(N * N) - P
        Lp = np.linalg.pinv(L)
        # spectral pseudo-inverse
        V = psi(N)
        lam = np.cos(np.pi * np.arange(N) / N)
        Lp_spec = np.zeros_like(L)
        for k in range(N):
            for j in range(N):
                if k == 0 and j == 0:
                    continue
                v = np.kron(V[k], V[j])
                Lp_spec += np.outer(v, v) / (q / 2 * (2 - lam[k] - lam[j]))
        err_pinv = max(err_pinv, np.abs(Lp - Lp_spec).max() / np.abs(Lp).max())
        for a in (N * N - 1, 0, (N * N) // 2):
            h = N * N * (Lp[a, a] - Lp[:, a])
            Lh = L @ h
            mask = np.arange(N * N) != a
            err_poisson = max(err_poisson, np.abs(Lh[mask] - 1).max())
            # direct hitting-time vector
            keep = np.where(mask)[0]
            mvec = np.linalg.solve(np.eye(len(keep)) - P[np.ix_(keep, keep)], np.ones(len(keep)))
            err_hit = max(err_hit, np.abs(h[keep] - mvec).max() / mvec.max())
record("L1a spectral pseudo-inverse", "L^+ = sum' psi psi^T/(1-lambda) equals numpy pinv(I-P)", err_pinv, "N in {2,3,5,8}, 3 q", 1e-11, "rel")
record("L1b Poisson equation", "(L h)(x) = 1 for x != a, h = N^2(L+_aa - L+_xa)", err_poisson, "3 targets each", 1e-10)
record("L1c hitting vector", "h(x) = E_x[T_a] from (I-Q)m=1, all x", err_hit, "3 targets each", 1e-11, "rel")

# ---------------------------------------------------------------- Lemma 3 (ring Green's function)
rng = np.random.default_rng(20261001)
err_ring = err_half = err_E1 = mp.mpf(0)
for N in (2, 3, 4, 5, 8, 13, 24):
    sigmas = [mp.mpf(str(1 + s)) for s in (1e-3, 0.07, 0.5, 1.0, 2.0)] + [mp.mpf(str(1 + rng.random() * 3))]
    for sg in sigmas:
        phi = mp.acosh(sg)
        for m in range(0, 2 * N + 1):
            full = mp.fsum(mp.cos(m * mp.pi * j / N) / (sg - mp.cos(mp.pi * j / N)) for j in range(2 * N))
            rhs = 2 * N * mp.cosh((N - m) * phi) / (mp.sinh(phi) * mp.sinh(N * phi))
            err_ring = max(err_ring, abs(full - rhs) / max(1, abs(rhs)))
            half = mp.fsum((1 if j == 0 else 2) * mp.cos(m * mp.pi * j / N) / (sg - mp.cos(mp.pi * j / N))
                           for j in range(N))
            rhs_h = rhs - (-1) ** m / (sg + 1)
            err_half = max(err_half, abs(half - rhs_h) / max(1, abs(rhs_h)))
            # closed form = Giuggioli (2020) Eq. (E1), sum over k=1..N-1
            s1 = mp.fsum(mp.cos(m * mp.pi * j / N) / (sg - mp.cos(mp.pi * j / N)) for j in range(1, N))
            e1 = N * mp.cosh(abs(N - m) * phi) / (mp.sinh(phi) * mp.sinh(N * phi)) \
                + mp.mpf(1) / 2 * (1 / (1 - sg) + (-1) ** (m + 1) / (1 + sg))
            err_E1 = max(err_E1, abs(s1 - e1) / max(1, abs(e1)))
record("L3a ring Green's function", "sum_{j<2N} cos(m x_j)/(s-cos x_j) = 2N cosh((N-m)phi)/(sinh phi sinh N phi)",
       err_ring, "N in {2,3,4,5,8,13,24}, 6 sigma, 0<=m<=2N", 1e-30, "abs/max(1,|rhs|)")
record("L3b half-range form", "sum_{j<N} alpha_j cos(m x_j)/(s-cos x_j) = ... - (-1)^m/(s+1)", err_half, "same", 1e-30, "abs/max(1,|rhs|)")
record("L3c closed form = PRX (E1)", "sum_{k=1}^{N-1} cos(m x_k)/(s-cos x_k) closed form", err_E1, "same", 1e-30, "rel")

# ---------------------------------------------------------------- Lemma 5 (F_m) = PRX (E2)
err_F = mp.mpf(0)
err_F_alt = mp.mpf(0)
for N in range(2, 31):
    for m in range(0, 2 * N + 1):
        s = mp.fsum(mp.cos(m * mp.pi * j / N) / (1 - mp.cos(mp.pi * j / N)) for j in range(1, N))
        F = M.F_m(N, m)
        err_F = max(err_F, abs(s - mp.mpf(F.numerator) / F.denominator))
        Fth = (Fraction(N * N) + Fraction(1, 2)) / 3 + m * (Fraction(m, 2) - N) + Fraction((-1) ** (m + 1) - 1, 4)
        err_F_alt = max(err_F_alt, abs(mp.mpf((F - Fth).numerator) / (F - Fth).denominator))
record("L5 F_m closed form", "sum_{j=1}^{N-1} cos(m x_j)/(1-cos x_j) = (N-m)^2/2 - N^2/6 - 1/12 - (-1)^m/4",
       err_F, "N=2..30, 0<=m<=2N", 1e-30)
record("L5b equivalent form", "PRX (E2) form equals the above exactly (rational)", err_F_alt, "same", 1e-40)

# ---------------------------------------------------------------- Lemma 4 (1D MFPT and k=0 sector)
err_1d = err_sector = err_cot = mp.mpf(0)
for N in range(2, 61):
    # m(n) = N(N-1) - n(n-1) solves the 1D hitting equations for the q=1 cancelled-move chain
    T = T1d(N)
    mvec = np.array([N * (N - 1) - n * (n - 1) for n in range(1, N + 1)], dtype=float)
    res = (mvec - 1 - T @ mvec)[:-1]
    err_1d = max(err_1d, mp.mpf(float(np.abs(res).max())))
    sec = mp.fsum(2 * mp.cos(mp.pi * j / (2 * N)) ** 2 * (1 - (-1) ** j) / (1 - mp.cos(mp.pi * j / N))
                  for j in range(1, N))
    err_sector = max(err_sector, abs(sec - N * (N - 1)))
    cot = mp.fsum(mp.cot(mp.pi * j / (2 * N)) ** 2 for j in range(1, N, 2))
    err_cot = max(err_cot, abs(cot - mp.mpf(N * (N - 1)) / 2))
record("L4a 1D hitting time", "m(n)=N(N-1)-n(n-1) solves m = 1 + T m, m(N)=0", err_1d, "N=2..60", 1e-9)
record("L4b k=0 sector", "sum_{j>=1} alpha_j c_j^2 (1-(-1)^j)/(1-cos x_j) = N(N-1)", err_sector, "N=2..60", 1e-30)
record("L4c odd cot^2 sum", "sum_{j odd<N} cot^2(pi j/2N) = N(N-1)/2", err_cot, "N=2..60", 1e-30)

# ---------------------------------------------------------------- Proposition (G_k) and hyperbolic rewriting
err_G = err_B = err_half_angle = err_c2 = mp.mpf(0)
for N in (2, 3, 4, 5, 6, 9, 16, 31, 64):
    err_c2 = max(err_c2, abs(mp.fsum(mp.cos(mp.pi * k / (2 * N)) ** 2 for k in range(1, N)) - mp.mpf(N - 1) / 2))
    for k in range(1, N):
        sg = 2 - mp.cos(mp.pi * k / N)
        phi = mp.acosh(sg)
        G = mp.fsum((1 if j == 0 else 2) * mp.cos(mp.pi * j / (2 * N)) ** 2 * (1 - (-1) ** (k + j))
                    / (sg - mp.cos(mp.pi * j / N)) for j in range(N))
        Gc = N * (1 + sg) * (mp.cosh(N * phi) - (-1) ** k) / (mp.sinh(phi) * mp.sinh(N * phi)) - N
        err_G = max(err_G, abs(G - Gc) / abs(Gc))
        Bk = mp.cosh(N * phi) + mp.cosh((N - 1) * phi) - (-1) ** k * (1 + mp.cosh(phi))
        lhs = Bk / (mp.sinh(phi) * mp.sinh(N * phi))
        t = mp.tanh(N * phi / 2)
        g = 1 / t if k % 2 else t
        rhs = mp.coth(phi / 2) * g - 1
        err_B = max(err_B, abs(lhs - rhs))
        err_B = max(err_B, abs(N * lhs - Gc) / abs(Gc))
        sk = mp.sin(mp.pi * k / (2 * N))
        err_half_angle = max(err_half_angle, abs(mp.sinh(phi / 2) - sk),
                             abs(mp.coth(phi / 2) - mp.sqrt(1 + sk ** 2) / sk))
record("P1 inner sum G_k", "G_k = N(1+s_k)(cosh N phi-(-1)^k)/(sinh phi sinh N phi) - N", err_G, "N in {2..64 sel.}, all k", 1e-30, "rel")
record("P2 hyperbolic rewriting", "B_k/(sinh phi sinh N phi) = coth(phi/2) g_k - 1 = G_k/N", err_B, "same", 1e-28)
record("P3 half-angle", "sinh(phi_k/2) = sin(pi k/2N), coth(phi_k/2)=sqrt(1+s^2)/s", err_half_angle, "same", 1e-30)
record("P4 sum c_k^2", "sum_{k=1}^{N-1} cos^2(pi k/2N) = (N-1)/2", err_c2, "same", 1e-30)

# ---------------------------------------------------------------- Step 5 of Theorem 2 and (6.3)
err_odd = err_H = err_63 = mp.mpf(0)
for N in (2, 3, 4, 5, 6, 7, 10, 15, 32, 33):
    err_odd = max(err_odd, abs(mp.fsum(mp.cos(mp.pi * k / (2 * N)) ** 2 for k in range(1, N, 2)) - mp.mpf(N) / 4))
    for k in range(1, N, 2):
        sg = 2 - mp.cos(mp.pi * k / N); phi = mp.acosh(sg)
        H = mp.fsum((1 if j == 0 else 2) * mp.cos(mp.pi * j / (2 * N)) ** 2 / (sg - mp.cos(mp.pi * j / N))
                    for j in range(0, N, 2))
        Hc = mp.mpf(N) / 2 * (mp.coth(phi / 2) * mp.coth(N * phi / 2) - 1)
        err_H = max(err_H, abs(H - Hc) / abs(Hc))
    for u in range(1, N + 1):
        for v in range(1, N + 1):
            lhs = mp.fsum(2 * (mp.cos(mp.pi * j * (2 * v - 1) / (2 * N)) ** 2
                               - mp.cos(mp.pi * j * (2 * u - 1) / (2 * N)) * mp.cos(mp.pi * j * (2 * v - 1) / (2 * N)))
                          / (1 - mp.cos(mp.pi * j / N)) for j in range(1, N))
            err_63 = max(err_63, abs(lhs - M.tau_1d(N, u, v)))
record("P5 odd sum of c_k^2", "sum_{k odd<N} cos^2(pi k/2N) = N/4 (both parities of N)", err_odd, "N in {2,...,33}", 1e-30)
record("P6 even-index inner sum H_k", "H_k = (N/2)[coth(phi/2) coth(N phi/2) - 1], k odd", err_H, "same", 1e-30, "rel")
record("L4d 1D spectral hitting time (6.3)", "sum_j alpha_j[cos^2 th_j(v)-cos th_j(u)cos th_j(v)]/(1-cos x_j) = tau_1(u,v)", err_63, "same, all u,v", 1e-28)

# ---------------------------------------------------------------- Theorems 1-2 end to end (mp, 40 digits)
err_T = mp.mpf(0)
for N in (2, 3, 4, 5, 6, 7, 10, 17, 32, 45):
    d = M.double_sum_mp(N, 1, 40)
    t = M.single_sum_form_a_mp(N, 1, 40)
    c = M.clean_single_mp(N, 1, 40)
    err_T = max(err_T, abs(d - t) / d, abs(d - c) / d)
record("T2 double = form (a) = clean", "three closed forms agree", err_T, "N in {2,...,45}, q=1", 1e-32, "rel")

# ---------------------------------------------------------------- Corollary: resistance / Essam-Wu single sum
err_R = err_EW = 0.0
par_even = par_odd = mp.mpf(0)
for N in range(2, 25):
    # graph Laplacian of the N x N grid, unit resistors
    P1 = M.transition_matrix(N, 1.0).toarray()
    Lg = 4 * (np.eye(N * N) - P1)           # = D - A (self-loops cancel)
    Lp = np.linalg.pinv(Lg)
    s, a = 0, N * N - 1
    R = Lp[s, s] + Lp[a, a] - 2 * Lp[s, a]
    Tq = float(M.clean_single_mp(N, 1, 30))
    err_R = max(err_R, abs(Tq - 2 * N * N * R) / Tq)
    terms = []
    for k in range(1, N):
        y = mp.pi * k / (2 * N)
        sk = mp.sin(y)
        t = mp.tanh(N * mp.asinh(sk))
        terms.append(mp.cos(y) ** 2 * mp.sqrt(1 + sk ** 2) / sk * (1 / t if k % 2 else t))
    odd = mp.fsum(terms[0::2])
    even = mp.fsum(terms[1::2])
    dpar = abs(2 * odd / N - 1 - 2 * even / N)
    if N % 2 == 0:
        par_even = max(par_even, dpar)
        # Essam & Wu (2009), Eqs (6),(15),(16) at M=N even, r=s=1
        S = mp.fsum(mp.cos(qq * mp.pi / N) ** 2 * mp.sqrt(1 + mp.sin(qq * mp.pi / N) ** 2) / mp.sin(qq * mp.pi / N)
                    * mp.tanh(N * mp.asinh(mp.sin(qq * mp.pi / N))) for qq in range(1, N // 2 + 1))
        R_EW = 1 + 4 * S / N
        err_EW = max(err_EW, abs(float(R_EW) - R) / R)
    else:
        par_odd = max(par_odd, dpar)
record("C1 commute-time identity", "q T = 2 N^2 R_eff(corner,corner), R_eff from pinv(D-A)", err_R, "N=2..24", 1e-10, "rel")
record("C2 Essam-Wu single sum", "R = 1 + (4/N) sum_{q<=N/2} D_q tanh(.) equals R_eff (N even)", err_EW, "N=2..24 even", 1e-10, "rel")
record("C3 parity identity (N even)", "(2/N) S_odd = 1 + (2/N) S_even", par_even, "N=2..24 even", 1e-28)
record("C3b parity identity (N odd)", "(2/N) S_odd = 1 + (2/N) S_even also for odd N (Theorem 2c)", par_odd, "N=3..23 odd", 1e-28)

# ---------------------------------------------------------------- generating-function route (Remark)
err_gf = 0.0
gf_inverted = None
for N in (4, 7):
    q = 0.8
    V = psi(N)
    lam1 = np.cos(np.pi * np.arange(N) / N)
    lam = 1 - q + q / 2 * (lam1[:, None] + lam1[None, :])
    s_idx, a_idx = 0, N - 1

    def Pz(x, y, z):
        return float(np.sum(V[:, x[0]][:, None] * V[:, x[1]][None, :] * V[:, y[0]][:, None] * V[:, y[1]][None, :]
                            / (1 - z * lam)))
    Tref = M.direct_mfpt(N, q)
    for eps in (1e-5, 1e-6):
        z = 1 - eps
        Fz = Pz((s_idx, s_idx), (a_idx, a_idx), z) / Pz((a_idx, a_idx), (a_idx, a_idx), z)
        est = (1 - Fz) / (1 - z)
        err_gf = max(err_gf, abs(est - Tref) / Tref / eps)   # error should be O(eps)
        gf_inverted = (1 - z) / (1 - Fz) * Tref          # the reciprocal ratio gives 1/T
record("R1 renewal / GF route", "(1-F(z))/(1-z) -> T as z->1 (error = O(1-z), ratio reported)", err_gf,
       "N in {4,7}, q=0.8", 1e4, "rel/eps")
results.append(dict(check="R1b reciprocal ratio", statement="(1-z)/(1-F(z)) tends to 1/T, not T; value*T reported",
                    max_error=float(gf_inverted), error_kind="value", scope="N=7", tolerance=None, passed=None))
print(f"[INFO] the reciprocal ratio gives 1/T: [(1-z)/(1-F)]*T = {gf_inverted:.6f} (should be 1)")

# ---------------------------------------------------------------- constants used in Theorem 3
mp.mp.dps = 50
p = -mp.exp(-mp.pi)
lam1 = mp.nsum(lambda k: k * p ** k / (1 + p ** k), [1, mp.inf])
lam2 = mp.nsum(lambda k: k ** 2 * p ** k / (1 + p ** k) ** 2, [1, mp.inf])
g14 = mp.gamma(mp.mpf(1) / 4)
E4i = 3 * g14 ** 8 / (2 * mp.pi) ** 6
record("A1 Lambert series 1", "sum_k k p^k/(1+p^k) = -1/24, p=-e^{-pi}", abs(lam1 + mp.mpf(1) / 24), "50 digits", 1e-45)
record("A2 Lambert series 2", "sum_k k^2 p^k/(1+p^k)^2 = -E4(i)/36", abs(lam2 + E4i / 36), "50 digits", 1e-45)
# E4(i) itself from its q-series
E4q = 1 + 240 * mp.nsum(lambda n: n ** 3 * mp.exp(-2 * mp.pi * n) / (1 - mp.exp(-2 * mp.pi * n)), [1, mp.inf])
record("A3 E4(i)", "1+240 sum n^3 q^n/(1-q^n) at q=e^{-2pi} = 3 Gamma(1/4)^8/(2pi)^6", abs(E4q - E4i), "50 digits", 1e-45)
E2q = 1 - 24 * mp.nsum(lambda n: n * mp.exp(-2 * mp.pi * n) / (1 - mp.exp(-2 * mp.pi * n)), [1, mp.inf])
record("A4 E2(i)", "E2(i) = 3/pi", abs(E2q - 3 / mp.pi), "50 digits", 1e-45)
E2h = 1 - 24 * mp.nsum(lambda n: n * p ** n / (1 - p ** n), [1, mp.inf])
record("A5 E2((1+i)/2)", "E2((1+i)/2) = 6/pi", abs(E2h - 6 / mp.pi), "50 digits", 1e-45)
Eser = mp.nsum(lambda k: (-2 * p ** k / (1 + p ** k)) / k, [1, mp.inf])
Eclosed = mp.mpf(5) / 2 * mp.log(2) + mp.mpf(3) / 2 * mp.log(mp.pi) - 2 * mp.log(g14) - mp.pi / 4
record("A6 log-series E", "sum_k (g_k^inf - 1)/k = (5/2)ln2 + (3/2)ln pi - 2 ln Gamma(1/4) - pi/4", abs(Eser - Eclosed), "50 digits", 1e-45)
fr = lambda y: mp.cos(y) ** 2 * mp.sqrt(1 + mp.sin(y) ** 2) / mp.sin(y) - 1 / y
Ir = mp.quad(fr, mp.linspace(0, mp.pi / 2, 9))
Ir_closed = mp.mpf(3) / 2 * mp.log(2) - mp.log(mp.pi) - mp.mpf(1) / 2
record("A7 integral I_r", "int_0^{pi/2} (f(y) - 1/y) dy = (3/2)ln2 - ln pi - 1/2", abs(Ir - Ir_closed), "quad, 50 digits", 1e-40)
d0 = mp.diff(fr, mp.mpf("1e-12"), h=mp.mpf("1e-14")) if False else mp.limit(lambda y: fr(y) / y, 0)
record("A8 r'(0)", "r'(0) = -1/3", abs(d0 + mp.mpf(1) / 3), "limit", 1e-12)
d1 = mp.diff(fr, mp.pi / 2)
record("A9 r'(pi/2)", "r'(pi/2) = 4/pi^2", abs(d1 - 4 / mp.pi ** 2), "numerical derivative", 1e-25)
C2, C0 = M.asymptotic_constants(50)
C2_series = 8 / mp.pi * (mp.euler + Ir + Eser)
C0_series = mp.pi / 18 + mp.mpf(2) / 3 * (mp.pi ** 2 * (-2 * lam2) - mp.pi * (-2 * lam1))
record("A10 C_2 closed form", "C_2 = (8/pi)(gamma + I_r + E) equals Gamma(1/4) form", abs(C2 - C2_series), "50 digits", 1e-40)
record("A11 C_0 closed form", "C_0 = pi/18 + (2/3)[pi^2 S2 - pi S1] equals Gamma(1/4)^8/(576 pi^4)", abs(C0 - C0_series), "50 digits", 1e-40)
# theta-function form of Izmailian & Huang (2010) at xi = rho = 1
th2 = mp.jtheta(2, 0, mp.exp(-mp.pi)); th3 = mp.jtheta(3, 0, mp.exp(-mp.pi)); th4 = mp.jtheta(4, 0, mp.exp(-mp.pi))
c0_IH = 2 / mp.pi * (2 * mp.log(8 / mp.pi) + 2 * mp.euler - 1 - mp.log(2) - mp.pi / 2 - 2 * mp.log(th2 * th4))
c2_IH = mp.pi / 72 * (2 * (th4 ** 4 - th2 ** 4) + 2 * mp.pi * th3 ** 4 * th4 ** 4)
record("A12 Izmailian-Huang c0", "C_2/2 equals IH Eq.(42) at xi=rho=1 (theta functions, nome e^{-pi})", abs(C2 / 2 - c0_IH), "50 digits", 1e-40)
record("A13 Izmailian-Huang c2", "C_0/2 equals IH Eq.(43) at xi=rho=1", abs(C0 / 2 - c2_IH), "50 digits", 1e-40)
print("C_2 =", mp.nstr(C2, 40))
print("C_0 =", mp.nstr(C0, 40))
print("C_2/2 (resistance c0) =", mp.nstr(C2 / 2, 25), "  Essam-Wu: 0.077318893909458")
print("C_0/2 (resistance c2) =", mp.nstr(C0 / 2, 25), "  Essam-Wu: 0.266070441638478")

n_fail = sum(1 for r in results if r["passed"] is False)
summary = dict(n_checks=len(results), n_failed=n_fail,
               C2=mp.nstr(C2, 45), C0=mp.nstr(C0, 45), checks=results)
with open(OUT, "w") as f:
    json.dump(summary, f, indent=1)
print(f"\n{len(results)} checks, {n_fail} failed. Written {_os.path.join(_R, 'data', 'msc_proof', 'identity_checks.json')}")
