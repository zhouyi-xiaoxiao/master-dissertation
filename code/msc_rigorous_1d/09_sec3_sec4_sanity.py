"""Numerical sanity checks for statements of Sections 3 and 4 of note R1 (not proofs):

 (1) Prop. 3.9: 2(h_2 - h_1) = n(n-1) - n(2n-1) q + (n^2 + n/2 - 1) q^2;  h_1 = n - (n-1/2) q   (exact rationals)
 (2) Lemma 3.8 (slowing down): f_q(t) = sum_k f_{q0}(k) C(t-1,k-1) p^k (1-p)^(t-k), p = q/q0     (exact rationals)
 (3) Theorem 3.10 (q = 1 parity formula) against exact stepping; vanishing pattern; class unimodality
 (4) Theorem 3.6: log-concavity for q <= 2/3 (exact rationals, several N); TP2 kernel claim M_t(x,y) >= 0
 (5) Section 4: image (theta) form of G;  LT of G = 1/cosh sqrt(2s);  G = density with mean 1;
     Lemma 5.1/5.2 elementary bounds on a grid.
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import os, sys, json
from fractions import Fraction as Fr
from math import comb
import mpmath as mp
sys.path.insert(0, os.path.dirname(__file__))
from lib1d import pmf_exact, Q_matrix
mp.mp.dps = 40
res = {}

# (1)
ok = True
for N in range(4, 12):
    n = N - 1
    for q in [Fr(1, 3), Fr(4, 5), Fr(1)]:
        f = pmf_exact(N, q, 1, n + 3)
        base = (q / 2) ** n
        h1, h2 = f[n + 1] / base, f[n + 2] / base
        ok &= (h1 == n - (n - Fr(1, 2)) * q)
        ok &= (2 * (h2 - h1) == n * (n - 1) - n * (2 * n - 1) * q + (n * n + Fr(n, 2) - 1) * q * q)
res["prop38_h1_h2_identities"] = ok
print("(1) h1,h2 identities:", ok)

# (2)
ok = True
for N, x0 in [(4, 1), (6, 1), (6, 3)]:
    for q0, q in [(Fr(1), Fr(1, 2)), (Fr(4, 5), Fr(1, 3)), (Fr(1), Fr(9, 10))]:
        p = q / q0
        T = 30
        a = pmf_exact(N, q0, x0, T)
        fq = pmf_exact(N, q, x0, T)
        for t in range(1, T + 1):
            s = sum(a[k] * comb(t - 1, k - 1) * p ** k * (1 - p) ** (t - k) for k in range(1, t + 1))
            ok &= (s == fq[t])
res["lemma37_compounding_identity"] = ok
print("(2) compounding identity:", ok)

# (3)
ok = True; okuni = True; okvan = True
for N in [3, 4, 5, 8, 13]:
    W = 2 * N - 1
    for x0 in range(1, N):
        d = N - x0; d1 = N + x0 - 1
        T = 6 * N * N
        f = pmf_exact(N, Fr(1), x0, T)
        for t in range(1, min(60, T)):
            D = d if (t - d) % 2 == 0 else d1
            s = mp.mpf(0)
            for k in range(1, N):
                th = k * mp.pi / W
                s += mp.sin(th) * mp.sin(D * th) * mp.cos(th) ** (t - 1)
            s *= mp.mpf(2) / W
            ok &= abs(s - mp.mpf(f[t].numerator) / f[t].denominator) < mp.mpf(10) ** -30
            if t < d or (t < d1 and (t - d1) % 2 == 0):
                okvan &= (f[t] == 0)
        for D in (d, d1):
            seq = [f[t] for t in range(D, T, 2)]
            i = max(range(len(seq)), key=lambda j: seq[j])
            okuni &= all(seq[j] < seq[j + 1] for j in range(i - 1)) and all(seq[j] > seq[j + 1] for j in range(i + 1, len(seq) - 1)) and all(s > 0 for s in seq)
res["thm39_parity_formula"] = ok; res["thm39_vanishing"] = okvan; res["thm39_class_unimodal"] = okuni
print("(3) parity formula:", ok, " vanishing:", okvan, " class unimodality:", okuni)

# (4)
ok = True; oktp = True
for N in [3, 4, 6, 9]:
    for q in [Fr(1, 10), Fr(1, 2), Fr(2, 3)]:
        T = 5 * N * N
        f = pmf_exact(N, q, 1, T)
        ok &= all(f[t] ** 2 >= f[t - 1] * f[t + 1] for t in range(2, T))
        n = N - 1
        Q = Q_matrix(N, q, exact=True)
        K = [[Fr(1) if x == n - 1 else Fr(0) for x in range(n)]]
        for t in range(1, 25):
            K.append([sum(Q[x][z] * K[-1][z] for z in range(n)) for x in range(n)])
            oktp &= all(K[t][x] * K[t - 1][y] - K[t - 1][x] * K[t][y] >= 0 for x in range(n) for y in range(x + 1, n))
res["thm35_logconcave_q_le_2_3"] = ok; res["thm35_claim_M_nonneg"] = oktp
# and failure just above 2/3 for the kernel claim is allowed; log-concavity itself fails above q_lc(N)
f = pmf_exact(4, Fr(79, 100), 1, 40)
res["logconcavity_fails_N4_q079"] = not all(f[t] ** 2 >= f[t - 1] * f[t + 1] for t in range(2, 39))
print("(4) log-concave q<=2/3:", ok, " M_t>=0:", oktp, " fails at N=4,q=0.79:", res["logconcavity_fails_N4_q079"])

# (5)
def Gser(u, j=0):
    s = mp.mpf(0)
    for k in range(1, 400, 2):
        y = k * mp.pi / 4
        s += (1 if ((k - 1) // 2) % 2 == 0 else -1) * 2 * y * (-2 * y * y) ** j * mp.e ** (-2 * u * y * y)
    return s
def Gimg(u):
    return sum((-1) ** n * (2 * n + 1) / mp.sqrt(2 * mp.pi * u ** 3) * mp.e ** (-(2 * n + 1) ** 2 / (2 * u)) for n in range(-60, 60))
e1 = max(abs(Gser(u) - Gimg(u)) for u in [mp.mpf(x) / 100 for x in (5, 10, 20, 33, 50, 100, 300)])
e2 = max(abs(mp.quad(lambda u: mp.e ** (-s * u) * Gimg(u), [0, 0.05, 0.3, 1, 5, 40, 200]) - 1 / mp.cosh(mp.sqrt(2 * s))) for s in [mp.mpf(0), mp.mpf(1) / 2, mp.mpf(3)])
mean = mp.quad(lambda u: u * Gimg(u), [0, 0.05, 0.3, 1, 5, 40, 300])
res["G_image_vs_series_max_err"] = float(e1); res["G_laplace_vs_sech_max_err"] = float(e2); res["G_mean"] = float(mean)
print("(5) image form err", mp.nstr(e1, 3), " LT err", mp.nstr(e2, 3), " mean", mp.nstr(mean, 15))

# elementary bounds (Lemma 5.1, 5.2) on a grid
worst = {}
def upd(name, val, bound):
    worst[name] = max(worst.get(name, -1e9), float(val - bound))
for i in range(0, 2001):
    z = mp.mpf(i) / 2000 * mp.pi ** 2 / 4
    if z == 0: z = mp.mpf(10) ** -12
    S = lambda zz: mp.sin(mp.sqrt(zz)) / mp.sqrt(zz)
    S0, S1, S2 = S(z), mp.diff(S, z), mp.diff(S, z, 2)
    C = lambda zz: mp.cos(mp.sqrt(zz)) ** 2
    C1, C2 = mp.diff(C, z), mp.diff(C, z, 2)
    be = lambda zz: S(zz) ** 2
    b1, b2 = mp.diff(be, z), mp.diff(be, z, 2)
    upd("S<=1", S0, 1); upd("S>=2/pi", 2 / mp.pi, S0); upd("|S'|<=1/6", abs(S1), mp.mpf(1) / 6); upd("S'<=0", S1, 0)
    upd("|S''|<=1/60", abs(S2), mp.mpf(1) / 60); upd("S''>=0", -S2, 0)
    upd("|C'|<=1", abs(C1), 1); upd("C'<=0", C1, 0); upd("|C''|<=2/3", abs(C2), mp.mpf(2) / 3); upd("C''>=0", -C2, 0)
    upd("|beta'|<=1/3", abs(b1), mp.mpf(1) / 3); upd("|beta''|<=4/45", abs(b2), mp.mpf(4) / 45)
    for j, a1, a2 in [(0, mp.mpf(7) / 6, mp.mpf(61) / 60), (1, mp.mpf(9) / 6, mp.mpf(113) / 60)]:
        A = lambda zz: S(zz) ** (2 * j + 1) * C(zz)
        upd(f"|A_{j}|<=1", abs(A(z)), 1); upd(f"|A_{j}'|<=al1", abs(mp.diff(A, z)), a1); upd(f"|A_{j}''|<=al2", abs(mp.diff(A, z, 2)), a2)
# B bounds on the low range, q <= qbar = 4/5
qbar = mp.mpf(4) / 5
Lam = lambda w: -mp.log(1 - w) / w if w > 0 else mp.mpf(1)
wb = qbar / 2
l0 = Lam(wb); l1 = (wb / (1 - wb) + mp.log(1 - wb)) / wb ** 2; l2 = 1 / (wb * (1 - wb) ** 2) - 2 / (wb ** 2 * (1 - wb)) - 2 * mp.log(1 - wb) / wb ** 3
B1 = max(l0 / 3, 2 * qbar * l1); B2 = max(mp.mpf(4) / 45 * l0 + 4 * qbar ** 2 * l2, mp.mpf(8) / 3 * qbar * l1)
for qi in range(1, 9):
    q = mp.mpf(qi) / 10
    for i in range(0, 201):
        z = mp.mpf(i) / 200 * mp.pi ** 2 / 36
        if z == 0: z = mp.mpf(10) ** -10
        B = lambda zz: (mp.sin(mp.sqrt(zz)) ** 2 / zz) * Lam(2 * q * mp.sin(mp.sqrt(zz)) ** 2)
        upd("|B'|<=b1(4/5)", abs(mp.diff(B, z)), B1); upd("|B''|<=b2(4/5)", abs(mp.diff(B, z, 2)), B2)
        upd("B>=beta*", 9 / mp.pi ** 2, B(z))
res["elementary_bounds_worst_violation(<=0 means OK)"] = worst
print("elementary bounds: max violation (must be <= ~1e-9):", max(worst.values()))
res["b1(4/5)"] = float(B1); res["b2(4/5)"] = float(B2)
json.dump(res, open(_os.path.join(_R, 'data', 'msc_rigorous_1d', '09_sanity.json'), "w"), indent=1)
