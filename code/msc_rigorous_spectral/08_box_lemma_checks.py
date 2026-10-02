"""Numerical sanity checks (floating point / mpmath) for the lemmas of Part III of note R2
(reflecting box, corner-to-corner):

  A. arrival profile:   0 <= 1 - u(t) <= W e^{-Lam1 t},   0 <= Lam1 (m - tau) <= 1 + ln W           (Lemma 7.2)
  B. K2 bound:          Lam1^2 K2 <= Z_d(N) = sum_{0<|k|_inf<N} |k|^-4                              (Lemma 7.4)
  C. tau lower bound:   tau >= 2 n (1 - 1/n)^2                                                      (Lemma 7.7)
  D. elementary inequalities used in the proofs
  E. harmonic numbers:  H_{N-1} = ln N + gamma - 1/(2N) - eta_N, 1/(12(N+1)^2) < eta_N < 1/(12N^2)+1/(6N^3)
  F. d = 3: exact row formula  R(k1) - (N/pi) J(theta_k1) = (N/pi) int_0^pi P(theta_k1, t) dt        (Lemma 10.5)
  G. d = 3: D_N <= 2 B_D N, sum_k e(k) <= 8 B_e N                                                   (Lemma 10.6)
  H. d = 3: J_b = g1 - g3 with g1, g3 increasing; eps_b(N) inside the proved enclosure               (Lemmas 10.3, 10.4)
  I. final enclosures of tau_2, m_2, tau_3 (Theorems 9.1, 10.1)
  J. tail of the neglected poles (Theorem 12.1): brute force for small N

Output: ../data/08_box_lemma_checks.json, log on stdout.
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import os, sys, json, itertools, math
import numpy as np
import mpmath as mp
sys.path.insert(0, os.path.dirname(__file__))
from t2lib import corner_spectral_data, build_L, site_index
mp.mp.dps = 30
OUT = {}
ok_all = True
def report(name, ok, info=""):
    global ok_all
    ok_all &= bool(ok)
    print(f"[{'OK ' if ok else 'FAIL'}] {name} {info}", flush=True)
    OUT[name] = {"ok": bool(ok), "info": info}

# ---------------------------------------------------------------- A. arrival profile
worst_u = 0.0; rows = []
for d in (1, 2, 3, 4):
    for N in ((2, 3, 4, 7, 12, 30, 80) if d <= 2 else (2, 3, 4, 7, 12) if d == 3 else (2, 3, 5)):
        lam, w, sg = corner_spectral_data(N, d)
        nz = lam > 1e-14
        Lam1 = (1 - np.cos(np.pi / N)) / d
        W = 2 * d * np.cos(np.pi / (2 * N)) ** 2
        tau = np.sum(w[nz] / lam[nz]); m = np.sum((1 - sg[nz]) * w[nz] / lam[nz])
        ts = np.concatenate([np.linspace(0, 3 / Lam1, 400), np.linspace(3 / Lam1, 12 / Lam1, 100)])
        ok = True
        for t in ts:
            one_minus_u = -np.sum(sg[nz] * w[nz] * np.exp(-lam[nz] * t))
            bound = min(1.0, W * np.exp(-Lam1 * t))
            if not (-1e-10 <= one_minus_u <= bound + 1e-10):
                ok = False
            worst_u = max(worst_u, one_minus_u - bound)
        val = Lam1 * (m - tau)
        rows.append((d, N, float(val), float(1 + np.log(W))))
        ok &= (0 <= val <= 1 + np.log(W))
        if not ok:
            report(f"A arrival profile d={d} N={N}", False)
report("A arrival profile: 0<=1-u<=min(1,W e^{-Lam1 t}) and 0<=Lam1(m-tau)<=1+ln W", all(0 <= r[2] <= r[3] for r in rows) and worst_u <= 1e-9,
       "max Lam1(m-tau)/(1+lnW) = %.4f" % max(r[2] / r[3] for r in rows))
OUT["A_rows"] = rows

# ---------------------------------------------------------------- B. K2 bound, C. tau lower bound
def Zd_N(d, N):
    tot = 0.0
    for k in itertools.product(range(N), repeat=d):
        if any(k):
            tot += 2 ** sum(1 for v in k if v) / (sum(v * v for v in k)) ** 2
    return tot
rowsB = []; okB = True; okC = True
for d, Ns in ((1, (2, 5, 20)), (2, (2, 3, 5, 10, 30, 60)), (3, (2, 3, 5, 10, 16)), (4, (2, 3, 5, 7))):
    for N in Ns:
        lam, w, sg = corner_spectral_data(N, d)
        nz = lam > 1e-14
        Lam1 = (1 - np.cos(np.pi / N)) / d
        K2 = np.sum(w[nz] / lam[nz] ** 2); tau = np.sum(w[nz] / lam[nz]); n = N ** d
        z = Zd_N(d, N)
        okB &= Lam1 ** 2 * K2 <= z * (1 + 1e-12)
        okC &= tau >= 2 * n * (1 - 1 / n) ** 2 * (1 - 1e-12)
        rowsB.append((d, N, float(Lam1 ** 2 * K2), float(z), float(tau / (2 * n * (1 - 1 / n) ** 2))))
report("B Lam1^2 K2 <= Z_d(N)", okB, "max ratio %.4f" % max(r[2] / r[3] for r in rowsB))
report("C tau >= 2n(1-1/n)^2", okC, "min ratio %.4f" % min(r[4] for r in rowsB))
OUT["BC_rows"] = rowsB

# ---------------------------------------------------------------- D. elementary inequalities
v = np.linspace(1e-6, np.pi / 2, 200001)
okD = np.all((np.sin(v) / v) ** 2 >= np.cos(v) - 1e-15)
x = np.linspace(0, 1, 100001)
okD &= np.all((1 - x) ** 4 * (1 + 2 * x) <= 1 + 1e-15)
rho = np.linspace(1e-9, np.sqrt(2), 100001); kap3 = np.sqrt(2) * np.arcsinh(np.sqrt(2))
okD &= np.all(2 * np.arcsinh(rho) >= kap3 * rho - 1e-14)
okD &= np.all(v - np.arcsinh(np.sin(v)) <= v ** 3 / 3 + 1e-15)
okD &= np.all(np.sqrt(1 - x) >= 1 - x / 2 - x ** 2 / 2 - 1e-15) and np.all(np.sqrt(1 - x) <= 1 - x / 2 + 1e-15)
z = np.linspace(0, 50, 100001)
okD &= np.all(np.sqrt(1 + z) <= 1 + z / 2 - z ** 2 / 8 + z ** 3 / 16 + 1e-12)
# a(s) = e^{-s}(I0+I1): sqrt(2/(pi s))(1 - 1/(8s) - 3/(32 s^2)) <= a(s) <= sqrt(2/(pi s)) (all s>0);  a(s) <= sqrt(2/(pi s))(1-1/(8s)) + e^{-2s}/pi (s>=1)
for s in (0.05, 0.3, 1, 2, 5, 20, 100, 400):
    a = float(mp.besseli(0, s) * mp.exp(-s) + mp.besseli(1, s) * mp.exp(-s)); b = math.sqrt(2 / (math.pi * s))
    okD &= (b * (1 - 1 / (8 * s) - 3 / (32 * s * s)) <= a) and a <= b          # valid for all s > 0
    if s >= 1:                                                                  # upper bound proved for s >= 1 only
        okD &= a <= b * (1 - 1 / (8 * s)) + math.exp(-2 * s) / math.pi + 1e-15
report("D elementary inequalities ((sin v/v)^2>=cos v, (1-x)^4(1+2x)<=1, 2asinh(rho)>=kap3 rho, x-asinh(sin x)<=x^3/3, sqrt bounds, Bessel bounds)", okD)

# ---------------------------------------------------------------- E. harmonic numbers
okE = True
for N in (2, 3, 5, 10, 100, 10 ** 4, 10 ** 6):
    eta = -(mp.harmonic(N - 1) - mp.log(N) - mp.euler + mp.mpf(1) / (2 * N))
    okE &= (mp.mpf(1) / (12 * (N + 1) ** 2) < eta < mp.mpf(1) / (12 * N * N) + mp.mpf(1) / (6 * N ** 3)) and eta < mp.mpf(1) / (6 * N * N)
report("E harmonic number enclosure", okE)

# ---------------------------------------------------------------- F. exact row formula in d = 3
def Kfun(s):
    s = mp.mpf(s)
    f = lambda ph: 4 * mp.cos(ph) ** 2 * mp.sqrt((1 + s * s + mp.sin(ph) ** 2) / (s * s + mp.sin(ph) ** 2))     # t = sin(ph)
    return mp.quad(f, [0, min(s, mp.mpf(1) / 2), 1, mp.pi / 2])
def Hfun(t1, t2):
    c1, c2 = mp.cos(t1), mp.cos(t2); S2 = (1 - c1) / 2 + (1 - c2) / 2; S = mp.sqrt(S2)
    return (1 + c1) * (1 + c2) * (mp.sqrt(1 + S2) / S - 1)
def Pfun(t1, t2, N):
    c1, c2 = mp.cos(t1), mp.cos(t2); S2 = (1 - c1) / 2 + (1 - c2) / 2; S = mp.sqrt(S2)
    phi = 2 * mp.asinh(S)
    return (1 + c1) * (1 + c2) * mp.sqrt(1 + S2) / S * 2 / mp.expm1(2 * N * phi)
maxerrF = 0
for N in (3, 6, 11):
    for k1 in (1, 2, N - 1):
        t1 = mp.pi * k1 / N
        R = sum((mp.mpf(1) / 2 if k2 == 0 else 1) * Hfun(t1, mp.pi * k2 / N) for k2 in range(N))
        J = 2 * (1 - mp.sin(t1 / 2) ** 2) * (Kfun(mp.sin(t1 / 2)) - mp.pi)
        e_direct = R - N / mp.pi * J
        e_formula = N / mp.pi * mp.quad(lambda t: Pfun(t1, t, N), mp.linspace(0, mp.pi, 7))
        maxerrF = max(maxerrF, abs(e_direct - e_formula) / abs(e_formula))
report("F exact row formula e(k1) = (N/pi) int_0^pi P", maxerrF < 1e-12, "max rel err %.2e" % float(maxerrF))

# ---------------------------------------------------------------- G. D_N and sum_k e(k) bounds
kap3m = mp.sqrt(2) * mp.asinh(mp.sqrt(2))
B_D = mp.mpf(0)
for a in range(-40, 41):
    for b in range(-40, 41):
        if a or b:
            r = mp.sqrt(a * a + b * b); B_D += 1 / (r * mp.expm1(2 * kap3m * r))
B_e = sum(mp.sqrt(mp.pi / (4 * kap3m * k)) * mp.exp(-2 * kap3m * k) / (1 - mp.exp(-2 * kap3m * k)) for k in range(1, 40))
print("   kappa_3 = %.10f  B_D = %.10f  B_e = %.10f  ->  3(8 B_e + 2 B_D) = %.6f" % (kap3m, B_D, B_e, 3 * (8 * B_e + 2 * B_D)))
OUT["kappa3"] = float(kap3m); OUT["B_D"] = float(B_D); OUT["B_e"] = float(B_e)
def pieces3(N):
    k = np.arange(N); c = np.cos(np.pi * k / N); wt = np.where(k == 0, 0.5, 1.0)
    C1, C2 = np.meshgrid(c, c, indexing="ij"); Wt = np.outer(wt, wt)
    S2 = (1 - C1) / 2 + (1 - C2) / 2; S2[0, 0] = 1.0; S = np.sqrt(S2)
    cothh = np.sqrt(1 + S2) / S; phi = 2 * np.arcsinh(S)
    P = (1 + C1) * (1 + C2) * Wt; P[0, 0] = 0
    with np.errstate(over="ignore"):
        DN = np.sum(P * cothh * 2 / np.expm1(2 * N * phi))
    Hm = (1 + C1) * (1 + C2) * (cothh - 1)
    Rrow = np.sum(Hm[1:, :] * wt[None, :], axis=1)
    T2 = np.sum(P * (cothh - 1))
    sg = np.outer((-1.0) ** k, (-1.0) ** k)
    with np.errstate(over="ignore"):
        alt = np.sum(P * sg * cothh / np.sinh(N * phi))
    return DN, Rrow, T2, alt
okG = True; rowsG = []
Jcache = {}
def Jtheta(t1):
    s = mp.sin(t1 / 2)
    return 2 * (1 - s * s) * (Kfun(s) - mp.pi)
for N in (2, 3, 5, 8, 16, 32):
    DN, Rrow, T2, alt = pieces3(N)
    sume = float(sum(mp.mpf(float(Rrow[k1 - 1])) - N / mp.pi * Jtheta(mp.pi * k1 / N) for k1 in range(1, N)))
    okG &= (0 <= DN <= 2 * float(B_D) * N) and (0 <= sume <= 8 * float(B_e) * N)
    rowsG.append((N, DN / N, sume / N))
    print(f"   N={N:3d}: D_N/N = {DN/N:.6f} (<= {2*float(B_D):.4f}),  sum_k e(k)/N = {sume/N:.6f} (<= {8*float(B_e):.4f})")
report("G D_N <= 2 B_D N and sum e(k) <= 8 B_e N", okG)
OUT["G_rows"] = rowsG

# ---------------------------------------------------------------- H. J_b = g1 - g3
K1 = Kfun(1)
ss = [mp.mpf(i) / 200 for i in range(1, 201)]
Kv = [Kfun(s) for s in ss]
g1 = [2 * (Kv[i] + 4 * mp.log(ss[i]) - mp.pi) for i in range(200)]
g3 = [2 * ss[i] ** 2 * (Kv[i] - mp.pi) for i in range(200)]
okH = all(g1[i] < g1[i + 1] for i in range(199)) and all(g3[i] < g3[i + 1] for i in range(199))
g1_0 = 12 * mp.log(2) - 4 - 2 * mp.pi
okH &= g1[0] > g1_0 and abs(2 * (Kfun(mp.mpf("1e-6")) + 4 * mp.log(mp.mpf("1e-6")) - mp.pi) - g1_0) < 1e-4 and g3[0] > 0
print("   K(1) = %.12f, g1(0+) = 12 ln2 - 4 - 2pi = %.10f, g1(1) = g3(1) = %.10f" % (K1, g1_0, 2 * (K1 - mp.pi)))
eb_lo = -2 * (K1 - mp.pi); eb_hi = -g1_0 + 2 * (K1 - mp.pi)
print("   proved enclosure of eps_b: [%.6f, %.6f];  limit pi + 2 - 6 ln 2 = %.6f" % (eb_lo, eb_hi, mp.pi + 2 - 6 * mp.log(2)))
def Jb(t):
    s = mp.sin(t / 2)
    if s == 0:
        return g1_0
    Ks = Kfun(s)
    return 2 * (Ks + 4 * mp.log(s) - mp.pi) - 2 * s * s * (Ks - mp.pi)
intJb = mp.quad(Jb, mp.linspace(0, mp.pi, 17))
ebs = []
for N in (2, 3, 4, 8, 16, 64):
    eb = sum(Jb(mp.pi * k / N) for k in range(1, N)) - N / mp.pi * intJb
    ebs.append((N, float(eb)))
    okH &= eb_lo <= eb <= eb_hi
print("   eps_b(N):", ebs)
report("H g1, g3 increasing; eps_b(N) inside proved enclosure", okH)
OUT["K1"] = float(K1); OUT["eps_b"] = ebs; OUT["eps_b_enclosure"] = [float(eb_lo), float(eb_hi)]
C3 = 3 / mp.pi ** 2 * (intJb + 8 * mp.pi * mp.log(2))
print("   C_3 = (3/pi^2) int_0^pi J = %.12f" % C3)

# ---------------------------------------------------------------- I. final enclosures
IPsi = -mp.mpf(1) / 2 + mp.mpf(3) / 2 * mp.log(2) - mp.log(mp.pi)
Lam = mp.nsum(lambda k: 1 / (k * mp.expm1(2 * mp.pi * k)), [1, mp.inf])
c_tau = 8 / mp.pi * (mp.euler + IPsi + 2 * Lam) - mp.mpf(2) / 3
A0 = -1 - 24 * mp.log(2) / mp.pi + 12 / mp.pi * (mp.euler + IPsi)
M2 = mp.mpf("2.35")
R2_lo = mp.mpf(2) / 3 - 8 / (6 * mp.pi) - mp.pi ** 2 * M2 / 12 - 4 * mp.mpf("0.1102")
R2_hi = mp.mpf(2) / 3 + mp.pi ** 2 * M2 / 12 + 4 * mp.mpf("0.2301")
lo3 = A0 + 3 / mp.pi * eb_lo; hi3 = A0 + 3 / mp.pi * eb_hi + 3 * (8 * B_e + 2 * B_D)
rho_lo = 1 - 2 / mp.pi - mp.pi ** 2 * M2 / 8; rho_hi = 1 + mp.pi ** 2 * M2 / 8
print("   2D: R in [%.4f, %.4f];  3D: (tau_3 - C_3 N^3) in [%.4f N^2 %+.4f, %.4f N^2 %+.4f];  A_0 = %.6f" % (R2_lo, R2_hi, lo3, rho_lo, hi3, rho_hi, A0))
OUT["R2_enclosure"] = [float(R2_lo), float(R2_hi)]; OUT["tau3_enclosure_N2"] = [float(lo3), float(hi3)]; OUT["rho_enclosure"] = [float(rho_lo), float(rho_hi)]
okI = True; rowsI = []
def tau2_m2(N):
    Nm = mp.mpf(N); t = mp.mpf(0); alt = mp.mpf(0)
    for k in range(1, N):
        x = mp.pi * k / (2 * Nm); s = mp.sin(x); Phi = mp.cos(x) ** 2 * mp.sqrt(1 + s * s) / s; y = 2 * Nm * mp.asinh(s)
        t += Phi * mp.coth(y); alt += (-1) ** k * Phi / mp.sinh(y)
    tau2 = 4 * Nm * t - mp.mpf(2) / 3 * (Nm * Nm - 1)
    return tau2, tau2 + mp.mpf(2) / 3 * (Nm * Nm - 1) - 4 * Nm * alt
for N in (2, 3, 4, 5, 7, 10, 30, 100, 1000, 5000):
    t2, m2 = tau2_m2(N)
    R = t2 - 8 / mp.pi * N * N * mp.log(N) - c_tau * N * N
    Rp = m2 - t2 - 4 * mp.log(2) / mp.pi * N * N
    okI &= (R2_lo <= R <= R2_hi) and (-mp.mpf(2) / 3 - 8.384 <= Rp <= -mp.mpf(2) / 3 + 8.384)
    rowsI.append(("2D", N, float(R), float(Rp)))
for N in (2, 3, 4, 6, 10, 20, 40, 80, 200):
    DN, Rrow, T2, alt = pieces3(N)
    tau3 = 3 * (N - 1) * (2 * N - 1) / 3 + 3 * N * T2 + 3 * N * DN
    diff = tau3 - float(C3) * N ** 3
    m_minus_tau = (N * N - 1) - 3 * N * alt
    Lam1 = (1 - math.cos(math.pi / N)) / 3
    okI &= (float(lo3) * N * N + float(rho_lo) <= diff <= float(hi3) * N * N + float(rho_hi)) and (0 <= m_minus_tau <= (1 + math.log(6)) / Lam1)
    rowsI.append(("3D", N, diff / N ** 2, m_minus_tau / N ** 2))
for r in rowsI:
    print("   ", r)
report("I enclosures of tau_2 (R), m_2 - tau_2, tau_3 - C_3 N^3, m_3 - tau_3", okI)
OUT["I_rows"] = rowsI

# ---------------------------------------------------------------- J. tail of the neglected poles
def tail_check(N, d, q_list=(0.3, 0.5, 0.8, 1.0)):
    n = N ** d
    L = build_L(N, d)
    a = site_index((N,) * d, N, d); x0 = site_index((1,) * d, N, d)
    keep = [i for i in range(n) if i != a]
    La = L[np.ix_(keep, keep)]
    mu, Psi = np.linalg.eigh(La)
    kx = keep.index(x0)
    one = np.ones(n - 1)
    coef = Psi[kx, :] * (Psi.T @ one)                  # residue carried by each eigenvector of L_a
    lam, w, sg = corner_spectral_data(N, d)
    nz = lam > 1e-14
    Lam1 = (1 - np.cos(np.pi / N)) / d
    tau = np.sum(w[nz] / lam[nz]); K2 = np.sum(w[nz] / lam[nz] ** 2)
    X = Lam1 * tau; kap = K2 / tau ** 2
    if X <= 1:
        return True, 0.0
    beta = X / (X - 1)
    # the two lowest poles of the K_0-sector: smallest eigenvalue, and the unique one in (Lam1, 2 Lam1)
    order = np.argsort(mu)
    j0 = order[0]
    in1 = [j for j in order if Lam1 * (1 + 1e-9) < mu[j] < 2 * Lam1 * (1 - 1e-9) and abs(coef[j]) > 1e-13]
    r0, s0 = coef[j0], mu[j0]
    r1 = sum(coef[j] for j in in1); s1 = mu[in1[0]] if in1 else 2 * Lam1
    rest = np.ones(n - 1, dtype=bool); rest[j0] = False; rest[in1] = False      # all eigenvectors except the two slowest poles
    # (remainders are computed as direct sums over 'rest', not by subtraction, to avoid cancellation)
    t0 = 1 / Lam1
    npt0 = np.sum(w * np.exp(-lam * t0))               # n p_{t0}(x0,x0)
    npt0_bound = (1 + N * np.sqrt(np.pi * d / (2 * t0))) ** d
    ok = npt0 <= npt0_bound * (1 + 1e-12)
    C = beta * np.sqrt(kap) * np.sqrt(npt0_bound)
    worst = 0.0
    for t in np.linspace(t0 / 2, 12 * t0, 60):
        rem = np.sum(coef[rest] * np.exp(-mu[rest] * t))
        bound = C * np.exp(-2 * Lam1 * (t - t0 / 2))
        worst = max(worst, abs(rem) / bound)
        remf = np.sum(coef[rest] * mu[rest] * np.exp(-mu[rest] * t))
        if t - t0 / 2 >= 1 / (2 * Lam1):
            boundf = C * 2 * Lam1 * np.exp(-2 * Lam1 * (t - t0 / 2))
            worst = max(worst, abs(remf) / boundf)
    ok &= worst <= 1 + 1e-9
    # discrete time, q <= 1/2
    for q in q_list:
        nu = 1 - q * mu
        vis2 = (np.abs(coef) > 1e-13) & (mu > 2 * Lam1 * (1 + 1e-9))
        if vis2.any():                                    # Corollary 12.2(c): |nu_j| < 1 - 2 q Lam1 for j >= 2
            ok &= np.max(np.abs(nu[vis2])) < 1 - 2 * q * Lam1
        tq0 = int(np.ceil(1 / (q * Lam1) / 2)) * 2        # even t0
        npq = np.sum(w * (1 - q * lam) ** tq0)
        Cq = beta * np.sqrt(kap) * np.sqrt(npq)
        for t in range(tq0 // 2, 14 * tq0, max(1, tq0 // 3)):
            rem = np.sum(coef[rest] * nu[rest] ** t)
            bound = Cq * (1 - 2 * q * Lam1) ** (t - tq0 // 2)
            worst = max(worst, abs(rem) / bound)
        ok &= worst <= 1 + 1e-9
    return ok, worst
okJ = True; wJ = []
for (N, d) in ((3, 2), (4, 2), (6, 2), (9, 2), (14, 2), (20, 2), (3, 3), (4, 3), (6, 3), (8, 3)):
    ok, worst = tail_check(N, d)
    okJ &= ok; wJ.append((N, d, float(worst)))
print("   tail: max |remainder|/bound per case:", wJ)
report("J tail of the neglected poles (continuous time; discrete time q = 0.3, 0.5, 0.8, 1.0)", okJ)
OUT["J_rows"] = wJ

# ---------------------------------------------------------------- K. general dimension (Lemma 7.7, Corollary 7.11)
okK = True; rowsK = []
for d, Ns in ((3, (3, 6, 12)), (4, (2, 3, 5, 8)), (5, (2, 3, 4))):
    for N in Ns:
        lam, w, sg = corner_spectral_data(N, d)
        nz = lam > 1e-14
        n = N ** d
        Lam1 = (1 - np.cos(np.pi / N)) / d
        tau = np.sum(w[nz] / lam[nz]); m = np.sum((1 - sg[nz]) * w[nz] / lam[nz]); K2 = np.sum(w[nz] / lam[nz] ** 2)
        X = Lam1 * tau
        G = lambda x: -1 / x + np.sum(w[nz] / (lam[nz] - x))
        from scipy.optimize import brentq
        s0 = brentq(G, Lam1 * 1e-12, Lam1 * (1 - 1e-13), xtol=1e-300, rtol=1e-15)
        z = Zd_N(d, N)
        zb = 2 * d * 3 ** (d - 1) * sum(j ** (d - 5.0) for j in range(1, N))
        c1 = X >= 4 / d * N ** (d - 2) * (1 - N ** (-d)) ** 2 * (1 - 1e-12)
        c2 = tau <= d * d * 3 ** (d - 1) / (d - 2) * N ** d
        c3 = z <= zb * (1 + 1e-12)
        c4 = (X <= 1) or (abs(s0 * tau - 1) <= X / (X - 1) * z / X ** 2 * (1 + 1e-9))
        c5 = abs(s0 * m - 1) <= (1 + np.log(2 * d)) / X
        okK &= c1 and c2 and c3 and c4 and c5
        rowsK.append((d, N, float(X), float(1 - s0 * tau), float(s0 * m - 1), float(z)))
print("   general d rows (d, N, Xbar, 1-s0*tau, s0*m-1, Z_d(N)):", rowsK)
report("K general dimension: Xbar lower bound, tau upper bound, Z_d(N) bound, |s0 tau-1|, |s0 m-1| (Lemma 7.7, Cor. 7.11)", okK)
OUT["K_rows"] = rowsK

# ---------------------------------------------------------------- L. stationary start and sizes of residues (Prop. 5.4)
def uniform_check(N, d, a, x0):
    n = N ** d
    L = build_L(N, d)
    ia = site_index(a, N, d); ix = site_index(x0, N, d)
    keep = [i for i in range(n) if i != ia]
    La = L[np.ix_(keep, keep)]
    mu, Psi = np.linalg.eigh(La)
    one = np.ones(n - 1); kx = keep.index(ix)
    # group eigenvalues; residues per distinct eigenvalue
    c1 = Psi.T @ one
    rho = c1 ** 2 / n                                   # contribution of each eigenvector to varrho
    r = Psi[kx, :] * c1
    groups = []
    i = 0
    while i < n - 1:
        j = i
        while j + 1 < n - 1 and abs(mu[j + 1] - mu[i]) < 1e-10: j += 1
        groups.append((r[i:j + 1].sum(), rho[i:j + 1].sum()))
        i = j + 1
    rr = np.array([g[0] for g in groups]); vr = np.array([g[1] for g in groups])
    vis = vr > 1e-13
    okk = abs(vr.sum() - (1 - 1 / n)) < 1e-10 and np.all(np.abs(rr[~vis]) < 1e-9)
    okk &= np.sum(rr[vis] ** 2 / (n * vr[vis])) <= 1 + 1e-9 and np.sum(np.abs(rr)) <= np.sqrt(n - 1) + 1e-9
    return okk, float(np.sum(rr[vis] ** 2 / (n * vr[vis]))), float(np.sum(np.abs(rr)) / np.sqrt(n - 1))
okL = True; rowsL = []
for (N, d, a, x0) in ((5, 1, (5,), (1,)), (4, 2, (4, 4), (1, 1)), (6, 2, (6, 6), (1, 1)), (7, 2, (4, 4), (1, 1)), (6, 2, (2, 5), (1, 1)), (4, 3, (4, 4, 4), (1, 1, 1)), (4, 3, (2, 3, 4), (1, 1, 1))):
    okk, v1, v2 = uniform_check(N, d, a, x0); okL &= okk; rowsL.append((N, d, v1, v2))
print("   (N, d, sum r_j^2/(n rho_j), sum|r_j|/sqrt(n-1)):", rowsL)
report("L stationary-start weights positive, sum = 1-1/n; sum r_j^2/(n rho_j) <= 1; sum |r_j| <= sqrt(n-1) (Prop. 5.4)", okL)

print("ALL OK:", ok_all)
OUT["all_ok"] = bool(ok_all)
json.dump(OUT, open(_os.path.join(_R, 'data', 'msc_rigorous_spectral', '08_box_lemma_checks.json'), "w"), indent=1, default=float)
