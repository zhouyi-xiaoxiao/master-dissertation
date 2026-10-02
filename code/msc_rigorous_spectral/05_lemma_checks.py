"""Lemma-level numerical sanity checks for the lattice-sum proofs (Sections 8-10 of note R2).

Labels: "L7.1/L7.2" = Lemmas 8.1/8.2 (one summation in closed form, cotangent sums); "L8" = Lemmas 7.1(d), 9.5 (Psi);
"Theorem 8" = Theorem 9.1 (d = 2); "Section 9" = Section 10 (d = 3).

Each block prints the quantity appearing in a lemma and the bound claimed in the text.
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import os, sys, json
import numpy as np
import mpmath as mp
mp.mp.dps = 30
out = {}
kap = float(2 * mp.asinh(1))

# ---------- Lemma 7.1: cycle Green function -------------------------------------------------
def cycle_check(N, phi):
    th = np.pi * np.arange(2 * N) / N
    errs = []
    for j in range(0, 2 * N + 1):
        lhs = np.sum(np.cos(j * th) / (np.cosh(phi) - np.cos(th))) / (2 * N)
        rhs = np.cosh((N - j) * phi) / (np.sinh(phi) * np.sinh(N * phi))
        errs.append(abs(lhs - rhs) / (abs(rhs) + 1e-9))
    k = np.arange(N)
    eps = np.where(k == 0, 0.5, 1.0)
    thk = np.pi * k / N
    s1 = np.sum(eps * (1 + np.cos(thk)) / (np.cosh(phi) - np.cos(thk)))
    r1 = N * (1 / np.tanh(phi / 2) / np.tanh(N * phi) - 1)
    s2 = np.sum(eps * (-1.0) ** k * (1 + np.cos(thk)) / (np.cosh(phi) - np.cos(thk)))
    r2 = N / np.tanh(phi / 2) / np.sinh(N * phi)
    return max(errs), abs(s1 - r1) / r1, abs(s2 - r2) / (abs(r2) + 1e-9)
worst = max(max(cycle_check(N, phi)) for N in (2, 3, 7, 16) for phi in (0.05, 0.3, 1.0, 2.5))
print("L7.1 cycle Green function, max err (relative to |rhs|+1e-9):", worst); out["cycle_lemma_maxerr"] = worst
# cot^2 sums
for N in (2, 3, 10, 57):
    k = np.arange(1, N); c2 = 1 / np.tan(np.pi * k / (2 * N)) ** 2
    assert abs(c2.sum() - (N - 1) * (2 * N - 1) / 3) < 1e-8 * N * N and abs(((-1.0) ** k * c2).sum() + (N * N - 1) / 3) < 1e-8 * N * N
print("L7.2 cot^2 sums OK")

# ---------- Lemma 8.x: the function Phi, Psi -------------------------------------------------
Phi = lambda x: np.cos(x) ** 2 * np.sqrt(1 + np.sin(x) ** 2) / np.sin(x)
x = np.linspace(1e-4, np.pi / 2, 200001)
Psi = Phi(x) - 1 / x
s = np.sin(x)
Q = -s * (1 + s ** 2 - s ** 4) / (1 + (1 - s ** 2) * np.sqrt(1 + s ** 2))
print("L8 decomposition Psi = (csc x - 1/x) + Q(sin x): max err", np.max(np.abs(Psi - (1 / s - 1 / x + Q))))
print("L8 Psi/x range:", (Psi / x).min(), (Psi / x).max(), "  claimed [-1.09, 0]")
out["Psi_over_x_range"] = [float((Psi / x).min()), float((Psi / x).max())]
asr = np.arcsinh(np.sin(x))
print("L8 asinh(sin x)/x range:", (asr / x).min(), (asr / x).max(), " claimed [2asinh(1)/pi = %.5f, 1]" % (kap / np.pi))
print("L8 (x - asinh(sin x))/x^3 max:", ((x - asr) / x ** 3).max(), " claimed <= 1/3")
out["x_minus_asinhsin_over_x3_max"] = float(((x - asr) / x ** 3).max())

# ---------- Theorem 8: remainder pieces in d = 2 ---------------------------------------------
IPsi = float(-mp.mpf(1) / 2 + mp.mpf(3) / 2 * mp.log(2) - mp.log(mp.pi))
Lam = float(mp.nsum(lambda k: 1 / (k * (mp.exp(2 * mp.pi * k) - 1)), [1, mp.inf]))
Salt = float(mp.log(2) / 2 - mp.pi / 12)
rows = []
for N in [2, 3, 4, 5, 7, 10, 20, 50, 100, 300, 1000, 3000, 10000, 100000]:
    k = np.arange(1, N); xk = np.pi * k / (2 * N); yk = 2 * N * np.arcsinh(np.sin(xk))
    P = Phi(xk)
    e1 = np.sum(P - 1 / xk) - (2 * N / np.pi) * IPsi - 1 / np.pi
    Ec = np.sum(P * 2 * np.exp(-2 * yk) / (1 - np.exp(-2 * yk))) - (2 * N / np.pi) * 2 * Lam
    with np.errstate(over="ignore"):
        AN = np.sum((-1.0) ** (k + 1) * P * 2 * np.exp(-yk) / (1 - np.exp(-2 * yk)))
    eA = AN - (2 * N / np.pi) * Salt
    HNm1 = float(mp.harmonic(N - 1)); thp = -(HNm1 - np.log(N) - np.euler_gamma + 1 / (2 * N)) * 6 * N * N
    rows.append((N, e1 * N, Ec * N, eA * N, thp))
    print(f"N={N:6d}  N*e1={e1*N:+.5f} (|.|<= pi^2 M2/48 = {np.pi**2*2.35/48:.4f})  N*E_c={Ec*N:+.5f} (in [-0.1102,0.2301])  "
          f"N*(A_N-(2N/pi)Sig)={eA*N:+.5f} (|.|<=2.096)  theta'={thp:.4f} (in (0,1))")
out["2d_remainder_pieces"] = rows
# constants used in the bounds
S1 = sum(k * k / np.sinh(kap * k) ** 2 for k in range(1, 60))
S2 = sum(2 * k / (np.exp(2 * kap * k) - 1) for k in range(1, 60))
S3 = sum(k * k * np.cosh(kap * k) / np.sinh(kap * k) ** 2 for k in range(1, 80))
S4 = sum(k / np.sinh(kap * k) for k in range(1, 80))
print("constants: sum k^2/sinh^2(kappa k) =", S1, " sum 2k/(e^{2 kappa k}-1) =", S2, " sum k^2 cosh/sinh^2 =", S3, " sum k/sinh =", S4)
out["consts_2d"] = dict(S1=S1, S2=S2, S3=S3, S4=S4)

# ---------- Section 9: d = 3 pieces ------------------------------------------------------------
def K(B):
    B = mp.mpf(B)
    f = lambda v: 2 * mp.sqrt(max(2 - v * v, 0) * (2 + B + v * v) / (v * v + B))      # u = v^2
    return mp.quad(f, sorted(set([mp.mpf(0), min(mp.sqrt(B), mp.sqrt(2)) if B > 0 else mp.mpf('1e-10'), mp.mpf(1) / 2, mp.mpf(1), mp.sqrt(2)])))
def Kr(B):
    B = mp.mpf(B)
    if B == 0:
        return -2 * (1 - mp.log(2))
    f = lambda v: 2 * (mp.sqrt(max(2 - v * v, 0) * (2 + B + v * v)) - 2) / mp.sqrt(v * v + B)
    return mp.quad(f, sorted(set([mp.mpf(0), min(mp.sqrt(B), mp.sqrt(2)), mp.mpf(1) / 2, mp.mpf(1), mp.sqrt(2)])))
Bs = [0, 1e-6, 1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.3, 0.6, 1.0, 1.5, 2.0]
Krs = [Kr(B) for B in Bs]
print("K_r(B):", [float(v) for v in Krs])
print("K_r increasing:", all(Krs[i] < Krs[i + 1] for i in range(len(Krs) - 1)), " K_r(0) = -2(1-ln2) =", float(-2 * (1 - mp.log(2))),
      " K_r(2) =", float(Krs[-1]), "< pi:", Krs[-1] < mp.pi)
print("check K(B) = 4 asinh(sqrt(2/B)) + K_r(B) at B=0.3:", float(K(0.3) - 4 * mp.asinh(mp.sqrt(2 / mp.mpf(0.3))) - Kr(0.3)))
out["K_r"] = dict(B=Bs, Kr=[float(v) for v in Krs])
# J(theta) = int_0^pi H dtheta2 = (1+c)(K(B)-pi)
def H(t1, t2):
    c1, c2 = mp.cos(t1), mp.cos(t2)
    S = mp.sqrt((1 - c1) / 2 + (1 - c2) / 2)
    return (1 + c1) * (1 + c2) * (mp.sqrt(1 + S * S) / S - 1)
for t1 in (0.2, 1.0, 2.5):
    Jd = mp.quad(lambda t2: H(t1, t2), [0, 0.5, 1.5, mp.pi])
    Jf = (1 + mp.cos(t1)) * (K(1 - mp.cos(t1)) - mp.pi)
    print(f"J({t1}) direct {float(Jd):.12f} formula {float(Jf):.12f}")
# J_b pieces and the Riemann-sum discrepancy eps_b(N) = sum_k J_b(theta_k) - (N/pi) int J_b
def Jb(t):
    s = mp.sin(t / 2)
    if s == 0:
        return 8 * mp.log(2) + 2 * (Kr(0) - mp.pi)
    return 8 * s * s * mp.log(s) + 8 * (1 - s * s) * mp.log(1 + mp.sqrt(1 + s * s)) + 2 * (1 - s * s) * (Kr(2 * s * s) - mp.pi)
intJb = mp.quad(Jb, mp.linspace(0, mp.pi, 9))
C3 = mp.mpf("4.32046016937353691333984222844")
print("int_0^pi J_b =", float(intJb), "  check: int J = pi^2 C3/3 :", float(intJb + 8 * mp.pi * mp.log(2)), float(mp.pi ** 2 * C3 / 3))
for N in (4, 16, 64):
    eb = sum(Jb(mp.pi * k / N) for k in range(1, N)) - N / mp.pi * intJb
    print(f"N={N}: eps_b = {float(eb):+.5f}   (proved enclosure [-2.2062, 4.1716], Lemma 10.4; limit -J_b(0)/2 = {float(-Jb(0)/2):.5f})")
out["Jb0"] = float(Jb(0)); out["intJb"] = float(intJb)

# row corrections e(k1) and D_N
def pieces3(N):
    k = np.arange(N); c = np.cos(np.pi * k / N); wt = np.where(k == 0, 0.5, 1.0)
    C1, C2 = np.meshgrid(c, c, indexing="ij"); Wt = np.outer(wt, wt)
    S2 = (1 - C1) / 2 + (1 - C2) / 2; S2[0, 0] = 1.0; S = np.sqrt(S2)
    cothh = np.sqrt(1 + S2) / S; phi = 2 * np.arcsinh(S)
    P = (1 + C1) * (1 + C2) * Wt; P[0, 0] = 0
    e2 = np.exp(-2 * N * phi)
    DN = np.sum(P * cothh * 2 * e2 / (1 - e2))
    Hm = (1 + C1) * (1 + C2) * (cothh - 1)
    rows_sum = np.sum(Hm[1:, :] * wt[None, :], axis=1)          # R(k1), k1 = 1..N-1
    return DN, rows_sum
for N in (8, 32, 128):
    DN, R = pieces3(N)
    es = []
    for k1 in range(1, min(N, 6)):
        t1 = np.pi * k1 / N
        J = float((1 + mp.cos(t1)) * (K(1 - mp.cos(t1)) - mp.pi))
        e = R[k1 - 1] - N / np.pi * J
        xk = np.pi * k1 / (2 * N); yk = 2 * N * np.arcsinh(np.sin(xk))
        kap3 = np.sqrt(2) * np.arcsinh(np.sqrt(2))
        bound = 8 * N * np.sqrt(np.pi / (4 * kap3 * k1)) * np.exp(-2 * kap3 * k1) / (1 - np.exp(-2 * kap3 * k1))   # Lemma 10.6
        es.append((k1, e, bound))
    print(f"N={N}: D_N/N={DN/N:.5f} (proved <= 2 B_D = 0.3957, Lemma 10.6);  e(k1) vs bound:", ", ".join(f"k={a}: {b:.4f}<={c_:.4f}" for a, b, c_ in es))
json.dump(out, open(_os.path.join(_R, 'data', 'msc_rigorous_spectral', '05_lemma_checks.json'), "w"), indent=1, default=float)
