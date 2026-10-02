"""Sanity checks (float64 / mpmath) for the d = 3 remainder analysis (Lemmas 10.8-10.11, Theorem 10.1(c),(d)).

 A. closed form K(s) = (4+2s^2) K(k) - (2+2s^2) E(k), k = 1/(1+s^2), against quadrature; g_00, g_11 formulas.
 B. M' = 4B/s, B' = 2s(K-E), M'' = 8(K-E) - M'/s, formula for 4 J_b''; J_b'(0) = J_b'(pi) = 0; crude bounds near 0.
 C. comparison lemma: |A - A_inf| <= 4|x|, 0 <= pi|kappa| - y <= pi|kappa||x|^2/3, |P_N - P_inf| <= Q(|kappa|)/N, alternating analogue.
 D. for N = 2..N_max: eps_b(N), delta_N, Alt_N, R_3(N), R_mtau3(N) against the certified bounds of script 16.
Output: ../data/17_d3_remainder_checks.json ; prints ALL OK.
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import os, sys, json, math
import numpy as np
import mpmath as mp
from scipy.special import ellipk, ellipe
HERE = os.path.dirname(__file__)
C06 = json.load(open(_os.path.join(_R, 'data', 'msc_rigorous_spectral', '06_certified_constants.json')))
C16 = json.load(open(_os.path.join(_R, 'data', 'msc_rigorous_spectral', '16_certified_d3_remainder.json')))
C3 = float(C06["C_3"]["mid"]); c_tau3 = float(C06["c_tau_3D"]["mid"]); c_mt3 = float(C06["c_m_minus_tau_3D"]["mid"]); E_inf = float(C06["E_inf"]["mid"])
V = C16["V upper"]["upper"]; c_delta = C16["c_delta"]["upper"]; c_alt = C16["c_alt"]["upper"]
kap3 = math.sqrt(2) * math.asinh(math.sqrt(2))
OK = True; OUT = {}
def check(name, cond, info=""):
    global OK
    OK &= bool(cond)
    print(("[OK ] " if cond else "[FAIL] ") + name + ("  " + info if info else ""), flush=True)

# ---------------------------------------------------------------- A
mp.mp.dps = 30
def K_quad(s):
    return 4 * mp.quad(lambda t: mp.sqrt((1 - t * t) * (1 + s * s + t * t) / (s * s + t * t)), mp.linspace(0, 1, 9))
def K_cl(s):
    m = (1 / (1 + s * s)) ** 2
    return (4 + 2 * s * s) * mp.ellipk(m) - (2 + 2 * s * s) * mp.ellipe(m)
errA = max(abs(K_quad(mp.mpf(s)) - K_cl(mp.mpf(s))) for s in (0.03, 0.2, 0.5, 0.9, 1.0))
check("A. K(s) = (4+2s^2)K(k) - (2+2s^2)E(k) vs quadrature (5 values of s)", errA < 1e-20, f"max err {mp.nstr(errA, 3)}")
def g_num(m_, n_, E):
    f = lambda a, b: mp.cos(m_ * a) * mp.cos(n_ * b) / (E - mp.cos(a) - mp.cos(b))
    return mp.quad(f, [0, mp.pi / 2, mp.pi], [0, mp.pi / 2, mp.pi]) / mp.pi ** 2
mp.mp.dps = 15
errg = 0
for E in (2.3, 3.0, 4.0):
    k = 2 / mp.mpf(E); Kk = mp.ellipk(k * k); Ek = mp.ellipe(k * k)
    g00 = 2 / (mp.pi * E) * Kk; g11 = 2 / (mp.pi * E) * ((2 / k ** 2 - 1) * Kk - 2 / k ** 2 * Ek); g10 = (E * g00 - 1) / 2
    errg = max(errg, abs(g_num(0, 0, E) - g00), abs(g_num(1, 1, E) - g11), abs(g_num(1, 0, E) - g10))
check("A. g_00 = 2K/(pi E), g_11 = (2/(pi E))[(2/k^2-1)K - (2/k^2)E], 2 g_10 = E g_00 - 1 (k = 2/E) vs 2D quadrature", errg < 1e-8, f"max err {mp.nstr(errg, 3)}")

# ---------------------------------------------------------------- B
mp.mp.dps = 40
eK = lambda s: mp.ellipk((1 / (1 + s * s)) ** 2); eE = lambda s: mp.ellipe((1 / (1 + s * s)) ** 2)
Mf = lambda s: K_cl(s) + 4 * mp.log(s)
Bf = lambda s: 1 - (1 + s * s) * eE(s) + s * s * (s * s + 2) / (1 + s * s) * eK(s)
def Jb(th):
    s = mp.sin(th / 2); return 2 * (1 - s * s) * (K_cl(s) - mp.pi) + 8 * mp.log(s)
def four_Jb2(th):
    s = mp.sin(th / 2)
    return (-4 * (1 - 2 * s * s) * (Mf(s) - mp.pi) - 40 * (1 - s * s) * Bf(s) - 8 * (1 - s * s) ** 2 * Bf(s) / s ** 2
            + 16 * (1 - s * s) ** 2 * (eK(s) - eE(s)) + 16 * (1 - 2 * s * s) * mp.log(s) + 24 - 32 * s * s)
errB = 0
for s in (mp.mpf("1e-4"), mp.mpf("0.05"), mp.mpf("0.4"), mp.mpf("0.9")):
    errB = max(errB, abs(mp.diff(Mf, s) - 4 * Bf(s) / s), abs(mp.diff(Bf, s) - 2 * s * (eK(s) - eE(s))),
               abs(mp.diff(Mf, s, 2) - (8 * (eK(s) - eE(s)) - 4 * Bf(s) / s ** 2)) * s)
for th in (mp.mpf("1e-3"), mp.mpf("0.2"), mp.mpf(1), mp.mpf("2.5")):
    errB = max(errB, abs(four_Jb2(th) - 4 * mp.diff(Jb, th, 2)) / 100)
check("B. M' = 4B/s, B' = 2s(K-E), M'' = 8(K-E) - M'/s, closed formula for 4 J_b''", errB < 1e-20, f"max err {mp.nstr(errB, 3)}")
d0 = mp.diff(Jb, mp.mpf("1e-12")); dpi = mp.diff(Jb, mp.pi, direction=-1)
check("B. J_b'(0+) = 0 = J_b'(pi)", abs(d0) < 1e-9 and abs(dpi) < 1e-9, f"{mp.nstr(d0, 3)}, {mp.nstr(dpi, 3)}")
okb = True
for e in range(1, 12):
    s = mp.mpf(10) ** (-e) * (1 if e > 1 else mp.mpf("0.5"))
    L = mp.log(1 / s)
    okb &= abs(four_Jb2(2 * mp.asin(s))) <= 47 + 41 * L and 0 < Bf(s) <= s * s * (mp.pi / 2 - mp.mpf(1) / 2 + L)
    okb &= mp.log(4) <= eK(s) + mp.log(s * mp.sqrt(s * s + 2) / (1 + s * s)) <= mp.pi / 2 and 1 <= eE(s) <= mp.pi / 2
check("B. near 0: |4 J_b''| <= 47 + 41 ln(1/s), 0 < B <= s^2(pi/2 - 1/2 + ln(1/s)), ln 4 <= K + ln k' <= pi/2, 1 <= E <= pi/2", okb)
mp.mp.dps = 15

# ---------------------------------------------------------------- C
rng = np.random.default_rng(17)
okC = True; worst = [0, 0, 0, 0]
for N in (2, 3, 5, 10, 50, 400):
    for _ in range(4000):
        kp = rng.uniform(0, N, 2)
        if rng.random() < 0.3: kp = np.floor(kp)
        nk = math.hypot(*kp)
        if nk < 1: continue
        x = math.pi * kp / (2 * N); nx = math.hypot(*x); s = np.sin(x); rho = math.hypot(*s)
        A = 4 * (1 - s[0] ** 2) * (1 - s[1] ** 2) * math.sqrt(1 + rho * rho) / rho; Ainf = 4 / nx
        y = 2 * N * math.asinh(rho)
        r1 = abs(A - Ainf) / (4 * nx); r2 = (math.pi * nk - y) / (math.pi * nk * nx * nx / 3)
        okC &= r1 <= 1 + 1e-12 and -1e-12 <= r2 <= 1 + 1e-12 and y >= kap3 * nk * (1 - 1e-12)
        worst[0] = max(worst[0], r1); worst[1] = max(worst[1], r2)
        if kap3 * nk > 150:          # both sides below 1e-130: skip the (overflowing) ratio test
            continue
        Fc = lambda z: 2 / math.expm1(2 * z)
        PN = A * Fc(y); Pinf = Ainf * Fc(math.pi * nk)
        Q = 2 * math.pi * nk * Fc(kap3 * nk) + 2 * math.pi ** 2 / 3 * nk ** 2 / math.sinh(kap3 * nk) ** 2
        r3 = abs(PN - Pinf) * N / Q
        Qa = 2 * math.pi * nk / math.sinh(kap3 * nk) + 2 * math.pi ** 2 / 3 * nk ** 2 * math.cosh(kap3 * nk) / math.sinh(kap3 * nk) ** 2
        r4 = abs(A / math.sinh(y) - Ainf / math.sinh(math.pi * nk)) * N / Qa
        okC &= r3 <= 1 + 1e-9 and r4 <= 1 + 1e-9
        worst[2] = max(worst[2], r3); worst[3] = max(worst[3], r4)
check("C. Lemma 10.11 (10.14)-(10.15): |A-A_inf| <= 4|x|, 0 <= pi|k|-y <= pi|k||x|^2/3, |P_N-P_inf| <= Q/N, alternating analogue (24000 random points)", okC,
      "max ratios " + ", ".join(f"{w:.3f}" for w in worst))

# ---------------------------------------------------------------- D
def pieces(N):
    k = np.arange(N); th = np.pi * k / N; c = np.cos(th); epsp = np.where(k == 0, 0.5, 1.0)
    C1, C2 = np.meshgrid(c, c, indexing="ij"); S1 = np.sin(th / 2)[:, None]; S2 = np.sin(th / 2)[None, :]
    rho = np.sqrt(S1 ** 2 + S2 ** 2); rho[0, 0] = 1.0
    phi = 2 * np.arcsinh(rho)
    coth_half = np.sqrt(1 + rho ** 2) / rho
    w = np.outer(epsp, epsp) * (1 + C1) * (1 + C2); w[0, 0] = 0.0
    H = w * (coth_half - 1); PN = w * coth_half * (-2 / np.expm1(-2 * N * phi) - 1 - 1 + 1)       # coth(y) - 1 = 2/(e^{2y}-1)
    PN = w * coth_half * (2 / np.expm1(2 * N * phi))
    sg = np.outer((-1.0) ** k, (-1.0) ** k)
    Alt = np.sum(sg * w * coth_half / np.sinh(N * phi))
    tau = (N - 1) * (2 * N - 1) + 3 * N * (H.sum() + PN.sum())
    m = tau + (N * N - 1) - 3 * N * Alt
    # J(theta_k), k >= 1, via the closed form
    s = np.sin(th[1:] / 2); mm = (1 / (1 + s * s)) ** 2
    Kc = (4 + 2 * s * s) * ellipk(mm) - (2 + 2 * s * s) * ellipe(mm)
    J = 2 * (1 - s * s) * (Kc - np.pi)
    eps_b = J.sum() - N / math.pi * (math.pi ** 2 / 3 * C3) + 4 * math.log(N) + 8 * math.log(2)
    rows = H[1:, :].sum(axis=1) / 1.0          # sum_{k2} eps'_{k2} H(theta_k1, theta_k2)   (eps'_{k1} = 1 for k1 >= 1)
    eN = rows - N / math.pi * J
    D = PN.sum()
    return tau, m, Alt, eps_b, eN.sum(), D
eps_inf = math.pi + 2 - 6 * math.log(2)
A_inf = (1 - c_mt3) / 3
Vg = C16["V_g upper"]["upper"]; R3lo = C16["R_3 lower"]["lower"]; R3hi = C16["R_3 upper"]["upper"]
d_inf = float(C16["delta_inf"]["lower"]); a_inf = float(C16["alpha_inf"]["lower"]); c0 = C16["c0_tau3"]["lower"]; c0m = C16["c0_mtau3"]["lower"]
z3 = float(mp.zeta(3))
okD = True; rowsD = []
for N in list(range(2, 41)) + [50, 64, 80, 100, 128, 160, 200, 256, 320, 400, 512, 640, 800]:
    tau, m, Alt, eps_b, se, D = pieces(N)
    R3 = tau - C3 * N ** 3 - c_tau3 * N * N
    Rm = m - tau - c_mt3 * N * N
    delta = D + se - E_inf * N
    rN = eps_b - eps_inf - z3 / (4 * N * N)
    bound_r = math.sqrt(3) * math.pi ** 2 / 216 * Vg / N ** 2
    r = dict(N=N, R3=R3, R_mtau3=Rm, eps_b_minus_limit=eps_b - eps_inf, r_N=rN, bound_r=bound_r, weaker_bound_eps=math.pi * V / (12 * N),
             N_delta_N=N * delta, c_delta=c_delta, N_Alt_err=N * (Alt - N * A_inf), c_alt=c_alt)
    tol = 2e-11 * N                          # float64 cancellation in the N-sums
    good = (abs(rN) <= bound_r + tol and abs(eps_b - eps_inf) <= math.pi * V / (12 * N) + tol and abs(delta) <= c_delta / N + tol
            and abs(Alt - N * A_inf) <= c_alt / N + tol and R3lo <= R3 <= R3hi and -82.4 <= Rm <= 80.4
            and N * N - 1 <= m - tau and se > 0 and D > 0 and Alt < 0)
    okD &= good; rowsD.append(r)
    if N in (2, 3, 5, 10, 20, 40, 100, 200, 400, 800):
        print(f"   N={N:4d} R_3={R3:+.6f} R_3'={Rm:+.6f} | N^2 r_N={N*N*rN:+.2e} (|.| <= {N*N*bound_r:.3f}) | N delta_N={N*delta:+.6f} (lim {d_inf:.6f}, <= {c_delta:.3f})"
              f" | N(Alt-N A_inf)={N*(Alt-N*A_inf):+.6f} (lim {a_inf:.6f}, <= {c_alt:.2f})", flush=True)
check("D. Thm 10.1(c): r_N, delta_N, Alt_N, R_3, R_3' within the certified bounds, N = 2..40 and up to 800", okD)
last = rowsD[-1]
check("D. Thm 10.1(d): limits R_3 -> c0_tau3, R_3' -> c0_mtau3, N delta_N -> delta_inf, N(Alt_N - A_inf N) -> alpha_inf (N = 800)",
      abs(last["R3"] - c0) < 2e-5 and abs(last["R_mtau3"] - c0m) < 2e-5 and abs(last["N_delta_N"] - d_inf) < 1e-5 and abs(last["N_Alt_err"] - a_inf) < 1e-5,
      f"R_3(800)={last['R3']:.6f} vs {c0:.6f}; R_3'(800)={last['R_mtau3']:.6f} vs {c0m:.6f}")
# monotone approach of R_3, R_3' (observation)
R3s = [r["R3"] for r in rowsD]; Rms = [r["R_mtau3"] for r in rowsD]
OUT["R3_increasing_obs"] = bool(all(R3s[i] < R3s[i + 1] + 1e-7 for i in range(len(R3s) - 1)))
OUT["R3prime_increasing_obs"] = bool(all(Rms[i] < Rms[i + 1] + 1e-7 for i in range(len(Rms) - 1)))
print("   observation: R_3(N) increasing:", OUT["R3_increasing_obs"], " R_3'(N) increasing:", OUT["R3prime_increasing_obs"])
OUT["D"] = rowsD; OUT["all_ok"] = bool(OK)
json.dump(OUT, open(_os.path.join(_R, 'data', 'msc_rigorous_spectral', '17_d3_remainder_checks.json'), "w"), indent=1, default=float)
print("ALL OK:", OK)
