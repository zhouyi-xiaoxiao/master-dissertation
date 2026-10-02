"""Numerical check of the limit statements of Theorem 11.3 (first-order pole structure, d = 2, 3):

   Lam1^2 K2 -> Z_d,     Y(sigma_1) - Y_0 -> Z'_d = sum_{|k|^2>=2} 1/(|k|^2(|k|^2-1)),     W/delta - Xbar -> Z'_d - 2d - 1,
   Xbar * r_1 -> -2d,    Xbar*(r_0 - 1) -> mu_inf,    Lam1 (m - tau) -> mu_inf   (mu_inf = pi ln 2 for d=2, (pi^2/6) c_{m tau 3} for d=3)

Floating point (double), exploiting the symmetry k_1 <= ... <= k_d.  Output ../data/10_asymptotic_checks.json
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import os, sys, json, math, itertools
import numpy as np
from scipy.optimize import brentq

def Zp(d, R):
    r = np.arange(-R, R + 1)
    if d == 2:
        A, B = np.meshgrid(r, r, indexing="ij"); q = (A * A + B * B).astype(float).ravel()
    else:
        A, B, C = np.meshgrid(r, r, r, indexing="ij"); q = (A * A + B * B + C * C).astype(float).ravel()
    q4 = q[q >= 1]; q2 = q[q >= 2]
    # tail corrections: int_{|x|>R+1/2} |x|^-4 dx
    tail = math.pi / (R + 0.5) ** 2 if d == 2 else 4 * math.pi / (R + 0.5)
    return np.sum(1 / q4 ** 2) + tail, np.sum(1 / (q2 * (q2 - 1))) + tail

def data(N, d):
    k = np.arange(N); c = np.cos(np.pi * k / N)
    w1 = np.where(k == 0, 1.0, 2.0) * (1 + c) / 2
    if d == 2:
        i, j = np.triu_indices(N)
        lam = ((1 - c[i]) + (1 - c[j])) / 2; w = w1[i] * w1[j] * np.where(i == j, 1.0, 2.0); sg = (-1.0) ** (i + j)
    else:
        idx = np.array([(a, b, e) for a in range(N) for b in range(a, N) for e in range(b, N)])
        a, b, e = idx[:, 0], idx[:, 1], idx[:, 2]
        mult = np.where((a == b) & (b == e), 1.0, np.where((a == b) | (b == e), 3.0, 6.0))
        lam = ((1 - c[a]) + (1 - c[b]) + (1 - c[e])) / 3; w = w1[a] * w1[b] * w1[e] * mult; sg = (-1.0) ** (a + b + e)
    nz = lam > 1e-15
    return lam[nz], w[nz], sg[nz]

def run(N, d):
    lam, w, sg = data(N, d)
    Lam1 = (1 - math.cos(math.pi / N)) / d; W = 2 * d * math.cos(math.pi / (2 * N)) ** 2
    tau = np.sum(w / lam); m = np.sum((1 - sg) * w / lam); K2 = np.sum(w / lam ** 2)
    X = Lam1 * tau
    G = lambda s: -1 / s + np.sum(w / (lam - s))
    Gx = lambda s: -1 / s + np.sum(sg * w / (lam - s))
    dG = lambda s: 1 / s ** 2 + np.sum(w / (lam - s) ** 2)
    s0 = brentq(G, Lam1 * 1e-9, Lam1 * (1 - 1e-12), xtol=1e-300, rtol=1e-15)
    s1 = brentq(G, Lam1 * (1 + 1e-10), 2 * Lam1 * (1 - 1e-12), xtol=1e-300, rtol=1e-15)
    r0 = -Gx(s0) / (s0 * dG(s0)); r1 = -Gx(s1) / (s1 * dG(s1))
    delta = s1 / Lam1 - 1
    hi = lam > Lam1 * 1.5
    Y1 = Lam1 * np.sum(w[hi] / (lam[hi] - s1)); Y0 = Lam1 * np.sum(w[hi] / lam[hi])
    return dict(N=N, d=d, X=X, L1sqK2=Lam1 ** 2 * K2, Y1_minus_Y0=Y1 - Y0, W_over_delta_minus_X=W / delta - X,
                X_r1=X * r1, X_r0_minus_1=X * (r0 - 1), Lam1_m_minus_tau=Lam1 * (m - tau), one_minus_s0tau_X2=(1 - s0 * tau) * X ** 2,
                s0m_minus_1_X=(s0 * m - 1) * X, delta=delta)

out = {}
C14 = json.load(open(_os.path.join(_R, 'data', 'msc_rigorous_spectral', '14_certified_epstein.json')))
Z2, Zp2, Z3, Zp3 = (float(C14[k]["lower"]) for k in ("Z_2", "Z_2'", "Z_3", "Z_3'"))       # certified values (Lemma 7.6)
print("certified: Z_2 = %.8f  Z'_2 = %.8f   Z_3 = %.8f  Z'_3 = %.8f" % (Z2, Zp2, Z3, Zp3))
Z2c, Zp2c = Zp(2, 3000)                                                                     # crude float cross-check in d = 2
assert abs(Z2c - Z2) < 1e-5 and abs(Zp2c - Zp2) < 1e-5
out["Z"] = dict(Z2=Z2, Zp2=Zp2, Z3=Z3, Zp3=Zp3)
mu_inf = {2: math.pi * math.log(2), 3: math.pi ** 2 / 6 * 1.5315848840778}
lim = {2: dict(L1sqK2=Z2, Y1_minus_Y0=Zp2, W_over_delta_minus_X=Zp2 - 5, X_r1=-4.0, X_r0_minus_1=mu_inf[2], Lam1_m_minus_tau=mu_inf[2], one_minus_s0tau_X2=Z2, s0m_minus_1_X=mu_inf[2]),
       3: dict(L1sqK2=Z3, Y1_minus_Y0=Zp3, W_over_delta_minus_X=Zp3 - 7, X_r1=-6.0, X_r0_minus_1=mu_inf[3], Lam1_m_minus_tau=mu_inf[3], one_minus_s0tau_X2=Z3, s0m_minus_1_X=mu_inf[3])}
# tolerances at the largest N (the convergence is O(1/Xbar): logarithmic in d = 2, O(1/N) in d = 3)
tol = {2: dict(L1sqK2=1e-3, Y1_minus_Y0=0.6, W_over_delta_minus_X=0.7, X_r1=0.02, X_r0_minus_1=0.1, Lam1_m_minus_tau=1e-4, one_minus_s0tau_X2=0.1, s0m_minus_1_X=0.15),
       3: dict(L1sqK2=0.2, Y1_minus_Y0=0.07, W_over_delta_minus_X=0.07, X_r1=0.04, X_r0_minus_1=0.02, Lam1_m_minus_tau=2e-4, one_minus_s0tau_X2=0.2, s0m_minus_1_X=0.02)}
print("limits predicted: d=2: Z'-2d-1 = %.4f, mu_inf = %.5f ;  d=3: Z'-2d-1 = %.4f, mu_inf = %.5f" % (Zp2 - 5, mu_inf[2], Zp3 - 7, mu_inf[3]))
rows = []; OK = True
for d, Ns in ((2, (10, 30, 100, 300, 1000, 2500)), (3, (6, 12, 25, 50, 100, 160))):
    rr = []
    for N in Ns:
        r = run(N, d); rows.append(r); rr.append(r)
        print(f"d={d} N={N:5d} X={r['X']:.3f} L1^2K2={r['L1sqK2']:.5f} Y(s1)-Y0={r['Y1_minus_Y0']:.5f} W/delta-X={r['W_over_delta_minus_X']:+.5f} "
              f"X*r1={r['X_r1']:.5f} X*(r0-1)={r['X_r0_minus_1']:.5f} Lam1(m-tau)={r['Lam1_m_minus_tau']:.5f} (1-s0 tau)X^2={r['one_minus_s0tau_X2']:.5f} "
              f"(s0 m-1)X={r['s0m_minus_1_X']:.5f}", flush=True)
    for key, L in lim[d].items():
        first, last = abs(rr[0][key] - L), abs(rr[-1][key] - L)
        good = last <= tol[d][key] and last < first
        OK &= good
        print(f"   [{'OK ' if good else 'FAIL'}] d={d} {key}: limit {L:.5f}, distance {first:.4f} (N={Ns[0]}) -> {last:.5f} (N={Ns[-1]}), tolerance {tol[d][key]}")
out["rows"] = rows; out["all_ok"] = bool(OK)
json.dump(out, open(_os.path.join(_R, 'data', 'msc_rigorous_spectral', '10_asymptotic_checks.json'), "w"), indent=1, default=float)
print("ALL OK:", OK)
