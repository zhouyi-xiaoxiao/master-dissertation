"""Sanity check of Theorems 6.2-6.8 (explicit two-sided bounds on sigma_0, sigma_1, r_0, r_1)
for the corner-to-corner geometry in d = 2, 3 (floating point; the certified version is 07_certify_poles.py).
A bound is tested only when its hypothesis holds (Y0 + theta > 1 for delta_+, (1+delta_+) rho < 1 for delta_-, r_1).

Unit rate.  Quantities:
  tau  = n * sum_{k != 0} phi_k(a)^2 / lambda_k            (mean hitting time from a uniform start)
  m    = mean hitting time from the opposite corner
  K2   = n * sum_{k != 0} phi_k(a)^2 / lambda_k^2
  Lam1, Lam2 = first two non-zero visible eigenvalues,  W = n * w_1
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import os, sys, json
import numpy as np
from scipy.optimize import brentq
sys.path.insert(0, os.path.dirname(__file__))
from t2lib import corner_spectral_data, group_visible


def exact_poles(N, d):
    lam, w, sg = corner_spectral_data(N, d)
    Lam, Wg, SW = group_visible(lam, w, sg)
    n = N ** d
    G = lambda s: np.sum(Wg / (Lam - s))
    Gx = lambda s: np.sum(SW / (Lam - s))
    dG = lambda s: np.sum(Wg / (Lam - s) ** 2)
    def root(i):
        gap = Lam[i + 1] - Lam[i]
        e = 1e-9 * gap
        while G(Lam[i] + e) > 0: e /= 8
        lo = Lam[i] + e
        e = 1e-9 * gap
        while G(Lam[i + 1] - e) < 0: e /= 8
        hi = Lam[i + 1] - e
        return brentq(G, lo, hi, xtol=1e-300, rtol=1e-15, maxiter=1000)
    s0, s1 = root(0), root(1)
    r = lambda s: -Gx(s) / (s * dG(s))
    return Lam, Wg, SW, s0, s1, r(s0), r(s1)


def bounds(N, d):
    lam, w, sg = corner_spectral_data(N, d)
    nz = lam > 1e-14
    tau = np.sum(w[nz] / lam[nz])
    m = np.sum((1 - sg[nz]) * w[nz] / lam[nz])
    K2 = np.sum(w[nz] / lam[nz] ** 2)
    Lam1 = (1 - np.cos(np.pi / N)) / d
    Lam2 = 2 * Lam1 if d >= 2 else (1 - np.cos(2 * np.pi / N))
    W = 2 * d * np.cos(np.pi / (2 * N)) ** 2
    A = W
    X = Lam1 * tau
    kap = K2 / tau ** 2
    beta = 1 / (1 - 1 / X)
    mu = (m - tau) / tau
    # (4a)
    t_lo = 2 / (1 + np.sqrt(1 + 4 * beta * kap))
    t_hi = 2 / (1 + np.sqrt(1 + 4 * kap))
    # (4b)
    Y0 = X - W
    th = Lam1 ** 2 * K2 - W
    rho = Lam1 / Lam2
    hyp_a = (Y0 + th - 1) > 0
    d_hi = W / (Y0 + th - 1) if hyp_a else np.nan
    ok_b = hyp_a and (1 + d_hi) * rho < 1
    d_lo = W / (Y0 + (1 + d_hi) * th / (1 - (1 + d_hi) * rho) - 1 / (1 + d_hi)) if ok_b else np.nan
    # (4c)  r0 in [(1 + t mu - beta kap t^2)/(1 + beta^2 kap t^2), (1 + t mu + beta kap t^2)/(1 + kap t^2)]
    r0_lo = (1 + t_lo * mu - beta * kap * t_hi ** 2) / (1 + beta ** 2 * kap * t_hi ** 2)
    r0_hi = (1 + t_hi * mu + beta * kap * t_hi ** 2) / (1 + kap * t_lo ** 2)
    # (4d)  r1 = -delta [A + delta (Yx - 1/(1+delta))] / [(1+delta) W + delta^2 (1/(1+delta) + (1+delta) theta(sigma1))]
    def r1_of(delta, Yx, th_s):
        return -delta * (A + delta * (Yx - 1 / (1 + delta))) / ((1 + delta) * W + delta ** 2 * (1 / (1 + delta) + (1 + delta) * th_s))
    if ok_b:
        Yx0 = A - Lam1 * (m - tau)
        Ex = (1 + d_hi) * th / (1 - (1 + d_hi) * rho)
        th_lo, th_hi = th, th / (1 - (1 + d_hi) * rho) ** 2
        cands = [r1_of(dd, Yx0 + e, t) for dd in (d_lo, d_hi) for e in (-Ex, Ex) for t in (th_lo, th_hi)]
        # monotone in each argument separately on the box => extremes at the corners (checked in the proof)
        r1_lo, r1_hi = min(cands), max(cands)
    else:
        r1_lo = r1_hi = np.nan
    return dict(N=N, d=d, tau=tau, m=m, K2=K2, Lam1=Lam1, Lam2=Lam2, W=W, X=X, kappa=kap, beta=beta, mu=mu,
                t_lo=t_lo, t_hi=t_hi, delta_lo=d_lo, delta_hi=d_hi, theta=th, r0_lo=r0_lo, r0_hi=r0_hi, r1_lo=r1_lo, r1_hi=r1_hi)


def main():
    rows = []
    allok = True
    for d, Ns in ((2, [2, 3, 4, 5, 6, 8, 10, 14, 20, 28, 35, 50, 70, 100, 140]), (3, [2, 3, 4, 5, 6, 8, 10, 14, 20, 28, 40])):
        for N in Ns:
            b = bounds(N, d)
            Lam, Wg, SW, s0, s1, r0, r1 = exact_poles(N, d)
            t = s0 * b["tau"]
            delta = s1 / b["Lam1"] - 1
            ok = (b["t_lo"] <= t * (1 + 1e-12) and t <= b["t_hi"] * (1 + 1e-12))
            okd = (np.isnan(b["delta_hi"]) or delta <= b["delta_hi"] * (1 + 1e-10)) and (np.isnan(b["delta_lo"]) or b["delta_lo"] <= delta * (1 + 1e-10))
            ok0 = b["r0_lo"] <= r0 * (1 + 1e-10) and r0 <= b["r0_hi"] * (1 + 1e-10)
            ok1 = np.isnan(b["r1_lo"]) or (b["r1_lo"] <= r1 + 1e-10 and r1 <= b["r1_hi"] + 1e-10)
            allok &= ok and okd and ok0 and ok1
            b.update(sigma0=s0, sigma1=s1, r0=r0, r1=r1, t=t, delta=delta, sigma0_m=s0 * b["m"], ok=bool(ok and okd and ok0 and ok1))
            rows.append(b)
            print(f"d={d} N={N:4d} X={b['X']:.3f} kap={b['kappa']:.4f} | s0*tau={t:.6f} in [{b['t_lo']:.6f},{b['t_hi']:.6f}] | "
                  f"delta={delta:.5f} in [{b['delta_lo']:.5f},{b['delta_hi']:.5f}] | r0={r0:.5f} in [{b['r0_lo']:.5f},{b['r0_hi']:.5f}] | "
                  f"r1={r1:.5f} in [{b['r1_lo']:.5f},{b['r1_hi']:.5f}] | s0*m={s0*b['m']:.5f} | ok={ok and okd and ok0 and ok1}")
    print("ALL OK:", allok)
    json.dump({"all_ok": bool(allok), "rows": rows}, open(_os.path.join(_R, 'data', 'msc_rigorous_spectral', '03_general_bounds_check.json'), "w"), indent=1,
              default=float)


if __name__ == "__main__":
    main()
