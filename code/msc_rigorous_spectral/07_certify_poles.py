"""Certified enclosures (Arb ball arithmetic) of sigma_0, sigma_1, r_0, r_1, tau, m for the corner-to-corner
geometry in d = 2, 3, and a certified check of the inequalities of Theorems 6.2-6.8 (general bounds).

Certification of a pole: G is strictly increasing between consecutive poles (Theorem 2.2(i)), so if
Lam_j < s_lo < s_hi < Lam_{j+1} and G(s_lo) < 0 < G(s_hi) (both verified in ball arithmetic), then the unique
zero sigma_j of G in (Lam_j, Lam_{j+1}) lies in (s_lo, s_hi).

Output: ../data/07_certified_poles.json (+ table printed).
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import os, sys, json, math, itertools
from collections import Counter
import numpy as np
from scipy.optimize import brentq
from flint import arb, ctx
sys.path.insert(0, os.path.dirname(__file__))
from t2lib import corner_spectral_data, group_visible
ctx.prec = 256
pi = arb.pi()


def spectral_arb(N, d):
    """list of (lambda_k, n*phi_k(a)^2 * multiplicity, sign) over sorted multi-indices."""
    c = [(pi * k / N).cos() for k in range(N)]
    w1 = [(1 + c[k]) / 2 * (1 if k == 0 else 2) for k in range(N)]       # eps_k cos^2(theta_k/2)
    out = []
    for k in itertools.combinations_with_replacement(range(N), d):
        cnt = Counter(k); mult = math.factorial(d)
        for v in cnt.values(): mult //= math.factorial(v)
        lam = sum((1 - c[j]) for j in k) / d
        w = arb(mult)
        for j in k: w *= w1[j]
        out.append((lam, w, (-1) ** sum(k), k))
    return out


def certify(N, d, eta_bits=150):
    data = spectral_arb(N, d)
    nz = [(l, w, s) for (l, w, s, k) in data if any(k)]
    tau = sum(w / l for l, w, s in nz)
    m = sum((1 - s) * w / l for l, w, s in nz)
    K2 = sum(w / (l * l) for l, w, s in nz)
    Lam1 = (1 - (pi / N).cos()) / d
    Lam2 = 2 * Lam1 if d >= 2 else (1 - (2 * pi / N).cos())
    W = 2 * d * (pi / (2 * N)).cos() ** 2
    def nG(s):   return -1 / s + sum(w / (l - s) for l, w, _ in nz)
    def nGx(s):  return -1 / s + sum(sg * w / (l - s) for l, w, sg in nz)
    def ndG(s):  return 1 / (s * s) + sum(w / (l - s) ** 2 for l, w, _ in nz)
    # floating-point starting values
    lamf, wf, sgf = corner_spectral_data(N, d)
    Lg, Wg, SW = group_visible(lamf, wf, sgf)
    Gf = lambda s: np.sum(Wg / (Lg - s))
    def froot(i):
        gap = Lg[i + 1] - Lg[i]
        e = 1e-9 * gap
        while Gf(Lg[i] + e) > 0: e /= 8
        lo = Lg[i] + e
        e = 1e-9 * gap
        while Gf(Lg[i + 1] - e) < 0: e /= 8
        return brentq(Gf, lo, Lg[i + 1] - e, xtol=1e-300, rtol=1e-15, maxiter=1000)
    res = {}
    brackets = [(arb(0), Lam1), (Lam1, Lam2)]
    for j in (0, 1):
        s = arb(froot(j))
        for _ in range(4):                       # Newton on midpoints (not rigorous; only produces a candidate)
            s = (s - nG(s) / ndG(s)).mid()
        eta = arb(2) ** (-eta_bits)
        s_lo, s_hi = (s * (1 - eta)).mid(), (s * (1 + eta)).mid()
        ok = (brackets[j][0] < s_lo) and (s_hi < brackets[j][1]) and (nG(s_lo) < 0) and (nG(s_hi) > 0)
        assert ok, f"certification failed N={N} d={d} j={j}"
        S = s_lo.union(s_hi)
        r = -nGx(S) / (S * ndG(S))
        res[f"sigma{j}"] = S; res[f"r{j}"] = r
    # ---- theorem inequalities (Section 6) -------------------------------------------------------------
    X = Lam1 * tau; kap = K2 / tau ** 2; mu = (m - tau) / tau
    chk = {}
    t = res["sigma0"] * tau
    chk["basic"] = bool(X / (1 + X) < t and t < 1)
    hyp_a = bool(X > 1)
    if hyp_a:
        beta = X / (X - 1)
        t_lo = 2 / (1 + (1 + 4 * beta * kap).sqrt()); t_hi = 2 / (1 + (1 + 4 * kap).sqrt())
        chk["sigma0"] = bool(t_lo < t and t < t_hi)
        r0_lo = (1 + t_lo * mu - beta * kap * t_hi ** 2) / (1 + beta ** 2 * kap * t_hi ** 2)
        r0_hi = (1 + t_hi * mu + beta * kap * t_hi ** 2) / (1 + kap * t_lo ** 2)
        chk["r0"] = bool(r0_lo < res["r0"] and res["r0"] < r0_hi)
    else:
        t_lo = t_hi = r0_lo = r0_hi = None
    chk["mu_range"] = bool(mu > 0 and Lam1 * (m - tau) < 1 + W.log())
    chk["sigma0_m"] = bool(X / (1 + X) < res["sigma0"] * m and res["sigma0"] * m < 1 + (1 + W.log()) / X)
    delta = res["sigma1"] / Lam1 - 1
    Y0 = X - W; th = Lam1 ** 2 * K2 - W; rho = Lam1 / Lam2
    d_lo = d_hi = r1_lo = r1_hi = None
    if bool(Y0 + th > 1):
        d_hi = W / (Y0 + th - 1)
        chk["delta_upper"] = bool(delta < d_hi)
        if bool((1 + d_hi) * rho < 1):
            om = (1 + d_hi) / (1 - (1 + d_hi) * rho)
            d_lo = W / (Y0 + om * th - 1 / (1 + d_hi))
            chk["delta_lower"] = bool(d_lo < delta)
            Yx0 = W - Lam1 * (m - tau)
            y_lo = Yx0 - om * th - 1; y_hi = Yx0 + om * th - 1 / (1 + d_hi)
            n_lo = W + (d_lo * y_lo).min(d_hi * y_lo)        # ball containing the min / max: rigorous
            n_hi = W + (d_lo * y_hi).max(d_hi * y_hi)
            D_lo = (1 + d_lo) * W + d_lo ** 2 * (1 / (1 + d_hi) + (1 + d_lo) * th)
            D_hi = (1 + d_hi) * W + d_hi ** 2 * (1 + om ** 2 * th / (1 + d_hi))
            if bool(n_lo > 0):
                r1_hi = -d_lo * n_lo / D_hi       # upper bound of r_1 (least negative)
                r1_lo = -d_hi * n_hi / D_lo
                chk["r1"] = bool(r1_lo < res["r1"] and res["r1"] < r1_hi)
    chk["r1_negative"] = bool(res["r1"] < 0)
    chk["r0_positive"] = bool(res["r0"] > 0)
    f = lambda v: None if v is None else float(v.mid())
    row = dict(N=N, d=d, tau=tau.str(25), m=m.str(25), K2=K2.str(20), X=f(X), kappa=f(kap), mu=f(mu),
               sigma0=res["sigma0"].str(30), sigma1=res["sigma1"].str(30), r0=res["r0"].str(25), r1=res["r1"].str(25),
               sigma0_tau=f(t), sigma0_m=f(res["sigma0"] * m), delta=f(delta),
               t_lo=f(t_lo), t_hi=f(t_hi), delta_lo=f(d_lo), delta_hi=f(d_hi), r0_lo=f(r0_lo), r0_hi=f(r0_hi), r1_lo=f(r1_lo), r1_hi=f(r1_hi),
               theta=f(th), L1sqK2=f(Lam1 ** 2 * K2), Lam1_m_minus_tau=f(Lam1 * (m - tau)), checks=chk,
               tau_f=f(tau), m_f=f(m))
    return row


def main():
    rows = []
    plan = [(2, [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16, 20, 25, 35, 50, 70, 100, 150, 200]),
            (3, [2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25, 30, 40])]
    allok = True
    for d, Ns in plan:
        for N in Ns:
            r = certify(N, d)
            rows.append(r)
            bad = [k for k, v in r["checks"].items() if not v]
            allok &= not bad
            print(f"d={d} N={N:4d} s0*tau={r['sigma0_tau']:.8f} s0*m={r['sigma0_m']:.8f} delta={r['delta']:.6f} "
                  f"r0={float(arb(r['r0']).mid()):.8f} r1={float(arb(r['r1']).mid()):.8f} X={r['X']:.4f} "
                  f"checks: {'ALL OK ('+str(len(r['checks']))+')' if not bad else 'FAILED '+str(bad)}", flush=True)
            json.dump({"rows": rows}, open(_os.path.join(_R, 'data', 'msc_rigorous_spectral', '07_certified_poles.json'), "w"), indent=1)
    print("ALL CERTIFIED CHECKS OK:", allok)


if __name__ == "__main__":
    main()
