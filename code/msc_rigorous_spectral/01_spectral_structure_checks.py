"""Sanity checks for Theorems 1-3 (spectral representation, interlacing, residues).

Brute force: build L and the killed generator L_a, diagonalise, and compare with
the zeros of the secular function G_aa(s) = sum_k phi_k(a)^2/(lambda_k - s).
Run:  python 01_spectral_structure_checks.py   (writes ../data/01_spectral_structure_checks.json)
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import itertools, json, os, sys
import numpy as np
from scipy.optimize import brentq
from scipy.linalg import expm
sys.path.insert(0, os.path.dirname(__file__))
from t2lib import build_L, site_index

OUT = _os.path.join(_R, 'data', 'msc_rigorous_spectral', '01_spectral_structure_checks.json')
TOL = 1e-9


def graph_dist(N, d, x, a):
    return sum(abs(xi - ai) for xi, ai in zip(x, a))


def check_case(N, d, a, x0, q=0.8):
    n = N ** d
    L = build_L(N, d)
    ia, ix = site_index(a, N, d), site_index(x0, N, d)
    lam, Phi = np.linalg.eigh(L)
    # distinct eigenvalues and projector weights at a
    groups = []
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(lam[j + 1] - lam[i]) < 1e-10:
            j += 1
        groups.append((lam[i:j + 1].mean(), np.arange(i, j + 1)))
        i = j + 1
    Lam = np.array([g[0] for g in groups])
    mult = np.array([len(g[1]) for g in groups])
    w = np.array([np.sum(Phi[ia, g[1]] ** 2) for g in groups])        # ||E_i e_a||^2
    wx = np.array([np.sum(Phi[ia, g[1]] * Phi[ix, g[1]]) for g in groups])  # <E_i e_a, e_x>
    vis = w > 1e-12
    LamV, wV, wxV = Lam[vis], w[vis], wx[vis]
    p = len(LamV) - 1
    G = lambda s: np.sum(wV / (LamV - s))
    Gx = lambda s: np.sum(wxV / (LamV - s))
    dG = lambda s: np.sum(wV / (LamV - s) ** 2)
    sig = []
    for i in range(p):
        gap = LamV[i + 1] - LamV[i]
        lo, hi = LamV[i] + 1e-13 * max(1, gap) , LamV[i + 1] - 1e-13 * max(1, gap)
        # shrink until signs are opposite (G -> -inf at left pole, +inf at right pole)
        e = 1e-12 * gap
        while G(LamV[i] + e) > 0: e /= 4
        lo = LamV[i] + e
        e = 1e-12 * gap
        while G(LamV[i + 1] - e) < 0: e /= 4
        hi = LamV[i + 1] - e
        sig.append(brentq(G, lo, hi, xtol=1e-15, rtol=1e-15, maxiter=500))
    sig = np.array(sig)
    # predicted spectrum of L_a
    pred = list(sig)
    for lam_i, m_i, v in zip(Lam, mult, vis):
        pred += [lam_i] * (m_i - 1 if v else m_i)
    pred = np.sort(np.array(pred))
    keep = [i for i in range(n) if i != ia]
    La = L[np.ix_(keep, keep)]
    mu, Psi = np.linalg.eigh(La)
    res = {}
    res["n"] = n
    res["p_visible_poles"] = int(p)
    res["count_ok"] = bool(len(pred) == n - 1)
    res["spec_err"] = float(np.max(np.abs(pred - mu))) if len(pred) == n - 1 else None
    res["strict_interlace"] = bool(np.all(sig > LamV[:-1]) and np.all(sig < LamV[1:]))
    res["min_interlace_margin_rel"] = float(np.min(np.minimum(sig - LamV[:-1], LamV[1:] - sig) / (LamV[1:] - LamV[:-1])))
    # residues: formula vs projector
    r_formula = np.array([-Gx(s) / (s * dG(s)) for s in sig])
    kx = keep.index(ix)
    one = np.ones(n - 1)
    r_proj = []
    for s in sig:
        sel = np.abs(mu - s) < 1e-8
        Pj = Psi[:, sel] @ Psi[:, sel].T
        r_proj.append(Pj[kx, :] @ one)
    r_proj = np.array(r_proj)
    res["residue_err"] = float(np.max(np.abs(r_formula - r_proj)))
    # total weight of e_x^T Pi 1 on eigenvalues that are NOT zeros of G must vanish
    tot = (Psi[kx, :] * (Psi.T @ one))
    notzero = np.array([np.min(np.abs(sig - m)) > 1e-8 for m in mu])
    res["weight_off_secular_zeros"] = float(np.max(np.abs(tot[notzero]))) if notzero.any() else 0.0
    res["sum_r_minus_1"] = float(np.sum(r_formula) - 1)
    res["r0"] = float(r_formula[0])
    res["sigma0_is_min_eig"] = bool(abs(sig[0] - mu[0]) < 1e-10 and (len(mu) < 2 or (mu[1] - mu[0]) > 1e-10))
    # MFPT and second moment identities
    m_direct = np.linalg.solve(La, one)[kx]
    res["mfpt_err_rel"] = float(abs(np.sum(r_formula / sig) - m_direct) / m_direct)
    # time-domain checks
    tt = [0.3 / sig[0], 1.0 / sig[0]]
    ferr = 0.0
    b = La @ one
    for t in tt:
        f_direct = (expm(-t * La) @ b)[kx]
        f_spec = np.sum(r_formula * sig * np.exp(-sig * t))
        ferr = max(ferr, abs(f_direct - f_spec) / abs(f_direct))
    res["cont_density_err_rel"] = float(ferr)
    Q = np.eye(n - 1) - q * La
    v = one.copy(); S_prev = 1.0; derr = 0.0
    nu = 1 - q * sig
    for t in range(1, 60):
        v = Q @ v
        S = v[kx]
        f_direct = S_prev - S
        f_spec = np.sum(r_formula * (1 - nu) * nu ** (t - 1))
        derr = max(derr, abs(f_direct - f_spec))
        S_prev = S
    res["disc_pmf_err_abs"] = float(derr)
    # vanishing moments and sign changes
    D = graph_dist(N, d, x0, a)
    res["D"] = int(D)
    scale = sig.max()
    mom = [float(np.sum(r_formula * (sig / scale) ** i)) for i in range(1, D)]
    res["max_abs_vanishing_moment_scaled"] = float(max(abs(m_) for m_ in mom)) if mom else 0.0
    momD = float(np.sum(r_formula * sig ** D))
    res["sign_moment_D_times_(-1)^(D-1)"] = float(np.sign(momD) * (-1) ** (D - 1))
    rs = r_formula[np.abs(r_formula) > 1e-11]
    res["n_nonzero_res"] = int(len(rs))
    res["sign_changes"] = int(np.sum(rs[1:] * rs[:-1] < 0))
    res["sign_changes_ge_D_minus_1"] = bool(res["sign_changes"] >= D - 1)
    res["r1"] = float(r_formula[1]) if p > 1 else None
    return res


def main():
    cases = []
    # 1D end-to-end and interior targets
    for N in (2, 3, 5, 8, 12):
        cases.append((N, 1, (N,), (1,)))
    cases.append((9, 1, (5,), (1,)))    # centre target: half the modes invisible... (disconnected complement!)
    # 2D: corner-corner, centre target, generic target, accidental degeneracy N = 6
    for N in (2, 3, 4, 5, 6, 7, 8, 10, 12):
        cases.append((N, 2, (N, N), (1, 1)))
    cases += [(5, 2, (3, 3), (1, 1)), (7, 2, (4, 4), (1, 1)), (6, 2, (2, 5), (1, 1)), (6, 2, (6, 6), (3, 2)),
              (9, 2, (5, 5), (1, 1)), (9, 2, (2, 5), (9, 9)), (6, 2, (3, 3), (6, 6))]
    # 3D
    for N in (2, 3, 4, 5, 6):
        cases.append((N, 3, (N, N, N), (1, 1, 1)))
    cases += [(5, 3, (3, 3, 3), (1, 1, 1)), (4, 3, (2, 3, 4), (1, 1, 1))]
    out = []
    worst = {}
    for (N, d, a, x0) in cases:
        r = check_case(N, d, a, x0)
        r.update({"N": N, "d": d, "a": a, "x0": x0})
        out.append(r)
        print(f"N={N} d={d} a={a} x0={x0}: p={r['p_visible_poles']} spec_err={r['spec_err']:.1e} "
              f"interl={r['strict_interlace']} res_err={r['residue_err']:.1e} sum_r-1={r['sum_r_minus_1']:.1e} "
              f"r0={r['r0']:.4f} r1={r['r1']} D={r['D']} signchg={r['sign_changes']}/{r['n_nonzero_res']-1} "
              f"mfpt_err={r['mfpt_err_rel']:.1e} dens_err={r['cont_density_err_rel']:.1e} pmf_err={r['disc_pmf_err_abs']:.1e} "
              f"vanmom={r['max_abs_vanishing_moment_scaled']:.1e} offzero={r['weight_off_secular_zeros']:.1e}")
    ok = all(r["count_ok"] and r["spec_err"] < 1e-8 and r["strict_interlace"] and r["residue_err"] < 1e-7
             and abs(r["sum_r_minus_1"]) < 1e-8 and r["r0"] > 0 and r["sign_changes_ge_D_minus_1"] for r in out)
    print("ALL OK:", ok)
    with open(OUT, "w") as f:
        json.dump({"all_ok": ok, "cases": out}, f, indent=1)


if __name__ == "__main__":
    main()
