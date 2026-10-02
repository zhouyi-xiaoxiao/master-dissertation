"""Numerical check of Theorem 5.6 and Remark 5.8 (sign rule for a reflection-symmetric start-target pair).

(a) r_j(x0) = -2 G^+(sigma_j) / (sigma_j G'(sigma_j)) = 2 G^-(sigma_j)/(sigma_j G'(sigma_j)),  G = G^+ + G^-;
(b) the smallest eigenvalue eta_1 of the walk with BOTH corners absorbing satisfies eta_1 <= Lam_1 and G^+(eta_1) = 0;
(c) sign r_j = - sign G^+(sigma_j) for every pole (brute force residues from the eigenvectors of L_a).
Corner-to-corner, d = 2, 3.  Output: ../data/12_sign_rule_check.json
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import os, sys, json
import numpy as np
from scipy.optimize import brentq
sys.path.insert(0, os.path.dirname(__file__))
from t2lib import build_L, site_index, corner_spectral_data, group_visible

rows = []; allok = True
for (N, d) in ((3, 2), (4, 2), (5, 2), (6, 2), (8, 2), (11, 2), (3, 3), (4, 3), (5, 3)):
    n = N ** d
    L = build_L(N, d)
    a = site_index((N,) * d, N, d); x0 = site_index((1,) * d, N, d)
    lam, w, sg = corner_spectral_data(N, d)
    Lam1 = (1 - np.cos(np.pi / N)) / d
    even = sg > 0; odd = ~even
    # G^+ = <v+,(L-s)^-1 v+>, v+ = (e_a+e_x0)/2:  <v+,E_k v+> = phi_k(a)^2 (1+(-1)^{|k|})^2/4 = phi_k(a)^2 for even |k|, 0 for odd |k|
    Gp = lambda s: np.sum(w[even] / (lam[even] - s)) / n
    Gm = lambda s: np.sum(w[odd] / (lam[odd] - s)) / n
    G = lambda s: np.sum(w / (lam - s)) / n
    dG = lambda s: np.sum(w / (lam - s) ** 2) / n
    # brute force residues
    keep = [i for i in range(n) if i != a]
    La = L[np.ix_(keep, keep)]
    mu, Psi = np.linalg.eigh(La)
    kx = keep.index(x0)
    coef = Psi[kx, :] * (Psi.T @ np.ones(n - 1))
    # group by distinct eigenvalue
    res = []
    i = 0
    while i < n - 1:
        j = i
        while j + 1 < n - 1 and abs(mu[j + 1] - mu[i]) < 1e-10: j += 1
        res.append((mu[i:j + 1].mean(), coef[i:j + 1].sum())); i = j + 1
    poles = [(s, r) for s, r in res if abs(r) > 1e-10]
    err_a = 0.0; sign_ok = True
    Lg, Wg = group_visible(lam, w)
    for s, r in poles:
        # refine the pole as a zero of G between the neighbouring visible eigenvalues
        idx = np.searchsorted(Lg, s)
        lo, hi = Lg[idx - 1], Lg[idx]
        e = 1e-11 * (hi - lo)
        try:
            s_ref = brentq(G, lo + e, hi - e, xtol=1e-300, rtol=1e-15)
        except ValueError:
            s_ref = s
        ra = -2 * Gp(s_ref) / (s_ref * dG(s_ref)); rb = 2 * Gm(s_ref) / (s_ref * dG(s_ref))
        err_a = max(err_a, abs(ra - r) / max(abs(r), 1e-12), abs(rb - r) / max(abs(r), 1e-12))
        sign_ok &= (np.sign(r) == -np.sign(Gp(s_ref)))
    # (b)
    keep2 = [i for i in range(n) if i not in (a, x0)]
    eta1 = np.linalg.eigvalsh(L[np.ix_(keep2, keep2)])[0]
    ok_b = eta1 <= Lam1 * (1 + 1e-12) and abs(Gp(eta1)) < 1e-8 * abs(Gp(eta1 * 0.5))
    neg = sum(1 for s, r in poles if r < 0)
    rows.append(dict(N=N, d=d, n_poles=len(poles), n_negative=neg, err_formula=err_a, sign_rule=bool(sign_ok), eta1_over_Lam1=eta1 / Lam1, ok_b=bool(ok_b)))
    allok &= err_a < 1e-6 and sign_ok and ok_b
    print(f"N={N} d={d}: poles with weight {len(poles)}, negative residues {neg}, max rel err of (5.4) {err_a:.1e}, sign rule {sign_ok}, "
          f"eta_1/Lam_1 = {eta1/Lam1:.6f} (<=1), G+(eta_1)=0: {ok_b}")
print("ALL OK:", allok)
json.dump({"all_ok": bool(allok), "rows": rows}, open(_os.path.join(_R, 'data', 'msc_rigorous_spectral', '12_sign_rule_check.json'), "w"), indent=1, default=float)
