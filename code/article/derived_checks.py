#!/usr/bin/env python
"""Derived numbers quoted in the article -- each block names the statement it supports.

  A. Supplementary Section S9.1: the "resistance factor" (G_aa - G_oa) relative to the clean lattice when ONE site is blocked,
     from the single-defect maps  (T_x / T_clean) * (n_clean / n_x),  N = 15, 25, 35.
  B. Supplementary Section S2.2: parity branches of the chain at q = 1 (maxima of f over even t and over odd t), N = 51, 201.
  C. Supplementary Section S2.4: runs of the second time-stepping code with q <= 0.9, by geometry.
  D. Supplementary Section S9.7 / Table appC_tables:tab-deterministic: periodic arrays, all phases.
  E. Section 8.3 and Supplementary Section S9.3: measured conductivity and diffusivity of the site-diluted lattice against the published
     low-density laws (Watson and Leath 1974: sigma/sigma_0 = 1 - pi p + pi p^2/2, effective-medium theory;
     Ernst, Nieuwenhuizen and van Velthoven 1987: exact expansion 1 - pi p + 1.2858 p^2 and
     D/D_0 = 1 - (pi - 1) p - 0.8558 p^2).
  G. Shape family: extent of the configurations of sample A (ratio and coefficient of variation).
  H. Section 8.3 and Supplementary Section S9.3: the three estimates of the time-dilation factor Theta(p).
  I. Section 5.1 and Supplementary Section S6.1: the three explicit bounds on the continuum density Phi used in Proposition s5_modes:prop-1d-limit,
     and the convergence of the rescaled lattice densities to Phi.

Inputs : data/msc_defects, code/msc_defects_verify/data, code/msc_modes_verify/data
Output : data/article/derived_checks.json
Run    : python derived_checks.py        (a few seconds)
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import csv
import glob
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = _R            # 
DD = _os.path.join(_R, 'data', 'msc_defects')
DV = _os.path.join(_R, 'data', 'msc_defects_verify')
MV = _os.path.join(_R, 'data', 'msc_modes_verify')
OUT = _os.path.join(_R, 'data', 'article', 'derived_checks.json')
out = {"generated_by": "code/article/derived_checks.py"}


def read_csv(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


# ----------------------------------------------------------------------------- A. resistance factor, one blocked site
A = {}
for N in (15, 25, 35):
    rows = read_csv(os.path.join(DD, "sensitivity_N%d.csv" % N))
    n0 = N * N
    fac = []
    for r in rows:
        i, j = int(r["i"]), int(r["j"])
        if i < 0:
            continue
        chi = float(r["chi_mfpt"])
        # T = (4 n / q)(G_aa - G_oa):  (G_aa - G_oa)_x / (G_aa - G_oa)_clean = (T_x / T_clean) * n0 / (n0 - 1)
        fac.append(((1.0 + chi) * n0 / (n0 - 1.0), chi, i, j))
    fac.sort()
    below = [f for f in fac if f[0] < 1.0]
    A[str(N)] = {
        "n_sites": len(fac), "n_sites_with_resistance_factor_below_1": len(below),
        "min_resistance_factor": fac[0][0], "min_at_site": [fac[0][2], fac[0][3]], "chi_T_there": fac[0][1],
        "volume_term_minus_1_over_n": -1.0 / n0,
        "site_1_1": next({"resistance_factor": f[0], "chi_T": f[1]} for f in fac if (f[2], f[3]) == (1, 1)),
        "max_resistance_factor": fac[-1][0],
        "sites_below_1": [[f[2], f[3], f[0]] for f in below],
        "sites_below_1_max_chebyshev_distance_from_start": max(max(f[2], f[3]) for f in below),
    }
out["A_resistance_factor_single_blocked_site"] = A
# three-site example: path  a - o - x  (target a, start o, pendant x), unit conductances, pinv by hand
L3 = np.array([[1.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 1.0]])   # order a, o, x
G3 = np.linalg.pinv(L3)
L2 = np.array([[1.0, -1.0], [-1.0, 1.0]])
G2 = np.linalg.pinv(L2)
out["A_three_site_example"] = {"G_aa_minus_G_oa_with_pendant_site": float(G3[0, 0] - G3[1, 0]),
                               "G_aa_minus_G_oa_pendant_blocked": float(G2[0, 0] - G2[1, 0])}

# ----------------------------------------------------------------------------- B. parity branches at q = 1, chain
B = {}
for N in (51, 201):
    om = (2 * np.arange(1, N) - 1) * np.pi / (2 * N - 1)
    lam = np.cos(om)                                                    # q = 1
    w = 2.0 / (2 * N - 1) * (-1.0) ** (np.arange(1, N) + 1) * np.cos(om / 2) * np.sin(om)
    tmax = int(1.2 * N * N)
    t = np.arange(1, tmax + 1)
    # f(t) = sum_m w_m lam_m^(t-1); evaluate by blocks to limit memory
    f = np.zeros(tmax + 1)
    for a in range(0, tmax, 4000):
        tt = t[a:a + 4000]
        f[tt] = (w[None, :] * lam[None, :] ** (tt[:, None] - 1)).sum(axis=1)
    fe, fo = f.copy(), f.copy()
    fe[1::2] = -1.0
    fo[0::2] = -1.0
    te, to = int(np.argmax(fe)), int(np.argmax(fo))
    two = 0.5 * (f[1:-1] + f[2:])
    B[str(N)] = {"argmax_all_t": int(np.argmax(f)), "argmax_even_t": te, "argmax_odd_t": to,
                 "max_even": float(f[te]), "max_odd": float(f[to]),
                 "branch_maxima_rel_difference": float(abs(f[te] - f[to]) / max(f[te], f[to])),
                 "larger_branch_maximum_over_smaller_minus_1": float(max(f[te], f[to]) / min(f[te], f[to]) - 1.0),
                 "argmax_two_step_average_first_index": int(np.argmax(two)) + 1}
out["B_parity_branches_chain_q1"] = B

# ----------------------------------------------------------------------------- C. second stepping code, by geometry
rows = [json.loads(l) for l in open(_os.path.join(_R, 'data', 'msc_modes_verify', 'discrete.jsonl')) if l.strip()]
sel = [r for r in rows if r["q"] <= 0.9]
C = {"n_runs_q_le_0.9": len(sel), "by_geometry": {}, "by_geometry_and_d": {}}
for r in sel:
    C["by_geometry"][r["geom"]] = C["by_geometry"].get(r["geom"], 0) + 1
    k = "%s_d%d" % (r["geom"], r["d"])
    C["by_geometry_and_d"][k] = C["by_geometry_and_d"].get(k, 0) + 1
pt = [r for r in sel if r["geom"] != "EXIT"]
C["n_point_target_q_le_0.9"] = len(pt)
C["n_exit_q_le_0.9"] = len(sel) - len(pt)
C["point_target_all_one_local_max"] = bool(all(r["n_local_max"] == 1 for r in pt))
C["point_target_all_monotone_after_mode"] = bool(all(r["monotone_after_mode"] for r in pt))
C["exit_all_one_local_max"] = bool(all(r["n_local_max"] == 1 for r in sel if r["geom"] == "EXIT"))
C["exit_all_monotone_after_mode"] = bool(all(r["monotone_after_mode"] for r in sel if r["geom"] == "EXIT"))
out["C_second_stepper_runs"] = C

# ----------------------------------------------------------------------------- D. periodic arrays, all phases
per = read_csv(_os.path.join(_R, 'data', 'msc_defects_verify', 'v06_periodic.csv'))
det = {r["name"]: r for r in read_csv(_os.path.join(_R, 'data', 'msc_defects', 'structured_deterministic.csv'))}
D = {}
for a in (2, 3, 4):
    ph = [r for r in per if int(r["a"]) == a]
    mf = np.array([float(r["mfpt_fac"]) for r in ph])
    mo = np.array([float(r["mode_fac"]) for r in ph])
    tab = det["periodic_a%d" % a]
    tab_m = float(tab["mfpt"]) / float(det["clean"]["mfpt"])
    k = int(np.argmin(np.abs(mf - tab_m)))
    D[str(a)] = {
        "n_phases": len(ph),
        "mean_factor_min": float(mf.min()), "mean_factor_max": float(mf.max()), "mean_factor_average": float(mf.mean()),
        "mode_factor_min": float(mo.min()), "mode_factor_max": float(mo.max()), "mode_factor_average": float(mo.mean()),
        "tabulated_phase_offset": [int(ph[k]["oi"]), int(ph[k]["oj"])], "tabulated_mean_factor": tab_m,
        "tabulated_blocked_sites": int(float(tab["n_blocked"])) if "n_blocked" in tab else int(ph[k]["blocked"]),
        "blocked_sites_by_phase_min_max": [int(min(int(r["blocked"]) for r in ph)), int(max(int(r["blocked"]) for r in ph))],
        "tabulated_is_smallest_mean_factor": bool(abs(tab_m - mf.min()) < 1e-9),
        "tabulated_is_smallest_mode_factor": bool(abs(float(ph[k]["mode_fac"]) - mo.min()) < 1e-9),
        "rank_of_tabulated_mean_factor_from_smallest": int(np.sum(mf < tab_m - 1e-12)) + 1,
        "phases": [{"offset": [int(r["oi"]), int(r["oj"])], "blocked": int(r["blocked"]),
                    "mean_factor": float(r["mfpt_fac"]), "mode_factor": float(r["mode_fac"]),
                    "target_neighbour_blocked": int(r["tgt_nbr_blocked"])} for r in ph],
    }
out["D_periodic_phases"] = D

# ----------------------------------------------------------------------------- E. dilute laws
hom = json.load(open(_os.path.join(_R, 'data', 'msc_defects', 'homogenization.json')))
E = []
ENVE_SIGMA2, ENVE_D2 = 1.2858, -0.8558
for r in hom["finite_p"]:
    p = r["p"]
    if p <= 0 or p > 0.2001:
        continue
    sig, ci = r["sigma"], 1.96 * r["sigma_sem"]
    D_ = r["D_ratio"]
    D_ci = ci / r["P_inf"]
    lin_D = 1.0 - (math.pi - 1.0) * p
    lin_s = 1.0 - math.pi * p
    emt_s = 1.0 - math.pi * p + math.pi * p * p / 2.0
    # exact low-density expansion of Ernst, Nieuwenhuizen and van Velthoven, J. Phys. A 20, 5335 (1987),
    # their Eqs (4.2) and (4.3) and Table 2 (row b = 1):  Sigma/Sigma_0 = 1 - pi c + 1.2858 c^2 + ...,
    # D/D_0 = 1 - (pi - 1) c - 0.8558 c^2 + ...
    enve_s = 1.0 - math.pi * p + ENVE_SIGMA2 * p * p
    enve_D = 1.0 - (math.pi - 1.0) * p + ENVE_D2 * p * p
    E.append({"p": p, "sigma": sig, "sigma_ci95": ci, "P_inf": r["P_inf"], "D_ratio": D_, "D_ci95": D_ci,
              "sigma_Watson_Leath_EMT": emt_s, "sigma_minus_EMT": sig - emt_s,
              "sigma_second_order_coeff": (sig - lin_s) / p ** 2, "sigma_second_order_coeff_ci95": ci / p ** 2,
              "D_minus_linear": D_ - lin_D, "D_second_order_coeff": (D_ - lin_D) / p ** 2,
              "D_second_order_coeff_ci95": D_ci / p ** 2,
              "sigma_exact_to_second_order_ENvV1987": enve_s, "sigma_minus_exact_second_order": sig - enve_s,
              "sigma_implied_third_order_coeff": (sig - enve_s) / p ** 3,
              "sigma_implied_third_order_coeff_ci95": ci / p ** 3,
              "D_exact_to_second_order_ENvV1987": enve_D, "D_minus_exact_second_order": D_ - enve_D,
              "D_implied_third_order_coeff": (D_ - enve_D) / p ** 3,
              "D_implied_third_order_coeff_ci95": D_ci / p ** 3})
# inverse-variance weighted mean of the measured second-order coefficient of sigma over 0.06 <= p <= 0.14
sel = [r_ for r_ in E if 0.0599 <= r_["p"] <= 0.1401]
wts = [1.0 / (r_["sigma_second_order_coeff_ci95"] / 1.96) ** 2 for r_ in sel]
wm = sum(w * r_["sigma_second_order_coeff"] for w, r_ in zip(wts, sel)) / sum(wts)
wci = 1.96 / math.sqrt(sum(wts))
out["E_dilute_laws"] = {"rows": E, "EMT_second_order_coeff_sigma": math.pi / 2.0,
                        "exact_second_order_coeff_sigma_ENvV1987": ENVE_SIGMA2,
                        "exact_second_order_coeff_D_ENvV1987": ENVE_D2,
                        "consistency_(1-c)*D_gives_sigma_coeff": math.pi - 1.0 + ENVE_D2,
                        "sigma_second_order_coeff_weighted_mean_p0.06_to_0.14": wm,
                        "sigma_second_order_coeff_weighted_mean_ci95": wci,
                        "max_abs_sigma_minus_exact_second_order_p_le_0.10": max(
                            abs(r_["sigma_minus_exact_second_order"]) for r_ in E if r_["p"] <= 0.1001),
                        "max_abs_sigma_minus_exact_second_order_p_le_0.18": max(
                            abs(r_["sigma_minus_exact_second_order"]) for r_ in E if r_["p"] <= 0.1801),
                        "n_densities_p_le_0.18_with_sigma_within_ci95_of_exact_second_order": sum(
                            1 for r_ in E if r_["p"] <= 0.1801
                            and abs(r_["sigma_minus_exact_second_order"]) <= r_["sigma_ci95"]),
                        "n_densities_p_le_0.18": sum(1 for r_ in E if r_["p"] <= 0.1801),
                        "max_abs_D_minus_exact_second_order_p_le_0.10": max(
                            abs(r_["D_minus_exact_second_order"]) for r_ in E if r_["p"] <= 0.1001),
                        "source_of_exact_coefficients": "Ernst, Nieuwenhuizen and van Velthoven, J. Phys. A 20, 5335 "
                                                        "(1987), Eqs (4.2), (4.3), Table 2 (b = 1): 1.2858, -0.8558",
                        "note": "sigma: 192 x 192 tori, 12 placements; ci95 = 1.96 s.e.m."}

# ----------------------------------------------------------------------------- G. shape family: extent of the data
pts = []
for f in sorted(glob.glob(_os.path.join(_R, 'data', 'msc_defects', 'sweep', '*.csv'))):
    for r in read_csv(f):
        pts.append((float(r["ratio"]), float(r["cv"]), "sweep", float(r["p"])))
for r in read_csv(_os.path.join(_R, 'data', 'msc_defects', 'structured_local.csv')):
    pts.append((float(r["ratio"]), float(r["cv"]), "local", 0.0))
for r in read_csv(_os.path.join(_R, 'data', 'msc_defects', 'structured_deterministic.csv')):
    pts.append((float(r["ratio"]), float(r["cv"]), "det", 0.0))
P = np.array([(a, b) for a, b, _, _ in pts])
outside = [(a, b, k, p) for a, b, k, p in pts if not (0.04 <= a <= 0.36 and 0.80 <= b <= 0.985)]
out["G_shape_family_extent"] = {
    "n_configurations": len(pts), "ratio_min": float(P[:, 0].min()), "ratio_max": float(P[:, 0].max()),
    "cv_min": float(P[:, 1].min()), "cv_max": float(P[:, 1].max()),
    "n_outside_frame_0.04-0.36_x_0.80-0.985": len(outside),
    "outside_by_kind": {k: sum(1 for o in outside if o[2] == k) for k in ("sweep", "local", "det")},
    "outside_sweep_min_p": min((o[3] for o in outside if o[2] == "sweep"), default=None),
}

# ----------------------------------------------------------------------------- H. three estimates of Theta
v4 = json.load(open(_os.path.join(_R, 'data', 'msc_defects_verify', 'v04_homog.json')))["random"]
v4b = json.load(open(_os.path.join(_R, 'data', 'msc_defects_verify', 'v04b_theta.json')))
H = {}
for p in (0.1, 0.2):
    r192 = next(r for r in hom["finite_p"] if abs(r["p"] - p) < 1e-9)
    r160 = v4["L160_p%.2f" % p] if ("L160_p%.2f" % p) in v4 else v4["L160_p%.1f" % p]
    r200 = v4b[str(p)]
    H[str(p)] = {"Theta_192_12_placements": r192["time_factor"],
                 "Theta_160_10_placements": r160["Theta"], "Theta_160_ci95": r160["Theta_ci"],
                 "Theta_200_40_placements": r200["Theta"], "Theta_200_ci95": r200["Theta_ci"]}
out["H_three_estimates_of_Theta"] = H

# ----------------------------------------------------------------------------- I. bounds used in Proposition s5_modes:prop-1d-limit
import mpmath as mp
mp.mp.dps = 30


def Phi(xi, terms=60):
    return mp.pi * mp.fsum((-1) ** (m + 1) * (2 * m - 1) * mp.exp(-(2 * m - 1) ** 2 * mp.pi ** 2 * xi / 4)
                           for m in range(1, terms + 1))


u = lambda m, xi: (2 * m - 1) * mp.exp(-(2 * m - 1) ** 2 * mp.pi ** 2 * xi / 4)
xs = mp.findroot(lambda x: mp.diff(Phi, x), mp.mpf(1) / 6)
out["I_continuum_density_bounds"] = {
    "Phi(0.05)": float(Phi(mp.mpf("0.05"))), "upper_bound_three_terms_at_0.05": float(mp.pi * (u(1, 0.05) - u(2, 0.05) + u(3, 0.05))),
    "Phi(1/6)": float(Phi(mp.mpf(1) / 6)), "lower_bound_two_terms_at_1/6": float(mp.pi * (u(1, mp.mpf(1) / 6) - u(2, mp.mpf(1) / 6))),
    "Phi(1/2)": float(Phi(mp.mpf(1) / 2)), "upper_bound_one_term_at_1/2": float(mp.pi * u(1, mp.mpf(1) / 2)),
    "term_ratio_u3_over_u2_at_0.05": float(u(3, 0.05) / u(2, 0.05)), "term_ratio_u2_over_u1_at_0.05": float(u(2, 0.05) / u(1, 0.05)),
    "term_ratio_u2_over_u1_at_1/2": float(u(2, 0.5) / u(1, 0.5)),
    "xi_star": mp.nstr(xs, 15), "two_xi_star": mp.nstr(2 * xs, 12), "Phi(xi_star)": float(Phi(xs)),
    "statement": "Phi(0.05) < 0.40 < 1.84 < Phi(1/6) and Phi(1/2) < 0.92",
}
# rescaled lattice densities against Phi: sup over xi in [0.05, 1] of |Phi_N - Phi| (continuous time, unit rate)
conv = {}
for N in (10, 100, 1000):
    l = N - 0.5
    m = np.arange(1, N)
    om = (2 * m - 1) * np.pi / (2 * l)
    c = 2 * l * (-1.0) ** (m + 1) * np.cos(om / 2) * np.sin(om)
    a = 2 * l * l * (1 - np.cos(om))
    xi = np.linspace(0.05, 1.0, 400)
    PhiN = (c[None, :] * np.exp(-a[None, :] * xi[:, None])).sum(axis=1)
    Ph = np.array([float(Phi(mp.mpf(float(x)))) for x in xi])
    conv[str(N)] = {"sup_abs_diff_on_[0.05,1]": float(np.abs(PhiN - Ph).max()),
                    "min_a_over_(2m-1)^2": float((a / (2 * m - 1) ** 2).min()),
                    "max_abs_c_over_(2m-1)pi": float((np.abs(c) / ((2 * m - 1) * np.pi)).max())}
out["I_continuum_density_bounds"]["convergence_of_rescaled_density"] = conv

json.dump(out, open(OUT, "w"), indent=1)
for k in out:
    if k != "generated_by":
        s = json.dumps(out[k])
        print(k, s[:700] + (" ..." if len(s) > 700 else ""))
