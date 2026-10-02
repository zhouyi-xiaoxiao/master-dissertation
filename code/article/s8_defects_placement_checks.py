#!/usr/bin/env python
"""Derived numbers quoted in Section 8.4 and Supplementary Sections S9.5-S9.10 (placement, mechanism, other defect types).

Everything here is recomputed from the saved research results (no new first-passage solves except
the small exact checks of items 5 and 8):

 1. stratification by the number of open neighbours of the target (both samples);
 2. single-defect susceptibilities at N = 35: shares of the summed susceptibility inside the
    4 x 4 boxes that the corner-thinned placement law thins, first-order slopes with uniform and
    corner-thinned weights (weights rebuilt from the generator's definition), predicted
    d(ratio)/dp;
 3. localised placements: summary of sample B from its raw file, elasticities of the mode with
    respect to the mean for near-target ensembles (both samples);
 4. periodic arrays: range and mean of the MFPT and mode factors over all phases;
 5. bulk time factor of a periodic array, by an exact rational solve of the unit-cell problem,
    and the random-placement factor Theta(p) interpolated at the same densities;
 6. two-time-scale heuristic: maximiser, elasticity, share of the logarithmic shift of the mode
    that it reproduces, and the same shifts from the closed form of Section 6;
 7. shape family: quadratic fit of CV against mode/MFPT on sample A (all 13,631 configurations);
 8. chain with barriers: formula against a direct linear solve on random chains;
 9. permeable obstacles: distance of the ratio from the clean value, and the 1/(1-p) jump.

Inputs : data/msc_defects/*, data/msc_defects_verify/*
Output : ../data/s8_defects_placement_checks.json
Usage  : python s8_defects_placement_checks.py
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import glob
import json
import math
import os

import numpy as np
import pandas as pd
import sympy as sy
from scipy.interpolate import CubicSpline

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = _R            # 
A = _os.path.join(_R, 'data', 'msc_defects')               # sample A
B = _os.path.join(_R, 'data', 'msc_defects_verify')     # sample B
OUT = _os.path.join(_R, 'data', 'article', 's8_defects_placement_checks.json')

N, q = 35, 0.8
CLEAN_MFPT, CLEAN_MODE = 14100.808049917203, 2493
CLEAN_RATIO = CLEAN_MODE / CLEAN_MFPT


def ci95(x):
    x = np.asarray(x, float)
    return float(1.96 * x.std(ddof=1) / math.sqrt(len(x)))


out = {"clean": dict(mfpt=CLEAN_MFPT, mode=CLEAN_MODE, ratio=CLEAN_RATIO)}

# ------------------------------------------------------------------ 1. stratification
st = pd.read_csv(_os.path.join(_R, 'data', 'msc_defects', 'summary_stratified.csv'))
rows = {}
for (scheme, p), g in st.groupby(["scheme", "p"]):
    if p > 0.2001:
        continue
    one, two = g[g.deg_t == 1].iloc[0], g[g.deg_t == 2].iloc[0]
    rows[f"{scheme}_p{p:.1f}"] = dict(mfpt_increase=float(one.mfpt / two.mfpt - 1),
                                      mode_increase=float(one["mode"] / two["mode"] - 1),
                                      K_one=int(one.K), K_two=int(two.K))
out["stratified_sample_A"] = rows
vm = json.load(open(_os.path.join(_R, 'data', 'msc_defects_verify', 'v09_mechanism.json')))
rows = {}
for p in ("p0.10", "p0.20"):
    for sch in ("uniform", "cornerthinned"):
        one, two = vm[p][f"{sch}_deg1"], vm[p][f"{sch}_deg2"]
        rows[f"{sch}_{p}"] = dict(mfpt_increase=one["mfpt"] / two["mfpt"] - 1,
                                  mode_increase=one["mode"] / two["mode"] - 1,
                                  K_one=one["K"], K_two=two["K"])
out["stratified_sample_B"] = rows
out["spearman_mfpt_Gaa"] = dict(
    sample_A={f"{s}_p{p}": float(c) for s, p, c in
              [(r.scheme, r.p, r.spearman) for r in
               pd.read_csv(_os.path.join(_R, 'data', 'msc_defects', 'summary_correlations.csv')).query("y == 'mfpt' and x == 'G_tt'").itertuples()]},
    sample_B={"uniform_p0.1": vm["p0.10"]["G_aa"]["mfpt"], "uniform_p0.2": vm["p0.20"]["G_aa"]["mfpt"]})

# ------------------------------------------------------------------ 2. susceptibilities
sens = pd.read_csv(_os.path.join(_R, 'data', 'msc_defects', 'sensitivity_N35.csv')).iloc[1:]      # row 0 is the clean lattice
i, j = sens.i.to_numpy(), sens.j.to_numpy()
cm, cmo = sens.chi_mfpt.to_numpy(), sens.chi_mode.to_numpy()
radius = max(3, N // 8)                                                   # = 4, as in the corner-thinned placement law
in_t = (i >= N - radius) & (j >= N - radius)
in_s = (i < radius) & (j < radius)
w = np.ones((N, N))
for a in range(radius):
    for b in range(radius):
        w[a, b] *= math.exp(-2 * (radius - math.hypot(a, b)) / radius)
for a in range(N - radius, N):
    for b in range(N - radius, N):
        w[a, b] *= math.exp(-2 * (radius - math.hypot(N - 1 - a, N - 1 - b)) / radius)
w[0, 0] = w[N - 1, N - 1] = 0.0
wn = w / w.sum()
wu = np.ones((N, N)); wu[0, 0] = wu[N - 1, N - 1] = 0.0; wu /= wu.sum()
dist_t = np.hypot(N - 1 - i, N - 1 - j)
fo = json.load(open(_os.path.join(_R, 'data', 'msc_defects', 'summary_first_order.json')))
out["susceptibility_N35"] = dict(
    n_sites=int(len(sens)),
    frac_lower_mfpt=float((cm < 0).mean()), frac_lower_mode=float((cmo < 0).mean()),
    max_chi_mfpt=float(cm.max()), max_chi_mode=float(cmo.max()), min_chi_mfpt=float(cm.min()),
    minus_one_over_n=-1.0 / N ** 2,
    share_mfpt_within={r: float(cm[dist_t <= r].sum() / cm.sum()) for r in (3, 6, 10)},
    share_mode_within={r: float(cmo[dist_t <= r].sum() / cmo.sum()) for r in (3, 6, 10)},
    box_side=radius,
    share_mfpt_target_box=float(cm[in_t].sum() / cm.sum()), share_mode_target_box=float(cmo[in_t].sum() / cmo.sum()),
    share_mfpt_start_box=float(cm[in_s].sum() / cm.sum()), share_mode_start_box=float(cmo[in_s].sum() / cmo.sum()),
    thinned_weight_relative_to_uniform_in_target_box=float(wn[i[in_t], j[in_t]].mean() / wu[i[in_t], j[in_t]].mean()),
    slope_mfpt_uniform=float(N * N * (wu[i, j] * cm).sum()), slope_mode_uniform=float(N * N * (wu[i, j] * cmo).sum()),
    slope_mfpt_thinned=float(N * N * (wn[i, j] * cm).sum()), slope_mode_thinned=float(N * N * (wn[i, j] * cmo).sum()),
    pred_dlnratio_dp_uniform=fo["uniform"]["pred_slope_ratio"], pred_dlnratio_dp_thinned=fo["smart"]["pred_slope_ratio"],
    pred_dratio_dp_thinned=CLEAN_RATIO * fo["smart"]["pred_slope_ratio"],
    pred_dratio_dp_uniform=CLEAN_RATIO * fo["uniform"]["pred_slope_ratio"])

pe = round(0.02 * N * N) / N ** 2                                         # 25 blocked sites
vs = pd.read_csv(_os.path.join(_R, 'data', 'msc_defects_verify', 'v03_summary.csv'))
sec = {}
for name_b, name in (("uniform", "uniform"), ("cornerthinned", "thinned")):
    r = vs[(vs.scheme == name_b) & np.isclose(vs.p, 0.02)].iloc[0]
    sec[name] = dict(mfpt=float((r.mfpt_fac - 1) / pe), mfpt_ci=float(r.mfpt_fac_ci / pe),
                     mode=float((r.mode_fac - 1) / pe), mode_ci=float(r.mode_fac_ci / pe))
out["secants_p002"] = dict(
    p_eff=pe,
    sample_A=dict(uniform=dict(mfpt=fo["uniform"]["measured_slope_mfpt_p002"], mfpt_ci=fo["uniform"]["measured_slope_mfpt_ci"],
                               mode=fo["uniform"]["measured_slope_mode_p002"], mode_ci=fo["uniform"]["measured_slope_mode_ci"]),
                  thinned=dict(mfpt=fo["smart"]["measured_slope_mfpt_p002"], mfpt_ci=fo["smart"]["measured_slope_mfpt_ci"],
                               mode=fo["smart"]["measured_slope_mode_p002"], mode_ci=fo["smart"]["measured_slope_mode_ci"])),
    sample_B=sec)
out["susceptibility_other_sizes_sample_B"] = {
    n_: {k: json.load(open(os.path.join(B, f"v05_sens_N{n_}.json")))[k]
         for k in ("frac_lower_mfpt", "slope_mfpt_uniform", "slope_mode_uniform")} for n_ in (15, 25, 35)}

# ------------------------------------------------------------------ 3. localised placements
loc_b = pd.read_csv(_os.path.join(_R, 'data', 'msc_defects_verify', 'v06_local.csv'))
rows = []
for (kind, M), g in loc_b.groupby(["kind", "M"]):
    mf, mo = g.mfpt / CLEAN_MFPT, g["mode"] / CLEAN_MODE
    rows.append(dict(kind=kind, M=int(M), K=int(len(g)), acceptance=float(len(g) / g.attempts.sum()),
                     mfpt_factor=float(mf.mean()), mfpt_factor_ci=ci95(mf),
                     mode_factor=float(mo.mean()), mode_factor_ci=ci95(mo),
                     ratio=float(g.ratio.mean()), ratio_ci=ci95(g.ratio),
                     frac_mfpt_below_clean=float((g.mfpt < CLEAN_MFPT).mean()),
                     elasticity=float(math.log(mo.mean()) / math.log(mf.mean()))))
out["local_sample_B"] = rows
loc_a = pd.read_csv(_os.path.join(_R, 'data', 'msc_defects', 'summary_structured_local.csv'))
out["near_target_elasticity"] = dict(
    sample_A={int(r.M): float(math.log(r.mode_factor) / math.log(r.mfpt_factor)) for r in loc_a[loc_a.kind == "near_target"].itertuples()},
    sample_B={r["M"]: r["elasticity"] for r in rows if r["kind"] == "near_target"})
out["power_law_check_M40_near_target"] = dict(
    note="clean ratio x (MFPT factor of the M = 40 near-target ensemble)^(-3/4)",
    sample_A=float(CLEAN_RATIO * loc_a[(loc_a.kind == "near_target") & (loc_a.M == 40)].mfpt_factor.iloc[0] ** (-0.75)))

det = pd.read_csv(_os.path.join(_R, 'data', 'msc_defects', 'structured_deterministic.csv')).set_index("name")
c0 = det.loc["clean"]
rows = {}
for name in ("block12_centre", "block6_near_start", "target_box_R5_g1", "start_box_R5_g1", "wall_mid_gap_centre_g1"):
    r = det.loc[name]
    rows[name] = dict(blocked=int(r.n_blocked), volume_factor=float(r.n_cluster / c0.n_cluster),
                      Gdiff_factor=float((r.G_tt - r.G_st) / (c0.G_tt - c0.G_st)),
                      mfpt_factor=float(r.mfpt / c0.mfpt), mode_factor=float(r["mode"] / c0["mode"]),
                      ratio=float(r.ratio), cv=float(r.cv))
serp = det[det.index.str.startswith("serpentine")]
rows["serpentines"] = dict(ratio=[float(serp.ratio.min()), float(serp.ratio.max())], cv=[float(serp.cv.min()), float(serp.cv.max())])
out["deterministic"] = rows

# ------------------------------------------------------------------ 4. periodic arrays
per = pd.read_csv(_os.path.join(_R, 'data', 'msc_defects_verify', 'v06_periodic.csv'))
rows = {}
for a, g in per.groupby("a"):
    nb = g[g.tgt_nbr_blocked == 1]
    rows[int(a)] = dict(phases=int(len(g)), blocked_fraction=[float(g.p.min()), float(g.p.max())],
                        mfpt_factor=[float(g.mfpt_fac.min()), float(g.mfpt_fac.max()), float(g.mfpt_fac.mean())],
                        mode_factor=[float(g.mode_fac.min()), float(g.mode_fac.max()), float(g.mode_fac.mean())],
                        ratio=[float(g.ratio.min()), float(g.ratio.max())],
                        mfpt_factor_target_neighbour_blocked=float(nb.mfpt_fac.max()) if len(nb) else None)
out["periodic_phases"] = rows

# ------------------------------------------------------------------ 5. bulk factor of a periodic array (exact)
def array_conductivity(a):
    """One blocked site per a x a cell of the square lattice with unit conductances: exact
    conductivity sigma/sigma_0 from the unit-cell problem on the a x a torus (rational arithmetic).
    sigma = (1/a^2) min_u sum_bonds (E.e + u_j - u_i)^2 with E = e_1."""
    sites = [(x, y) for x in range(a) for y in range(a) if (x, y) != (0, 0)]
    idx = {s: k for k, s in enumerate(sites)}
    n = len(sites)
    L = sy.zeros(n, n); b = sy.zeros(n, 1); nx = 0
    for (x, y) in sites:
        for (dx, dy) in ((1, 0), (0, 1)):
            t = ((x + dx) % a, (y + dy) % a)
            if t not in idx or t == (x, y):
                continue
            i_, j_ = idx[(x, y)], idx[t]
            L[i_, i_] += 1; L[j_, j_] += 1; L[i_, j_] -= 1; L[j_, i_] -= 1
            if dx == 1:                       # bond along the field: term (1 + u_j - u_i)^2
                nx += 1; b[i_] += 1; b[j_] -= 1
    # minimise  nx + 2 b.u~ + u^T L u  with u~ sign convention: (1 + u_j - u_i)^2 -> gradient L u = b
    Lr, br = L[1:, 1:], b[1:, :]              # ground the first open site
    u = Lr.LUsolve(br)
    energy = nx - (br.T * u)[0, 0]
    return sy.nsimplify(energy / a ** 2)

hom = json.load(open(_os.path.join(_R, 'data', 'msc_defects', 'homogenization.json')))
hp = np.array([r["p"] for r in hom["finite_p"]]); ht = np.array([r["time_factor"] for r in hom["finite_p"]])
rows = {}
for a in (2, 3, 4):
    s = array_conductivity(a)
    theta = (1 - sy.Rational(1, a * a)) / s
    rows[a] = dict(sigma=str(s), theta=str(theta), theta_float=float(theta), density=1 / a ** 2,
                   theta_random_linear_interp=float(np.interp(1 / a ** 2, hp, ht)),
                   theta_random_cubic_interp=float(CubicSpline(hp, ht)(1 / a ** 2)))
out["periodic_bulk"] = rows

# ------------------------------------------------------------------ 6. two time scales
tm = vm["two_mode"]
T1, tau2, c, tau = tm["T1"], tm["tau2"], tm["c"], tm["tau_eff"]
t_formula = tau * math.log(c * (1 + T1 / tau))
elast = (tau / t_formula) * (T1 / tau) / (1 + T1 / tau)           # tau and c held fixed
# with tau_1 and c held fixed instead, tau = (1/tau_1 - 1/T_cap)^-1 follows T_cap: d tau / d T_cap = -tau^2 / T_cap^2
_dtau = -tau ** 2 / T1 ** 2
_L = math.log(c * (1 + T1 / tau))
elast_tau1 = (T1 / t_formula) * (_dtau * _L + tau * ((1 + _dtau) / (tau + T1) - _dtau / tau))
mu1 = math.sin(math.pi / (2 * N)) ** 2                    # slowest relaxation rate, unit rate, d = 2
W = 4 * math.cos(math.pi / (2 * N)) ** 2
def closed_form_mode(mfpt):                               # Section 6, eq-L1 (discrete-time mode)
    X = mu1 * q * mfpt
    return mfpt * math.log(W * X) / (X + W - 1), X
m0, X0 = closed_form_mode(CLEAN_MFPT)
tw = pd.read_csv(_os.path.join(_R, 'data', 'msc_defects_verify', 'v09b_twotime.csv'))
cases = []
for r in tw.itertuples():
    mcf, _ = closed_form_mode(CLEAN_MFPT * r.mfpt_fac)
    cases.append(dict(case=r.case, mfpt_factor=float(r.mfpt_fac), T1_factor=float(r.T1_fac),
                      mode_factor_exact=float(r.mode_fac_exact), elasticity_exact=float(r.elasticity_exact),
                      mode_factor_frozen_tau_c=float(r.pred_tau540),
                      share_of_log_shift_frozen=float(math.log(r.pred_tau540) / math.log(r.mode_fac_exact)),
                      mode_factor_closed_form_sec6=float(mcf / m0),
                      tau_first_contributing=float(r.tau_sub2 if abs(r.relw_sub1) < 0.5 else r.tau_sub1)))
out["two_time"] = dict(T_cap=T1, tau2=tau2, tau=tau, c=c, slowest_reflecting_time=1 / (q * mu1),
                       weight_of_621_mode=vm["clean_top_modes_lam_w_tau"][1][1],
                       weight_of_leading_mode=vm["clean_top_modes_lam_w_tau"][0][1],
                       mode_two_term_discrete=tm["mode_two_mode_truncation"], mode_three_terms=tm["mode_with_3_modes"],
                       t_formula=t_formula, elasticity_formula=elast, elasticity_formula_tau1_c_fixed=elast_tau1,
                       T_cap_over_tau=T1 / tau,
                       closed_form_sec6=dict(X=X0, W=W, mode=m0, rel_err=m0 / CLEAN_MODE - 1,
                                             second_time_constant=1 / (1 / (1 / (q * mu1)) + W / CLEAN_MFPT),
                                             elasticity=1 / math.log(W * X0) + (W - 1) / (X0 + W - 1)),
                       cases=cases,
                       cross_placement_slope_p02=dict(sample_B=vm["p0.20"]["loglog_slope_mode_on_mfpt"]))
g = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(_os.path.join(_R, 'data', 'msc_defects', 'sweep', 'sweep_N35_uniform_*.csv')))])
g = g[np.isclose(g.p, 0.20)]
out["two_time"]["cross_placement_slope_p02"]["sample_A"] = float(np.polyfit(np.log(g.mfpt), np.log(g["mode"]), 1)[0])

# ------------------------------------------------------------------ 7. shape family (sample A)
fam = [pd.read_csv(f)[["ratio", "cv"]] for f in sorted(glob.glob(_os.path.join(_R, 'data', 'msc_defects', 'sweep', '*.csv')))]
fam += [pd.read_csv(_os.path.join(_R, 'data', 'msc_defects', 'structured_local.csv'))[["ratio", "cv"]],
        pd.read_csv(_os.path.join(_R, 'data', 'msc_defects', 'structured_deterministic.csv'))[["ratio", "cv"]]]
F = pd.concat(fam)
co = np.polyfit(F.ratio, F.cv, 2)
res = F.cv - np.polyval(co, F.ratio)
sweep_all = pd.concat([pd.read_csv(f)[["p", "ratio", "cv"]] for f in sorted(glob.glob(_os.path.join(_R, 'data', 'msc_defects', 'sweep', '*.csv')))])
big = sweep_all[np.abs(sweep_all.cv - np.polyval(co, sweep_all.ratio)) > 0.02]
n_big_other = int((np.abs(res) > 0.02).sum()) - len(big)
cl = pd.read_csv(_os.path.join(_R, 'data', 'msc_defects', 'size_clean.csv'))
clres = (cl.sd / cl.mfpt) - np.polyval(co, cl["mode"] / cl.mfpt)
out["shape_family"] = dict(
    sample_A=dict(n=int(len(F)), pearson=float(np.corrcoef(F.ratio, F.cv)[0, 1]),
                  quad_c0_c1_c2=[float(co[2]), float(co[1]), float(co[0])],
                  rms=float(np.sqrt(np.mean(res ** 2))), max_abs_resid=float(np.abs(res).max()),
                  frac_resid_gt_0p02=float((np.abs(res) > 0.02).mean()),
                  n_resid_gt_0p02=int((np.abs(res) > 0.02).sum()),
                  n_resid_gt_0p02_in_random_sweep=int(len(big)), n_resid_gt_0p02_elsewhere=n_big_other,
                  min_p_of_resid_gt_0p02=float(big.p.min()),
                  composition=dict(random_sweep=int(len(sweep_all)), localised=int(len(fam[-2])), deterministic=int(len(fam[-1]))),
                  clean_resid={int(n_): float(r_) for n_, r_ in zip(cl.N, clres)},
                  chain_1d_point=[1 / 3, math.sqrt(2 / 3)],
                  chain_1d_resid=float(math.sqrt(2 / 3) - np.polyval(co, 1 / 3))),
    sample_B=dict(n=vm["family_recheck"]["n"], pearson=vm["family_recheck"]["pearson"],
                  quad_c0_c1_c2=vm["family_recheck"]["quad"][::-1], rms=vm["family_recheck"]["rms"],
                  max_abs_resid=vm["family_recheck"]["max_abs_resid"]))

# ------------------------------------------------------------------ 8. chain with barriers
rng = np.random.default_rng(20261001)
worst = 0.0
for _ in range(200):
    n = int(rng.integers(3, 80))
    wk = rng.uniform(0.01, 0.5, n - 1)                    # w_{k-1} + w_k <= 1
    Q = np.zeros((n - 1, n - 1))
    for k in range(n - 1):
        left = wk[k - 1] if k > 0 else 0.0
        Q[k, k] = 1 - left - wk[k]
        if k > 0:
            Q[k, k - 1] = left
        if k < n - 2:
            Q[k, k + 1] = wk[k]
    h = np.linalg.solve(np.eye(n - 1) - Q, np.ones(n - 1))
    worst = max(worst, abs(h[0] / np.sum(np.arange(1, n) / wk) - 1))
bj = json.load(open(_os.path.join(_R, 'data', 'msc_defects', 'summary_beyond.json')))
vb = json.load(open(_os.path.join(_R, 'data', 'msc_defects_verify', 'v08_beyond.json')))
out["chain_barriers"] = dict(max_rel_err_formula_200_random_chains=float(worst),
                             clean_N100=100 * 99 / q,
                             disorder_average_kappa02={str(p): 100 * 99 / q * (1 + 4 * p) for p in (0.05, 0.1, 0.2, 0.3, 0.5)},
                             ratio_range_sample_A=[min(r["ratio"] for r in bj["1d_random"][1:]), max(r["ratio"] for r in bj["1d_random"][1:])],
                             ratio_range_sample_B=[min(v["ratio"] for v in vb["1d_random"].values()), max(v["ratio"] for v in vb["1d_random"].values())])

# ------------------------------------------------------------------ 9. permeable obstacles
pa = {r["kappa"]: r for r in bj["2d_permeable"]}
pb = {float(k): v for k, v in vb["perm"].items()}
out["permeable"] = dict(
    sample_A={str(k): dict(excess=pa[k]["ratio"] / CLEAN_RATIO - 1, excess_over_ci=(pa[k]["ratio"] - CLEAN_RATIO) / pa[k]["ratio_ci"]) for k in (0.0, 0.001, 0.01, 0.1, 0.3, 0.6)},
    sample_B={str(k): dict(excess=pb[k]["ratio"] / CLEAN_RATIO - 1, excess_over_ci=(pb[k]["ratio"] - CLEAN_RATIO) / pb[k]["ratio_ci"]) for k in (0.0, 0.001, 0.01, 0.1, 0.3, 0.6)},
    inert_mean_over_1_minus_p=dict(sample_A=pa[0.0]["mfpt"] / 0.9, sample_B=pb[0.0]["mfpt"] / 0.9),
    mean_at_kappa_0p001=dict(sample_A=pa[0.001]["mfpt"], sample_B=pb[0.001]["mfpt"]))

os.makedirs(_os.path.join(_R, 'data', 'article'), exist_ok=True)
json.dump(out, open(OUT, "w"), indent=1)
print(json.dumps(out, indent=1))
