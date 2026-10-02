#!/usr/bin/env python
"""Section 5 (s5_modes): numbers, tables and re-checks with separately written code.

Everything quoted in sections/s5_modes.tex is collected here from the research result
files, and the statements that the section proves or tabulates are re-checked with
code written for this script (no research library is imported):

 1. Proposition s5_modes:prop-1d  -- eigenvectors of the killed chain, the closed-form
    PMF against explicit matrix powers, normalisation and mean.
 2. Exact 1D modes at q = 0.8 from the closed form (N = 50, 100, 200, 1000) and the two
    large-N modes (N = 1e5, 1e6) as local maxima of the closed form in 60-digit arithmetic.
 3. Continuum constants of the 1D law (mpmath).
 4. Exact 2D / 3D modes by a separately written time stepper (2D: N = 35, 100, 200;
    3D: N = 35, 40).
 5. Rows of Table s5_modes:tab-modes, the quantities q t*/N^2 and t*/T, local exponents
    and local slopes quoted in the text.
 6. Condensed rows of Table s5_modes:tab-fits.

Inputs  (relative to ):
    data/msc_modes/discrete_modes.jsonl, laplace_modes.jsonl, fit_results.json
    data/msc_modes_verify/bigN_1d_mp.json, constants.json
Output: data/article/s5_modes_tables.json
Usage:  python s5_modes_tables.py          (about 1 minute, < 200 MB; the stepper re-check of d = 2, N = 401 and
                                            d = 3, N = 100 is then COPIED from the existing output file)
        python s5_modes_tables.py --big    (re-steps these two cases as well; about 9 minutes more, < 200 MB)
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import json
import math
import os
import sys
import time

import numpy as np
import mpmath as mp

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = _R            # 
RD = _os.path.join(_R, 'data', 'msc_modes')
RV = _os.path.join(_R, 'data', 'msc_modes_verify')
OUT = _os.path.join(_R, 'data', 'article', 's5_modes_tables.json')
Q = 0.8

D = [json.loads(l) for l in open(_os.path.join(_R, 'data', 'msc_modes', 'discrete_modes.jsonl')) if l.strip()]
L = [json.loads(l) for l in open(_os.path.join(_R, 'data', 'msc_modes', 'laplace_modes.jsonl')) if l.strip()]
FITS = json.load(open(_os.path.join(_R, 'data', 'msc_modes', 'fit_results.json')))
BIG = json.load(open(_os.path.join(_R, 'data', 'msc_modes_verify', 'bigN_1d_mp.json')))
CONST = json.load(open(_os.path.join(_R, 'data', 'msc_modes_verify', 'constants.json')))

out = {"q": Q, "generated_by": "code/article/s5_modes_tables.py"}


def disc(d, N):
    r = [x for x in D if x["d"] == d and x["geo"] == "CC" and x["q"] == Q and x["N"] == N]
    return r[0] if r else None


def cont(d, N):
    r = [x for x in L if x["d"] == d and x["geo"] == "CC" and x["N"] == N]
    return r[0] if r else None


# ---------------------------------------------------------------------------
# 1. Proposition prop-1d: structure of the proof and the closed form
# ---------------------------------------------------------------------------
def killed_chain(N, q):
    """Q on the sites 0..N-2 (target N-1 removed) and the absorption vector r."""
    n = N - 1
    K = np.zeros((n, n))
    for j in range(n):
        if j == 0:
            K[0, 0] += 0.5                     # cancelled move at the reflecting end
        else:
            K[j, j - 1] += 0.5
        if j + 1 <= n - 1:
            K[j, j + 1] += 0.5                 # (the move N-2 -> N-1 leaves the killed chain)
    Qm = (1 - q) * np.eye(n) + q * K
    r = np.zeros(n)
    r[n - 1] = q / 2
    return K, Qm, r


def pmf_closed(N, q, t):
    """Eq. (s5_modes:eq-1d-pmf); t is an integer array."""
    m = np.arange(1, N)
    om = (2 * m - 1) * np.pi / (2 * N - 1)
    lam = 1 - q + q * np.cos(om)
    w = (2 * q / (2 * N - 1)) * (-1.0) ** (m + 1) * np.cos(om / 2) * np.sin(om)
    t = np.asarray(t, dtype=np.int64)
    return (w[:, None] * np.power(lam[:, None], (t - 1)[None, :])).sum(axis=0)


chk = []
for N, q in ((5, 0.8), (8, 0.8), (20, 0.8), (20, 1.0), (50, 0.3), (50, 0.8)):
    K, Qm, r = killed_chain(N, q)
    m = np.arange(1, N)
    om = (2 * m - 1) * np.pi / (2 * N - 1)
    j = np.arange(N - 1)
    V = np.cos(np.outer(j + 0.5, om))                       # columns v_m
    eig_res = np.abs(K @ V - V * np.cos(om)[None, :]).max()
    norm_dev = np.abs((V * V).sum(axis=0) - (2 * N - 1) / 4).max()
    gram = V.T @ V
    orth_dev = np.abs(gram - np.diag(np.diag(gram))).max()
    tmax = int(40 * N * N / q)
    f_mat = np.empty(tmax)
    v = np.zeros(N - 1)
    v[0] = 1.0
    for t in range(1, tmax + 1):                            # f(t) = e_0^T Q^{t-1} r
        f_mat[t - 1] = v @ r
        v = v @ Qm
    f_cl = pmf_closed(N, q, np.arange(1, tmax + 1))
    tt = np.arange(1, tmax + 1)
    chk.append({
        "N": N, "q": q,
        "max_abs_eigen_residual": float(eig_res),
        "max_abs_norm_minus_(2N-1)/4": float(norm_dev),
        "max_abs_offdiag_gram": float(orth_dev),
        "max_abs_diff_closed_vs_matrix_power": float(np.abs(f_cl - f_mat).max()),
        "one_minus_sum_f": float(1 - f_cl.sum()),
        "sum_t_f": float((tt * f_cl).sum()),
        "N(N-1)/q": N * (N - 1) / q,
        "argmax_closed": int(tt[np.argmax(f_cl)]),
        "argmax_matrix_power": int(tt[np.argmax(f_mat)]),
    })
out["prop_1d_checks"] = chk

# ---------------------------------------------------------------------------
# 2. exact 1D modes from the closed form
# ---------------------------------------------------------------------------
modes_1d = []
for N in (50, 100, 200, 1000):
    T = N * (N - 1) / Q
    t_hi = int(1.2 * T)
    f_all = []
    for a in range(1, t_hi + 1, 4000):
        tt = np.arange(a, min(a + 4000, t_hi + 1))
        f_all.append(pmf_closed(N, Q, tt))
    f_all = np.concatenate(f_all)
    k = int(np.argmax(f_all))
    # f(t) = 0 exactly for t < N - 1 and the closed form returns round-off noise there, so local
    # maxima are counted only where f exceeds 1e-9 of its maximum
    big_enough = f_all[1:-1] > 1e-9 * f_all[k]
    interior = (f_all[1:-1] > f_all[:-2]) & (f_all[1:-1] >= f_all[2:]) & big_enough
    stored = disc(1, N)
    modes_1d.append({
        "N": N, "mode_closed_form": k + 1, "mode_stored": stored["mode"] if stored else None,
        "n_local_maxima_with_f>1e-9fmax_up_to_1.2T": int(interior.sum()),
        "T": T, "ratio": (k + 1) / T, "q_mode_over_N2": Q * (k + 1) / N ** 2,
    })
out["modes_1d_closed_form"] = modes_1d

# the research runs already compared time stepping with the closed form at every 1D size
d1rows = [x for x in D if x["d"] == 1 and x["geo"] == "CC" and x["q"] == Q]
out["d1_stepper_vs_closed_form_all_sizes"] = {
    "n_sizes": len(d1rows),
    "all_modes_equal": bool(all(x["mode"] == x["mode_closed_form_check"] for x in d1rows)),
    "max_abs_pmf_difference": max(x["closed_form_max_abs_diff_window"] for x in d1rows)}
# shortest run length in units of the mode, all corner-to-corner runs at q = 0.8
out["min_run_length_over_mode"] = {str(d): min(x["t_end"] / x["mode"] for x in D if x["d"] == d and x["geo"] == "CC" and x["q"] == Q)
                                   for d in (1, 2, 3)}

# large N: the two modes of the 50-digit check are local maxima of the closed form
mp.mp.dps = 60


def f_mp(N, q, t, terms=60):
    s = mp.mpf(0)
    q = mp.mpf(q)
    for m in range(1, terms + 1):
        om = (2 * m - 1) * mp.pi / (2 * N - 1)
        lam = 1 - q + q * mp.cos(om)
        s += (-1) ** (m + 1) * mp.cos(om / 2) * mp.sin(om) * mp.power(lam, t - 1)
    return 2 * q / (2 * N - 1) * s


big = []
for row in BIG:
    N = row["N"]
    t0 = row["recheck"]
    T = mp.mpf(N) * (N - 1) / mp.mpf("0.8")
    fm, f0, fp = (f_mp(N, "0.8", t0 - 1), f_mp(N, "0.8", t0), f_mp(N, "0.8", t0 + 1))
    # neglected terms: |lam_m|^(t-1) <= exp(-q (1 - cos om_m) (t-1)); m = 61 gives < 1e-1000
    big.append({
        "N": N, "mode": t0,
        "is_local_max_of_closed_form_60_digits": bool(f0 > fm and f0 > fp),
        "f(mode)-f(mode-1)": mp.nstr(f0 - fm, 5), "f(mode+1)-f(mode)": mp.nstr(fp - f0, 5),
        "ratio_mode_over_T": mp.nstr(t0 / T, 12),
        "q_mode_over_(N-1/2)^2": mp.nstr(mp.mpf("0.8") * t0 / (mp.mpf(N) - mp.mpf("0.5")) ** 2, 12),
    })
out["modes_1d_large_N"] = big

# ---------------------------------------------------------------------------
# 3. continuum constants of the 1D law
# ---------------------------------------------------------------------------
mp.mp.dps = 30


def Phi(x, der=0):
    """Phi(xi) = pi sum_{m>=1} (-1)^{m+1} (2m-1) exp(-(2m-1)^2 pi^2 xi/4) and its xi-derivative."""
    s = mp.mpf(0)
    for m in range(1, 60):
        a = (2 * m - 1) ** 2 * mp.pi ** 2 / 4
        s += (-1) ** (m + 1) * (2 * m - 1) * (-a) ** der * mp.exp(-a * x)
    return mp.pi * s


def Surv(x):
    s = mp.mpf(0)
    for m in range(1, 60):
        a = (2 * m - 1) ** 2 * mp.pi ** 2 / 4
        s += (-1) ** (m + 1) / (2 * m - 1) * mp.exp(-a * x)
    return 4 / mp.pi * s


xi_star = mp.findroot(lambda x: Phi(x, 1), mp.mpf("0.1666"))
xi_med = mp.findroot(lambda x: Surv(x) - mp.mpf("0.5"), mp.mpf("0.38"))
# moments by term-wise integration of the series: int xi Phi = (16/pi^3) sum (-1)^{m+1}/(2m-1)^3 = 1/2,
# int xi^2 Phi = (128/pi^5) sum (-1)^{m+1}/(2m-1)^5 = 5/12
mean_xi = 16 / mp.pi ** 3 * mp.nsum(lambda m: (-1) ** (m + 1) / (2 * m - 1) ** 3, [1, mp.inf])
m2_xi = 128 / mp.pi ** 5 * mp.nsum(lambda m: (-1) ** (m + 1) / (2 * m - 1) ** 5, [1, mp.inf])
cv = mp.sqrt(m2_xi - mean_xi ** 2) / mean_xi
out["continuum_1d"] = {
    "xi_star": mp.nstr(xi_star, 15),
    "ratio_2_xi_star": mp.nstr(2 * xi_star, 15),
    "one_third_minus_ratio": mp.nstr(mp.mpf(1) / 3 - 2 * xi_star, 6),
    "mean_xi": mp.nstr(mean_xi, 12), "second_moment_xi": mp.nstr(m2_xi, 12),
    "cdf_at_mode": mp.nstr(1 - Surv(xi_star), 10),
    "median_over_mean": mp.nstr(xi_med / mean_xi, 10),
    "cv": mp.nstr(cv, 10), "sqrt(2/3)": mp.nstr(mp.sqrt(mp.mpf(2) / 3), 10),
    "max_density_times_mean": mp.nstr(Phi(xi_star) * mean_xi, 10),
    "survival_at_mean": mp.nstr(Surv(mean_xi), 10),
    "single_image_mode_xi": "1/6 (maximiser of xi^(-3/2) exp(-1/(4 xi)))",
    "stored_in_research_files": {k: CONST[k] for k in CONST if k.startswith("1d_")},
}


# image (short-time) form of the same density: Phi(xi) = pi^(-1/2) xi^(-3/2) sum_{k>=0} (-1)^k (2k+1) exp(-(2k+1)^2/(4 xi));
# its first term peaks at exactly xi = 1/6
def Phi_images(x, kmax=40):
    return x ** mp.mpf("-1.5") / mp.sqrt(mp.pi) * mp.fsum(
        (-1) ** k * (2 * k + 1) * mp.exp(-mp.mpf((2 * k + 1) ** 2) / (4 * x)) for k in range(kmax))


out["continuum_1d"]["image_series_max_rel_diff_vs_spectral_series"] = mp.nstr(
    max(abs(Phi_images(mp.mpf(x)) / Phi(mp.mpf(x)) - 1) for x in ("0.05", "0.1", "0.1666", "0.3", "0.5", "1")), 3)
# d/dxi [xi^(-3/2) exp(-1/(4 xi))] = xi^(-3/2) exp(-1/(4 xi)) [-3/(2 xi) + 1/(4 xi^2)]: zero at xi = 1/6
out["continuum_1d"]["first_image_term_maximiser"] = mp.nstr(
    mp.findroot(lambda x: -mp.mpf(3) / (2 * x) + 1 / (4 * x * x), mp.mpf("0.16")), 15)
out["continuum_1d"]["one_sixth_minus_xi_star"] = mp.nstr(mp.mpf(1) / 6 - xi_star, 6)
out["continuum_1d"]["ratio_0.64_over_limit"] = mp.nstr(mp.mpf("0.64") / (2 * xi_star), 5)
# lattice ratios in continuous time (unit rate) approaching 2 xi*
out["lattice_ratio_1d_continuous_time"] = [
    {"N": r["N"], "mode_over_T": r["mode"] / r["mfpt"], "mode_over_(N-1/2)^2": r["mode"] / (r["N"] - 0.5) ** 2}
    for r in sorted([x for x in L if x["d"] == 1 and x["geo"] == "CC"], key=lambda x: x["N"])
    if r["N"] in (10, 100, 1000, 3620, 10240, 100000, 1000000)]


# ---------------------------------------------------------------------------
# 4. separately written time stepper (2D, 3D)
# ---------------------------------------------------------------------------
def step_modes(d, N, q, t_end):
    p = np.zeros((N,) * d)
    p[(0,) * d] = 1.0
    target = (N - 1,) * d
    c = q / (2 * d)
    f = np.empty(t_end)
    full = [slice(None)] * d
    for t in range(t_end):
        new = (1 - q) * p
        for ax in range(d):
            hi, lo, last, first = list(full), list(full), list(full), list(full)
            hi[ax] = slice(1, None)
            lo[ax] = slice(0, -1)
            last[ax] = slice(-1, None)
            first[ax] = slice(0, 1)
            hi, lo, last, first = tuple(hi), tuple(lo), tuple(last), tuple(first)
            new[hi] += c * p[lo]            # moves in the + direction
            new[last] += c * p[last]        # cancelled at the far face
            new[lo] += c * p[hi]            # moves in the - direction
            new[first] += c * p[first]      # cancelled at the near face
        f[t] = new[target]                  # mass arriving at the target at step t+1
        new[target] = 0.0
        p = new
    k = int(np.argmax(f))
    interior = (f[1:-1] > f[:-2]) & (f[1:-1] >= f[2:])
    nb = (float(f[k - 1] / f[k] - 1), float(f[k + 1] / f[k] - 1))      # relative depth of the two neighbours
    return k + 1, int(interior.sum()), float(f[k]), float(p.sum()), nb


step = []
cases = [(2, 35), (2, 100), (2, 200), (3, 35), (3, 40)]
if "--big" in sys.argv:
    cases += [(2, 401), (3, 100)]
elif os.path.exists(OUT):                      # keep the results of an earlier --big run
    try:
        step = [r for r in json.load(open(OUT)).get("stepper_recheck", []) if (r["d"], r["N"]) in ((2, 401), (3, 100))]
    except Exception:
        step = []
for d, N in cases:
    stored = disc(d, N)
    t0 = time.time()
    mode, nmax, fmax, surv, nb = step_modes(d, N, Q, int(1.4 * stored["mode"]))
    step.append({"d": d, "N": N, "mode_this_script": mode, "mode_stored": stored["mode"],
                 "n_local_maxima_up_to_1.4_modes": nmax, "f_max": fmax,
                 "f(mode-1)/f(mode)-1": nb[0], "f(mode+1)/f(mode)-1": nb[1],
                 "survival_at_end_of_run": surv, "wall_s": round(time.time() - t0, 1)})
    print(step[-1], flush=True)
step.sort(key=lambda r: (r["d"], r["N"]))
out["stepper_recheck"] = step

# ---------------------------------------------------------------------------
# 5. table rows and quoted quantities
# ---------------------------------------------------------------------------
rows = []
for d, sizes in ((1, (5, 50, 100, 200, 1000)), (2, (5, 35, 100, 200, 401)), (3, (5, 35, 40, 100))):
    for N in sizes:
        r = disc(d, N)
        rows.append({"d": d, "N": N, "time": "discrete, q=0.8", "mode_steps": r["mode"],
                     "T_steps": r["mfpt_exact"], "q_mode_over_N2": Q * r["mode"] / N ** 2,
                     "ratio": r["mode"] / r["mfpt_exact"],
                     "n_local_maxima": r["n_local_maxima"], "ties_exact": r["ties_exact"]})
for b in big[1:]:
    N = b["N"]
    T = N * (N - 1) / Q
    rows.append({"d": 1, "N": N, "time": "discrete, q=0.8 (closed form, 50-digit check)",
                 "mode_steps": b["mode"], "T_steps": T, "q_mode_over_N2": Q * b["mode"] / N ** 2,
                 "ratio": b["mode"] / T})
for d, sizes in ((2, (1810, 20480, 1048576, 16777216)), (3, (320, 1280, 4096))):
    for N in sizes:
        r = cont(d, N)
        rows.append({"d": d, "N": N, "time": "continuous, unit rate", "mode_unit_rate": r["mode"],
                     "qT": r["mfpt"], "q_mode_over_N2": r["mode"] / N ** 2, "ratio": r["mode"] / r["mfpt"]})
rows.sort(key=lambda r: (r["d"], r["N"]))
out["tab_modes_rows"] = rows

out["counts"] = {
    "discrete_CC_q0.8": {str(d): len([x for x in D if x["d"] == d and x["geo"] == "CC" and x["q"] == Q]) for d in (1, 2, 3)},
    "continuous_CC": {str(d): len([x for x in L if x["d"] == d and x["geo"] == "CC"]) for d in (1, 2, 3)},
    "discrete_CC_q0.8_all_one_local_max_no_tie": bool(all(
        x["n_local_maxima"] == 1 and x["ties_exact"] == 1 for x in D if x["geo"] == "CC" and x["q"] == Q)),
    "continuous_CC_all_one_sign_change": bool(all(
        x.get("n_sign_changes_down", 1) == 1 and x.get("n_sign_changes_up", 0) == 0 for x in L if x["geo"] == "CC" and x["d"] > 1)),
}

# monotonicity along the computed grids (d = 2, 3): q t*/N^2 increases, t*/T decreases
mono = {}
for d in (2, 3):
    dd = sorted([x for x in D if x["d"] == d and x["geo"] == "CC" and x["q"] == Q], key=lambda x: x["N"])
    cc = sorted([x for x in L if x["d"] == d and x["geo"] == "CC"], key=lambda x: x["N"])
    a = np.array([Q * x["mode"] / x["N"] ** 2 for x in dd])
    b = np.array([x["mode"] / x["mfpt_exact"] for x in dd])
    a2 = np.array([x["mode"] / x["N"] ** 2 for x in cc])
    b2 = np.array([x["mode"] / x["mfpt"] for x in cc])
    mono[str(d)] = {
        "n_discrete": len(dd), "N_discrete_range": [dd[0]["N"], dd[-1]["N"]],
        "discrete_q_mode_over_N2_strictly_increasing": bool(np.all(np.diff(a) > 0)),
        "discrete_ratio_strictly_decreasing": bool(np.all(np.diff(b) < 0)),
        "n_continuous": len(cc), "N_continuous_range": [cc[0]["N"], cc[-1]["N"]],
        "continuous_mode_over_N2_strictly_increasing": bool(np.all(np.diff(a2) > 0)),
        "continuous_ratio_strictly_decreasing": bool(np.all(np.diff(b2) < 0)),
        "max_rel_width_of_mode_bracket_continuous": max(x.get("bracket_rel_width", 0.0) for x in cc),
    }
out["monotonicity"] = mono
# separation T / t* (inverse of the ratio column)
out["separation_T_over_mode"] = [{"d": r["d"], "N": r["N"], "time": r["time"], "T_over_mode": 1 / r["ratio"]} for r in rows]
# prefactor growth in 2D over 5 <= N <= 200
out["d2_prefactor_growth_N5_to_200"] = (disc(2, 200)["mode"] / 200 ** 2) / (disc(2, 5)["mode"] / 5 ** 2)
# 2D: constant of the formal law compared with ln ln N at the largest size
out["d2_ln_8pi"] = math.log(8 * math.pi)
out["d2_lnlnN_at_largest_N"] = math.log(math.log(16777216))
out["four_over_pi2"] = 4 / math.pi ** 2
out["six_over_pi2"] = 6 / math.pi ** 2

E = FITS["effective_exponents"]
A = FITS["asymptotics"]


def pick(xs, ys, targets):
    xs = np.array(xs)
    return [{"N_mid": float(xs[int(np.argmin(np.abs(np.log(xs / t))))]),
             "value": float(ys[int(np.argmin(np.abs(np.log(xs / t))))])} for t in targets]


out["local_exponents"] = {
    "d1_mode": pick(E["1"]["N_geo_mean"], E["1"]["a_eff_mode"], (6, 134, 3000)),
    "d2_mode": pick(E["2"]["N_geo_mean"], E["2"]["a_eff_mode"], (6, 134, 3e6)),
    "d2_mean": pick(E["2"]["N_geo_mean"], E["2"]["a_eff_mfpt"], (6, 134, 3e6)),
    "d3_mode": pick(E["3"]["N_geo_mean"], E["3"]["a_eff_mode"], (6, 134, 3238)),
    "d3_mean": pick(E["3"]["N_geo_mean"], E["3"]["a_eff_mfpt"], (6, 134, 3238)),
}
s2 = A["2d_slope_d(mode/N^2)/d(lnlnN)"]
s3 = A["3d_slope_d(mode/N^2)/d(lnN)"]
out["local_slopes"] = {
    "d2_vs_lnlnN": pick(s2["N_mid"], s2["slope"], (6, 1.5e6, 1.2e7)), "d2_limit": s2["theory_limit"],
    "d3_vs_lnN": pick(s3["N_mid"], s3["slope"], (6, 134, 3238)), "d3_limit": s3["theory_limit"],
}

# ---------------------------------------------------------------------------
# 6. condensed fit table
# ---------------------------------------------------------------------------
F = FITS["fits"]


def fitrow(block, key):
    r = F[block][key]
    return {"form": r["form"], "n_points": r["n_points"], "params": r["params"], "ci95_halfwidth": r["ci95_halfwidth"],
            "rms_ln_residual": r["rms_ln_residual"], "dAICc": r["dAICc"],
            "holdout_N": r.get("holdout_N"), "holdout_rel_error": r.get("holdout_rel_error")}


fit = {"2d_fit_10_200": {k: fitrow("2d_mode_fit_10_200_extrapolated", k) for k in ("C2", "P", "L2", "PL", "LL2", "LN2")},
       "3d_fit_10_100": {k: fitrow("3d_mode_fit_10_100_extrapolated", k) for k in ("C3", "C2", "P", "L2", "PL", "LN2")}}
# closed form (Eq. s6_mechanism:eq-L1) at the hold-out sizes
fit["closed_form_L1_rel_error"] = {
    "2d": {str(N): cont(2, N)["pred_L1"] / cont(2, N)["mode"] - 1 for N in (1280, 10240, 163840, 16777216)},
    "3d": {str(N): cont(3, N)["pred_L1"] / cont(3, N)["mode"] - 1 for N in (320, 1280, 4096)}}
fit["power_law_exponent_discrete_short_range"] = {
    "1d_N10_200": [F["1d_mode_discrete_short_range"]["P"]["params"]["a"], F["1d_mode_discrete_short_range"]["P"]["ci95_halfwidth"]["a"]],
    "2d_N11_200": [F["2d_mode_discrete_short_range"]["P"]["params"]["a"], F["2d_mode_discrete_short_range"]["P"]["ci95_halfwidth"]["a"]],
    "3d_N11_100": [F["3d_mode_discrete_short_range"]["P"]["params"]["a"], F["3d_mode_discrete_short_range"]["P"]["ci95_halfwidth"]["a"]]}
fit["power_law_exponent_2d_continuous_by_window"] = {
    "N10_200": [F["2d_mode_fit_10_200_extrapolated"]["P"]["params"]["a"], F["2d_mode_fit_10_200_extrapolated"]["P"]["ci95_halfwidth"]["a"]],
    "N10_1.7e7": [F["2d_mode_fit_all"]["P"]["params"]["a"], F["2d_mode_fit_all"]["P"]["ci95_halfwidth"]["a"]],
    "N1000_1.7e7": [F["2d_mode_fit_large"]["P"]["params"]["a"], F["2d_mode_fit_large"]["P"]["ci95_halfwidth"]["a"]]}
fit["dAICc_lnln_form_minus_best_by_window_2d"] = {
    b: {"LL2": F[b]["LL2"]["dAICc"], "PL": F[b]["PL"]["dAICc"]}
    for b in ("2d_mode_fit_10_200_extrapolated", "2d_mode_fit_all", "2d_mode_fit_large", "2d_mode_discrete_short_range")}
fit["N3_fit_overprediction_factor_at_4096"] = 1 + F["3d_mode_fit_10_100_extrapolated"]["C3"]["holdout_rel_error"][-1]
out["tab_fits"] = fit

os.makedirs(_os.path.join(_R, 'data', 'article'), exist_ok=True)
json.dump(out, open(OUT, "w"), indent=1)
print("written", OUT)
