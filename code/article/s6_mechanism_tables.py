#!/usr/bin/env python
"""Numbers, table rows and consistency checks for Section 6 (s6_mechanism).

Part A reads stored results only (no new first-passage computation):
  data/msc_modes/laplace_modes.jsonl   exact continuous-time modes, poles, residues, stored predictions
  data/msc_modes/discrete_modes.jsonl  exact discrete-time modes (time stepping)
  data/msc_modes/medians.json          medians, survival at the mode and at the mean
  data/msc_modes/full_tail.json        full-support discrete PMFs (coefficient of variation)
  data/msc_modes/fit_results.json      continuum exit constants, local slopes
  data/article/closed_form_checks.json             first accuracy check of the two closed-form variants (closed_form_checks.py)
and evaluates
  A1  accuracy of L0, (eq-L1), (eq-L1-simple), two and three exact poles (corner to corner, corner to centre)
  A2  the ingredients of the formal two-pole argument (nu_0 T, nu_1/mu, b_1/b_0) and the error budget of (eq-L1)
  A3  nu_1 t* against ln(2dX)
  A4  the formal leading laws (eq-asym-2d), (eq-asym-3d) against the exact modes
  A5  other placements (corner->centre, centre->corner, exit) and the activity scan
Part B recomputes, with code/msc_modes/fptlib.py, the height of the density at the mode and
the window in which the density exceeds 99 % of its maximum at the sizes used in the shape table
(the stored data/msc_modes/profile_stats.json uses a different set of sizes), and cross-checks the
survival at the mode against medians.json.
Part C checks Proposition (factorisation) on small lattices by dense linear algebra and measures the
convergence of the arrival profile u to theta_4(0, exp(-x))^d.

Output: ../data/s6_mechanism_tables.json   (plus a printed summary)
Usage:  python s6_mechanism_tables.py
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import json
import math
import os
import sys

import numpy as np
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = _R            # 
OUT = _os.path.join(_R, 'data', 'article', 's6_mechanism_tables.json')
sys.path.insert(0, _os.path.join(_R, 'code', 'msc_modes'))
import fptlib as F  # noqa: E402


def jl(name):
    return [json.loads(l) for l in open(os.path.join(_R, 'data', 'msc_modes', name)) if l.strip()]


LAP = jl("laplace_modes.jsonl")
DIS = jl("discrete_modes.jsonl")
MED = json.load(open(_os.path.join(_R, 'data', 'msc_modes', 'medians.json')))
TAIL = json.load(open(_os.path.join(_R, 'data', 'msc_modes', 'full_tail.json')))
PROF = json.load(open(_os.path.join(_R, 'data', 'msc_modes', 'profile_stats.json')))
FITS = json.load(open(_os.path.join(_R, 'data', 'msc_modes', 'fit_results.json')))
PLAN = json.load(open(_os.path.join(_R, 'data', 'article', 'closed_form_checks.json')))

out = {"sources": ["data/msc_modes/laplace_modes.jsonl", "data/msc_modes/discrete_modes.jsonl",
                   "data/msc_modes/medians.json", "data/msc_modes/full_tail.json",
                   "data/msc_modes/profile_stats.json", "data/msc_modes/fit_results.json",
                   "data/article/closed_form_checks.json"]}


def sel(d, geo):
    return sorted((r for r in LAP if r["d"] == d and r["geo"] == geo), key=lambda r: r["N"])


# ---------------------------------------------------------------------------------------------
# A1. accuracy of the predictions
# ---------------------------------------------------------------------------------------------
def preds(r):
    d, N, T, t = r["d"], r["N"], r["mfpt"], r["mode"]
    mu, W, A = r["p"][0], r["W"][0], r["A_u"][0]
    X = mu * T
    nu0, nu1 = r["nu"][0], r["nu"][1]
    b0, b1 = r["res"][0], r["res"][1]
    two = math.log(-b1 * nu1 / (b0 * nu0)) / (nu1 - nu0)       # maximiser of b0 e^{-nu0 t} + b1 e^{-nu1 t}
    return {
        "N": N, "X": X, "mu": mu, "W": W, "A": A,
        "L0": math.log(A * X) / mu / t - 1,
        "L1": T * math.log(A * X) / (X + W - 1) / t - 1,
        "L1_simple": T * math.log(2 * d * X) / (X + 2 * d - 1) / t - 1,
        "two_pole": two / t - 1,
        "two_pole_stored": r["pred_2pole"] / t - 1,
        "three_pole": r["pred_3pole"] / t - 1,
        "L1_stored": r["pred_L1"] / t - 1,
    }


acc = {}
for d in (2, 3):
    for geo in ("CC", "C2M"):
        v = [preds(r) for r in sel(d, geo)]
        big = [x for x in v if x["N"] >= 10]
        o = {"n_sizes": len(v), "n_sizes_N>=10": len(big), "N_list_N>=10": [x["N"] for x in big]}
        for key in ("L0", "L1", "L1_simple", "two_pole", "three_pole"):
            m = max(big, key=lambda x: abs(x[key]))
            o[key] = {"max_abs_N>=10": abs(m[key]), "at_N": m["N"], "signed": m[key],
                      "at_smallest_N>=10": big[0][key], "at_largest_N": big[-1][key],
                      "max_abs_N>=35": max(abs(x[key]) for x in big if x["N"] >= 35),
                      "max_abs_N>=100": max(abs(x[key]) for x in big if x["N"] >= 100)}
        o["max_abs_diff_two_pole_vs_stored"] = max(abs(x["two_pole"] - x["two_pole_stored"]) for x in v)
        o["max_abs_diff_L1_vs_stored"] = max(abs(x["L1"] - x["L1_stored"]) for x in v)
        o["rows"] = [{k: x[k] for k in ("N", "X", "L0", "L1", "L1_simple", "two_pole", "three_pole")} for x in v]
        acc[f"{d}d_{geo}"] = o
# cross-check against the numbers of closed_form_checks.py
for key in ("2d_CC", "3d_CC", "2d_C2M", "3d_C2M"):
    acc[key]["closed_form_check_L1"] = PLAN[key]["L1_weights"]["max_abs_rel_err_N>=10"]
    acc[key]["closed_form_check_L1_simple"] = PLAN[key]["L1_plain"]["max_abs_rel_err_N>=10"]
    assert abs(acc[key]["L1"]["max_abs_N>=10"] - acc[key]["closed_form_check_L1"]) < 1e-12
    assert abs(acc[key]["L1_simple"]["max_abs_N>=10"] - acc[key]["closed_form_check_L1_simple"]) < 1e-12
acc["2d_CC_asymptotic_input"] = PLAN["2d_CC_fully_asymptotic_L1_plain"]
out["accuracy"] = acc

# corner -> centre: which reflecting mode is the slowest one seen by the target, and its weights
c2m = []
for d in (2, 3):
    for r in sel(d, "C2M"):
        N = r["N"]
        mu1 = 2.0 / d * math.sin(math.pi / (2 * N)) ** 2
        c2m.append({"d": d, "N": N, "mu_over_mu1": r["p"][0] / mu1,
                    "mu_minus_(2/d)sin^2(pi/N)": r["p"][0] - 2.0 / d * math.sin(math.pi / N) ** 2,
                    "W_minus_2d": r["W"][0] - 2 * d,
                    "A_minus_2d_cos(pi/N)": r["A_u"][0] - 2 * d * math.cos(math.pi / N)})
out["c2m_weights_check"] = {
    "max_abs_mu_dev": max(abs(x["mu_minus_(2/d)sin^2(pi/N)"]) for x in c2m),
    "max_abs_W_dev": max(abs(x["W_minus_2d"]) for x in c2m),
    "max_abs_A_dev": max(abs(x["A_minus_2d_cos(pi/N)"]) for x in c2m),
    "mu_over_mu1_range": [min(x["mu_over_mu1"] for x in c2m), max(x["mu_over_mu1"] for x in c2m)]}
cc_w = []
for d in (2, 3):
    for r in sel(d, "CC"):
        N = r["N"]
        w = 2 * d * math.cos(math.pi / (2 * N)) ** 2
        cc_w.append(max(abs(r["W"][0] - w), abs(r["A_u"][0] - w),
                        abs(r["p"][0] - 2.0 / d * math.sin(math.pi / (2 * N)) ** 2)))
out["cc_weights_check_max_abs_dev"] = max(cc_w)

# ---------------------------------------------------------------------------------------------
# A2. ingredients of the two-pole argument and error budget of (eq-L1)
# ---------------------------------------------------------------------------------------------
ingr = {}
for d in (2, 3):
    rows = []
    for r in sel(d, "CC"):
        T, mu, W, A = r["mfpt"], r["p"][0], r["W"][0], r["A_u"][0]
        X = mu * T
        nu0, nu1 = r["nu"][0], r["nu"][1]
        b0, b1 = r["res"][0], r["res"][1]
        arg_exact = -b1 * nu1 / (b0 * nu0)
        rate_exact = nu1 - nu0
        rows.append({
            "N": r["N"], "X": X, "nu0_T": nu0 * T, "b0_T": b0 * T, "b1_over_b0": b1 / b0,
            "minus_b1_over_A_b0": -b1 / (A * b0),
            "nu1_over_mu": nu1 / mu, "(nu1/mu-1)X/W": (nu1 / mu - 1) * X / W,
            # error budget of (eq-L1):  t_L1/t* = [ln(AX)/ln(arg_exact)] * [rate_exact/rate_L1] * [t_2pole/t*]
            "log_argument_exact_over_AX": arg_exact / (A * X),
            "factor_log": math.log(A * X) / math.log(arg_exact),
            "factor_rate": rate_exact / (mu * (X + W - 1) / X),
            "factor_third_pole": math.log(arg_exact) / rate_exact / r["mode"],
            "nu1_tmode": nu1 * r["mode"], "ln_2dX": math.log(2 * d * X), "mu_tmode": mu * r["mode"],
        })
    ingr[f"{d}d_CC"] = rows
out["ingredients"] = ingr

# ---------------------------------------------------------------------------------------------
# A4. formal leading laws against exact modes
# ---------------------------------------------------------------------------------------------
K3 = 4.320460169                         # data/msc_modes_verify/extra.json, constants.json
C2 = PLAN["c2_closed_form"]
XC = PLAN["X_const_2d"]
asy = {"ln_8pi": math.log(8 * math.pi), "exp_8pi": math.exp(8 * math.pi),
       "4_over_pi2": 4 / math.pi ** 2, "6_over_pi2": 6 / math.pi ** 2,
       "ln_pi2_K3": math.log(math.pi ** 2 * K3), "6/(pi^2 K3)": 6 / (math.pi ** 2 * K3),
       "pi2_K3_over_6": math.pi ** 2 * K3 / 6, "X_const_2d": XC}
rows2 = []
for r in sel(2, "CC"):
    N = r["N"]
    if N < 10:
        continue
    exact = r["mode"] / N ** 2
    lead = (4 / math.pi ** 2) * (math.log(math.log(N)) + math.log(8 * math.pi))
    ratio_exact = r["mode"] / r["mfpt"]
    ratio_lead = math.log(8 * math.pi * math.log(N)) / (2 * math.pi * math.log(N))
    rows2.append({"N": N, "q_tmode_over_N2_exact": exact, "leading_law": lead, "rel_err": lead / exact - 1,
                  "lnlnN": math.log(math.log(N)), "ratio_exact": ratio_exact, "ratio_leading": ratio_lead,
                  "ratio_rel_err": ratio_lead / ratio_exact - 1})
rows3 = []
for r in sel(3, "CC"):
    N = r["N"]
    if N < 10:
        continue
    exact = r["mode"] / N ** 2
    lead = (6 / math.pi ** 2) * (math.log(N) + math.log(math.pi ** 2 * K3))
    ratio_exact = r["mode"] / r["mfpt"]
    ratio_lead = 6 / (math.pi ** 2 * K3) * (math.log(N) + math.log(math.pi ** 2 * K3)) / N
    rows3.append({"N": N, "q_tmode_over_N2_exact": exact, "leading_law": lead, "rel_err": lead / exact - 1,
                  "ratio_exact": ratio_exact, "ratio_leading": ratio_lead, "ratio_rel_err": ratio_lead / ratio_exact - 1})
asy["2d"] = rows2
asy["3d"] = rows3
sl = FITS["asymptotics"]["2d_slope_d(mode/N^2)/d(lnlnN)"]
asy["2d_local_slope_vs_lnlnN"] = {"N_mid_last4": sl["N_mid"][-4:], "slope_last4": sl["slope"][-4:],
                                  "theory_limit": sl["theory_limit"]}
out["asymptotes"] = asy

# ---------------------------------------------------------------------------------------------
# A5. other placements, exit, activity
# ---------------------------------------------------------------------------------------------
geo = {}
for d in (2, 3):
    for g in ("CC", "C2M", "M2C"):
        rows = []
        for r in sel(d, g):
            row = {"N": r["N"], "tmode_over_N2": r["mode"] / r["N"] ** 2, "ratio": r["mode"] / r["mfpt"],
                   "A": r["A_u"][0], "b1_sign": "+" if r["res"][1] > 0 else "-",
                   "n_sign_changes_down": r["n_sign_changes_down"], "n_sign_changes_up": r["n_sign_changes_up"]}
            if r.get("pred_L1") is not None:
                row["L1"] = r["pred_L1"] / r["mode"] - 1
                row["three_pole"] = r["pred_3pole"] / r["mode"] - 1
            rows.append(row)
        geo[f"{d}d_{g}"] = rows
        dd = sorted((x for x in DIS if x["d"] == d and x["geo"] == g and x["q"] == 0.8 and x.get("mode_over_mfpt")),
                    key=lambda x: x["N"])
        geo[f"{d}d_{g}_discrete_q0.8"] = [{"N": x["N"], "mode": x["mode"], "ratio": x["mode_over_mfpt"]} for x in dd]
ex = {}
for d in (1, 2, 3):
    for q in (0.8, 0.9):
        rows = sorted((x for x in DIS if x["d"] == d and x["geo"] == "EXIT" and x["q"] == q), key=lambda x: x["N"])
        ex[f"{d}d_q{q}"] = [{"N": x["N"], "mode": x["mode"], "mfpt": x["mfpt_exact"], "ratio": x["mode_over_mfpt"],
                             "n_local_maxima": x.get("n_local_maxima")} for x in rows]
    ex[f"{d}d_continuum"] = FITS["continuum_exit"][str(d)]["mode_over_mean"]
geo["exit"] = ex
out["geometries"] = geo

act = []
for x in sorted((x for x in DIS if x["geo"] == "CC" and (x.get("group") == "qscan" or x["q"] == 0.8)),
                key=lambda x: (x["d"], x["N"], x["q"])):
    if (x["d"], x["N"]) in ((2, 21), (2, 51), (2, 101), (3, 11), (3, 21), (3, 31), (1, 51), (1, 201)):
        c = [r for r in LAP if r["d"] == x["d"] and r["geo"] == "CC" and r["N"] == x["N"]]
        act.append({"d": x["d"], "N": x["N"], "q": x["q"], "mode": x["mode"], "q_mode": x["q"] * x["mode"],
                    "tmode_c_unit_rate": c[0]["mode"] if c else None})
out["activity"] = act

# ---------------------------------------------------------------------------------------------
# shape: stored statistics
# ---------------------------------------------------------------------------------------------
out["medians"] = MED
out["cv_full_support_q0.8"] = [{"d": x["d"], "geo": x["geo"], "N": x["N"], "q": x["q"], "cv": x["cv"],
                                "P_T_le_mode": x["P_T_le_mode"], "median": x["median"], "mode": x["mode"],
                                "mean": x["mean_from_pmf"]} for x in TAIL if x["geo"] == "CC"]
out["profile_stats_stored"] = [{"d": x["d"], "N": x["N"], "g_max_T": x["g_max_times_mfpt"],
                                "band99": [x["t_lo_0.99"] / x["mode"], x["t_hi_0.99"] / x["mode"]],
                                "P_T_le_mode": 1 - x["S_at_mode"], "S_T": x["S_at_mfpt"]} for x in PROF]

# ---------------------------------------------------------------------------------------------
# B. density height and 99 % window at the sizes of the shape table (recomputed)
# ---------------------------------------------------------------------------------------------
MT = 20
idx = {(r["d"], r["geo"], r["N"]): r for r in LAP}
SHAPE = {1: [1000], 2: [35, 200, 163840], 3: [40, 100, 1280]}
shape = []
for d, Ns in SHAPE.items():
    for N in Ns:
        r = idx[(d, "CC", N)]
        T, mode = r["mfpt"], r["mode"]
        if d == 1:
            ch = F.Chain1D(N, 1.0)
            gfun = lambda tt, ch=ch: ch.g_cont(np.atleast_1d(tt))
            S = lambda t, ch=ch: float((ch.W / ch.mu * np.exp(-ch.mu * t)).sum())
        else:
            L = F.LaplaceFP(N, d, 1.0, F.corner(N, d), F.far_corner(N, d))
            L.prepare(2.0 * MT * MT / (5.0 * 1e-3 * mode) * 1.05)
            gfun = lambda tt, L=L: F.talbot_invert(L.Fhat, np.atleast_1d(tt), M=MT)
            S = lambda t, L=L: float(F.talbot_invert(lambda s: (1.0 - L.Fhat(s)) / s, [t], M=MT)[0])
        gmax = float(gfun(mode)[0])
        fl = lambda tt: float(gfun(tt)[0]) - 0.99 * gmax
        lo = brentq(fl, mode * 0.2, mode, rtol=1e-10)
        hi = brentq(fl, mode, 60.0 * mode, rtol=1e-10)
        m = [x for x in MED if x["d"] == d and x["N"] == N][0]
        rec = {"d": d, "N": N, "ratio": mode / T, "median_over_T": m["median_over_mfpt"],
               "P_T_le_mode": m["P_T_le_mode"], "S_T": m["S_at_mfpt"], "S_mode": m["S_at_mode"],
               "g_max_T": gmax * T, "band99_lo": lo / mode, "band99_hi": hi / mode,
               "S_mode_recomputed": S(mode), "S_T_recomputed": S(T)}
        rec["abs_diff_S_mode"] = abs(rec["S_mode_recomputed"] - m["S_at_mode"])
        shape.append(rec)
        print("shape", rec, flush=True)
out["shape_table"] = shape
# consistency with the stored profile statistics at the sizes that overlap
ov = []
for x in PROF:
    for y in shape:
        if x["d"] == y["d"] and x["N"] == y["N"]:
            ov.append({"d": x["d"], "N": x["N"], "g_max_T_diff": abs(x["g_max_times_mfpt"] - y["g_max_T"]),
                       "band_lo_diff": abs(x["t_lo_0.99"] / x["mode"] - y["band99_lo"]),
                       "band_hi_diff": abs(x["t_hi_0.99"] / x["mode"] - y["band99_hi"])})
out["shape_overlap_with_profile_stats"] = ov


# ---------------------------------------------------------------------------------------------
# C. factorisation checked by dense linear algebra; convergence of the arrival profile
# ---------------------------------------------------------------------------------------------
def P1_dense(N, d):
    """q = 1 transition matrix of the walk with cancelled moves on {0..N-1}^d (dense)."""
    n = N ** d
    idx_ = np.arange(n).reshape((N,) * d)
    P = np.zeros((n, n))
    for site in np.ndindex(*(N,) * d):
        i = idx_[site]
        for ax in range(d):
            for sg in (-1, 1):
                nb = list(site)
                nb[ax] += sg
                if 0 <= nb[ax] < N:
                    P[i, idx_[tuple(nb)]] += 1.0 / (2 * d)
                else:
                    P[i, i] += 1.0 / (2 * d)
    return P, idx_


fac = []
for (N, d, o, a) in ((6, 2, (0, 0), (5, 5)), (5, 2, (1, 3), (2, 2)), (4, 3, (0, 0, 0), (3, 3, 3)), (4, 3, (1, 0, 2), (3, 1, 0))):
    P, idx_ = P1_dense(N, d)
    n = N ** d
    ia, io = idx_[a], idx_[o]
    keep = [i for i in range(n) if i != ia]
    Qm = P[np.ix_(keep, keep)]
    r_abs = P[keep, ia]                               # one-step absorption rates (unit jump rate)
    worst = 0.0
    for s in (0.003, 0.05, 0.7, 0.02 + 0.3j):
        # first-passage transforms from every start: (s + I - Q) F = r
        Fx = np.linalg.solve((s + 1) * np.eye(n - 1) - Qm, r_abs.astype(complex))
        Fall = np.ones(n, dtype=complex)
        Fall[keep] = Fx                                # start on the target: T = 0, transform 1
        G = np.linalg.inv((s + 1) * np.eye(n) - P)     # reflecting resolvent  \hat P(x, s | y)
        u_hat = n * G[ia, io]
        h_hat = 1.0 / (n * s * G[ia, ia])
        worst = max(worst, abs(Fall[io] - s * u_hat * h_hat) / abs(Fall[io]),
                    abs(Fall.mean() - h_hat) / abs(h_hat),
                    abs(G[ia, :].sum() - 1 / s) * abs(s))
    fac.append({"N": N, "d": d, "start": o, "target": a, "max_rel_dev": worst})
out["factorisation_check"] = fac


def theta4_pow(x, d):
    k = np.arange(1, 200)[:, None]
    return (1 + 2 * ((-1.0) ** k * np.exp(-k ** 2 * x[None, :])).sum(axis=0)) ** d


def u_cc(N, d, x):
    """arrival profile for corner to corner at t = x / mu_1 (unit rate)."""
    k = np.arange(1, N)[:, None]
    mu1 = 2.0 / d * math.sin(math.pi / (2 * N)) ** 2
    eps = 2.0 / d * np.sin(np.pi * k / (2 * N)) ** 2
    c2 = np.cos(np.pi * k / (2 * N)) ** 2
    one = 1 + 2 * ((-1.0) ** k * c2 * np.exp(-eps * x[None, :] / mu1)).sum(axis=0)
    return one ** d


xg = np.linspace(0.05, 12, 240)
out["arrival_profile_convergence"] = [
    {"d": d, "N": N, "max_abs_dev_from_theta4^d": float(np.max(np.abs(u_cc(N, d, xg) - theta4_pow(xg, d))))}
    for d in (2, 3) for N in (10, 100, 1000)]

with open(OUT, "w") as f:
    json.dump(out, f, indent=1)

# ---------------------------------------------------------------------------------------------
# printed summary
# ---------------------------------------------------------------------------------------------
print("\n== accuracy (max |rel err| over computed sizes N >= 10)")
for key, o in acc.items():
    if "rows" not in o:
        continue
    print(key, "sizes N>=10:", o["n_sizes_N>=10"],
          {k: ("%.3e at N=%d" % (o[k]["max_abs_N>=10"], o[k]["at_N"])) for k in ("L0", "L1", "L1_simple", "two_pole", "three_pole")})
    print("    largest N:", {k: "%.3e" % o[k]["at_largest_N"] for k in ("L0", "L1", "L1_simple", "two_pole", "three_pole")},
          "| L1 N>=35: %.2e N>=100: %.2e | L1s N>=35: %.2e N>=100: %.2e" %
          (o["L1"]["max_abs_N>=35"], o["L1"]["max_abs_N>=100"], o["L1_simple"]["max_abs_N>=35"], o["L1_simple"]["max_abs_N>=100"]))
print("\n== accuracy rows")
for key in ("2d_CC", "3d_CC"):
    for x in acc[key]["rows"]:
        print(key, "N=%d X=%.3f L0=%+.3f%% L1=%+.4f%% L1s=%+.4f%% 2p=%+.4f%% 3p=%+.2e" %
              (x["N"], x["X"], 100 * x["L0"], 100 * x["L1"], 100 * x["L1_simple"], 100 * x["two_pole"], x["three_pole"]))
print("\n== ingredients")
for key, rows in ingr.items():
    for x in rows:
        if x["N"] in (5, 10, 35, 40, 100, 113, 1280, 4096, 1048576, 4194304, 16777216):
            print(key, {k: (round(v, 5) if isinstance(v, float) else v) for k, v in x.items()})
print("\n== asymptotes")
print({k: v for k, v in asy.items() if not isinstance(v, (list, dict))})
for x in rows2:
    if x["N"] in (10, 100, 1280, 10240, 1048576, 16777216):
        print("2d", {k: round(v, 5) for k, v in x.items()})
for x in rows3:
    if x["N"] in (10, 40, 100, 1280, 4096):
        print("3d", {k: round(v, 5) for k, v in x.items()})
print(asy["2d_local_slope_vs_lnlnN"])
print("\n== checks")
print("c2m weights:", out["c2m_weights_check"], "cc weights dev:", out["cc_weights_check_max_abs_dev"])
print("factorisation:", fac)
print("arrival profile:", out["arrival_profile_convergence"])
print("shape overlap:", ov)
print("\n== geometries")
for key in ("2d_C2M", "2d_M2C", "3d_C2M", "3d_M2C"):
    for x in geo[key]:
        if x["N"] in (11, 41, 61, 161, 201, 641, 1281, 10241, 655361):
            print(key, {k: (round(v, 6) if isinstance(v, float) else v) for k, v in x.items()})
for key in ("2d_M2C_discrete_q0.8", "3d_M2C_discrete_q0.8", "2d_C2M_discrete_q0.8", "3d_C2M_discrete_q0.8"):
    print(key, geo[key][-1])
print("exit:", {k: (v[-1] if isinstance(v, list) else v) for k, v in ex.items()})
print("\n== activity")
for x in act:
    print(x)
print("\nwrote", OUT)


# ---------------------------------------------------------------------------------------------
# LaTeX rows of the three tables of Section 6 (copied by hand into sections/s6_mechanism.tex)
# ---------------------------------------------------------------------------------------------
def sci(x, nd=1):
    """signed number in the form +a.b x 10^{e} for LaTeX math mode."""
    if x == 0:
        return "0"
    e = int(math.floor(math.log10(abs(x))))
    m = x / 10 ** e
    if abs(round(m, nd)) >= 10:
        m /= 10
        e += 1
    return ("%+.*f" % (nd, m))+ r"\times10^{%d}" % e


def pct(x, nd):
    return "%+.*f" % (nd, 100 * x)


def Nfmt(N):
    s = "%d" % N
    if len(s) <= 4:
        return s
    parts = []
    while s:
        parts.append(s[-3:])
        s = s[:-3]
    return r"\,".join(reversed(parts))


def Xfmt(X):
    """X with three decimals; integer part grouped in threes when it has five or more digits."""
    whole, frac = ("%.3f" % X).split(".")
    return Nfmt(int(whole)) + "." + frac


lines = ["% generated by code/article/s6_mechanism_tables.py -- do not edit by hand", "% --- tab-accuracy"]
ROWS_ACC = {2: [10, 35, 113, 1280, 10240, 1048576, 16777216], 3: [10, 35, 100, 320, 1280, 4096]}
for d in (2, 3):
    lines.append("%% d = %d" % d)
    for x in acc[f"{d}d_CC"]["rows"]:
        if x["N"] in ROWS_ACC[d]:
            lines.append(r"%s & %s & $%s$ & $%s$ & $%s$ & $%s$ & $%s$ \\" % (
                Nfmt(x["N"]), Xfmt(x["X"]), pct(x["L0"], 3), pct(x["L1"], 4), pct(x["L1_simple"], 4),
                pct(x["two_pole"], 4), sci(x["three_pole"])))
lines.append("% --- tab-shape")
for r in shape:
    lines.append(r"%d & %s & %.4f & %.4f & %.4f & %.4f & %.4f & $[%.2f,\,%.2f]$ \\" % (
        r["d"], Nfmt(r["N"]), r["ratio"], r["P_T_le_mode"], r["median_over_T"], r["S_T"],
        r["g_max_T"], r["band99_lo"], r["band99_hi"]))
lines.append("% --- tab-geometries (a) point targets")
ROWS_GEO = {(2, "C2M"): [11, 161, 10241, 655361], (2, "M2C"): [11, 161, 10241, 655361],
            (3, "C2M"): [11, 161, 1281], (3, "M2C"): [11, 161, 1281]}
for (d, g), Ns in ROWS_GEO.items():
    for x in geo[f"{d}d_{g}"]:
        if x["N"] in Ns:
            l1 = ("$%s$" % pct(x["L1"], 4)) if "L1" in x else "--"
            lines.append(r"%d & %s & %s & %.4f & %.5f & $%s$ & %s \\" % (
                d, {"C2M": r"\CtM", "M2C": r"\MtC"}[g], Nfmt(x["N"]), x["tmode_over_N2"], x["ratio"],
                "-" if x["b1_sign"] == "-" else "+", l1))
lines.append("% --- tab-geometries (b) exit")
for d in (1, 2, 3):
    lat = ex[f"{d}d_q0.8"][-1]
    mf = "%.1f" % lat["mfpt"]
    lines.append(r"%d & \EXIT & %d & %.6f & %.5f & %s & %s \\" % (
        d, lat["N"], ex[f"{d}d_continuum"], lat["ratio"], Nfmt(lat["mode"]), Nfmt(int(mf.split(".")[0])) + "." + mf.split(".")[1]))
with open(_os.path.join(_R, 'data', 'article', 's6_mechanism_rows.tex'), "w") as f:
    f.write("\n".join(lines) + "\n")
print("\n".join(lines))
