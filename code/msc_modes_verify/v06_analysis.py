#!/usr/bin/env python
"""Check by the second implementation, part 6: compare with the results of the main implementation, test the closed forms,
redo the scaling fits.  Output: ../data/analysis.json and ../data/tables.md."""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import json, os, sys, collections
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vspec

HERE = os.path.dirname(os.path.abspath(__file__))
V = _os.path.join(_R, 'data', 'msc_modes_verify')
R = _os.path.join(_R, 'data', 'msc_modes')          # data of the main implementation (read only)
A = {}
md = []


def jl(p):
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


recheck_d = jl(_os.path.join(_R, 'data', 'msc_modes_verify', 'discrete.jsonl'))
recheck_p = jl(_os.path.join(_R, 'data', 'msc_modes_verify', 'poles.jsonl'))
their_d = jl(_os.path.join(_R, 'data', 'msc_modes', 'discrete_modes.jsonl'))
their_l = jl(_os.path.join(_R, 'data', 'msc_modes', 'laplace_modes.jsonl'))
misc = json.load(open(_os.path.join(_R, 'data', 'msc_modes_verify', 'misc.json'))) if os.path.exists(_os.path.join(_R, 'data', 'msc_modes_verify', 'misc.json')) else {}

# ------------------------------------------------------------------------------------------
# 1. discrete modes: this check vs the main implementation (every overlapping case)
tmap = {}
for r in their_d:
    tmap.setdefault((r["d"], r["N"], round(r["q"], 6), r["geo"]), r)
cmp_rows, mism = [], []
seen = set()
for r in recheck_d:
    k = (r["d"], r["N"], round(r["q"], 6), r["geom"])
    if k in tmap and k not in seen:
        seen.add(k)
        t = tmap[k]
        cmp_rows.append((k, r["mode"], t["mode"]))
        if r["mode"] != t["mode"]:
            mism.append((k, r["mode"], t["mode"]))
A["discrete_compare"] = dict(n_overlap=len(cmp_rows), n_mismatch=len(mism), mismatches=[[str(k), a, b] for k, a, b in mism])
A["discrete_unimodal"] = dict(
    n_runs=len(recheck_d),
    n_runs_q_le_09=len([r for r in recheck_d if r["q"] <= 0.9]),
    multi_max_q_le_09=[r["key"] for r in recheck_d if r["q"] <= 0.9 and r["n_local_max"] != 1],
    multi_max_q_1=[(r["key"], r["n_local_max"]) for r in recheck_d if r["q"] > 0.95 and r["n_local_max"] != 1],
    ties=[r["key"] for r in recheck_d if r["ties"] != 1],
    not_monotone_after_mode=[r["key"] for r in recheck_d if not r["monotone_after_mode"] and r["q"] <= 0.9],
    full_support_runs=[dict(key=r["key"], n_max=r["n_local_max"], n_min=r["n_local_min"], cv=r.get("cv"), cdf_at_mode=r["cdf_at_mode"],
                            median=r.get("median"), mean=r.get("mean_from_pmf"), S_at_mean=r.get("S_at_mean"), mode=r["mode"])
                       for r in recheck_d if r["stop"] <= 0])

# ------------------------------------------------------------------------------------------
# 2. exact MFPT for the discrete CC runs of this check (spectral, validated) and ratios
def mfpt_cc(d, N, q):
    if d == 1:
        return N * (N - 1) / q
    return vspec.Corner(d, N).mfpt() / q

cc = {}
for r in recheck_d:
    if r["geom"] == "CC" and abs(r["q"] - 0.8) < 1e-9 and r["stop"] > 0 and "long" not in r["key"]:
        cc[(r["d"], r["N"])] = r
tab = collections.defaultdict(list)
for (d, N), r in sorted(cc.items()):
    m = mfpt_cc(d, N, 0.8)
    tab[d].append(dict(N=N, mode=r["mode"], mfpt=m, ratio=r["mode"] / m, qmode_over_N2=0.8 * r["mode"] / N ** 2,
                       mode_parabolic=r["mode_parabolic"], cdf_at_mode=r["cdf_at_mode"]))
A["cc_q0.8"] = {str(d): v for d, v in tab.items()}

# ------------------------------------------------------------------------------------------
# 3. pole-expansion modes vs Talbot modes of the main implementation; vs the stepper of this check
tl = {}
for r in their_l:
    if r["geo"] == "CC":
        tl[(r["d"], r["N"])] = r
pol = {}
for r in recheck_p:
    if "conv" in r["key"] or "farcheck" in r["key"]:
        continue
    pol[(r["d"], r["N"])] = r
rows = []
for k, r in sorted(pol.items()):
    row = dict(d=k[0], N=k[1], mode=r["mode"], mfpt=r["mfpt"], ratio=r["mode"] / r["mfpt"], mode_over_N2=r["mode"] / k[1] ** 2,
               X=r["X"], n_poles=r["n_poles"],
               L0_err=r["L0"] / r["mode"] - 1, L1_err=r["L1"] / r["mode"] - 1, L1simple_err=r["L1_simple"] / r["mode"] - 1,
               pole2_err=r.get("mode_2pole", np.nan) / r["mode"] - 1, pole3_err=r.get("mode_3pole", np.nan) / r["mode"] - 1,
               cdf_at_mode=r["cdf_at_mode"], median_over_mfpt=r["median"] / r["mfpt"], S_at_mfpt=r["S_at_mfpt"],
               gmax_mfpt=r["gmax"] * r["mfpt"], band99=[r["band99_lo"], r["band99_hi"]],
               nu0_mfpt=r["nu"][0] * r["mfpt"], nu1_over_mu1=r["nu"][1] / r["mu1"], a1_over_a0=r["res"][1] / r["res"][0])
    if k in tl:
        row["their_mode"] = tl[k]["mode"]; row["rel_diff_mode"] = r["mode"] / tl[k]["mode"] - 1
        row["rel_diff_mfpt"] = r["mfpt"] / tl[k]["mfpt"] - 1
    if k in cc:
        row["disc_mode_stepper"] = cc[k]["mode"]; row["disc_mode_poles"] = r["mode_disc_q0.8"]
        row["disc_real_poles"] = r["mode_disc_real_q0.8"]; row["disc_parabolic_stepper"] = cc[k]["mode_parabolic"]
        row["offset_q*disc-cont"] = 0.8 * cc[k]["mode"] - r["mode"]
    rows.append(row)
A["poles_cc"] = rows
for d in (2, 3):
    rr = [x for x in rows if x["d"] == d]
    A[f"summary_d{d}"] = dict(
        max_rel_diff_mode_vs_talbot=max([abs(x["rel_diff_mode"]) for x in rr if "rel_diff_mode" in x], default=None),
        max_rel_diff_mfpt=max([abs(x["rel_diff_mfpt"]) for x in rr if "rel_diff_mfpt" in x], default=None),
        n_compared=len([x for x in rr if "rel_diff_mode" in x]),
        disc_mode_mismatch_poles_vs_stepper=[(x["N"], x["disc_mode_poles"], x["disc_mode_stepper"]) for x in rr
                                             if "disc_mode_stepper" in x and x["disc_mode_poles"] != x["disc_mode_stepper"]],
        max_abs_diff_real_maximiser_N_ge_30=max([abs(x["disc_real_poles"] - x["disc_parabolic_stepper"]) for x in rr
                                                 if "disc_mode_stepper" in x and x["N"] >= 30 and x["disc_parabolic_stepper"]], default=None),
        offsets=[(x["N"], x["offset_q*disc-cont"]) for x in rr if "disc_mode_stepper" in x],
        L1_max_abs_err_N_ge_10=max(abs(x["L1_err"]) for x in rr if x["N"] >= 10),
        L1simple_max_abs_err_N_ge_10=max(abs(x["L1simple_err"]) for x in rr if x["N"] >= 10),
        L1_argmax_N=max((x for x in rr if x["N"] >= 10), key=lambda x: abs(x["L1_err"]))["N"],
        L1_max_abs_err_N_ge_100=max(abs(x["L1_err"]) for x in rr if x["N"] >= 100),
        L0_err_range_N_ge_10=[max(x["L0_err"] for x in rr if x["N"] >= 10), min(x["L0_err"] for x in rr if x["N"] >= 10)],
        pole2_max_abs_err_N_ge_10=max(abs(x["pole2_err"]) for x in rr if x["N"] >= 10),
        pole3_max_abs_err_N_ge_10=max(abs(x["pole3_err"]) for x in rr if x["N"] >= 10))

# ------------------------------------------------------------------------------------------
# 4. fits
def ols(Xm, y):
    n, k = Xm.shape
    beta, *_ = np.linalg.lstsq(Xm, y, rcond=None)
    resid = y - Xm @ beta
    rss = float(resid @ resid)
    s2 = rss / max(n - k, 1)
    cov = s2 * np.linalg.inv(Xm.T @ Xm)
    ci = stats.t.ppf(0.975, max(n - k, 1)) * np.sqrt(np.diag(cov))
    aicc = n * np.log(rss / n) + 2 * k + (2 * k * (k + 1) / (n - k - 1) if n - k - 1 > 0 else np.inf)
    return beta, ci, np.sqrt(rss / n), aicc


def fit_models(N, y, models, extrap=None):
    """Least squares in ln y.  Returns dict model -> (params, ci, rms, aicc, extrapolation errors)."""
    N = np.asarray(N, float); ly = np.log(np.asarray(y, float)); lN = np.log(N); llN = np.log(lN)
    one = np.ones_like(N)
    out = {}
    for mname in models:
        if mname == "A N^2":
            b, ci, rms, aicc = ols(one[:, None], ly - 2 * lN); pred = lambda n, b=b: np.exp(b[0]) * n ** 2
        elif mname == "A N^3":
            b, ci, rms, aicc = ols(one[:, None], ly - 3 * lN); pred = lambda n, b=b: np.exp(b[0]) * n ** 3
        elif mname == "A N^a":
            b, ci, rms, aicc = ols(np.c_[one, lN], ly); pred = lambda n, b=b: np.exp(b[0]) * n ** b[1]
        elif mname == "A N^2 (ln N)^b":
            b, ci, rms, aicc = ols(np.c_[one, llN], ly - 2 * lN); pred = lambda n, b=b: np.exp(b[0]) * n ** 2 * np.log(n) ** b[1]
        elif mname == "A N^a (ln N)^b":
            b, ci, rms, aicc = ols(np.c_[one, lN, llN], ly); pred = lambda n, b=b: np.exp(b[0]) * n ** b[1] * np.log(n) ** b[2]
        elif mname in ("N^2 (A lnln N + B)", "N^2 (A ln N + B)", "N^3 (A + B/N)", "A (N-delta)^2"):
            if mname == "N^2 (A lnln N + B)":
                fn = lambda n, a, b_: np.log(n ** 2 * (a * np.log(np.log(n)) + b_)); p0 = (0.5, 1.0)
            elif mname == "N^2 (A ln N + B)":
                fn = lambda n, a, b_: np.log(n ** 2 * (a * np.log(n) + b_)); p0 = (0.5, 1.0)
            elif mname == "N^3 (A + B/N)":
                fn = lambda n, a, b_: np.log(n ** 3 * (a + b_ / n)); p0 = (4.0, -1.0)
            else:
                fn = lambda n, a, b_: np.log(a * (n - b_) ** 2); p0 = (0.4, 0.5)
            p, pc = curve_fit(fn, N, ly, p0=p0, maxfev=20000)
            resid = ly - fn(N, *p); n_, k = len(N), 2
            rss = float(resid @ resid); rms = np.sqrt(rss / n_)
            ci = stats.t.ppf(0.975, n_ - k) * np.sqrt(np.diag(pc))
            aicc = n_ * np.log(rss / n_) + 2 * k + 2 * k * (k + 1) / (n_ - k - 1)
            b = p; pred = lambda n, p=p, fn=fn: np.exp(fn(np.asarray(n, float), *p))
        out[mname] = dict(params=[float(x) for x in b], ci95=[float(x) for x in ci], rms_ln=float(rms), aicc=float(aicc))
        if extrap:
            out[mname]["extrap_err"] = {str(int(n)): float(pred(float(n)) / yv - 1) for n, yv in extrap.items()}
    amin = min(v["aicc"] for v in out.values())
    for v in out.values():
        v["dAICc"] = v["aicc"] - amin
    return out


F = {}
# short-range fits on exact discrete modes (q = 0.8), N >= 10 from the size lists of the discrete runs
for d, models in ((1, ["A N^2", "A N^a", "A (N-delta)^2"]),
                  (2, ["A N^2", "A N^a", "A N^2 (ln N)^b", "A N^a (ln N)^b", "N^2 (A lnln N + B)", "N^2 (A ln N + B)"]),
                  (3, ["A N^3", "A N^2", "A N^a", "A N^2 (ln N)^b", "A N^a (ln N)^b", "N^2 (A ln N + B)"])):
    hi = 200 if d < 3 else 100
    pts = [(x["N"], x["mode"]) for x in tab[d] if 10 <= x["N"] <= hi]
    if len(pts) > 4:
        N_, y_ = zip(*pts)
        F[f"discrete_d{d}_short_range"] = dict(N=list(N_), fits=fit_models(N_, y_, models))
        # window dependence of the pure power-law exponent
        wins = {}
        for lo_, hi_ in ((5, 30), (10, 60), (30, hi), (10, hi), (5, hi)):
            sel = [(n, y) for n, y in zip(*zip(*[(x["N"], x["mode"]) for x in tab[d]])) if lo_ <= n <= hi_]
            if len(sel) > 3:
                n2, y2 = zip(*sel)
                ff = fit_models(n2, y2, ["A N^a"])["A N^a"]
                wins[f"{lo_}-{hi_}"] = [ff["params"][1], ff["ci95"][1]]
        F[f"discrete_d{d}_window_dependence_of_exponent"] = wins

# continuous-time (pole) modes: fit on the short-range window, extrapolate
for d, lo, hi, models, ex_list in ((2, 10, 200, ["A N^2", "A N^a", "A N^2 (ln N)^b", "A N^a (ln N)^b", "N^2 (A lnln N + B)", "N^2 (A ln N + B)"],
                                    (1280, 10240, 163840, 16777216)),
                                   (3, 10, 100, ["A N^3", "A N^2", "A N^a", "A N^2 (ln N)^b", "A N^a (ln N)^b", "N^2 (A ln N + B)"],
                                    (320, 1280, 4096))):
    rr = [x for x in rows if x["d"] == d]
    pts = [(x["N"], x["mode"]) for x in rr if lo <= x["N"] <= hi]
    ex = {x["N"]: x["mode"] for x in rr if x["N"] in ex_list}
    if len(pts) > 4:
        N_, y_ = zip(*pts)
        F[f"cont_d{d}_mode_fit_{lo}-{hi}"] = dict(N=list(N_), fits=fit_models(N_, y_, models, ex))
        ptsm = [(x["N"], x["mfpt"]) for x in rr if lo <= x["N"] <= hi]
        exm = {x["N"]: x["mfpt"] for x in rr if x["N"] in ex_list}
        mm = ["A N^2", "A N^a", "A N^2 (ln N)^b", "N^2 (A ln N + B)"] if d == 2 else ["A N^3", "A N^a", "N^3 (A + B/N)"]
        F[f"cont_d{d}_mfpt_fit_{lo}-{hi}"] = dict(fits=fit_models(*zip(*ptsm), mm, exm))
        # closed-form extrapolation errors at the same sizes
        F[f"cont_d{d}_L1_err_at_extrap"] = {str(x["N"]): x["L1_err"] for x in rr if x["N"] in ex_list}
    # local exponents (centred finite differences on the log grid)
    Ns = np.array([x["N"] for x in rr], float); mo = np.array([x["mode"] for x in rr]); mf = np.array([x["mfpt"] for x in rr])
    if len(Ns) > 3:
        le = [(float(np.sqrt(Ns[i - 1] * Ns[i + 1])), float(np.log(mo[i + 1] / mo[i - 1]) / np.log(Ns[i + 1] / Ns[i - 1])),
               float(np.log(mf[i + 1] / mf[i - 1]) / np.log(Ns[i + 1] / Ns[i - 1]))) for i in range(1, len(Ns) - 1)]
        F[f"local_exponents_d{d}"] = le
        # local slopes of mode/N^2 against ln ln N (2D) or ln N (3D)
        xx = np.log(np.log(Ns)) if d == 2 else np.log(Ns)
        yy = mo / Ns ** 2
        F[f"local_slope_d{d}"] = [(float(np.sqrt(Ns[i - 1] * Ns[i + 1])), float((yy[i + 1] - yy[i - 1]) / (xx[i + 1] - xx[i - 1]))) for i in range(1, len(Ns) - 1)]
A["fits"] = F

json.dump(A, open(_os.path.join(_R, 'data', 'msc_modes_verify', 'analysis.json'), "w"), indent=1, default=float)

# ---- compact printed report -------------------------------------------------------------------
print("discrete compare:", A["discrete_compare"])
print("unimodal:", {k: v for k, v in A["discrete_unimodal"].items() if k != "full_support_runs"})
for d in (2, 3):
    if f"summary_d{d}" in A:
        print(f"summary d={d}:", json.dumps(A[f"summary_d{d}"], default=float))
for k, v in F.items():
    if "fits" in v if isinstance(v, dict) else False:
        print("\n", k)
        for m, f in v["fits"].items():
            print(f"   {m:22s} params={np.round(f['params'], 5)} ci={np.round(f['ci95'], 5)} rms={f['rms_ln']:.2e} dAICc={f['dAICc']:.1f}",
                  " extrap:", {n: f"{100 * e:+.2f}%" for n, e in f.get("extrap_err", {}).items()})
    else:
        print("\n", k, v if not isinstance(v, list) else [tuple(np.round(x, 4)) for x in v[:: max(1, len(v) // 12)]])
