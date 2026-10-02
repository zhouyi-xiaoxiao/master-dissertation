"""
04_fit_scaling.py -- scaling analysis of the exact modes and means.

Inputs : data/discrete_modes.jsonl (exact discrete-time PMF, q = 0.8)
         data/laplace_modes.jsonl  (exact continuous-time density, unit rate)
Outputs: data/fit_results.json, data/tables.md

The data are deterministic (no sampling noise).  'Confidence intervals' below
are ordinary least-squares 95% intervals (Student t) computed from the scatter
of the residuals about the fitted model; they quantify how well a functional
form describes the exact data over the stated window, not statistical noise.
Model comparison uses AICc/BIC (Gaussian residuals in ln mode) and, more
tellingly, out-of-sample extrapolation to sizes far outside the fit window.
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import json, os, sys, math
import numpy as np
from scipy import stats
from scipy.optimize import least_squares, brentq
import mpmath as mp

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = _os.path.join(_R, 'data', 'msc_modes')
Drows = [json.loads(l) for l in open(_os.path.join(_R, 'data', 'msc_modes', 'discrete_modes.jsonl')) if l.strip()]
Lrows = [json.loads(l) for l in open(_os.path.join(_R, 'data', 'msc_modes', 'laplace_modes.jsonl')) if l.strip()]
OUT = {}
MD = []


def sel(rows, **kw):
    out = [r for r in rows if all(r.get(k) == v for k, v in kw.items())]
    return sorted(out, key=lambda r: r["N"])


# ---------------------------------------------------------------------------
# generic fitting utilities (fits are done on y = ln(quantity))
# ---------------------------------------------------------------------------
def ols(X, y):
    n, k = X.shape
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    rss = float(res @ res)
    dof = max(n - k, 1)
    cov = rss / dof * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    tq = stats.t.ppf(0.975, dof)
    return beta, se, tq * se, res, rss


def info(n, k, rss):
    kk = k + 1                                   # + variance
    rss = max(rss, 1e-300)
    aic = n * math.log(rss / n) + 2 * kk
    aicc = aic + (2 * kk * (kk + 1)) / max(n - kk - 1, 1)
    bic = n * math.log(rss / n) + kk * math.log(n)
    return aicc, bic


def nl_fit(fun, p0, N, y):
    r = least_squares(lambda p: fun(p, N) - y, p0, xtol=1e-14, ftol=1e-14, gtol=1e-14)
    res = r.fun
    n, k = len(y), len(p0)
    rss = float(res @ res)
    J = r.jac
    dof = max(n - k, 1)
    try:
        cov = rss / dof * np.linalg.inv(J.T @ J)
        se = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        se = np.full(k, np.nan)
    tq = stats.t.ppf(0.975, dof)
    return r.x, se, tq * se, res, rss


MODELS = {}


def model(name, npar, desc):
    def deco(f):
        MODELS[name] = (f, npar, desc)
        return f
    return deco


@model("P", 2, "A N^a")
def m_P(N, y):
    x = np.log(N)
    b, se, ci, res, rss = ols(np.c_[np.ones_like(x), x], y)
    return dict(params={"lnA": b[0], "a": b[1]}, ci95={"lnA": ci[0], "a": ci[1]}, res=res, rss=rss,
                predict=lambda M: b[0] + b[1] * np.log(M))


@model("L2", 2, "A N^2 (ln N)^b")
def m_L2(N, y):
    x = np.log(N)
    z = np.log(x)
    b, se, ci, res, rss = ols(np.c_[np.ones_like(x), z], y - 2 * x)
    return dict(params={"lnA": b[0], "b": b[1]}, ci95={"lnA": ci[0], "b": ci[1]}, res=res, rss=rss,
                predict=lambda M: b[0] + 2 * np.log(M) + b[1] * np.log(np.log(M)))


@model("PL", 3, "A N^a (ln N)^b")
def m_PL(N, y):
    x = np.log(N)
    z = np.log(x)
    b, se, ci, res, rss = ols(np.c_[np.ones_like(x), x, z], y)
    return dict(params={"lnA": b[0], "a": b[1], "b": b[2]}, ci95={"lnA": ci[0], "a": ci[1], "b": ci[2]}, res=res,
                rss=rss, predict=lambda M: b[0] + b[1] * np.log(M) + b[2] * np.log(np.log(M)))


@model("C2", 1, "A N^2 (pure quadratic)")
def m_C2(N, y):
    x = np.log(N)
    b, se, ci, res, rss = ols(np.ones((len(x), 1)), y - 2 * x)
    return dict(params={"lnA": b[0]}, ci95={"lnA": ci[0]}, res=res, rss=rss, predict=lambda M: b[0] + 2 * np.log(M))


@model("C3", 1, "A N^3 (pure cubic)")
def m_C3(N, y):
    x = np.log(N)
    b, se, ci, res, rss = ols(np.ones((len(x), 1)), y - 3 * x)
    return dict(params={"lnA": b[0]}, ci95={"lnA": ci[0]}, res=res, rss=rss, predict=lambda M: b[0] + 3 * np.log(M))


@model("LN2", 2, "N^2 (A ln N + B)")
def m_LN2(N, y):
    f = lambda p, M: 2 * np.log(M) + np.log(np.abs(p[0] * np.log(M) + p[1]))
    p, se, ci, res, rss = nl_fit(f, [0.5, 1.0], N, y)
    return dict(params={"A": p[0], "B": p[1]}, ci95={"A": ci[0], "B": ci[1]}, res=res, rss=rss,
                predict=lambda M: f(p, M))


@model("LL2", 2, "N^2 (A ln ln N + B)")
def m_LL2(N, y):
    f = lambda p, M: 2 * np.log(M) + np.log(np.abs(p[0] * np.log(np.log(M)) + p[1]))
    p, se, ci, res, rss = nl_fit(f, [0.5, 1.0], N, y)
    return dict(params={"A": p[0], "B": p[1]}, ci95={"A": ci[0], "B": ci[1]}, res=res, rss=rss,
                predict=lambda M: f(p, M))


@model("LN3", 2, "N^3 (A + B/N)")
def m_LN3(N, y):
    f = lambda p, M: 3 * np.log(M) + np.log(np.abs(p[0] + p[1] / M))
    p, se, ci, res, rss = nl_fit(f, [1.0, 0.0], N, y)
    return dict(params={"A": p[0], "B": p[1]}, ci95={"A": ci[0], "B": ci[1]}, res=res, rss=rss,
                predict=lambda M: f(p, M))


@model("Q2", 2, "A (N - delta)^2")
def m_Q2(N, y):
    f = lambda p, M: np.log(np.abs(p[0])) + 2 * np.log(M - p[1])
    p, se, ci, res, rss = nl_fit(f, [0.3, 0.5], N, y)
    return dict(params={"A": p[0], "delta": p[1]}, ci95={"A": ci[0], "delta": ci[1]}, res=res, rss=rss,
                predict=lambda M: f(p, M))


def run_models(names, N, val, Nhold=None, valhold=None):
    N = np.asarray(N, float); y = np.log(np.asarray(val, float))
    out = {}
    for nm in names:
        f, k, desc = MODELS[nm]
        r = f(N, y)
        aicc, bic = info(len(y), k, r["rss"])
        o = {"form": desc, "n_points": len(y), "n_params": k,
             "params": {a: float(b) for a, b in r["params"].items()},
             "ci95_halfwidth": {a: float(b) for a, b in r["ci95"].items()},
             "rms_ln_residual": float(np.sqrt(r["rss"] / len(y))), "max_abs_ln_residual": float(np.abs(r["res"]).max()),
             "AICc": aicc, "BIC": bic}
        if Nhold is not None:
            pred = r["predict"](np.asarray(Nhold, float))
            rel = np.exp(pred - np.log(np.asarray(valhold, float))) - 1.0
            o["holdout_N"] = [int(v) for v in Nhold]
            o["holdout_rel_error"] = [float(v) for v in rel]
        out[nm] = o
    best = min(out, key=lambda k_: out[k_]["AICc"])
    for nm in out:
        out[nm]["dAICc"] = out[nm]["AICc"] - out[best]["AICc"]
    return out


def md_models(title, res):
    MD.append(f"\n**{title}**\n")
    hold = any("holdout_N" in v for v in res.values())
    hdr = "| model | form | fitted parameters (95% CI half-width) | rms ln-resid | max ln-resid | dAICc |"
    if hold:
        hN = next(v["holdout_N"] for v in res.values() if "holdout_N" in v)
        hdr += " " + " | ".join(f"extrap. err N={n}" for n in hN) + " |"
    MD.append(hdr)
    MD.append("|" + "---|" * (hdr.count("|") - 1))
    for nm, v in res.items():
        ps = ", ".join(f"{k}={v['params'][k]:.5g} (±{v['ci95_halfwidth'][k]:.2g})" for k in v["params"])
        line = f"| {nm} | {v['form']} | {ps} | {v['rms_ln_residual']:.2e} | {v['max_abs_ln_residual']:.2e} | {v['dAICc']:.1f} |"
        if hold:
            line += " " + " | ".join(f"{100*e:+.2f}%" for e in v["holdout_rel_error"]) + " |"
        MD.append(line)


# ---------------------------------------------------------------------------
# 0. continuum constants
# ---------------------------------------------------------------------------
mp.mp.dps = 30
NTERM = 60   # exp(-(2n+1)^2 pi^2 tau) terms; ample for tau >= 1e-3 at 30 digits
# 1D: reflecting at 0, absorbing at L, start at 0.  tau = D t / L^2.
# density g(tau) ~ sum_{m>=1} (-1)^(m+1) (2m-1) exp(-(2m-1)^2 pi^2 tau/4);  mean tau = 1/2.
def dg1(tau):
    # derivative of the theta-series density (analytic term-by-term)
    return mp.fsum((-1) ** (m + 1) * (2 * m - 1) ** 3 * mp.e ** (-(2 * m - 1) ** 2 * mp.pi ** 2 * tau / 4) for m in range(1, NTERM))
tau1 = mp.findroot(dg1, (mp.mpf("0.16"), mp.mpf("0.17")), solver="anderson", tol=1e-30)
OUT["continuum_1d"] = {"tau_mode": float(tau1), "tau_mean": 0.5, "mode_over_mean": float(2 * tau1),
                       "one_third_minus_ratio": float(mp.mpf(1) / 3 - 2 * tau1),
                       "note": "mode solves g'(tau)=0 for the theta-series density; 1/6 is the semi-infinite (single image) value"}
# Dirichlet exit from the centre of a d-cube (independent coordinates), unit interval, D = 1:
def Sk(tau, k):
    # k-th derivative of S1(tau) = (4/pi) sum (-1)^n/(2n+1) exp(-(2n+1)^2 pi^2 tau)
    return 4 / mp.pi * mp.fsum((-1) ** n / mp.mpf(2 * n + 1) * (-(2 * n + 1) ** 2 * mp.pi ** 2) ** k
                               * mp.e ** (-(2 * n + 1) ** 2 * mp.pi ** 2 * tau) for n in range(0, NTERM))
ex = {}
for d in (1, 2, 3):
    # g_d = -d S^(d-1) S';  g_d' = 0  <=>  (d-1) S'^2 + S S'' = 0
    cond = lambda t: (d - 1) * Sk(t, 1) ** 2 + Sk(t, 0) * Sk(t, 2)
    tm = mp.findroot(cond, (mp.mpf("0.02"), mp.mpf("0.06")), solver="anderson", tol=1e-30)
    # for tau < 0.01 the eigen-series converges slowly; there S1 = 1 - (exponentially small), use
    # the image form S1(tau) = 1 - 2 sum_k (-1)^k erfc((2k+1)/(4 sqrt(tau)))
    def S1_any(t):
        if t < mp.mpf("0.02"):
            if t == 0:
                return mp.mpf(1)
            return 1 - 2 * mp.fsum((-1) ** k * mp.erfc((2 * k + 1) / (4 * mp.sqrt(t))) for k in range(0, 8))
        return Sk(t, 0)
    mean = mp.quad(lambda t: S1_any(t) ** d, [0, 0.02, 0.05, 0.2, 1, 6])
    ex[d] = {"tau_mode": float(tm), "tau_mean": float(mean), "mode_over_mean": float(tm / mean),
             "lattice_q_mode_over_(N-1)^2": float(2 * d * tm)}
OUT["continuum_exit"] = ex

# ---------------------------------------------------------------------------
# 1. tables of exact values (discrete time, q = 0.8)
# ---------------------------------------------------------------------------
OUT["exact_value_tables"] = {}
for d in (1, 2, 3):
    rows = sel(Drows, d=d, geo="CC", q=0.8)
    tab = [{"N": r["N"], "mode_exact": r["mode"], "mfpt_exact": r["mfpt_exact"], "ratio_exact": r["mode_over_mfpt"]}
           for r in rows]
    OUT["exact_value_tables"][str(d)] = tab
    MD.append(f"\n**Table T{d}. Exact values, d = {d}, corner-to-corner, q = 0.8 (discrete time).**\n")
    MD.append("| N | exact mode | exact MFPT | exact mode/MFPT |")
    MD.append("|---|---|---|---|")
    for e in tab:
        MD.append(f"| {e['N']} | {e['mode_exact']} | {e['mfpt_exact']:.6g} | {e['ratio_exact']:.4f} |")

# ---------------------------------------------------------------------------
# 2. model comparison
# ---------------------------------------------------------------------------
OUT["fits"] = {}
# Fits use the half-octave size grid (plus the sizes of the tables above); the few extra sizes computed for the
# q-scan / N = 4001 comparison (group "extra_pts") are excluded so that the size grid stays near log-uniform.
Lfit = [r for r in Lrows if r.get("group") != "extra_pts"]
cont = {d: sel(Lfit, d=d, geo="CC") for d in (1, 2, 3)}

# --- 1D ---
r1 = [r for r in cont[1] if r["N"] >= 10]
N1 = [r["N"] for r in r1]; M1 = [r["mode"] for r in r1]
dd = sel(Drows, d=1, geo="CC", q=0.8)
dd = [r for r in dd if 10 <= r["N"] <= 200]
OUT["fits"]["1d_mode_discrete_short_range"] = run_models(["P", "C2", "Q2"], [r["N"] for r in dd], [r["mode"] for r in dd])
OUT["fits"]["1d_mode_continuous_all"] = run_models(["P", "C2", "Q2"], N1, M1)
md_models("1D mode, exact discrete PMF, q = 0.8, N = 10..200 (short range)", OUT["fits"]["1d_mode_discrete_short_range"])
md_models(f"1D mode, continuous time (unit rate), N = 10..{max(N1)}", OUT["fits"]["1d_mode_continuous_all"])

# --- 2D ---
c2 = cont[2]
Nall = np.array([r["N"] for r in c2]); Mall = np.array([r["mode"] for r in c2]); Tall = np.array([r["mfpt"] for r in c2])
inwin = (Nall >= 10) & (Nall <= 200)
hold = np.isin(Nall, [1280, 10240, 163840, 16777216])
names2 = ["C2", "P", "L2", "PL", "LL2", "LN2"]
OUT["fits"]["2d_mode_fit_10_200_extrapolated"] = run_models(names2, Nall[inwin], Mall[inwin], Nall[hold], Mall[hold])
md_models("2D mode (continuous time, unit rate): fit on N = 10..200, extrapolated", OUT["fits"]["2d_mode_fit_10_200_extrapolated"])
big = Nall >= 10
OUT["fits"]["2d_mode_fit_all"] = run_models(names2, Nall[big], Mall[big])
md_models(f"2D mode (continuous time): fit on all N = 10..{Nall.max()}", OUT["fits"]["2d_mode_fit_all"])
large = Nall >= 1000
OUT["fits"]["2d_mode_fit_large"] = run_models(names2, Nall[large], Mall[large])
md_models(f"2D mode (continuous time): fit on N = 1000..{Nall.max()}", OUT["fits"]["2d_mode_fit_large"])
d2 = [r for r in sel(Drows, d=2, geo="CC", q=0.8) if 10 <= r["N"] <= 200]
OUT["fits"]["2d_mode_discrete_short_range"] = run_models(names2, [r["N"] for r in d2], [r["mode"] for r in d2])
md_models("2D mode, exact discrete PMF, q = 0.8, N = 11..200 (short range)", OUT["fits"]["2d_mode_discrete_short_range"])
OUT["fits"]["2d_mfpt_fit_10_200_extrapolated"] = run_models(["C2", "P", "L2", "LN2"], Nall[inwin], Tall[inwin], Nall[hold], Tall[hold])
md_models("2D MFPT (unit rate): fit on N = 10..200, extrapolated", OUT["fits"]["2d_mfpt_fit_10_200_extrapolated"])

# --- 3D ---
c3 = cont[3]
N3 = np.array([r["N"] for r in c3]); M3 = np.array([r["mode"] for r in c3]); T3 = np.array([r["mfpt"] for r in c3])
inwin3 = (N3 >= 10) & (N3 <= 100)
hold3 = np.isin(N3, [320, 1280, 4096])
names3 = ["C3", "C2", "P", "L2", "PL", "LN2"]
OUT["fits"]["3d_mode_fit_10_100_extrapolated"] = run_models(names3, N3[inwin3], M3[inwin3], N3[hold3], M3[hold3])
md_models("3D mode (continuous time, unit rate): fit on N = 10..100, extrapolated", OUT["fits"]["3d_mode_fit_10_100_extrapolated"])
OUT["fits"]["3d_mode_fit_all"] = run_models(names3, N3[N3 >= 10], M3[N3 >= 10])
md_models(f"3D mode (continuous time): fit on all N = 10..{N3.max()}", OUT["fits"]["3d_mode_fit_all"])
d3 = [r for r in sel(Drows, d=3, geo="CC", q=0.8) if 10 <= r["N"] <= 100]
OUT["fits"]["3d_mode_discrete_short_range"] = run_models(names3, [r["N"] for r in d3], [r["mode"] for r in d3])
md_models("3D mode, exact discrete PMF, q = 0.8, N = 11..100 (short range)", OUT["fits"]["3d_mode_discrete_short_range"])
OUT["fits"]["3d_mfpt_fit_10_100_extrapolated"] = run_models(["C3", "P", "LN3"], N3[inwin3], T3[inwin3], N3[hold3], T3[hold3])
md_models("3D MFPT (unit rate): fit on N = 10..100, extrapolated", OUT["fits"]["3d_mfpt_fit_10_100_extrapolated"])

# ---------------------------------------------------------------------------
# 3. effective (local) exponents
# ---------------------------------------------------------------------------
eff = {}
for d in (1, 2, 3):
    rr = cont[d]
    N = np.array([r["N"] for r in rr], float); M = np.array([r["mode"] for r in rr]); T = np.array([r["mfpt"] for r in rr])
    # use roughly half-octave spaced subsequence
    keep = [0]
    for i in range(1, len(N)):
        if N[i] / N[keep[-1]] >= 1.35:
            keep.append(i)
    N, M, T = N[keep], M[keep], T[keep]
    Ng = np.sqrt(N[1:] * N[:-1])
    am = np.diff(np.log(M)) / np.diff(np.log(N))
    at = np.diff(np.log(T)) / np.diff(np.log(N))
    eff[str(d)] = {"N_geo_mean": Ng.tolist(), "a_eff_mode": am.tolist(), "a_eff_mfpt": at.tolist()}
OUT["effective_exponents"] = eff

# ---------------------------------------------------------------------------
# 4. asymptotic constants and accuracy of the closed-form predictions
# ---------------------------------------------------------------------------
asy = {}
# 2D MFPT: MFPT_1 = (8/pi) N^2 ln N + c2 N^2 + ...
c2seq = (Tall - 8 / np.pi * Nall ** 2 * np.log(Nall)) / Nall ** 2
asy["2d_mfpt_const_sequence"] = {"N": Nall.tolist(), "c2": c2seq.tolist()}
asy["2d_mfpt_const_limit_estimate"] = float(c2seq[-1])
mu1_2 = np.array([r["p"][0] for r in c2])
X2 = mu1_2 * Tall
asy["2d_X_minus_2pi_lnN"] = (X2 - 2 * np.pi * np.log(Nall)).tolist()
# slopes
lnln = np.log(np.log(Nall))
m_over = Mall / Nall ** 2
asy["2d_slope_d(mode/N^2)/d(lnlnN)"] = {"N_mid": np.sqrt(Nall[1:] * Nall[:-1]).tolist(),
                                       "slope": (np.diff(m_over) / np.diff(lnln)).tolist(), "theory_limit": 4 / np.pi ** 2}
# 3D
C3seq = T3 / N3 ** 3
asy["3d_C3_sequence"] = {"N": N3.tolist(), "MFPT/N^3": C3seq.tolist()}
# Richardson (assume C3 + c/N): use last two
C3lim = (C3seq[-1] * N3[-1] - C3seq[-2] * N3[-2]) / (N3[-1] - N3[-2])
asy["3d_C3_limit_richardson"] = float(C3lim)
asy["3d_C3_1overN_coeff"] = float((C3seq[-1] - C3lim) * N3[-1])
m3 = M3 / N3 ** 2
asy["3d_slope_d(mode/N^2)/d(lnN)"] = {"N_mid": np.sqrt(N3[1:] * N3[:-1]).tolist(),
                                      "slope": (np.diff(m3) / np.diff(np.log(N3))).tolist(), "theory_limit": 6 / np.pi ** 2}
OUT["asymptotics"] = asy

acc = {}
for (d, geo) in [(2, "CC"), (3, "CC"), (2, "C2M"), (3, "C2M")]:
    rr = sel(Lfit, d=d, geo=geo)
    a = {"N": [r["N"] for r in rr]}
    for k in ("pred_L0", "pred_L1", "pred_2pole", "pred_3pole"):
        a[k + "_rel_err"] = [r[k] / r["mode"] - 1 for r in rr]
    acc[f"{d}d_{geo}"] = a
    MD.append(f"\n**Accuracy of mode predictions, d = {d}, {geo} (continuous time).**  max |rel. error| over N >= 10: "
              + ", ".join(f"{k[5:]}: {max(abs(e) for e, n in zip(a[k + '_rel_err'], a['N']) if n >= 10):.2e}"
                          for k in ("pred_L0", "pred_L1", "pred_2pole", "pred_3pole")))
OUT["prediction_accuracy"] = acc

# closed-form L1 with the asymptotic X (no exact MFPT needed)
def L1_2d_asym(N, c2):
    X = 2 * np.pi * np.log(N) + np.pi ** 2 / 4 * c2
    return 4 * N ** 2 / np.pi ** 2 * X * np.log(4 * X) / (X + 3), X
p_as, X_as = L1_2d_asym(Nall, asy["2d_mfpt_const_limit_estimate"])
OUT["closed_form_2d_asymptotic_X"] = {"N": Nall.tolist(), "rel_err_mode": (p_as / Mall - 1).tolist(),
                                      "ratio_pred": (np.log(4 * X_as) / (X_as + 3)).tolist(), "ratio_exact": (Mall / Tall).tolist()}

# discrete (q=0.8) vs continuous offset
off = []
for r in Drows:
    if r["geo"] in ("CC", "C2M", "M2C") and r["q"] == 0.8:
        key = (r["d"], r["geo"], r["N"])
        m = [x for x in Lrows if (x["d"], x["geo"], x["N"]) == key]
        if m:
            off.append({"d": r["d"], "geo": r["geo"], "N": r["N"], "mode_discrete": r["mode"],
                        "mode_cont_over_q": m[0]["mode"] / 0.8, "offset_steps": r["mode"] - m[0]["mode"] / 0.8,
                        "offset_refined": r["mode_refined"] - m[0]["mode"] / 0.8})
OUT["discrete_vs_continuous_offsets"] = off
OUT["max_abs_offset_steps"] = max(abs(o["offset_steps"]) for o in off)

# independent semi-analytic check of the DISCRETE modes: three-pole formula in discrete time.
# F~(z) = F^((1-z)/z)  =>  f(t) = sum_j q a_j (1 - q nu_j)^(t-1)  (a_j, nu_j: unit-rate residues/poles)
chk = []
for r in Drows:
    if r["geo"] in ("CC", "C2M") and r["q"] == 0.8 and r["d"] > 1:
        m = [x for x in Lrows if (x["d"], x["geo"], x["N"]) == (r["d"], r["geo"], r["N"])]
        if not m:
            continue
        nus = 0.8 * np.array(m[0]["nu"]); a = 0.8 * np.array(m[0]["res"])
        lam = 1.0 - nus
        dfun = lambda t: float((a * np.log(lam) * lam ** (t - 1.0)).sum())
        try:
            t3 = brentq(dfun, 0.5 * r["mode"], 2.0 * r["mode"], rtol=1e-14)
        except ValueError:
            continue
        chk.append({"d": r["d"], "geo": r["geo"], "N": r["N"], "mode_argmax": r["mode"], "mode_refined": r["mode_refined"],
                    "three_pole_discrete": t3, "diff_refined_minus_3pole": r["mode_refined"] - t3})
OUT["discrete_three_pole_check"] = chk
big = [c for c in chk if c["N"] >= 30]
OUT["discrete_three_pole_check_max_abs_diff_N>=30"] = max(abs(c["diff_refined_minus_3pole"]) for c in big)
print("discrete 3-pole check, max |refined - 3pole| for N>=30:", OUT["discrete_three_pole_check_max_abs_diff_N>=30"])

# perturbative pole structure: nu_0*tau -> 1, (nu_1/mu_1 - 1)(X-1)/W -> 1, a_1/a_0 -> -W
ps = {}
for (d, geo) in [(2, "CC"), (3, "CC")]:
    rr = sel(Lfit, d=d, geo=geo)
    ps[f"{d}d"] = {"N": [r["N"] for r in rr],
                   "X": [r["p"][0] * r["mfpt"] for r in rr],
                   "nu0_tau": [r["nu"][0] * r["mfpt"] for r in rr],
                   "delta1_times_(X-1)/W": [(r["nu"][1] / r["p"][0] - 1) * (r["p"][0] * r["mfpt"] - 1) / r["W"][0] for r in rr],
                   "a1_over_a0": [r["res"][1] / r["res"][0] for r in rr],
                   "W": [r["W"][0] for r in rr]}
OUT["pole_structure"] = ps

json.dump(OUT, open(_os.path.join(_R, 'data', 'msc_modes', 'fit_results.json'), "w"), indent=1)
open(_os.path.join(_R, 'data', 'msc_modes', 'tables.md'), "w").write("\n".join(MD) + "\n")
print("\n".join(MD))
print(json.dumps({k: OUT[k] for k in ("continuum_1d", "continuum_exit")}, indent=1))
print("2d c2 limit", asy["2d_mfpt_const_limit_estimate"], "3d C3", asy["3d_C3_limit_richardson"], asy["3d_C3_1overN_coeff"])
print("max offset", OUT["max_abs_offset_steps"])
