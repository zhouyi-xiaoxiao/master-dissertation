"""Cross-validation of the final theorems against exact-mode data computed separately (not a proof):
 (a) data/msc_modes/discrete_modes.jsonl (1D rows, produced by the floating-point study, code/msc_modes),
 (b) fresh float64 computations for a grid of (N, q), including q in (4/5, 1) with N >= 601,
 (c) the global structure of Theorem 7.7 for q = 0.9, N = 601: f increasing on [t_a, t_-+1], decreasing on [t_+, inf).
Checks: Theorem 6.3 window (tau - E, tau + 1 + E), E = K/(q L^2); Theorem 7.4 (q = 1); Corollary 6.4 bounds.
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import os, sys, json, math
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from lib1d import pmf_stepper

c = 0.33328426549494789874; kappa = 0.086273779378239017; rho = 1.9911786618652829
b = 0.33250858299642786; s2 = -2.2503062886311552
here = os.path.dirname(__file__)
res = {"independent_rows": [], "grid": [], "notes": []}


def chi(q):
    return 0.5 if q <= 0.8 else min(0.5, -math.log(2 * q - 1) / q)


def hcond(q, L):
    return math.log(4 * (math.pi / 2) ** 3) + 8 * math.log(L) - 0.15 * chi(q) * L * L <= math.log(1e-12)


def check(N, q, modes):
    L = N - 0.5
    out = dict(N=N, q=q, modes=modes)
    if q == 1.0:
        tau1 = c * L * L - b * L + 1 + s2
        K = 10.3
        out["theorem"] = "7.4"
        out["covered"] = True
        out["ok"] = all((m - (N - 1)) % 2 == 0 and tau1 - K / L < m < tau1 + 2 + K / L for m in modes)
        out["offset"] = [m - tau1 for m in modes]
        return out
    tau = (c * L * L - kappa) / q + 1 - rho
    if q <= 0.8:
        K = 1.78; covered = N >= 3
    else:
        K = 1.56; covered = (N >= 601 and hcond(q, L))
    E = K / (q * L * L)
    out["theorem"] = "6.3/7.7"; out["covered"] = covered
    out["ok"] = all(tau - E < m < tau + 1 + E for m in modes)
    out["offset"] = [m - tau for m in modes]
    # Corollary 6.4 (eq. (24))
    out["cor_e3"] = all(abs(q * m / L ** 2 - c) < (0.0863 + 0.9912 * q) / L ** 2 + K / L ** 4 for m in modes)
    return out


# (a) independent data
src = _os.path.join(_R, 'data', 'msc_modes', 'discrete_modes.jsonl')
bad_cov = 0; n_cov = 0; n_unc = 0; unc_fail = []
for line in open(src):
    r = json.loads(line)
    if r.get("d") != 1 or r.get("geo") != "CC":      # corner-to-corner rows only (reflecting end -> absorbing end)
        continue
    lo, hi = r["argmax_band_1e-12"]
    modes = [r["mode"]]
    o = check(r["N"], float(r["q"]), modes)
    res["independent_rows"].append(o)
    if o["covered"]:
        n_cov += 1
        bad_cov += (not o["ok"])
    else:
        n_unc += 1
        if not o["ok"]:
            unc_fail.append((r["N"], r["q"], r["mode"], round(o["offset"][0], 3)))
print(f"(a) independent 1D rows: covered by a theorem: {n_cov}, violations: {bad_cov};  not covered (crossover regime): {n_unc}, "
      f"of which outside the window: {unc_fail}")
res["independent_summary"] = dict(covered=n_cov, violations=bad_cov, uncovered=n_unc, uncovered_outside_window=unc_fail)

# (b) fresh grid
viol = 0; tot = 0
for q in [0.05, 0.2, 1 / 3, 0.5, 2 / 3, 0.75, 0.8]:
    for N in list(range(3, 61)) + [80, 127, 200, 333, 500]:
        L = N - 0.5
        T = int(0.6 * L * L / q) + 60
        f = pmf_stepper(N, q, 1, T)
        m = f.max()
        modes = [int(t) for t in np.where(f >= m * (1 - 1e-13))[0]]
        o = check(N, q, modes); tot += 1
        if not (o["ok"] and o["cor_e3"]):
            viol += 1; res["grid"].append(o)
for q in [0.9, 0.99, 0.999]:
    for N in [601, 750, 1000]:
        L = N - 0.5
        T = int(0.5 * L * L / q) + 60
        f = pmf_stepper(N, q, 1, T)
        m = f.max()
        modes = [int(t) for t in np.where(f >= m * (1 - 1e-13))[0]]
        o = check(N, q, modes); tot += 1
        o["h_condition"] = hcond(q, L)
        res["grid"].append(o)
        if o["covered"] and not o["ok"]:
            viol += 1
        print("   q=%.3f N=%d modes=%s offset=%s covered=%s ok=%s" % (q, N, modes, [round(x, 4) for x in o["offset"]], o["covered"], o["ok"]))
print(f"(b) fresh grid: {tot} cases, violations of covered statements: {viol}")
res["grid_summary"] = dict(cases=tot, violations=viol)

# (c) structure for q = 0.9, N = 601
N, q = 601, 0.9
L = N - 0.5
T = int(1.2 * L * L / q)
f = pmf_stepper(N, q, 1, T)
tau = (c * L * L - kappa) / q + 1 - rho
E = 1.56 / (q * L * L)
ta = int(math.ceil(0.15 * L * L / q)) + 1
tm, tp = int(math.floor(tau - E)), int(math.ceil(tau + E))
inc = bool(np.all(np.diff(f[ta:tm + 2]) > 0)); dec = bool(np.all(np.diff(f[tp:T]) < 0))
left = bool(f[1:ta].max() < f.max())
nonuni = bool(f[N - 1] > f[N] < f[N + 1])
print(f"(c) q=0.9 N=601: increasing on [t_a, t_-+1]: {inc}; decreasing on [t_+, T]: {dec}; left tail below max: {left}; "
      f"non-unimodal at the left end f(N-1) > f(N) < f(N+1): {nonuni}")
res["structure_q09_N601"] = dict(increasing=inc, decreasing=dec, left_below=left, nonunimodal_left_end=nonuni)
json.dump(res, open(_os.path.join(_R, 'data', 'msc_rigorous_1d', '20_validation.json'), "w"), indent=1)
