"""
03b_profiles.py -- exact continuous-time first-passage densities on rescaled
time grids (for the collapse figure), survival probability at the mode, and
'plateau' widths (times at which g = 0.99 g_max and 0.5 g_max on both sides).
Unit jump rate.  Output: data/profiles.npz and data/profile_stats.json.
Requires data/laplace_modes.jsonl (from 03_run_laplace.py).
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import json, os, sys, math
import numpy as np
from scipy.optimize import brentq
sys.path.insert(0, os.path.dirname(__file__))
import fptlib as F

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = _os.path.join(_R, 'data', 'msc_modes')
MT = 20
rows = [json.loads(l) for l in open(_os.path.join(_R, 'data', 'msc_modes', 'laplace_modes.jsonl')) if l.strip()]
idx = {(r["d"], r["geo"], r["N"]): r for r in rows}

SEL = {
    1: [10, 40, 160, 640, 10240],
    2: [10, 40, 160, 640, 10240, 163840, 1048576],
    3: [10, 20, 40, 80, 160, 640, 4096],
}
XG = np.concatenate([np.linspace(0.05, 12.0, 240), np.linspace(12.2, 60.0, 120)])  # x = mu_1 t
YG = np.concatenate([np.geomspace(1e-4, 0.02, 40), np.linspace(0.025, 4.0, 160)])   # y = t / MFPT

prof = {}
stats = []
for d, Ns in SEL.items():
    for N in Ns:
        r = idx.get((d, "CC", N))
        if r is None:
            print("missing", d, N); continue
        tau, mode = r["mfpt"], r["mode"]
        a = 1.0 / d
        mu1 = 2.0 * a * math.sin(math.pi / (2.0 * N)) ** 2
        t = XG / mu1
        s0, a0 = F.corner(N, d), F.far_corner(N, d)
        if d == 1:
            ch = F.Chain1D(N, 1.0)
            g = ch.g_cont(t)
            gfun = lambda tt: ch.g_cont(np.atleast_1d(tt))
            # survival: S(t) = sum W/mu exp(-mu t)
            Sm = float((ch.W / ch.mu * np.exp(-ch.mu * mode)).sum())
        else:
            L = F.LaplaceFP(N, d, 1.0, s0, a0)
            L.prepare(2.0 * MT * MT / (5.0 * t.min()) * 1.05)
            g = F.talbot_invert(L.Fhat, t, M=MT)
            gfun = lambda tt: F.talbot_invert(L.Fhat, np.atleast_1d(tt), M=MT)
            Sm = float(F.talbot_invert(lambda s: (1.0 - L.Fhat(s)) / s, [mode], M=MT)[0])
        gmax = float(gfun(mode)[0])
        st = {"d": d, "N": N, "mode": mode, "mfpt": tau, "g_max": gmax, "g_max_times_mfpt": gmax * tau,
              "S_at_mode": Sm, "S_at_mfpt": None, "mu1": mu1}
        if d > 1:
            st["S_at_mfpt"] = float(F.talbot_invert(lambda s: (1.0 - L.Fhat(s)) / s, [tau], M=MT)[0])
        else:
            st["S_at_mfpt"] = float((ch.W / ch.mu * np.exp(-ch.mu * tau)).sum())
        for lev in (0.99, 0.9, 0.5):
            fl = lambda tt: float(gfun(tt)[0]) - lev * gmax
            try:
                lo = brentq(fl, mode * 1e-3, mode, rtol=1e-10)
            except ValueError:
                lo = float("nan")
            try:
                hi = brentq(fl, mode, 60.0 * tau, rtol=1e-10)
            except ValueError:
                hi = float("nan")
            st[f"t_lo_{lev}"] = lo
            st[f"t_hi_{lev}"] = hi
        stats.append(st)
        prof[f"d{d}_N{N}_x"] = XG
        prof[f"d{d}_N{N}_g_tau"] = g * tau
        prof[f"d{d}_N{N}_y"] = YG
        prof[f"d{d}_N{N}_g_tau_y"] = np.asarray(gfun(YG * tau)) * tau
        print(d, N, "gmax*tau=%.5f S(mode)=%.5f S(mfpt)=%.5f  99%%-band=[%.3f,%.3f] mode" %
              (gmax * tau, Sm, st["S_at_mfpt"], st["t_lo_0.99"] / mode, st["t_hi_0.99"] / mode), flush=True)

np.savez_compressed(_os.path.join(_R, 'data', 'msc_modes', 'profiles.npz'), **prof)
json.dump(stats, open(_os.path.join(_R, 'data', 'msc_modes', 'profile_stats.json'), "w"), indent=1)
print("done")
