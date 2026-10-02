"""
03c_medians.py -- medians and survival at the mode / at the MFPT for the
corner-to-corner problem (continuous time, unit rate), to quantify how
'typical' the modal time is.  Output: data/medians.json
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
SEL = {1: [10, 35, 100, 1000], 2: [10, 35, 100, 200, 1280, 10240, 163840], 3: [10, 35, 40, 100, 320, 1280]}
out = []
for d, Ns in SEL.items():
    for N in Ns:
        r = idx.get((d, "CC", N))
        if r is None:
            continue
        tau, mode = r["mfpt"], r["mode"]
        if d == 1:
            ch = F.Chain1D(N, 1.0)
            S = lambda t: float((ch.W / ch.mu * np.exp(-ch.mu * t)).sum())
        else:
            L = F.LaplaceFP(N, d, 1.0, F.corner(N, d), F.far_corner(N, d))
            L.prepare(2.0 * MT * MT / (5.0 * 0.5 * mode) * 1.05)
            S = lambda t: float(F.talbot_invert(lambda s: (1.0 - L.Fhat(s)) / s, [t], M=MT)[0])
        med = brentq(lambda t: S(t) - 0.5, mode, 3 * tau, rtol=1e-10)
        rec = {"d": d, "N": N, "mode": mode, "median": med, "mfpt": tau, "mode_over_mfpt": mode / tau,
               "median_over_mfpt": med / tau, "S_at_mode": S(mode), "P_T_le_mode": 1 - S(mode), "S_at_mfpt": S(tau)}
        out.append(rec)
        print(rec, flush=True)
json.dump(out, open(_os.path.join(_R, 'data', 'msc_modes', 'medians.json'), "w"), indent=1)
