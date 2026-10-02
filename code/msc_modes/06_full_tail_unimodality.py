"""
06_full_tail_unimodality.py -- iterate the exact discrete-time PMF until the
survival probability is below 1e-12 and count local maxima over the WHOLE
support (not just a window around the mode).  Also re-derives the mean from
the PMF.  q = 0.8 and q = 0.5.  Output: data/full_tail.json
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import fptlib as F

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = _os.path.join(_R, 'data', 'msc_modes', 'full_tail.json')
CASES = ([(1, "CC", N, q) for N in (5, 21, 51, 101) for q in (0.8, 0.5)]
         + [(2, "CC", N, q) for N in (5, 9, 15, 21, 29, 35) for q in (0.8, 0.5)]
         + [(3, "CC", N, 0.8) for N in (5, 7, 9, 11, 15)]
         + [(2, g, N, 0.8) for g in ("C2M", "M2C") for N in (5, 11, 21)]
         + [(3, g, N, 0.8) for g in ("C2M", "M2C") for N in (5, 11)])
res = []
for (d, geo, N, q) in CASES:
    s, a = F.GEOMETRIES[geo][0](N, d), F.GEOMETRIES[geo][1](N, d)
    st = F.Stepper(N, d, q, s, [a])
    f = [0.0]
    while True:
        f.append(st.step())
        if (st.t & 1023) == 0 and st.survival() < 1e-12:
            break
    f = np.array(f)
    t = np.arange(f.size)
    sg = np.sign(np.diff(f[1:]))
    nz = sg != 0
    sgn = sg[nz]
    n_max = int(((sgn[:-1] > 0) & (sgn[1:] < 0)).sum())
    n_min = int(((sgn[:-1] < 0) & (sgn[1:] > 0)).sum())
    mode = int(np.argmax(f))
    mean = float((t * f).sum())
    var = float((t * t * f).sum() - mean ** 2)
    rec = {"d": d, "geo": geo, "N": N, "q": q, "t_end": int(f.size - 1), "S_end": st.survival(), "mass": float(f.sum()),
           "mode": mode, "n_local_maxima_full_support": n_max, "n_local_minima_full_support": n_min,
           "mean_from_pmf": mean, "cv": float(np.sqrt(var) / mean),
           "P_T_le_mode": float(f[: mode + 1].sum()), "median": int(np.searchsorted(np.cumsum(f), 0.5))}
    res.append(rec)
    print(rec, flush=True)
json.dump(res, open(OUT, "w"), indent=1)
