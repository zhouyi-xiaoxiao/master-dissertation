"""Theorem 2.3 (window from the bound functions Phi_-, Phi_+ with EXACT (float) pole data) versus the exact
continuous-time modes of the floating-point study (Talbot inversion), d = 2, 3.  Prints X_s (tau - tau_cf) for tau = tau_-, tau*, tau_+."""
import json, sys
import numpy as np
from r4lib import scalars, window, exact_modes_C, mu_real, DATA
mC = exact_modes_C()
NS = {2: [5, 7, 10, 14, 20, 28, 35, 50, 80, 100, 160, 200, 320, 400, 640, 905, 1280, 1810],
      3: [5, 7, 10, 14, 20, 28, 35, 50, 80, 100, 160]}
out = []
for d in (2, 3):
    for N in NS[d]:
        s = scalars(d, N)
        lo, hi = window(s)
        rec = dict(d=d, N=N, Xs=float(s['Xs']), X=float(s['X']), theta=float(s['theta']), tau_cf=float(s['tau_cf']), t_c=float(s['t_c']),
                   tau_minus=lo, tau_plus=hi)
        if lo is not None and hi is not None:
            rec['Xs(tau_minus-tau_cf)'] = float(s['Xs'] * (lo - s['tau_cf'])); rec['Xs(tau_plus-tau_cf)'] = float(s['Xs'] * (hi - s['tau_cf']))
        if (d, N) in mC:
            ts = mC[(d, N)][0] * mu_real(d, N)
            rec['tau_star'] = float(ts); rec['Xs(tau_star-tau_cf)'] = float(s['Xs'] * (ts - s['tau_cf']))
            rec['X_float_study_relerr'] = float(mC[(d, N)][1] * mu_real(d, N) / s['X'] - 1)
            if lo is not None and hi is not None:
                rec['inside'] = bool(lo < ts < hi)
        rec['Kd_formula'] = float(1 - s['S1'] - 2 * s['EU'] + s['W'] + s['Ir'] - (d - 1) / d)
        out.append(rec)
        print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)
        json.dump(out, open(DATA + '/41_windows_exact.json', 'w'), indent=1)
