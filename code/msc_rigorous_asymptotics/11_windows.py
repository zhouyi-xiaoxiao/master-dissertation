"""Semi-explicit windows (Theorem A) from exact scalars vs exact modes."""
import json
import numpy as np
from t3lib import scalars, exact_modes, mu_real, DATA
from windows import window
modes = exact_modes()
cases = [(2, N) for N in (5, 8, 10, 14, 20, 35, 50, 100, 200, 400, 640, 1280)] + \
        [(3, N) for N in (5, 8, 10, 14, 20, 28, 40, 80, 160)]
out = []
for d, N in cases:
    s = scalars(d, N)
    w = window(s)
    rec = dict(d=d, N=N, X=s['X'], **w)
    if (d, N) in modes:
        ts = modes[(d, N)][0] * mu_real(d, N)
        rec['tau_star'] = ts
        if w.get('ok'):
            rec['inside'] = bool(w['t_minus'] <= ts <= w['t_plus'])
            rec['rel_width'] = (w['t_plus'] - w['t_minus']) / ts
    out.append(rec)
    print(json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)
json.dump(out, open(DATA + '/11_windows.json', 'w'), indent=1)
