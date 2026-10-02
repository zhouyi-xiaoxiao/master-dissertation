"""v04b -- more precise Theta(p) at p = 0.1, 0.2 (L = 200 torus, K = 40) for the large-N convergence test."""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import os, sys, json, re
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
src = open(_os.path.join(_R, 'code', 'msc_defects_verify', 'v04_homog.py')).read()
# reuse torus_sigma (function of this check) without running the rest of v04
ns = {}
start = src.index('def torus_sigma'); end = src.index('# (b) single blocked site')
exec('import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spla, scipy.sparse.csgraph as csg\n' + src[start:end], ns)
torus_sigma = ns['torus_sigma']
out = {}
for p in (0.10, 0.20):
    L, K = 200, 40
    rng = np.random.default_rng([13579, int(p * 100)])
    th = []; sg = []; pin = []
    for k in range(K):
        mask = np.zeros(L * L, bool); mask[rng.choice(L * L, int(round(p * L * L)), replace=False)] = True
        s, pinf, n = torus_sigma(~mask.reshape(L, L)); th.append(pinf / s); sg.append(s); pin.append(pinf)
    th = np.array(th)
    out[str(p)] = dict(L=L, K=K, Theta=float(np.mean(pin) / np.mean(sg)), Theta_ci=float(1.96 * th.std(ddof=1) / np.sqrt(K)), sigma=float(np.mean(sg)), sigma0_over_sigma=float(1 / np.mean(sg)), P_inf=float(np.mean(pin)))
    print(p, out[str(p)], flush=True)
json.dump(out, open(_os.path.join(_R, 'data', 'msc_defects_verify', 'v04b_theta.json'), 'w'), indent=1)
