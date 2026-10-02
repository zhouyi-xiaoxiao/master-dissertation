#!/usr/bin/env python
"""Supplementary Section S2.1 (route B): why the logarithm of each eigenvalue must be formed as log1p(-mu).

For the chain (start 0, target N-1) at q = 0.8 the integer mode is the smallest t with
    Delta(t) = f(t+1) - f(t) = -sum_m W_m mu_m lambda_m^(t-1) <= 0,      lambda_m = 1 - mu_m,
evaluated as in route B (code/msc_modes/03_run_laplace.py, run_point, d = 1): only the terms with
lambda_m > 0, lambda_m^(t-1) = exp((t-1) * L_m), and a Brent root of Delta, whose ceiling is the mode.
This script locates the root twice, with L_m = log1p(-mu_m) and with L_m = log(1 - mu_m) (the logarithm of the
rounded difference), and checks both integers in 50-digit arithmetic with the closed form of the increment
(the 60 slowest terms; the others are below 1e-400 at these times).  Minimal example of a general pitfall: once
mu_1 ~ N^-2 is below about 1e-9, 1 - mu_1 keeps too few digits of mu_1 for the integer mode.

Output: data/article/s3_methods_log1p_demo.json
Cost  : a few seconds per size (two Brent searches over N - 1 terms, and the 50-digit checks).  No random numbers.
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import json
import math
import os
import sys

import mpmath as mp
import numpy as np
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = _R            # 
OUT = _os.path.join(_R, 'data', 'article', 's3_methods_log1p_demo.json')
sys.path.insert(0, _os.path.join(_R, 'code', 'msc_modes'))
import fptlib as F  # noqa: E402

Q = 0.8
mp.mp.dps = 50


def located_mode(N, use_log1p):
    ch = F.Chain1D(N, Q)
    pos = ch.lam > 0
    Wp, mup = ch.W[pos], ch.mu[pos]
    lap = np.log1p(-mup) if use_log1p else np.log(1.0 - mup)

    def delta(t):
        return float(-(Wp * mup * np.exp((t - 1.0) * lap)).sum())
    t0 = (N - 0.5) ** 2 / 3.0
    tr = brentq(delta, 0.5 * t0 / Q, 1.6 * t0 / Q, rtol=1e-15)
    return int(math.ceil(tr))


def inc50(N, t, M=60):
    """proportional (positive factor) to f(t+1) - f(t), 50 digits"""
    q = mp.mpf(8) / 10
    s = mp.mpf(0)
    for m in range(1, M + 1):
        th = (2 * m - 1) * mp.pi / (2 * N - 1)
        mu = 2 * mp.sin(th / 2) ** 2
        lam = 1 - q * mu
        s += (-1) ** (m + 1) * mp.cos(th / 2) * mp.sin(th) * (-q * mu) * lam ** (t - 1)
    return s


def is_argmax(N, t):
    return bool(inc50(N, t - 1) > 0 and inc50(N, t) <= 0)


rows = []
for N in (100000, 1000000):
    a = located_mode(N, True)
    b = located_mode(N, False)
    r = dict(N=N, q=Q, mode_log1p=a, mode_log_of_rounded_difference=b, displacement_steps=b - a,
             mode_log1p_is_argmax_50_digits=is_argmax(N, a),
             log_of_rounded_difference_is_argmax_50_digits=is_argmax(N, b),
             mu1=float(F.Chain1D(N, Q).mu[0]))
    rows.append(r)
    print(r, flush=True)
json.dump({"generated_by": "code/article/s3_methods_log1p_demo.py", "rows": rows}, open(OUT, "w"), indent=1)
