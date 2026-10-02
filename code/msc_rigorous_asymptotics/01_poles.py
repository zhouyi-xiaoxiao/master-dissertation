"""Numerical exploration of the two slowest poles of hhat = 1/(1+sR) and of the derived constants."""
import json, sys
import numpy as np
from common import basic, poles01
out = []
cases = [(2, N) for N in (5, 10, 20, 35, 50, 100, 200, 400, 1000, 2000)] + [(3, N) for N in (5, 10, 20, 40, 80, 160)]
for d, N in cases:
    B = basic(d, N)
    mu, W, Xs, sig, pia = B['mu'], B['W'], B['Xs'], B['sigma'], B['pia']
    nu0, nu1, r0, r1 = poles01(B)
    b0, b1 = r0 / nu0, r1 / nu1
    beta = 1 - pia - b0
    Gam = r1 / (nu1 - mu)
    delta2 = r0 / (mu - nu0) - pia - Gam
    th = nu1 / mu - 1
    rec = dict(d=d, N=N, Xs=Xs, X=B['X'], sigma=sig, nu0_M=nu0 * B['M'], one_minus_b0_X2=(1 - b0) * Xs ** 2,
               theta_X_over_W=th * Xs / W, b1_X2_over_W=b1 * Xs ** 2 / W, beta2_X2=(beta - b1) * Xs ** 2,
               delta2_X2=delta2 * Xs ** 2, Gam_X=Gam * Xs, muEU=B['X'] - Xs)
    out.append(rec)
    print(' '.join(f'{k}={v:.6g}' if isinstance(v, float) else f'{k}={v}' for k, v in rec.items()), flush=True)
json.dump(out, open('../data/01_poles.json', 'w'), indent=1)
