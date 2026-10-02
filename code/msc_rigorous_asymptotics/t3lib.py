"""Float64 exploration library for the asymptotics of the mode (rigorous enclosures are in the cert_* scripts).

Continuous-time corner-to-corner walk on {1..N}^d, time unit chosen so that mu = eps_1 = 1.
  eps_k = sin^2(k x)/sin^2(x), x = pi/(2N); gamma_k = alpha_k cos^2(k x).
  mu_k = sum_i eps_{k_i}, w_k = prod gamma_{k_i}, c_k = (-1)^{|k|} w_k.
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import json, os
import numpy as np
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = _os.path.join(_R, 'data', 'msc_rigorous_asymptotics')
MODES = _os.path.join(_R, 'data', 'msc_modes', 'laplace_modes.jsonl')


def spec(d, N):
    k = np.arange(N)
    x = np.pi / (2 * N)
    eps = np.sin(k * x) ** 2 / np.sin(x) ** 2
    gam = np.where(k == 0, 1.0, 2.0) * np.cos(k * x) ** 2
    sg = np.where(k % 2 == 0, 1.0, -1.0) * gam
    if d == 1:
        mu, w, c = eps, gam, sg
    elif d == 2:
        mu = (eps[:, None] + eps[None, :]).ravel()
        w = (gam[:, None] * gam[None, :]).ravel()
        c = (sg[:, None] * sg[None, :]).ravel()
    elif d == 3:
        mu = (eps[:, None, None] + eps[None, :, None] + eps[None, None, :]).ravel()
        w = (gam[:, None, None] * gam[None, :, None] * gam[None, None, :]).ravel()
        c = (sg[:, None, None] * sg[None, :, None] * sg[None, None, :]).ravel()
    return mu[1:], w[1:], c[1:]


def scalars(d, N):
    mu, w, c = spec(d, N)
    x = np.pi / (2 * N)
    c2 = np.cos(x) ** 2
    W = 2 * d * c2
    pia = float(N) ** (-d)
    Xs = np.sum(w / mu)
    sigma = np.sum(w / mu ** 2)
    hi = mu > 1.5                      # mu_k >= 2
    S1 = np.sum(w[hi] / (mu[hi] * (mu[hi] - 1)))
    S2 = np.sum(w[hi] / (mu[hi] - 1) ** 2)
    EU = -np.sum(c / mu)
    eH = (2.0 / 3.0) * (N * N - 1) * np.sin(x) ** 2

    def D(nu):
        return 1.0 - nu * np.sum(w / (mu - nu))

    def rho(nu):
        return 1.0 / np.sum(w * mu / (mu - nu) ** 2)
    nu0 = brentq(D, 1e-14, 1 - 1e-13, xtol=1e-300, rtol=1e-15)
    nu1 = brentq(D, 1 + 1e-13, 2 - 1e-13, xtol=1e-300, rtol=1e-15)
    rho0, rho1 = rho(nu0), rho(nu1)
    th = nu1 - 1
    Gam = rho1 / th
    b0, b1 = rho0 / nu0, rho1 / nu1
    delta2 = rho0 / (1 - nu0) - pia - Gam
    beta2 = 1 - pia - b0 - b1
    m0 = 1 - nu0 * np.sum(c / (mu - nu0))
    k1 = np.sum(c[hi] * mu[hi] / (mu[hi] - nu1)) / W
    return dict(d=d, N=N, x=x, c2=c2, W=W, pia=pia, Xs=Xs, sigma=sigma, S1=S1, S2=S2, EU=EU, eH=eH, X=Xs + EU,
                nu0=nu0, nu1=nu1, theta=th, rho0=rho0, rho1=rho1, Gam=Gam, b0=b0, b1=b1, delta2=delta2,
                beta2=beta2, m0=m0, k1=k1)


def exact_modes():
    """exact continuous-time modes (unit rate q=1) from the floating-point study: dict (d,N)->(mode, mfpt)."""
    out = {}
    with open(MODES) as f:
        for line in f:
            r = json.loads(line)
            if r.get('geo') == 'CC' and r.get('rate', 1.0) == 1.0:
                out[(r['d'], r['N'])] = (r['mode'], r['mfpt'])
    return out


def mu_real(d, N, q=1.0):
    return (q / d) * (1 - np.cos(np.pi / N))
