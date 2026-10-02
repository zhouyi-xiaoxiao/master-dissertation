"""Unified float64 library for the asymptotics of the mode (exploration / sanity checks; rigorous enclosures live in the cert_* scripts).

Reflecting walk on {1..N}^d, corner (1..1) -> corner (N..N).
Scaled spectral data (time unit 1/mu, mu = slowest reflecting rate = (2q/d) sin^2 x, x = pi/2N):
  eps_k = sin^2(kx)/sin^2 x,  gamma_0 = 1, gamma_k = 2 cos^2(kx),
  mu_k = sum_i eps_{k_i},  w_k = prod gamma_{k_i},  c_k = (-1)^{|k|} w_k   (k != 0).
Two time scales:  'C' continuous (E_kappa(t) = exp(-kappa t)),  'D' discrete (E_kappa(t) = (1-kappa)^t, t integer).
All pole data (nu_j, rho_j, m0, kappa1, ...) are the same numbers in both scales up to the trivial factor mu.
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import json, os
import numpy as np
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = _os.path.join(_R, 'data', 'msc_rigorous_asymptotics')
LOGS = _os.path.join(_R, 'out', 'logs', 'msc_rigorous_asymptotics')
MODES_C = _os.path.join(_R, 'data', 'msc_modes', 'laplace_modes.jsonl')
MODES_D = _os.path.join(_R, 'data', 'msc_modes', 'discrete_modes.jsonl')


def spec(d, N):
    """scaled spectrum: arrays mu, w, c over k in {0..N-1}^d minus 0."""
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


def consts(d, N):
    x = np.pi / (2 * N)
    A = 2 * np.cos(x) ** 2
    eps2 = 4 * np.cos(x) ** 2
    gam2 = 2 * np.cos(2 * x) ** 2
    return dict(d=d, N=N, x=x, A=A, W=d * A, eps2=eps2, gam2=gam2, C1=d * (d - 1) * A * A, C2=d * gam2 * eps2,
                pia=float(N) ** (-d), W2=d * (d - 1) / 2 * A * A, W3=(A ** 3 if d == 3 else 0.0))


def scalars(d, N, near=True):
    """exact (float) scaled scalars."""
    mu, w, c = spec(d, N)
    s = consts(d, N)
    W, pia = s['W'], s['pia']
    hi = mu > 1.5
    Xs = np.sum(w / mu)
    sigma = np.sum(w / mu ** 2)
    S1 = np.sum(w[hi] / (mu[hi] * (mu[hi] - 1)))
    EU = -np.sum(c / mu)

    def D(nu):
        return 1.0 - nu * np.sum(w / (mu - nu))

    def rho(nu):
        return 1.0 / np.sum(w * mu / (mu - nu) ** 2)
    nu0 = brentq(D, 1e-14, 1 - 1e-13, xtol=1e-300, rtol=1e-15)
    nu1 = brentq(D, 1 + 1e-13, 2 - 1e-13, xtol=1e-300, rtol=1e-15)
    rho0, rho1 = rho(nu0), rho(nu1)
    th = nu1 - 1
    m0 = 1 - nu0 * np.sum(c / (mu - nu0))
    kap1 = np.sum(c[hi] * mu[hi] ** 2 / (mu[hi] - nu1))
    a1 = rho1 * (W / th + kap1)
    a0 = rho0 * nu0 * m0
    out = dict(s, Xs=Xs, sigma=sigma, S1=S1, EU=EU, X=Xs + EU, nu0=nu0, nu1=nu1, theta=th, rho0=rho0, rho1=rho1,
               Gam=rho1 / th, b0=rho0 / nu0, b1=rho1 / nu1, m0=m0, kap1=kap1, a1=a1, a0=a0)
    if near:
        # near poles: in (2, 3) [d=3 only, level 3 exists], (2 or 3, eps2)
        lv = [2.0] + ([3.0] if d == 3 else []) + [s['eps2']]
        nn, rr = [], []
        for lo, hi_ in zip(lv[:-1], lv[1:]):
            if hi_ - lo < 1e-9:
                continue
            nu = brentq(D, lo * (1 + 1e-13), hi_ * (1 - 1e-13), xtol=1e-300, rtol=1e-15)
            nn.append(nu); rr.append(rho(nu))
        out['near_nu'] = nn; out['near_rho'] = rr
        out['beta2'] = 1 - pia - out['b0'] - out['b1']
        out['beta_far'] = out['beta2'] - sum(r / n for r, n in zip(rr, nn))
    return out


def make_E(scale, mu):
    """E(kappa_scaled, tau): exponential of the time scale; tau in real time units (steps for 'D')."""
    if scale == 'C':
        return lambda kap, tau: np.exp(-kap * mu * tau)
    return lambda kap, tau: (1.0 - kap * mu) ** tau


def mu_real(d, N, q=1.0):
    return (2.0 * q / d) * np.sin(np.pi / (2 * N)) ** 2


# ---------------------------------------------------------------- Theorem A bound functions (scaled coefficients)
def rbar(s, E, tau):
    return s['C1'] * E(2.0, tau) + s['C2'] * E(s['eps2'], tau)                       # r / mu


def r1bar(s, E, tau):
    return 2 * s['C1'] * E(2.0, tau) + s['C2'] * s['eps2'] * E(s['eps2'], tau)       # r1 / mu^2


def Qlow(s, E, tau):
    d, A, e2, g2 = s['d'], s['A'], s['eps2'], s['gam2']
    return d * (d - 1) * (A * A * E(2.0, tau) / 2 - 2 * A * g2 * e2 * E(1 + e2, tau) / (1 + e2)
                          - (d - 2) * A ** 3 * E(3.0, tau) / 3)


def LB(s, E, tau, nq=0.0):
    """lower bound for G / mu^2 (scaled coefficients)."""
    nu0, nu1, rho0, rho1, e2 = s['nu0'], s['nu1'], s['rho0'], s['rho1'], s['eps2']
    C1, C2 = s['C1'], s['C2']
    T1 = 2 * C1 * E(2.0, tau) / (2 - nu1) + C2 * e2 * E(e2, tau) / (e2 - nu1)
    J = C1 * E(2.0, tau) / (2 - nu0) + C2 * E(e2, tau) / (e2 - nu0)
    return (s['a1'] * E(nu1, tau) - rho1 * T1 - rho0 * nu0 * (s['m0'] * E(nu0, tau) + J) - rho0 * rbar(s, E, tau)
            - nq * E(2.0, tau))


def P2bar(s, E, tau, tautil, mu, nq=0.0, Rfar=None):
    """upper bound for P2 / mu^2.  tautil = scaled 'time' mu*tau (C) or mu*tau/(1-2mu) (D)."""
    W, C1, C2, e2, pia = s['W'], s['C1'], s['C2'], s['eps2'], s['pia']
    near = 0.0
    for nu, rh in zip(s['near_nu'], s['near_rho']):
        near += rh * (2 * C1 * min(tautil, 1 / (nu - 2)) + C2 * e2 / (e2 - nu))
    if Rfar is None:
        Rfar = s['d'] / (4 * s['N'] ** s['d'] * np.sin(s['x']) ** 2)      # sum of all rho_j, scaled
    return (E(2.0, tau) * (2 * W * s['beta2'] + near + 2 * C1 * (e2 / (e2 - 2)) * s['beta_far'] + nq)
            + E(e2, tau) * C2 * e2 * tautil * Rfar + pia * r1bar(s, E, tau))


def UB(s, E, tau, tautil, mu, nq=0.0):
    return s['a1'] * E(s['nu1'], tau) - s['a0'] * E(s['nu0'], tau) + P2bar(s, E, tau, tautil, mu, nq)


def exact_modes_C():
    out = {}
    with open(MODES_C) as f:
        for line in f:
            r = json.loads(line)
            if r.get('geo') == 'CC' and r.get('rate', 1.0) == 1.0:
                out[(r['d'], r['N'])] = (r['mode'], r['mfpt'])
    return out


def exact_modes_D():
    out = {}
    with open(MODES_D) as f:
        for line in f:
            r = json.loads(line)
            if r.get('geo') == 'CC':
                out[(r['d'], r['N'], r['q'])] = (r['mode'], r.get('mfpt_exact'))
    return out
