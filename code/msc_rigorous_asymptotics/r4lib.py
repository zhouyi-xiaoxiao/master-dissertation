"""Float64 library for the asymptotics of the mode (used for SANITY CHECKS of every lemma; rigorous enclosures are in certlib.py and the *_cert_*.py scripts).

Notation = note R3.  Scaled time (unit 1/mu, mu = (2q/d) sin^2 x, x = pi/(2N)), continuous time unless stated.
  eps_k = sin^2(kx)/sin^2 x,  gamma_0 = 1, gamma_k = 2 cos^2(kx)
  mu_k = sum_i eps_{k_i},  w_k = prod_i gamma_{k_i},  c_k = (-1)^{|k|} w_k      (k in {0..N-1}^d minus 0)
  A = gamma_1, W = d A, eps2 = eps_2 = 4cos^2 x, gam2 = gamma_2, C1 = d(d-1)A^2, C2 = d gam2 eps2, W2 = C1/2, W3 = A^3
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
    C1 = d * (d - 1) * A * A
    return dict(d=d, N=N, x=x, A=A, W=d * A, eps2=eps2, gam2=gam2, C1=C1, C2=d * gam2 * eps2,
                W2=C1 / 2, W3=(A ** 3 if d == 3 else 0.0), pia=float(N) ** (-d),
                Rtot=d / (4 * float(N) ** d * np.sin(x) ** 2))


def scalars(d, N):
    """exact (float) scalars: lattice sums and pole data."""
    mu, w, c = spec(d, N)
    s = consts(d, N)
    W, pia = s['W'], s['pia']
    hi = mu > 1.5
    Xs = np.sum(w / mu)
    sigma = np.sum(w / mu ** 2)
    EU = -np.sum(c / mu)

    def D(nu):
        return 1.0 - nu * np.sum(w / (mu - nu))

    def rho(nu):
        return 1.0 / np.sum(w * mu / (mu - nu) ** 2)
    nu0 = brentq(D, 1e-14, 1 - 1e-13, xtol=1e-300, rtol=1e-15)
    nu1 = brentq(D, 1 + 1e-13, 2 - 1e-13, xtol=1e-300, rtol=1e-15)
    rho0, rho1 = rho(nu0), rho(nu1)
    th = nu1 - 1
    m0 = 1 - nu0 * np.sum(c / (mu - nu0))                       # = -sum c mu/(mu-nu0)  (uses sum c = -1)
    kap1 = np.sum(c[hi] * mu[hi] ** 2 / (mu[hi] - nu1))         # = int_0^inf r1 e^{nu1 s} ds
    S1 = np.sum(w[hi] / (mu[hi] * (mu[hi] - 1)))
    Snu = np.sum(w[hi] / (mu[hi] * (mu[hi] - nu1)))
    S2nu = np.sum(w[hi] / (mu[hi] - nu1) ** 2)
    Ir = np.sum(c[hi] * mu[hi] / (mu[hi] - 1))                  # int_0^inf r e^{s} ds
    out = dict(s, Xs=Xs, y=1 / Xs, sigma=sigma, EU=EU, X=Xs + EU, nu0=nu0, nu1=nu1, theta=th, rho0=rho0, rho1=rho1,
               b0=rho0 / nu0, b1=rho1 / nu1, m0=m0, kap1=kap1, S1=S1, Snu=Snu, S2nu=S2nu, Ir=Ir,
               a1=W * rho1 / th, a0=rho0 * nu0 * m0)
    # near poles (between level 2 and eps2)
    lv = [2.0] + ([3.0] if d == 3 else []) + [s['eps2']]
    nn, rr = [], []
    for lo, hi_ in zip(lv[:-1], lv[1:]):
        if hi_ - lo < 1e-9:
            continue
        nu = brentq(D, lo * (1 + 1e-13), hi_ * (1 - 1e-13), xtol=1e-300, rtol=1e-15)
        nn.append(nu); rr.append(rho(nu))
    out['near_nu'] = nn; out['near_rho'] = rr
    out['betahat'] = 1 - out['b0'] - out['b1']                 # = beta_2 + pi_a
    out['tau_cf'] = out['X'] * np.log(2 * d * out['X']) / (out['X'] + 2 * d - 1)
    out['t_c'] = np.log(out['a1'] / out['a0']) / (nu1 - nu0)
    return out


# ------------------------------------------------------------------ bound functions of Theorem A (exact pole data)
def ell(s, t):
    return s['A'] * np.exp(-t) - s['gam2'] * np.exp(-s['eps2'] * t)


def g_ell(s, t):
    l = ell(s, t)
    return l if s['d'] == 2 else 2 * l - l * l


def rbar(s, t):
    return s['C1'] * np.exp(-2 * t) + s['C2'] * np.exp(-s['eps2'] * t)


def r1bar(s, t):
    return 2 * s['C1'] * np.exp(-2 * t) + s['C2'] * s['eps2'] * np.exp(-s['eps2'] * t)


def rlow(s, t):
    return s['W'] * np.exp(-t) * g_ell(s, t)


def _F(a, t):
    return -np.expm1(-a * t) / a


def Lr(s, t, th=0.0):
    """lower bound for int_0^t r(s') e^{(1+th) s'} ds'  (uses r >= W e^{-s} g_ell(s))."""
    A, g2, e2, W = s['A'], s['gam2'], s['eps2'], s['W']
    if s['d'] == 2:
        return W * (A * _F(1 - th, t) - g2 * _F(e2 - th, t))
    return W * (2 * A * _F(1 - th, t) - 2 * g2 * _F(e2 - th, t) - A * A * _F(2 - th, t)
                + 2 * A * g2 * _F(1 + e2 - th, t) - g2 * g2 * _F(2 * e2 - th, t))


def kappa_lo(s, t, th=None):
    """lower bound for int_0^t r1 e^{nu1 s} ds = W - r(t) e^{nu1 t} + nu1 int_0^t r e^{nu1 s} ds."""
    th = s['theta'] if th is None else th
    return max(0.0, s['W'] - rbar(s, t) * np.exp((1 + th) * t) + (1 + th) * Lr(s, t, th))


def Iplus(s, th):
    W, C1, C2, e2 = s['W'], s['C1'], s['C2'], s['eps2']
    sx = np.log(C1 / W)
    return W * np.expm1(th * sx) / th + C1 * np.exp(-(1 - th) * sx) / (1 - th) + C2 * np.exp(-(e2 - 1 - th) * sx) / (e2 - 1 - th)


def kappa_hi(s, th=None):
    th = s['theta'] if th is None else th
    return s['W'] + (1 + th) * Iplus(s, th)


def Eminus_hi(s, t):
    nu0, e2 = s['nu0'], s['eps2']
    return (2 * s['C1'] * np.exp(-(2 - nu0) * t) / (2 - nu0) + s['C2'] * e2 * np.exp(-(e2 - nu0) * t) / (e2 - nu0)) / (nu0 * s['m0'])


def Eminus_lo(s, t):
    return max(0.0, s['W'] * np.exp(-(1 - s['nu0']) * t) * g_ell(s, t) / (s['nu0'] * s['m0']))


def Prest_hi(s, t):
    W, C1, C2, e2 = s['W'], s['C1'], s['C2'], s['eps2']
    cb = 2 * W + 2 * C1 * e2 / (e2 - 2)
    near = 0.0
    for nu, rh in zip(s['near_nu'], s['near_rho']):
        near += rh * (2 * C1 * min(t, 1 / (nu - 2)) + C2 * e2 * min(t, 1 / (e2 - nu)))
    return np.exp(-2 * t) * (cb * s['betahat'] + near) + np.exp(-e2 * t) * C2 * e2 * (s['pia'] + t * s['Rtot'])


def Phi_minus(s, t):
    return s['a1'] * np.exp(-s['nu1'] * t) * (1 + s['theta'] * kappa_lo(s, t) / s['W']) - s['a0'] * np.exp(-s['nu0'] * t) * (1 + Eminus_hi(s, t))


def Phi_plus(s, t):
    return (s['a1'] * np.exp(-s['nu1'] * t) * (1 + s['theta'] * kappa_hi(s) / s['W']) + Prest_hi(s, t)
            - s['a0'] * np.exp(-s['nu0'] * t) * (1 + Eminus_lo(s, t)))


def window(s):
    """(tau_minus, tau_plus): sup{t: Phi_minus > 0 near t_c} and inf{t: Phi_plus < 0}; float root finding."""
    tc = s['t_c']
    lo, hi = None, None
    if Phi_minus(s, tc - 3.0) > 0 or True:
        # largest t with Phi_minus(t) > 0: scan down from tc + 1
        ts = np.linspace(max(0.5, tc - 3), tc + 1.5, 4000)
        vm = np.array([Phi_minus(s, t) for t in ts])
        pos = np.where(vm > 0)[0]
        if len(pos):
            i = pos[-1]
            lo = brentq(lambda t: Phi_minus(s, t), ts[i], ts[i + 1]) if i + 1 < len(ts) else ts[i]
        vp = np.array([Phi_plus(s, t) for t in ts])
        neg = np.where(vp < 0)[0]
        if len(neg):
            i = neg[0]
            hi = brentq(lambda t: Phi_plus(s, t), ts[i - 1], ts[i]) if i > 0 else ts[i]
    return lo, hi


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


def mu_real(d, N, q=1.0):
    return (2.0 * q / d) * np.sin(np.pi / (2 * N)) ** 2
