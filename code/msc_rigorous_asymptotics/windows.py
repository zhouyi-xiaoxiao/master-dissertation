"""Semi-explicit window of Theorem A (continuous time, units mu=1), float evaluation from a dict of scalars.

Required keys: d, N, c2, x, W, pia, nu0, nu1, theta, rho0, Gam, b0, beta2, m0, k1  (exact values or valid bounds:
the caller is responsible for passing a1 as lower/upper bounds where needed; here everything is treated as exact).
"""
import numpy as np
from scipy.optimize import brentq


def constants(s):
    d, c2, x, W = s['d'], s['c2'], s['x'], s['W']
    eps2 = 4 * c2
    gam2 = 2 * np.cos(2 * x) ** 2
    ta = 1 / eps2
    tb = (2 * eps2 - 1) / (eps2 * (eps2 - 1))
    A = 2 * (d - 1) * c2                 # rho_bar(t) = A e^{-t} + B e^{-kap t},  t >= ta
    B = 4 * np.cos(2 * x) ** 2
    kap = eps2 - 1
    Cq_inf = 2 * d * (d - 1) * (2 * c2) ** 2
    D0 = d * (eps2 + 2 * c2)
    D1 = d * gam2 * eps2 ** 2
    Cq_p = D0 * np.exp(2 * tb) + D1 * np.exp(-(eps2 - 2) * tb) * eps2 / (eps2 - 2)
    return dict(eps2=eps2, gam2=gam2, ta=ta, tb=tb, A=A, B=B, kap=kap, Cq_inf=Cq_inf, D0=D0, D1=D1, Cq_p=Cq_p)


def rho_bar(s, t):
    c = constants(s)
    return c['A'] * np.exp(-t) + c['B'] * np.exp(-c['kap'] * t)


def window(s, k1_lo=None, k1_hi=None, m0_lo=None, m0_hi=None):
    """return dict with t1, t_minus, t_plus (mu=1 units) or raises if the conditions cannot be met."""
    c = constants(s)
    d, W, pia = s['d'], s['W'], s['pia']
    nu0, nu1, th, rho0, Gam, b0, beta2 = s['nu0'], s['nu1'], s['theta'], s['rho0'], s['Gam'], s['b0'], s['beta2']
    k1_lo = s['k1'] if k1_lo is None else k1_lo
    k1_hi = s['k1'] if k1_hi is None else k1_hi
    m0_lo = s['m0'] if m0_lo is None else m0_lo
    m0_hi = s['m0'] if m0_hi is None else m0_hi
    a1_lo = Gam * nu1 * W * (1 + th * k1_lo)
    a1_hi = Gam * nu1 * W * (1 + th * k1_hi)
    a0_lo = rho0 * nu0 * m0_lo
    a0_hi = rho0 * nu0 * m0_hi
    A, B, kap = c['A'], c['B'], c['kap']
    CA = A * W * (Gam * nu1 * th / (1 - th) + rho0 * nu0 / (2 - nu0) + rho0 + Gam * th)
    CB = B * W * (Gam * nu1 * th / (kap - th) + rho0 * nu0 / (kap + 1 - nu0) + rho0 + Gam * th)

    def psiL(t):
        return a1_lo - a0_hi * np.exp((nu1 - nu0) * t) - CA * np.exp(-(1 - th) * t) - CB * np.exp(-(kap - th) * t)

    def P2bar(t):
        return np.exp(-2 * t) * (pia * (c['Cq_inf'] + c['D1'] * np.exp(-(c['eps2'] - 2) * c['tb']))
                                 + beta2 * (2 * W + c['Cq_inf'] * (1 + 2 * t) + c['Cq_p']))

    def psiU(t):
        return a1_hi * np.exp(-(nu1 - nu0) * t) - a0_lo + P2bar(t) * np.exp(nu0 * t)
    # early region: need t1 >= max(ta, tb) with W e^{-t1}(1-rho_bar(t1)) > eta1 and psiL(t1) > 0
    delta2 = s['delta2']

    def eps_early(t):
        return (delta2 + pia) / rho0 + (Gam * th / rho0) * max(nu1 * t - 1.0, 0.0)

    def early_ok(t):
        e = eps_early(t)
        return e < 1 and W * np.exp(-t) * (1 - rho_bar(s, t)) > nu0 / (1 - e)
    tgrid = np.linspace(c['ta'], 12, 6000)
    ok = [early_ok(t) and psiL(t) > 0 for t in tgrid]
    if not any(ok):
        return dict(ok=False, reason='no t1')
    t1 = tgrid[ok.index(True)]
    eta1 = nu0 / (1 - eps_early(t1))
    # upper root of psiL (concave): maximise first
    tt = np.linspace(t1, 40, 20000)
    vals = psiL(tt)
    imax = int(np.argmax(vals))
    hi = tt[np.where(vals > 0)[0][-1]]
    tm = brentq(psiL, hi, hi + 1.0) if psiL(hi + 1.0) < 0 else hi
    tm = tm * (1 - 1e-12)
    # t_plus: psiU < 0 from there on (psiU decreasing for t >= 1)
    valsU = psiU(tt)
    neg = np.where(valsU < 0)[0]
    if len(neg) == 0:
        return dict(ok=False, reason='psiU never negative')
    # first index after which all negative
    pos = np.where(valsU >= 0)[0]
    lo = tt[pos[-1]] if len(pos) else t1
    tp = brentq(psiU, lo, lo + (tt[1] - tt[0]) * 2) if len(pos) else t1
    tau_c_lo = np.log(a1_lo / a0_hi) / (nu1 - nu0)
    tau_c_hi = np.log(a1_hi / a0_lo) / (nu1 - nu0)
    return dict(ok=True, t1=float(t1), t_minus=float(tm), t_plus=float(tp), eta1=float(eta1), tau_c_lo=float(tau_c_lo),
                tau_c_hi=float(tau_c_hi), CA_over_a1=float(CA / a1_lo), P2bar_rel=float(P2bar(tp) * np.exp(nu0 * tp) / a0_lo))
