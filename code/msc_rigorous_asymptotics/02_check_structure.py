"""Numerical sanity check (small N, full diagonalisation) of the exact structure used in the proof:
  (a) decomposition  g = pi(a) u' + u' * h_ac   (T = U + T_pi)
  (b) u = v^d, v' = hypoexponential density, bounds v' <= 2c^2 mu e^{-mu t}, -v'' <= mu v', q>=0
  (c) formula (STAR) and the two-pole coefficients a0', a1'
Continuous time, activity q=1.
"""
import sys, json
import numpy as np
from scipy.integrate import quad
from common import basic

def killed_generator(d, N, q=1.0):
    """Symmetric generator L (positive semidefinite) of the reflecting walk; return L restricted to non-target sites."""
    n = N ** d
    idx = np.arange(n).reshape((N,) * d)
    L = np.zeros((n, n))
    rate = q / (2 * d)
    for ax in range(d):
        a = np.moveaxis(idx, ax, 0)
        src = a[:-1].ravel(); dst = a[1:].ravel()
        L[src, src] += rate; L[dst, dst] += rate
        L[src, dst] -= rate; L[dst, src] -= rate
    tgt = n - 1; start = 0
    keep = np.arange(n - 1)
    return L[np.ix_(keep, keep)], start

def run(d, N):
    B = basic(d, N)
    mu, A, W, pia, c2 = B['mu'], B['A'], B['W'], B['pia'], B['c2']
    eps = B['eps'][1:]
    k = np.arange(1, N)
    ck2 = np.cos(np.pi * k / (2 * N)) ** 2
    sgn = (-1.0) ** (k + 1)
    def v(t):   return 1 - np.sum(2 * sgn * ck2 * np.exp(-eps * t))
    def v1(t):  return np.sum(2 * sgn * ck2 * eps * np.exp(-eps * t))
    def v2(t):  return -np.sum(2 * sgn * ck2 * eps ** 2 * np.exp(-eps * t))
    def u1(t):  return d * v(t) ** (d - 1) * v1(t)
    def u2(t):  return d * (d - 1) * v(t) ** (d - 2) * v1(t) ** 2 + d * v(t) ** (d - 1) * v2(t)
    La, start = killed_generator(d, N)
    nu, V = np.linalg.eigh(La)
    one = np.ones(La.shape[0])
    cx = V[start, :]; c1 = V.T @ one
    b = c1 ** 2 / N ** d                    # P_pi(T>t) = sum b_j e^{-nu_j t}
    rho = b * nu
    aF = nu * cx * c1                       # g(t) = sum aF_j e^{-nu_j t}
    def g(t):  return np.sum(aF * np.exp(-nu * t))
    def g1(t): return -np.sum(aF * nu * np.exp(-nu * t))
    def hac(t): return np.sum(rho * np.exp(-nu * t))
    res = dict(d=d, N=N, sum_b=float(b.sum()), one_minus_pia=1 - pia)
    # (a) decomposition at a few times
    errs = []
    for tau in (0.5, 1.5, 3.0, 6.0):
        t = tau / mu
        conv = quad(lambda s: u1(s) * hac(t - s), 0, t, limit=400, epsabs=0, epsrel=1e-11)[0]
        errs.append(abs(pia * u1(t) + conv - g(t)) / g(t))
    res['decomp_relerr_max'] = max(errs)
    # (b) bounds on grid
    taus = np.linspace(0.02, 12, 600)
    bad = 0; qmin = 1e9
    for tau in taus:
        t = tau / mu
        if v1(t) > 2 * c2 * mu * np.exp(-tau) * (1 + 1e-12) or v1(t) < -1e-18: bad += 1
        if -v2(t) > mu * v1(t) * (1 + 1e-9) + 1e-18: bad += 1
        qq = u2(t) + A * mu * np.exp(-tau)
        qmin = min(qmin, qq / (A * mu * np.exp(-tau)))
    res['bound_violations'] = bad; res['q_min_rel'] = qmin
    # (c) poles: visible ones (b_j>0), sorted
    vis = np.where(b > 1e-14)[0]
    j0, j1 = vis[0], vis[1]
    nu0, nu1, rho0, rho1 = nu[j0], nu[j1], rho[j0], rho[j1]
    th = nu1 / mu - 1
    Gam = rho1 / (nu1 - mu)
    # identity hhat(-mu) = 0
    res['hhat_at_minus_mu'] = float(pia + np.sum(rho[vis] / (nu[vis] - mu)))
    m0 = quad(lambda s: u1(s) * np.exp(nu0 * s), 0, 60 / mu, limit=800, epsrel=1e-12)[0]
    k1 = quad(lambda tau: (1 - u1(tau / mu) / (A * np.exp(-tau))) * np.exp(th * tau), 0, 40, limit=800, epsrel=1e-11)[0]
    a0p = rho0 * nu0 * m0
    a1p = nu1 * Gam * A * (1 + th * k1)
    # compare with residues of F: aggregate degenerate eigenvalues equal to nu0 / nu1
    a0F = np.sum(aF[np.abs(nu - nu0) < 1e-9 * mu]); a1F = np.sum(aF[np.abs(nu - nu1) < 1e-9 * mu])
    res['a0p_vs_residue'] = float(a0p / (a0F * nu0) - 1)
    res['a1p_vs_residue'] = float(a1p / (-a1F * nu1) - 1)
    res.update(theta=float(th), k1=float(k1), m0=float(m0), Xs=float(B['Xs']), Lambda=float(a1p / a0p),
               tau_c=float(np.log(a1p / a0p) / (nu1 - nu0) * mu))
    # exact mode by root of g1
    from scipy.optimize import brentq
    tc = np.log(a1p / a0p) / (nu1 - nu0)
    ts = brentq(g1, 0.3 * tc, 3 * tc, xtol=1e-14 * tc)
    res['tau_star'] = float(ts * mu); res['X_times_(tau_c-tau_star)'] = float((tc - ts) * mu * B['Xs'])
    # (STAR): check E(t) = g' - a1' e^{-nu1 t} + a0' e^{-nu0 t} sign/size
    Es = []
    for tau in (2.0, 3.0, res['tau_star'], 6.0, 8.0):
        t = tau / mu
        E = g1(t) - a1p * np.exp(-nu1 * t) + a0p * np.exp(-nu0 * t)
        Es.append((tau, E / (rho0 * mu * A * np.exp(-2 * tau))))   # normalised by rho0 * A mu e^{-2 tau}
    res['E_norm'] = Es
    return res

if __name__ == '__main__':
    out = []
    for d, N in [(2, 8), (2, 14), (2, 24), (2, 36), (3, 5), (3, 8), (3, 11)]:
        r = run(d, N); out.append(r)
        print(json.dumps(r), flush=True)
    json.dump(out, open('../data/02_check_structure.json', 'w'), indent=1)
