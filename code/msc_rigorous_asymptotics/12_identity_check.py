"""Brute-force check (dense eigendecomposition, small N) of the exact identity and of every bound used in Theorem A.
Continuous time, units mu = 1.
 (a) g'(t) = Gam nu1 e^{-nu1 t}[W + th I1(t)] - rho0 nu0 e^{-nu0 t}[m0 + J(t)] - (rho0+Gam th) r(t) + P2(t)
 (b) 0 <= P2 <= P2bar (t >= tb);  (c) 0 <= r <= W e^{-t} rho_bar (t >= ta); q >= 0; q <= Cq_inf e^{-2t} + qtilde bounds
 (d) early bound  g' >= u'[rho0 - delta2 - pia - Gam th (nu1 t - 1)_+] - rho0 nu0 u
 (e) value bound  g <= rho0 u + [Gam th t + delta2 + pia] u'
 (f) g' e^{t} + g e^{t} >= 0  (i.e. (g e^{t})' >= 0)
"""
import json, sys
import numpy as np
from t3lib import spec, scalars, mu_real, DATA
from windows import constants, rho_bar


def killed(d, N):
    n = N ** d
    idx = np.arange(n).reshape((N,) * d)
    L = np.zeros((n, n))
    rate = 1.0 / (2 * d)
    for ax in range(d):
        a = np.moveaxis(idx, ax, 0)
        src = a[:-1].ravel(); dst = a[1:].ravel()
        L[src, src] += rate; L[dst, dst] += rate
        L[src, dst] -= rate; L[dst, src] -= rate
    keep = np.arange(n - 1)             # target = last site (N,..,N); start = site 0
    return L[np.ix_(keep, keep)] / mu_real(d, N)


def run(d, N):
    s = scalars(d, N)
    cst = constants(s)
    mu, w, c = spec(d, N)
    W, pia = s['W'], s['pia']
    nu0, nu1, th, rho0, rho1, Gam, m0 = s['nu0'], s['nu1'], s['theta'], s['rho0'], s['rho1'], s['Gam'], s['m0']
    La = killed(d, N)
    lam, V = np.linalg.eigh(La)
    one = np.ones(La.shape[0])
    cx = V[0, :]; c1 = V.T @ one
    b = c1 ** 2 / N ** d
    aF = lam * cx * c1
    g = lambda t: float(np.sum(aF * np.exp(-lam * t)))
    g1 = lambda t: float(-np.sum(aF * lam * np.exp(-lam * t)))
    hi = mu > 1.5
    u = lambda t: 1 + float(np.sum(c * np.exp(-mu * t)))
    u1 = lambda t: -float(np.sum(c * mu * np.exp(-mu * t)))
    r = lambda t: float(np.sum(c[hi] * mu[hi] * np.exp(-mu[hi] * t)))
    q = lambda t: float(np.sum(c[hi] * mu[hi] ** 2 * np.exp(-mu[hi] * t)))
    I1 = lambda t: float(np.sum(c[hi] * mu[hi] * (1 - np.exp(-(mu[hi] - nu1) * t)) / (mu[hi] - nu1)))      # int_0^t r e^{nu1 s}
    J = lambda t: float(np.sum(c[hi] * mu[hi] * np.exp(-(mu[hi] - nu0) * t) / (mu[hi] - nu0)))             # int_t^inf r e^{nu0 s}
    # visible poles >= 2 (aggregate)
    vis = b > 1e-13
    lv, bv = lam[vis], b[vis]
    j2 = lv > 1.9
    nus, rhos = lv[j2], (bv * lv)[j2]

    def qconv(nu, t):
        return float(np.sum(c[hi] * mu[hi] ** 2 * (np.exp(-nu * t) - np.exp(-mu[hi] * t)) / (mu[hi] - nu)))

    def P2(t):
        return pia * q(t) + sum(rh * (W * np.exp(-nu * t) / (nu - 1) + qconv(nu, t)) for nu, rh in zip(nus, rhos))

    def P2bar(t):
        return np.exp(-2 * t) * (pia * (cst['Cq_inf'] + cst['D1'] * np.exp(-(cst['eps2'] - 2) * cst['tb']))
                                 + s['beta2'] * (2 * W + cst['Cq_inf'] * (1 + 2 * t) + cst['Cq_p']))
    res = dict(d=d, N=N, n_poles_ge2=int(len(nus)), sum_rho_check=float(abs(pia + np.sum(bv) - 1)))
    ident, viol = 0.0, []
    for t in np.concatenate([np.linspace(0.02, 3, 60), np.linspace(3, 14, 45)]):
        lhs = g1(t)
        rhs = Gam * nu1 * np.exp(-nu1 * t) * (W + th * I1(t)) - rho0 * nu0 * np.exp(-nu0 * t) * (m0 + J(t)) \
            - (rho0 + Gam * th) * r(t) + P2(t)
        scale = rho0 * nu0 + abs(lhs)
        ident = max(ident, abs(lhs - rhs) / scale)
        tol = 1e-9 * scale
        if P2(t) < -tol: viol.append(('P2<0', t))
        if t >= cst['tb'] and P2(t) > P2bar(t) * (1 + 1e-9) + tol: viol.append(('P2>P2bar', t, P2(t) / P2bar(t)))
        if r(t) < -1e-11: viol.append(('r<0', t))
        if t >= cst['ta'] and r(t) > W * np.exp(-t) * rho_bar(s, t) * (1 + 1e-10) + 1e-13: viol.append(('r>bar', t))
        if q(t) < -1e-9: viol.append(('q<0', t, q(t)))
        qb = cst['Cq_inf'] * np.exp(-2 * t) + (cst['D1'] * np.exp(-cst['eps2'] * t) if t >= cst['tb'] else cst['D0'])
        if q(t) > qb * (1 + 1e-10) + 1e-12: viol.append(('q>bar', t, q(t) / qb))
        early = u1(t) * (rho0 - s['delta2'] - pia - Gam * th * max(nu1 * t - 1, 0)) - rho0 * nu0 * u(t)
        if lhs < early - tol: viol.append(('early', t))
        val = rho0 * u(t) + (Gam * th * t + s['delta2'] + pia) * u1(t)
        if g(t) > val * (1 + 1e-10) + 1e-14: viol.append(('value', t, g(t) / val))
        if lhs + g(t) < -tol: viol.append(('g e^t decreasing', t))
    res['identity_max_relerr'] = ident
    res['violations'] = viol
    return res


if __name__ == '__main__':
    out = []
    for d, N in [(2, 6), (2, 9), (2, 14), (2, 22), (2, 30), (3, 4), (3, 5), (3, 7), (3, 9)]:
        rr = run(d, N); out.append(rr); print(json.dumps(rr, default=str), flush=True)
    json.dump(out, open(DATA + '/12_identity_check.json', 'w'), indent=1, default=str)
