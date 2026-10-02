"""Brute-force sanity check (dense eigendecomposition + independent time stepping) of every structural statement
used in Theorem A, in BOTH time scales (C = continuous, D = discrete with activity q).

 (a) g = sum_j a_j E_{nu_j}  agrees with direct time stepping (D) / matrix exponential (C)
 (b) exact identity for G = D g
 (c) r >= 0, r <= rbar, qt >= 0, qt <= qbar, qt >= qlow, r >= rlow, J in [0, Jbar]
 (d) LB <= G <= UB (UB with exact near poles), P2 >= -(negative pole bound)
 (e) value bound, u + phi/mu = 1 - Q, Q >= Qlow, Q nonincreasing
 (f) pole bounds of Lemma 'poles'
Usage: python 21_bruteforce.py           (writes data/21_bruteforce.json)
"""
import json, sys
import numpy as np
from scipy.linalg import expm
from u3lib import spec, scalars, consts, make_E, mu_real, rbar, r1bar, Qlow, LB, UB, DATA


def L1(d, N):
    """unit-rate generator I - P_1 of the reflecting walk (dense)."""
    n = N ** d
    idx = np.arange(n).reshape((N,) * d)
    L = np.zeros((n, n))
    rate = 1.0 / (2 * d)
    for ax in range(d):
        a = np.moveaxis(idx, ax, 0)
        src = a[:-1].ravel(); dst = a[1:].ravel()
        L[src, src] += rate; L[dst, dst] += rate
        L[src, dst] -= rate; L[dst, src] -= rate
    return L


def run(d, N, scale, q):
    s = scalars(d, N)
    mu = mu_real(d, N, q)
    E = make_E(scale, mu)
    mus, w, c = spec(d, N)                       # scaled
    n = N ** d
    La = q * L1(d, N)[:n - 1, :n - 1]            # target = last site, start = site 0
    lam, V = np.linalg.eigh(La)
    cx = V[0, :]; c1 = V.T @ np.ones(n - 1)
    b = c1 ** 2 / n
    a = lam * cx * c1
    nus = lam / mu                                # scaled poles
    if scale == 'C':
        Ep = lambda tau: np.exp(-lam * tau)
    else:
        Ep = lambda tau: (1 - lam) ** tau
    g = lambda tau: float(np.sum(a * Ep(tau)))
    G = lambda tau: float(-np.sum(a * lam * Ep(tau)))
    res = dict(d=d, N=N, scale=scale, q=q, mu=mu, Xs=s['Xs'])
    # ---- (a) independent evaluation
    if scale == 'D':
        P = np.eye(n - 1) - La
        p = np.zeros(n - 1); p[0] = 1.0
        surv_prev = 1.0; f = []
        for t in range(1, 400):
            p = p @ P
            sv = p.sum(); f.append(surv_prev - sv); surv_prev = sv
        f = np.array(f)                           # f[t-1] = f(t)
        err = max(abs(f[t] - g(t)) for t in range(0, 399))   # g(tau) = f(tau+1)
        res['a_stepper_abs_err'] = float(err); res['a_fmax'] = float(f.max())
    else:
        errs = []
        for tau in (0.5 / mu, 2.0 / mu, 5.0 / mu):
            M = expm(-La * tau)
            gd = float(M[0, :] @ (La @ np.ones(n - 1)))
            errs.append(abs(gd - g(tau)) / abs(gd))
        res['a_expm_rel_err'] = float(max(errs))
    # ---- visible poles
    vis = b > 1e-12
    nv, bv = nus[vis], b[vis]
    order = np.argsort(nv); nv, bv = nv[order], bv[order]
    rv = bv * nv                                  # scaled rho
    res['sum_b_plus_pia'] = float(bv.sum() + s['pia'])
    res['nu0_err'] = float(abs(nv[0] - s['nu0'])); res['nu1_err'] = float(abs(nv[1] - s['nu1']))
    res['rho0_err'] = float(abs(rv[0] - s['rho0'])); res['rho1_err'] = float(abs(rv[1] - s['rho1']))
    res['max_pole_real'] = float(lam.max())
    W, pia, C1, C2, e2 = s['W'], s['pia'], s['C1'], s['C2'], s['eps2']
    hi = mus > 1.5
    mh, ch = mus[hi], c[hi]
    Ek = lambda tau: E(mus, tau)
    u = lambda tau: 1 + float(np.sum(c * Ek(tau)))
    phi = lambda tau: -float(np.sum(c * mus * Ek(tau)))            # / mu
    r = lambda tau: float(np.sum(ch * mh * E(mh, tau)))            # / mu
    r1 = lambda tau: float(np.sum(ch * mh ** 2 * E(mh, tau)))      # / mu^2
    J = lambda tau: float(np.sum(ch * mh * E(mh, tau) / (mh - s['nu0'])))
    nu0, rho0, m0 = s['nu0'], s['rho0'], s['m0']
    neg = nv * mu > 1.0                           # poles with negative discrete eigenvalue
    Rneg = mu * float(rv[neg].sum()) if scale == 'D' else 0.0   # REAL-unit sum of rho_j over negative poles
    nq = 0.0
    if scale == 'D' and neg.any():
        nq = Rneg * (W / (1 - mu) + (2 * C1 + C2 * e2) / (2 * (1 - q - mu)))
    res['n_neg_poles'] = int(neg.sum()) if scale == 'D' else 0
    res['nq'] = nq

    def conv_r1(nu, tau):                         # (r1 * E_nu)(tau) / mu
        return float(np.sum(ch * mh ** 2 * (E(nu, tau) - E(mh, tau)) / (mh - nu)))

    def P2(tau):
        tot = pia * r1(tau)
        for nu, rh in zip(nv[2:], rv[2:]):
            tot += rh * (W / (nu - 1) * E(nu, tau) + conv_r1(nu, tau))
        return tot

    if scale == 'C':
        taus = np.concatenate([np.linspace(0.02, 3, 40), np.linspace(3, 14, 30)]) / mu
    else:
        tmax = int(14 / mu)
        taus = np.unique(np.concatenate([np.arange(0, min(40, tmax)), np.linspace(0, tmax, 90).astype(int)]))
    viol = []
    ident = 0.0
    Qprev = None
    for tau in taus:
        tautil = mu * tau if scale == 'C' else mu * tau / (1 - 2 * mu)
        Gs = G(tau) / mu ** 2                     # scaled G
        gs = g(tau) / mu
        # identity
        lead = sum(rh * W / (nu - 1) * E(nu, tau) for nu, rh in zip(nv[1:], rv[1:]))
        convs = sum(rh * conv_r1(nu, tau) for nu, rh in zip(nv[1:], rv[1:]))
        rhs = lead - rho0 * nu0 * m0 * E(nu0, tau) - rho0 * nu0 * J(tau) - rho0 * r(tau) + pia * r1(tau) + convs
        sc = abs(Gs) + rho0 * nu0
        ident = max(ident, abs(Gs - rhs) / sc)
        tol = 1e-9 * sc
        rt, r1t, qt, ph, ut = r(tau), r1(tau), r1(tau) - r(tau), phi(tau), u(tau)
        if rt < -1e-10: viol.append(('r<0', float(tau), rt))
        if rt > rbar(s, E, tau) * (1 + 1e-9) + 1e-12: viol.append(('r>rbar', float(tau), rt / rbar(s, E, tau)))
        if qt < -1e-9: viol.append(('qt<0', float(tau), qt))
        qb = C1 * E(2.0, tau) + C2 * (e2 - 1) * E(e2, tau)
        if qt > qb * (1 + 1e-9) + 1e-11: viol.append(('qt>qbar', float(tau), qt / qb))
        A, g2 = s['A'], s['gam2']
        ql = d * (d - 1) * (A * A * E(2.0, tau) - 2 * A * g2 * e2 * E(1 + e2, tau) - (d - 2) * A ** 3 * E(3.0, tau))
        if qt < ql - 1e-9 * max(1, abs(ql)): viol.append(('qt<qlow', float(tau), qt, ql))
        if d == 2:
            rl = W * (A * E(2.0, tau) - g2 * E(1 + e2, tau))
        else:
            rl = W * (2 * A * E(2.0, tau) - A * A * E(3.0, tau) - 2 * g2 * E(1 + e2, tau) - g2 * g2 * E(1 + 2 * e2, tau))
        if rt < rl - 1e-9: viol.append(('r<rlow', float(tau), rt, rl))
        Jb = C1 * E(2.0, tau) / (2 - nu0) + C2 * E(e2, tau) / (e2 - nu0)
        if J(tau) < -1e-10 or J(tau) > Jb * (1 + 1e-9) + 1e-12: viol.append(('J', float(tau), J(tau), Jb))
        # LB / UB
        lb = LB(s, E, tau, nq); ub = UB(s, E, tau, tautil, mu, nq)
        if Gs < lb - tol: viol.append(('G<LB', float(tau), Gs, lb))
        if Gs > ub + tol: viol.append(('G>UB', float(tau), Gs, ub))
        p2 = P2(tau)
        if p2 < -nq * E(2.0, tau) - tol: viol.append(('P2<-neg', float(tau), p2))
        # value bound  g <= rho0 [u + phi/(1-nu0)] (+ neg-pole correction), scaled
        vb = rho0 * (ut + ph / (1 - nu0)) + (Rneg / (2 * (1 - q)) if scale == 'D' else 0.0)
        if gs > vb * (1 + 1e-9) + 1e-13: viol.append(('value', float(tau), gs, vb))
        Q = 1 - ut - ph
        if Q < Qlow(s, E, tau) - 1e-9: viol.append(('Q<Qlow', float(tau), Q, Qlow(s, E, tau)))
        if Qprev is not None and Q > Qprev + 1e-9: viol.append(('Q increasing', float(tau)))
        Qprev = Q
        glow = rho0 * E(nu0, tau) * (1 - W * E(1.0, tau)) - (Rneg / (2 * (1 - q)) if scale == 'D' else 0.0)
        if gs < glow - 1e-12: viol.append(('g<glow', float(tau), gs, glow))
    res['identity_max_relerr'] = float(ident)
    # ---- (f) pole bounds
    Xs, sig = s['Xs'], s['sigma']
    y = 1 / Xs
    sig2 = sig - W
    th = s['theta']
    chk = {}
    chk['nu0_hi'] = nu0 <= y * (1 + 1e-12)
    chk['nu0_lo'] = nu0 >= y * (1 - sig * y * y / (1 - y)) - 1e-12
    chk['b0'] = (s['b0'] <= 1 + 1e-12) and (1 - s['b0'] <= sig * nu0 ** 2 / (1 - nu0) ** 2 + 1e-12)
    thbar = W / (Xs - W - 1)
    S1bar = 2 * sig2
    ebar = thbar + (1 + thbar) * S1bar / (1 - thbar) if thbar < 1 else np.inf
    chk['theta_hi'] = th <= thbar * (1 + 1e-12)
    chk['theta_lo'] = th >= W / (Xs - W - 1 + ebar) * (1 - 1e-12)
    S2nu = float(np.sum(w[hi] / (mh - s['nu1']) ** 2))
    chk['S2nu'] = S2nu <= 4 * sig2 / (1 - th) ** 2 * (1 + 1e-12)
    iG = 1 / s['Gam']
    chk['Gam_identity'] = abs(iG - (W * (1 + th) / th + th / (1 + th) + th * (1 + th) * S2nu)) < 1e-8 * iG
    beta2 = 1 - pia - s['b0'] - s['b1']
    delta2 = float(np.sum(rv[2:] / (nv[2:] - 1)))
    chk['delta2'] = delta2 <= 2 * beta2 + 1e-12
    chk['sumrule_mu'] = abs(pia + float(np.sum(rv / (nv - 1)))) < 1e-9
    chk['sum_rho'] = abs(float(rv.sum()) - d / (4 * n * np.sin(s['x']) ** 2)) < 1e-9
    chk['m0_lo'] = m0 >= 1 + nu0 * s['EU'] - 1e-12
    chk['m0_hi'] = m0 <= W ** nu0 / (1 - nu0) + 1e-12
    res['pole_checks_failed'] = [k for k, v in chk.items() if not v]
    res['n_violations'] = len(viol); res['violations'] = viol[:12]
    return res


if __name__ == '__main__':
    cases = [(2, 6, 'C', 1.0), (2, 10, 'C', 1.0), (2, 14, 'C', 1.0), (3, 4, 'C', 1.0), (3, 6, 'C', 1.0),
             (2, 6, 'D', 0.3), (2, 6, 'D', 0.5), (2, 10, 'D', 0.6), (2, 14, 'D', 0.8), (2, 20, 'D', 0.85),
             (3, 4, 'D', 0.5), (3, 6, 'D', 0.6), (3, 9, 'D', 0.7)]
    out = []
    for cs in cases:
        rr = run(*cs); out.append(rr)
        print(json.dumps(rr, default=str), flush=True)
        json.dump(out, open(DATA + '/21_bruteforce.json', 'w'), indent=1, default=str)
