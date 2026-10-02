"""Sanity check (float, brute force, small N) of the continuous-time statements of Parts 1-2 of note R3:
 (U)  B = g'/phi is nonincreasing; g' has exactly one zero                      [Theorem 1.9]
 (S)  structure: v' >= 0, alternating bounds on v', 1-v; 0 <= r <= rbar, r >= rlow, r <= r1 <= r1bar  [Lemmas 1.6, 1.7]
 (I)  exact identity for g'                                                       [Proposition 2.1]
 (W)  Phi_minus <= g' <= Phi_plus on a grid; kappa bounds; E_- bounds; P_rest bound   [Theorem 2.3]
Dense eigendecomposition of the killed generator gives g, g' exactly (float).
"""
import json, sys
import numpy as np
from r4lib import (spec, scalars, consts, rbar, r1bar, rlow, kappa_lo, kappa_hi, Eminus_hi, Eminus_lo, Prest_hi,
                   Phi_minus, Phi_plus, Lr, DATA)


def L1(d, N):
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


def run(d, N):
    s = scalars(d, N)
    mus, w, c = spec(d, N)
    n = N ** d
    mu = (2.0 / d) * np.sin(s['x']) ** 2
    La = L1(d, N)[:n - 1, :n - 1] / mu              # scaled killed generator; target = last site, start = site 0
    lam, V = np.linalg.eigh(La)
    cx = V[0, :]; c1 = V.T @ np.ones(n - 1)
    a = lam * cx * c1                               # g(t) = sum a_j e^{-lam_j t}
    b = c1 ** 2 / n
    g1 = lambda t: float(-np.sum(a * lam * np.exp(-lam * t)))
    g1noise = lambda t: 1e-12 * float(np.sum(np.abs(a * lam) * np.exp(-lam * t)))      # float cancellation level
    W, pia, e2 = s['W'], s['pia'], s['eps2']
    hi = mus > 1.5
    mh, ch = mus[hi], c[hi]
    phi = lambda t: -float(np.sum(c * mus * np.exp(-mus * t)))
    r = lambda t: float(np.sum(ch * mh * np.exp(-mh * t)))
    r1 = lambda t: float(np.sum(ch * mh ** 2 * np.exp(-mh * t)))
    # 1D quantities
    k = np.arange(1, N)
    eps = np.sin(k * s['x']) ** 2 / np.sin(s['x']) ** 2
    gam = 2 * np.cos(k * s['x']) ** 2
    sg = (-1.0) ** (k + 1)
    v = lambda t: 1 - float(np.sum(sg * gam * np.exp(-eps * t)))
    vp = lambda t: float(np.sum(sg * gam * eps * np.exp(-eps * t)))
    A, g2 = s['A'], s['gam2']
    res = dict(d=d, N=N, Xs=s['Xs'])
    viol = []
    # ---- visible poles
    vis = b > 1e-12
    nv, bv = lam[vis], b[vis]
    o = np.argsort(nv); nv, bv = nv[o], bv[o]; rv = bv * nv
    nu0, nu1, rho0, rho1, th, m0 = s['nu0'], s['nu1'], s['rho0'], s['rho1'], s['theta'], s['m0']
    res['pole_err'] = float(max(abs(nv[0] - nu0), abs(nv[1] - nu1), abs(rv[0] - rho0), abs(rv[1] - rho1)))
    res['betahat_err'] = float(abs(s['betahat'] - (pia + bv[2:].sum())))
    res['Rtot_err'] = float(abs(s['Rtot'] - rv.sum()))
    J = lambda t: float(np.sum(ch * mh * np.exp(-(mh - nu0) * t) / (mh - nu0)))          # int_t^inf r e^{nu0 s} ds

    def conv_r1(nu, t):
        return float(np.sum(ch * mh ** 2 * (np.exp(-nu * t) - np.exp(-mh * t)) / (mh - nu)))
    ts = np.concatenate([np.linspace(0.03, 3, 60), np.linspace(3.05, 14, 60)])
    Bprev = None; nzero = 0; sprev = None
    ident = 0.0
    kap1_exact = s['kap1']
    for t in ts:
        G = g1(t); ph = phi(t)
        sc = abs(G) + rho0 * nu0 * np.exp(-nu0 * t)
        tol = 1e-8 * sc
        # (U)
        if abs(G) > 50 * g1noise(t) and ph > 1e-250:
            B = G / ph
            if Bprev is not None and B > Bprev + 1e-7 * (abs(Bprev) + 1e-12):
                viol.append(('B increasing', float(t), B, Bprev))
            Bprev = B
        if abs(G) > 50 * g1noise(t):
            sg_ = np.sign(G)
            if sprev is not None and sg_ != sprev:
                nzero += 1
            sprev = sg_
        # (S)
        vt, vpt = v(t), vp(t)
        if vpt < -1e-12 or vpt > A * np.exp(-t) * (1 + 1e-10) + 1e-13: viol.append(('vp', float(t)))
        if vpt < A * np.exp(-t) - g2 * e2 * np.exp(-e2 * t) - 1e-10: viol.append(('vp low', float(t)))
        if 1 - vt > A * np.exp(-t) + 1e-10 or 1 - vt < A * np.exp(-t) - g2 * np.exp(-e2 * t) - 1e-10: viol.append(('1-v', float(t)))
        rt, r1t = r(t), r1(t)
        if rt < -1e-9 or rt > rbar(s, t) * (1 + 1e-9) + 1e-11: viol.append(('r', float(t), rt, rbar(s, t)))
        if rt < rlow(s, t) - 1e-9: viol.append(('rlow', float(t), rt, rlow(s, t)))
        if r1t < rt - 1e-8 or r1t > r1bar(s, t) * (1 + 1e-9) + 1e-10: viol.append(('r1', float(t), r1t, r1bar(s, t)))
        # (I) identity
        lead = sum(rh * W / (nu - 1) * np.exp(-nu * t) for nu, rh in zip(nv[1:], rv[1:]))
        convs = sum(rh * conv_r1(nu, t) for nu, rh in zip(nv[1:], rv[1:]))
        rhs = lead - rho0 * nu0 * m0 * np.exp(-nu0 * t) - rho0 * nu0 * np.exp(-nu0 * t) * J(t) - rho0 * rt + pia * r1t + convs
        ident = max(ident, abs(G - rhs) / sc)
        # (W)
        if G < Phi_minus(s, t) - tol: viol.append(('G<Phi-', float(t), G, Phi_minus(s, t)))
        if G > Phi_plus(s, t) + tol: viol.append(('G>Phi+', float(t), G, Phi_plus(s, t)))
        kap_t = conv_r1(nu1, t) * np.exp(nu1 * t)       # int_0^t r1 e^{nu1 s}
        if kap_t < kappa_lo(s, t) - 1e-7 or kap_t > kappa_hi(s) + 1e-7: viol.append(('kappa', float(t), kap_t, kappa_lo(s, t), kappa_hi(s)))
        Em = (rho0 * nu0 * np.exp(-nu0 * t) * J(t) + rho0 * rt) / (s['a0'] * np.exp(-nu0 * t))
        if Em < Eminus_lo(s, t) - 1e-9 or Em > Eminus_hi(s, t) * (1 + 1e-9) + 1e-12: viol.append(('E-', float(t), Em))
        Prest = sum(rh * W / (nu - 1) * np.exp(-nu * t) for nu, rh in zip(nv[2:], rv[2:])) + pia * r1t \
            + sum(rh * conv_r1(nu, t) for nu, rh in zip(nv[2:], rv[2:]))
        if Prest < -tol or Prest > Prest_hi(s, t) * (1 + 1e-9) + tol: viol.append(('Prest', float(t), Prest, Prest_hi(s, t)))
    res['identity_max_relerr'] = float(ident)
    res['n_sign_changes_gprime'] = nzero
    res['kap1'] = float(kap1_exact); res['kap_hi'] = float(kappa_hi(s)); res['kap_lo_inf'] = float(kappa_lo(s, 40.0))
    res['Ir'] = float(s['Ir']); res['Lr_inf'] = float(Lr(s, 40.0))
    res['n_violations'] = len(viol); res['violations'] = viol[:10]
    return res


if __name__ == '__main__':
    cases = [(2, 6), (2, 10), (2, 16), (2, 24), (2, 36), (3, 4), (3, 6), (3, 9), (3, 12)]
    out = []
    for cs in cases:
        rr = run(*cs); out.append(rr)
        print(json.dumps(rr, default=str), flush=True)
        json.dump(out, open(DATA + '/40_check_thmA.json', 'w'), indent=1, default=str)
