"""Sanity check (float, brute force, small N) of the DISCRETE-time statements of Section 7 (q <= 1/2):
 (a) P(T = t+1) = mu * sum_j a_j (1-mu nu_j)^t  against direct time stepping              [Prop. 1.5]
 (b) discrete identity for G = D g                                                         [Prop. 7.1]
 (c) 0 <= W g_frak <= r <= rbar,  r <= r1 <= r1bar,  phi >= 0,  phi(t+1) >= (1-mu) phi(t)   [Lemma 7.3]
 (d) [exact check of log-concavity/unimodality: see 71_check_discrete_unimodal.py]          [Theorem 7.4]
 (e) kappa_lo <= kappa <= kappa_hi, E_- bounds, P_rest bound, Phi_-^D <= G <= Phi_+^D       [Lemma 7.6, Theorem 7.7]
Scaled quantities: E_k(t) = (1 - mu k)^t,  (Df)(t) = (f(t+1)-f(t))/mu.
"""
import json, sys, math
import numpy as np
from r4lib import spec, scalars, mu_real, DATA


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


def run(d, N, q):
    s = scalars(d, N)
    mus, w, c = spec(d, N)
    n = N ** d
    mu = mu_real(d, N, q)
    E = lambda k, t: (1.0 - mu * k) ** t
    La = L1(d, N)[:n - 1, :n - 1] * q / mu            # scaled killed generator
    lam, V = np.linalg.eigh(La)
    cx = V[0, :]; c1 = V.T @ np.ones(n - 1)
    a = lam * cx * c1
    b = c1 ** 2 / n
    g = lambda t: float(np.sum(a * E(lam, t)))
    G = lambda t: float(-np.sum(a * lam * E(lam, t)))
    res = dict(d=d, N=N, q=q, mu=mu, Xs=s['Xs'], max_mu_lam=float(mu * lam.max()))
    # (a) time stepping
    P = np.eye(n - 1) - mu * La
    p = np.zeros(n - 1); p[0] = 1.0
    sprev = 1.0; f = []
    tmax_step = int(min(6000, 8.0 / mu))
    for t in range(1, tmax_step):
        p = p @ P
        sv = p.sum(); f.append(sprev - sv); sprev = sv
    f = np.array(f)
    res['a_err'] = float(max(abs(f[t] - mu * g(t)) for t in range(0, len(f), max(1, len(f) // 300))))
    res['mode_stepper'] = int(np.argmax(f)) + 1
    W, pia, e2, g2, A, C1, C2 = s['W'], s['pia'], s['eps2'], s['gam2'], s['A'], s['C1'], s['C2']
    hi = mus > 1.5
    mh, ch = mus[hi], c[hi]
    phi = lambda t: -float(np.sum(c * mus * E(mus, t)))
    r = lambda t: float(np.sum(ch * mh * E(mh, t)))
    r1 = lambda t: float(np.sum(ch * mh ** 2 * E(mh, t)))
    vis = b > 1e-12
    nv, bv = lam[vis], b[vis]
    o = np.argsort(nv); nv, bv = nv[o], bv[o]; rv = bv * nv
    nu0, nu1, rho0, rho1, th, m0 = s['nu0'], s['nu1'], s['rho0'], s['rho1'], s['theta'], s['m0']
    a1, a0 = s['a1'], s['a0']
    EJ = lambda t: float(np.sum(ch * mh * E(mh, t) / (mh - nu0)))           # E_{nu0}(t) J(t)

    def conv_r1(nu, t):
        return float(np.sum(ch * mh ** 2 * (E(nu, t) - E(mh, t)) / (mh - nu)))

    def gfrak(t):
        if d == 2:
            return A * E(2, t) - g2 * E(1 + e2, t)
        return 2 * A * E(2, t) - 2 * g2 * E(1 + e2, t) - A * A * E(3, t) + 2 * A * g2 * E(2 + e2, t) - g2 * g2 * E(1 + 2 * e2, t)

    rbar = lambda t: C1 * E(2, t) + C2 * E(e2, t)
    r1bar = lambda t: 2 * C1 * E(2, t) + C2 * e2 * E(e2, t)
    # kappa bounds
    S = max(1, math.ceil(math.log((d - 1) * A) / mu)); sx = mu * S
    kap_hi = W / (1 - mu * nu1) + nu1 * (W * (math.exp(th * sx / (1 - nu1 * mu)) - 1) / th + C1 * math.exp(-(1 - th) * sx) / (1 - th)
                                          + C2 * math.exp(-(e2 - 1 - th) * sx) / (e2 - 1 - th))

    def geo(aa, t):          # mu sum_{s=1}^{t-1} E_a(s)/E_{nu1}(s+1) = (rho - rho^t)/(a - nu1)
        rho = (1 - aa * mu) / (1 - nu1 * mu)
        return (rho - rho ** t) / (aa - nu1)

    def kap_lo(t):
        if d == 2:
            integ = A * geo(2, t) - g2 * geo(1 + e2, t)
        else:
            integ = 2 * A * geo(2, t) - 2 * g2 * geo(1 + e2, t) - A * A * geo(3, t) + 2 * A * g2 * geo(2 + e2, t) - g2 * g2 * geo(1 + 2 * e2, t)
        return max(0.0, W - rbar(t) / E(nu1, t) + nu1 * W * integ)

    def Em_hi(t):
        return (2 * C1 * E(2, t) / (2 - nu0) + C2 * e2 * E(e2, t) / (e2 - nu0)) / (nu0 * m0 * E(nu0, t))

    def Em_lo(t):
        return max(0.0, W * gfrak(t) / (nu0 * m0 * E(nu0, t)))

    def Prest_hi(t):
        tau = mu * t
        cb = 2 * W + 2 * C1 * e2 / (e2 - 2)
        near = 0.0
        for nu, rh in zip(s['near_nu'], s['near_rho']):
            near += rh * (2 * C1 * min(tau / (1 - 2 * mu), 1 / (nu - 2)) + C2 * e2 * min(tau / (1 - nu * mu), 1 / (e2 - nu)))
        return E(2, t) * (cb * s['betahat'] + near) + E(e2, t) * C2 * e2 * (pia + tau * s['Rtot'] / (1 - e2 * mu))

    viol = []
    ident = 0.0
    tmax = int(14 / mu)
    ts = np.unique(np.concatenate([np.arange(0, min(60, tmax)), np.linspace(0, tmax, 160).astype(int)]))
    for t in ts:
        t = int(t)
        Gt = G(t)
        sc = abs(Gt) + rho0 * nu0 * E(nu0, t)
        tol = 1e-8 * sc
        rt, r1t, ph = r(t), r1(t), phi(t)
        if ph < -1e-10: viol.append(('phi<0', t))
        if phi(t + 1) < (1 - mu) * ph - 1e-10: viol.append(('D2', t))
        if rt < -1e-9 or rt > rbar(t) * (1 + 1e-9) + 1e-11: viol.append(('r', t, rt, rbar(t)))
        if rt < W * gfrak(t) - 1e-9: viol.append(('r<W gfrak', t, rt, W * gfrak(t)))
        if r1t < rt - 1e-8 or r1t > r1bar(t) * (1 + 1e-9) + 1e-10: viol.append(('r1', t, r1t, r1bar(t)))
        lead = sum(rh * W / (nu - 1) * E(nu, t) for nu, rh in zip(nv[1:], rv[1:]))
        convs = sum(rh * conv_r1(nu, t) for nu, rh in zip(nv[1:], rv[1:]))
        rhs = lead - rho0 * nu0 * m0 * E(nu0, t) - rho0 * nu0 * EJ(t) - rho0 * rt + pia * r1t + convs
        ident = max(ident, abs(Gt - rhs) / sc)
        if t >= 1:
            kap_t = conv_r1(nu1, t) / E(nu1, t)
            if kap_t < kap_lo(t) - 1e-7 or kap_t > kap_hi + 1e-7: viol.append(('kappa', t, kap_t, kap_lo(t), kap_hi))
            Em = (rho0 * nu0 * EJ(t) + rho0 * rt) / (a0 * E(nu0, t))
            if Em < Em_lo(t) - 1e-9 or Em > Em_hi(t) * (1 + 1e-9) + 1e-12: viol.append(('E-', t, Em, Em_lo(t), Em_hi(t)))
            Prest = sum(rh * W / (nu - 1) * E(nu, t) for nu, rh in zip(nv[2:], rv[2:])) + pia * r1t \
                + sum(rh * conv_r1(nu, t) for nu, rh in zip(nv[2:], rv[2:]))
            if Prest < -tol or Prest > Prest_hi(t) * (1 + 1e-9) + tol: viol.append(('Prest', t, Prest, Prest_hi(t)))
            Phim = a1 * E(nu1, t) * (1 + th * kap_lo(t) / W) - a0 * E(nu0, t) * (1 + Em_hi(t))
            Phip = a1 * E(nu1, t) * (1 + th * kap_hi / W) + Prest_hi(t) - a0 * E(nu0, t) * (1 + Em_lo(t))
            if Gt < Phim - tol: viol.append(('G<Phi-', t, Gt, Phim))
            if Gt > Phip + tol: viol.append(('G>Phi+', t, Gt, Phip))
    res['identity_max_relerr'] = float(ident)
    # (d) log-concavity of phi and unimodality: float evaluation of the eigen-sums is too noisy at early times;
    #     see 71_check_discrete_unimodal.py for the exact (integer arithmetic) check.
    res['n_violations'] = len(viol); res['violations'] = [tuple(map(float, v[1:])) if False else str(v) for v in viol[:8]]
    # exact discrete mode vs closed form
    tstar = res['mode_stepper']
    res['X(mu t*-tau_cf)'] = float(s['X'] * (mu * tstar - s['tau_cf']))
    return res


if __name__ == '__main__':
    cases = [(2, 6, 0.5), (2, 10, 0.5), (2, 10, 0.2), (2, 16, 0.5), (2, 24, 0.5), (2, 24, 0.3), (3, 4, 0.5), (3, 6, 0.5), (3, 6, 0.25), (3, 9, 0.5)]
    out = []
    for cs in cases:
        rr = run(*cs); out.append(rr)
        print(json.dumps(rr, default=str), flush=True)
        json.dump(out, open(DATA + '/70_check_discrete.json', 'w'), indent=1, default=str)
