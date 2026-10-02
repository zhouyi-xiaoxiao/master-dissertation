"""Sanity check (float, brute force, small N) of the statements of Section 7.5 for 1/2 < q < 1, N >= N_1(q) (sin 2x <= (1-q)/q):
 (a) one-dimensional pairing lemma: v_{n+1}-v_n >= 0, pi_2 >= 0, pi_3 >= 0 and the bounds of Lemma 7.2             [Lemma 7.10]
 (b) phi >= 0, phi(t+1) >= (1-mu)phi(t), W gfrak <= r <= rbar, r <= r1 <= r1bar, qtilde >= qlow                    [Lemma 7.3 for q<1]
 (c) negative poles: |P_neg| <= n_q                                                                                 [Lemma 7.11(a)]
 (d) Phi_-^D - n_q <= Dg <= Phi_+^D + n_q   (E_-^lo replaced by 0 is NOT needed for the bound itself)               [Lemma 7.12(A)]
 (e) region C: Dg >= phi (B_1 - omega_q) - omega_q'                                                                 [Lemma 7.12(B)]
 (f) region D: g <= rho0 [1 - Q + nu0 phi/(1-nu0)] + neg_v,  Q >= Q_lo,  g >= rho0 E_{nu0} (1 - W E_1) - neg_v      [Lemma 7.12(C)]
"""
import json, sys, math
import numpy as np
from r4lib import spec, scalars, mu_real, DATA


def L1(d, N):
    n = N ** d
    idx = np.arange(n).reshape((N,) * d)
    L = np.zeros((n, n)); rate = 1.0 / (2 * d)
    for ax in range(d):
        a = np.moveaxis(idx, ax, 0)
        src = a[:-1].ravel(); dst = a[1:].ravel()
        L[src, src] += rate; L[dst, dst] += rate; L[src, dst] -= rate; L[dst, src] -= rate
    return L


def one_dim(N, q, nmax):
    """coefficients pi_1, pi_2, pi_3 of prod_{k>=j} p_k z/(1-beta_k z), by power-series multiplication in 40-digit arithmetic
    (the float recursion loses accuracy when many beta_k are negative)."""
    from mpmath import mp, mpf
    mp.dps = 40
    x = mp.pi / (2 * N)
    p = [2 * mpf(q) * mp.sin(k * x) ** 2 for k in range(1, N)]
    beta = [1 - pp for pp in p]
    out = {}
    for j in (1, 2, 3):
        ser = [mpf(0)] * (nmax + 1); ser[0] = mpf(1)
        for kk in range(j - 1, N - 1):
            new = [mpf(0)] * (nmax + 1)
            for n in range(1, nmax + 1):
                new[n] = beta[kk] * new[n - 1] + p[kk] * ser[n - 1]
            ser = new
        out[j] = np.array([float(z) for z in ser])
        out[('min', j)] = float(min(ser))
    return out, np.array([float(z) for z in p]), np.array([float(z) for z in beta])


def run(d, N, q):
    s = scalars(d, N)
    mus, w, c = spec(d, N)
    n = N ** d
    mu = mu_real(d, N, q)
    x = s['x']
    res = dict(d=d, N=N, q=q, mu=mu, N1_condition=bool(math.sin(2 * x) <= (1 - q) / q), Xs=s['Xs'])
    viol = []
    # (a) one-dimensional
    nmax = min(1500, int(12 / (d * mu)))
    pis, p, beta = one_dim(N, q, nmax)
    A, g2, e2, W, C1, C2, pia = s['A'], s['gam2'], s['eps2'], s['W'], s['C1'], s['C2'], s['pia']
    for j in (1, 2, 3):
        if pis[('min', j)] < 0: viol.append(('pi_%d<0' % j, pis[('min', j)]))
    psi = pis[1][1:] / p[0]                        # psi(n) = pi_1(n+1)/p_1
    nn = np.arange(len(psi))
    if np.any(psi > A * beta[0] ** nn * (1 + 1e-9) + 1e-10): viol.append(('psi>A beta1^n',))
    if np.any(psi < (A * beta[0] ** nn - g2 * e2 * beta[1] ** nn) * (1 + 1e-9 * np.sign(-(A * beta[0] ** nn - g2 * e2 * beta[1] ** nn))) - 1e-13): viol.append(('psi lower',))
    m = np.arange(1, nmax + 1)
    if np.any(pis[2][1:] > p[1] * g2 * (e2 - 1) * beta[1] ** (m - 1) * (1 + 1e-9) + 1e-10): viol.append(('pi_2 upper',))
    res['n_negative_beta'] = int(np.sum(beta < 0))
    # d-dimensional
    E = lambda k, t: (1.0 - mu * k) ** t
    La = L1(d, N)[:n - 1, :n - 1] * q / mu
    lam, V = np.linalg.eigh(La)
    cx = V[0, :]; c1 = V.T @ np.ones(n - 1)
    a = lam * cx * c1
    b = c1 ** 2 / n
    g = lambda t: float(np.sum(a * E(lam, t)))
    G = lambda t: float(-np.sum(a * lam * E(lam, t)))
    hi = mus > 1.5
    mh, ch = mus[hi], c[hi]
    phi = lambda t: -float(np.sum(c * mus * E(mus, t)))
    u = lambda t: 1 + float(np.sum(c * E(mus, t)))
    r = lambda t: float(np.sum(ch * mh * E(mh, t)))
    r1 = lambda t: float(np.sum(ch * mh ** 2 * E(mh, t)))
    vis = b > 1e-12
    nv, bv = lam[vis], b[vis]
    o = np.argsort(nv); nv, bv = nv[o], bv[o]; rv = bv * nv
    nu0, nu1, rho0, rho1, th, m0 = s['nu0'], s['nu1'], s['rho0'], s['rho1'], s['theta'], s['m0']
    a1, a0 = s['a1'], s['a0']
    negp = mu * nv >= 1.0
    res['n_neg_poles'] = int(negp.sum()); res['R_neg_mu'] = float(mu * rv[negp].sum()); res['pia_q_over_2'] = float(pia * q / 2)
    rq = 2 * q - 1
    EJ = lambda t: float(np.sum(ch * mh * E(mh, t) / (mh - nu0)))

    def conv_r1(nu, t):
        return float(np.sum(ch * mh ** 2 * (E(nu, t) - E(mh, t)) / (mh - nu)))

    def gfrak(t):
        if d == 2:
            return A * E(2, t) - g2 * E(1 + e2, t)
        return 2 * A * E(2, t) - 2 * g2 * E(1 + e2, t) - A * A * E(3, t) + 2 * A * g2 * E(2 + e2, t) - g2 * g2 * E(1 + 2 * e2, t)
    rbar = lambda t: C1 * E(2, t) + C2 * E(e2, t)
    r1bar = lambda t: 2 * C1 * E(2, t) + C2 * e2 * E(e2, t)
    qlow = lambda t: d * (d - 1) * (A * A * E(2, t) - 2 * A * g2 * e2 * E(1 + e2, t) - (d - 2) * A ** 3 * E(3, t))
    Qlow = lambda t: d * (d - 1) * (A * A * E(2, t) / 2 - 2 * A * g2 * e2 * E(1 + e2, t) / (1 + e2) - (d - 2) * A ** 3 * E(3, t) / 3)
    nq = lambda t: (pia * q / 2) * ((W / (1 - mu) + 2 * C1 / (2 * (1 - q) - 2 * mu)) * E(2, t) + C2 * e2 * E(e2, t) / (2 * (1 - q) - e2 * mu))
    S = max(1, math.ceil(math.log((d - 1) * A) / mu)); sx = mu * S
    kap_hi = W / (1 - mu * nu1) + nu1 * (W * (math.exp(th * sx / (1 - nu1 * mu)) - 1) / th + C1 * math.exp(-(1 - th) * sx) / (1 - th)
                                          + C2 * math.exp(-(e2 - 1 - th) * sx) / (e2 - 1 - th))

    def geo(aa, t):
        rho = (1 - aa * mu) / (1 - nu1 * mu)
        return (rho - rho ** t) / (aa - nu1)

    def kap_lo(t):
        if d == 2:
            integ = A * geo(2, t) - g2 * geo(1 + e2, t)
        else:
            integ = 2 * A * geo(2, t) - 2 * g2 * geo(1 + e2, t) - A * A * geo(3, t) + 2 * A * g2 * geo(2 + e2, t) - g2 * g2 * geo(1 + 2 * e2, t)
        return max(0.0, W - rbar(t) / E(nu1, t) + nu1 * W * integ)
    Em_hi = lambda t: (2 * C1 * E(2, t) / (2 - nu0) + C2 * e2 * E(e2, t) / (e2 - nu0)) / (nu0 * m0 * E(nu0, t))

    def Prest_hi(t):
        tau = mu * t
        cb = 2 * W + 2 * C1 * e2 / (e2 - 2)
        near = 0.0
        for nu, rh in zip(s['near_nu'], s['near_rho']):
            near += rh * (2 * C1 * min(tau / (1 - 2 * mu), 1 / (nu - 2)) + C2 * e2 * min(tau / (1 - nu * mu), 1 / (e2 - nu)))
        return E(2, t) * (cb * s['betahat'] + near) + E(e2, t) * C2 * e2 * (pia + tau * s['Rtot'] / (1 - e2 * mu))
    omq = (pia * q / 2) / (2 * (1 - q) - mu)
    omq1 = lambda t: (pia * q / 2) * (C1 * E(2, t) / (2 * (1 - q) - 2 * mu) + C2 * (e2 - 1) * E(e2, t) / (2 * (1 - q) - e2 * mu))
    negv = pia * q / (4 * (1 - q))
    B1 = lambda t: (a1 * nu1 * E(nu1, t) / W - rho0 * nu0 * E(nu0, t) / (1 - nu0)) / E(1, t)
    tmax = int(14 / mu)
    ts = np.unique(np.concatenate([np.arange(0, min(80, tmax)), np.linspace(0, tmax, 200).astype(int)]))
    Qprev = None
    for t in ts:
        t = int(t)
        Gt = G(t); gt = g(t)
        sc = abs(Gt) + rho0 * nu0 * E(nu0, t)
        tol = 1e-8 * sc
        rt, r1t, ph, ut = r(t), r1(t), phi(t), u(t)
        qt = (phi(t + 1) - (1 - mu) * ph) / mu
        if ph < -1e-10: viol.append(('phi<0', t))
        if qt < -1e-8: viol.append(('qtilde<0', t, qt))
        if qt < qlow(t) - 1e-8: viol.append(('qtilde<qlow', t, qt, qlow(t)))
        if rt < -1e-9 or rt > rbar(t) * (1 + 1e-9) + 1e-11: viol.append(('r', t, rt, rbar(t)))
        if rt < W * gfrak(t) - 1e-9: viol.append(('r<W gfrak', t, rt, W * gfrak(t)))
        if r1t < rt - 1e-8 or r1t > r1bar(t) * (1 + 1e-9) + 1e-10: viol.append(('r1', t, r1t, r1bar(t)))
        Q = 1 - ut - ph
        if Q < Qlow(t) - 1e-9: viol.append(('Q<Qlow', t, Q, Qlow(t)))
        if Qprev is not None and Q > Qprev + 1e-9: viol.append(('Q increasing', t))
        Qprev = Q
        if t >= 1:
            # (c) negative poles
            Pneg = sum(rh * (W / (nu - 1) * E(nu, t) + conv_r1(nu, t)) for nu, rh in zip(nv[negp], rv[negp]))
            if abs(Pneg) > nq(t) * (1 + 1e-9) + tol: viol.append(('Pneg', t, Pneg, nq(t)))
            # (d) window
            Phim = a1 * E(nu1, t) * (1 + th * kap_lo(t) / W) - a0 * E(nu0, t) * (1 + Em_hi(t)) - nq(t)
            Phip = a1 * E(nu1, t) * (1 + th * kap_hi / W) + Prest_hi(t) - a0 * E(nu0, t) + nq(t)
            if Gt < Phim - tol: viol.append(('G<Phi-', t, Gt, Phim))
            if Gt > Phip + tol: viol.append(('G>Phi+', t, Gt, Phip))
            # (e) region C
            lowC = ph * (B1(t) - omq) - omq1(t)
            if Gt < lowC - tol: viol.append(('regionC', t, Gt, lowC))
        # (f) value bounds
        up = rho0 * (1 - Q + nu0 * ph / (1 - nu0)) + negv
        if gt > up * (1 + 1e-9) + 1e-13: viol.append(('value up', t, gt, up))
        lo = rho0 * E(nu0, t) * (1 - W * E(1, t)) - negv
        if gt < lo - 1e-12: viol.append(('value lo', t, gt, lo))
    res['n_violations'] = len(viol); res['violations'] = [str(v) for v in viol[:8]]
    return res


if __name__ == '__main__':
    cases = [(2, 13, 0.8), (2, 20, 0.8), (2, 30, 0.8), (2, 30, 0.9), (2, 16, 0.6), (3, 13, 0.8), (3, 8, 0.7), (2, 10, 0.5), (2, 36, 0.9)]
    out = []
    for cs in cases:
        rr = run(*cs); out.append(rr)
        print(json.dumps(rr, default=str), flush=True)
        json.dump(out, open(DATA + '/75_check_q_large.json', 'w'), indent=1, default=str)
