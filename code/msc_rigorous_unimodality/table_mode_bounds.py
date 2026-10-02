"""
table_mode_bounds.py -- numerical illustration (float) of Corollary 4.13:
    m_u  <  t*  <=  m_1  <=  m(q/M)
for the continuous-time first-passage density with unit rate q = 1, where
    u(t)   = P_{x0}(X_t = a) = h(t)^d              (unkilled walk; h = 1D propagator, rate 1/d)
    m_u    = maximiser of u'                        (zero of u'')
    t*     = mode of g                              (zero of g', from the killed spectrum)
    m(a)   = zero of J_a(t) = u'(t) - a int_0^t exp(-a(t-s)) u'(s) ds  beyond m_u
    m_1    = m(alpha_1),  alpha_1 = smallest eigenvalue of the killed Laplacian A
    M      = max_x E_x[T]  (so that alpha_1 >= 1/M)
Also prints the closed-form approximation of Section 6 of the article, MFPT*ln(2dX)/(X+2d-1), X = beta_1*MFPT.

Not part of any proof.  Output: data/table_mode_bounds.json
"""
from __future__ import annotations
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository

import json
import os
import sys

import numpy as np
from scipy.optimize import brentq
import scipy.sparse as sp
import scipy.sparse.linalg as spla

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = _os.path.join(_R, 'data', 'msc_rigorous_unimodality')


def laplacian_sparse(N, d):
    n = N ** d
    idx = np.arange(n).reshape((N,) * d)
    rows, cols, vals = [], [], []
    w = 1.0 / (2 * d)
    diag = np.zeros(n)
    for ax in range(d):
        lo = [slice(None)] * d; hi = [slice(None)] * d
        lo[ax] = slice(0, N - 1); hi[ax] = slice(1, N)
        a = idx[tuple(lo)].ravel(); b = idx[tuple(hi)].ravel()
        rows += [a, b]; cols += [b, a]; vals += [-w * np.ones(a.size), -w * np.ones(a.size)]
        np.add.at(diag, a, w); np.add.at(diag, b, w)
    rows.append(np.arange(n)); cols.append(np.arange(n)); vals.append(diag)
    L = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
    return L, idx


def h_coeffs(N, d, j, k):
    """1D propagator j->k (1-based) with rate c = 1/d: h(t) = sum coef_m exp(-rate_m t)."""
    m = np.arange(N)
    cm = np.where(m == 0, 1.0 / N, 2.0 / N)
    coef = cm * np.cos(m * np.pi * (j - 0.5) / N) * np.cos(m * np.pi * (k - 0.5) / N)
    rate = (1.0 / d) * (1 - np.cos(m * np.pi / N))
    keep = np.abs(coef) > 1e-15
    return coef[keep], rate[keep]


def u_expansion(N, d, j, k):
    """u = h^d as a single exponential sum (coefficients C, rates R)."""
    c, r = h_coeffs(N, d, j, k)
    C, R = c.copy(), r.copy()
    for _ in range(d - 1):
        C = np.outer(C, c).ravel()
        R = np.add.outer(R, r).ravel()
    return C, R


def run(geo, d, N):
    cc = (N + 1) // 2
    if geo == "CC":
        s, a = (1,) * d, (N,) * d
    elif geo == "C2M":
        s, a = (1,) * d, (cc,) * d
    else:
        s, a = (cc,) * d, (N,) * d
    L, idx = laplacian_sparse(N, d)
    n = N ** d
    ia = idx[tuple(x - 1 for x in a)]
    ix = idx[tuple(x - 1 for x in s)]
    keep = np.array([i for i in range(n) if i != ia])
    A = L[keep][:, keep].toarray()
    rvec = -np.asarray(L[keep][:, ia].todense()).ravel()
    al, Phi = np.linalg.eigh(A)
    pos = int(np.where(keep == ix)[0][0])
    cg = Phi[pos, :] * (Phi.T @ rvec)
    alpha1 = float(al[0])
    Mvec = np.linalg.solve(A, np.ones(n - 1))
    Mmax = float(Mvec.max())
    mfpt = float(Mvec[pos])
    gp = lambda t: float(np.sum(-al * cg * np.exp(-al * t)))
    C, R = u_expansion(N, d, s[0], a[0])
    u1 = lambda t: float(np.sum(-R * C * np.exp(-R * t)))
    u2 = lambda t: float(np.sum(R * R * C * np.exp(-R * t)))

    def J(t, aa):
        # u'(t) - a * int_0^t exp(-a(t-s)) u'(s) ds ; term: -R C (e^{-R t}-e^{-a t})/(a-R)
        den = aa - R
        safe = np.abs(den) > 1e-12
        integ = np.sum((-R[safe] * C[safe]) * (np.exp(-R[safe] * t) - np.exp(-aa * t)) / den[safe])
        if (~safe).any():
            integ += np.sum((-R[~safe] * C[~safe]) * t * np.exp(-aa * t))
        return u1(t) - aa * integ

    scale = d * N * N
    # m_u: first zero of u'' (u'' > 0 near 0)
    grid = np.linspace(scale * 0.03, 3.0 * scale, 3000)
    v = np.array([u2(t) for t in grid])
    i0 = int(np.argmax(v < 0))
    m_u = brentq(u2, grid[i0 - 1], grid[i0], xtol=1e-10)
    # mode of g
    grid2 = np.linspace(m_u, 40.0 * scale, 6000)
    v = np.array([gp(t) for t in grid2])
    i0 = int(np.argmax(v < 0))
    tstar = brentq(gp, grid2[i0 - 1], grid2[i0], xtol=1e-10)
    nchg = int(np.sum(np.sign(v[1:]) != np.sign(v[:-1])))

    def m_of(aa):
        vv = np.array([J(t, aa) for t in grid2])
        k0 = int(np.argmax(vv < 0))
        return brentq(lambda t: J(t, aa), grid2[k0 - 1], grid2[k0], xtol=1e-10)

    m1 = m_of(alpha1)
    mM = m_of(1.0 / Mmax)
    beta1 = (1.0 / d) * (1 - np.cos(np.pi / N)) if geo == "CC" else (1.0 / d) * (1 - np.cos(2 * np.pi / N))
    X = beta1 * mfpt
    formal = mfpt * np.log(2 * d * X) / (X + 2 * d - 1) if geo == "CC" else float("nan")
    rec = dict(geo=geo, d=d, N=N, alpha1=alpha1, mfpt=mfpt, Mmax=Mmax, m_u=m_u, mode=tstar, m_1=m1, m_M=mM,
               sign_changes_gprime_on_grid=nchg, ok=bool(m_u < tstar <= m1 * (1 + 1e-9) and m1 <= mM * (1 + 1e-9)),
               formal_two_pole=formal)
    return rec


if __name__ == "__main__":
    out = []
    cases = [("CC", 2, 8), ("CC", 2, 16), ("CC", 2, 32), ("CC", 2, 64), ("CC", 3, 8), ("CC", 3, 12), ("CC", 3, 16),
             ("C2M", 2, 9), ("C2M", 2, 33), ("M2C", 2, 9), ("M2C", 2, 33), ("C2M", 3, 9), ("M2C", 3, 9), ("CC", 1, 16)]
    for geo, d, N in cases:
        rec = run(geo, d, N)
        out.append(rec)
        print("%s d=%d N=%d: m_u=%.2f  mode=%.2f  m_1=%.2f  m(1/M)=%.2f  [m_1/mode=%.4f, m(1/M)/mode=%.4f]  formal=%.2f  ok=%s"
              % (geo, d, N, rec["m_u"], rec["mode"], rec["m_1"], rec["m_M"], rec["m_1"] / rec["mode"],
                 rec["m_M"] / rec["mode"], rec["formal_two_pole"], rec["ok"]), flush=True)
        with open(_os.path.join(_R, 'data', 'msc_rigorous_unimodality', 'table_mode_bounds.json'), "w") as fh:
            json.dump(out, fh, indent=1)
