"""Common exact quantities for the reflecting lattice walk, corner-to-corner.
All rates are for activity q (default q=1): mu_k = (q/d) sum_i (1-cos(pi k_i/N)).
"""
import numpy as np
from scipy.optimize import brentq

def spectrum(d, N, q=1.0):
    """Return arrays (mu_k, w_k, par_k) over k in {0..N-1}^d minus 0 (flattened)."""
    k = np.arange(N)
    x = np.pi * k / (2 * N)
    eps = (2 * q / d) * np.sin(x) ** 2
    al = np.where(k == 0, 1.0, 2.0) * np.cos(x) ** 2
    par = k % 2
    if d == 1:
        mu, w, p = eps, al, par
    elif d == 2:
        mu = (eps[:, None] + eps[None, :]).ravel()
        w = (al[:, None] * al[None, :]).ravel()
        p = ((par[:, None] + par[None, :]) % 2).ravel()
    elif d == 3:
        mu = (eps[:, None, None] + eps[None, :, None] + eps[None, None, :]).ravel()
        w = (al[:, None, None] * al[None, :, None] * al[None, None, :]).ravel()
        p = ((par[:, None, None] + par[None, :, None] + par[None, None, :]) % 2).ravel()
    return mu[1:], w[1:], p[1:], eps

def basic(d, N, q=1.0):
    mu_k, w_k, par, eps = spectrum(d, N, q)
    mu = eps[1]
    c2 = np.cos(np.pi / (2 * N)) ** 2
    M = np.sum(w_k / mu_k)                 # stationary-start MFPT  E_pi T
    S2 = np.sum(w_k / mu_k ** 2)
    Tcc = 2 * np.sum(w_k[par == 1] / mu_k[par == 1])   # corner-to-corner MFPT
    return dict(d=d, N=N, q=q, mu=mu, c2=c2, W=2 * d * c2, A=2 * d * c2 * mu, M=M, S2=S2,
                Xs=mu * M, X=mu * Tcc, Tcc=Tcc, sigma=mu ** 2 * S2, pia=1.0 / N ** d,
                mu_k=mu_k, w_k=w_k, par=par, eps=eps)

def poles01(B):
    """nu_0, nu_1, rho_0, rho_1 from 1 + s R(s) = 0 (float, brentq)."""
    mu_k, w_k, mu = B['mu_k'], B['w_k'], B['mu']
    def F(nu):   # 1 - nu R(-nu)
        return 1.0 - nu * np.sum(w_k / (mu_k - nu))
    def rho(nu):
        return 1.0 / (1.0 / nu + nu * np.sum(w_k / (mu_k - nu) ** 2))
    nu0 = brentq(F, mu * 1e-12, mu * (1 - 1e-12), xtol=1e-300, rtol=1e-15)
    lo, hi = mu * (1 + 1e-13), 2 * mu * (1 - 1e-13)
    nu1 = brentq(F, lo, hi, xtol=1e-300, rtol=1e-15)
    return nu0, nu1, rho(nu0), rho(nu1)
