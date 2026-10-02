"""Common helpers for the one-dimensional computations.

Conventions: sites 1..N, absorbing target N, reflecting wall behind site 1
(a cancelled move = stay), start x0, d = N - x0, L = N - 1/2.
"""
from fractions import Fraction
import numpy as np


def Q_matrix(N, q, exact=False):
    """Substochastic transition matrix on transient sites 1..N-1."""
    n = N - 1
    if exact:
        q = Fraction(q)
        Q = [[Fraction(0)] * n for _ in range(n)]
        for j in range(n):
            Q[j][j] = 1 - q
            if j > 0:
                Q[j][j - 1] = q / 2
            if j < n - 1:
                Q[j][j + 1] = q / 2
        Q[0][0] = 1 - q / 2
        return Q
    Q = np.zeros((n, n))
    for j in range(n):
        Q[j, j] = 1 - q
        if j > 0:
            Q[j, j - 1] = q / 2
        if j < n - 1:
            Q[j, j + 1] = q / 2
    Q[0, 0] = 1 - q / 2
    return Q


def pmf_stepper(N, q, x0, tmax):
    """f(t), t = 1..tmax, by forward stepping (float64, all terms >= 0)."""
    n = N - 1
    v = np.zeros(n)
    v[x0 - 1] = 1.0
    f = np.zeros(tmax + 1)
    a, b = 1 - q, q / 2
    for t in range(1, tmax + 1):
        f[t] = b * v[n - 1]
        w = a * v
        w[1:] += b * v[:-1]
        w[:-1] += b * v[1:]
        w[0] += b * v[0]
        v = w
    return f


def pmf_exact(N, q, x0, tmax):
    """Exact rational f(t), t=1..tmax (list indexed by t; f[0]=0)."""
    q = Fraction(q)
    n = N - 1
    v = [Fraction(0)] * n
    v[x0 - 1] = Fraction(1)
    a, b = 1 - q, q / 2
    f = [Fraction(0)] * (tmax + 1)
    for t in range(1, tmax + 1):
        f[t] = b * v[n - 1]
        w = [a * x for x in v]
        for j in range(1, n):
            w[j] += b * v[j - 1]
        for j in range(0, n - 1):
            w[j] += b * v[j + 1]
        w[0] += b * v[0]
        v = w
    return f


def spectral(N, q, x0):
    """theta_m, lambda_m, coefficient C_m with f(t) = sum C_m lambda_m^(t-1)."""
    m = np.arange(1, N)
    th = (2 * m - 1) * np.pi / (2 * N - 1)
    lam = 1 - q * (1 - np.cos(th))
    d = N - x0
    C = (2 * q / (2 * N - 1)) * np.sin(th) * np.sin(d * th)
    return th, lam, C
