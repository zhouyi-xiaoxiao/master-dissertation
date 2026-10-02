"""Common helpers for the note on the rigorous spectral structure (d >= 2).

Conventions (unit rate; the lazy walk with activity q has generator q*L in continuous
time and transition matrix P = I - q*L in discrete time):
  L = I - P_1, P_1 = reflecting ("cancelled move") nearest-neighbour walk on {1..N}^d.
  L is real symmetric, L 1 = 0, eigenpairs
     lambda_k = (1/d) sum_i (1 - cos(pi k_i / N)),  k in {0..N-1}^d
     phi_k(x) = prod_i c_{k_i} cos(pi k_i (x_i - 1/2)/N), c_0 = N^{-1/2}, c_k = (2/N)^{1/2}.
"""
import itertools
import numpy as np


def build_L(N, d):
    """Dense generator L = I - P_1 on {1..N}^d (index = mixed radix, 0-based coords)."""
    n = N ** d
    L = np.zeros((n, n))
    idx = lambda x: int(np.ravel_multi_index(x, (N,) * d))
    for x in itertools.product(range(N), repeat=d):
        i = idx(x)
        for ax in range(d):
            for s in (-1, 1):
                y = list(x)
                y[ax] += s
                if 0 <= y[ax] < N:
                    j = idx(tuple(y))
                    L[i, i] += 1.0 / (2 * d)
                    L[i, j] -= 1.0 / (2 * d)
    return L


def site_index(x, N, d):
    """x given with 1-based coordinates."""
    return int(np.ravel_multi_index(tuple(c - 1 for c in x), (N,) * d))


def box_modes(N, d):
    """All k, lambda_k and the 1D eigenvector table V[k, x] (x = 0..N-1 <-> site x+1)."""
    k = np.arange(N)
    x = np.arange(N)
    V = np.cos(np.pi * np.outer(k, x + 0.5) / N) * np.sqrt(2.0 / N)
    V[0, :] = np.sqrt(1.0 / N)
    lam1 = (1 - np.cos(np.pi * k / N))
    return k, lam1, V


def corner_weights_1d(N):
    """n-normalised 1D weights at the corner site N: eps_k cos^2(pi k / 2N), and signs (-1)^k."""
    k = np.arange(N)
    eps = np.where(k == 0, 1.0, 2.0)
    w = eps * np.cos(np.pi * k / (2 * N)) ** 2
    sgn = (-1.0) ** k
    lam1 = 1 - np.cos(np.pi * k / N)
    return w, sgn, lam1


def corner_spectral_data(N, d):
    """Flattened arrays over k in {0..N-1}^d: lambda_k, n*phi_k(a)^2, (-1)^{|k|}  (a = corner (N..N), x0 = (1..1))."""
    w1, s1, l1 = corner_weights_1d(N)
    lam = np.zeros((N,) * d)
    w = np.ones((N,) * d)
    sg = np.ones((N,) * d)
    for ax in range(d):
        shape = [1] * d
        shape[ax] = N
        lam = lam + l1.reshape(shape) / d
        w = w * w1.reshape(shape)
        sg = sg * s1.reshape(shape)
    return lam.ravel(), w.ravel(), sg.ravel()


def group_visible(lam, w, sg=None, tol=1e-11):
    """Group eigenvalues into distinct values; return Lambda_i, W_i = sum of n*phi^2, and (optional) signed weights."""
    order = np.argsort(lam, kind="stable")
    lam, w = lam[order], w[order]
    sgw = (w * sg[order]) if sg is not None else None
    Lam, W, SW = [], [], []
    i = 0
    while i < len(lam):
        j = i
        while j + 1 < len(lam) and abs(lam[j + 1] - lam[i]) <= tol * max(1.0, abs(lam[i])):
            j += 1
        Lam.append(lam[i:j + 1].mean())
        W.append(w[i:j + 1].sum())
        if sgw is not None:
            SW.append(sgw[i:j + 1].sum())
        i = j + 1
    Lam, W = np.array(Lam), np.array(W)
    if sg is not None:
        return Lam, W, np.array(SW)
    return Lam, W
