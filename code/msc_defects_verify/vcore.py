"""vcore.py -- second implementation, written separately for the check, of the exact blocked-site first-passage machinery.

Written from the model definition only (lazy nearest-neighbour walk, move probability q, an attempted
move that leaves the lattice or lands on a blocked site is cancelled, absorbing target).  It does not
import or call any code of the main implementation.

Mode certificate used here (different from the main implementation's spectral envelope):
  Q symmetric, u_a = Q^a e_s, v_b = Q^b r (r = one-step absorption vector).  Then
      f(a+b+1) = u_a . v_b                                   (exact)
      f(t) <= ||u_a|| ||v_b||   for every t >= a+b+1          (Cauchy-Schwarz and ||Q||_2 <= 1)
  so stepping u and v together gives f(2a+1) = u_a.v_a, f(2a+2) = u_{a+1}.v_a for all a, and the
  search stops at the first a with ||u_a|| ||v_a|| < max_{t<=2a+2} f(t).  No eigen-decomposition.
"""
import os
for k in ('VECLIB_MAXIMUM_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(k, '1')
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from collections import deque


def cluster_index(open_mask, start):
    """BFS from start over open sites (nearest-neighbour).  Returns idx array (-1 = not in cluster), n."""
    shape = open_mask.shape
    d = open_mask.ndim
    idx = -np.ones(shape, dtype=np.int64)
    if not open_mask[start]:
        raise ValueError('start blocked')
    dq = deque([tuple(start)])
    idx[tuple(start)] = 0
    n = 1
    while dq:
        x = dq.popleft()
        for ax in range(d):
            for s in (-1, 1):
                y = list(x); y[ax] += s
                if 0 <= y[ax] < shape[ax]:
                    y = tuple(y)
                    if open_mask[y] and idx[y] < 0:
                        idx[y] = n; n += 1
                        dq.append(y)
    return idx, n


def bonds(idx):
    """List of open-open nearest-neighbour bonds (i, j) inside the cluster (each once)."""
    d = idx.ndim
    I = []; J = []
    for ax in range(d):
        a = np.moveaxis(idx, ax, 0)
        lo = a[:-1].ravel(); hi = a[1:].ravel()
        ok = (lo >= 0) & (hi >= 0)
        I.append(lo[ok]); J.append(hi[ok])
    return np.concatenate(I), np.concatenate(J)


def laplacian(idx, n, w=None):
    """Unit-conductance (or weighted) graph Laplacian of the cluster."""
    I, J = bonds(idx)
    if w is None:
        w = np.ones(len(I))
    A = sp.coo_matrix((np.concatenate([w, w]), (np.concatenate([I, J]), np.concatenate([J, I]))),
                      shape=(n, n)).tocsr()
    deg = np.asarray(A.sum(axis=1)).ravel()
    return (sp.diags(deg) - A).tocsr()


def transition(idx, n, q, w=None):
    """P = I - (q/2d) L : symmetric, cancelled moves become self-loops."""
    d = idx.ndim
    L = laplacian(idx, n, w)
    return (sp.identity(n, format='csr') - (q / (2 * d)) * L).tocsr()


def reduce_target(P, a):
    n = P.shape[0]
    keep = np.r_[0:a, a + 1:n]
    Q = P[keep][:, keep].tocsr()
    r = np.asarray(P[keep][:, [a]].todense()).ravel()
    return Q, r, keep


def moments(Q, s_red):
    """Exact mean and variance of the first-passage time from reduced index s_red."""
    n = Q.shape[0]
    lu = spla.splu((sp.identity(n, format='csc') - Q.tocsc()))
    m = lu.solve(np.ones(n))
    m2 = lu.solve(1.0 + 2.0 * (Q @ m))
    return m[s_red], m2[s_red] - m[s_red] ** 2, m


def certified_mode(Q, r, s_red, chk=32, amax=20_000_000, want_pmf=False):
    """Certified global mode of f(t) = e_s^T Q^{t-1} r, t >= 1.
    Returns dict(mode, fmode, t_cert, peak_sub (parabolic sub-step position), pmf (optional))."""
    n = Q.shape[0]
    u = np.zeros(n); u[s_red] = 1.0
    v = r.copy()
    fl = []                                  # f(1), f(2), ...
    fmax = -1.0; tmax = 0
    a = 0
    while True:
        f_odd = float(u @ v)                 # f(2a+1)
        u_next = Q @ u
        f_even = float(u_next @ v)           # f(2a+2)
        fl.append(f_odd); fl.append(f_even)
        if f_odd > fmax:
            fmax = f_odd; tmax = 2 * a + 1
        if f_even > fmax:
            fmax = f_even; tmax = 2 * a + 2
        if a % chk == 0 and a > 0:
            B = np.sqrt(u @ u) * np.sqrt(v @ v)   # bounds f(t) for all t >= 2a+1
            if B < fmax:
                break
        u = u_next
        v = Q @ v
        a += 1
        if a > amax:
            raise RuntimeError('mode not certified')
    f = np.array(fl)
    out = dict(mode=int(tmax), fmode=float(fmax), t_cert=2 * a + 1)
    m = tmax
    if 2 <= m < len(f):
        y0, y1, y2 = f[m - 2], f[m - 1], f[m]
        den = (y0 - 2 * y1 + y2)
        out['peak_sub'] = m + (0.5 * (y0 - y2) / den if den != 0 else 0.0)
    else:
        out['peak_sub'] = float(m)
    if want_pmf:
        out['pmf'] = f
    return out


def solve_config(open_mask, q, start=None, target=None, w_fun=None, want_pmf=False, extras=False):
    """Everything for one configuration.  open_mask: True = open site."""
    N = open_mask.shape[0]; d = open_mask.ndim
    start = tuple(start) if start is not None else (0,) * d
    target = tuple(target) if target is not None else (N - 1,) * d
    idx, n = cluster_index(open_mask, start)
    a = idx[target]
    if a < 0:
        return None
    P = transition(idx, n, q)
    Q, r, keep = reduce_target(P, a)
    s_red = 0 if a > 0 else None   # start has cluster index 0; target index a > 0
    mean, var, mvec = moments(Q, s_red)
    md = certified_mode(Q, r, s_red, want_pmf=want_pmf)
    rec = dict(n=n, mfpt=float(mean), sd=float(np.sqrt(var)), cv=float(np.sqrt(var) / mean),
               mode=md['mode'], fmode=md['fmode'], t_cert=md['t_cert'], peak_sub=md['peak_sub'],
               ratio=md['mode'] / float(mean), n_open=int(open_mask.sum()))
    if want_pmf:
        rec['pmf'] = md['pmf']
    if extras:
        rec.update(_extras(idx, n, a, q, d))
    return rec


def _extras(idx, n, a, q, d):
    """Resistance identity pieces: G = pseudo-inverse of the cluster Laplacian (via grounded solve)."""
    L = laplacian(idx, n)
    # solve L g = e_a - 1/n  with sum(g) = 0  -> g = G[:, a]
    keep = np.r_[1:n]                       # ground node 0 (the start)
    lu = spla.splu(L[keep][:, keep].tocsc())
    b = -np.ones(n) / n; b[a] += 1.0
    g = np.zeros(n); g[keep] = lu.solve(b[keep]); g -= g.mean()
    return dict(G_aa=float(g[a]), G_sa=float(g[0]), mfpt_res=float((2 * d * n / q) * (g[a] - g[0])))


def connected(open_mask, start, target):
    idx, n = cluster_index(open_mask, start)
    return idx[tuple(target)] >= 0


def ci95(x):
    x = np.asarray(x, float)
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0
