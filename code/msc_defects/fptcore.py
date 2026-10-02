"""
fptcore.py -- exact first-passage statistics for the lazy nearest-neighbour walk
on a finite hypercubic lattice {0..N-1}^d with blocked ("inert") sites.

Model (inert defects, Section 8 of the article):
  * at each step the walker picks one of the 2d lattice directions with
    probability q/(2d) each and rests with probability 1-q;
  * an attempted move off the lattice OR onto a blocked site is cancelled
    (walker stays put);
  * the target site is absorbing; the start site is s.
The walker can only ever visit the open cluster (4-/2d-connectivity) that
contains s, so the state space is that cluster minus the target.  P restricted
to the cluster is symmetric (doubly stochastic), hence reversible with uniform
stationary law.  Q = P restricted to transient states (symmetric, substochastic),
r = one-step absorption probabilities into the target.

Exact quantities:
  MFPT      : (I-Q) m1 = 1                        (sparse LU)
  2nd mom.  : (I-Q) m2 = 1 + 2 Q m1               (E[T^2])
  PMF       : f(t) = e_s^T Q^{t-1} r = sum_k w_k lam_k^{t-1},  w_k = phi_k(s) (phi_k . r)
  survival  : S(t) = e_s^T Q^t 1  = sum_k c_k lam_k^t,          c_k = phi_k(s) (phi_k . 1)
  mode      : argmax_{t>=1} f(t), certified globally by the decreasing envelope
              E(t) = sum_k |w_k| |lam_k|^{t-1}  >= |f(t')| for all t' >= t.
For large state spaces a time-stepping PMF with a Lanczos-based certificate is
provided (pmf_stats_timestep).
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy import ndimage

# ----------------------------------------------------------------------------
# lattice / chain construction
# ----------------------------------------------------------------------------

def cross_structure(d):
    return ndimage.generate_binary_structure(d, 1)


def cluster_of(open_mask, site):
    """Boolean mask of the open cluster (nearest-neighbour connectivity) containing `site`."""
    lab, _ = ndimage.label(open_mask, structure=cross_structure(open_mask.ndim))
    l0 = lab[site]
    if l0 == 0:
        raise ValueError("site is blocked")
    return lab == l0


def connected(open_mask, s, t):
    if not open_mask[s] or not open_mask[t]:
        return False
    lab, _ = ndimage.label(open_mask, structure=cross_structure(open_mask.ndim))
    return lab[s] == lab[t]


def build_chain(open_mask, q, start, target, bond_weight=None):
    """Build the transient chain.

    open_mask : bool ndarray of shape (N,)*d ; True = accessible site.
    bond_weight : optional callable(site_flat_a, site_flat_b) -> multiplier in [0,1] (vectorised
                  over arrays) for partially permeable bonds (default 1).
    Returns dict with Q (csr, transient x transient), r (absorption vector), s (index of start),
    n_cluster (sites in start cluster, incl. target), sites (flat indices of transient states),
    cluster (bool mask), P_off (off-diagonal hop probability array per bond, for diagnostics).
    """
    open_mask = np.asarray(open_mask, dtype=bool)
    d = open_mask.ndim
    shape = open_mask.shape
    start = tuple(start); target = tuple(target)
    clus = cluster_of(open_mask, start)
    if not clus[target]:
        raise ValueError("target not in start cluster")
    flat = np.flatnonzero(clus.ravel())
    n = flat.size
    idx_of = -np.ones(clus.size, dtype=np.int64)
    idx_of[flat] = np.arange(n)
    coords = np.array(np.unravel_index(flat, shape))  # d x n
    rows, cols, vals = [], [], []
    stay = np.ones(n)
    hop = q / (2 * d)
    for ax in range(d):
        for sgn in (+1, -1):
            nb = coords.copy()
            nb[ax] += sgn
            inside = (nb[ax] >= 0) & (nb[ax] < shape[ax])
            src = np.flatnonzero(inside)
            nbf = np.ravel_multi_index(tuple(nb[:, src]), shape)
            ok = clus.ravel()[nbf]
            src = src[ok]
            dst = idx_of[nbf[ok]]
            w = np.full(src.size, hop)
            if bond_weight is not None:
                w = w * bond_weight(flat[src], flat[dst])
            rows.append(src); cols.append(dst); vals.append(w)
            np.subtract.at(stay, src, w)
    rows = np.concatenate(rows); cols = np.concatenate(cols); vals = np.concatenate(vals)
    rows = np.concatenate([rows, np.arange(n)]); cols = np.concatenate([cols, np.arange(n)])
    vals = np.concatenate([vals, stay])
    P = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    it = idx_of[np.ravel_multi_index(target, shape)]
    is_ = idx_of[np.ravel_multi_index(start, shape)]
    keep = np.ones(n, dtype=bool); keep[it] = False
    trans = np.flatnonzero(keep)
    Q = P[trans][:, trans].tocsr()
    r = np.asarray(P[trans][:, [it]].todense()).ravel()
    s = int(np.searchsorted(trans, is_))
    assert trans[s] == is_
    return dict(Q=Q, r=r, s=s, n_cluster=int(n), sites=flat[trans], cluster=clus,
                P=P, t_index=int(it), s_index_full=int(is_))


# ----------------------------------------------------------------------------
# moments by linear solves
# ----------------------------------------------------------------------------

def moments(Q, s=None):
    n = Q.shape[0]
    A = (sp.identity(n, format='csc') - Q.tocsc())
    lu = spla.splu(A)
    one = np.ones(n)
    m1 = lu.solve(one)
    m2 = lu.solve(one + 2.0 * (Q @ m1))
    if s is None:
        return m1, m2
    return float(m1[s]), float(m2[s] - m1[s] ** 2), m1


# ----------------------------------------------------------------------------
# spectral PMF (dense), certified mode, quantiles
# ----------------------------------------------------------------------------

class SpectralFPT:
    def __init__(self, Q, r, s):
        Qd = Q.toarray()
        # Q is symmetric for this model; enforce exact symmetry to kill roundoff
        asym = np.max(np.abs(Qd - Qd.T))
        if asym > 1e-14:
            raise ValueError(f"Q not symmetric (max asym {asym})")
        lam, phi = np.linalg.eigh(0.5 * (Qd + Qd.T))
        self.lam = lam
        a = phi[s, :]
        self.w = a * (phi.T @ r)          # f(t) = sum w lam^(t-1)
        self.c = a * phi.sum(axis=0)      # S(t) = sum c lam^t
        self.r_s = float(r[s])
        self.abslam = np.abs(lam)
        self.absw = np.abs(self.w)
        self.lam1 = float(lam[-1])
        self.lam2 = float(lam[-2]) if lam.size > 1 else float('nan')

    def _pow(self, e):
        e = np.asarray(e, dtype=np.int64)
        return np.power(self.lam[None, :], e[:, None])

    def f(self, t):
        t = np.atleast_1d(np.asarray(t, dtype=np.int64))
        out = self._pow(t - 1) @ self.w
        out[t == 1] = self.r_s
        out[t < 1] = 0.0
        return out

    def S(self, t):
        t = np.atleast_1d(np.asarray(t, dtype=np.int64))
        out = self._pow(t) @ self.c
        out[t == 0] = 1.0
        return out

    def envelope(self, t):
        """sum |w_k| |lam_k|^(t-1): upper bound on |f(t')| for all t' >= t."""
        return float(np.sum(self.absw * self.abslam ** (t - 1)))

    def mode(self, chunk=4096, tmax=10**8):
        t0 = 1
        best_t, best_f = 1, -np.inf
        while t0 < tmax:
            ts = np.arange(t0, t0 + chunk)
            fv = self.f(ts)
            k = int(np.argmax(fv))
            if fv[k] > best_f:
                best_f, best_t = float(fv[k]), int(ts[k])
            t0 += chunk
            if self.envelope(t0) < best_f:
                return best_t, best_f, t0
        raise RuntimeError("mode certificate not reached")

    def quantile(self, alpha, t_hint):
        """Smallest integer t with S(t) <= 1-alpha (i.e. P(T<=t) >= alpha)."""
        target = 1.0 - alpha
        lo, hi = 0, max(2, int(t_hint))
        while self.S(hi)[0] > target:
            lo, hi = hi, hi * 2
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if self.S(mid)[0] > target:
                lo = mid
            else:
                hi = mid
        return hi

    def mfpt(self):
        return float(np.sum(self.c / (1.0 - self.lam)))


def fpt_summary(open_mask, q, start, target, want_series=False, series_tmax=None):
    """All exact statistics for one configuration (dense spectral route)."""
    ch = build_chain(open_mask, q, start, target)
    Q, r, s = ch['Q'], ch['r'], ch['s']
    mfpt, var, m1 = moments(Q, s)
    sp_ = SpectralFPT(Q, r, s)
    mode, fmode, tcert = sp_.mode()
    med = sp_.quantile(0.5, mfpt)
    q10 = sp_.quantile(0.1, mfpt)
    q90 = sp_.quantile(0.9, mfpt)
    out = dict(n_cluster=ch['n_cluster'], mfpt=mfpt, sd=float(np.sqrt(var)), var=var,
               mode=int(mode), f_mode=fmode, median=int(med), q10=int(q10), q90=int(q90),
               S_mode=float(sp_.S(mode)[0]), S_mfpt=float(sp_.S(int(round(mfpt)))[0]),
               lam1=sp_.lam1, tau1=float(-1.0 / np.log(sp_.lam1)),
               A1=float(sp_.c[-1]), mfpt_spectral=sp_.mfpt(), t_cert=int(tcert),
               lam2=sp_.lam2, w1=float(sp_.w[-1]))
    out['ratio'] = out['mode'] / out['mfpt']
    out['cv'] = out['sd'] / out['mfpt']
    if want_series:
        tm = series_tmax or int(3 * mfpt)
        ts = np.arange(1, tm + 1)
        out['series_t'] = ts
        out['series_f'] = sp_.f(ts)
    return out, ch


# ----------------------------------------------------------------------------
# time stepping (independent route, used for validation and large N)
# ----------------------------------------------------------------------------

def pmf_timestep(Q, r, s, tmax):
    n = Q.shape[0]
    x = np.zeros(n); x[s] = 1.0
    f = np.empty(tmax); S = np.empty(tmax)
    QT = Q.T.tocsr()
    for t in range(tmax):
        f[t] = x @ r
        x = QT @ x
        S[t] = x.sum()
    return f, S  # f[t-1] = f(t), S[t-1] = S(t)


def pmf_stats_timestep(Q, r, s, mfpt, block=2000):
    """Certified mode by time stepping for large n.  Certificate: with Q symmetric,
    f(t) = alpha1 lam1^(t-1) + eps(t), |eps(t)| <= lam*^(t-1) * sqrt(1-a1^2) * sqrt(|r|^2-b1^2),
    where lam* = max(|lam2|, |lam_min|).  Stop once alpha1 lam1^(t-1) + bound < f_max."""
    n = Q.shape[0]
    vals, vecs = spla.eigsh(Q, k=2, which='LA', tol=1e-12)
    order = np.argsort(vals)[::-1]
    lam1, lam2 = vals[order[0]], vals[order[1]]
    phi1 = vecs[:, order[0]]
    if phi1.sum() < 0:
        phi1 = -phi1
    lmin = spla.eigsh(Q, k=1, which='SA', tol=1e-10)[0][0]
    lstar = max(abs(lam2), abs(lmin)) * (1 + 1e-9)
    a1 = phi1[s]; b1 = phi1 @ r
    alpha1 = a1 * b1
    C = np.sqrt(max(0.0, 1 - a1 ** 2)) * np.sqrt(max(0.0, r @ r - b1 ** 2))
    x = np.zeros(n); x[s] = 1.0
    QT = Q.T.tocsr()
    fs = []
    best_t, best_f = 0, -np.inf
    t = 0
    while True:
        for _ in range(block):
            t += 1
            ft = x @ r
            fs.append(ft)
            if ft > best_f:
                best_f, best_t = ft, t
            x = QT @ x
        bound = alpha1 * lam1 ** t + C * lstar ** t  # for all t' >= t+1
        if bound < best_f:
            break
    f = np.array(fs)
    S = 1.0 - np.cumsum(f)
    return dict(mode=int(best_t), f_mode=float(best_f), lam1=float(lam1), lam2=float(lam2),
                lmin=float(lmin), t_cert=int(t), f=f, S=S, alpha1=float(alpha1),
                A1=float(a1 * phi1.sum()))


# ----------------------------------------------------------------------------
# effective resistance / Tetali decomposition
# ----------------------------------------------------------------------------

def green_pinv(ch):
    """Pseudo-inverse G of the unit-conductance graph Laplacian on the start cluster
    (including the target).  Returns G and the full-cluster indices of s and t."""
    P = ch['P']
    n = P.shape[0]
    A = P.copy().tolil()
    A.setdiag(0)
    A = A.tocsr()
    A.data[:] = 1.0
    A.eliminate_zeros()
    deg = np.asarray(A.sum(axis=1)).ravel()
    L = np.diag(deg) - A.toarray()
    J = np.full((n, n), 1.0 / n)
    G = np.linalg.inv(L + J) - J
    return G


def tetali_parts(ch, q, d=2):
    """MFPT = (2 n d / q) (G_tt - G_st)   [exact for this reversible chain].
    Returns dict with n, G_tt, G_ss, G_st, R_st (unit), and the implied MFPT."""
    G = green_pinv(ch)
    s = ch['s_index_full']; t = ch['t_index']; n = ch['n_cluster']
    Gtt, Gss, Gst = G[t, t], G[s, s], G[s, t]
    mfpt = 2 * n * d / q * (Gtt - Gst)
    return dict(n=n, G_tt=float(Gtt), G_ss=float(Gss), G_st=float(Gst),
                R_st=float(Gss + Gtt - 2 * Gst), mfpt_tetali=float(mfpt),
                half_commute=float(n * d / q * (Gss + Gtt - 2 * Gst)),
                asym=float(n * d / q * (Gtt - Gss)))


def chemical_distance(cluster, s, t):
    """BFS graph distance inside the cluster."""
    from collections import deque
    shape = cluster.shape
    d = cluster.ndim
    dist = -np.ones(shape, dtype=np.int64)
    dist[s] = 0
    dq = deque([s])
    while dq:
        u = dq.popleft()
        if u == t:
            return int(dist[u])
        for ax in range(d):
            for sg in (1, -1):
                v = list(u); v[ax] += sg; v = tuple(v)
                if 0 <= v[ax] < shape[ax] and cluster[v] and dist[v] < 0:
                    dist[v] = dist[u] + 1
                    dq.append(v)
    return -1


def reflecting_relaxation(ch):
    """Second-largest eigenvalue mu2 of the full reflecting chain P on the start cluster (no
    absorption) and the relaxation time tau_rel = -1/ln(mu2)."""
    Pd = ch['P'].toarray()
    mu = np.linalg.eigvalsh(0.5 * (Pd + Pd.T))
    mu2 = float(mu[-2])
    return mu2, float(-1.0 / np.log(mu2))


def tetali_parts_sparse(ch, q, d=2):
    """Same as tetali_parts but with two sparse solves (any system size)."""
    P = ch['P'].tocsr()
    n = P.shape[0]
    A = P - sp.diags(P.diagonal())
    A.eliminate_zeros()
    A.data[:] = 1.0
    deg = np.asarray(A.sum(axis=1)).ravel()
    L = (sp.diags(deg) - A).tocsc()
    s = ch['s_index_full']; t = ch['t_index']
    pin = 0 if (s != 0 and t != 0) else [i for i in range(3) if i not in (s, t)][0]
    keep = np.ones(n, bool); keep[pin] = False
    fi = np.flatnonzero(keep)
    lu = spla.splu(L[fi][:, fi])
    def gcol(j):
        b = np.full(n, -1.0 / n); b[j] += 1.0
        x = np.zeros(n); x[fi] = lu.solve(b[fi])
        return x - x.mean()
    gt = gcol(t); gs = gcol(s)
    Gtt, Gss, Gst = gt[t], gs[s], gt[s]
    return dict(n=n, G_tt=float(Gtt), G_ss=float(Gss), G_st=float(Gst),
                R_st=float(Gss + Gtt - 2 * Gst), mfpt_tetali=float(2 * n * d / q * (Gtt - Gst)))


def timestep_summary(open_mask, q, start, target, block=2000, want_median=True):
    """Exact statistics by time stepping (large systems): MFPT/variance by sparse LU, certified
    mode with a shift-invert Lanczos certificate, median by continuing the iteration."""
    ch = build_chain(open_mask, q, start, target)
    Q, r, s = ch['Q'], ch['r'], ch['s']
    n = Q.shape[0]
    mfpt, var, m1 = moments(Q, s)
    A = (sp.identity(n, format='csc') - Q.tocsc())
    lu = spla.splu(A)
    op = spla.LinearOperator((n, n), matvec=lu.solve, dtype=float)   # (I-Q)^{-1}
    vals, vecs = spla.eigsh(op, k=2, which='LA', tol=1e-13)
    order = np.argsort(vals)[::-1]
    lam1 = 1.0 - 1.0 / vals[order[0]]; lam2 = 1.0 - 1.0 / vals[order[1]]
    phi1 = vecs[:, order[0]]
    if phi1.sum() < 0:
        phi1 = -phi1
    lmin = spla.eigsh(Q, k=1, which='SA', tol=1e-8)[0][0]
    lstar = max(abs(lam2), abs(lmin) * (1 + 1e-6))
    a1 = phi1[s]; b1 = phi1 @ r
    alpha1 = a1 * b1
    C = np.sqrt(max(0.0, 1 - a1 ** 2)) * np.sqrt(max(0.0, r @ r - b1 ** 2))
    x = np.zeros(n); x[s] = 1.0
    QT = Q.T.tocsr()
    best_t, best_f = 0, -np.inf
    t = 0; S = 1.0; median = None; certified = False; t_cert = None
    while True:
        for _ in range(block):
            t += 1
            ft = x @ r
            if ft > best_f:
                best_f, best_t = ft, t
            x = QT @ x
            S -= ft
            if median is None and S <= 0.5:
                median = t
        if not certified:
            bound = alpha1 * lam1 ** t + C * lstar ** t
            if bound < best_f:
                certified = True; t_cert = t
        if certified and (median is not None or not want_median):
            break
    out = dict(n_cluster=ch['n_cluster'], mfpt=mfpt, sd=float(np.sqrt(var)), mode=int(best_t),
               f_mode=float(best_f), median=median, lam1=float(lam1), lam2=float(lam2),
               tau1=float(-1 / np.log(lam1)), t_cert=int(t_cert), A1=float(a1 * phi1.sum()),
               w1=float(alpha1))
    out['ratio'] = out['mode'] / mfpt; out['cv'] = out['sd'] / mfpt
    return out, ch
