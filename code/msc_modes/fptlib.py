"""
fptlib.py -- exact first-passage machinery for the lazy nearest-neighbour walk
on the box {0,...,N-1}^d with cancellation-reflecting boundaries and one (or a
set of) absorbing target site(s).

Model (Section 2 of the article):
  * each step: with prob. q attempt a move to one of the 2d neighbours
    (uniformly, prob q/(2d) each); a move leaving the box is cancelled
    (walker stays); with prob 1-q rest.
  * target site(s) absorbing.  Q = substochastic matrix on transient sites.
  * S(t) = e_s^T Q^t 1,  f(t) = S(t-1) - S(t).

Three independent routes are implemented:
  (A) time stepping of the transient occupation vector (exact up to
      floating-point round-off; f(t) is computed as the one-step flux into
      the target, NOT as a difference of survival values);
  (B) closed-form spectral solution for d = 1 (end-to-end chain);
  (C) Laplace-domain renewal formula F^(s) = P^(a,s|x0)/P^(a,s|a) for the
      continuous-time walk (jump rate q), with the reflecting-box propagator
      evaluated through the exact 1D lattice resolvent (closed form), which
      reduces the d-fold spectral sum to a (d-1)-fold sum, followed by
      fixed-Talbot numerical inversion.

Also: exact MFPT by sparse linear solve and by the s->0 limit of (C).

Everything is deterministic (no random numbers are used anywhere).
"""
from __future__ import annotations

import math
import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------

def corner(N, d):
    return tuple([0] * d)


def far_corner(N, d):
    return tuple([N - 1] * d)


def centre(N, d):
    if N % 2 == 0:
        raise ValueError("centre requires odd N")
    return tuple([(N - 1) // 2] * d)


GEOMETRIES = {
    # name: (start(N,d), target(N,d))
    "CC": (corner, far_corner),          # corner -> opposite corner
    "C2M": (corner, centre),             # corner start -> centre (middle) target
    "M2C": (centre, far_corner),         # centre start -> corner target
}


# ---------------------------------------------------------------------------
# (A) time stepping
# ---------------------------------------------------------------------------

class Stepper:
    """Exact time stepping of rho(t) = Q^T-propagated occupation (Q is
    symmetric here, so rho(t)_i = P(X_t=i, T>t)).

    targets: list of absorbing sites (tuples).  Supports a point target or
    a set (e.g. the whole boundary layer for the Dirichlet-exit problem).
    """

    def __init__(self, N, d, q, start, targets, dtype=np.float64):
        self.N, self.d, self.q = N, d, q
        shape = (N,) * d
        self.shape = shape
        w = q / (2 * d)
        self.w = w
        # number of blocked (cancelled) directions at each site
        nblk = np.zeros(shape, dtype=np.int8)
        for ax in range(d):
            idx0 = [slice(None)] * d
            idx0[ax] = 0
            nblk[tuple(idx0)] += 1
            idx1 = [slice(None)] * d
            idx1[ax] = N - 1
            nblk[tuple(idx1)] += 1
        self.c_stay = ((1.0 - q) + w * nblk).astype(dtype)
        self.tmask = np.zeros(shape, dtype=bool)
        for tg in targets:
            self.tmask[tuple(tg)] = True
        self.single_target = len(targets) == 1
        self.target0 = tuple(targets[0])
        self.rho = np.zeros(shape, dtype=dtype)
        if self.tmask[tuple(start)]:
            raise ValueError("start is a target")
        self.rho[tuple(start)] = 1.0
        self.new = np.zeros(shape, dtype=dtype)
        self.tmp = np.zeros(shape, dtype=dtype)
        self.t = 0
        # slices for shifts
        self._sl = []
        for ax in range(d):
            lo = [slice(None)] * d
            hi = [slice(None)] * d
            lo[ax] = slice(0, N - 1)
            hi[ax] = slice(1, N)
            self._sl.append((tuple(lo), tuple(hi)))

    def step(self):
        """Advance one step; return f(t) = probability absorbed at this step."""
        rho, new, tmp = self.rho, self.new, self.tmp
        np.multiply(self.c_stay, rho, out=new)
        np.multiply(rho, self.w, out=tmp)
        for lo, hi in self._sl:
            new[hi] += tmp[lo]   # move +1 along axis
            new[lo] += tmp[hi]   # move -1 along axis
        if self.single_target:
            f = float(new[self.target0])
            new[self.target0] = 0.0
        else:
            f = float(new[self.tmask].sum())
            new[self.tmask] = 0.0
        self.rho, self.new = new, rho
        self.t += 1
        return f

    def survival(self):
        return float(self.rho.sum())


def run_pmf(N, d, q, start, targets, t_max=None, stop_rule=None,
            max_steps=10**8, log_every=0, time_limit=None):
    """Iterate and return f[1..T] (index 0 unused, f[0]=0).

    stop_rule(t, f_arr_view, fmax, tmax_arg) -> bool, evaluated every 256 steps.
    """
    st = Stepper(N, d, q, start, targets)
    cap = 1 << 16
    f = np.zeros(cap)
    fmax, targmax = -1.0, 0
    t0 = time.time()
    t = 0
    while True:
        t += 1
        if t >= cap:
            cap *= 2
            f2 = np.zeros(cap)
            f2[: len(f)] = f
            f = f2
        v = st.step()
        f[t] = v
        if v > fmax:
            fmax, targmax = v, t
        if t_max is not None and t >= t_max:
            break
        if (t & 255) == 0:
            if stop_rule is not None and stop_rule(t, f, fmax, targmax):
                break
            if time_limit is not None and time.time() - t0 > time_limit:
                raise TimeoutError(f"time limit at t={t}")
            if log_every and (t % log_every) < 256:
                print(f"  N={N} d={d} t={t} argmax={targmax} S={st.survival():.6f} "
                      f"{time.time()-t0:.1f}s", flush=True)
        if t >= max_steps:
            break
    return f[: t + 1].copy(), st.survival(), time.time() - t0


# ---------------------------------------------------------------------------
# sparse Q and MFPT by linear solve
# ---------------------------------------------------------------------------

def build_Q(N, d, q, targets):
    """Sparse symmetric Q on transient sites; returns (Q, index_map) where
    index_map maps flat site index -> transient index (-1 for targets)."""
    shape = (N,) * d
    n = N ** d
    flat = np.arange(n).reshape(shape)
    tmask = np.zeros(shape, dtype=bool)
    for tg in targets:
        tmask[tuple(tg)] = True
    tidx = -np.ones(n, dtype=np.int64)
    trans = ~tmask.ravel()
    tidx[trans] = np.arange(trans.sum())
    w = q / (2 * d)
    rows, cols, vals = [], [], []
    diag = np.full(shape, 1.0 - q)
    for ax in range(d):
        for sgn in (+1, -1):
            src = [slice(None)] * d
            dst = [slice(None)] * d
            if sgn > 0:
                src[ax] = slice(0, N - 1); dst[ax] = slice(1, N)
                edge = [slice(None)] * d; edge[ax] = N - 1
            else:
                src[ax] = slice(1, N); dst[ax] = slice(0, N - 1)
                edge = [slice(None)] * d; edge[ax] = 0
            diag[tuple(edge)] += w
            s = flat[tuple(src)].ravel()
            t_ = flat[tuple(dst)].ravel()
            rows.append(s); cols.append(t_)
    rows = np.concatenate(rows); cols = np.concatenate(cols)
    keep = trans[rows] & trans[cols]
    r = tidx[rows[keep]]; c = tidx[cols[keep]]
    dflat = diag.ravel()[trans]
    m = trans.sum()
    Q = sp.coo_matrix((np.full(r.size, w), (r, c)), shape=(m, m))
    Q = (Q + sp.diags(dflat)).tocsr()
    return Q, tidx


def mfpt_linear_solve(N, d, q, start, targets):
    Q, tidx = build_Q(N, d, q, targets)
    m = Q.shape[0]
    A = (sp.identity(m, format="csc") - Q.tocsc())
    b = np.ones(m)
    x = spla.spsolve(A, b)
    flat = np.ravel_multi_index(tuple(start), (N,) * d)
    return float(x[tidx[flat]])


# ---------------------------------------------------------------------------
# (B) closed-form spectral solution, d = 1, start 0, target N-1
# ---------------------------------------------------------------------------

class Chain1D:
    """Q on sites 0..N-2 (target N-1), reflecting (cancel) at 0.
    Eigenvectors v_j = cos(theta (j+1/2)), theta_m=(2m-1)pi/(2N-1), m=1..N-1,
    lambda_m = 1-q+q cos theta_m.
    f(t) = sum_m W_m lambda_m^(t-1),
    W_m = (2q/(2N-1)) (-1)^(m+1) cos(theta_m/2) sin(theta_m).
    Continuous time (jump rate q): g(t) = sum_m (W_m) exp(-q(1-cos theta_m) t)
    (the flux factor q/2 is already inside W_m).
    """

    def __init__(self, N, q):
        self.N, self.q = N, q
        m = np.arange(1, N)
        th = (2 * m - 1) * np.pi / (2 * N - 1)
        self.theta = th
        self.W = (2 * q / (2 * N - 1)) * np.where(m % 2 == 1, 1.0, -1.0) * np.cos(th / 2) * np.sin(th)
        self.mu = 2.0 * q * np.sin(th / 2.0) ** 2   # q(1-cos theta), continuous-time rates
        self.lam = 1.0 - self.mu

    def f_discrete(self, t):
        t = np.atleast_1d(np.asarray(t, dtype=float))
        lam = self.lam
        sgn = np.sign(lam)
        la = np.log(np.abs(lam))
        out = np.zeros(t.shape)
        for i, tt in enumerate(t):
            e = tt - 1
            terms = self.W * np.exp(e * la) * np.where(sgn < 0, (-1.0) ** (e % 2), 1.0)
            out[i] = terms.sum()
        return out

    def f_discrete_smooth(self, t):
        """positive-eigenvalue part only, analytic in real t (negative-lambda
        part is bounded by sum|W| (2q-1)^(t-1), negligible for t >~ 200)."""
        t = np.atleast_1d(np.asarray(t, dtype=float))
        pos = self.lam > 0
        W, la = self.W[pos], np.log(self.lam[pos])
        return np.array([(W * np.exp((tt - 1) * la)).sum() for tt in t])

    def neg_bound(self, t):
        neg = self.lam < 0
        if not neg.any():
            return 0.0
        return float(np.abs(self.W[neg]).sum() * np.abs(self.lam[neg]).max() ** (t - 1))

    def g_cont(self, t):
        t = np.atleast_1d(np.asarray(t, dtype=float))
        return np.array([(self.W * np.exp(-self.mu * tt)).sum() for tt in t])

    def dg_cont(self, t):
        t = np.atleast_1d(np.asarray(t, dtype=float))
        return np.array([(-self.mu * self.W * np.exp(-self.mu * tt)).sum() for tt in t])

    def mfpt(self):
        return self.N * (self.N - 1) / self.q


# ---------------------------------------------------------------------------
# (C) Laplace domain: exact 1D lattice resolvent and renewal formula
# ---------------------------------------------------------------------------

def phi_of(x):
    """phi with cosh(phi) = 1 + x, Re(phi) >= 0, accurate for small |x|."""
    return 2.0 * np.arcsinh(np.sqrt(x / 2.0))


def G1(sigma, n, n0, N, a):
    """Resolvent [(sigma + a (I - P1))^{-1}]_{n,n0} of the 1D reflecting
    (cancellation) chain on 0..N-1, where P1 moves left/right w.p. 1/2 each
    (cancelled at the ends).  sigma: complex array.  Closed form:
      cosh(phi(n<+1/2)) cosh(phi(N-1/2-n>)) / ((a/2) sinh(phi) sinh(N phi)),
      cosh(phi) = 1 + sigma/a.
    Evaluated in an overflow-free scaled form (Re phi >= 0)."""
    sigma = np.asarray(sigma, dtype=complex)
    lo, hi = min(n, n0), max(n, n0)
    A = lo + 0.5
    B = N - 0.5 - hi
    ph = phi_of(sigma / a)
    num = np.exp(ph * (A + B - N)) * (1 + np.exp(-2 * ph * A)) * (1 + np.exp(-2 * ph * B))
    den = 2.0 * np.sinh(ph) * (-np.expm1(-2.0 * N * ph))
    return num / den / (a / 2.0)


def cos_coeffs(N, n, n0):
    """c_k = (alpha_k/N) cos(pi k(2n+1)/2N) cos(pi k(2n0+1)/2N), k=0..N-1.
    Arguments are reduced modulo 2*pi in exact integer arithmetic."""
    k = np.arange(N, dtype=np.int64)
    alpha = np.where(k == 0, 1.0, 2.0)
    m1 = (k * (2 * int(n) + 1)) % (4 * N)
    m2 = (k * (2 * int(n0) + 1)) % (4 * N)
    return alpha / N * np.cos(np.pi * m1 / (2.0 * N)) * np.cos(np.pi * m2 / (2.0 * N))


class LaplaceFP:
    """Continuous-time walk with total jump rate q (generator q(P1 - I)),
    P1 = the q=1 walk with cancellation.  Point target a, start x0.
    Propagator in Laplace space (a_ = q/d):
       P^(n,s|n0) = sum_{k_1..k_{d-1}} prod_i c_{k_i} * G1(s + shift_k, n_d, n0_d),
       shift_k = a_ * sum_i (1 - cos(pi k_i/N)),
    F^(s) = P^(a,s|x0) / P^(a,s|a).   (d-1)-fold sum instead of d-fold.

    Large-N acceleration (exact to round-off): terms with shift > KAPPA*rho
    ('far' terms) are analytic in |s| < KAPPA*rho; their sum is replaced by
    its degree-(NFAR-1) Taylor polynomial about s=0, obtained from NFAR
    samples on the circle |s| = rho (aliasing/truncation error
    ~ KAPPA^-NFAR ~ 1e-21).  Only 'near' terms are summed for every s.
    """
    KAPPA = 20.0
    NFAR = 16

    def __init__(self, N, d, q, start, target):
        self.N, self.d, self.q = N, d, q
        self.start, self.target = tuple(start), tuple(target)
        self.a = q / d
        a = self.a
        k = np.arange(N)
        ek = 2.0 * a * np.sin(np.pi * k / (2.0 * N)) ** 2      # a(1-cos(pi k/N)), no cancellation
        self.ek = ek
        if d == 1:
            shift = np.zeros(1); w_ax = np.ones(1); w_aa = np.ones(1)
        elif d == 2:
            shift = ek
            w_ax = cos_coeffs(N, target[0], start[0])
            w_aa = cos_coeffs(N, target[0], target[0])
        elif d == 3:
            c_ax = [cos_coeffs(N, target[i], start[i]) for i in range(2)]
            c_aa = [cos_coeffs(N, target[i], target[i]) for i in range(2)]
            sym = (target[0] == target[1]) and (start[0] == start[1])
            if sym:
                i, j = np.triu_indices(N)
                mult = np.where(i == j, 1.0, 2.0)
                shift = ek[i] + ek[j]
                w_ax = c_ax[0][i] * c_ax[0][j] * mult
                w_aa = c_aa[0][i] * c_aa[0][j] * mult
            else:
                shift = (ek[:, None] + ek[None, :]).ravel()
                w_ax = (c_ax[0][:, None] * c_ax[1][None, :]).ravel()
                w_aa = (c_aa[0][:, None] * c_aa[1][None, :]).ravel()
        else:
            raise NotImplementedError
        order = np.argsort(shift, kind="stable")
        self.shift = shift[order]
        self.w_ax = w_ax[order]
        self.w_aa = w_aa[order]
        self.rho = None

    # -- far-field Taylor polynomials ------------------------------------
    def prepare(self, rho):
        self.rho = float(rho)
        cut = self.KAPPA * self.rho
        self.n_near = int(np.searchsorted(self.shift, cut, side="right"))
        n = self.NFAR
        sm = self.rho * np.exp(2j * np.pi * np.arange(n) / n)
        self.poly = {}
        for key, w, (nn, n0) in (("ax", self.w_ax, (self.target[-1], self.start[-1])),
                                 ("aa", self.w_aa, (self.target[-1], self.target[-1]))):
            sh = self.shift[self.n_near:]
            ww = w[self.n_near:]
            T = np.zeros(n, dtype=complex)
            if sh.size:
                nz = ww != 0.0
                sh, ww = sh[nz], ww[nz]
                chunk = max(1, int(2e6 // n))
                for i0 in range(0, sh.size, chunk):
                    sig = sm[:, None] + sh[None, i0:i0 + chunk]
                    T += (G1(sig, nn, n0, self.N, self.a) * ww[None, i0:i0 + chunk]).sum(axis=1)
            # Taylor coefficients in powers of (s/rho)
            self.poly[key] = np.fft.fft(T) / n

    def _P(self, s, key):
        if key == "ax":
            w, nn, n0 = self.w_ax, self.target[-1], self.start[-1]
        else:
            w, nn, n0 = self.w_aa, self.target[-1], self.target[-1]
        sh, ww = self.shift[: self.n_near], w[: self.n_near]
        nz = ww != 0.0
        sh, ww = sh[nz], ww[nz]
        out = np.zeros(s.shape, dtype=complex)
        chunk = max(1, int(2e6 // max(1, sh.size)))
        for i0 in range(0, s.size, chunk):
            ss = s[i0:i0 + chunk]
            sig = ss[:, None] + sh[None, :]
            out[i0:i0 + chunk] = (G1(sig, nn, n0, self.N, self.a) * ww[None, :]).sum(axis=1)
        # far part: polynomial in x = s/rho (Horner)
        c = self.poly[key]
        x = s / self.rho
        far = np.zeros(s.shape, dtype=complex)
        for cj in c[::-1]:
            far = far * x + cj
        return out + far

    def Fhat(self, s):
        s = np.atleast_1d(np.asarray(s, dtype=complex))
        smax = float(np.abs(s).max())
        if self.rho is None or smax > self.rho * (1 + 1e-12):
            self.prepare(smax)
        return self._P(s, "ax") / self._P(s, "aa")

    def mfpt(self):
        """exact MFPT = |Omega| * [R_aa(0) - R_ax(0)], R = propagator with the
        zero mode removed.  The single term containing the zero mode (all
        k=0) is handled by the 1D pseudo-Green function (direct O(N) sum)."""
        N, a = self.N, self.a
        vol = float(N) ** self.d
        k = np.arange(1, N)

        def R(w, n, n0):
            idx = np.arange(1, self.shift.size)          # shift[0] = 0 is the zero-mode term
            val = 0.0
            chunk = 2000000
            for i0 in range(0, idx.size, chunk):
                ii = idx[i0:i0 + chunk]
                ww = w[ii]
                nz = ww != 0.0
                val += (G1(self.shift[ii][nz].astype(complex), n, n0, N, a).real * ww[nz]).sum()
            ck = cos_coeffs(N, n, n0)[1:]
            pg = (ck / self.ek[1:]).sum()
            return val + w[0] * pg

        Raa = R(self.w_aa, self.target[-1], self.target[-1])
        Rax = R(self.w_ax, self.target[-1], self.start[-1])
        return vol * (Raa - Rax)


# ---------------------------------------------------------------------------
# fixed-Talbot inversion (Abate & Valko 2004)
# ---------------------------------------------------------------------------

def talbot_nodes(M):
    k = np.arange(1, M)
    th = k * np.pi / M
    cot = 1.0 / np.tan(th)
    s_unit = th * (cot + 1j)                     # s/r
    sig = th + (th * cot - 1.0) * cot            # d s/d theta factor
    return s_unit, sig


def talbot_invert(Fhat, t, M=24, deriv=0):
    """Invert Laplace transform at times t (array).  deriv=1 inverts s*Fhat
    (i.e. returns g'(t) assuming g(0)=0)."""
    t = np.atleast_1d(np.asarray(t, dtype=float))
    s_unit, sig = talbot_nodes(M)
    out = np.zeros(t.shape)
    # evaluate all nodes for all t in one batched call
    r = 2.0 * M / (5.0 * t)
    S = np.concatenate([r[:, None].astype(complex), r[:, None] * s_unit[None, :]], axis=1)
    Fv = Fhat(S.ravel()).reshape(S.shape)
    if deriv == 1:
        Fv = Fv * S
    e = np.exp(t[:, None] * S)
    term0 = 0.5 * (Fv[:, 0] * e[:, 0]).real
    rest = (e[:, 1:] * Fv[:, 1:] * (1 + 1j * sig[None, :])).real.sum(axis=1)
    out = r / M * (term0 + rest)
    return out


# ---------------------------------------------------------------------------
# weak-target (small-target) leading-order mode prediction
# ---------------------------------------------------------------------------

def weak_target_prediction(N, d, q, start, target, mfpt):
    """u(t) = |Omega| p_refl(a,t|x0) = prod_i [1 + 2 sum_k c'_k e^{-a(1-cos(pi k/N))t}]
    with c'_k = cos(theta_k(a_i)) cos(theta_k(x0_i)).  At late times
    u ~ 1 - A e^{-mu t} with mu the slowest rate with nonzero amplitude.
    Leading-order mode:  A mu exp(-mu t*) = 1/MFPT  =>  t* = ln(A mu MFPT)/mu.
    Returns (t*, A, mu)."""
    a = q / d
    k = np.arange(1, N)
    rate = 2.0 * a * np.sin(np.pi * k / (2.0 * N)) ** 2
    amps = {}
    for i in range(d):
        ck = np.cos(np.pi * k * (2 * target[i] + 1) / (2 * N)) * np.cos(np.pi * k * (2 * start[i] + 1) / (2 * N))
        # first k with nonzero amplitude
        j = np.where(np.abs(ck) > 1e-12)[0][0]
        amps.setdefault(rate[j], 0.0)
        amps[rate[j]] += 2 * ck[j]
    mu = min(amps)
    A = -amps[mu]
    if A <= 0:
        return float("nan"), A, mu
    return math.log(A * mu * mfpt) / mu, A, mu
