"""Second implementation (check): spectral / resolvent tools for the reflecting hypercube.

Unit-rate continuous-time walk (generator P1 - I, P1 = the q = 1 one-step matrix of the
model).  Along each axis the 1D reflecting chain has rate a = 1/d, eigenvalues
e_k = a (1 - cos(pi k / N)) and eigenvectors sqrt(alpha_k / N) cos(pi k (2n+1) / 2N).

Corner target a = (N-1,..,N-1), corner start x0 = (0,..,0):
    phi_k(a)^2        = prod_i w_{k_i},            w_k = (alpha_k / N) cos^2(pi k / 2N)
    phi_k(a) phi_k(x0) = prod_i (-1)^{k_i} w_{k_i}

One axis is summed in closed form with the end-site resolvent of the 1D chain,
    Gaa(sig) = [sig + L1]^{-1}_{N-1,N-1},   Gax(sig) = [sig + L1]^{-1}_{N-1,0},
written with E = exp(-phi), cosh(phi) = 1 + sig/a  (derived here; checked against dense
inverses in v02_validate.py):
    Gaa = (2/a) E (1 + E^{2N-1}) / ((1 - E)(1 - E^{2N}))
    Gax = (2/a) E^N (1 + E)      / ((1 - E)(1 - E^{2N}))

Everything is written for general (complex) sig so that it also works on the negative
real axis between poles.
"""
import numpy as np
from scipy.optimize import brentq


def e1d(k, N, d):
    """1D eigenvalue a(1-cos(pi k/N)) = 2a sin^2(pi k / 2N), a = 1/d."""
    return (2.0 / d) * np.sin(np.pi * np.asarray(k, dtype=float) / (2.0 * N)) ** 2


def w1d(k, N):
    k = np.asarray(k)
    alpha = np.where(k == 0, 1.0, 2.0)
    return alpha / N * np.cos(np.pi * k / (2.0 * N)) ** 2


def _E(sig, a):
    sig = np.asarray(sig, dtype=complex)
    phi = 2.0 * np.arcsinh(np.sqrt(sig / (2.0 * a)))
    return phi


def G_end(sig, N, d, deriv=False):
    """Return (Gaa, Gax[, dGaa/dsig]) for the 1D chain with rate a = 1/d."""
    a = 1.0 / d
    phi = _E(sig, a)
    E = np.exp(-phi)
    omE = -np.expm1(-phi)                 # 1 - E
    E2N = np.exp(-2.0 * N * phi)
    omE2N = -np.expm1(-2.0 * N * phi)     # 1 - E^{2N}
    E2Nm1 = np.exp(-(2.0 * N - 1.0) * phi)
    Gaa = (2.0 / a) * E * (1.0 + E2Nm1) / (omE * omE2N)
    Gax = (2.0 / a) * np.exp(-N * phi) * (1.0 + E) / (omE * omE2N)
    if not deriv:
        return Gaa, Gax
    dlog = -1.0 - (2.0 * N - 1.0) * E2Nm1 / (1.0 + E2Nm1) - E / omE - 2.0 * N * E2N / omE2N
    sinh_phi = np.sinh(phi)
    dGaa = Gaa * dlog / (a * sinh_phi)
    return Gaa, Gax, dGaa


class Corner:
    """Corner-to-corner problem in d = 2 or 3 (unit rate)."""

    def __init__(self, d, N, far_factor=None):
        self.d, self.N = d, N
        k = np.arange(N)
        self.e = e1d(k, N, d)
        self.w = w1d(k, N)
        self.sgn = np.where(k % 2 == 0, 1.0, -1.0)
        if d == 2:
            self.shift = self.e
            self.wt = self.w
            self.sg = self.sgn
        elif d == 3:
            i, j = np.triu_indices(N)
            mult = np.where(i == j, 1.0, 2.0)
            self.shift = self.e[i] + self.e[j]
            self.wt = mult * self.w[i] * self.w[j]
            self.sg = self.sgn[i] * self.sgn[j]
        else:
            raise ValueError
        self.mu1 = float(self.e[1])
        self.near = None
        if far_factor is not None:
            self._setup_far(far_factor)

    # ---- raw sums -------------------------------------------------------------------
    def _sums(self, s, idx=None):
        sh = self.shift if idx is None else self.shift[idx]
        wt = self.wt if idx is None else self.wt[idx]
        sg = self.sg if idx is None else self.sg[idx]
        tot = np.zeros(3, dtype=complex)
        CH = 1 << 20
        for lo in range(0, len(sh), CH):
            Gaa, Gax, dGaa = G_end(s + sh[lo:lo + CH], self.N, self.d, deriv=True)
            w_ = wt[lo:lo + CH]
            tot[0] += np.sum(w_ * Gaa)
            tot[1] += np.sum(w_ * sg[lo:lo + CH] * Gax)
            tot[2] += np.sum(w_ * dGaa)
        return tot

    def _setup_far(self, far_factor, smax_units=40.0, ncheb=16):
        """Split terms into 'near' (shift <= far_factor*mu1) and 'far'; far part is replaced by a
        Chebyshev interpolant in s on [-smax, 0] (far terms are analytic and nearly constant there)."""
        self.near = np.nonzero(self.shift <= far_factor * self.mu1)[0]
        far = np.nonzero(self.shift > far_factor * self.mu1)[0]
        self.smax = smax_units * self.mu1
        x = np.cos(np.pi * (np.arange(ncheb) + 0.5) / ncheb)       # Chebyshev nodes on [-1,1]
        s_nodes = -0.5 * self.smax * (1.0 - x)
        vals = np.array([self._sums(s, far).real for s in s_nodes])   # (ncheb, 3)
        self.cheb = [np.polynomial.chebyshev.Chebyshev.fit(x, vals[:, c], ncheb - 1, domain=[-1, 1]) for c in range(3)]

    def PQ(self, s):
        """(P_aa, P_ax, dP_aa/ds) (each multiplied by |Omega|^0, i.e. plain spectral sums)."""
        if self.near is None:
            return self._sums(s).real
        assert -self.smax <= s <= 0.0
        x = 1.0 + 2.0 * s / self.smax
        far = np.array([c(x) for c in self.cheb])
        return self._sums(s, self.near).real + far

    # ---- MFPT ------------------------------------------------------------------------
    def mfpt(self):
        """Exact unit-rate MFPT = |Omega| sum_{k != 0} [phi_k(a)^2 - phi_k(a)phi_k(x0)] / mu_k."""
        d, N = self.d, self.N
        nz = self.shift > 0
        Gaa, Gax = G_end(self.shift[nz], N, d)
        tot = np.sum(self.wt[nz] * (Gaa.real - self.sg[nz] * Gax.real))
        # remaining axis-only terms: w_0^(d-1) * sum_{k>=1} w_k (1-(-1)^k)/e_k
        k = np.arange(1, N)
        D0 = np.sum(self.w[1:] * (1.0 - self.sgn[1:]) / self.e[1:])
        tot += (1.0 / N) ** (d - 1) * D0
        return float(N ** d * tot)

    # ---- poles and residues ----------------------------------------------------------
    def distinct_poles(self, numax_units):
        """Sorted distinct reflecting eigenvalues mu <= numax (all carry weight at a corner)."""
        N, d = self.N, self.d
        kmax = int(np.ceil(np.sqrt(numax_units))) + 2
        kmax = min(kmax, N - 1)
        ks = np.arange(kmax + 1)
        e = self.e[: kmax + 1]
        if d == 2:
            vals = (e[:, None] + e[None, :]).ravel()
        else:
            vals = (e[:, None, None] + e[None, :, None] + e[None, None, :]).ravel()
        vals = np.sort(vals[vals <= numax_units * self.mu1 * (1 + 1e-12)])
        out = [vals[0]]
        for v in vals[1:]:
            if v - out[-1] > 1e-9 * self.mu1:
                out.append(v)
        return np.array(out)

    def poles(self, numax_units=12.0, verbose=False):
        """Zeros -nu_j of P_aa on the negative real axis with nu_j < numax, and residues a_j of F^."""
        mus = self.distinct_poles(numax_units)
        nus, res = [], []
        f = lambda nu: self.PQ(-nu)[0]
        for lo, hi in zip(mus[:-1], mus[1:]):
            gap = hi - lo
            eps = 1e-9
            a_, b_ = lo + eps * gap, hi - eps * gap
            fa, fb = f(a_), f(b_)
            if not (fa < 0 < fb):
                if verbose:
                    print("  skipped interval", lo / self.mu1, hi / self.mu1, fa, fb)
                continue
            nu = brentq(f, a_, b_, xtol=1e-15 * hi, rtol=1e-15, maxiter=200)
            Paa, Pax, dPaa = self.PQ(-nu)
            nus.append(nu); res.append(Pax / dPaa)
        return np.array(nus), np.array(res)


def density_mode(nus, res, t_lo=None, t_hi=None):
    """Maximiser of g(t) = sum_j res_j exp(-nu_j t) (unit rate); returns (t*, g(t*))."""
    gp = lambda t: -np.sum(res * nus * np.exp(-nus * t))
    if t_lo is None:
        # heuristic bracket around the two-pole balance point
        t0 = np.log(-res[1] * nus[1] / (res[0] * nus[0])) / (nus[1] - nus[0])
        t_lo, t_hi = 0.5 * t0, 2.0 * t0
    assert gp(t_lo) > 0 > gp(t_hi), (gp(t_lo), gp(t_hi))
    t = brentq(gp, t_lo, t_hi, xtol=1e-14 * t_hi, rtol=1e-15)
    return t, float(np.sum(res * np.exp(-nus * t)))


def discrete_mode_from_poles(nus, res, q, t_guess):
    """argmax over integers of f(t) = sum_j q a_j (1 - q nu_j)^(t-1) near t_guess, and the real-t maximiser.

    Only poles with 1 - q nu_j > 0 enter the real-t interpolation; the integer comparison uses all poles."""
    lam = 1.0 - q * nus
    pos = lam > 0
    def f_int(t):
        return np.sum(q * res * lam ** int(t - 1))
    def fp(t):
        return np.sum(q * res[pos] * np.log(lam[pos]) * lam[pos] ** (t - 1.0))
    t = brentq(fp, 0.5 * t_guess, 2.0 * t_guess, xtol=1e-12 * t_guess)
    cand = [c for c in range(int(np.floor(t)) - 2, int(np.floor(t)) + 4) if c >= 1]
    best = max(cand, key=f_int)
    return best, t
