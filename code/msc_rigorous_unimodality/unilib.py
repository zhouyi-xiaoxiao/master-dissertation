"""
unilib.py -- exact machinery for the note on unimodality of first-passage laws.

Model: lazy nearest-neighbour walk on the box {0,..,N-1}^d (sites are 0-based
in code; the text uses {1,..,N}), move probability q = qn/qd; each of the 2d
directions has probability w = q/(2d); a move leaving the box is cancelled
(walker stays); single absorbing target a; start x0.

EXACT INTEGER REPRESENTATION.  With the common one-step denominator
    D = 2*d*qd / g            (g = gcd of all integer weights below)
the one-step weights are the integers
    move  : m  = qn / g
    stay  : s(x) = (2*d*(qd-qn) + qn*nblk(x)) / g ,  nblk(x) = number of
            blocked directions at x (0..d),
and  rho_t = R_t / D^t  with R_t an integer vector (numpy object array of
Python ints, arbitrary precision).  The first-passage PMF is
    f(t) = F_t / D^t ,  F_t = m * sum_{y ~ a} R_{t-1}(y).
Signs of differences are decided exactly:
    sign(f(t+1) - f(t)) = sign(F_{t+1} - D*F_t).

Nothing here uses floating point unless a function name says so.
"""
from __future__ import annotations

import itertools
import math
from fractions import Fraction

import numpy as np


# --------------------------------------------------------------------------
# exact integer stepper
# --------------------------------------------------------------------------
class ExactStepper:
    def __init__(self, N, d, qn, qd, start, target):
        assert 0 < qn <= qd
        self.N, self.d = N, d
        self.shape = (N,) * d
        m = qn
        base = 2 * d * (qd - qn)
        g = math.gcd(m, base) if base else m
        g = math.gcd(g, qn)  # qn*nblk for nblk=1
        self.m = m // g
        self.D = 2 * d * qd // g
        assert (2 * d * qd) % g == 0
        nblk = np.zeros(self.shape, dtype=object)
        for ax in range(d):
            i0 = [slice(None)] * d
            i0[ax] = 0
            nblk[tuple(i0)] += 1
            i1 = [slice(None)] * d
            i1[ax] = N - 1
            nblk[tuple(i1)] += 1
        stay = np.empty(self.shape, dtype=object)
        for idx in np.ndindex(*self.shape):
            stay[idx] = (base + qn * int(nblk[idx])) // g
        self.stay = stay
        self.start = tuple(start)
        self.target = tuple(target)
        assert self.start != self.target
        R = np.zeros(self.shape, dtype=object)
        for idx in np.ndindex(*self.shape):
            R[idx] = 0
        R[self.start] = 1
        self.R = R
        self.t = 0
        self._sl = []
        for ax in range(d):
            lo = [slice(None)] * d
            hi = [slice(None)] * d
            lo[ax] = slice(0, N - 1)
            hi[ax] = slice(1, N)
            self._sl.append((tuple(lo), tuple(hi)))

    def apply_Q(self, R):
        """Return (D*Q) R  restricted to non-target sites, and the integer
        flux into the target, both exact."""
        new = self.stay * R
        mR = R * self.m if self.m != 1 else R
        for lo, hi in self._sl:
            new[hi] += mR[lo]
            new[lo] += mR[hi]
        flux = new[self.target]
        # the target row/column is deleted: mass that entered the target is
        # the flux; R(target) is always 0 so stay*R contributes 0 there.
        new[self.target] = 0
        return new, flux

    def step(self):
        """Advance one step.  Returns integer F_t with f(t) = F_t / D^t."""
        self.R, flux = self.apply_Q(self.R)
        self.t += 1
        return flux


def exact_pmf_numerators(N, d, qn, qd, start, target, T):
    """[F_1, ..., F_T] with f(t) = F_t / D^t; also returns D."""
    st = ExactStepper(N, d, qn, qd, start, target)
    F = [st.step() for _ in range(T)]
    return F, st.D


def sign_pattern_from_numerators(F, D):
    """signs of f(t+1)-f(t), t = 0..T-1 (f(0)=0), exact."""
    out = []
    prev = 0  # F_0 = 0
    for Ft in F:
        diff = Ft - D * prev
        out.append((diff > 0) - (diff < 0))
        prev = Ft
    return out


def count_sign_changes(signs):
    """number of sign changes of a +-1/0 sequence, zeros discarded
    (this is S^- in Karlin's notation)."""
    nz = [s for s in signs if s != 0]
    return sum(1 for i in range(len(nz) - 1) if nz[i] != nz[i + 1])


# --------------------------------------------------------------------------
# float helpers (exploration only)
# --------------------------------------------------------------------------
def site_index(N, d):
    sites = list(itertools.product(range(N), repeat=d))
    idx = {s: i for i, s in enumerate(sites)}
    return sites, idx


def laplacian_q1(N, d):
    """L_1 = I - P_1 for the q=1 walk on the full box (dense float),
    symmetric, row sums zero."""
    sites, idx = site_index(N, d)
    n = len(sites)
    w = 1.0 / (2 * d)
    L = np.zeros((n, n))
    for s in sites:
        i = idx[s]
        for ax in range(d):
            for dx in (-1, 1):
                y = list(s)
                y[ax] += dx
                if 0 <= y[ax] < N:
                    j = idx[tuple(y)]
                    L[i, j] -= w
                    L[i, i] += w
    return L, sites, idx


def killed(L, idx, target):
    k = idx[tuple(target)]
    keep = [i for i in range(L.shape[0]) if i != k]
    return L[np.ix_(keep, keep)], keep


def spectral_data(N, d, start, target):
    """eigenvalues sigma_k of the killed q=1 Laplacian and the residues
    c_k = phi_k(x0) * sigma_k * (1.phi_k)   (so that the continuous-time
    density with total jump rate 1 is  sum_k c_k exp(-sigma_k t) and the
    discrete PMF at activity q is  q * sum_k c_k (1-q sigma_k)^(t-1))."""
    L, sites, idx = laplacian_q1(N, d)
    Lk, keep = killed(L, idx, target)
    sig, V = np.linalg.eigh(Lk)
    pos = keep.index(idx[tuple(start)])
    c = V[pos, :] * sig * V.sum(axis=0)
    return sig, c
