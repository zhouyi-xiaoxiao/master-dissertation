"""fpcore.py -- reference (verifying) implementation.  Trusted base: Python integers only.

Model.  Lazy nearest-neighbour walk on the box {0,...,N-1}^d with move probability q = qn/qd:
each of the 2d directions has probability q/(2d); a move that would leave the box is cancelled
(the walker stays); with probability 1-q the walker rests.  Target a = (N-1,...,N-1) is absorbing,
start x0 = (0,...,0).  Q = transition matrix restricted to the non-target sites.

Integer form.  Q = M/D with  M = cm*(A + B) + cs*I  (A = adjacency of the box minus the target,
B = diag(number of blocked directions)),  cm/D = q/(2d),  cs/D = 1-q.

Occupation vector v_1 = e_{x0}, v_{t+1} = Q v_t;   f(t) = P(T = t) = (cm/D) * s_t,
s_t = sum of v_t over the d neighbours of the target.

Two phases (both produce integers only):
  exact phase  (t <= ts):  X_t = D^(t-1) v_t  (integer vector),  X_{t+1} = M X_t.
  rounded phase (t >= ts): L_ts = floor(2^P X_ts / D^(ts-1)), U_ts = ceil(2^P X_ts / D^(ts-1)),
                           L_{t+1} = floor(M L_t / D),  U_{t+1} = ceil(M U_t / D),
                           so that  L_t <= 2^P v_t <= U_t  entrywise (Lemma 7 of note R5).
  ts = first t with floor(2^P s_t) >= 2^G  (a heuristic; any ts gives valid enclosures).

The run stops at the first time T at which  v_{T+1} < v_T  is certified at EVERY non-target site
(U_{T+1} < L_T entrywise, or X_{T+1} < D X_T in the exact phase).  By Lemma 1 (tail lemma) f is then
strictly decreasing on [T, infinity).
"""
import hashlib
from math import gcd
import numpy as np


def coeffs(d, qn, qd):
    """Integers (cm, cs, D) with q/(2d) = cm/D and 1-q = cs/D, in lowest terms."""
    g = gcd(gcd(qn, 2 * d * (qd - qn)), 2 * d * qd)
    return qn // g, 2 * d * (qd - qn) // g, 2 * d * qd // g


def neighbours_of_target(d, N):
    out = []
    for ax in range(d):
        idx = [N - 1] * d
        idx[ax] = N - 2
        out.append(tuple(idx))
    return out


def apply_M(X, d, N, cm, cs):
    """M X for an object (Python-int) array X of shape (N,)*d with X[target] == 0."""
    S = np.zeros_like(X)
    for ax in range(d):
        Xm = np.moveaxis(X, ax, 0)
        Sm = np.moveaxis(S, ax, 0)
        Sm[1:] += Xm[:-1]      # arrivals by a +1 move along this axis
        Sm[0] += Xm[0]         # the -1 move at the wall is cancelled: walker stays
        Sm[:-1] += Xm[1:]      # arrivals by a -1 move
        Sm[-1] += Xm[-1]       # the +1 move at the wall is cancelled
    W = cm * S + cs * X
    W[(N - 1,) * d] = 0        # mass entering the target is absorbed
    return W


def digest(*arrays):
    """sha256 of the decimal representation of integer arrays (C order)."""
    h = hashlib.sha256()
    for A in arrays:
        h.update((",".join(str(int(x)) for x in A.ravel()) + ";").encode())
    return h.hexdigest()


class Checker:
    """Consumes the stream of per-step data and assembles the certificate."""

    def __init__(self, d, N, qn, qd, P, G):
        self.cm, self.cs, self.D = coeffs(d, qn, qd)
        self.meta = dict(d=d, N=N, q=f"{qn}/{qd}", cm=self.cm, cs=self.cs, D=self.D, P=P, G=G)
        self.P = P
        self.FL = [None]   # FL[t] <= 2^P s_t <= FU[t]   (index 0 unused)
        self.FU = [None]
        self.rel = [None]  # rel[t] in '<', '>', '=', '?' : certified relation between f(t) and f(t+1)
        self.exact_prev = None   # (s_t, t) while in the exact phase

    def exact_step(self, t, s_t):
        """Exact phase: s_t = sum of X_t over target neighbours (scale D^(t-1))."""
        Dp = self.D ** (t - 1)
        self.FL.append((s_t << self.P) // Dp)
        self.FU.append(-((-(s_t << self.P)) // Dp))
        if self.exact_prev is not None:
            a, b = self.D * self.exact_prev, s_t          # f(t-1) vs f(t)  <=>  D s_{t-1} vs s_t
            self.rel.append('<' if a < b else '>' if a > b else '=')
        self.exact_prev = s_t

    def rounded_step(self, t, sL, sU, first=False):
        """Rounded phase: sL <= 2^P s_t <= sU.  first=True at t = ts (FL, FU already stored)."""
        if first:
            self.FL[t], self.FU[t] = sL, sU
            self.exact_prev = None
            return
        if self.FU[t - 1] < sL:
            self.rel.append('<')
        elif self.FL[t - 1] > sU:
            self.rel.append('>')
        else:
            self.rel.append('?')
        self.FL.append(sL)
        self.FU.append(sU)

    def finish(self, T, ts, tail_margin, dig):
        """T = tail time (v_{T+1} < v_T entrywise certified); data known for t = 1..T+1."""
        FL, FU, rel = self.FL, self.FU, self.rel
        tstar = max(range(1, T + 1), key=lambda t: (FL[t], -t))
        mode_ok = all(FL[tstar] > FU[t] for t in range(1, T + 1) if t != tstar)
        t0 = next(t for t in range(1, T + 2) if FU[t] > 0)
        zeros_ok = all(rel[t] == '=' for t in range(1, t0 - 1))
        inc_ok = all(rel[t] == '<' for t in range(max(t0 - 1, 1), tstar))
        dec_ok = all(rel[t] == '>' for t in range(tstar, T))
        # certified strict local maxima at times t <= T (f(T) > f(T+1) holds by the tail check)
        r = lambda t: '>' if t == T else rel[t]
        nmax = sum(1 for t in range(1, T + 1) if r(t) == '>' and (t == 1 or rel[t - 1] == '<'))
        undecided = sum(1 for t in range(1, T) if rel[t] == '?')
        c = dict(self.meta)
        c.update(ts=ts, t0=t0, mode=tstar, T_tail=T, mode_certified=bool(mode_ok),
                 unimodal_certified=bool(mode_ok and zeros_ok and inc_ok and dec_ok),
                 certified_local_maxima_upto_T=nmax, undecided_adjacent_pairs=undecided,
                 tail_margin=str(tail_margin), digest=dig,
                 F_enclosures={str(t): [str(FL[t]), str(FU[t])]
                               for t in (tstar - 1, tstar, tstar + 1) if 1 <= t <= T + 1})
        if tstar > 1:
            c["gap_left"] = str(FL[tstar] - FU[tstar - 1])
        c["gap_right"] = str(FL[tstar] - FU[tstar + 1])
        return c


def switch_time_reached(s_t, t, D, P, G):
    return ((s_t << P) // D ** (t - 1)) >> G > 0


def exact_phase(d, N, qn, qd, P, G, tmax):
    """Exact integer phase.  Returns ('done', certificate) if the tail condition is met before the
    switch, else ('switch', X_ts, ts, checker)."""
    cm, cs, D = coeffs(d, qn, qd)
    nb = neighbours_of_target(d, N)
    target = (N - 1,) * d
    ck = Checker(d, N, qn, qd, P, G)
    X = np.zeros((N,) * d, dtype=object)
    X[...] = 0
    X[(0,) * d] = 1
    t = 1
    while True:
        s = sum(X[i] for i in nb)
        ck.exact_step(t, s)
        if switch_time_reached(s, t, D, P, G):
            return 'switch', X, t, ck
        Xn = apply_M(X, d, N, cm, cs)
        mask = (Xn < D * X)
        mask[target] = True
        if mask.all():                      # tail certified inside the exact phase
            ck.exact_step(t + 1, sum(Xn[i] for i in nb))
            margin = min((D * X - Xn)[m] for m in np.ndindex(X.shape) if m != target)
            return 'done', ck.finish(t, None, f"exact:{margin}", digest(X, Xn)), None, None
        X = Xn
        t += 1
        if t > tmax:
            raise RuntimeError("tmax reached in exact phase")


def to_fixed_point(X, ts, D, P):
    """L = floor(2^P X / D^(ts-1)), U = ceil(2^P X / D^(ts-1))."""
    Dp = D ** (ts - 1)
    return (X * (1 << P)) // Dp, -((-(X * (1 << P))) // Dp)


def certify(d, N, qn, qd, P=112, G=48, tmax=10 ** 8):
    """Reference certificate, Python integers only."""
    cm, cs, D = coeffs(d, qn, qd)
    nb = neighbours_of_target(d, N)
    target = (N - 1,) * d
    kind, X, ts, ck = exact_phase(d, N, qn, qd, P, G, tmax)
    if kind == 'done':
        return X
    L, U = to_fixed_point(X, ts, D, P)
    ck.rounded_step(ts, sum(L[i] for i in nb), sum(U[i] for i in nb), first=True)
    t = ts
    while True:
        Ln = apply_M(L, d, N, cm, cs) // D
        Un = -((-apply_M(U, d, N, cm, cs)) // D)
        ck.rounded_step(t + 1, sum(Ln[i] for i in nb), sum(Un[i] for i in nb))
        diff = L - Un                      # need > 0 at every non-target site
        diff[target] = 1
        if (diff > 0).all():
            diff[target] = diff.max()
            return ck.finish(t, ts, diff.min(), digest(L, Un))
        L, U = Ln, Un
        t += 1
        if t > tmax:
            raise RuntimeError("tmax reached")


if __name__ == "__main__":
    import json, sys, time
    d, N, qn, qd = (int(x) for x in sys.argv[1:5])
    t0 = time.time()
    c = certify(d, N, qn, qd)
    c["seconds"] = round(time.time() - t0, 2)
    print(json.dumps(c))
