"""
check_lc_threshold.py -- exact evidence for Conjecture 9.4 (log-concavity of the free-propagator increments beyond q = 1/2).

Part A (d = 1, end-to-end, exact at the threshold).
    With m = N-1 and eigenvalues lambda_k = 1-q+q cos(k pi/N), the increments of the 1D propagator are
        N * [h_q(n) - h_q(n-1)] = prod_k(1-lambda_k) * s_{n-m},   s_j = h_j(lambda_1..lambda_m),
    and  s_j = (q/2)^j W_{m-1+j}(y),  W_n(y) = ((y I + Adj(P_m))^n)(1,m),  y = 2(1-q)/q,
    (P_m = path with m vertices).  The first log-concavity inequality s_1^2 >= s_0 s_2 is  y^2 >= 2/m,
    i.e. q <= q_1(N) = 1/(1+1/sqrt(2(N-1))).   Here we verify EXACTLY, in the ring Z[sqrt(2m)], that at the threshold
    y^2 = 2/m all inequalities  W_n^2 >= W_{n-1} W_{n+1}  hold for n-m+1 = j = 1..J.
    (By Lemma 4.10(b) -- binomial thinning -- log-concavity on the window then holds for every q <= q_1(N).)

Part B (d = 2, 3, corner to corner): exact test of log-concavity of phi(t) = u(t)-u(t-1), u(t) = P_q^t(x0,a), on windows,
    for q in {3/5, 4/5, 9/10}.

Output: data/check_lc_threshold.json
"""
from __future__ import annotations
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from explore12_discrete_lc import lc_report  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = _os.path.join(_R, 'data', 'msc_rigorous_unimodality')


class Z2m:
    """a + b*sqrt(R), a, b Python ints."""
    __slots__ = ("a", "b", "R")

    def __init__(self, a, b, R):
        self.a, self.b, self.R = a, b, R

    def __add__(self, o):
        return Z2m(self.a + o.a, self.b + o.b, self.R)

    def __sub__(self, o):
        return Z2m(self.a - o.a, self.b - o.b, self.R)

    def __mul__(self, o):
        return Z2m(self.a * o.a + self.b * o.b * self.R, self.a * o.b + self.b * o.a, self.R)

    def scale(self, k):
        return Z2m(self.a * k, self.b * k, self.R)


def part_a(N, J):
    m = N - 1
    R = 2 * m                       # (m*y)^2 = 2m at the threshold y^2 = 2/m
    yy = Z2m(0, 1, R)               # m*y = sqrt(2m)
    zero = Z2m(0, 0, R)
    v = [zero] * m
    v[0] = Z2m(1, 0, R)
    W = []
    for n in range(m - 1 + J + 2):
        W.append(v[m - 1])
        new = []
        for i in range(m):
            acc = yy * v[i]
            if i > 0:
                acc = acc + v[i - 1].scale(m)
            if i + 1 < m:
                acc = acc + v[i + 1].scale(m)
            new.append(acc)
        v = new
    ok = True
    first_fail = None
    n_zero = 0
    for j in range(1, J + 1):
        n = m - 1 + j
        D = W[n] * W[n] - W[n - 1] * W[n + 1]
        assert D.b == 0             # parity: Delta is a polynomial in y^2
        if D.a < 0:
            ok = False
            first_fail = j
            break
        if D.a == 0:
            n_zero += 1
    # positivity of W
    pos = all((w.a > 0 and w.b == 0) or (w.a == 0 and w.b > 0) for w in W[m - 1:])
    return dict(N=N, m=m, J=J, lc_at_threshold=ok, first_fail=first_fail, equalities=n_zero, W_positive=pos)


if __name__ == "__main__":
    out = dict(part_a=[], part_b=[])
    for N in range(3, 41):
        J = min(4 * N * N, 500)
        rec = part_a(N, J)
        out["part_a"].append(rec)
        print("A", rec, flush=True)
        with open(_os.path.join(_R, 'data', 'msc_rigorous_unimodality', 'check_lc_threshold.json'), "w") as fh:
            json.dump(out, fh, indent=1)
    for d, Ns in [(2, list(range(2, 21)) + [24]), (3, list(range(2, 11)))]:
        for N in Ns:
            T = int(5 * d * N * N) + 40
            for (qn, qd) in [(3, 5), (4, 5), (9, 10)]:
                r = lc_report(N, d, qn, qd, (0,) * d, (N - 1,) * d, T)
                rec = dict(d=d, N=N, q="%d/%d" % (qn, qd), T=T, log_concave_on_window=(r["n_viol"] == 0 and r["n_neg"] == 0),
                           first_violations=r["first_viol"])
                out["part_b"].append(rec)
                print("B", rec, flush=True)
                with open(_os.path.join(_R, 'data', 'msc_rigorous_unimodality', 'check_lc_threshold.json'), "w") as fh:
                    json.dump(out, fh, indent=1)
