"""Remark 3.17(a) (exact rational arithmetic; NOT a proof): where the TP2-type minors of the proof of
Theorem 3.6 become negative when q > 2/3.

   K_t(x) = (Q^t)_{x,n},   M_t(x,y) = K_t(x) K_{t-1}(y) - K_{t-1}(x) K_t(y)   (1 <= x < y <= n = N-1, t >= 1);
   f(t)^2 - f(t-1) f(t+1) = (q/2)^3 M_{t-1}(1,2)   (so f is log-concave iff M_t(1,2) >= 0 for all t).

For q = 4/5 (and 3/4, 7/10) and 5 <= N <= NMAX we list all (n-x, n-y, t) with M_t(x,y) < 0, for t <= T(N) = N^2.
Observed: for q = 4/5 the negative minors occur only within distance 6 of the absorbing end and for t <= 10, the set
of triples (n-x, n-y, t) is the same for all N >= 12, and M_t(1,2) >= 0 for every N >= 5 (equality at N = 5, t = 4):
f is log-concave for q = 4/5, N >= 5, in the tested range.
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import os, sys, json
from fractions import Fraction as Fr

NMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 24
res = {}
for q in (Fr(4, 5), Fr(3, 4), Fr(7, 10)):
    a, b = 1 - q, q / 2
    rows = []
    for N in range(5, NMAX + 1):
        n = N - 1
        T = N * N
        prev = [Fr(0)] * (n + 2)
        prev[n] = Fr(1)
        neg = []
        min12 = None
        for t in range(1, T + 1):
            cur = [Fr(0)] * (n + 2)
            for x in range(1, n + 1):
                left = prev[x - 1] if x > 1 else prev[1]
                right = prev[x + 1] if x < n else Fr(0)
                cur[x] = a * prev[x] + b * (left + right)
            for x in range(1, n):
                cx, px = cur[x], prev[x]
                if cx == 0 and px == 0:
                    continue
                for y in range(x + 1, n + 1):
                    M = cx * prev[y] - px * cur[y]
                    if M < 0:
                        neg.append((n - x, n - y, t))
                    if x == 1 and y == 2 and (cx != 0):
                        # normalised: M / (K_t(1) K_{t-1}(2)) in [.,1]; record the minimum of the sign only
                        if M < 0:
                            min12 = "NEGATIVE at t=%d" % t
                        elif M == 0 and prev[2] != 0 and min12 is None:
                            min12 = "zero at t=%d" % t
            prev = cur
        rows.append(dict(N=N, T=T, n_negative=len(neg), max_t=max([t for _, _, t in neg], default=0),
                         max_dist=max([d for d, _, _ in neg], default=0), M12=min12 or "positive whenever K_{t-1}(2) > 0",
                         negatives=sorted(neg)))
        print(float(q), {k: v for k, v in rows[-1].items() if k != "negatives"}, flush=True)
    univ = [r["negatives"] for r in rows if r["N"] >= 12]
    res[str(q)] = dict(rows=rows, same_set_for_N_ge_12=all(u == univ[0] for u in univ) if univ else None,
                       universal_set=univ[0] if univ else None)
    print("q =", q, ": same set of (n-x, n-y, t) for all N >= 12:", res[str(q)]["same_set_for_N_ge_12"], "; set:", univ[0] if univ else None)
json.dump(res, open(_os.path.join(_R, 'data', 'msc_rigorous_1d', '30_tp2_pattern.json'), "w"), indent=1)
