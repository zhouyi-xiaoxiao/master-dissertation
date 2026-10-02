"""Exact integer arithmetic (no rounding): first-passage PMF of the simple walk (q = 1) with reflecting walls (cancelled moves),
corner to opposite corner, d = 2.  4^t P(T = t) is an integer (number of move sequences of length t hitting the target first at t).
Reports the sign pattern of P(T = t+1) - P(T = t) near the global maximiser.  Illustrates Remark 7.19(ii): for q = 1 the PMF is
not unimodal in general.  Output: logs/82_q1_parity_example.log (stdout)."""
from fractions import Fraction
import sys


def pmf_counts(N, tmax):
    d = 2
    cnt = {(0, 0): 1}                         # number of sequences (out of 4^t) at each transient site
    tgt = (N - 1, N - 1)
    out = [0]
    for t in range(1, tmax + 1):
        new = {}
        for (i, j), c in cnt.items():
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                a, b = i + di, j + dj
                if not (0 <= a < N and 0 <= b < N):
                    a, b = i, j                  # cancelled move
                new[(a, b)] = new.get((a, b), 0) + c
        out.append(new.pop(tgt, 0))
        cnt = new
    return out


for N in (8, 9, 10, 11):
    tmax = 14 * N * N // 4 + 200
    f = pmf_counts(N, tmax)                    # f[t] = 4^t P(T = t)
    P = [Fraction(f[t], 4 ** t) for t in range(tmax + 1)]
    tstar = max(range(1, tmax + 1), key=lambda t: P[t])
    diffs = [P[t + 1] - P[t] for t in range(1, tmax)]
    changes = sum(1 for a, b in zip(diffs, diffs[1:]) if (a > 0) != (b > 0) and a != 0 and b != 0)
    print('N=%d  global maximiser t*=%d  sign changes of P(T=t+1)-P(T=t) on [1,%d]: %d' % (N, tstar, tmax, changes))
    if changes > 1:
        for t in range(tstar - 2, tstar + 4):
            print('    P(T=%d) = %.11f' % (t, float(P[t])))
    sys.stdout.flush()
