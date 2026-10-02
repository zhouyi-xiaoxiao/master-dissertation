"""Proposition 8.8 (discrete time, general start, q > q_N): exact checks of the failure of unimodality.

(a) Symbolic check (sympy, q a symbol), for 3 <= N <= 9 and every start 2 <= x0 <= N-1 (d = N - x0), of the three
    first nonzero values used in Proposition 8.8:
      f(d) = (q/2)^d,  f(d+1) = (q/2)^d d (1-q),  f(d+2) = (q/2)^d [ d(d+1)/2 (1-q)^2 + d q^2/4 ],
    and of the factorisation  4 (f(d+2) - f(d+1)) / (q/2)^d = d [ (2d+3) r^2 - 6 r + 1 ],  r = 1 - q.
(b) The exact examples quoted in the text (Fractions): N=3, x0=2, q=9/10; N=4, x0=2, q=4/5; N=5, x0=2, q=7/10;
    N=6, x0=3, q=7/10.  A failure certificate is a triple j < k < l with f(k) < min(f(j), f(l)), which proves that
    the sequence is not unimodal (no tail argument is needed for a negative statement).
(c) Exact scan: 3 <= N <= 13, 2 <= x0 <= N-1, q in {3/5, 2/3, 3/4, 4/5, 9/10, 1}; t <= 6 N^2; counts the pairs
    with a failure certificate (only failures are certified; absence of a certificate in the finite horizon proves
    nothing and is reported as 'no certificate found').
Output: data/36_start_failure.json, log: logs/36_start_failure.log.
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import os, json
from fractions import Fraction as Fr
import sympy as sp

here = os.path.dirname(os.path.abspath(__file__))


def pmf(N, x0, q, tmax):
    """exact PMF f(1..tmax) of the lazy walk on {1..N}, reflecting at 1, absorbing at N, start x0."""
    n = N - 1
    p = [Fr(0)] * (n + 2)
    p[x0] = Fr(1)
    h = q / 2
    out = []
    for _ in range(tmax):
        new = [Fr(0)] * (n + 2)
        for x in range(1, n + 1):
            if p[x] == 0:
                continue
            stay = (1 - h) if x == 1 else (1 - q)
            new[x] += p[x] * stay
            if x > 1:
                new[x - 1] += p[x] * h
            new[x + 1] += p[x] * h          # x + 1 = N is absorption
        out.append(new[N] if N == n + 1 else Fr(0))
        new[N] = Fr(0)
        p = new
    return out


def failure_certificate(f):
    """return (j,k,l) with j<k<l and f[k] < min(f[j], f[l]) (1-based times), or None."""
    m = len(f)
    # running max from the left, then look for a later larger value after a dip
    best_left = 0
    for k in range(1, m - 1):
        if f[best_left] < f[k - 1]:
            best_left = k - 1
        if f[k] < f[best_left]:
            for l in range(k + 1, m):
                if f[l] > f[k]:
                    return (best_left + 1, k + 1, l + 1)
    return None


def symbolic_first_values():
    q = sp.symbols("q", positive=True)
    rows = []
    for N in range(3, 10):
        n = N - 1
        Q = sp.zeros(n, n)
        for j in range(n):
            Q[j, j] = 1 - q / 2 if j == 0 else 1 - q
            if j > 0:
                Q[j, j - 1] = q / 2
            if j < n - 1:
                Q[j, j + 1] = q / 2
        Ps = [sp.eye(n)]
        for t in range(1, n + 2):
            Ps.append(sp.expand(Ps[-1] * Q))
        for x0 in range(2, N):
            d = N - x0
            f = lambda t: sp.expand(q / 2 * Ps[t - 1][x0 - 1, n - 1])
            r = 1 - q
            base = (q / 2) ** d
            ok = all(sp.expand(f(t)) == 0 for t in range(1, d))
            ok &= sp.expand(f(d) - base) == 0
            ok &= sp.expand(f(d + 1) - base * d * r) == 0
            ok &= sp.expand(f(d + 2) - base * (sp.Rational(d * (d + 1), 2) * r ** 2 + d * q ** 2 / 4)) == 0
            ok &= sp.expand(4 * (f(d + 2) - f(d + 1)) - base * d * ((2 * d + 3) * r ** 2 - 6 * r + 1)) == 0
            rows.append(dict(N=N, x0=x0, d=d, ok=bool(ok)))
    return rows


def q_fail(d):
    """threshold of Proposition 8.8: f(d) > f(d+1) < f(d+2) for all q in (q_fail(d), 1]."""
    if d == 1:
        return 0.8
    if d == 2:
        return (4 + 2 ** 0.5) / 7
    return 1 - 1 / d


if __name__ == "__main__":
    res = {}
    res["symbolic"] = symbolic_first_values()
    print("(a) symbolic first values, all starts x0 >= 2, N <= 9:", all(r["ok"] for r in res["symbolic"]),
          len(res["symbolic"]), "pairs")
    ex = []
    for (N, x0, q) in [(3, 2, Fr(9, 10)), (4, 2, Fr(4, 5)), (5, 2, Fr(7, 10)), (6, 3, Fr(7, 10))]:
        f = pmf(N, x0, q, 6 * N * N)
        cert = failure_certificate(f)
        qN = 1 / (2 * float(__import__("math").cos(__import__("math").pi / (2 * N - 1))) ** 2)
        ex.append(dict(N=N, x0=x0, q=str(q), q_N=qN, first_values=[str(v) for v in f[:6]],
                       first_values_float=[float(v) for v in f[:6]], certificate=cert,
                       cert_values=[str(f[i - 1]) for i in cert] if cert else None))
        print("(b)", ex[-1])
    res["examples"] = ex
    scan = []
    nfail = 0
    nnone = 0
    for N in range(3, 14):
        for x0 in range(2, N):
            for q in [Fr(3, 5), Fr(2, 3), Fr(3, 4), Fr(4, 5), Fr(9, 10), Fr(1)]:
                f = pmf(N, x0, q, 6 * N * N)
                cert = failure_certificate(f)
                if cert:
                    nfail += 1
                    scan.append(dict(N=N, x0=x0, q=str(q), cert=cert))
                else:
                    nnone += 1
    res["scan"] = dict(range="3<=N<=13, 2<=x0<=N-1, q in {3/5,2/3,3/4,4/5,9/10,1}, t<=6N^2",
                       pairs_with_failure_certificate=nfail, pairs_without_certificate=nnone, failures=scan)
    print("(c) failure certificates:", nfail, " no certificate found:", nnone)
    print("(c) failures with d = N - x0 = 1 and q > 4/5:",
          sum(1 for r in scan if r["N"] - r["x0"] == 1 and Fr(r["q"]) > Fr(4, 5)),
          "of", sum(1 for N in range(3, 14) for q in [Fr(9, 10), Fr(1)]))
    print("(c) smallest q with a failure certificate:", min(Fr(r["q"]) for r in scan))
    expl = [r for r in scan if float(Fr(r["q"])) > q_fail(r["N"] - r["x0"])]
    pred = [(N, x0, str(q)) for N in range(3, 14) for x0 in range(2, N) for q in
            [Fr(3, 5), Fr(2, 3), Fr(3, 4), Fr(4, 5), Fr(9, 10), Fr(1)] if float(q) > q_fail(N - x0)]
    found = set((r["N"], r["x0"], r["q"]) for r in scan)
    res["scan"]["predicted_by_prop"] = len(pred)
    res["scan"]["predicted_and_certified"] = sum(1 for p in pred if p in found)
    res["scan"]["certified_not_predicted"] = [r for r in scan if float(Fr(r["q"])) <= q_fail(r["N"] - r["x0"])]
    print("(c) pairs predicted by Prop. 8.8:", len(pred), " of these certified:", res["scan"]["predicted_and_certified"],
          " failures not predicted:", res["scan"]["certified_not_predicted"])
    json.dump(res, open(_os.path.join(_R, 'data', 'msc_rigorous_1d', '36_start_failure.json'), "w"), indent=1)
    allok = (all(r["ok"] for r in res["symbolic"]) and all(e["certificate"] for e in ex)
             and res["scan"]["predicted_and_certified"] == len(pred))
    print("ALL OK:", allok)
