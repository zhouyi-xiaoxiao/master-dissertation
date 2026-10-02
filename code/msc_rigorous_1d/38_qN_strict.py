"""Theorem 3.5 at the endpoint q = q_N (numerical sanity check of the strict pattern).

For 3 <= N <= 25, q = q_N = 1/(2 cos^2(pi/(2N-1))) (60 digits; then lambda_n = 0 up to 1e-58), and EVERY start x0,
computes f_{x0}(t), 1 <= t <= 8 N^2, by the recursion K_t = Q K_{t-1}, K_0 = e_n, f_x(t) = (q/2) K_{t-1}(x), and checks
   0 = f(1) = ... = f(d-1) < f(d) < ... < f(t*) >= f(t*+1) > f(t*+2) > ...   (t <= 8 N^2),
reporting the smallest relative gap |f(t+1) - f(t)| / f(t) among the strict inequalities (it must be far above 1e-50).
Output: data/38_qN_strict.json; log: logs/38_qN_strict.log.
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import os, json
import mpmath as mp

mp.mp.dps = 60
here = os.path.dirname(os.path.abspath(__file__))
rows = []
allok = True
worst_gap = mp.inf
for N in range(3, 26):
    n = N - 1
    q = 1 / (2 * mp.cos(mp.pi / (2 * N - 1)) ** 2)
    h = q / 2
    T = 8 * N * N
    K = [mp.mpf(0)] * n
    K[n - 1] = mp.mpf(1)
    F = [[] for _ in range(n)]
    for t in range(1, T + 1):
        for x in range(n):
            F[x].append(h * K[x])
        new = [mp.mpf(0)] * n
        for x in range(n):
            stay = (1 - h) if x == 0 else (1 - q)
            s = stay * K[x]
            if x > 0:
                s += h * K[x - 1]
            if x < n - 1:
                s += h * K[x + 1]
            new[x] = s
        K = new
    nties = 0
    for x0 in range(1, N):
        d = N - x0
        f = F[x0 - 1]
        ok = all(f[t - 1] == 0 or abs(f[t - 1]) < mp.mpf(10) ** -55 for t in range(1, d))
        ok &= f[d - 1] > 0
        t = d
        while t < T and f[t] > f[t - 1]:            # f(t+1) > f(t)
            worst_gap = min(worst_gap, (f[t] - f[t - 1]) / f[t - 1])
            t += 1
        tstar = t
        if t < T and f[t] == f[t - 1]:
            nties += 1
        t += 1
        while t < T:
            ok &= f[t] < f[t - 1]
            worst_gap = min(worst_gap, (f[t - 1] - f[t]) / f[t - 1])
            t += 1
        allok &= bool(ok)
        rows.append(dict(N=N, x0=x0, d=d, mode=tstar, ok=bool(ok)))
    print(f"N={N}: q_N={mp.nstr(q, 12)}  starts={N-1}  all strict patterns ok={all(r['ok'] for r in rows if r['N']==N)}",
          flush=True)
print("smallest relative gap among strict inequalities:", mp.nstr(worst_gap, 5))
print("ALL OK:", allok)
json.dump(dict(all_ok=allok, smallest_relative_gap=mp.nstr(worst_gap, 8), rows=rows),
          open(_os.path.join(_R, 'data', 'msc_rigorous_1d', '38_qN_strict.json'), "w"), indent=1)
