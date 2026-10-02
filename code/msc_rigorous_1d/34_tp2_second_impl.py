"""Re-check of Theorem 3.15(1) with a second, separately written implementation (different algorithm; exact integers;
same project, same machine).

For q = 4/5 and N in {4, 5, 9, 13, 14, 37, 80}: A = 10 Q as a dense numpy object matrix (Python ints), A^k by
numpy.linalg.matrix_power (exact for object dtype), and for every pair of rows x < y the vector of ratios is not used;
instead ALL 2x2 minors are formed at once as  outer(P[x], P[y]) - outer(P[x], P[y]).T  (entries [z, w] = P[x,z] P[y,w] -
P[x,w] P[y,z]) and the upper triangle (z < w) is tested for negativity.   k = 8, ..., 19.
Also: f log-concave for N in {5, 9, 13, 14, 37} by a direct exact computation of f(t), t <= 3 N^2, from A^t (without using M_t).
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import os, sys, json
import numpy as np

res = {}
def Amat(N):
    n = N - 1
    A = np.zeros((n, n), dtype=object)
    for i in range(n):
        A[i, i] = 2
        if i > 0:
            A[i, i - 1] = 4
        if i < n - 1:
            A[i, i + 1] = 4
    A[0, 0] = 6
    return A

allok = True
for N in (4, 5, 9, 13, 14, 37, 80):
    A = Amat(N); n = N - 1
    iu = np.triu_indices(n, k=1)
    neg_by_k = {}
    P = np.linalg.matrix_power(A, 7)
    for k in range(8, 20):
        P = P.dot(A)
        assert (P == np.linalg.matrix_power(A, k)).all() if k in (8, 19) else True
        neg = 0
        for x in range(n):
            for y in range(x + 1, n):
                O = np.outer(P[x], P[y])
                D = O - O.T
                neg += int((D[iu] < 0).sum())
        neg_by_k[k] = neg
    ok = all(neg_by_k[k] == 0 for k in range(10, 20))
    allok &= ok
    res[str(N)] = neg_by_k
    print("N =", N, "negative 2x2 minors of A^k, k = 8..19:", [neg_by_k[k] for k in range(8, 20)], "TP2 for 10 <= k <= 19:", ok, flush=True)

lc = True
for N in (5, 9, 13, 14, 37):
    A = Amat(N); n = N - 1
    T = 3 * N * N
    v = np.zeros(n, dtype=object); v[0] = 1
    g = []
    for t in range(1, T + 1):
        g.append(int(v[n - 1]))          # f(t) = (q/2) g_t / 10^(t-1)
        v = v.dot(A)
    h = [g[t - 1] * 10 ** (T - t) for t in range(1, T + 1)]
    lc &= all(h[i] * h[i] >= h[i - 1] * h[i + 1] for i in range(1, T - 1))
print("f log-concave (t <= 3N^2) for N in (5, 9, 13, 14, 37):", lc)
res["logconcave"] = lc; res["ALL_OK"] = bool(allok and lc)
print("ALL OK:", res["ALL_OK"])
json.dump(res, open(_os.path.join(_R, 'data', 'msc_rigorous_1d', '34_tp2_second_impl.json'), "w"), indent=1)
