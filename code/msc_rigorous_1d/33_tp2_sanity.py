"""Sanity checks (exact arithmetic; not part of the proofs) for Section 3.7 (Lemmas 3.12-3.14, Theorems 3.15, 3.16):
 (1) Lemma 3.12: Cauchy-Binet for 2x2 minors of a product, and M_t(x,y) = sum_{u<v} (Q^(t-1))[x,y|u,v] M_1(u,v);
 (2) Lemma 3.14 (locality): for q = 4/5 and k in {10, 19}, the set of all nontrivial 2x2 minors of A^k for N = 97 and
     N = 131 equals the set for N_0 = 80 (as sets of integers), and Q^k is TP2 for N = 97, 131 (direct test, all minors);
 (3) Theorem 3.15 by brute force: q = 4/5, 5 <= N <= 40: f(t)^2 >= f(t-1) f(t+1) for all t <= 3 N^2 (exact rationals -> integers);
 (4) Theorem 3.16 by brute force: q = 17/20 and 9/10, 3 <= N <= 40: number of sign changes '- then +' of the increments
     f(t+1) - f(t), t <= 4 N^2 + 60 (exact), compared with the verdicts of script 31.
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import os, sys, json, random
from fractions import Fraction as Fr

random.seed(5)
here = os.path.dirname(os.path.abspath(__file__))
res = {}


def tridiag(n, qn, qd):
    A = [[0] * n for _ in range(n)]
    for i in range(n):
        A[i][i] = 2 * (qd - qn)
        if i > 0:
            A[i][i - 1] = qn
        if i < n - 1:
            A[i][i + 1] = qn
    A[0][0] = 2 * qd - qn
    return A


def matmul(X, Y):
    n = len(X)
    return [[sum(X[i][l] * Y[l][j] for l in range(n)) for j in range(n)] for i in range(n)]


def minor(P, x, y, z, w):
    return P[x][z] * P[y][w] - P[x][w] * P[y][z]


# (1)
ok1 = True
for trial in range(20):
    n = 6
    X = [[random.randint(-3, 5) for _ in range(n)] for _ in range(n)]
    Y = [[random.randint(-3, 5) for _ in range(n)] for _ in range(n)]
    XY = matmul(X, Y)
    for _ in range(30):
        x, y = sorted(random.sample(range(n), 2)); z, w = sorted(random.sample(range(n), 2))
        s = sum(minor(X, x, y, u, v) * minor(Y, u, v, z, w) for u in range(n) for v in range(u + 1, n))
        ok1 &= (s == minor(XY, x, y, z, w))
A = tridiag(7, 4, 5)
Pw = [[1 if i == j else 0 for j in range(7)] for i in range(7)]
K = [[1 if i == 6 else 0 for i in range(7)]]
for t in range(1, 9):
    K.append([sum(A[i][l] * K[-1][l] for l in range(7)) for i in range(7)])          # K_t = A^t e_n
    # M_t = C_2(A^(t-1)) M_1
    for x in range(7):
        for y in range(x + 1, 7):
            Mt = K[t][x] * K[t - 1][y] - K[t - 1][x] * K[t][y]
            s = sum(minor(Pw, x, y, u, v) * (K[1][u] * K[0][v] - K[0][u] * K[1][v]) for u in range(7) for v in range(u + 1, 7))
            ok1 &= (Mt == s)
    Pw = matmul(Pw, A)
res["lemma312_identities"] = ok1
print("(1) Cauchy-Binet and M_t = C_2(Q^(t-1)) M_1:", ok1)


# (2)
def power(A, k):
    n = len(A)
    P = [row[:] for row in A]
    for kk in range(2, k + 1):
        R = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(max(0, i - kk), min(n, i + kk + 1)):
                s = P[i][j] * A[j][j]
                if j > 0:
                    s += P[i][j - 1] * A[j - 1][j]
                if j < n - 1:
                    s += P[i][j + 1] * A[j + 1][j]
                R[i][j] = s
        P = R
    return P


def nontrivial_minors(P, k):
    n = len(P)
    vals = set(); neg = 0
    for x in range(n):
        for y in range(x + 1, min(n, x + 2 * k)):
            lo = max(0, y - k); hi = min(n - 1, x + k)
            for z in range(lo, hi):
                for w in range(z + 1, hi + 1):
                    if P[x][w] != 0 and P[y][z] != 0:
                        d = minor(P, x, y, z, w)
                        vals.add(d); neg += d < 0
    return vals, neg


ok2 = True
for k in (10, 19):
    ref, neg0 = nontrivial_minors(power(tridiag(79, 4, 5), k), k)
    for N in (97, 131):
        vals, neg = nontrivial_minors(power(tridiag(N - 1, 4, 5), k), k)
        ok2 &= (vals == ref) and neg == 0 and neg0 == 0
res["locality_sets_equal_and_tp2"] = ok2
print("(2) locality: sets of nontrivial minors for N = 97, 131 equal those for N = 80 (k = 10, 19), all >= 0:", ok2)


# (3), (4)
def pmf_int(N, qn, qd, T):
    """integers F[t] = f(t) * (2 qd)^T * (2qd/qn)... : returns g[t] with f(t) = (qn/(2qd)) g[t] / (2qd)^(t-1); and scaled values
    h[t] = g[t] * (2qd)^(T - t) (common scaling), t = 1..T."""
    n = N - 1
    v = [0] * n; v[0] = 1
    a = 2 * (qd - qn); b = qn
    g = [0] * (T + 1)
    for t in range(1, T + 1):
        g[t] = v[n - 1]
        w = [a * x for x in v]
        for j in range(1, n):
            w[j] += b * v[j - 1]
        for j in range(0, n - 1):
            w[j] += b * v[j + 1]
        w[0] += b * v[0]
        v = w
    return [0] + [g[t] * (2 * qd) ** (T - t) for t in range(1, T + 1)]


ok3 = True
for N in range(5, 41):
    T = 3 * N * N
    h = pmf_int(N, 4, 5, T)
    ok3 &= all(h[t] * h[t] >= h[t - 1] * h[t + 1] for t in range(2, T))
res["q45_logconcave_bruteforce_N5_40"] = ok3
print("(3) q = 4/5: f log-concave for 5 <= N <= 40 (t <= 3 N^2, exact):", ok3)

ok4 = True
detail = {}
for qn, qd in ((17, 20), (9, 10)):
    d31 = {r["N"]: r["verdict"] for r in json.load(open(os.path.join(_R, 'data', 'msc_rigorous_1d', f"31_tp2_powers_q{qn}_{qd}.json")))["rows"]}
    notuni = []
    for N in range(3, 41):
        T = 4 * N * N + 60
        h = pmf_int(N, qn, qd, T)
        seen_minus = False; bad = False
        for t in range(1, T):
            d = h[t + 1] - h[t]
            if d < 0:
                seen_minus = True
            elif d > 0 and seen_minus:
                bad = True; break
        if bad:
            notuni.append(N)
        verdict = d31[N]
        consistent = (bad == (verdict == "NOT unimodal")) or verdict.startswith("undetermined")
        ok4 &= consistent
    detail[f"{qn}/{qd}"] = notuni
res["bruteforce_not_unimodal_N_le_40"] = detail; res["verdicts_consistent"] = ok4
print("(4) brute force: not unimodal for N in", detail, "; consistent with script 31:", ok4)
res["ALL_OK"] = ok1 and ok2 and ok3 and ok4
print("ALL OK:", res["ALL_OK"])
json.dump(res, open(_os.path.join(_R, 'data', 'msc_rigorous_1d', '33_tp2_sanity.json'), "w"), indent=1)
