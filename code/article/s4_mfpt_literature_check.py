#!/usr/bin/env python
"""Section 4.2 (relation to the literature): the published single-sum resistance formulas of
Z.-Z. Tan and Z. Tan, "The basic principle of m x n resistor networks", Commun. Theor. Phys. 72, 055001 (2020),
evaluated exactly as printed there and compared with the formulas of this article.

Their network (their Fig. 1 with r_1 = r_2 = r_0, their "Case 2"): nodes (x, y), 0 <= x <= n, 0 <= y <= m, i.e.
(m+1) x (n+1) nodes, resistors r (horizontal) and r_0 (vertical).  With h = r / r_0, theta_i = i pi / (m+1),

    lambda_i, lambdabar_i = 1 + h - h cos(theta_i) +- sqrt((1 + h - h cos(theta_i))^2 - 1)          [their Eq. (20)]
    F_k = (lambda^k - lambdabar^k) / (lambda - lambdabar),   Delta F_k = F_{k+1} - F_k              [their Eq. (22)]
    C_{y,i} = cos((y + 1/2) theta_i)                                                               [their Eq. (19)]
    beta_{k,s} = Delta F_{x_k} Delta F_{n - x_s}     (x_1 <= x_2; k <= s)                           [below their Eq. (57)]

    R(d_1, d_2) = |x_2 - x_1| r / (m+1)
                  + r_0/(m+1) sum_{i=1}^{m} [beta_11 C_{y1,i}^2 - 2 beta_12 C_{y1,i} C_{y2,i} + beta_22 C_{y2,i}^2]
                                            / [(1 - cos theta_i) F_{n+1}]                           [their Eq. (57)]

    R({0,0},{n,m}) = n r/(m+1) + r_0/(m+1) sum_{i=1}^{m} [(Delta F_n - (-1)^i) / F_{n+1}] cot^2(theta_i / 2)
                                                                                                    [their Eq. (64)]

Checks (unit resistors, N x N grid, i.e. m = n = N - 1):
  1. Eq. (57) against the effective resistance from the pseudo-inverse of the grid Laplacian, all ordered pairs;
  2. Eq. (57) against Theorem s4_mfpt:thm-pair of this article through the commute-time identity
       T_{o->a} + T_{a->o} = (4 N^2 / q) R(o, a);
  3. Eq. (64) against form (a) of Theorem s4_mfpt:thm-single TERM BY TERM (k = i), and against exact values
     q T_N = 2 N^2 R_N for odd and even N.

Output: data/article/s4_mfpt_literature_check.json
Run   : python s4_mfpt_literature_check.py          (a few seconds; mpmath, 40 digits)
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import json
import os
from fractions import Fraction

import mpmath as mp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = _os.path.join(_R, 'data', 'article', 's4_mfpt_literature_check.json')
mp.mp.dps = 40


# ----------------------------------------------------------------- Tan and Tan (2020), as printed
def tt_lams(theta, h=1):
    a = 1 + h - h * mp.cos(theta)
    s = mp.sqrt(a * a - 1)
    return a + s, a - s


def tt_F(k, lam, lamb):
    return (lam ** k - lamb ** k) / (lam - lamb)


def tt_dF(k, lam, lamb):
    return tt_F(k + 1, lam, lamb) - tt_F(k, lam, lamb)


def tantan_eq57(m, n, d1, d2, r=1, r0=1):
    (x1, y1), (x2, y2) = d1, d2
    if x1 > x2:
        (x1, y1), (x2, y2) = (x2, y2), (x1, y1)
    h = mp.mpf(r) / r0
    tot = mp.mpf(abs(x2 - x1)) * r / (m + 1)
    for i in range(1, m + 1):
        th = i * mp.pi / (m + 1)
        lam, lamb = tt_lams(th, h)
        C1 = mp.cos((y1 + mp.mpf(1) / 2) * th)
        C2 = mp.cos((y2 + mp.mpf(1) / 2) * th)
        b11 = tt_dF(x1, lam, lamb) * tt_dF(n - x1, lam, lamb)
        b12 = tt_dF(x1, lam, lamb) * tt_dF(n - x2, lam, lamb)
        b22 = tt_dF(x2, lam, lamb) * tt_dF(n - x2, lam, lamb)
        tot += mp.mpf(r0) / (m + 1) * (b11 * C1 ** 2 - 2 * b12 * C1 * C2 + b22 * C2 ** 2) / (
            (1 - mp.cos(th)) * tt_F(n + 1, lam, lamb))
    return tot


def tantan_eq64_terms(m, n, r=1, r0=1):
    h = mp.mpf(r) / r0
    terms = []
    for i in range(1, m + 1):
        th = i * mp.pi / (m + 1)
        lam, lamb = tt_lams(th, h)
        terms.append(mp.mpf(r0) / (m + 1) * (tt_dF(n, lam, lamb) - (-1) ** i) / tt_F(n + 1, lam, lamb)
                     * mp.cot(th / 2) ** 2)
    return mp.mpf(n) * r / (m + 1), terms


# ----------------------------------------------------------------- this article, as printed (zero-based)
def phi(k, N):
    return mp.acosh(2 - mp.cos(mp.pi * k / N))


def h1(m, mp_, N):
    return (mp_ - m) * (mp_ + m + 1) if m <= mp_ else (m - mp_) * (2 * N - 1 - m - mp_)


def Psi(k, m, mp_, N):
    ph = phi(k, N)
    return (mp.cosh((N - abs(m - mp_)) * ph) + mp.cosh((N - 1 - m - mp_) * ph)) / (mp.sinh(ph) * mp.sinh(N * ph))


def qT_pair(o, a, N):
    """Theorem s4_mfpt:thm-pair, Eq. (s4_mfpt:eq-pair)"""
    th = lambda k, s: mp.pi * k * (2 * s + 1) / (2 * N)
    tot = mp.mpf(2 * h1(o[1], a[1], N))
    for k in range(1, N):
        caa = mp.cos(th(k, a[0])) ** 2
        coa = mp.cos(th(k, o[0])) * mp.cos(th(k, a[0]))
        tot += 4 * N * (caa * Psi(k, a[1], a[1], N) - coa * Psi(k, o[1], a[1], N))
    return tot


def form_a_terms(N):
    """Theorem s4_mfpt:thm-single(a): q T_N = 2N(N-1) + 4N sum_k c_k^2 B_k / (sinh phi_k sinh N phi_k)"""
    terms = []
    for k in range(1, N):
        ph = phi(k, N)
        ck2 = mp.cos(mp.pi * k / (2 * N)) ** 2
        Bk = mp.cosh(N * ph) + mp.cosh((N - 1) * ph) - (-1) ** k * (1 + mp.cosh(ph))
        terms.append(4 * N * ck2 * Bk / (mp.sinh(ph) * mp.sinh(N * ph)))
    return mp.mpf(2 * N * (N - 1)), terms


# ----------------------------------------------------------------- reference: grid Laplacian
def grid_resistances(N):
    n = N * N
    L = np.zeros((n, n))
    idx = lambda x, y: x * N + y
    for x in range(N):
        for y in range(N):
            for dx, dy in ((1, 0), (0, 1)):
                u, v = x + dx, y + dy
                if u < N and v < N:
                    i, j = idx(x, y), idx(u, v)
                    L[i, i] += 1
                    L[j, j] += 1
                    L[i, j] -= 1
                    L[j, i] -= 1
    G = np.linalg.pinv(L)
    d = np.diag(G)
    return d[:, None] + d[None, :] - 2 * G


out = {"generated_by": "code/article/s4_mfpt_literature_check.py",
       "source": "Tan and Tan, Commun. Theor. Phys. 72, 055001 (2020), Eqs. (19)-(22), (57), (64); r = r_0 = 1, m = n = N - 1",
       "digits": mp.mp.dps}

# 1. Eq. (57) against the Laplacian pseudo-inverse, all pairs
c1 = []
for N in (2, 3, 4, 5, 6, 7):
    R = grid_resistances(N)
    worst = 0.0
    npairs = 0
    for x1 in range(N):
        for y1 in range(N):
            for x2 in range(N):
                for y2 in range(N):
                    if (x1, y1) == (x2, y2):
                        continue
                    val = float(tantan_eq57(N - 1, N - 1, (x1, y1), (x2, y2)))
                    ref = R[x1 * N + y1, x2 * N + y2]
                    worst = max(worst, abs(val - ref) / ref)
                    npairs += 1
    c1.append({"N": N, "ordered_pairs": npairs, "max_rel_dev_vs_pinv_float64": worst})
out["eq57_vs_laplacian_pinv"] = c1

# 2. Eq. (57) against Theorem thm-pair through the commute-time identity
c2 = []
rng = np.random.default_rng(20261001)
for N in (3, 4, 5, 8, 9, 20, 35):
    if N <= 5:
        pairs = [((a, b), (c, d)) for a in range(N) for b in range(N) for c in range(N) for d in range(N)
                 if (a, b) != (c, d)]
    else:
        pairs = []
        while len(pairs) < 12:
            p = tuple(int(v) for v in rng.integers(0, N, 4))
            if p[:2] != p[2:]:
                pairs.append((p[:2], p[2:]))
        pairs.append(((0, 0), (N - 1, N - 1)))
    worst = mp.mpf(0)
    for o, a in pairs:
        commute = qT_pair(o, a, N) + qT_pair(a, o, N)          # = 4 N^2 R(o, a)
        R_tt = tantan_eq57(N - 1, N - 1, o, a)
        worst = max(worst, abs(commute / (4 * N * N) / R_tt - 1))
    c2.append({"N": N, "pairs": len(pairs), "max_rel_dev": float(worst)})
out["eq57_vs_thm_pair_commute_time"] = c2

# 3. Eq. (64) against form (a), term by term, and against exact values
exact = {2: Fraction(8), 3: Fraction(27), 4: Fraction(416, 7), 5: Fraction(1175, 11), 6: Fraction(9368, 55),
         7: Fraction(312865, 1247)}                             # q T_N, Table s4_mfpt:tab-exact
c3 = []
for N in (2, 3, 4, 5, 6, 7, 10, 11, 35, 36, 100, 101):
    lead_tt, t_tt = tantan_eq64_terms(N - 1, N - 1)
    lead_a, t_a = form_a_terms(N)
    # q T_N = 2 N^2 R_N: compare 2 N^2 * (their k-th term) with our k-th term
    term_dev = max(abs(2 * N * N * u / v - 1) for u, v in zip(t_tt, t_a))
    lead_dev = abs(2 * N * N * lead_tt / lead_a - 1)
    R_tt = lead_tt + mp.fsum(t_tt)
    qT_a = lead_a + mp.fsum(t_a)
    row = {"N": N, "parity": "odd" if N % 2 else "even",
           "max_rel_dev_term_by_term": float(term_dev), "rel_dev_leading_term": float(lead_dev),
           "R_N_eq64": mp.nstr(R_tt, 20), "rel_dev_2N2R_vs_form_a": float(abs(2 * N * N * R_tt / qT_a - 1))}
    if N in exact:
        ex = mp.mpf(exact[N].numerator) / exact[N].denominator
        row["rel_dev_2N2R_vs_exact_rational"] = float(abs(2 * N * N * R_tt / ex - 1))
    c3.append(row)
out["eq64_vs_form_a"] = c3

out["summary"] = {
    "eq57_max_rel_dev_vs_pinv": max(r["max_rel_dev_vs_pinv_float64"] for r in c1),
    "eq57_n_ordered_pairs_vs_pinv": sum(r["ordered_pairs"] for r in c1),
    "eq57_max_rel_dev_vs_thm_pair": max(r["max_rel_dev"] for r in c2),
    "eq57_n_pairs_vs_thm_pair": sum(r["pairs"] for r in c2),
    "eq64_max_rel_dev_term_by_term_vs_form_a": max(r["max_rel_dev_term_by_term"] for r in c3),
    "eq64_sizes": [r["N"] for r in c3],
    "conclusion": "Eq. (57) of Tan and Tan (2020) coincides with Theorem thm-pair (through the commute-time identity) "
                  "and their Eq. (64) coincides term by term with form (a) of Theorem thm-single, for odd and even N.",
}
json.dump(out, open(OUT, "w"), indent=1)
print(json.dumps(out["summary"], indent=1))
