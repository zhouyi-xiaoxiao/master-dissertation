"""
Checks behind the tables and the quoted numbers of Section 4 (s4_mfpt).

Everything is evaluated in the ZERO-BASED form printed in the article (sites 0..N-1), with
code written for this file only (no import from the research scripts), and compared with
the stored research results where these exist.

  1. exact rational solves of (I-Q)m = 1 for N = 2..12 (banded elimination in Fractions)
     against forms (a), (b), (c) and the cosine double sum at 50 digits        -> tab-exact
  2. the pair formula (thm-pair) and h_1 against exact rational solves, all ordered pairs, N = 2..5
  3. closed-form coefficients C_2 .. C_{-8} to 30 digits                        -> tab-coeff
  4. truncation errors of the expansion at N = 10, 35, 100, 1000, 4001          -> tab-trunc
  5. double precision: first N at which form (a) is not finite; form (b) vs 40 digits
  6. small-k behaviour of phi_k
  7. X = mu_1 q T_N minus 2 pi ln N                                             -> eq-X2d
  8. three dimensions: double sum (prop-3d) against a sparse solve at small N; q T/N^3,
     two-term law and X = mu_1 q T for N up to 4096                             -> obs-3d, eq-X3d

Output: data/article/s4_mfpt_checks.json
Run:    python code/article/s4_mfpt_checks.py          (about 10 s)
"""
from __future__ import annotations
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository

import csv
import json
import os
import time
import warnings
from fractions import Fraction

import mpmath as mp
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla

HERE = os.path.dirname(os.path.abspath(__file__))
ART = _R
ROOT = _R
OUT = {}
t0 = time.time()


# ----------------------------------------------------------------------------- exact solves
def exact_hitting_times(N, target):
    """q*T_{x->target} for all x on {0..N-1}^2, exact rationals.
    (I - P) m = 1 off the target with I - P = (q/4)(D - A): solve (D - A)_restricted (q m) = 4."""
    idx = lambda x, y: x * N + y
    sites = [(x, y) for x in range(N) for y in range(N) if (x, y) != target]
    pos = {s: i for i, s in enumerate(sites)}
    n = len(sites)
    A = [dict() for _ in range(n)]
    b = [Fraction(4)] * n
    for (x, y), i in pos.items():
        deg = 0
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            u, v = x + dx, y + dy
            if 0 <= u < N and 0 <= v < N:
                deg += 1
                if (u, v) != target:
                    A[i][pos[(u, v)]] = Fraction(-1)
        A[i][i] = Fraction(deg)
    # Gaussian elimination on the sparse (banded) rows; the matrix is symmetric positive definite
    for i in range(n):
        piv = A[i][i]
        for j in [j for j in list(A[i].keys()) if j > i]:
            # eliminate column i from row j (symmetry: A[j][i] is present iff A[i][j] is)
            if i not in A[j]:
                continue
            f = A[j][i] / piv
            del A[j][i]
            for c, v in A[i].items():
                if c > i:
                    A[j][c] = A[j].get(c, Fraction(0)) - f * v
            b[j] -= f * b[i]
    m = [Fraction(0)] * n
    for i in range(n - 1, -1, -1):
        s = b[i] - sum(v * m[c] for c, v in A[i].items() if c > i)
        m[i] = s / A[i][i]
    res = {s: m[i] for s, i in pos.items()}
    res[target] = Fraction(0)
    return res


# ----------------------------------------------------------------------------- closed forms (mpmath)
def forms_mp(N):
    k = range(1, N)
    s = {i: mp.sin(mp.pi * i / (2 * N)) for i in range(0, N)}
    c = {i: mp.cos(mp.pi * i / (2 * N)) for i in range(0, N)}
    phi = {i: 2 * mp.asinh(s[i]) for i in k}
    # (a)
    a = 2 * N * (N - 1)
    for i in k:
        B = mp.cosh(N * phi[i]) + mp.cosh((N - 1) * phi[i]) - (-1) ** i * (1 + mp.cosh(phi[i]))
        a += 4 * N * c[i] ** 2 * B / (mp.sinh(phi[i]) * mp.sinh(N * phi[i]))
    # (b)
    b = mp.mpf(0)
    for i in k:
        g = mp.coth(N * mp.asinh(s[i])) if i % 2 else mp.tanh(N * mp.asinh(s[i]))
        b += c[i] ** 2 * mp.sqrt(1 + s[i] ** 2) / s[i] * g
    b *= 4 * N
    # (c)
    So = sum(c[i] ** 2 * mp.coth(phi[i] / 2) * mp.coth(N * phi[i] / 2) for i in k if i % 2)
    Se = sum((c[i] ** 2 * mp.coth(phi[i] / 2) * mp.tanh(N * phi[i] / 2) for i in k if i % 2 == 0), mp.mpf(0))
    # cosine double sum (eq-cosine-cc, d = 2)
    al = lambda i: 1 if i == 0 else 2
    d = mp.mpf(0)
    for i in range(N):
        for j in range(N):
            if (i + j) % 2:
                d += al(i) * al(j) * c[i] ** 2 * c[j] ** 2 / (s[i] ** 2 + s[j] ** 2)
    d *= 2
    return dict(a=a, b=b, c_odd=8 * N * So - 2 * N ** 2, c_even=2 * N ** 2 + 8 * N * Se, Sdiff=So - Se, double=d)


def form_b_mp(N):
    tot = mp.mpf(0)
    for i in range(1, N):
        y = mp.pi * i / (2 * N)
        si = mp.sin(y)
        x = N * mp.asinh(si)
        g = mp.coth(x) if i % 2 else mp.tanh(x)
        tot += mp.cos(y) ** 2 * mp.sqrt(1 + si ** 2) / si * g
    return 4 * N * tot


def h1(m, mp_, N):
    return (mp_ - m) * (mp_ + m + 1) if m <= mp_ else (m - mp_) * (2 * N - 1 - m - mp_)


def pair_mp(N, o, a):
    th = lambda k, m: mp.pi * k * (2 * m + 1) / (2 * N)
    tot = mp.mpf(0)
    for k in range(1, N):
        phi = 2 * mp.asinh(mp.sin(mp.pi * k / (2 * N)))
        Psi = lambda m, mm: (mp.cosh((N - abs(m - mm)) * phi) + mp.cosh((N - 1 - m - mm) * phi)) / (mp.sinh(phi) * mp.sinh(N * phi))
        Ck = lambda m, mm: mp.cos(th(k, m)) * mp.cos(th(k, mm))
        tot += Ck(a[0], a[0]) * Psi(a[1], a[1]) - Ck(o[0], a[0]) * Psi(o[1], a[1])
    return 2 * h1(o[1], a[1], N) + 4 * N * tot


# ----------------------------------------------------------------------------- 1. exact values
mp.mp.dps = 50
ref = {}
for r in csv.DictReader(open(_os.path.join(_R, 'data', 'msc_proof', 'verify_exact.csv'))):
    if r["q"] == "1":
        ref[int(r["N"])] = Fraction(int(r["exact_num"]), int(r["exact_den"]))
rows = []
worst = mp.mpf(0)
for N in range(2, 13):
    ex = exact_hitting_times(N, (N - 1, N - 1))[(0, 0)]
    f = forms_mp(N)
    exm = mp.mpf(ex.numerator) / ex.denominator
    rel = {k: float(abs(v - exm) / exm) for k, v in f.items() if k != "Sdiff"}
    worst = max(worst, max(rel.values()))
    rows.append(dict(N=N, qT_exact=f"{ex.numerator}/{ex.denominator}", qT_decimal=mp.nstr(exm, 12),
                     T_at_q_0p8=mp.nstr(exm / mp.mpf("0.8"), 12),
                     equals_research_value=(ex == ref[N]), rel_dev_forms=rel,
                     Sodd_minus_Seven_minus_N_over_2=float(f["Sdiff"] - mp.mpf(N) / 2)))
OUT["exact_corner"] = dict(wu2004_example4_R_4x4="13/7", two_N2_R_at_N4=str(2 * 16 * Fraction(13, 7)),
                           rows=rows, worst_rel_dev_any_form=float(worst),
                           all_equal_research_values=all(r["equals_research_value"] for r in rows))
print("1. exact values N=2..12: worst rel dev of any form", float(worst),
      "| equal to research fractions:", OUT["exact_corner"]["all_equal_research_values"], f"[{time.time()-t0:.1f}s]")

# ----------------------------------------------------------------------------- 2. pair formula
mp.mp.dps = 40
worst = 0.0; cases = 0; worst_1d = 0
for N in range(2, 6):
    for a in [(x, y) for x in range(N) for y in range(N)]:
        ex = exact_hitting_times(N, a)
        for o, v in ex.items():
            if o == a:
                continue
            val = pair_mp(N, o, a)
            exm = mp.mpf(v.numerator) / v.denominator
            worst = max(worst, float(abs(val - exm) / exm)); cases += 1
OUT["pair_formula"] = dict(N_range=[2, 5], ordered_pairs=cases, worst_rel_dev=worst)
# one-dimensional chain: q T_{m->m'} = h_1(m, m') (exact integers)
ok1d = True
for N in (2, 3, 7, 12):
    for a in range(N):
        # solve tridiagonal (I - T) h = 1 off the target, exact
        idxs = [m for m in range(N) if m != a]
        n = len(idxs); pos = {m: i for i, m in enumerate(idxs)}
        M = [[Fraction(0)] * n for _ in range(n)]; b = [Fraction(1)] * n
        for m in idxs:
            i = pos[m]; M[i][i] = Fraction(1)
            for mm in (m - 1, m + 1):
                if 0 <= mm < N:
                    if mm != a:
                        M[i][pos[mm]] -= Fraction(1, 2)
                else:
                    M[i][i] -= Fraction(1, 2)
        # dense exact elimination (n <= 11)
        for i in range(n):
            p = M[i][i]
            for j in range(i + 1, n):
                if M[j][i] != 0:
                    f = M[j][i] / p
                    for c in range(i, n):
                        M[j][c] -= f * M[i][c]
                    b[j] -= f * b[i]
        x = [Fraction(0)] * n
        for i in range(n - 1, -1, -1):
            x[i] = (b[i] - sum(M[i][c] * x[c] for c in range(i + 1, n))) / M[i][i]
        for m in idxs:
            ok1d &= (x[pos[m]] == h1(m, a, N))
OUT["one_dimension"] = dict(h1_equals_exact_solve=bool(ok1d), sizes=[2, 3, 7, 12])
print("2. pair formula:", OUT["pair_formula"], "| 1D h_1 exact:", ok1d, f"[{time.time()-t0:.1f}s]")

# ----------------------------------------------------------------------------- 3. coefficients
mp.mp.dps = 60
varpi = mp.gamma(mp.mpf(1) / 4) ** 2 / (2 * mp.sqrt(2 * mp.pi))
Xi = varpi ** 4 / mp.pi ** 2
Cf = {
    2: 8 / mp.pi * (mp.euler + mp.log(4 * mp.sqrt(2) / varpi) - mp.mpf(1) / 2 - mp.pi / 4),
    0: Xi / 9,
    -2: -mp.pi * Xi * (25 * Xi + 648) / 10800,
    -4: mp.pi ** 2 * Xi ** 2 * (1225 * Xi + 12879) / 1905120,
    -6: -mp.pi ** 3 * Xi ** 2 * (79625 * Xi ** 2 + 1215000 * Xi + 960498) / 435456000,
    -8: mp.pi ** 4 * Xi ** 3 * (7703465 * Xi ** 2 + 92129400 * Xi + 118943883) / 63228211200,
}
series = json.load(open(_os.path.join(_R, 'data', 'msc_proof', 'asymptotic_series.json')))
summ = json.load(open(_os.path.join(_R, 'data', 'msc_proof', 'asymptotics_summary.json')))
stored = {2: summ["C2"], 0: series["coefficients"]["1"]["closed_value"], -2: series["coefficients"]["2"]["closed_value"],
          -4: series["coefficients"]["3"]["closed_value"], -6: series["coefficients"]["4"]["closed_value"],
          -8: series["coefficients"]["5"]["closed_value"]}
OUT["coefficients"] = dict(varpi=mp.nstr(varpi, 25), Xi=mp.nstr(Xi, 25),
                           values={str(k): mp.nstr(v, 25) for k, v in Cf.items()},
                           abs_diff_from_stored={str(k): float(abs(Cf[k] - mp.mpf(stored[k]))) for k in Cf},
                           resistance_halves={str(k): mp.nstr(Cf[k] / 2, 20) for k in (2, 0, -2)})
print("3. coefficients:", OUT["coefficients"]["values"])

# ----------------------------------------------------------------------------- 4. truncation errors
def trunc(N, terms):
    v = 8 / mp.pi * N ** 2 * mp.log(N)
    powers = [2, 0, -2, -4, -6, -8]
    for j in range(terms - 1):
        v += Cf[powers[j]] * mp.mpf(N) ** powers[j]
    return v


ref_tr = {int(r["N"]): r for r in csv.DictReader(open(_os.path.join(_R, 'data', 'msc_proof', 'asymptotic_truncation_errors.csv')))}
tr = []; worst = 0.0
for N in (10, 35, 100, 1000, 4001):
    T = form_b_mp(N)
    e = [float(abs(trunc(N, t) - T) / T) for t in range(1, 6)]
    for t, key in zip(range(5), ["rel_err_1term", "rel_err_2terms", "rel_err_3terms", "rel_err_4terms", "rel_err_5terms"]):
        worst = max(worst, abs(e[t] - float(ref_tr[N][key])) / e[t])
    tr.append(dict(N=N, qT=mp.nstr(T, 25), rel_err_terms_1_to_5=e))
OUT["truncation"] = dict(rows=tr, worst_rel_diff_from_research_table=worst)
print("4. truncation errors: worst rel. diff. from research table", worst, f"[{time.time()-t0:.1f}s]")

# ----------------------------------------------------------------------------- 5. double precision
def form_a_f64(N):
    k = np.arange(1, N)
    c2 = np.cos(np.pi * k / (2 * N)) ** 2
    phi = np.arccosh(2 - np.cos(np.pi * k / N))
    B = np.cosh(N * phi) + np.cosh((N - 1) * phi) - (-1.0) ** k * (1 + np.cosh(phi))
    return 2 * N * (N - 1) + 4 * N * np.sum(c2 * B / (np.sinh(phi) * np.sinh(N * phi)))


def form_b_f64(N):
    k = np.arange(1, N)
    s = np.sin(np.pi * k / (2 * N))
    x = N * np.arcsinh(s)
    g = np.where(k % 2 == 1, 1 / np.tanh(x), np.tanh(x))
    return 4 * N * np.sum(np.cos(np.pi * k / (2 * N)) ** 2 * np.sqrt(1 + s * s) / s * g)


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    first_bad = next(N for N in range(2, 2000) if not np.isfinite(form_a_f64(N)))
    a402 = form_a_f64(402)
mp.mp.dps = 40
fb = {}
for N in (10, 100, 402, 403, 1000, 6000):
    T = form_b_mp(N)
    fb[str(N)] = float(abs(mp.mpf(float(form_b_f64(N))) - T) / T)
T402 = form_b_mp(402)
OUT["float64"] = dict(first_N_form_a_not_finite=int(first_bad),
                      form_a_rel_err_at_N402=float(abs(mp.mpf(float(a402)) - T402) / T402),
                      form_b_rel_err_vs_40_digits=fb, max_form_b_rel_err=max(fb.values()),
                      arcosh3=float(mp.acosh(3)), ln_DBL_MAX=float(np.log(np.finfo(float).max)),
                      N_times_phi_max_at_402_403=[float(402 * 2 * mp.asinh(mp.sin(mp.pi * 401 / 804))),
                                                 float(403 * 2 * mp.asinh(mp.sin(mp.pi * 402 / 806)))])
print("5. float64:", OUT["float64"], f"[{time.time()-t0:.1f}s]")

# ----------------------------------------------------------------------------- 6. phi_k for small k/N
mp.mp.dps = 30
ph = []
for N, k in ((100, 1), (1000, 1), (10000, 1), (1000, 50)):
    phi = 2 * mp.asinh(mp.sin(mp.pi * k / (2 * N)))
    ph.append(dict(N=N, k=k, phi_N_over_pi_k=mp.nstr(phi * N / (mp.pi * k), 15),
                   cubic_coefficient=mp.nstr((phi - mp.pi * k / N) / (mp.mpf(k) / N) ** 3, 10)))
OUT["phi_small_k"] = dict(rows=ph, cubic_coefficient_limit_minus_pi3_over_12=mp.nstr(-mp.pi ** 3 / 12, 10))
print("6. phi_k:", ph)

# ----------------------------------------------------------------------------- 7. X in two dimensions
mp.mp.dps = 40
x2 = []
for N in (10, 35, 100, 1000, 10000, 100000):
    T = form_b_mp(N)
    X = mp.sin(mp.pi / (2 * N)) ** 2 * T
    x2.append(dict(N=N, X=mp.nstr(X, 12), X_minus_2pi_lnN=mp.nstr(X - 2 * mp.pi * mp.log(N), 12)))
OUT["X_2d"] = dict(rows=x2, constant_pi2_C2_over_4=mp.nstr(mp.pi ** 2 * Cf[2] / 4, 15))
print("7. X (d=2):", OUT["X_2d"], f"[{time.time()-t0:.1f}s]")

# ----------------------------------------------------------------------------- 8. three dimensions
def qT3_double_sum(N):
    k = np.arange(N)
    s2 = np.sin(np.pi * k / (2 * N)) ** 2
    c2 = np.cos(np.pi * k / (2 * N)) ** 2
    al = np.where(k == 0, 1.0, 2.0)
    w = al * c2
    tot = 0.0
    for i in range(N):
        r2 = s2[i] + s2
        if i == 0:
            r2 = r2.copy(); r2[0] = 1.0                 # placeholder, (0,0) excluded below
        half = np.arcsinh(np.sqrt(r2))                   # varphi_{kj}/2
        x = N * half
        par = (i + k) % 2
        g = np.where(par == 1, 1 / np.tanh(x), np.tanh(x))
        term = w[i] * w * (1 / np.tanh(half)) * g
        if i == 0:
            term[0] = 0.0
        tot += float(np.sum(term))
    return 3 * N * (tot - N * (N - 1))


def qT3_direct(N):
    """Sparse solve of (I - Q) m = 1 on {0..N-1}^3 at q = 1, corner (0,0,0) -> (N-1,N-1,N-1)."""
    idx = lambda x, y, z: (x * N + y) * N + z
    n = N ** 3
    rows_, cols, vals = [], [], []
    for x in range(N):
        for y in range(N):
            for z in range(N):
                i = idx(x, y, z); stay = 0.0
                for d in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
                    u, v, w_ = x + d[0], y + d[1], z + d[2]
                    if 0 <= u < N and 0 <= v < N and 0 <= w_ < N:
                        rows_.append(i); cols.append(idx(u, v, w_)); vals.append(1 / 6)
                    else:
                        stay += 1 / 6
                rows_.append(i); cols.append(i); vals.append(stay)
    P = sps.csr_matrix((vals, (rows_, cols)), shape=(n, n))
    keep = np.arange(n - 1)                               # target is the last index
    A = (sps.identity(n - 1, format="csc") - P[keep][:, keep]).tocsc()
    return float(spla.spsolve(A, np.ones(n - 1))[0])


const = json.load(open(_os.path.join(_R, 'data', 'msc_modes_verify', 'constants.json')))
extra = json.load(open(_os.path.join(_R, 'data', 'msc_modes_verify', 'extra.json')))
K3 = float(const["G3_000"]) + 3 * float(const["G3_100"]) + 3 * float(const["G3_110"]) + float(const["G3_111"])
K3p = float(extra["C3prime_closed_form"])
small = []
for N in (2, 3, 5, 8, 12):
    d, s_ = qT3_direct(N), qT3_double_sum(N)
    small.append(dict(N=N, qT_direct=d, qT_double_sum=s_, rel_dev=abs(d - s_) / d))
big = []
for N in (10, 35, 100, 400, 1000, 4096):
    T = qT3_double_sum(N)
    X = (2 / 3) * np.sin(np.pi / (2 * N)) ** 2 * T
    big.append(dict(N=N, qT=T, qT_over_N3=T / N ** 3,
                    rel_dev_two_term=(K3 * N ** 3 + K3p * N ** 2 - T) / T,
                    remainder_after_two_terms=T - K3 * N ** 3 - K3p * N ** 2,
                    X=X, X_minus_linear_law=X - (np.pi ** 2 / 6) * (K3 * N + K3p)))
OUT["three_dimensions"] = dict(K3_sum_of_Green_values=K3, K3_prime_closed_form=K3p,
                               K3_stored=const["C3_from_capacity_of_2x2x2_block"],
                               pi2_K3_over_6=np.pi ** 2 * K3 / 6, pi2_K3prime_over_6=np.pi ** 2 * K3p / 6,
                               double_sum_vs_direct=small, large_N=big)
print("8. 3D:", json.dumps(OUT["three_dimensions"], indent=1), f"[{time.time()-t0:.1f}s]")

OUT["seconds"] = round(time.time() - t0, 1)
json.dump(OUT, open(_os.path.join(_R, 'data', 'article', 's4_mfpt_checks.json'), "w"), indent=1)
print("wrote data/article/s4_mfpt_checks.json")
