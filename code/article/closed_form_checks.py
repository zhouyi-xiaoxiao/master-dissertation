#!/usr/bin/env python
"""Consistency checks of the closed forms used in the article.

1. Closed-form mode approximation: accuracy of the two variants that occur in the
   analysis (with and without the cos^2(pi/2N) factor) against the exact
   continuous-time modes stored in data/msc_modes/laplace_modes.jsonl.
2. The MFPT formulas of Section 4 and Supplementary Section S3 in their zero-based form
   (sites 0..N-1, as in the article): general-d cosine sum, single sum for an arbitrary
   pair in d = 2, overflow-free corner-to-corner single sum, and the one-index
   reduction in d = 3, each against a direct sparse solve of (I - Q) m = 1.
3. Closed-form first-passage PMF of the one-dimensional chain against time stepping.
4. Simplified (overflow-free) corner-to-corner double sum in d = 3.
5. Closed-form resolvent of the one-dimensional reflecting chain against a dense inverse.

Output: ../data/closed_form_checks.json
Usage:  python closed_form_checks.py
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import json
import math
import os
import itertools

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = _R          # 
MODES = _os.path.join(_R, 'data', 'msc_modes', 'laplace_modes.jsonl')
OUT = _os.path.join(_R, 'data', 'article', 'closed_form_checks.json')

out = {}

# ---------------------------------------------------------------------------
# 1. closed-form mode approximation, two variants
# ---------------------------------------------------------------------------
rows = [json.loads(l) for l in open(MODES)]
C2_2D = (8 / math.pi) * (0.5772156649015329 + 4 * math.log(2) + 0.5 * math.log(math.pi)
                         - 2 * math.lgamma(0.25)) - 2 - 4 / math.pi
out["c2_closed_form"] = C2_2D


def variants(r):
    d, N, T, t_exact = r["d"], r["N"], r["mfpt"], r["mode"]
    mu = r["p"][0]                      # slowest reflecting rate seen by the target (unit rate)
    W = r["W"][0]
    A = r["A_u"][0]                     # amplitude in u(t) ~ 1 - A exp(-mu t)
    X = mu * T
    res = {"N": N, "X": X}
    # derived first-order form with the exact weights (W = A = 2d cos^2(pi/2N) for CC)
    res["L1_weights"] = T * math.log(A * X) / (X + W - 1) / t_exact - 1
    # simplified form, W = A = 2d
    res["L1_plain"] = T * math.log(2 * d * X) / (X + 2 * d - 1) / t_exact - 1
    res["L0"] = math.log(A * X) / mu / t_exact - 1
    res["stored_L1"] = r["pred_L1"] / t_exact - 1
    return res


for d in (2, 3):
    for geo in ("CC", "C2M"):
        sel = sorted((r for r in rows if r["d"] == d and r["geo"] == geo), key=lambda r: r["N"])
        v = [variants(r) for r in sel]
        key = f"{d}d_{geo}"
        o = {"n_sizes": len(v), "N_min": v[0]["N"], "N_max": v[-1]["N"]}
        for name in ("L1_weights", "L1_plain", "L0", "stored_L1"):
            big = [(abs(x[name]), x["N"], x[name]) for x in v if x["N"] >= 10]
            m = max(big)
            o[name] = {"max_abs_rel_err_N>=10": m[0], "at_N": m[1], "signed": m[2],
                       "at_largest_N": v[-1][name]}
            for nmin in (35, 100):
                bb = [abs(x[name]) for x in v if x["N"] >= nmin]
                if bb:
                    o[name][f"max_abs_rel_err_N>={nmin}"] = max(bb)
        o["table"] = [{k: (x[k] if k == "N" else float(f"{x[k]:.6g}")) for k in
                       ("N", "X", "L0", "L1_weights", "L1_plain")} for x in v]
        out[key] = o

# 2D CC with the asymptotic X = 2 pi ln N + const and the two-term MFPT (no exact MFPT needed)
XCONST = (math.pi ** 2 / 4) * C2_2D
out["X_const_2d"] = XCONST
v = []
for r in sorted((r for r in rows if r["d"] == 2 and r["geo"] == "CC"), key=lambda r: r["N"]):
    N = r["N"]
    if N < 10:
        continue
    Tas = (8 / math.pi) * N * N * math.log(N) + C2_2D * N * N
    Xas = 2 * math.pi * math.log(N) + XCONST
    pred = Tas * math.log(4 * Xas) / (Xas + 3)
    v.append((abs(pred / r["mode"] - 1), N, pred / r["mode"] - 1))
out["2d_CC_fully_asymptotic_L1_plain"] = {"max_abs_rel_err_N>=10": max(v)[0], "at_N": max(v)[1],
                                         "at_largest_N": v[-1][2]}

# ---------------------------------------------------------------------------
# 2. zero-based MFPT formulas against direct solves
# ---------------------------------------------------------------------------

def build_P(N, d, q):
    """Lazy walk with cancelled moves on {0..N-1}^d."""
    n = N ** d
    idx = np.arange(n).reshape((N,) * d)
    rows_, cols_, vals_ = [], [], []
    diag = np.full(n, 1.0 - q)
    for ax in range(d):
        for sgn in (-1, 1):
            src = [slice(None)] * d
            dst = [slice(None)] * d
            if sgn == 1:
                src[ax] = slice(0, N - 1); dst[ax] = slice(1, N)
            else:
                src[ax] = slice(1, N); dst[ax] = slice(0, N - 1)
            s_ = idx[tuple(src)].ravel(); t_ = idx[tuple(dst)].ravel()
            rows_.append(s_); cols_.append(t_); vals_.append(np.full(s_.size, q / (2 * d)))
            # cancelled moves at the wall
            wall = [slice(None)] * d
            wall[ax] = N - 1 if sgn == 1 else 0
            diag[idx[tuple(wall)].ravel()] += q / (2 * d)
    P = sp.coo_matrix((np.concatenate(vals_), (np.concatenate(rows_), np.concatenate(cols_))), shape=(n, n)).tocsr()
    return P + sp.diags(diag), idx


def mfpt_solve(N, d, q, s, a):
    P, idx = build_P(N, d, q)
    n = N ** d
    ia, is_ = idx[tuple(a)], idx[tuple(s)]
    keep = np.array([i for i in range(n) if i != ia])
    Q = P[keep][:, keep]
    m = spla.spsolve((sp.identity(n - 1) - Q).tocsc(), np.ones(n - 1))
    pos = {k: j for j, k in enumerate(keep)}
    return m[pos[is_]] if is_ != ia else 0.0


def theta(k, m, N):          # zero-based
    return math.pi * k * (2 * m + 1) / (2 * N)


def cosine_sum(N, d, s, a):
    """q*T = (d/2) sum'_k prod alpha [prod cos^2 th(a_i) - prod cos th(s_i) cos th(a_i)] / sum_i s_{k_i}^2"""
    tot = 0.0
    for k in itertools.product(range(N), repeat=d):
        if all(ki == 0 for ki in k):
            continue
        al = np.prod([1.0 if ki == 0 else 2.0 for ki in k])
        ca = np.prod([math.cos(theta(ki, ai, N)) ** 2 for ki, ai in zip(k, a)])
        cs = np.prod([math.cos(theta(ki, si, N)) * math.cos(theta(ki, ai, N)) for ki, si, ai in zip(k, s, a)])
        den = sum(math.sin(math.pi * ki / (2 * N)) ** 2 for ki in k)
        tot += al * (ca - cs) / den
    return 0.5 * d * tot


def tau1(u, v, N):           # zero-based one-dimensional hitting time, q = 1
    return (v - u) * (v + u + 1) if u <= v else (u - v) * (2 * N - 1 - u - v)


def psi(phi, u, v, N):       # zero-based
    return (math.cosh((N - abs(u - v)) * phi) + math.cosh((N - 1 - u - v) * phi)) / (math.sinh(phi) * math.sinh(N * phi))


def single_sum_pair(N, s, a):
    tot = 2.0 * tau1(s[1], a[1], N)
    for k in range(1, N):
        phi = 2 * math.asinh(math.sin(math.pi * k / (2 * N)))
        Caa = math.cos(theta(k, a[0], N)) ** 2
        Csa = math.cos(theta(k, s[0], N)) * math.cos(theta(k, a[0], N))
        tot += 4 * N * (Caa * psi(phi, a[1], a[1], N) - Csa * psi(phi, s[1], a[1], N))
    return tot


def single_sum_corner(N):
    tot = 0.0
    for k in range(1, N):
        sk = math.sin(math.pi * k / (2 * N)); ck = math.cos(math.pi * k / (2 * N))
        g = 1 / math.tanh(N * math.asinh(sk)) if k % 2 else math.tanh(N * math.asinh(sk))
        tot += ck * ck * math.sqrt(1 + sk * sk) / sk * g
    return 4 * N * tot


def double_sum_corner_3d(N):
    """q*T for corner to opposite corner in d = 3 after summing one index with the ring identity.
    q*T = (3/2) sum' a_k a_j a_l c_k^2 c_j^2 c_l^2 [1-(-1)^{k+j+l}] / (s_k^2+s_j^2+s_l^2)
        = 3 sum_{k,j} a_k a_j c_k^2 c_j^2 G_{kj},
    G_{kj} = sum_l a_l c_l^2 [1-(-1)^{k+j+l}] / (sigma_{kj} - cos x_l),  sigma_{kj} = 3 - cos x_k - cos x_j,
    G_{00} = N(N-1) (one-dimensional hitting time),
    G_{kj} = N (1+sigma)[cosh N phi - (-1)^{k+j}] / (sinh phi sinh N phi) - N  otherwise (ring identity)."""
    tot = 0.0
    for k in range(N):
        for j in range(N):
            al = (1.0 if k == 0 else 2.0) * (1.0 if j == 0 else 2.0)
            ck2 = math.cos(math.pi * k / (2 * N)) ** 2; cj2 = math.cos(math.pi * j / (2 * N)) ** 2
            if k == 0 and j == 0:
                # sum_{l>=1} a_l c_l^2 [1-(-1)^l] / (1 - cos x_l) = N(N-1)
                inner = N * (N - 1)
            else:
                sig = 3 - math.cos(math.pi * k / N) - math.cos(math.pi * j / N)   # = 1 + 2 s_k^2 + 2 s_j^2
                phi = math.acosh(sig)
                par = -1.0 if (k + j) % 2 else 1.0
                # G = sum_l a_l c_l^2 [1 - par (-1)^l] / (sig - cos x_l)
                #   = N (1+sig) [cosh N phi - par] / (sinh phi sinh N phi) - N
                x = N * phi
                ratio = (1 - par * 2 * math.exp(-x) / (1 + math.exp(-2 * x))) / math.tanh(x) if x < 700 else 1.0
                inner = N * (1 + sig) * ratio / math.sinh(phi) - N
            tot += al * ck2 * cj2 * inner
    return 3.0 * tot


chk = []
rng = np.random.default_rng(20261001)
for N, q in ((4, 1.0), (7, 0.8), (12, 0.3)):
    for _ in range(6):
        s = tuple(int(x) for x in rng.integers(0, N, 2)); a = tuple(int(x) for x in rng.integers(0, N, 2))
        if s == a:
            continue
        ref = q * mfpt_solve(N, 2, q, s, a)
        chk.append({"d": 2, "N": N, "q": q, "s": s, "a": a, "qT_solve": ref,
                    "rel_cosine_sum": cosine_sum(N, 2, s, a) / ref - 1,
                    "rel_single_sum_pair": single_sum_pair(N, s, a) / ref - 1})
out["zero_based_pairs_d2"] = {"max_abs_rel_cosine_sum": max(abs(c["rel_cosine_sum"]) for c in chk),
                              "max_abs_rel_single_sum_pair": max(abs(c["rel_single_sum_pair"]) for c in chk),
                              "cases": len(chk)}
cc = []
for N in (2, 3, 5, 10, 35):
    ref = 0.8 * mfpt_solve(N, 2, 0.8, (0, 0), (N - 1, N - 1))
    cc.append({"N": N, "qT_solve": ref, "rel_single_sum_corner": single_sum_corner(N) / ref - 1,
               "rel_pair_formula": single_sum_pair(N, (0, 0), (N - 1, N - 1)) / ref - 1})
out["zero_based_corner_d2"] = cc
c1 = []
for N in (5, 30):
    ref = 0.8 * mfpt_solve(N, 1, 0.8, (0,), (N - 1,))
    c1.append({"N": N, "qT_solve": ref, "N(N-1)": N * (N - 1), "rel_cosine_sum": cosine_sum(N, 1, (0,), (N - 1,)) / ref - 1})
out["d1"] = c1
c3 = []
for N in (3, 5, 8, 12):
    ref = 0.8 * mfpt_solve(N, 3, 0.8, (0, 0, 0), (N - 1,) * 3)
    c3.append({"N": N, "qT_solve": ref,
               "rel_cosine_triple_sum": (cosine_sum(N, 3, (0, 0, 0), (N - 1,) * 3) / ref - 1) if N <= 8 else None,
               "rel_double_sum_reduction": double_sum_corner_3d(N) / ref - 1})
out["d3_corner"] = c3
s3 = (1, 0, 2); a3 = (3, 4, 1)
ref = 0.5 * mfpt_solve(5, 3, 0.5, s3, a3)
out["d3_general_pair"] = {"N": 5, "s": s3, "a": a3, "rel_cosine_sum": cosine_sum(5, 3, s3, a3) / ref - 1}
out["d3_large_N_double_sum"] = {str(N): double_sum_corner_3d(N) / N ** 3 for N in (50, 100, 400)}


# ---------------------------------------------------------------------------
# 3. one-dimensional closed-form PMF (reflecting end 0 -> absorbing end N-1)
#    f(t) = (2q/(2N-1)) sum_{m=1}^{N-1} (-1)^{m+1} cos(th_m/2) sin(th_m) lam_m^{t-1},
#    th_m = (2m-1) pi/(2N-1), lam_m = 1 - q + q cos(th_m)
# ---------------------------------------------------------------------------
def pmf_1d_closed(N, q, tmax):
    m = np.arange(1, N)
    th = (2 * m - 1) * math.pi / (2 * N - 1)
    lam = 1 - q + q * np.cos(th)
    w = (2 * q / (2 * N - 1)) * (-1.0) ** (m + 1) * np.cos(th / 2) * np.sin(th)
    t = np.arange(1, tmax + 1)
    return (w[None, :] * lam[None, :] ** (t[:, None] - 1)).sum(axis=1)


def pmf_1d_step(N, q, tmax):
    rho = np.zeros(N - 1); rho[0] = 1.0
    f = np.zeros(tmax)
    for t in range(tmax):
        new = (1 - q) * rho
        new[1:] += 0.5 * q * rho[:-1]
        new[:-1] += 0.5 * q * rho[1:]
        new[0] += 0.5 * q * rho[0]           # cancelled move at the reflecting end
        f[t] = 0.5 * q * rho[-1]             # flux into the target N-1
        rho = new
    return f


c1d = []
for N, q in ((5, 0.8), (20, 0.8), (50, 0.5), (100, 0.8)):
    tmax = 6 * N * N
    fc = pmf_1d_closed(N, q, tmax); fs = pmf_1d_step(N, q, tmax)
    c1d.append({"N": N, "q": q, "max_abs_diff": float(np.max(np.abs(fc - fs))),
                "argmax_closed": int(np.argmax(fc)) + 1, "argmax_step": int(np.argmax(fs)) + 1,
                "mean_truncated_at_6N2": float(np.sum(np.arange(1, tmax + 1) * fs)), "N(N-1)/q": N * (N - 1) / q})
out["d1_closed_form_pmf"] = c1d


# ---------------------------------------------------------------------------
# 4. simplified (overflow-free) corner-to-corner double sum in d = 3:
#    q T = 3N [ sum_{(k,j) != (0,0)} a_k a_j c_k^2 c_j^2 coth(phi_kj/2) g_kj - N(N-1) ],
#    cosh(phi_kj) = 3 - cos(pi k/N) - cos(pi j/N), g = coth(N phi/2) (k+j odd), tanh(N phi/2) (k+j even)
# ---------------------------------------------------------------------------
def double_sum_corner_3d_simplified(N):
    k = np.arange(N)
    al = np.where(k == 0, 1.0, 2.0)
    c2 = np.cos(np.pi * k / (2 * N)) ** 2
    s2 = np.sin(np.pi * k / (2 * N)) ** 2
    half = np.arcsinh(np.sqrt(s2[:, None] + s2[None, :]))          # phi_kj / 2, since sinh^2(phi/2) = s_k^2 + s_j^2
    half[0, 0] = 1.0                                               # placeholder, excluded below
    par_odd = ((k[:, None] + k[None, :]) % 2 == 1)
    th = np.tanh(N * half)
    g = np.where(par_odd, 1.0 / th, th)
    term = (al[:, None] * al[None, :]) * (c2[:, None] * c2[None, :]) / np.tanh(half) * g
    term[0, 0] = 0.0
    return 3.0 * N * (term.sum() - N * (N - 1))


out["d3_corner_simplified"] = [
    {"N": N, "rel_vs_unsimplified": double_sum_corner_3d_simplified(N) / double_sum_corner_3d(N) - 1}
    for N in (3, 5, 8, 12, 50, 100)]
out["d3_qT_over_N3_simplified"] = {str(N): double_sum_corner_3d_simplified(N) / N ** 3 for N in (100, 1000, 4096)}


# ---------------------------------------------------------------------------
# 5. resolvent of the one-dimensional reflecting chain (zero-based), used in the Laplace-space route:
#    [(sigma + a (I - T))^{-1}]_{n,n0} = cosh(phi(n< + 1/2)) cosh(phi(N - 1/2 - n>)) / ((a/2) sinh(phi) sinh(N phi)),
#    cosh(phi) = 1 + sigma/a
# ---------------------------------------------------------------------------
def resolvent_check(N=7, a=0.4):
    Tm = np.zeros((N, N))
    for m in range(N):
        for dm in (-1, 1):
            if 0 <= m + dm < N:
                Tm[m, m + dm] = 0.5
            else:
                Tm[m, m] += 0.5
    worst = 0.0
    for sig in (0.3, 2.0, 0.01 + 0.2j):
        R = np.linalg.inv(sig * np.eye(N) + a * (np.eye(N) - Tm))
        ph = np.arccosh(1 + sig / a)
        for n in range(N):
            for n0 in range(N):
                lo, hi = min(n, n0), max(n, n0)
                f = np.cosh(ph * (lo + 0.5)) * np.cosh(ph * (N - 0.5 - hi)) / ((a / 2) * np.sinh(ph) * np.sinh(N * ph))
                worst = max(worst, abs(f - R[n, n0]) / abs(R[n, n0]))
    return worst


out["resolvent_1d_max_rel_err"] = resolvent_check()

with open(OUT, "w") as f:
    json.dump(out, f, indent=1)
for k, v in out.items():
    if isinstance(v, dict) and "table" in v:
        v = {kk: vv for kk, vv in v.items() if kk != "table"}
    print(k, json.dumps(v)[:1500])
