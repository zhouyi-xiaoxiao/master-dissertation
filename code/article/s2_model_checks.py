#!/usr/bin/env python
"""Numerical checks of every statement of Section 2 (s2_model).

Each block tests one statement of the section against a direct dense or sparse
computation on small lattices; nothing here is used as a proof.

 1. model:       P of eq. (s2_model:eq-P) is symmetric and doubly stochastic;
                 I - P = q (I - P_1) = (q/2d) * (graph Laplacian);  the mirror-reflection
                 rule gives a non-symmetric matrix.
 2. structure:   (a) q T independent of q;  (b) PGF identity F~_q(z) = F^_1((1-z)/(qz));
                 (c) f(t) = sum_j q b_j (1 - q nu_j)^(t-1) against time stepping and
                     g(t) = sum_j b_j exp(-nu_j t) against the matrix exponential,
                     sum_j b_j/nu_j = 1, sum_j b_j/nu_j^2 = q T;
                 (d) smallest eigenvalue of Q is >= 1 - 2q (non-negative for q <= 1/2).
 3. renewal:     e_o^T (s + I - Q_1)^(-1) r_1 = P^(a,s|o) / P^(a,s|a) with the cosine propagator.
 4. interlacing: zeros of P^(a,s|a) = eigenvalues nu of I - Q_1 with E_nu r_1 != 0; exactly one
                 between consecutive distinct reflecting rates with non-zero weight at the target;
                 1/nu_0 = mean first-passage time from the quasi-stationary start.
 5. two spectra: reflecting gap q mu_1, killed gap q nu_0 and 1/T for d = 2, N = 35 and
                 d = 3, N = 40 (corner to corner, q = 0.8);  X = mu_1 q T.
 6. pinv:        T = n (L+_aa - L+_xa) = (2 d n / q)(G_aa - G_xa); commute-time identity;
                 T = lim (1 - F~(z))/(1 - z), whereas the reciprocal tends to 1/T.

Output: ../data/s2_model_checks.json
Usage:  python s2_model_checks.py        (about 20 s, < 1 GB)
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import itertools
import json
import math
import os

import numpy as np
import scipy.linalg as la
import scipy.optimize as opt
import scipy.sparse as sp
import scipy.sparse.linalg as spla

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = _os.path.join(_R, 'data', 'article', 's2_model_checks.json')
out = {}


# ---------------------------------------------------------------------------
# building blocks
# ---------------------------------------------------------------------------
def sites(N, d):
    return list(itertools.product(range(N), repeat=d))


def build_P(N, d, q, rule="cancel"):
    """Dense transition matrix. rule = 'cancel' (the model) or 'mirror' (bounce back)."""
    S = sites(N, d)
    idx = {x: i for i, x in enumerate(S)}
    P = np.zeros((len(S), len(S)))
    for x in S:
        i = idx[x]
        P[i, i] += 1.0 - q
        for ax in range(d):
            for sgn in (-1, 1):
                y = list(x)
                y[ax] += sgn
                if 0 <= y[ax] < N:
                    P[i, idx[tuple(y)]] += q / (2 * d)
                elif rule == "cancel":
                    P[i, i] += q / (2 * d)
                else:                                   # mirror: sent to the interior neighbour
                    y[ax] = x[ax] - sgn
                    P[i, idx[tuple(y)]] += q / (2 * d)
    return P, idx


def laplacian(N, d):
    S = sites(N, d)
    idx = {x: i for i, x in enumerate(S)}
    Lg = np.zeros((len(S), len(S)))
    for x in S:
        for ax in range(d):
            y = list(x)
            y[ax] += 1
            if y[ax] < N:
                i, j = idx[x], idx[tuple(y)]
                Lg[i, i] += 1
                Lg[j, j] += 1
                Lg[i, j] -= 1
                Lg[j, i] -= 1
    return Lg, idx


def killed(P, ia):
    keep = [i for i in range(P.shape[0]) if i != ia]
    Q = P[np.ix_(keep, keep)]
    r = P[keep, ia].copy()
    return Q, r, keep


def phi(N, d):
    """Orthonormal cosine eigenvectors (columns) and unit-rate reflecting rates mu_k."""
    S = sites(N, d)
    ks = sites(N, d)
    V = np.ones((len(S), len(ks)))
    mu = np.zeros(len(ks))
    for c, k in enumerate(ks):
        for ax in range(d):
            alpha = 1.0 if k[ax] == 0 else 2.0
            col = np.array([math.sqrt(alpha / N) * math.cos(math.pi * k[ax] * (2 * x[ax] + 1) / (2 * N))
                            for x in S])
            V[:, c] *= col
            mu[c] += (2.0 / d) * math.sin(math.pi * k[ax] / (2 * N)) ** 2
    return V, mu


def weighted_levels(mu, w, tol=1e-12):
    """Group equal rates; return distinct rates with summed weights, dropping zero weights."""
    order = np.argsort(mu)
    lev, wt = [], []
    for i in order:
        if lev and abs(mu[i] - lev[-1]) < tol:
            wt[-1] += w[i]
        else:
            lev.append(mu[i])
            wt.append(w[i])
    lev, wt = np.array(lev), np.array(wt)
    keepm = wt > 1e-14
    return lev[keepm], wt[keepm]


# ---------------------------------------------------------------------------
# 1. model
# ---------------------------------------------------------------------------
res = []
for d, N in ((1, 7), (2, 5), (2, 6), (3, 4)):
    for q in (0.3, 0.8, 1.0):
        P, idx = build_P(N, d, q)
        P1, _ = build_P(N, d, 1.0)
        Lg, _ = laplacian(N, d)
        I = np.eye(P.shape[0])
        Pm, _ = build_P(N, d, q, rule="mirror")
        res.append({
            "d": d, "N": N, "q": q,
            "symmetry_maxabs": float(np.abs(P - P.T).max()),
            "row_sum_err": float(np.abs(P.sum(1) - 1).max()),
            "col_sum_err": float(np.abs(P.sum(0) - 1).max()),
            "min_entry": float(P.min()),
            "I-P_minus_q(I-P1)_maxabs": float(np.abs((I - P) - q * (I - P1)).max()),
            "I-P_minus_(q/2d)Laplacian_maxabs": float(np.abs((I - P) - q / (2 * d) * Lg).max()),
            "irreducible": bool((np.linalg.matrix_power(P + I, P.shape[0]) > 0).all()),
            "mirror_rule_asymmetry_maxabs": float(np.abs(Pm - Pm.T).max()),
            "mirror_rule_col_sum_err": float(np.abs(Pm.sum(0) - 1).max()),
        })
out["model"] = res

# ---------------------------------------------------------------------------
# 2. structure
# ---------------------------------------------------------------------------
res = []
cases = [(1, 9, (0,), (8,)), (2, 6, (0, 0), (5, 5)), (2, 7, (0, 0), (3, 3)), (2, 6, (1, 4), (3, 2)),
         (3, 4, (0, 0, 0), (3, 3, 3)), (3, 5, (2, 2, 2), (4, 4, 4))]
for d, N, o, a in cases:
    P1, idx = build_P(N, d, 1.0)
    Q1, r1, keep = killed(P1, idx[a])
    io = keep.index(idx[o])
    m = Q1.shape[0]
    Im = np.eye(m)
    nu, U = la.eigh(Im - Q1)
    b = U[io, :] * (U.T @ r1)                      # residues (per eigenvector)
    T1 = float(np.linalg.solve(Im - Q1, np.ones(m))[io])
    rec = {"d": d, "N": N, "o": o, "a": a, "qT_unit_rate": T1,
           "sum_b_over_nu_minus_1": float(abs((b / nu).sum() - 1)),
           "sum_b_over_nu2_rel_err": float(abs((b / nu ** 2).sum() / T1 - 1)),
           "nu_min": float(nu.min()), "nu_max": float(nu.max())}
    # continuous-time density against the matrix exponential
    ts = np.array([0.3, 1.0, 5.0, 0.2 * T1, T1])
    g_spec = np.array([(b * np.exp(-nu * t)).sum() for t in ts])
    g_expm = np.array([la.expm(-(Im - Q1) * t)[io, :] @ r1 for t in ts])
    rec["g_spectral_vs_expm_maxabs"] = float(np.abs(g_spec - g_expm).max())
    per_q = []
    for q in (0.3, 0.5, 0.8, 1.0):
        P, _ = build_P(N, d, q)
        Q, r, _ = killed(P, idx[a])
        T = float(np.linalg.solve(Im - Q, np.ones(m))[io])
        # (c) time stepping against the spectral sum
        rho = np.zeros(m)
        rho[io] = 1.0
        nstep = 400
        f_step = np.empty(nstep)
        for t in range(nstep):
            f_step[t] = rho @ r
            rho = Q.T @ rho
        tt = np.arange(1, nstep + 1)
        f_spec = np.array([(q * b * (1 - q * nu) ** (t - 1)).sum() for t in tt])
        # (b) transform identity
        zd = []
        for z in (0.5, 0.9, 0.99, 0.6 + 0.3j, -0.4 + 0.2j):
            Ft = z * (np.linalg.solve(np.eye(m) - z * Q, r)[io])
            s = (1 - z) / (q * z)
            Fh = np.linalg.solve(s * Im + Im - Q1, r1)[io]
            zd.append(abs(Ft - Fh))
        eigQ = la.eigvalsh(Q)
        per_q.append({"q": q, "qT": q * T, "qT_rel_dev_from_unit_rate": abs(q * T / T1 - 1),
                      "r_minus_q_r1_maxabs": float(np.abs(r - q * r1).max()),
                      "pmf_spectral_vs_stepping_maxabs": float(np.abs(f_spec - f_step).max()),
                      "transform_identity_maxabs": float(max(zd)),
                      "min_eig_Q": float(eigQ.min()), "bound_1-2q": 1 - 2 * q,
                      "max_eig_Q": float(eigQ.max())})
    rec["per_q"] = per_q
    res.append(rec)
out["structure"] = res

# ---------------------------------------------------------------------------
# 3 + 4. renewal ratio, interlacing, quasi-stationary mean
# ---------------------------------------------------------------------------
res = []
cases = [(1, 9, (0,), (8,)), (2, 6, (0, 0), (5, 5)), (2, 7, (0, 0), (3, 3)), (2, 7, (3, 3), (6, 6)),
         (2, 6, (1, 4), (3, 2)), (3, 4, (0, 0, 0), (3, 3, 3)), (3, 5, (0, 0, 0), (2, 2, 2))]
for d, N, o, a in cases:
    P1, idx = build_P(N, d, 1.0)
    n = P1.shape[0]
    ia, io_full = idx[a], idx[o]
    Q1, r1, keep = killed(P1, ia)
    io = keep.index(io_full)
    m = n - 1
    Im = np.eye(m)
    V, mu = phi(N, d)
    # cosine vectors are eigenvectors of P_1
    eig_res = float(np.abs((np.eye(n) - P1) @ V - V * mu).max())
    # renewal ratio
    dev = []
    for s in (0.01, 0.3, 2.0, 0.2 + 0.7j):
        Fh = np.linalg.solve(s * Im + Im - Q1, r1)[io]
        Pao = (V[ia, :] * V[io_full, :] / (s + mu)).sum()
        Paa = (V[ia, :] ** 2 / (s + mu)).sum()
        dev.append(abs(Fh - Pao / Paa))
    # weighted reflecting levels at the target, zeros of P^(a,s|a)
    lev, wt = weighted_levels(mu, V[ia, :] ** 2)
    Paa_fun = lambda s: (wt / (s + lev)).sum()
    zeros = []
    for i in range(len(lev) - 1):
        lo, hi = -lev[i + 1], -lev[i]
        eps = 1e-13 * max(1.0, abs(lo))
        zeros.append(-opt.brentq(Paa_fun, lo + eps, hi - eps, xtol=1e-15, rtol=1e-15, maxiter=500))
    zeros = np.sort(np.array(zeros))
    # visible eigenvalues of I - Q_1: eigenprojection does not annihilate r_1
    nu, U = la.eigh(Im - Q1)
    order = np.argsort(nu)
    nu, U = nu[order], U[:, order]
    groups, cur = [], [0]
    for j in range(1, m):
        if abs(nu[j] - nu[cur[-1]]) < 1e-10:
            cur.append(j)
        else:
            groups.append(cur)
            cur = [j]
    groups.append(cur)
    visible, resid_o = [], []
    for gidx in groups:
        proj_r = U[:, gidx] @ (U[:, gidx].T @ r1)
        if np.linalg.norm(proj_r) > 1e-9:
            visible.append(nu[gidx].mean())
            resid_o.append(proj_r[io])
    visible = np.array(visible)
    same = len(visible) == len(zeros)
    # quasi-stationary start
    h = np.linalg.solve(Im - Q1, np.ones(m))
    v = np.abs(U[:, 0])
    res.append({
        "d": d, "N": N, "o": o, "a": a,
        "cosine_eigenvector_residual": eig_res,
        "renewal_ratio_maxabs": float(max(dev)),
        "n_weighted_reflecting_levels": int(len(lev)),
        "n_zeros_of_Paa": int(len(zeros)),
        "n_visible_killed_rates": int(len(visible)),
        "zeros_vs_visible_rates_max_rel": float(np.abs(zeros / visible - 1).max()) if same else None,
        "strict_interlacing": bool(np.all(lev[:-1] < zeros) and np.all(zeros < lev[1:])),
        "n_rates_with_nonzero_residue_for_this_start": int(np.sum(np.abs(resid_o) > 1e-10)),
        "nu0": float(nu[0]), "first_weighted_reflecting_rate": float(lev[1]),
        "mu1": float((2.0 / d) * math.sin(math.pi / (2 * N)) ** 2),
        "perron_vector_positive": bool(v.min() > 0),
        "inv_nu0_minus_quasistationary_mean_rel": float(abs((v @ h) / v.sum() * nu[0] - 1)),
        "min_max_mfpt_times_nu0": [float(h.min() * nu[0]), float(h.max() * nu[0])],
    })
out["renewal_interlacing"] = res

# ---------------------------------------------------------------------------
# 5. the two spectra at the sizes quoted in the text (corner to corner, q = 0.8)
# ---------------------------------------------------------------------------
res = []
q = 0.8
for d, N in ((2, 35), (3, 40)):
    k1 = np.arange(N)
    al = np.where(k1 == 0, 1.0, 2.0)
    c2 = np.cos(math.pi * k1 / (2 * N)) ** 2          # N * psi_k(a)^2 / alpha_k at the corner
    s2 = np.sin(math.pi * k1 / (2 * N)) ** 2
    sign = (-1.0) ** k1                              # psi_k(o) psi_k(a) = alpha_k c_k^2 (-1)^k / N
    if d == 2:
        W = np.multiply.outer(al * c2, al * c2)
        Sg = np.multiply.outer(sign, sign)
        MU = (2.0 / d) * np.add.outer(s2, s2)
    else:
        W = np.einsum("i,j,k->ijk", al * c2, al * c2, al * c2)
        Sg = np.einsum("i,j,k->ijk", sign, sign, sign)
        MU = (2.0 / d) * (s2[:, None, None] + s2[None, :, None] + s2[None, None, :])
    W, Sg, MU = W.ravel(), Sg.ravel(), MU.ravel()
    n = N ** d
    nz = MU > 0
    qT = float((W[nz] * (1 - Sg[nz]) / MU[nz]).sum())      # n (L+_aa - L+_oa) at unit rate
    mu1 = (2.0 / d) * math.sin(math.pi / (2 * N)) ** 2
    nPaa = lambda s: (W / (s + MU)).sum()                  # n * P^(a,s|a)
    nu0 = -opt.brentq(nPaa, -mu1 * (1 - 1e-12), -1e-14 * mu1, xtol=1e-300, rtol=1e-15, maxiter=500)
    rec = {"d": d, "N": N, "q": q, "qT": qT, "T": qT / q, "mu1": mu1, "X": mu1 * qT,
           "nu0_from_zero_of_Paa": nu0, "nu0_times_qT": nu0 * qT, "mu1_over_nu0": mu1 / nu0,
           "reflecting_gap_q_mu1": q * mu1, "killed_gap_q_nu0": q * nu0, "one_over_T": q / qT,
           "leading_form_pi2_over_2dN2": math.pi ** 2 / (2 * d * N ** 2)}
    # independent value of nu_0: smallest eigenvalue of the sparse matrix I - Q_1
    one = sp.identity(N, format="csr")
    Tm = sp.diags([0.5 * np.ones(N - 1), 0.5 * np.ones(N - 1)], [-1, 1], format="lil")
    Tm[0, 0] = 0.5
    Tm[N - 1, N - 1] = 0.5
    Tm = Tm.tocsr()
    P1 = None
    for ax in range(d):
        term = None
        for bx in range(d):
            f = Tm if bx == ax else one
            term = f if term is None else sp.kron(term, f, format="csr")
        P1 = term if P1 is None else P1 + term
    P1 = (P1 / d).tocsr()
    keep = np.arange(n - 1)                                  # target = last site (N-1,...,N-1)
    A = (sp.identity(n - 1, format="csc") - P1[keep][:, keep]).tocsc()
    lu = spla.splu(A)
    op = spla.LinearOperator(A.shape, matvec=lu.solve)
    val = spla.eigsh(op, k=1, which="LM", tol=1e-13)[0][0]   # largest eigenvalue of (I - Q_1)^(-1)
    rec["nu0_from_sparse_eigensolver"] = float(1.0 / val)
    rec["qT_from_sparse_solve"] = float(lu.solve(np.ones(n - 1))[0])
    res.append(rec)
out["two_spectra"] = res

# ---------------------------------------------------------------------------
# 6. pseudo-inverse, resistance, commute time, generating-function limit
# ---------------------------------------------------------------------------
res = []
for d, N, q in ((1, 8, 0.8), (2, 5, 0.8), (2, 6, 0.3), (3, 4, 1.0)):
    P, idx = build_P(N, d, q)
    n = P.shape[0]
    Lp = np.linalg.pinv(np.eye(n) - P, hermitian=True)
    Lg, _ = laplacian(N, d)
    G = np.linalg.pinv(Lg, hermitian=True)
    worst_pinv = worst_res = worst_comm = 0.0
    hit = {}
    for a in sites(N, d):
        ia = idx[a]
        Q, r, keep = killed(P, ia)
        h = np.linalg.solve(np.eye(n - 1) - Q, np.ones(n - 1))
        for j, i in enumerate(keep):
            hit[(i, ia)] = h[j]
            worst_pinv = max(worst_pinv, abs(n * (Lp[ia, ia] - Lp[i, ia]) / h[j] - 1))
            worst_res = max(worst_res, abs(2 * d * n / q * (G[ia, ia] - G[i, ia]) / h[j] - 1))
    for (i, j), hij in hit.items():
        R = G[i, i] + G[j, j] - 2 * G[i, j]
        worst_comm = max(worst_comm, abs((hij + hit[(j, i)]) / (2 * d * n / q * R) - 1))
    res.append({"d": d, "N": N, "q": q, "ordered_pairs": len(hit),
                "pinv_formula_max_rel": worst_pinv, "resistance_formula_max_rel": worst_res,
                "commute_identity_max_rel": worst_comm})
out["pinv_resistance_commute"] = res

d, N, q = 2, 20, 0.8
P, idx = build_P(N, d, q)
n = P.shape[0]
Q, r, keep = killed(P, idx[(N - 1, N - 1)])
io = keep.index(idx[(0, 0)])
T = float(np.linalg.solve(np.eye(n - 1) - Q, np.ones(n - 1))[io])
lim = []
for eps in (1e-4, 1e-6, 1e-8):
    z = 1 - eps
    Ft = z * np.linalg.solve(np.eye(n - 1) - z * Q, r)[io]
    lim.append({"1-z": eps, "(1-F)/(1-z)": float((1 - Ft) / eps), "(1-z)/(1-F)": float(eps / (1 - Ft))})
out["generating_function_limit"] = {"d": d, "N": N, "q": q, "T": T, "one_over_T": 1 / T, "values": lim}

with open(OUT, "w") as fh:
    json.dump(out, fh, indent=1)


# ---------------------------------------------------------------------------
# short report
# ---------------------------------------------------------------------------
def mx(key, block):
    return max(r[key] for r in out[block])


print("model: sym %.1e rows %.1e cols %.1e  q-scaling %.1e  Laplacian %.1e  mirror asym %.2f" % (
    mx("symmetry_maxabs", "model"), mx("row_sum_err", "model"), mx("col_sum_err", "model"),
    mx("I-P_minus_q(I-P1)_maxabs", "model"), mx("I-P_minus_(q/2d)Laplacian_maxabs", "model"),
    min(r["mirror_rule_asymmetry_maxabs"] for r in out["model"])))
print("structure: qT dev %.1e  pmf %.1e  g %.1e  transform %.1e  norm %.1e" % (
    max(p["qT_rel_dev_from_unit_rate"] for r in out["structure"] for p in r["per_q"]),
    max(p["pmf_spectral_vs_stepping_maxabs"] for r in out["structure"] for p in r["per_q"]),
    mx("g_spectral_vs_expm_maxabs", "structure"),
    max(p["transform_identity_maxabs"] for r in out["structure"] for p in r["per_q"]),
    mx("sum_b_over_nu_minus_1", "structure")))
print("  min eig Q - (1-2q):", min(p["min_eig_Q"] - p["bound_1-2q"] for r in out["structure"] for p in r["per_q"]))
for r in out["renewal_interlacing"]:
    print("interlacing", r["d"], r["N"], r["o"], r["a"], r["renewal_ratio_maxabs"], r["n_weighted_reflecting_levels"],
          r["n_zeros_of_Paa"], r["n_visible_killed_rates"], r["zeros_vs_visible_rates_max_rel"],
          r["strict_interlacing"], r["n_rates_with_nonzero_residue_for_this_start"],
          r["inv_nu0_minus_quasistationary_mean_rel"], r["nu0"] / r["first_weighted_reflecting_rate"],
          r["first_weighted_reflecting_rate"] / r["mu1"])
for r in out["two_spectra"]:
    print(json.dumps(r))
for r in out["pinv_resistance_commute"]:
    print(json.dumps(r))
print(json.dumps(out["generating_function_limit"]))
