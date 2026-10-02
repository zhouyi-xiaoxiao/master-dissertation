#!/usr/bin/env python
"""Check by the second implementation, part 2: cross-validation of every building block.

Writes ../data/validate.json.  All checks are against brute-force dense/sparse linear algebra
built directly from the master equation of the model (Section 2 of the article).
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import json, os, subprocess, sys, itertools
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.linalg as la
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vspec

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = _os.path.join(_R, 'data', 'msc_modes_verify')
OUT = {}


def build_P(d, N, q):
    """Full one-step matrix of the lazy reflecting walk (cancelled moves stay). Row-stochastic & symmetric."""
    n = N ** d
    idx = np.arange(n).reshape((N,) * d)
    rows, cols, vals = [], [], []
    diag = np.full(n, 1.0 - q)
    for ax in range(d):
        for sgn in (-1, 1):
            src = [slice(None)] * d
            dst = [slice(None)] * d
            if sgn == 1:
                src[ax] = slice(0, N - 1); dst[ax] = slice(1, N)
                edge = [slice(None)] * d; edge[ax] = N - 1
            else:
                src[ax] = slice(1, N); dst[ax] = slice(0, N - 1)
                edge = [slice(None)] * d; edge[ax] = 0
            r = idx[tuple(src)].ravel(); c = idx[tuple(dst)].ravel()
            rows.append(r); cols.append(c); vals.append(np.full(len(r), q / (2 * d)))
            diag[idx[tuple(edge)].ravel()] += q / (2 * d)       # cancelled move
    P = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)).tocsr()
    P = P + sp.diags(diag)
    return P.tocsr(), idx


def transient(P, targets):
    n = P.shape[0]
    keep = np.setdiff1d(np.arange(n), targets)
    Q = P[keep][:, keep].tocsc()
    r = np.asarray(P[keep][:, targets].sum(axis=1)).ravel()    # one-step absorption probability
    return Q, r, keep


def mfpt_solve(d, N, q, start, target):
    P, idx = build_P(d, N, q)
    Q, r, keep = transient(P, np.array([target]))
    m = spla.spsolve(sp.identity(Q.shape[0], format="csc") - Q, np.ones(Q.shape[0]))
    return m[np.searchsorted(keep, start)]


def single_sum_form_a(N, q):
    k = np.arange(1, N)
    phi = np.arccosh(2.0 - np.cos(np.pi * k / N))
    B = np.cosh(N * phi) + np.cosh((N - 1) * phi) - (-1.0) ** k * (1 + np.cosh(phi))
    return 2 * N * (N - 1) / q + 4 * N / q * np.sum(np.cos(np.pi * k / (2 * N)) ** 2 * B / (np.sinh(phi) * np.sinh(N * phi)))


# ---------------------------------------------------------------------------------------------
# 0. Q symmetric, row sums, model sanity
P, idx = build_P(2, 6, 0.8)
OUT["P_symmetric_maxabs"] = float(abs(P - P.T).max())
OUT["P_rowsum_err"] = float(abs(np.asarray(P.sum(axis=1)).ravel() - 1).max())

# 1. closed-form end-site resolvent vs dense inverse (positive, negative-real and complex sigma)
errs = []
for d in (1, 2, 3):
    for N in (2, 3, 7, 12):
        a = 1.0 / d
        L = np.zeros((N, N))
        for i in range(N):
            for j in (i - 1, i + 1):
                if 0 <= j < N:
                    L[i, i] += a / 2; L[i, j] -= a / 2
        ev = np.linalg.eigvalsh(L)
        e_formula = np.sort(vspec.e1d(np.arange(N), N, d))
        errs.append(("eig", d, N, float(abs(ev - e_formula).max())))
        for sig in (0.37, 3e-5, -0.013 * a, -0.61 * a, -1.37 * a, 0.2 + 0.5j, -0.3 + 1e-3j):
            G = np.linalg.inv(sig * np.eye(N) + L)
            Gaa, Gax, dGaa = vspec.G_end(np.array([sig]), N, d, deriv=True)
            G2 = G @ G
            errs.append(("G", d, N, str(sig), float(abs(Gaa[0] - G[N - 1, N - 1]) / abs(G[N - 1, N - 1])),
                         float(abs(Gax[0] - G[N - 1, 0]) / abs(G[N - 1, 0])),
                         float(abs(dGaa[0] + G2[N - 1, N - 1]) / abs(G2[N - 1, N - 1]))))
OUT["resolvent_checks_max_rel_err"] = max(max(e[4:]) for e in errs if e[0] == "G")
OUT["eig_formula_max_abs_err"] = max(e[3] for e in errs if e[0] == "eig")

# 2. corner sums vs dense resolvent of the full d-dim generator
errs2 = []
for d, N in ((2, 6), (2, 9), (3, 4), (3, 5)):
    P1, idx = build_P(d, N, 1.0)
    Lfull = (sp.identity(N ** d) - P1).toarray()
    a_site = N ** d - 1; x_site = 0
    C = vspec.Corner(d, N)
    for s in (0.21, 1e-3, -0.4 * C.mu1, -1.5 * C.mu1, -3.3 * C.mu1):
        R = np.linalg.inv(s * np.eye(N ** d) + Lfull)
        R2 = R @ R
        Paa, Pax, dPaa = C.PQ(s)
        errs2.append((d, N, s / C.mu1, abs(Paa - R[a_site, a_site]) / abs(R[a_site, a_site]),
                      abs(Pax - R[a_site, x_site]) / abs(R[a_site, x_site]),
                      abs(dPaa + R2[a_site, a_site]) / abs(R2[a_site, a_site])))
OUT["corner_sum_checks_max_rel_err"] = float(max(max(e[3:]) for e in errs2))

# 3. MFPT three-way: spectral single/double sum, sparse solve, single sum form (a)
rows = []
for q in (0.3, 0.8, 1.0):
    for N in list(range(2, 41)) + [60, 100, 150, 200]:
        C = vspec.Corner(2, N)
        m_spec = C.mfpt() / q
        m_solve = mfpt_solve(2, N, q, 0, N * N - 1)
        m_35 = single_sum_form_a(N, q)
        rows.append((q, N, m_spec, m_solve, m_35))
r = np.array(rows)
OUT["mfpt2d_spec_vs_solve_max_rel"] = float(np.max(abs(r[:, 2] - r[:, 3]) / r[:, 3]))
OUT["mfpt2d_form_a_vs_solve_max_rel"] = float(np.max(abs(r[:, 4] - r[:, 3]) / r[:, 3]))
OUT["mfpt2d_form_a_vs_spec_max_rel"] = float(np.max(abs(r[:, 4] - r[:, 2]) / r[:, 2]))
OUT["mfpt2d_examples"] = {f"q{q}_N{int(N)}": [a, b, c] for q, N, a, b, c in rows if N in (5, 35, 200)}
rows3 = []
for N in range(2, 17):
    C = vspec.Corner(3, N)
    m_spec = C.mfpt() / 0.8
    m_solve = mfpt_solve(3, N, 0.8, 0, N ** 3 - 1)
    rows3.append((N, m_spec, m_solve))
r3 = np.array(rows3)
OUT["mfpt3d_spec_vs_solve_max_rel"] = float(np.max(abs(r3[:, 1] - r3[:, 2]) / r3[:, 2]))
OUT["mfpt3d_examples"] = {f"N{int(N)}": [a, b] for N, a, b in rows3 if N in (5, 15)}
# 1D
r1 = [(N, mfpt_solve(1, N, 0.8, 0, N - 1), N * (N - 1) / 0.8) for N in (2, 5, 50, 200)]
OUT["mfpt1d_solve_vs_N(N-1)/q"] = r1

# 4. poles/residues vs dense eigen-decomposition of the killed generator, and density vs expm
pole_checks = []
for d, N in ((2, 8), (2, 13), (3, 5)):
    P1, idx = build_P(d, N, 1.0)
    Q1, r1v, keep = transient(P1, np.array([N ** d - 1]))
    K = (sp.identity(Q1.shape[0]) - Q1).toarray()          # killed generator (unit rate), symmetric
    ev, V = np.linalg.eigh(K)
    amp = (V.T @ r1v) * V[np.searchsorted(keep, 0), :]     # residue of each killed eigenmode in g(t)
    C = vspec.Corner(d, N)
    nus, res = C.poles(numax_units=14.0)
    # match each pole to the killed eigenvalue cluster (sum residues of degenerate eigenvalues)
    de, da = [], []
    for nu, a_ in zip(nus, res):
        sel = abs(ev - nu) < 1e-9
        de.append(abs(ev[sel][0] - nu) / nu if sel.any() else np.inf)
        da.append(abs(amp[sel].sum() - a_) / abs(a_) if sel.any() else np.inf)
    # eigenvalues with non-negligible residue that were missed below numax?
    missed = [float(e) for e, am in zip(ev, amp) if e < nus[-1] and abs(am) > 1e-12 * abs(amp).max()
              and np.min(abs(nus - e)) > 1e-9]
    # density at a few times, all eigenmodes
    ts = np.array([0.5, 1.0, 2.0, 4.0]) / C.mu1
    g_exact = np.array([np.sum(amp * np.exp(-ev * t)) for t in ts])
    g_poles = np.array([np.sum(res * np.exp(-nus * t)) for t in ts])
    # exact continuous-time mode from the full eigen-expansion
    from scipy.optimize import brentq
    gp = lambda t: -np.sum(amp * ev * np.exp(-ev * t))
    tm_poles, _ = vspec.density_mode(nus, res)
    tm_exact = brentq(gp, 0.5 * tm_poles, 2 * tm_poles, xtol=1e-13 * tm_poles)
    pole_checks.append(dict(d=d, N=N, n_poles=len(nus), max_rel_err_nu=float(max(de)), max_rel_err_res=float(max(da)),
                            missed=missed, g_rel_err=[float(x) for x in abs(g_poles - g_exact) / g_exact],
                            mode_exact=float(tm_exact), mode_poles=float(tm_poles),
                            mfpt_spec=C.mfpt(), mfpt_from_eig=float(np.sum(amp / ev ** 2))))
OUT["pole_checks"] = pole_checks

# 5. z-transform identity  F~_q(z) = F^_1((1-z)/(z q)), and stepper vs sparse matrix powers
S = _os.path.join(_R, 'data', 'msc_modes_verify', 'pmf', '_val.bin')
zid = []
for d, N, q in ((2, 9, 0.8), (3, 5, 0.8), (1, 12, 0.5), (2, 9, 1.0)):
    out = subprocess.run([_os.path.join(_R, 'code', 'msc_modes_verify', 'stepper'), str(d), str(N), repr(q), "CC", "0", "10", "100000000", S],
                         capture_output=True, text=True, check=True).stdout
    f = np.fromfile(S, dtype=np.float64)
    # sparse-matrix reference: f(t) = r^T Q^{t-1} e_s
    Pq, idx = build_P(d, N, q)
    Qm, rv, keep = transient(Pq, np.array([N ** d - 1]))
    v = np.zeros(Qm.shape[0]); v[0] = 1.0
    fref = []
    for t in range(1, 400):
        fref.append(rv @ v); v = Qm @ v
    zid.append(dict(d=d, N=N, q=q, stepper_vs_sparse_maxabs=float(np.max(abs(f[:399] - np.array(fref)))), sum_f=float(f.sum())))
    if d >= 2:
        C = vspec.Corner(d, N)
        t = np.arange(1, len(f) + 1)
        for z in (0.5, 0.9, 0.99, 0.6 + 0.3j):
            lhs = np.sum(f * z ** t)
            s = (1 - z) / (z * q)
            # use complex-capable raw sums
            tot = C._sums(s)
            rhs = tot[1] / tot[0]
            zid[-1][f"z={z}"] = float(abs(lhs - rhs))
os.remove(S)
OUT["ztransform_identity_and_stepper"] = zid

with open(_os.path.join(_R, 'data', 'msc_modes_verify', 'validate.json'), "w") as fh:
    json.dump(OUT, fh, indent=1, default=str)
print(json.dumps(OUT, indent=1, default=str))
