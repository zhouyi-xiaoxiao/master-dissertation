#!/usr/bin/env python
"""Check by the second implementation, part 5: Abate-Whitt formulas, MFPT cross-checks by linear solves,
MFPT constants, other geometries.  Output: ../data/misc.json (sections are cached)."""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import json, os, sys, time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vspec

HERE = os.path.dirname(os.path.abspath(__file__))
OUTF = _os.path.join(_R, 'data', 'msc_modes_verify', 'misc.json')
OUT = json.load(open(OUTF)) if os.path.exists(OUTF) else {}


def save():
    json.dump(OUT, open(OUTF, "w"), indent=1)


def build_P(d, N, q):
    n = N ** d
    idx = np.arange(n).reshape((N,) * d)
    rows, cols, vals = [], [], []
    diag = np.full(n, 1.0 - q)
    for ax in range(d):
        for sgn in (-1, 1):
            src = [slice(None)] * d; dst = [slice(None)] * d; edge = [slice(None)] * d
            if sgn == 1:
                src[ax] = slice(0, N - 1); dst[ax] = slice(1, N); edge[ax] = N - 1
            else:
                src[ax] = slice(1, N); dst[ax] = slice(0, N - 1); edge[ax] = 0
            r = idx[tuple(src)].ravel(); c = idx[tuple(dst)].ravel()
            rows.append(r); cols.append(c); vals.append(np.full(len(r), q / (2 * d)))
            diag[idx[tuple(edge)].ravel()] += q / (2 * d)
    P = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)).tocsr()
    return (P + sp.diags(diag)).tocsr(), idx


def sites(d, N, geom):
    idx = np.arange(N ** d).reshape((N,) * d)
    c = (N - 1) // 2
    if geom == "CC":
        return 0, np.array([N ** d - 1])
    if geom == "C2M":
        return 0, np.array([idx[(c,) * d]])
    if geom == "M2C":
        return int(idx[(c,) * d]), np.array([N ** d - 1])
    if geom == "EXIT":
        mask = np.zeros((N,) * d, bool)
        for ax in range(d):
            sl = [slice(None)] * d; sl[ax] = 0; mask[tuple(sl)] = True
            sl[ax] = N - 1; mask[tuple(sl)] = True
        return int(idx[(c,) * d]), idx[mask]


def mfpt(d, N, q, geom, method="auto"):
    P, idx = build_P(d, N, q)
    start, targets = sites(d, N, geom)
    keep = np.setdiff1d(np.arange(N ** d), targets)
    A = (sp.identity(len(keep), format="csr") - P[keep][:, keep]).tocsr()
    b = np.ones(len(keep))
    if method == "auto":
        method = "lu" if (d < 3 and len(keep) < 200000) or len(keep) < 20000 else "cg"
    if method == "lu":
        m = spla.spsolve(A.tocsc(), b)
        info = "sparse LU"
    else:
        Dinv = 1.0 / A.diagonal()
        M = spla.LinearOperator(A.shape, lambda x: Dinv * x)
        m, flag = spla.cg(A, b, rtol=1e-13, atol=0.0, maxiter=200000, M=M)
        res = np.linalg.norm(A @ m - b) / np.linalg.norm(b)
        info = f"CG flag={flag} relres={res:.2e}"
    return float(m[np.searchsorted(keep, start)]), info


# ---- A. Abate-Whitt: fixed-radius rule (r = 0.9, K = 54) vs the correct rule -----------------------
if "abate_whitt" not in OUT:
    d, N, q = 2, 35, 0.8
    C = vspec.Corner(d, N)

    def Ftilde(z):
        s = (1 - z) / (z * q)
        tot = C._sums(s)
        return tot[1] / tot[0]

    def printed(t, r=0.9, K=54):
        k = np.arange(1, K + 1)
        return r ** (-t) / t * (0.5 * Ftilde(r).real + sum((-1) ** kk * Ftilde(r * np.exp(1j * np.pi * kk / t)).real for kk in k))

    def correct(t, gamma=10.0):
        r = 10 ** (-gamma / (2 * t))
        k = np.arange(1, 2 * t + 1)
        return sum((-1) ** kk * Ftilde(r * np.exp(1j * np.pi * kk / t)).real for kk in k) / (2 * t * r ** t)

    # reference: pole expansion (validated against the stepper elsewhere) and the stepper itself
    import subprocess
    S = _os.path.join(_R, 'data', 'msc_modes_verify', 'pmf', '_aw.bin')
    subprocess.run([_os.path.join(_R, 'code', 'msc_modes_verify', 'stepper'), "2", "35", "0.8", "CC", "1.5", "6100", "100000000", S], check=True, capture_output=True)
    f = np.fromfile(S); os.remove(S)
    rows = []
    for t in (20, 54, 100, 400, 1000, 2493, 6000):
        p = printed(t)
        c = correct(t) if t <= 2493 else None
        rows.append(dict(t=t, exact_stepper=float(f[t - 1]), printed_eq_2_17=float(p), correct_rule=None if c is None else float(c),
                         rel_err_correct=None if c is None else float(abs(c - f[t - 1]) / f[t - 1])))
        print(rows[-1], flush=True)
    OUT["abate_whitt"] = rows
    save()

# ---- B. spectral MFPT vs linear solves (3D, large) --------------------------------------------
if "mfpt_cg_3d" not in OUT:
    rows = []
    for N in (20, 40, 100):
        t0 = time.time()
        m_cg, info = mfpt(3, N, 0.8, "CC", method="cg")
        m_sp = vspec.Corner(3, N).mfpt() / 0.8
        rows.append(dict(N=N, mfpt_cg=m_cg, mfpt_spectral=m_sp, rel_diff=abs(m_cg - m_sp) / m_sp, info=info, wall=time.time() - t0))
        print(rows[-1], flush=True)
    OUT["mfpt_cg_3d"] = rows
    save()

# ---- C. MFPT for the other geometries ---------------------------------------------------------
if "mfpt_geom" not in OUT:
    rows = []
    for d, Ns in ((2, (11, 21, 41, 101, 201)), (3, (11, 21, 41, 61))):
        for N in Ns:
            for geom in ("C2M", "M2C"):
                m, info = mfpt(d, N, 0.8, geom)
                rows.append(dict(d=d, N=N, q=0.8, geom=geom, mfpt=m, info=info)); print(rows[-1], flush=True)
    for d, Ns, qq in ((1, (51, 101, 201, 401), 0.8), (2, (21, 51, 101, 201), 0.8), (3, (11, 21, 31, 51), 0.8),
                      (1, (51,), 0.9), (2, (51,), 0.9), (3, (31,), 0.9)):
        for N in Ns:
            m, info = mfpt(d, N, qq, "EXIT")
            rows.append(dict(d=d, N=N, q=qq, geom="EXIT", mfpt=m, info=info)); print(rows[-1], flush=True)
    OUT["mfpt_geom"] = rows
    save()

# ---- D. MFPT constants ------------------------------------------------------------------------
if "mfpt_constants" not in OUT:
    c2 = []
    for N in (10, 100, 1000, 10 ** 4, 10 ** 5, 10 ** 6, 2 ** 22, 2 ** 24):
        m = vspec.Corner(2, N).mfpt()
        c2.append(dict(N=N, mfpt_unit=m, c2_seq=(m - 8 / np.pi * N * N * np.log(N)) / N ** 2,
                       X=float(vspec.e1d(1, N, 2) * m), X_minus_2pi_lnN=float(vspec.e1d(1, N, 2) * m - 2 * np.pi * np.log(N))))
        print(c2[-1], flush=True)
    c3 = []
    for N in (10, 20, 40, 80, 160, 320, 640, 1280, 2560, 4096):
        m = vspec.Corner(3, N).mfpt()
        c3.append(dict(N=N, mfpt_unit=m, m_over_N3=m / N ** 3, X=float(vspec.e1d(1, N, 3) * m)))
        print(c3[-1], flush=True)
    # Richardson: m/N^3 = C3 + C3'/N + C3''/N^2 ; successive doublings
    seq = {r["N"]: r["m_over_N3"] for r in c3}
    rich = []
    for N in (40, 80, 160, 320, 640, 1280):
        a, b = seq[N], seq[2 * N]
        C3 = 2 * b - a                    # removes 1/N
        C3p = (a - b) * 2 * N             # slope estimate (leading)
        rich.append(dict(N=N, C3_est=C3, C3prime_est=C3p))
    # second-order: three sizes N,2N,4N -> fit C3 + C3'/N + C3''/N^2 exactly
    rich2 = []
    for N in (40, 80, 160, 320, 640):
        A = np.array([[1, 1 / n, 1 / n ** 2] for n in (N, 2 * N, 4 * N)])
        sol = np.linalg.solve(A, np.array([seq[N], seq[2 * N], seq[4 * N]]))
        rich2.append(dict(N=N, C3=sol[0], C3prime=sol[1], C3second=sol[2]))
    OUT["mfpt_constants"] = dict(c2=c2, c3=c3, richardson1=rich, richardson2=rich2)
    save()

print(json.dumps({k: (v if k != "mfpt_constants" else "...") for k, v in OUT.items()}, indent=1)[:6000])
