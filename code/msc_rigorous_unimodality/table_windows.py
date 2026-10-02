"""table_windows.py -- numerical illustration (float) of Theorems 5.6 and 5.10 for corner-to-corner, continuous time, unit rate (q=1):
   m_N  = first zero of h''   (mode of the hypoexponential density),
   t_N  = first zero of (d-1) h'^2 + h h''  (end of the convexity interval of u = h^d),
   mode = mode of g_1,
   T_N  = ln(4 sqrt(n-1)/alpha_1^2)/(beta_1-alpha_1)   (Theorem 5.6(ii)),
   Tbar = explicit bound of Theorem 5.6(iii).
alpha_1 from sparse eigensolver; mode of g from positive time stepping at q=1/8 rescaled (float)."""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import json, math, os, sys
import numpy as np
import scipy.sparse as sp, scipy.sparse.linalg as spla
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
def alpha1(N, d):
    n = N ** d
    idx = np.arange(n).reshape((N,) * d)
    rows, cols, vals = [], [], []
    diag = np.zeros(n)
    w = 1.0 / (2 * d)
    for ax in range(d):
        lo = [slice(None)] * d; hi = [slice(None)] * d
        lo[ax] = slice(0, N - 1); hi[ax] = slice(1, N)
        i = idx[tuple(lo)].ravel(); j = idx[tuple(hi)].ravel()
        rows += [i, j]; cols += [j, i]; vals += [-w * np.ones(i.size)] * 2
        np.add.at(diag, i, w); np.add.at(diag, j, w)
    L = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)).tocsr() + sp.diags(diag)
    keep = np.arange(n - 1)   # target = last index = far corner
    A = L[keep][:, keep].tocsc()
    val = spla.eigsh(A, k=1, sigma=0, which="LM", return_eigenvectors=False)
    return float(val[0])
def hfun(N, d):
    ks = np.arange(0, N)
    lam = (1 - np.cos(ks * np.pi / N)) / d
    c = np.array([(1 if k == 0 else 2) / N * math.cos(k * math.pi * 0.5 / N) * math.cos(k * math.pi * (N - 0.5) / N) for k in ks])
    return lambda t, der=0: float(((-lam) ** der * c * np.exp(-lam * t)).sum())
def first_zero(fn, t_hi, npts=4001):
    # the spectral sum suffers cancellation for t << N^2; the zero sought lies in [0.05, 1] * t_hi
    ts = np.linspace(0.05 * t_hi, t_hi, npts)
    vals = np.array([fn(t) for t in ts])
    assert vals[0] > 0
    i = int(np.argmax(vals < 0)); assert vals[i] < 0
    a, b = ts[i - 1], ts[i]
    for _ in range(60):
        m = 0.5 * (a + b)
        if fn(m) > 0: a = m
        else: b = m
    return 0.5 * (a + b)
def mode_cont(N, d, qq=1.0 / 16):
    from explore02 import runs_float
    T = int(3.0 * N * N * d / qq * (1 if d == 2 else 0.8))
    f = runs_float(N, d, qq, (0,) * d, (N - 1,) * d, T)
    return float(np.argmax(f)) * qq
rows = []
for (d, Ns) in ((2, (4, 8, 16, 32, 64)), (3, (4, 8, 16, 30))):
    for N in Ns:
        n = N ** d
        h = hfun(N, d)
        mN = first_zero(lambda t: h(t, 2), 1.0 * N * N * d)
        tN = first_zero(lambda t: (d - 1) * h(t, 1) ** 2 + h(t) * h(t, 2), 1.5 * N * N * d)
        al = alpha1(N, d); be = (1 - math.cos(math.pi / N)) / d
        TN = math.log(4 * math.sqrt(n - 1) / al ** 2) / (be - al)
        M = 2 * d * d * (N - 1) * n
        Tbar = (math.log(4) + 0.5 * math.log(n - 1) + 2 * math.log(M)) / (be - 1 / (2 * (n - 1)))
        md = mode_cont(N, d)
        rec = dict(d=d, N=N, m_N=mN, t_N=tN, mode=md, T_N=TN, Tbar_N=Tbar, alpha_1=al, beta_1=be,
                   m_N_over_dN2=mN / (d * N * N), t_N_over_dN2=tN / (d * N * N), mode_over_N2=md / N ** 2, T_N_over_N2=TN / N ** 2)
        rows.append(rec)
        print("d=%d N=%3d  m_N=%10.2f  t_N=%10.2f  mode=%10.2f  T_N=%11.1f  Tbar=%11.1f   (m_N/dN^2=%.4f, t_N/dN^2=%.4f, mode/N^2=%.3f, T_N/N^2=%.2f)" %
              (d, N, mN, tN, md, TN, Tbar, mN / (d * N * N), tN / (d * N * N), md / N ** 2, TN / N ** 2), flush=True)
json.dump(rows, open(_os.path.join(_R, 'data', 'msc_rigorous_unimodality', 'table_windows.json'), "w"), indent=1)
