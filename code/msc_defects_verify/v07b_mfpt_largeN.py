"""v07b -- does the MFPT factor converge to Theta(p)?  MFPT only (one sparse solve per placement), uniform
placement, p = 0.1 and 0.2, N = 35..280, K placements; plus the exact decomposition MFPT = (4n/q)(G_aa - G_sa).
Output: data/v07b_mfpt_largeN.csv"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd, scipy.sparse as sp, scipy.sparse.linalg as spla
from multiprocessing import Pool
import vcore as vc
HERE = os.path.dirname(os.path.abspath(__file__)); D = _os.path.join(_R, 'data', 'msc_defects_verify')
q = 0.8

def work(args):
    N, p, rep = args
    rng = np.random.default_rng(np.random.SeedSequence([27182, N, int(round(p * 1e4)), rep])); M = int(round(p * N * N))
    while True:
        m = np.zeros(N * N, bool)
        if M: m[rng.choice(N * N - 2, M, replace=False) + 1] = True
        m = m.reshape(N, N)
        idx, n = vc.cluster_index(~m, (0, 0))
        if idx[N - 1, N - 1] >= 0: break
    a = idx[N - 1, N - 1]
    L = vc.laplacian(idx, n)
    keep = np.r_[1:n]
    lu = spla.splu(L[keep][:, keep].tocsc())
    b = -np.ones(n) / n; b[a] += 1.0
    g = np.zeros(n); g[keep] = lu.solve(b[keep]); g -= g.mean()
    om = ~m
    return dict(N=N, p=p, rep=rep, n=n, G_aa=g[a], G_sa=g[0], mfpt=(4 * n / q) * (g[a] - g[0]), deg_t=int(om[N - 2, N - 1]) + int(om[N - 1, N - 2]))

if __name__ == '__main__':
    fn = _os.path.join(_R, 'data', 'msc_defects_verify', 'v07b_mfpt_largeN.csv')
    done = pd.read_csv(fn) if os.path.exists(fn) else pd.DataFrame()
    t0 = time.time()
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 2) as pool:
        for N, K in ((35, 800), (50, 800), (70, 800), (100, 600), (140, 400), (200, 300), (280, 200)):
            for p in (0.0, 0.10, 0.20):
                if len(done) and ((done.N == N) & (np.isclose(done.p, p))).any(): continue
                k = 1 if p == 0 else K
                df = pd.DataFrame(pool.map(work, [(N, p, r) for r in range(k)], chunksize=2))
                done = pd.concat([done, df]); done.to_csv(fn, index=False)
                print(N, p, k, df.mfpt.mean(), time.time() - t0, flush=True)
