"""v11 -- is uniform disorder a pure time dilation of the WHOLE law?  Kolmogorov distances between
(i) the ensemble-averaged survival function rescaled by Theta(p) and the clean one, and
(ii) each placement's law in units of its own MFPT and the clean law in units of the clean MFPT.
Compared with structured placements.  N = 35, dense spectral route.  Output: data/v11_shape.json"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, scipy.linalg as la
from multiprocessing import Pool
import vcore as vc, v06_structured as v6
HERE = os.path.dirname(os.path.abspath(__file__)); D = _os.path.join(_R, 'data', 'msc_defects_verify')
N, q = 35, 0.8
X = np.linspace(0.0, 6.0, 601)          # time in units of the (clean or own) MFPT

def surv(mask):
    idx, n = vc.cluster_index(~mask, (0, 0)); a = idx[N - 1, N - 1]
    P = vc.transition(idx, n, q); Q, r, _ = vc.reduce_target(P, a)
    lam, V = la.eigh(Q.toarray()); c = V[0] * V.sum(axis=0)
    mfpt = float(np.sum(c / (1 - lam)))
    def S(t):                             # t array (rounded to integers)
        t = np.round(t).astype(np.int64)
        return (c[None, :] * np.power(lam[None, :], t[:, None])).sum(axis=1)
    return mfpt, S

M0, S0 = surv(np.zeros((N, N), bool))
S0x = S0(X * M0)

def uni(args):
    p, rep, theta = args
    rng = np.random.default_rng(np.random.SeedSequence([987654321, N, 11, int(round(p * 1e4)), rep])); M = int(round(p * N * N))
    while True:
        sites = rng.choice(N * N - 2, M, replace=False) + 1
        m = np.zeros(N * N, bool); m[sites] = True; m = m.reshape(N, N)
        if vc.connected(~m, (0, 0), (N - 1, N - 1)): break
    mf, S = surv(m)
    return mf, S(X * M0 * theta), float(np.max(np.abs(S(X * mf) - S0x)))

if __name__ == '__main__':
    TH = json.load(open(_os.path.join(_R, 'data', 'msc_defects_verify', 'v04b_theta.json')))
    res = {}
    with Pool(3) as pool:
        for p in (0.1, 0.2):
            th = TH[str(p)]['Theta']
            R = pool.map(uni, [(p, r, th) for r in range(200)], chunksize=4)
            Sav = np.mean([r[1] for r in R], axis=0); ks_own = np.array([r[2] for r in R])
            res[f'uniform_p{p}'] = dict(K=200, theta=th, KS_ensemble_rescaled_by_theta=float(np.max(np.abs(Sav - S0x))), KS_own_mfpt_mean=float(ks_own.mean()),
                                        KS_own_mfpt_median=float(np.median(ks_own)), KS_own_mfpt_max=float(ks_own.max()),
                                        KS_ensemble_rescaled_by_mean_mfpt=None)
            print(p, res[f'uniform_p{p}'], flush=True)
    dd = v6.det()
    for k in ('target_box_R5_g1', 'start_box_R8_g1', 'wall_mid_gap_centre_g1', 'serpentine_k3_g3', 'block12_centre', 'periodic_a3'):
        mf, S = surv(dd[k]); res[k] = dict(KS_own_mfpt=float(np.max(np.abs(S(X * mf) - S0x))), mfpt_fac=mf / M0)
        print(k, res[k], flush=True)
    # reference scale: clean lattices of other sizes vs N = 35 (own-MFPT units)
    json.dump(res, open(_os.path.join(_R, 'data', 'msc_defects_verify', 'v11_shape.json'), 'w'), indent=1)
