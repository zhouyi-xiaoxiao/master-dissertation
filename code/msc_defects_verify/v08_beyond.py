"""v08 -- other defect types (own code): 1D permeable bonds, 2D permeable obstacles, literal Eq.(4.1) traps.
Output: data/v08_beyond.json"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import os, sys, json, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spla
from multiprocessing import Pool
import vcore as vc
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = _os.path.join(_R, 'data', 'msc_defects_verify', 'v08_beyond.json')
q = 0.8
ci = lambda x: float(1.96 * np.std(x, ddof=1) / np.sqrt(len(x)))

# ---------- 1D
def chain(kappa):
    """sites 0..N-1, bond k joins k,k+1 with hop probability kappa_k q/2; reflecting at 0; absorbing N-1."""
    N = len(kappa) + 1; w = np.asarray(kappa) * q / 2
    diag = np.ones(N); diag[:-1] -= w; diag[1:] -= w
    P = sp.diags([w, diag, w], [-1, 0, 1]).tocsr()
    Q = P[:N - 1][:, :N - 1].tocsr(); r = np.zeros(N - 1); r[-1] = w[-1]
    mean, var, _ = vc.moments(Q, 0); md = vc.certified_mode(Q, r, 0)
    return mean, np.sqrt(var), md['mode']

def one_1d(args):
    p, rep = args
    rng = np.random.default_rng([808, int(round(p * 1e4)), rep])
    kap = np.where(rng.random(99) < p, 0.2, 1.0)
    m, s, mo = chain(kap); formula = (2 / q) * np.sum(np.arange(1, 100) / kap)
    return m, s, mo, formula

# ---------- 2D permeable / traps
N = 35
def placement(rep, p=0.10):
    rng = np.random.default_rng([909, rep]); M = int(round(p * N * N))
    while True:
        m = np.zeros(N * N, bool); m[rng.choice(N * N - 2, M, replace=False) + 1] = True; m = m.reshape(N, N)
        if vc.connected(~m, (0, 0), (N - 1, N - 1)): return m

def perm(args):
    rep, kappa = args
    mask = placement(rep)
    if kappa == 0:
        r = vc.solve_config(~mask, q); return r['mfpt'], r['mode'], r['cv']
    om = np.ones((N, N), bool); idx, n = vc.cluster_index(om, (0, 0))
    I, J = vc.bonds(idx)
    flat = np.zeros(n, bool); flat[idx[mask]] = True
    w = np.where(flat[I] | flat[J], kappa, 1.0)
    P = vc.transition(idx, n, q, w); a = idx[N - 1, N - 1]
    Q, rr, _ = vc.reduce_target(P, a)
    mean, var, _ = vc.moments(Q, 0); md = vc.certified_mode(Q, rr, 0, chk=64)
    return mean, md['mode'], np.sqrt(var) / mean

def trap(args):
    rep, Delta = args
    mask = placement(rep)
    om = np.ones((N, N), bool); idx, n = vc.cluster_index(om, (0, 0))
    P = vc.transition(idx, n, q).tolil(); 
    for k in idx[mask]: P[k, k] -= Delta
    P = P.tocsr(); a = idx[N - 1, N - 1]
    Q, rr, _ = vc.reduce_target(P, a); m = Q.shape[0]
    lu = spla.splu(sp.identity(m, format='csc') - Q.tocsc())
    h = lu.solve(rr); H = h[0]                 # hitting probability from each state
    g = lu.solve(h)                            # sum_t t f(t) = e_s (I-Q)^-2 r
    md = vc.certified_mode(Q, rr, 0)
    return H, g[0] / H, md['mode']

if __name__ == '__main__':
    res = {}
    # 1D checks
    Nc = 100
    m0, s0, mo0 = chain(np.ones(Nc - 1))
    res['1d_clean'] = dict(mfpt=m0, formula=Nc * (Nc - 1) / q, mode=mo0, ratio=mo0 / m0, cv=s0 / m0)
    sing = []
    for b in range(99):
        k = np.ones(99); k[b] = 0.1
        m, s, mo = chain(k); sing.append((b, m, (2 / q) * np.sum(np.arange(1, 100) / k), mo, mo / m))
    sing = np.array(sing)
    res['1d_single_k0.1'] = dict(max_rel_err_formula=float(np.max(np.abs(sing[:, 1] / sing[:, 2] - 1))), ratio_min=float(sing[:, 4].min()), bond_min=int(sing[np.argmin(sing[:, 4]), 0]),
                                 ratio_max=float(sing[:, 4].max()), bond_max=int(sing[np.argmax(sing[:, 4]), 0]))
    with Pool(3) as pool:
        res['1d_random'] = {}
        for p in (0.05, 0.1, 0.2, 0.3, 0.5):
            R = np.array(pool.map(one_1d, [(p, r) for r in range(400)], chunksize=10))
            res['1d_random'][str(p)] = dict(mfpt=float(R[:, 0].mean()), mfpt_ci=ci(R[:, 0]), exact_avg=12375 * (1 + 4 * p), mode=float(R[:, 2].mean()), mode_ci=ci(R[:, 2]),
                                            ratio=float((R[:, 2] / R[:, 0]).mean()), ratio_ci=ci(R[:, 2] / R[:, 0]), ratio_sd=float((R[:, 2] / R[:, 0]).std(ddof=1)),
                                            cv=float((R[:, 1] / R[:, 0]).mean()), max_rel_err_formula=float(np.max(np.abs(R[:, 0] / R[:, 3] - 1))))
            print('1d', p, res['1d_random'][str(p)], flush=True)
        json.dump(res, open(OUT, 'w'), indent=1)
        res['perm'] = {}
        for kappa in (0, 1e-3, 1e-2, 0.1, 0.3, 0.6):
            R = np.array(pool.map(perm, [(r, kappa) for r in range(100)], chunksize=2))
            res['perm'][str(kappa)] = dict(mfpt=float(R[:, 0].mean()), mfpt_ci=ci(R[:, 0]), mode=float(R[:, 1].mean()), mode_ci=ci(R[:, 1]),
                                           ratio=float((R[:, 1] / R[:, 0]).mean()), ratio_ci=ci(R[:, 1] / R[:, 0]), cv=float(R[:, 2].mean()))
            print('perm', kappa, res['perm'][str(kappa)], flush=True)
            json.dump(res, open(OUT, 'w'), indent=1)
        # per-placement jump: mfpt(kappa=1e-3)/mfpt(inert)
        res['traps'] = {}
        om = np.ones((N, N), bool); idx, n = vc.cluster_index(om, (0, 0)); P = vc.transition(idx, n, q); Qc, rc, _ = vc.reduce_target(P, idx[N - 1, N - 1])
        for Delta in (1e-5, 1e-4, 1e-3, 1e-2):
            R = np.array(pool.map(trap, [(r, Delta) for r in range(100)], chunksize=2))
            z = 1 - 0.1 * Delta
            mf = z * spla.spsolve((sp.identity(Qc.shape[0], format='csc') - z * Qc.tocsc()), rc)[0]
            res['traps'][str(Delta)] = dict(H=float(R[:, 0].mean()), H_ci=ci(R[:, 0]), H_meanfield=float(mf), cond_mean=float(R[:, 1].mean()), cond_mean_ci=ci(R[:, 1]),
                                            mode=float(R[:, 2].mean()), ratio=float((R[:, 2] / R[:, 1]).mean()), ratio_of_means=float(R[:, 2].mean() / R[:, 1].mean()))
            print('trap', Delta, res['traps'][str(Delta)], flush=True)
            json.dump(res, open(OUT, 'w'), indent=1)
    print(json.dumps({k: res[k] for k in ('1d_clean', '1d_single_k0.1')}, indent=1))
