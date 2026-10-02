"""07_beyond_inert.py -- defect types other than the blocked site.

(a) 1D.  A blocked site between start and target disconnects them, so the inert model is empty in
    1D.  The meaningful 1D analogue is a partially permeable bond: hop probability kappa*q/2 across
    it (the remaining attempt probability is added to the self-loops, so P stays symmetric).
    Exact:  MFPT_{0 -> N-1} = (2/q) sum_{k=0}^{N-2} (k+1)/kappa_k .
    Experiments (N=100, q=0.8): one barrier (kappa=0.1) at every position; random barriers
    (each bond a barrier with probability p, kappa=0.2), K=400 placements per p.
(b) 2D partially permeable obstacles: every bond touching a defect site has hop probability
    kappa*q/4 (kappa=0: inert site; kappa=1: clean).  N=35, p=0.10, the first 100 uniform placements
    of the main sweep, kappa in {0, 1e-3, 1e-2, 0.1, 0.3, 0.6, 1}.
(c) partial traps: T -> T - Delta e_n e_n^T on defect sites, i.e. a
    walker standing on a defect site is killed with probability Delta per step.
    N=35, p=0.10, same placements, Delta in {1e-5, 1e-4, 1e-3, 1e-2}: hitting probability H,
    conditional mean and certified mode of the (defective) first-passage PMF.
Outputs: data/beyond_1d_single.csv, data/beyond_1d_random.csv, data/beyond_2d_permeable.csv,
data/beyond_2d_traps.csv
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
import numpy as np, pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from multiprocessing import Pool
import fptcore as fc, ensemble

DATA = _os.path.join(_R, 'data', 'msc_defects')
q = 0.8

# ------------------------------------------------------------------ (a) 1D
def chain_1d(N, kappa):
    """kappa: array of N-1 bond permeabilities. Returns Q (transient 0..N-2), r, s=0."""
    w = (q / 2) * np.asarray(kappa, float)
    stay = np.ones(N)
    stay[:-1] -= w; stay[1:] -= w
    P = sp.diags([w, stay, w], [-1, 0, 1], format='csr')
    Q = P[:N - 1][:, :N - 1].tocsr()
    r = np.zeros(N - 1); r[N - 2] = w[N - 2]
    return Q, r

def stats_1d(kappa):
    N = len(kappa) + 1
    Q, r = chain_1d(N, kappa)
    mf, var, _ = fc.moments(Q, 0)
    S = fc.SpectralFPT(Q, r, 0)
    mode, fm, tc = S.mode()
    exact = (2 / q) * np.sum(np.arange(1, N) / np.asarray(kappa))
    return dict(mfpt=mf, mfpt_formula=float(exact), sd=float(np.sqrt(var)), mode=int(mode),
                ratio=mode / mf, cv=float(np.sqrt(var)) / mf, median=int(S.quantile(0.5, mf)))

def one_d_random(args):
    N, p, kap, rep = args
    rng = np.random.default_rng(np.random.SeedSequence([777, N, int(round(p * 1000)), rep]))
    kappa = np.where(rng.random(N - 1) < p, kap, 1.0)
    out = stats_1d(kappa)
    out.update(N=N, p=p, kappa=kap, rep=rep, n_barriers=int((kappa < 1).sum()),
               weighted_pos=float(np.sum(np.arange(1, N) * (kappa < 1)) / max(1, (kappa < 1).sum()) / N))
    return out

# ------------------------------------------------------------------ (b), (c) 2D
N2 = 35
def placement(rep, p=0.10):
    ss = np.random.SeedSequence([20261001, N2, ensemble.SCHEME_ID['uniform'], int(round(p * 10000)), rep])
    mask, _ = ensemble.draw_connected(N2, p, 'uniform', ss)
    return mask

def permeable(args):
    rep, kap = args
    mask = placement(rep)
    S_, T_ = (0, 0), (N2 - 1, N2 - 1)
    if kap == 0.0:
        om = ~mask; bw = None
    else:
        om = np.ones((N2, N2), bool)
        dflat = mask.ravel()
        bw = lambda a, b: np.where(dflat[a] | dflat[b], kap, 1.0)
    ch = fc.build_chain(om, q, S_, T_, bond_weight=bw)
    mf, var, _ = fc.moments(ch['Q'], ch['s'])
    Sp = fc.SpectralFPT(ch['Q'], ch['r'], ch['s'])
    mode, fm, tc = Sp.mode()
    return dict(rep=rep, kappa=kap, mfpt=mf, sd=float(np.sqrt(var)), mode=int(mode), ratio=mode / mf,
                cv=float(np.sqrt(var)) / mf, n_states=ch['n_cluster'])

def traps(args):
    rep, Delta = args
    mask = placement(rep)
    S_, T_ = (0, 0), (N2 - 1, N2 - 1)
    ch = fc.build_chain(np.ones((N2, N2), bool), q, S_, T_)
    Q = ch['Q'].tolil()
    sites = ch['sites']                       # flat lattice index of each transient state
    trap = mask.ravel()[sites]
    assert Delta <= 1 - q + 1e-15             # diagonal must stay non-negative
    d = Q.diagonal()
    d[trap] -= Delta
    Q.setdiag(d); Q = Q.tocsr()
    n = Q.shape[0]
    lu = spla.splu((sp.identity(n, format='csc') - Q.tocsc()))
    h = lu.solve(ch['r'])                     # hitting probabilities
    g = lu.solve(h)                           # sum_t t f(t) = e_s^T (I-Q)^-2 r
    H = float(h[ch['s']]); cm = float(g[ch['s']] / H)
    Sp = fc.SpectralFPT(Q, ch['r'], ch['s'])
    mode, fm, tc = Sp.mode()
    return dict(rep=rep, Delta=Delta, H=H, cond_mean=cm, mode=int(mode), ratio=mode / cm)

if __name__ == '__main__':
    t0 = time.time()
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    N = 100
    base = stats_1d(np.ones(N - 1))
    print('1D clean N=100', base, flush=True)
    rows = []
    for k in range(N - 1):
        kap = np.ones(N - 1); kap[k] = 0.1
        r_ = stats_1d(kap); r_.update(N=N, pos=k, kappa=0.1); rows.append(r_)
    df = pd.DataFrame(rows); df.to_csv(_os.path.join(_R, 'data', 'msc_defects', 'beyond_1d_single.csv'), index=False)
    print('1D single barrier: max |MFPT-formula| =', float(np.max(np.abs(df.mfpt - df.mfpt_formula))))
    print(df.iloc[[0, 10, 25, 49, 75, 90, 98]][['pos', 'mfpt', 'mode', 'ratio']].to_string(), flush=True)
    with Pool(workers) as pool:
        jobs = [(N, p, 0.2, rep) for p in (0.0, 0.05, 0.10, 0.20, 0.30, 0.50) for rep in range(1 if p == 0 else 400)]
        df = pd.DataFrame(pool.map(one_d_random, jobs, chunksize=25))
        df.to_csv(_os.path.join(_R, 'data', 'msc_defects', 'beyond_1d_random.csv'), index=False)
        print(df.groupby('p').agg(mfpt=('mfpt', 'mean'), mode=('mode', 'mean'), ratio=('ratio', 'mean'),
                                  ratio_sem=('ratio', lambda x: x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0),
                                  cv=('cv', 'mean')).to_string(), flush=True)
        jobs = [(rep, kap) for kap in (0.0, 1e-3, 1e-2, 0.1, 0.3, 0.6, 1.0) for rep in range(100)]
        df = pd.DataFrame(pool.map(permeable, jobs, chunksize=5))
        df.to_csv(_os.path.join(_R, 'data', 'msc_defects', 'beyond_2d_permeable.csv'), index=False)
        print(df.groupby('kappa').agg(mfpt=('mfpt', 'mean'), mode=('mode', 'mean'), ratio=('ratio', 'mean'),
                                      ratio_sem=('ratio', lambda x: x.std(ddof=1) / np.sqrt(len(x))),
                                      cv=('cv', 'mean')).to_string(), flush=True)
        jobs = [(rep, D) for D in (1e-5, 1e-4, 1e-3, 1e-2) for rep in range(100)]
        df = pd.DataFrame(pool.map(traps, jobs, chunksize=5))
        df.to_csv(_os.path.join(_R, 'data', 'msc_defects', 'beyond_2d_traps.csv'), index=False)
        print(df.groupby('Delta').agg(H=('H', 'mean'), cond_mean=('cond_mean', 'mean'), mode=('mode', 'mean'),
                                      ratio=('ratio', 'mean'),
                                      ratio_sem=('ratio', lambda x: x.std(ddof=1) / np.sqrt(len(x)))).to_string(), flush=True)
    print(f'done {time.time()-t0:.0f}s')
