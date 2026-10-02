"""v07 -- size dependence (own code, own seeds): clean lattices N=10..200 and uniform random defects
p=0.1, 0.2 for N in {15,25,50,70,100,140}.  Output: data/v07_clean.csv, data/v07_size/*.csv"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
from multiprocessing import Pool
import vcore as vc
HERE = os.path.dirname(os.path.abspath(__file__)); D = _os.path.join(_R, 'data', 'msc_defects_verify')
q = 0.8

def clean(N):
    r = vc.solve_config(np.ones((N, N), bool), q); r['N'] = N; return r

def work(args):
    N, p, rep = args
    rng = np.random.default_rng(np.random.SeedSequence([31337, N, int(round(p * 1e4)), rep])); M = int(round(p * N * N)); att = 0
    while True:
        att += 1
        m = np.zeros(N * N, bool); m[rng.choice(N * N - 2, M, replace=False) + 1] = True; m = m.reshape(N, N)
        if vc.connected(~m, (0, 0), (N - 1, N - 1)): break
    r = vc.solve_config(~m, q); r.update(N=N, p=p, rep=rep, attempts=att); return r

if __name__ == '__main__':
    os.makedirs(_os.path.join(_R, 'data', 'msc_defects_verify', 'v07_size'), exist_ok=True)
    t0 = time.time()
    with Pool(3) as pool:
        fn = _os.path.join(_R, 'data', 'msc_defects_verify', 'v07_clean.csv')
        if not os.path.exists(fn):
            c = pd.DataFrame(pool.map(clean, [10, 15, 20, 25, 35, 50, 70, 100, 140, 200])); c.to_csv(fn, index=False)
        c = pd.read_csv(fn).set_index('N'); print(c[['mfpt', 'mode', 'ratio', 'cv']], flush=True)
        for N, K in ((15, 200), (25, 200), (50, 200), (70, 100), (100, 60), (140, 30)):
            for p in (0.10, 0.20):
                fn = os.path.join(_R, 'data', 'msc_defects_verify', 'v07_size', f'N{N}_p{p:.2f}.csv')
                if not os.path.exists(fn):
                    df = pd.DataFrame(pool.map(work, [(N, p, r) for r in range(K)], chunksize=1)); df.to_csv(fn, index=False)
                df = pd.read_csv(fn); ci = lambda x: 1.96 * x.std(ddof=1) / np.sqrt(len(x))
                print(f'N={N} p={p} K={len(df)} mfpt_fac={df.mfpt.mean() / c.mfpt[N]:.3f}+-{ci(df.mfpt) / c.mfpt[N]:.3f} '
                      f'mode_fac={df["mode"].mean() / c["mode"][N]:.3f}+-{ci(df["mode"]) / c["mode"][N]:.3f} '
                      f'ratio={df.ratio.mean():.4f}+-{ci(df.ratio):.4f} shift={df.ratio.mean() - c.ratio[N]:+.4f} '
                      f'rom_shift={df["mode"].mean() / df.mfpt.mean() - c.ratio[N]:+.4f} [{time.time() - t0:.0f}s]', flush=True)
