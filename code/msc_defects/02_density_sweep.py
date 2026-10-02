"""02_density_sweep.py -- exact defect-density sweep with many random placements.

usage: python 02_density_sweep.py N K [workers] [scheme ...]
For each scheme in {uniform, smart} and each defect fraction p, draws K placements with exactly
M=round(p N^2) blocked sites, conditioned on start-target connectivity (rejection), and records the
exact MFPT, SD, certified mode, quantiles, Tetali resistance parts, ... for every placement.
Checkpoint: one CSV per (N, scheme, p) in data/sweep/ ; finished files are skipped on re-run.
Seeds: SeedSequence([20261001, N, scheme_id, round(1e4 p), rep]) -> fully deterministic.
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
import numpy as np, pandas as pd
from multiprocessing import Pool
import ensemble

P_GRID = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.25, 0.30, 0.35]
q = 0.8

def main():
    N = int(sys.argv[1]); K = int(sys.argv[2])
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    schemes = sys.argv[4:] or ['uniform', 'smart']
    pgrid = P_GRID
    if os.environ.get('PGRID'):
        pgrid = [float(x) for x in os.environ['PGRID'].split(',')]
    outdir = _os.path.join(_R, 'data', 'msc_defects', 'sweep')
    os.makedirs(outdir, exist_ok=True)
    t0 = time.time()
    with Pool(workers) as pool:
        for scheme in schemes:
            for p in pgrid:
                fn = os.path.join(outdir, f'sweep_N{N}_{scheme}_p{p:.2f}.csv')
                if os.path.exists(fn):
                    continue
                k = 1 if p == 0 else K
                jobs = [(N, q, p, scheme, rep) for rep in range(k)]
                recs = pool.map(ensemble.worker, jobs, chunksize=max(1, k // (workers * 8)))
                df = pd.DataFrame(recs)
                df.to_csv(fn + '.tmp', index=False)
                os.replace(fn + '.tmp', fn)
                print(f'N={N} {scheme} p={p:.2f} K={k} MFPT={df.mfpt.mean():.1f} mode={df["mode"].mean():.1f} '
                      f'ratio={df.ratio.mean():.4f}+-{df.ratio.std(ddof=1)/np.sqrt(k) if k>1 else 0:.4f} '
                      f'acc={k/df.attempts.sum():.3f} [{time.time()-t0:.0f}s]', flush=True)

if __name__ == '__main__':
    main()
