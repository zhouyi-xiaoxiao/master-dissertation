"""05_size_dependence.py -- how the defect effect depends on lattice size N.

(A) clean lattice, corner-to-corner, q=0.8, N = 10..200: exact MFPT, certified mode, median.
(B) uniform random blocked sites at p = 0.10 and 0.20 for N = 15, 25, 35, 50, 70, 100, 140
    (K placements each, conditioned on start-target connectivity).
All by the time-stepping route (timestep_summary); N=35 reproduces the dense-route sweep exactly
because the seeds coincide.  Output: data/size_clean.csv, data/size_defects.csv (checkpointed per
(N,p) in data/size/).
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
import numpy as np, pandas as pd
from multiprocessing import Pool
import fptcore as fc, ensemble

q = 0.8
DATA = _os.path.join(_R, 'data', 'msc_defects')
os.makedirs(_os.path.join(_R, 'data', 'msc_defects', 'size'), exist_ok=True)

def clean(N):
    summ, ch = fc.timestep_summary(np.ones((N, N), bool), q, (0, 0), (N - 1, N - 1))
    tp = fc.tetali_parts_sparse(ch, q)
    summ.update(N=N, G_tt=tp['G_tt'], G_st=tp['G_st'])
    return summ

if __name__ == '__main__':
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    t0 = time.time()
    with Pool(workers) as pool:
        fn = _os.path.join(_R, 'data', 'msc_defects', 'size_clean.csv')
        if not os.path.exists(fn):
            Ns = [10, 15, 20, 25, 35, 50, 70, 100, 140, 200]
            recs = pool.map(clean, Ns, chunksize=1)
            pd.DataFrame(recs).to_csv(fn, index=False)
            for r in recs:
                print(f"clean N={r['N']:4d} MFPT={r['mfpt']:.3f} mode={r['mode']} MFPT/mode={r['mfpt']/r['mode']:.4f} "
                      f"lnN={np.log(r['N']):.4f}", flush=True)
        plan = [(15, 200), (25, 200), (35, 200), (50, 200), (70, 100), (100, 80), (140, 40)]
        for N, K in plan:
            for p in (0.10, 0.20):
                f2 = os.path.join(_R, 'data', 'msc_defects', 'size', f'size_N{N}_p{p:.2f}.csv')
                if os.path.exists(f2):
                    continue
                recs = pool.map(ensemble.worker_large, [(N, q, p, 'uniform', rep) for rep in range(K)], chunksize=1)
                df = pd.DataFrame(recs)
                df.to_csv(f2 + '.tmp', index=False); os.replace(f2 + '.tmp', f2)
                print(f"N={N} p={p} K={K} MFPT={df.mfpt.mean():.1f} mode={df['mode'].mean():.1f} "
                      f"ratio={df.ratio.mean():.4f}+-{df.ratio.std(ddof=1)/np.sqrt(K):.4f} [{time.time()-t0:.0f}s]", flush=True)
    import glob
    pd.concat([pd.read_csv(f) for f in sorted(glob.glob(_os.path.join(_R, 'data', 'msc_defects', 'size', '*.csv')))]).to_csv(
        _os.path.join(_R, 'data', 'msc_defects', 'size_defects.csv'), index=False)
