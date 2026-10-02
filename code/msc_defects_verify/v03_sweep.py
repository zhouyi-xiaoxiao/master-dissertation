"""v03 -- second density sweep (separately written code), N=35, q=0.8, K placements per p, own seeds, own solver.
usage: python v03_sweep.py K workers [scheme ...]
Schemes: 'uniform' (M=round(pN^2) sites uniform over all sites but the two corners),
         'cornerthinned'  (corner-thinned placement law, generator generate_defects_smart of code/msc_defects/ensemble.py).
Rejection until start and target are connected.  One CSV per (scheme, p) in data/v03_sweep/."""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
from multiprocessing import Pool
import vcore as vc
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _os.path.join(_R, 'code', 'msc_defects'))
from ensemble import generate_defects_smart as gen     # corner-thinned placement law (Section 8 of the article)
P_GRID = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.25, 0.30, 0.35]
q = 0.8
SID = {'uniform': 11, 'cornerthinned': 22}


def draw(N, p, scheme, rng):
    M = int(round(p * N * N)); att = 0
    while True:
        att += 1
        if scheme == 'uniform':
            sites = rng.choice(N * N - 2, M, replace=False) + 1
            mask = np.zeros(N * N, bool); mask[sites] = True; mask = mask.reshape(N, N)
        else:
            mask = gen(N, p, int(rng.integers(0, 2 ** 31 - 1)))
        if vc.connected(~mask, (0, 0), (N - 1, N - 1)):
            return mask, att


def work(args):
    N, p, scheme, rep = args
    rng = np.random.default_rng(np.random.SeedSequence([987654321, N, SID[scheme], int(round(p * 1e4)), rep]))
    mask, att = draw(N, p, scheme, rng)
    rec = vc.solve_config(~mask, q, extras=True)
    om = ~mask
    rec.update(N=N, p=p, scheme=scheme, rep=rep, attempts=att, n_blocked=int(mask.sum()),
               deg_t=int(om[N - 2, N - 1]) + int(om[N - 1, N - 2]),
               blk_t6=int(mask[N - 7:, N - 7:].sum()), blk_s6=int(mask[:7, :7].sum()))
    return rec


def main():
    K = int(sys.argv[1]); workers = int(sys.argv[2]); schemes = sys.argv[3:] or ['uniform', 'cornerthinned']
    N = int(os.environ.get('NLAT', 35))
    pgrid = [float(x) for x in os.environ['PGRID'].split(',')] if os.environ.get('PGRID') else P_GRID
    out = _os.path.join(_R, 'data', 'msc_defects_verify', 'v03_sweep'); os.makedirs(out, exist_ok=True)
    t0 = time.time()
    with Pool(workers) as pool:
        for p in pgrid:
            for scheme in schemes:
                fn = os.path.join(out, f'N{N}_{scheme}_p{p:.2f}.csv')
                if os.path.exists(fn):
                    continue
                recs = pool.map(work, [(N, p, scheme, r) for r in range(K)], chunksize=4)
                df = pd.DataFrame(recs); df.to_csv(fn + '.tmp', index=False); os.replace(fn + '.tmp', fn)
                print(f'{scheme} p={p:.2f} mfpt={df.mfpt.mean():.1f} mode={df["mode"].mean():.1f} '
                      f'ratio={df.ratio.mean():.4f}+-{vc.ci95(df.ratio):.4f} acc={K / df.attempts.sum():.3f} '
                      f'[{time.time() - t0:.0f}s]', flush=True)

if __name__ == '__main__':
    main()
