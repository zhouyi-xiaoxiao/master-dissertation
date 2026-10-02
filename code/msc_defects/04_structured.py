"""04_structured.py -- structured (non-random) and spatially localised defect placements.

N=35, q=0.8, start (0,0), target (N-1,N-1).  Two families:
 (D) deterministic geometries: walls with gaps, serpentines, enclosures of the target / start,
     compact blocks, periodic obstacle arrays -> data/structured_deterministic.csv (+ PMFs in
     data/structured_pmfs.npz)
 (L) random placements of M blocked sites confined to a region (near target / near start / centre
     / anywhere), K placements each -> data/structured_local.csv
All statistics exact (certified mode).  Seeds fixed.
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
import numpy as np, pandas as pd
from multiprocessing import Pool
import fptcore as fc
import ensemble

N, q = 35, 0.8
S, T = (0, 0), (N - 1, N - 1)
DATA = _os.path.join(_R, 'data', 'msc_defects')


def wall(c, gap_lo, g, axis=0):
    m = np.zeros((N, N), bool)
    idx = [j for j in range(N) if not (gap_lo <= j < gap_lo + g)]
    if axis == 0:
        m[c, idx] = True
    else:
        m[idx, c] = True
    return m


def serpentine(k, g=3):
    m = np.zeros((N, N), bool)
    for w in range(1, k + 1):
        c = int(round(w * N / (k + 1)))
        gap_lo = N - g if w % 2 == 1 else 0
        m |= wall(c, gap_lo, g)
    return m


def box(corner, R, gap_sites):
    """L-shaped wall at Chebyshev distance R from a corner; gap_sites = offsets along one arm."""
    m = np.zeros((N, N), bool)
    for a in range(R + 1):
        m[R, a] = True; m[a, R] = True
    for a in gap_sites:
        m[R, a] = False
    if corner == 'target':
        m = m[::-1, ::-1]
    return m


def block(lo, k):
    m = np.zeros((N, N), bool); m[lo:lo + k, lo:lo + k] = True
    return m


def periodic(a, off):
    m = np.zeros((N, N), bool)
    ii = np.arange(N)
    sel = ii[ii % a == off]
    m[np.ix_(sel, sel)] = True
    return m


def deterministic_configs():
    c = {}
    c['clean'] = np.zeros((N, N), bool)
    mid = N // 2
    for g in (1, 3, 9):
        c[f'wall_mid_gap_centre_g{g}'] = wall(mid, mid - g // 2, g)
    c['wall_mid_gap_startedge_g3'] = wall(mid, 0, 3)
    c['wall_mid_gap_targetedge_g3'] = wall(mid, N - 3, 3)
    c['wall_nearstart_gap_centre_g3'] = wall(5, mid - 1, 3)
    c['wall_neartarget_gap_centre_g3'] = wall(N - 6, mid - 1, 3)
    for k in (1, 2, 3, 4, 6):
        c[f'serpentine_k{k}_g3'] = serpentine(k)
    for R in (3, 5, 8):
        c[f'target_box_R{R}_g1'] = box('target', R, [1])
        c[f'start_box_R{R}_g1'] = box('start', R, [1])
    c['target_box_R5_g3'] = box('target', 5, [0, 1, 2])
    c['start_box_R5_g3'] = box('start', 5, [0, 1, 2])
    c['both_box_R5_g1'] = box('target', 5, [1]) | box('start', 5, [1])
    c['block6_near_target'] = block(N - 9, 6)
    c['block6_centre'] = block(mid - 3, 6)
    c['block6_near_start'] = block(3, 6)
    c['block12_centre'] = block(mid - 6, 12)
    c['periodic_a4'] = periodic(4, 1)
    c['periodic_a3'] = periodic(3, 2)
    c['periodic_a2'] = periodic(2, 1)
    return c


def region_sites(kind, radius):
    ii, jj = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    if kind == 'near_target':
        d2 = (ii - (N - 1)) ** 2 + (jj - (N - 1)) ** 2
    elif kind == 'near_start':
        d2 = ii ** 2 + jj ** 2
    elif kind == 'centre':
        d2 = (ii - N // 2) ** 2 + (jj - N // 2) ** 2
    else:
        d2 = np.zeros((N, N))
    sel = d2 <= radius ** 2
    sel[S] = False; sel[T] = False
    return np.flatnonzero(sel.ravel())


REGIONS = {'near_target': 12.0, 'near_start': 12.0, 'centre': 6.2, 'anywhere': 1e9}
KIND_ID = {'near_target': 1, 'near_start': 2, 'centre': 3, 'anywhere': 4}


def local_worker(args):
    kind, M, rep = args
    sites = region_sites(kind, REGIONS[kind])
    rng = np.random.default_rng(np.random.SeedSequence([4242, KIND_ID[kind], M, rep]))
    attempts = 0
    while True:
        attempts += 1
        m = np.zeros(N * N, bool)
        m[rng.choice(sites, M, replace=False)] = True
        m = m.reshape(N, N)
        if fc.connected(~m, S, T):
            break
    rec = ensemble.analyze(m, q)
    rec.update(kind=kind, M=M, rep=rep, attempts=attempts, region_sites=int(sites.size))
    return rec


def main():
    t0 = time.time()
    rows = []
    pmfs = {}
    masks = {}
    for name, m in deterministic_configs().items():
        assert not m[S] and not m[T]
        rec = ensemble.analyze(m, q)
        rec['name'] = name
        rows.append(rec)
        summ, ch = fc.fpt_summary(~m, q, S, T, want_series=True, series_tmax=int(5 * rec['mfpt']))
        t = summ['series_t']; f = summ['series_f']
        # thin to <= 6000 points (keep every point up to 3x mode, then geometric thinning)
        keep = np.unique(np.concatenate([np.arange(0, min(len(t), 3 * rec['mode'])),
                                         np.geomspace(1, len(t), 3000).astype(int) - 1]))
        if keep.size > 12000:
            keep = np.unique(np.concatenate([np.linspace(0, min(len(t), 3 * rec['mode']) - 1, 6000).astype(int),
                                             np.geomspace(1, len(t), 3000).astype(int) - 1]))
        pmfs[name + '__t'] = t[keep]; pmfs[name + '__f'] = f[keep]
        masks[name] = m
        print(f"{name:32s} M={rec['n_blocked']:4d} n={rec['n_cluster']:4d} MFPT={rec['mfpt']:10.1f} "
              f"mode={rec['mode']:7d} ratio={rec['ratio']:.4f} CV={rec['cv']:.4f} med/MFPT={rec['median']/rec['mfpt']:.3f} "
              f"tau_rel={rec['tau_rel']:.0f} chem={rec['chem_dist']}", flush=True)
    pd.DataFrame(rows).to_csv(_os.path.join(_R, 'data', 'msc_defects', 'structured_deterministic.csv'), index=False)
    np.savez_compressed(_os.path.join(_R, 'data', 'msc_defects', 'structured_pmfs.npz'), **pmfs)
    np.savez_compressed(_os.path.join(_R, 'data', 'msc_defects', 'structured_masks.npz'), **masks)

    fn = _os.path.join(_R, 'data', 'msc_defects', 'structured_local.csv')
    K = 200
    jobs = [(kind, M, rep) for kind in REGIONS for M in (10, 20, 30, 40) for rep in range(K)]
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 5) as pool:
        recs = pool.map(local_worker, jobs, chunksize=20)
    df = pd.DataFrame(recs)
    df.to_csv(fn, index=False)
    g = df.groupby(['kind', 'M']).agg(mfpt=('mfpt', 'mean'), mode=('mode', 'mean'), ratio=('ratio', 'mean'),
                                       ratio_sem=('ratio', lambda x: x.std(ddof=1) / np.sqrt(len(x))),
                                       acc=('attempts', lambda x: len(x) / x.sum()))
    print(g.to_string())
    print(f'done in {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
