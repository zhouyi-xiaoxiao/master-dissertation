"""v06 -- structured placements, own implementation of the geometries of the main implementation (code/msc_defects).
(D) deterministic geometries  (L) M blocked sites confined to a region (K=200, own seeds)
(P) periodic arrays with every phase of the array relative to the lattice.
Output: data/v06_det.csv, v06_local.csv, v06_periodic.csv"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
from multiprocessing import Pool
import vcore as vc
HERE = os.path.dirname(os.path.abspath(__file__)); D = _os.path.join(_R, 'data', 'msc_defects_verify')
N, q = 35, 0.8
mid = N // 2

def vwall(col, gap_lo, g):
    m = np.zeros((N, N), bool); m[col, :] = True; m[col, gap_lo:gap_lo + g] = False; return m

def serp(k, g=3):
    m = np.zeros((N, N), bool)
    for w in range(1, k + 1):
        c = int(round(w * N / (k + 1)))
        m |= vwall(c, N - g if w % 2 == 1 else 0, g)
    return m

def lbox(corner, R, gaps):
    m = np.zeros((N, N), bool)
    m[R, :R + 1] = True; m[:R + 1, R] = True
    for a in gaps: m[R, a] = False
    return m[::-1, ::-1].copy() if corner == 'target' else m

def blk(lo, k):
    m = np.zeros((N, N), bool); m[lo:lo + k, lo:lo + k] = True; return m

def per(a, oi, oj):
    m = np.zeros((N, N), bool); m[oi::a, oj::a] = True; m[0, 0] = False; m[-1, -1] = False; return m

def det():
    c = {'clean': np.zeros((N, N), bool)}
    for g in (1, 3, 9): c[f'wall_mid_gap_centre_g{g}'] = vwall(mid, mid - g // 2, g)
    c['wall_mid_gap_startedge_g3'] = vwall(mid, 0, 3); c['wall_mid_gap_targetedge_g3'] = vwall(mid, N - 3, 3)
    c['wall_nearstart_gap_centre_g3'] = vwall(5, mid - 1, 3); c['wall_neartarget_gap_centre_g3'] = vwall(N - 6, mid - 1, 3)
    for k in (1, 2, 3, 4, 6): c[f'serpentine_k{k}_g3'] = serp(k)
    for R in (3, 5, 8):
        c[f'target_box_R{R}_g1'] = lbox('target', R, [1]); c[f'start_box_R{R}_g1'] = lbox('start', R, [1])
    c['target_box_R5_g3'] = lbox('target', 5, [0, 1, 2]); c['start_box_R5_g3'] = lbox('start', 5, [0, 1, 2])
    c['both_box_R5_g1'] = lbox('target', 5, [1]) | lbox('start', 5, [1])
    c['block6_near_target'] = blk(N - 9, 6); c['block6_centre'] = blk(mid - 3, 6); c['block6_near_start'] = blk(3, 6)
    c['block12_centre'] = blk(mid - 6, 12)
    c['periodic_a4'] = per(4, 1, 1); c['periodic_a3'] = per(3, 2, 2); c['periodic_a2'] = per(2, 1, 1)
    return c

def region(kind):
    ii, jj = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    if kind == 'near_target': sel = (ii - N + 1) ** 2 + (jj - N + 1) ** 2 <= 144
    elif kind == 'near_start': sel = ii ** 2 + jj ** 2 <= 144
    elif kind == 'centre': sel = (ii - mid) ** 2 + (jj - mid) ** 2 <= 6.2 ** 2
    else: sel = np.ones((N, N), bool)
    sel[0, 0] = False; sel[-1, -1] = False
    return np.flatnonzero(sel.ravel())
KID = {'near_target': 1, 'near_start': 2, 'centre': 3, 'anywhere': 4}

def local(args):
    kind, M, rep = args
    sites = region(kind)
    rng = np.random.default_rng(np.random.SeedSequence([55555, KID[kind], M, rep])); att = 0
    while True:
        att += 1
        m = np.zeros(N * N, bool); m[rng.choice(sites, M, replace=False)] = True; m = m.reshape(N, N)
        if vc.connected(~m, (0, 0), (N - 1, N - 1)): break
    r = vc.solve_config(~m, q)
    r.update(kind=kind, M=M, rep=rep, attempts=att, region_sites=len(sites))
    return r

if __name__ == '__main__':
    rows = []
    for name, m in det().items():
        r = vc.solve_config(~m, q); r.update(name=name, blocked=int(m.sum())); rows.append(r)
    d = pd.DataFrame(rows); c = d.iloc[0]
    d['mfpt_fac'] = d.mfpt / c.mfpt; d['mode_fac'] = d['mode'] / c['mode']
    d.to_csv(_os.path.join(_R, 'data', 'msc_defects_verify', 'v06_det.csv'), index=False)
    tr = pd.read_csv(_os.path.join(_R, 'data', 'msc_defects', 'structured_deterministic.csv')).set_index('name')
    d['tr_mfpt'] = d.name.map(tr.mfpt); d['tr_mode'] = d.name.map(tr['mode']); d['tr_blocked'] = d.name.map(tr.n_blocked)
    pd.set_option('display.width', 250)
    print(d[['name', 'blocked', 'tr_blocked', 'n', 'mfpt', 'tr_mfpt', 'mode', 'tr_mode', 'ratio', 'cv', 'mfpt_fac', 'mode_fac']].to_string())
    print('max rel diff mfpt', np.max(np.abs(d.mfpt / d.tr_mfpt - 1)), ' modes equal:', int((d['mode'] == d.tr_mode).sum()), '/', len(d))
    # periodic, all phases
    rows = []
    for a in (2, 3, 4):
        for oi in range(a):
            for oj in range(a):
                m = per(a, oi, oj)
                if not vc.connected(~m, (0, 0), (N - 1, N - 1)): continue
                r = vc.solve_config(~m, q); r.update(a=a, oi=oi, oj=oj, blocked=int(m.sum()), p=m.sum() / N ** 2,
                                                     mfpt_fac=r['mfpt'] / c.mfpt, mode_fac=r['mode'] / c['mode'],
                                                     tgt_nbr_blocked=int(m[N - 2, N - 1]) + int(m[N - 1, N - 2])); rows.append(r)
    pp = pd.DataFrame(rows); pp.to_csv(_os.path.join(_R, 'data', 'msc_defects_verify', 'v06_periodic.csv'), index=False)
    print(pp[['a', 'oi', 'oj', 'blocked', 'p', 'tgt_nbr_blocked', 'mfpt_fac', 'mode_fac', 'ratio']].round(4).to_string())
    jobs = [(k, M, r) for k in KID for M in (10, 20, 30, 40) for r in range(200)]
    with Pool(3) as pool:
        recs = pool.map(local, jobs, chunksize=8)
    L = pd.DataFrame(recs); L.to_csv(_os.path.join(_R, 'data', 'msc_defects_verify', 'v06_local.csv'), index=False)
    ci = lambda x: 1.96 * x.std(ddof=1) / np.sqrt(len(x))
    for (k, M), g in L.groupby(['kind', 'M']):
        print(f'{k:12s} M={M} acc={len(g) / g.attempts.sum():.3f} mfpt_fac={g.mfpt.mean() / c.mfpt:.4f}+-{ci(g.mfpt) / c.mfpt:.4f} '
              f'mode_fac={g["mode"].mean() / c["mode"]:.4f}+-{ci(g["mode"]) / c["mode"]:.4f} ratio={g.ratio.mean():.4f}+-{ci(g.ratio):.4f} '
              f'frac_below={np.mean(g.mfpt < c.mfpt):.3f} elasticity={np.log(g["mode"].mean() / c["mode"]) / np.log(g.mfpt.mean() / c.mfpt):.3f}')
