"""06_sensitivity.py -- single-defect susceptibility maps (exact first-order / dilute theory).

For every site x other than start and target, block x alone and compute the exact MFPT, SD and
certified mode.  chi_A(x) = [A(x) - A(clean)] / A(clean) for A = MFPT, mode.  In the dilute limit a
placement law with site weights w(x) (sum_x w = 1) gives
        d ln<A> / dp |_{p=0}  =  N^2 * sum_x w(x) chi_A(x).
The mode is an integer; a sub-step peak position is also recorded from the parabola through
f(m-1), f(m), f(m+1) so that single-defect shifts of a few steps are resolved.
Output: data/sensitivity_N{N}.csv (one row per blocked site).   usage: 06_sensitivity.py [N] [workers]
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
import numpy as np, pandas as pd
from multiprocessing import Pool
import fptcore as fc

q = 0.8
DATA = _os.path.join(_R, 'data', 'msc_defects')

def one(args):
    N, x = args
    om = np.ones((N, N), bool)
    if x is not None:
        om[x] = False
    ch = fc.build_chain(om, q, (0, 0), (N - 1, N - 1))
    mf, var, _ = fc.moments(ch['Q'], ch['s'])
    S = fc.SpectralFPT(ch['Q'], ch['r'], ch['s'])
    mode, fm, tc = S.mode()
    f3 = S.f(np.array([mode - 1, mode, mode + 1]))
    den = f3[0] - 2 * f3[1] + f3[2]
    mode_interp = mode + 0.5 * (f3[0] - f3[2]) / den if den != 0 else float(mode)
    tp = fc.tetali_parts_sparse(ch, q)
    mu2, tau_rel = fc.reflecting_relaxation(ch)
    return dict(N=N, i=-1 if x is None else x[0], j=-1 if x is None else x[1], mfpt=mf,
                sd=float(np.sqrt(var)), mode=int(mode), mode_interp=float(mode_interp), f_mode=fm,
                G_tt=tp['G_tt'], G_st=tp['G_st'], tau_rel=tau_rel, median=int(S.quantile(0.5, mf)))

if __name__ == '__main__':
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 35
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    t0 = time.time()
    sites = [None] + [(i, j) for i in range(N) for j in range(N) if (i, j) not in ((0, 0), (N - 1, N - 1))]
    with Pool(workers) as pool:
        recs = pool.map(one, [(N, x) for x in sites], chunksize=16)
    df = pd.DataFrame(recs)
    c = df.iloc[0]
    df['chi_mfpt'] = df.mfpt / c.mfpt - 1
    df['chi_mode'] = df.mode_interp / c.mode_interp - 1
    df['chi_sd'] = df.sd / c.sd - 1
    df.to_csv(os.path.join(DATA, f'sensitivity_N{N}.csv'), index=False)
    d = df.iloc[1:]
    print(f'N={N} clean MFPT={c.mfpt:.3f} mode={int(c["mode"])} (interp {c.mode_interp:.3f})')
    print('uniform-weight slopes  d ln MFPT/dp = %.4f   d ln mode/dp = %.4f   (pi-1 = %.4f)' %
          (N * N * d.chi_mfpt.mean(), N * N * d.chi_mode.mean(), np.pi - 1))
    print('fraction of sites whose blocking LOWERS the MFPT: %.3f ; lowers the mode: %.3f' %
          ((d.chi_mfpt < 0).mean(), (d.chi_mode < 0).mean()))
    print('max chi_mfpt = %.4f at (%d,%d); max chi_mode = %.4f at (%d,%d)' % (
        d.chi_mfpt.max(), *d.loc[d.chi_mfpt.idxmax(), ['i', 'j']], d.chi_mode.max(), *d.loc[d.chi_mode.idxmax(), ['i', 'j']]))
    print(f'{time.time()-t0:.0f}s')
