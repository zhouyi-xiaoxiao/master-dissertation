"""v05 -- single-defect susceptibility maps (own code): block each admissible site alone, exact MFPT and
(sub-step) mode.  usage: python v05_sensitivity.py N workers.   Output: data/v05_sens_N{N}.csv + json summary."""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
from multiprocessing import Pool
import vcore as vc
HERE = os.path.dirname(os.path.abspath(__file__))
q = 0.8

def one(args):
    N, i, j = args
    om = np.ones((N, N), bool); om[i, j] = False
    r = vc.solve_config(om, q)
    return dict(i=i, j=j, mfpt=r['mfpt'], mode=r['mode'], peak_sub=r['peak_sub'], n=r['n'])

def cornerthinned_weights(N):
    w = np.ones((N, N)); radius = max(3, N // 8)
    for i in range(radius):
        for j in range(radius):
            w[i, j] *= np.exp(-2 * (radius - np.sqrt(i ** 2 + j ** 2)) / radius)
    for i in range(N - radius, N):
        for j in range(N - radius, N):
            w[i, j] *= np.exp(-2 * (radius - np.sqrt((N - 1 - i) ** 2 + (N - 1 - j) ** 2)) / radius)
    w[0, 0] = 0; w[N - 1, N - 1] = 0
    return w / w.sum()

def main():
    N = int(sys.argv[1]); workers = int(sys.argv[2])
    fn = os.path.join(_R, 'data', 'msc_defects_verify', f'v05_sens_N{N}.csv')
    c = vc.solve_config(np.ones((N, N), bool), q)
    if not os.path.exists(fn):
        jobs = [(N, i, j) for i in range(N) for j in range(N) if (i, j) not in ((0, 0), (N - 1, N - 1))]
        with Pool(workers) as pool:
            recs = pool.map(one, jobs, chunksize=8)
        pd.DataFrame(recs).to_csv(fn, index=False)
    d = pd.read_csv(fn)
    d['chi_mfpt'] = d.mfpt / c['mfpt'] - 1; d['chi_mode'] = d.peak_sub / c['peak_sub'] - 1
    d['chi_mode_int'] = d['mode'] / c['mode'] - 1
    dt = np.sqrt((N - 1 - d.i) ** 2 + (N - 1 - d.j) ** 2)
    W = cornerthinned_weights(N); wu = np.ones((N, N)); wu[0, 0] = wu[-1, -1] = 0; wu /= wu.sum()
    ii = d.i.values; jj = d.j.values
    out = dict(N=N, clean_mfpt=c['mfpt'], clean_mode=c['mode'], clean_peak_sub=c['peak_sub'],
               frac_lower_mfpt=float((d.chi_mfpt < 0).mean()), frac_lower_mode=float((d.chi_mode < 0).mean()),
               frac_lower_mode_integer=float((d.chi_mode_int < 0).mean()),
               max_chi_mfpt=float(d.chi_mfpt.max()), argmax_mfpt=[int(d.i[d.chi_mfpt.idxmax()]), int(d.j[d.chi_mfpt.idxmax()])],
               max_chi_mode=float(d.chi_mode.max()), max_chi_mode_integer=float(d.chi_mode_int.max()),
               min_chi_mfpt=float(d.chi_mfpt.min()), argmin_mfpt=[int(d.i[d.chi_mfpt.idxmin()]), int(d.j[d.chi_mfpt.idxmin()])],
               minus_one_over_n=-1.0 / (N * N),
               slope_mfpt_uniform=float(N * N * np.sum(wu[ii, jj] * d.chi_mfpt)), slope_mode_uniform=float(N * N * np.sum(wu[ii, jj] * d.chi_mode)),
               slope_mode_uniform_integer=float(N * N * np.sum(wu[ii, jj] * d.chi_mode_int)),
               slope_mfpt_cornerthinned=float(N * N * np.sum(W[ii, jj] * d.chi_mfpt)), slope_mode_cornerthinned=float(N * N * np.sum(W[ii, jj] * d.chi_mode)))
    for r in (3, 4, 6, 10):
        out[f'share_mfpt_within_{r}'] = float(d.chi_mfpt[dt <= r].sum() / d.chi_mfpt.sum())
        out[f'share_mode_within_{r}'] = float(d.chi_mode[dt <= r].sum() / d.chi_mode.sum())
    # the 4x4 corner boxes actually thinned by the corner-thinned weights
    box_t = (d.i >= N - max(3, N // 8)) & (d.j >= N - max(3, N // 8)); box_s = (d.i < max(3, N // 8)) & (d.j < max(3, N // 8))
    out['share_mfpt_in_target_box'] = float(d.chi_mfpt[box_t].sum() / d.chi_mfpt.sum()); out['share_mode_in_target_box'] = float(d.chi_mode[box_t].sum() / d.chi_mode.sum())
    out['share_mfpt_in_start_box'] = float(d.chi_mfpt[box_s].sum() / d.chi_mfpt.sum()); out['share_mode_in_start_box'] = float(d.chi_mode[box_s].sum() / d.chi_mode.sum())
    out['cornerthinned_weight_ratio_in_target_box'] = float(W[N - max(3, N // 8):, N - max(3, N // 8):].sum() / wu[N - max(3, N // 8):, N - max(3, N // 8):].sum())
    d.to_csv(fn, index=False)
    json.dump(out, open(os.path.join(_R, 'data', 'msc_defects_verify', f'v05_sens_N{N}.json'), 'w'), indent=1)
    print(json.dumps(out, indent=1))

if __name__ == '__main__':
    main()
