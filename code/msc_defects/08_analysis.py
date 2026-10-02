"""08_analysis.py -- aggregate the raw per-placement records into the summary tables of the defect study.

Reads data/sweep/*.csv, data/homogenization.json, data/sensitivity_N35.csv (if present),
data/structured_*.csv, data/size_*.csv.  Writes
  data/summary_sweep_N35.csv      mean +/- 95% CI per (scheme, p), time factors, EMA comparison
  data/summary_correlations.csv   which configuration descriptors control mean / mode
  data/summary_stratified.csv     stratification by the number of open neighbours of the target
  data/summary_first_order.json   dilute slopes: measured vs single-defect susceptibility vs pi-1
  data/summary_size.csv           finite-size table
Light: no chain is solved here.
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import os, sys, json, glob
import numpy as np, pandas as pd
from scipy import stats

DATA = _os.path.join(_R, 'data', 'msc_defects')
rng = np.random.default_rng(12345)

def ci(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return 0.0
    return float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))

def boot_ratio_of_means(a, b, B=4000):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) < 2:
        return float(a.mean() / b.mean()), 0.0, 0.0
    idx = rng.integers(0, len(a), (B, len(a)))
    r = a[idx].mean(1) / b[idx].mean(1)
    return float(a.mean() / b.mean()), float(np.quantile(r, 0.025)), float(np.quantile(r, 0.975))

hom = json.load(open(_os.path.join(_R, 'data', 'msc_defects', 'homogenization.json')))
hp = np.array([r['p'] for r in hom['finite_p']]); htf = np.array([r['time_factor'] for r in hom['finite_p']])
hsig = np.array([r['sigma'] for r in hom['finite_p']]); hpinf = np.array([r['P_inf'] for r in hom['finite_p']])
def hom_tf(p):
    return float(np.interp(p, hp, htf))

sweep = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(_os.path.join(_R, 'data', 'msc_defects', 'sweep', 'sweep_N35_*.csv')))],
                  ignore_index=True)
base = sweep[sweep.p == 0].iloc[0]
rows = []
for (scheme, p), g in sweep.groupby(['scheme', 'p']):
    K = len(g)
    r = dict(scheme=scheme, p=p, K=K, acceptance=K / g.attempts.sum(),
             pocket_free_frac=g.pocket_free.mean())
    for k in ('mfpt', 'mode', 'ratio', 'sd', 'cv', 'median', 'n_cluster', 'G_tt', 'G_st', 'tau_rel',
              'S_mode', 'S_mfpt', 'chem_dist', 'A1', 'tau1'):
        r[k] = float(g[k].mean()); r[k + '_ci'] = ci(g[k]); r[k + '_std'] = float(g[k].std(ddof=1)) if K > 1 else 0.0
    r['median_over_mfpt'] = float((g['median'] / g.mfpt).mean())
    rom, lo, hi = boot_ratio_of_means(g['mode'], g.mfpt)
    r['ratio_of_means'] = rom; r['ratio_of_means_lo'] = lo; r['ratio_of_means_hi'] = hi
    r['mfpt_factor'] = r['mfpt'] / base.mfpt; r['mfpt_factor_ci'] = r['mfpt_ci'] / base.mfpt
    r['mode_factor'] = r['mode'] / base['mode']; r['mode_factor_ci'] = r['mode_ci'] / base['mode']
    r['tau_rel_factor'] = r['tau_rel'] / base.tau_rel
    r['hom_factor'] = hom_tf(p)
    r['dilute_factor'] = 1 / (1 - (np.pi - 1) * p) if (np.pi - 1) * p < 1 else np.nan
    # exact Tetali split of the mean: MFPT = (4/q) n (G_tt - G_st)
    r['n_factor'] = r['n_cluster'] / base.n_cluster
    r['Gdiff_factor'] = float((g.G_tt - g.G_st).mean() / (base.G_tt - base.G_st))
    r['tetali_max_rel_err'] = float(np.max(np.abs(g.mfpt_tetali / g.mfpt - 1)))
    # geometric-mean (typical) values
    r['mfpt_geo'] = float(np.exp(np.log(g.mfpt).mean())); r['mode_geo'] = float(np.exp(np.log(g['mode']).mean()))
    rows.append(r)
summ = pd.DataFrame(rows).sort_values(['scheme', 'p'])
summ.to_csv(_os.path.join(_R, 'data', 'msc_defects', 'summary_sweep_N35.csv'), index=False)
pd.set_option('display.width', 250)
print(summ[['scheme', 'p', 'K', 'acceptance', 'mfpt', 'mfpt_ci', 'mode', 'mode_ci', 'ratio', 'ratio_ci',
            'ratio_of_means', 'cv', 'mfpt_factor', 'mode_factor', 'hom_factor']].to_string(index=False))

# ---- monotone trend test of the ratio vs p (Spearman over all placements, p<=0.2)
trend = {}
for scheme in ('uniform', 'smart'):
    g = sweep[(sweep.scheme == scheme) & (sweep.p <= 0.2001)]
    rho, pv = stats.spearmanr(g.p, g.ratio)
    sl = stats.linregress(g.p, g.ratio)
    trend[scheme] = dict(spearman_rho=float(rho), p_value=float(pv), ols_slope=float(sl.slope),
                         ols_slope_se=float(sl.stderr), n=int(len(g)))
print(json.dumps(trend, indent=1))

# ---- correlations at fixed p
desc = ['n_cluster', 'G_tt', 'G_st', 'G_ss', 'tau_rel', 'blk_t_r3', 'blk_t_r6', 'blk_s_r3', 'blk_s_r6',
        'deg_t', 'deg_s', 'chem_dist']
rows = []
for scheme in ('uniform', 'smart'):
    for p in (0.10, 0.20):
        g = sweep[(sweep.scheme == scheme) & np.isclose(sweep.p, p)]
        for y in ('mfpt', 'mode', 'ratio'):
            for x in desc:
                if g[x].std() == 0:
                    continue
                rho = stats.spearmanr(g[x], g[y])[0]
                rows.append(dict(scheme=scheme, p=p, y=y, x=x, spearman=float(rho)))
        # two-variable log-linear regressions
        for y in ('mfpt', 'mode'):
            X = np.column_stack([np.ones(len(g)), np.log(g.n_cluster * g.G_tt), np.log(g.tau_rel)])
            beta, res, rk, sv = np.linalg.lstsq(X, np.log(g[y]), rcond=None)
            pred = X @ beta
            r2 = 1 - np.sum((np.log(g[y]) - pred) ** 2) / np.sum((np.log(g[y]) - np.log(g[y]).mean()) ** 2)
            rows.append(dict(scheme=scheme, p=p, y='log ' + y, x='log(n G_tt), log(tau_rel) [exponents]',
                             spearman=np.nan, exp_nGtt=float(beta[1]), exp_taurel=float(beta[2]), R2=float(r2)))
corr = pd.DataFrame(rows)
corr.to_csv(_os.path.join(_R, 'data', 'msc_defects', 'summary_correlations.csv'), index=False)
print(corr[corr.x.str.startswith('log')].to_string(index=False))
print(corr[(corr.scheme == 'uniform') & (corr.p == 0.2) & corr.spearman.notna()].pivot(index='x', columns='y', values='spearman').round(3).to_string())

# ---- stratify by target degree
rows = []
for scheme in ('uniform', 'smart'):
    for p in (0.10, 0.20, 0.30):
        g = sweep[(sweep.scheme == scheme) & np.isclose(sweep.p, p)]
        for dt, h in g.groupby('deg_t'):
            rows.append(dict(scheme=scheme, p=p, deg_t=int(dt), K=len(h), mfpt=h.mfpt.mean(), mfpt_ci=ci(h.mfpt),
                             mode=h['mode'].mean(), mode_ci=ci(h['mode']), ratio=h.ratio.mean(), ratio_ci=ci(h.ratio)))
strat = pd.DataFrame(rows)
strat.to_csv(_os.path.join(_R, 'data', 'msc_defects', 'summary_stratified.csv'), index=False)
print(strat.round(4).to_string(index=False))

# ---- first-order (dilute) theory from the single-defect maps
fo = {}
fn = _os.path.join(_R, 'data', 'msc_defects', 'sensitivity_N35.csv')
if os.path.exists(fn):
    sens = pd.read_csv(fn)
    N = 35
    d = sens.iloc[1:]
    chiM = np.zeros((N, N)); chim = np.zeros((N, N))
    chiM[d.i, d.j] = d.chi_mfpt; chim[d.i, d.j] = d.chi_mode
    w_uni = np.ones((N, N)); w_uni[0, 0] = 0; w_uni[-1, -1] = 0; w_uni /= w_uni.sum()
    w = np.ones((N, N)); radius = max(3, N // 8)
    for i in range(radius):
        for j in range(radius):
            dist = np.sqrt(i ** 2 + j ** 2)
            w[i, j] *= np.exp(-2 * (radius - dist) / radius)
    for i in range(N - radius, N):
        for j in range(N - radius, N):
            dist = np.sqrt((N - 1 - i) ** 2 + (N - 1 - j) ** 2)
            w[i, j] *= np.exp(-2 * (radius - dist) / radius)
    w[0, 0] = 0; w[-1, -1] = 0; w_smart = w / w.sum()
    for name, ww in (('uniform', w_uni), ('smart', w_smart)):
        g = summ[(summ.scheme == name) & np.isclose(summ.p, 0.02)].iloc[0]
        M = int(round(0.02 * N * N)); pe = M / N ** 2
        fo[name] = dict(pred_slope_mfpt=float(N * N * np.sum(ww * chiM)), pred_slope_mode=float(N * N * np.sum(ww * chim)),
                        measured_slope_mfpt_p002=float((g.mfpt_factor - 1) / pe),
                        measured_slope_mfpt_ci=float(g.mfpt_factor_ci / pe),
                        measured_slope_mode_p002=float((g.mode_factor - 1) / pe),
                        measured_slope_mode_ci=float(g.mode_factor_ci / pe))
        fo[name]['pred_slope_ratio'] = fo[name]['pred_slope_mode'] - fo[name]['pred_slope_mfpt']
    fo['pi_minus_1'] = float(np.pi - 1)
    fo['frac_sites_lowering_mfpt'] = float((d.chi_mfpt < 0).mean())
    fo['frac_sites_lowering_mode'] = float((d.chi_mode < 0).mean())
    fo['max_chi_mfpt'] = dict(value=float(d.chi_mfpt.max()), site=[int(d.loc[d.chi_mfpt.idxmax(), 'i']), int(d.loc[d.chi_mfpt.idxmax(), 'j'])])
    fo['max_chi_mode'] = dict(value=float(d.chi_mode.max()), site=[int(d.loc[d.chi_mode.idxmax(), 'i']), int(d.loc[d.chi_mode.idxmax(), 'j'])])
    fo['min_chi_mfpt'] = dict(value=float(d.chi_mfpt.min()), site=[int(d.loc[d.chi_mfpt.idxmin(), 'i']), int(d.loc[d.chi_mfpt.idxmin(), 'j'])])
    # share of the total MFPT susceptibility carried by sites within r of the target
    ii, jj = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    dt = np.sqrt((ii - N + 1) ** 2 + (jj - N + 1) ** 2); ds = np.sqrt(ii ** 2 + jj ** 2)
    for r_ in (3, 6, 10):
        fo[f'share_mfpt_within_{r_}_of_target'] = float(chiM[dt <= r_].sum() / chiM.sum())
        fo[f'share_mode_within_{r_}_of_target'] = float(chim[dt <= r_].sum() / chim.sum())
        fo[f'share_mfpt_within_{r_}_of_start'] = float(chiM[ds <= r_].sum() / chiM.sum())
        fo[f'share_mode_within_{r_}_of_start'] = float(chim[ds <= r_].sum() / chim.sum())
    json.dump(fo, open(_os.path.join(_R, 'data', 'msc_defects', 'summary_first_order.json'), 'w'), indent=1)
    print(json.dumps(fo, indent=1))

# ---- size dependence
fc_ = _os.path.join(_R, 'data', 'msc_defects', 'size_clean.csv'); fd_ = _os.path.join(_R, 'data', 'msc_defects', 'size_defects.csv')
if os.path.exists(fc_) and os.path.exists(fd_):
    cl = pd.read_csv(fc_); de = pd.read_csv(fd_)
    rows = []
    for _, c in cl.iterrows():
        rows.append(dict(N=int(c.N), p=0.0, K=1, mfpt=c.mfpt, mode=c['mode'], ratio=c['mode'] / c.mfpt, ratio_ci=0.0,
                         inv_ratio=c.mfpt / c['mode'], inv_ratio_ci=0.0, cv=c.sd / c.mfpt, lnN=np.log(c.N)))
    for (N_, p), g in de.groupby(['N', 'p']):
        c = cl[cl.N == N_].iloc[0]
        rows.append(dict(N=int(N_), p=p, K=len(g), mfpt=g.mfpt.mean(), mfpt_ci=ci(g.mfpt), mode=g['mode'].mean(),
                         mode_ci=ci(g['mode']), ratio=g.ratio.mean(), ratio_ci=ci(g.ratio),
                         inv_ratio=(g.mfpt / g['mode']).mean(), inv_ratio_ci=ci(g.mfpt / g['mode']),
                         cv=g.cv.mean(), lnN=np.log(N_), mfpt_factor=g.mfpt.mean() / c.mfpt,
                         mfpt_factor_ci=ci(g.mfpt) / c.mfpt, mode_factor=g['mode'].mean() / c['mode'],
                         mode_factor_ci=ci(g['mode']) / c['mode'], hom_factor=hom_tf(p),
                         ratio_shift=g.ratio.mean() - c['mode'] / c.mfpt))
    sz = pd.DataFrame(rows).sort_values(['p', 'N'])
    sz.to_csv(_os.path.join(_R, 'data', 'msc_defects', 'summary_size.csv'), index=False)
    print(sz.round(4).to_string(index=False))
    fits = {}
    for p, g in sz.groupby('p'):
        g = g[g.N >= 25]
        sl = stats.linregress(g.lnN, g.inv_ratio)
        fits[str(p)] = dict(slope=float(sl.slope), slope_se=float(sl.stderr), intercept=float(sl.intercept),
                            intercept_se=float(sl.intercept_stderr), Ns=[int(x) for x in g.N])
    json.dump(fits, open(_os.path.join(_R, 'data', 'msc_defects', 'summary_size_fits.json'), 'w'), indent=1)
    print(json.dumps(fits, indent=1))

# ---- beyond-inert summaries (1D barriers, permeable obstacles, literal Eq.(4.1) traps)
sys.path.insert(0, os.path.dirname(__file__))
import fptcore as fc
out = {}
f1 = _os.path.join(_R, 'data', 'msc_defects', 'beyond_1d_random.csv')
if os.path.exists(f1):
    d1 = pd.read_csv(f1)
    out['1d_random'] = []
    for p, g in d1.groupby('p'):
        out['1d_random'].append(dict(p=p, K=len(g), mfpt=g.mfpt.mean(), mfpt_ci=ci(g.mfpt),
                                     mfpt_exact_mean=12375.0 * (1 + p * (1 / 0.2 - 1)),
                                     mode=g['mode'].mean(), mode_ci=ci(g['mode']), ratio=g.ratio.mean(),
                                     ratio_ci=ci(g.ratio), ratio_std=float(g.ratio.std(ddof=1)) if len(g) > 1 else 0.0,
                                     cv=g.cv.mean()))
    s1 = pd.read_csv(_os.path.join(_R, 'data', 'msc_defects', 'beyond_1d_single.csv'))
    out['1d_single'] = dict(ratio_min=float(s1.ratio.min()), pos_min=int(s1.pos[s1.ratio.idxmin()]),
                            ratio_max=float(s1.ratio.max()), pos_max=int(s1.pos[s1.ratio.idxmax()]),
                            max_abs_err_formula=float(np.max(np.abs(s1.mfpt - s1.mfpt_formula))))
f2 = _os.path.join(_R, 'data', 'msc_defects', 'beyond_2d_permeable.csv')
if os.path.exists(f2):
    d2 = pd.read_csv(f2)
    out['2d_permeable'] = [dict(kappa=k, K=len(g), mfpt=g.mfpt.mean(), mfpt_ci=ci(g.mfpt), mode=g['mode'].mean(),
                                mode_ci=ci(g['mode']), ratio=g.ratio.mean(), ratio_ci=ci(g.ratio), cv=g.cv.mean())
                           for k, g in d2.groupby('kappa')]
f3 = _os.path.join(_R, 'data', 'msc_defects', 'beyond_2d_traps.csv')
if os.path.exists(f3):
    d3 = pd.read_csv(f3)
    N = 35; q = 0.8
    ch = fc.build_chain(np.ones((N, N), bool), q, (0, 0), (N - 1, N - 1))
    S = fc.SpectralFPT(ch['Q'], ch['r'], ch['s'])
    out['2d_traps'] = []
    for D, g in d3.groupby('Delta'):
        z = 1 - 0.10 * D                      # mean-field: uniform killing probability p*Delta per step
        H_mf = float(np.sum(S.w / (1 - S.lam * z)))
        cm_mf = float(np.sum(S.w / (1 - S.lam * z) ** 2) / H_mf)
        out['2d_traps'].append(dict(Delta=D, K=len(g), H=g.H.mean(), H_ci=ci(g.H), cond_mean=g.cond_mean.mean(),
                                    cond_mean_ci=ci(g.cond_mean), mode=g['mode'].mean(), mode_ci=ci(g['mode']),
                                    ratio=g.ratio.mean(), ratio_ci=ci(g.ratio), H_meanfield=H_mf,
                                    cond_mean_meanfield=cm_mf))
json.dump(out, open(_os.path.join(_R, 'data', 'msc_defects', 'summary_beyond.json'), 'w'), indent=1)
print(json.dumps(out, indent=1))

# ---- structured local ensembles
fl = _os.path.join(_R, 'data', 'msc_defects', 'structured_local.csv')
if os.path.exists(fl):
    dl = pd.read_csv(fl)
    rows = []
    for (kind, M), g in dl.groupby(['kind', 'M']):
        rows.append(dict(kind=kind, M=M, K=len(g), acceptance=len(g) / g.attempts.sum(),
                         mfpt_factor=g.mfpt.mean() / base.mfpt, mfpt_factor_ci=ci(g.mfpt) / base.mfpt,
                         mode_factor=g['mode'].mean() / base['mode'], mode_factor_ci=ci(g['mode']) / base['mode'],
                         ratio=g.ratio.mean(), ratio_ci=ci(g.ratio), cv=g.cv.mean(),
                         frac_mfpt_below_clean=float((g.mfpt < base.mfpt).mean())))
    sl = pd.DataFrame(rows)
    sl.to_csv(_os.path.join(_R, 'data', 'msc_defects', 'summary_structured_local.csv'), index=False)
    print(sl.round(4).to_string(index=False))
