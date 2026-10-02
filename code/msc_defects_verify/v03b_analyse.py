"""v03b -- aggregate the sweep of this check, compare with the main implementation's per-placement CSVs."""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import os, glob, json, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
D = _os.path.join(_R, 'data', 'msc_defects_verify')
TR = _os.path.join(_R, 'data', 'msc_defects', 'sweep')
H = json.load(open(_os.path.join(_R, 'data', 'msc_defects_verify', 'v04_homog.json')))['random']
theta = {round(v['p'], 2): v['Theta'] for k, v in H.items() if v['L'] == 160}
clean = dict(mfpt=14100.808049917203, mode=2493, ratio=2493 / 14100.808049917203, cv=0.9182769845672402)
rng = np.random.default_rng(1)

def boot_ratio_of_means(mode, mfpt, B=4000):
    n = len(mode); ii = rng.integers(0, n, (B, n))
    r = mode[ii].mean(axis=1) / mfpt[ii].mean(axis=1)
    return np.percentile(r, [2.5, 97.5])

ci = lambda x: 1.96 * np.std(x, ddof=1) / np.sqrt(len(x))
rows = []
for scheme, tscheme in (('uniform', 'uniform'), ('cornerthinned', 'smart')):
    for fn in sorted(glob.glob(os.path.join(_R, 'data', 'msc_defects_verify', 'v03_sweep', f'N35_{scheme}_p*.csv'))):
        df = pd.read_csv(fn); p = float(fn.split('_p')[-1][:-4])
        tr = pd.read_csv(os.path.join(TR, f'sweep_N35_{tscheme}_p{p:.2f}.csv'))
        lo, hi = boot_ratio_of_means(df['mode'].values.astype(float), df.mfpt.values)
        row = dict(scheme=scheme, p=p, K=len(df), acc=len(df) / df.attempts.sum(),
                   mfpt=df.mfpt.mean(), mfpt_ci=ci(df.mfpt), mode=df['mode'].mean(), mode_ci=ci(df['mode']),
                   ratio=df.ratio.mean(), ratio_ci=ci(df.ratio), rom=df['mode'].mean() / df.mfpt.mean(), rom_lo=lo, rom_hi=hi,
                   cv=df.cv.mean(), mfpt_fac=df.mfpt.mean() / clean['mfpt'], mfpt_fac_ci=ci(df.mfpt) / clean['mfpt'],
                   mode_fac=df['mode'].mean() / clean['mode'], mode_fac_ci=ci(df['mode']) / clean['mode'], theta=theta.get(round(p, 2)),
                   n_frac=df.n.mean() / 1225, res_fac=((df.G_aa - df.G_sa).mean()) / (2.081608920467689 + 0.2205638223760005),
                   ident_maxrel=float(np.max(np.abs(df.mfpt_res / df.mfpt - 1))),
                   tr_mfpt=tr.mfpt.mean(), tr_mfpt_ci=ci(tr.mfpt), tr_mode=tr['mode'].mean(), tr_mode_ci=ci(tr['mode']),
                   tr_ratio=tr.ratio.mean(), tr_ratio_ci=ci(tr.ratio), tr_cv=tr.cv.mean(), tr_rom=tr['mode'].mean() / tr.mfpt.mean())
        # z-scores of the difference between the two independent samples
        row['z_mfpt'] = (row['mfpt'] - row['tr_mfpt']) / np.hypot(row['mfpt_ci'], row['tr_mfpt_ci']) * 1.96
        row['z_mode'] = (row['mode'] - row['tr_mode']) / np.hypot(row['mode_ci'], row['tr_mode_ci']) * 1.96
        row['z_ratio'] = (row['ratio'] - row['tr_ratio']) / np.hypot(row['ratio_ci'], row['tr_ratio_ci']) * 1.96
        # pooled
        allr = np.r_[df.ratio.values, tr.ratio.values]
        row['pooled_ratio'] = allr.mean(); row['pooled_ratio_ci'] = ci(allr)
        pm = np.r_[df['mode'].values, tr['mode'].values].astype(float); pf = np.r_[df.mfpt.values, tr.mfpt.values]
        lo, hi = boot_ratio_of_means(pm, pf)
        row['pooled_rom'] = pm.mean() / pf.mean(); row['pooled_rom_lo'] = lo; row['pooled_rom_hi'] = hi
        row['pooled_mean_log_ratio'] = float(np.mean(np.log(allr / clean['ratio']))); row['pooled_mean_log_ratio_ci'] = ci(np.log(allr / clean['ratio']))
        row['frac_ratio_below_clean'] = float(np.mean(allr < clean['ratio']))
        row['median_ratio'] = float(np.median(allr))
        rows.append(row)
S = pd.DataFrame(rows)
S.to_csv(_os.path.join(_R, 'data', 'msc_defects_verify', 'v03_summary.csv'), index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
print(S[['scheme', 'p', 'acc', 'mfpt', 'mfpt_ci', 'tr_mfpt', 'mode', 'mode_ci', 'tr_mode', 'ratio', 'ratio_ci', 'tr_ratio', 'tr_ratio_ci', 'cv', 'tr_cv']].round(4).to_string())
print(S[['scheme', 'p', 'z_mfpt', 'z_mode', 'z_ratio', 'mfpt_fac', 'mfpt_fac_ci', 'mode_fac', 'mode_fac_ci', 'theta', 'rom', 'rom_lo', 'rom_hi', 'tr_rom', 'n_frac', 'res_fac', 'ident_maxrel']].round(4).to_string())
print(S[['scheme', 'p', 'pooled_ratio', 'pooled_ratio_ci', 'pooled_rom', 'pooled_rom_lo', 'pooled_rom_hi', 'pooled_mean_log_ratio', 'pooled_mean_log_ratio_ci', 'frac_ratio_below_clean', 'median_ratio']].round(4).to_string())
# OLS slope of ratio on p for p<=0.2 (per placement), both samples
for scheme, tscheme in (('uniform', 'uniform'), ('cornerthinned', 'smart')):
    for name, pat in (('recheck', os.path.join(_R, 'data', 'msc_defects_verify', 'v03_sweep', f'N35_{scheme}_p*.csv')), ('main_implementation', os.path.join(TR, f'sweep_N35_{tscheme}_p*.csv'))):
        A = pd.concat([pd.read_csv(f) for f in glob.glob(pat)])
        if 'p' not in A: continue
        A = A[(A.p <= 0.2001)]
        if name == 'recheck':   # add the clean point as the main implementation did (one row)
            A = pd.concat([A[['p', 'ratio']], pd.DataFrame(dict(p=[0.0], ratio=[clean['ratio']]))])
        x = A.p.values; y = A.ratio.values
        b, a = np.polyfit(x, y, 1); resid = y - (a + b * x); se = np.sqrt(resid.var(ddof=2) / ((x - x.mean()) ** 2).sum())
        print(scheme, name, 'n=', len(x), 'OLS slope of ratio on p (p<=0.2): %.4f +- %.4f (1.96 se)' % (b, 1.96 * se))
