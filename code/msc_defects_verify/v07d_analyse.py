"""v07d -- size dependence summary: mode study (v07 + v07c) and MFPT-only large-N study (v07b)."""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import os, glob, json, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); D = _os.path.join(_R, 'data', 'msc_defects_verify')
TH = json.load(open(_os.path.join(_R, 'data', 'msc_defects_verify', 'v04b_theta.json')))
theta = {0.1: TH['0.1']['Theta'], 0.2: TH['0.2']['Theta']}; s0s = {0.1: TH['0.1']['sigma0_over_sigma'], 0.2: TH['0.2']['sigma0_over_sigma']}
ci = lambda x: 1.96 * np.std(x, ddof=1) / np.sqrt(len(x))
c = pd.read_csv(_os.path.join(_R, 'data', 'msc_defects_verify', 'v07_clean.csv')).set_index('N')
print('clean: MFPT/mode - ln N'); print((c.mfpt / c['mode'] - np.log(c.index.values)).round(4).to_dict())
x = np.log(c.index.values[c.index.values >= 25]); y = (c.mfpt / c['mode']).values[c.index.values >= 25]
print('fit N>=25: slope %.4f intercept %.4f' % tuple(np.polyfit(x, y, 1)))
rng = np.random.default_rng(5)
rows = []
TRK = pd.read_csv(_os.path.join(_R, 'data', 'msc_defects', 'size_defects.csv')) if os.path.exists(_os.path.join(_R, 'data', 'msc_defects', 'size_defects.csv')) else None
for N in (15, 25, 35, 50, 70, 100, 140, 200):
    for p in (0.1, 0.2):
        fs = glob.glob(os.path.join(_R, 'data', 'msc_defects_verify', 'v07_size', f'N{N}_p{p:.2f}*.csv'))
        if N == 35: fs = [os.path.join(_R, 'data', 'msc_defects_verify', 'v03_sweep', f'N35_uniform_p{p:.2f}.csv')]
        if not fs: continue
        df = pd.concat([pd.read_csv(f) for f in fs])
        K = len(df); ii = rng.integers(0, K, (2000, K)); mo = df['mode'].values.astype(float); mf = df.mfpt.values
        rom = mo[ii].mean(1) / mf[ii].mean(1)
        row = dict(N=N, p=p, K=K, mfpt_fac=df.mfpt.mean() / c.mfpt[N], mfpt_fac_ci=ci(df.mfpt) / c.mfpt[N], mode_fac=mo.mean() / c['mode'][N], mode_fac_ci=ci(mo) / c['mode'][N],
                   theta=theta[p], ratio=df.ratio.mean(), ratio_ci=ci(df.ratio), shift=df.ratio.mean() - c.ratio[N], rel_shift=df.ratio.mean() / c.ratio[N] - 1, rel_shift_ci=ci(df.ratio) / c.ratio[N],
                   rom_rel_shift=mo.mean() / mf.mean() / c.ratio[N] - 1, rom_rel_lo=np.percentile(rom, 2.5) / c.ratio[N] - 1, rom_rel_hi=np.percentile(rom, 97.5) / c.ratio[N] - 1)
        if TRK is not None and 'N' in TRK:
            t = TRK[(TRK.N == N) & np.isclose(TRK.p, p)]
            if len(t): row.update(tr_K=len(t), tr_ratio=t.ratio.mean(), tr_ratio_ci=ci(t.ratio), tr_mode_fac=t['mode'].mean() / c['mode'][N], tr_mfpt_fac=t.mfpt.mean() / c.mfpt[N])
        rows.append(row)
S = pd.DataFrame(rows); S.to_csv(_os.path.join(_R, 'data', 'msc_defects_verify', 'v07_summary.csv'), index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
print(S.round(4).to_string())
# MFPT-only
B = pd.read_csv(_os.path.join(_R, 'data', 'msc_defects_verify', 'v07b_mfpt_largeN.csv'))
rows = []
for N in sorted(B.N.unique()):
    b0 = B[(B.N == N) & (B.p == 0)].iloc[0]
    for p in (0.1, 0.2):
        g = B[(B.N == N) & np.isclose(B.p, p)]
        res0 = b0.G_aa - b0.G_sa
        rows.append(dict(N=N, p=p, K=len(g), mfpt_fac=g.mfpt.mean() / b0.mfpt, ci=ci(g.mfpt) / b0.mfpt, over_theta=g.mfpt.mean() / b0.mfpt / theta[p], over_theta_ci=ci(g.mfpt) / b0.mfpt / theta[p],
                         n_frac=g.n.mean() / N ** 2, res_fac=(g.G_aa - g.G_sa).mean() / res0, s0s=s0s[p], Gaa_fac=g.G_aa.mean() / b0.G_aa, Gaa_clean=b0.G_aa, Gsa_clean=b0.G_sa,
                         dGaa_minus_hom=(g.G_aa.mean() - s0s[p] * b0.G_aa), dGsa_minus_hom=(g.G_sa.mean() - s0s[p] * b0.G_sa), frac_deg1=(g.deg_t == 1).mean()))
M = pd.DataFrame(rows); M.to_csv(_os.path.join(_R, 'data', 'msc_defects_verify', 'v07b_summary.csv'), index=False)
print(M.round(4).to_string())
