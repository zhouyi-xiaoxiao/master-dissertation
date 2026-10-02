"""s7_defects_density_pooled.py -- every number of Section 8 that is not quoted verbatim from a research file.

Recomputes, from the per-placement result files of the two independent 400-placement samples,
  sample A = data/msc_defects/sweep/*.csv            (placement laws 'uniform', 'smart')
  sample B = data/msc_defects_verify/v03_sweep/*.csv (placement laws 'uniform', 'cornerthinned')
('smart' / 'cornerthinned' = the corner-thinned placement law of the article):

  1. per (law, p): mean +- 95 % interval (1.96 sd / sqrt K) of MFPT, mode, per-placement ratio mode/MFPT, CV, for
     sample A, sample B and the pooled 800 placements; ratio of ensemble means <mode>/<MFPT> with a percentile
     bootstrap 95 % interval; time factors relative to the clean lattice; acceptance probability of the
     start-target connectivity test; fraction of placements whose ratio is below the clean value;
  2. ordinary least-squares trend of the per-placement ratio against the realised blocked fraction M/N^2 for
     p <= 0.20 (and p <= 0.12), with 95 % intervals from classical and heteroscedasticity-consistent (HC3)
     standard errors;
  3. maximum relative deviation of the identity T = (4 n / q)(G_aa - G_oa) over all placements;
  4. comparison of the time factors with the homogenised factor Theta(p) (192 x 192 tori);
  5. the rows of the tables of Section 8 and Supplementary Section S9 (validation, homogenisation, size dependence)
     assembled from the research result files, with inverse-variance weighted means where the text quotes them.

Nothing is simulated here: the script only aggregates stored exact per-placement results.
Output: data/article/s7_defects_density_pooled.json, s7_defects_density_pooled.csv,
        s7_defects_density_tables.txt (LaTeX rows, for transcription into the section).
Run:    python code/article/s7_defects_density_pooled.py      (about 10 s, < 300 MB)
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import os, json, glob
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ART = _R                                   # the repository root
ROOT = _R                  # repository root
RA = _os.path.join(_R, 'data', 'msc_defects')
RB = _os.path.join(_R, 'data', 'msc_defects_verify')
OUT = _os.path.join(_R, 'data', 'article')
os.makedirs(OUT, exist_ok=True)

N, Q = 35, 0.8
PGRID = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.25, 0.30, 0.35]
LAWS = {'uniform': ('uniform', 'uniform'), 'corner_thinned': ('smart', 'cornerthinned')}   # article name -> (A name, B name)
Z = 1.959963984540054


def ci(x):
    x = np.asarray(x, float)
    return float(Z * x.std(ddof=1) / np.sqrt(x.size))


def ms(x):
    return float(np.mean(x)), ci(x)


def boot_rom(mode, mfpt, rng, B=4000):
    """Percentile bootstrap 95 % interval of the ratio of ensemble means <mode>/<MFPT>."""
    mode = np.asarray(mode, float); mfpt = np.asarray(mfpt, float); K = mode.size
    idx = rng.integers(0, K, size=(B, K))
    r = mode[idx].mean(1) / mfpt[idx].mean(1)
    return float(np.quantile(r, 0.025)), float(np.quantile(r, 0.975))


def ols(x, y):
    """Slope of y on x with classical and HC3 standard errors."""
    x = np.asarray(x, float); y = np.asarray(y, float); n = x.size
    X = np.column_stack([np.ones(n), x])
    XtXi = np.linalg.inv(X.T @ X)
    beta = XtXi @ X.T @ y
    res = y - X @ beta
    s2 = res @ res / (n - 2)
    se_cl = np.sqrt(np.diag(s2 * XtXi))
    h = np.einsum('ij,jk,ik->i', X, XtXi, X)
    meat = (X * (res / (1 - h))[:, None] ** 2).T @ X
    se_hc = np.sqrt(np.diag(XtXi @ meat @ XtXi))
    return dict(n=int(n), intercept=float(beta[0]), slope=float(beta[1]),
                slope_se_classical=float(se_cl[1]), slope_ci95_classical=float(Z * se_cl[1]),
                slope_se_hc3=float(se_hc[1]), slope_ci95_hc3=float(Z * se_hc[1]))


# ------------------------------------------------------------------ clean lattice
cleanA = pd.read_csv(_os.path.join(_R, 'data', 'msc_defects', 'sweep', 'sweep_N35_uniform_p0.00.csv')).iloc[0]
v01 = json.load(open(_os.path.join(_R, 'data', 'msc_defects_verify', 'v01_validation.json')))
valA = json.load(open(_os.path.join(_R, 'data', 'msc_defects', 'validation.json')))
T0, M0 = float(v01['clean_N35']['mfpt']), float(v01['clean_N35']['mode'])
R0 = M0 / T0
assert int(cleanA['mode']) == int(M0) and abs(cleanA['mfpt'] / T0 - 1) < 1e-11

res = {'meta': dict(N=N, q=Q, clean_mfpt=T0, clean_mode=M0, clean_ratio=R0, clean_cv=float(v01['clean_N35']['cv']),
                    interval='mean +- 1.96 sd/sqrt(K) over placements unless stated',
                    sources=['data/msc_defects/sweep/*.csv', 'data/msc_defects_verify/v03_sweep/*.csv'])}

# ------------------------------------------------------------------ homogenised factor Theta(p)
hom = json.load(open(_os.path.join(_R, 'data', 'msc_defects', 'homogenization.json')))
THETA = {round(r['p'], 3): r['time_factor'] for r in hom['finite_p']}

# ------------------------------------------------------------------ 1. pooled sweep statistics
rng = np.random.default_rng(20261001)
rows, per = [], {}
ident = {'A': dict(n=0, maxrel=0.0), 'B': dict(n=0, maxrel=0.0)}
ident['A']['n'] += 2                                             # the two clean rows of sample A
for nm in ('uniform', 'smart'):
    c = pd.read_csv(os.path.join(_R, 'data', 'msc_defects', 'sweep', f'sweep_N35_{nm}_p0.00.csv')).iloc[0]
    ident['A']['maxrel'] = max(ident['A']['maxrel'], abs(c['mfpt_tetali'] / c['mfpt'] - 1))
for law, (na, nb) in LAWS.items():
    for p in PGRID:
        A = pd.read_csv(os.path.join(_R, 'data', 'msc_defects', 'sweep', f'sweep_N35_{na}_p{p:.2f}.csv'))
        B = pd.read_csv(os.path.join(_R, 'data', 'msc_defects_verify', 'v03_sweep', f'N35_{nb}_p{p:.2f}.csv'))
        assert len(A) == 400 and len(B) == 400
        assert (A.n_blocked == A.n_blocked.iloc[0]).all() and (B.n_blocked == A.n_blocked.iloc[0]).all()
        Mblk = int(A.n_blocked.iloc[0])
        ident['A']['n'] += len(A); ident['B']['n'] += len(B)
        ident['A']['maxrel'] = max(ident['A']['maxrel'], float(np.abs(A.mfpt_tetali / A.mfpt - 1).max()))
        ident['B']['maxrel'] = max(ident['B']['maxrel'], float(np.abs(B.mfpt_res / B.mfpt - 1).max()))
        d = {'A': dict(mfpt=A.mfpt.values, mode=A['mode'].values.astype(float), cv=A.cv.values, n=A.n_cluster.values, att=A.attempts.values),
             'B': dict(mfpt=B.mfpt.values, mode=B['mode'].values.astype(float), cv=B.cv.values, n=B.n.values, att=B.attempts.values)}
        d['P'] = {k: np.concatenate([d['A'][k], d['B'][k]]) for k in d['A']}
        per[(law, p)] = d
        row = dict(law=law, p=p, M=Mblk, p_realised=Mblk / N ** 2, theta_L192=THETA.get(round(p, 3)))
        for s in ('A', 'B', 'P'):
            x = d[s]; ratio = x['mode'] / x['mfpt']
            row[f'K_{s}'] = int(ratio.size)
            row[f'accept_{s}'] = float(ratio.size / x['att'].sum())
            for key, arr in (('mfpt', x['mfpt']), ('mode', x['mode']), ('ratio', ratio), ('cv', x['cv']),
                             ('mfpt_fac', x['mfpt'] / T0), ('mode_fac', x['mode'] / M0), ('n_frac', x['n'] / N ** 2)):
                m, c = ms(arr); row[f'{key}_{s}'] = m; row[f'{key}_ci_{s}'] = c
            row[f'mfpt_relsd_{s}'] = float(x['mfpt'].std(ddof=1) / x['mfpt'].mean())      # placement-to-placement relative s.d.
            row[f'mode_relsd_{s}'] = float(x['mode'].std(ddof=1) / x['mode'].mean())
            row[f'rom_{s}'] = float(x['mode'].mean() / x['mfpt'].mean())
            lo, hi = boot_rom(x['mode'], x['mfpt'], rng); row[f'rom_lo_{s}'] = lo; row[f'rom_hi_{s}'] = hi
            row[f'frac_ratio_below_clean_{s}'] = float(np.mean(ratio < R0))
        rA = d['A']['mode'] / d['A']['mfpt']; rB = d['B']['mode'] / d['B']['mfpt']
        row['z_ratio_A_vs_B'] = float((rB.mean() - rA.mean()) / np.sqrt(rA.var(ddof=1) / rA.size + rB.var(ddof=1) / rB.size))
        row['ratio_excess_P_pct'] = 100 * (row['ratio_P'] / R0 - 1)
        row['rom_excess_P_pct'] = 100 * (row['rom_P'] / R0 - 1)
        if row['theta_L192']:
            row['mfpt_fac_over_theta_P_pct'] = 100 * (row['mfpt_fac_P'] / row['theta_L192'] - 1)
            row['mode_fac_over_theta_P_pct'] = 100 * (row['mode_fac_P'] / row['theta_L192'] - 1)
            row['mfpt_fac_over_theta_A_pct'] = 100 * (row['mfpt_fac_A'] / row['theta_L192'] - 1)
            row['mode_fac_over_theta_A_pct'] = 100 * (row['mode_fac_A'] / row['theta_L192'] - 1)
        row['alpha_implied_P'] = (1 - 1 / row['mfpt_fac_P']) / row['p_realised']     # 1/(1 - alpha p) = MFPT factor
        rows.append(row)
sweep = pd.DataFrame(rows)
sweep.to_csv(_os.path.join(_R, 'data', 'article', 's7_defects_density_pooled.csv'), index=False)
res['sweep'] = rows
res['identity_check'] = {k: dict(placements=int(v['n']), max_rel_dev=float(v['maxrel'])) for k, v in ident.items()}

# summary statements about the uniform law
u = sweep[sweep.law == 'uniform']; c = sweep[sweep.law == 'corner_thinned']
s = {}
s['uniform_max_pooled_ratio_excess_pct_p_le_0.12'] = float(u[u.p <= 0.12].ratio_excess_P_pct.max())
s['uniform_p_of_max_excess_p_le_0.12'] = float(u[u.p <= 0.12].sort_values('ratio_excess_P_pct').p.iloc[-1])
s['uniform_max_pooled_ratio_excess_pct_p_le_0.14'] = float(u[u.p <= 0.14].ratio_excess_P_pct.max())
s['uniform_rom_excess_pct_by_p'] = {f'{p:.2f}': float(v) for p, v in zip(u.p, u.rom_excess_P_pct)}
s['uniform_ratio_excess_pct_by_p'] = {f'{p:.2f}': float(v) for p, v in zip(u.p, u.ratio_excess_P_pct)}
s['uniform_min_rom_excess_pct_p_le_0.14'] = float(u[u.p <= 0.14].rom_excess_P_pct.min())
s['uniform_frac_below_clean_range_p_le_0.20'] = [float(u[u.p <= 0.2].frac_ratio_below_clean_P.min()), float(u[u.p <= 0.2].frac_ratio_below_clean_P.max())]
s['corner_thinned_pooled_ratio_monotone_in_p'] = bool(np.all(np.diff(np.r_[R0, c.ratio_P.values]) > 0))
s['corner_thinned_ratio_excess_pct_by_p'] = {f'{p:.2f}': float(v) for p, v in zip(c.p, c.ratio_excess_P_pct)}
s['uniform_pooled_ratio_monotone_in_p'] = bool(np.all(np.diff(np.r_[R0, u.ratio_P.values]) > 0))
s['mean_of_ratios_above_ratio_of_means_in_all_cells'] = bool((sweep.ratio_P > sweep.rom_P).all())
s['uniform_relsd_mfpt_and_mode_by_p'] = {f'{p:.2f}': [float(a), float(b)] for p, a, b in zip(u.p, u.mfpt_relsd_P, u.mode_relsd_P)}
s['min_ratio_any_placement'] = float(min((d['P']['mode'] / d['P']['mfpt']).min() for d in per.values()))
s['max_ratio_any_placement'] = float(max((d['P']['mode'] / d['P']['mfpt']).max() for d in per.values()))
s['max_ratio_any_placement_p_le_0.20'] = float(max((d['P']['mode'] / d['P']['mfpt']).max() for (l, p), d in per.items() if p <= 0.2))
s['n_placements_total'] = int(sum(d['P']['mfpt'].size for d in per.values()))
s['uniform_alpha_implied_range_p_le_0.20'] = [float(u[u.p <= 0.2].alpha_implied_P.min()), float(u[u.p <= 0.2].alpha_implied_P.max())]
s['corner_thinned_alpha_implied_range_p_le_0.20'] = [float(c[c.p <= 0.2].alpha_implied_P.min()), float(c[c.p <= 0.2].alpha_implied_P.max())]
for nm, g in (('uniform', u), ('corner_thinned', c)):
    gg = g[g.p <= 0.2]
    s[f'{nm}_max_abs_mfpt_fac_over_theta_pct_p_le_0.20_pooled'] = float(gg.mfpt_fac_over_theta_P_pct.abs().max())
    s[f'{nm}_max_abs_mode_fac_over_theta_pct_p_le_0.20_pooled'] = float(gg.mode_fac_over_theta_P_pct.abs().max())
    s[f'{nm}_mfpt_fac_over_theta_pct_by_p_pooled'] = {f'{p:.2f}': float(v) for p, v in zip(g.p, g.mfpt_fac_over_theta_P_pct)}
    s[f'{nm}_mode_fac_over_theta_pct_by_p_pooled'] = {f'{p:.2f}': float(v) for p, v in zip(g.p, g.mode_fac_over_theta_P_pct)}
    s[f'{nm}_max_abs_mfpt_fac_over_theta_pct_p_le_0.20_A'] = float(gg.mfpt_fac_over_theta_A_pct.abs().max())
    s[f'{nm}_max_abs_mode_fac_over_theta_pct_p_le_0.20_A'] = float(gg.mode_fac_over_theta_A_pct.abs().max())
res['statements'] = s

# ------------------------------------------------------------------ 2. OLS trends of the per-placement ratio
tr = {}
for law in LAWS:
    for pmax in (0.20, 0.12):
        for smp in ('A', 'B', 'P'):
            xs, ys = [0.0], [R0]                                  # the clean lattice enters once
            for p in PGRID:
                if p > pmax + 1e-9:
                    continue
                d = per[(law, p)][smp]
                M = int(sweep[(sweep.law == law) & np.isclose(sweep.p, p)].M.iloc[0])
                r = d['mode'] / d['mfpt']
                xs += [M / N ** 2] * r.size; ys += list(r)
            tr[f'{law}_pmax{pmax:.2f}_{smp}'] = ols(xs, ys)
res['ols_trend_ratio_vs_p'] = tr

# ------------------------------------------------------------------ 5a. validation table
A1 = valA['V1_uniform_N35']; B1 = v01['clean_N35']
valsA = [A1['mfpt_solve'], A1['mfpt_spectral'], A1['mfpt_cosine_double_sum'], A1['mfpt_tetali']]
valsB = [B1['mfpt'], B1['mfpt_spectral'], B1['mfpt_cosine_double_sum'], B1['mfpt_res']]
A2 = valA['V2_random_N35_p015']
res['validation'] = dict(
    clean_mfpt_four_routes_A=valsA, clean_mfpt_four_routes_B=valsB,
    clean_mfpt_max_rel_spread_A=float((max(valsA) - min(valsA)) / T0), clean_mfpt_max_rel_spread_B=float((max(valsB) - min(valsB)) / T0),
    clean_mfpt_max_rel_spread_all=float((max(valsA + valsB) - min(valsA + valsB)) / T0),
    clean_mode=[A1['mode_certified'], A1['mode_timestep'], B1['mode'], B1['mode_spectral']],
    clean_sd=B1['sd'], clean_cv=B1['cv'], clean_median=B1['median'], clean_S_mode=B1['S_mode'], clean_S_mfpt=B1['S_mfpt'],
    clean_pmf_max_abs_diff_two_routes_B=B1['max_abs_diff_pmf_two_routes'],
    random_p015=dict(n=A2['n_cluster'], mfpt=[A2['mfpt_solve'], A2['mfpt_spectral'], A2['mfpt_tetali']],
                     max_rel_spread=float((max(A2['mfpt_solve'], A2['mfpt_spectral'], A2['mfpt_tetali']) - min(A2['mfpt_solve'], A2['mfpt_spectral'], A2['mfpt_tetali'])) / A2['mfpt_solve']),
                     mode=[A2['mode_dense'], A2['mode_timestep']], peak_rel_diff=float(abs(A2['f_mode_dense'] / A2['f_mode_timestep'] - 1))),
    mc_A={k: valA['V3_montecarlo_N15_p015'][k] for k in ('walkers', 'n_blocked', 'exact_mfpt', 'mc_mean', 'mc_se', 'z', 'chi2', 'dof', 'p_value')},
    mc_B={k: v01['mc'][k] for k in ('walkers', 'blocked', 'exact_mfpt', 'mc_mean', 'mc_se', 'z', 'chi2', 'dof')},
    eq41=v01['eq41'])

# 400 placements of sample A regenerated from their seeds and re-solved with the second code
pc = pd.read_csv(_os.path.join(_R, 'data', 'msc_defects_verify', 'v03c_perconfig.csv'))
res['validation']['resolved_sample_A_placements'] = dict(
    n=int(len(pc)), same_cluster_size=int((pc.n == pc.tr_n).sum()), same_rejection_count=int((pc.att == pc.tr_att).sum()),
    same_mode=int((pc['mode'] == pc.tr_mode).sum()), mfpt_max_rel_dev=float((pc.mfpt / pc.tr_mfpt - 1).abs().max()),
    sd_max_rel_dev=float((pc.sd / pc.tr_sd - 1).abs().max()), peak_max_rel_dev=float((pc.fmode / pc.tr_fmode - 1).abs().max()))

# ------------------------------------------------------------------ 5c. homogenisation table
vh = json.load(open(_os.path.join(_R, 'data', 'msc_defects_verify', 'v04_homog.json')))
vb = json.load(open(_os.path.join(_R, 'data', 'msc_defects_verify', 'v04b_theta.json')))
ht = []
for r in hom['finite_p']:
    p = r['p']
    b = next((v for v in vh['random'].values() if v['L'] == 160 and abs(v['p'] - p) < 1e-9), None)
    ht.append(dict(p=p, L=r['L'], samples=r['samples'], sigma=r['sigma'], sigma_ci=Z * r['sigma_sem'], P_inf=r['P_inf'], D_ratio=r['D_ratio'],
                   dilute_D=1 - (np.pi - 1) * p, theta=r['time_factor'],
                   theta_B_L160=None if b is None else b['Theta'], theta_B_L160_ci=None if b is None else b['Theta_ci'],
                   D_minus_dilute=r['D_ratio'] - (1 - (np.pi - 1) * p)))
res['homogenisation'] = dict(single_site_torus=hom['single_site'], single_site_torus_B=vh['single'], green=vh['green'],
                             finite_p=ht, theta_L200_K40=vb)

# ------------------------------------------------------------------ 5d. size dependence
Zs = pd.read_csv(_os.path.join(_R, 'data', 'msc_defects_verify', 'v07_summary.csv')); Ms = pd.read_csv(_os.path.join(_R, 'data', 'msc_defects_verify', 'v07b_summary.csv'))
sz = dict(mode=[], mfpt=[])
for _, r in Zs.iterrows():
    sz['mode'].append(dict(N=int(r.N), p=float(r.p), K=int(r.K), theta=float(r.theta), mode_fac=float(r.mode_fac), mode_fac_ci=float(r.mode_fac_ci),
                           mode_fac_over_theta=float(r.mode_fac / r.theta), mode_fac_over_theta_ci=float(r.mode_fac_ci / r.theta),
                           rel_shift_pct=100 * float(r.rel_shift), rel_shift_ci_pct=100 * float(r.rel_shift_ci),
                           rom_rel_shift_pct=100 * float(r.rom_rel_shift), rom_lo_pct=100 * float(r.rom_rel_lo), rom_hi_pct=100 * float(r.rom_rel_hi),
                           ratio=float(r.ratio), ratio_ci=float(r.ratio_ci),
                           A_K=None if r.tr_K != r.tr_K else int(r.tr_K), A_mode_fac=None if r.tr_mode_fac != r.tr_mode_fac else float(r.tr_mode_fac),
                           A_ratio=None if r.tr_ratio != r.tr_ratio else float(r.tr_ratio)))
for _, r in Ms.iterrows():
    sz['mfpt'].append(dict(N=int(r.N), p=float(r.p), K=int(r.K), mfpt_fac=float(r.mfpt_fac), mfpt_fac_ci=float(r.ci),
                           over_theta=float(r.over_theta), over_theta_ci=float(r.over_theta_ci), n_frac=float(r.n_frac), res_fac=float(r.res_fac), s0s=float(r.s0s)))
wm = {}
for p in (0.1, 0.2):
    g = Ms[np.isclose(Ms.p, p)]; w = 1 / (g.over_theta_ci / Z) ** 2
    wm[f'mfpt_over_theta_p{p}'] = dict(Ns=[int(x) for x in g.N], mean=float((w * g.over_theta).sum() / w.sum()), ci95=float(Z / np.sqrt(w.sum())),
                                       min=float(g.over_theta.min()), max=float(g.over_theta.max()))
    g = Zs[np.isclose(Zs.p, p)]
    wm[f'mode_over_theta_p{p}'] = dict(Ns=[int(x) for x in g.N], max_abs_dev_pct=float(100 * (g.mode_fac / g.theta - 1).abs().max()),
                                       N_of_max=int(g.N.iloc[int(np.argmax((g.mode_fac / g.theta - 1).abs().values))]),
                                       min=float(g.mode_fac.min()), max=float(g.mode_fac.max()))
    wm[f'rel_shift_pct_p{p}'] = {int(n): [100 * float(a), 100 * float(b)] for n, a, b in zip(g.N, g.rel_shift, g.rel_shift_ci)}
    wm[f'rom_rel_shift_pct_p{p}'] = {int(n): 100 * float(a) for n, a in zip(g.N, g.rom_rel_shift)}
sz['summary'] = wm
sz['kolmogorov'] = json.load(open(_os.path.join(_R, 'data', 'msc_defects_verify', 'v11_shape.json')))
res['size_dependence'] = sz

json.dump(res, open(_os.path.join(_R, 'data', 'article', 's7_defects_density_pooled.json'), 'w'), indent=1)

# ------------------------------------------------------------------ LaTeX rows (transcribed verbatim into the section)
L = []
def f(x, nd): return f'{x:.{nd}f}'
def fi(x):
    """integer with thin-space groups for five or more digits (four-digit numbers are not grouped)."""
    t = f'{x:,.0f}'
    return t.replace(',', '') if len(t.replace(',', '').lstrip('-')) <= 4 else t.replace(',', '\\,')
def pm(x, c, nd): return f'${f(x, nd)}\\pm{f(c, nd)}$'
L.append('% ---- tab-sweep: p & acceptance & <T>/T_clean & <t*>/t*_clean & Theta & mean of ratios & ratio of means [95 % bootstrap] & CV   (pooled, 800 placements)')
for law in LAWS:
    L.append(f'% {law}')
    for _, r in sweep[sweep.law == law].iterrows():
        L.append(f"{r.p:.2f} & {r.accept_P:.2f} & {pm(r.mfpt_fac_P, r.mfpt_fac_ci_P, 3)} & {pm(r.mode_fac_P, r.mode_fac_ci_P, 3)} & {f(r.theta_L192, 3)} & "
                 f"{pm(r.ratio_P, r.ratio_ci_P, 4)} & ${f(r.rom_P, 4)}$ $[{f(r.rom_lo_P, 4)},\\,{f(r.rom_hi_P, 4)}]$ & {f(r.cv_P, 3)} \\\\")
L.append('% ---- (not printed) per-sample values: law, p, <T>, <t*>, ratio A, ratio B, z, acceptance A, B')
for law in LAWS:
    for _, r in sweep[sweep.law == law].iterrows():
        L.append(f"% {law} {r.p:.2f}: T {fi(r.mfpt_P)}+-{fi(r.mfpt_ci_P)}  t* {fi(r.mode_P)}+-{fi(r.mode_ci_P)}  ratio A {f(r.ratio_A, 4)}+-{f(r.ratio_ci_A, 4)}  B {f(r.ratio_B, 4)}+-{f(r.ratio_ci_B, 4)}"
                 f"  z={r.z_ratio_A_vs_B:+.1f}  acc A {r.accept_A:.3f} B {r.accept_B:.3f}")
L.append('% ---- tab-homog: p & Sigma/Sigma_0 & P_inf & D_eff/D_0 & 1-(pi-1)p & Theta (L=192, 12) & Theta (L=160, 10)')
for r in ht:
    if round(r['p'], 3) not in (0.02, 0.04, 0.06, 0.08, 0.10, 0.14, 0.20, 0.25, 0.30, 0.35):
        continue
    nd = 4 if r['theta_B_L160_ci'] < 0.005 else (3 if r['theta_B_L160_ci'] < 0.05 else 2)
    L.append(f"{r['p']:.2f} & {pm(r['sigma'], r['sigma_ci'], 4)} & {f(r['P_inf'], 4)} & {f(r['D_ratio'], 4)} & {f(r['dilute_D'], 4)} & {f(r['theta'], 3)} & "
             f"{pm(r['theta_B_L160'], r['theta_B_L160_ci'], nd)} \\\\")
L.append("% ---- tab-size: N & K & mode factor & shift of mean of ratios (%) & shift of ratio of means (%) [95 % bootstrap] & K' & <T>/(Theta T_clean)")
mm = {(m['N'], m['p']): m for m in sz['mode']}; tt = {(m['N'], m['p']): m for m in sz['mfpt']}
for pv in (0.1, 0.2):
    L.append(f'% p = {pv}, Theta = {vb[str(pv)]["Theta"]:.4f} +- {vb[str(pv)]["Theta_ci"]:.4f}')
    for n in (15, 25, 35, 50, 70, 100, 140, 200, 280):
        a, c = mm.get((n, pv)), tt.get((n, pv))
        left = (f"{a['K']} & {pm(a['mode_fac'], a['mode_fac_ci'], 3)} & ${a['rel_shift_pct']:+.1f}\\pm{a['rel_shift_ci_pct']:.1f}$ & "
                f"${a['rom_rel_shift_pct']:+.1f}$ $[{a['rom_lo_pct']:+.1f},\\,{a['rom_hi_pct']:+.1f}]$") if a else "-- & -- & -- & --"
        right = f"{c['K']} & {pm(c['over_theta'], c['over_theta_ci'], 3)}" if c else "-- & --"
        L.append(f"{n} & {left} & {right} \\\\")
open(_os.path.join(_R, 'data', 'article', 's7_defects_density_tables.txt'), 'w').write('\n'.join(L) + '\n')

# ------------------------------------------------------------------ console summary
print('clean: T =', T0, 'mode =', M0, 'ratio =', R0)
print('identity check:', res['identity_check'])
for k, v in tr.items():
    print(f"OLS {k}: n={v['n']} slope={v['slope']:+.4f}  95% classical +-{v['slope_ci95_classical']:.4f}  HC3 +-{v['slope_ci95_hc3']:.4f}")
for k, v in s.items():
    print(k, v)
print(json.dumps(wm, indent=1))
print('re-solved placements:', res['validation']['resolved_sample_A_placements'])
print('validation spreads:', res['validation']['clean_mfpt_max_rel_spread_A'], res['validation']['clean_mfpt_max_rel_spread_B'],
      res['validation']['clean_mfpt_max_rel_spread_all'], res['validation']['random_p015'])
print(open(_os.path.join(_R, 'data', 'article', 's7_defects_density_tables.txt')).read())
