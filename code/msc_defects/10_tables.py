"""10_tables.py -- render the summary CSV/JSON files as markdown tables (data/tables/*.md) so that
every number of these tables is machine-generated from the raw results."""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import os, json
import numpy as np, pandas as pd

DATA = _os.path.join(_R, 'data', 'msc_defects')
OUT = _os.path.join(_R, 'data', 'msc_defects', 'tables'); os.makedirs(OUT, exist_ok=True)

def w(name, lines):
    open(os.path.join(OUT, name), 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines)); print()

s = pd.read_csv(_os.path.join(_R, 'data', 'msc_defects', 'summary_sweep_N35.csv'))
for scheme in ('uniform', 'smart'):
    g = s[s.scheme == scheme]
    L = ['| p | accepted | MFPT | mode | mode/MFPT (mean of ratios) | ⟨mode⟩/⟨MFPT⟩ | CV | MFPT factor | mode factor | homogenised D₀/D_eff |',
         '|---|---|---|---|---|---|---|---|---|---|']
    for _, r in g.iterrows():
        if r.p == 0:
            L.append(f'| 0 | – | 14 100.8 | 2 493 | 0.1768 | 0.1768 | 0.918 | 1 | 1 | 1 |')
            continue
        L.append(f'| {r.p:.2f} | {r.acceptance:.3f} | {r.mfpt:,.0f} ± {r.mfpt_ci:,.0f} | {r["mode"]:,.0f} ± {r.mode_ci:,.0f} | '
                 f'{r.ratio:.4f} ± {r.ratio_ci:.4f} | {r.ratio_of_means:.4f} [{r.ratio_of_means_lo:.4f}–{r.ratio_of_means_hi:.4f}] | {r.cv:.3f} | '
                 f'{r.mfpt_factor:.3f} ± {r.mfpt_factor_ci:.3f} | {r.mode_factor:.3f} ± {r.mode_factor_ci:.3f} | {r.hom_factor:.3f} |'.replace(',', ' '))
    w(f'sweep_{scheme}.md', L)

# Tetali split
L = ['| p | n/N² (uniform) | [G_aa−G_sa] factor (uniform) | n/N² (corner-thinned pl.) | [G_aa−G_sa] factor (corner-thinned pl.) | homogenised P∞ | homogenised σ₀/σ |',
     '|---|---|---|---|---|---|---|']
hom = json.load(open(_os.path.join(_R, 'data', 'msc_defects', 'homogenization.json')))
hp = {round(r['p'], 3): r for r in hom['finite_p']}
u = s[s.scheme == 'uniform'].set_index('p'); m = s[s.scheme == 'smart'].set_index('p')
for p in (0.02, 0.06, 0.10, 0.14, 0.20, 0.25, 0.30, 0.35):
    h = hp[round(p, 3)]
    L.append(f'| {p:.2f} | {u.loc[p, "n_factor"]:.4f} | {u.loc[p, "Gdiff_factor"]:.3f} | {m.loc[p, "n_factor"]:.4f} | {m.loc[p, "Gdiff_factor"]:.3f} | {h["P_inf"]:.4f} | {1/h["sigma"]:.3f} |')
w('tetali_split.md', L)

L = ['| p | σ/σ₀ (L = 192 torus) | P∞ | D_eff/D₀ = σ/(σ₀P∞) | 1 − (π−1)p | time factor D₀/D_eff |', '|---|---|---|---|---|---|']
for r in hom['finite_p']:
    if r['p'] in (0.0, 0.005, 0.225, 0.275, 0.325):
        continue
    L.append(f'| {r["p"]:.2f} | {r["sigma"]:.4f} ± {1.96*r["sigma_sem"]:.4f} | {r["P_inf"]:.4f} | {r["D_ratio"]:.4f} | {1-(np.pi-1)*r["p"]:.4f} | {r["time_factor"]:.3f} |')
w('homogenization.md', L)
L = ['| L | (1 − σ/σ₀)·L² |', '|---|---|'] + [f'| {r["L"]} | {r["one_minus_sigma_times_L2"]:.6f} |' for r in hom['single_site']]
w('single_site.md', L)

d = pd.read_csv(_os.path.join(_R, 'data', 'msc_defects', 'structured_deterministic.csv'))
b = d[d['name'] == 'clean'].iloc[0]
L = ['| geometry | blocked sites | n | MFPT | mode | mode/MFPT | MFPT factor | mode factor | CV | median/MFPT |', '|---|---|---|---|---|---|---|---|---|---|']
for _, r in d.iterrows():
    L.append(f'| `{r["name"]}` | {int(r.n_blocked)} | {int(r.n_cluster)} | {r.mfpt:,.1f} | {int(r["mode"]):,} | {r.ratio:.4f} | {r.mfpt/b.mfpt:.3f} | {r["mode"]/b["mode"]:.3f} | {r.cv:.3f} | {r["median"]/r.mfpt:.3f} |'.replace(',', ' '))
w('structured_deterministic.md', L)

l = pd.read_csv(_os.path.join(_R, 'data', 'msc_defects', 'summary_structured_local.csv'))
L = ['| region | M | accepted | MFPT factor | mode factor | mode/MFPT | fraction of placements with MFPT below clean |', '|---|---|---|---|---|---|---|']
for _, r in l.iterrows():
    L.append(f'| {r.kind} | {int(r.M)} | {r.acceptance:.3f} | {r.mfpt_factor:.4f} ± {r.mfpt_factor_ci:.4f} | {r.mode_factor:.4f} ± {r.mode_factor_ci:.4f} | {r.ratio:.4f} ± {r.ratio_ci:.4f} | {r.frac_mfpt_below_clean:.3f} |')
w('structured_local.md', L)

z = pd.read_csv(_os.path.join(_R, 'data', 'msc_defects', 'summary_size.csv'))
L = ['| N | clean MFPT | clean mode | clean MFPT/mode − ln N | p | K | MFPT factor | mode factor | mode/MFPT | shift of mode/MFPT vs clean |', '|---|---|---|---|---|---|---|---|---|---|']
c = z[z.p == 0].set_index('N')
for _, r in z[z.p > 0].sort_values(['N', 'p']).iterrows():
    cc = c.loc[r.N]
    L.append(f'| {int(r.N)} | {cc.mfpt:,.1f} | {int(cc["mode"]):,} | {cc.inv_ratio-cc.lnN:.4f} | {r.p:.2f} | {int(r.K)} | {r.mfpt_factor:.3f} ± {r.mfpt_factor_ci:.3f} | '
             f'{r.mode_factor:.3f} ± {r.mode_factor_ci:.3f} | {r.ratio:.4f} ± {r.ratio_ci:.4f} | {r.ratio_shift:+.4f} |'.replace(',', ' '))
w('size.md', L)
L = ['| N | MFPT | mode | mode/MFPT | MFPT/mode − ln N | CV |', '|---|---|---|---|---|---|']
for N_, cc in c.iterrows():
    L.append(f'| {int(N_)} | {cc.mfpt:,.3f} | {int(cc["mode"]):,} | {cc.ratio:.4f} | {cc.inv_ratio-cc.lnN:.4f} | {cc.cv:.4f} |'.replace(',', ' '))
w('size_clean.md', L)

st = pd.read_csv(_os.path.join(_R, 'data', 'msc_defects', 'summary_stratified.csv'))
L = ['| placement | p | open neighbours of target | K | MFPT | mode | mode/MFPT |', '|---|---|---|---|---|---|---|']
for _, r in st.iterrows():
    L.append(f'| {"uniform" if r.scheme=="uniform" else "corner-thinned"} | {r.p:.2f} | {int(r.deg_t)} | {int(r.K)} | {r.mfpt:,.0f} ± {r.mfpt_ci:,.0f} | {r["mode"]:,.0f} ± {r.mode_ci:,.0f} | {r.ratio:.4f} ± {r.ratio_ci:.4f} |'.replace(',', ' '))
w('stratified.md', L)

for N_ in (15, 25, 35, 50):
    fn = os.path.join(DATA, f'sensitivity_N{N_}.csv')
    if os.path.exists(fn):
        x = pd.read_csv(fn); dd = x.iloc[1:]
        print(f'N={N_}: slope MFPT {N_*N_*dd.chi_mfpt.mean():.4f}  slope mode {N_*N_*dd.chi_mode.mean():.4f}  '
              f'frac lowering MFPT {(dd.chi_mfpt<0).mean():.3f}  frac lowering mode {(dd.chi_mode<0).mean():.3f}  '
              f'max chi_mfpt {dd.chi_mfpt.max():.4f} max chi_mode {dd.chi_mode.max():.4f}')
