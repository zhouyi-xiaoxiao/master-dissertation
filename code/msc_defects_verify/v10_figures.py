"""v10 -- verification figures (vector PDF + PNG) from the data of this check, with the data of the main implementation overlaid."""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import os, sys, json, glob
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, SymLogNorm
import scipy.linalg as la
import vcore as vc
HERE = os.path.dirname(os.path.abspath(__file__)); D = _os.path.join(_R, 'data', 'msc_defects_verify'); F = _os.path.join(_R, 'out', 'figures', 'msc_defects_verify'); TD = _os.path.join(_R, 'data', 'msc_defects')
os.makedirs(F, exist_ok=True)
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'      # validated categorical slots 1-3
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'lines.linewidth': 1.6, 'legend.frameon': False,
                     'pdf.fonttype': 42, 'savefig.bbox': 'tight', 'figure.dpi': 150, 'axes.titlesize': 9.5, 'axes.titleweight': 'bold'})
def save(fig, name):
    fig.savefig(os.path.join(F, name + '.pdf')); fig.savefig(os.path.join(F, name + '.png'), dpi=220); plt.close(fig)
ci = lambda x: 1.96 * np.std(x, ddof=1) / np.sqrt(len(x))
CL = dict(mfpt=14100.808049917203, mode=2493.0); CL['ratio'] = CL['mode'] / CL['mfpt']

# ------------------------------------------------------------------ Fig 1: density sweep
S = pd.read_csv(_os.path.join(_R, 'data', 'msc_defects_verify', 'v03_summary.csv'))
H = json.load(open(_os.path.join(_R, 'data', 'msc_defects_verify', 'v04_homog.json')))['random']
hp = np.array(sorted(v['p'] for v in H.values() if v['L'] == 160)); hth = np.array([H[f'L160_p{p:.2f}']['Theta'] for p in hp])
fig, ax = plt.subplots(1, 3, figsize=(11.2, 3.5))
pp = np.linspace(0, 0.36, 200)
for a, key, trkey, ttl, yl in ((ax[0], 'mfpt', 'tr_mfpt', '(a) mean first-passage time', 'MFPT / clean MFPT'), (ax[1], 'mode', 'tr_mode', '(b) mode', 'mode / clean mode')):
    a.plot(np.r_[0, hp], np.r_[1, hth], color=INK, lw=1.3, label=r'homogenised $\Theta(p)=P_\infty\sigma_0/\sigma$')
    a.plot(pp[pp <= 0.2], 1 + (np.pi - 1) * pp[pp <= 0.2], color=MUTED, lw=1.1, ls='--', label=r'dilute $1+(\pi-1)p$')
    for sch, col, mk, lab in (('uniform', BLUE, 'o', 'uniform placement'), ('cornerthinned', ORANGE, 's', 'corner-thinned placement')):
        g = S[S.scheme == sch]
        a.errorbar(g.p, g[key] / CL[key], yerr=g[key + '_ci'] / CL[key], fmt=mk, color=col, ms=4.5, lw=1, capsize=2, label=lab + ' (check)')
        a.plot(g.p + 0.004, g[trkey] / CL[key], mk, mfc='none', mec=col, ms=4.5, mew=1, label=lab + ' (main implementation)')
    a.set_yscale('log'); a.set_yticks([1, 2, 3, 4, 6, 8]); a.set_yticklabels(['1', '2', '3', '4', '6', '8']); a.set_ylim(0.95, 8.5)
    a.set_xlabel('blocked fraction $p$'); a.set_ylabel(yl); a.set_title(ttl, loc='left')
ax[0].legend(fontsize=6.6, loc='upper left')
a = ax[2]
a.axhline(CL['ratio'], color=INK, lw=1.0); a.text(0.355, CL['ratio'] - 0.0035, 'clean lattice 0.1768', ha='right', va='top', fontsize=7, color=MUTED)
for sch, col, mk, lab in (('uniform', BLUE, 'o', 'uniform'), ('cornerthinned', ORANGE, 's', 'corner-thinned placement')):
    g = S[S.scheme == sch]
    a.errorbar(g.p, g.ratio, yerr=g.ratio_ci, fmt=mk, color=col, ms=4.5, lw=1, capsize=2, label=lab + ': mean of ratios (check)')
    a.plot(g.p + 0.004, g.tr_ratio, mk, mfc='none', mec=col, ms=4.5, mew=1, label=lab + ': mean of ratios (main implementation)')
    a.plot(g.p, g.pooled_rom, '-', color=col, lw=1.0, alpha=0.8, label=lab + r': $\langle$mode$\rangle/\langle$MFPT$\rangle$, pooled')
a.set_xlabel('blocked fraction $p$'); a.set_ylabel('mode / MFPT'); a.set_title('(c) mode-to-mean ratio', loc='left'); a.legend(fontsize=6.3, loc='upper left')
fig.tight_layout(); save(fig, 'vfig1_density_sweep')

# ------------------------------------------------------------------ Fig 2: susceptibility maps
s = pd.read_csv(_os.path.join(_R, 'data', 'msc_defects_verify', 'v05_sens_N35.csv')); N = 35
div = LinearSegmentedColormap.from_list('div', [BLUE, '#f0efec', ORANGE])
fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.9))
for a, col, ttl in ((ax[0], 'chi_mfpt', r'(a) $N^2\chi_{\rm MFPT}$'), (ax[1], 'chi_mode', r'(b) $N^2\chi_{\rm mode}$')):
    Z = np.full((N, N), np.nan); Z[s.i, s.j] = N * N * s[col]
    im = a.imshow(Z.T, origin='lower', cmap=div, norm=SymLogNorm(linthresh=1.0, vmin=-500, vmax=500))
    a.set_title(ttl, loc='left'); a.grid(False); a.set_xlabel('$x$'); a.set_ylabel('$y$')
    a.plot(0, 0, marker='o', color=INK, ms=4); a.plot(N - 1, N - 1, marker='*', color=INK, ms=7)
    cb = fig.colorbar(im, ax=a, shrink=0.82, ticks=[-100, -10, -1, 0, 1, 10, 100]); cb.outline.set_visible(False)
    a.text(1.5, 0.3, 'start', fontsize=7, color=INK); a.text(N - 2.5, N - 3.2, 'target', fontsize=7, color=INK, ha='right')
fig.suptitle('Relative change of the statistic when one site is blocked (N = 35); blue = decrease', fontsize=8.5, color=MUTED, y=0.99)
fig.tight_layout(); save(fig, 'vfig2_susceptibility')

# ------------------------------------------------------------------ Fig 3: size dependence
M = pd.read_csv(_os.path.join(_R, 'data', 'msc_defects_verify', 'v07b_summary.csv')); Z = pd.read_csv(_os.path.join(_R, 'data', 'msc_defects_verify', 'v07_summary.csv'))
fig, ax = plt.subplots(1, 3, figsize=(11.2, 3.4))
for p, col, mk in ((0.1, BLUE, 'o'), (0.2, ORANGE, 's')):
    g = M[np.isclose(M.p, p)]; ax[0].errorbar(g.N * (1.0 if p == 0.1 else 1.04), g.over_theta, yerr=g.over_theta_ci, fmt=mk + '-', color=col, ms=4.5, lw=1, capsize=2, label=f'$p$ = {p}')
    g = Z[np.isclose(Z.p, p)]
    ax[1].errorbar(g.N * (1.0 if p == 0.1 else 1.04), g.mode_fac / g.theta, yerr=g.mode_fac_ci / g.theta, fmt=mk + '-', color=col, ms=4.5, lw=1, capsize=2, label=f'$p$ = {p} (check)')
    if 'tr_mode_fac' in g: ax[1].plot(g.N * 0.96, g.tr_mode_fac / g.theta, mk, mfc='none', mec=col, ms=4.5, label=f'$p$ = {p} (main implementation)')
    ax[2].errorbar(g.N * (1.0 if p == 0.1 else 1.04), 100 * g.rel_shift, yerr=100 * g.rel_shift_ci, fmt=mk + '-', color=col, ms=4.5, lw=1, capsize=2, label=f'$p$ = {p}: mean of ratios')
    ax[2].plot(g.N * (1.0 if p == 0.1 else 1.04), 100 * g.rom_rel_shift, mk + ':', mfc='none', mec=col, color=col, ms=4.5, lw=1, label=f'$p$ = {p}: ratio of means')
for a, ttl, yl in ((ax[0], r'(a) MFPT factor / $\Theta(p)$', r'$\langle$MFPT$\rangle$ / (clean MFPT $\times\ \Theta$)'), (ax[1], r'(b) mode factor / $\Theta(p)$', r'$\langle$mode$\rangle$ / (clean mode $\times\ \Theta$)'),
                   (ax[2], '(c) shift of mode/MFPT relative to clean', 'relative shift (%)')):
    a.set_xscale('log'); a.set_xlabel('lattice size $N$'); a.set_ylabel(yl); a.set_title(ttl, loc='left'); a.legend(fontsize=6.8)
    a.set_xticks([15, 25, 35, 50, 70, 100, 140, 200, 280]); a.set_xticklabels(['15', '25', '35', '50', '70', '100', '140', '200', '280'], fontsize=7); a.minorticks_off()
ax[0].axhline(1, color=INK, lw=0.9); ax[1].axhline(1, color=INK, lw=0.9); ax[2].axhline(0, color=INK, lw=0.9)
fig.tight_layout(); save(fig, 'vfig3_size_dependence')

# ------------------------------------------------------------------ Fig 4: periodic phases and localised placements
P = pd.read_csv(_os.path.join(_R, 'data', 'msc_defects_verify', 'v06_periodic.csv')); Lc = pd.read_csv(_os.path.join(_R, 'data', 'msc_defects_verify', 'v06_local.csv')); TL = pd.read_csv(_os.path.join(_R, 'data', 'msc_defects', 'structured_local.csv'))
fig, ax = plt.subplots(1, 2, figsize=(10.2, 3.6))
a = ax[0]
for per, col, mk, th in ((2, BLUE, 'o', 1.5), (3, ORANGE, 's', 56 / 45), (4, AQUA, '^', 255 / 224)):
    g = P[P.a == per]; a.plot(g.mfpt_fac, g.mode_fac, mk, color=col, ms=5, mec='white', mew=0.6, label=f'one blocked site per {per}×{per} cell, all phases')
    a.plot(th, th, mk, mfc='none', mec=col, ms=10, mew=1.3)
for nm, (oi, oj, per) in {'main implementation a2': (1, 1, 2), 'main implementation a3': (2, 2, 3), 'main implementation a4': (1, 1, 4)}.items():
    g = P[(P.a == per) & (P.oi == oi) & (P.oj == oj)].iloc[0]; a.annotate('phase used\nby the main implementation' if per == 3 else '', (g.mfpt_fac, g.mode_fac), xytext=(1.02, 1.30), fontsize=7, color=MUTED,
                                                                         arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.6) if per == 3 else None)
a.plot([1, 1.6], [1, 1.6], color=MUTED, lw=0.8, ls=':'); a.text(1.47, 1.42, 'equal factors', fontsize=7, color=MUTED, rotation=36, ha='right', va='top')
a.plot([], [], 'o', mfc='none', mec=MUTED, ms=9, label=r'bulk $\Theta$ of the array (torus)')
a.set_xlabel('MFPT / clean MFPT'); a.set_ylabel('mode / clean mode'); a.set_title('(a) periodic arrays, N = 35: phase matters', loc='left'); a.legend(fontsize=6.8, loc='lower right')
a = ax[1]
a.axhline(CL['ratio'], color=INK, lw=0.9)
for kind, col, mk, lab in (('near_target', BLUE, 'o', 'within 12 of target'), ('near_start', ORANGE, 's', 'within 12 of start'), ('centre', AQUA, '^', 'centre disc'), ('anywhere', MUTED, 'D', 'anywhere')):
    g = Lc[Lc.kind == kind].groupby('M').ratio.agg(mean='mean', cihw=ci); a.errorbar(g.index, g['mean'], yerr=g['cihw'], fmt=mk + '-', color=col, ms=4.5, lw=1, capsize=2, label=lab)
    t = TL[TL.kind == kind].groupby('M').ratio.mean(); a.plot(t.index + 0.8, t.values, mk, mfc='none', mec=col, ms=4.5)
a.plot([], [], 'o', mfc='none', mec=MUTED, ms=4.5, label='open symbols: main implementation')
a.set_xlabel('number of blocked sites $M$'); a.set_ylabel('mode / MFPT'); a.set_title('(b) same number of defects, different regions', loc='left'); a.legend(fontsize=6.8, loc='upper left', bbox_to_anchor=(1.01, 1.0)); a.set_xlim(8, 43)
a.set_xticks([10, 20, 30, 40])
fig.tight_layout(); save(fig, 'vfig4_structured')

# ------------------------------------------------------------------ Fig 5: two-mode structure and elasticities
N, q = 35, 0.8
om = np.ones((N, N), bool); idx, n = vc.cluster_index(om, (0, 0)); a_ = idx[N - 1, N - 1]
Pm = vc.transition(idx, n, q); Q, r, _ = vc.reduce_target(Pm, a_)
lam, V = la.eigh(Q.toarray()); w = V[0] * (V.T @ r); o = np.argsort(-lam); sig = [k for k in o if abs(w[k]) > 1e-12 * abs(w[o[0]])]
t = np.arange(1, 12001)
full = (w[None, :] * lam[None, :] ** (t[:, None] - 1)).sum(1) if False else None
ex = vc.solve_config(om, q, want_pmf=True)['pmf']
u = np.zeros(n - 1); u[0] = 1.0; f = []
for _ in range(12000):
    f.append(u @ r); u = Q @ u
f = np.array(f)
f2 = sum(w[k] * lam[k] ** (t - 1) for k in sig[:2]); f3 = sum(w[k] * lam[k] ** (t - 1) for k in sig[:3])
fig, ax = plt.subplots(1, 2, figsize=(8.8, 3.5))
a = ax[0]
a.plot(t, f * 1e5, color=INK, lw=1.8, label='exact PMF (mode 2493)')
a.plot(t, f2 * 1e5, color=BLUE, lw=1.3, ls='--', label=r'two modes: $\tau$ = 12 935 and 518 (mode 2506)')
a.plot(t, f3 * 1e5, color=ORANGE, lw=1.3, ls=':', label='three modes (mode 2493)')
a.set_ylim(0, 7.5); a.set_xlim(0, 12000); a.set_xlabel('time $t$ (steps)'); a.set_ylabel(r'$f(t)\times 10^{5}$'); a.set_title('(a) clean lattice: spectral truncations', loc='left'); a.legend(fontsize=7, loc='lower right')
a.text(0.97, 0.95, r'the $\tau_{\rm rel}$ = 621 mode is odd under $x\leftrightarrow y$:' + '\nzero weight for a corner start', transform=a.transAxes, fontsize=7, color=MUTED, ha='right', va='top')
T = pd.read_csv(_os.path.join(_R, 'data', 'msc_defects_verify', 'v09b_twotime.csv')).iloc[:5]
a = ax[1]; x = np.arange(len(T))
a.plot(x, T.elasticity_exact, 'o', color=INK, ms=6, label='exact: ln(mode factor)/ln(MFPT factor)')
a.plot(x, T.elast_tau621, 's', color=BLUE, ms=5, label=r'heuristic, $\tau$ = 621 ($c$ calibrated on clean mode)')
a.plot(x, T.elast_tau540, '^', color=ORANGE, ms=5, label=r'heuristic, $\tau$ = 540, $c$ = 4.15 (from the spectrum)')
a.axhline(0.25, color=MUTED, lw=0.9, ls=':'); a.text(4.4, 0.253, '1/4', fontsize=7, color=MUTED, ha='right')
a.set_xticks(x); a.set_xticklabels(['one target\nneighbour', 'box R=3', 'box R=5', 'box R=8', 'box R=5\n3-site door'], fontsize=7)
a.set_ylim(0.15, 0.34); a.set_ylabel('elasticity of mode w.r.t. mean'); a.set_title('(b) target-local obstacles', loc='left'); a.legend(fontsize=6.8, loc='upper right')
fig.tight_layout(); save(fig, 'vfig5_two_time_scales')
print('figures written')
