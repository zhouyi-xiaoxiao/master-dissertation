"""s7_defects_density_figures.py -- the two regenerated figures of Section 8 (vector PDF).

  figures/s7_defects_density_sweep.pdf   density sweep at N = 35, q = 0.8: time factors of the mean and of the mode and
                                         the ratio mode/MFPT against the blocked fraction p, for the two placement laws
                                         and the two independent 400-placement samples.
                                         Adapted from code/msc_defects/09_figures.py (fig2) and
                                         code/msc_defects_verify/v10_figures.py (vfig1); legends renamed
                                         ('corner-thinned', 'sample A', 'sample B').
  figures/s7_defects_density_size.pdf    size dependence under uniform placement, p = 0.1 and 0.2.
                                         Adapted from code/msc_defects_verify/v10_figures.py (vfig3);
                                         legends 'sample A / sample B'.

Inputs (nothing is recomputed here except means and 95 % intervals of stored per-placement results):
  data/article/s7_defects_density_pooled.csv            (written by s7_defects_density_pooled.py; run that first)
  data/msc_defects/homogenization.json             Theta(p), 192 x 192 tori
  data/msc_defects_verify/v07_summary.csv, v07b_summary.csv, v04b_theta.json
  data/msc_defects/summary_size.csv                sample A, size dependence

Palette: categorical slots blue / orange (checked with the palette validator for colour-vision deficiency);
identity is double-encoded (colour + marker shape for the placement law or density, filled/open for the sample).
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
ART = _R
ROOT = _R
RA = _os.path.join(_R, 'data', 'msc_defects')
RB = _os.path.join(_R, 'data', 'msc_defects_verify')
FIG = _os.path.join(_R, 'figures')
DATA = _os.path.join(_R, 'data', 'article')

BLUE, ORANGE = '#2a78d6', '#eb6834'
INK, INK2, MUTED, GRID = '#0b0b0b', '#52514e', '#8a8985', '#e4e3df'
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.labelsize': 8.5, 'axes.titlesize': 9,
    'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7.0,
    'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.7,
    'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.grid': True, 'axes.axisbelow': True,
    'lines.linewidth': 1.3, 'legend.frameon': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.bbox': 'tight',
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
})


def panel(ax, letter):
    ax.text(-0.02, 1.04, f'({letter})', transform=ax.transAxes, fontsize=8.5, ha='right', va='bottom', color=INK)


# =========================================================================== figure: density sweep
S = pd.read_csv(_os.path.join(_R, 'data', 'article', 's7_defects_density_pooled.csv'))
hom = json.load(open(_os.path.join(_R, 'data', 'msc_defects', 'homogenization.json')))
hp = np.array([r['p'] for r in hom['finite_p']]); hth = np.array([r['time_factor'] for r in hom['finite_p']])
R0 = 2493.0 / 14100.808049917203

LAW = {'uniform': (BLUE, 'o', 'uniform'), 'corner_thinned': (ORANGE, 's', 'corner-thinned')}
DX = 0.0035
fig, axs = plt.subplots(1, 3, figsize=(6.6, 2.75), gridspec_kw=dict(wspace=0.42))
pp = np.linspace(0, 0.36, 300)
for ax, key, ylab in ((axs[0], 'mfpt_fac', r'$\langle T\rangle\,/\,T_{\rm clean}$'), (axs[1], 'mode_fac', r'$\langle t^{*}\rangle\,/\,t^{*}_{\rm clean}$')):
    ax.plot(hp[hp <= 0.35], hth[hp <= 0.35], color=INK, lw=1.2, zorder=2)
    dm = pp[(np.pi - 1) * pp < 0.72]
    ax.plot(dm, 1 / (1 - (np.pi - 1) * dm), color=INK2, lw=1.0, ls='--', zorder=2)
    for law, (c, mk, lab) in LAW.items():
        g = S[S.law == law]
        ax.errorbar(g.p - DX, g[key + '_A'], yerr=g[key + '_ci_A'], color=c, marker=mk, ms=3.8, ls='', elinewidth=0.8, capsize=1.4,
                    mec='white', mew=0.5, zorder=4)
        ax.errorbar(g.p + DX, g[key + '_B'], yerr=g[key + '_ci_B'], color=c, marker=mk, ms=3.6, ls='', elinewidth=0.8, capsize=1.4,
                    mfc='white', mec=c, mew=0.9, zorder=4)
    ax.set_yscale('log'); ax.set_ylim(0.95, 15); ax.set_xlim(-0.01, 0.365)
    ax.set_yticks([1, 2, 3, 4, 6, 8]); ax.set_yticklabels(['1', '2', '3', '4', '6', '8']); ax.minorticks_off()
    ax.set_xlabel('blocked fraction $p$'); ax.set_ylabel(ylab)
h_ref = [Line2D([], [], color=INK, lw=1.2), Line2D([], [], color=INK2, lw=1.0, ls='--')]
axs[0].legend(h_ref, [r'$\Theta(p)=P_\infty\Sigma_0/\Sigma$ (tori)', r'dilute law $1/[1-(\pi-1)p]$'],
              loc='upper left', handlelength=2.0, borderaxespad=0.2)
h_s = [Line2D([], [], color=BLUE, marker='o', ms=4, ls='', mec='white', mew=0.5), Line2D([], [], color=ORANGE, marker='s', ms=4, ls='', mec='white', mew=0.5),
       Line2D([], [], color=INK2, marker='o', ms=4, ls=''), Line2D([], [], color=INK2, marker='o', ms=3.8, ls='', mfc='white', mew=0.9)]
axs[1].legend(h_s, ['uniform', 'corner-thinned', 'sample A (filled)', 'sample B (open)'], loc='upper left', handlelength=1.2, borderaxespad=0.2)
ax = axs[2]
ax.axhline(R0, color=MUTED, lw=0.8)
for law, (c, mk, lab) in LAW.items():
    g = S[S.law == law]
    ax.errorbar(g.p - DX, g.ratio_A, yerr=g.ratio_ci_A, color=c, marker=mk, ms=3.8, ls='', elinewidth=0.8, capsize=1.4, mec='white', mew=0.5, zorder=4)
    ax.errorbar(g.p + DX, g.ratio_B, yerr=g.ratio_ci_B, color=c, marker=mk, ms=3.6, ls='', elinewidth=0.8, capsize=1.4, mfc='white', mec=c, mew=0.9, zorder=4)
    ax.plot(np.r_[0, g.p], np.r_[R0, g.rom_P], color=c, lw=1.0, zorder=3)
h_c = [Line2D([], [], color=INK2, marker='o', ms=3.8, ls=''), Line2D([], [], color=INK2, lw=1.0)]
ax.legend(h_c, ['mean of per-placement ratios', r'$\langle t^{*}\rangle/\langle T\rangle$, pooled'],
          loc='upper left', handlelength=1.6, borderaxespad=0.2, bbox_to_anchor=(0.0, 0.93))
ax.text(0.36, R0 - 0.0018, 'clean lattice 0.1768', color=INK2, fontsize=7.0, va='top', ha='right')
ax.set_ylim(0.165, 0.302); ax.set_xlim(-0.01, 0.365)
ax.set_xlabel('blocked fraction $p$'); ax.set_ylabel(r'$t^{*}/T$')
for a, l in zip(axs, 'abc'):
    panel(a, l)
fig.savefig(_os.path.join(_R, 'figures', 's7_defects_density_sweep.pdf')); plt.close(fig)
print('wrote s7_defects_density_sweep.pdf')

# =========================================================================== figure: size dependence
M = pd.read_csv(_os.path.join(_R, 'data', 'msc_defects_verify', 'v07b_summary.csv')); Zs = pd.read_csv(_os.path.join(_R, 'data', 'msc_defects_verify', 'v07_summary.csv'))
A = pd.read_csv(_os.path.join(_R, 'data', 'msc_defects', 'summary_size.csv'))
TH = json.load(open(_os.path.join(_R, 'data', 'msc_defects_verify', 'v04b_theta.json')))
fig, axs = plt.subplots(1, 3, figsize=(6.6, 2.65), gridspec_kw=dict(wspace=0.56))
for p, col, mk in ((0.1, BLUE, 'o'), (0.2, ORANGE, 's')):
    th = TH[str(p)]['Theta']; band = TH[str(p)]['Theta_ci'] / th
    sh = 1.0 if p == 0.1 else 1.045
    g = M[np.isclose(M.p, p)]
    axs[0].errorbar(g.N * sh, g.over_theta, yerr=g.over_theta_ci, color=col, marker=mk, ms=3.8, lw=0.9, elinewidth=0.8, capsize=1.4, mec='white', mew=0.5,
                    label=f'$p$ = {p}, sample B')
    g = Zs[np.isclose(Zs.p, p)]
    axs[1].errorbar(g.N * sh, g.mode_fac / th, yerr=g.mode_fac_ci / th, color=col, marker=mk, ms=3.8, lw=0.9, elinewidth=0.8, capsize=1.4, mec='white', mew=0.5,
                    label=f'$p$ = {p}, sample B')
    a = A[np.isclose(A.p, p)]
    axs[1].errorbar(a.N * sh * 0.93, a.mode_factor / th, yerr=a.mode_factor_ci / th, color=col, marker=mk, ms=3.6, ls='', elinewidth=0.6, capsize=1.2,
                    mfc='white', mec=col, mew=0.9, label=f'$p$ = {p}, sample A')
    axs[2].errorbar(g.N * sh, 100 * g.rel_shift, yerr=100 * g.rel_shift_ci, color=col, marker=mk, ms=3.8, lw=0.9, elinewidth=0.8, capsize=1.4, mec='white', mew=0.5,
                    label=f'$p$ = {p}: mean of ratios')
    axs[2].plot(g.N * sh, 100 * g.rom_rel_shift, color=col, marker=mk, ms=3.6, lw=0.9, ls=':', mfc='white', mec=col, mew=0.9,
                label=f'$p$ = {p}: ratio of means')
    for ax in axs[:2]:
        ax.axhspan(1 - band, 1 + band, color=col, alpha=0.10, lw=0, zorder=1)
for ax, yl in ((axs[0], r'$\langle T\rangle\,/\,(\Theta\,T_{\rm clean})$'), (axs[1], r'$\langle t^{*}\rangle\,/\,(\Theta\,t^{*}_{\rm clean})$'),
               (axs[2], r'shift of $t^{*}/T$ relative to clean (%)')):
    ax.set_xscale('log'); ax.set_xlabel('lattice size $N$'); ax.set_ylabel(yl)
    ax.set_xticks([15, 25, 35, 50, 70, 100, 140, 200, 280]); ax.set_xticklabels(['15', '', '35', '', '70', '', '140', '', '280'])
    ax.minorticks_off(); ax.set_xlim(12.5, 330)
axs[0].axhline(1, color=INK, lw=0.8); axs[1].axhline(1, color=INK, lw=0.8); axs[2].axhline(0, color=INK, lw=0.8)
axs[0].set_ylim(0.912, 1.06); axs[1].set_ylim(0.912, 1.06); axs[2].set_ylim(-3.5, 25)
axs[0].legend(loc='lower left', handlelength=1.6, borderaxespad=0.2)
axs[1].legend(loc='lower right', handlelength=1.6, borderaxespad=0.2, ncol=1)
axs[2].legend(loc='upper right', handlelength=1.9, borderaxespad=0.2)
for a, l in zip(axs, 'abc'):
    panel(a, l)
fig.savefig(_os.path.join(_R, 'figures', 's7_defects_density_size.pdf')); plt.close(fig)
print('wrote s7_defects_density_size.pdf')
