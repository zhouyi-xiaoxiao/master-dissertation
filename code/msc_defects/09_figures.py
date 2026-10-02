"""09_figures.py -- publication figures (vector PDF + PNG) from the saved raw results.

Palette: validated categorical slots (blue, orange, aqua, yellow) in fixed order; series identity
is always double-encoded (marker shape / line style + direct label or legend).  Diverging maps use
blue <-> red with a neutral grey midpoint.  No dual axes.
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import os, sys, json, glob
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, SymLogNorm, ListedColormap
import fptcore as fc, ensemble

DATA = _os.path.join(_R, 'data', 'msc_defects'); FIG = _os.path.join(_R, 'out', 'figures', 'msc_defects')
os.makedirs(FIG, exist_ok=True)

BLUE, ORANGE, AQUA, YELLOW = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
INK, INK2, MUTED, GRID = '#0b0b0b', '#52514e', '#8a8985', '#e4e3df'
DIV = LinearSegmentedColormap.from_list('div', ['#104281', '#2a78d6', '#9ec5f4', '#f0efec', '#f3a9a8', '#e34948', '#8f1f1e'])

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.labelsize': 8.5, 'axes.titlesize': 9,
    'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7.5,
    'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.7,
    'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.grid': True, 'axes.axisbelow': True,
    'lines.linewidth': 1.5, 'lines.markersize': 4.5, 'legend.frameon': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.bbox': 'tight', 'savefig.dpi': 300,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
})

def save(fig, name):
    fig.savefig(os.path.join(FIG, name + '.pdf'))
    fig.savefig(os.path.join(FIG, name + '.png'), dpi=300)
    plt.close(fig)
    print('wrote', name)

def panel(ax, letter):
    ax.text(-0.02, 1.04, f'({letter})', transform=ax.transAxes, fontsize=9, fontweight='bold',
            ha='right', va='bottom', color=INK)

N, q = 35, 0.8
S_, T_ = (0, 0), (N - 1, N - 1)
summ = pd.read_csv(_os.path.join(_R, 'data', 'msc_defects', 'summary_sweep_N35.csv'))
hom = json.load(open(_os.path.join(_R, 'data', 'msc_defects', 'homogenization.json')))
hp = np.array([r['p'] for r in hom['finite_p']]); htf = np.array([r['time_factor'] for r in hom['finite_p']])
base_mfpt, base_mode = 14100.808049937992, 2493

def draw_lattice(ax, mask, title):
    img = np.where(mask.T, 1.0, 0.0)
    ax.imshow(img, origin='lower', cmap=ListedColormap(['#f6f5f2', '#3b3a37']), interpolation='nearest',
              extent=(-0.5, N - 0.5, -0.5, N - 0.5))
    ax.plot([0], [0], marker='o', ms=4.5, color=BLUE, mec='white', mew=0.6, zorder=5)
    ax.plot([N - 1], [N - 1], marker='*', ms=7.5, color=ORANGE, mec='white', mew=0.5, zorder=5)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(True); s.set_linewidth(0.6)
    ax.set_title(title, fontsize=7.5, pad=3)

# =============================================================================== Fig 1
def fig1():
    pm = np.load(_os.path.join(_R, 'data', 'msc_defects', 'structured_pmfs.npz'))
    masks = np.load(_os.path.join(_R, 'data', 'msc_defects', 'structured_masks.npz'))
    det = pd.read_csv(_os.path.join(_R, 'data', 'msc_defects', 'structured_deterministic.csv')).set_index('name')
    ss = np.random.SeedSequence([20261001, N, 1, 2000, 0])
    rmask, _ = ensemble.draw_connected(N, 0.20, 'uniform', ss)
    rs, _ = fc.fpt_summary(~rmask, q, S_, T_, want_series=True, series_tmax=200000)
    cases = [('clean', 'clean lattice', INK2, '-', None),
             ('random', 'random, p = 0.20', BLUE, '-', rmask),
             ('target_box_R5_g1', 'target enclosed (10 sites)', ORANGE, '-', masks['target_box_R5_g1']),
             ('start_box_R8_g1', 'start enclosed (16 sites)', AQUA, '-', masks['start_box_R8_g1'])]
    fig = plt.figure(figsize=(7.2, 3.5))
    gs = fig.add_gridspec(2, 4, width_ratios=[2.6, 2.6, 1.35, 1.35], wspace=0.12, hspace=0.22)
    ax = fig.add_subplot(gs[:, :2])
    for key, lab, col, ls, m in cases:
        if key == 'random':
            t, f = rs['series_t'], rs['series_f']; mode, mf = rs['mode'], rs['mfpt']
        else:
            t, f = pm[key + '__t'], pm[key + '__f']; mode, mf = det.loc[key, 'mode'], det.loc[key, 'mfpt']
        ax.plot(t, f * 1e4, color=col, ls=ls, lw=1.4, label=f'{lab}  (mode/MFPT = {mode / mf:.3f})')
        fm = np.interp(mode, t, f); fT = np.interp(mf, t, f)
        ax.plot([mode], [fm * 1e4], 'o', color=col, ms=5.5, mec='white', mew=0.8, zorder=6)
        ax.plot([mf], [fT * 1e4], 'v', color=col, ms=5.5, mec='white', mew=0.8, zorder=6)
    ax.set_xscale('log'); ax.set_xlim(150, 2.2e5); ax.set_ylim(0, 1.08)
    ax.set_xlabel('time t (steps)'); ax.set_ylabel(r'first-passage probability $f(t)$  ($\times 10^{-4}$)')
    h, l = ax.get_legend_handles_labels()
    from matplotlib.lines import Line2D
    h += [Line2D([], [], marker='o', color=MUTED, ls='', ms=5), Line2D([], [], marker='v', color=MUTED, ls='', ms=5)]
    l += ['mode (exact argmax)', 'mean (MFPT)']
    ax.legend(h, l, loc='upper left', fontsize=6.6, handlelength=1.6, ncol=1, borderaxespad=0.3)
    panel(ax, 'a')
    thumbs = [(rmask, 'random, p = 0.20'), (masks['target_box_R5_g1'], 'target enclosed'),
              (masks['start_box_R8_g1'], 'start enclosed'), (masks['serpentine_k4_g3'], 'serpentine (4 walls)')]
    for k, (m, ttl) in enumerate(thumbs):
        a = fig.add_subplot(gs[k // 2, 2 + k % 2])
        draw_lattice(a, m, ttl)
        if k == 0:
            a.text(-0.04, 1.12, '(b)', transform=a.transAxes, fontsize=9, fontweight='bold', ha='right', va='bottom')
    save(fig, 'fig1_pmf_examples')

# =============================================================================== Fig 2
def fig2():
    fo = json.load(open(_os.path.join(_R, 'data', 'msc_defects', 'summary_first_order.json')))
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.75), gridspec_kw=dict(wspace=0.36))
    pp = np.linspace(0, 0.35, 200)
    style = {'uniform': (BLUE, 'o', 'uniform placement'), 'smart': (ORANGE, 's', 'corner-thinned placement\n(fewer defects near corners)')}
    for ax, key, ylab in ((axs[0], 'mfpt_factor', 'MFPT(p) / MFPT(0)'), (axs[1], 'mode_factor', 'mode(p) / mode(0)')):
        ax.plot(hp[hp <= 0.35], htf[hp <= 0.35], color=INK, lw=1.2, label=r'homogenised, $D_0/D_{\rm eff}(p)$')
        dm = pp[(np.pi - 1) * pp < 0.72]
        ax.plot(dm, 1 / (1 - (np.pi - 1) * dm), color=INK2, lw=1.0, ls='--', label=r'dilute, $1/[1-(\pi-1)p]$')
        for sch in ('uniform', 'smart'):
            g = summ[summ.scheme == sch]; c, mk, lab = style[sch]
            ax.errorbar(g.p, g[key], yerr=g[key + '_ci'], color=c, marker=mk, ms=3.8, lw=0.9, capsize=1.5,
                        mec='white', mew=0.5, label=lab.split('\n')[0], zorder=4)
        ax.set_yscale('log'); ax.set_ylim(0.95, 8.5); ax.set_xlim(-0.005, 0.36)
        ax.set_yticks([1, 2, 3, 4, 6, 8]); ax.set_yticklabels(['1', '2', '3', '4', '6', '8']); ax.minorticks_off()
        ax.set_xlabel('defect fraction p'); ax.set_ylabel(ylab)
    axs[0].legend(loc='upper left', fontsize=6.3, handlelength=1.8, borderaxespad=0.2)
    ax = axs[2]
    for sch in ('uniform', 'smart'):
        g = summ[summ.scheme == sch]; c, mk, lab = style[sch]
        ax.errorbar(g.p, g.ratio, yerr=g.ratio_ci, color=c, marker=mk, ms=3.8, lw=0.9, capsize=1.5, mec='white', mew=0.5, zorder=4)
        sl = fo[sch]['pred_slope_ratio']
        x = np.linspace(0, 0.08, 10)
        ax.plot(x, 0.176798 * (1 + sl * x), color=c, lw=0.9, ls='--', zorder=3)
    ax.axhline(0.176798, color=MUTED, lw=0.7, ls='-')
    from matplotlib.lines import Line2D
    hh = [Line2D([], [], color=BLUE, marker='o', ms=3.8, lw=0.9, mec='white', mew=0.5),
          Line2D([], [], color=ORANGE, marker='s', ms=3.8, lw=0.9, mec='white', mew=0.5),
          Line2D([], [], color=INK2, lw=0.9, ls='--')]
    ax.legend(hh, ['uniform placement', 'corner-thinned placement', 'first-order theory'],
              loc='upper left', fontsize=6.2, handlelength=1.9, borderaxespad=0.2, bbox_to_anchor=(0.0, 0.95))
    ax.text(0.355, 0.1752, 'clean lattice 0.1768', color=INK2, fontsize=6.3, va='top', ha='right')
    ax.set_ylim(0.165, 0.292); ax.set_xlim(-0.005, 0.36)
    ax.set_xlabel('defect fraction p'); ax.set_ylabel('mode / MFPT')
    for a, l in zip(axs, 'abc'):
        panel(a, l)
    save(fig, 'fig2_density_sweep')

# =============================================================================== Fig 3
def fig3():
    d = pd.read_csv(_os.path.join(_R, 'data', 'msc_defects', 'sensitivity_N35.csv')).iloc[1:]
    A = np.full((N, N), np.nan); B = np.full((N, N), np.nan)
    A[d.i, d.j] = N * N * d.chi_mfpt; B[d.i, d.j] = N * N * d.chi_mode
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.7), gridspec_kw=dict(wspace=0.28))
    norm = SymLogNorm(linthresh=1.0, vmin=-500, vmax=500, base=10)
    titles = [r'$N^2\chi_{\rm MFPT}(x)$', r'$N^2\chi_{\rm mode}(x)$', r'$N^2[\chi_{\rm mode}-\chi_{\rm MFPT}](x)$']
    for ax, M, ttl, l in zip(axs, (A, B, B - A), titles, 'abc'):
        im = ax.imshow(M.T, origin='lower', cmap=DIV, norm=norm, interpolation='nearest')
        ax.plot([0], [0], marker='o', ms=4.5, color='white', mec=INK, mew=0.8)
        ax.plot([N - 1], [N - 1], marker='*', ms=8, color='white', mec=INK, mew=0.8)
        ax.set_xticks([0, 17, 34]); ax.set_yticks([0, 17, 34]); ax.grid(False)
        ax.set_title(ttl, fontsize=8.5)
        ax.set_xlabel('i')
        panel(ax, l)
    axs[0].set_ylabel('j')
    cb = fig.colorbar(im, ax=axs, shrink=0.85, pad=0.02, ticks=[-100, -10, -1, 0, 1, 10, 100])
    cb.ax.set_yticklabels(['−100', '−10', '−1', '0', '1', '10', '100'])
    cb.set_label('relative change per unit defect fraction', fontsize=7.5)
    cb.outline.set_linewidth(0.5)
    save(fig, 'fig3_sensitivity_maps')

# =============================================================================== Fig 4
def fig4():
    det = pd.read_csv(_os.path.join(_R, 'data', 'msc_defects', 'structured_deterministic.csv'))
    loc = pd.read_csv(_os.path.join(_R, 'data', 'msc_defects', 'summary_structured_local.csv'))
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.9), gridspec_kw=dict(wspace=0.38, width_ratios=[1.35, 1, 1]))
    ax = axs[0]
    def group(name):
        if name.startswith(('target_box', 'block6_near_target', 'wall_neartarget')):
            return 'near target', ORANGE, 's'
        if name.startswith(('start_box', 'block6_near_start', 'wall_nearstart')):
            return 'near start', AQUA, '^'
        if name.startswith(('wall_mid', 'serpentine')):
            return 'walls / serpentines', BLUE, 'o'
        return 'other', MUTED, 'D'
    seen = set()
    x0, x1 = 0.33, 14
    xx = np.array([x0, x1])
    r0 = base_mode / base_mfpt
    for fac, lab in ((1, 'ratio unchanged'), (2, '×2'), (0.5, '×½')):
        ax.plot(xx, fac * xx, color=GRID if fac != 1 else MUTED, lw=0.9, ls='-' if fac == 1 else '--', zorder=1)
    ax.text(11.5, 10.2, 'ratio unchanged', fontsize=6.0, color=INK2, rotation=41, ha='right', va='top')
    ax.text(4.3, 9.6, 'ratio ×2', fontsize=6.0, color=INK2, rotation=41, ha='center', va='bottom')
    ax.text(12.5, 5.5, 'ratio ×½', fontsize=6.0, color=INK2, rotation=41, ha='right', va='top')
    for _, r in det.iterrows():
        if r['name'] == 'clean':
            continue
        g, c, mk = group(r['name'])
        ax.plot(r.mfpt / base_mfpt, r['mode'] / base_mode, marker=mk, color=c, ms=5, mec='white', mew=0.6, ls='',
                label=g if g not in seen else None, zorder=4)
        seen.add(g)
    lab = {'target_box_R5_g1': ('target enclosed', (3, -11), 'left'), 'start_box_R8_g1': ('start enclosed', (-5, 1), 'right'),
           'serpentine_k6_g3': ('serpentine, 6 walls', (-4, -12), 'right'),
           'wall_neartarget_gap_centre_g3': ('wall near target', (5, 1), 'left'),
           'wall_mid_gap_centre_g1': ('mid wall, 1-site gap', (6, -2), 'left')}
    for nm, (txt, off, ha) in lab.items():
        r = det[det['name'] == nm].iloc[0]
        ax.annotate(txt, (r.mfpt / base_mfpt, r['mode'] / base_mode), xytext=off, textcoords='offset points',
                    fontsize=6.3, color=INK, ha=ha)
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(x0, x1); ax.set_ylim(0.85, 26)
    ax.set_xticks([1, 2, 4, 8]); ax.set_xticklabels(['1', '2', '4', '8']); ax.set_yticks([1, 2, 4, 8, 16]); ax.set_yticklabels(['1', '2', '4', '8', '16'])
    ax.minorticks_off()
    ax.set_xlabel('MFPT / MFPT(clean)'); ax.set_ylabel('mode / mode(clean)')
    ax.legend(loc='upper left', fontsize=6.2, handletextpad=0.2, borderaxespad=0.3)
    st = {'anywhere': (BLUE, 'o', 'anywhere'), 'near_target': (ORANGE, 's', 'near target'),
          'near_start': (AQUA, '^', 'near start'), 'centre': (YELLOW, 'D', 'centre')}
    for ax, key, ylab in ((axs[1], 'mfpt_factor', 'factor relative to clean lattice'), (axs[2], 'ratio', 'mode / MFPT')):
        for kind, (c, mk, lab_) in st.items():
            g = loc[loc.kind == kind].sort_values('M')
            if key == 'mfpt_factor':
                ax.errorbar(g.M, g.mfpt_factor, yerr=g.mfpt_factor_ci, color=c, marker=mk, ms=4, lw=1.1, capsize=1.5, mec='white', mew=0.5, label=lab_)
                ax.errorbar(g.M, g.mode_factor, yerr=g.mode_factor_ci, color=c, marker=mk, ms=4, lw=1.1, ls='--', capsize=1.5, mfc='white', mew=0.9)
            else:
                ax.errorbar(g.M, g.ratio, yerr=g.ratio_ci, color=c, marker=mk, ms=4, lw=1.1, capsize=1.5, mec='white', mew=0.5)
                ax.text(41.5, g.ratio.iloc[-1], lab_, color=INK, fontsize=6.5, va='center')
        ax.set_xlabel('blocked sites M in the region'); ax.set_ylabel(ylab); ax.set_xticks([10, 20, 30, 40])
    axs[1].set_yscale('log'); axs[1].set_yticks([1, 1.5, 2, 3]); axs[1].set_yticklabels(['1', '1.5', '2', '3']); axs[1].minorticks_off()
    from matplotlib.lines import Line2D
    h, l = axs[1].get_legend_handles_labels()
    h = [x[0] if isinstance(x, tuple) or hasattr(x, 'lines') else x for x in h]
    h += [Line2D([], [], color=INK2, lw=1.1), Line2D([], [], color=INK2, lw=1.1, ls='--')]
    l += ['MFPT', 'mode']
    axs[1].legend(h, l, loc='upper left', fontsize=6.2, handlelength=1.8, borderaxespad=0.1)
    axs[2].axhline(r0, color=MUTED, lw=0.7); axs[2].set_xlim(8, 52)
    for a, l_ in zip(axs, 'abc'):
        panel(a, l_)
    save(fig, 'fig4_structured')

# =============================================================================== Fig 5
def fig5():
    sw = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(_os.path.join(_R, 'data', 'msc_defects', 'sweep', 'sweep_N35_uniform_*.csv')))])
    g = sw[np.isclose(sw.p, 0.20)]
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.7), gridspec_kw=dict(wspace=0.4))
    ax = axs[0]
    gm = 4 * g.n_cluster / q * g.G_tt
    for dt, c, mk, lab in ((2, BLUE, 'o', 'both target neighbours open'), (1, ORANGE, 's', 'one target neighbour blocked')):
        h = g[g.deg_t == dt]
        ax.plot(4 * h.n_cluster / q * h.G_tt / 1e4, h.mfpt / 1e4, ls='', marker=mk, ms=2.8, color=c, alpha=0.75, mec='none', label=lab)
    lim = [1.0, 7.5]
    ax.plot(lim, lim, color=MUTED, lw=0.8)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel(r'$(4n/q)\,G_{aa}$  ($\times10^4$ steps)'); ax.set_ylabel(r'MFPT  ($\times10^4$ steps)')
    ax.legend(loc='upper left', fontsize=6.2, handletextpad=0.1, borderaxespad=0.1)
    ax = axs[1]
    for dt, c, mk in ((2, BLUE, 'o'), (1, ORANGE, 's')):
        h = g[g.deg_t == dt]
        ax.plot(h.mfpt / 1e4, h['mode'] / 1e3, ls='', marker=mk, ms=2.8, color=c, alpha=0.75, mec='none')
    b = np.polyfit(np.log(g.mfpt), np.log(g['mode']), 1)
    xx = np.linspace(g.mfpt.min(), g.mfpt.max(), 50)
    ax.plot(xx / 1e4, np.exp(np.polyval(b, np.log(xx))) / 1e3, color=INK, lw=1.0)
    xm, ym = np.exp(np.log(g.mfpt).mean()), np.exp(np.log(g['mode']).mean())
    ax.plot(xx / 1e4, ym * (xx / xm) / 1e3, color=MUTED, lw=0.9, ls='--')
    ax.text(0.97, 0.06, f'fitted slope {b[0]:.2f}', transform=ax.transAxes, fontsize=6.5, ha='right', color=INK)
    ax.text(0.40, 0.93, 'slope 1\n(pure time dilation)', transform=ax.transAxes, fontsize=6.3, ha='right', va='top', color=INK2)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xticks([2, 3, 4, 6]); ax.set_xticklabels(['2', '3', '4', '6']); ax.set_yticks([3, 4, 5, 6, 8]); ax.set_yticklabels(['3', '4', '5', '6', '8'])
    ax.minorticks_off()
    ax.set_xlabel(r'MFPT  ($\times10^4$ steps)'); ax.set_ylabel(r'mode  ($\times10^3$ steps)')
    ax = axs[2]
    u = summ[summ.scheme == 'uniform']; s = summ[summ.scheme == 'smart']
    hpv = np.array([r['p'] for r in hom['finite_p']]); hs = np.array([r['sigma'] for r in hom['finite_p']])
    ax.plot(hpv[hpv <= 0.3], 1 / hs[hpv <= 0.3], color=INK, lw=1.1, label=r'homogenised $\sigma_0/\sigma(p)$')
    ax.plot(u.p[u.p <= 0.3], u.Gdiff_factor[u.p <= 0.3], color=BLUE, marker='o', ms=3.6, lw=0.9, mec='white', mew=0.5, label='uniform')
    ax.plot(s.p[s.p <= 0.3], s.Gdiff_factor[s.p <= 0.3], color=ORANGE, marker='s', ms=3.6, lw=0.9, mec='white', mew=0.5, label='corner-thinned placement')
    ax.set_yscale('log'); ax.set_yticks([1, 2, 3, 5]); ax.set_yticklabels(['1', '2', '3', '5']); ax.minorticks_off()
    ax.set_xlabel('defect fraction p'); ax.set_ylabel(r'$[G_{aa}-G_{sa}]$ relative to clean lattice')
    ax.legend(loc='upper left', fontsize=6.3, borderaxespad=0.1)
    for a, l_ in zip(axs, 'abc'):
        panel(a, l_)
    save(fig, 'fig5_mechanism')

# =============================================================================== Fig 6
def fig6():
    sz = pd.read_csv(_os.path.join(_R, 'data', 'msc_defects', 'summary_size.csv'))
    fig, axs = plt.subplots(1, 2, figsize=(6.2, 2.8), gridspec_kw=dict(wspace=0.36))
    ax = axs[0]
    for p, c, mk, lab in ((0.0, INK2, 'D', 'clean'), (0.10, BLUE, 'o', 'p = 0.10'), (0.20, ORANGE, 's', 'p = 0.20')):
        g = sz[np.isclose(sz.p, p)].sort_values('N')
        ax.errorbar(g.N, g.inv_ratio, yerr=g.inv_ratio_ci, color=c, marker=mk, ms=4, lw=1.0, capsize=1.5, mec='white', mew=0.5, label=lab)
    ax.set_xscale('log'); ax.set_xticks([10, 20, 50, 100, 200]); ax.set_xticklabels(['10', '20', '50', '100', '200']); ax.minorticks_off()
    ax.set_xlabel('lattice size N'); ax.set_ylabel('MFPT / mode'); ax.legend(loc='upper left')
    ax = axs[1]
    for p, c, mk in ((0.10, BLUE, 'o'), (0.20, ORANGE, 's')):
        g = sz[np.isclose(sz.p, p)].sort_values('N')
        ax.errorbar(g.N, g.mfpt_factor, yerr=g.mfpt_factor_ci, color=c, marker=mk, ms=4, lw=1.0, capsize=1.5, mec='white', mew=0.5)
        ax.errorbar(g.N, g.mode_factor, yerr=g.mode_factor_ci, color=c, marker=mk, ms=4, lw=1.0, ls='--', capsize=1.5, mfc='white', mew=0.9)
        ax.axhline(g.hom_factor.iloc[0], color=MUTED, lw=0.8)
        ax.text(150, g.hom_factor.iloc[0], f' homogenised\n p = {p:.2f}', fontsize=6.3, color=INK2, va='center', ha='left')
    from matplotlib.lines import Line2D
    ax.legend([Line2D([], [], color=INK2, lw=1.0, marker='o', ms=3.5), Line2D([], [], color=INK2, lw=1.0, ls='--', marker='o', mfc='white', ms=3.5)],
              ['MFPT(p)/MFPT(0)', 'mode(p)/mode(0)'], loc='center left', fontsize=6.5, bbox_to_anchor=(0.0, 0.5))
    ax.set_xscale('log'); ax.set_xlim(12, 330); ax.set_xticks([15, 35, 70, 140]); ax.set_xticklabels(['15', '35', '70', '140']); ax.minorticks_off()
    ax.set_xlabel('lattice size N'); ax.set_ylabel('time factor')
    for a, l_ in zip(axs, 'ab'):
        panel(a, l_)
    save(fig, 'fig6_size_dependence')

# =============================================================================== Fig 7
def fig7():
    s1 = pd.read_csv(_os.path.join(_R, 'data', 'msc_defects', 'beyond_1d_single.csv'))
    bj = json.load(open(_os.path.join(_R, 'data', 'msc_defects', 'summary_beyond.json')))
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.7), gridspec_kw=dict(wspace=0.46))
    ax = axs[0]
    m0, mode0 = 12375.0, 4124
    ax.plot(s1.pos + 1, s1.mfpt / m0 - 1, color=BLUE, lw=1.4, label='MFPT (exact formula)')
    ax.plot(s1.pos + 1, s1['mode'] / mode0 - 1, color=ORANGE, lw=1.2, ls='--', label='mode')
    ax.set_xlabel('barrier position k (bond k–k+1)'); ax.set_ylabel('relative shift')
    ax.legend(loc='upper left', fontsize=6.5); ax.set_title('1D: one barrier', fontsize=7.5)
    ax = axs[1]
    pm_ = pd.DataFrame(bj['2d_permeable'])
    xk = pm_.kappa.replace(0.0, 1e-4)
    ax.errorbar(xk, pm_.mfpt / base_mfpt, yerr=pm_.mfpt_ci / base_mfpt, color=BLUE, marker='o', ms=4, lw=1.0, capsize=1.5, mec='white', mew=0.5, label='MFPT / MFPT(clean)')
    ax.errorbar(xk, pm_['mode'] / base_mode, yerr=pm_.mode_ci / base_mode, color=ORANGE, marker='s', ms=4, lw=1.0, ls='--', capsize=1.5, mec='white', mew=0.5, label='mode / mode(clean)')
    ax.set_xscale('log'); ax.set_xticks([1e-4, 1e-3, 1e-2, 1e-1, 1]); ax.set_xticklabels(['0\n(inert)', '$10^{-3}$', '$10^{-2}$', '0.1', '1']); ax.minorticks_off()
    ax.set_xlabel('obstacle permeability κ'); ax.set_ylabel('factor'); ax.set_ylim(0.97, 1.66); ax.legend(loc='upper right', fontsize=6.3)
    ax.set_title('2D: permeable obstacles', fontsize=7.5)
    ax = axs[2]
    tr = pd.DataFrame(bj['2d_traps'])
    ax.plot(tr.Delta, tr.H, color=BLUE, marker='o', ms=4, lw=1.0, mec='white', mew=0.5, label='hitting probability H')
    ax.plot(tr.Delta, tr.cond_mean / base_mfpt, color=ORANGE, marker='s', ms=4, lw=1.0, ls='--', mec='white', mew=0.5, label='conditional mean / MFPT(clean)')
    ax.plot(tr.Delta, tr['ratio'], color=AQUA, marker='^', ms=4, lw=1.0, ls='-.', mec='white', mew=0.5, label='mode / conditional mean')
    ax.plot(tr.Delta, tr.H_meanfield, color=MUTED, lw=0.8, ls=':', label='H, mean-field killing')
    ax.set_xscale('log'); ax.set_ylim(0, 1.5); ax.set_xlabel('killing probability Δ per step')
    ax.set_ylabel('dimensionless value'); ax.legend(loc='upper center', fontsize=6.0, ncol=1, borderaxespad=0.1)
    ax.set_title('2D: partial traps, Eq. (4.1)', fontsize=7.5)
    for a, l_ in zip(axs, 'abc'):
        panel(a, l_)
    save(fig, 'fig7_beyond_inert')

# =============================================================================== Fig 8
def fig8():
    sw = pd.concat([pd.read_csv(f_) for f_ in sorted(glob.glob(_os.path.join(_R, 'data', 'msc_defects', 'sweep', '*.csv')))])
    det = pd.read_csv(_os.path.join(_R, 'data', 'msc_defects', 'structured_deterministic.csv'))
    loc = pd.read_csv(_os.path.join(_R, 'data', 'msc_defects', 'structured_local.csv'))
    cl = pd.read_csv(_os.path.join(_R, 'data', 'msc_defects', 'size_clean.csv'))
    d1 = pd.read_csv(_os.path.join(_R, 'data', 'msc_defects', 'beyond_1d_random.csv'))
    fig, ax = plt.subplots(figsize=(3.5, 2.9))
    ax.plot(sw.ratio, sw.cv, ls='', marker='.', ms=1.6, color=BLUE, alpha=0.35, rasterized=True, label='random placements, N = 35')
    ax.plot(loc.ratio, loc.cv, ls='', marker='.', ms=1.6, color=AQUA, alpha=0.45, rasterized=True, label='localised placements')
    ax.plot(det.ratio, det.cv, ls='', marker='s', ms=3.4, color=ORANGE, mec='white', mew=0.4, label='walls, enclosures, blocks')
    ax.plot(cl['mode'] / cl.mfpt, cl.sd / cl.mfpt, color=INK, lw=1.0, marker='D', ms=2.8, mec='white', mew=0.3, label='clean lattices, N = 10–200')
    ax.plot([d1.ratio.iloc[0]], [d1.cv.iloc[0]], marker='*', ms=8, color=YELLOW, mec=INK, mew=0.5, ls='', label='clean 1D chain, N = 100')
    ax.set_xlim(0.04, 0.36); ax.set_ylim(0.80, 0.985)
    ax.set_xlabel('mode / MFPT'); ax.set_ylabel('coefficient of variation  SD / MFPT')
    ax.legend(loc='lower left', fontsize=6.3, handletextpad=0.3, markerscale=1.0)
    ax.annotate('capture-limited\n(exponential-like)', (0.075, 0.970), xytext=(0.125, 0.975), fontsize=6.3, color=INK2, va='center')
    ax.annotate('transit-limited\n(1D-like)', (0.33, 0.82), xytext=(0.275, 0.905), fontsize=6.3, color=INK2, va='center')
    save(fig, 'fig8_shape_family')

if __name__ == '__main__':
    which = sys.argv[1:] or ['1', '2', '3', '4', '5', '6', '7', '8']
    for w in which:
        globals()['fig' + w]()
