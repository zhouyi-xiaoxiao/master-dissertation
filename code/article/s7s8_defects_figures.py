#!/usr/bin/env python
"""Figures of Section 8 and Supplementary Section S9 drawn at the text width of the article (16 cm) and in its notation.

    figures/s7_defects_density_pmf.pdf           exact PMFs of four configurations              (Suppl. S9.1)
    figures/s8_defects_placement_maps.pdf        single-defect susceptibilities                 (Suppl. S9.6)
    figures/s8_defects_placement_structured.pdf  deterministic geometries; localised placements (Sec. 8.4)    
    figures/s8_defects_placement_twotime.pdf     two time scales                                (Suppl. S9.8)
    figures/s8_defects_placement_shape.pdf       coefficient of variation against t*/T          (Suppl. S9.9)

Same data and quantities as code/msc_defects/09_figures.py (its fig1, fig3, fig4, fig8) and
code/msc_defects_verify/v10_figures.py (its vfig5), redrawn so that nothing is rescaled in the
article (no text below 7 pt) and every label uses the symbols of the article.  Differences from those figures:
  * the PMF figure shows only configurations whose PMF is plotted;
  * the two-time-scale figure shows all SIX deterministic cases of v09b_twotime.csv, including the wall near
    the target, for which the frozen-tau heuristic fails;
  * the shape-family figure shows all 13 631 configurations of sample A and the fitted quadratic of
    Supplementary Section S9.9.

What is computed here: the PMF of one uniform placement (p = 0.20, the first placement of the sweep, regenerated
from its seed with the research library) and the spectrum of the clean 35 x 35 lattice (dense symmetric
eigenproblem of order 1224).  Everything else is read from the stored result files.

Inputs : data/msc_defects/{structured_pmfs.npz, structured_masks.npz, structured_deterministic.csv,
         sensitivity_N35.csv, summary_structured_local.csv, structured_local.csv, sweep/*.csv, size_clean.csv,
         beyond_1d_random.csv}, data/msc_defects_verify/v09b_twotime.csv,
         data/article/s8_defects_placement_checks.json (quadratic of the shape family)
Usage  : python s7s8_defects_figures.py          (about 20 s)
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, SymLogNorm
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = _R            # 
DATA = _os.path.join(_R, 'data', 'msc_defects')
VDATA = _os.path.join(_R, 'data', 'msc_defects_verify')
LIB = _os.path.join(_R, 'code', 'msc_defects')
ADATA = _os.path.join(_R, 'data', 'article')
FIG = _os.path.join(_R, 'figures')
sys.path.insert(0, LIB)
import fptcore as fc          # noqa: E402  (research library: exact PMF of one configuration)
import ensemble               # noqa: E402  (research library: seeded placements)

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8985", "#e4e3df"
DIV = LinearSegmentedColormap.from_list("div", ["#104281", "#2a78d6", "#9ec5f4", "#f0efec", "#f3a9a8", "#e34948",
                                                "#8f1f1e"])
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 8.5, "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.2,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.7,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.grid": True, "axes.axisbelow": True,
    "lines.linewidth": 1.4, "lines.markersize": 4.5, "legend.frameon": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "mathtext.fontset": "dejavusans",
    "figure.facecolor": "white", "axes.facecolor": "white",
})
W = 6.3                                                                 # inches = text width of the article
SMALL = 7.0
N, q = 35, 0.8
S_, T_ = (0, 0), (N - 1, N - 1)
base_mfpt, base_mode = 14100.808049937992, 2493


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    plt.close(fig)
    print("wrote", name + ".pdf")


def panel(ax, text):
    ax.set_title(text, loc="left")


def draw_lattice(ax, mask, title):
    img = np.where(mask.T, 1.0, 0.0)
    ax.imshow(img, origin="lower", cmap=ListedColormap(["#f6f5f2", "#3b3a37"]), interpolation="nearest",
              extent=(-0.5, N - 0.5, -0.5, N - 0.5))
    ax.plot([0], [0], marker="o", ms=4.5, color=BLUE, mec="white", mew=0.6, zorder=5)
    ax.plot([N - 1], [N - 1], marker="*", ms=7.5, color=ORANGE, mec="white", mew=0.5, zorder=5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.6)
    ax.set_title(title, fontsize=SMALL, pad=3)


# =============================================================================== exact PMFs
def fig_pmf():
    pm = np.load(_os.path.join(_R, 'data', 'msc_defects', 'structured_pmfs.npz'))
    masks = np.load(_os.path.join(_R, 'data', 'msc_defects', 'structured_masks.npz'))
    det = pd.read_csv(_os.path.join(_R, 'data', 'msc_defects', 'structured_deterministic.csv')).set_index("name")
    ss = np.random.SeedSequence([20261001, N, 1, 2000, 0])             # first uniform placement of the sweep, p = 0.20
    rmask, _ = ensemble.draw_connected(N, 0.20, "uniform", ss)
    rs, _ = fc.fpt_summary(~rmask, q, S_, T_, want_series=True, series_tmax=200000)
    cases = [("clean", "clean lattice", INK2, None),
             ("random", "uniform placement, $p=0.20$", BLUE, rmask),
             ("target_box_R5_g1", "target enclosed (10 sites)", ORANGE, masks["target_box_R5_g1"]),
             ("start_box_R8_g1", "start enclosed (16 sites)", AQUA, masks["start_box_R8_g1"])]
    fig = plt.figure(figsize=(W, 3.1))
    gs = fig.add_gridspec(3, 2, width_ratios=[4.6, 1.0], wspace=0.04, hspace=0.30,
                          left=0.085, right=0.995, top=0.94, bottom=0.125)
    ax = fig.add_subplot(gs[:, 0])
    for key, lab, col, m in cases:
        if key == "random":
            t, f = rs["series_t"], rs["series_f"]
            mode, mf = rs["mode"], rs["mfpt"]
        else:
            t, f = pm[key + "__t"], pm[key + "__f"]
            mode, mf = det.loc[key, "mode"], det.loc[key, "mfpt"]
        ax.plot(t, f * 1e4, color=col, lw=1.4, label=f"{lab}  ($t^{{*}}/T={mode / mf:.3f}$)")
        fm = np.interp(mode, t, f)
        fT = np.interp(mf, t, f)
        ax.plot([mode], [fm * 1e4], "o", color=col, ms=5.5, mec="white", mew=0.8, zorder=6)
        ax.plot([mf], [fT * 1e4], "v", color=col, ms=5.5, mec="white", mew=0.8, zorder=6)
    ax.set_xscale("log")
    ax.set_xlim(150, 2.2e5)
    ax.set_ylim(0, 1.12)
    ax.set_xlabel("time $t$ (steps)")
    ax.set_ylabel(r"first-passage probability $f(t)$  ($\times10^{-4}$)")
    h, l = ax.get_legend_handles_labels()
    h += [Line2D([], [], marker="o", color=MUTED, ls="", ms=5), Line2D([], [], marker="v", color=MUTED, ls="", ms=5)]
    l += ["mode $t^{*}$ (exact maximiser)", "mean $T$"]
    ax.legend(h, l, loc="upper left", fontsize=SMALL, handlelength=1.6, ncol=1, borderaxespad=0.3)
    panel(ax, "(a)")
    thumbs = [(rmask, "(b) uniform, $p=0.20$"), (masks["target_box_R5_g1"], "target enclosed"),
              (masks["start_box_R8_g1"], "start enclosed")]
    for k, (m, ttl) in enumerate(thumbs):
        a = fig.add_subplot(gs[k, 1])
        draw_lattice(a, m, ttl)
    save(fig, "s7_defects_density_pmf")


# =============================================================================== susceptibility maps
def fig_maps():
    d = pd.read_csv(_os.path.join(_R, 'data', 'msc_defects', 'sensitivity_N35.csv')).iloc[1:]
    A = np.full((N, N), np.nan)
    B = np.full((N, N), np.nan)
    A[d.i, d.j] = N * N * d.chi_mfpt
    B[d.i, d.j] = N * N * d.chi_mode
    fig, axs = plt.subplots(1, 3, figsize=(W, 2.45), gridspec_kw=dict(wspace=0.22, left=0.07, right=0.87,
                                                                        top=0.90, bottom=0.17))
    norm = SymLogNorm(linthresh=1.0, vmin=-500, vmax=500, base=10)
    titles = [r"(a) $N^{2}\chi_{T}(\mathbf{x})$", r"(b) $N^{2}\chi_{t^{*}}(\mathbf{x})$",
              r"(c) $N^{2}[\chi_{t^{*}}-\chi_{T}](\mathbf{x})$"]
    for ax, M, ttl in zip(axs, (A, B, B - A), titles):
        im = ax.imshow(M.T, origin="lower", cmap=DIV, norm=norm, interpolation="nearest")
        ax.plot([0], [0], marker="o", ms=4.5, color="white", mec=INK, mew=0.8)
        ax.plot([N - 1], [N - 1], marker="*", ms=8, color="white", mec=INK, mew=0.8)
        ax.set_xticks([0, 17, 34])
        ax.set_yticks([0, 17, 34])
        ax.grid(False)
        ax.set_xlabel("$x_{1}$")
        panel(ax, ttl)
    axs[0].set_ylabel("$x_{2}$")
    cax = fig.add_axes([0.89, 0.19, 0.016, 0.69])
    cb = fig.colorbar(im, cax=cax, ticks=[-100, -10, -1, 0, 1, 10, 100])
    cb.ax.set_yticklabels(["−100", "−10", "−1", "0", "1", "10", "100"])
    cb.set_label("relative change per unit defect fraction", fontsize=SMALL)
    cb.outline.set_linewidth(0.5)
    save(fig, "s8_defects_placement_maps")


# =============================================================================== placement decides the direction
def fig_structured():
    det = pd.read_csv(_os.path.join(_R, 'data', 'msc_defects', 'structured_deterministic.csv'))
    loc = pd.read_csv(_os.path.join(_R, 'data', 'msc_defects', 'summary_structured_local.csv'))
    fig, axs = plt.subplots(1, 3, figsize=(W, 2.75), gridspec_kw=dict(wspace=0.46, width_ratios=[1.3, 1, 1],
                                                                       left=0.07, right=0.93, top=0.90, bottom=0.17))
    ax = axs[0]

    def group(name):
        if name.startswith(("target_box", "block6_near_target", "wall_neartarget")):
            return "near target", ORANGE, "s"
        if name.startswith(("start_box", "block6_near_start", "wall_nearstart")):
            return "near start", AQUA, "^"
        if name.startswith(("wall_mid", "serpentine")):
            return "walls, serpentines", BLUE, "o"
        return "other", MUTED, "D"

    seen = set()
    x0, x1 = 0.45, 14
    xx = np.array([x0, x1])
    r0 = base_mode / base_mfpt
    for fac in (1, 2, 0.5):
        ax.plot(xx, fac * xx, color=GRID if fac != 1 else MUTED, lw=0.9, ls="-" if fac == 1 else "--", zorder=1)
    ax.text(7.6, 6.4, "ratio unchanged", fontsize=SMALL, color=INK2, rotation=40, ha="right", va="top")
    ax.text(4.4, 10.6, r"$\times2$", fontsize=SMALL, color=INK2, ha="right", va="bottom")
    ax.text(13.0, 4.6, r"$\times\frac{1}{2}$", fontsize=SMALL, color=INK2, ha="right", va="top")
    for _, r in det.iterrows():
        if r["name"] == "clean":
            continue
        g, c, mk = group(r["name"])
        ax.plot(r.mfpt / base_mfpt, r["mode"] / base_mode, marker=mk, color=c, ms=4.6, mec="white", mew=0.6, ls="",
                label=g if g not in seen else None, zorder=4)
        seen.add(g)
    lab = {"target_box_R5_g1": ("target enclosed", (3.1, 1.02), "left", "center"),
           "start_box_R8_g1": ("start\nenclosed", (0.5, 2.25), "left", "center"),
           "serpentine_k6_g3": ("serpentine,\n6 walls", (13.3, 8.6), "right", "center"),
           "wall_neartarget_gap_centre_g3": ("wall near target", (3.1, 1.95), "left", "center"),
           "wall_mid_gap_centre_g1": ("mid wall,\n1-site gap", (0.5, 4.7), "left", "center")}
    for nm, (txt, pos, ha, va) in lab.items():
        r = det[det["name"] == nm].iloc[0]
        ax.annotate(txt, (r.mfpt / base_mfpt, r["mode"] / base_mode), xytext=pos, textcoords="data",
                    fontsize=SMALL, color=INK, ha=ha, va=va,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5, shrinkA=1, shrinkB=3))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(x0, x1)
    ax.set_ylim(0.85, 26)
    ax.set_xticks([1, 2, 4, 8])
    ax.set_xticklabels(["1", "2", "4", "8"])
    ax.set_yticks([1, 2, 4, 8, 16])
    ax.set_yticklabels(["1", "2", "4", "8", "16"])
    ax.minorticks_off()
    ax.set_xlabel(r"$T/T_{\mathrm{clean}}$")
    ax.set_ylabel(r"$t^{*}/t^{*}_{\mathrm{clean}}$")
    ax.legend(loc="upper left", fontsize=SMALL, handletextpad=0.2, borderaxespad=0.1, labelspacing=0.25)
    st = {"anywhere": (BLUE, "o", "anywhere"), "near_target": (ORANGE, "s", "near target"),
          "near_start": (AQUA, "^", "near start"), "centre": (YELLOW, "D", "centre")}
    for ax, key, ylab in ((axs[1], "mfpt_factor", "factor relative to clean lattice"), (axs[2], "ratio", r"$t^{*}/T$")):
        for kind, (c, mk, lab_) in st.items():
            g = loc[loc.kind == kind].sort_values("M")
            if key == "mfpt_factor":
                ax.errorbar(g.M, g.mfpt_factor, yerr=g.mfpt_factor_ci, color=c, marker=mk, ms=3.8, lw=1.1, capsize=1.5,
                            mec="white", mew=0.5, label=lab_)
                ax.errorbar(g.M, g.mode_factor, yerr=g.mode_factor_ci, color=c, marker=mk, ms=3.8, lw=1.1, ls="--",
                            capsize=1.5, mfc="white", mew=0.9)
            else:
                ax.errorbar(g.M, g.ratio, yerr=g.ratio_ci, color=c, marker=mk, ms=3.8, lw=1.1, capsize=1.5,
                            mec="white", mew=0.5)
                ax.text(41.8, g.ratio.iloc[-1], lab_, color=INK, fontsize=SMALL, va="center")
        ax.set_xlabel("blocked sites $M$ in the region")
        ax.set_ylabel(ylab)
        ax.set_xticks([10, 20, 30, 40])
    axs[1].set_yscale("log")
    axs[1].set_yticks([1, 1.5, 2, 3])
    axs[1].set_yticklabels(["1", "1.5", "2", "3"])
    axs[1].minorticks_off()
    h, l = axs[1].get_legend_handles_labels()
    h = [x[0] if isinstance(x, tuple) or hasattr(x, "lines") else x for x in h]
    h += [Line2D([], [], color=INK2, lw=1.1), Line2D([], [], color=INK2, lw=1.1, ls="--")]
    l += ["mean $T$", "mode $t^{*}$"]
    axs[1].legend(h, l, loc="upper left", fontsize=SMALL, handlelength=1.8, borderaxespad=0.1, labelspacing=0.25)
    axs[2].axhline(r0, color=MUTED, lw=0.7)
    axs[2].set_xlim(8, 58)
    for a, l_ in zip(axs, "abc"):
        panel(a, f"({l_})")
    save(fig, "s8_defects_placement_structured")


# =============================================================================== two time scales
def clean_Q():
    idx = -np.ones((N, N), dtype=int)
    k = 0
    for i in range(N):
        for j in range(N):
            if (i, j) != T_:
                idx[i, j] = k
                k += 1
    Qm = np.zeros((k, k))
    r = np.zeros(k)
    for i in range(N):
        for j in range(N):
            a = idx[i, j]
            if a < 0:
                continue
            Qm[a, a] += 1 - q
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                u, v = i + di, j + dj
                if not (0 <= u < N and 0 <= v < N):
                    Qm[a, a] += q / 4                                  # cancelled move
                elif (u, v) == T_:
                    r[a] += q / 4
                else:
                    Qm[a, idx[u, v]] += q / 4
    return Qm, r, idx[S_]


def fig_twotime():
    Qm, r, o = clean_Q()
    lam, V = np.linalg.eigh(Qm)
    w = V[o] * (V.T @ r)
    order = np.argsort(-lam)
    sig = [k for k in order if abs(w[k]) > 1e-12 * abs(w[order[0]])]
    t = np.arange(1, 12001)
    u = np.zeros(Qm.shape[0])
    u[o] = 1.0
    f = np.empty(t.size)
    for i in range(t.size):
        f[i] = u @ r
        u = Qm @ u
    f2 = sum(w[k] * lam[k] ** (t - 1) for k in sig[:2])
    f3 = sum(w[k] * lam[k] ** (t - 1) for k in sig[:3])
    tau = [-1.0 / np.log(lam[k]) for k in sig[:2]]
    m_exact, m2, m3 = int(t[np.argmax(f)]), int(t[np.argmax(f2)]), int(t[np.argmax(f3)])
    fig, ax = plt.subplots(1, 2, figsize=(W, 2.8), gridspec_kw=dict(wspace=0.27, left=0.065, right=0.995,
                                                                     top=0.90, bottom=0.25,
                                                                     width_ratios=[1.0, 1.35]))
    a = ax[0]
    a.plot(t, f * 1e5, color=INK, lw=1.7, label=f"exact PMF (maximum at $t={m_exact}$)")
    k2, k3 = (f2 > -2e-6) & (f2 < 9e-5), (f3 > -2e-6) & (f3 < 9e-5)      # truncations are far off scale at small t
    k2 &= t >= t[~k2].max(initial=0)
    k3 &= t >= t[~k3].max(initial=0)
    a.plot(t[k2], f2[k2] * 1e5, color=BLUE, lw=1.2, ls="--", label=f"two slowest terms ({m2})")
    a.plot(t[k3], f3[k3] * 1e5, color=ORANGE, lw=1.2, ls=":", label=f"three slowest terms ({m3})")
    a.set_ylim(0, 7.5)
    a.set_xlim(0, 12000)
    a.set_xlabel("time $t$ (steps)")
    a.set_ylabel(r"$f(t)\times10^{5}$")
    panel(a, "(a) clean lattice")
    a.legend(fontsize=SMALL, loc="lower right", borderaxespad=0.2)
    T = pd.read_csv(_os.path.join(_R, 'data', 'msc_defects_verify', 'v09b_twotime.csv'))
    a = ax[1]
    x = np.arange(len(T))
    a.plot(x, T.elasticity_exact, "o", color=INK, ms=5.5, label="exact")
    a.plot(x, T.elast_tau540, "^", color=ORANGE, ms=5, label=r"heuristic, $\tau=540$, $c=4.15$")
    a.plot(x, T.elast_tau621, "s", color=BLUE, ms=4.5, label=r"heuristic, $\tau=621$ (for comparison)")
    a.axhline(0.25, color=MUTED, lw=0.9, ls=":")
    a.text(len(T) - 0.55, 0.262, "1/4", fontsize=SMALL, color=MUTED, ha="right")
    a.set_xticks(x)
    a.set_xticklabels(["one\ntarget\nneighbour", "enclosure\n$R=3$", "enclosure\n$R=5$", "enclosure\n$R=8$",
                       "$R=5$,\n3-site\ndoor", "wall\nnear\ntarget"], fontsize=SMALL)
    a.set_ylim(0.13, 0.74)
    a.set_xlim(-0.5, len(T) - 0.5)
    a.set_ylabel(r"$\ln(t^{*}\ \mathrm{factor})\,/\,\ln(T\ \mathrm{factor})$")
    panel(a, "(b) obstacles next to the target")
    a.legend(fontsize=SMALL, loc="upper left", borderaxespad=0.2)
    save(fig, "s8_defects_placement_twotime")
    return {"tau_two_slowest_terms": tau, "max_exact": m_exact, "max_two_terms": m2, "max_three_terms": m3}


# =============================================================================== family of shapes
def fig_shape():
    sw = pd.concat([pd.read_csv(f_) for f_ in sorted(glob.glob(_os.path.join(_R, 'data', 'msc_defects', 'sweep', '*.csv')))])
    det = pd.read_csv(_os.path.join(_R, 'data', 'msc_defects', 'structured_deterministic.csv'))
    loc = pd.read_csv(_os.path.join(_R, 'data', 'msc_defects', 'structured_local.csv'))
    cl = pd.read_csv(_os.path.join(_R, 'data', 'msc_defects', 'size_clean.csv'))
    d1 = pd.read_csv(_os.path.join(_R, 'data', 'msc_defects', 'beyond_1d_random.csv'))
    chk = json.load(open(_os.path.join(_R, 'data', 'article', 's8_defects_placement_checks.json')))["shape_family"]["sample_A"]
    coef = chk["quad_c0_c1_c2"]
    n_all = len(sw) + len(det) + len(loc)
    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    lo = sw[sw.p <= 0.2001]
    hi = sw[sw.p > 0.2001]
    # drawing order: the widely scattered p >= 0.25 placements first, then p <= 0.2 and the localised placements
    # on top (each of the three groups stays visible); the legend lists them in the order of the caption
    h_hi, = ax.plot(hi.ratio, hi.cv, ls="", marker=".", ms=1.8, color="#7a5cc9", alpha=0.35, rasterized=True,
                    label=r"random placements, $p\geq0.25$", zorder=2)
    h_loc, = ax.plot(loc.ratio, loc.cv, ls="", marker=".", ms=1.8, color=AQUA, alpha=0.45, rasterized=True,
                     label="localised placements", zorder=2.2)
    h_lo, = ax.plot(lo.ratio, lo.cv, ls="", marker=".", ms=1.8, color=BLUE, alpha=0.35, rasterized=True,
                    label=r"random placements, $p\leq0.2$", zorder=2.4)
    ax.plot(det.ratio, det.cv, ls="", marker="s", ms=3.2, color=ORANGE, mec="white", mew=0.4,
            label="deterministic geometries", zorder=3)
    ax.plot(cl["mode"] / cl.mfpt, cl.sd / cl.mfpt, color=INK, lw=1.0, marker="D", ms=2.6, mec="white", mew=0.3,
            label="clean lattices, $N=10$–$200$", zorder=3.2)
    ax.plot([d1.ratio.iloc[0]], [d1.cv.iloc[0]], marker="*", ms=8, color=YELLOW, mec=INK, mew=0.5, ls="",
            label="clean chain, $N=100$", zorder=4)
    if coef is not None:
        xr = np.linspace(0.045, 0.47, 200)
        ax.plot(xr, coef[0] + coef[1] * xr + coef[2] * xr ** 2, color=MUTED, lw=0.9, ls=(0, (4, 2)),
                label="fitted quadratic", zorder=1)
    xmin, xmax = min(sw.ratio.min(), det.ratio.min(), loc.ratio.min()), max(sw.ratio.max(), det.ratio.max(), loc.ratio.max())
    ymin, ymax = min(sw.cv.min(), det.cv.min(), loc.cv.min()), max(sw.cv.max(), det.cv.max(), loc.cv.max())
    ax.set_xlim(0.03, 0.49)
    ax.set_ylim(0.71, 1.14)
    assert 0.03 < xmin and xmax < 0.49 and 0.71 < ymin and ymax < 1.14, "a configuration lies outside the frame"
    ax.set_xlabel(r"$t^{*}/T$")
    ax.set_ylabel("coefficient of variation")
    hh, ll = ax.get_legend_handles_labels()
    first = [h_lo, h_hi, h_loc]
    hh2 = first + [h for h in hh if h not in first]
    ll2 = [h.get_label() for h in hh2]
    dots = {BLUE, "#7a5cc9", AQUA}
    hh2 = [Line2D([], [], ls="", marker="o", ms=3.2, color=h.get_color()) if h.get_color() in dots else h
           for h in hh2]
    ax.legend(hh2, ll2, loc="upper right", fontsize=SMALL, handletextpad=0.4, labelspacing=0.3, borderaxespad=0.2)
    fig.subplots_adjust(left=0.135, right=0.985, top=0.975, bottom=0.135)
    save(fig, "s8_defects_placement_shape")
    return {"n_configurations_plotted": int(n_all), "ratio_range": [float(xmin), float(xmax)],
            "cv_range": [float(ymin), float(ymax)], "frame": [0.03, 0.49, 0.71, 1.14],
            "quadratic_drawn": coef}


if __name__ == "__main__":
    fig_pmf()
    fig_maps()
    fig_structured()
    info = {"generated_by": "code/article/s7s8_defects_figures.py", "two_time": fig_twotime(),
            "shape_family": fig_shape()}
    json.dump(info, open(_os.path.join(_R, 'data', 'article', 's7s8_defects_figures.json'), "w"), indent=1)
    print(json.dumps(info))
