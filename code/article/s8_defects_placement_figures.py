#!/usr/bin/env python
"""Two figures of Supplementary Section S9, drawn with the notation of the article.

Both figures are redrawn from the saved research results; no first-passage problem is solved
here.  The plotting code follows code/msc_defects/09_figures.py (fig5, fig7) with the
labels changed to the notation of the article:

  figures/s8_defects_placement_mechanism.pdf   (fig5_mechanism there: legend "corner-thinned";
        G_sa -> G_oa; sigma -> Sigma)
  figures/s8_defects_placement_beyond.pdf      (fig7_beyond_inert there: panel (c) titled "partial traps";
        panel (a) abscissa is the zero-based bond index k)

Inputs : data/msc_defects/{sweep/sweep_N35_uniform_*.csv, summary_sweep_N35.csv,
         homogenization.json, beyond_1d_single.csv, summary_beyond.json}
Usage  : python s8_defects_placement_figures.py
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import glob
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = _R            # 
DATA = _os.path.join(_R, 'data', 'msc_defects')
FIG = _os.path.join(_R, 'figures')

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8985", "#e4e3df"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 8.5, "axes.titlesize": 9,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.7,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.grid": True, "axes.axisbelow": True,
    "lines.linewidth": 1.5, "lines.markersize": 4.5, "legend.frameon": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight", "savefig.dpi": 300,
    "figure.facecolor": "white", "axes.facecolor": "white",
})
N, q = 35, 0.8
base_mfpt, base_mode = 14100.808049937992, 2493


def panel(ax, letter):
    ax.text(-0.02, 1.04, f"({letter})", transform=ax.transAxes, fontsize=8.5,
            ha="right", va="bottom", color=INK)


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    plt.close(fig)
    print("wrote", name + ".pdf")


def mechanism():
    summ = pd.read_csv(_os.path.join(_R, 'data', 'msc_defects', 'summary_sweep_N35.csv'))
    hom = json.load(open(_os.path.join(_R, 'data', 'msc_defects', 'homogenization.json')))
    sw = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(_os.path.join(_R, 'data', 'msc_defects', 'sweep', 'sweep_N35_uniform_*.csv')))])
    g = sw[np.isclose(sw.p, 0.20)]
    fig, axs = plt.subplots(1, 3, figsize=(6.5, 2.6), gridspec_kw=dict(wspace=0.46))
    ax = axs[0]
    for dt, c, mk, lab in ((2, BLUE, "o", "both neighbours of $\\mathbf{a}$ open"), (1, ORANGE, "s", "one neighbour of $\\mathbf{a}$ blocked")):
        h = g[g.deg_t == dt]
        ax.plot(4 * h.n_cluster / q * h.G_tt / 1e4, h.mfpt / 1e4, ls="", marker=mk, ms=2.8, color=c, alpha=0.75, mec="none", label=lab)
    lim = [1.0, 7.5]
    ax.plot(lim, lim, color=MUTED, lw=0.8)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel(r"$(4n/q)\,G_{aa}$  ($\times10^4$ steps)"); ax.set_ylabel(r"$T$  ($\times10^4$ steps)")
    ax.legend(loc="upper left", fontsize=7.0, handletextpad=0.1, borderaxespad=0.1)
    ax = axs[1]
    for dt, c, mk in ((2, BLUE, "o"), (1, ORANGE, "s")):
        h = g[g.deg_t == dt]
        ax.plot(h.mfpt / 1e4, h["mode"] / 1e3, ls="", marker=mk, ms=2.8, color=c, alpha=0.75, mec="none")
    b = np.polyfit(np.log(g.mfpt), np.log(g["mode"]), 1)
    xx = np.linspace(g.mfpt.min(), g.mfpt.max(), 50)
    ax.plot(xx / 1e4, np.exp(np.polyval(b, np.log(xx))) / 1e3, color=INK, lw=1.0)
    xm, ym = np.exp(np.log(g.mfpt).mean()), np.exp(np.log(g["mode"]).mean())
    ax.plot(xx / 1e4, ym * (xx / xm) / 1e3, color=MUTED, lw=0.9, ls="--")
    ax.text(0.97, 0.06, f"fitted slope {b[0]:.2f}", transform=ax.transAxes, fontsize=7.0, ha="right", color=INK)
    ax.text(0.03, 0.97, "dashed: slope 1\n(pure time dilation)", transform=ax.transAxes, fontsize=7.0, ha="left", va="top", color=INK2)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks([2, 3, 4, 6]); ax.set_xticklabels(["2", "3", "4", "6"]); ax.set_yticks([3, 4, 5, 6, 8]); ax.set_yticklabels(["3", "4", "5", "6", "8"])
    ax.minorticks_off()
    ax.set_xlabel(r"$T$  ($\times10^4$ steps)"); ax.set_ylabel(r"$t^{*}$  ($\times10^3$ steps)")
    ax = axs[2]
    u = summ[summ.scheme == "uniform"]; s = summ[summ.scheme == "smart"]
    hpv = np.array([r["p"] for r in hom["finite_p"]]); hs = np.array([r["sigma"] for r in hom["finite_p"]])
    ax.plot(hpv[hpv <= 0.3], 1 / hs[hpv <= 0.3], color=INK, lw=1.1, label=r"$\Sigma_0/\Sigma(p)$ (tori)")
    ax.plot(u.p[u.p <= 0.3], u.Gdiff_factor[u.p <= 0.3], color=BLUE, marker="o", ms=3.6, lw=0.9, mec="white", mew=0.5, label="uniform")
    ax.plot(s.p[s.p <= 0.3], s.Gdiff_factor[s.p <= 0.3], color=ORANGE, marker="s", ms=3.6, lw=0.9, mec="white", mew=0.5, label="corner-thinned")
    ax.set_yscale("log"); ax.set_yticks([1, 2, 3, 5]); ax.set_yticklabels(["1", "2", "3", "5"]); ax.minorticks_off()
    ax.set_xlabel(r"defect fraction $p$"); ax.set_ylabel(r"$[G_{aa}-G_{oa}]$ relative to clean lattice")
    ax.legend(loc="upper left", fontsize=7.0, borderaxespad=0.2, handlelength=1.6)
    for a, l_ in zip(axs, "abc"):
        panel(a, l_)
    save(fig, "s8_defects_placement_mechanism")
    return float(b[0])


def beyond():
    s1 = pd.read_csv(_os.path.join(_R, 'data', 'msc_defects', 'beyond_1d_single.csv'))
    bj = json.load(open(_os.path.join(_R, 'data', 'msc_defects', 'summary_beyond.json')))
    fig, axs = plt.subplots(1, 3, figsize=(6.5, 2.6), gridspec_kw=dict(wspace=0.52))
    ax = axs[0]
    m0, mode0 = 12375.0, 4124
    ax.plot(s1.pos, s1.mfpt / m0 - 1, color=BLUE, lw=1.4, label="mean $T$")
    ax.plot(s1.pos, s1["mode"] / mode0 - 1, color=ORANGE, lw=1.2, ls="--", label="mode $t^{*}$")
    ax.set_xlabel(r"barrier on bond $k$"); ax.set_ylabel("relative shift")
    ax.legend(loc="upper left", fontsize=7.0, borderaxespad=0.2); ax.set_title("(a) chain, one barrier", loc="left", fontsize=8.5)
    ax = axs[1]
    pm_ = pd.DataFrame(bj["2d_permeable"])
    xk = pm_.kappa.replace(0.0, 1e-4)
    ax.errorbar(xk, pm_.mfpt / base_mfpt, yerr=pm_.mfpt_ci / base_mfpt, color=BLUE, marker="o", ms=4, lw=1.0, capsize=1.5, mec="white", mew=0.5, label=r"$T/T_{\rm clean}$")
    ax.errorbar(xk, pm_["mode"] / base_mode, yerr=pm_.mode_ci / base_mode, color=ORANGE, marker="s", ms=4, lw=1.0, ls="--", capsize=1.5, mec="white", mew=0.5, label=r"$t^{*}/t^{*}_{\rm clean}$")
    ax.set_xscale("log"); ax.set_xticks([1e-4, 1e-3, 1e-2, 1e-1, 1]); ax.set_xticklabels(["0\n(inert)", "$10^{-3}$", "$10^{-2}$", "0.1", "1"]); ax.minorticks_off()
    ax.set_xlabel(r"obstacle permeability $\kappa$"); ax.set_ylabel("factor"); ax.set_ylim(0.97, 1.66); ax.legend(loc="upper right", fontsize=7.0)
    ax.set_title("(b) permeable obstacles", loc="left", fontsize=8.5)
    ax = axs[2]
    tr = pd.DataFrame(bj["2d_traps"])
    ax.plot(tr.Delta, tr.H, color=BLUE, marker="o", ms=4, lw=1.0, mec="white", mew=0.5, label=r"$P_{\rm hit}$")
    ax.plot(tr.Delta, tr.cond_mean / base_mfpt, color=ORANGE, marker="s", ms=4, lw=1.0, ls="--", mec="white", mew=0.5, label=r"cond. mean$\,/\,T_{\rm clean}$")
    ax.plot(tr.Delta, tr["ratio"], color=AQUA, marker="^", ms=4, lw=1.0, ls="-.", mec="white", mew=0.5, label=r"$t^{*}/\,$cond. mean")
    ax.plot(tr.Delta, tr.H_meanfield, color=MUTED, lw=0.8, ls=":", label=r"$P_{\rm hit}$, uniform killing")
    ax.set_xscale("log"); ax.set_ylim(0, 1.82); ax.set_xlabel(r"killing probability $\Delta$ per step")
    ax.set_ylabel("dimensionless value"); ax.legend(loc="upper right", fontsize=7.0, ncol=1, borderaxespad=0.1, handlelength=1.6)
    ax.set_title("(c) partial traps", loc="left", fontsize=8.5)
    save(fig, "s8_defects_placement_beyond")


if __name__ == "__main__":
    slope = mechanism()
    beyond()
    print("cross-placement slope (sample A, uniform, p = 0.20):", round(slope, 4))
