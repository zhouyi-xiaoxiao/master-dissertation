#!/usr/bin/env python
"""Section 5 (s5_modes): Figures s5_modes:fig-mode-mean, fig-ratio, fig-exponents.

Adapted from code/msc_modes/05_make_figures.py (Figs 1, 2 and 4 there): same
data, same quantities; the labels use the wording of the article and the sizes are chosen
for the text width of the article (16 cm).
No computation: everything plotted is read from the saved result files.

Inputs  (relative to ):
    data/msc_modes/discrete_modes.jsonl   exact discrete-time modes (time stepping)
    data/msc_modes/laplace_modes.jsonl    exact continuous-time modes, MFPTs, closed form
    data/msc_modes/fit_results.json       local exponents (effective_exponents)
Outputs: figures/s5_modes_mode_mean.pdf, s5_modes_ratio.pdf, s5_modes_exponents.pdf
Usage:   python s5_modes_figures.py
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = _R            # 
DATA = _os.path.join(_R, 'data', 'msc_modes')
FIG = _os.path.join(_R, 'figures')
os.makedirs(FIG, exist_ok=True)

# palette and style of the research figures (kept so that all figures of the article match)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, FAINT = "#0b0b0b", "#52514e", "#b8b7b0"
DC = {1: BLUE, 2: ORANGE, 3: AQUA}
DM = {1: "o", 2: "s", 3: "^"}
plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8.5, "axes.titlesize": 8.5, "legend.fontsize": 7.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "lines.linewidth": 1.4, "lines.markersize": 4,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "axes.grid": True, "grid.color": "#e6e5e0",
    "grid.linewidth": 0.5, "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
    "mathtext.fontset": "dejavusans",
})

D = [json.loads(l) for l in open(_os.path.join(_R, 'data', 'msc_modes', 'discrete_modes.jsonl')) if l.strip()]
L = [json.loads(l) for l in open(_os.path.join(_R, 'data', 'msc_modes', 'laplace_modes.jsonl')) if l.strip()]
FITS = json.load(open(_os.path.join(_R, 'data', 'msc_modes', 'fit_results.json')))
Q = 0.8


def sel(rows, **kw):
    return sorted([r for r in rows if all(r.get(k) == v for k, v in kw.items())], key=lambda r: r["N"])


def arr(rows, key):
    return np.array([r[key] for r in rows], float)


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("saved", name + ".pdf")


NMAX = {1: 1.2e4, 2: 1.2e5, 3: 4500}

# ------------------------------------------------------------------ mode and mean against N
fig, axs = plt.subplots(1, 3, figsize=(6.6, 2.55))
for ax, d in zip(axs, (1, 2, 3)):
    c = [r for r in sel(L, d=d, geo="CC") if r["N"] <= NMAX[d]]
    N = arr(c, "N")
    ax.plot(N, arr(c, "mfpt") / Q, color=BLUE)                       # unit-rate values / q = steps
    ax.plot(N, arr(c, "mode") / Q, color=ORANGE)
    dd = sel(D, d=d, geo="CC", q=Q)
    ax.plot(arr(dd, "N"), arr(dd, "mode"), ls="none", marker="o", mfc="white", mec=ORANGE, mew=0.9, ms=4, zorder=3)
    if d > 1:
        ax.plot(N, arr(c, "pred_L1") / Q, color=INK, ls=(0, (4, 2)), lw=0.9)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("lattice size $N$")
    ax.set_title({1: "(a) $d=1$", 2: "(b) $d=2$", 3: "(c) $d=3$"}[d], loc="left")
axs[0].set_ylabel("time (steps, $q=0.8$)")
handles = [
    Line2D([], [], color=BLUE, label="mean $T$, exact"),
    Line2D([], [], color=ORANGE, label="mode $t^{*}$, exact (continuous time)"),
    Line2D([], [], ls="none", marker="o", mfc="white", mec=ORANGE, mew=0.9, ms=4,
           label="mode $t^{*}$, maximiser of the exact PMF (discrete time)"),
    Line2D([], [], color=INK, ls=(0, (4, 2)), lw=0.9, label="mode, closed-form approximation ($d=2,3$)"),
]
fig.tight_layout(w_pad=0.8)
fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=2, handlelength=2.0,
           columnspacing=1.5)
save(fig, "s5_modes_mode_mean")

# ------------------------------------------------------------------ mode / MFPT against N
fig, ax = plt.subplots(figsize=(6.3, 3.3))
for d in (1, 2, 3):
    c = sel(L, d=d, geo="CC")
    N = arr(c, "N")
    ax.plot(N, arr(c, "mode") / arr(c, "mfpt"), color=DC[d], label=f"$d={d}$, continuous time")
    dd = sel(D, d=d, geo="CC", q=Q)
    ax.plot(arr(dd, "N"), arr(dd, "mode_over_mfpt"), ls="none", marker=DM[d], mfc="white", mec=DC[d], mew=0.8,
            ms=3.6, label=f"$d={d}$, discrete time, $q=0.8$")
    if d > 1:
        ax.plot(N, arr(c, "pred_L1") / arr(c, "mfpt"), color=INK, ls=(0, (4, 2)), lw=0.9,
                label="closed-form approximation" if d == 2 else None)
ax.text(1.5e3, 0.37, "$d=1$: 0.33328", color=BLUE, fontsize=7.5, va="bottom")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(4, 2.5e7)
ax.set_ylim(3e-4, 1.6)
ax.set_xlabel("lattice size $N$")
ax.set_ylabel(r"$t^{*}/T$")
ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.0), handlelength=2.0, borderaxespad=0.3)
save(fig, "s5_modes_ratio")

# ------------------------------------------------------------------ local exponents
fig, axs = plt.subplots(1, 2, figsize=(6.3, 2.6), sharex=True)
E = FITS["effective_exponents"]
for d in (1, 2, 3):
    e = E[str(d)]
    axs[0].plot(e["N_geo_mean"], e["a_eff_mode"], color=DC[d], marker=DM[d], ms=2.6, lw=1.1, label=f"$d={d}$")
    axs[1].plot(e["N_geo_mean"], e["a_eff_mfpt"], color=DC[d], marker=DM[d], ms=2.6, lw=1.1, label=f"$d={d}$")
for ax in axs:
    ax.axhline(2, color=MUTED, lw=0.8, ls=":")
    ax.axhline(3, color=MUTED, lw=0.8, ls=":")
for ax, t in zip(axs, ("(a) mode $t^{*}$", "(b) mean $T$")):
    ax.set_xscale("log")
    ax.set_xlabel("lattice size $N$")
    ax.set_title(t, loc="left")
    ax.legend(loc="center right")
axs[0].set_ylabel(r"local exponent $\mathrm{d}\ln(\cdot)/\mathrm{d}\ln N$")
axs[0].set_ylim(1.95, 3.05)
axs[1].set_ylim(1.95, 3.25)
fig.tight_layout()
save(fig, "s5_modes_exponents")
print("done")
