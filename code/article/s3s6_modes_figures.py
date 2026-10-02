#!/usr/bin/env python
"""Figures of Sections 3, 5 and 6 drawn at the text width of the article (16 cm) and in its notation.

    figures/s3_methods_parity.pdf        parity at q = 1; discrete against continuous time      (Suppl. S2.2)
    figures/s5_modes_compensated.pdf     compensated plots of mean and mode                     (Suppl. S6.2)
    figures/s6_mechanism_accuracy.pdf    accuracy of the predictions of the mode                (Suppl. S7.2)
    figures/s6_mechanism_density.pdf     the two regimes of the first-passage density           (Sec. 6.4)
    figures/s6_mechanism_geometries.pdf  other placements of start and target                   (Suppl. S7.4)

Same data and quantities as code/msc_modes/05_make_figures.py (its Figs 7, 3, 6, 5, 8); the figures
are redrawn here so that nothing is rescaled in the article (no text below 7 pt) and every label uses the symbols
of the article (t*, T, t*_c, mu_1, X, A, W) instead of working names.  No first-passage problem is solved: all
values are read from the stored result files, except the continuous-time density of the chain in the parity
figure, which is the closed form of Proposition s5_modes:prop-1d.

Inputs : data/msc_modes/{discrete_modes.jsonl, laplace_modes.jsonl, fit_results.json, profiles.npz,
         profile_stats.json, pmf/pmf_d1_CC_N51_q1.0.npz, pmf/pmf_d2_CC_N160_q0.8.npz, pmf/pmf_d3_CC_N40_q0.8.npz}
Usage  : python s3s6_modes_figures.py          (a few seconds)
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import json
import math
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = _R            # 
DATA = _os.path.join(_R, 'data', 'msc_modes')
FIG = _os.path.join(_R, 'figures')
os.makedirs(FIG, exist_ok=True)

BLUE, ORANGE, AQUA, YELLOW, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7"
INK, MUTED, FAINT = "#0b0b0b", "#52514e", "#b8b7b0"
DC = {1: BLUE, 2: ORANGE, 3: AQUA}
DM = {1: "o", 2: "s", 3: "^"}
plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8.5, "axes.titlesize": 8.5, "legend.fontsize": 7.2,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "lines.linewidth": 1.4, "lines.markersize": 4,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "axes.grid": True, "grid.color": "#e6e5e0",
    "grid.linewidth": 0.5, "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
    "mathtext.fontset": "dejavusans",
})
W = 6.3                                                                 # inches = text width of the article
SMALL = 7.0                                                             # smallest font used

D = [json.loads(l) for l in open(_os.path.join(_R, 'data', 'msc_modes', 'discrete_modes.jsonl')) if l.strip()]
L = [json.loads(l) for l in open(_os.path.join(_R, 'data', 'msc_modes', 'laplace_modes.jsonl')) if l.strip()]
FITS = json.load(open(_os.path.join(_R, 'data', 'msc_modes', 'fit_results.json')))
Q = 0.8
C2 = 8 / math.pi * (0.5772156649015329 + math.log(4 * math.sqrt(2) / 2.6220575542921198) - 0.5 - math.pi / 4)


def sel(rows, **kw):
    return sorted([r for r in rows if all(r.get(k) == v for k, v in kw.items())], key=lambda r: r["N"])


def arr(rows, key):
    return np.array([r[key] for r in rows], float)


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    plt.close(fig)
    print("saved", name + ".pdf")


# ------------------------------------------------------------------ parity; discrete against continuous time
def chain_density(N, t):
    """unit-rate first-passage density of the chain, reflecting end to absorbing end (Proposition prop-1d)"""
    m = np.arange(1, N)
    om = (2 * m - 1) * np.pi / (2 * N - 1)
    b = 2.0 / (2 * N - 1) * (-1.0) ** (m + 1) * np.cos(om / 2) * np.sin(om)
    return (b[None, :] * np.exp(-(1 - np.cos(om))[None, :] * t[:, None])).sum(axis=1)


fig, axs = plt.subplots(1, 2, figsize=(W, 2.5))
ax = axs[0]
z = np.load(_os.path.join(_R, 'data', 'msc_modes', 'pmf', 'pmf_d1_CC_N51_q1.0.npz'))
t, f = z["t"], z["f"]
w = (t >= 700) & (t <= 1000)
ev = w & (t % 2 == 0)
od = w & (t % 2 == 1)
ax.plot(t[ev], f[ev] * 1e4, ls="none", marker="o", ms=2.0, color=BLUE, label="$f(t)$, even $t$")
ax.plot(t[od], f[od] * 1e4, ls="none", marker="s", ms=2.0, color=ORANGE, label="$f(t)$, odd $t$")
fb = 0.5 * (f[:-1] + f[1:])
tb = t[:-1] + 0.5
wb = (tb >= 700) & (tb <= 1000)
ax.plot(tb[wb], fb[wb] * 1e4, color=AQUA, lw=1.3, label="two-step average")
tc = np.linspace(700, 1000, 200)
ax.plot(tc, chain_density(51, tc) * 1e4, color=INK, lw=0.9, ls=(0, (4, 2)), label="continuous time, $g(t)$")
ax.axvline(float(t[np.argmax(f)]), color=BLUE, lw=0.8, ls=":")
ax.axvline(float(tb[np.argmax(fb)]), color=AQUA, lw=0.8, ls=":")
ax.set_ylim(3.05, 3.78)
ax.set_xlabel("time $t$ (steps)")
ax.set_ylabel(r"probability per step ($\times10^{-4}$)")
ax.set_title("(a) $d=1$, $N=51$, $q=1$", loc="left")
ax.legend(loc="lower center", ncol=2, columnspacing=1.0, handletextpad=0.4)
ax = axs[1]
off = FITS["discrete_vs_continuous_offsets"]
for d in (1, 2, 3):
    o = sorted([x for x in off if x["d"] == d and x["geo"] == "CC"], key=lambda x: x["N"])
    ax.plot([x["N"] for x in o], [x["offset_refined"] for x in o], color=DC[d], marker=DM[d], ms=3.5, lw=1.0,
            label=f"$d={d}$")
ax.set_xscale("log")
ax.set_xlabel("lattice size $N$")
ax.set_ylabel(r"$t^{*}-t^{*}_{\mathrm{c}}/q$  (steps), $q=0.8$")
ax.set_title("(b) discrete against continuous time", loc="left")
ax.legend()
fig.tight_layout()
save(fig, "s3_methods_parity")

# ------------------------------------------------------------------ compensated plots
fig, axs = plt.subplots(2, 2, figsize=(W, 4.2))
c2 = sel(L, d=2, geo="CC")
N2, M2, T2 = arr(c2, "N"), arr(c2, "mode"), arr(c2, "mfpt")
c3 = sel(L, d=3, geo="CC")
N3, M3, T3 = arr(c3, "N"), arr(c3, "mode"), arr(c3, "mfpt")
ax = axs[0, 0]
ax.plot(np.log(N2), T2 / N2 ** 2, ls="none", marker="o", color=BLUE, ms=3.2, label="exact")
xx = np.linspace(np.log(5), np.log(N2.max()), 50)
ax.plot(xx, 8 / np.pi * xx + C2, color=INK, lw=0.9, ls=(0, (4, 2)), label=r"$(8/\pi)\ln N+C_{2}$")
ax.set_xlabel(r"$\ln N$")
ax.set_ylabel(r"$q\,T/N^{2}$")
ax.set_title("(a) $d=2$, mean", loc="left")
ax.legend()
ax = axs[0, 1]
ax.plot(np.log(np.log(N2)), M2 / N2 ** 2, ls="none", marker="o", color=ORANGE, ms=3.2,
        label=r"exact, continuous time ($t^{*}_{\mathrm{c}}$)")
ax.plot(np.log(np.log(N2)), arr(c2, "pred_L1") / N2 ** 2, color=INK, lw=0.9, ls=(0, (4, 2)),
        label="closed-form approximation")
d2 = sel(D, d=2, geo="CC", q=Q)
ax.plot(np.log(np.log(arr(d2, "N"))), Q * arr(d2, "mode") / arr(d2, "N") ** 2, ls="none", marker="s", mfc="white",
        mec=ORANGE, mew=0.8, ms=3.2, label="exact discrete PMF, $q=0.8$")
z0 = np.log(np.log(N2[-1]))
y0 = M2[-1] / N2[-1] ** 2
zz = np.array([z0 - 0.9, z0])
ax.plot(zz, y0 + 4 / np.pi ** 2 * (zz - z0) - 0.12, color=MUTED, lw=0.9)
ax.text(zz.mean(), y0 + 4 / np.pi ** 2 * (zz.mean() - z0) - 0.24, r"slope $4/\pi^{2}$", color=MUTED,
        fontsize=SMALL, ha="left")
ax.set_xlabel(r"$\ln\ln N$")
ax.set_ylabel(r"$q\,t^{*}/N^{2}$")
ax.set_title("(b) $d=2$, mode", loc="left")
ax.set_ylim(top=2.78)
ax.legend(loc="upper left")
ax = axs[1, 0]
ax.plot(1 / N3, T3 / N3 ** 3, ls="none", marker="^", color=BLUE, ms=3.2, label="exact")
K3 = FITS["asymptotics"]["3d_C3_limit_richardson"]
K3p = FITS["asymptotics"]["3d_C3_1overN_coeff"]
xx = np.linspace(0, 0.2, 50)
ax.plot(xx, K3 + K3p * xx, color=INK, lw=0.9, ls=(0, (4, 2)), label=r"$K_{3}+K_{3}'/N$")
ax.set_xlabel(r"$1/N$")
ax.set_ylabel(r"$q\,T/N^{3}$")
ax.set_title("(c) $d=3$, mean", loc="left")
ax.legend()
ax = axs[1, 1]
ax.plot(np.log(N3), M3 / N3 ** 2, ls="none", marker="^", color=ORANGE, ms=3.2,
        label=r"exact, continuous time ($t^{*}_{\mathrm{c}}$)")
ax.plot(np.log(N3), arr(c3, "pred_L1") / N3 ** 2, color=INK, lw=0.9, ls=(0, (4, 2)), label="closed-form approximation")
d3 = sel(D, d=3, geo="CC", q=Q)
ax.plot(np.log(arr(d3, "N")), Q * arr(d3, "mode") / arr(d3, "N") ** 2, ls="none", marker="s", mfc="white", mec=ORANGE,
        mew=0.8, ms=3.2, label="exact discrete PMF, $q=0.8$")
x0 = np.log(N3[-1])
y0 = M3[-1] / N3[-1] ** 2
xx = np.array([x0 - 2.5, x0])
ax.plot(xx, y0 + 6 / np.pi ** 2 * (xx - x0) - 0.45, color=MUTED, lw=0.9)
ax.text(xx.mean(), y0 + 6 / np.pi ** 2 * (xx.mean() - x0) - 1.05, r"slope $6/\pi^{2}$", color=MUTED, fontsize=SMALL)
ax.set_xlabel(r"$\ln N$")
ax.set_ylabel(r"$q\,t^{*}/N^{2}$")
ax.set_title("(d) $d=3$, mode", loc="left")
ax.set_ylim(top=8.9)
ax.legend(loc="upper left")
fig.tight_layout()
save(fig, "s5_modes_compensated")

# ------------------------------------------------------------------ accuracy of the predictions
fig, axs = plt.subplots(1, 2, figsize=(W, 2.9), sharey=True)
series = [("pred_L0", r"leading order, $\ln(AX)/\mu_{1}$", BLUE, "o"),
          ("pred_L1", r"closed form, $qT\,\ln(AX)/(X+W-1)$", ORANGE, "s"),
          ("pred_2pole", "exact two-exponential truncation", AQUA, "^"),
          ("pred_3pole", "exact three-exponential truncation", YELLOW, "D")]
for ax, d in zip(axs, (2, 3)):
    c = sel(L, d=d, geo="CC")
    N = arr(c, "N")
    M = arr(c, "mode")
    for key, lab, col_, mk in series:
        ax.plot(N, np.abs(arr(c, key) / M - 1), color=col_, marker=mk, ms=2.8, lw=1.0, label=lab)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("lattice size $N$")
    ax.set_title(f"({'ab'[d - 2]}) $d={d}$", loc="left")
axs[0].set_ylabel(r"$|$predicted $t^{*}_{\mathrm{c}}$ / exact $t^{*}_{\mathrm{c}}-1|$")
axs[0].set_ylim(3e-12, 3)
axs[0].legend(loc="lower left", handlelength=1.6, borderaxespad=0.2)
fig.tight_layout()
save(fig, "s6_mechanism_accuracy")

# ------------------------------------------------------------------ the two regimes of the density
P = np.load(_os.path.join(_R, 'data', 'msc_modes', 'profiles.npz'))
PS = json.load(open(_os.path.join(_R, 'data', 'msc_modes', 'profile_stats.json')))


def theta4_pow(x, d):
    k = np.arange(1, 400)[:, None]
    th = 1 + 2 * ((-1.0) ** k * np.exp(-k ** 2 * x[None, :])).sum(axis=0)
    return th ** d


def nlabel(N):
    return f"$N={N}$" if N < 10000 else r"$N=%s$" % f"{N:,}".replace(",", r"\,")


fig, axs = plt.subplots(2, 2, figsize=(W, 4.4))
for col, d in enumerate((2, 3)):
    Ns = sorted(int(k.split("_")[1][1:]) for k in P.files if k.startswith(f"d{d}_") and k.endswith("_x"))
    cmap = plt.get_cmap("Blues" if d == 2 else "Greens")
    for i, N in enumerate(Ns):
        x = P[f"d{d}_N{N}_x"]
        y = P[f"d{d}_N{N}_g_tau"]
        col_ = cmap(0.35 + 0.6 * i / max(1, len(Ns) - 1))
        axs[0, col].plot(x, y, color=col_, lw=1.1)                       # colours as in the lower panels
        axs[1, col].plot(P[f"d{d}_N{N}_y"], P[f"d{d}_N{N}_g_tau_y"], color=col_, lw=1.1, label=nlabel(N))
    xx = np.linspace(0.05, 12, 400)
    axs[0, col].plot(xx, theta4_pow(xx, d), color=INK, lw=1.0, ls=(0, (4, 2)),
                     label=r"$\theta_{4}(\mathrm{e}^{-x})^{%d}$" % d)
    tt = np.linspace(0, 4, 100)
    axs[1, col].plot(tt, np.exp(-tt), color=INK, lw=1.0, ls=(0, (4, 2)), label=r"$\mathrm{e}^{-t/(qT)}$")
    Nd = 160 if d == 2 else 40
    fn = os.path.join(_R, 'data', 'msc_modes', 'pmf', f"pmf_d{d}_CC_N{Nd}_q0.8.npz")
    if os.path.exists(fn):
        z = np.load(fn)
        mu1q = Q * 2.0 / d * math.sin(math.pi / (2 * Nd)) ** 2
        sl = slice(None, None, max(1, len(z["t"]) // 60))
        axs[0, col].plot(mu1q * z["t"][sl], z["f"][sl] * float(z["mfpt"]), ls="none", marker="o", mfc="white",
                         mec=ORANGE, mew=0.8, ms=2.8, label=f"discrete, $N={Nd}$, $q=0.8$")
    axs[0, col].set_xlim(0, 12)
    axs[0, col].set_ylim(0, 1.05)
    axs[0, col].set_xlabel(r"diffusive time $x=\mu_{1}t$")
    axs[0, col].set_ylabel(r"$q\,T\,g(t)$")
    axs[0, col].set_title(f"({'ab'[col]}) $d={d}$: arrival", loc="left")
    axs[0, col].legend(loc="lower right", ncol=1, handlelength=1.6, labelspacing=0.3)
    axs[1, col].set_xlim(0, 4)
    axs[1, col].set_yscale("log")
    axs[1, col].set_ylim(1e-2, 1.5)
    axs[1, col].set_xlabel(r"$t/(qT)$")
    axs[1, col].set_ylabel(r"$q\,T\,g(t)$")
    axs[1, col].set_title(f"({'cd'[col]}) $d={d}$: exponential regime", loc="left")
    axs[1, col].legend(loc="lower left", ncol=2, handlelength=1.6, labelspacing=0.3, columnspacing=1.0)
fig.tight_layout()
save(fig, "s6_mechanism_density")

# ------------------------------------------------------------------ other placements
fig, axs = plt.subplots(1, 2, figsize=(W, 2.7), sharey=True)
GEO = [("CC", "corner to opposite corner", BLUE, "o"), ("C2M", "corner to centre target", ORANGE, "s"),
       ("M2C", "centre to corner target", AQUA, "^")]
EX = FITS["continuum_exit"]
for ax, d in zip(axs, (2, 3)):
    for geo, lab, col_, mk in GEO:
        c = sel(L, d=d, geo=geo)
        ax.plot(arr(c, "N"), arr(c, "mode") / arr(c, "mfpt"), color=col_, lw=1.2, label=lab)
        dd = sel(D, d=d, geo=geo, q=Q)
        ax.plot(arr(dd, "N"), arr(dd, "mode_over_mfpt"), ls="none", marker=mk, mfc="white", mec=col_, mew=0.8, ms=3.2)
    ee = [r for r in sel(D, d=d, geo="EXIT", q=Q) if r["mode_over_mfpt"]]
    ax.plot(arr(ee, "N"), arr(ee, "mode_over_mfpt"), ls="none", marker="D", color=VIOLET, ms=3.2,
            label="centre to absorbing boundary (exit)")
    ax.axhline(EX[str(d)]["mode_over_mean"], color=VIOLET, lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("lattice size $N$")
    ax.set_title(f"({'ab'[d - 2]}) $d={d}$", loc="left")
axs[0].set_ylabel(r"$t^{*}/T$")
axs[0].legend(loc="lower left", handlelength=1.6)
fig.tight_layout()
save(fig, "s6_mechanism_geometries")
print("all figures done")
