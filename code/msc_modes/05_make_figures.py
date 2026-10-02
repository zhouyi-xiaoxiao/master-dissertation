"""
05_make_figures.py -- publication figures (vector PDF + PNG) from the saved
results.  Reads data/*.jsonl, data/fit_results.json, data/profiles.npz and
the stored PMFs.  No computation beyond cheap closed forms.
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import json, os, sys, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
import fptlib as F

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = _os.path.join(_R, 'data', 'msc_modes')
FIG = _os.path.join(_R, 'out', 'figures', 'msc_modes')
os.makedirs(FIG, exist_ok=True)

# categorical palette (fixed order; first three validate for all-pairs CVD separation)
BLUE, ORANGE, AQUA, YELLOW, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7"
INK, MUTED, FAINT = "#0b0b0b", "#52514e", "#b8b7b0"
DC = {1: BLUE, 2: ORANGE, 3: AQUA}
DM = {1: "o", 2: "s", 3: "^"}
plt.rcParams.update({
    "font.size": 8.5, "axes.labelsize": 9, "axes.titlesize": 9, "legend.fontsize": 7.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "lines.linewidth": 1.5, "lines.markersize": 4,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "axes.grid": True, "grid.color": "#e6e5e0",
    "grid.linewidth": 0.5, "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
    "figure.dpi": 120, "mathtext.fontset": "dejavusans",
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
    fig.savefig(os.path.join(FIG, name + ".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("saved", name)


NMAX = {1: 1.2e4, 2: 1.2e5, 3: 4500}

# ------------------------------------------------------------------ Fig 1
fig, axs = plt.subplots(1, 3, figsize=(10.2, 3.3))
for ax, d in zip(axs, (1, 2, 3)):
    c = [r for r in sel(L, d=d, geo="CC") if r["N"] <= NMAX[d]]
    N = arr(c, "N")
    ax.plot(N, arr(c, "mfpt") / Q, color=BLUE, label="MFPT, exact")
    ax.plot(N, arr(c, "mode") / Q, color=ORANGE, label="mode, exact (continuous time)")
    dd = sel(D, d=d, geo="CC", q=Q)
    ax.plot(arr(dd, "N"), arr(dd, "mode"), ls="none", marker="o", mfc="white", mec=ORANGE, mew=1.0, ms=4.5,
            label="mode, exact PMF argmax (discrete)")
    if d > 1:
        ax.plot(N, arr(c, "pred_L1") / Q, color=INK, ls=(0, (4, 2)), lw=0.9, label="mode, closed form (L1)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("lattice size $N$")
    ax.set_title({1: "(a) $d=1$: end to end", 2: "(b) $d=2$: corner to corner", 3: "(c) $d=3$: corner to corner"}[d],
                 loc="left")
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi * (hi / lo) ** 0.55)
    ax.legend(loc="upper left", handlelength=1.8)
axs[0].set_ylabel("time steps ($q=0.8$)")
save(fig, "fig1_mode_mean_vs_N")

# ------------------------------------------------------------------ Fig 2
fig, ax = plt.subplots(figsize=(7.4, 3.9))
for d in (1, 2, 3):
    c = sel(L, d=d, geo="CC")
    N = arr(c, "N"); rat = arr(c, "mode") / arr(c, "mfpt")
    ax.plot(N, rat, color=DC[d], label=f"$d={d}$, exact (continuous time)")
    dd = sel(D, d=d, geo="CC", q=Q)
    ax.plot(arr(dd, "N"), arr(dd, "mode_over_mfpt"), ls="none", marker=DM[d], mfc="white", mec=DC[d], mew=0.9, ms=4,
            label=f"$d={d}$, exact discrete PMF, $q=0.8$")
    if d > 1:
        ax.plot(N, arr(c, "pred_L1") / arr(c, "mfpt"), color=INK, ls=(0, (4, 2)), lw=0.9,
                label="closed form (L1)" if d == 2 else None)
ax.text(3e5, 0.36, "1D: 0.33328", color=BLUE, fontsize=7.5, va="bottom")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(4, 2e7); ax.set_ylim(3e-4, 1.3)
ax.set_xlabel("lattice size $N$"); ax.set_ylabel("mode / MFPT")
ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), ncol=1)
save(fig, "fig2_mode_over_mfpt")

# ------------------------------------------------------------------ Fig 3 (compensated plots)
fig, axs = plt.subplots(2, 2, figsize=(8.6, 6.2))
c2 = sel(L, d=2, geo="CC"); N2 = arr(c2, "N"); M2 = arr(c2, "mode"); T2 = arr(c2, "mfpt")
c3 = sel(L, d=3, geo="CC"); N3 = arr(c3, "N"); M3 = arr(c3, "mode"); T3 = arr(c3, "mfpt")
ax = axs[0, 0]
ax.plot(np.log(N2), T2 / N2 ** 2, ls="none", marker="o", color=BLUE, ms=3.5, label="exact")
xx = np.linspace(np.log(5), np.log(N2.max()), 50)
cc = FITS["asymptotics"]["2d_mfpt_const_limit_estimate"]
ax.plot(xx, 8 / np.pi * xx + cc, color=INK, lw=0.9, ls=(0, (4, 2)), label=r"$(8/\pi)\ln N + %.4f$" % cc)
ax.set_xlabel(r"$\ln N$"); ax.set_ylabel(r"$q\,\mathrm{MFPT}/N^2$"); ax.set_title("(a) $d=2$ mean", loc="left"); ax.legend()
ax = axs[0, 1]
ax.plot(np.log(np.log(N2)), M2 / N2 ** 2, ls="none", marker="o", color=ORANGE, ms=3.5, label="exact mode")
ax.plot(np.log(np.log(N2)), arr(c2, "pred_L1") / N2 ** 2, color=INK, lw=0.9, ls=(0, (4, 2)), label="closed form (L1)")
d2 = sel(D, d=2, geo="CC", q=Q)
ax.plot(np.log(np.log(arr(d2, "N"))), Q * arr(d2, "mode") / arr(d2, "N") ** 2, ls="none", marker="s", mfc="white",
        mec=ORANGE, mew=0.8, ms=3.5, label="exact discrete PMF, $q=0.8$")
z0 = np.log(np.log(N2[-1])); y0 = M2[-1] / N2[-1] ** 2
zz = np.array([z0 - 0.9, z0])
ax.plot(zz, y0 + 4 / np.pi ** 2 * (zz - z0) - 0.12, color=MUTED, lw=0.9)
ax.text(zz.mean(), y0 + 4 / np.pi ** 2 * (zz.mean() - z0) - 0.20, r"slope $4/\pi^2$", color=MUTED, fontsize=7.5, ha="left")
ax.set_xlabel(r"$\ln\ln N$"); ax.set_ylabel(r"$q\,\mathrm{mode}/N^2$"); ax.set_title("(b) $d=2$ mode", loc="left"); ax.legend(loc="upper left")
ax = axs[1, 0]
ax.plot(1 / N3, T3 / N3 ** 3, ls="none", marker="^", color=BLUE, ms=3.5, label="exact")
C3 = FITS["asymptotics"]["3d_C3_limit_richardson"]; C3b = FITS["asymptotics"]["3d_C3_1overN_coeff"]
xx = np.linspace(0, 0.2, 50)
ax.plot(xx, C3 + C3b * xx, color=INK, lw=0.9, ls=(0, (4, 2)), label=r"$%.4f %+.3f/N$" % (C3, C3b))
ax.set_xlabel(r"$1/N$"); ax.set_ylabel(r"$q\,\mathrm{MFPT}/N^3$"); ax.set_title("(c) $d=3$ mean", loc="left"); ax.legend()
ax = axs[1, 1]
ax.plot(np.log(N3), M3 / N3 ** 2, ls="none", marker="^", color=ORANGE, ms=3.5, label="exact mode")
ax.plot(np.log(N3), arr(c3, "pred_L1") / N3 ** 2, color=INK, lw=0.9, ls=(0, (4, 2)), label="closed form (L1)")
d3 = sel(D, d=3, geo="CC", q=Q)
ax.plot(np.log(arr(d3, "N")), Q * arr(d3, "mode") / arr(d3, "N") ** 2, ls="none", marker="s", mfc="white", mec=ORANGE,
        mew=0.8, ms=3.5, label="exact discrete PMF, $q=0.8$")
x0 = np.log(N3[-1]); y0 = M3[-1] / N3[-1] ** 2
xx = np.array([x0 - 2.5, x0])
ax.plot(xx, y0 + 6 / np.pi ** 2 * (xx - x0) - 0.45, color=MUTED, lw=0.9)
ax.text(xx.mean(), y0 + 6 / np.pi ** 2 * (xx.mean() - x0) - 0.95, r"slope $6/\pi^2$", color=MUTED, fontsize=7.5)
ax.set_xlabel(r"$\ln N$"); ax.set_ylabel(r"$q\,\mathrm{mode}/N^2$"); ax.set_title("(d) $d=3$ mode", loc="left"); ax.legend(loc="upper left")
fig.tight_layout()
save(fig, "fig3_compensated")

# ------------------------------------------------------------------ Fig 4 (local exponents)
fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.3), sharex=True)
E = FITS["effective_exponents"]
for d in (1, 2, 3):
    e = E[str(d)]
    axs[0].plot(e["N_geo_mean"], e["a_eff_mode"], color=DC[d], marker=DM[d], ms=3, label=f"$d={d}$")
    axs[1].plot(e["N_geo_mean"], e["a_eff_mfpt"], color=DC[d], marker=DM[d], ms=3, label=f"$d={d}$")
axs[0].axhline(2, color=MUTED, lw=0.8, ls=":"); axs[0].axhline(3, color=MUTED, lw=0.8, ls=":")
axs[1].axhline(2, color=MUTED, lw=0.8, ls=":"); axs[1].axhline(3, color=MUTED, lw=0.8, ls=":")
for ax, t in zip(axs, ("(a) mode", "(b) MFPT")):
    ax.set_xscale("log"); ax.set_xlabel("lattice size $N$"); ax.set_title(t, loc="left"); ax.legend(loc="center right")
axs[0].set_ylabel(r"local exponent $\mathrm{d}\ln(\cdot)/\mathrm{d}\ln N$")
axs[0].set_ylim(1.95, 3.05); axs[1].set_ylim(1.95, 3.25)
fig.tight_layout()
save(fig, "fig4_local_exponents")

# ------------------------------------------------------------------ Fig 5 (collapse)
P = np.load(_os.path.join(_R, 'data', 'msc_modes', 'profiles.npz'))
PS = json.load(open(_os.path.join(_R, 'data', 'msc_modes', 'profile_stats.json')))


def theta(x, d):
    k = np.arange(1, 400)[:, None]
    th = 1 + 2 * ((-1.0) ** k * np.exp(-k ** 2 * x[None, :])).sum(axis=0)
    return th ** d


fig, axs = plt.subplots(2, 2, figsize=(8.8, 6.4))
for col, d in enumerate((2, 3)):
    Ns = sorted(int(k.split("_")[1][1:]) for k in P.files if k.startswith(f"d{d}_") and k.endswith("_x"))
    cmap = plt.get_cmap("Blues" if d == 2 else "Greens")
    for i, N in enumerate(Ns):
        x = P[f"d{d}_N{N}_x"]; y = P[f"d{d}_N{N}_g_tau"]
        col_ = cmap(0.35 + 0.6 * i / max(1, len(Ns) - 1))
        st = [s for s in PS if s["d"] == d and s["N"] == N][0]
        X = st["mu1"] * st["mfpt"]
        axs[0, col].plot(x, y, color=col_, lw=1.2, label=f"$N={N}$")
        axs[1, col].plot(P[f"d{d}_N{N}_y"], P[f"d{d}_N{N}_g_tau_y"], color=col_, lw=1.2, label=f"$N={N}$")
    xx = np.linspace(0.05, 12, 400)
    axs[0, col].plot(xx, theta(xx, d), color=INK, lw=1.0, ls=(0, (4, 2)), label=r"$\vartheta_4(0,e^{-x})^{%d}$ ($N\to\infty$)" % d)
    tt = np.linspace(0, 4, 100)
    axs[1, col].plot(tt, np.exp(-tt), color=INK, lw=1.0, ls=(0, (4, 2)), label=r"$e^{-t/\mathrm{MFPT}}$")
    # discrete overlay
    Nd = 160 if d == 2 else 40
    fn = os.path.join(_R, 'data', 'msc_modes', 'pmf', f"pmf_d{d}_CC_N{Nd}_q0.8.npz")
    if os.path.exists(fn):
        z = np.load(fn)
        mu1q = Q * 2.0 / d * math.sin(math.pi / (2 * Nd)) ** 2
        sl = slice(None, None, max(1, len(z["t"]) // 60))
        axs[0, col].plot(mu1q * z["t"][sl], z["f"][sl] * float(z["mfpt"]), ls="none", marker="o", mfc="white", mec=ORANGE,
                         mew=0.8, ms=3, label=f"discrete PMF, $N={Nd}$, $q=0.8$")
    axs[0, col].set_xlim(0, 12); axs[0, col].set_ylim(0, 1.05)
    axs[0, col].set_xlabel(r"$x=\mu_1 t$  (diffusive time)"); axs[0, col].set_ylabel(r"$\mathrm{MFPT}\times g(t)$")
    axs[0, col].set_title(f"({'ab'[col]}) $d={d}$: arrival regime", loc="left")
    axs[0, col].legend(loc="lower right", ncol=1)
    axs[1, col].set_xlim(0, 4); axs[1, col].set_yscale("log"); axs[1, col].set_ylim(1e-2, 1.5)
    axs[1, col].legend(loc="lower left", ncol=2)
    axs[1, col].set_xlabel(r"$t/\mathrm{MFPT}$"); axs[1, col].set_ylabel(r"$\mathrm{MFPT}\times g(t)$")
    axs[1, col].set_title(f"({'cd'[col]}) $d={d}$: exponential regime", loc="left")
    axs[1, col].legend(loc="lower left", ncol=2)
fig.tight_layout()
save(fig, "fig5_rescaled_density_collapse")

# ------------------------------------------------------------------ Fig 6 (prediction accuracy)
fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.4), sharey=True)
series = [("pred_L0", r"L0: $\ln(W X)/\mu_1$", BLUE, "o"), ("pred_L1", r"L1: $\mathrm{MFPT}\,\ln(WX)/(X+W-1)$", ORANGE, "s"),
          ("pred_2pole", "two poles (exact $\\nu_{0,1}$, residues)", AQUA, "^"), ("pred_3pole", "three poles", YELLOW, "D")]
for ax, d in zip(axs, (2, 3)):
    c = sel(L, d=d, geo="CC"); N = arr(c, "N"); M = arr(c, "mode")
    for key, lab, col_, mk in series:
        ax.plot(N, np.abs(arr(c, key) / M - 1), color=col_, marker=mk, ms=3, lw=1.1, label=lab)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("lattice size $N$")
    ax.set_title(f"({'ab'[d-2]}) $d={d}$, corner to corner", loc="left")
axs[0].set_ylabel("|predicted mode / exact mode $-$ 1|")
axs[0].legend(loc="lower left")
fig.tight_layout()
save(fig, "fig6_prediction_accuracy")

# ------------------------------------------------------------------ Fig 7 (parity, discrete vs continuous)
fig, axs = plt.subplots(1, 2, figsize=(8.8, 3.4))
ax = axs[0]
z = np.load(_os.path.join(_R, 'data', 'msc_modes', 'pmf', 'pmf_d1_CC_N51_q1.0.npz'))
t, f = z["t"], z["f"]
w = (t >= 700) & (t <= 1000)
ev = w & (t % 2 == 0); od = w & (t % 2 == 1)
ax.plot(t[ev], f[ev], ls="none", marker="o", ms=2.2, color=BLUE, label="$f(t)$, even $t$")
ax.plot(t[od], f[od], ls="none", marker="s", ms=2.2, color=ORANGE, label="$f(t)$, odd $t$")
fb = 0.5 * (f[:-1] + f[1:]); tb = t[:-1] + 0.5
wb = (tb >= 700) & (tb <= 1000)
ax.plot(tb[wb], fb[wb], color=AQUA, lw=1.3, label="two-step average")
ch = F.Chain1D(51, 1.0)
tc = np.linspace(700, 1000, 200)
ax.plot(tc, ch.g_cont(tc), color=INK, lw=0.9, ls=(0, (4, 2)), label="continuous-time density")
ax.axvline(float(t[np.argmax(f)]), color=BLUE, lw=0.8, ls=":")
ax.axvline(float(tb[np.argmax(fb)]), color=AQUA, lw=0.8, ls=":")
ax.set_xlabel("$t$"); ax.set_ylabel("probability per step")
ax.set_title("(a) parity at $q=1$ ($d=1$, $N=51$)", loc="left"); ax.legend(loc="lower center", ncol=2)
ax = axs[1]
off = FITS["discrete_vs_continuous_offsets"]
for d in (1, 2, 3):
    o = sorted([x for x in off if x["d"] == d and x["geo"] == "CC"], key=lambda x: x["N"])
    ax.plot([x["N"] for x in o], [x["offset_refined"] for x in o], color=DC[d], marker=DM[d], ms=3.5, lw=1.0,
            label=f"$d={d}$")
ax.set_xscale("log"); ax.set_xlabel("lattice size $N$")
ax.set_ylabel(r"mode$_{\mathrm{discrete}}(q{=}0.8)$ $-$ mode$_{\mathrm{cont}}/q$  (steps)")
ax.set_title("(b) discrete vs continuous time", loc="left"); ax.legend()
fig.tight_layout()
save(fig, "fig7_parity_and_discrete_offset")

# ------------------------------------------------------------------ Fig 8 (geometries)
fig, axs = plt.subplots(1, 2, figsize=(8.8, 3.5), sharey=True)
GEO = [("CC", "corner $\\to$ opposite corner", BLUE, "o"), ("C2M", "corner $\\to$ centre target", ORANGE, "s"),
       ("M2C", "centre $\\to$ corner target", AQUA, "^")]
EX = FITS["continuum_exit"]
for ax, d in zip(axs, (2, 3)):
    for geo, lab, col_, mk in GEO:
        c = sel(L, d=d, geo=geo)
        ax.plot(arr(c, "N"), arr(c, "mode") / arr(c, "mfpt"), color=col_, lw=1.3, label=lab)
        dd = sel(D, d=d, geo=geo, q=Q)
        ax.plot(arr(dd, "N"), arr(dd, "mode_over_mfpt"), ls="none", marker=mk, mfc="white", mec=col_, mew=0.8, ms=3.5)
    ee = sel(D, d=d, geo="EXIT", q=Q)
    ee = [r for r in ee if r["mode_over_mfpt"]]
    ax.plot(arr(ee, "N"), arr(ee, "mode_over_mfpt"), ls="none", marker="D", color=VIOLET, ms=3.5,
            label="centre $\\to$ absorbing boundary (exit)")
    ax.axhline(EX[str(d)]["mode_over_mean"], color=VIOLET, lw=0.8, ls=":")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("lattice size $N$")
    ax.set_title(f"({'ab'[d-2]}) $d={d}$", loc="left")
axs[0].set_ylabel("mode / MFPT"); axs[0].legend(loc="lower left")
fig.tight_layout()
save(fig, "fig8_geometries")
print("all figures done")
