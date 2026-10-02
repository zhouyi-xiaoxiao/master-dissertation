#!/usr/bin/env python
"""Check by the second implementation, part 9: figures (vector PDF + PNG) from the data of this check only."""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import json, os, subprocess, sys
import numpy as np
import mpmath as mp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vspec

HERE = os.path.dirname(os.path.abspath(__file__))
V = _os.path.join(_R, 'data', 'msc_modes_verify')
FIG = _os.path.join(_R, 'out', 'figures', 'msc_modes_verify')
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "legend.fontsize": 7.5, "figure.dpi": 150,
                     "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
CB = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000"]
q = 0.8
A = json.load(open(_os.path.join(_R, 'data', 'msc_modes_verify', 'analysis.json')))
X = json.load(open(_os.path.join(_R, 'data', 'msc_modes_verify', 'extra.json')))
C = json.load(open(_os.path.join(_R, 'data', 'msc_modes_verify', 'constants.json')))
pol = {d: [r for r in A["poles_cc"] if r["d"] == d] for d in (2, 3)}
disc = {int(d): v for d, v in A["cc_q0.8"].items()}
c2 = float(X["c2_closed_form"]); C3 = float(X["C3_closed_form"]); C3p = float(X["C3prime_closed_form"])
r1 = float(C["1d_mode_over_mfpt"])


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, name + ".png"), bbox_inches="tight", dpi=220)
    plt.close(fig)


# ------------------------------------------------------------------ Fig 1: mode and mean vs N
fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.4))
for ax, d in zip(axs, (1, 2, 3)):
    Nd = np.array([x["N"] for x in disc[d]]); md = np.array([x["mode"] for x in disc[d]]); mf = np.array([x["mfpt"] for x in disc[d]])
    ax.loglog(Nd, mf, "s", ms=3.5, color=CB[0], label="MFPT, exact")
    ax.loglog(Nd, md, "o", ms=3.5, color=CB[1], label=r"mode, exact $\arg\max_t f(t)$")
    if d == 1:
        Ng = np.logspace(np.log10(5), 3, 100)
        ax.loglog(Ng, r1 * (Ng - 0.5) ** 2 / q, "-", lw=1, color=CB[1], label=r"$0.333284\,(N-\frac{1}{2})^2/q$")
        ax.loglog(Ng, Ng * (Ng - 1) / q, "-", lw=1, color=CB[0], label=r"$N(N-1)/q$")
    else:
        Np = np.array([x["N"] for x in pol[d]]); sel = Np <= (2000 if d == 2 else 400)
        ax.loglog(Np[sel], np.array([x["mode"] for x in pol[d]])[sel] / q, "-", lw=1, color=CB[1], label="mode, pole expansion (cont. time)")
        ax.loglog(Np[sel], np.array([x["mfpt"] for x in pol[d]])[sel] / q, "-", lw=1, color=CB[0])
        L1 = np.array([x["mode"] * (1 + x["L1_err"]) for x in pol[d]])[sel] / q
        ax.loglog(Np[sel], L1, "--", lw=1, color=CB[6], label=r"closed form $\mathrm{MFPT}\,\ln(2dX)/(X+2d-1)$")
    ax.set_xlabel("lattice size $N$"); ax.set_title(f"$d={d}$, corner to corner, $q=0.8$", fontsize=9)
    ax.legend(frameon=False, loc="lower right", fontsize=6.5)
axs[0].set_ylabel("first-passage time (steps)")
save(fig, "figV1_mode_mean_vs_N")

# ------------------------------------------------------------------ Fig 2: ratio
fig, ax = plt.subplots(figsize=(5.4, 3.8))
for d, c in zip((1, 2, 3), CB[:3]):
    Nd = np.array([x["N"] for x in disc[d]]); rr = np.array([x["ratio"] for x in disc[d]])
    ax.loglog(Nd, rr, "o", ms=3.5, color=c, label=f"$d={d}$: exact discrete PMF")
    if d > 1:
        Np = np.array([x["N"] for x in pol[d]])
        ax.loglog(Np, [x["ratio"] for x in pol[d]], "-", lw=1, color=c, label=f"$d={d}$: pole expansion")
        ax.loglog(Np, [x["ratio"] * (1 + x["L1_err"]) for x in pol[d]], "--", lw=0.9, color="k")
ax.axhline(r1, color=CB[0], lw=0.8)
ax.plot([], [], "--", lw=0.9, color="k", label="closed form (no fitted parameter)")
ax.set_xlabel("lattice size $N$"); ax.set_ylabel("mode / MFPT"); ax.set_xlim(4, 3e7); ax.set_ylim(3e-4, 1.0)
ax.legend(frameon=False, loc="lower left")
save(fig, "figV2_mode_over_mfpt")

# ------------------------------------------------------------------ Fig 3: compensated plots
fig, axs = plt.subplots(2, 2, figsize=(9, 6.2))
Np = np.array([x["N"] for x in pol[2]], float); mf = np.array([x["mfpt"] for x in pol[2]]); mo = np.array([x["mode"] for x in pol[2]])
ax = axs[0, 0]
ax.semilogx(Np, mf / Np ** 2 - 8 / np.pi * np.log(Np), "o-", ms=3, lw=0.8, color=CB[0])
ax.axhline(c2, color="k", ls="--", lw=0.8, label=r"$c_2=\frac{8}{\pi}[\gamma+4\ln2+\frac{1}{2}\ln\pi-2\ln\Gamma(\frac{1}{4})]-2-\frac{4}{\pi}$" + f"\n$=${c2:.9f}")
ax.set_xlabel("$N$"); ax.set_ylabel(r"$q\,\mathrm{MFPT}/N^2-(8/\pi)\ln N$"); ax.set_title("(a) 2D mean: exact constant", fontsize=9); ax.legend(frameon=False)
ax = axs[0, 1]
ax.plot(np.log(np.log(Np)), mo / Np ** 2, "o", ms=3, color=CB[1], label="exact (pole expansion)")
Nd = np.array([x["N"] for x in disc[2]], float)
ax.plot(np.log(np.log(Nd)), [x["qmode_over_N2"] for x in disc[2]], "x", ms=4, color=CB[6], label="exact discrete argmax, $q=0.8$")
ax.plot(np.log(np.log(Np)), [x["mode"] * (1 + x["L1_err"]) / x["N"] ** 2 for x in pol[2]], "--", lw=1, color="k", label="closed form")
xx = np.log(np.log(Np)); ax.plot(xx, mo[-1] / Np[-1] ** 2 + 4 / np.pi ** 2 * (xx - xx[-1]), ":", color=CB[2], label=r"asymptotic slope $4/\pi^2$")
ax.set_xlabel(r"$\ln\ln N$"); ax.set_ylabel(r"$q\,\mathrm{mode}/N^2$"); ax.set_title(r"(b) 2D mode is not $\propto N^2$", fontsize=9); ax.legend(frameon=False)
Np3 = np.array([x["N"] for x in pol[3]], float); mf3 = np.array([x["mfpt"] for x in pol[3]]); mo3 = np.array([x["mode"] for x in pol[3]])
ax = axs[1, 0]
ax.plot(1 / Np3, mf3 / Np3 ** 3, "o", ms=3, color=CB[0], label="exact")
xs = np.linspace(0, 0.2, 50); ax.plot(xs, C3 + C3p * xs, "--", color="k", lw=0.8, label=f"$C_3+C_3'/N$, $C_3=${C3:.6f}, $C_3'=${C3p:.6f}")
ax.set_xlabel("$1/N$"); ax.set_ylabel(r"$q\,\mathrm{MFPT}/N^3$"); ax.set_title("(c) 3D mean: exact constants", fontsize=9); ax.legend(frameon=False)
ax = axs[1, 1]
ax.semilogx(Np3, mo3 / Np3 ** 2, "o", ms=3, color=CB[1], label="exact (pole expansion)")
Nd3 = np.array([x["N"] for x in disc[3]], float)
ax.semilogx(Nd3, [x["qmode_over_N2"] for x in disc[3]], "x", ms=4, color=CB[6], label="exact discrete argmax, $q=0.8$")
ax.semilogx(Np3, 6 / np.pi ** 2 * (np.log(Np3) + np.log(np.pi ** 2 * C3)), "--", lw=1, color="k", label=r"$(6/\pi^2)(\ln N+\ln\pi^2C_3)$")
ax.set_xlabel("$N$"); ax.set_ylabel(r"$q\,\mathrm{mode}/N^2$"); ax.set_title(r"(d) 3D mode $\propto N^2\ln N$, not $N^3$", fontsize=9); ax.legend(frameon=False)
fig.tight_layout(); save(fig, "figV3_compensated")

# ------------------------------------------------------------------ Fig 4: rescaled PMFs
STEP = _os.path.join(_R, 'code', 'msc_modes_verify', 'stepper'); cache = _os.path.join(_R, 'data', 'msc_modes_verify', 'pmf', 'collapse.npz')
cases = [(1, 20), (1, 50), (1, 200), (2, 20), (2, 35), (2, 100), (2, 200), (3, 10), (3, 20), (3, 40), (3, 80)]
if os.path.exists(cache):
    Z = dict(np.load(cache))
else:
    Z = {}
    for d, N in cases:
        mfp = N * (N - 1) / q if d == 1 else vspec.Corner(d, N).mfpt() / q
        tmin = int(4 * mfp) if d == 1 else int(min(4 * mfp, 40 / (q * vspec.e1d(1, N, d))))
        tmp = _os.path.join(_R, 'data', 'msc_modes_verify', 'pmf', '_c.bin')
        subprocess.run([STEP, str(d), str(N), repr(q), "CC", "1.5", str(tmin), "2000000000", tmp], check=True, capture_output=True)
        f = np.fromfile(tmp); os.remove(tmp)
        idx = np.unique(np.round(np.logspace(0, np.log10(len(f)), 1500)).astype(int)) - 1
        Z[f"t_{d}_{N}"] = idx + 1.0; Z[f"f_{d}_{N}"] = f[idx]; Z[f"m_{d}_{N}"] = np.array([mfp])
    np.savez_compressed(cache, **Z)
fig, axs = plt.subplots(2, 3, figsize=(10.5, 6))
for col, d in enumerate((1, 2, 3)):
    ax1, ax2 = axs[0, col], axs[1, col]
    Ns = [N for dd, N in cases if dd == d]
    for N, c in zip(Ns, CB):
        t = Z[f"t_{d}_{N}"]; f = Z[f"f_{d}_{N}"]; mfp = float(Z[f"m_{d}_{N}"][0])
        mu1 = q * float(vspec.e1d(1, N, d))
        ax1.plot(mu1 * t, mfp * f, "-", lw=1, color=c, label=f"$N={N}$")
        ax2.semilogy(t / mfp, mfp * f, "-", lw=1, color=c, label=f"$N={N}$")
        im = np.argmax(f); ax1.plot(mu1 * t[im], mfp * f[im], "o", ms=3, color=c)
    xs = np.linspace(0.05, 14, 400)
    if d > 1:
        ax1.plot(xs, [float(mp.jtheta(4, 0, mp.e ** (-x)) ** d) for x in xs], "k--", lw=0.9, label=r"$\vartheta_4(0,e^{-\mu_1 t})^d$")
        ax2.semilogy(np.linspace(0, 4, 50), np.exp(-np.linspace(0, 4, 50)), "k--", lw=0.9, label=r"$e^{-t/\mathrm{MFPT}}$")
    ax1.set_xlim(0, 14); ax1.set_xlabel(r"$\mu_1 t$  (diffusive time)"); ax2.set_xlabel(r"$t/\mathrm{MFPT}$"); ax2.set_xlim(0, 4); ax2.set_ylim(1e-2, 2)
    ax1.set_title(f"$d={d}$: arrival regime (dots: modes)", fontsize=9); ax2.set_title(f"$d={d}$: tail regime", fontsize=9)
    ax1.legend(frameon=False); ax2.legend(frameon=False)
axs[0, 0].set_ylabel(r"$\mathrm{MFPT}\times f(t)$"); axs[1, 0].set_ylabel(r"$\mathrm{MFPT}\times f(t)$")
fig.tight_layout(); save(fig, "figV4_rescaled_pmf")

# ------------------------------------------------------------------ Fig 5: verification summary
fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.3))
ax = axs[0]
for d, c in ((2, CB[1]), (3, CB[2])):
    rr = [(x["N"], abs(x["rel_diff_mode"])) for x in pol[d] if "rel_diff_mode" in x]
    ax.loglog(*zip(*[(n, max(v, 1e-17)) for n, v in rr]), "o-", ms=3, lw=0.7, color=c, label=f"$d={d}$")
ax.set_xlabel("$N$"); ax.set_ylabel("|pole expansion / Talbot − 1|"); ax.set_title("(a) two inversion routes, cont.-time mode", fontsize=9); ax.legend(frameon=False)
ax = axs[1]
for d, ls in ((2, "-"), (3, "--")):
    Np_ = [x["N"] for x in pol[d] if x["N"] >= 5]
    for key, c, lab in (("L0_err", CB[4], "$L_0$"), ("L1_err", CB[0], "closed form $L_1$"), ("pole2_err", CB[2], "2 exact poles"), ("pole3_err", CB[3], "3 exact poles")):
        ax.loglog(Np_, [abs(x[key]) for x in pol[d] if x["N"] >= 5], ls, lw=1, color=c, label=lab + f" ($d={d}$)")
ax.axhline(0.01, color="grey", lw=0.6, ls=":"); ax.set_xlabel("$N$"); ax.set_ylabel("|prediction / exact mode − 1|")
ax.set_title("(b) accuracy of mode predictions", fontsize=9); ax.legend(frameon=False, ncol=2, fontsize=6)
ax = axs[2]
for d, c in ((2, CB[1]), (3, CB[2])):
    off = A[f"summary_d{d}"]["offsets"]
    ax.semilogx([o[0] for o in off], [o[1] / q for o in off], "o", ms=4, color=c, label=f"$d={d}$")
ax.set_xlabel("$N$"); ax.set_ylabel(r"$t^*_{\rm disc}(q)-t^*_{\rm cont}(1)/q$  (steps)"); ax.set_title("(c) discrete minus continuous mode, $q=0.8$", fontsize=9); ax.legend(frameon=False)
fig.tight_layout(); save(fig, "figV5_verification")
print("figures written to", FIG)
