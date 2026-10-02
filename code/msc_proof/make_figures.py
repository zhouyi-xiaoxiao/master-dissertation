"""
Figures of the mean first-passage part of the analysis (vector PDF + PNG, English labels).

  fig1_verification   float64 agreement of direct solve / double sum / single sum, and the
                      overflow of form (a) of the single sum
  fig2_asymptotics    relative error of the truncated large-N expansion with closed-form
                      coefficients
  fig3_landscape      MFPT to the corner target from every start site (Theorem 3 single sum)

Reads results/*.csv produced by verify_mfpt.py and asymptotics.py; fig3 computes its own data
and writes results/landscape_check.json.
"""
from __future__ import annotations
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository

import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mpmath as mp
import numpy as np

import mfpt2d as M

HERE = os.path.dirname(os.path.abspath(__file__))
RES = _os.path.join(_R, 'data', 'msc_proof')
FIG = _os.path.join(_R, 'out', 'figures', 'msc_proof')

# palette: first slots of a CVD-validated categorical order (fixed order, never cycled)
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK = "#0b0b0b"; INK2 = "#52514e"; GRID = "#e4e3df"; NEUTRAL = "#8a8983"
plt.rcParams.update({
    "font.size": 8.5, "axes.labelsize": 9, "axes.titlesize": 9, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.linewidth": 0.6,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.grid": True, "axes.axisbelow": True,
    "lines.linewidth": 1.6, "lines.markersize": 4.5, "pdf.fonttype": 42, "ps.fonttype": 42,
    "figure.dpi": 150, "savefig.dpi": 300, "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
})


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, name + ".png"), bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


# ------------------------------------------------------------------ Fig 1
rows = list(csv.DictReader(open(_os.path.join(_R, 'data', 'msc_proof', 'verify_float.csv'))))
N = np.array([int(r["N"]) for r in rows])
e_dir = np.array([float(r["rel_direct_vs_clean"]) if r["rel_direct_vs_clean"] not in ("", "nan") else np.nan for r in rows])
e_dbl = np.array([float(r["rel_double_vs_clean"]) for r in rows])
e_th = np.array([float(r["rel_form_a_vs_clean"]) if r["rel_form_a_vs_clean"] not in ("", "nan") else np.nan for r in rows])
floor = 1e-17
fig, ax = plt.subplots(figsize=(5.2, 3.3))
m = ~np.isnan(e_dir)
ax.loglog(N[m], np.maximum(e_dir[m], floor), "-o", lw=1.0, color=C[0], label="direct solve $(I-Q)m=1$ vs single sum (S)")
ax.loglog(N, np.maximum(e_dbl, floor), "-s", lw=1.0, color=C[1], label="cosine double sum (D) vs single sum (S)")
m = ~np.isnan(e_th)
ax.loglog(N[m], np.maximum(e_th[m], floor), "-^", lw=1.0, color=C[2], label="form (a) vs single sum (S)")
ax.axvline(403, color=NEUTRAL, lw=0.8)
ax.text(430, 3e-16, "form (a)\noverflows in float64\nfor $N\\geq 403$", fontsize=7.5, color=INK2, va="bottom")
ax.set_xlabel("lattice side $N$")
ax.set_ylabel("relative difference (float64)")
ax.set_ylim(5e-18, 1e-8)
ax.set_xlim(1.6, 9000)
ax.legend(loc="upper left", frameon=False)
ax.set_title("Three independent routes to the corner-to-corner MFPT agree", loc="left")
save(fig, "fig1_verification")

# ------------------------------------------------------------------ Fig 2
mp.mp.dps = 50
rows = list(csv.DictReader(open(_os.path.join(_R, 'data', 'msc_proof', 'asymptotics.csv'))))
seen = set(); data = []
for r in rows:
    n = int(r["N"])
    if n in seen:
        continue
    seen.add(n); data.append((n, mp.mpf(r["qT"])))
data.sort()
g14 = mp.gamma(mp.mpf(1) / 4)
varpi = g14 ** 2 / (2 * mp.sqrt(2 * mp.pi))
X = varpi ** 4 / mp.pi ** 2
C2 = 8 / mp.pi * (mp.euler + mp.log(4 * mp.sqrt(2) / varpi) - mp.mpf(1) / 2 - mp.pi / 4)
coef = [X / 9, -mp.pi * X * (25 * X + 648) / 10800, mp.pi ** 2 * X ** 2 * (1225 * X + 12879) / 1905120]


def trunc(n, terms):
    v = 8 / mp.pi * n ** 2 * mp.log(n)
    if terms >= 2:
        v += C2 * n ** 2
    for j in range(3, terms + 1):
        v += coef[j - 3] * mp.mpf(n) ** (2 - 2 * (j - 2))
    return v


Ns = np.array([d[0] for d in data], dtype=float)
fig, ax = plt.subplots(figsize=(5.8, 3.6))
labels = ["1 term: $(8/\\pi)N^2\\ln N$", "2 terms: $+\\,C_2N^2$", "3 terms: $+\\,C_0$",
          "4 terms: $+\\,C_{-2}N^{-2}$", "5 terms: $+\\,C_{-4}N^{-4}$"]
table = {}
for t in range(1, 6):
    err = np.array([float(abs(trunc(n, t) - T) / T) for n, T in data])
    table[t] = err
    ok = err > 1e-33
    ax.loglog(Ns[ok], err[ok], "-o", color=C[t - 1], label=labels[t - 1], markersize=3.2)
ax.text(Ns[-1] * 1.25, table[1][-1] * 4.5, "1 term", fontsize=7.5, color=INK2, va="center")
ax.text(Ns[-1] * 1.25, table[2][-1], "2 terms", fontsize=7.5, color=INK2, va="center")
ax.text(Ns[-1] * 1.25, table[3][-1], "3 terms", fontsize=7.5, color=INK2, va="center")
ax.set_xlim(1.6, 1.5e7)
ax.set_ylim(1e-33, 3)
ax.set_xlabel("lattice side $N$")
ax.set_ylabel("relative error of truncated expansion")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), frameon=False, ncol=2)
ax.set_title("Large-$N$ expansion with closed-form coefficients vs exact single sum", loc="left")
save(fig, "fig2_asymptotics")
with open(_os.path.join(_R, 'data', 'msc_proof', 'asymptotic_truncation_errors.csv'), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["N", "rel_err_1term", "rel_err_2terms", "rel_err_3terms", "rel_err_4terms", "rel_err_5terms"])
    for i, (n, T) in enumerate(data):
        w.writerow([n] + [f"{table[t][i]:.6e}" for t in range(1, 6)])


# ------------------------------------------------------------------ Fig 3
def landscape(N, q, a):
    """MFPT to target a from every start site via the Theorem-3 single sum (float64, overflow-free)."""
    a1, a2 = a
    k = np.arange(1, N)[:, None]
    n = np.arange(1, N + 1)[None, :]
    phi = 2 * np.arcsinh(np.sin(np.pi * k / (2 * N)))
    cth = np.cos(np.pi * k * (2 * n - 1) / (2 * N))                 # cos theta_k(n)
    A = cth * cth[:, [a1 - 1]]                                       # C_k(s1, a1)

    def Psi(u, v):
        m1 = np.abs(u - v); m2 = u + v - 1
        return (np.exp(-m1 * phi) + np.exp(-(2 * N - m1) * phi) + np.exp(-m2 * phi) + np.exp(-(2 * N - m2) * phi)) \
            / ((1 - np.exp(-2 * N * phi)) * np.sinh(phi))
    B = Psi(n, a2)                                                   # Psi_k(s2, a2)
    const = float(np.sum(cth[:, a1 - 1] ** 2 * Psi(a2, a2)[:, 0]))
    s2 = np.arange(1, N + 1)
    tau = np.where(s2 <= a2, (a2 - s2) * (a2 + s2 - 1), (s2 - a2) * (2 * N + 1 - a2 - s2)).astype(float)
    return 2 / q * (tau[None, :] + 2 * N * (const - A.T @ B))


N0, q0 = 35, 0.8
Tmap = landscape(N0, q0, (N0, N0))
# independent check: direct solve for every start
import scipy.sparse as sps, scipy.sparse.linalg as spla
P = M.transition_matrix(N0, q0)
ia = N0 * N0 - 1
A_ = (sps.identity(ia, format="csc") - P[:ia, :ia]).tocsc()
mvec = np.append(spla.spsolve(A_, np.ones(ia)), 0.0).reshape(N0, N0)
relerr = np.abs(Tmap - mvec)[mvec > 0] / mvec[mvec > 0]
chk = dict(N=N0, q=q0, target=[N0, N0], max_rel_err_vs_direct=float(relerr.max()),
           corner_value=float(Tmap[0, 0]), target_value=float(Tmap[-1, -1]))
# a second, interior target
Tmap2 = landscape(N0, q0, (9, 23))
i2 = (9 - 1) * N0 + (23 - 1)
keep = np.array([i for i in range(N0 * N0) if i != i2])
m2 = np.zeros(N0 * N0)
m2[keep] = spla.spsolve((sps.identity(len(keep), format="csc") - P[keep][:, keep]).tocsc(), np.ones(len(keep)))
m2 = m2.reshape(N0, N0)
chk["interior_target"] = [9, 23]
chk["interior_max_rel_err_vs_direct"] = float((np.abs(Tmap2 - m2)[m2 > 0] / m2[m2 > 0]).max())
json.dump(chk, open(_os.path.join(_R, 'data', 'msc_proof', 'landscape_check.json'), "w"), indent=1)
print(chk)

fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1), constrained_layout=True)
for ax, Tm, tgt, ttl in ((axes[0], Tmap, (N0, N0), "target at corner $(N,N)$"),
                         (axes[1], Tmap2, (9, 23), "target at interior site $(9,23)$")):
    im = ax.imshow(Tm.T, origin="lower", cmap="Blues", extent=(0.5, N0 + 0.5, 0.5, N0 + 0.5), vmin=0)
    ax.grid(False)
    ax.plot([tgt[0]], [tgt[1]], marker="s", color=C[1], markersize=5, markeredgecolor="white", markeredgewidth=0.8)
    ax.annotate("target", tgt, xytext=(-28, -14) if tgt[0] > 20 else (8, 6), textcoords="offset points", fontsize=7.5, color=INK)
    ax.set_xlabel("start $s_1$"); ax.set_ylabel("start $s_2$")
    ax.set_title(ttl, loc="left")
    cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label("MFPT $T_{s\\to a}$ (steps)")
    cb.outline.set_linewidth(0.4)
fig.suptitle(f"MFPT from every start site by the single sum of Theorem 3 ($N={N0}$, $q={q0}$)", x=0.01, ha="left", fontsize=9)
save(fig, "fig3_landscape")
