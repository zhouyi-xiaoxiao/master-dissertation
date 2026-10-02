"""
Figures of Section 4 (s4_mfpt), "Exact mean first-passage times".

Adapted from code/msc_proof/make_figures.py with the labels of the article
(form (a) / form (b)) and ZERO-BASED site coordinates.

  figures/s4_mfpt_verification.pdf   float64 agreement of three routes to q T_N and the
                                     overflow of form (a)                 (s4_mfpt:fig-verif)
  figures/s4_mfpt_asymptotics.pdf    relative error of the truncated large-N expansion
                                                                          (s4_mfpt:fig-asym)
  figures/s4_mfpt_landscape.pdf      MFPT from every start site by the single sum for an
                                     arbitrary pair                       (s4_mfpt:fig-landscape)

Inputs (read only):
  data/msc_proof/verify_float.csv          float64 comparisons (research run)
  data/msc_proof_verify/v5_extra.json      float64 form (b) vs 40-digit values
  data/msc_proof/asymptotics.csv           q T_N to 50 digits (form (b))
Outputs:
  figures/s4_mfpt_*.pdf
  data/article/s4_mfpt_truncation_errors.csv       data plotted in fig-asym
  data/article/s4_mfpt_landscape_check.json        pair formula vs sparse direct solve

Run:  python code/article/s4_mfpt_figures.py        (about 10 s)
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
import scipy.sparse as sps
import scipy.sparse.linalg as spla

HERE = os.path.dirname(os.path.abspath(__file__))
ART = _R
ROOT = _R            # repository root
FIG = _os.path.join(_R, 'figures')
DATA = _os.path.join(_R, 'data', 'article')

# palette and style of the research figures (fixed categorical order)
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
    path = os.path.join(FIG, name + ".pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.relpath(path, ROOT))


def num(x):
    return float(x) if x not in ("", "nan") else np.nan


# ------------------------------------------------------------------ fig-verif
rows = list(csv.DictReader(open(_os.path.join(_R, 'data', 'msc_proof', 'verify_float.csv'))))
N = np.array([int(r["N"]) for r in rows])
e_dir = np.array([num(r["rel_direct_vs_clean"]) for r in rows])
e_dbl = np.array([num(r["rel_double_vs_clean"]) for r in rows])
e_a = np.array([num(r["rel_form_a_vs_clean"]) for r in rows])
first_nan = int(min(n for n, e in zip(N, e_a) if np.isnan(e)))
assert first_nan == 403, first_nan
v5 = json.load(open(_os.path.join(_R, 'data', 'msc_proof_verify', 'v5_extra.json')))["float64_formS_vs_mp"]
N_b = np.array(sorted(int(k) for k in v5)); e_b = np.array([v5[str(n)] for n in N_b])

floor = 1e-17                      # exact zeros of the float64 comparison are drawn at this level
fig, ax = plt.subplots(figsize=(5.4, 3.5))
m = ~np.isnan(e_dir)
ax.loglog(N[m], np.maximum(e_dir[m], floor), "-o", lw=1.0, color=C[0],
          label=r"direct solve of $(I-Q)h=\mathbf{1}$ vs form (b)")
ax.loglog(N, np.maximum(e_dbl, floor), "-s", lw=1.0, color=C[1], label="cosine double sum vs form (b)")
m = ~np.isnan(e_a)
ax.loglog(N[m], np.maximum(e_a[m], floor), "-^", lw=1.0, color=C[2], label="form (a) vs form (b)")
ax.loglog(N_b, e_b, "D", ms=4.0, color=C[3], label="form (b): float64 vs 40-digit arithmetic")
ax.axvline(403, color=NEUTRAL, lw=0.8)
ax.text(430, 6e-16, "form (a) overflows\nin float64 for $N\\geq 403$", fontsize=7.5, color=INK2, va="bottom")
ax.set_xlabel("lattice side $N$")
ax.set_ylabel("relative difference (float64)")
ax.set_ylim(5e-18, 3e-7)
ax.set_xlim(1.6, 9000)
ax.legend(loc="upper left", frameon=False)
save(fig, "s4_mfpt_verification")

# ------------------------------------------------------------------ fig-asym
mp.mp.dps = 50
rows = list(csv.DictReader(open(_os.path.join(_R, 'data', 'msc_proof', 'asymptotics.csv'))))
seen = set(); data = []
for r in rows:
    n = int(r["N"])
    if n in seen:
        continue
    seen.add(n); data.append((n, mp.mpf(r["qT"])))
data.sort()
varpi = mp.gamma(mp.mpf(1) / 4) ** 2 / (2 * mp.sqrt(2 * mp.pi))            # lemniscate constant
Xi = varpi ** 4 / mp.pi ** 2
C2 = 8 / mp.pi * (mp.euler + mp.log(4 * mp.sqrt(2) / varpi) - mp.mpf(1) / 2 - mp.pi / 4)
coef = [Xi / 9, -mp.pi * Xi * (25 * Xi + 648) / 10800, mp.pi ** 2 * Xi ** 2 * (1225 * Xi + 12879) / 1905120]


def trunc(n, terms):
    v = 8 / mp.pi * n ** 2 * mp.log(n)
    if terms >= 2:
        v += C2 * n ** 2
    for j in range(3, terms + 1):
        v += coef[j - 3] * mp.mpf(n) ** (2 - 2 * (j - 2))
    return v


Ns = np.array([d[0] for d in data], dtype=float)
fig, ax = plt.subplots(figsize=(5.8, 3.7))
labels = ["1 term: $(8/\\pi)N^2\\ln N$", "2 terms: $+\\,C_2N^2$", "3 terms: $+\\,C_0$",
          "4 terms: $+\\,C_{-2}N^{-2}$", "5 terms: $+\\,C_{-4}N^{-4}$"]
table = {}
for t in range(1, 6):
    err = np.array([float(abs(trunc(n, t) - T) / T) for n, T in data])
    table[t] = err
    ok = err > 1e-33                               # below: precision floor of the stored values
    ax.loglog(Ns[ok], err[ok], "-o", color=C[t - 1], label=labels[t - 1], markersize=3.2)
ax.text(Ns[-1] * 1.25, table[1][-1] * 4.5, "1 term", fontsize=7.5, color=INK2, va="center")
ax.text(Ns[-1] * 1.25, table[2][-1], "2 terms", fontsize=7.5, color=INK2, va="center")
ax.text(Ns[-1] * 1.25, table[3][-1], "3 terms", fontsize=7.5, color=INK2, va="center")
ax.set_xlim(1.6, 1.5e7)
ax.set_ylim(1e-33, 3)
ax.set_xlabel("lattice side $N$")
ax.set_ylabel("relative error")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), frameon=False, ncol=2)
save(fig, "s4_mfpt_asymptotics")
with open(_os.path.join(_R, 'data', 'article', 's4_mfpt_truncation_errors.csv'), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["N", "rel_err_1term", "rel_err_2terms", "rel_err_3terms", "rel_err_4terms", "rel_err_5terms"])
    for i, (n, T) in enumerate(data):
        w.writerow([n] + [f"{table[t][i]:.6e}" for t in range(1, 6)])
# consistency with the research table (same input, same formulas)
ref = {int(r["N"]): r for r in csv.DictReader(open(_os.path.join(_R, 'data', 'msc_proof', 'asymptotic_truncation_errors.csv')))}
worst = 0.0
for i, (n, T) in enumerate(data):
    for t, key in zip(range(1, 5), ["rel_err_1term", "rel_err_2terms", "rel_err_3terms", "rel_err_4terms"]):
        a, b = table[t][i], float(ref[n][key])
        if b > 1e-30:
            worst = max(worst, abs(a - b) / b)
print("truncation table vs research table, worst relative difference:", worst)
assert worst < 1e-5


# ------------------------------------------------------------------ fig-landscape (zero-based)
def transition_matrix(N, q):
    """Lazy walk on {0..N-1}^2 built from the rules: with probability q pick one of the four
    directions uniformly; a move that would leave the lattice is cancelled."""
    idx = lambda x, y: x * N + y
    P = sps.lil_matrix((N * N, N * N))
    for x in range(N):
        for y in range(N):
            i = idx(x, y)
            P[i, i] += 1 - q
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                u, v = x + dx, y + dy
                if 0 <= u < N and 0 <= v < N:
                    P[i, idx(u, v)] += q / 4
                else:
                    P[i, i] += q / 4
    return P.tocsr()


def landscape(N, q, a):
    """T_{o->a} for every start o by the single sum of Theorem thm-pair (zero-based, float64,
    overflow-free:  cosh((N-m)phi)/sinh(N phi) = (e^{-m phi} + e^{-(2N-m) phi})/(1 - e^{-2N phi}) )."""
    a1, a2 = a
    k = np.arange(1, N)[:, None]
    n = np.arange(0, N)[None, :]
    phi = 2 * np.arcsinh(np.sin(np.pi * k / (2 * N)))
    cth = np.cos(np.pi * k * (2 * n + 1) / (2 * N))                 # cos theta_k(m), theta_k(m) = pi k (2m+1)/(2N)
    Ck = cth * cth[:, [a1]]                                          # C_k(o1, a1)

    def Psi(m, mp_):
        m1 = np.abs(m - mp_); m2 = m + mp_ + 1
        return (np.exp(-m1 * phi) + np.exp(-(2 * N - m1) * phi) + np.exp(-m2 * phi) + np.exp(-(2 * N - m2) * phi)) \
            / ((1 - np.exp(-2 * N * phi)) * np.sinh(phi))
    Pk = Psi(n, a2)                                                  # Psi_k(o2, a2)
    const = float(np.sum(cth[:, a1] ** 2 * Psi(a2, a2)[:, 0]))
    o2 = np.arange(0, N)
    h1 = np.where(o2 <= a2, (a2 - o2) * (a2 + o2 + 1), (o2 - a2) * (2 * N - 1 - o2 - a2)).astype(float)
    return (2 * h1[None, :] + 4 * N * (const - Ck.T @ Pk)) / q       # [o1, o2]


def direct(N, q, a):
    P = transition_matrix(N, q)
    ia = a[0] * N + a[1]
    keep = np.array([i for i in range(N * N) if i != ia])
    A = (sps.identity(len(keep), format="csc") - P[keep][:, keep]).tocsc()
    m = np.zeros(N * N); m[keep] = spla.spsolve(A, np.ones(len(keep)))
    return m.reshape(N, N)


N0, q0 = 35, 0.8
targets = [(N0 - 1, N0 - 1), (8, 22)]                                # zero-based: corner and an interior site
maps, chk = [], dict(N=N0, q=q0, indexing="zero-based", targets=[])
for a in targets:
    Tm = landscape(N0, q0, a); Td = direct(N0, q0, a)
    rel = np.abs(Tm - Td)[Td > 0] / Td[Td > 0]
    chk["targets"].append(dict(target=list(a), max_rel_err_vs_direct=float(rel.max()),
                               T_from_corner_00=float(Tm[0, 0]), T_max=float(Tm.max()),
                               value_at_target=float(Tm[a])))
    maps.append(Tm)
json.dump(chk, open(_os.path.join(_R, 'data', 'article', 's4_mfpt_landscape_check.json'), "w"), indent=1)
print(json.dumps(chk, indent=1))

fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), constrained_layout=True)
for ax, Tm, tgt, ttl in ((axes[0], maps[0], targets[0], f"target at the corner $({N0 - 1},{N0 - 1})$"),
                         (axes[1], maps[1], targets[1], f"target at the interior site $({targets[1][0]},{targets[1][1]})$")):
    im = ax.imshow(Tm.T, origin="lower", cmap="Blues", extent=(-0.5, N0 - 0.5, -0.5, N0 - 0.5), vmin=0)
    ax.grid(False)
    ax.plot([tgt[0]], [tgt[1]], marker="s", color=C[1], markersize=5, markeredgecolor="white", markeredgewidth=0.8)
    ax.annotate("target", tgt, xytext=(-28, -14) if tgt[0] > 20 else (8, 6), textcoords="offset points",
                fontsize=7.5, color=INK)
    ax.set_xticks([0, 10, 20, 30]); ax.set_yticks([0, 10, 20, 30])
    ax.set_xlabel("start coordinate $o_1$"); ax.set_ylabel("start coordinate $o_2$")
    ax.set_title(ttl, loc="left")
    cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label(r"$T_{\mathbf{o}\to\mathbf{a}}$ (steps)")
    cb.outline.set_linewidth(0.4)
save(fig, "s4_mfpt_landscape")
