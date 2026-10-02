#!/usr/bin/env python
"""s3_methods_validation.py -- checks behind Section 3 of the article and Supplementary Section S2.

Corner-to-corner first passage of the lazy walk on the reflecting square lattice,
N = 35, q = 0.8 (the baseline case of the article).

  A. Exact PMF by time stepping of the killed chain (route A), to S(t) < 1e-13.
  B. Lemma (resolvent of the one-dimensional reflecting chain): closed form against
     dense inverses, for positive, complex and negative real arguments.
  C. Generating function: single-sum renewal ratio against (i) a direct linear solve
     of (I - zQ) and (ii) the series sum_t f(t) z^t.
  D. Inversion of the generating function at many t:
       trapezoidal rule with a fixed radius r = 0.9 truncated after K = 54 nodes,
       all 2t nodes with the fixed radius r = 0.9,
       Abate-Whitt rule (2t nodes, r = 10^(-gamma/(2t))), gamma = 8, 10, 12;
     plus a scan over every integer t in [300, 700] to locate where the fixed-radius
     rules break down.
  E. Monte Carlo of the literal walker rule (draw U; if U < q pick one of the four
     directions; a move off the lattice is cancelled): 8 independent batches of
     500,000 walkers (seeds spawned from one fixed seed; every batch is reported),
     against the exact PMF: z-score of the sample mean, chi-square on equiprobable
     bins, fraction arrived by the modal time, Kolmogorov-Smirnov distance.
  F. Far-term Taylor acceleration of the Laplace-space sums against brute-force sums
     (2D N = 3000, 3D N = 120).
  H. Product form of the one-dimensional transforms (proposition on the chain)
     against direct solves.
  G. Bookkeeping of the stored research runs that Section 3 summarises (no new
     computation): number of runs, number of local maxima, extent of the grids on
     which the sign of g'(t) was scanned, parity branches at q = 1.

Outputs (paths relative to ):
  data/s3_methods_validation.json      all numbers quoted in Section 3 from this script
  data/s3_methods_mc_times.npz         the Monte Carlo arrival times (int32, batch by batch)
  figures/s3_methods_validation.pdf    Figure (a) PMF vs Monte Carlo, (b) inversion errors

Usage:  python s3_methods_validation.py            (about 4 minutes on 4 cores, < 1 GB)
        python s3_methods_validation.py --reuse-mc (loads the Monte Carlo arrival times from
                                                    data/s3_methods_mc_times.npz instead of re-simulating
                                                    them; the sample is deterministic, so the output is the same)
Deterministic: the batch seeds are SeedSequence(20261001).spawn(8); the results do not
depend on the number of worker processes (environment variable S3_NPROC, default 4).
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ART = _R
ROOT = _R          # 
for cand in (os.environ.get("FPTLIB_DIR", ""),
             _os.path.join(_R, 'code', 'msc_modes'),
             HERE):
    if cand and os.path.exists(os.path.join(cand, "fptlib.py")):
        sys.path.insert(0, cand)
        break
import fptlib as F  # noqa: E402  (code/msc_modes/fptlib.py)

OUT_JSON = _os.path.join(_R, 'data', 'article', 's3_methods_validation.json')
OUT_NPZ = _os.path.join(_R, 'data', 'article', 's3_methods_mc_times.npz')
OUT_FIG = _os.path.join(_R, 'figures', 's3_methods_validation.pdf')

N, D, Q = 35, 2, 0.8
SEED = 20261001
N_BATCH = 8
N_PER_BATCH = 500_000
N_WALKERS = N_BATCH * N_PER_BATCH
N_PROC = int(os.environ.get("S3_NPROC", "4"))
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.1f}s]", *a, flush=True)


# ---------------------------------------------------------------------------
# Monte Carlo worker (module level so that it can be run in worker processes)
# ---------------------------------------------------------------------------
def mc_batch(args):
    """Literal walker rule of the model.  Each step and each walker: draw U uniform on
    [0,1); if U < q choose one of the four directions with probability 1/4 each and move
    unless the move would leave the lattice (then stay); otherwise rest.  The walker is
    removed when it first occupies the target (N-1, N-1).  Returns the arrival times."""
    seed_seq, n = args
    rng = np.random.default_rng(seed_seq)
    dxs = np.array([1, -1, 0, 0], dtype=np.int8)
    dys = np.array([0, 0, 1, -1], dtype=np.int8)
    x = np.zeros(n, dtype=np.int8)
    y = np.zeros(n, dtype=np.int8)
    out, t = [], 0
    while x.size:
        t += 1
        m = x.size
        move = rng.random(m, dtype=np.float32) < Q            # U < q: attempt a move
        k = rng.integers(0, 4, m, dtype=np.uint8)             # one of the 2d = 4 directions
        x += dxs[k] * move
        y += dys[k] * move
        np.clip(x, 0, N - 1, out=x)                           # a move off the lattice is cancelled
        np.clip(y, 0, N - 1, out=y)
        hit = (x == N - 1) & (y == N - 1)
        nh = int(np.count_nonzero(hit))
        if nh:
            out.append(np.full(nh, t, dtype=np.int32))
            keep = ~hit
            x, y = x[keep], y[keep]
    return np.concatenate(out)


def chain_T(n):
    """q = 1 one-dimensional chain: +-1 with probability 1/2, cancelled at the two ends."""
    T = np.zeros((n, n))
    for m in range(n):
        for mp in (m - 1, m + 1):
            if 0 <= mp < n:
                T[m, mp] = 0.5
            else:
                T[m, m] += 0.5
    return T


def G1_plain(sigma, n, n0, Nn, a):
    """formula of the Lemma exactly as printed (no overflow protection)."""
    phi = np.arccosh(1.0 + sigma / a + 0j)
    lo, hi = min(n, n0), max(n, n0)
    return (np.cosh(phi * (lo + 0.5)) * np.cosh(phi * (Nn - 0.5 - hi))
            / ((a / 2.0) * np.sinh(phi) * np.sinh(Nn * phi)))


class BruteFP(F.LaplaceFP):
    KAPPA = 1e300                                          # every term is 'near': plain sums


def main():
    res = {"case": {"N": N, "d": D, "q": Q, "start": [0, 0], "target": [N - 1, N - 1]}}

    # -----------------------------------------------------------------------
    # A. exact PMF by time stepping
    # -----------------------------------------------------------------------
    start, target = F.corner(N, D), F.far_corner(N, D)
    st = F.Stepper(N, D, Q, start, [target])
    f = [0.0]
    while True:
        f.append(st.step())
        if (st.t & 1023) == 0 and st.survival() < 1e-13:
            break
    f = np.array(f)                       # f[t], t = 0..t_end, f[0] = 0
    tt = np.arange(f.size)
    cdf = np.cumsum(f)
    mode = int(np.argmax(f))
    mean = float((tt * f).sum())
    var = float((tt.astype(float) ** 2 * f).sum() - mean ** 2)
    sd = math.sqrt(var)
    mfpt_solve = F.mfpt_linear_solve(N, D, Q, start, [target])
    is_max = (f[1:-1] > f[:-2]) & (f[1:-1] >= f[2:])
    band = np.flatnonzero(f >= 0.99 * f[mode])            # f is unimodal here: a single interval
    res["exact_pmf"] = {
        "t_end": int(f.size - 1), "S_end": st.survival(), "one_minus_sum_f": float(1.0 - f.sum()),
        "mode": mode, "f_mode": float(f[mode]), "mean_from_pmf": mean, "mfpt_linear_solve": mfpt_solve,
        "sd": sd, "cv": sd / mean, "cdf_at_mode": float(cdf[mode]),
        "first_nonzero_t": int(np.flatnonzero(f > 0)[0]),
        "n_local_maxima_full_support": int(is_max.sum()),
        "mode_over_mean": mode / mfpt_solve,
        "band_f_ge_0.99_fmax": [int(band.min()), int(band.max())],
        "band_f_ge_0.99_fmax_over_mode": [float(band.min() / mode), float(band.max() / mode)],
        "band_is_one_interval": bool(band.size == band.max() - band.min() + 1),
    }
    log("A", res["exact_pmf"])

    # -----------------------------------------------------------------------
    # B. Lemma: closed-form resolvent of the 1D reflecting chain vs dense inverse
    # -----------------------------------------------------------------------
    worst_plain, worst_scaled, ncase = 0.0, 0.0, 0
    for Nn in (2, 3, 7, 20, 64):
        T1 = chain_T(Nn)
        eig = 1.0 - np.cos(np.pi * np.arange(Nn) / Nn)          # eigenvalues of I - T
        for a in (0.4, 1.0 / 3.0, 0.5, 1.0):
            sig_list = [0.013, 0.7, 5.0, 0.3 + 0.8j, -0.2 + 1.5j, 2.0 - 0.1j]
            # negative real arguments strictly between consecutive poles -a*eig_k
            for k in range(min(Nn - 1, 3)):
                sig_list.append(-a * (0.35 * eig[k] + 0.65 * eig[k + 1]))
            for sig in sig_list:
                M = np.linalg.inv(sig * np.eye(Nn) + a * (np.eye(Nn) - T1))
                scale = np.abs(M).max()
                for n in range(Nn):
                    for n0 in range(Nn):
                        e1 = abs(G1_plain(sig, n, n0, Nn, a) - M[n, n0]) / scale
                        e2 = abs(F.G1(np.array([sig]), n, n0, Nn, a)[0] - M[n, n0]) / scale
                        worst_plain, worst_scaled = max(worst_plain, e1), max(worst_scaled, e2)
                        ncase += 1
    res["lemma_resolvent"] = {
        "entries_compared": ncase, "max_rel_err_formula_as_printed": float(worst_plain),
        "max_rel_err_overflow_free_form": float(worst_scaled),
        "note": "relative to the largest entry of the dense inverse; N in {2,3,7,20,64}; a in {0.4,1/3,0.5,1}; "
                "sigma positive, complex and negative real between poles"}
    log("B", res["lemma_resolvent"])

    # -----------------------------------------------------------------------
    # C. generating function: single sum vs direct solve vs series
    # -----------------------------------------------------------------------
    L = F.LaplaceFP(N, D, Q, start, target)           # continuous-time walk with jump rate q

    def Ftil(z):
        """PGF of the discrete walk = Laplace transform of the rate-q walk at s = (1 - z)/z."""
        z = np.atleast_1d(np.asarray(z, dtype=complex))
        return L.Fhat((1.0 - z) / z)

    Qm, tidx = F.build_Q(N, D, Q, [target])
    m = Qm.shape[0]
    e0 = np.zeros(m)
    e0[tidx[np.ravel_multi_index(start, (N,) * D)]] = 1.0
    rvec = np.asarray(1.0 - Qm.sum(axis=1)).ravel()   # one-step absorption probabilities
    zs = [0.5, 0.9, 0.99, 0.999, 0.9 * np.exp(0.3j), 0.99 * np.exp(0.05j), 0.995 * np.exp(2.0j), -0.9]
    c_direct, c_series, rel_real = 0.0, 0.0, {}
    for z in zs:
        A = (sp.identity(m, format="csc", dtype=complex) - z * Qm.tocsc().astype(complex))
        direct = z * (e0 @ spla.spsolve(A, rvec.astype(complex)))
        series = (f * z ** tt).sum()
        single = Ftil(z)[0]
        c_direct = max(c_direct, abs(single - direct))
        c_series = max(c_series, abs(single - series))
        if np.isreal(z) and z > 0:                    # positive series: accurate to round-off in relative terms
            rel_real[str(z)] = {"Ftil": float(abs(series)), "rel_diff": float(abs(single - series) / abs(series))}
    res["generating_function"] = {"n_z": len(zs), "max_abs_diff_single_sum_vs_direct_solve": float(c_direct),
                                  "max_abs_diff_single_sum_vs_series": float(c_series),
                                  "positive_real_z_rel_diff_single_sum_vs_series": rel_real,
                                  "Ftil_at_0.9": float(Ftil(0.9)[0].real)}
    log("C", res["generating_function"])

    # -----------------------------------------------------------------------
    # D. inversion rules
    # -----------------------------------------------------------------------
    def aw_correct(t, gamma=10.0):
        """Abate & Whitt (1992): f(t) ~ (1/(2 t r^t)) sum_{k=1}^{2t} (-1)^k Re F~(r e^{i pi k/t}),
        r = 10^(-gamma/(2t)).  Exact value of the sum: sum_{j>=0} f((2j+1)t) 10^(-gamma j)."""
        r = 10.0 ** (-gamma / (2.0 * t))
        k = np.arange(1, 2 * t + 1)
        Fz = Ftil(r * np.exp(1j * np.pi * k / t))
        return float((np.where(k % 2 == 0, 1.0, -1.0) * Fz.real).sum() / (2.0 * t)) * 10.0 ** (gamma / 2.0)

    def aw_fixed_r_all_nodes(t, r=0.9):
        """all 2t nodes, but a fixed radius r = 0.9."""
        k = np.arange(1, 2 * t + 1)
        Fz = Ftil(r * np.exp(1j * np.pi * k / t))
        s = float((np.where(k % 2 == 0, 1.0, -1.0) * Fz.real).sum() / (2.0 * t))
        return s * math.exp(-t * math.log(r))

    def fixed_r_K54(t, r=0.9, K=54):
        """trapezoidal rule with a fixed radius, truncated after K nodes:
        F(t) = r^(-t)/t [ F~(r)/2 + sum_{k=1}^{K} (-1)^k Re F~(r e^{i pi k/t}) ]."""
        k = np.arange(1, K + 1)
        Fz = Ftil(r * np.exp(1j * np.pi * k / t))
        s = 0.5 * Ftil(r)[0].real + float((np.where(k % 2 == 0, 1.0, -1.0) * Fz.real).sum())
        return s / t * math.exp(-t * math.log(r))

    def alias_sum(t, gamma):
        """exact aliasing error of the Abate-Whitt rule: sum_{j>=1} f((2j+1)t) 10^(-gamma j)."""
        tot, j = 0.0, 1
        while (2 * j + 1) * t < f.size:
            tot += f[(2 * j + 1) * t] * 10.0 ** (-gamma * j)
            j += 1
        return tot

    t_grid = sorted(set([int(round(v)) for v in np.geomspace(100, 6000, 70)]
                        + [200, 400, 700, 1000, mode, 4000, 6000]))
    rows = []
    for t in t_grid:
        ex = float(f[t])
        p = fixed_r_K54(t)
        fr = aw_fixed_r_all_nodes(t)
        c8, c10, c12 = aw_correct(t, 8.0), aw_correct(t, 10.0), aw_correct(t, 12.0)
        rows.append({"t": t, "exact": ex, "fixed_r_K54": p, "fixed_r_all_nodes": fr,
                     "correct_gamma8": c8, "correct_gamma10": c10, "correct_gamma12": c12,
                     "rel_fixed_r_K54": abs(p - ex) / ex, "rel_fixed_r": abs(fr - ex) / ex,
                     "rel_correct_gamma8": abs(c8 - ex) / ex, "rel_correct_gamma10": abs(c10 - ex) / ex,
                     "rel_correct_gamma12": abs(c12 - ex) / ex,
                     "alias_rel_gamma10": alias_sum(t, 10.0) / ex})
    by_t = {r_["t"]: r_ for r_ in rows}
    # dense scan over every integer t in [100, 700]
    td = np.arange(100, 701)
    relp = np.array([abs(fixed_r_K54(int(t)) - f[t]) / f[t] for t in td])
    relf = np.array([abs(aw_fixed_r_all_nodes(int(t)) - f[t]) / f[t] for t in td])

    def last_below(rel, thr):
        bad = np.flatnonzero(rel >= thr)
        bad = bad[td[bad] >= 200]
        return int(td[bad[0]] - 1)

    def first_always_above(rel, thr):
        ok = np.flatnonzero(rel < thr)
        return int(td[ok[-1]] + 1)

    sel = (td >= 150) & (td <= 400)
    big = [r_ for r_ in rows if r_["t"] >= 1000]
    res["inversion"] = {"rows": rows, "summary": {
        "fixed_r_K54_at_400": float(by_t[400]["fixed_r_K54"]), "rel_fixed_r_K54_at_400": float(by_t[400]["rel_fixed_r_K54"]),
        "fixed_r_K54_at_1000": float(by_t[1000]["fixed_r_K54"]), "fixed_r_K54_at_mode": float(by_t[mode]["fixed_r_K54"]),
        "exact_at_400": by_t[400]["exact"], "exact_at_1000": by_t[1000]["exact"], "exact_at_mode": by_t[mode]["exact"],
        "fixed_r_all_nodes_at_mode": float(by_t[mode]["fixed_r_all_nodes"]),
        "rel_correct_gamma10_at_mode": by_t[mode]["rel_correct_gamma10"],
        "rel_correct_gamma8_at_mode": by_t[mode]["rel_correct_gamma8"],
        "rel_correct_gamma12_at_mode": by_t[mode]["rel_correct_gamma12"],
        "alias_rel_gamma10_at_mode": by_t[mode]["alias_rel_gamma10"],
        "max_rel_correct_gamma10_t>=1000": max(r_["rel_correct_gamma10"] for r_ in big),
        "max_rel_correct_gamma10_t>=300": max(r_["rel_correct_gamma10"] for r_ in rows if r_["t"] >= 300),
        "max_rel_alias_minus_observed_gamma10_t<=1000": max(
            abs(r_["rel_correct_gamma10"] - r_["alias_rel_gamma10"]) / r_["alias_rel_gamma10"]
            for r_ in rows if r_["t"] <= 1000),
        "dense_scan_range": [100, 700],
        "fixed_r_K54_max_rel_150<=t<=400": float(relp[sel].max()),
        "fixed_r_K54_last_t_before_rel_exceeds_1e-2": last_below(relp, 1e-2),
        "fixed_r_K54_rel_above_1_for_all_t_from": first_always_above(relp, 1.0),
        "fixed_r_all_nodes_max_rel_150<=t<=400": float(relf[sel].max()),
        "fixed_r_all_nodes_last_t_before_rel_exceeds_1e-2": last_below(relf, 1e-2),
        "fixed_r_all_nodes_rel_above_1_for_all_t_from": first_always_above(relf, 1.0),
        "r_pow_t_at_1000": 0.9 ** 1000, "log10_r_pow_minus_t_at_mode": -mode * math.log10(0.9),
    }}
    log("D", res["inversion"]["summary"])

    # -----------------------------------------------------------------------
    # E. Monte Carlo of the literal walker rule: 8 independent batches
    # -----------------------------------------------------------------------
    tm0 = time.time()
    seeds = np.random.SeedSequence(SEED).spawn(N_BATCH)
    if "--reuse-mc" in sys.argv and os.path.exists(OUT_NPZ):
        z = np.load(OUT_NPZ)
        assert int(z["seed"]) == SEED and int(z["N"]) == N and float(z["q"]) == Q
        batches = [z["times"][z["batch"] == b] for b in range(N_BATCH)]
        assert all(b.size == N_PER_BATCH for b in batches)
        log("E  Monte Carlo arrival times loaded from", OUT_NPZ)
    else:
        with ProcessPoolExecutor(max_workers=N_PROC) as ex:
            batches = list(ex.map(mc_batch, [(s_, N_PER_BATCH) for s_ in seeds]))
    log(f"E  Monte Carlo done: {N_BATCH} x {N_PER_BATCH} walkers, {time.time() - tm0:.1f}s")
    times = np.concatenate(batches)
    assert times.size == N_WALKERS
    if "--reuse-mc" not in sys.argv:
        np.savez_compressed(OUT_NPZ, times=times, batch=np.repeat(np.arange(N_BATCH), N_PER_BATCH).astype(np.int8),
                            seed=SEED, N=N, q=Q)

    NB = 100                                               # (nearly) equiprobable bins with integer edges
    edges = np.array([0] + [int(np.searchsorted(cdf, j / NB)) for j in range(1, NB)])
    p_bin = np.diff(np.concatenate([cdf[edges], [1.0]]))
    hist_edges = np.concatenate([edges + 0.5, [np.inf]])
    cdf_ext = np.concatenate([cdf, np.ones(max(0, int(times.max()) + 2 - cdf.size))])
    p_mode = float(cdf[mode])

    def stats_of(tm):
        n = tm.size
        obs = np.histogram(tm, bins=hist_edges)[0]
        expct = n * p_bin
        chi2 = float(((obs - expct) ** 2 / expct).sum())
        frac = float((tm <= mode).mean())
        ts = np.sort(tm)
        ecdf_hi = np.arange(1, n + 1) / n
        ks = float(max(np.abs(ecdf_hi - cdf_ext[ts]).max(), np.abs(ecdf_hi - 1.0 / n - cdf_ext[ts - 1]).max()))
        return {"n": int(n), "mean": float(tm.mean()), "sd": float(tm.std(ddof=1)),
                "z_mean": float((tm.mean() - mfpt_solve) / (sd / math.sqrt(n))),
                "chi2": chi2, "chi2_dof": NB - 1, "chi2_p": float(stats.chi2.sf(chi2, NB - 1)),
                "chi2_min_expected": float(expct.min()),
                "frac_arrived_by_mode": frac,
                "z_frac_by_mode": float((frac - p_mode) / math.sqrt(p_mode * (1 - p_mode) / n)),
                "ks_distance": ks, "ks_sqrt_n_D": ks * math.sqrt(n),
                "ks_p_conservative": float(stats.kstwobign.sf(ks * math.sqrt(n))),
                "min_time": int(tm.min()), "max_time": int(tm.max())}

    pooled = stats_of(times)
    per_batch = [stats_of(b) for b in batches]
    chi2_sum = float(sum(b["chi2"] for b in per_batch))
    # histogram for the figure: bins of width 200 steps up to t = 16000
    BW = 200
    hb = np.arange(0, 16000 + BW, BW)
    hc = np.histogram(times, bins=hb + 0.5)[0]
    h_exact = np.array([f[lo + 1:hi + 1].sum() for lo, hi in zip(hb[:-1], hb[1:])]) / BW
    h_mc = hc / (N_WALKERS * BW)
    h_se = np.sqrt(hc) / (N_WALKERS * BW)
    pull = (hc - N_WALKERS * BW * h_exact) / np.sqrt(N_WALKERS * BW * h_exact * (1 - BW * h_exact))
    res["monte_carlo"] = {
        "n_walkers": N_WALKERS, "n_batches": N_BATCH, "n_per_batch": N_PER_BATCH, "seed": SEED,
        "rule": "U<q then uniform direction among 4; off-lattice move cancelled",
        "exact_mean": mfpt_solve, "exact_sd": sd, "exact_cdf_at_mode": p_mode,
        "pooled": pooled, "per_batch": per_batch,
        "per_batch_z_mean": [b["z_mean"] for b in per_batch],
        "per_batch_z_frac_by_mode": [b["z_frac_by_mode"] for b in per_batch],
        "per_batch_chi2": [b["chi2"] for b in per_batch],
        "sum_of_batch_chi2": chi2_sum, "sum_of_batch_chi2_dof": N_BATCH * (NB - 1),
        "sum_of_batch_chi2_p": float(stats.chi2.sf(chi2_sum, N_BATCH * (NB - 1))),
        "hist_bin_width": BW, "hist_n_bins": int(hc.size), "hist_max_abs_pull": float(np.abs(pull).max()),
        "hist_n_abs_pull_gt_2": int((np.abs(pull) > 2).sum()),
        "hist_edges": hb.tolist(), "hist_counts": hc.tolist(),
    }
    log("E pooled", pooled)
    log("E per-batch z(mean)", np.round(res["monte_carlo"]["per_batch_z_mean"], 2).tolist())
    log("E per-batch z(frac by mode)", np.round(res["monte_carlo"]["per_batch_z_frac_by_mode"], 2).tolist())
    log("E per-batch chi2", np.round(res["monte_carlo"]["per_batch_chi2"], 1).tolist(),
        "sum", round(chi2_sum, 1), "p", res["monte_carlo"]["sum_of_batch_chi2_p"])

    # -----------------------------------------------------------------------
    # F. far-term Taylor acceleration vs brute-force sums
    # -----------------------------------------------------------------------
    far = []
    for (NN, dd) in ((3000, 2), (120, 3)):
        s0, a0 = F.corner(NN, dd), F.far_corner(NN, dd)
        La, Lb = F.LaplaceFP(NN, dd, 1.0, s0, a0), BruteFP(NN, dd, 1.0, s0, a0)
        mu1 = (2.0 / dd) * math.sin(math.pi / (2 * NN)) ** 2
        worst, nfar = 0.0, []
        for tfac in (0.5, 2.0, 8.0):                           # Talbot nodes (M = 20) for t = tfac / mu1
            tq = tfac / mu1
            s_unit, _ = F.talbot_nodes(20)
            r = 2.0 * 20 / (5.0 * tq)
            S = np.concatenate([[r + 0j], r * s_unit])
            va, vb = La.Fhat(S), Lb.Fhat(S)
            worst = max(worst, float(np.abs(va / vb - 1.0).max()))
            nfar.append(int(La.shift.size - La.n_near))
        far.append({"N": NN, "d": dd, "max_rel_diff_Fhat": worst, "n_terms": int(La.shift.size),
                    "n_far_terms": nfar})
    res["far_term_acceleration"] = far
    log("F", far)

    # -----------------------------------------------------------------------
    # H. product form of the one-dimensional transforms (Proposition on the chain)
    # -----------------------------------------------------------------------
    worst_eig, worst_prod = 0.0, 0.0
    for (N1, q1) in ((2, 0.8), (5, 0.8), (9, 0.5), (12, 1.0), (40, 0.8), (40, 0.3)):
        n1 = N1 - 1
        Q1 = np.zeros((n1, n1))
        for m1 in range(n1):
            Q1[m1, m1] = 1.0 - q1 + (q1 / 2.0 if m1 == 0 else 0.0)
            if m1 + 1 < n1:
                Q1[m1, m1 + 1] = Q1[m1 + 1, m1] = q1 / 2.0
        om = (2 * np.arange(1, N1) - 1) * np.pi / (2 * N1 - 1)
        lam = 1.0 - q1 + q1 * np.cos(om)
        worst_eig = max(worst_eig, float(np.abs(np.sort(lam) - np.linalg.eigvalsh(Q1)).max()))
        r1 = np.zeros(n1)
        r1[-1] = q1 / 2.0
        for z in (0.3, 0.9, 0.99, 0.5 + 0.4j, -0.7):
            direct = z * np.linalg.solve(np.eye(n1) - z * Q1, r1.astype(complex))[0]
            prod_z = np.prod((1.0 - lam) * z / (1.0 - lam * z))
            s_ = (1.0 - z) / (q1 * z)
            prod_s = np.prod((1.0 - np.cos(om)) / (s_ + 1.0 - np.cos(om)))
            worst_prod = max(worst_prod, abs(direct - prod_z) / abs(direct), abs(direct - prod_s) / abs(direct))
    res["prop_1d_product"] = {"cases": "N in {2,5,9,12,40}, q in {0.3,0.5,0.8,1}, five values of z each",
                              "max_abs_err_eigenvalues": worst_eig, "max_rel_err_products_vs_direct_solve": float(worst_prod)}
    log("H", res["prop_1d_product"])

    # -----------------------------------------------------------------------
    # G. bookkeeping of the stored research runs summarised in Section 3
    # -----------------------------------------------------------------------
    rdata = None
    for cand in (_os.path.join(_R, 'data', 'msc_modes'),):
        if os.path.exists(os.path.join(cand, "laplace_modes.jsonl")):
            rdata = cand
            break
    if rdata is None:
        log("G skipped: data/msc_modes not found")
    else:
        lm = [json.loads(l_) for l_ in open(os.path.join(rdata, "laplace_modes.jsonl")) if l_.strip()]
        tal = [r_ for r_ in lm if "grid_tmax" in r_]                 # d = 2, 3 (Talbot route)
        ratio_T = np.array([r_["grid_tmax"] / r_["mfpt"] for r_ in tal])
        ratio_m = np.array([r_["grid_tmax"] / r_["mode"] for r_ in tal])
        lo_m = np.array([r_["grid_tmin"] / r_["mode"] for r_ in tal])
        worst = min(tal, key=lambda r_: r_["grid_tmax"] / r_["mfpt"])
        g_scan = {
            "n_cases_total": len(lm), "n_cases_1d_closed_form": len(lm) - len(tal), "n_cases_d2_d3_talbot": len(tal),
            "n_one_sign_change_down_and_none_up": int(sum(1 for r_ in lm if r_["n_sign_changes_down"] == 1
                                                          and r_["n_sign_changes_up"] == 0)),
            "d2_d3_grid_end_over_T_min": float(ratio_T.min()), "d2_d3_grid_end_over_T_max": float(ratio_T.max()),
            "d2_d3_n_grid_end_ge_14T": int((ratio_T >= 14.0).sum()),
            "d2_d3_n_grid_capped_at_15T": int((ratio_T >= 15.0 / 2 ** 0.25).sum()),   # last grid point within one step of 15 T
            "d2_d3_n_grid_end_ge_1T": int((ratio_T >= 1.0).sum()),
            "d2_d3_grid_end_over_mode_min": float(ratio_m.min()), "d2_d3_grid_end_over_mode_max": float(ratio_m.max()),
            "d2_d3_grid_start_over_mode_min": float(lo_m.min()), "d2_d3_grid_start_over_mode_max": float(lo_m.max()),
            "d2_d3_smallest_grid_end_case": {"d": worst["d"], "geo": worst["geo"], "N": worst["N"]},
            "cc_grid_end_over_T": {f"{r_['d']}d_N{r_['N']}": float(r_["grid_tmax"] / r_["mfpt"]) for r_ in tal
                                   if r_["geo"] == "CC" and r_["N"] in (35, 100, 4096, 16777216)},
            "max_bracket_rel_width": float(max(r_["bracket_rel_width"] for r_ in tal)),
            "max_N_2d": max(r_["N"] for r_ in tal if r_["d"] == 2), "max_N_3d": max(r_["N"] for r_ in tal if r_["d"] == 3),
        }
        dm = [json.loads(l_) for l_ in open(os.path.join(rdata, "discrete_modes.jsonl")) if l_.strip()]
        pt = [r_ for r_ in dm if r_["geo"] != "EXIT" and r_["q"] <= 0.9]
        ptd = [r_ for r_ in pt if r_["d"] >= 2]
        g_disc = {
            "n_runs": len(dm), "n_exact_ties": int(sum(1 for r_ in dm if r_["ties_exact"] != 1)),
            "n_point_target_q_le_0.9": len(pt),
            "n_point_target_q_le_0.9_one_local_max": int(sum(1 for r_ in pt if r_["n_local_maxima"] == 1)),
            "n_point_target_q_le_0.9_monotone_after_mode": int(sum(1 for r_ in pt if r_["monotone_after_mode"])),
            "n_point_target_q_le_0.9_d_ge_2": len(ptd),
            "tail_factors_d_ge_2": sorted(set(r_["tail_factor"] for r_ in ptd)),
            "tail_factors_1d_and_exit": sorted(set(r_["tail_factor"] for r_ in dm if r_["d"] == 1 or r_["geo"] == "EXIT")),
            "q1_cases": [{"d": r_["d"], "N": r_["N"], "argmax": r_["mode"], "two_step": r_["mode_two_step_avg"],
                          "n_local_maxima": r_["n_local_maxima"], "first_local_maxima": r_["local_maxima_first10"][:4]}
                         for r_ in dm if r_["q"] == 1.0],
            "largest_cc": {str(d_): {k_: max((r_ for r_ in dm if r_["d"] == d_ and r_["geo"] == "CC"),
                                             key=lambda r_: r_["N"])[k_] for k_ in ("N", "t_end", "mode")}
                           for d_ in (1, 2, 3)},
        }
        ft = json.load(open(os.path.join(rdata, "full_tail.json")))
        g_full = {"n_runs": len(ft), "n_runs_d_ge_2": int(sum(1 for r_ in ft if r_["d"] >= 2)),
                  "n_one_max_no_min": int(sum(1 for r_ in ft if r_["n_local_maxima_full_support"] == 1
                                              and r_["n_local_minima_full_support"] == 0)),
                  "q_values": sorted(set(r_["q"] for r_ in ft)), "max_S_end": float(max(r_["S_end"] for r_ in ft)),
                  "max_N_by_d": {str(d_): max(r_["N"] for r_ in ft if r_["d"] == d_) for d_ in (1, 2, 3)}}
        g_par = {}
        pz = os.path.join(rdata, "pmf", "pmf_d1_CC_N51_q1.0.npz")
        if os.path.exists(pz):
            zz = np.load(pz)
            t1, f1 = zz["t"], zz["f"]
            ev, od = f1[t1 % 2 == 0].max(), f1[t1 % 2 == 1].max()
            g_par = {"N": 51, "max_even_branch": float(ev), "max_odd_branch": float(od),
                     "relative_gap_of_branch_maxima": float(ev / od - 1.0)}
        fr = json.load(open(os.path.join(rdata, "fit_results.json")))
        offs = fr["discrete_vs_continuous_offsets"]
        g_off = {"n_cases": len(offs), "min_offset_steps": float(min(o["offset_steps"] for o in offs)),
                 "max_offset_steps": float(max(o["offset_steps"] for o in offs)),
                 "max_rel_offset_N_ge_100": float(max(abs(o["offset_steps"]) / o["mode_discrete"]
                                                      for o in offs if o["N"] >= 100)),
                 "refined_offset_cc": {str(d_): [[o["N"], round(o["offset_refined"], 3)] for o in
                                                 sorted((o for o in offs if o["d"] == d_ and o["geo"] == "CC"),
                                                        key=lambda o: o["N"])] for d_ in (1, 2, 3)}}
        res["stored_runs_summary"] = {"continuous_time_sign_scan": g_scan, "discrete_runs": g_disc,
                                      "full_support_runs": g_full, "parity_q1_1d": g_par,
                                      "discrete_vs_continuous_offset_q0.8": g_off,
                                      "sources": "data/msc_modes/{laplace_modes.jsonl, discrete_modes.jsonl, "
                                                 "full_tail.json, fit_results.json, pmf/pmf_d1_CC_N51_q1.0.npz}"}
        log("G", json.dumps(res["stored_runs_summary"])[:3000])

    res["wall_seconds"] = time.time() - T0
    with open(OUT_JSON, "w") as fh:
        json.dump(res, fh, indent=1)
    log("wrote", OUT_JSON)

    # -----------------------------------------------------------------------
    # figure
    # -----------------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker  # noqa: F401

    INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d2"
    C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"                # blue, orange, aqua
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 0.6,
                         "pdf.fonttype": 42, "font.family": "DejaVu Sans"})
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.2, 3.1))

    # (a) exact PMF and Monte Carlo histogram
    tp = np.arange(1, 16001)
    ax.plot(tp, f[1:16001] * 1e5, color=C1, lw=1.6, label="exact PMF (time stepping)", zorder=3)
    mid = 0.5 * (hb[:-1] + hb[1:]) + 0.5
    ax.errorbar(mid, h_mc * 1e5, yerr=2 * h_se * 1e5, fmt="o", ms=2.6, mfc="white", mec=INK, mew=0.6,
                ecolor=MUTED, elinewidth=0.6, capsize=0, zorder=4,
                label=r"Monte Carlo, $4\times10^{6}$ walkers ($\pm2$ s.e.)")
    ax.axvline(mode, color=MUTED, lw=0.8, ls=(0, (4, 2)))
    ax.axvline(mfpt_solve, color=MUTED, lw=0.8, ls=(0, (1, 1.5)))
    ax.text(mode + 250, 7.55, rf"mode $t^{{*}}={mode}$", color=INK, fontsize=8, ha="left", va="bottom")
    ax.text(mfpt_solve - 250, 7.55, rf"mean $T={mfpt_solve:.1f}$", color=INK, fontsize=8, ha="right", va="bottom")
    ax.set_xlim(0, 16500)
    ax.set_ylim(0, 8.2)
    ax.set_xlabel("time $t$ (steps)")
    ax.set_ylabel(r"$f(t)\times10^{5}$")
    ax.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", bbox_to_anchor=(0.84, 0.03), frameon=False, fontsize=7.5, handlelength=1.8)
    ax.set_title("(a)", loc="left", fontsize=9)

    # (b) relative error of the inversion rules (the two fixed-radius curves leave the frame near t = 700)
    pr = [r_ for r_ in rows if r_["t"] >= 110]
    tg = np.array([r_["t"] for r_ in pr])
    bx.plot(tg, [r_["rel_fixed_r_K54"] for r_ in pr], color=C2, lw=1.6, marker="s", ms=2.8,
            label=r"fixed $r=0.9$, $K=54$ nodes")
    bx.plot(tg, [r_["rel_fixed_r"] for r_ in pr], color=C3, lw=1.6, ls=(0, (4, 1.5)), marker="^", ms=2.8,
            label=r"$2t$ nodes, $r=0.9$")
    bx.plot(tg, [r_["rel_correct_gamma10"] for r_ in pr], color=C1, lw=1.6, marker="o", ms=2.8,
            label=r"$r=10^{-\gamma/2t}$, $\gamma=10$")
    bx.axhline(1.0, color=MUTED, lw=0.6)
    bx.axvline(mode, color=MUTED, lw=0.8, ls=(0, (4, 2)))
    bx.text(mode * 0.94, 2e-13, r"$t^{*}$", color=INK, fontsize=8, ha="right", va="bottom")
    bx.set_xscale("log")
    bx.set_yscale("log")
    bx.set_xlim(105, 6500)
    bx.set_ylim(1e-13, 1e12)
    bx.set_yticks([1e-12, 1e-8, 1e-4, 1.0, 1e4, 1e8, 1e12])
    bx.set_xticks([200, 500, 1000, 2000, 5000])
    bx.set_xticklabels(["200", "500", "1000", "2000", "5000"])
    bx.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    bx.set_xlabel("time $t$ (steps)")
    bx.set_ylabel("relative error of the inverted $f(t)$")
    bx.grid(True, which="major", color=GRID, lw=0.5)
    bx.set_axisbelow(True)
    leg = bx.legend(loc="center right", bbox_to_anchor=(1.02, 0.37), frameon=True, fontsize=7.5, handlelength=2.2,
                    borderaxespad=0.2, facecolor="white", edgecolor="none", framealpha=1.0)
    leg.set_zorder(5)
    bx.set_title("(b)", loc="left", fontsize=9)
    for a_ in (ax, bx):
        for sp_ in ("top", "right"):
            a_.spines[sp_].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_FIG)
    log("wrote", OUT_FIG)


if __name__ == "__main__":
    main()
