"""
02_run_discrete.py -- exact discrete-time first-passage PMFs by time stepping
(and the closed-form chain in 1D); modes are argmax_t f(t) of the exact PMF.
No mean-derived fallbacks, no smoothing, no acceptance windows.

Usage:  python 02_run_discrete.py GROUP [GROUP ...]
Groups: see GROUPS below.  Results are appended to data/discrete_modes.jsonl
(one JSON object per line; existing (d,geo,N,q) keys are skipped, so the
script is restartable).  Subsampled PMFs go to data/pmf/*.npz.
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import json, os, sys, time, math
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import fptlib as F

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = _os.path.join(_R, 'data', 'msc_modes')
OUT = _os.path.join(_R, 'data', 'msc_modes', 'discrete_modes.jsonl')
PMF = _os.path.join(_R, 'data', 'msc_modes', 'pmf')
os.makedirs(PMF, exist_ok=True)

SIZES_1D2D = [5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 35, 40, 45, 50, 55, 60, 70, 80, 90, 100,
               120, 140, 160, 180, 200]
SIZES_3D = [5, 7, 9, 11, 13, 15, 17, 19, 25, 30, 35, 40, 50, 60, 70, 80, 100]
ODD = [5, 7, 9, 11, 15, 21, 29, 35, 45, 61, 81, 101, 141, 201]
ODD3 = [5, 7, 9, 11, 15, 21, 29, 41, 61]

GROUPS = {
    # name: list of (d, geo, N, q, tail_factor)
    "1d": [(1, "CC", N, 0.8, 3.0) for N in SIZES_1D2D + [300, 400, 600, 800, 1000]],
    "2d_cc": [(2, "CC", N, 0.8, 1.5) for N in SIZES_1D2D],
    "2d_cc_big": [(2, "CC", N, 0.8, 1.35) for N in [241, 301, 401]],
    "2d_other": [(2, g, N, 0.8, 1.5) for g in ("C2M", "M2C") for N in ODD],
    "3d_cc": [(3, "CC", N, 0.8, 1.5) for N in SIZES_3D if N <= 60],
    "3d_cc_big": [(3, "CC", N, 0.8, 1.35) for N in [70, 80, 100]],
    "3d_other": [(3, g, N, 0.8, 1.5) for g in ("C2M", "M2C") for N in ODD3],
    "qscan": ([(2, "CC", N, q, 1.5) for N in (21, 51, 101) for q in (0.3, 0.5, 0.9, 1.0)]
              + [(3, "CC", N, q, 1.5) for N in (11, 21, 31) for q in (0.3, 0.5, 0.9, 1.0)]
              + [(1, "CC", N, q, 3.0) for N in (51, 201) for q in (0.3, 0.5, 0.9, 1.0)]),
    "exit": [(d, "EXIT", N, q, 4.0) for q in (0.8, 0.9) for d in (1, 2, 3)
             for N in ([11, 21, 31, 51, 101, 201, 401] if d == 1 else [11, 21, 31, 51, 101, 201] if d == 2
                       else [11, 21, 31, 51])],
}


def load_done():
    done = set()
    if os.path.exists(OUT):
        with open(OUT) as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    done.add((r["d"], r["geo"], r["N"], r["q"]))
    return done


def geometry(d, geo, N):
    if geo == "EXIT":
        # Dirichlet exit problem (exit geometry):
        # start at the centre, every boundary-layer site is absorbing.
        start = F.centre(N, d)
        shape = (N,) * d
        idx = np.indices(shape)
        onb = np.zeros(shape, dtype=bool)
        for ax in range(d):
            onb |= (idx[ax] == 0) | (idx[ax] == N - 1)
        targets = [tuple(int(v) for v in p) for p in np.argwhere(onb)]
        return start, targets
    s, a = F.GEOMETRIES[geo][0](N, d), F.GEOMETRIES[geo][1](N, d)
    return s, [a]


def analyse(f):
    """mode statistics from an exact PMF array f[0..T]"""
    T = f.size - 1
    fmax = float(f.max())
    m = int(np.argmax(f))                       # smallest maximiser
    ties_exact = int((f == fmax).sum())
    near = np.where(f >= fmax * (1 - 1e-12))[0]
    # local maxima (strict rise then fall), ignoring zeros
    df = np.diff(f[1:])
    sgn = np.sign(df)
    up_down = np.where((sgn[:-1] > 0) & (sgn[1:] < 0))[0] + 2
    # refined (real-valued) maximiser: symmetric 3-point parabola with half-width h
    # (h ~ sqrt(m)/4 keeps the skewness bias << 1 step while the second
    #  difference stays far above round-off)
    h = max(1, int(math.sqrt(m) / 4))
    ref = float("nan")
    if m - h >= 1 and m + h <= T:
        den = f[m - h] - 2 * f[m] + f[m + h]
        if den != 0:
            ref = m + 0.5 * h * (f[m - h] - f[m + h]) / den
    # two-step average (parity-smoothed) mode
    fb = 0.5 * (f[1:-1] + f[2:])
    m2 = int(np.argmax(fb)) + 1                 # window {m2, m2+1}
    # half-maximum crossing on the rise
    rise = int(np.argmax(f >= 0.5 * fmax))
    after = f[m:]
    mono_after = bool(np.all(np.diff(after) <= 0))
    return {
        "mode": m, "f_max": fmax, "ties_exact": ties_exact,
        "argmax_band_1e-12": [int(near.min()), int(near.max())],
        "n_local_maxima": int(up_down.size), "local_maxima_first10": [int(x) for x in up_down[:10]],
        "mode_refined": ref, "refine_h": h, "mode_two_step_avg": m2,
        "t_half_rise": rise, "monotone_after_mode": mono_after,
        "f_end_over_fmax": float(f[-1] / fmax),
    }


def exact_mfpt(d, geo, N, q, start, targets):
    if geo == "EXIT":
        if N ** d <= 200000:
            return F.mfpt_linear_solve(N, d, q, start, targets), "sparse linear solve"
        return None, "not computed"
    if d == 1:
        return N * (N - 1) / q, "closed form N(N-1)/q"
    return float(F.LaplaceFP(N, d, q, start, targets[0]).mfpt()), "reflecting-resolvent formula"


def run_one(d, geo, N, q, tail):
    start, targets = geometry(d, geo, N)
    t0 = time.time()

    def stop(t, f, fmax, targmax):
        return fmax > 0.0 and t >= tail * targmax + 64

    f, Send, wall = F.run_pmf(N, d, q, start, targets, stop_rule=stop, log_every=50000)
    rec = {"d": d, "geo": geo, "N": N, "q": q, "start": list(start),
           "target": list(targets[0]) if len(targets) == 1 else f"{len(targets)} boundary sites",
           "t_end": int(f.size - 1), "S_end": Send, "mass_captured": float(f.sum()), "tail_factor": tail}
    rec.update(analyse(f))
    mf, how = exact_mfpt(d, geo, N, q, start, targets)
    rec["mfpt_exact"] = mf
    rec["mfpt_method"] = how
    rec["mode_over_mfpt"] = rec["mode"] / mf if mf else None
    if d == 1 and geo == "CC":
        # independent closed-form check of the argmax
        ch = F.Chain1D(N, q)
        win = np.arange(max(1, rec["mode"] - 5), rec["mode"] + 6)
        fc = ch.f_discrete(win)
        rec["mode_closed_form_check"] = int(win[np.argmax(fc)])
        rec["closed_form_max_abs_diff_window"] = float(np.abs(fc - f[win]).max())
    rec["wall_s"] = time.time() - t0
    # save subsampled PMF
    T = f.size - 1
    step = max(1, T // 6000)
    tt = np.arange(1, T + 1, step)
    m = rec["mode"]
    lo, hi = max(1, m - 300), min(T, m + 300)
    np.savez_compressed(os.path.join(PMF, f"pmf_d{d}_{geo}_N{N}_q{q}.npz"),
                        t=tt, f=f[tt], t_win=np.arange(lo, hi + 1), f_win=f[lo:hi + 1],
                        mode=m, mfpt=mf if mf else np.nan)
    return rec


def main():
    groups = sys.argv[1:]
    done = load_done()
    for g in groups:
        for (d, geo, N, q, tail) in GROUPS[g]:
            if (d, geo, N, q) in done:
                continue
            rec = run_one(d, geo, N, q, tail)
            rec["group"] = g
            with open(OUT, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            done.add((d, geo, N, q))
            print(f"[{g}] d={d} {geo} N={N} q={q}: mode={rec['mode']} refined={rec['mode_refined']:.2f} "
                  f"mfpt={rec['mfpt_exact']} ratio={rec['mode_over_mfpt']} nlocmax={rec['n_local_maxima']} "
                  f"mono={rec['monotone_after_mode']} wall={rec['wall_s']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
