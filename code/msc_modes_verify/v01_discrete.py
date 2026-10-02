#!/usr/bin/env python
"""Check by the second implementation, part 1: exact discrete-time first-passage PMFs.

Drives the C stepper (stepper.c, written for this verification) and post-processes
the full f(t) arrays.  Mode = smallest argmax_t f(t); no smoothing, no fallbacks.
Results are appended to ../data/discrete.jsonl (restartable: finished keys are skipped).
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import json, os, subprocess, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = _os.path.join(_R, 'data', 'msc_modes_verify')
PMF = _os.path.join(_R, 'data', 'msc_modes_verify', 'pmf')
os.makedirs(PMF, exist_ok=True)
OUT = _os.path.join(_R, 'data', 'msc_modes_verify', 'discrete.jsonl')
STEPPER = _os.path.join(_R, 'code', 'msc_modes_verify', 'stepper')

SIZES_1D = [5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 35, 40, 45, 50, 55, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200]
SIZES_2D = SIZES_1D
SIZES_3D = [5, 7, 9, 11, 13, 15, 17, 19, 25, 30, 35, 40, 50, 60, 70, 80, 100]


def done_keys():
    keys = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            r = json.loads(line)
            keys.add(r["key"])
    return keys


def analyse(f):
    """f[0] = f(t=1).  Returns a dict of mode diagnostics."""
    T = len(f)
    i = int(np.argmax(f))               # first maximiser
    fmax = float(f[i])
    ties = int(np.sum(f == fmax))
    # strict local maxima / minima on the support (ignore leading zeros)
    nz = np.nonzero(f > 0)[0]
    lo = nz[0] if len(nz) else 0
    g = f[lo:]
    dif = np.sign(np.diff(g))
    # collapse zero differences (plateaus) by forward filling
    for k in range(1, len(dif)):
        if dif[k] == 0:
            dif[k] = dif[k - 1]
    sc = np.diff(dif)
    n_max = int(np.sum(sc < 0)) + (1 if len(dif) and dif[0] < 0 else 0)
    n_min = int(np.sum(sc > 0))
    # parabolic refinement of the peak (3-point with span h chosen so that curvature is resolved)
    h = max(1, int(round(0.02 * (i + 1))))
    ref = None
    if i - h >= 0 and i + h < T:
        y0, y1, y2 = f[i - h], f[i], f[i + h]
        den = (y0 - 2 * y1 + y2)
        if den != 0:
            ref = (i + 1) + 0.5 * h * (y0 - y2) / den
    # two-step average
    f2 = 0.5 * (f[:-1] + f[1:])
    i2 = int(np.argmax(f2))
    cum = np.cumsum(f)
    res = dict(T=T, mode=i + 1, fmax=fmax, ties=ties, n_local_max=n_max, n_local_min=n_min,
               mode_parabolic=ref, parab_h=h, mode_two_step_first=i2 + 1,
               cdf_at_mode=float(cum[i]), mass=float(cum[-1]),
               monotone_after_mode=bool(np.all(np.diff(f[i:]) <= 0)),
               f_end_over_fmax=float(f[-1] / fmax))
    if cum[-1] >= 0.5:
        res["median"] = int(np.searchsorted(cum, 0.5) + 1)
    return res, cum


def run(d, N, q, geom, stop, tmin, tmax, keep=False, tag=""):
    key = f"{d}|{N}|{q}|{geom}|{stop}|{tmin}{tag}"
    if key in done_keys():
        return
    fn = os.path.join(PMF, f"f_d{d}_N{N}_q{q}_{geom}_s{stop}_{tmin}.bin")
    t0 = time.time()
    out = subprocess.run([STEPPER, str(d), str(N), repr(q), geom, str(stop), str(tmin), str(tmax), fn],
                         capture_output=True, text=True, check=True).stdout
    meta = json.loads(out)
    f = np.fromfile(fn, dtype=np.float64)
    res, cum = analyse(f)
    res.update(key=key, d=d, N=N, q=q, geom=geom, stop=stop, S_remaining=meta["S_remaining"],
               wall_s=time.time() - t0)
    if stop <= 0 or res["mass"] > 1 - 1e-9:
        t = np.arange(1, len(f) + 1, dtype=np.float64)
        m1 = float(np.sum(t * f)); m2 = float(np.sum(t * t * f))
        res.update(mean_from_pmf=m1, cv=float(np.sqrt(m2 - m1 * m1) / m1),
                   S_at_mean=float(1 - cum[int(np.floor(m1)) - 1]))
    if keep:
        # thinned copy for figures (<= ~4000 points) + exact window round the mode
        stride = max(1, len(f) // 4000)
        np.savez_compressed(fn.replace(".bin", ".npz"), t=np.arange(1, len(f) + 1)[::stride], f=f[::stride],
                            mode=res["mode"], cdf=cum[::stride])
    os.remove(fn)
    with open(OUT, "a") as fh:
        fh.write(json.dumps(res) + "\n")
    print(key, "mode", res["mode"], "T", res["T"], f"{res['wall_s']:.1f}s", flush=True)


if __name__ == "__main__":
    groups = sys.argv[1:] or ["small"]
    BIG = 2_000_000_000
    for grp in groups:
        if grp == "small":       # quick sanity set
            run(1, 5, 0.8, "CC", 3.0, 200, BIG)
            run(2, 35, 0.8, "CC", 1.5, 200, BIG, keep=True)
            run(3, 40, 0.8, "CC", 1.5, 200, BIG, keep=True)
        if grp == "1d":
            for N in SIZES_1D + [300, 400, 600, 800, 1000]:
                run(1, N, 0.8, "CC", 3.0, 200, BIG)
        if grp == "2d":
            for N in SIZES_2D + [241, 301]:
                run(2, N, 0.8, "CC", 1.5, 200, BIG)
        if grp == "2d_big":
            run(2, 401, 0.8, "CC", 1.5, 200, BIG)
        if grp == "3d":
            for N in SIZES_3D[:-3]:
                run(3, N, 0.8, "CC", 1.5, 200, BIG)
        if grp == "3d_big":
            for N in SIZES_3D[-3:]:
                run(3, N, 0.8, "CC", 1.5, 200, BIG)
        if grp == "qscan":
            for q in (0.3, 0.5, 0.9, 1.0):
                for N in (51, 201):
                    run(1, N, q, "CC", 3.0, 200, BIG, keep=(q == 1.0))
                for N in (21, 51, 101):
                    run(2, N, q, "CC", 1.5, 200, BIG)
                for N in (11, 21, 31):
                    run(3, N, q, "CC", 1.5, 200, BIG)
        if grp == "geom":
            for N in (11, 21, 41, 101, 201):
                run(2, N, 0.8, "C2M", 1.5, 200, BIG)
                run(2, N, 0.8, "M2C", 1.5, 200, BIG)
            for N in (11, 21, 41, 61):
                run(3, N, 0.8, "C2M", 1.5, 200, BIG)
                run(3, N, 0.8, "M2C", 1.5, 200, BIG)
        if grp == "exit":
            for N in (51, 101, 201, 401):
                run(1, N, 0.8, "EXIT", 3.0, 200, BIG)
            for N in (21, 51, 101, 201):
                run(2, N, 0.8, "EXIT", 3.0, 200, BIG)
            for N in (11, 21, 31, 51):
                run(3, N, 0.8, "EXIT", 3.0, 200, BIG)
            # exit geometry, q = 0.9
            run(1, 51, 0.9, "EXIT", 3.0, 200, BIG)
            run(2, 51, 0.9, "EXIT", 3.0, 200, BIG)
            run(3, 31, 0.9, "EXIT", 3.0, 200, BIG)
        if grp == "full":       # full support: unimodality, CV, median, S(MFPT)
            for N in (5, 21, 51, 101):
                run(1, N, 0.8, "CC", 0, 10, BIG, keep=(N == 51), tag="full")
            for N in (5, 9, 15, 21, 35):
                run(2, N, 0.8, "CC", 0, 10, BIG, keep=(N == 35), tag="full")
            for N in (5, 9, 15):
                run(3, N, 0.8, "CC", 0, 10, BIG, keep=(N == 15), tag="full")
            for N in (5, 11, 21):
                run(2, N, 0.8, "C2M", 0, 10, BIG, tag="full")
                run(2, N, 0.8, "M2C", 0, 10, BIG, tag="full")
            for N in (5, 11):
                run(3, N, 0.8, "C2M", 0, 10, BIG, tag="full")
                run(3, N, 0.8, "M2C", 0, 10, BIG, tag="full")
            for q in (0.5, 0.9, 1.0):
                run(2, 15, q, "CC", 0, 10, BIG, tag="full")
                run(1, 21, q, "CC", 0, 10, BIG, tag="full")
        if grp == "n40_long":   # 3D N=40 to ~3 MFPT for median and S(MFPT)
            run(3, 40, 0.8, "CC", 1.5, 1_100_000, BIG, keep=True, tag="long")
