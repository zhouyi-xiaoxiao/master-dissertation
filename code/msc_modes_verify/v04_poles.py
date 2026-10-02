#!/usr/bin/env python
"""Check by the second implementation, part 4: continuous-time modes at large N from the pole expansion.

For corner-to-corner in d = 2, 3 the first-passage density is g(t) = sum_j a_j exp(-nu_j t), where
-nu_j are the zeros of P^(a,s|a) on the negative real axis and a_j = P^(a,-nu_j|x0)/dP^(a,s|a)/ds.
This script finds all poles with nu_j < NUMAX*mu1 by bracketed root finding on the resolvent sum
(single sum in 2D, double sum in 3D; one axis summed in closed form), and maximises the
truncated exponential sum.  This is a different inversion route from the Talbot rule used by the
main implementation.  Output: ../data/poles.jsonl (restartable).
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import json, os, sys, time
import numpy as np
from scipy.optimize import brentq
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vspec

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = _os.path.join(_R, 'data', 'msc_modes_verify', 'poles.jsonl')

N2 = [5, 7, 10, 14, 20, 28, 35, 40, 50, 57, 80, 100, 113, 160, 200, 226, 320, 400, 401, 453, 640, 905, 1280, 1810,
      2560, 3620, 4001, 5120, 7241, 10240, 20480, 40960, 81920, 163840, 327680, 655360, 1048576, 2097152,
      4194304, 8388608, 16777216]
N3 = [5, 7, 10, 14, 20, 28, 35, 40, 50, 57, 80, 100, 113, 160, 200, 226, 320, 453, 640, 905, 1280, 1810, 2560, 4096]


def done():
    s = set()
    if os.path.exists(OUT):
        for l in open(OUT):
            r = json.loads(l); s.add(r["key"])
    return s


def run(d, N, numax=14.0, far=None, tag=""):
    key = f"{d}|{N}|{numax}|{far}{tag}"
    if key in done():
        return
    t0 = time.time()
    C = vspec.Corner(d, N, far_factor=far)
    mfpt = C.mfpt()
    nus, res = C.poles(numax_units=numax)
    mu1 = C.mu1
    X = mu1 * mfpt
    W = 2 * d * np.cos(np.pi / (2 * N)) ** 2
    out = dict(key=key, d=d, N=N, numax=numax, far=far, mfpt=mfpt, mu1=mu1, X=X, n_poles=len(nus),
               nu=[float(x) for x in nus], res=[float(x) for x in res])
    # mode with all poles and with the first 2, 3 poles
    t_all, g_all = vspec.density_mode(nus, res)
    out["mode"] = t_all; out["gmax"] = g_all
    for J in (2, 3, 4, 6):
        if len(nus) >= J:
            tJ, _ = vspec.density_mode(nus[:J], res[:J], 0.5 * t_all, 2.0 * t_all)
            out[f"mode_{J}pole"] = tJ
    # closed forms under review
    out["L0"] = float(np.log(W * X) / mu1)
    out["L1"] = float(mfpt * np.log(W * X) / (X + W - 1.0))
    out["L1_simple"] = float(mfpt * np.log(2 * d * X) / (X + 2 * d - 1.0))      # form quoted in the summary
    # discrete-time mode at q = 0.8 from the same poles
    q = 0.8
    best, treal = vspec.discrete_mode_from_poles(nus, res, q, t_all / q)
    out["mode_disc_q0.8"] = best; out["mode_disc_real_q0.8"] = treal
    # shape statistics (need only t >= ~mode, where the truncated sum is converged)
    cdf = lambda t: 1.0 - np.sum(res / nus * np.exp(-nus * t))
    out["cdf_at_mode"] = float(cdf(t_all))
    out["S_at_mfpt"] = float(1.0 - cdf(mfpt))
    out["median"] = float(brentq(lambda t: cdf(t) - 0.5, t_all, 3 * mfpt))
    g = lambda t: np.sum(res * np.exp(-nus * t))
    out["band99_hi"] = float(brentq(lambda t: g(t) - 0.99 * g_all, t_all, 50 * mfpt) / t_all)
    try:
        out["band99_lo"] = float(brentq(lambda t: g(t) - 0.99 * g_all, 0.3 * t_all, t_all) / t_all)
    except Exception:
        out["band99_lo"] = None
    out["sum_res_over_nu"] = float(np.sum(res / nus))
    out["wall_s"] = time.time() - t0
    with open(OUT, "a") as fh:
        fh.write(json.dumps(out) + "\n")
    print(key, f"mode/N^2={t_all / N**2:.6f} mode/mfpt={t_all / mfpt:.6f} npoles={len(nus)} {out['wall_s']:.1f}s", flush=True)


if __name__ == "__main__":
    groups = sys.argv[1:] or ["2d", "3d"]
    for grp in groups:
        if grp == "2d":
            for N in N2:
                run(2, N, far=None if N <= 50000 else 400.0)
        if grp == "3d":
            for N in N3:
                run(3, N, numax=11.0, far=None if N <= 400 else 400.0)
        if grp == "farcheck":       # same N with and without the far-field Chebyshev split
            run(2, 4001, far=400.0, tag="farcheck")
            run(2, 40960, far=400.0, tag="farcheck")
            run(3, 160, numax=11.0, far=400.0, tag="farcheck")
            run(3, 320, numax=11.0, far=400.0, tag="farcheck")
        if grp == "conv":           # convergence in the number of poles (band edges need more poles)
            run(2, 35, numax=40.0, tag="conv")
            run(2, 640, numax=40.0, tag="conv")
            run(3, 40, numax=30.0, tag="conv")
            run(3, 640, numax=30.0, far=400.0, tag="conv")
