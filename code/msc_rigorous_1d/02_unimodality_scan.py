"""Exploration (numerical observation, NOT a proof): threshold q*(N) = sup{q : f_q unimodal},
log-concavity threshold q_lc(N), and where unimodality first fails.

float64 stepper; all terms nonnegative so relative errors are ~ t * 1e-16.
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from lib1d import pmf_stepper


def analyse(N, q, x0=1):
    L = N - 0.5
    tmax = int(6 * L * L / q) + 50
    f = pmf_stepper(N, q, x0, tmax)[N - x0:]          # from first nonzero value
    f = f[f > 1e-290]
    tol = 1e-11
    r = f[1:] / f[:-1]
    inc = r > 1 + tol
    dec = r < 1 - tol
    # unimodal: no 'inc' after a 'dec'
    first_dec = np.argmax(dec) if dec.any() else len(r)
    unimodal = not inc[first_dec:].any()
    viol_t = None
    if not unimodal:
        viol_t = int(first_dec + np.argmax(inc[first_dec:])) + (N - x0)
    # log-concavity: r non-increasing
    lc = bool(np.all(r[1:] <= r[:-1] * (1 + 1e-10)))
    lc_viol = None
    if not lc:
        lc_viol = int(np.argmax(r[1:] > r[:-1] * (1 + 1e-10))) + (N - x0)
    return unimodal, viol_t, lc, lc_viol


def threshold(N, pred, lo=0.5, hi=1.0, it=40):
    if pred(hi):
        return 1.0
    for _ in range(it):
        mid = 0.5 * (lo + hi)
        if pred(mid):
            lo = mid
        else:
            hi = mid
    return lo


rows = []
for N in list(range(3, 31)) + [40, 50, 60, 80]:
    qs = threshold(N, lambda q: analyse(N, q)[0])
    ql = threshold(N, lambda q: analyse(N, q)[2])
    qN = 1 / (2 * np.cos(np.pi / (2 * N - 1)) ** 2)
    early = (N - 2) / (N - 1.5)
    u, vt, lc, lv = analyse(N, min(qs + 1e-6, 1.0))
    rows.append(dict(N=N, q_star=qs, q_logconcave=ql, q_nonneg_spectrum=qN, q_early=early, first_violation_t=vt))
    print(f"N={N:3d} q*={qs:.9f}  q_lc={ql:.9f}  q_N(spec>=0)={qN:.9f}  (N-2)/(N-3/2)={early:.9f}  viol at t={vt} (N-1={N-1})", flush=True)

json.dump(rows, open(_os.path.join(_R, 'data', 'msc_rigorous_1d', '02_unimodality_scan.json'), "w"), indent=1)
