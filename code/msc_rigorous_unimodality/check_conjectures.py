"""
check_conjectures.py -- EXACT (integer) evidence for Conjecture 9.1 on long
finite windows.  NOT a proof (the window is finite); recorded as numerical
observation with exact arithmetic.

For geometry GEO, dimension d, activity q = qn/qd and each N, over
t = 0 .. T = K * (cone time + 1):

  (P1')  R_{t+1}(x) R_t(x+e_i) <= R_{t+1}(x+e_i) R_t(x)     for all x, i
         (growth factor rho_{t+1}/rho_t coordinatewise nondecreasing; only
          meaningful for the corner target, where 'toward the target' is the
          coordinatewise order)                                  [flag --p1]
  (LC)   F_t^2 >= F_{t-1} F_{t+1}                              (log-concave PMF)
  (IFR)  F_t S_t <= F_{t+1} S_{t-1}  with S_t = sum_x R_t(x)   (hazard nondecreasing)

Usage: python check_conjectures.py GEO d qn qd K N1 N2 ... [--p1]
Output appended to data/conjectures_<GEO>_d<d>_q<qn>-<qd>.jsonl
"""
from __future__ import annotations
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from unilib import ExactStepper  # noqa: E402
from cert_unimodal import geometry  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = _os.path.join(_R, 'data', 'msc_rigorous_unimodality')


def run(geo, N, d, qn, qd, K, do_p1):
    t0 = time.time()
    start, target = geometry(geo, N, d)
    st = ExactStepper(N, d, qn, qd, start, target)
    D = st.D
    sl = st._sl
    T_cone = None
    T_end = None
    Fm2, Fm1 = None, 0
    Sm1 = 1
    lc_fail, ifr_fail, p1_fail = [], [], []
    t = 0
    while True:
        Rold = st.R
        F = st.step()
        t += 1
        S = int(st.R.sum())
        # LC at index t-1: F_{t-1}^2 >= F_{t-2} F_t
        if Fm2 is not None and Fm1 > 0 and Fm1 * Fm1 < Fm2 * F and len(lc_fail) < 5:
            lc_fail.append(t - 1)
        # IFR: hazard_{t-1} <= hazard_t  <=>  F_{t-1} * S_{t-1} <= F_t * S_{t-2};  here Sm1 = S_{t-1}, Sm2 = S_{t-2}
        if t >= 2 and Fm1 * Sm1 > F * Sm2 and len(ifr_fail) < 5:
            ifr_fail.append(t)
        if do_p1:
            for lo, hi in sl:
                if bool(np.any(st.R[lo] * Rold[hi] > st.R[hi] * Rold[lo])) and len(p1_fail) < 5:
                    p1_fail.append(t)
        if T_cone is None and bool(np.all(st.R <= Rold * D)):
            T_cone = t - 1
            T_end = K * (T_cone + 1)
        Fm2, Fm1 = Fm1, F
        Sm2, Sm1 = Sm1, S
        if T_end is not None and t >= T_end:
            break
    return dict(geo=geo, d=d, N=N, q=f"{qn}/{qd}", T_cone=T_cone, window=t,
                logconcave=not lc_fail, lc_fail_t=lc_fail,
                ifr=not ifr_fail, ifr_fail_t=ifr_fail,
                p1prime=(not p1_fail) if do_p1 else None, p1_fail_t=p1_fail,
                seconds=round(time.time() - t0, 1))


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    do_p1 = "--p1" in sys.argv
    geo, d, qn, qd, K = args[0], int(args[1]), int(args[2]), int(args[3]), int(args[4])
    out = os.path.join(DATA, f"conjectures_{geo}_d{d}_q{qn}-{qd}.jsonl")
    for N in [int(v) for v in args[5:]]:
        rec = run(geo, N, d, qn, qd, K, do_p1)
        with open(out, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)
