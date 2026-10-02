"""
End-to-end verification of the single-sum MFPT identity.

Part A  exact rational MFPT (banded Gaussian elimination in Fractions, built
        from the model rules only) versus 60-digit evaluations of
          - the canonical cosine double sum (Theorem 1),
          - form (a) of the single sum,
          - the clean form (Theorem 2b) and the two half sums (Theorem 2c).
        N = 2..N_EXACT for q = 1; q in {4/5, 3/10, 1/7} for N <= 16;
        also exact check that q*T is independent of q.
Part B  float64: sparse direct solve / CG versus the float64 sums up to N = 1000,
        float64 double sum versus clean single sum up to N = 6000,
        and the overflow of form (a) at N >= 403.
Part C  50-digit: form (a) vs clean form vs double sum at larger N.

Outputs: results/verify_exact.csv, results/verify_float.csv, results/verify_mp.csv,
         results/verify_summary.json
"""
from __future__ import annotations
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository

import csv
import json
import os
import time
from fractions import Fraction

import mpmath as mp
import numpy as np

import mfpt2d as M

HERE = os.path.dirname(os.path.abspath(__file__))
RES = _os.path.join(_R, 'data', 'msc_proof')
N_EXACT = 40
DPS = 60


def half_sums_mp(N, q, dps):
    """Theorem 2c: q T = 8N S_odd - 2N^2 = 2N^2 + 8N S_even."""
    with mp.workdps(dps):
        So = mp.mpf(0); Se = mp.mpf(0)
        for k in range(1, N):
            y = mp.pi * k / (2 * N)
            sk = mp.sin(y)
            t = mp.tanh(N * mp.asinh(sk))
            a = mp.cos(y) ** 2 * mp.sqrt(1 + sk ** 2) / sk
            if k % 2:
                So += a / t
            else:
                Se += a * t
        return (8 * N * So - 2 * N * N) / mp.mpf(q), (2 * N * N + 8 * N * Se) / mp.mpf(q)


summary = {}

# ------------------------------------------------------------------ Part A
t0 = time.time()
rows = []
worst = mp.mpf(0)
exact_q1 = {}
for N in range(2, N_EXACT + 1):
    qs = [Fraction(1)]
    if N <= 16:
        qs += [Fraction(4, 5), Fraction(3, 10), Fraction(1, 7)]
    for q in qs:
        T = M.exact_mfpt_rational(N, q)
        if q == 1:
            exact_q1[N] = T
        else:
            assert T * q == exact_q1[N], "q*T must be independent of q exactly"
        with mp.workdps(DPS):
            Tm = mp.mpf(T.numerator) / T.denominator
            qm = mp.mpf(q.numerator) / q.denominator
            d = M.double_sum_mp(N, qm, DPS)
            th = M.single_sum_form_a_mp(N, qm, DPS)
            cl = M.clean_single_mp(N, qm, DPS)
            ho, he = half_sums_mp(N, qm, DPS)
            errs = [abs(v - Tm) / Tm for v in (d, th, cl, ho, he)]
            worst = max(worst, *errs)
            rows.append(dict(N=N, q=str(q), exact_num=str(T.numerator), exact_den=str(T.denominator),
                             exact_float=mp.nstr(Tm, 25),
                             relerr_double=mp.nstr(errs[0], 3), relerr_form_a=mp.nstr(errs[1], 3),
                             relerr_clean=mp.nstr(errs[2], 3), relerr_half_odd=mp.nstr(errs[3], 3),
                             relerr_half_even=mp.nstr(errs[4], 3)))
with open(_os.path.join(_R, 'data', 'msc_proof', 'verify_exact.csv'), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
summary["partA"] = dict(N_range=[2, N_EXACT], q_values=["1", "4/5", "3/10", "1/7 (N<=16)"], dps=DPS,
                        n_cases=len(rows), worst_relative_error=float(worst),
                        q_scaling_exact=True, seconds=round(time.time() - t0, 1))
print(f"Part A: {len(rows)} exact cases, worst rel. error of any closed form vs exact rational = {mp.nstr(worst, 3)}"
      f"  [{time.time() - t0:.1f}s]")
print("   sample exact values (q=1): ", {N: str(exact_q1[N]) for N in (2, 3, 4, 5, 6)})

# ------------------------------------------------------------------ Part B
t0 = time.time()
rows = []
worstB = 0.0
for N in (2, 3, 5, 10, 20, 35, 50, 100, 200, 300, 402, 403, 500, 700, 1000):
    q = 0.8
    if N <= 300:
        direct = M.direct_mfpt(N, q); how = "spsolve"
    else:
        direct = M.direct_mfpt_cg(N, q); how = "cg"
    ds = M.double_sum_np(N, q)
    cl = M.clean_single_np(N, q)
    th = M.single_sum_form_a_np(N, q)
    e_dir = abs(direct - cl) / cl
    e_ds = abs(ds - cl) / cl
    e_th = abs(th - cl) / cl if np.isfinite(th) else float("nan")
    worstB = max(worstB, e_dir, e_ds)
    rows.append(dict(N=N, q=q, direct=repr(direct), direct_method=how, double_sum=repr(ds), clean_single=repr(cl),
                     form_a_float64=repr(th), rel_direct_vs_clean=e_dir, rel_double_vs_clean=e_ds,
                     rel_form_a_vs_clean=e_th))
    print(f"   N={N:5d} direct({how})={direct:.10e} clean={cl:.10e} rel={e_dir:.1e}  dbl rel={e_ds:.1e}  (3.5) float64: {th}")
for N in (2000, 4000, 6000):
    q = 1.0
    ds = M.double_sum_np(N, q)
    cl = M.clean_single_np(N, q)
    e_ds = abs(ds - cl) / cl
    worstB = max(worstB, e_ds)
    rows.append(dict(N=N, q=q, direct="", direct_method="", double_sum=repr(ds), clean_single=repr(cl),
                     form_a_float64=repr(M.single_sum_form_a_np(N, q)), rel_direct_vs_clean="",
                     rel_double_vs_clean=e_ds, rel_form_a_vs_clean=float("nan")))
    print(f"   N={N:5d} double={ds:.12e} clean={cl:.12e} rel={e_ds:.1e}")
with open(_os.path.join(_R, 'data', 'msc_proof', 'verify_float.csv'), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
first_nan = next(N for N in range(380, 430) if not np.isfinite(M.single_sum_form_a_np(N, 1.0)))
summary["partB"] = dict(worst_relative_error=worstB, first_N_where_form_a_overflows_in_float64=first_nan,
                        seconds=round(time.time() - t0, 1))
print(f"Part B: worst float64 rel. difference = {worstB:.2e}; form (a) first overflows at N = {first_nan}"
      f"  [{time.time() - t0:.1f}s]")

# ------------------------------------------------------------------ Part C
t0 = time.time()
rows = []
worstC = mp.mpf(0)
for N in (64, 128, 256, 403, 512, 1000):
    with mp.workdps(50):
        cl = M.clean_single_mp(N, 1, 50)
        th = M.single_sum_form_a_mp(N, 1, 50 + int(N * 0.8))   # form (a) loses ~0.77N digits to cosh/sinh cancellation
        ho, he = half_sums_mp(N, 1, 50)
        e_th = abs(th - cl) / cl
        e_h = max(abs(ho - cl), abs(he - cl)) / cl
        row = dict(N=N, clean=mp.nstr(cl, 40), rel_form_a=mp.nstr(e_th, 3), rel_half_sums=mp.nstr(e_h, 3))
        if N <= 512:
            ds = M.double_sum_mp(N, 1, 50)
            e_d = abs(ds - cl) / cl
            row["rel_double"] = mp.nstr(e_d, 3)
            worstC = max(worstC, e_d)
        else:
            row["rel_double"] = ""
        worstC = max(worstC, e_th, e_h)
        rows.append(row)
        print("   ", row)
with open(_os.path.join(_R, 'data', 'msc_proof', 'verify_mp.csv'), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
summary["partC"] = dict(worst_relative_error=float(worstC), seconds=round(time.time() - t0, 1))
print(f"Part C: worst 50-digit rel. difference = {mp.nstr(worstC, 3)}  [{time.time() - t0:.1f}s]")

with open(_os.path.join(_R, 'data', 'msc_proof', 'verify_summary.json'), "w") as f:
    json.dump(summary, f, indent=1)
