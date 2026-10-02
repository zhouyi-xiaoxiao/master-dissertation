"""
Theorem 3 (arbitrary start s and target a on the reflecting square): numerical check.

  q T_{s->a} = 2 tau_1(s2,a2)
             + 4N sum_{k=1}^{N-1} [ cos^2 th_k(a1) Psi_k(a2,a2) - cos th_k(s1) cos th_k(a1) Psi_k(s2,a2) ]

Compared against (i) the exact rational solve for all ordered pairs at N <= 5 and
(ii) the general double sum and the exact rational solve for seeded random pairs
at larger N. Output: results/general_pairs.csv, results/general_pairs_summary.json
"""
from __future__ import annotations
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository

import csv
import itertools
import json
import os
from fractions import Fraction

import mpmath as mp
import numpy as np

import mfpt2d as M

HERE = os.path.dirname(os.path.abspath(__file__))
RES = _os.path.join(_R, 'data', 'msc_proof')
DPS = 40
rows = []
worst = mp.mpf(0)
n_all = 0
# (i) exhaustive, small N
for N in (2, 3, 4, 5):
    sites = list(itertools.product(range(1, N + 1), repeat=2))
    for a in sites:
        # one banded solve per target would be cheaper; N is tiny so solve per pair
        for s in sites:
            if s == a:
                continue
            for q in (Fraction(1), Fraction(4, 5)):
                Tx = M.exact_mfpt_rational(N, q, s, a)
                with mp.workdps(DPS):
                    Tm = mp.mpf(Tx.numerator) / Tx.denominator
                    qm = mp.mpf(q.numerator) / q.denominator
                    v = M.general_single_mp(N, qm, s, a, DPS)
                    worst = max(worst, abs(v - Tm) / Tm)
                n_all += 1
print(f"exhaustive N<=5: {n_all} ordered pairs x q, worst rel err = {mp.nstr(worst, 3)}")

# (ii) random pairs
rng = np.random.default_rng(20261001)
worst2 = mp.mpf(0)
for N in (6, 8, 11, 16, 23, 30):
    for _ in range(12):
        s = tuple(int(v) for v in rng.integers(1, N + 1, size=2))
        a = tuple(int(v) for v in rng.integers(1, N + 1, size=2))
        if s == a:
            continue
        Tx = M.exact_mfpt_rational(N, Fraction(1), s, a)
        with mp.workdps(DPS):
            Tm = mp.mpf(Tx.numerator) / Tx.denominator
            v = M.general_single_mp(N, 1, s, a, DPS)
            d = M.general_double_sum_mp(N, 1, s, a, DPS)
            e1 = abs(v - Tm) / Tm
            e2 = abs(d - Tm) / Tm
            worst2 = max(worst2, e1, e2)
            rows.append(dict(N=N, start=str(s), target=str(a), exact=str(Tx), exact_float=mp.nstr(Tm, 20),
                             rel_single=mp.nstr(e1, 3), rel_double=mp.nstr(e2, 3)))
print(f"random pairs: {len(rows)} cases, worst rel err = {mp.nstr(worst2, 3)}")
with open(_os.path.join(_R, 'data', 'msc_proof', 'general_pairs.csv'), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
with open(_os.path.join(_R, 'data', 'msc_proof', 'general_pairs_summary.json'), "w") as f:
    json.dump(dict(exhaustive_cases=n_all, exhaustive_worst_rel=float(worst),
                   random_cases=len(rows), random_worst_rel=float(worst2), dps=DPS, seed=20261001), f, indent=1)
