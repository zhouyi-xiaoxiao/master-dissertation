#!/usr/bin/env python
"""Companion note R0, Section 4 (one dimension, discrete time): the counts behind the exact-mode formula t* = ceil(tau).

For q = 4/5 and q = 1/2 the window of the computer-assisted theorem on the discrete-time mode of the chain contains
exactly one integer, and then t* = ceil(tau), whenever dist(tau, Z) >= E, where (note R1, Section 6)
    L = N - 1/2,   tau = (c L^2 - kappa)/q + 1 - rho,   E = K/(q L^2),   K = 1.78 (N <= 40), 1.68 (N >= 41).
On the ranges of the certified exact modes of the chain (data/msc_rigorous_certified/c128_d1_q4-5.jsonl: N = 2..1500
and 2000; c128_d1_q1-2.jsonl: N = 2..1000) this script counts, for N >= 3, the sizes at which dist(tau, Z) >= E holds,
lists the others, and checks that ceil(tau) equals the certified mode wherever the hypothesis holds.  The same counts
are given for the sub-ranges used in note R1 (N <= 600 and N = 800, 1000 for q = 4/5; N <= 300 for q = 1/2).
Constants c, kappa, rho: midpoints of their enclosures in data/msc_rigorous_1d/03_constants.json (radius below 1e-60,
far below the smallest distance |dist(tau, Z) - E| that occurs).  No proof rests on this script; it recomputes numbers
that the text quotes.

Input : data/msc_rigorous_1d/03_constants.json; data/msc_rigorous_certified/c128_d1_q4-5.jsonl, c128_d1_q1-2.jsonl
Output: data/checks/r0_1d_exact_formula_counts.json
Run   : python checks_1d_exact_formula.py          (a few seconds)
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import json
import os
import re

import mpmath as mp

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = _R
CONST = _os.path.join(_R, 'data', 'msc_rigorous_1d', '03_constants.json')
CERT_Q45 = _os.path.join(_R, 'data', 'msc_rigorous_certified', 'c128_d1_q4-5.jsonl')
CERT_Q12 = _os.path.join(_R, 'data', 'msc_rigorous_certified', 'c128_d1_q1-2.jsonl')
OUT = _os.path.join(_R, 'data', 'checks', 'r0_1d_exact_formula_counts.json')

mp.mp.dps = 50


def mid(ball):
    """midpoint of an Arb ball printed as '[m +/- r]'"""
    m = re.match(r"\[\s*([-+0-9.eE]+)\s*\+/-\s*([0-9.eE+-]+)\s*\]", ball)
    assert m and mp.mpf(m.group(2)) < mp.mpf("1e-55"), ball
    return mp.mpf(m.group(1))


C = json.load(open(CONST))
c, kappa, rho = mp.mpf(C["c_mid_80_digits"]), mid(C["kappa"]), mid(C["rho"])


def counts(path, q, sub_range):
    rows = [json.loads(line) for line in open(path) if line.strip()]
    n_all = n_hyp = n_agree = n_sub = n_sub_hyp = 0
    near, min_margin = [], None
    for r in rows:
        N = int(r["N"])
        if N < 3:
            continue
        n_all += 1
        L = mp.mpf(N) - mp.mpf(1) / 2
        K = mp.mpf("1.78") if N <= 40 else mp.mpf("1.68")
        E = K / (q * L * L)
        tau = (c * L * L - kappa) / q + 1 - rho
        dist = min(tau - mp.floor(tau), mp.ceil(tau) - tau)
        margin = abs(dist - E)
        min_margin = margin if min_margin is None else min(min_margin, margin)
        sub = sub_range(N)
        n_sub += sub
        if dist >= E:
            n_hyp += 1
            n_sub_hyp += sub
            n_agree += int(mp.ceil(tau)) == int(r["mode"])
        else:
            near.append(N)
    Ns = sorted(int(r["N"]) for r in rows)
    return {"certificate_file": os.path.basename(path), "N_range": [Ns[0], Ns[-1]], "n_records": len(rows),
            "n_cases_N_ge_3": n_all, "n_hypothesis_holds": n_hyp, "n_formula_equals_certified_mode": n_agree,
            "N_with_tau_within_E_of_an_integer": near,
            "sub_range": sub_range.__doc__, "n_cases_sub_range": n_sub, "n_hypothesis_holds_sub_range": n_sub_hyp,
            "min_abs_dist_minus_E": mp.nstr(min_margin, 6)}


def sub45(N):
    """N <= 600 and N = 800, 1000"""
    return N <= 600 or N in (800, 1000)


def sub12(N):
    """N <= 300"""
    return N <= 300


out = {"provenance": "computed by code/article/checks_1d_exact_formula.py from data/msc_rigorous_1d/03_constants.json "
                     "and the certified exact modes data/msc_rigorous_certified/c128_d1_q4-5.jsonl, c128_d1_q1-2.jsonl",
       "rule": "t* = ceil(tau) when dist(tau, Z) >= E; L = N - 1/2, tau = (c L^2 - kappa)/q + 1 - rho, "
               "E = K/(q L^2), K = 1.78 for N <= 40 and 1.68 for N >= 41",
       "q=4/5": counts(CERT_Q45, mp.mpf(4) / 5, sub45),
       "q=1/2": counts(CERT_Q12, mp.mpf(1) / 2, sub12)}
for k in ("q=4/5", "q=1/2"):
    v = out[k]
    print(k, v["n_hypothesis_holds"], "of", v["n_cases_N_ge_3"], "; formula = certified mode:",
          v["n_formula_equals_certified_mode"], "; exceptions:", v["N_with_tau_within_E_of_an_integer"],
          "; sub-range:", v["n_hypothesis_holds_sub_range"], "of", v["n_cases_sub_range"])
out["all_agree"] = all(out[k]["n_formula_equals_certified_mode"] == out[k]["n_hypothesis_holds"]
                       for k in ("q=4/5", "q=1/2"))
json.dump(out, open(OUT, "w"), indent=1)
print("all agree:", out["all_agree"], "->", os.path.relpath(OUT, HERE))
