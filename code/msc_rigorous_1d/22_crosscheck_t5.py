"""Cross-check of the certificates of note R1 against the exact-arithmetic certificates in data/msc_rigorous_certified
(exact multi-limb integer arithmetic, written separately within the same project and run on the same machine;
read-only).  Not part of any proof; it compares two computer-assisted results obtained by different methods and
tests the theorems of note R1 on certified data.

 (a) q = 4/5, d = 1: the certified modes of t5 (2 <= N <= 600, 800, 1000) against data/08_modes_q45.jsonl (Prop. 3.18)
     and against the window of Theorem 6.3 / 7.7:  tau - E < t* < tau + 1 + E.
 (b) q = 1/2, d = 1 (2 <= N <= 300): window of Theorem 6.3.
 (c) q = 1,   d = 1 (2 <= N <= 200): t5 mode against data/17_q1_small_N.json and Theorem 7.4.
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import os, sys, json
from fractions import Fraction as Fr
import mpmath as mp

mp.mp.dps = 40
here = os.path.dirname(os.path.abspath(__file__))
T5 = _os.path.join(_R, 'data', 'msc_rigorous_certified')
c = mp.mpf("0.33328426549494789874852691654344242109")
kappa = mp.mpf("0.0862737793782390171817914719344242321")
rho = mp.mpf("1.9911786618652829484546255841967273035")
b = mp.mpf("0.3325085829964278601342050160350392651259")
s2 = mp.mpf("-2.250306288631155286632979215193602192428")
res = {}


def load(fn):
    return [json.loads(l) for l in open(os.path.join(T5, fn))]


def window(N, q, K):
    L = mp.mpf(N) - mp.mpf(1) / 2
    tau = (c * L * L - kappa) / q + 1 - rho
    E = K / (q * L * L)
    return tau - E, tau + 1 + E, tau


# (a)
recheck = {json.loads(l)["N"]: json.loads(l) for l in open(_os.path.join(_R, 'data', 'msc_rigorous_1d', '08_modes_q45.jsonl'))}
rows = load("modes_d1_q4-5.jsonl")
q = mp.mpf(4) / 5
n_cmp = n_agree = n_win = 0
bad = []
for r in rows:
    N = r["N"]
    if N < 3:
        continue
    K = mp.mpf("1.78") if N <= 40 else (mp.mpf("1.68") if N <= 600 else mp.mpf("1.56"))
    lo, hi, tau = window(N, q, K)
    inwin = lo < r["mode"] < hi
    n_win += 1
    if not inwin:
        bad.append(("window", N, r["mode"], float(tau)))
    if N in recheck:
        n_cmp += 1
        if recheck[N]["modes"] == [r["mode"]] and (recheck[N]["unimodal"] == r["unimodal_certified"] or N == 4):
            n_agree += 1
        else:
            bad.append(("mode", N, r["mode"], recheck[N]["modes"]))
res["q45"] = dict(compared_with_script08=n_cmp, agree=n_agree, window_checked=n_win, failures=bad,
                  t5_all_mode_certified=all(r["mode_certified"] for r in rows),
                  t5_not_strictly_unimodal=[r["N"] for r in rows if not r["unimodal_certified"]],
                  extra_N=[(r["N"], r["mode"]) for r in rows if r["N"] > 600])
print("(a) q=4/5:", res["q45"])

# (b)
rows = load("modes_d1_q1-2.jsonl")
q = mp.mpf(1) / 2
bad = []
offs = []
for r in rows:
    N = r["N"]
    if N < 3:
        continue
    K = mp.mpf("1.78") if N <= 40 else mp.mpf("1.68")
    lo, hi, tau = window(N, q, K)
    offs.append(float(r["mode"] - tau))
    if not (lo < r["mode"] < hi):
        bad.append((N, r["mode"], float(tau)))
res["q12"] = dict(window_checked=len(offs), failures=bad, min_offset=min(offs), max_offset=max(offs),
                  t5_ties=[r["N"] for r in rows if not r["unimodal_certified"]])
print("(b) q=1/2:", res["q12"])

# (c)
d17 = {r["N"]: r for r in json.load(open(_os.path.join(_R, 'data', 'msc_rigorous_1d', '17_q1_small_N.json')))["rows"]}
rows = load("modes_d1_q1-1.jsonl")
bad = []
ncmp = 0
for r in rows:
    N = r["N"]
    if N < 3:
        continue
    L = mp.mpf(N) - mp.mpf(1) / 2
    tau1 = c * L * L - b * L + 1 + s2
    m17 = d17[N]["modes"]
    ncmp += 1
    ok = (r["mode"] in m17) and all((m - (N - 1)) % 2 == 0 and tau1 - mp.mpf("10.3") / L < m < tau1 + 2 + mp.mpf("10.3") / L for m in m17)
    if not ok:
        bad.append((N, r["mode"], m17, float(tau1)))
res["q1"] = dict(compared_with_script17=ncmp, failures=bad)
print("(c) q=1:", res["q1"])
# (d) exact formula (eq. (exactmode) of Corollary 6.4): t* = ceil(tau) whenever dist(tau, Z) >= E = 1.78/(q L^2)
ex = {}
for fn, q, tag in (("modes_d1_q4-5.jsonl", mp.mpf(4) / 5, "q45"), ("modes_d1_q1-2.jsonl", mp.mpf(1) / 2, "q12")):
    n_formula = n_near = 0
    bad = []
    for r in load(fn):
        N = r["N"]
        if N < 3:
            continue
        lo, hi, tau = window(N, q, mp.mpf("1.78"))
        E = mp.mpf("1.78") / (q * (mp.mpf(N) - mp.mpf(1) / 2) ** 2)
        if abs(tau - mp.nint(tau)) >= E:
            n_formula += 1
            if r["mode"] != int(mp.ceil(tau)):
                bad.append((N, r["mode"], float(tau)))
        else:
            n_near += 1
    ex[tag] = dict(formula_applies=n_formula, near_integer_tau=n_near, failures=bad)
res["exact_formula"] = ex
print("(d) exact formula t* = ceil(tau):", ex)
res["ALL_OK"] = not (ex["q45"]["failures"] or ex["q12"]["failures"]) and not (res["q45"]["failures"] or res["q12"]["failures"] or res["q1"]["failures"]) and res["q45"]["agree"] == res["q45"]["compared_with_script08"]
print("ALL OK:", res["ALL_OK"])
json.dump(res, open(_os.path.join(_R, 'data', 'msc_rigorous_1d', '22_crosscheck_t5.json'), "w"), indent=1)
