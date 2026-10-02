"""
Large-N expansion of the corner-to-corner MFPT (Theorem 4.5 of the article, proof in Supplementary Section S4):

  q T_N = (8/pi) N^2 ln N + C_2 N^2 + C_0 + C_{-2} N^{-2} + O(N^{-4}),

  C_2    = (8/pi) [ gamma + ln(4 sqrt(2)/varpi) - 1/2 - pi/4 ]
  C_0    = varpi^4 / (9 pi^2)                 ( = Gamma(1/4)^8 / (576 pi^4) )
  C_{-2} = -3 varpi^4/(50 pi) - varpi^8/(432 pi^3)
  varpi  = Gamma(1/4)^2 / (2 sqrt(2 pi))      (lemniscate constant)

The script evaluates q T_N exactly (50 digits, overflow-free single sum) for
N up to 2^19 on three sequences (powers of 2, powers of 3, powers of 10),
forms the successive residuals and estimates the next coefficient C_{-4}
by Richardson extrapolation.

Outputs: results/asymptotics.csv, results/asymptotics_summary.json
"""
from __future__ import annotations
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository

import csv
import json
import os
import time

import mpmath as mp

import mfpt2d as M

HERE = os.path.dirname(os.path.abspath(__file__))
RES = _os.path.join(_R, 'data', 'msc_proof')
DPS = 50
mp.mp.dps = DPS

g14 = mp.gamma(mp.mpf(1) / 4)
varpi = g14 ** 2 / (2 * mp.sqrt(2 * mp.pi))
C2 = 8 / mp.pi * (mp.euler + mp.log(4 * mp.sqrt(2) / varpi) - mp.mpf(1) / 2 - mp.pi / 4)
C0 = varpi ** 4 / (9 * mp.pi ** 2)
Cm2 = -3 * varpi ** 4 / (50 * mp.pi) - varpi ** 8 / (432 * mp.pi ** 3)
C2_lib, C0_lib = M.asymptotic_constants(DPS)
assert abs(C2 - C2_lib) < mp.mpf(10) ** (-45) and abs(C0 - C0_lib) < mp.mpf(10) ** (-45)

# independent series for C_{-2} (Supplementary Section S4, before the modular evaluation)
p = -mp.exp(-mp.pi)
L3 = mp.nsum(lambda k: k ** 3 * p ** k / (1 + p ** k), [1, mp.inf])
L4 = mp.nsum(lambda k: k ** 4 * p ** k / (1 + p ** k) ** 2, [1, mp.inf])
L5 = mp.nsum(lambda k: k ** 5 * p ** k * (1 - p ** k) / (1 + p ** k) ** 3, [1, mp.inf])
Cm2_series = (-47 * mp.pi ** 3 / 21600 - mp.pi ** 5 / 18 * L5 + 5 * mp.pi ** 4 / 18 * L4
              + 47 * mp.pi ** 3 / 90 * L3)
e4 = 3 * varpi ** 4 / mp.pi ** 4
checks = dict(
    Lambda3_minus_closed=mp.nstr(abs(L3 - (1 - 6 * e4) / 240), 3),
    Lambda4_minus_closed=mp.nstr(abs(L4 + e4 / (20 * mp.pi)), 3),
    Lambda5_minus_closed=mp.nstr(abs(L5 - (-e4 / (8 * mp.pi ** 2) + e4 ** 2 / 216)), 3),
    Cm2_series_minus_closed=mp.nstr(abs(Cm2_series - Cm2), 3),
)
print("constant checks:", checks)
print("C_2    =", mp.nstr(C2, 40))
print("C_0    =", mp.nstr(C0, 40))
print("C_{-2} =", mp.nstr(Cm2, 40), "  (Essam-Wu 2*c4 = -1.069558947686132)")


seqs = {
    "2^j": [2 ** j for j in range(1, 20)],
    "3^j": [3 ** j for j in range(1, 12)],
    "10^j": [10 ** j for j in range(1, 6)] + [35, 200, 403, 4001],
}
rows = []
t0 = time.time()
for name, Ns in seqs.items():
    for N in Ns:
        T = M.clean_single_mp(N, 1, DPS)
        lead = 8 / mp.pi * N ** 2 * mp.log(N)
        r1 = T - lead
        r2 = r1 - C2 * N ** 2
        r3 = r2 - C0
        r4 = r3 - Cm2 / mp.mpf(N) ** 2
        rows.append(dict(sequence=name, N=N, qT=mp.nstr(T, 40),
                         rel_err_1term=mp.nstr(r1 / T, 6), rel_err_2term=mp.nstr(r2 / T, 6),
                         res_after_N2=mp.nstr(r2, 20), res_after_const=mp.nstr(r3, 12),
                         N2_times_res3=mp.nstr(r3 * N ** 2, 20),
                         N4_times_res4=mp.nstr(r4 * mp.mpf(N) ** 4, 15)))
    print(f"   sequence {name} done [{time.time() - t0:.1f}s]")
rows.sort(key=lambda r: r["N"])
with open(_os.path.join(_R, 'data', 'msc_proof', 'asymptotics.csv'), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)

# Richardson estimate of C_{-4} from the 2^j sequence (residual r4 ~ C_{-4}/N^4 + C_{-6}/N^6 ...)
pw = [r for r in rows if r["sequence"] == "2^j" and r["N"] >= 64]
vals = [mp.mpf(r["N4_times_res4"]) for r in pw]
rich = [(4 * vals[i + 1] - vals[i]) / 3 for i in range(len(vals) - 1)]   # removes the N^-2 correction
Cm4_est = rich[-1]
Cm4_spread = abs(rich[-1] - rich[-2])
# fit of C_{-2} treated as unknown: N^2 * r3 -> C_{-2} + C_{-4}/N^2
v3 = [mp.mpf(r["N2_times_res3"]) for r in pw]
rich3 = [(4 * v3[i + 1] - v3[i]) / 3 for i in range(len(v3) - 1)]
summary = dict(
    dps=DPS, C2=mp.nstr(C2, 45), C0=mp.nstr(C0, 45), Cm2=mp.nstr(Cm2, 45),
    varpi=mp.nstr(varpi, 45), constant_checks=checks,
    Cm2_numerical_Richardson=mp.nstr(rich3[-1], 25),
    Cm2_numerical_minus_closed=mp.nstr(abs(rich3[-1] - Cm2), 3),
    Cm4_numerical_Richardson=mp.nstr(Cm4_est, 15), Cm4_last_two_differ_by=mp.nstr(Cm4_spread, 3),
    essam_wu=dict(c0="0.077318893909458", c2="0.266070441638478", c4="-0.534779473843066",
                  ours_c0=mp.nstr(C2 / 2, 20), ours_c2=mp.nstr(C0 / 2, 20), ours_c4=mp.nstr(Cm2 / 2, 20)),
    largest_N=max(r["N"] for r in rows),
    max_abs_res4_N_ge_64=mp.nstr(max(abs(mp.mpf(r["N4_times_res4"])) for r in rows if r["N"] >= 64), 6),
    seconds=round(time.time() - t0, 1),
)
with open(_os.path.join(_R, 'data', 'msc_proof', 'asymptotics_summary.json'), "w") as f:
    json.dump(summary, f, indent=1)
print(json.dumps(summary, indent=1))
for r in rows:
    if r["N"] in (10, 35, 100, 200, 1000, 4001, 10000, 100000, 524288):
        print(r["N"], r["qT"][:24], "rel1", r["rel_err_1term"], "rel2", r["rel_err_2term"],
              "N^2*r3", r["N2_times_res3"][:16], "N^4*r4", r["N4_times_res4"])
