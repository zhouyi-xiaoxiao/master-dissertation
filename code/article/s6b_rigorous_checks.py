#!/usr/bin/env python
"""Section 7 (s6b_rigorous) and the additions to Section 4: consistency checks between the theorems and the
computations of Sections 3-6, one deduction of Section 7 computed again, and the few derived numbers quoted.

 1. Certified modes.  Every corner-to-corner discrete-time mode of the article (time stepping, route A of
    Section 3; activities q = 0.8, 0.5, 1) that falls in a range certified in S5 (Theorem 7.15) is compared with
    the certified integer mode (exact certificates; Python integers and the C engine).
 2. One dimension, q = 4/5.  For every size of the article's chain runs at q = 0.8 (and the two large sizes
    N = 1e5, 1e6 and the size N = 10 240 of Section 3) the window of Theorem 7.13(c) (computer-assisted),
    tau - E < t* < tau + 1 + E with tau = (c L^2 - kappa)/q + 1 - rho, L = N - 1/2, is evaluated in 50-digit
    arithmetic; where dist(tau, Z) >= E the theorem gives t* = ceil(tau), which is compared with the stored mode.
 3. Corollary 7.22 (closed form within 0.3 % in d = 3 for every N >= 10; within 1 % in d = 2 for large N):
    the bound B(N) of its proof computed again in interval arithmetic (mpmath.iv) from the window constants of
    Theorem 7.20(a), (b) (the computation of S0, Corollary 8.15).
 4. Section 4, d = 3: the certified constants c_tau3, c_mtau3 of S2 (Theorem 4.8) against the expressions
    -(6/pi) xi_H and (6/pi^2) varkappa of Numerical observation 4.9 (14 digits available).
 5. The continuous-time modes of Sections 5-6 against the windows of Theorem 7.18 and the laws of Theorem 7.19
    (double precision; a consistency check, not a proof).
 6. Derived constants: the two-sided forms of Theorem 7.19, the number of certified blocks of Theorem 7.20(c),
    and the bounds of Section 4 obtained by adding certified bounds of S2.

Inputs  (relative to ):
    data/msc_modes/discrete_modes.jsonl, laplace_modes.jsonl
    data/msc_modes_verify/bigN_1d_mp.json, extra.json
    data/msc_rigorous_certified/modes_d*_q*.jsonl, c128_d*_q*.jsonl
    data/msc_rigorous_1d/03_constants.json
    data/msc_rigorous_asymptotics/80_tables.json, 78_cert_allq_d2.jsonl, 78_cert_allq_d3.jsonl
    data/msc_rigorous_spectral/06_certified_constants.json, 16_certified_d3_remainder.json
    data/article/s3_methods_certificate_chain_N10240.json
Output: data/article/s6b_rigorous_checks.json
Usage:  python s6b_rigorous_checks.py          (a few seconds, < 100 MB)
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import json
import math
import os

import mpmath as mp
from mpmath import iv

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = _R            # 
RD = _os.path.join(_R, 'data', 'msc_modes')
RV = _os.path.join(_R, 'data', 'msc_modes_verify')
CERTS = _os.path.join(_R, 'data', 'msc_rigorous_certified')
R1 = _os.path.join(_R, 'data', 'msc_rigorous_1d')
R2 = _os.path.join(_R, 'data', 'msc_rigorous_spectral')
R3 = _os.path.join(_R, 'data', 'msc_rigorous_asymptotics')
ADATA = _os.path.join(_R, 'data', 'article')
OUT = _os.path.join(_R, 'data', 'article', 's6b_rigorous_checks.json')

out = {"generated_by": "code/article/s6b_rigorous_checks.py"}

# ------------------------------------------------------------------------------------------- 1. certified modes
QSTR = {0.8: "4-5", 0.5: "1-2", 1.0: "1-1"}


def certified(d, qs):
    """N -> (mode, unimodal_certified, file) from the certificate files; Python-integer files take precedence"""
    res = {}
    for kind in ("c128", "modes"):                        # later entries overwrite: Python-integer certificates win
        p = os.path.join(CERTS, f"{kind}_d{d}_q{qs}.jsonl")
        if not os.path.exists(p):
            continue
        for line in open(p):
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("mode_certified"):
                res[r["N"]] = (r["mode"], bool(r.get("unimodal_certified")), os.path.basename(p))
    return res


D = [json.loads(l) for l in open(_os.path.join(_R, 'data', 'msc_modes', 'discrete_modes.jsonl')) if l.strip()]
rows, n_cmp, n_agree, n_outside = [], 0, 0, 0
per = {}
for q in (0.8, 0.5, 1.0):
    for d in (1, 2, 3):
        cert = certified(d, QSTR[q])
        key = f"d{d}_q{q}"
        per[key] = {"n_runs": 0, "n_in_certified_range": 0, "n_agree": 0, "n_unimodal_certified": 0}
        for r in D:
            if r["geo"] != "CC" or r["d"] != d or r["q"] != q:
                continue
            per[key]["n_runs"] += 1
            N = r["N"]
            if N not in cert:
                n_outside += 1
                continue
            m, uni, f = cert[N]
            n_cmp += 1
            per[key]["n_in_certified_range"] += 1
            ok = (m == r["mode"])
            n_agree += ok
            per[key]["n_agree"] += ok
            per[key]["n_unimodal_certified"] += uni
            rows.append({"d": d, "q": q, "N": N, "mode_time_stepping": r["mode"], "mode_certified": m,
                         "agree": ok, "unimodal_certified": uni, "certificate_file": f})
out["certified_modes"] = {"n_compared": n_cmp, "n_agree": n_agree, "n_outside_certified_ranges": n_outside,
                          "by_case": per, "rows": rows}

# ------------------------------------------------------------------------------------------- 2. 1D window, q = 4/5
mp.mp.dps = 50
C1 = json.load(open(_os.path.join(_R, 'data', 'msc_rigorous_1d', '03_constants.json')))
c = mp.mpf(C1["c_mid_80_digits"])
kap = mp.mpf(C1["kappa"].strip("[").split("+/-")[0])
rho = mp.mpf(C1["rho"].strip("[").split("+/-")[0])
q = mp.mpf(4) / 5


def window(N):
    L = mp.mpf(N) - mp.mpf(1) / 2
    tau = (c * L ** 2 - kap) / q + 1 - rho
    E = (mp.mpf("1.68") if N >= 41 else mp.mpf("1.78")) / (q * L ** 2)
    dist = min(tau - mp.floor(tau), mp.ceil(tau) - tau)
    return tau, E, dist


chk = []
cases = [(r["N"], r["mode"], "discrete_modes.jsonl") for r in D if r["geo"] == "CC" and r["d"] == 1 and r["q"] == 0.8]
for b in json.load(open(_os.path.join(_R, 'data', 'msc_modes_verify', 'bigN_1d_mp.json'))):
    if b["N"] in (100000, 1000000):
        cases.append((b["N"], b["recheck"], "bigN_1d_mp.json"))
ch = json.load(open(_os.path.join(_R, 'data', 'article', 's3_methods_certificate_chain_N10240.json')))
cases.append((10240, ch["mode"], "s3_methods_certificate_chain_N10240.json"))
n_det, n_det_agree, n_win = 0, 0, 0
for N, m, src in sorted(cases):
    tau, E, dist = window(N)
    inside = bool(tau - E < m < tau + 1 + E)
    n_win += inside
    det = bool(dist >= E)
    pred = int(mp.ceil(tau)) if det else None
    if det:
        n_det += 1
        n_det_agree += (pred == m)
    chk.append({"N": N, "stored_mode": m, "source": src, "tau": mp.nstr(tau, 20), "E": mp.nstr(E, 6),
                "dist_tau_Z": mp.nstr(dist, 6), "in_window": inside, "window_determines_mode": det,
                "ceil_tau": pred, "agree": (pred == m) if det else None})
out["one_dimension_q4-5_window"] = {"n_cases": len(chk), "n_in_window": n_win, "n_window_determines_mode": n_det,
                                    "n_agree_where_determined": n_det_agree, "rows": chk,
                                    "offset_kappa_over_q_minus_1_plus_rho": mp.nstr(kap / q - 1 + rho, 12)}

# ------------------------------------------------------------------------------------------- 3. Corollary 7.22
iv.dps = 40
T80 = json.load(open(_os.path.join(_R, 'data', 'msc_rigorous_asymptotics', '80_tables.json')))


def rows_of(key):
    """(N0, K-, K+) rows of a window table of the asymptotics note, as decimal strings"""
    return [(int(r[0]), repr(float(r[1])), repr(float(r[2]))) for r in T80[key]["table"]]


def B(d, qv, N, Km, Kp):
    N = iv.mpf(N)
    pi = iv.pi
    Xlo = 2 * pi * iv.log(N) - iv.mpf("0.24") if d == 2 else iv.mpf("7.106860") * N - iv.mpf("7.63")
    mu_hi = (2 * qv / d) * (pi / (2 * N)) ** 2
    taucf = Xlo * iv.log(2 * d * Xlo) / (Xlo + 2 * d - 1)
    Km = iv.mpf(Km); Kp = iv.mpf(Kp)
    num = Kp / Xlo + 2 * mu_hi
    if Km > 0:
        num2 = Km / Xlo
        num = iv.mpf([max(num.a, num2.a), max(num.b, num2.b)])
        den = taucf - Km / Xlo
    else:
        den = taucf
    return num / den


tables = {(3, "4/5"): "T714_q0p8_d3", (3, "1/2"): "T78_d3", (2, "4/5"): "T714_q0p8_d2", (2, "1/2"): "T78_d2"}
target = {2: "0.0100", 3: "0.0030"}
cor = {}
for (d, qs), key in tables.items():
    qn, qd = qs.split("/")
    qv = iv.mpf(qn) / iv.mpf(qd)
    res = []
    rws = [r for r in rows_of(key) if r[0] >= 10]
    for (N0, Km, Kp) in rws:
        b = B(d, qv, N0, Km, Kp)
        res.append({"N0": N0, "K-": Km, "K+": Kp, "B_upper": mp.nstr(b.b, 8),
                    "below_target": bool(b.b < iv.mpf(target[d]).a)})
    N0, Km, Kp = [r for r in rws if r[0] <= 1000][-1]
    lo, hi = N0, 10 ** 12
    if B(d, qv, lo, Km, Kp).b < iv.mpf(target[d]).a:
        Nstar = lo
    else:
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if B(d, qv, mid, Km, Kp).b < iv.mpf(target[d]).a:
                hi = mid
            else:
                lo = mid
        Nstar = hi
    cor[f"d{d}_q{qn}-{qd}"] = {"window_table": key, "target": target[d], "rows": res,
                              "first_N_below_target_with_row_1000": Nstar}
out["corollary_closed_form_all_N"] = cor

# ------------------------------------------------------------------------------------------- 4. d = 3 constants
C06 = json.load(open(_os.path.join(_R, 'data', 'msc_rigorous_spectral', '06_certified_constants.json')))
C16 = json.load(open(_os.path.join(_R, 'data', 'msc_rigorous_spectral', '16_certified_d3_remainder.json')))
EX = json.load(open(_os.path.join(_R, 'data', 'msc_modes_verify', 'extra.json')))
mp.mp.dps = 30
ctau3 = mp.mpf(C06["c_tau_3D"]["mid"])
cmtau3 = mp.mpf(C16["pi^2 c_mtau3/6"]["str"].strip("[").split("+/-")[0]) * 6 / mp.pi ** 2
xiH = mp.mpf(EX["hasimoto_xi"]); vk = mp.mpf(EX["alternating_sum_b"])
out["d3_constants"] = {
    "C_3": C06["C_3"]["str"], "c_tau3": C06["c_tau_3D"]["str"], "pi2_c_mtau3_over_6": C16["pi^2 c_mtau3/6"]["str"],
    "C_3pp": C16["C_3''"]["str"],
    "minus_6_over_pi_xiH": mp.nstr(-6 / mp.pi * xiH, 16), "abs_diff_c_tau3": mp.nstr(abs(ctau3 + 6 / mp.pi * xiH), 3),
    "six_over_pi2_varkappa": mp.nstr(6 / mp.pi ** 2 * vk, 16),
    "abs_diff_c_mtau3": mp.nstr(abs(cmtau3 - 6 / mp.pi ** 2 * vk), 3),
    "K3prime_series": mp.nstr(ctau3 + cmtau3, 15),
    "K3prime_xi_varkappa": mp.nstr(-6 / mp.pi * xiH + 6 / mp.pi ** 2 * vk, 15),
    "digits_of_xiH_and_varkappa_available": 14,
}

# ------------------------------------------------------------------------------------------- 5. continuous time
# The exact continuous-time modes of Sections 5-6 (route C, unit rate) against the window of Theorem 7.18 and the
# two-sided laws of Theorem 7.19 (a consistency check in double precision, not a proof).
LAP = [json.loads(l) for l in open(_os.path.join(_R, 'data', 'msc_modes', 'laplace_modes.jsonl')) if l.strip()]
c3 = (math.pi ** 2 / 6) * float(C06["C_3"]["mid"])
cont = {}
for d in (2, 3):
    tab = sorted((t for t in T80[f"T51_d{d}"]["table"] if t[0] >= (12 if d == 2 else 10)), key=lambda r: r[0])  # printed rows
    n_w = n_w_ok = n_l = n_l_ok = 0
    worst = []
    for r in sorted((x for x in LAP if x["geo"] == "CC" and x["d"] == d), key=lambda x: x["N"]):
        N = r["N"]
        mu1 = (2.0 / d) * math.sin(math.pi / (2 * N)) ** 2
        X = mu1 * r["mfpt"]
        taucf = X * math.log(2 * d * X) / (X + 2 * d - 1)
        val = X * (mu1 * r["mode"] - taucf)
        rows_ok = [t for t in tab if t[0] <= N]
        if rows_ok:
            N0, Km, Kp = rows_ok[-1][0], rows_ok[-1][1], rows_ok[-1][2]
            ok = (-Km < val < Kp)
            n_w += 1; n_w_ok += ok
            worst.append({"N": N, "X_times_scaled_error": round(val, 4), "row_N0": N0, "K-": Km, "K+": Kp, "inside": ok})
        if N >= 20:
            if d == 2:
                lead = math.log(math.log(N)) + math.log(8 * math.pi)
                low = (4 / math.pi ** 2) * N ** 2 * (math.log(math.log(N)) + 2.27)
                up = (4 / math.pi ** 2) * N ** 2 * (math.log(math.log(N)) + 3.2242) * (1 + 0.83 / N ** 2)
            else:
                lead = math.log(N) + math.log(6 * c3)
                low = (6 / math.pi ** 2) * N ** 2 * (math.log(N) + 3.43)
                up = (6 / math.pi ** 2) * N ** 2 * (math.log(N) + 3.7529) * (1 + 0.83 / N ** 2)
            ok2 = (low <= r["mode"] <= up)
            n_l += 1; n_l_ok += ok2
    cont[f"d{d}"] = {"n_sizes_in_window_range": n_w, "n_inside_window": n_w_ok,
                     "n_sizes_N_ge_20": n_l, "n_inside_two_sided_law": n_l_ok, "rows": worst}
out["continuous_time_against_windows"] = cont

# ------------------------------------------------------------------------------------------- 6. derived constants
# (a) the two-sided forms of Theorem 7.19 from its remainder bounds: for N >= 20,
#     ln(8 pi) + R_N >= 2.27 (d = 2), ln(6 c_3) + R_N >= 3.43 (d = 3), with the lower bounds on R_N of the theorem,
#     and ln(8 pi) <= 3.2242, ln(6 c_3) <= 3.7529 (upper forms; R_N < 0).  The lower bounds on R_N increase with N
#     (checked on N = 20 ... 10^7 and in the limit 0); interval arithmetic, c_3 in [7.106860, 7.106880].
iv.dps = 30
c3lo, c3hi = iv.mpf("7.106860"), iv.mpf("7.106880")


def lowR(d, N):
    N = iv.mpf(N)
    if d == 2:
        return -(iv.mpf("4.6") + 3 * iv.log(8 * iv.pi * iv.log(N) + iv.mpf("2.4"))) / (2 * iv.pi * iv.log(N) - iv.mpf("0.24"))
    return -(iv.mpf("9.2") + 5 * iv.log(6 * c3hi * N)) / (c3lo * N - iv.mpf("7.63"))


grid = [20 + k for k in range(0, 2000)] + [int(10 ** (k / 20)) for k in range(67, 141)]
lc = {}
for d, lead_lo, lead_hi, low_c, up_c in ((2, iv.log(8 * iv.pi), iv.log(8 * iv.pi), "2.27", "3.2242"),
                                          (3, iv.log(6 * c3lo), iv.log(6 * c3hi), "3.43", "3.7529")):
    vals = [lowR(d, N) for N in grid]
    increasing = all(vals[i + 1].a > vals[i].a for i in range(len(vals) - 1))
    lc[f"d{d}"] = {"lead_lower": mp.nstr(lead_lo.a, 10), "lead_upper": mp.nstr(lead_hi.b, 10),
                   "R_lower_bound_at_N20": mp.nstr(vals[0].a, 8),
                   "lead_plus_R_lower_at_N20_ge_" + low_c: bool((lead_lo + vals[0]).a >= iv.mpf(low_c).b),
                   "lead_upper_le_" + up_c: bool(lead_hi.b <= iv.mpf(up_c).a),
                   "R_lower_bound_increasing_on_grid": increasing}
out["two_sided_law_constants"] = lc

# (b) number of blocks that pass in the certificate of the window for every q < 1 (Theorem 7.20(c))
blocks = {}
for d in (2, 3):
    rr = [json.loads(l) for l in open(os.path.join(R3, f"78_cert_allq_d{d}.jsonl")) if l.strip()]
    ok = [r for r in rr if r.get("ok")]
    blocks[f"d{d}"] = {"blocks": len(rr), "passing": len(ok), "smallest_N_passing": min(r["Na"] for r in ok)}
out["every_q_certificate_blocks"] = blocks

# (c) Section 4: bounds for every N obtained by adding the certified bounds of S2 (Thms 9.1, 10.1, Cors 11.1, 11.2): q T_N - (8/pi) N^2 ln N - C_2 N^2 in [-2.14 - 9.06, 3.53 + 7.72];
#     q T_N^(3) - K_3 N^3 - K_3' N^2 in [-17.4 - 82.4, 19.4 + 80.4];
#     X - 2 pi ln N - (pi^2/4) C_2: |.| <= (8.8 + 5.2 ln N)/N^2 + 24.2/N^2;
#     X - (pi^2/6)(K_3 N + K_3'): >= -5.85/N - (21.3 + 137.7)/N^2 and <= (31.92 + 137.7)/N^2.
F = lambda x: mp.mpf(x)
out["section4_combined_bounds"] = {
    "d2_remainder_interval": [mp.nstr(-F("2.14") - F("9.06"), 6), mp.nstr(F("3.53") + F("7.72"), 6)],
    "d3_remainder_interval": [mp.nstr(-F("17.4") - F("82.4"), 6), mp.nstr(F("19.4") + F("80.4"), 6)],
    "d2_X_bound_constant_term": mp.nstr(F("8.8") + F("24.2"), 6), "d2_X_bound_lnN_coefficient": "5.2",
    "d3_X_lower_1_over_N2": mp.nstr(F("21.3") + F("137.7"), 6), "d3_X_upper_1_over_N2": mp.nstr(F("31.92") + F("137.7"), 6),
    "d2_pi2_over_4_C2": mp.nstr(mp.pi ** 2 / 4 * F(C06["c_m_2D"]["mid"]), 12) if "c_m_2D" in C06 else None,
}

os.makedirs(_os.path.join(_R, 'data', 'article'), exist_ok=True)
json.dump(out, open(OUT, "w"), indent=1)
print(json.dumps({k: (v if k != "certified_modes" else {kk: vv for kk, vv in v.items() if kk != "rows"})
                  for k, v in out.items() if k not in ("one_dimension_q4-5_window",)}, indent=1)[:4000])
w = out["one_dimension_q4-5_window"]
print("1D window:", {k: v for k, v in w.items() if k != "rows"})
for r in w["rows"]:
    if r["N"] >= 1000 or not r["agree"]:
        print("  ", r)
