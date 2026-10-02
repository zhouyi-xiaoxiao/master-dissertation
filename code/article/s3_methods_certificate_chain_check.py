#!/usr/bin/env python
"""Supplementary Section S2.3: certificate of the global mode for the chain at N = 10 240, q = 0.8 (second half).

At this size the maximum of the first-passage PMF is too flat for double-precision time stepping to decide the
integer maximiser: neighbouring values differ by a relative 1.7e-16, and the stepped values carry an accumulated
round-off of about 1e-10.  The certificate therefore has three parts, none of which compares stepped values more
finely than the worst-case round-off of the stepping.

 (1) t >= t_cert.  The compiled stepper  s3_methods_certificate_chain.c  gives the Cauchy-Schwarz bound of
     Proposition s3_methods:prop-certificate(b):  f(t) <= ||u_k|| ||v_k||  for every t >= t_cert = 2k+1, and this
     bound lies below the computed maximum by the relative margin  margin_rel  (1.1e-6).
 (2) t < t_cert outside the stored window |t - t*| <= 20000.  The largest stepped value there is
     f_max_outside_stored_window_over_f_max (= 1 - 1.6e-7) times the computed maximum.
 (3) |t - t*| <= 20000.  Here the stepper is not used.  The closed form of Proposition s5_modes:prop-1d,

        f(t)          = 2q/(2N-1) sum_{m=1}^{N-1} (-1)^(m+1) cos(w_m/2) sin(w_m) lam_m^(t-1),
        f(t+1) - f(t) = 2q/(2N-1) sum_{m=1}^{N-1} (-1)^(m+1) cos(w_m/2) sin(w_m) lam_m^(t-1) (lam_m - 1),
        w_m = (2m-1) pi/(2N-1),   lam_m = 1 - q + q cos(w_m),

     is evaluated in 60-digit arithmetic with its M_TERMS slowest terms; the omitted terms are bounded by
     2q/(2N-1) (N-1-M_TERMS) 2 Lam^(t-1), Lam = max_{m > M_TERMS} |lam_m|, which is below 1e-290 at these times.
     Check: f(t+1) - f(t) > 0 for every t in [t* - 20000, t* - 1] and < 0 for every t in [t*, t* + 20000], with
     |increment| above the tail bound.  Hence f(t) < f(t*) for every t != t* of the window.

A-priori round-off of the stepping (used for (1) and (2)).  Every operation of the stepper acts on non-negative
numbers: one application of Q is  out = a x_j + c (x_{j-1} + x_{j+1})  with three rounded operations, and the
stored coefficients a = fl(1 - fl(0.8)), c = fl(0.5 fl(0.8)) differ from 1/5 and 2/5 by relative amounts of at most
2u (u = 2^-53).  By monotonicity every component of the computed u_k, v_k is within the factor
[(1 + u)^3 (1 + 2u)]^(+-k), about (1 + 5u)^(+-k), of its exact value, and a dot product of n non-negative terms in four accumulators adds at most (n/4 + 5) u.  With
k <= (t_cert - 1)/2 this bounds the relative error of every stepped f(t), of the norms and of their product by
delta (3.4e-8), barring underflow, which affects only components below 1e-300.  Conditions (1) and (2) are
required to hold with the factor (1 + delta)/(1 - delta), i.e. for the exact values.

Measured round-off.  If the stepper was run with a probe file (fourth argument), the values f(t) written there are
compared with the closed form (all N-1 terms): block "stepper_against_closed_form".  The measured error (1e-10
over the whole run) is far below the a-priori bound; it is reported, not used.

The complete sum (all N-1 terms, no truncation) is evaluated for the 104 values of t around the mode at which the
stepped values are within 1e-12 of their maximum; this checks the truncation and gives the true gap between the
maximum and its neighbours.

Input : data/article/s3_methods_certificate_chain_N10240.json       (output of the compiled stepper)
        data/article/s3_methods_certificate_chain_N10240_probes.txt (its probe file; optional)
Output: data/article/s3_methods_certificate_chain.json
Run   : python s3_methods_certificate_chain_check.py                    (about 2 minutes; mpmath)
        python s3_methods_certificate_chain_check.py <stepper-output.json> <output.json> [<probe-file>]  (another size)
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import json
import os
import sys
import time

import mpmath as mp

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = _os.path.join(_R, 'data', 'article')
INP = sys.argv[1] if len(sys.argv) > 1 else _os.path.join(_R, 'data', 'article', 's3_methods_certificate_chain_N10240.json')
OUT = sys.argv[2] if len(sys.argv) > 2 else _os.path.join(_R, 'data', 'article', 's3_methods_certificate_chain.json')
PROBE = sys.argv[3] if len(sys.argv) > 3 else _os.path.join(_R, 'data', 'article', 's3_methods_certificate_chain_N10240_probes.txt')
mp.mp.dps = 60
M_TERMS = 24

t00 = time.time()
st = json.load(open(INP))
N, tstar, tcert, half = st["N"], st["mode"], st["t_cert"], st["stored_window_half_width"]
q = mp.mpf(4) / 5 if abs(st["q"] - 0.8) < 1e-15 else mp.mpf(st["q"])       # 0.8 exactly, not its double
assert 1 < tstar - half and tstar + half < tcert

pref = 2 * q / (2 * N - 1)


def term(m):
    w = (2 * m - 1) * mp.pi / (2 * N - 1)
    return (-1) ** (m + 1) * mp.cos(w / 2) * mp.sin(w), 1 - q + q * mp.cos(w)


# ---------------------------------------------------------------- (3) the stored window, 60 digits, M_TERMS terms
M = min(M_TERMS, N - 1)
cl = [term(m) for m in range(1, M + 1)]
coef_M, lam_M = [c for c, _ in cl], [l for _, l in cl]
if M < N - 1:
    Lam = max(abs(term(M + 1)[1]), abs(term(N - 1)[1]))                   # lam_m decreases with m
else:
    Lam = mp.mpf(0)
w_lo, w_hi = tstar - half, tstar + half
tail_inc = pref * (N - 1 - M) * 2 * Lam ** (w_lo - 1)                     # bound on the omitted part of f(t+1)-f(t)
pw = [l ** (w_lo - 1) for l in lam_M]                                      # lam_m^(t-1) at t = w_lo
rising = falling = True
min_abs_inc_over_tail = None
min_abs_inc = None
n_window = 0
for t in range(w_lo, w_hi + 1):
    inc = pref * mp.fsum(c * p * (l - 1) for c, p, l in zip(coef_M, pw, lam_M))   # f(t+1) - f(t), truncated
    if t <= tstar - 1:
        rising = rising and (inc - tail_inc > 0)
    else:
        falling = falling and (inc + tail_inc < 0)
    a = abs(inc)
    if min_abs_inc is None or a < min_abs_inc:
        min_abs_inc = a
    pw = [p * l for p, l in zip(pw, lam_M)]
    n_window += 1
f_star_M = pref * mp.fsum(c * l ** (tstar - 1) for c, l in zip(coef_M, lam_M))
f_edge = {tt: pref * mp.fsum(c * l ** (tt - 1) for c, l in zip(coef_M, lam_M)) for tt in (w_lo, w_hi + 1)}

# ---------------------------------------------------------------- all N-1 terms around the mode (no truncation)
coef, lam = [], []
for m in range(1, N):
    c, l = term(m)
    coef.append(c)
    lam.append(l)
lo, hi = st["plateau_first_t"] - 2, st["plateau_last_t"] + 1
assert w_lo <= lo < tstar <= hi <= w_hi
pw = [l ** (lo - 1) for l in lam]
rows, f_at = [], {}
for t in range(lo, hi + 1):
    f = pref * mp.fsum(c * p for c, p in zip(coef, pw))
    inc = pref * mp.fsum(c * p * (l - 1) for c, p, l in zip(coef, pw, lam))
    f_at[t] = f
    rows.append({"t": t, "increment_over_f": mp.nstr(inc / f, 12), "sign": 1 if inc > 0 else (-1 if inc < 0 else 0)})
    pw = [p * l for p, l in zip(pw, lam)]
rising_full = all(r["sign"] == 1 for r in rows if r["t"] <= tstar - 1)
falling_full = all(r["sign"] == -1 for r in rows if r["t"] >= tstar)
fmax_mp = f_at[tstar]
trunc_rel_diff = abs(f_star_M - fmax_mp) / fmax_mp

# ---------------------------------------------------------------- a-priori round-off bound of the stepper
u = mp.mpf(2) ** -53
kmax = (tcert - 1) // 2                                                    # applications of Q to each of u_k, v_k
n_sites = N - 1
step = (1 + u) ** 3 * (1 + 2 * u)                                          # three roundings, coefficients within 2u
delta = step ** (2 * kmax) * (1 + u) ** (n_sites // 4 + 6) - 1
factor = (1 + delta) / (1 - delta)
bound_ratio = mp.mpf(st["bound_at_t_cert"]) / mp.mpf(st["f_max"])
outside_ratio = mp.mpf(st["f_max_outside_stored_window_over_f_max"])
cond1 = bool(bound_ratio * factor < 1)
cond2 = bool(outside_ratio * factor < 1)
cond3 = bool(rising and falling)

# ---------------------------------------------------------------- measured round-off: probes against the closed form
probe_block = None
if os.path.exists(PROBE):
    probes = [(int(a), float(b)) for a, b in (ln.split() for ln in open(PROBE) if ln.strip())]
    err = {}
    for t, fs in probes:
        f = f_at.get(t)
        if f is None:
            f = pref * mp.fsum(c * l ** (t - 1) for c, l in zip(coef, lam))
        err[t] = (mp.mpf(fs) - f) / f                                      # signed relative error of the stepper
    e0 = err[tstar]
    near = [t for t in err if abs(t - tstar) <= 60]
    window = [t for t in err if abs(t - tstar) <= half]
    below = [t for t in err if t < tcert]
    probe_block = {
        "probe_file": "data/article/" + os.path.basename(PROBE), "n_probes": len(probes), "digits": mp.mp.dps,
        "rel_err_at_mode": float(e0),
        "near_mode": {"half_width": 60, "n": len(near),
                      "max_abs_variation_of_rel_err": float(max(abs(err[t] - e0) for t in near))},
        "stored_window": {"half_width": half, "n": len(window),
                          "max_abs_variation_of_rel_err": float(max(abs(err[t] - e0) for t in window))},
        "whole_run": {"n": len(below), "max_abs_rel_err": float(max(abs(err[t]) for t in below)),
                      "rows": [{"t": t, "t_over_mode": round(t / tstar, 4), "rel_err": float(err[t])}
                               for t in sorted(err) if abs(t - tstar) > half]},
        "max_abs_rel_err_below_a_priori_bound": bool(max(abs(err[t]) for t in below) < delta),
    }

out = {
    "generated_by": "code/article/s3_methods_certificate_chain_check.py",
    "N": N, "q": float(q), "mode": tstar, "T": N * (N - 1) / float(q), "mode_over_T": tstar / (N * (N - 1) / float(q)),
    "stepper": st,
    "a_priori_roundoff": {
        "unit_roundoff": float(u), "per_step_factor": "(1 + u)^3 (1 + 2u)", "applications_of_Q_per_vector": kmax,
        "dot_product_terms": n_sites, "delta": float(delta), "factor_(1+delta)/(1-delta)_minus_1": float(factor - 1),
        "note": "relative error bound of every stepped f(t), of ||u_k|| ||v_k|| and of the computed maximum "
                "(all operations on non-negative numbers; underflow below 1e-300 disregarded)",
    },
    "part1_beyond_t_cert": {"bound_over_f_max_computed": float(bound_ratio),
                            "one_minus_bound_over_f_max": float(1 - bound_ratio),
                            "holds_with_a_priori_roundoff": cond1},
    "part2_below_t_cert_outside_window": {"window_half_width": half,
                                           "largest_f_over_f_max_computed": float(outside_ratio),
                                           "one_minus_largest_f_over_f_max": float(1 - outside_ratio),
                                           "holds_with_a_priori_roundoff": cond2,
                                           "closed_form_f_over_f_max_minus_1_at_window_edges": {
                                               str(tt): float(f_edge[tt] / f_star_M - 1) for tt in f_edge}},
    "part3_window_closed_form": {
        "digits": mp.mp.dps, "terms_kept": M, "t_first": w_lo, "t_last": w_hi, "n_t": n_window,
        "largest_omitted_abs_lambda": float(Lam),
        "log10_tail_bound_on_increment": float(mp.log10(tail_inc)) if tail_inc > 0 else None,
        "log10_smallest_abs_increment": float(mp.log10(min_abs_inc)),
        "increments_positive_for_t_below_mode": bool(rising),
        "increments_negative_from_mode_on": bool(falling),
        "f_mode_truncated_vs_all_terms_rel_diff": float(trunc_rel_diff),
    },
    "multi_precision": {
        "digits": mp.mp.dps, "terms": "all N-1", "t_first": lo, "t_last": hi, "n_t": len(rows),
        "increments_positive_for_t_below_mode": bool(rising_full),
        "increments_negative_from_mode_on": bool(falling_full),
        "f_max_50_digits": mp.nstr(fmax_mp, 25),
        "stepper_f_max_rel_diff_from_50_digits": float(abs(mp.mpf(st["f_max"]) - fmax_mp) / fmax_mp),
        "gap_rel_to_larger_neighbour_50_digits": mp.nstr(
            (fmax_mp - max(f_at[tstar - 1], f_at[tstar + 1])) / fmax_mp, 6),
        "rows": rows,
    },
    "stepper_against_closed_form": probe_block,
    "certified_global_maximiser": bool(
        st["certified"] and st["mode_equals_expected"] and cond1 and cond2 and cond3
        and rising_full and falling_full),
    "statement": ("(1) f(t') < max f for every t' >= t_cert: Cauchy-Schwarz bound, relative margin margin_rel; "
                  "(2) every t < t_cert outside the stored window lies below max f by the relative margin "
                  "1 - f_max_outside_stored_window_over_f_max; both margins exceed the a-priori round-off bound "
                  "delta of the stepping with the factor (1+delta)/(1-delta); (3) inside the window the closed form, "
                  "in 60-digit arithmetic with a bounded truncation, rises strictly up to the mode and falls "
                  "strictly after it."),
    "wall_s": round(time.time() - t00, 1),
}
json.dump(out, open(OUT, "w"), indent=1)
print(json.dumps({k: v for k, v in out.items()
                  if k not in ("stepper", "multi_precision", "stepper_against_closed_form")}, indent=1))
print("all-terms rows", lo, hi, "rising", rising_full, "falling", falling_full)
if probe_block:
    print("stepper against closed form:", json.dumps({k: v for k, v in probe_block.items() if k != "whole_run"}),
          "largest error over the run", probe_block["whole_run"]["max_abs_rel_err"])
