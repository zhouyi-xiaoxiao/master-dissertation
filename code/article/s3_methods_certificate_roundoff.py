#!/usr/bin/env python
"""Section 3.2 (s3_methods) and Supplementary Section S2.3: an a-priori round-off bound for the integer mode of
every run of route A (time stepping) that is not covered by the exact certificates of Theorem 7.15.

Route A (code/msc_modes/fptlib.py, class Stepper) computes, per step and per site,
    new(x) = c_stay(x) rho(x) + sum over existing neighbours y of  w rho(y),     w = q/(2d),
and f(t) = new(target) (point target) or the sum of new over the target sites (EXIT geometry).  Every quantity is
non-negative.  Per site there are at most 2d+1 products, each rounded once, and at most 2d additions, so every
computed component equals  sum_y Q(x,y) rho~(y) (1 + eta_y)  with
    |eta_y| <= delta := (1+u)^(2d+1) (1+e) - 1,      u = 2^-53,
where e is the largest relative error of the stored coefficients w~ and c_stay~ against the exact values for the
exact activity q (q = 0.3, 0.8, 0.9 are not binary numbers; the errors are computed exactly with fractions).
Because Q is non-negative, induction gives, componentwise and for every t,
    (1-delta)^t (1-u)^(m-1) f(t) <= f~(t) <= (1+delta)^t (1+u)^(m-1) f(t)          (m = number of target sites),
apart from underflow, which adds an absolute error below t * n * 2^-1074 (n = number of sites), i.e. below 1e-300.
(For the EXIT geometry the sum over the m target sites is bounded by the worst case of recursive summation.)

The computed maximiser t* is then the maximiser of the exact PMF on {1, ..., t_cert - 1} if, for every other t in
that range,
    f~(t) * U(t*) < f~(t*) * L(t),      U(t) = (1+delta)^t (1+u)^(m-1),   L(t) = (1-delta)^t (1-u)^(m-1),
with the absolute underflow term added on the left.  Beyond t_cert the Cauchy-Schwarz bound of
s3_methods_certificate.py has a relative margin of at least 1e-6; its round-off (vectors as above, dot products of
n terms with relative error at most n u) is checked to be below one tenth of the recorded margin.

This script re-runs route A, with the library that produced the stored modes, for every run that is not
corner-to-corner in a range of Theorem 7.15 (the corner-to-corner runs in those ranges have modes certified in exact
arithmetic; data/article/s6b_rigorous_checks.json, certified_modes), up to t_cert - 1 of
data/article/s3_methods_certificate.jsonl, and applies the test above.

Input : data/msc_modes/discrete_modes.jsonl, data/article/s3_methods_certificate.jsonl,
        data/article/s6b_rigorous_checks.json
Output: data/article/s3_methods_certificate_roundoff.json
Cost  : about one minute of computing in total (96 runs); memory below 200 MB.  No random numbers.
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import json
import os
import sys
from fractions import Fraction

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = _R            # 
CERT = _os.path.join(_R, 'data', 'article', 's3_methods_certificate.jsonl')
CHECKS = _os.path.join(_R, 'data', 'article', 's6b_rigorous_checks.json')
OUT = _os.path.join(_R, 'data', 'article', 's3_methods_certificate_roundoff.json')
sys.path.insert(0, _os.path.join(_R, 'code', 'msc_modes'))
import fptlib as F  # noqa: E402

U = Fraction(1, 2 ** 53)
TINY = 2.0 ** -1074


def exact_q(q):
    return Fraction(repr(q))           # 0.8 -> 4/5, 0.3 -> 3/10, 0.9 -> 9/10, 0.5 -> 1/2, 1.0 -> 1


def coefficient_error(d, q):
    """largest relative error of the coefficients w and c_stay(nblk) as computed by fptlib.Stepper"""
    qe = exact_q(q)
    w_f = q / (2 * d)
    w_e = qe / (2 * d)
    errs = [abs(Fraction(w_f) - w_e) / w_e]
    for nb in range(0, 2 * d + 1):
        c_f = float(np.float64((1.0 - q) + w_f * np.int8(nb)))
        c_e = (1 - qe) + w_e * nb
        if c_e == 0:
            assert c_f == 0.0
            continue
        errs.append(abs(Fraction(c_f) - c_e) / c_e)
    return max(errs)


def run_targets(rec):
    N, d, geo = rec["N"], rec["d"], rec["geo"]
    if geo == "EXIT":
        tg = set()
        for x in np.ndindex(*(N,) * d):
            if any(c in (0, N - 1) for c in x):
                tg.add(tuple(int(c) for c in x))
        return sorted(tg)
    return [tuple(rec["target"])]


def main():
    stored = {}
    for line in open(_os.path.join(_R, 'data', 'msc_modes', 'discrete_modes.jsonl')):
        if line.strip():
            r = json.loads(line)
            stored[(r["d"], r["geo"], r["N"], r["q"])] = r
    cert = [json.loads(l) for l in open(CERT) if l.strip()]
    rows_ex = json.load(open(CHECKS))["certified_modes"]["rows"]
    exact = {(r["d"], round(r["q"], 3), r["N"]) for r in rows_ex if r["agree"]}
    out_rows = []
    n_exact = 0
    for c in cert:
        key = (c["d"], c["geo"], c["N"], c["q"])
        if c["geo"] == "CC" and (c["d"], round(c["q"], 3), c["N"]) in exact:
            n_exact += 1
            continue
        rec = stored[key]
        d, N, q = c["d"], c["N"], c["q"]
        targets = run_targets(rec)
        m = len(targets)
        n = N ** d
        e = coefficient_error(d, q)
        delta = (1 + U) ** (2 * d + 1) * (1 + e) - 1
        dl = float(delta) * (1 + 1e-12)                                   # rounded up
        tc = c["t_cert"]
        f, _, _ = F.run_pmf(N, d, q, tuple(rec["start"]), targets, t_max=tc - 1)
        ft = f[1:tc]
        tm = int(np.argmax(ft)) + 1
        fmax = float(ft[tm - 1])
        t = np.arange(1, tc, dtype=float)
        lu = float(np.log1p(float(U)))
        logU = t * np.log1p(dl) + (m - 1) * lu
        logL = t * np.log1p(-dl) + (m - 1) * np.log1p(-float(U))
        Ut = float(np.exp(logU[tm - 1]))
        absu = tc * n * (2 * d + 1 + m) * TINY
        others = np.ones(tc - 1, dtype=bool)
        others[tm - 1] = False
        lhs = ft[others] * Ut + absu
        rhs = fmax * np.exp(logL[others])
        ok = bool(np.all(lhs < rhs))
        gap_all = float(np.min((fmax - ft[others]) / fmax))
        # the relative round-off allowance at the mode, 1 - L(t_cert)/U(t*)  (for the summary)
        allow = 1.0 - float(np.exp(logL[-1] - logU[tm - 1]))
        # tail: round-off of the Cauchy-Schwarz bound of s3_methods_certificate.py (same stencil, dot products of n terms)
        kc = (tc - 1) // 2
        tail_err = float((1 + dl) ** (2 * kc + 2) * (1 + float(U)) ** (n + 4) - 1) * 2 + allow
        tail_ok = bool(tail_err < 0.1 * c["margin_rel"])
        out_rows.append({
            "d": d, "geo": c["geo"], "N": N, "q": q, "mode_stored": rec["mode"], "mode_recomputed": tm,
            "mode_equals_stored": bool(tm == rec["mode"]), "t_cert": tc, "n_target_sites": m,
            "coefficient_rel_error": float(e), "delta_per_step": dl,
            "min_rel_gap_to_any_other_t": gap_all,
            "roundoff_allowance_rel": allow, "gap_over_allowance": gap_all / allow,
            "argmax_certified_against_roundoff": ok,
            "tail_roundoff_rel": tail_err, "tail_margin_rel": c["margin_rel"], "tail_certified_against_roundoff": tail_ok,
            "certified": bool(ok and tail_ok and tm == rec["mode"]),
        })
        print(f"d={d} {c['geo']:4s} N={N:4d} q={q} mode={tm} gap={gap_all:.3g} allow={allow:.3g} ok={ok} tail={tail_ok}",
              flush=True)
    bad = [r for r in out_rows if not r["certified"]]
    summ = {
        "generated_by": "code/article/s3_methods_certificate_roundoff.py",
        "statement": ("a-priori round-off bound of route A (time stepping, fptlib.Stepper): every computed f(t) lies within "
                      "the factors (1 -/+ delta)^t (1 -/+ u)^(m-1) of the exact value, delta = (1+u)^(2d+1)(1+e) - 1; the "
                      "computed maximiser is certified when every other computed value, inflated by these factors, stays "
                      "below the deflated maximum"),
        "unit_roundoff": float(U),
        "n_runs_route_A": len(cert),
        "n_runs_corner_to_corner_certified_exactly": n_exact,
        "n_runs_checked_here": len(out_rows),
        "n_certified_against_roundoff": len(out_rows) - len(bad),
        "not_certified": [{k: r[k] for k in ("d", "geo", "N", "q", "mode_stored", "min_rel_gap_to_any_other_t",
                                              "roundoff_allowance_rel")} for r in bad],
        "min_rel_gap_checked": min(r["min_rel_gap_to_any_other_t"] for r in out_rows),
        "max_roundoff_allowance_rel": max(r["roundoff_allowance_rel"] for r in out_rows),
        "min_gap_over_allowance": min(r["gap_over_allowance"] for r in out_rows),
        "max_delta_per_step": max(r["delta_per_step"] for r in out_rows),
        "max_tail_roundoff_rel": max(r["tail_roundoff_rel"] for r in out_rows),
        "rows": out_rows,
    }
    json.dump(summ, open(OUT, "w"), indent=1)
    print(json.dumps({k: v for k, v in summ.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()
