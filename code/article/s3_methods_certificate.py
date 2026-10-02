#!/usr/bin/env python
"""Section 3 (s3_methods): certificate that the reported discrete-time modes are GLOBAL maximisers.

Proposition s3_methods:prop-certificate(b):  for the symmetric killed matrix Q, the start vector e_o and the
one-step absorption vector r = (I - Q) 1,

        f(t') <= || Q^j e_o || * || Q^l r ||        for all  t' >= j + l + 1                (Cauchy-Schwarz).

This script re-computes the exact first-passage PMF of every run of route A (time stepping,
data/msc_modes/discrete_modes.jsonl) with code written for this purpose (no research library is
imported) and applies the bound with j = l = k:

    u_k = Q^k e_o,   v_k = Q^k r,
    f(2k+1) = u_k . v_k,        f(2k+2) = u_{k+1} . v_k          (all terms non-negative: no cancellation),
    B_k     = ||u_k|| ||v_k||   >=  f(t')  for every  t' >= 2k+1 .

The run stops at the first k (checked every CHK steps) with  B_k < (1 - MARGIN) * max_{t <= 2k} f(t);  then the
global maximum of f over all t >= 1 is attained on {1, ..., 2k} only, and the mode is the smallest maximiser
found there by exhaustive comparison.  The cost is that of ONE time-stepping run of length t_cert = 2k+1,
because the two vectors are advanced alternately.

Round-off.  The certificate itself has the relative margin MARGIN = 1e-6, far above double-precision error.
The integer argmax is decided by comparing computed values of f; as an empirical measure of their round-off
error the script evaluates f(2k+1) a second time from a different pair of vectors, u_{k+1} . v_{k-1}, and
records the largest relative difference in a window around the mode ("noise"), together with the relative gap
between the maximum and the larger of its two neighbours ("gap").

Input : data/msc_modes/discrete_modes.jsonl          (list of runs, stored modes for comparison)
Output: data/article/s3_methods_certificate.jsonl          (one record per run; restartable)
        data/article/s3_methods_certificate.json           (summary, written by  --summary  or at the end)
Usage : python s3_methods_certificate.py               all runs not yet present (about 20 minutes in total)
        python s3_methods_certificate.py small         runs with N^d * t_end < 2e9        (about 2 minutes)
        python s3_methods_certificate.py mid | big     the rest (d = 2, N >= 241 and d = 3, N >= 60 are "big")
        python s3_methods_certificate.py --summary     only rebuild the summary from the .jsonl file
Memory: below 200 MB.  No random numbers.
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = _R            # 
RD = _os.path.join(_R, 'data', 'msc_modes')
OUTL = _os.path.join(_R, 'data', 'article', 's3_methods_certificate.jsonl')
OUTS = _os.path.join(_R, 'data', 'article', 's3_methods_certificate.json')

CHK = 16            # the bound is tested every CHK steps of k
MARGIN = 1e-6       # relative margin required of the certificate
MAXFAC = 6.0        # give up when 2k exceeds MAXFAC * (stored length of the route-A run)


def apply_Q(x, out, N, d, q, tmask):
    """out <- Q x  for the cancelled-move walk on {0..N-1}^d with the target sites (boolean mask) removed"""
    c = q / (2 * d)
    np.multiply(x, 1.0 - q, out=out)
    for ax in range(d):
        lo = [slice(None)] * d
        hi = [slice(None)] * d
        lo[ax] = slice(0, N - 1)
        hi[ax] = slice(1, N)
        lo, hi = tuple(lo), tuple(hi)
        out[hi] += c * x[lo]
        out[lo] += c * x[hi]
        e0 = [slice(None)] * d
        e1 = [slice(None)] * d
        e0[ax] = 0
        e1[ax] = N - 1
        e0, e1 = tuple(e0), tuple(e1)
        out[e0] += c * x[e0]                   # cancelled move at the lower wall
        out[e1] += c * x[e1]                   # cancelled move at the upper wall
    out[tmask] = 0.0
    return out


def geometry(rec):
    N, d, geo = rec["N"], rec["d"], rec["geo"]
    tmask = np.zeros((N,) * d, dtype=bool)
    if geo == "EXIT":
        for ax in range(d):
            e0 = [slice(None)] * d
            e1 = [slice(None)] * d
            e0[ax] = 0
            e1[ax] = N - 1
            tmask[tuple(e0)] = True
            tmask[tuple(e1)] = True
        start = tuple(rec["start"])
    else:
        tmask[tuple(rec["target"])] = True
        start = tuple(rec["start"])
    return start, tmask


def certify(rec):
    N, d, q = rec["N"], rec["d"], rec["q"]
    start, tmask = geometry(rec)
    shape = (N,) * d
    u = np.zeros(shape)
    u[start] = 1.0
    ind = tmask.astype(float)
    v = np.zeros(shape)
    # r(x) = sum_{a in target} P(x, a):  apply P (not Q) to the indicator of the target, then restrict
    apply_Q(ind, v, N, d, q, np.zeros(shape, dtype=bool))
    v[tmask] = 0.0
    u2 = np.empty(shape)
    v2 = np.empty(shape)
    v_prev = None
    tmax_allowed = int(MAXFAC * rec["t_end"]) + 64
    f = np.zeros(tmax_allowed + 4)            # f[t], t >= 1
    alt = {}                                  # second evaluation of f(2k+1), k >= 1
    fmax = -1.0
    k = 0
    t0 = time.time()
    cert = None
    while 2 * k + 2 <= tmax_allowed:
        fo = float(np.vdot(u, v))             # f(2k+1)
        apply_Q(u, u2, N, d, q, tmask)        # u_{k+1}
        fe = float(np.vdot(u2, v))            # f(2k+2)
        if v_prev is not None:
            alt[2 * k + 1] = float(np.vdot(u2, v_prev))      # f(2k+1) = u_{k+1} . v_{k-1}
        f[2 * k + 1] = fo
        f[2 * k + 2] = fe
        if k > 0 and k % CHK == 0:
            # fmax is the running maximum over t <= 2k (strictly before t_cert = 2k+1)
            B = float(np.sqrt(np.vdot(u, u) * np.vdot(v, v)))
            if B < (1.0 - MARGIN) * fmax:
                cert = {"k": k, "t_cert": 2 * k + 1, "bound": B, "margin_rel": (fmax - B) / fmax}
                break
        fmax = max(fmax, fo, fe)
        apply_Q(v, v2, N, d, q, tmask)        # v_{k+1}
        if v_prev is None:
            v_prev = np.empty(shape)
        u, u2 = u2, u
        v_prev, v, v2 = v, v2, v_prev         # v_prev <- v_k ; v <- v_{k+1}
        k += 1
    wall = time.time() - t0
    out = {"d": d, "geo": rec["geo"], "N": N, "q": q, "group": rec.get("group"),
           "mode_stored": rec["mode"], "f_max_stored": rec["f_max"], "wall_s": round(wall, 2)}
    if cert is None:
        out.update({"certified": False, "t_reached": 2 * k})
        return out
    tc = cert["t_cert"]
    ft = f[1:tc]                                               # f(1), ..., f(t_cert - 1)
    # smallest maximiser, ties
    tmode = int(np.argmax(ft)) + 1
    fmax = float(ft[tmode - 1])
    n_ties = int(np.sum(ft == fmax))
    # local maxima on 1 .. t_cert-1 (interior points, ignoring values below 1e-9 of the maximum)
    inner = ft[1:-1]
    big = inner > 1e-9 * fmax
    locmax = (inner > ft[:-2]) & (inner >= ft[2:]) & big
    locmin = (inner < ft[:-2]) & (inner <= ft[2:]) & big
    lm_t = (np.nonzero(locmax)[0] + 2).tolist()
    mono_after = bool(np.all(np.diff(ft[tmode - 1:]) <= 0))
    # gap to the neighbours and empirical round-off near the mode
    left = ft[tmode - 2] if tmode >= 2 else 0.0
    right = ft[tmode] if tmode < tc - 1 else 0.0
    gap = (fmax - max(left, right)) / fmax
    win = max(8, int(0.02 * tmode))
    noise = 0.0
    for t in range(max(3, tmode - win), min(tc - 1, tmode + win) + 1):
        if t in alt and f[t] > 0:
            noise = max(noise, abs(alt[t] / f[t] - 1.0))
    noise_all = max((abs(alt[t] / f[t] - 1.0) for t in alt if t < tc and f[t] > 1e-9 * fmax), default=0.0)
    # argmax when the odd values are replaced by their second evaluation
    f_alt = ft.copy()
    for t, val in alt.items():
        if t < tc:
            f_alt[t - 1] = val
    tmode_alt = int(np.argmax(f_alt)) + 1
    out.update({
        "certified": True, "t_cert": tc, "t_cert_over_mode": tc / tmode,
        "bound_at_t_cert": cert["bound"], "margin_rel": cert["margin_rel"],
        "mode": tmode, "f_max": fmax, "n_ties_exact": n_ties,
        "mode_equals_stored": bool(tmode == rec["mode"]),
        "f_max_rel_diff_from_stored": abs(fmax / rec["f_max"] - 1.0),
        "n_local_maxima_below_t_cert": len(lm_t), "local_maxima_first10": lm_t[:10],
        "n_local_minima_below_t_cert": int(np.sum(locmin)),
        "monotone_from_mode_to_t_cert": mono_after,
        "gap_rel_to_larger_neighbour": gap,
        "roundoff_rel_near_mode": noise, "roundoff_rel_all_t": noise_all,
        "mode_with_second_evaluation": tmode_alt,
        "argmax_resolved": bool(noise < 0.5 * gap and tmode_alt == tmode),
    })
    return out


def key(r):
    return (r["d"], r["geo"], r["N"], r["q"])


def load_done():
    done = {}
    if os.path.exists(OUTL):
        for line in open(OUTL):
            if line.strip():
                r = json.loads(line)
                done[key(r)] = r
    return done


def size_class(rec):
    work = rec["N"] ** rec["d"] * rec["t_end"]
    if work < 2e9:
        return "small"
    if (rec["d"] == 2 and rec["N"] >= 241) or (rec["d"] == 3 and rec["N"] >= 60):
        return "big"
    return "mid"


def summary():
    done = load_done()
    rows = list(done.values())
    S = {"generated_by": "code/article/s3_methods_certificate.py",
         "statement": "f(t') <= ||Q^k e_o|| ||Q^k r|| for t' >= 2k+1 (Cauchy-Schwarz, Proposition "
                      "s3_methods:prop-certificate(b)); certificate with relative margin 1e-6",
         "n_runs_in_route_A": len([json.loads(l) for l in open(_os.path.join(_R, 'data', 'msc_modes', 'discrete_modes.jsonl')) if l.strip()]),
         "n_runs_done": len(rows)}

    def block(sel):
        r = [x for x in rows if sel(x)]
        c = [x for x in r if x["certified"]]
        if not r:
            return {"n": 0}
        b = {"n": len(r), "n_certified": len(c),
             "n_mode_equals_stored": sum(x["mode_equals_stored"] for x in c),
             "not_certified": [[x["d"], x["geo"], x["N"], x["q"]] for x in r if not x["certified"]]}
        if c:
            b.update({
                "t_cert_over_mode_min": min(x["t_cert_over_mode"] for x in c),
                "t_cert_over_mode_max": max(x["t_cert_over_mode"] for x in c),
                "n_single_local_maximum_below_t_cert": sum(x["n_local_maxima_below_t_cert"] == 1 for x in c),
                "n_monotone_from_mode_to_t_cert": sum(x["monotone_from_mode_to_t_cert"] for x in c),
                "n_exact_ties": sum(x["n_ties_exact"] > 1 for x in c),
                "max_f_max_rel_diff_from_stored": max(x["f_max_rel_diff_from_stored"] for x in c),
                "min_gap_rel": min(x["gap_rel_to_larger_neighbour"] for x in c),
                "max_roundoff_rel_near_mode": max(x["roundoff_rel_near_mode"] for x in c),
                "n_argmax_resolved": sum(x["argmax_resolved"] for x in c),
                "argmax_not_resolved": [[x["d"], x["geo"], x["N"], x["q"], x["gap_rel_to_larger_neighbour"],
                                         x["roundoff_rel_near_mode"]] for x in c if not x["argmax_resolved"]],
                "wall_s_total": round(sum(x["wall_s"] for x in c), 1),
            })
        return b

    S["all_runs"] = block(lambda x: True)
    S["point_target_q_le_0.9"] = block(lambda x: x["geo"] != "EXIT" and x["q"] <= 0.9)
    S["CC_q0.8"] = block(lambda x: x["geo"] == "CC" and x["q"] == 0.8)
    for d in (1, 2, 3):
        S["CC_q0.8_d%d" % d] = block(lambda x, d=d: x["geo"] == "CC" and x["q"] == 0.8 and x["d"] == d)
    S["exit"] = block(lambda x: x["geo"] == "EXIT")
    S["q1"] = block(lambda x: x["q"] == 1.0)
    S["examples"] = {}
    for (d, N) in [(1, 1000), (2, 35), (2, 100), (2, 200), (2, 401), (3, 40), (3, 100)]:
        x = done.get((d, "CC", N, 0.8))
        if x and x["certified"]:
            S["examples"]["d%d_N%d" % (d, N)] = {k: x[k] for k in (
                "mode", "t_cert", "t_cert_over_mode", "margin_rel", "gap_rel_to_larger_neighbour",
                "roundoff_rel_near_mode", "argmax_resolved", "wall_s")}
    json.dump(S, open(OUTS, "w"), indent=1)
    print(json.dumps({k: v for k, v in S.items() if k != "examples"}, indent=1))


if __name__ == "__main__":
    args = [a for a in sys.argv[1:]]
    if "--summary" in args:
        summary()
        sys.exit(0)
    runs = [json.loads(l) for l in open(_os.path.join(_R, 'data', 'msc_modes', 'discrete_modes.jsonl')) if l.strip()]
    classes = [a for a in args if a in ("small", "mid", "big")]
    done = load_done()
    todo = [r for r in runs if key(r) not in done and (not classes or size_class(r) in classes)]
    todo.sort(key=lambda r: r["N"] ** r["d"] * r["t_end"])
    print("runs to do:", len(todo), flush=True)
    for r in todo:
        res = certify(r)
        with open(OUTL, "a") as fh:
            fh.write(json.dumps(res) + "\n")
        print(res["d"], res["geo"], res["N"], res["q"], "certified" if res["certified"] else "NOT certified",
              res.get("mode"), res.get("t_cert"), "%.1fs" % res["wall_s"], flush=True)
    summary()
