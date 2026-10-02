"""
cert_pairs.py -- EXHAUSTIVE exact classification of all (start, target) pairs of
the box {0..N-1}^d (targets reduced by the hyperoctahedral symmetry of the box:
sorted coordinates, each <= (N-1)/2; all starts != target).

For each pair:
  1. exact cone certificate at q = 1/2  (cert_unimodal.certify)
       -> 'U12'  : PMF unimodal at q = 1/2  (hence for all q <= 1/2 and in
                   continuous time, by the thinning theorem)
       -> 'M12'  : PMF NOT unimodal at q = 1/2 (exact '+ - +' pattern)
  2. for the M12 pairs: exact certificate at q = 1/4, 1/8, 1/16, 1/32, 1/64
       -> first q at which the PMF is unimodal => continuous-time density unimodal
  3. if still multimodal at q = 1/64: locate extrema of the continuous-time
     density in floating point and PROVE non-unimodality by the exact
     three-point certificate (cert_continuous.three_point_certificate).

Usage: python cert_pairs.py d N [N ...]
Output: data/pairs_d<d>_N<N>.json
"""
from __future__ import annotations
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository

import itertools
import json
import os
import sys
import time
from fractions import Fraction

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cert_unimodal import certify  # noqa: E402
from cert_continuous import three_point_certificate  # noqa: E402
from unilib import laplacian_q1, killed  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = _os.path.join(_R, 'data', 'msc_rigorous_unimodality')


def float_extrema(N, d, start, target):
    """float: sign-change times of g_1'(t) (continuous time, unit rate)."""
    L, sites, idx = laplacian_q1(N, d)
    Lk, keep = killed(L, idx, target)
    sig, V = np.linalg.eigh(Lk)
    pos = keep.index(idx[tuple(start)])
    c = V[pos, :] * sig * V.sum(axis=0)
    tg = np.concatenate([[0.0], np.geomspace(1e-3, 80.0 * N * N * d, 20000)])
    E = np.exp(-np.outer(tg, sig))
    fp = -(E * (c * sig)).sum(axis=1)
    scale = np.abs(E * (c * sig)).sum(axis=1)
    sg = np.where(np.abs(fp) < 1e-9 * scale, 0, np.sign(fp))
    keepi = sg != 0
    nz = sg[keepi]
    tt = tg[keepi]
    ii = np.where(nz[1:] != nz[:-1])[0]
    return [float(tt[i + 1]) for i in ii]


def classify(N, d):
    sites = list(itertools.product(range(N), repeat=d))
    res = []
    for a in sites:
        if any(ai > (N - 1) / 2 for ai in a) or list(a) != sorted(a):
            continue
        for s in sites:
            if s == a:
                continue
            r = certify(N, d, 1, 2, s, a)
            rec = dict(start=list(s), target=list(a), q12=r["unimodal"], T_cone12=r["T_cone"], mode12=r["mode"])
            if r["unimodal"] is True:
                rec["cont"] = "unimodal (thinning from q=1/2)"
            elif r["unimodal"] is False:
                rec["runs12"] = r["sign_runs"][:12]
                rec["cont"] = None
                for k in (4, 8, 16, 32, 64):
                    rk = certify(N, d, 1, k, s, a)
                    if rk["unimodal"] is True:
                        rec["cont"] = f"unimodal (thinning from q=1/{k})"
                        rec["first_unimodal_q"] = f"1/{k}"
                        break
                if rec["cont"] is None:
                    ext = float_extrema(N, d, s, a)
                    rec["float_extrema"] = ext
                    if len(ext) >= 3:
                        t1, t2, t3 = (Fraction(round(ext[0] * 8), 8), Fraction(round(ext[1] * 8), 8),
                                      Fraction(round(ext[2] * 8), 8))
                        cert = three_point_certificate(N, d, s, a, t1, t2, t3)
                        rec["three_point"] = cert
                        if cert["non_unimodal_proved"]:
                            rec["cont"] = "NOT unimodal (three-point certificate)"
                    if rec["cont"] is None:
                        rec["cont"] = "UNDECIDED"
            else:
                rec["cont"] = "UNDECIDED (no cone within Tmax)"
            res.append(rec)
    return res


if __name__ == "__main__":
    d = int(sys.argv[1])
    for N in [int(v) for v in sys.argv[2:]]:
        out = os.path.join(DATA, f"pairs_d{d}_N{N}.json")
        if os.path.exists(out):
            print("skip", out)
            continue
        t0 = time.time()
        res = classify(N, d)
        summ = {}
        for r in res:
            key = ("q12:" + str(r["q12"]), "cont:" + r["cont"].split(" (")[0])
            summ[str(key)] = summ.get(str(key), 0) + 1
        json.dump(dict(d=d, N=N, n_pairs=len(res), summary=summ, pairs=res), open(out, "w"))
        print("d", d, "N", N, "pairs", len(res), summ, "sec %.1f" % (time.time() - t0), flush=True)
        for r in res:
            if r["q12"] is not True:
                print("   ", r["start"], "->", r["target"], "q12", r["q12"], "| cont:", r["cont"], flush=True)
