"""
cert_unimodal.py -- EXACT (integer-arithmetic) certificate of (non-)unimodality
of the first-passage PMF f(t), t >= 1, of the lazy walk on {0..N-1}^d with a
single absorbing target, for one rational activity q = qn/qd.

What is proved by one run (see note R4, Lemma 6.1 "cone certificate"):

  * R_t := D^t Q^t e_{x0} is an exact integer vector, F_t := D^t f(t) an exact
    integer (unilib.ExactStepper).
  * s_j := sign(f(j) - f(j-1)) = sign(F_j - D F_{j-1}),  j = 1..t  (f(0)=0).
  * CONE at time T:  R_{T+1} <= D R_T componentwise  (i.e. rho_{T+1} <= rho_T).
    Because Q >= 0 entrywise this propagates:  rho_{u+1} <= rho_u for all
    u >= T, hence f(u+2) <= f(u+1) for all u >= T.
  * Therefore, if the cone holds at T and the explicit signs s_1..s_{T+1} are
    of the form (>=0)*(<=0)*, the PMF is unimodal on its whole infinite support.
  * If the explicit signs contain a pattern  + ... - ... +  the PMF is NOT
    unimodal (this needs no cone).

Usage:
  python cert_unimodal.py GEO d qn qd N1 N2 ...      [--out file.jsonl] [--lc]
GEO in {CC, C2M, M2C} or "PAIR:x0_0,x0_1,..:a_0,a_1,.." (0-based coordinates).
Records are appended to data/cert_<GEO>_d<d>_q<qn>-<qd>.jsonl ; existing
records are skipped (checkpointing).
"""
from __future__ import annotations
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from unilib import ExactStepper  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = _os.path.join(_R, 'data', 'msc_rigorous_unimodality')


def geometry(geo, N, d):
    c = (N - 1) // 2
    if geo == "CC":
        return (0,) * d, (N - 1,) * d
    if geo == "C2M":
        assert N % 2 == 1
        return (0,) * d, (c,) * d
    if geo == "M2C":
        assert N % 2 == 1
        return (c,) * d, (N - 1,) * d
    if geo.startswith("PAIR:"):
        _, s, a = geo.split(":")
        return tuple(int(v) for v in s.split(",")), tuple(int(v) for v in a.split(","))
    raise ValueError(geo)


def certify(N, d, qn, qd, start, target, Tmax=None, check_lc=False):
    """Run the exact certificate.  Returns a dict (JSON-serialisable)."""
    t0 = time.time()
    st = ExactStepper(N, d, qn, qd, start, target)
    D = st.D
    if Tmax is None:
        Tmax = 400 * N * N * d * qd // qn + 1000
    # neighbours of the target (for the cheap pre-check)
    nbrs = []
    for ax in range(d):
        for dx in (-1, 1):
            y = list(target)
            y[ax] += dx
            if 0 <= y[ax] < N:
                nbrs.append(tuple(y))
    runs = []          # [sign, first_j, last_j]
    Fprev = 0          # F_0
    Fprev2 = None
    lc_ok = True
    lc_first_fail = None
    T_cone = None
    first_nonpos_after_pos = None
    seen_pos = False
    t = 0
    while t < Tmax:
        Rold = st.R
        F = st.step()
        t += 1
        diff = F - D * Fprev
        s = (diff > 0) - (diff < 0)
        if runs and runs[-1][0] == s:
            runs[-1][2] = t
        else:
            runs.append([s, t, t])
        if s > 0:
            seen_pos = True
        if check_lc and Fprev2 is not None and Fprev > 0:
            # f(t-1)^2 >= f(t-2) f(t)  <=>  F_{t-1}^2 >= F_{t-2} F_t
            if Fprev * Fprev < Fprev2 * F:
                if lc_ok:
                    lc_first_fail = t - 1
                lc_ok = False
        Fprev2, Fprev = Fprev, F
        # cone test at T = t-1 :  R_t <= D R_{t-1}
        ok = all(st.R[y] <= D * Rold[y] for y in nbrs)
        if ok and bool(np.all(st.R <= Rold * D)):
            T_cone = t - 1
            break
    nz = [r[0] for r in runs if r[0] != 0]
    changes = sum(1 for i in range(len(nz) - 1) if nz[i] != nz[i + 1])
    # explicit part is (>=0)*(<=0)*  <=>  nz is +...+ -...- (at most one change, + first)
    explicit_ok = changes == 0 or (changes == 1 and nz[0] > 0)
    unimodal = None
    if T_cone is not None and explicit_ok:
        unimodal = True
    elif changes >= 2 or (changes == 1 and nz[0] < 0 and seen_pos):
        unimodal = False
    # '+ - +' somewhere means non-unimodal irrespective of the cone
    pat = "".join("+" if v > 0 else "-" for v in nz)
    if "+-+" in pat.replace("++", "+").replace("--", "-") or _has_pmp(nz):
        unimodal = False
    # mode: last j with positive sign (first maximiser) when unimodal
    mode = None
    last_pos = [r for r in runs if r[0] > 0]
    if unimodal and last_pos:
        mode = last_pos[-1][2]
    elif unimodal:
        mode = 1
    rec = dict(
        d=d, N=N, q=f"{qn}/{qd}", start=list(start), target=list(target), D=D,
        steps=t, T_cone=T_cone, sign_runs=runs if len(runs) <= 60 else runs[:30] + [["..."]] + runs[-30:],
        n_sign_changes_explicit=changes, unimodal=unimodal, mode=mode,
        bits_last=int(Fprev.bit_length()) if Fprev else 0,
        seconds=round(time.time() - t0, 2),
    )
    if check_lc:
        rec["logconcave_on_explicit_window"] = lc_ok
        rec["lc_first_fail_t"] = lc_first_fail
    return rec


def _has_pmp(nz):
    """True iff the nonzero sign list contains + ... - ... + (two sign changes
    starting from +) -- proves non-unimodality."""
    state = 0
    for v in nz:
        if state == 0 and v > 0:
            state = 1
        elif state == 1 and v < 0:
            state = 2
        elif state == 2 and v > 0:
            return True
    return False


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = [a for a in sys.argv[1:] if a.startswith("--")]
    geo, d, qn, qd = args[0], int(args[1]), int(args[2]), int(args[3])
    Ns = [int(v) for v in args[4:]]
    check_lc = "--lc" in flags
    tag = geo.replace(":", "_").replace(",", "")
    out = os.path.join(DATA, f"cert_{tag}_d{d}_q{qn}-{qd}.jsonl")
    for fl in flags:
        if fl.startswith("--out="):
            out = fl.split("=", 1)[1]
    done = set()
    if os.path.exists(out):
        for line in open(out):
            try:
                done.add(json.loads(line)["N"])
            except Exception:
                pass
    for N in Ns:
        if N in done:
            continue
        start, target = geometry(geo, N, d)
        rec = certify(N, d, qn, qd, start, target, check_lc=check_lc)
        rec["geo"] = geo
        with open(out, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(geo, "d", d, "N", N, "q", rec["q"], "unimodal", rec["unimodal"], "mode", rec["mode"],
              "T_cone", rec["T_cone"], "changes", rec["n_sign_changes_explicit"],
              "lc", rec.get("logconcave_on_explicit_window"), "bits", rec["bits_last"], "sec", rec["seconds"], flush=True)


if __name__ == "__main__":
    main()
