"""
cert_lc1d.py -- rigorous certificate of the ONE-DIMENSIONAL log-concavity statement of Section 9.2 of note R4
(Computer-assisted Theorem 9.7: tail Lemma 9.5 + finite window), for one value of the laziness parameter y and a range of sizes.

Objects.  y >= 0;  two families of m x m Jacobi matrices J = y I + T:
  E2E  (end to end, N = m+1 sites of the reflecting path):  T = Adj(P_m),
        eigenvalues  lam_k = y + 2 cos(k pi/(m+1)),  weights c_k = (-1)^(k-1) (2/(m+1)) sin^2(k pi/(m+1)).
  FOLD (end to centre, N = 2m+1 odd, folded chain with m+1 states):  T = Adj(P_m) - e_m e_m^T,
        eigenvalues  lam_k = y + 2 cos(2 k pi/(2m+1)),  weights c_k = (-1)^(k-1) (4/(2m+1)) sin(phi_k) sin(phi_k/2),
        phi_k = 2 k pi/(2m+1).
  W_n := (J^n)(1,m) = sum_k c_k lam_k^n   (n >= 0;  W_n = 0 for n < m-1, W_{m-1} = 1).
  delta h_q(n) = K_q^{n+1}(1,N) - K_q^n(1,N)  [resp. K_q^{n+1}(1,c) - K_q^n(1,c)]  =  (q/2)^{n+1} * W_n,  y = 2(1-q)/q.
  Claim certified:   C_n := W_n^2 - W_{n-1} W_{n+1} > 0  for all n >= m   (">= 0, = 0 only at n = m" in threshold mode).

Method.
  (T) Tail (Lemma 9.5): C_n = - sum_{k<l} c_k c_l (lam_k lam_l)^(n-1) (lam_k - lam_l)^2.  With
        L_n = |c_1 c_2| (lam_1 lam_2)^(n-1) (lam_1-lam_2)^2,   Tot_n = sum_{k<l} |c_k c_l| |lam_k lam_l|^(n-1) (lam_k-lam_l)^2,
      if lam_1 > lam_2 > max_{k>=3} |lam_k|, c_1 > 0 > c_2 and 2 L_{n0} > Tot_{n0}, then C_n > 0 for all n >= n0.
      Tot_n = (sum p)(sum p lam^2) - (sum p lam)^2 with p_k = |c_k| |lam_k|^(n-1).   Verified in Arb ball arithmetic.
  (W) Window m <= n < n0: C_n evaluated from the spectral sum in Arb ball arithmetic (normalised by lam_1^n);
      the certificate requires the ball of C_n / lam_1^(2n) to be strictly positive.
  (P) Positivity: W_n > 0 is certified for m-1 <= n < n0 (Arb balls) and for n >= n0 by
      c_1 lam_1^n0 > sum_{k>=2} |c_k| |lam_k|^n0  (field "tail_positivity").
      [The files cert_lc1d_E2E_1-2_{a,b,c}.jsonl were produced before (P) was added; for E2E positivity is automatic
       because J is entrywise nonnegative.]
  (X) Cross-check: for n <= n_x the numbers W_n are ALSO computed exactly (integers for rational y, Z[sqrt(D)] in
      threshold mode) by the matrix recursion; the exact value must lie in the Arb ball and the exact sign of C_n
      must be > 0 (or = 0 at n = m in threshold mode).

Usage:  python cert_lc1d.py GEOM YMODE M_LO M_HI [STEP] [OUTFILE]
        GEOM in {E2E, FOLD};  YMODE = 'p/r' (e.g. 1/2) or 'thr' (threshold: first inequality is an equality)
Output: one JSON line per m appended to OUTFILE (default data/cert_lc1d_<GEOM>_<YMODE>.jsonl).
"""
from __future__ import annotations
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository

import json
import math
import os
import sys
import time
from fractions import Fraction

from flint import arb, ctx

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = _os.path.join(_R, 'data', 'msc_rigorous_unimodality')


# ----------------------------------------------------------------------------------------------------------
# exact arithmetic in Z[sqrt(D)]  (elements = pairs (a, b) meaning a + b sqrt(D));  D = 0 means plain integers
# ----------------------------------------------------------------------------------------------------------
def zs_sign(a, b, D):
    """exact sign of a + b sqrt(D), D >= 2 not a perfect square (or b == 0)"""
    if b == 0 or D == 0:
        return (a > 0) - (a < 0)
    if a == 0:
        return (b > 0) - (b < 0)
    sa, sb = (a > 0) - (a < 0), (b > 0) - (b < 0)
    if sa == sb:
        return sa
    # opposite signs: compare a^2 with b^2 D
    lhs, rhs = a * a, b * b * D
    if lhs == rhs:
        return 0
    return sa if lhs > rhs else sb


def zs_mul(x, y, D):
    return (x[0] * y[0] + D * x[1] * y[1], x[0] * y[1] + x[1] * y[0])


class ExactJ:
    """exact powers of the scaled matrix  S*J = alpha I + S*T  applied to e_1, entries in Z[sqrt(D)].

    rational y = p/r :  S = r, alpha = p, D = 0.
    E2E threshold  y = sqrt(2/m):            S = m,  alpha = sqrt(2m)          -> D = 2m,   alpha = (0, 1)
    FOLD threshold y = (1+sqrt(1+2m))/m:     S = m,  alpha = 1 + sqrt(1+2m)    -> D = 1+2m, alpha = (1, 1)
    If D is a perfect square the element is collapsed to integers.
    """

    def __init__(self, geom, m, ymode):
        self.geom, self.m = geom, m
        if ymode == "thr":
            self.S = m
            if geom == "E2E":
                D, al = 2 * m, (0, 1)
            else:
                D, al = 1 + 2 * m, (1, 1)
            s = math.isqrt(D)
            if s * s == D:
                al = (al[0] + al[1] * s, 0)
                D = 0
            self.D, self.alpha = D, al
        else:
            p, r = ymode
            self.S, self.D, self.alpha = r, 0, (p, 0)
        self.v = [(0, 0)] * m
        self.v[0] = (1, 0)
        self.n = 0

    def step(self):
        m, S, D, al = self.m, self.S, self.D, self.alpha
        v = self.v
        new = []
        for x in range(m):
            a, b = v[x]
            # alpha * v[x]
            if D:
                ra, rb = al[0] * a + D * al[1] * b, al[0] * b + al[1] * a
            else:
                ra, rb = al[0] * a, 0
            sa = sb = 0
            if x > 0:
                sa += v[x - 1][0]; sb += v[x - 1][1]
            if x < m - 1:
                sa += v[x + 1][0]; sb += v[x + 1][1]
            if self.geom == "FOLD" and x == m - 1:
                sa -= a; sb -= b
            new.append((ra + S * sa, rb + S * sb))
        self.v = new
        self.n += 1

    def W(self):
        return self.v[self.m - 1]


# ----------------------------------------------------------------------------------------------------------
def spectrum(geom, m, ymode):
    """returns (y, lam[], c[]) as arb at current precision"""
    pi = arb.pi()
    if geom == "E2E":
        th = [pi * k / (m + 1) for k in range(1, m + 1)]
        base = [2 * t.cos() for t in th]
        c = [(2 * t.sin() ** 2) / (m + 1) * (1 if k % 2 == 0 else -1) for k, t in enumerate(th)]
        if ymode == "thr":
            y = (arb(2) / m).sqrt()
        else:
            y = arb(ymode[0]) / ymode[1]
    elif geom == "FOLD":
        N = 2 * m + 1
        ph = [2 * pi * k / N for k in range(1, m + 1)]
        base = [2 * t.cos() for t in ph]
        c = [4 * t.sin() * (t / 2).sin() / N * (1 if k % 2 == 0 else -1) for k, t in enumerate(ph)]
        if ymode == "thr":
            y = (1 + arb(1 + 2 * m).sqrt()) / m
        else:
            y = arb(ymode[0]) / ymode[1]
    else:
        raise ValueError(geom)
    lam = [y + b for b in base]
    return y, lam, c


def ipow(x, n):
    """x^n for an integer n >= 0 by binary powering (valid for balls containing 0, unlike arb.__pow__)"""
    result = arb(1)
    base = x
    while n:
        if n & 1:
            result = result * base
        n >>= 1
        if n:
            base = base * base
    return result


def tail_ok(lam, c, n):
    """certified test of  2 L_n > Tot_n  (Lemma 9.5).  Normalised by (lam_1 lam_2)^(n-1)."""
    l1, l2 = lam[0], lam[1]
    den = (l1 * l2).sqrt()  # p_k normalised: |c_k| (|lam_k|/sqrt(l1 l2))^(n-1)
    p = [abs(ck) * ipow(abs(lk) / den, n - 1) for ck, lk in zip(c, lam)]
    s0 = arb(0); s1 = arb(0); s2 = arb(0)
    for pk, lk in zip(p, lam):
        s0 += pk; s1 += pk * lk; s2 += pk * lk * lk
    tot = s0 * s2 - s1 * s1
    L = p[0] * p[1] * (l1 - l2) ** 2
    return bool(2 * L - tot > 0)


def certify(geom, m, ymode, verbose=False):
    t0 = time.time()
    thr = ymode == "thr"
    prec = int(1.7 * m) + 320
    ctx.prec = prec
    y, lam, c = spectrum(geom, m, ymode)
    rec = dict(geom=geom, m=m, N=(m + 1 if geom == "E2E" else 2 * m + 1),
               ymode=("thr" if thr else "%d/%d" % ymode), y=float(y.mid()), prec=prec)
    # q corresponding to y:  y = 2(1-q)/q  ->  q = 2/(2+y)
    rec["q"] = float((2 / (2 + y)).mid())
    # ---- hypotheses of the tail lemma
    hyp = bool(lam[0] > lam[1]) and bool(lam[1] > 0) and bool(c[0] > 0) and bool(c[1] < 0)
    for k in range(2, m):
        hyp = hyp and bool(lam[1] > abs(lam[k]))
    rec["tail_hypotheses"] = hyp
    if not hyp:
        rec["status"] = "FAIL: tail hypotheses"
        return rec
    # ---- find n0 with certified 2 L_n0 > Tot_n0
    lo = m + 1
    hi = max(m + 1, 2 * m)
    while not tail_ok(lam, c, hi):
        lo = hi
        hi *= 2
        if hi > 400 * (m + 2) ** 2:
            rec["status"] = "FAIL: no tail time found"
            return rec
    while hi - lo > max(1, lo // 200):       # coarse bisection is enough (any certified n0 works)
        mid = (lo + hi) // 2
        if tail_ok(lam, c, mid):
            hi = mid
        else:
            lo = mid
    n0 = hi
    assert tail_ok(lam, c, n0)
    rec["n0"] = n0
    # positivity of W_n for n >= n0:  c_1 lam_1^n > sum_{k>=2} |c_k| |lam_k|^n  at n = n0 (then for all larger n)
    rest = arb(0)
    for ck, lk in zip(c[1:], lam[1:]):
        rest += abs(ck) * ipow(abs(lk) / lam[0], n0)
    rec["tail_positivity"] = bool(c[0] - rest > 0)
    rec["n0_over_N2"] = n0 / rec["N"] ** 2
    # ---- window: n = m .. n0-1, spectral sums in Arb (normalised by lam_1^n)
    # W'_n := W_n / lam_1^n = sum_k c_k (lam_k/lam_1)^n.   All |lam_k/lam_1| <= 1 (tail hypotheses).
    # Adaptive evaluation (rigorous): at "checkpoints" the working precision is re-chosen and modes k whose
    # term is negligible are DROPPED; a dropped mode contributes at most its bound at the time of dropping for all
    # later n (|ratio| <= 1), and the sum E of these bounds is added to every later W'_n as a ball [-E, E].
    ctx.prec = prec
    ratio_full = [lk / lam[0] for lk in lam]
    state = dict(active=list(range(m)), E=arb(0), prec=prec)

    def reinit(top):
        """p_k = c_k ratio_k^top for the active modes at the current precision"""
        return [c[k] * ipow(ratio_full[k], top) for k in state["active"]]

    def total(pv):
        s = arb(0)
        for a in pv:
            s += a
        if state["E"] != 0:
            s = s + state["E"] * arb(0, 1)
        return s

    def checkpoint(top, pv, Wtop):
        """choose precision and prune; returns new p-vector (for index top) at the new precision"""
        ctx.prec = prec
        absW = abs(Wtop)
        lowW = absW.lower()
        if not (lowW > 0):
            return pv                       # cannot prune safely; keep everything
        keep = []; maxp = arb(0)
        E = state["E"]
        thresh = lowW * arb(2) ** (-240)
        for k, pk in zip(state["active"], pv):
            up = abs(pk).upper()
            if up < thresh:
                E = E + up
            else:
                keep.append(k)
                if up > maxp:
                    maxp = up
        state["active"] = keep
        state["E"] = E
        # precision: cancellation bits + 320
        ratio_c = (maxp * len(keep) + arb(2) ** (-100000)) / lowW      # heuristic only (choice of precision)
        bits = max(0.0, float((ratio_c.log() / arb(2).log()).mid()))
        newprec = min(prec, int(bits) + 320)
        state["prec"] = newprec
        ctx.prec = newprec
        return reinit(top)

    ratio = None

    def advance(pv):
        return [a * ratio_full[k] for a, k in zip(pv, state["active"])]

    p = reinit(m - 1)
    Wm1 = total(p)                       # W'_{m-1}
    p = advance(p); W0 = total(p)        # W'_m
    p = advance(p); Wp1 = total(p)       # W'_{m+1}
    n = m
    next_cp = 2 * m + 8                  # first checkpoint (index of the newest W, i.e. n+1)
    n_checkpoints = 0
    min_active = m
    # exact cross-check machinery
    n_x = min(n0, 3 * m + 60) if m <= 1200 else min(n0, m + 1500)
    ex = ExactJ(geom, m, ymode)
    for _ in range(m - 1):
        ex.step()
    eW = [ex.W()]                        # exact scaled W at n = m-1 :  S^(n) W_n
    ex.step(); eW.append(ex.W())
    ex.step(); eW.append(ex.W())         # eW = [W_{n-1}, W_n, W_{n+1}] (scaled by S^{n-1}, S^n, S^{n+1})
    D, S = ex.D, ex.S
    sqrtD = arb(D).sqrt() if D else arb(0)
    lamS = lam[0] * S

    min_margin = None; arg_margin = None
    n_nonpos = 0 if bool(Wm1 > 0) else 1          # positivity of W'_n, n = m-1 .. n0-1 (certified balls)
    n_fail = 0; first_fail = None
    n_zero_exact = 0
    cross_ok = True; cross_n = 0
    equality_at_m = None
    while n < n0:
        C = W0 * W0 - Wm1 * Wp1
        certified = bool(C > 0)
        if not bool(W0 > 0):
            n_nonpos += 1
        if n <= n_x:
            # exact value of W_n in ball?   W'_n = (a + b sqrtD) / (S lam_1)^n
            a, b = eW[1]
            exact_arb = (arb(a) + arb(b) * sqrtD) / ipow(lamS, n)
            if not exact_arb.overlaps(W0):
                cross_ok = False
            # exact sign of C_n
            sq = zs_mul(eW[1], eW[1], D)
            pr = zs_mul(eW[0], eW[2], D)
            sgn = zs_sign(sq[0] - pr[0], sq[1] - pr[1], D)
            if sgn == 0:
                n_zero_exact += 1
                if n == m:
                    equality_at_m = True
                    certified = True     # exact equality, allowed only at n = m (threshold mode)
                else:
                    cross_ok = False
            elif sgn < 0:
                cross_ok = False
            else:
                if not certified:
                    # exact arithmetic decides (ball too wide); count as certified by the exact computation
                    certified = True
            cross_n = n
            ex.step()
            eW = [eW[1], eW[2], ex.W()]
        if not certified:
            n_fail += 1
            if first_fail is None:
                first_fail = n
        else:
            if not (thr and n == m):
                mg = float((C / (W0 * W0)).mid())
                if min_margin is None or mg < min_margin:
                    min_margin, arg_margin = mg, n
        # advance
        if n + 1 >= next_cp:
            p = checkpoint(n + 1, p, Wp1)
            n_checkpoints += 1
            min_active = min(min_active, len(p))
            next_cp = int(1.25 * (n + 1)) + 8
        p = advance(p)
        Wm1, W0, Wp1 = W0, Wp1, total(p)
        n += 1
    ctx.prec = prec
    rec.update(window=[m, n0 - 1], n_checked=n0 - m, min_rel_margin=min_margin, argmin_n=arg_margin,
               n_not_certified=n_fail, first_not_certified=first_fail,
               exact_crosscheck_upto=cross_n, exact_crosscheck_ok=cross_ok,
               exact_equalities=n_zero_exact, equality_at_first=bool(equality_at_m),
               n_not_certified_positive=n_nonpos,
               n_checkpoints=n_checkpoints, min_active_modes=min_active, final_prec=state["prec"],
               dropped_bound=float(state["E"].upper()) if state["E"] != 0 else 0.0,
               seconds=round(time.time() - t0, 2))
    if thr:
        ok = (n_fail == 0) and cross_ok and n_zero_exact == 1 and bool(equality_at_m)
    else:
        ok = (n_fail == 0) and cross_ok and n_zero_exact == 0
    ok = ok and n_nonpos == 0 and rec["tail_positivity"]
    rec["status"] = "LOGCONCAVE (certified)" if ok else "NOT CERTIFIED"
    return rec


def main():
    geom = sys.argv[1]
    ym = sys.argv[2]
    if ym == "thr":
        ymode = "thr"; tag = "thr"
    else:
        fr = Fraction(ym)
        ymode = (fr.numerator, fr.denominator); tag = "%d-%d" % ymode
    lo, hi = int(sys.argv[3]), int(sys.argv[4])
    step = int(sys.argv[5]) if len(sys.argv) > 5 and sys.argv[5].isdigit() else 1
    out = sys.argv[-1] if sys.argv[-1].endswith(".jsonl") else os.path.join(DATA, "cert_lc1d_%s_%s.jsonl" % (geom, tag))
    for m in range(lo, hi + 1, step):
        rec = certify(geom, m, ymode)
        with open(out, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
