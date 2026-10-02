"""
cert_continuous.py -- rigorous enclosure of the CONTINUOUS-time first-passage
density, and rigorous three-point certificates of non-unimodality.

Continuous-time walk with unit total jump rate (each direction at rate 1/(2d);
the density for activity q is g_q(t) = q g_1(q t)).  By uniformisation
(note R4, Theorem 2.5(b)):

    g_1(t) = sum_{n>=0} e^{-t} t^n / n!  * a(n),      a(n) := f_{q=1}(n+1),

with f_{q=1} the discrete PMF at q = 1, which is an exact rational
F_{n+1} / (2d)^{n+1}.  For rational t and a cut-off M > t - 2:

    P_M(t) := sum_{n<=M} t^n/n! a(n)                (exact Fraction)
    0 <= e^{t} g_1(t) - P_M(t) <= t^{M+1}/(M+1)! * 1/(1 - t/(M+2))   (a(n) <= 1).

To compare g_1(t1) and g_1(t2) we compare  P(t1) e^{t2-t1}  with  P(t2),
using exact rational lower/upper bounds for exp(delta), delta rational > 0:
    L_K = sum_{k<=K} delta^k/k!,   U_K = L_K + delta^{K+1}/(K+1)!/(1-delta/(K+2)).

Everything is exact rational arithmetic (fractions.Fraction); no floats enter a
decision.
"""
from __future__ import annotations
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository

import json
import os
import sys
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from unilib import ExactStepper  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = _os.path.join(_R, 'data', 'msc_rigorous_unimodality')


def a_sequence(N, d, start, target, M):
    """a(n) = f_{q=1}(n+1), n = 0..M, as exact Fractions."""
    st = ExactStepper(N, d, 1, 1, start, target)
    out = []
    Dp = 1
    for n in range(M + 1):
        F = st.step()
        Dp *= st.D
        out.append(Fraction(F, Dp))
    return out


def scaled_density_bounds(a, t):
    """Return (lo, hi) exact Fractions with lo <= e^t g_1(t) <= hi."""
    M = len(a) - 1
    assert M + 2 > t
    term = Fraction(1)
    P = Fraction(0)
    for n in range(M + 1):
        if n > 0:
            term = term * t / n
        P += term * a[n]
    tail = term * t / (M + 1) / (1 - Fraction(t) / (M + 2))
    return P, P + tail


def exp_bounds(delta, K=None):
    """exact rational bounds L <= exp(delta) <= U for rational delta >= 0."""
    delta = Fraction(delta)
    assert delta >= 0
    if K is None:
        K = int(delta) * 4 + 60
    term = Fraction(1)
    L = Fraction(1)
    for k in range(1, K + 1):
        term = term * delta / k
        L += term
    U = L + term * delta / (K + 1) / (1 - delta / (K + 2))
    return L, U


def strictly_greater(a, t_big, t_small):
    """Rigorous test  g_1(t_big) > g_1(t_small)  (returns True only if PROVED).
    Here t_big / t_small are just two times; 'big' refers to the value of g."""
    lo_b, hi_b = scaled_density_bounds(a, t_big)
    lo_s, hi_s = scaled_density_bounds(a, t_small)
    # g(t) = e^{-t} X(t),  X in [lo,hi].  Want e^{-tb} lo_b > e^{-ts} hi_s.
    if t_big <= t_small:
        # lo_b * e^{ts - tb} > hi_s  ; use lower bound of exp
        L, U = exp_bounds(Fraction(t_small) - Fraction(t_big))
        return lo_b * L > hi_s
    else:
        # lo_b > hi_s * e^{tb - ts} ; use upper bound of exp
        L, U = exp_bounds(Fraction(t_big) - Fraction(t_small))
        return lo_b > hi_s * U


def three_point_certificate(N, d, start, target, t1, t2, t3, M=None):
    """Prove g(t1) > g(t2) < g(t3) with t1 < t2 < t3 (rational)."""
    t1, t2, t3 = Fraction(t1), Fraction(t2), Fraction(t3)
    assert t1 < t2 < t3
    if M is None:
        M = int(t3) * 3 + 80
    a = a_sequence(N, d, start, target, M)
    ok1 = strictly_greater(a, t1, t2)
    ok3 = strictly_greater(a, t3, t2)
    vals = {}
    for nm, t in (("t1", t1), ("t2", t2), ("t3", t3)):
        lo, hi = scaled_density_bounds(a, t)
        import math
        vals[nm] = dict(t=str(t), g_float=float(lo) * math.exp(-float(t)),
                        rel_width=float((hi - lo) / hi) if hi > 0 else None)
    return dict(d=d, N=N, start=list(start), target=list(target), M=M,
                proved_g_t1_gt_g_t2=bool(ok1), proved_g_t3_gt_g_t2=bool(ok3),
                non_unimodal_proved=bool(ok1 and ok3), values=vals)


if __name__ == "__main__":
    # the catalogue of continuous-time counterexamples (times from float scan,
    # rounded to simple rationals)
    cases = [
        # d, N, start, target, t1, t2, t3
        (3, 3, (2, 1, 1), (0, 1, 1), Fraction(21, 10), Fraction(3), Fraction(13)),
        (3, 4, (2, 2, 2), (1, 1, 1), Fraction(13, 4), Fraction(23, 2), Fraction(19)),
        (2, 8, (5, 5), (2, 3), Fraction(21, 2), Fraction(49, 2), Fraction(59, 2)),
        (2, 9, (6, 4), (2, 3), Fraction(27, 2), Fraction(57, 2), Fraction(41)),
        (2, 9, (6, 3), (2, 4), Fraction(27, 2), Fraction(26), Fraction(87, 2)),
        (2, 9, (6, 4), (2, 4), Fraction(23, 2), Fraction(30), Fraction(87, 2)),
        (2, 10, (7, 6), (3, 4), Fraction(17), Fraction(65, 2), Fraction(52)),
    ]
    out = []
    for (d, N, s, a, t1, t2, t3) in cases:
        rec = three_point_certificate(N, d, s, a, t1, t2, t3)
        out.append(rec)
        print(json.dumps(rec), flush=True)
    json.dump(out, open(_os.path.join(_R, 'data', 'msc_rigorous_unimodality', 'continuous_counterexamples.json'), "w"), indent=1)
