"""exact_examples.py -- exact rational values quoted in note R4 (Section 8 and Section 4)."""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import json, os, sys
from fractions import Fraction
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from unilib import ExactStepper
HERE = os.path.dirname(os.path.abspath(__file__))
def pmf(N, d, qn, qd, s, a, T):
    st = ExactStepper(N, d, qn, qd, s, a); out = [Fraction(0)]; Dp = 1
    for _ in range(T):
        F = st.step(); Dp *= st.D; out.append(Fraction(F, Dp))
    return out, st.D
res = {}
def ex(name, N, d, qn, qd, s, a, ts, T, claim):
    f, D = pmf(N, d, qn, qd, s, a, T)
    vals = {str(t): dict(exact=str(f[t]) if f[t].denominator.bit_length() < 400 else None,
                         numerator_times_D_pow_t=str(f[t] * D ** t) if (f[t] * D ** t).numerator.bit_length() < 700 else "see cert files",
                         D=D, float=float(f[t])) for t in ts}
    ok = claim(f)
    res[name] = dict(d=d, N=N, q=f"{qn}/{qd}", start_0based=list(s), target_0based=list(a), values=vals, claim_verified_exactly=bool(ok))
    print(name, ok, {t: float(f[t]) for t in ts}, flush=True)
ex("1D_N4_q1", 4, 1, 1, 1, (0,), (3,), [3, 4, 5], 8, lambda f: f[3] > f[4] < f[5])
ex("3D_N3_facecentres_q1/2", 3, 3, 1, 2, (2, 1, 1), (0, 1, 1), [2, 6, 25], 40, lambda f: f[2] > f[6] < f[25])
ex("3D_N4_q1/2", 4, 3, 1, 2, (2, 2, 2), (1, 1, 1), [5, 20, 38], 60, lambda f: f[5] > f[20] < f[38])
ex("2D_N8_q1/2", 8, 2, 1, 2, (5, 5), (2, 3), [19, 46, 63], 80, lambda f: f[19] > f[46] < f[63])
ex("2D_N9_cornercorner_q1", 9, 2, 1, 1, (0, 0), (8, 8), [108, 109, 110, 111], 120, lambda f: f[108] > f[109] < f[110] and f[110] > f[111])
ex("2D_N3_M2C_q1/2_not_logconcave", 3, 2, 1, 2, (1, 1), (2, 2), [4, 5, 6], 12, lambda f: f[5] ** 2 < f[4] * f[6])
ex("2D_N4_q1/2_artefact", 4, 2, 1, 2, (1, 3), (1, 1), [4, 5, 8], 12, lambda f: f[4] > f[5] < f[8])
json.dump(res, open(_os.path.join(_R, 'data', 'msc_rigorous_unimodality', 'exact_examples.json'), "w"), indent=1)
