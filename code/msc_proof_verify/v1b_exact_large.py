"""V1b: exact rational corner-to-corner MFPT for larger N (beyond the main implementation's N<=40)."""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import json, time, os
from fractions import Fraction
import mpmath as mp
import vlib
mp.mp.dps = 60
OUT = _os.path.join(_R, 'data', 'msc_proof_verify', 'v1b_exact_large.json')
rows = []
t0 = time.time()
for N in list(range(23, 33)) + [35, 38, 41, 44, 47, 50]:
    h = vlib.exact_hitting_times(N, Fraction(1), (N - 1, N - 1))
    T = h[(0, 0)]
    Tm = mp.mpf(T.numerator) / mp.mpf(T.denominator)
    So, Se = vlib.half_sums(N)
    r = {'N': N, 'qT': mp.nstr(Tm, 45), 'num_digits': len(str(T.numerator)),
         'rel_D': float(abs(vlib.qT_double_corner(N) - Tm) / Tm),
         'rel_form_a': float(abs(vlib.qT_form_a(N) - Tm) / Tm),
         'rel_S': float(abs(4 * N * (So + Se) - Tm) / Tm),
         'rel_Hodd': float(abs(8 * N * So - 2 * N * N - Tm) / Tm),
         'rel_Heven': float(abs(2 * N * N + 8 * N * Se - Tm) / Tm),
         # also check transposition symmetry and the neighbour-of-target values via Thm 3
         'rel_thm3_from_(1,N)': float(abs(vlib.qT_thm3(N, (1, N), (N, N)) - mp.mpf(h[(0, N - 1)].numerator) / h[(0, N - 1)].denominator) / Tm),
         'rel_thm3_from_centre': float(abs(vlib.qT_thm3(N, (N // 2 + 1, N // 2 + 1), (N, N)) - mp.mpf(h[(N // 2, N // 2)].numerator) / h[(N // 2, N // 2)].denominator) / Tm),
         'elapsed': round(time.time() - t0, 1)}
    rows.append(r)
    json.dump(rows, open(OUT, 'w'), indent=1)
    print(r, flush=True)
