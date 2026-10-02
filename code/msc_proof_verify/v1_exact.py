"""V1: exact rational hitting times (built from the model rules) versus every closed
form in Theorems 1-3.  Checkpoints to results/v1_exact.json after every case."""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import json, time, random, sys, os
from fractions import Fraction
import mpmath as mp
import vlib

OUT = _os.path.join(_R, 'data', 'msc_proof_verify', 'v1_exact.json')
mp.mp.dps = 50
res = {'dps': 50, 'corner': [], 'q_independence': [], 'pairs_exhaustive': [], 'pairs_random': []}


def save():
    with open(OUT, 'w') as f:
        json.dump(res, f, indent=1)


def frac2mp(fr):
    return mp.mpf(fr.numerator) / mp.mpf(fr.denominator)


def rel(a, b):
    return float(abs(a - b) / abs(b))


t0 = time.time()
NMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 22
# ---- (A) corner-to-corner, q=1, N=2..NMAX; other q for N<=10
for N in range(2, NMAX + 1):
    tgt = (N - 1, N - 1)
    h = vlib.exact_hitting_times(N, Fraction(1), tgt)
    T = h[(0, 0)]
    Tm = frac2mp(T)
    row = {'N': N, 'qT_exact': f'{T.numerator}/{T.denominator}', 'qT_float': float(T),
           'rel_D': rel(vlib.qT_double_corner(N), Tm),
           'rel_form_a': rel(vlib.qT_form_a(N), Tm),
           'rel_S': rel(vlib.qT_S(N), Tm)}
    So, Se = vlib.half_sums(N)
    row['rel_Hodd'] = rel(8 * N * So - 2 * N * N, Tm)
    row['rel_Heven'] = rel(2 * N * N + 8 * N * Se, Tm)
    row['Sodd_minus_Seven_minus_N/2'] = float(abs(So - Se - mp.mpf(N) / 2))
    # symmetry T(s->a) for corner = same from the other direction (point reflection)
    row['elapsed'] = round(time.time() - t0, 1)
    res['corner'].append(row)
    save()
    print(row, flush=True)
    if N <= 10:
        for q in (Fraction(4, 5), Fraction(3, 10), Fraction(1, 7), Fraction(1, 97)):
            hq = vlib.exact_hitting_times(N, q, tgt)
            res['q_independence'].append({'N': N, 'q': str(q), 'q*T == T(q=1)': bool(q * hq[(0, 0)] == T)})
        save()

# ---- (B) all ordered pairs, N<=5 (q=1 and q=3/7): Theorem 1 (G) and Theorem 3
mp.mp.dps = 40
worstG = 0.0
worst3 = 0.0
count = 0
for N in range(2, 6):
    for q in (Fraction(1), Fraction(3, 7)):
        for a0 in range(N):
            for a1 in range(N):
                h = vlib.exact_hitting_times(N, q, (a0, a1))
                for s0 in range(N):
                    for s1 in range(N):
                        if (s0, s1) == (a0, a1):
                            continue
                        ex = frac2mp(q * h[(s0, s1)])
                        s = (s0 + 1, s1 + 1)
                        a = (a0 + 1, a1 + 1)
                        e3 = rel(vlib.qT_thm3(N, s, a), ex)
                        worst3 = max(worst3, e3)
                        if q == 1:
                            eG = rel(vlib.qT_double_general(N, s, a), ex)
                            worstG = max(worstG, eG)
                        count += 1
    res['pairs_exhaustive'] = {'N_max_done': N, 'cases': count, 'worst_rel_thm3': worst3, 'worst_rel_G': worstG}
    save()
    print('pairs N', N, count, worst3, worstG, flush=True)

# ---- (C) random pairs for larger N (incl. boundary/edge/interior, both orderings)
random.seed(424242)
for N in (6, 7, 9, 11, 14, 17):
    for rep in range(3):
        a = (random.randrange(N), random.randrange(N))
        h = vlib.exact_hitting_times(N, Fraction(1), a)
        w3 = 0.0
        wG = 0.0
        starts = random.sample([x for x in h if x != a], 6)
        for s in starts:
            ex = frac2mp(h[s])
            w3 = max(w3, rel(vlib.qT_thm3(N, (s[0] + 1, s[1] + 1), (a[0] + 1, a[1] + 1)), ex))
        s = starts[0]
        wG = rel(vlib.qT_double_general(N, (s[0] + 1, s[1] + 1), (a[0] + 1, a[1] + 1)), frac2mp(h[s]))
        res['pairs_random'].append({'N': N, 'target': a, 'starts': starts, 'worst_rel_thm3': w3, 'rel_G_one': wG})
        save()
        print('random', N, a, w3, wG, flush=True)
res['seconds'] = round(time.time() - t0, 1)
save()
print('done', res['seconds'])
