"""check_ext_lemmas.py -- numerical sanity checks of the lemmas of Section 'A second engine and extended ranges'.
None of these checks is part of a proof.  Output: certs/ext_lemma_checks.json  (exit status 1 on any failure).

 A  Lemma 'time of the first possible passage': v_t(y) > 0  <=>  t - 1 >= |y|_1  (exact rationals, small cases).
 B  Lemma 'symmetry reduction': the reduced and the full C program give identical F^-_t, F^+_t, relf_t, T, t_sw;
    the full final vectors are symmetric and their restriction to the sorted sites is the reduced vector.
 C  Lemma 'integer routines': the two division identities on random and extreme W (Python integers).
 D  Lemma 'floating enclosures' (a): fsum against exact rationals on random terms: direction of rounding,
    zero iff sum zero, format, exponent window [E-4, E+5], relative error < 2^-60.
 E  Proposition 'soundness of the C certificates' end to end against exact rational arithmetic on small cases:
    F^-_t <= 2^P sigma_t <= F^+_t for all t <= T+1; every early relation relf_t is true; '=' only for exact zeros;
    v_{T+1} < v_T exactly at every site (T is a tail time); t_0 = d(N-1); gamma bound valid.
 F  Proposition 'one dimension, q close to 1: not unimodal, for all N >= 4': the closed forms of f(N-1), f(N), f(N+1)
    against exact rational iteration for several N and q; the formula for min G and its positivity for n < 400.
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import itertools, json, os, random, struct, subprocess, sys
from fractions import Fraction
from math import gcd
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = _os.path.join(_R, 'proofs', 'R5_certified_computations')
WORK = _os.path.join(_R, 'out', 'work', 'msc_rigorous_certified')
sys.path.insert(0, HERE)
import c128_ref, c128_post

random.seed(20261001)
res, bad = {}, 0


def record(name, ok, detail):
    global bad
    res[name] = dict(ok=bool(ok), **detail)
    bad += not ok
    print(("OK    " if ok else "FAIL  ") + name, detail, flush=True)


def chain(d, N, q):
    """sites of Omega', exact Q as dict of lists (y -> [(y', prob)]), neighbours of a."""
    a = (N - 1,) * d
    sites = [x for x in itertools.product(range(N), repeat=d) if x != a]
    Q = {}
    for y in sites:
        row = {y: 1 - q} if q < 1 else {}
        for k in range(d):
            for s in (-1, 1):
                z = list(y)
                z[k] += s
                z = tuple(z)
                if not all(0 <= c <= N - 1 for c in z):
                    row[y] = row.get(y, 0) + q / (2 * d)
                elif z != a:
                    row[z] = row.get(z, 0) + q / (2 * d)
        Q[y] = list(row.items())
    nb = [tuple(N - 1 - (j == k) for j in range(d)) for k in range(d)]
    return sites, Q, nb


def step(Q, v):
    return {y: sum(p * v[z] for z, p in row) for y, row in Q.items()}       # Q symmetric: (Qv)(y) = sum_z Q(y,z) v(z)


# ------------------------------------------------------------------ A
cases_A = [(1, 6), (1, 9), (2, 3), (2, 5), (3, 2), (3, 3)]
okA, nA = True, 0
for q in (Fraction(4, 5), Fraction(1, 2), Fraction(1)):
    for d, N in cases_A:
        sites, Q, nb = chain(d, N, q)
        v = {y: Fraction(int(y == (0,) * d)) for y in sites}
        for t in range(1, d * (N - 1) + 5):
            for y in sites:
                okA &= (v[y] > 0) == (t - 1 >= sum(y))
                nA += 1
            okA &= (sum(v[y] for y in nb) > 0) == (t >= d * (N - 1))
            v = step(Q, v)
record("A first passage / supports", okA, dict(site_time_pairs=nA, cases=len(cases_A) * 3))

# ------------------------------------------------------------------ B
def run_c(d, N, qn, qd, red):
    p = _os.path.join(_R, 'out', 'work', 'msc_rigorous_certified', 'chk_ext.bin')
    subprocess.run([_os.path.join(_R, 'out', 'work', 'msc_rigorous_certified', 'fp128'), str(d), str(N), str(qn), str(qd), str(red), p], check=True, stdout=subprocess.DEVNULL)
    r = c128_post.read_bin(p)
    os.remove(p)
    return r


okB, nB = True, 0
for (d, N, qn, qd) in [(2, 9, 4, 5), (2, 30, 4, 5), (2, 17, 1, 1), (2, 24, 1, 2), (3, 6, 4, 5), (3, 12, 4, 5), (3, 9, 1, 2), (3, 8, 1, 1)]:
    r, f = run_c(d, N, qn, qd, 1), run_c(d, N, qn, qd, 0)
    same = all(r[k] == f[k] for k in ("FL", "FU", "relf", "T", "tsw"))
    full = N ** d
    coords = lambda i: tuple((i // N ** k) % N for k in range(d))
    reps = [i for i in range(full) if c128_post.sorted_ok(i, N, d)]
    idx = {coords(i): j for j, i in enumerate(reps)}
    sym = all(f["L"][i] == r["L"][idx[tuple(sorted(coords(i)))]] and f["U"][i] == r["U"][idx[tuple(sorted(coords(i)))]] for i in range(full))
    okB &= same and sym
    nB += 1
record("B symmetry reduction (reduced = full)", okB, dict(cases=nB))

# ------------------------------------------------------------------ C
okC, nC = True, 0
for D in range(1, 16):
    Ws = [0, 1, D - 1, D, 2 ** 58 - 1, 2 ** 58, 2 ** 116 - 1, D * 2 ** 112, D * 2 ** 112 + D - 1] + [random.randrange(2 ** 116) for _ in range(2000)]
    for W in Ws:
        w1, w0 = W >> 58, W & (2 ** 58 - 1)
        q1, r1 = divmod(w1, D)
        c = (r1 << 58) | w0
        okC &= w1 < 2 ** 58 and c < 2 ** 62 and (q1 << 58) + c // D == W // D
        nC += 1
    Ws = [0, 1, 2 ** 126 - 1, 2 ** 125, 15 * 2 ** 121 + 14] + [random.randrange(2 ** 126) for _ in range(2000)]
    for W in Ws:
        a2, a1, a0 = W >> 84, (W >> 42) & (2 ** 42 - 1), W & (2 ** 42 - 1)
        q2, r = divmod(a2, D)
        t1 = (r << 42) | a1
        q1, r = divmod(t1, D)
        t0 = (r << 42) | a0
        q0 = t0 // D
        okC &= max(a2, t1, t0) < 2 ** 46 and (q2 << 84) + (q1 << 42) + q0 == W // D
        nC += 1
record("C division routines", okC, dict(tests=nC))

# ------------------------------------------------------------------ D
okD, nD, worst = True, 0, Fraction(0)
val = lambda x: Fraction(x[0]) * Fraction(2) ** x[1]


def rnd_float():
    k = random.random()
    if k < 0.15:
        return (0, 0)
    m = random.choice([2 ** 62, 2 ** 63 - 1, random.randrange(2 ** 62, 2 ** 63)])
    return (m, random.choice([-62, -63, -70, -200, -3000, random.randrange(-400, -50)]) + random.randrange(0, 80) * (random.random() < 0.5))


for _ in range(60000):
    k = random.randrange(1, 8)
    cs_ = [random.randrange(0, 4) for _ in range(k)]
    while sum(cs_) > 15:
        cs_[random.randrange(k)] = 0
    xs = [rnd_float() for _ in range(k)]
    Dv = random.randrange(1, 16)
    s = sum(c * val(x) for c, x in zip(cs_, xs)) / Dv
    lo, hi = c128_ref.fsum(list(zip(cs_, xs)), Dv, False), c128_ref.fsum(list(zip(cs_, xs)), Dv, True)
    okD &= val(lo) <= s <= val(hi) and (lo[0] == 0) == (s == 0) == (hi[0] == 0)
    for z in (lo, hi):
        okD &= z == (0, 0) or 2 ** 62 <= z[0] < 2 ** 63
    if s > 0:
        E = max(x[1] for c, x in zip(cs_, xs) if c and x[0])
        okD &= E - 4 <= lo[1] <= E + 5 and E - 4 <= hi[1] <= E + 5
        w = (val(hi) - val(lo)) / s
        worst = max(worst, w)
        okD &= w < Fraction(1, 2 ** 58)
        okD &= c128_ref.less(lo, hi) or lo == hi
    nD += 1
record("D fsum against exact rationals", okD, dict(tests=nD, max_relative_width=float(worst)))

# ------------------------------------------------------------------ E
okE, nE, detail = True, 0, []
for (qn, qd) in ((4, 5), (1, 2), (1, 1)):
    q = Fraction(qn, qd)
    for d, N in [(1, 2), (1, 3), (1, 4), (1, 5), (1, 9), (1, 14), (2, 2), (2, 3), (2, 4), (2, 6), (2, 9), (3, 2), (3, 3), (3, 4)]:
        r = run_c(d, N, qn, qd, 1)
        T, P = r["T"], r["P"]
        sites, Q, nb = chain(d, N, q)
        v = {y: Fraction(int(y == (0,) * d)) for y in sites}
        sig, vs = [None], [None]
        for t in range(1, T + 3):
            sig.append(sum(v[y] for y in nb))
            vs.append(v)
            v = step(Q, v)
        ok = all(r["FL"][t - 1] <= sig[t] * 2 ** P <= r["FU"][t - 1] for t in range(1, T + 2))
        for t in range(1, T + 1):
            c = r["relf"][t - 1]
            ok &= (c == "?" or (c == "<" and sig[t] < sig[t + 1]) or (c == ">" and sig[t] > sig[t + 1])
                   or (c == "=" and sig[t] == 0 == sig[t + 1]))
        ok &= all(vs[T + 1][y] < vs[T][y] for y in sites)                           # T is a tail time
        ok &= next(t for t in range(1, T + 2) if r["FU"][t - 1] > 0) == d * (N - 1) == next(t for t in range(1, T + 2) if sig[t] > 0)
        gam = max(Fraction(r["U"][i], r["L"][i]) for i in range(r["n"]) if i != r["tgt"])
        ok &= gam < 1 and all(vs[T + 1][y] <= gam * vs[T][y] for y in sites)
        first_exact_tail = next(t for t in range(1, T + 1) if all(vs[t + 1][y] < vs[t][y] for y in sites))
        okE &= ok
        nE += 1
        detail.append(dict(d=d, N=N, q=f"{qn}/{qd}", T=T, first_exact_tail_time=first_exact_tail, ok=bool(ok)))
record("E soundness of the C data against exact rationals", okE,
       dict(cases=nE, cases_where_T_exceeds_exact_tail_time=[(x["d"], x["N"], x["q"]) for x in detail if x["T"] != x["first_exact_tail_time"]]))

# ------------------------------------------------------------------ F  Proposition 'one dimension, q close to 1: not unimodal'
okF, nF, minG = True, 0, None
for n in range(2, 400):                              # the closed form of min G and its positivity
    u = Fraction(1, 2 * n + 1)
    G = Fraction(n - 1, 4) - n * u + Fraction(2 * n * n + 5 * n + 1, 4) * u * u
    okF &= G == Fraction(n * (2 * n * n - 3 * n - 1), 2 * (2 * n + 1) ** 2) > 0 and Fraction(2 * n, 2 * n * n + 5 * n + 1) >= u
for N in (4, 5, 6, 7, 8, 9, 10, 12, 16):
    thr, n = Fraction(2 * N - 4, 2 * N - 3), N - 2
    for q in (Fraction(1), (thr + 1) / 2, thr + Fraction(1, 1000), thr, thr - Fraction(1, 1000), Fraction(1, 2)):
        sites, Q, nb = chain(1, N, q)
        v = {y: Fraction(int(y == (0,))) for y in sites}
        f = [None]
        for t in range(1, N + 3):
            f.append(q / 2 * v[nb[0]])
            v = step(Q, v)
        m, u = (q / 2) ** (N - 1), 1 - q
        r0, r = 1 - q / 2, 1 - q
        G = Fraction(n - 1, 4) - n * u + Fraction(2 * n * n + 5 * n + 1, 4) * u * u
        okF &= all(x == 0 for x in f[1:N - 1]) and f[N - 1] == m and f[N] == m * (r0 + n * r)
        okF &= f[N + 1] == m * (r0 * r0 + n * r0 * r + Fraction(n * (n + 1), 2) * r * r + n * (q / 2) ** 2) and f[N + 1] - f[N] == m * G
        okF &= (f[N - 1] > f[N]) == (q > thr)
        if q >= thr:
            okF &= f[N + 1] > f[N]
            minG = G if minG is None else min(minG, G)
        nF += 1
record("F non-unimodality in 1D for q > (2N-4)/(2N-3): exact f(N-1), f(N), f(N+1)", okF, dict(cases=nF, smallest_G_tested=float(minG)))

json.dump(dict(failures=bad, checks=res), open(_os.path.join(_R, 'data', 'msc_rigorous_certified', 'ext_lemma_checks.json'), "w"), indent=1)
print("ALL OK" if not bad else f"{bad} FAILURES")
sys.exit(1 if bad else 0)
