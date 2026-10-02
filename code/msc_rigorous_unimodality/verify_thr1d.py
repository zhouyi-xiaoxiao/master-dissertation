"""
verify_thr1d.py -- sanity checks Z1..Z11 of the following statements of note R4
(proofs/R4_unimodality/: Corollary 2.6(4), Lemma 3.1, Theorem 4.1(d),
Proposition 4.4 (N = 4 by hand), Proposition 7.3, Computer-assisted Theorem 7.4).

Everything here is written from the definitions and does NOT call cert_thr1d.certify
(except Z8, which audits the stored records, and Z7, which compares).
Output: data/verify_thr1d.json ; exit code 0 iff all checks pass.
"""
from __future__ import annotations
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository

import itertools
import json
import math
import os
import random
import sys
from fractions import Fraction as Fr

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = _os.path.join(_R, 'data', 'msc_rigorous_unimodality')
RES = {}
ALLOK = True


def report(name, ok, **info):
    global ALLOK
    ALLOK = ALLOK and bool(ok)
    RES[name] = dict(ok=bool(ok), **info)
    print(("PASS " if ok else "FAIL ") + name, json.dumps(info, default=str)[:400], flush=True)


# ---------------------------------------------------------------- definition-level PMF, d = 1
def pmf1d(N, x0, q, tmax):
    """f(0..tmax), walk on {1..N}, target N, start x0; exact Fractions, straight from the definition."""
    q = Fr(q)
    S = N - 1
    rho = [Fr(0)] * (S + 2)          # 1..S used
    rho[x0] = Fr(1)
    f = [Fr(0)]
    for _ in range(tmax):
        f.append(q / 2 * rho[S])
        new = [Fr(0)] * (S + 2)
        for x in range(1, S + 1):
            p = rho[x]
            if p == 0:
                continue
            # right
            if x + 1 <= S:
                new[x + 1] += q / 2 * p
            # (x+1 == N: absorbed, accounted in f)
            # left, cancelled at the wall
            if x - 1 >= 1:
                new[x - 1] += q / 2 * p
            else:
                new[x] += q / 2 * p
            new[x] += (1 - q) * p
        rho = new
    return f


def diffsigns_int(N, x0, qn, qd, tmax):
    """signs of f(j)-f(j-1), j=1..tmax, exact integers (own scaling: common denominator (2 qd)^t)."""
    S = N - 1
    Dn = 2 * qd
    R = [0] * (S + 2)
    R[x0] = 1
    Fprev = 0
    out = []
    for _ in range(tmax):
        F = qn * R[S]
        dl = F - Dn * Fprev
        out.append((dl > 0) - (dl < 0))
        Fprev = F
        new = [0] * (S + 2)
        for x in range(1, S + 1):
            p = R[x]
            if not p:
                continue
            if x + 1 <= S:
                new[x + 1] += qn * p
            if x - 1 >= 1:
                new[x - 1] += qn * p
            else:
                new[x] += qn * p
            new[x] += 2 * (qd - qn) * p
        R = new
    return out


def pattern(signs):
    nz = [s for s in signs if s]
    runs = [k for k, _ in itertools.groupby(nz)]
    return runs


# ---------------------------------------------------------------- Z1, Z2: Proposition 7.3
def z1_z2():
    bad = 0
    n = 0
    for M in range(1, 11):
        N = 2 * M + 1
        for q in (Fr(1, 3), Fr(1, 2), Fr(2, 3), Fr(3, 4), Fr(4, 5), Fr(9, 10), Fr(1)):
            f = pmf1d(N, M + 1, q, M + 2)
            eta = 1 - q
            c0 = (q / 2) ** M
            ok = (all(f[t] == 0 for t in range(M)) and f[M] == c0 and f[M + 1] == c0 * M * eta
                  and f[M + 2] == c0 * (Fr(M * (M + 1), 2) * eta ** 2 + M * q * q / 4))
            psi = (2 * M + 3) * eta ** 2 - 6 * eta + 1
            ok = ok and (f[M + 2] - f[M + 1] == c0 * Fr(M, 4) * psi)
            ok = ok and ((f[M] > f[M + 1]) == (q > 1 - Fr(1, M)))
            n += 1
            bad += not ok
    report("Z1 closed forms of Proposition 7.3 (exact)", bad == 0, cases=n, violations=bad)
    # Z2: the dip above the elementary thresholds, grid of rational q
    bad = 0
    n = 0
    for M in range(1, 31):
        N = 2 * M + 1
        if M == 1:
            thr = Fr(4, 5)
        elif M == 2:
            thr = Fr(7734591, 10 ** 7)        # > (4+sqrt2)/7 = 0.77345908...
            assert (7 * thr - 4) ** 2 > 2
        else:
            thr = 1 - Fr(1, M)
        for k in range(1, 41):
            q = thr + (1 - thr) * Fr(k, 40)
            f = pmf1d(N, M + 1, q, M + 2)
            n += 1
            bad += not (f[M] > f[M + 1] < f[M + 2])
        # Psi_M(1/M) = (M-1)(M-3)/M^2
        assert (2 * M + 3) * Fr(1, M) ** 2 - 6 * Fr(1, M) + 1 == Fr((M - 1) * (M - 3), M * M)
    report("Z2 dip f(M)>f(M+1)<f(M+2) above the elementary threshold (exact, M<=30)", bad == 0, cases=n, violations=bad)


# ---------------------------------------------------------------- Z3: hand computations
def z3():
    # N = 4 end to end at q = 4/5
    Q5 = [[3, 2, 0], [2, 1, 2], [0, 2, 1]]
    R = [1, 0, 0]
    Rs = [R]
    for _ in range(5):
        R = [sum(Q5[i][j] * R[j] for j in range(3)) for i in range(3)]
        Rs.append(R)
    ok = Rs[2] == [13, 8, 4] and Rs[3] == [55, 42, 20] and Rs[4] == [249, 192, 104] and Rs[5] == [1131, 898, 488]
    ok = ok and all(Rs[5][i] <= 5 * Rs[4][i] for i in range(3))
    f = pmf1d(4, 1, Fr(4, 5), 60)
    ok = ok and f[1] == f[2] == 0 and f[3] == f[4] == Fr(8, 125) and f[5] == Fr(208, 3125) and f[6] == Fr(976, 15625)
    ok = ok and all(f[t + 1] <= f[t] for t in range(5, 60))
    # quadratic for N = 4:  Phi = 19/4 q^2 - 15/2 q + 3  has negative discriminant
    ok = ok and (Fr(15, 2) ** 2 - 4 * Fr(19, 4) * 3 < 0)
    # N = 3 centre -> corner at q = 4/5
    Q5 = [[3, 2], [2, 1]]
    R0 = [0, 1]
    R1 = [sum(Q5[i][j] * R0[j] for j in range(2)) for i in range(2)]
    R2 = [sum(Q5[i][j] * R1[j] for j in range(2)) for i in range(2)]
    ok = ok and R1 == [2, 1] and R2 == [8, 5] and all(R2[i] <= 5 * R1[i] for i in range(2))
    g = pmf1d(3, 2, Fr(4, 5), 60)
    ok = ok and g[1] == Fr(2, 5) and g[2] == g[3] == Fr(2, 25) and all(g[t + 1] <= g[t] for t in range(1, 60))
    g2 = pmf1d(3, 2, Fr(801, 1000), 4)
    ok = ok and g2[1] > g2[2] < g2[3]
    report("Z3 hand computations (N=4 end-to-end and N=3 centre->corner at q=4/5)", ok)


# ---------------------------------------------------------------- Z4: brackets, long windows
def z4():
    info = {}
    ok = True
    recs = {}
    p = _os.path.join(_R, 'data', 'msc_rigorous_unimodality', 'cert_thr1d_bisect.jsonl')
    for line in open(p):
        r = json.loads(line)
        recs[r["N"]] = r
    for N in list(range(5, 20)) + [21]:
        r = recs[N]
        W = 12 * N * N + 200
        lo = pattern(diffsigns_int(N, 1, r["lo"], r["den"], W))
        hi = pattern(diffsigns_int(N, 1, r["hi"], r["den"], W))
        bd = pattern(diffsigns_int(N, 1, 2 * N - 4, 2 * N - 3, W))
        good = lo == [1, -1] and hi[:3] == [1, -1, 1] and bd[:3] == [1, -1, 1] and Fr(r["hi"], r["den"]) < Fr(2 * N - 4, 2 * N - 3)
        ok = ok and good
        info[N] = [r["lo"], r["hi"], good]
    for N in (2, 3):
        W = 400
        ok = ok and pattern(diffsigns_int(N, 1, 1, 1, W)) == [1, -1]
    for N in (4, 20, 22, 23, 30, 40, 55):
        W = 12 * N * N + 200
        ok = ok and pattern(diffsigns_int(N, 1, 2 * N - 4, 2 * N - 3, W)) == [1, -1]
        # just above the bound: not unimodal
        ok = ok and pattern(diffsigns_int(N, 1, 2 * (2 * N - 4) + 1, 2 * (2 * N - 3), 3 * N))[:3] == [1, -1, 1]
    report("Z4 end-to-end brackets and bound on long exact windows (own integer code)", ok, brackets=info)
    # centre -> corner
    d = json.load(open(_os.path.join(_R, 'data', 'msc_rigorous_unimodality', 'cert_thr1d_m2c.json')))
    ok = True
    info = {}
    for r in d["brackets"]:
        N = r["N"]
        if N > 23:
            continue
        c = (N + 1) // 2
        W = 12 * N * N + 200
        lo = pattern(diffsigns_int(N, c, r["lo"], r["den"], W))
        hi = pattern(diffsigns_int(N, c, r["hi"], r["den"], W))
        good = lo in ([1, -1], [-1]) and (hi[:3] == [1, -1, 1] or hi[:2] == [-1, 1])
        ok = ok and good
        info[N] = [r["lo"], r["hi"], good]
    for N in (7, 9, 17, 21, 23, 25, 31):
        M = (N - 1) // 2
        W = 12 * N * N + 200
        pt = pattern(diffsigns_int(N, M + 1, M - 1, M, W))
        ok = ok and pt in ([1, -1], [-1])
    for N in (11, 13, 15, 19):
        M = (N - 1) // 2
        pt = pattern(diffsigns_int(N, M + 1, M - 1, M, 12 * N * N))
        ok = ok and len(pt) >= 3
    report("Z4b centre->corner brackets and elementary bound on long exact windows", ok, brackets=info)


# ---------------------------------------------------------------- Z5: N = 21 counter-example, and N = 11 centre
def z5():
    f = pmf1d(21, 1, Fr(38, 39), 146)
    ok = f[142] > f[143] < f[144]
    info = dict(rel_dip=float((f[142] - f[143]) / f[143]), rel_rise=float((f[144] - f[143]) / f[143]),
                tie_first=bool(f[20] == f[21]))
    g = pmf1d(11, 6, Fr(4, 5), 12)
    ok = ok and g[9] == Fr(1216, 78125) and g[10] == Fr(149952, 9765625) and g[11] == Fr(151104, 9765625) and g[9] > g[10] < g[11]
    g = pmf1d(5, 3, Fr(4, 5), 5)
    ok = ok and (g[2], g[3], g[4]) == (Fr(4, 25), Fr(8, 125), Fr(44, 625))
    g = pmf1d(7, 4, Fr(4, 5), 6)
    ok = ok and (g[3], g[4], g[5]) == (Fr(8, 125), Fr(24, 625), Fr(144, 3125))
    g = pmf1d(9, 5, Fr(4, 5), 7)
    ok = ok and (g[4], g[5], g[6]) == (Fr(16, 625), Fr(64, 3125), Fr(416, 15625))
    report("Z5 exact witnesses: N=21 end-to-end at 38/39; N=5,7,9,11 centre->corner at 4/5", ok, **info)


# ---------------------------------------------------------------- Z6: q*(5) = (4+sqrt2)/7, centre -> corner
def z6():
    import mpmath as mp
    import sympy as sp
    mp.mp.dps = 80
    q = (4 + mp.sqrt(2)) / 7

    def pm(qv, tmax):
        S = 4
        rho = [mp.mpf(0)] * 6
        rho[3] = mp.mpf(1)
        f = [mp.mpf(0)]
        for _ in range(tmax):
            f.append(qv / 2 * rho[S])
            new = [mp.mpf(0)] * 6
            for x in range(1, S + 1):
                p = rho[x]
                if x + 1 <= S:
                    new[x + 1] += qv / 2 * p
                if x - 1 >= 1:
                    new[x - 1] += qv / 2 * p
                else:
                    new[x] += qv / 2 * p
                new[x] += (1 - qv) * p
            rho = new
        return f
    f = pm(q, 400)
    ok = f[2] > f[3] and abs(f[3] - f[4]) < mp.mpf(10) ** (-70) and all(f[t + 1] < f[t] for t in range(4, 400))
    f2 = pm(q + mp.mpf(10) ** (-9), 6)
    ok = ok and f2[2] > f2[3] < f2[4]
    # symbolic: f(4) - f(3) vanishes at q = (4+sqrt2)/7
    qs = sp.symbols("q")
    eta = 1 - qs
    expr = sp.expand((qs / 2) ** 2 * (3 * eta ** 2 + 2 * qs ** 2 / 4) - (qs / 2) ** 2 * 2 * eta)
    val = sp.simplify(expr.subs(qs, (4 + sp.sqrt(2)) / 7))
    ok = ok and val == 0
    d = json.load(open(_os.path.join(_R, 'data', 'msc_rigorous_unimodality', 'cert_thr1d_m2c.json')))
    ok = ok and d["N5_sqrt2"]["unimodal"] is True and d["N5_sqrt2"]["signs"][:4] == [0, 1, -1, 0]
    report("Z6 q*(5) = (4+sqrt2)/7 for centre->corner (80 digits + symbolic tie + Z[sqrt2] record)", ok,
           f3_minus_f4=mp.nstr(f[3] - f[4], 5))


# ---------------------------------------------------------------- Z7: engines agree with definition-level code
def z7():
    sys.path.insert(0, HERE)
    from cert_thr1d import certify, certify_flint
    random.seed(7)
    bad = 0
    n = 0
    for _ in range(60):
        N = random.randint(2, 14)
        x0 = random.randint(1, N - 1)
        qd = random.randint(1, 12)
        qn = random.randint(1, qd)
        a = certify(N, x0, qn, qd)
        b = certify_flint(N, x0, qn, qd)
        W = max(a["steps"] + 5, 40)
        sg = diffsigns_int(N, x0, qn, qd, 30 * N * N + 100)
        pt = pattern(sg)
        truth_unimodal = pt in ([1, -1], [-1], [1])
        same = (a["verdict"], a["T_cone"], a["mode"], a["witness"]) == (b["verdict"], b["T_cone"], b["mode"], b["witness"])
        n += 1
        if not same or a["verdict"] is None or a["verdict"] != truth_unimodal:
            bad += 1
            continue
        if a["verdict"]:
            # after the cone time the exact differences are all <= 0
            T = a["T_cone"]
            if any(s > 0 for s in sg[T + 1:]):
                bad += 1
    report("Z7 both certificate engines vs definition-level long windows (60 random cases)", bad == 0, cases=n, violations=bad)


# ---------------------------------------------------------------- Z8: audit of the stored records
def z8():
    def load(name):
        out = {}
        p = os.path.join(DATA, name)
        if not os.path.exists(p):
            return out
        for line in open(p):
            r = json.loads(line)
            out[r["N"]] = r
        return out
    bf = load("cert_thr1d_boundf.jsonl")
    bp = load("cert_thr1d_bound.jsonl")
    mb = load("cert_thr1d_m2cbound.jsonl")
    Nmax = max(bf) if bf else 0
    complete = sorted(bf) == list(range(4, Nmax + 1))
    notuni = sorted(N for N, r in bf.items() if r["unimodal"] is False)
    uni = sorted(N for N, r in bf.items() if r["unimodal"] is True)
    ok = complete and notuni == list(range(5, 20)) + [21] and uni == [4, 20] + list(range(22, Nmax + 1))
    front = all(r["T_cone"] == r["mode"] - 1 for r in bf.values() if r["unimodal"])
    agree = all((bp[N]["unimodal"], bp[N]["T_cone"], bp[N]["mode"], bp[N]["witness"]) ==
                (bf[N]["unimodal"], bf[N]["T_cone"], bf[N]["mode"], bf[N]["witness"]) for N in bp if N in bf)
    ok = ok and agree and front
    Mmax = max(mb) if mb else 0
    mcomplete = sorted(mb) == list(range(7, Mmax + 1, 2))
    mnot = sorted(N for N, r in mb.items() if r["unimodal"] is False)
    ok = ok and mcomplete and mnot == [11, 13, 15, 19] and all(r["unimodal"] is not None for r in mb.values())
    report("Z8 audit of records (end-to-end bound, both engines; centre->corner bound)", ok,
           e2e_range=[4, Nmax], e2e_not_unimodal=notuni, python_engine_overlap=[min(bp) if bp else None, max(bp) if bp else None],
           engines_agree=agree, cone_at_mode_minus_1=front, m2c_range=[7, Mmax], m2c_not_unimodal=mnot,
           max_bits=max((r.get("bits") or 0) for r in bf.values()) if bf else None,
           max_steps=max(r["steps"] for r in bf.values()) if bf else None)


# ---------------------------------------------------------------- Z9: Theorem 4.1(d)
def box_Q(N, d, a):
    """dense killed matrix (2d) * P' as integer lists; sites = tuples != a."""
    sites = [x for x in itertools.product(range(N), repeat=d) if x != a]
    idx = {x: i for i, x in enumerate(sites)}
    n = len(sites)
    Mx = [[0] * n for _ in range(n)]
    for x in sites:
        i = idx[x]
        for ax in range(d):
            for s in (-1, 1):
                y = list(x)
                y[ax] += s
                y = tuple(y)
                if not (0 <= y[ax] < N):
                    Mx[i][i] += 1
                elif y != a:
                    Mx[i][idx[y]] += 1
    return sites, idx, Mx


def z9():
    import numpy as np
    random.seed(11)
    bad = 0
    n = 0
    minC = []
    for _ in range(30):
        d = random.choice((1, 2, 2, 3))
        N = random.randint(3, 6 if d == 1 else (5 if d == 2 else 3))
        while True:
            a = tuple(random.randrange(N) for _ in range(d))
            x0 = tuple(random.randrange(N) for _ in range(d))
            D = sum(abs(u - v) for u, v in zip(a, x0))
            if D >= 2:
                break
        sites, idx, Mx = box_Q(N, d, a)
        nn = len(sites)
        r = [sum(1 for ax in range(d) for s in (-1, 1)
                 if tuple(x[k] + (s if k == ax else 0) for k in range(d)) == a) for x in sites]
        # f_1(t) * (2d)^t = e_x0^T Mx^{t-1} r
        v = [0] * nn
        v[idx[x0]] = 1
        fl = []
        for t in range(1, D + 1):
            fl.append(sum(v[i] * r[i] for i in range(nn)))
            v = [sum(Mx[j][i] * v[j] for j in range(nn)) for i in range(nn)]     # row vector times M
        # number of shortest paths
        G = math.factorial(D)
        for u, w in zip(a, x0):
            G //= math.factorial(abs(u - w))
        ok = fl[0] == 0 and all(x == 0 for x in fl[:-1]) and fl[-1] == G
        # residues (float): some C_j < 0  (component of x0)
        A = np.eye(nn) - np.array(Mx, dtype=float) / (2 * d)
        w, V = np.linalg.eigh(A)
        rr = np.array(r, dtype=float) / (2 * d)
        C = V[idx[x0], :] * (V.T @ rr)
        # group equal eigenvalues
        Cg = []
        k = 0
        while k < nn:
            j = k
            s = 0.0
            while j < nn and abs(w[j] - w[k]) < 1e-9:
                s += C[j]
                j += 1
            Cg.append(s)
            k = j
        ok = ok and min(Cg) < -1e-12 and abs(sum(Cg)) < 1e-10      # sum C_j = f(1)/q = 0
        minC.append(min(Cg))
        n += 1
        bad += not ok
    report("Z9 Theorem 4.1(d): f(1)=0<f(D)=G(2d)^-D (exact), total mass sum C_j = 0, some C_j<0 (float)", bad == 0,
           cases=n, violations=bad, largest_min_residue=max(minC))


# ---------------------------------------------------------------- Z10: Lemma 3.1 (P'^m > 0, m = 2(n'-1))
def z10():
    bad = 0
    n = 0
    cases = [(1, N) for N in range(2, 8)] + [(2, 2), (2, 3), (2, 4), (3, 2)]
    for d, N in cases:
        for a in itertools.product(range(N), repeat=d):
            sites, idx, Mx = box_Q(N, d, a)
            # connected components
            nn = len(sites)
            seen = set()
            for s0 in range(nn):
                if s0 in seen:
                    continue
                comp = [s0]
                seen.add(s0)
                k = 0
                while k < len(comp):
                    i = comp[k]
                    k += 1
                    for j in range(nn):
                        if Mx[i][j] and j != i and j not in seen:
                            seen.add(j)
                            comp.append(j)
                nc = len(comp)
                if nc == 1:
                    ok = Mx[comp[0]][comp[0]] > 0
                else:
                    sub = [[Mx[i][j] for j in comp] for i in comp]
                    ok = any(sub[i][i] > 0 for i in range(nc))       # a wall site in the component
                    m = 2 * (nc - 1)
                    # boolean power
                    B = [[1 if sub[i][j] else 0 for j in range(nc)] for i in range(nc)]
                    Pw = [[1 if i == j else 0 for j in range(nc)] for i in range(nc)]
                    for _ in range(m):
                        Pw = [[1 if any(Pw[i][k2] and B[k2][j] for k2 in range(nc)) else 0 for j in range(nc)] for i in range(nc)]
                    ok = ok and all(all(row) for row in Pw)
                n += 1
                bad += not ok
    report("Z10 Lemma 3.1: every component has a wall site and P'^(2(n'-1)) > 0 entrywise", bad == 0, components=n, violations=bad)


# ---------------------------------------------------------------- Z11: Corollary 2.6(4)
def z11():
    random.seed(5)
    bad = 0
    for _ in range(3000):
        L = random.randint(1, 14)
        diffs = [random.choice((-1, 0, 1)) for _ in range(L)]
        nz = [s for s in diffs if s]
        if not nz or nz[-1] != -1:
            continue
        S = sum(1 for i in range(len(nz) - 1) if nz[i] != nz[i + 1])
        # local maxima: number of (+ -> -) changes, plus one if the first nonzero difference is -
        lm = sum(1 for i in range(len(nz) - 1) if nz[i] == 1 and nz[i + 1] == -1) + (1 if nz[0] == -1 else 0)
        bad += lm != -(-(S + 1) // 2)
    report("Z11 Corollary 2.6(4): number of local maxima = ceil((S+1)/2)", bad == 0, violations=bad)


# ---------------------------------------------------------------- Z12: Remark 7.5(1), shape of the failure
def z12():
    """For the exceptional N (end to end, q = (2N-4)/(2N-3)), exact window of 12N^2+200 steps.
    Checks: the nonzero signs of f(j)-f(j-1) are (+...+) then a block t_1 <= j <= m of strictly alternating
    single signs - + - + ... - + (k dips), then (- ... -) to the end of the window.  Reports k, the dip times,
    m (last increase) and the global mode (argmax of f over the window, exact)."""
    ok = True
    info = {}
    for N in list(range(5, 20)) + [21]:
        qn, qd = 2 * N - 4, 2 * N - 3
        W = 12 * N * N + 200
        sg = diffsigns_int(N, 1, qn, qd, W)
        js = [j + 1 for j, s in enumerate(sg) if s]
        m = max(j for j in js if sg[j - 1] > 0)
        dips = [j for j in js if sg[j - 1] < 0 and j < m]
        t1 = dips[0]
        block = list(range(t1, m + 1))
        alt = all(sg[j - 1] == (-1 if (j - t1) % 2 == 0 else 1) for j in block)
        before = all(sg[j - 1] >= 0 for j in range(1, t1))
        after = all(sg[j - 1] < 0 for j in range(m + 1, W + 1))
        # exact f up to m+1 (scaled): F_j = Dn^j f(j)
        S = N - 1
        Dn = 2 * qd
        R = [0] * (S + 2)
        R[1] = 1
        F = [0]
        for _ in range(m + 2):
            F.append(qn * R[S])
            new = [0] * (S + 2)
            for x in range(1, S + 1):
                pp = R[x]
                if not pp:
                    continue
                if x + 1 <= S:
                    new[x + 1] += qn * pp
                if x - 1 >= 1:
                    new[x - 1] += qn * pp
                else:
                    new[x] += qn * pp
                new[x] += 2 * (qd - qn) * pp
            R = new
        best = max(range(1, m + 2), key=lambda j: Fr(F[j], Dn ** j))
        good = alt and before and after and (m - t1) % 2 == 1
        ok = ok and good
        info[N] = dict(dips=len(dips), dip_times=dips, last_increase=m, global_mode=best, shape_ok=good)
    report("Z12 Remark 7.5(1): shape of the failure at the bound for the exceptional N (exact windows)", ok, cases=info)


if __name__ == "__main__":
    only = sys.argv[1:]
    tests = dict(z12=z1_z2, z3=z3, z4=z4, z5=z5, z6=z6, z7=z7, z8=z8, z9=z9, z10=z10, z11=z11, zr=z12)
    for k, fn in tests.items():
        if only and k not in only:
            continue
        fn()
    if not only:
        json.dump(RES, open(_os.path.join(_R, 'data', 'msc_rigorous_unimodality', 'verify_thr1d.json'), "w"), indent=1, default=str)
    print("ALL PASS" if ALLOK else "SOME FAILED")
    sys.exit(0 if ALLOK else 1)
