"""Certified (Arb) signs of ALL residues for small boxes, corner to corner (Proposition 14.1); consistency checks of Theorem 5.10.

For each case the eigenvalues lambda_k = 1 - (1/d) sum_i cos(pi k_i/N) are grouped EXACTLY: with zeta = e^{i pi/N} (a primitive
2N-th root of unity, minimal polynomial Phi_{2N}), 2 sum_i cos(pi k_i/N) = sum_i (zeta^{k_i} + zeta^{2N-k_i}), so two eigenvalues are
equal iff the polynomials sum_i (x^{k_i} + x^{2N-k_i}) agree modulo Phi_{2N}(x) (exact integer arithmetic, sympy).  The distinct
values Lam_0 < ... < Lam_p are then separated in ball arithmetic,
each zero sigma_j of G is bracketed by a certified sign change (G is increasing between consecutive poles, Theorem 2.2(i)),
and r_j = -G_x(sigma_j)/(sigma_j G'(sigma_j)) is evaluated on the bracket.  Output: ../data/13_certified_residue_signs.json
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import os, sys, json, itertools, math
from collections import Counter, defaultdict
import numpy as np
from scipy.optimize import brentq
from flint import arb, ctx
import sympy as sp
ctx.prec = 300
pi = arb.pi()

def case(N, d):
    c = [(pi * k / N).cos() for k in range(N)]
    cf = [math.cos(math.pi * k / N) for k in range(N)]
    w1 = [(1 + c[k]) / 2 * (1 if k == 0 else 2) for k in range(N)]
    x = sp.symbols("x")
    Phi = sp.Poly(sp.cyclotomic_poly(2 * N, x), x)
    def exact_key(k):
        P = sp.Poly(sum(x ** j + x ** (2 * N - j) for j in k), x)
        return tuple(sp.rem(P, Phi).all_coeffs())
    groups = defaultdict(lambda: [None, arb(0), arb(0), set()])  # exact key -> [lam ball, weight, signed weight, parities of |k|]
    for k in itertools.combinations_with_replacement(range(N), d):
        cnt = Counter(k); mult = math.factorial(d)
        for v in cnt.values(): mult //= math.factorial(v)
        lam = sum((1 - c[j]) for j in k) / d
        key = exact_key(k)
        w = arb(mult)
        for j in k: w *= w1[j]
        g = groups[key]
        if g[0] is None: g[0] = lam
        else: assert g[0].overlaps(lam)          # consistency (equality is already exact by the key)
        g[1] += w; g[2] += w * (-1) ** sum(k); g[3].add(sum(k) % 2)
    keys = sorted(groups, key=lambda kk: float(groups[kk][0].mid()))
    Lam = [groups[k][0] for k in keys]; W = [groups[k][1] for k in keys]; SW = [groups[k][2] for k in keys]
    PAR = [frozenset(groups[k][3]) for k in keys]                # exact: the grouping of the eigenvalues is exact
    for i in range(len(Lam) - 1):
        assert Lam[i] < Lam[i + 1], "eigenvalues not separated"
    p = len(Lam) - 1
    Lf = np.array([float(x.mid()) for x in Lam]); Wf = np.array([float(x.mid()) for x in W])
    G = lambda s: sum(W[i] / (Lam[i] - s) for i in range(p + 1))
    Gx = lambda s: sum(SW[i] / (Lam[i] - s) for i in range(p + 1))
    dG = lambda s: sum(W[i] / (Lam[i] - s) ** 2 for i in range(p + 1))
    Gf = lambda s: np.sum(Wf / (Lf - s))
    signs = []; vals = []
    for j in range(p):
        gap = Lf[j + 1] - Lf[j]; e = 1e-9 * gap
        while Gf(Lf[j] + e) > 0: e /= 8
        lo = Lf[j] + e; e = 1e-9 * gap
        while Gf(Lf[j + 1] - e) < 0: e /= 8
        s = arb(brentq(Gf, lo, Lf[j + 1] - e, xtol=1e-300, rtol=1e-15))
        for _ in range(5):
            s = (s - G(s) / dG(s)).mid()
        eta = arb(2) ** (-200)
        s_lo, s_hi = (s * (1 - eta)).mid(), (s * (1 + eta)).mid()
        assert Lam[j] < s_lo and s_hi < Lam[j + 1] and G(s_lo) < 0 and G(s_hi) > 0, f"pole {j} not certified"
        S = s_lo.union(s_hi)
        r = -Gx(S) / (S * dG(S))
        if r > 0: signs.append("+")
        elif r < 0: signs.append("-")
        else: signs.append("?")
        vals.append(r.str(12))
    return p, "".join(signs), vals, PAR

out = []
# the list contains d = 3, N = 5, 6 and every (d, N) quoted in Observation 14.2 of note R2.  Theorem 5.10 is proved without
# computation; the assertions below are consistency checks of its statements against the certified signs.
CASES = ((2, 2), (3, 2), (4, 2), (5, 2), (6, 2), (7, 2), (8, 2), (10, 2), (11, 2), (12, 2),
         (2, 3), (3, 3), (4, 3), (5, 3), (6, 3), (2, 4), (3, 4), (4, 4), (2, 5), (3, 5))          # pairs (N, d)
def thm510b(N, d, p):
    """Theorem 5.10(b): index i and the common parity (0 = purely even, 1 = purely odd) of Lam_i, Lam_{i+1}; None if d = 1 or N = 2"""
    if d == 1 or N == 2: return None
    D = d * (N - 1)
    return (p - 3, D % 2) if N != 4 else (p - 4, (D - 1) % 2)
def thm510c(N, d):
    """Theorem 5.10(c): the pair (index l+1, excluded signs of (r_l, r_{l+1})); None if not covered"""
    if d == 2 and N >= 3: return (3, "+-")
    if d >= 4 and N >= 4: return (5, "+-")
    if d == 3 and N >= 7: return (7, "+-")
    return None
SG = {"+": 1, "-": -1}
for (N, d) in CASES:
    p, sg, vals, PAR = case(N, d)
    changes = sum(1 for i in range(len(sg) - 1) if sg[i] != sg[i + 1])
    alt = "".join("+" if j % 2 == 0 else "-" for j in range(p))
    alternates = (sg == alt)
    row = dict(N=N, d=d, p=p, signs=sg, sign_changes=changes, D_minus_1=d * (N - 1) - 1, negative=sg.count("-"), alternates=alternates, residues=vals)
    msg = ""
    b = thm510b(N, d, p)
    if b is not None:
        i, par = b
        assert 1 <= i and i + 1 <= p - 1, "Theorem 5.10(b): index range"
        assert PAR[i] == frozenset({par}) and PAR[i + 1] == frozenset({par}), "Theorem 5.10(b): purity/parity of Lam_i, Lam_{i+1} (exact)"
        t = [SG[c] for c in sg[i - 1:i + 2]]
        mono = (t[0] <= t[1] <= t[2]) if par == 0 else (t[0] >= t[1] >= t[2])
        assert mono, "Theorem 5.10(b) violated?!"
        row["thm_5_10b"] = {"i": i, "parity": "even" if par == 0 else "odd", "certified_signs_i-1_i_i+1": sg[i - 1:i + 2]}
        msg += f"; Thm 5.10(b): Lam_{i}, Lam_{i+1} purely {'even' if par == 0 else 'odd'}, signs of (r_{i-1}, r_{i}, r_{i+1}) = {sg[i-1:i+2]}"
    c = thm510c(N, d)
    if c is not None:
        i, pat = c
        assert 1 <= i <= p - 1 and alt[i - 1:i + 1] == pat, "Theorem 5.10(c): the named pair must be the one required by alternation"
        assert PAR[i] == frozenset({0}), "Theorem 5.10(c): Lam_i purely even (exact)"
        assert sg[i - 1:i + 1] != pat, "Theorem 5.10(c) violated?!"
        row["thm_5_10c_pair"] = {"i": i, "excluded": pat, "certified": sg[i - 1:i + 1]}
        msg += f"; Thm 5.10(c): (r_{i-1}, r_{i}) = {sg[i-1:i+1]} (excluded: {pat})"
    if N == 2:
        assert alternates and p == d, "Theorem 5.10(a): N = 2"
    else:
        assert not alternates, "Theorem 5.10(b)"
    out.append(row)
    print(f"d={d} N={N}: p={p} certified signs {sg}  sign changes {changes} (alternation would need {p-1}; lower bound D-1 = {d*(N-1)-1}); "
          f"negative {sg.count('-')}; alternates: {alternates}" + msg, flush=True)
trip = sorted({r["thm_5_10b"]["certified_signs_i-1_i_i+1"] + "/" + r["thm_5_10b"]["parity"] for r in out if "thm_5_10b" in r})
print("signs of the triple of Theorem 5.10(b) over all rows with N >= 3 (signs/parity):", trip)
ok = all("?" not in r["signs"] for r in out)
print("ALL SIGNS CERTIFIED:", ok)
assert ok
json.dump({"all_certified": ok, "rows": out}, open(_os.path.join(_R, 'data', 'msc_rigorous_spectral', '13_certified_residue_signs.json'), "w"), indent=1)
