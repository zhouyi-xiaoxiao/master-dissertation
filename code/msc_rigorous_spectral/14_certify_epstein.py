"""Certified (Arb) enclosures of the Epstein-type lattice sums of Lemma 7.6 (theta splitting):

    Z_d(s) = sum_{k in Z^d \\ 0} |k|^{-2s}      (s > d/2),
    Z_d    = Z_d(2),
    Z_d'   = sum_{|k|^2 >= 2} 1/(|k|^2 (|k|^2 - 1)) = sum_{j=2}^{J} (Z_d(j) - 2d) + sum_{|k|^2>=2} |k|^{-2J}/(|k|^2-1).

Formula (Lemma 7.6; Riemann's splitting with the Jacobi theta transformation theta(1/t) = sqrt(t) theta(t)):

    pi^{-s} Gamma(s) Z_d(s) = 1/(s - d/2) - 1/s + sum_{k != 0} [ c^{-s} Gamma(s, c) + c^{s-d/2} Gamma(d/2 - s, c) ],  c = pi |k|^2.

Truncation |k|_inf <= R with the tail bound proved in the text:
    sum_{|k|_inf > R} (...) <= 4 d (1.1)^{d-1} * 1.001 * exp(-pi (R+1)^2) / (pi R^2 - s + 1).
Output: ../data/14_certified_epstein.json
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import os, json, math, itertools
from collections import Counter
from flint import arb, ctx
ctx.prec = 300
pi = arb.pi()
OUT = {}

def enc(name, x, note=""):
    OUT[name] = {"str": x.str(40), "lower": x.lower().str(45, radius=False), "upper": x.upper().str(45, radius=False), "note": note}
    print(f"ENCLOSURE {name:14s} = {x.str(40)}   {note}", flush=True)

def Gup(s, c):
    """upper incomplete gamma Gamma(s, c) for arb s, c > 0  (python-flint: z.gamma_upper(s) = Gamma(s, z))"""
    return c.gamma_upper(s)

# sanity: Gamma(-1/2, x) = 2 (x^{-1/2} e^{-x} - sqrt(pi) erfc(sqrt x)),  Gamma(2, x) = (1+x) e^{-x}
x = arb(2)
assert Gup(arb(-1) / 2, x).overlaps(2 * (x.rsqrt() * (-x).exp() - pi.sqrt() * x.sqrt().erfc()))
assert Gup(arb(2), x).overlaps((1 + x) * (-x).exp())

def shells(d, R):
    """dict n -> number of k in Z^d with |k|_inf <= R and |k|^2 = n (n >= 1)"""
    cnt = Counter()
    for k in itertools.combinations_with_replacement(range(R + 1), d):
        if not any(k): continue
        c = Counter(k); perms = math.factorial(d)
        for v in c.values(): perms //= math.factorial(v)
        signs = 2 ** sum(1 for v in k if v)
        cnt[sum(v * v for v in k)] += perms * signs
    return cnt

def Z(d, s, R=7):
    s = arb(s); hd = arb(d) / 2
    tot = 1 / (s - hd) - 1 / s
    for n, mult in shells(d, R).items():
        c = pi * n
        tot += mult * (c ** (-s) * Gup(s, c) + c ** (s - hd) * Gup(hd - s, c))
    tail = 4 * d * arb("1.1") ** (d - 1) * arb("1.001") * (-pi * (R + 1) ** 2).exp() / (pi * R * R - s + 1)
    tot += arb(0, 1) * tail          # [tot - tail, tot + tail] (the neglected terms are positive; a two-sided ball is harmless)
    return tot * pi ** s / s.gamma()

def direct_tail_sum(d, J, R=40):
    """sum_{|k|^2 >= 2} |k|^{-2J}/(|k|^2 - 1): partial sum over |k|_inf <= R plus tail <= 4 d 3^{d-1} R^{d-2-2J}/(2J+2-d)."""
    tot = arb(0)
    for n, mult in shells(d, R).items():
        if n >= 2:
            tot += arb(mult) / (arb(n) ** J * (n - 1))
    tail = arb(4 * d * 3 ** (d - 1)) * arb(R) ** (d - 2 - 2 * J) / (2 * J + 2 - d)
    return tot + arb(0, 1) * tail

res = {}
for d in (2, 3):
    Zd = Z(d, 2)
    enc(f"Z_{d}", Zd, "sum_{k != 0} |k|^{-4} (theta splitting)")
    J = 8
    Zp = sum(Z(d, j) - 2 * d for j in range(2, J + 1)) + direct_tail_sum(d, J)
    enc(f"Z_{d}'", Zp, "sum_{|k|^2>=2} 1/(|k|^2(|k|^2-1))")
    res[d] = (Zd, Zp)
    # cross-checks -------------------------------------------------------------------------------
    # (i) truncation radius: R = 6 and R = 8 give overlapping balls
    assert Z(d, 2, R=6).overlaps(Zd) and Z(d, 2, R=8).overlaps(Zd)
    # (ii) direct summation for s = 4, 6 (fast convergence): partial sum + integral-comparison tail (Lemma 7.3)
    for s in (4, 6):
        Rr = 60
        part = sum(arb(mult) / arb(n) ** s for n, mult in shells(d, Rr).items())
        tail = arb(2 * d * 3 ** (d - 1)) * arb(Rr) ** (d - 2 * s) / (2 * s - d)       # sum_{j>R} 2d 3^{d-1} j^{d-1-2s}
        direct = part + arb(0, 1) * tail
        assert direct.overlaps(Z(d, s)), (d, s)
    # (iii) J-independence of Z_d'
    Zp2 = sum(Z(d, j) - 2 * d for j in range(2, 6)) + direct_tail_sum(d, 5, R=60)
    assert Zp2.overlaps(Zp)

# closed form in d = 2 (Lorenz-Hardy): Z_2 = 4 zeta(2) G -- comparison only
Z2c = 4 * arb(2).zeta() * arb.const_catalan()
assert Z2c.overlaps(res[2][0])
enc("Z_2_closed", Z2c, "4 zeta(2) Catalan (comparison)")
# consistency with the elementary enclosures of script 06
C06 = json.load(open(_os.path.join(_R, 'data', 'msc_rigorous_spectral', '06_certified_constants.json')))
assert arb(C06["Z_3_lower"]["mid"]) < res[3][0] and res[3][0] < arb(C06["Z_3_upper"]["mid"])
assert arb(C06["Z_2_lower"]["mid"]) < res[2][0] and res[2][0] < arb(C06["Z_2_upper"]["mid"])
# rounded constants quoted in the text
assert arb("6.02681") < res[2][0] < arb("6.02682")
assert arb("16.53231") < res[3][0] < arb("16.53232")
assert arb("3.2259") < res[2][1] < arb("3.2260")
print("Z_2' =", res[2][1].str(20), " Z_3' =", res[3][1].str(20))
# the decimal strings printed in Lemma 7.6, (7.11): TRUNCATED (not rounded) expansions; every digit shown is certified,
# i.e. printed < Z < printed + one unit in the last printed place
TRUNC = {"Z_2": ("6.02681203969194012", res[2][0]), "Z_3": ("16.5323159597616696", res[3][0]),
         "Z_2'": ("3.22591076832683889", res[2][1]), "Z_3'": ("14.7022972380072310", res[3][1])}
OUT["truncated_digits"] = {}
for name, (txt, ball) in TRUNC.items():
    ulp = arb(10) ** (-len(txt.split(".")[1]))
    assert arb(txt) < ball and ball < arb(txt) + ulp, name
    OUT["truncated_digits"][name] = txt
    print(f"CERTIFIED truncated digits: {name} = {txt}...  (printed < value < printed + 1e-{len(txt.split('.')[1])})")
# correctly ROUNDED strings are not always truncations; two such strings are checked here:
assert res[3][0] < arb("16.53231595976167") and res[2][1] < arb("3.22591076832684")
assert abs(res[3][0] - arb("16.53231595976167")) < arb("5e-15") and abs(res[2][1] - arb("3.22591076832684")) < arb("5e-15")
OUT["rounded"] = {"Z_2": [6.02681, 6.02682], "Z_3": [16.53231, 16.53232]}
json.dump(OUT, open(_os.path.join(_R, 'data', 'msc_rigorous_spectral', '14_certified_epstein.json'), "w"), indent=1)
print("ALL CERTIFIED CHECKS OK")
