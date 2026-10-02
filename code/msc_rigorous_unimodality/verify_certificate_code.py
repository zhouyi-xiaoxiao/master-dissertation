"""verify_certificate_code.py -- separately written checks of the exact machinery behind the computer-assisted theorems.

 (A) ExactStepper (numpy object arrays) versus a separately written dense implementation with fractions.Fraction
     built directly from the definition of the walk (dictionary of sites, explicit neighbour loop).
 (B) A separately written pure-Python implementation of the whole cone certificate (dict-based integers) re-derives the
     sign runs, the cone time and the verdict for a sample of cases, including non-unimodal ones.
 (C) Cross-check of the modes against the separately written floating-point code of code/msc_modes
     (data/full_tail.json: q = 0.8 and 0.5, geometries CC, C2M, M2C).
 (D) After the cone time, the PMF is checked (exactly) to be nonincreasing for another 3*T steps.
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import itertools, json, os, sys
from fractions import Fraction
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from unilib import ExactStepper
from cert_unimodal import certify, geometry
HERE = os.path.dirname(os.path.abspath(__file__))
RES = []
def rec(name, ok, detail=""):
    RES.append(dict(check=name, ok=bool(ok), detail=detail)); print(("PASS " if ok else "FAIL ") + name + ((" :: " + detail) if detail else ""), flush=True)

def pmf_definition(N, d, q, s, a, T):
    """f(1..T) straight from the definition, Fractions, dict of sites."""
    sites = list(itertools.product(range(N), repeat=d))
    rho = {x: Fraction(0) for x in sites}; rho[s] = Fraction(1)
    w = q / (2 * d); out = [Fraction(0)]
    for _ in range(T):
        new = {x: Fraction(0) for x in sites}
        for x in sites:
            if rho[x] == 0 or x == a: continue
            new[x] += (1 - q) * rho[x]
            for ax in range(d):
                for dx in (-1, 1):
                    y = list(x); y[ax] += dx
                    if 0 <= y[ax] < N: new[tuple(y)] += w * rho[x]
                    else: new[x] += w * rho[x]
        out.append(new[a]); new[a] = Fraction(0); rho = new
    return out

def A_check():
    ok = True
    for (d, N, s, a) in [(1, 5, (0,), (4,)), (2, 3, (0, 0), (2, 2)), (2, 4, (1, 3), (1, 1)), (2, 5, (0, 0), (2, 2)), (3, 3, (2, 1, 1), (0, 1, 1)), (3, 3, (0, 0, 0), (2, 2, 2))]:
        for q in (Fraction(1), Fraction(499, 500), Fraction(4, 5), Fraction(1, 2), Fraction(1, 64)):
            T = 30
            f_def = pmf_definition(N, d, q, s, a, T)
            st = ExactStepper(N, d, q.numerator, q.denominator, s, a); Dp = 1
            for t in range(1, T + 1):
                F = st.step(); Dp *= st.D
                if Fraction(F, Dp) != f_def[t]: ok = False
    rec("A ExactStepper == definition (Fractions), 6 geometries x 5 activities x 30 steps", ok)

def independent_certificate(N, d, qn, qd, s, a, Tmax=200000):
    """pure-Python re-implementation: integer weights straight from q=qn/qd with denominator 2*d*qd (no gcd reduction)."""
    sites = list(itertools.product(range(N), repeat=d)); Dn = 2 * d * qd
    nb = {}
    for x in sites:
        lst = []; blocked = 0
        for ax in range(d):
            for dx in (-1, 1):
                y = list(x); y[ax] += dx
                if 0 <= y[ax] < N: lst.append(tuple(y))
                else: blocked += 1
        nb[x] = (lst, 2 * d * (qd - qn) + qn * blocked)
    R = {x: 0 for x in sites}; R[s] = 1
    signs = []; Fprev = 0
    for t in range(1, Tmax + 1):
        new = {x: 0 for x in sites}
        for x in sites:
            v = R[x]
            if v == 0: continue
            lst, stay = nb[x]
            new[x] += stay * v
            for y in lst: new[y] += qn * v
        F = new[a]; new[a] = 0
        diff = F - Dn * Fprev; signs.append((diff > 0) - (diff < 0)); Fprev = F
        cone = all(new[x] <= Dn * R[x] for x in sites if x != a)
        R = new
        if cone:
            return signs, t - 1
    return signs, None

def B_check():
    ok = True; det = []
    cases = [("CC", 2, 9, 1, 1), ("CC", 2, 9, 499, 500), ("CC", 2, 12, 499, 500), ("CC", 2, 11, 1, 1), ("CC", 3, 6, 1, 1), ("CC", 1, 4, 1, 1), ("CC", 1, 4, 4, 5),
             ("C2M", 2, 7, 4, 5), ("M2C", 2, 7, 4, 5), ("M2C", 3, 5, 4, 5), ("PAIR:2,1,1:0,1,1", 3, 3, 1, 2), ("PAIR:5,5:2,3", 2, 8, 1, 2), ("CC", 4, 3, 1, 1)]
    for (geo, d, N, qn, qd) in cases:
        s, a = geometry(geo, N, d)
        signs, Tc = independent_certificate(N, d, qn, qd, s, a)
        r = certify(N, d, qn, qd, s, a)
        runs = []
        for j, sg in enumerate(signs, start=1):
            if runs and runs[-1][0] == sg: runs[-1][2] = j
            else: runs.append([sg, j, j])
        same = (runs == r["sign_runs"]) and (Tc == r["T_cone"])
        nz = [v for v in signs if v != 0]
        ch = sum(1 for i in range(len(nz) - 1) if nz[i] != nz[i + 1])
        verdict = (ch == 0) or (ch == 1 and nz[0] > 0)
        same = same and (verdict == bool(r["unimodal"]))
        ok = ok and same
        det.append(f"{geo} d{d} N{N} q={qn}/{qd}: T_cone={Tc} unimodal={verdict} {'ok' if same else 'MISMATCH'}")
    rec("B separately written pure-Python cone certificate reproduces sign runs, cone time and verdict (13 cases)", ok, " | ".join(det))

def C_check():
    path = _os.path.join(_R, 'data', 'msc_modes', 'full_tail.json')
    old = json.load(open(path)); ok = True; n = 0; det = []
    for r in old:
        qf = Fraction(r["q"]).limit_denominator(10)
        if r["N"] > 21 and r["d"] >= 2: continue
        if r["N"] > 60: continue
        s, a = geometry(r["geo"], r["N"], r["d"])
        c = certify(r["N"], r["d"], qf.numerator, qf.denominator, s, a)
        good = c["unimodal"] is True and c["mode"] == r["mode"] and r["n_local_maxima_full_support"] == 1
        ok = ok and good; n += 1
        if not good: det.append(f"{r['geo']} d{r['d']} N{r['N']} q{r['q']}: old mode {r['mode']} new {c['mode']}")
    rec(f"C modes agree with the separately written float code of msc_modes/data/full_tail.json ({n} cases)", ok, "; ".join(det))

def D_check():
    ok = True
    for (geo, d, N, qn, qd) in [("CC", 2, 7, 499, 500), ("CC", 2, 9, 1, 1), ("CC", 3, 5, 1, 1), ("C2M", 2, 9, 4, 5), ("M2C", 3, 5, 4, 5), ("CC", 1, 9, 4, 5)]:
        s, a = geometry(geo, N, d)
        r = certify(N, d, qn, qd, s, a)
        Tc = r["T_cone"]
        st = ExactStepper(N, d, qn, qd, s, a); Fs = [0]
        for _ in range(4 * Tc + 50): Fs.append(st.step())
        D = st.D
        for u in range(Tc, 4 * Tc + 48):
            if Fs[u + 2] > D * Fs[u + 1]: ok = False     # f(u+2) <= f(u+1) for all u >= T_cone
    rec("D PMF nonincreasing after the cone time on a further 3*T steps (exact, 6 cases)", ok)

if __name__ == "__main__":
    A_check(); B_check(); C_check(); D_check()
    json.dump(RES, open(_os.path.join(_R, 'data', 'msc_rigorous_unimodality', 'verify_certificate_code.json'), "w"), indent=1)
    print("ALL PASS" if all(r["ok"] for r in RES) else "SOME FAIL")
