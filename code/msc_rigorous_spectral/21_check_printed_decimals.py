"""Convention check: every decimal string of the text that is followed by dots
(`1.2345\\ldots`) must be the TRUNCATION of the constant it stands for, i.e. |printed| <= |value| < |printed| + one unit in the last place.

The script extracts all such strings from tex/*.tex and, for each, looks for a constant in the catalogue below (certified balls from the
data files of scripts 06, 09, 14, 16, 19 and directly computed Arb values) for which the truncation property is CERTIFIED.
A string that matches no constant is a failure.  Output: ../data/21_printed_decimals.json
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import os, re, sys, json, glob
from flint import arb, ctx
ctx.prec = 300
HERE = os.path.dirname(__file__)
D = lambda f: json.load(open(os.path.join(_R, 'data', 'msc_rigorous_spectral', f)))
pi = arb.pi(); ln2 = arb(2).log(); gam = arb.const_euler(); p2 = pi ** 2
CAT = {}
def add(name, ball):
    CAT[name] = ball
def from_str(s):                       # "[mid +/- rad]" -> ball, inflated to cover the rounding of the printed string
    m = re.match(r"\[(\S+) \+/- (\S+)\]", s)
    if m: return arb(m.group(1)) + arb(0, 1) * arb(m.group(2)) * arb("1.01") + arb(0, 1) * arb(10) ** -60
    return arb(s) + arb(0, 1) * arb(10) ** -60
for k, v in D("06_certified_constants.json").items():
    if isinstance(v, dict) and "str" in v: add("06:" + k, from_str(v["str"]))
for k, v in D("09_certified_part3.json")["constants"].items(): add("09:" + k, from_str(v["str"]))
for k, v in D("14_certified_epstein.json").items():
    if isinstance(v, dict) and "str" in v: add("14:" + k, from_str(v["str"]))
for k, v in D("16_certified_d3_remainder.json").items():
    if isinstance(v, dict) and "str" in v: add("16:" + k, from_str(v["str"]))
    elif isinstance(v, str) and v.startswith("["): add("16:" + k, from_str(v))
for k, v in D("19_certified_d2_limits.json")["constants"].items(): add("19:" + k, from_str(v["str"]))
c_tau = CAT["06:c_tau_2D"]; C3 = CAT["06:C_3"]; c_tau3 = CAT["06:c_tau_3D"]; c_mt3 = CAT["06:c_m_minus_tau_3D"]; K1 = CAT["09:K(1)"]
G14 = (arb(1) / 4).gamma()
add("pi ln 2", pi * ln2)
add("(pi^2/4) c_tau", p2 / 4 * c_tau); add("(pi^2/6) C_3", p2 / 6 * C3); add("(pi^2/6) c_tau3", p2 / 6 * c_tau3); add("(pi^2/6) c_mtau3", p2 / 6 * c_mt3)
add("C_3' = c_tau3 + c_mtau3", c_tau3 + c_mt3)
add("K(1) - pi", K1 - pi); add("6 ln 2 - 2 - pi", 6 * ln2 - 2 - pi)
add("mu_-(3) lower bound (pi^2/6)(1-1/9)(1-pi^2/108)", p2 / 6 * (1 - arb(1) / 9) * (1 - p2 / 108))
add("c_* = 1 + pi^{3/2}/2", 1 + pi ** arb("1.5") / 2)
add("Gamma(1/4)^8/(576 pi^4)", G14 ** 8 / (576 * pi ** 4))
add("7.1069 - (pi^2/6) C_3", arb("7.1069") - p2 / 6 * C3)
add("R_tau(2) = 5 - (32/pi) ln 2 - 4 c_tau", 5 - 32 / pi * ln2 - 4 * c_tau)
add("R_mtau(2) = 3 - 16 ln 2/pi", 3 - 16 * ln2 / pi)
def Xm(d, N):
    aN = 1 - p2 / (12 * arb(N) ** 2)
    if d == 2: return aN * (2 * pi * arb(N).log() + p2 / 4 * c_tau - p2 / 4 * arb("2.14") / arb(N) ** 2)
    return aN * p2 / 6 * (C3 * N + c_tau3 - arb("17.4") / arb(N) ** 2)
add("X_-(d=2,N=2)", Xm(2, 2)); add("X_-(d=2,N=3)", Xm(2, 3)); add("X_-(d=3,N=3)", Xm(3, 3))

found = []; fails = []
pat = re.compile(r"(?<![\d.])(-?\d+\.\d+)\\ldots(\\cdot10\^\{(-?\d+)\})?")
for f in sorted(glob.glob(_os.path.join(_R, 'proofs', 'R2_spectral_structure', '*.tex'))):  # the note
    txt = open(f).read()
    for m in pat.finditer(txt):
        s = m.group(1); ex = int(m.group(3)) if m.group(3) else 0
        nd = len(s.split(".")[1])
        scale = arb(10) ** ex
        lo = arb(s.lstrip("-")) * scale; ulp = arb(10) ** (-nd) * scale
        neg = s.startswith("-")
        hits = []
        for name, v in CAT.items():
            av = -v if neg else v
            if lo <= av and av < lo + ulp and (av > 0 or lo == 0):
                hits.append(name)
        ctxt = txt[max(0, m.start() - 40):m.end()].replace("\n", " ")
        found.append({"file": os.path.basename(f), "printed": m.group(0), "matches": hits})
        if not hits:
            fails.append((os.path.basename(f), m.group(0), ctxt))
uniq = {}
for r in found: uniq.setdefault(r["printed"], set()).update(r["matches"])
for s in sorted(uniq): print(f"{s:34s} <- {sorted(uniq[s])[:3]}")
print(f"{len(found)} printed truncated decimals ({len(uniq)} distinct), {len(fails)} without a certified match")
for f in fails: print("FAIL", f)
json.dump({"n": len(found), "n_fail": len(fails), "items": found}, open(_os.path.join(_R, 'data', 'msc_rigorous_spectral', '21_printed_decimals.json'), "w"), indent=1)
print("ALL OK" if not fails else "FAILURES")
sys.exit(1 if fails else 0)
