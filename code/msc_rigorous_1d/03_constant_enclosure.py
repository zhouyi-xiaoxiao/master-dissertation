"""Rigorous enclosure of the continuum constants (Theorems 4.3 and 4.6 of note R1).

  c      = unique maximiser of G on (0, infinity)
  G(c), G''(c), G'''(c), G''''(c)
  kappa  = 3/4 + (c/6) G'''(c)/G''(c)
  rho    = -(c/2) G'''(c)/G''(c)

Method: G'(c_mid - r) > 0 > G'(c_mid + r) certified in ball arithmetic  =>  (G is unimodal with a unique
maximiser, Theorem 4.3)  c in (c_mid - r, c_mid + r).  Also certifies G'(3/10) > 0 > G'(2/5) (used in the
proof of Theorem 4.3) and G'' < 0 on the enclosure.
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
import mpmath as mp
import rig
from rig import arb, G, interval
from flint import ctx

rig.set_prec(400)
mp.mp.dps = 90


def Gp_mp(u):
    s = mp.mpf(0)
    for k in range(1, 121, 2):
        y = k * mp.pi / 4
        sg = 1 if ((k - 1) // 2) % 2 == 0 else -1
        s += sg * 2 * y * (-2 * y * y) * mp.e ** (-2 * u * y * y)
    return s


c_mid = mp.findroot(Gp_mp, mp.mpf("0.3332842654949479"))
c_str = mp.nstr(c_mid, 80)
r = arb("1e-70")
cm = arb(c_str)
lo, hi = cm - r, cm + r
g_lo, g_hi = G(1, lo), G(1, hi)
assert g_lo > 0 and g_hi < 0, "sign change not certified"
cball = interval(lo, hi)
G2 = G(2, cball)
assert G2 < 0
out = {
    "c_mid_80_digits": c_str,
    "c_radius": "1e-70",
    "certified": {
        "G'(c-r)>0": bool(g_lo > 0), "G'(c+r)<0": bool(g_hi < 0), "G''<0 on [c-r,c+r]": bool(G2 < 0),
        "G'(3/10)>0": bool(G(1, arb(3) / 10) > 0), "G'(2/5)<0": bool(G(1, arb(2) / 5) < 0),
    },
}
ctx.prec = 400
G0, G3, G4 = G(0, cball), G(3, cball), G(4, cball)
kappa = arb(3) / 4 + cball / 6 * G3 / G2
rho = -cball / 2 * G3 / G2
for name, val in [("c", cball), ("G(c)", G0), ("G''(c)", G2), ("G'''(c)", G3), ("G''''(c)", G4),
                  ("kappa", kappa), ("rho", rho), ("rho-1", rho - 1), ("c/4-kappa", cball / 4 - kappa)]:
    out[name] = val.str(60)
    print(f"{name:10s} = {val.str(55)}")
print(out["certified"])
# also the single-image value 1/3: c - 1/3
print("c - 1/3 =", (cball - arb(1) / 3).str(20))
out["c-1/3"] = (cball - arb(1) / 3).str(30)
json.dump(out, open(_os.path.join(_R, 'data', 'msc_rigorous_1d', '03_constants.json'), "w"), indent=1)
