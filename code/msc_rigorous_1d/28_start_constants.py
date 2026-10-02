"""Certified ranges and values of the general-start constants (Theorem 8.12, Corollary 8.14):
     c(xi),  rho(xi) = -(c/2) G'''(c)/G''(c),  kappa(xi) = 1/2 - rho/3,  G''(c)     (G = G_xi, c = c(xi)),
 (a) hulls over xi in [0, 1/2] from NB balls (ball arithmetic; c(xi) enclosed via Theorem 8.3(2));
 (b) values at xi = 0, 1/10, ..., 1/2 (exact xi, tight enclosures).
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
import core, start
from core import arb, interval
from start import Gxi, c_enclosure

core.set_prec(160)
NB = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
hull = {}
def upd(name, ball):
    lo, hi = float(ball.lower()), float(ball.upper())
    a = hull.get(name, (lo, hi))
    hull[name] = (min(a[0], lo), max(a[1], hi))

for i in range(NB):
    Xi = interval(arb(i) / (2 * NB), arb(i + 1) / (2 * NB))
    C, _ = c_enclosure(Xi)
    G2, G3 = Gxi(2, C, Xi), Gxi(3, C, Xi)
    assert G2 < 0
    rho = -C / 2 * G3 / G2
    kap = arb(1) / 2 - rho / 3
    upd("c", C); upd("rho", rho); upd("kappa", kap); upd("G2", G2)
    upd("kappa+q(rho-1), q=1/2", kap + (rho - 1) / 2); upd("kappa+q(rho-2), q=1/2", kap + (rho - 2) / 2)
    upd("c/(1-xi^2)", C / (1 - Xi * Xi))
print("hulls over xi in [0, 1/2] (%d balls):" % NB)
for k, v in hull.items():
    print("  %-26s in [%.6f, %.6f]" % (k, v[0], v[1]))
rows = []
for num in range(0, 6):
    xi = arb(num) / 10
    C, _ = c_enclosure(xi)
    G2, G3 = Gxi(2, C, xi), Gxi(3, C, xi)
    rho = -C / 2 * G3 / G2
    kap = arb(1) / 2 - rho / 3
    rows.append(dict(xi=num / 10, c=C.str(20), rho=rho.str(16), kappa=kap.str(16), G2=G2.str(14)))
    print(rows[-1])
json.dump(dict(NB=NB, hulls=hull, values=rows), open(_os.path.join(_R, 'data', 'msc_rigorous_1d', '28_start_constants.json'), "w"), indent=1)
