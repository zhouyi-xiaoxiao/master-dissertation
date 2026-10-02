"""Theorem 5.7 (continuous-time mode), analytic part: for all N >= N0,
        | u_N - c + kappa/L^2 | < K / L^4 ,        u_N = q t*_cont / L^2,  L = N - 1/2.

For eps = 1/L^2 in (0, eps0], s in [-K, K], u = c - kappa*eps + s*eps^2 (d := kappa - s*eps):

  L^4 G_N'(u) = s G''(c) + Gamma(s, eps),
  Gamma = (1/2) G'''(c) d^2 - H1'(c) d + eps [ -(1/6) G''''(xi) d^3 + (1/2) H1''(xi') d^2 ] + R(u),
  H1 = (3/4) G'' + (u/6) G''',   R = remainder of Proposition 5.4 (j = 1, continuous time).

Certificate:  K |G''(c)| > sup |Gamma|  over s in [-K, K], eps in [0, eps0]
  =>  G_N'(c - kappa eps - K eps^2) > 0 > G_N'(c - kappa eps + K eps^2)  =>  claim (G_N' has a single zero, Thm 3.3).
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
import core
from core import arb, G, interval, R_enclosure, ub

core.set_prec(200)
const = json.load(open(_os.path.join(_R, 'data', 'msc_rigorous_1d', '03_constants.json')))
c = arb(const["c_mid_80_digits"][:60], "1e-55")
G2c, G3c, G4c = G(2, c), G(3, c), G(4, c)
kappa = arb(3) / 4 + c / 6 * G3c / G2c
H1p_c = arb(11) / 12 * G3c + c / 6 * G4c

out = {}
for N0, k0 in [(21, 9), (31, 9), (51, 11), (101, 13), (201, 13), (301, 15), (2001, 15)]:
    L0 = arb(2 * N0 - 1) / 2
    eps = interval(arb(0), 1 / (L0 * L0))
    K = arb("1.54")
    for it in range(300):
        sK = interval(-K, K)
        d = kappa - sK * eps
        U = c - d * eps
        R, Bnd, parts = R_enclosure(1, U, L0, k0)
        G4U = G(4, U)
        H1pp = arb(13) / 12 * G4U + U / 6 * G(5, U)
        gam = G3c * d * d / 2 - H1p_c * d + eps * (-G4U * d ** 3 / 6 + H1pp * d * d / 2) + R
        need = ub(gam) / abs(G2c)
        if K > need:
            break
        K = K * arb("1.01")
    ok = bool(K * abs(G2c) > ub(gam))
    kappa2 = -gam / G2c
    print(f"N0={N0:5d} k0={k0:2d}: Gamma in {gam.str(8)}  Bnd={Bnd.str(5)}  K={float(K.upper()):.5f} certified={ok}  "
          f"kappa2 := lim L^4(u_N-c+kappa/L^2) in {kappa2.str(8)}", flush=True)
    out[str(N0)] = dict(k0=k0, Gamma=gam.str(12), Gamma_lower=float(gam.lower()), Gamma_upper=float(gam.upper()),
                        Bnd=Bnd.str(6), K=float(K.upper()), certified=ok, kappa2=kappa2.str(12),
                        kappa2_lower=float(kappa2.lower()), kappa2_upper=float(kappa2.upper()), Gamma_negative=bool(gam < 0))
json.dump(out, open(_os.path.join(_R, 'data', 'msc_rigorous_1d', '12_cont_theorem.json'), "w"), indent=1)
