"""Theorem 7.4 (q = 1: location of the mode, parity classes), analytic part N >= N1.

eta = 1/L in (0, eta0].  Class 'near' (t = N-1 mod 2, upper signs) / 'far' (t = N mod 2, lower signs):
   (1/2) L^4 (f(t+2) - f(t)) = G'(v) -+ eta calE'(v) + eta^2 J(v) + eta^3 r,   J = (5/4) G'' - (v/3) G''',
   |r| <= (rho_E + eta0 rho_P)/2                                               (Lemma 7.3, v = (t-1)/L^2).
With v = c -+ b eta + s eta^2 (b = -calE'(c)/G''(c)), delta = -+ b + s eta:
   (1/2) L^6 (f(t+2) - f(t)) = (s - s2) G''(c) + eta Gamma1,     s2 = -Gamma0/G''(c),
   Gamma0 = (1/2) G'''(c) b^2 + b calE''(c) + J(c),
   Gamma1 = (1/2) G'''(c) (-+ 2 b s + s^2 eta) -+ calE''(c) s + (1/6) G''''(xi) delta^3 -+ (1/2) calE'''(xi') delta^2
            + J'(xi'') delta + r.
Certificate 1:  K1 |G''(c)| > sup |Gamma1|  over s in [s2 - K1 eta0, s2 + K1 eta0], eta in [0, eta0], both classes
   => the unique zero t0 of the class increment satisfies |t0 - (c L^2 -+ b L + 1 + s2)| < K1/L.
Certificate 2 (near class wins):  with s_n, s_f in [s2 - K1 eta0, s2 + 2 + K1 eta0],
   L^2 (f(t_near) - f(t_far))/eta = -2 calE(c) + eta X > 0.
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
import core, q1
from core import arb, G, interval, ub

core.set_prec(200)
const = json.load(open(_os.path.join(_R, 'data', 'msc_rigorous_1d', '03_constants.json')))
c = arb(const["c_mid_80_digits"][:60], "1e-55")
G0c, G2c, G3c = G(0, c), G(2, c), G(3, c)
E0c, E1c, E2c = q1.calE(0, c), q1.calE(1, c), q1.calE(2, c)
b = -E1c / G2c
Jc = arb(5) / 4 * G2c - c / 3 * G3c
Gam0 = G3c * b * b / 2 + b * E2c + Jc
s2 = -Gam0 / G2c
absG2 = abs(G2c)
out = dict(b=b.str(40), s2=s2.str(40), calE_c=E0c.str(30), calE1_c=E1c.str(30), calE2_c=E2c.str(30), Gamma0=Gam0.str(30),
           tau1_offset=(1 + s2).str(30), rows={})
print("b =", b.str(30), " s2 =", s2.str(30), " calE(c) =", E0c.str(20))

for N1 in [21, 31, 51, 101, 201, 601]:
    L0 = arb(2 * N1 - 1) / 2
    eta0 = 1 / L0
    eta = interval(arb(0), eta0)
    K1 = arb(5)
    ok = False
    for it in range(60):
        sZ = s2 + interval(-K1 * eta0, K1 * eta0)                 # s-range for the zero
        sM = interval(s2 - K1 * eta0, s2 + 2 + K1 * eta0)         # s-range for the class modes
        smax = ub(sM)
        half = (b + smax * eta0) * eta0
        V = c + interval(-half, half)
        rP, rE, rP0, rE0 = q1.remainders(arb(V.lower()), arb(V.upper()), L0)
        rbd = (rE + eta0 * rP) / 2
        r = arb(0, rbd.upper())
        G4V = G(4, V)
        E3V = q1.calE(3, V)
        JpV = arb(11) / 12 * G(3, V) - V / 3 * G4V
        worst = arb(0)
        for sgn in (-1, +1):                                       # -1: near (v = c - b eta + ...), +1: far
            delta = sgn * b + sZ * eta
            gam1 = G3c / 2 * (sgn * 2 * b * sZ + sZ * sZ * eta) + sgn * E2c * sZ + G4V * delta ** 3 / 6 \
                + sgn * E3V * delta ** 2 / 2 + JpV * delta + r
            w = ub(gam1)
            if w > worst:
                worst = w
        need = worst / absG2
        if K1 > need:
            ok = True
            break
        K1 = need * arb("1.02")
    # certificate 2
    dn = -b + sM * eta
    df = b + sM * eta
    G2V = G(2, V)
    E1V = q1.calE(1, V)
    X = (G2V * dn * dn - G2V * df * df) / 2 - (E1V * dn + E1V * df) + arb(0, (2 * (rP0 + eta0 * rE0)).upper())
    Delta = -2 * E0c + eta * X
    ok2 = bool(Delta > 0)
    # the zero must lie beyond the trivial zeros: tau1 - K1/L > N - 2, checked at L0 (c L^2 - (b+1) L increasing for L >= 3)
    ok3 = bool(c * L0 * L0 - b * L0 + 1 + s2 - K1 * eta0 > L0 - arb(3) / 2) and bool(2 * c * L0 > b + 1)
    print(f"N1={N1:4d}: rho_P={float(rP.mid()):.1f} rho_E={float(rE.mid()):.1f} rho_P0={float(rP0.mid()):.2f} rho_E0={float(rE0.mid()):.2f} "
          f"|r|<={float(rbd.mid()):.1f}  K1={float(K1.upper()):.3f} cert1={ok}  Delta in {Delta.str(6)} cert2={ok2} cert3={ok3}", flush=True)
    out["rows"][str(N1)] = dict(K1=float(K1.upper()), cert1=ok, cert2=ok2, cert3=ok3, rho_P=float(rP.upper()), rho_E=float(rE.upper()),
                                rho_P0=float(rP0.upper()), rho_E0=float(rE0.upper()), Delta=Delta.str(10),
                                Delta_lower=float(Delta.lower()), Delta_upper=float(Delta.upper()), r_bound=float(rbd.upper()),
                                window=[float(V.lower()), float(V.upper())])
json.dump(out, open(_os.path.join(_R, 'data', 'msc_rigorous_1d', '16_q1_theorem.json'), "w"), indent=1)
