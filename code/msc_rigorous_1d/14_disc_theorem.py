"""Theorem 6.1 (discrete time, local sign analysis of the increment), computer-assisted constants.

Notation: eps = 1/L^2, tau = (c L^2 - kappa)/q + 1 - rho,  D(t) = f(t+1) - f(t),
          frakD(t) = (L^4/q^2) D(t) = S_1 (lattice sum, j = 1),  v = q (t-1)/L^2.
For q in a ball [qlo, qhi] subset (0, 4/5] and all L >= L0 we certify a constant Kd with

   (A) frakD(t) > 0 for all integers t in [tau - Kd/(q L^2) - 1, tau - Kd/(q L^2)],
   (B) frakD(t) < 0 for all integers t in [tau + Kd/(q L^2), tau + Kd/(q L^2) + 1].

Reduction (proof of Theorem 6.1): with t = tau + sigma, s = q sigma L^2, d = kappa + q rho - s eps, v = c - d eps,

   L^4 frakD = s G''(c) + (1/2) G'''(c) d^2 - H'(c) d + eps[-(1/6) G''''(xi) d^3 + (1/2) H''(xi') d^2] + R,
   H = (3/4) G'' + (v/6)(1-3q) G''',  H(c) = (kappa + q rho) G''(c).

The part M(s) = s G''(c) + (1/2) G'''(c) d^2 - H'(c) d is decreasing in s if |G''(c)| > eps |G'''(c) d - H'(c)|
("slope condition"), so on window (A) it is >= M(-Kd) and on window (B) it is <= M(Kd).

Usage: 14_disc_theorem.py N0 k0 qmax(num/den) nsub [tag]
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import os, sys, json, time
sys.path.insert(0, os.path.dirname(__file__))
import core
from core import arb, G, interval, R_enclosure, ub

core.set_prec(128)
const = json.load(open(_os.path.join(_R, 'data', 'msc_rigorous_1d', '03_constants.json')))
c = arb(const["c_mid_80_digits"][:40], "1e-36")
G2c, G3c, G4c = G(2, c), G(3, c), G(4, c)
kappa = arb(3) / 4 + c / 6 * G3c / G2c
rho = -c / 2 * G3c / G2c
absG2 = abs(G2c)


def certify(qlo, qhi, L0, k0, Kd):
    q = interval(arb(qlo), arb(qhi))
    eps0 = 1 / (L0 * L0)
    eps = interval(arb(0), eps0)
    base = kappa + q * rho
    dfull = base + interval(-(q + Kd * eps0), q + Kd * eps0)   # all d on both windows
    V = c - dfull * eps
    R, Bnd, parts = R_enclosure(1, V, L0, k0, q=q, qbar=arb(qhi))
    G4V, G5V = G(4, V), G(5, V)
    Hp_c = arb(3) / 4 * G3c + (1 - 3 * q) / 6 * G3c + c / 6 * (1 - 3 * q) * G4c
    Hpp = arb(3) / 4 * G4V + (1 - 3 * q) / 3 * G4V + V / 6 * (1 - 3 * q) * G(5, V)
    common = eps * (-G4V * dfull ** 3 / 6 + Hpp * dfull ** 2 / 2) + R
    d_lo = base + Kd * eps          # s = -Kd
    d_hi = base - Kd * eps          # s = +Kd
    val_lo = Kd * absG2 + G3c * d_lo ** 2 / 2 - Hp_c * d_lo + common
    val_hi = -Kd * absG2 + G3c * d_hi ** 2 / 2 - Hp_c * d_hi + common
    MG = ub(G3c * dfull - Hp_c)
    slope_ok = absG2 * L0 * L0 > MG
    gam = G3c * base ** 2 / 2 - Hp_c * base + common            # Gamma at s = 0
    return bool(val_lo > 0), bool(val_hi < 0), bool(slope_ok), gam, Bnd


def min_Kd(qlo, qhi, L0, k0, guess=2.5, kmax=60.0):
    Kd = arb(str(guess))
    for _ in range(40):
        a, b, s, gam, Bnd = certify(qlo, qhi, L0, k0, Kd)
        need = ub(gam) / absG2 * arb("1.01") + arb("0.001")
        if a and b and s:
            if need < Kd * arb("0.97"):
                a2, b2, s2, gam2, Bnd2 = certify(qlo, qhi, L0, k0, need)
                if a2 and b2 and s2:
                    return need, gam2, Bnd2
            return Kd, gam, Bnd
        Kd = Kd * arb("1.5")
        if Kd > kmax:
            return None, gam, Bnd
    return None, gam, Bnd


if __name__ == "__main__":
    N0 = int(sys.argv[1]) if len(sys.argv) > 1 else 51
    k0 = int(sys.argv[2]) if len(sys.argv) > 2 else 11
    qmax_num, qmax_den = map(int, (sys.argv[3] if len(sys.argv) > 3 else "4/5").split("/"))
    nsub = int(sys.argv[4]) if len(sys.argv) > 4 else 240
    tag = sys.argv[5] if len(sys.argv) > 5 else f"N0_{N0}_q{qmax_num}_{qmax_den}"
    if len(sys.argv) > 6:
        core.HIGH_TOL = sys.argv[6]          # tolerance mode (Theorem 7.7): B_high <= this number is a hypothesis
    L0 = arb(2 * N0 - 1) / 2
    rows = []
    worst = 0.0
    worst23 = 0.0
    t0 = time.time()
    for i in range(nsub):
        qlo = arb(qmax_num * i) / (qmax_den * nsub)
        qhi = arb(qmax_num * (i + 1)) / (qmax_den * nsub)
        Kd, gam, Bnd = min_Kd(qlo, qhi, L0, k0)
        gamma_coef = (-gam / G2c)     # ~ q L^2 (t0 - tau)
        kd = float(Kd.upper()) if Kd is not None else None
        rows.append(dict(qlo=float(qlo.mid()), qhi=float(qhi.mid()), Kd=kd, gamma=gamma_coef.str(6), Bnd=float(Bnd.upper())))
        if kd is None:
            print("FAILED on", rows[-1]); worst = float("inf")
        else:
            worst = max(worst, kd)
            if float(qhi.upper()) <= 2 / 3 + 1e-12:
                worst23 = max(worst23, kd)
        if i % max(1, nsub // 24) == 0 or i == nsub - 1:
            print(rows[-1], flush=True)
    print(f"N0={N0} k0={k0} q in (0,{qmax_num}/{qmax_den}]: certified Kd = {worst}   (q <= 2/3: {worst23})   [{time.time()-t0:.1f}s]")
    json.dump(dict(N0=N0, k0=k0, qmax=f"{qmax_num}/{qmax_den}", nsub=nsub, Kd=worst, Kd_q_le_2_3=worst23, rows=rows),
              open(os.path.join(_R, 'data', 'msc_rigorous_1d', f"14_disc_{tag}.json"), "w"), indent=1)
