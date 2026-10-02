"""Numerical sanity checks (not proofs) for Section 7 of note R1:
 (1) Lemma 7.1: L^2 f(t) = P -+ E/L and L^4 (f(t+2) - f(t)) = P1 -+ E1/L against the exact PMF (q = 1);
 (2) Lemma 7.2: closed forms and bounds for M', M'', (S c2)', (S S4)', (S^3 S4)', (S^3 c2)', (S^3 c2)'' on a grid;
 (3) Lemma 7.3: the true remainders R_P, R_E, R_P0, R_E0 (50-digit evaluation) lie within the certified bounds rho_*;
 (4) Lemma 4.1 (tail bound) on a few cases;
 (5) Theorem 7.4: predicted class zeros  c L^2 -+ b L + 1 + s2  against the zeros of Psi_D found numerically.
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import os, sys, json
from fractions import Fraction as Fr
sys.path.insert(0, os.path.dirname(__file__))
import mpmath as mp
import core, q1
from core import arb
from lib1d import pmf_exact
core.set_prec(200)
mp.mp.dps = 50
res = {}


def sums(N, t):
    L = mp.mpf(N) - mp.mpf(1) / 2
    P = E = P1 = E1 = mp.mpf(0)
    for k in range(1, N):
        yh = k * mp.pi / 2; x = yh / L; S = mp.sin(x) / x
        ct = mp.cos(x) ** (t - 1)
        if k % 2 == 1:
            sg = 1 if ((k - 1) // 2) % 2 == 0 else -1
            P += sg * yh * S * mp.cos(x / 2) * ct; P1 += -sg * yh ** 3 * S ** 3 * mp.cos(x / 2) * ct
        else:
            tk = 1 if (k // 2) % 2 == 0 else -1
            S4 = mp.sin(x / 2) / (x / 2)
            E += tk * yh ** 2 / 2 * S * S4 * ct; E1 += -tk * yh ** 4 / 2 * S ** 3 * S4 * ct
    return P, E, P1, E1


# (1)
worst = mp.mpf(0)
for N in [3, 6, 11, 20]:
    L = mp.mpf(N) - mp.mpf(1) / 2
    f = pmf_exact(N, Fr(1), 1, 3 * N * N)
    for t in range(1, 2 * N * N):
        P, E, P1, E1 = sums(N, t)
        sgn = -1 if (t - (N - 1)) % 2 == 0 else 1          # near: minus
        ft, ft2 = mp.mpf(f[t].numerator) / f[t].denominator, mp.mpf(f[t + 2].numerator) / f[t + 2].denominator
        worst = max(worst, abs(L ** 2 * ft - (P + sgn * E / L)), abs(L ** 4 * (ft2 - ft) - (P1 + sgn * E1 / L)))
res["lemma71_max_abs_err"] = float(worst)
print("(1) Lemma 7.1 identities, max abs error:", mp.nstr(worst, 3))

# (2)
S = lambda z: mp.sin(mp.sqrt(z)) / mp.sqrt(z)
c2 = lambda z: mp.cos(mp.sqrt(z) / 2)
S4 = lambda z: S(z / 4)
M = lambda z: -2 * mp.log(mp.cos(mp.sqrt(z))) / z
m1, m2 = [mp.mpf(x.str(30, radius=False)) for x in q1.consts()]
viol = mp.mpf("-1")
for i in range(1, 401):
    z = mp.mpf(i) / 400 * mp.pi ** 2 / 9
    x = mp.sqrt(z)
    M1 = (x * mp.tan(x) + 2 * mp.log(mp.cos(x))) / x ** 4
    M2 = (x * x / mp.cos(x) ** 2 - 5 * x * mp.tan(x) - 8 * mp.log(mp.cos(x))) / (2 * x ** 6)
    viol = max(viol, abs(M1 - mp.diff(M, z)) - mp.mpf(10) ** -25, abs(M2 - mp.diff(M, z, 2)) - mp.mpf(10) ** -20)
    viol = max(viol, 1 - M(z), M1 - m1, M2 - m2, -M1, -M2)
    for fn, bd in [(lambda w: S(w) * c2(w), mp.mpf(7) / 24), (lambda w: S(w) * S4(w), mp.mpf(5) / 24),
                   (lambda w: S(w) ** 3 * S4(w), mp.mpf(13) / 24), (lambda w: S(w) ** 3 * c2(w), mp.mpf(5) / 8)]:
        d = mp.diff(fn, z)
        viol = max(viol, abs(d) - bd, d)
    d2 = mp.diff(lambda w: S(w) ** 3 * c2(w), z, 2)
    viol = max(viol, d2 - mp.mpf(111) / 320, -d2)
res["lemma72_max_violation"] = float(viol)
print("(2) Lemma 7.2 bounds, max violation (<= 0 required up to 1e-20):", mp.nstr(viol, 3))


def Gm(j, u):
    return sum((1 if ((k - 1) // 2) % 2 == 0 else -1) * 2 * (k * mp.pi / 4) * (-2 * (k * mp.pi / 4) ** 2) ** j * mp.e ** (-2 * u * (k * mp.pi / 4) ** 2) for k in range(1, 300, 2))


def Em(j, u):
    return sum((1 if (k // 2) % 2 == 0 else -1) * ((k * mp.pi / 2) ** 2 / 2) * (-(k * mp.pi / 2) ** 2 / 2) ** j * mp.e ** (-u * (k * mp.pi / 2) ** 2 / 2) for k in range(2, 300, 2))


# (3)
ok3 = True
rows = []
for N0, (vlo, vhi) in [(21, ("0.30", "0.36")), (101, ("0.32", "0.345")), (601, ("0.2249", "0.2251")), (601, ("0.3327", "0.3339"))]:
    rP, rE, rP0, rE0 = [float(r.upper()) for r in q1.remainders(arb(vlo), arb(vhi), arb(2 * N0 - 1) / 2)]
    for N in [N0, 2 * N0 + 1]:
        L = mp.mpf(N) - mp.mpf(1) / 2
        for vt in [mp.mpf(vlo), (mp.mpf(vlo) + mp.mpf(vhi)) / 2, mp.mpf(vhi)]:
            t = 1 + vt * L * L                      # real t
            P, E, P1, E1 = sums(N, t)
            RP = L ** 4 * (P1 - 2 * Gm(1, vt) - (mp.mpf(5) / 2 * Gm(2, vt) - 2 * vt / 3 * Gm(3, vt)) / L ** 2)
            RE = L ** 2 * (E1 - 2 * Em(1, vt)); RP0 = L ** 2 * (P - Gm(0, vt)); RE0 = L ** 2 * (E - Em(0, vt))
            inside = abs(RP) <= rP and abs(RE) <= rE and abs(RP0) <= rP0 and abs(RE0) <= rE0
            ok3 &= bool(inside)
            rows.append(dict(N0=N0, N=N, v=float(vt), R_P=float(RP), rho_P=rP, R_E=float(RE), rho_E=rE, R_P0=float(RP0), rho_P0=rP0,
                             R_E0=float(RE0), rho_E0=rE0, inside=bool(inside)))
res["lemma73_all_inside"] = ok3; res["lemma73_rows"] = rows
print("(3) Lemma 7.3: all true remainders within the certified bounds:", ok3)
print("    e.g.", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in rows[-2].items()})

# (4)
ok4 = True
for p, a, K in [(1, mp.mpf("0.3"), 5), (7, mp.mpf("0.15"), 11), (3, mp.mpf("0.02"), 31)]:
    tail = sum((k * mp.pi / 4) ** p * mp.e ** (-2 * a * (k * mp.pi / 4) ** 2) for k in range(K, 4000, 2))
    r = (1 + mp.mpf(2) / K) ** p * mp.e ** (-a * mp.pi ** 2 * (K + 1) / 2)
    bound = (K * mp.pi / 4) ** p * mp.e ** (-2 * a * (K * mp.pi / 4) ** 2) / (1 - r)
    ok4 &= bool(r < 1 and tail <= bound)
res["lemma41_ok"] = ok4
print("(4) Lemma 4.1 tail bound:", ok4)

# (5)
c = mp.mpf("0.33328426549494789874852691654344242109703655907113")
G2 = Gm(2, c); b = -Em(1, c) / G2
s2 = -(Gm(3, c) * b * b / 2 + b * Em(2, c) + mp.mpf(5) / 4 * G2 - c / 3 * Gm(3, c)) / G2
out5 = []
for N in [30, 100, 300]:
    L = mp.mpf(N) - mp.mpf(1) / 2
    for name, sgn in (("near", -1), ("far", +1)):
        pred = c * L * L + sgn * b * L + 1 + s2
        t0 = mp.findroot(lambda t: (lambda q: q[2] + sgn * q[3] / L)(sums(N, t)), pred)
        out5.append(dict(N=N, cls=name, predicted=float(pred), t0=float(t0), L_times_diff=float(L * (t0 - pred))))
        print("(5) N=%d %s: predicted zero %.4f, actual %.4f, L*(diff) = %.4f" % (N, name, pred, t0, L * (t0 - pred)))
res["thm74_class_zeros"] = out5
res["b"] = float(b); res["s2"] = float(s2)
json.dump(res, open(_os.path.join(_R, 'data', 'msc_rigorous_1d', '21_q1_sanity.json'), "w"), indent=1)
