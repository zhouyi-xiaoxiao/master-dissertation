"""Re-computation of the constants that are derived by hand in note R1 (each number quoted in the text is printed
next to the expression it comes from, and the corresponding inequality of the text is asserted).

Sections: Prop. 3.9 (exact values), Lemma 5.2 (b_1, b_2), Cor. 5.8, Thm 6.1 (chi(9/10)), Thm 6.3 (N = 4, q = 4/5),
Cor. 6.4 and Remark 6.5, Lemma 7.2 (m_1, m_2), Thm 7.7 (condition (33)), Remark 7.9 (closed form in d = 1),
Remark 4.5 (image sizes), Prop. 8.8 (q_fail(2)).
Output: data/39_hand_constants.json; log: logs/39_hand_constants.log.
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import os, json
import mpmath as mp
import sympy as sp

mp.mp.dps = 30
here = os.path.dirname(os.path.abspath(__file__))
const = json.load(open(_os.path.join(_R, 'data', 'msc_rigorous_1d', '03_constants.json')))
c = mp.mpf(const["c_mid_80_digits"])
ball = lambda key: mp.mpf(const[key].strip("[").split(" +/-")[0])
G2, G3 = ball("G''(c)"), ball("G'''(c)")
kappa, rho = ball("kappa"), ball("rho")
out, ok = {}, True


def rec(name, val, cond=True):
    global ok
    out[name] = mp.nstr(val, 12) if not isinstance(val, (str, bool)) else val
    ok &= bool(cond)
    print(f"{name:55s} {out[name]}   {'ok' if cond else 'FAIL'}")


# Prop 3.9
q, n = sp.symbols("q n")
expr = sp.expand(2 * (sp.Rational(1, 2) * ((n - (n - sp.Rational(1, 2)) * q) ** 2 + (1 - q / 2) ** 2 + (n - 1) * (1 - q) ** 2
                                           + (n - 1) * q ** 2 / 2) - (n - (n - sp.Rational(1, 2)) * q)))
target = n * (n - 1) - n * (2 * n - 1) * q + (n ** 2 + n / 2 - 1) * q ** 2
rec("Prop3.9: 2(h2-h1) identity", str(sp.simplify(expr - target) == 0), sp.simplify(expr - target) == 0)
disc = sp.factor(sp.expand((n * (2 * n - 1)) ** 2 - 4 * n * (n - 1) * (n ** 2 + n / 2 - 1)))
rec("Prop3.9: discriminant", str(disc), sp.simplify(disc - n * (-2 * n ** 2 + 7 * n - 4)) == 0)
qq = sp.Rational(4, 5)
h1 = 3 - sp.Rational(5, 2) * qq
h2 = sp.Rational(1, 2) * (h1 ** 2 + (1 - qq / 2) ** 2 + 2 * (1 - qq) ** 2 + 2 * qq ** 2 / 2)
rec("Prop3.9: N=4,q=4/5: f(3)=f(4)=8/125, f(5)", str((qq / 2) ** 3 * h2),
    (qq / 2) ** 3 * h1 == sp.Rational(8, 125) and (qq / 2) ** 3 * h2 == sp.Rational(208, 3125))

# Lemma 5.2
Lam = lambda w: -mp.log(1 - w) / w
Lam1 = lambda w: (w / (1 - w) + mp.log(1 - w)) / w ** 2
Lam2 = lambda w: 1 / (w * (1 - w) ** 2) - 2 / (w ** 2 * (1 - w)) - 2 * mp.log(1 - w) / w ** 3
for qb, lim1, lim2 in [(mp.mpf(4) / 5, 1.559, 5.425), (mp.mpf(1), 2.455, 12.49)]:
    w = qb / 2
    b1 = max(Lam(w) / 3, 2 * qb * Lam1(w))
    b2 = max(mp.mpf(4) / 45 * Lam(w) + 4 * qb ** 2 * Lam2(w), mp.mpf(8) / 3 * qb * Lam1(w))
    rec(f"Lemma5.2: b1(qbar={mp.nstr(qb,2)}) < {lim1}", b1, b1 < lim1)
    rec(f"Lemma5.2: b2(qbar={mp.nstr(qb,2)}) < {lim2}", b2, b2 < lim2)

# Cor 5.8
rec("Cor5.8: c/4 - kappa", c / 4 - kappa, abs(c / 4 - kappa + mp.mpf("0.0029527130")) < 1e-10)
x = 1.78 + abs(c / 16 - kappa / 4)
rec("Cor5.8: 1.78+|c/16-kappa/4| < 1.7808; 25/24*1.7808 < 1.86", x, x < 1.7808 and 25 / 24 * 1.7808 < 1.86)

# Thm 6.1
chi = -mp.mpf(10) / 9 * mp.log(mp.mpf(4) / 5)
rec("Thm6.1: chi(9/10) = -(10/9) ln(4/5)", chi, abs(chi - 0.2479) < 1e-4 and chi < mp.mpf(1) / 2)

# Thm 6.3 exceptional case
L = mp.mpf(7) / 2
tau = (c * L ** 2 - kappa) / (mp.mpf(4) / 5) + 1 - rho
E = mp.mpf("1.78") / (mp.mpf(4) / 5 * L ** 2)
rec("Thm6.3: N=4, q=4/5: tau", tau, abs(tau - 4.0043) < 1e-4)
rec("Thm6.3: N=4, q=4/5: E; tau-E < 5 < tau+1+E", E, tau - E < 5 < tau + 1 + E)

# Cor 6.4
rec("Cor6.4: kappa + (rho-1)q coefficients", f"{mp.nstr(kappa, 8)} + {mp.nstr(rho - 1, 8)} q",
    kappa < 0.0863 and rho - 1 < 0.9912)
rec("Cor6.4: kappa-(2-rho) > 0.0774", kappa - (2 - rho), kappa - (2 - rho) > 0.0774)
rec("Cor6.4: L^2 > 1.78/0.0774 iff N >= 6", mp.mpf(1.78) / 0.0774, (5.5) ** 2 > 1.78 / 0.0774 > (4.5) ** 2)
rec("Cor6.4 (e3): 0.0863 + 1.78/6.25 <= 0.372", 0.0863 + 1.78 / 6.25, 0.0863 + 1.78 / 6.25 <= 0.372)
rec("Cor6.4 (e3): 1/L^2 <= 1.44/N^2 (equality at N=3)", mp.mpf(6) ** 2 / 5 ** 2, mp.mpf(36) / 25 == mp.mpf("1.44"))
coef = 25 / mp.mpf(24) * (1.78 + abs(c / 16) + 0.25 * (0.0863 + 0.9912))
rec("Cor6.4 (e4): L^-4 coefficient < 2.2", coef, abs(c / 16) < 0.0209 and coef < 2.2)
rec("Cor6.4 (e4): rho - 3/2", rho - 1.5, abs(rho - 1.5 - mp.mpf("0.49117")) < 1e-5)

# Remark 6.5
off45 = kappa / (mp.mpf(4) / 5) - 1 + rho
off12 = kappa / (mp.mpf(1) / 2) - 1 + rho
rec("Rem6.5: q=4/5 offset kappa/q - 1 + rho", off45, abs(off45 - mp.mpf("1.09902")) < 1e-5)
rec("Rem6.5: q=1/2 offset kappa/q - 1 + rho", off12, abs(off12 - mp.mpf("1.16372")) < 1e-5)

# Lemma 7.2
xx = mp.pi / 3
m1 = (xx * mp.tan(xx) + 2 * mp.log(mp.cos(xx))) / xx ** 4
m2 = (xx ** 2 / mp.cos(xx) ** 2 - 5 * xx * mp.tan(xx) - 8 * mp.log(mp.cos(xx))) / (2 * xx ** 6)
rec("Lemma7.2: m1 = M'(pi^2/9)", m1, abs(m1 - mp.mpf("0.35548")) < 1e-5)
rec("Lemma7.2: m2 = M''(pi^2/9)", m2, abs(m2 - mp.mpf("0.32707")) < 1e-5)
rec("Lemma7.2: 1/6+1/20+1/8+1/192 = 111/320", str(sp.Rational(1, 6) + sp.Rational(1, 20) + sp.Rational(1, 8) + sp.Rational(1, 192)),
    sp.Rational(1, 6) + sp.Rational(1, 20) + sp.Rational(1, 8) + sp.Rational(1, 192) == sp.Rational(111, 320))

# Thm 7.7, condition (33)
const32 = mp.log(4 * (mp.pi / 2) ** 3) + 12 * mp.log(10)
rec("Thm7.7: ln(4(pi/2)^3) + 12 ln 10", const32, abs(const32 - mp.mpf("30.37")) < 0.01)
for qv, Lv, lhs_min, rhs_max in [(0.999, 600.5, 108.4, 81.6), (0.9999, 2000.5, 120.08, 91.18)]:
    qv, Lv = mp.mpf(qv), mp.mpf(Lv)
    chiq = min(mp.mpf(1) / 2, -mp.log(2 * qv - 1) / qv)
    lhs, rhs = mp.mpf("0.15") * chiq * Lv ** 2, 8 * mp.log(Lv) + const32
    rec(f"Thm7.7: q={mp.nstr(qv,5)}, L={mp.nstr(Lv,6)}: 0.15 chi L^2 > 8 ln L + 30.37", f"{mp.nstr(lhs,6)} > {mp.nstr(rhs,6)}",
        lhs >= lhs_min - 0.01 and rhs <= rhs_max + 0.01 and lhs > rhs)

# Remark 7.9
lim = mp.log(mp.pi ** 2) / (1 + mp.pi ** 2 / 2)
rec("Rem7.9: ln(pi^2)/(1+pi^2/2)", lim, abs(lim - mp.mpf("0.385768")) < 1e-6)
rec("Rem7.9: ratio to c", lim / c, abs(lim / c - mp.mpf("1.15747")) < 1e-5)

# Remark 4.5
rec("Rem4.5: c - 1/3", c - mp.mpf(1) / 3, abs(c - mp.mpf(1) / 3 + mp.mpf("4.9068e-5")) < 1e-9)
rec("Rem4.5: 3 exp(-4/u) at u=1/3", 3 * mp.exp(-12), abs(3 * mp.exp(-12) - 2e-5) < 0.2e-5)

# Prop 8.8
qf2 = (4 + mp.sqrt(2)) / 7
rec("Prop8.8: q_fail(2) = (4+sqrt2)/7", qf2, abs(qf2 - mp.mpf("0.77346")) < 1e-5)

print("ALL OK:", ok)
json.dump(dict(all_ok=ok, values=out), open(_os.path.join(_R, 'data', 'msc_rigorous_1d', '39_hand_constants.json'), "w"), indent=1)
