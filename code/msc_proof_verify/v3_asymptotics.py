"""V3: large-N expansion (Theorem 4.5 of the article).  Separately written; uses no script of the main implementation.
 (a) q*T_N at 70 digits from form (S) for N=2^j, 3^j, 10^j and selected sizes (checkpointed);
 (b) 'blind' Richardson / linear-solve estimates of the coefficients, with tests for
     forbidden terms (N ln N, N, odd powers);
 (c) the classical constants used in the proof (I_r, E_1, Lambert series, Eisenstein values);
 (d) the coefficient formulas (8.2),(8.4),(8.5) evaluated *numerically* (no modular forms)
     and compared with the closed forms (3.8);
 (e) the literature constants (Essam-Wu 2009; Izmailian-Huang 2010 Eqs. (42),(43)).
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import json, os, sys, time
import mpmath as mp

HERE = os.path.dirname(os.path.abspath(__file__))
VAL = _os.path.join(_R, 'data', 'msc_proof_verify', 'v3_values.json')
OUT = _os.path.join(_R, 'data', 'msc_proof_verify', 'v3_asymptotics.json')
mp.mp.dps = 70
t0 = time.time()


def qT_S_fast(N):
    """form (S); g_k = 1 to working precision once e^{-2u} < 10^-80."""
    tot = mp.mpf(0)
    corr = mp.mpf(0)
    h = mp.pi / (2 * N)
    for k in range(1, N):
        s = mp.sin(k * h)
        w = (1 - s * s) * mp.sqrt(1 + s * s) / s
        tot += w
        if k <= 120:                       # N*asinh(s_k) >= 0.8813*k  ->  e^{-2u} <= e^{-211}
            u = N * mp.asinh(s)
            g = mp.coth(u) if k % 2 else mp.tanh(u)
            corr += w * (g - 1)
    return 4 * N * (tot + corr)


# ---------------------------------------------------------------- (a) values
vals = json.load(open(VAL)) if os.path.exists(VAL) else {}
Ns = sorted(set([2 ** j for j in range(3, 18)] + [3 ** j for j in range(2, 11)] + [10 ** j for j in range(1, 6)] + [35, 100, 1000, 4001]))
for N in Ns:
    if str(N) not in vals:
        vals[str(N)] = mp.nstr(qT_S_fast(N), 68)
        json.dump(vals, open(VAL, 'w'), indent=1)
        print('value', N, round(time.time() - t0, 1), flush=True)
V = {int(k): mp.mpf(v) for k, v in vals.items()}
# cross-check the fast evaluation against the plain formulas for a few N
import vlib
chk = max(abs(qT_S_fast(N) - vlib.qT_form_a(N)) / V[N] if N in V else 0 for N in (8, 9, 27, 64, 100))
chk2 = max(abs(V[N] - vlib.qT_double_corner(N)) / V[N] for N in (8, 9, 27, 64))

# ---------------------------------------------------------------- closed forms claimed
pi = mp.pi
varpi = mp.gamma(mp.mpf(1) / 4) ** 2 / (2 * mp.sqrt(2 * pi))
X = varpi ** 4 / pi ** 2
C = {2: 8 / pi * (mp.euler + mp.log(4 * mp.sqrt(2) / varpi) - mp.mpf(1) / 2 - pi / 4),
     0: X / 9,
     -2: -pi * X * (25 * X + 648) / 10800,
     -4: pi ** 2 * X ** 2 * (1225 * X + 12879) / 1905120,
     -6: -pi ** 3 * X ** 2 * (79625 * X ** 2 + 1215000 * X + 960498) / 435456000,
     -8: pi ** 4 * X ** 3 * (7703465 * X ** 2 + 92129400 * X + 118943883) / 63228211200}
R = {'dps': 70, 'fast_vs_form_a_rel': float(chk), 'fast_vs_double_sum_rel': float(chk2),
     'closed_forms': {str(k): mp.nstr(v, 40) for k, v in C.items()}, 'varpi': mp.nstr(varpi, 40), 'X': mp.nstr(X, 40)}

# ---------------------------------------------------------------- (b) blind estimates
def blind_solve(Nlist, powers, logs=()):
    """Solve exactly for coefficients c_p in  qT - (8/pi) N^2 ln N = sum_p c_p N^p (+ extra log terms)
    using len(powers)+len(logs) values of N.  powers: list of exponents; logs: list of (p) meaning N^p ln N."""
    m = len(powers) + len(logs)
    assert len(Nlist) == m
    A = mp.matrix(m, m)
    b = mp.matrix(m, 1)
    for i, N in enumerate(Nlist):
        Nm = mp.mpf(N)
        for j, p in enumerate(powers):
            A[i, j] = Nm ** p
        for j, p in enumerate(logs):
            A[i, len(powers) + j] = Nm ** p * mp.log(Nm)
        b[i] = V[N] - 8 / pi * Nm ** 2 * mp.log(Nm)
    x = mp.lu_solve(A, b)
    return [x[i] for i in range(m)]


fam2 = [2 ** j for j in range(3, 18)]
fam3 = [3 ** j for j in range(2, 11)]
blind = {}
# even powers only, as claimed: N^2, N^0, ..., N^{-2(m-2)}
for name, fam, m in (('2^j (top 12)', fam2[-12:], 12), ('2^j (top 9)', fam2[-9:], 9), ('3^j (all 9, odd N)', fam3, 9)):
    powers = [2 - 2 * i for i in range(m)]
    x = blind_solve(fam, powers)
    blind[name] = {str(p): {'estimate': mp.nstr(x[i], 30), 'minus_closed': (mp.nstr(x[i] - C[p], 5) if p in C else None)} for i, p in enumerate(powers[:7])}
R['blind_even_power_fit'] = blind
# allow forbidden terms: N ln N, N, N^-1, N^-3 and check that they vanish
powers = [2, 1, 0, -1, -2, -3, -4, -5, -6, -7, -8]
x = blind_solve(fam2[-12:], powers, logs=(1,))
R['forbidden_terms_2^j'] = {**{f'N^{p}': mp.nstr(x[i], 8) for i, p in enumerate(powers)}, 'N ln N': mp.nstr(x[len(powers)], 8)}
x = blind_solve(fam3[-8:], [2, 1, 0, -1, -2, -3, -4], logs=(1,))
R['forbidden_terms_3^j'] = {**{f'N^{p}': mp.nstr(x[i], 8) for i, p in enumerate([2, 1, 0, -1, -2, -3, -4])}, 'N ln N': mp.nstr(x[7], 8)}
# is the leading coefficient really 8/pi?  fit a free coefficient of N^2 ln N
def lead_fit(fam, m):
    powers = [2 - 2 * i for i in range(m - 1)]
    A = mp.matrix(m, m); b = mp.matrix(m, 1)
    for i, N in enumerate(fam[-m:]):
        Nm = mp.mpf(N)
        A[i, 0] = Nm ** 2 * mp.log(Nm)
        for j, p in enumerate(powers):
            A[i, j + 1] = Nm ** p
        b[i] = V[N]
    return mp.lu_solve(A, b)[0]
R['leading_coefficient_minus_8/pi'] = {'2^j': mp.nstr(lead_fit(fam2, 10) - 8 / pi, 5), '3^j': mp.nstr(lead_fit(fam3, 8) - 8 / pi, 5)}

# truncation errors at selected sizes
def series(N, terms):
    Nm = mp.mpf(N)
    s = 8 / pi * Nm ** 2 * mp.log(Nm)
    for p in [2, 0, -2, -4, -6, -8][:terms - 1]:
        s += C[p] * Nm ** p
    return s
tab = []
for N in (10, 35, 100, 1000, 4001, 10 ** 5):
    row = {'N': N, 'qT': mp.nstr(V[N], 25)}
    for terms in range(1, 8):
        row[f'rel_err_{terms}_terms'] = float(abs(series(N, terms) - V[N]) / V[N])
    tab.append(row)
R['truncation_table'] = tab
json.dump(R, open(OUT, 'w'), indent=1)
print('blind done', round(time.time() - t0, 1), flush=True)

# ---------------------------------------------------------------- (c) classical constants
mp.mp.dps = 50
f = lambda y: mp.cos(y) ** 2 * mp.sqrt(1 + mp.sin(y) ** 2) / mp.sin(y)
Ir_num = mp.quad(lambda y: f(y) - 1 / y, mp.linspace(0, mp.pi / 2, 9))
Ir_closed = mp.mpf(3) / 2 * mp.log(2) - mp.log(pi) - mp.mpf(1) / 2
p = -mp.exp(-pi)
U = lambda v: v / (1 + v)
E1_num = -2 * mp.nsum(lambda k: U(p ** k) / k, [1, mp.inf])
E1_closed = mp.log(2 * pi / varpi) - pi / 4


def Eis(wt, z):
    """E_2, E_4, E_6 as q-series at nome z=e^{2 pi i tau}."""
    c = {2: -24, 4: 240, 6: -504}[wt]
    return 1 + c * mp.nsum(lambda n: n ** (wt - 1) * z ** n / (1 - z ** n), [1, mp.inf])


zi = mp.exp(-2 * pi)
e4 = 3 * varpi ** 4 / pi ** 4
Lam1 = mp.nsum(lambda k: k * U(p ** k), [1, mp.inf])
dLam1 = mp.nsum(lambda k: k ** 2 * p ** k / (1 + p ** k) ** 2, [1, mp.inf])
Lam3 = mp.nsum(lambda k: k ** 3 * U(p ** k), [1, mp.inf])
dLam3 = mp.nsum(lambda k: k ** 4 * p ** k / (1 + p ** k) ** 2, [1, mp.inf])
d2Lam3 = mp.nsum(lambda k: k ** 5 * p ** k * (1 - p ** k) / (1 + p ** k) ** 3, [1, mp.inf])
R['classical'] = {
    'I_r quad - (8.3)': mp.nstr(Ir_num - Ir_closed, 5),
    'E_1 series - (8.8)': mp.nstr(E1_num - E1_closed, 5),
    'E2(i)-3/pi': mp.nstr(Eis(2, zi) - 3 / pi, 5), 'E4(i)-3varpi^4/pi^4': mp.nstr(Eis(4, zi) - e4, 5), 'E6(i)': mp.nstr(Eis(6, zi), 5),
    'E2(tau0)-6/pi': mp.nstr(Eis(2, p) - 6 / pi, 5), 'E4(tau0)+4E4(i)': mp.nstr(Eis(4, p) + 4 * e4, 5), 'E6(tau0)': mp.nstr(Eis(6, p), 5),
    'Lambda1(p)+1/24': mp.nstr(Lam1 + mp.mpf(1) / 24, 5), 'thetaLambda1(p)+e/36': mp.nstr(dLam1 + e4 / 36, 5),
    'Lambda3(p)-(1-6e)/240': mp.nstr(Lam3 - (1 - 6 * e4) / 240, 5), 'thetaLambda3(p)+e/(20pi)': mp.nstr(dLam3 + e4 / (20 * pi), 5),
    'theta2Lambda3(p)-(-e/(8pi^2)+e^2/216)': mp.nstr(d2Lam3 - (-e4 / (8 * pi ** 2) + e4 ** 2 / 216), 5),
    'theta3(e^-pi)-pi^(1/4)/Gamma(3/4)': mp.nstr(mp.jtheta(3, 0, mp.exp(-pi)) - pi ** 0.25 / mp.gamma(0.75), 5),
    'prod(1-e^{-2 pi n}) - e^{pi/12}Gamma(1/4)/(2 pi^{3/4})': mp.nstr(mp.qp(zi) - mp.exp(pi / 12) * mp.gamma(0.25) / (2 * pi ** 0.75), 5),
}
json.dump(R, open(OUT, 'w'), indent=1)
print('classical done', round(time.time() - t0, 1), flush=True)

# ---------------------------------------------------------------- (d) coefficient formulas, numerically
MMAX = 5
# exact rational Taylor series (own power-series arithmetic, truncated at y^DEG)
from fractions import Fraction as Fr
from math import factorial as fct
DEG = 2 * MMAX + 4
def smul(a, b):
    c = [Fr(0)] * (DEG + 1)
    for i, ai in enumerate(a):
        if ai:
            for j, bj in enumerate(b):
                if i + j > DEG:
                    break
                c[i + j] += ai * bj
    return c
def sinv(a):                      # 1/a, a[0] != 0
    b = [Fr(0)] * (DEG + 1)
    b[0] = 1 / a[0]
    for n in range(1, DEG + 1):
        b[n] = -sum(a[i] * b[n - i] for i in range(1, n + 1)) / a[0]
    return b
def scompose(coeffs, w):          # sum_j coeffs[j] w^j, w[0]==0
    out = [Fr(0)] * (DEG + 1)
    pw = [Fr(1)] + [Fr(0)] * DEG
    for cj in coeffs:
        out = [o + cj * x for o, x in zip(out, pw)]
        pw = smul(pw, w)
    return out
SIN = [Fr(0)] * (DEG + 1); COS = [Fr(0)] * (DEG + 1)
for i in range(DEG + 1):
    if i % 2:
        SIN[i] = Fr((-1) ** (i // 2), fct(i))
    else:
        COS[i] = Fr((-1) ** (i // 2), fct(i))
SIN2 = smul(SIN, SIN)
binom_half = [Fr(1)]
for j in range(1, DEG // 2 + 2):
    binom_half.append(binom_half[-1] * (Fr(1, 2) - (j - 1)) / j)
SQ = scompose(binom_half, SIN2)                       # sqrt(1+sin^2 y)
SINC = SIN[1:] + [Fr(0)]                              # sin(y)/y
yf = smul(smul(smul(COS, COS), SQ), sinv(SINC))       # y f(y) = phi(y^2)
phi = [mp.mpf(yf[2 * i].numerator) / yf[2 * i].denominator for i in range(MMAX + 1)]
phi_exact = [yf[2 * i] for i in range(MMAX + 1)]
# asinh(s) = sum (-1)^j (2j)!/(4^j j!^2 (2j+1)) s^(2j+1)  ->  asinh(sin y)/y
ash = [Fr(0)] * (DEG + 2)
for j in range(DEG // 2 + 1):
    ash[2 * j + 1] = Fr((-1) ** j * fct(2 * j), 4 ** j * fct(j) ** 2 * (2 * j + 1))
AS = scompose(ash[:DEG + 1], SIN)
om_exact = [AS[2 * i + 1] for i in range(MMAX + 1)]   # coefficients of asinh(sin y)/y in Y=y^2
Om = [mp.mpf(c.numerator) / c.denominator for c in om_exact]; Om[0] = mp.mpf(0)
def pmul(a, b):
    c = [mp.mpf(0)] * (MMAX + 1)
    for i, ai in enumerate(a):
        for j, bj in enumerate(b):
            if i + j <= MMAX:
                c[i + j] += ai * bj
    return c
# kappa[n][m] = [Y^m] phi(Y) (-Omega(Y))^n / n!
kappa = []
cur = phi[:]
for n in range(MMAX + 1):
    kappa.append([c / mp.factorial(n) for c in cur])
    cur = pmul(cur, [-c for c in Om])
# D^n U (v) = v P_n(v)/(1+v)^{n+1}, P_{n+1} = (P_n + v P_n')(1+v) - (n+1) v P_n
Ppoly = [[mp.mpf(1)]]
for n in range(MMAX):
    P = Ppoly[-1]
    d = [i * c for i, c in enumerate(P)]            # v P'
    a = [P[i] + d[i] for i in range(len(P))]
    new = [mp.mpf(0)] * (len(P) + 1)
    for i, c in enumerate(a):
        new[i] += c; new[i + 1] += c
    for i, c in enumerate(P):
        new[i + 1] -= (n + 1) * c
    Ppoly.append(new)
def DnU(n, v):
    return v * mp.polyval(Ppoly[n][::-1], v) / (1 + v) ** (n + 1)
# sanity: D^n U by numerical differentiation in t (v = e^t)
sanity = max(abs(DnU(n, mp.mpf('0.3')) - mp.diff(lambda t: U(mp.exp(t)), mp.log(mp.mpf('0.3')), n)) for n in range(1, 4))
coef = {}
for m in range(0, MMAX + 1):
    if m == 0:
        b = -16 / pi * mp.nsum(lambda k: U(p ** k) / k, [1, mp.inf])
        val = 8 / pi * (mp.euler + Ir_num) + b
    else:
        b = mp.mpf(0)
        for n in range(0, m + 1):
            e = 2 * m - 1 + n
            b += kappa[n][m] * pi ** e * mp.nsum(lambda k: k ** e * DnU(n, p ** k), [1, mp.inf])
        b *= -mp.mpf(16) / 4 ** m
        # a_m = -4 B_2m/(2m)! (pi/2)^(2m-1) r^(2m-1)(0);  r^(2m-1)(0) = (2m-1)! * [y^(2m-1)] r = (2m-1)! * phi_m
        a = -4 * mp.bernoulli(2 * m) / mp.factorial(2 * m) * (pi / 2) ** (2 * m - 1) * mp.factorial(2 * m - 1) * phi[m]
        val = a + b
    coef[2 - 2 * m] = val
R['coefficient_formula_numeric_vs_closed'] = {str(k): {'formula(8.2)+(8.4)': mp.nstr(v, 35), 'minus_closed_form': mp.nstr(v - C[k], 5)} for k, v in coef.items()}
R['DnU_sanity'] = mp.nstr(sanity, 5)
R['kappa_check'] = {'k01': mp.nstr(kappa[0][1], 12), 'k11': mp.nstr(kappa[1][1], 12), 'k02': mp.nstr(kappa[0][2], 12), 'k12': mp.nstr(kappa[1][2], 12), 'k22': mp.nstr(kappa[2][2], 12),
                    'r_taylor_y1,y3,y5 (exact)': [str(phi_exact[1]), str(phi_exact[2]), str(phi_exact[3])], 'Omega_Y1,Y2 (exact)': [str(om_exact[1]), str(om_exact[2])]}
json.dump(R, open(OUT, 'w'), indent=1)
print('coef done', round(time.time() - t0, 1), flush=True)

# ---------------------------------------------------------------- (e) literature constants
EW_c0 = 1 + 2 / pi * (2 * mp.euler - 1 + mp.log(2 / pi ** 2)) + 4 / pi * mp.nsum(lambda n: (mp.tanh(pi * n) - 1) / n, [1, mp.inf])
EW_dom = 1 + 2 / pi * (2 * mp.euler - 1 + mp.log(2 / pi ** 2))
th = lambda k, t: mp.jtheta(k, 0, mp.exp(-pi * t))
IH_c0 = 2 / pi * (2 * mp.log(8 / pi) + 2 * mp.euler - 1 - mp.log(2) - pi / 2 - 2 * mp.log(th(2, 1) * th(4, 1)))
g = lambda t: th(4, t) ** 4 - th(2, t) ** 4
IH_c2 = pi / 72 * (2 * g(1) + 2 * mp.diff(g, 1))
R['literature'] = {
    'EssamWu_c0_from_their_eq(43)_formula': mp.nstr(EW_c0, 25), 'EssamWu_dominant_part': mp.nstr(EW_dom, 20),
    'EssamWu_printed': {'c0': '0.077318893909458', 'c2': '0.266070441638478', 'c4': '-0.534779473843066', 'dominant': '0.082069879627328', 'correction': '-0.0047509857178700465'},
    'ours_C2/2': mp.nstr(C[2] / 2, 25), 'ours_C0/2': mp.nstr(C[0] / 2, 25), 'ours_C-2/2': mp.nstr(C[-2] / 2, 25),
    'C2/2 - EW formula': mp.nstr(C[2] / 2 - EW_c0, 5),
    'IzmailianHuang_printed': {'c0': '0.07731889390945876', 'c2': '0.26607044163847837'},
    'IH_eq42_at_rho=xi=1 - C2/2': mp.nstr(IH_c0 - C[2] / 2, 5), 'IH_eq43_at_rho=xi=1 - C0/2': mp.nstr(IH_c2 - C[0] / 2, 5),
}
R['seconds'] = round(time.time() - t0, 1)
json.dump(R, open(OUT, 'w'), indent=1)
print('done', R['seconds'])
