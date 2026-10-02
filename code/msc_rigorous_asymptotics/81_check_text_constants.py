"""Interval-arithmetic (mpmath.iv) checks of the small numerical claims made in the text of note R3 that are not produced by the
block certificates: Corollary 4.7 (E U bounds), display (6.1), Corollaries 7.15 and 7.18 (arithmetic of the error numerators),
Lemma 7.16 (monotonicity facts).  Every check prints PASS/FAIL; the script exits with an error if any check fails.
Output: logs/81_check_text_constants.log (stdout)."""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import json, os, sys
from mpmath import iv, mpf
iv.prec = 100
HERE = os.path.dirname(os.path.abspath(__file__))
TAB = json.load(open(_os.path.join(_R, 'data', 'msc_rigorous_asymptotics', '80_tables.json')))
C3 = json.load(open(_os.path.join(_R, 'data', 'msc_rigorous_asymptotics', '50_cert_constants.json')))['c3']
c3lo = iv.mpf(C3[0])
fails = []


def check(name, cond, info=''):
    ok = bool(cond)
    print(('PASS ' if ok else 'FAIL ') + name + ('   ' + str(info) if info else ''))
    if not ok:
        fails.append(name)


def le(a, b):            # a <= b, certainly
    return iv.mpf(a).b <= iv.mpf(b).a


def lt(a, b):
    return iv.mpf(a).b < iv.mpf(b).a


# ---------------- Corollary 4.7
x = iv.pi / 20
c = iv.cos(x) ** 2
f = 2 * iv.cos(2 * x) ** 2 * iv.exp(-4 * c * iv.log(2 * c))
check('Cor4.7: f(pi/20) = gamma_2 A^-eps_2 at N=10 <= 0.1333', le(f, '0.1333'), f)
check('Cor4.7: 1 + ln(2 cos^2(pi/20)) >= 1.668', le('1.668', 1 + iv.log(2 * c)))
check('Cor4.7: 1/cos(pi/10) <= 1.052', le(1 / iv.cos(2 * x), '1.052'))
check('Cor4.7: A(10) = 2cos^2(pi/20) >= 1.9507', le('1.9507', 2 * c))
check('Cor4.7: eps_2(10) = 4cos^2(pi/20) >= 3.9014', le('3.9014', 4 * c))
fb = iv.mpf('0.1333'); lnA = iv.log(iv.mpf('1.9507'))
EU2 = iv.mpf(3) / 2 - 2 * fb / iv.mpf('3.9014') + lnA * (1 - fb ** 2)
EU3 = iv.mpf(11) / 6 - 3 * fb / iv.mpf('3.9014') + lnA * (1 - fb ** 3)
check('Cor4.7: EU_lo(d=2) >= 2.0879 >= 2.08', le('2.0879', EU2), EU2)
check('Cor4.7: EU_lo(d=3) >= 2.3974 >= 2.39', le('2.3974', EU3), EU3)
check('Cor4.7: EU_hi(d=2) = ln 2 + 3/2 <= 2.20', le(iv.log(2) + iv.mpf(3) / 2, '2.20'))
check('Cor4.7: EU_hi(d=3) = ln 2 + 11/6 <= 2.53', le(iv.log(2) + iv.mpf(11) / 6, '2.53'))
check('Cor4.7: 3 pi + 0.585 <= 10.02', le(3 * iv.pi + iv.mpf('0.585'), '10.02'))
from fractions import Fraction as Fr
check('Cor4.7: lower X, d=2: -2.32 + 2.08 = -0.24 (exact decimals)', Fr('-2.32') + Fr('2.08') == Fr('-0.24'))
check('Cor4.7: upper X, d=2: -1.60 + 2.20 = 0.60 (exact decimals)', Fr('-1.60') + Fr('2.20') == Fr('0.60'))
check('Cor4.7: lower X, d=3: -(3 pi + 0.585) + 2.39 >= -7.63', le('-7.63', -(3 * iv.pi + iv.mpf('0.585')) + iv.mpf('2.39')))
check('Cor4.7: upper X, d=3: -8.38 + 2.53 = -5.85 (exact decimals)', Fr('-8.38') + Fr('2.53') == Fr('-5.85'))
# ---------------- (6.1)
u = iv.mpf('1.1e-3')
check('(6.1): (3-2u)/(1-u)^2 <= 3.01 at u = 1.1e-3 (increasing in u)', le((3 - 2 * u) / (1 - u) ** 2, '3.01'))
check('(6.1): 2 + 3.01*1.1e-3 <= 2.01', le(2 + iv.mpf('3.01') * u, '2.01'))
check('(6.1): u(N=20) = pi^2/9600 <= 1.03e-3', le(iv.pi ** 2 / 9600, '1.03e-3'))
check('(6.1): 2.01 pi^2/24 <= 0.83', le(iv.mpf('2.01') * iv.pi ** 2 / 24, '0.83'), iv.mpf('2.01') * iv.pi ** 2 / 24)


# ---------------- Corollaries 7.15 / 7.18 (error numerators; upper-bound condition at the smallest N of the range)
def tab(key, N0):
    for e in TAB[key]['table']:
        if e[0] == N0:
            return e[1], e[2]
    raise KeyError((key, N0))


def upper_ok_d2(Kp, N):          # 0.62 + K^+ < 3 ln(4X) X/(X+3) for X >= 2 pi ln N - 0.24 (increasing in X); also X <= 1.032 L needs N >= 20
    X = 2 * iv.pi * iv.log(N) - iv.mpf('0.24')
    return lt(iv.mpf('0.62') + iv.mpf(repr(Kp)), 3 * iv.log(4 * X) * X / (X + 3)), 3 * iv.log(4 * X) * X / (X + 3)


def upper_ok_d3(Kp, N):          # K^+ < 5 ln(6X) X/(X+5) for X >= c3 N - 7.63
    X = c3lo * N - iv.mpf('7.63')
    return lt(iv.mpf(repr(Kp)), 5 * iv.log(6 * X) * X / (X + 5)), 5 * iv.log(6 * X) * X / (X + 5)


# Cor 7.15: d=2, q<=0.95, N>=100 (max over q_max in {0.8,0.9,0.95} at N0=100) and q<=0.99, N>=312
Km2 = max(tab('T714_q%s_d2' % t, 100)[0] for t in ('0p8', '0p9', '0p95')); Kp2 = max(tab('T714_q%s_d2' % t, 100)[1] for t in ('0p8', '0p9', '0p95'))
Km2b, Kp2b = tab('T714_q0p99_d2', 312)
Km3 = max(tab('T714_q%s_d3' % t, 100)[0] for t in ('0p8', '0p9', '0p95')); Kp3 = max(tab('T714_q%s_d3' % t, 100)[1] for t in ('0p8', '0p9', '0p95'))
Km3b, Kp3b = tab('T714_q0p99_d3', 312)
print('Cor7.15 constants: d=2 N0=100 (%.3f, %.3f), N0=312 (%.3f, %.3f); d=3 N0=100 (%.3f, %.3f), N0=312 (%.3f, %.3f)' % (Km2, Kp2, Km2b, Kp2b, Km3, Kp3, Km3b, Kp3b))
check('Cor7.15 d=2: 0.24 + max K^- <= 1.72', le(iv.mpf('0.24') + iv.mpf(repr(max(Km2, Km2b))), '1.72'))
check('Cor7.15 d=2: upper bound condition at N=100', upper_ok_d2(max(Kp2, Kp2b), 100)[0], upper_ok_d2(max(Kp2, Kp2b), 100)[1])
check('Cor7.15 d=3: 7.63 + max K^- <= 9.28', le(iv.mpf('7.63') + iv.mpf(repr(max(Km3, Km3b))), '9.28'))
check('Cor7.15 d=3: upper bound condition at N=100', upper_ok_d3(max(Kp3, Kp3b), 100)[0], upper_ok_d3(max(Kp3, Kp3b), 100)[1])
# Cor 7.18: uniform in q; N >= max(N_*, N_1(q)) and N >= max(100, N_1(q))
for d, Nst, num_st, num_100 in ((2, 59, '2.33', '1.75'), (3, 12, '14.01', '9.31')):
    first = TAB['T717_d%d' % d]['first_ok']
    check('Cor7.18 d=%d: certificate passes from N_* = %d' % (d, Nst), first == Nst, first)
    for N0, num in ((Nst, num_st), (100, num_100)):
        Km, Kp = tab('T717_d%d' % d, N0)
        base = '0.24' if d == 2 else '7.63'
        check('Cor7.18 d=%d N0=%d: %s + K^- (%.3f) <= %s' % (d, N0, base, Km, num), le(iv.mpf(base) + iv.mpf(repr(Km)), num))
        okU, val = (upper_ok_d2 if d == 2 else upper_ok_d3)(Kp, N0)
        check('Cor7.18 d=%d N0=%d: upper bound condition (K^+ = %.3f)' % (d, N0, Kp), okU, val)
# ---------------- Lemma 7.16 monotonicity facts (x = pi/(2N) <= pi/20)
xm = iv.pi / 20
for a in (1, 2):
    for d in (2, 3):
        check('Lemma7.16(c): 4 cos 2x > (2a/d) sin 2x on x<=pi/20 (a=%d, d=%d)' % (a, d),
              lt(iv.mpf(2 * a) / d * iv.sin(2 * xm), 4 * iv.cos(2 * xm)))
check('Lemma7.16(c): sin 2x < d/2 (h increasing), x <= pi/20', lt(iv.sin(2 * xm), 1))
check('Lemma7.16(a): sin x <= d cos x, x <= pi/20', le(iv.sin(xm), 2 * iv.cos(xm)))
check('Lemma7.16(a): sin 2x < 1 for N >= 4', lt(iv.sin(iv.pi / 4), 1))
print('ALL PASS' if not fails else 'FAILURES: %s' % fails)
sys.exit(1 if fails else 0)
