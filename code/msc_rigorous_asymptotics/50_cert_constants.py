"""Rigorous enclosures (Arb ball arithmetic, python-flint) of the absolute constants of Section 4 of note R3.

 Z2      = sum_{k in Z^2\\0} |k|^-4 = 4 zeta(2) beta(2) = (2 pi^2/3) G           (closed form; cross-checked by partial sums)
 Z3      = sum_{k in Z^3\\0} |k|^-4   enclosure: partial sum over |k|_inf <= K plus [0, tail bound of Lemma 4.2]
 cbox2   = pi/2 + 1,  cbox3 = 24 int_[0,1]^2 (1+a^2+b^2)^-2                       (tail constants)
 J       = int_0^inf kappa^3,  kappa(z) = e^{-z}(I0(z)+I1(z));  c3 = pi^2 J / 2
 K2(Z1)  = int_0^{Z1} kappa^2,  c_kappa^{lo/hi}
 T_d(s0) = int_{s0}^inf [(1+omega_bar)^d - 1] ds,  omega_bar(s) = 2e^{-s} + 2e^{-3s} + 2 e^{-(64/pi^2-1)s}/(1-e^{-32 s/pi^2})
Output: data/50_cert_constants.json  (decimal strings of lower/upper bounds).
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import json, os, sys, time
from flint import arb, acb, ctx

ctx.prec = 80
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = _os.path.join(_R, 'data', 'msc_rigorous_asymptotics')
PI = arb.pi()


def lo(a):
    return float(a.lower())      # rounding of a rigorous lower endpoint; we always report with outward slack below


def hi(a):
    return float(a.upper())


def enc(a, nd=15):
    """(lower, upper) as floats pushed outward by 1e-15 relative."""
    l, u = float(a.lower()), float(a.upper())
    return [l - abs(l) * 1e-14 - 1e-300, u + abs(u) * 1e-14 + 1e-300]


def kappa_acb(z):
    return z.bessel_i(0, scaled=True) + z.bessel_i(1, scaled=True)


def integral(f, a, b, analytic_aware=False, **kw):
    """acb.integral of f over [a, b].  Entire and meromorphic integrands (kappa^3, kappa^2, (1+obar)^d - 1) may ignore
    the 'analytic' flag (poles give non-finite balls automatically); integrands with branch cuts must be written as
    f(z, an) and forward the flag (analytic_aware=True)."""
    if analytic_aware:
        r = acb.integral(lambda z, an: f(z, an), acb(a), acb(b), **kw)
    else:
        r = acb.integral(lambda z, an: f(z), acb(a), acb(b), **kw)
    assert r.imag.contains(0) or abs(float(r.imag.mid())) < 1e-20
    return r.real


out = {}
t0 = time.time()
# ---------------- Z2
Z2 = 4 * arb(2).zeta() * arb.const_catalan()
out['Z2'] = enc(Z2)
cbox2 = PI / 2 + 1
# cross-check Z2 by partial sum + tail bound (Lemma 4.2):  tail_2(K) <= cbox2/K^2 + 4/(3K^3)
K = 200
S = arb(0)
for a in range(1, K + 1):
    for b in range(0, a + 1):          # 0 <= b <= a, a >= 1: multiplicity 4 if b == 0 or b == a, else 8
        mult = 4 if (b == 0 or b == a) else 8
        S += arb(mult) / arb(a * a + b * b) ** 2
tail2 = cbox2 / K ** 2 + arb(4) / (3 * K ** 3)
out['Z2_partial_K200'] = enc(S)
out['Z2_check_upper'] = enc(S + tail2)
assert S < Z2 and Z2 < S + tail2, 'Z2 cross-check failed'
print('Z2', Z2, ' partial(200)+tail in', S, S + tail2, flush=True)

# ---------------- cbox3 = 24 * int_0^1 [ 1/(2c^2(c^2+1)) + atan(1/c)/(2c^3) ] da,  c = sqrt(1+a^2)
I_UNIT = acb(0, 1)


def inner(a, an):
    """branch cuts: sqrt on (-inf, 0], log on (-inf, 0]; the analytic flag is forwarded, so any evaluation ball that meets
    a cut returns a non-finite value and is subdivided.  atan(w) = (log(1 + i w) - log(1 - i w)) / (2 i) (principal branches)."""
    c2 = 1 + a * a
    c = c2.sqrt(analytic=an)
    w = 1 / c
    at = ((1 + I_UNIT * w).log(analytic=an) - (1 - I_UNIT * w).log(analytic=an)) / (2 * I_UNIT)
    return 1 / (2 * c2 * (c2 + 1)) + at / (2 * c2 * c)
I2 = integral(inner, 0, 1, analytic_aware=True)
cbox3 = 24 * I2
out['cbox2'] = enc(cbox2); out['cbox3'] = enc(cbox3)
print('cbox3', cbox3, flush=True)

# ---------------- Z3: partial sums over |k|_inf <= K (octant representatives with multiplicities)
def P3(K):
    S = arb(0)
    for a in range(0, K + 1):
        for b in range(0, K + 1):
            ab = a * a + b * b
            ma = 1 if a == 0 else 2
            mb = 1 if b == 0 else 2
            for c in range(0, K + 1):
                n2 = ab + c * c
                if n2 == 0:
                    continue
                S += arb(ma * mb * (1 if c == 0 else 2)) / arb(n2 * n2)
    return S
K3 = 60
P = P3(K3)
tail3 = cbox3 / K3 + 3 * cbox2 / K3 ** 2 + arb(2) / K3 ** 3
out['Z3_partial_K60'] = enc(P)
out['Z3'] = [enc(P)[0], enc(P + tail3)[1]]
print('Z3 in', P, P + tail3, ' t=%.1fs' % (time.time() - t0), flush=True)
json.dump(out, open(DATA + '/50_cert_constants.json', 'w'), indent=1)

# ---------------- J = int_0^inf kappa^3
ZJ = 2000
Jmain = integral(lambda z: kappa_acb(z) ** 3, 0, ZJ)
c32 = (2 / PI) ** arb('1.5')
tail_hi = c32 * 2 / arb(ZJ).sqrt()                        # kappa <= sqrt(2/(pi z))
# kappa >= sqrt(2/(pi z)) (1 - 1/(4z)) - e^{-2z}/(pi z)  >= sqrt(2/(pi z)) (1 - 1/(3z))  for z >= ZJ  (e^{-2z}/(pi z) tiny)
# => kappa^3 >= (2/(pi z))^{3/2} (1 - 1/z);   int_ZJ^inf z^{-3/2}(1 - 1/z) dz = 2 ZJ^{-1/2} - (2/3) ZJ^{-3/2}
zz = arb(ZJ)
assert (-2 * zz).exp() / (PI * zz) < (2 / (PI * zz)).sqrt() / (12 * zz)
tail_lo = c32 * (2 / zz.sqrt() - arb(2) / 3 / zz ** arb('1.5'))
Jlo = Jmain + tail_lo; Jhi = Jmain + tail_hi
J = Jlo.union(Jhi)
c3 = PI ** 2 * J / 2
out['J'] = enc(J); out['c3'] = enc(c3); out['ln_6c3'] = enc((6 * c3).log())
print('J', J, 'c3', c3, 'ln(6 c3)', (6 * c3).log(), ' t=%.1fs' % (time.time() - t0), flush=True)
json.dump(out, open(DATA + '/50_cert_constants.json', 'w'), indent=1)

# ---------------- K2(Z1), c_kappa
for Z1 in (1, 10, 20):
    K2 = integral(lambda z: kappa_acb(z) ** 2, 0, Z1)
    z1 = arb(Z1)
    ck_hi = K2 - 2 / PI * z1.log()
    ck_lo = ck_hi - 1 / (PI * z1) - (2 / PI).sqrt() / PI * (-2 * z1).exp() / z1 ** arb('1.5')
    out['K2_%d' % Z1] = enc(K2); out['ckappa_lo_%d' % Z1] = enc(ck_lo)[0]; out['ckappa_hi_%d' % Z1] = enc(ck_hi)[1]
    print('Z1', Z1, 'K2', K2, 'ck in [', ck_lo, ',', ck_hi, ']', flush=True)
out['kappa_1'] = enc(kappa_acb(acb(1)).real)

# ---------------- T_d(s0)
def obar(s):
    return 2 * (-s).exp() + 2 * (-3 * s).exp() + 2 * (-(64 / acb.pi() ** 2 - 1) * s).exp() / (1 - (-32 * s / acb.pi() ** 2).exp())
for d in (2, 3):
    for s0 in ('0.25', '0.5', '1'):
        Smax = 60
        main = integral(lambda s: (1 + obar(s)) ** d - 1, arb(s0), Smax)
        tail = arb(7) * (-arb(Smax)).exp()      # omega_bar <= 2.1 e^{-s} for s >= 60 => (1+ob)^d - 1 <= d ob (1+ob)^{d-1} <= 7 e^{-s}
        T = main.union(main + tail)
        out['T%d_s0_%s' % (d, s0)] = enc(T)
        print('T_%d(%s) =' % (d, s0), T, flush=True)
json.dump(out, open(DATA + '/50_cert_constants.json', 'w'), indent=1)
print('done t=%.1fs' % (time.time() - t0))
