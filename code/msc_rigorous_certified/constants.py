"""constants.py -- certified enclosures of the two constants (Theorems 23 and 25 of note R5).

(1) tau* : the unique maximiser on (0, infinity) of
        phi(tau) = sum_{m>=1} (-1)^(m+1) (2m-1) exp(-(2m-1)^2 pi^2 tau / 4);
    the 1D constant is  kappa_1 = 2 tau* = 0.33328426549...
    On (0, 1/4], sign phi'(tau) = sign B(tau),
        B(tau) = 1/4 - 3 tau/2 - E(tau),   E(tau) = sum_{n>=1} (-1)^(n+1) b_n(tau),
        b_n(tau) = (2n+1) ((2n+1)^2/4 - 3 tau/2) exp(-n(n+1)/tau),
    with b_1 - b_2 <= E <= b_1 - b_2 + b_3 (alternating series, decreasing positive terms).
(2) C_2 = (8/pi) [gamma + ln(4 sqrt2 / varpi) - 1/2 - pi/4],  varpi = Gamma(1/4)^2 / (2 sqrt(2 pi)).

Trusted base: Arb ball arithmetic (python-flint).  Every sign decision is repeated with mpmath.iv.
Output: certs/constants.json
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import json, os
from fractions import Fraction
from flint import arb, ctx, fmpq
from mpmath import iv

HERE = os.path.dirname(os.path.abspath(__file__))
ctx.prec = 512
iv.prec = 512
checks = {}


def Q(fr):                       # Fraction -> exact Arb ball
    return arb(fmpq(fr.numerator, fr.denominator))


def b_arb(n, tau):
    return (2 * n + 1) * (arb(fmpq((2 * n + 1) ** 2, 4)) - 3 * tau / 2) * (-arb(n * (n + 1)) / tau).exp()


def B_bounds_arb(fr):
    tau = Q(fr)
    b1, b2, b3 = (b_arb(n, tau) for n in (1, 2, 3))
    base = arb(fmpq(1, 4)) - 3 * tau / 2
    return base - (b1 - b2 + b3), base - (b1 - b2)      # (lower bound of B, upper bound of B)


def B_bounds_iv(fr):
    tau = iv.mpf(fr.numerator) / iv.mpf(fr.denominator)
    b = [(2 * n + 1) * (iv.mpf((2 * n + 1) ** 2) / 4 - 3 * tau / 2) * iv.exp(-iv.mpf(n * (n + 1)) / tau) for n in (1, 2, 3)]
    base = iv.mpf(1) / 4 - 3 * tau / 2
    return base - (b[0] - b[1] + b[2]), base - (b[0] - b[1])


def dec(fr, digits, up):
    sc = 10 ** digits
    n = fr.numerator * sc
    v = -((-n) // fr.denominator) if up else n // fr.denominator
    s = str(v).rjust(digits + 1, "0")
    return s[:-digits] + "." + s[-digits:]


# ---------------------------------------------------------------- (1) tau*
lo, hi = Fraction(1, 7), Fraction(1, 6)
assert B_bounds_arb(lo)[0] > 0 and B_bounds_arb(hi)[1] < 0
steps = 0
while hi - lo > Fraction(1, 10 ** 60):
    mid = (lo + hi) / 2
    bl, bu = B_bounds_arb(mid)
    if bl > 0:
        lo = mid
    elif bu < 0:
        hi = mid
    else:
        break
    steps += 1
# final sign certificates at the two rational endpoints, in both arithmetics
bl_lo = B_bounds_arb(lo)[0]
bu_hi = B_bounds_arb(hi)[1]
assert bl_lo > 0 and bu_hi < 0
ivl, ivu = B_bounds_iv(lo)[0], B_bounds_iv(hi)[1]
assert ivl.a > 0 and ivu.b < 0
checks["B(tau_lo) > 0 [Arb, mpmath.iv]"] = True
checks["B(tau_hi) < 0 [Arb, mpmath.iv]"] = True

# auxiliary numerical inequalities used in the uniqueness proof (Theorem 23 of note R5)
e8 = (-arb(8)).exp()
c1 = Fraction(441, 2)                       # 220.5
s_tail = e8 * (1 + 32 * (-arb(16)).exp() / (1 - (-arb(11)).exp()))   # sum_{n>=1} n^5 e^{-4n(n+1)} <= e^-8 (1 + ...)
checks["sum_n |b_n'| <= 220.5 * sum n^5 e^{-4n(n+1)} < 3/2"] = bool(Q(c1) * s_tail < arb(fmpq(3, 2)))
checks["value of that bound"] = float((Q(c1) * s_tail).upper())
checks["(47/9) e^{-16} < 1  (b_n decreasing in n on (0,1/4])"] = bool(arb(fmpq(47, 9)) * (-arb(16)).exp() < 1)
tau1 = arb(27).log() / (2 * arb.pi() ** 2)
checks["ln 27 / (2 pi^2) < 1/4"] = bool(tau1 < arb(fmpq(1, 4)))
checks["ln 27 / (2 pi^2) upper bound"] = float(tau1.upper())
checks["B(1/4) < 0"] = bool(B_bounds_arb(Fraction(1, 4))[1] < 0)
checks["iv: ln 27 / (2 pi^2) < 1/4"] = bool((iv.log(27) / (2 * iv.pi ** 2)).b < 0.25)

# sanity check of the theta transformation: both series for phi agree (enclosures overlap)
def phi_direct(fr, terms=60):
    tau = Q(fr)
    s = arb(0)
    for m in range(1, terms + 1):
        s += (-1) ** (m + 1) * (2 * m - 1) * (-(2 * m - 1) ** 2 * arb.pi() ** 2 * tau / 4).exp()
    tail = (2 * terms + 1) * (-(2 * terms + 1) ** 2 * arb.pi() ** 2 * tau / 4).exp()
    return s, tail
def phi_dual(fr, terms=60):
    tau = Q(fr)
    s = arb(0)
    for n in range(terms):
        s += (-1) ** n * (2 * n + 1) * (-arb((2 * n + 1) ** 2) / (4 * tau)).exp()
    tail = (2 * terms + 1) * (-arb((2 * terms + 1) ** 2) / (4 * tau)).exp()
    return s * (arb.pi() * tau) ** arb(fmpq(-3, 2)), tail * (arb.pi() * tau) ** arb(fmpq(-3, 2))
theta = {}
for fr in (Fraction(1, 20), Fraction(1, 10), lo, Fraction(1, 4), Fraction(1, 2), Fraction(2, 1)):
    a, ta = phi_direct(fr)
    b, tb = phi_dual(fr)
    diff = (a - b)
    bound = ta + tb
    theta[str(fr) if fr != lo else "tau_lo"] = dict(direct=a.str(30), dual=b.str(30),
                                                    agree=bool(abs(diff).upper() <= (bound + abs(diff).rad()).upper() + arb(10) ** -100))
checks["theta transformation sanity (direct vs dual series)"] = all(v["agree"] for v in theta.values())

kappa_lo, kappa_hi = 2 * lo, 2 * hi
out = {
    "tau_star": {"lo": f"{lo.numerator}/{lo.denominator}", "hi": f"{hi.numerator}/{hi.denominator}",
                 "lo_dec": dec(lo, 55, False), "hi_dec": dec(hi, 55, True), "bisection_steps": steps},
    "kappa_1 = 2 tau_star": {"lo_dec": dec(kappa_lo, 55, False), "hi_dec": dec(kappa_hi, 55, True),
                             "width": float(kappa_hi - kappa_lo)},
    "theta_sanity": theta,
}

# ---------------------------------------------------------------- (2) C_2
g14 = arb(fmpq(1, 4)).gamma()
varpi = g14 ** 2 / (2 * (2 * arb.pi()).sqrt())
C2a = 8 / arb.pi() * (arb.const_euler() + (4 * arb(2).sqrt() / varpi).log() - arb(fmpq(1, 2)) - arb.pi() / 4)
C2b = 8 / arb.pi() * (arb.const_euler() + 4 * arb(2).log() + arb.pi().log() / 2 - 2 * g14.log()) - 2 - 4 / arb.pi()
g14i = iv.gamma(iv.mpf(1) / 4)
varpii = g14i ** 2 / (2 * iv.sqrt(2 * iv.pi))
C2i = 8 / iv.pi * (iv.euler + iv.log(4 * iv.sqrt(2) / varpii) - iv.mpf(1) / 2 - iv.pi / 4)
checks["C2: the two closed forms overlap (Arb)"] = bool(C2a.overlaps(C2b))
from mpmath import mp, mpf
mp.prec = 600
alo, ahi = mpf(C2a.lower().str(120, radius=False)), mpf(C2a.upper().str(120, radius=False))
checks["C2: Arb and mpmath.iv enclosures overlap"] = bool(not (C2i.b < alo - mpf(10) ** -100 or C2i.a > ahi + mpf(10) ** -100))
out["C_2"] = {"arb": C2a.str(70), "arb_lower": C2a.lower().str(70, radius=False), "arb_upper": C2a.upper().str(70, radius=False),
              "mpmath_iv": [str(C2i.a), str(C2i.b)], "varpi": varpi.str(60)}
out["checks"] = checks
assert all(v for k, v in checks.items() if isinstance(v, bool)), checks
json.dump(out, open(_os.path.join(_R, 'data', 'msc_rigorous_certified', 'constants.json'), "w"), indent=1)
print(json.dumps(out, indent=1))
