"""mfpt2d_remainder.py -- certified remainder of the two-term 2D MFPT law (Theorem 26 and Proposition 27 of note R5)

    r_N := tau_N - (8/pi) N^2 ln N - C_2 N^2,   tau_N = q * MFPT (corner to corner, N x N),

(a) for 2 <= N <= N_A from the rigorous linear-system enclosures certs/mfpt_d2.jsonl (self-contained);
(b) for N_A < N <= N_B from the single sum  tau_N = 4N sum_{k=1}^{N-1} c_k^2 sqrt(1+s_k^2)/s_k * g_k,
    g_k = coth(N asinh s_k) (k odd), tanh(N asinh s_k) (k even), s_k = sin(pi k/2N), c_k = cos(pi k/2N)
    [Theorem 4.2(b) of the article, proved in its Supplementary Section S3 -- a separately proved input],
    evaluated in Arb ball arithmetic.  On the overlap range the two enclosures are required to intersect.
Trusted base: Arb (python-flint); Python integers.   Output: certs/mfpt2d_remainder.json
usage: mfpt2d_remainder.py [N_B]
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import json, os, sys
from fractions import Fraction
from flint import arb, ctx, fmpq
HERE = os.path.dirname(os.path.abspath(__file__))
CERT = _os.path.join(_R, 'data', 'msc_rigorous_certified')
ctx.prec = 200
NB = int(sys.argv[1]) if len(sys.argv) > 1 else 1500


def frac(s):
    n, dn = s.split("/")
    return Fraction(int(n), int(dn))


def ball(lo, hi):
    mid, rad = (lo + hi) / 2, (hi - lo) / 2
    return arb(fmpq(mid.numerator, mid.denominator)) + arb(0, 1) * arb(fmpq(rad.numerator, rad.denominator))


def bounds(x):
    out = []
    for e in (x.lower(), x.upper()):
        m, ex = e.man_exp()
        out.append(Fraction(int(m)) * Fraction(2) ** int(ex))
    return out


def dec(fr, digits, up):
    sc = 10 ** digits
    n = fr.numerator * sc
    v = -((-n) // fr.denominator) if up else n // fr.denominator
    sgn = "-" if v < 0 else ""
    s = str(abs(v)).rjust(digits + 1, "0")
    return sgn + s[:-digits] + "." + s[-digits:]


def tau_single_sum(N):
    tot = arb(0)
    for k in range(1, N):
        s, c = arb.sin_cos_pi_fmpq(fmpq(k, 2 * N))
        x = N * s.asinh()
        g = x.coth() if k % 2 else x.tanh()
        tot += c * c * (1 + s * s).sqrt() / s * g
    return 4 * N * tot


g14 = arb(fmpq(1, 4)).gamma()
varpi = g14 ** 2 / (2 * (2 * arb.pi()).sqrt())
C2 = 8 / arb.pi() * (arb.const_euler() + (4 * arb(2).sqrt() / varpi).log() - arb(fmpq(1, 2)) - arb.pi() / 4)
X = varpi ** 4 / arb.pi() ** 2
C0 = X / 9

rows = {}
for line in open(_os.path.join(_R, 'data', 'msc_rigorous_certified', 'mfpt_d2.jsonl')):
    r = json.loads(line)
    rows[r["N"]] = (frac(r["tau_lo"]), frac(r["tau_hi"]))
NA = 2                                             # self-contained part: the contiguous range 2..NA of the enclosure file
while NA + 1 in rows:
    NA += 1
extra_a = sorted(N for N in rows if N > NA)        # isolated larger N with a linear-system enclosure
res_a, res_b, overlap_ok = {}, {}, True
for N in range(2, NA + 1):
    tau = ball(*rows[N])
    rem = tau - 8 / arb.pi() * N * N * arb(N).log() - C2 * N * N
    res_a[N] = bounds(rem)
    if N <= 60 or N % 20 == 0:                     # cross-check the external single sum on the overlap
        overlap_ok &= bool(tau_single_sum(N).overlaps(tau))
res_x = {}
for N in extra_a:
    tau = ball(*rows[N])
    res_x[N] = bounds(tau - 8 / arb.pi() * N * N * arb(N).log() - C2 * N * N)
    overlap_ok &= bool(tau_single_sum(N).overlaps(tau))
for N in range(NA + 1, NB + 1):
    tau = tau_single_sum(N)
    rem = tau - 8 / arb.pi() * N * N * arb(N).log() - C2 * N * N
    res_b[N] = bounds(rem)

def summarise(res):
    if not res:
        return None
    lo = min(res.items(), key=lambda kv: kv[1][0])
    hi = max(res.items(), key=lambda kv: kv[1][1])
    mono = all(res[N][1] < res[N + 1][0] for N in res if N + 1 in res)
    return dict(N_min=min(res), N_max=max(res), min_remainder_lower=dec(lo[1][0], 12, False), at_N=lo[0],
                max_remainder_upper=dec(hi[1][1], 12, True), at_N_max=hi[0], strictly_increasing_in_N=mono)

out = dict(C2=C2.str(40), C0_reference=C0.str(30),
           self_contained=summarise(res_a), via_single_sum=summarise(res_b),
           self_contained_isolated={str(N): [dec(v[0], 12, False), dec(v[1], 12, True)] for N, v in res_x.items()},
           single_sum_agrees_with_linear_system_on_overlap=overlap_ok,
           sample={str(N): [dec(v[0], 15, False), dec(v[1], 15, True)] for N, v in {**res_a, **res_b}.items()
                   if N in (2, 3, 5, 10, 20, 35, 50, 100, 160, 200, 300, 500, 1000, 1500, 2000, 3000)})
json.dump(out, open(_os.path.join(_R, 'data', 'msc_rigorous_certified', 'mfpt2d_remainder.json'), "w"), indent=1)
print(json.dumps(out, indent=1))
