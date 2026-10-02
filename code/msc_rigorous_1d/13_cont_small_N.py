"""Theorem 5.7, computer-assisted part (uses core.py): for 3 <= N <= NMAX certify (ball arithmetic)

     G_N'(c - kappa/L^2 - K/L^4) > 0 > G_N'(c - kappa/L^2 + K/L^4),     K = 1.78,

which, G_N being strictly unimodal (Theorem 3.3), gives |u_N - c + kappa/L^2| < K/L^4.
Also certifies  -0.3614/L^2 < u_N - c < -kappa/L^2  for these N, and tabulates verified enclosures of u_N.
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
import mpmath as mp
import core as rig
from core import arb, G, GN, interval

rig.set_prec(160)
const = json.load(open(_os.path.join(_R, 'data', 'msc_rigorous_1d', '03_constants.json')))
c = arb(const["c_mid_80_digits"][:60], "1e-55")
kappa = arb(3) / 4 + c / 6 * G(3, c) / G(2, c)
K = arb("1.78")
NMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 300
mp.mp.dps = 40


def GNp_mp(u, N):
    L = mp.mpf(N) - mp.mpf(1) / 2
    s = mp.mpf(0)
    for m in range(1, N):
        th = (2 * m - 1) * mp.pi / (2 * N - 1)
        mu = 1 - mp.cos(th)
        s += (-1) ** (m + 1) * L * mp.sin(th) * mp.cos(th / 2) * (-L * L * mu) * mp.e ** (-L * L * mu * u)
    return s


rows = []
all_ok = True
worst = 0.0
for N in range(3, NMAX + 1):
    L = arb(2 * N - 1) / 2
    um = c - kappa / L ** 2 - K / L ** 4
    up = c - kappa / L ** 2 + K / L ** 4
    ok1 = GN(1, um, N) > 0
    ok2 = GN(1, up, N) < 0
    # one-sided: u_N < c - kappa/L^2, and u_N > c - 0.3614/L^2
    ok3 = GN(1, c - kappa / L ** 2, N) < 0
    ok4 = GN(1, c - arb("0.3614") / L ** 2, N) > 0
    row = dict(N=N, two_sided=bool(ok1 and ok2), upper=bool(ok3), lower=bool(ok4))
    if N <= 60 or N % 20 == 0:
        # verified enclosure of u_N: float root, then certify sign change on a tiny bracket
        r = mp.findroot(lambda u: GNp_mp(u, N), mp.mpf(str(float(c.mid()))) - mp.mpf(str(float(kappa.mid()))) / (N - 0.5) ** 2)
        rr = arb(mp.nstr(r, 35))
        d = arb("1e-25")
        okb = (GN(1, rr - d, N) > 0) and (GN(1, rr + d, N) < 0)
        coef = (interval(rr - d, rr + d) - c + kappa / L ** 2) * L ** 4
        row.update(u_N=mp.nstr(r, 25), bracket_certified=bool(okb), L4_coef=coef.str(12))
        worst = max(worst, float(coef.abs_upper()))
    rows.append(row)
    all_ok &= bool(ok1 and ok2 and ok3 and ok4)
    if N <= 12 or N % 50 == 0:
        print(row, flush=True)
print("ALL CERTIFIED for 3 <= N <=", NMAX, ":", all_ok, " max |L^4 coef| among tabulated:", worst)
json.dump(dict(NMAX=NMAX, K=1.78, all_certified=all_ok, rows=rows),
          open(_os.path.join(_R, 'data', 'msc_rigorous_1d', '13_cont_small_N.json'), "w"), indent=1)
