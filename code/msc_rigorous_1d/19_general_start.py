"""Section 8 (general start): the limit density
      G_xi(u) = sum_{k odd} sigma_k 2 y_k cos(2 y_k xi) exp(-2 u y_k^2),   xi = (x0 - 1/2)/L in [0, 1),
its image form (Lemma 8.1, numerical sanity check), certified enclosures of its unique maximiser c(xi)
(Theorem 8.3: G_xi is unimodal with a unique maximiser; a certified sign change + -> - of G_xi' brackets it),
and a numerical comparison with lattice modes (continuous time and discrete time q = 1/2), N = 400.
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import os, sys, json, math
sys.path.insert(0, os.path.dirname(__file__))
import mpmath as mp
import numpy as np
import core
from core import arb, sigma, pos_tail
from flint import ctx
from lib1d import pmf_stepper

core.set_prec(600)
mp.mp.dps = 60
PI = arb.pi()


def Gxi(j, u, xi, K=None):
    u = arb(u); xi = arb(xi)
    ulo = arb(u.lower())
    if K is None:
        K = int(4 / math.pi * math.sqrt(ctx.prec * 0.6931 / (2 * float(ulo)))) + 9
        K = max(K, int(2 * 0.9 / (float(ulo) * math.pi ** 2)) + 11)      # makes the tail ratio of Lemma 4.1 < 1/2
        K += (K + 1) % 2
    s = arb(0)
    for k in range(1, K, 2):
        y = k * PI / 4
        s += sigma(k) * 2 * y * (2 * y * xi).cos() * (-2 * y * y) ** j * (-2 * u * y * y).exp()
    tail = 2 ** (j + 1) * pos_tail(2 * j + 1, ulo, K)
    return s + arb(0, tail.upper())


def Gxi_mp(j, u, xi):
    K = int(4 / math.pi * math.sqrt(mp.mp.prec * 0.6931 / (2 * float(u)))) + 9
    s = mp.mpf(0)
    for k in range(1, K + 1, 2):
        y = k * mp.pi / 4
        s += sigma(k) * 2 * y * mp.cos(2 * y * xi) * (-2 * y * y) ** j * mp.e ** (-2 * u * y * y)
    return s


def Gimg(u, xi, nmax=40):
    h = lambda a: a / mp.sqrt(2 * mp.pi * u ** 3) * mp.e ** (-a * a / (2 * u))
    return sum(h(4 * n + 1 - xi) + h(4 * n + 1 + xi) for n in range(-nmax, nmax + 1))


def dGimg(u, xi):
    return mp.diff(lambda w: Gimg(w, xi, 12), u)


def find_mode(xi):
    """numerical maximiser of G_xi (bracketing on a geometric grid, then bisection-type solver); certified afterwards."""
    lo = (1 - xi) ** 2 / 3 * mp.mpf("0.5")
    us = [lo * mp.mpf("1.08") ** i for i in range(120)]
    prev = dGimg(us[0], xi)
    for a, b_ in zip(us[:-1], us[1:]):
        cur = dGimg(b_, xi)
        if prev > 0 and cur <= 0:
            r0 = mp.findroot(lambda u: dGimg(u, xi), (a, b_), solver="anderson")
            return mp.findroot(lambda u: Gxi_mp(1, u, xi), r0)
        prev = cur
    raise RuntimeError("no sign change found")


out = {"rows": []}
# (1) image identity
err = 0
for xi in [mp.mpf(0), mp.mpf("0.3"), mp.mpf("0.7"), mp.mpf("0.95")]:
    for u in [mp.mpf("0.01"), mp.mpf("0.1"), mp.mpf("0.4"), mp.mpf(2)]:
        err = max(err, abs(Gxi_mp(0, u, xi) - Gimg(u, xi)))
out["image_identity_max_abs_err"] = float(err)
print("image identity max abs err:", mp.nstr(err, 3))

# (2) certified c(xi)
xis = ["0", "1/10", "1/5", "3/10", "2/5", "1/2", "3/5", "7/10", "4/5", "9/10", "19/20", "99/100"]
for xs in xis:
    num, den = (xs.split("/") + ["1"])[:2]
    xi = arb(int(num)) / int(den)
    xim = mp.mpf(int(num)) / int(den)
    guess = (1 - xim) ** 2 / 3
    r = find_mode(xim)
    rr = arb(mp.nstr(r, 50))
    d = arb("1e-40")
    ok = bool(Gxi(1, rr - d, xi) > 0) and bool(Gxi(1, rr + d, xi) < 0)
    row = dict(xi=xs, c_xi=mp.nstr(r, 30), certified=ok, levy=mp.nstr(guess, 12), ratio_to_levy=mp.nstr(r / guess, 12),
               G_at_c=Gxi(0, rr, xi).str(15))
    out["rows"].append(row)
    print(row, flush=True)

# (3) numerical comparison with lattice modes (not a proof): N = 400, x0 = xi L + 1/2 rounded
N = 400; L = N - 0.5
cmp = []
for xs, row in zip(xis, out["rows"]):
    num, den = (xs.split("/") + ["1"])[:2]
    xi = int(num) / int(den)
    x0 = max(1, min(N - 1, int(round(xi * L + 0.5))))
    xi_eff = (x0 - 0.5) / L
    cxi = float(find_mode(mp.mpf(xi_eff)))
    q = 0.5
    tmax = int(1.2 * L * L / q * max(cxi, 0.002)) + 400
    f = pmf_stepper(N, q, x0, tmax)
    tstar = int(np.argmax(f))
    cmp.append(dict(xi=xs, x0=x0, xi_eff=xi_eff, c_xi_eff=cxi, discrete_q_half_mode=tstar, q_tstar_over_L2=q * tstar / L ** 2,
                    diff=q * tstar / L ** 2 - cxi))
    print(cmp[-1], flush=True)
out["lattice_comparison_N400_q_half"] = cmp
json.dump(out, open(_os.path.join(_R, 'data', 'msc_rigorous_1d', '19_general_start.json'), "w"), indent=1)
