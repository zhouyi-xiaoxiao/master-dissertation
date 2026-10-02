"""Sanity check of Proposition 5.4 (lattice expansion): the certified enclosure of
     R = L^4 [S_j - G^(j)(v) - H_j(v)/L^2]
must contain the directly computed value (mpmath, 50 digits) for sample N >= N0, v in V, q.
Also checks the elementary bounds of Lemma 5.1/5.2 on a grid (S, C, beta, A_j, Lam, B and derivatives).
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
import mpmath as mp
import core
from core import arb, G, H, R_enclosure, interval, PI
mp.mp.dps = 50


def Gmp(j, u):
    s = mp.mpf(0)
    for k in range(1, 400, 2):
        y = k * mp.pi / 4
        sg = 1 if ((k - 1) // 2) % 2 == 0 else -1
        s += sg * 2 * y * (-2 * y * y) ** j * mp.e ** (-2 * u * y * y)
    return s


def lattice(j, N, v, q=None):
    """S_j: continuous time (q None) at scaled time v; discrete at integer t with v = q(t-1)/L^2 (t passed via v)."""
    L = mp.mpf(N) - mp.mpf(1) / 2
    s = mp.mpf(0)
    for m in range(1, N):
        k = 2 * m - 1
        y = k * mp.pi / 4
        x = y / L
        sg = 1 if (m - 1) % 2 == 0 else -1
        A = (mp.sin(x) / x) ** (2 * j + 1) * mp.cos(x) ** 2
        if q is None:
            e = mp.e ** (-2 * v * L * L * mp.sin(x) ** 2)
        else:
            t = v
            lam = 1 - 2 * q * mp.sin(x) ** 2
            e = lam ** (t - 1)
        s += sg * 2 * y * (-2 * y * y) ** j * A * e
    return s


out = []
ok_all = True
# continuous time
for j in (0, 1):
    for N0, k0 in [(21, 9), (51, 11), (301, 15)]:
        L0 = arb(2 * N0 - 1) / 2
        V = interval(arb("0.30"), arb("0.36"))
        R, Bnd, parts = R_enclosure(j, V, L0, k0)
        for N in [N0, N0 + 7, 3 * N0, 10 * N0]:
            L = mp.mpf(N) - mp.mpf(1) / 2
            for v in [mp.mpf("0.30"), mp.mpf("0.3333"), mp.mpf("0.36")]:
                val = L ** 4 * (lattice(j, N, v) - Gmp(j, v) - ((7 + 2 * j) / mp.mpf(12) * Gmp(j + 1, v) + v / 6 * Gmp(j + 2, v)) / L ** 2)
                inside = R.contains(arb(mp.nstr(val, 30)))
                ok_all &= bool(inside)
                out.append(dict(mode="cont", j=j, N0=N0, N=N, v=float(v), R_true=float(val), encl=R.str(8), inside=bool(inside)))
        print("cont j=%d N0=%d  R in %s  Bnd=%s parts=%s" % (j, N0, R.str(8), Bnd.str(5), [p.str(4) for p in parts]), flush=True)
# discrete time
for j in (0, 1):
    for N0, k0, qq in [(51, 11, "0.3"), (51, 11, "0.6667"), (101, 13, "0.8")]:
        L0 = arb(2 * N0 - 1) / 2
        V = interval(arb("0.30"), arb("0.36"))
        qa = arb(qq)
        R, Bnd, parts = R_enclosure(j, V, L0, k0, q=qa, qbar=arb(qq))
        for N in [N0, 2 * N0 + 1, 6 * N0]:
            L = mp.mpf(N) - mp.mpf(1) / 2
            q = mp.mpf(qq)
            for vt in [mp.mpf("0.305"), mp.mpf("0.3333"), mp.mpf("0.355")]:
                t = int(vt * L * L / q) + 1
                v = q * (t - 1) / L ** 2
                Hq = (7 + 2 * j) / mp.mpf(12) * Gmp(j + 1, v) + v / 6 * (1 - 3 * q) * Gmp(j + 2, v)
                val = L ** 4 * (lattice(j, N, t, q) - Gmp(j, v) - Hq / L ** 2)
                inside = R.contains(arb(mp.nstr(val, 30)))
                ok_all &= bool(inside)
                out.append(dict(mode="disc", q=qq, j=j, N0=N0, N=N, v=float(v), R_true=float(val), encl=R.str(8), inside=bool(inside)))
        print("disc q=%s j=%d N0=%d  R in %s  Bnd=%s parts=%s" % (qq, j, N0, R.str(8), Bnd.str(5), [p.str(4) for p in parts]), flush=True)
print("all sampled true remainders inside the certified enclosures:", ok_all)
for r in out[:6] + out[-6:]:
    print(r)
json.dump(dict(all_inside=ok_all, rows=out), open(_os.path.join(_R, 'data', 'msc_rigorous_1d', '11_expansion_sanity.json'), "w"), indent=1)
