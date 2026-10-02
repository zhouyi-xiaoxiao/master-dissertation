"""Randomised test of Theorem 8.12 (general start, xi_N <= 1/2) against direct computation (NOT a proof).
 (A) discrete time, q <= 1/2: full float64 PMF, all maximisers must satisfy tau - E < t* < tau + 1 + E,
     tau = (c L^2 - kappa)/q + 1 - rho, E = 3.5/(q L^2), with c = c(xi_N), rho, kappa from the theta series (float).
 (B) continuous time: zero u_N of G_{N,x0}' (40-digit Newton): |u_N - c + kappa/L^2| < 3.5/L^4.
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import os, sys, json, random
import numpy as np
import mpmath as mp
sys.path.insert(0, os.path.dirname(__file__))
import start
from lib1d import pmf_stepper

random.seed(99)
mp.mp.dps = 40
here = os.path.dirname(os.path.abspath(__file__))


def consts(xi):
    c = start.c_float(xi)
    G2, G3 = start._Gf(2, c, xi), start._Gf(3, c, xi)
    rho = -c / 2 * G3 / G2
    return c, 0.5 - rho / 3, rho


viol = []; n = 0; ext = [1e9, -1e9]
cases = [(N, x0) for N in range(3, 26) for x0 in range(1, (2 * N + 1) // 4 + 1)]
cases += [(N, random.randint(1, (2 * N + 1) // 4)) for N in [random.randint(26, 260) for _ in range(150)]]
for N, x0 in cases:
    L = N - 0.5; xi = (x0 - 0.5) / L
    c, kap, rho = consts(xi)
    for q in {0.5, random.uniform(0.05, 0.5)}:
        T = int(4.0 * c * L * L / q) + 60
        f = pmf_stepper(N, q, x0, T)
        mx = f.max()
        modes = [int(t) for t in np.where(f >= mx * (1 - 1e-12))[0]]
        tau = (c * L * L - kap) / q + 1 - rho
        E = 3.5 / (q * L * L)
        n += 1
        for m in modes:
            ext = [min(ext[0], m - tau + E), max(ext[1], m - tau - 1 - E)]
        if not all(tau - E < m < tau + 1 + E for m in modes):
            viol.append((N, x0, q, modes, tau, E))
resA = {"cases": n, "violations": viol, "min of t*-(tau-E)": ext[0], "max of t*-(tau+1+E)": ext[1]}
print("(A)", resA, flush=True)


def GNx(u, N, x0, der):
    L = mp.mpf(N) - mp.mpf(1) / 2; xi = (x0 - mp.mpf(1) / 2) / L
    s = mp.mpf(0)
    for m in range(1, N):
        k = 2 * m - 1; y = k * mp.pi / 4; x = y / L
        S = mp.sin(x) / x
        mu = 2 * y * y * S * S
        e = mp.e ** (-mu * u)
        s += (-1) ** (m + 1) * 2 * y * S * mp.cos(x) * mp.cos(2 * y * xi) * (-mu) ** der * e
    return s


def Gx(j, u, xi):
    s = mp.mpf(0)
    for k in range(1, 600, 2):
        y = k * mp.pi / 4
        s += (1 if ((k - 1) // 2) % 2 == 0 else -1) * 2 * y * mp.cos(2 * y * xi) * (-2 * y * y) ** j * mp.e ** (-2 * u * y * y)
    return s


viol = []; rows = []
for N, x0 in [(3, 1), (4, 2), (6, 3), (8, 4), (12, 6), (22, 11), (60, 30), (101, 50), (150, 20), (240, 120)] + \
        [(N, random.randint(1, (2 * N + 1) // 4)) for N in [random.randint(5, 200) for _ in range(12)]]:
    L = mp.mpf(N) - mp.mpf(1) / 2; xi = (x0 - mp.mpf(1) / 2) / L
    c = mp.findroot(lambda u: Gx(1, u, xi), mp.mpf(start.c_float(float(xi))))
    rho = -c / 2 * Gx(3, c, xi) / Gx(2, c, xi); kap = mp.mpf(1) / 2 - rho / 3
    u = c - kap / L ** 2
    for _ in range(40):
        du = GNx(u, N, x0, 1) / GNx(u, N, x0, 2)
        u -= du
        if abs(du) < mp.mpf(10) ** -30:
            break
    dev = (u - c + kap / L ** 2) * L ** 4
    rows.append(dict(N=N, x0=x0, xi=float(xi), L4_dev=float(dev)))
    if not abs(dev) < 3.5:
        viol.append((N, x0, float(dev)))
resB = dict(cases=len(rows), violations=viol, max_abs_L4_dev=max(abs(r["L4_dev"]) for r in rows), rows=rows)
print("(B)", {k: v for k, v in resB.items() if k != "rows"})
ok = not (resA["violations"] or resB["violations"])
print("ALL OK:", ok)
json.dump(dict(A=resA, B=resB, ALL_OK=ok), open(_os.path.join(_R, 'data', 'msc_rigorous_1d', '29_start_stress.json'), "w"), indent=1)
