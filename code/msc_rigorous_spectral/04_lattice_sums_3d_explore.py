"""Exploration / sanity check of the exact reduced formulae and constants in d = 3 (corner target).

tau_3 = 3*I1 + 3N*T2star[H] + 3N*D_N,   I1 = (N-1)(2N-1)/3
  H(th1,th2) = (1+c1)(1+c2)(coth(phi/2)-1),   cosh(phi) = 3 - c1 - c2
  D_N        = sum' (1+c1)(1+c2) coth(phi/2) (coth(N phi)-1)
m_3 - tau_3 = (N^2-1) - 3N sum' (-1)^{k1+k2} (1+c1)(1+c2) coth(phi/2)/sinh(N phi)
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import os, sys, json
import numpy as np
import mpmath as mp
sys.path.insert(0, os.path.dirname(__file__))
from t2lib import corner_spectral_data
mp.mp.dps = 30


def brute3(N):
    lam, w, sg = corner_spectral_data(N, 3)
    nz = lam > 1e-14
    return np.sum(w[nz] / lam[nz]), np.sum((1 - sg[nz]) * w[nz] / lam[nz]), np.sum(w[nz] / lam[nz] ** 2)


def reduced3(N):
    k = np.arange(N)
    c = np.cos(np.pi * k / N)
    wt = np.where(k == 0, 0.5, 1.0)
    C1, C2 = np.meshgrid(c, c, indexing="ij")
    Wt = np.outer(wt, wt)
    sgn = np.outer((-1.0) ** k, (-1.0) ** k)
    S2 = (1 - C1) / 2 + (1 - C2) / 2            # sinh^2(phi/2) = s1^2 + s2^2
    S2[0, 0] = 1.0
    S = np.sqrt(S2)
    cothh = np.sqrt(1 + S2) / S
    phi = 2 * np.arcsinh(S)
    P = (1 + C1) * (1 + C2) * Wt
    P[0, 0] = 0.0
    with np.errstate(over="ignore"):
        e2 = np.exp(-2 * N * phi)
        cothN_m1 = 2 * e2 / (1 - e2)
        inv_sinh = 2 * np.exp(-N * phi) / (1 - e2)
    T2H = np.sum(P * (cothh - 1))
    DN = np.sum(P * cothh * cothN_m1)
    I1 = (N - 1) * (2 * N - 1) / 3
    tau = 3 * I1 + 3 * N * T2H + 3 * N * DN
    alt = np.sum(P * sgn * cothh * inv_sinh)
    m_minus_tau = (N * N - 1) - 3 * N * alt
    return tau, m_minus_tau, T2H, DN


# Watson-type constant: C3 = int_0^inf e^{-t} (I0(t/3)+I1(t/3))^3 dt
f = lambda s: (mp.besseli(0, s, ) + mp.besseli(1, s)) * mp.exp(-s)
C3 = 3 * (mp.quad(lambda s: f(s) ** 3, mp.linspace(0, 60, 31)) + mp.quad(lambda s: f(s) ** 3, [60, 200, 1000, 10000, mp.inf]))
u = lambda y: 3 * mp.quad(lambda s: mp.exp(-3 * s) * mp.besseli(y[0], s) * mp.besseli(y[1], s) * mp.besseli(y[2], s), mp.linspace(0, 60, 31)) \
    + 3 * mp.quad(lambda s: mp.exp(-3 * s) * mp.besseli(y[0], s) * mp.besseli(y[1], s) * mp.besseli(y[2], s), [60, 200, 1000, 10000, mp.inf])
print("C3 (Bessel integral) =", C3)
u0 = mp.sqrt(6) / (32 * mp.pi ** 3) * mp.gamma(mp.mpf(1) / 24) * mp.gamma(mp.mpf(5) / 24) * mp.gamma(mp.mpf(7) / 24) * mp.gamma(mp.mpf(11) / 24)
print("Watson u(0) closed form =", u0)

E_inf_1 = 4 / mp.pi * mp.nsum(lambda a, b: 0 if (a == 0 and b == 0) else 1 / (mp.sqrt(a * a + b * b) * (mp.exp(2 * mp.pi * mp.sqrt(a * a + b * b)) - 1)),
                              [-6, 6], [-6, 6])
E_inf_2 = 16 / mp.pi * sum(mp.quad(lambda v: 1 / (mp.sqrt(k * k + v * v) * (mp.exp(2 * mp.pi * mp.sqrt(k * k + v * v)) - 1)), [0, 1, 3, 8]) for k in range(1, 6))
c_tau3 = 3 / mp.pi * (4 * mp.euler - 4 * mp.log(mp.pi) + mp.pi - 8 * mp.log(2)) - 1 + 3 * (E_inf_1 + E_inf_2)
print("E_inf parts:", E_inf_1, E_inf_2, " predicted c_tau3 =", c_tau3)
alt_const = mp.nsum(lambda a, b: 0 if (a == 0 and b == 0) else (-1) ** (a + b) / (mp.sqrt(a * a + b * b) * mp.sinh(mp.pi * mp.sqrt(a * a + b * b))), [-12, 12], [-12, 12])
c_mt3 = 1 - 6 / mp.pi * alt_const
print("predicted (m-tau)/N^2 ->", c_mt3, "  predicted c_m3 =", c_tau3 + c_mt3)

rows = []
for N in [2, 3, 4, 5, 6, 8, 10, 14, 20, 30, 40]:
    tb, mb, K2 = brute3(N)
    tr, mmt, T2H, DN = reduced3(N)
    lam1 = (1 - np.cos(np.pi / N)) / 3
    rows.append(dict(N=N, tau=tr, m=tr + mmt, err_tau=abs(tb - tr) / tb, err_m=abs(mb - tr - mmt) / mb, lam1sqK2=lam1 ** 2 * K2))
    print(f"N={N:3d} tau={tr:.6f} m={tr+mmt:.6f} relerr {rows[-1]['err_tau']:.1e} {rows[-1]['err_m']:.1e} (tau-C3N^3)/N^2={(tr-float(C3)*N**3)/N**2:+.5f} "
          f"(m-tau)/N^2={mmt/N**2:.5f} L1^2K2={rows[-1]['lam1sqK2']:.4f} X={lam1*tr:.3f} D_N/N={DN/N:.5f}")
for N in [80, 160, 320, 640, 1280, 2560]:
    tr, mmt, T2H, DN = reduced3(N)
    rows.append(dict(N=N, tau=tr, m=tr + mmt))
    print(f"N={N:5d} (tau-C3N^3)/N^2={(tr-float(C3)*N**3)/N**2:+.6f} (m-tau)/N^2={mmt/N**2:.6f} (m-C3N^3)/N^2={(tr+mmt-float(C3)*N**3)/N**2:+.6f} D_N/N={DN/N:.6f}")
json.dump({"C3": str(C3), "c_tau3_pred": str(c_tau3), "c_mt3_pred": str(c_mt3), "rows": rows},
          open(_os.path.join(_R, 'data', 'msc_rigorous_spectral', '04_lattice_sums_3d_explore.json'), "w"), indent=1, default=float)
