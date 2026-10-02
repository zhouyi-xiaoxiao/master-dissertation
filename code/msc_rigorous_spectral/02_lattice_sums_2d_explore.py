"""Exploration / sanity check of the exact single-sum formulae and constants in d = 2.

tau_2(N) = mean hitting time of the corner from a uniform start (unit rate)
m_2(N)   = mean hitting time corner -> opposite corner (unit rate)
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import os, sys, json
import numpy as np
import mpmath as mp
sys.path.insert(0, os.path.dirname(__file__))
from t2lib import corner_spectral_data

mp.mp.dps = 40

def brute(N, d=2):
    lam, w, sg = corner_spectral_data(N, d)
    nz = lam > 1e-14
    tau = np.sum(w[nz] / lam[nz])
    m = np.sum((1 - sg[nz]) * w[nz] / lam[nz])
    K2 = np.sum(w[nz] / lam[nz] ** 2)
    return tau, m, K2

def single_sum(N):
    N = mp.mpf(N)
    tau = mp.mpf(0); alt = mp.mpf(0)
    for k in range(1, int(N)):
        x = mp.pi * k / (2 * N)
        s = mp.sin(x)
        Phi = mp.cos(x) ** 2 * mp.sqrt(1 + s * s) / s
        y = 2 * N * mp.asinh(s)
        tau += Phi * mp.coth(y)
        alt += (-1) ** k * Phi / mp.sinh(y)
    tau2 = 4 * N * tau - mp.mpf(2) / 3 * (N * N - 1)
    m2 = tau2 + mp.mpf(2) / 3 * (N * N - 1) - 4 * N * alt
    return tau2, m2

Lam = mp.nsum(lambda k: 1 / (k * (mp.exp(2 * mp.pi * k) - 1)), [1, mp.inf])
Lam_closed = -mp.pi / 12 - mp.log(mp.gamma(mp.mpf(1) / 4)) + mp.log(2) + mp.mpf(3) / 4 * mp.log(mp.pi)
Sig = mp.nsum(lambda k: (-1) ** (k + 1) / (k * mp.sinh(mp.pi * k)), [1, mp.inf])
Sig_closed = mp.log(2) / 2 - mp.pi / 12
IPsi = mp.quad(lambda x: mp.cos(x) ** 2 * mp.sqrt(1 + mp.sin(x) ** 2) / mp.sin(x) - 1 / x, [0, mp.pi / 4, mp.pi / 2])
IPsi_closed = -mp.mpf(1) / 2 + mp.mpf(3) / 2 * mp.log(2) - mp.log(mp.pi)
c_tau = 8 / mp.pi * (mp.euler + IPsi_closed + 2 * Lam_closed) - mp.mpf(2) / 3
c_tau_closed = 8 / mp.pi * (mp.euler - mp.mpf(1) / 2 + mp.mpf(7) / 2 * mp.log(2) + mp.log(mp.pi) / 2 - 2 * mp.log(mp.gamma(mp.mpf(1) / 4))) - 2
c_m_minus_tau = 4 * mp.log(2) / mp.pi
print("Lambda      ", Lam, Lam - Lam_closed)
print("Sigma_alt   ", Sig, Sig - Sig_closed)
print("I_Psi       ", IPsi, IPsi - IPsi_closed)
print("c_tau       ", c_tau, c_tau - c_tau_closed)
print("c_m - c_tau ", c_m_minus_tau, " c_m =", c_tau + c_m_minus_tau)

rows = []
for N in [2, 3, 4, 5, 8, 10, 16, 20, 35, 50, 100, 200, 400]:
    t_b, m_b, K2 = brute(N)
    t_s, m_s = single_sum(N)
    R_tau = t_s - (8 / mp.pi) * N ** 2 * mp.log(N) - c_tau * N ** 2
    R_m = (m_s - t_s) - c_m_minus_tau * N ** 2
    lam1 = 0.5 * (1 - np.cos(np.pi / N))
    rows.append(dict(N=N, tau=float(t_s), m=float(m_s), err_tau=float(abs(t_b - t_s) / t_s), err_m=float(abs(m_b - m_s) / m_s),
                     R_tau=float(R_tau), R_m=float(R_m), lam1sqK2=float(lam1 ** 2 * K2), Xbar=float(lam1 * t_s)))
    print(f"N={N:4d} tau={float(t_s):.6f} m={float(m_s):.6f} relerr(brute) {rows[-1]['err_tau']:.1e} {rows[-1]['err_m']:.1e} "
          f"R_tau={float(R_tau):+.6f} R_tau/N={float(R_tau)/N:+.5f} R_(m-tau)={float(R_m):+.6f} L1^2K2={rows[-1]['lam1sqK2']:.5f} Xbar={rows[-1]['Xbar']:.4f}")
for N in [1000, 4000, 16000]:
    t_s, m_s = single_sum(N)
    R_tau = t_s - (8 / mp.pi) * N ** 2 * mp.log(N) - c_tau * N ** 2
    R_m = (m_s - t_s) - c_m_minus_tau * N ** 2
    print(f"N={N:6d} R_tau={float(R_tau):+.6f} R_(m-tau)={float(R_m):+.6f}  c2_eff={(float(m_s)-(8/np.pi)*N*N*np.log(N))/N**2:.10f}")
json.dump({"Lambda": str(Lam), "Sigma_alt": str(Sig), "I_Psi": str(IPsi), "c_tau": str(c_tau), "c_m": str(c_tau + c_m_minus_tau), "rows": rows},
          open(_os.path.join(_R, 'data', 'msc_rigorous_spectral', '02_lattice_sums_2d_explore.json'), "w"), indent=1)
