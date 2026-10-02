#!/usr/bin/env python
"""Check by the second implementation, part 3: continuum constants (mpmath, 30 digits) and lattice Green constants.

 (a) 1D reflecting-start / absorbing-end:  mode/MFPT = 2 tau*, tau* = argmax of
     sum_{m>=1} (-1)^{m+1} (2m-1) exp(-(2m-1)^2 pi^2 tau / 4)                (tau = D t / L^2, MFPT = L^2/2D)
 (b) EXIT (centre start, absorbing box boundary) in d = 1,2,3: S_d = S_1^d,
     S_1(tau) = sum_m 4 (-1)^{m+1} / ((2m-1) pi) exp(-(2m-1)^2 pi^2 tau)       (tau = D_axis t / width^2)
 (c) 3D mean constant C3 from the capacity of a 2x2x2 block of sites in Z^3 (image argument):
     C3 = G(000) + 3 G(100) + 3 G(110) + G(111),  G = lattice Green function of the unit-rate walk.
 (d) other shape constants for the 1D law (CDF at mode, median/MFPT, CV, g_max*MFPT, S(MFPT)).
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import json, os
import mpmath as mp

mp.mp.dps = 30
OUT = {}
M = 60


def dens1(tau):      # reflecting start -> absorbing end, up to a constant factor
    return mp.fsum((-1) ** (m + 1) * (2 * m - 1) * mp.e ** (-(2 * m - 1) ** 2 * mp.pi ** 2 * tau / 4) for m in range(1, M))


def ddens1(tau):
    return mp.fsum(-(-1) ** (m + 1) * (2 * m - 1) ** 3 * mp.pi ** 2 / 4 * mp.e ** (-(2 * m - 1) ** 2 * mp.pi ** 2 * tau / 4) for m in range(1, M))


tau_star = mp.findroot(ddens1, mp.mpf(1) / 6)
OUT["1d_tau_star"] = mp.nstr(tau_star, 15)
OUT["1d_mode_over_mfpt"] = mp.nstr(2 * tau_star, 15)
OUT["1d_minus_one_third"] = mp.nstr(2 * tau_star - mp.mpf(1) / 3, 6)


def S1r(tau):        # survival, reflecting start -> absorbing end; tau = D t / L^2
    return mp.fsum(4 * (-1) ** (m + 1) / ((2 * m - 1) * mp.pi) * mp.e ** (-(2 * m - 1) ** 2 * mp.pi ** 2 * tau / 4) for m in range(1, M))


def g1r(tau):
    return mp.pi * dens1(tau)

mfpt_tau = mp.mpf(1) / 2
OUT["1d_cdf_at_mode"] = mp.nstr(1 - S1r(tau_star), 10)
OUT["1d_S_at_mfpt"] = mp.nstr(S1r(mfpt_tau), 10)
OUT["1d_gmax_times_mfpt"] = mp.nstr(g1r(tau_star) * mfpt_tau, 10)
med = mp.findroot(lambda t: S1r(t) - mp.mpf(1) / 2, 0.38)
OUT["1d_median_over_mfpt"] = mp.nstr(med / mfpt_tau, 10)
# second moment: E[T^2] = 2 int t S dt
m2 = 2 * mp.fsum(4 * (-1) ** (m + 1) / ((2 * m - 1) * mp.pi) * (4 / ((2 * m - 1) ** 2 * mp.pi ** 2)) ** 2 for m in range(1, 4000))
OUT["1d_cv"] = mp.nstr(mp.sqrt(m2 - mfpt_tau ** 2) / mfpt_tau, 10)
OUT["1d_cv_expected_sqrt(2/3)"] = mp.nstr(mp.sqrt(mp.mpf(2) / 3), 10)


# ---- EXIT -----------------------------------------------------------------------------------
def S1(tau):
    if tau < mp.mpf("0.02"):     # short-time (image) form to avoid slow convergence: S = 1 - 2 sum (-1)^n erfc((2n+1)/(4 sqrt(tau)))
        return 1 - 2 * mp.fsum((-1) ** n * mp.erfc((2 * n + 1) / (4 * mp.sqrt(tau))) for n in range(0, 12))
    return mp.fsum(4 * (-1) ** (m + 1) / ((2 * m - 1) * mp.pi) * mp.e ** (-(2 * m - 1) ** 2 * mp.pi ** 2 * tau) for m in range(1, M))


def S1p(tau, order):   # order-th derivative of the long-time series (fine for tau >= 0.01)
    return mp.fsum(4 * (-1) ** (m + 1) / ((2 * m - 1) * mp.pi) * (-(2 * m - 1) ** 2 * mp.pi ** 2) ** order
                   * mp.e ** (-(2 * m - 1) ** 2 * mp.pi ** 2 * tau) for m in range(1, M))


for d in (1, 2, 3):
    Sd = lambda t: S1(t) ** d
    mean = mp.quad(Sd, [0, 0.01, 0.05, 0.2, 1, 5, 30])
    # d^2/dt^2 S1^d = d(d-1) S1^(d-2) S1'^2 + d S1^(d-1) S1''  (density maximum <=> this vanishes)
    d2 = lambda t: d * (d - 1) * S1p(t, 0) ** (d - 2) * S1p(t, 1) ** 2 + d * S1p(t, 0) ** (d - 1) * S1p(t, 2)
    guess = {1: 0.04, 2: 0.03, 3: 0.025}[d]
    tm = mp.findroot(d2, (guess * 0.5, guess * 2.5), solver="illinois", tol=1e-22, maxsteps=200)
    OUT[f"exit_d{d}_mode_over_mfpt"] = mp.nstr(tm / mean, 10)
    OUT[f"exit_d{d}_mean_tau"] = mp.nstr(mean, 12)
    OUT[f"exit_d{d}_mode_tau"] = mp.nstr(tm, 12)
OUT["exit_d1_mean_tau_expected_1/8"] = "0.125"


# ---- lattice Green function of the unit-rate walk on Z^3 -----------------------------------
def G3(x, y, z):
    f = lambda t: mp.e ** (-t) * mp.besseli(x, t / 3) * mp.besseli(y, t / 3) * mp.besseli(z, t / 3)
    return mp.quad(f, mp.linspace(0, 200, 41) + [400, 800, 1600, 3200, 6400, 12800, 25600, 51200, 102400, mp.inf])

mp.mp.dps = 20
try:
    g000 = G3(0, 0, 0); g100 = G3(1, 0, 0); g110 = G3(1, 1, 0); g111 = G3(1, 1, 1)
    OUT["G3_000"] = mp.nstr(g000, 10); OUT["G3_100"] = mp.nstr(g100, 10)
    OUT["G3_110"] = mp.nstr(g110, 10); OUT["G3_111"] = mp.nstr(g111, 10)
    OUT["G3_000_Watson_expected"] = "1.5163860592"
    OUT["C3_from_capacity_of_2x2x2_block"] = mp.nstr(g000 + 3 * g100 + 3 * g110 + g111, 10)
except Exception as exc:      # the slowly decaying integrand makes this delicate; a k-space check is in v04
    OUT["G3_error"] = repr(exc)

here = os.path.dirname(os.path.abspath(__file__))
with open(_os.path.join(_R, 'data', 'msc_modes_verify', 'constants.json'), "w") as fh:
    json.dump(OUT, fh, indent=1)
print(json.dumps(OUT, indent=1))
