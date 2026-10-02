#!/usr/bin/env python
"""Check by the second implementation, part 7: remaining spot checks and exact constants.

 (a) 1D closed-form PMF vs stepper; 1D modes at very large N (discrete q = 0.8 and continuous).
 (b) convergence of pole-expansion results in the number of poles; far-field split vs full sums.
 (c) peak refinement of the stepped PMF vs the real-t maximiser of the pole formula.
 (d) location of the extra local maximum at q = 1 in 2D.
 (e) exact constants of the mean (image / torus argument; mpmath):
        2D: c2 = (8/pi)[gamma + 4 ln 2 + (1/2) ln pi - 2 ln Gamma(1/4)] - 2 - 4/pi
        3D: C3 = G(000)+3G(100)+3G(110)+G(111),  C3' = -(6/pi) xi + (6/pi^2) b,
            xi = 2.8372974795 (Hasimoto constant), b = -sum'(-1)^{i+j+k}/(i^2+j^2+k^2) = 2.5193561521
Output: ../data/extra.json
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import json, os, subprocess, sys
import numpy as np
import mpmath as mp
from scipy.optimize import brentq
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vspec

HERE = os.path.dirname(os.path.abspath(__file__))
V = _os.path.join(_R, 'data', 'msc_modes_verify')
OUT = {}
STEP = _os.path.join(_R, 'code', 'msc_modes_verify', 'stepper')
TMP = _os.path.join(_R, 'data', 'msc_modes_verify', 'pmf', '_x.bin')


def stepper(d, N, q, geom, stop, tmin):
    subprocess.run([STEP, str(d), str(N), repr(q), geom, str(stop), str(tmin), "2000000000", TMP], check=True, capture_output=True)
    f = np.fromfile(TMP); os.remove(TMP)
    return f


# ---- (a) 1D ---------------------------------------------------------------------------------
def f1d(N, q, t):
    m = np.arange(1, N)
    th = (2 * m - 1) * np.pi / (2 * N - 1)
    lam = 1 - q + q * np.cos(th)
    w = (2 * q / (2 * N - 1)) * (-1.0) ** (m + 1) * np.cos(th / 2) * np.sin(th)
    return np.array([np.sum(w * lam ** (tt - 1)) for tt in np.atleast_1d(t)])


f = stepper(1, 50, 0.8, "CC", 3.0, 200)
t = np.arange(1, len(f) + 1)
OUT["1d_closed_form_vs_stepper_maxabs"] = float(np.max(abs(f1d(50, 0.8, t) - f)))


def mode1d(N, q=None, M=60):
    """continuous-time (unit rate) if q is None, else discrete integer argmax at activity q."""
    m = np.arange(1, min(N, M + 1))
    th = (2 * m - 1) * np.pi / (2 * N - 1)
    mu = 2 * np.sin(th / 2) ** 2                      # 1 - cos(theta)
    w = (-1.0) ** (m + 1) * np.cos(th / 2) * np.sin(th)
    L2 = (N - 0.5) ** 2
    if q is None:
        gp = lambda tt: -np.sum(w * mu * np.exp(-mu * tt))
        return brentq(gp, 0.2 * L2, 0.5 * L2, xtol=1e-15 * L2, rtol=1e-15)
    keep = q * mu < 0.9                 # modes with larger rates are < 1e-300 at the modal time
    w, mu = w[keep], mu[keep]
    loglam = np.log1p(-q * mu)
    fp = lambda tt: np.sum(w * loglam * np.exp(loglam * (tt - 1)))
    tr = brentq(fp, 0.2 * L2 / q, 0.5 * L2 / q, xtol=1e-15 * L2, rtol=1e-15)
    # integer decision by the sign of the exact increment f(t+1)-f(t) = -q sum w mu lam^(t-1)
    inc = lambda tt: -np.sum(w * mu * np.exp(loglam * (tt - 1)))
    t0 = int(np.floor(tr))
    cand = t0 - 2
    while inc(cand) > 0:
        cand += 1
    return cand, tr


rows = []
for N in (100, 1000, 3620, 10240, 100000, 1000000):
    tc = mode1d(N)
    td, tr = mode1d(N, 0.8)
    rows.append(dict(N=N, cont_mode=tc, cont_over_L2=tc / (N - 0.5) ** 2, cont_ratio=tc / (N * (N - 1)),
                     disc_mode_q0_8=int(td), disc_ratio=td / (N * (N - 1) / 0.8)))
OUT["1d_large_N"] = rows

# ---- (b) convergence in the number of poles / far-field split ----------------------------------
P = [json.loads(l) for l in open(_os.path.join(_R, 'data', 'msc_modes_verify', 'poles.jsonl'))]
base = {(r["d"], r["N"]): r for r in P if "conv" not in r["key"] and "farcheck" not in r["key"]}
conv = []
for r in P:
    if "conv" in r["key"] or "farcheck" in r["key"]:
        b = base.get((r["d"], r["N"]))
        if b:
            conv.append(dict(key=r["key"], n_poles=r["n_poles"], n_poles_base=b["n_poles"],
                             rel_mode=r["mode"] / b["mode"] - 1, rel_mfpt=r["mfpt"] / b["mfpt"] - 1,
                             band_lo=[r["band99_lo"], b["band99_lo"]], band_hi=[r["band99_hi"], b["band99_hi"]],
                             cdf=[r["cdf_at_mode"], b["cdf_at_mode"]], median_rel=r["median"] / b["median"] - 1,
                             nu_rel_max=float(max(abs(np.array(r["nu"][:len(b["nu"])]) / np.array(b["nu"][:len(r["nu"])]) - 1)))))
OUT["pole_convergence_and_farfield"] = conv

# ---- (c) peak refinement ---------------------------------------------------------------------
ref = []
for d, N in ((2, 35), (2, 100), (2, 200), (3, 35), (3, 40), (3, 50)):
    f = stepper(d, N, 0.8, "CC", 1.3, 200)
    i = int(np.argmax(f))
    h = max(2, int(0.004 * i))
    tt = np.arange(i - h, i + h + 1)
    c = np.polyfit(tt - i, f[i - h:i + h + 1] / f[i], 3)
    # stationary point of the cubic near 0
    r = np.roots(np.polyder(c)); r = r[np.isreal(r)].real
    tstar = i + 1 + r[np.argmin(abs(r))]
    b = base[(d, N)]
    ref.append(dict(d=d, N=N, argmax=i + 1, cubic_fit_peak=float(tstar), pole_formula_real_peak=b["mode_disc_real_q0.8"],
                    diff_steps=float(tstar - b["mode_disc_real_q0.8"]), h=h))
OUT["peak_refinement"] = ref

# ---- (d) extra local maximum at q = 1 in 2D ----------------------------------------------------
f = stepper(2, 21, 1.0, "CC", 1.5, 200)
df = np.diff(f)
nz = np.nonzero(f > 0)[0][0]
locmax = [int(i + 1) for i in range(max(nz, 1), len(f) - 1) if f[i] > f[i - 1] and f[i] > f[i + 1]]
OUT["q1_2d_N21_local_maxima_t"] = locmax
OUT["q1_2d_N21_first_values"] = [[int(nz + 1 + k), float(f[nz + k])] for k in range(8)]
OUT["q1_2d_N21_argmax"] = int(np.argmax(f) + 1)

# ---- (e) exact constants -----------------------------------------------------------------------
mp.mp.dps = 25
c2 = 8 / mp.pi * (mp.euler + 4 * mp.log(2) + mp.log(mp.pi) / 2 - 2 * mp.loggamma(mp.mpf(1) / 4)) - 2 - 4 / mp.pi
OUT["c2_closed_form"] = mp.nstr(c2, 15)
OUT["X2D_constant_closed_form"] = mp.nstr(mp.pi ** 2 / 4 * c2, 12)
th3 = lambda x: mp.jtheta(3, 0, x)
def th4(x):
    """theta_4(0, x); for x close to 1 use the Jacobi imaginary transformation."""
    t_ = -mp.log(x)
    if t_ < 1:
        return mp.sqrt(mp.pi / t_) * mp.jtheta(2, 0, mp.e ** (-mp.pi ** 2 / t_))
    return mp.jtheta(4, 0, x)
# alternating sum b = -sum' (-1)^{i+j+k}/(i^2+j^2+k^2) = -int_0^inf [theta4(e^-t)^3 - 1] dt
b = -mp.quad(lambda t_: th4(mp.e ** (-t_)) ** 3 - 1, [0, 0.5, 1, 2, 5, 10, 30, 80])
# Hasimoto constant xi = -int_0^inf [theta3(e^{-pi u})^3 - 1 - u^{-3/2}] du
f_small = lambda u: u ** mp.mpf(-1.5) * (th3(mp.e ** (-mp.pi / u)) ** 3 - 1) - 1      # Jacobi-inverted form for u < 1
f_large = lambda u: th3(mp.e ** (-mp.pi * u)) ** 3 - 1
xi = -(mp.quad(f_small, [0, 0.25, 0.5, 1]) + mp.quad(f_large, [1, 2, 4, 10, 40]) - 2)
OUT["hasimoto_xi"] = mp.nstr(xi, 14)
OUT["alternating_sum_b"] = mp.nstr(b, 14)
C3p = -(6 / mp.pi) * xi + (6 / mp.pi ** 2) * b
OUT["C3prime_closed_form"] = mp.nstr(C3p, 12)
# 2D cross-check of the same route: g2(centre) = (1/4pi^2) int [theta4^2 - 1] dt  (expected -ln2/(4 pi))
g2c = mp.quad(lambda t_: th4(mp.e ** (-t_)) ** 2 - 1, [0, 0.5, 1, 2, 5, 10, 30, 80]) / (4 * mp.pi ** 2)
OUT["g2_centre"] = mp.nstr(g2c, 14); OUT["minus_ln2_over_4pi"] = mp.nstr(-mp.log(2) / (4 * mp.pi), 14)
misc = json.load(open(_os.path.join(_R, 'data', 'msc_modes_verify', 'misc.json')))
OUT["c2_numeric_N=2^24"] = misc["mfpt_constants"]["c2"][-1]["c2_seq"]
OUT["C3_numeric_richardson"] = misc["mfpt_constants"]["richardson2"][-1]["C3"]
OUT["C3prime_numeric_richardson"] = misc["mfpt_constants"]["richardson2"][-1]["C3prime"]
const = json.load(open(_os.path.join(_R, 'data', 'msc_modes_verify', 'constants.json')))
OUT["C3_closed_form"] = const["C3_from_capacity_of_2x2x2_block"]

json.dump(OUT, open(_os.path.join(_R, 'data', 'msc_modes_verify', 'extra.json'), "w"), indent=1)
print(json.dumps(OUT, indent=1))
