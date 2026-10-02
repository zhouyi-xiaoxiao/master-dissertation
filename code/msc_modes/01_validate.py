"""
01_validate.py -- cross-validation of the three independent routes in fptlib.
Writes data/validation.json.  Deterministic.
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import json, os, sys, time, math
import numpy as np
import scipy.sparse as sp
from scipy.special import gammaln
sys.path.insert(0, os.path.dirname(__file__))
import fptlib as F

OUT = _os.path.join(_R, 'data', 'msc_modes', 'validation.json')
res = {}


def full_pmf(N, d, q, start, target, tol=1e-13):
    """iterate until S(t) < tol"""
    st = F.Stepper(N, d, q, start, [target])
    f = [0.0]
    while True:
        f.append(st.step())
        if (st.t & 1023) == 0 and st.survival() < tol:
            break
    return np.array(f), st.survival()


# 1. time stepping vs explicit sparse-Q powers ------------------------------------
chk = []
for (N, d) in [(6, 2), (5, 3), (9, 1)]:
    q = 0.8
    s, a = F.corner(N, d), F.far_corner(N, d)
    Q, tidx = F.build_Q(N, d, q, [a])
    assert abs(Q - Q.T).max() < 1e-15, "Q must be symmetric"
    e = np.zeros(Q.shape[0]); e[tidx[np.ravel_multi_index(s, (N,) * d)]] = 1.0
    one = np.ones(Q.shape[0])
    v = e.copy(); Sprev = 1.0; fQ = [0.0]
    for t in range(1, 400):
        v = Q @ v
        Snow = v @ one
        fQ.append(Sprev - Snow); Sprev = Snow
    f, _, _ = F.run_pmf(N, d, q, s, [a], t_max=399)
    err = float(np.abs(np.array(fQ) - f).max())
    chk.append({"N": N, "d": d, "max_abs_diff_stepper_vs_Qpowers": err})
    print("stepper vs Q powers", N, d, err)
res["stepper_vs_Qpowers"] = chk

# 2. normalisation and mean: sum f = 1, sum t f = MFPT(linear solve) --------------
chk = []
for (N, d, q) in [(9, 1, 0.8), (12, 2, 0.8), (12, 2, 1.0), (15, 2, 0.5), (7, 3, 0.8), (8, 3, 0.9)]:
    s, a = F.corner(N, d), F.far_corner(N, d)
    f, Send = full_pmf(N, d, q, s, a)
    t = np.arange(f.size)
    m_pmf = float((t * f).sum())
    m_ls = F.mfpt_linear_solve(N, d, q, s, [a])
    m_lap = F.LaplaceFP(N, d, q, s, a).mfpt()
    row = {"N": N, "d": d, "q": q, "sum_f": float(f.sum()), "S_end": Send,
           "mfpt_from_pmf": m_pmf, "mfpt_linear_solve": m_ls, "mfpt_laplace_formula": float(m_lap),
           "rel_pmf_vs_ls": abs(m_pmf - m_ls) / m_ls, "rel_lap_vs_ls": abs(m_lap - m_ls) / m_ls,
           "argmax": int(np.argmax(f))}
    chk.append(row); print(row)
res["normalisation_and_mean"] = chk

# 3. 1D closed form vs time stepping ---------------------------------------------
chk = []
for (N, q) in [(5, 0.8), (20, 0.8), (50, 0.8), (50, 1.0), (50, 0.3), (200, 0.8)]:
    ch = F.Chain1D(N, q)
    f, _, _ = F.run_pmf(N, 1, q, (0,), [(N - 1,)], t_max=int(3 * N * N / q))
    tt = np.arange(1, f.size)
    fc = ch.f_discrete(tt)
    err = float(np.abs(fc - f[1:]).max())
    row = {"N": N, "q": q, "max_abs_diff": err, "argmax_stepper": int(np.argmax(f)),
           "argmax_closed_form": int(tt[np.argmax(fc)]), "mfpt_formula": ch.mfpt(),
           "mfpt_linear_solve": F.mfpt_linear_solve(N, 1, q, (0,), [(N - 1,)])}
    chk.append(row); print(row)
res["chain1d_closed_form_vs_stepper"] = chk

# 4. MFPT: Laplace formula vs linear solve vs single sum, form (a) -------------
def single_sum_form_a(N, q):
    """single sum, form (a), overflow-free evaluation"""
    k = np.arange(1, N)
    ph = np.arccosh(2 - np.cos(np.pi * k / N))
    # B_k / sinh(N ph)
    e2N = np.exp(-2 * N * ph)
    coth = (1 + e2N) / (1 - e2N)
    r1 = np.exp(-ph) * (1 + np.exp(-2 * (N - 1) * ph)) / (1 - e2N)
    r2 = (1 + np.cosh(ph)) * 2 * np.exp(-N * ph) / (1 - e2N)
    Bs = coth + r1 - np.where(k % 2 == 0, 1.0, -1.0) * r2
    return 2 * N * (N - 1) / q + 4 * N / q * (np.cos(np.pi * k / (2 * N)) ** 2 * Bs / np.sinh(ph)).sum()

chk = []
for N in [2, 3, 5, 12, 35, 60, 120, 200]:
    for q in [0.8, 1.0, 0.3]:
        s, a = F.corner(N, 2), F.far_corner(N, 2)
        m_ls = F.mfpt_linear_solve(N, 2, q, s, [a])
        m_lap = float(F.LaplaceFP(N, 2, q, s, a).mfpt())
        m_ss = float(single_sum_form_a(N, q))
        chk.append({"N": N, "q": q, "linear_solve": m_ls, "laplace": m_lap, "single_sum_form_a": m_ss,
                    "rel_lap": abs(m_lap - m_ls) / m_ls, "rel_ss": abs(m_ss - m_ls) / m_ls})
        print(chk[-1])
res["mfpt_2d_three_way"] = chk
chk = []
for (N, geo) in [(11, "CC"), (11, "C2M"), (11, "M2C"), (21, "CC"), (21, "C2M"), (21, "M2C")]:
    for d in (2, 3):
        if d == 3 and N > 15:
            continue
        q = 0.8
        s, a = F.GEOMETRIES[geo][0](N, d), F.GEOMETRIES[geo][1](N, d)
        m_ls = F.mfpt_linear_solve(N, d, q, s, [a])
        m_lap = float(F.LaplaceFP(N, d, q, s, a).mfpt())
        chk.append({"N": N, "d": d, "geo": geo, "linear_solve": m_ls, "laplace": m_lap,
                    "rel": abs(m_lap - m_ls) / m_ls})
        print(chk[-1])
for N in [5, 9, 15, 25]:
    q = 0.8
    s, a = F.corner(N, 3), F.far_corner(N, 3)
    m_ls = F.mfpt_linear_solve(N, 3, q, s, [a])
    m_lap = float(F.LaplaceFP(N, 3, q, s, a).mfpt())
    chk.append({"N": N, "d": 3, "geo": "CC", "linear_solve": m_ls, "laplace": m_lap, "rel": abs(m_lap - m_ls) / m_ls})
    print(chk[-1])
res["mfpt_geometries_laplace_vs_linear_solve"] = chk

# 5. continuous-time density: Talbot vs Erlang mixture of the q=1 discrete PMF ----
def erlang_mixture(f1, t):
    """g(t) = sum_n f1(n) t^(n-1) e^{-t}/(n-1)!   (unit jump rate)"""
    n = np.arange(1, f1.size)
    logk = (n - 1) * np.log(t) - t - gammaln(n)
    return float((f1[1:] * np.exp(logk)).sum())

chk = []
for (N, d, geo) in [(21, 2, "CC"), (21, 2, "C2M"), (21, 2, "M2C"), (9, 3, "CC"), (30, 1, "CC")]:
    s, a = F.GEOMETRIES[geo][0](N, d), F.GEOMETRIES[geo][1](N, d)
    f1, Send = full_pmf(N, d, 1.0, s, a, tol=1e-14)
    L = F.LaplaceFP(N, d, 1.0, s, a)
    tm = float(np.argmax(f1))
    for fac in [0.2, 0.5, 1.0, 2.0, 5.0, 20.0]:
        t = fac * tm
        ge = erlang_mixture(f1, t)
        for M in (16, 20, 24, 28, 32):
            gt = float(F.talbot_invert(L.Fhat, [t], M=M)[0])
            chk.append({"N": N, "d": d, "geo": geo, "t": t, "M": M, "erlang": ge, "talbot": gt,
                        "rel": abs(gt - ge) / ge})
        print(N, d, geo, "t=%.1f" % t, "erlang=%.6e" % ge,
              " ".join("M%d:%.1e" % (c["M"], c["rel"]) for c in chk[-5:]))
res["talbot_vs_erlang_mixture"] = chk

# 5b. 1D: Talbot vs closed-form continuous density
chk = []
for N in [20, 200, 2000]:
    ch = F.Chain1D(N, 1.0)
    L = F.LaplaceFP(N, 1, 1.0, (0,), (N - 1,))
    for tau in [0.05, 0.1, 1 / 6, 0.3, 1.0, 3.0]:
        t = tau * N * N * 2
        ge = float(ch.g_cont([t])[0])
        gt = float(F.talbot_invert(L.Fhat, [t], M=24)[0])
        chk.append({"N": N, "t": t, "closed": ge, "talbot": gt, "rel": abs(gt - ge) / abs(ge)})
        print(chk[-1])
res["talbot_vs_closed_form_1d"] = chk

# 6. discrete generating function identity F~(z) = F^((1-z)/z) -------------------
chk = []
for (N, d, q) in [(12, 2, 0.8), (7, 3, 0.8), (9, 1, 0.5)]:
    s, a = F.corner(N, d), F.far_corner(N, d)
    f, _ = full_pmf(N, d, q, s, a, tol=1e-15)
    L = F.LaplaceFP(N, d, q, s, a)
    for z in [0.5, 0.9, 0.99, 0.6 + 0.3j, -0.4 + 0.2j]:
        lhs = (f * z ** np.arange(f.size)).sum()
        rhs = L.Fhat(np.array([(1 - z) / z]))[0]
        chk.append({"N": N, "d": d, "q": q, "z": str(z), "abs_diff": float(abs(lhs - rhs))})
        print(chk[-1])
res["z_transform_identity"] = chk

# 7. correct Abate-Whitt (1992) lattice inversion reproduces f(t) ------------------
def abate_whitt(L, t, gamma=11):
    """f(t) ~ (1/(2 t r^t)) sum_{k=1}^{2t} (-1)^k Re F~(r e^{i pi k/t}),  r = 10^(-gamma/(2t));
    F~(z) = F^((1-z)/z).  Discretisation error <= ~10^-gamma."""
    r = 10 ** (-gamma / (2.0 * t))
    k = np.arange(1, 2 * t + 1)
    z = r * np.exp(1j * np.pi * k / t)
    Fz = L.Fhat((1 - z) / z)
    return float((np.where(k % 2 == 0, 1.0, -1.0) * Fz.real).sum() / (2 * t * r ** t))

chk = []
N, d, q = 35, 2, 0.8
s, a = F.corner(N, d), F.far_corner(N, d)
f, _, _ = F.run_pmf(N, d, q, s, [a], t_max=6000)
L = F.LaplaceFP(N, d, q, s, a)
for t in [50, 400, 1200, 2493, 2494, 4000, 6000]:
    aw = abate_whitt(L, t)
    chk.append({"N": N, "q": q, "t": t, "stepper": float(f[t]), "abate_whitt": aw, "rel": abs(aw - f[t]) / f[t]})
    print(chk[-1])
# a fixed-radius truncated trapezoidal rule, r=0.9, K=54 (Supplementary Section S2)
def fixed_radius_rule(L, t, r=0.9, K=54):
    k = np.arange(1, K + 1)
    z = r * np.exp(1j * np.pi * k / t)
    Fr = L.Fhat(np.array([(1 - r) / r]))[0].real
    Fz = L.Fhat((1 - z) / z)
    return float(r ** (-t) / t * (0.5 * Fr + (np.where(k % 2 == 0, 1.0, -1.0) * Fz.real).sum()))
for t in [50, 400, 2493]:
    chk.append({"N": N, "q": q, "t": t, "stepper": float(f[t]), "fixed_r_K54": fixed_radius_rule(L, t)})
    print(chk[-1])
res["abate_whitt_check"] = chk

with open(OUT, "w") as fh:
    json.dump(res, fh, indent=1)
print("wrote", OUT)
