"""Randomised stress test of the main theorems of note R1 against direct computation (NOT a proof; written
separately from the certificate scripts, within the same project, to try to falsify the statements).

 (A) Theorem 6.3 / Cor. 6.4 (discrete time, q <= 4/5): random (N, q), 3 <= N <= 140, q in [0.03, 0.8]:
     full float64 PMF by forward stepping, ALL maximisers (relative tolerance 1e-12) must satisfy
         tau - E < t* < tau + 1 + E,   E = 1.78/(q L^2)  (1.68 for N >= 41),
     and (23), (24), (25) of Corollary 6.4.
 (B) the same window for large N (150 <= N <= 4000) and q in (0, 0.8] via the spectral formula in 40-digit arithmetic
     on the integers of [tau - 4, tau + 5]: the largest value must be attained inside the window, with
     D(floor(tau - E)) > 0 > D(ceil(tau + E)).
 (C) Theorem 5.7 / Cor. 5.8 (continuous time): root of G_N' by 40-digit Newton for random N <= 6000:
         |u_N - c + kappa/L^2| < 1.78/L^4  (1.54 for N >= 201),  u_N < c - kappa/L^2,  L^4 * dev -> kappa_2 = -1.5309.
 (D) Theorem 7.4 (q = 1): N in {601, 777, 1000, 1500, 2200}: parity and window with K = 10.3; observed excess.
 (E) Theorem 7.7 (4/5 < q < 1, N >= 601, condition (33)): q in {0.85, 0.93, 0.97, 0.995}, N in {601, 640, 900}.
 (F) Theorem 6.3(b),(c) (4/5 < q <= 17/20, N >= 7;  4/5 < q <= 9/10, N >= 10): window with K = 1.78, Corollary 6.4, and
     unimodality (float; Theorem 3.16).
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import os, sys, json, math, random
import numpy as np
import mpmath as mp
sys.path.insert(0, os.path.dirname(__file__))
from lib1d import pmf_stepper

mp.mp.dps = 40
random.seed(20261001)
c = mp.mpf("0.33328426549494789874852691654344242109")
kappa = mp.mpf("0.0862737793782390171817914719344242321")
rho = mp.mpf("1.9911786618652829484546255841967273035")
b = mp.mpf("0.3325085829964278601342050160350392651259")
s2 = mp.mpf("-2.250306288631155286632979215193602192428")
cf, kf, rf = float(c), float(kappa), float(rho)
here = os.path.dirname(os.path.abspath(__file__))
res = {}


def cor64(N, q, m, K):
    L = N - 0.5
    x = q * m / L ** 2 - cf
    ok22 = -(kf + (rf - 1) * q) / L ** 2 - K / L ** 4 < x < -(kf - (2 - rf) * q) / L ** 2 + K / L ** 4
    ok23 = abs(x) < (0.0863 + 0.9912 * q) / L ** 2 + 1.78 / L ** 4 and abs(x) <= 1.44 * (0.372 + 0.992 * q) / N ** 2
    ET = N * (N - 1) / q
    ok24 = abs(m / ET - cf) <= (0.003 + 0.992 * q) / L ** 2 + 2.2 / L ** 4
    neg = (x < 0) or N < 6
    return ok22 and ok23 and ok24 and neg


# ---------------- (A) ----------------
viol = []
nA = 0
two = 0
extreme = [1e9, -1e9]
cases = [(N, q) for N in range(3, 41) for q in (0.8, 0.79, 2 / 3, 0.5, 0.31)]
cases += [(random.randint(3, 140), random.choice([random.uniform(0.03, 0.8), random.uniform(0.6, 0.8), 0.8])) for _ in range(260)]
for N, q in cases:
    L = N - 0.5
    T = int(0.75 * L * L / q) + 80
    f = pmf_stepper(N, q, 1, T)
    mx = f.max()
    modes = [int(t) for t in np.where(f >= mx * (1 - 1e-12))[0]]
    K = 1.78 if N <= 40 else 1.68
    tau = (cf * L * L - kf) / q + 1 - rf
    E = K / (q * L * L)
    ok = all(tau - E < m < tau + 1 + E for m in modes) and all(cor64(N, q, m, K) for m in modes)
    nA += 1
    two += len(modes) > 1
    for m in modes:
        extreme = [min(extreme[0], m - tau + E), max(extreme[1], m - tau - 1 - E)]
    if not ok:
        viol.append((N, q, modes, tau, E))
# (A2) fine scan in q for the smallest N (where E = 1.78/(q L^2) is not small): 400 values of q per N
nA2 = 0
for N in range(3, 13):
    L = N - 0.5
    for i in range(1, 401):
        q = 0.8 * i / 400
        T = int(1.2 * L * L / q) + 40
        f = pmf_stepper(N, q, 1, T)
        mx = f.max()
        modes = [int(t) for t in np.where(f >= mx * (1 - 1e-12))[0]]
        tau = (cf * L * L - kf) / q + 1 - rf
        E = 1.78 / (q * L * L)
        nA2 += 1
        if not all(tau - E < m < tau + 1 + E for m in modes):
            viol.append((N, q, modes, tau, E))
res["A"] = {"cases": nA, "fine_q_scan_cases_N_3_to_12": nA2, "violations": viol, "cases_with_several_near_maximisers": two,
            "min of t*-(tau-E) (must be > 0)": extreme[0], "max of t*-(tau+1+E) (must be < 0)": extreme[1]}
print("(A)", res["A"], flush=True)


# ---------------- (B) ----------------
def spec_vals(N, q, ts):
    """f(t) * (2N-1)/(2q) for integer t in ts (mp), x0 = 1."""
    W = 2 * N - 1
    th = [(2 * m - 1) * mp.pi / W for m in range(1, N)]
    co = [mp.sin(t_) * mp.sin((N - 1) * t_) for t_ in th]
    lam = [1 - q * (1 - mp.cos(t_)) for t_ in th]
    t0 = min(ts)
    base = [l ** (t0 - 1) for l in lam]
    out = {}
    for t in range(t0, max(ts) + 1):
        out[t] = mp.fsum(c_ * p for c_, p in zip(co, base))
        base = [p * l for p, l in zip(base, lam)]
    return out


viol = []
nB = 0
for _ in range(60):
    N = random.choice([random.randint(150, 600), random.randint(601, 4000)])
    q = mp.mpf(random.choice([random.uniform(0.002, 0.8), random.uniform(0.66, 0.8), 0.8, 0.5]))
    L = mp.mpf(N) - mp.mpf(1) / 2
    K = mp.mpf("1.68") if N <= 600 else mp.mpf("1.56")
    tau = (c * L * L - kappa) / q + 1 - rho
    E = K / (q * L * L)
    tm, tp = int(mp.floor(tau - E)), int(mp.ceil(tau + E))
    ts = list(range(tm - 3, tp + 5))
    v = spec_vals(N, q, ts)
    best = max(v.values())
    arg = [t for t in ts if v[t] == best]
    ok = all(tau - E < t < tau + 1 + E for t in arg) and v[tm + 1] > v[tm] and v[tp + 1] < v[tp] \
        and all(v[t + 1] > v[t] for t in range(tm - 3, tm + 1)) and all(v[t + 1] < v[t] for t in range(tp, tp + 4))
    nB += 1
    if not ok:
        viol.append((N, float(q), arg, float(tau), float(E)))
res["B"] = dict(cases=nB, violations=viol)
print("(B)", res["B"], flush=True)


# ---------------- (C) ----------------
def GNp(u, N, der=1):
    L = mp.mpf(N) - mp.mpf(1) / 2
    s = mp.mpf(0)
    for m in range(1, N):
        th = (2 * m - 1) * mp.pi / (2 * N - 1)
        mu = L * L * (1 - mp.cos(th))
        e = mp.e ** (-mu * u)
        if e < mp.mpf(10) ** -60 and m > 5:
            break
        s += (-1) ** (m + 1) * L * mp.sin(th) * mp.cos(th / 2) * (-mu) ** der * e
    return s


viol = []
rowsC = []
for N in [3, 4, 5, 6, 7, 10, 17, 40, 99, 200, 201, 350, 1000, 2500, 6000] + [random.randint(8, 3000) for _ in range(10)]:
    L = mp.mpf(N) - mp.mpf(1) / 2
    u = c - kappa / L ** 2
    for _ in range(60):
        du = GNp(u, N, 1) / GNp(u, N, 2)
        u -= du
        if abs(du) < mp.mpf(10) ** -32:
            break
    dev = (u - c + kappa / L ** 2) * L ** 4
    K = mp.mpf("1.78") if N < 201 else mp.mpf("1.54")
    ET_ratio = u / (1 - 1 / (4 * L * L))
    ok = abs(dev) < K and u < c - kappa / L ** 2 and u > c - mp.mpf("0.3614") / L ** 2 \
        and abs(ET_ratio - c - (c / 4 - kappa) / L ** 2) <= mp.mpf("1.86") / L ** 4
    rowsC.append(dict(N=N, L4_dev=float(dev), ok=bool(ok)))
    if not ok:
        viol.append((N, float(dev)))
res["C"] = dict(cases=len(rowsC), violations=viol, rows=rowsC)
print("(C)", dict(cases=len(rowsC), violations=viol, L4_dev_at_N6000=[r["L4_dev"] for r in rowsC if r["N"] == 6000]), flush=True)

# ---------------- (D) ----------------
viol = []
rowsD = []
for N in [601, 777, 1000, 1500, 2200]:
    L = N - 0.5
    T = int(0.36 * L * L) + 50
    f = pmf_stepper(N, 1.0, 1, T)
    mx = f.max()
    modes = [int(t) for t in np.where(f >= mx * (1 - 1e-12))[0]]
    tau1 = float(c * L * L - b * L + 1 + s2)
    ok = all((m - (N - 1)) % 2 == 0 and tau1 - 10.3 / L < m < tau1 + 2 + 10.3 / L for m in modes)
    ok = ok and all(abs(m / L ** 2 - cf + float(b) / L) < (1.2504 + 10.3 / L) / L ** 2 for m in modes)
    rowsD.append(dict(N=N, modes=modes, minus_tau1=[m - tau1 for m in modes], ok=ok))
    if not ok:
        viol.append((N, modes, tau1))
res["D"] = dict(cases=len(rowsD), violations=viol, rows=rowsD)
print("(D)", res["D"], flush=True)


# ---------------- (E) ----------------
def chi(q):
    return 0.5 if q <= 0.8 else min(0.5, -math.log(2 * q - 1) / q)


viol = []
rowsE = []
for q in [0.85, 0.93, 0.97, 0.995]:
    for N in [601, 640, 900]:
        L = N - 0.5
        hyp = math.log(4 * (math.pi / 2) ** 3) + 8 * math.log(L) - 0.15 * chi(q) * L * L <= math.log(1e-12)
        T = int(0.5 * L * L / q) + 60
        f = pmf_stepper(N, q, 1, T)
        mx = f.max()
        modes = [int(t) for t in np.where(f >= mx * (1 - 1e-12))[0]]
        tau = (cf * L * L - kf) / q + 1 - rf
        E = 1.56 / (q * L * L)
        ta = int(math.ceil(0.15 * L * L / q)) + 1
        tm, tp = int(math.floor(tau - E)), int(math.ceil(tau + E))
        inc = bool(np.all(np.diff(f[ta:tm + 2]) > 0)); dec = bool(np.all(np.diff(f[tp:T]) < 0))
        left = bool(f[1:ta].max() < mx)
        ok = all(tau - E < m < tau + 1 + E for m in modes) and inc and dec and left
        rowsE.append(dict(q=q, N=N, hypothesis_30=hyp, modes=modes, minus_tau=[m - tau for m in modes], ok=ok))
        if hyp and not ok:
            viol.append((q, N, modes, tau))
res["E"] = dict(cases=len(rowsE), violations=viol, rows=rowsE)
print("(E)", dict(cases=len(rowsE), violations=viol, all_hyp=all(r["hypothesis_30"] for r in rowsE)), flush=True)
# ---------------- (F) Theorem 6.3(b),(c): 4/5 < q <= 9/10 ----------------
viol = []; nF = 0; ext = [1e9, -1e9]; nonuni = []
casesF = [(N, q) for N in (7, 8, 9) for q in (0.801, 0.82, 0.84, 0.85)]
casesF += [(N, q) for N in range(10, 61) for q in (0.81, 0.86, 0.9)]
casesF += [(random.randint(10, 400), random.uniform(0.8, 0.9)) for _ in range(120)]
for N, q in casesF:
    L = N - 0.5
    T = int(0.75 * L * L / q) + 80
    f = pmf_stepper(N, q, 1, T)
    mx = f.max()
    modes = [int(t) for t in np.where(f >= mx * (1 - 1e-12))[0]]
    tau = (cf * L * L - kf) / q + 1 - rf
    E = 1.78 / (q * L * L)
    nF += 1
    for m in modes:
        ext = [min(ext[0], m - tau + E), max(ext[1], m - tau - 1 - E)]
    d = np.diff(f[N - 1:])
    sg = np.sign(np.where(np.abs(d) < 1e-13 * mx, 0, d)); sg = sg[sg != 0]
    if int(np.sum(sg[1:] != sg[:-1])) > 1:
        nonuni.append((N, q))
    if not (all(tau - E < m < tau + 1 + E for m in modes) and all(cor64(N, q, m, 1.78) for m in modes)):
        viol.append((N, q, modes, tau, E))
res["F"] = {"cases": nF, "violations": viol, "not_unimodal(float)": nonuni,
            "min of t*-(tau-E) (must be > 0)": ext[0], "max of t*-(tau+1+E) (must be < 0)": ext[1]}
print("(F)", res["F"], flush=True)
res["ALL_OK"] = not any(res[k]["violations"] for k in "ABCDEF") and not nonuni
print("ALL OK:", res["ALL_OK"])
json.dump(res, open(_os.path.join(_R, 'data', 'msc_rigorous_1d', '23_stress.json'), "w"), indent=1)
