"""Theorem 9.8 (d = 2: limits of the remainders R_tau(N), R_mtau(N)) -- certified constants and sanity checks.

 (A) certified (Arb) values of S_5..S_8, c_tau^(0), c_mtau^(0), C_2^(0); the decimal statements of (9.9), (9.10), (9.11) are asserted;
     the classical closed forms of Remark 9.9 are compared with the certified series values (overlap of balls, > 50 digits).
 (B) certified values of R_tau(N), R_mtau(N) for a list of N (exact single sums of Proposition 8.3 in ball arithmetic)
     and checks: they lie in the windows of Theorem 9.1, approach the limits, are increasing along the list,
     and (R - limit) N^2 is bounded (observed rate; not claimed in the text).  The list reaches N = 10^6; in addition
     strict monotonicity is certified for every consecutive pair N, N+1 with 2 <= N < 300.
 (C) sanity checks of every step of the proof: N e_1(N) -> 1/(6 pi) + pi/72 (Lemma 9.6), termwise majorants and pointwise limits
     of a_k, b_k, a_k', b_k', the limits of the four sums, and the expansion arsinh(sin x) = x - x^3/3 + O(x^5).
Output: ../data/19_certified_d2_limits.json     (all comparisons between balls are certified)
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import os, json, math, sys
from flint import arb, ctx
ctx.prec = 300
pi = arb.pi(); ln2 = arb(2).log(); gam = arb.const_euler()
OUT = {"constants": {}, "R": [], "checks": {}}
NFAIL = 0
def enc(name, x, note=""):
    OUT["constants"][name] = {"str": x.str(40), "lower": x.lower().str(45, radius=False), "upper": x.upper().str(45, radius=False), "note": note}
    print(f"ENCLOSURE {name:14s} = {x.str(40)}   {note}", flush=True)
def check(name, ok, info=""):
    global NFAIL
    OUT["checks"][name] = bool(ok)
    if not ok: NFAIL += 1
    print(("PASS " if ok else "FAIL ") + name + ("   " + info if info else ""), flush=True)

# ------------------------------------------------------------------------------------------------ (A) constants
KMAX = 60
def tail_bound(K):
    """sum_{k>K} 8 k^2 e^{-pi k} <= first term / (1 - ratio), ratio = ((K+2)/(K+1))^2 e^{-pi} < 1/20."""
    ratio = (arb(K + 2) / (K + 1)) ** 2 * (-pi).exp()
    assert ratio < arb(1) / 20
    return 8 * arb(K + 1) ** 2 * (-pi * (K + 1)).exp() / (1 - ratio)
TB = tail_bound(KMAX)
assert TB < arb(10) ** -75
# each summand is bounded by 8 k^2 e^{-pi k} (checked here for k <= KMAX; for all k >= 1 see the text)
S5 = S6 = S7 = S8 = arb(0)
for k in range(1, KMAX + 1):
    a = pi * k
    t5 = 2 * arb(k) / ((2 * a).exp() - 1); t6 = arb(k) ** 2 / a.sinh() ** 2
    t7 = arb(k) / a.sinh(); t8 = arb(k) ** 2 * a.cosh() / a.sinh() ** 2
    bnd = 8 * arb(k) ** 2 * (-a).exp()
    assert t5 < bnd and t6 < bnd and t7 < bnd and t8 < bnd
    sg = 1 if k % 2 == 1 else -1
    S5 += t5; S6 += t6; S7 += sg * t7; S8 += sg * t8
pm = arb(0, 1) * TB
S5, S6, S7, S8 = S5 + pm, S6 + pm, S7 + pm, S8 + pm
enc("S_5", S5, "sum 2k/(e^{2 pi k}-1)"); enc("S_6", S6, "sum k^2/sinh^2(pi k)")
enc("S_7", S7, "sum (-1)^{k+1} k/sinh(pi k)"); enc("S_8", S8, "sum (-1)^{k+1} k^2 cosh(pi k)/sinh^2(pi k)")
c_tau0 = arb(2) / 3 + pi / 18 - 2 * pi / 3 * S5 + 2 * pi ** 2 / 3 * S6
c_mt0 = -arb(2) / 3 - 2 * pi / 3 * S7 + 2 * pi ** 2 / 3 * S8
C20 = c_tau0 + c_mt0
enc("c_tau^(0)", c_tau0, "lim R_tau(N)"); enc("c_mtau^(0)", c_mt0, "lim R_mtau(N)"); enc("C_2^(0)", C20, "constant term of m_2")
# statements of the theorem / of (9.x)
assert arb("0.8830352") < c_tau0 < arb("0.8830353")
assert arb("-0.3508944") < c_mt0 < arb("-0.3508943")
assert arb("0.5321408") < C20 < arb("0.5321409")
assert arb("0.0037558") < S5 < arb("0.0037559") and arb("0.0075537") < S6 < arb("0.0075538")
assert arb("0.0795774") < S7 < arb("0.0795775") and arb("0.0733219") < S8 < arb("0.0733220")
print("CERTIFIED: the decimal values printed in (9.9) and in Theorem 9.8 (truncated expansions)")
# closed forms of Remark 9.9 (classical; comparison only)
G14 = (arb(1) / 4).gamma()
E4i = 3 * G14 ** 8 / (2 * pi) ** 6
cf = {"S_5": arb(1) / 12 - 1 / (4 * pi), "S_7": 1 / (4 * pi), "S_6": (E4i - 9 / pi ** 2) / 72,
      "S_8": 1 / (8 * pi ** 2) + G14 ** 8 / (512 * pi ** 6),
      "c_tau^(0)": arb(3) / 4 + G14 ** 8 / (2304 * pi ** 4), "c_mtau^(0)": -arb(3) / 4 + G14 ** 8 / (768 * pi ** 4),
      "C_2^(0)": G14 ** 8 / (576 * pi ** 4)}
ser = {"S_5": S5, "S_6": S6, "S_7": S7, "S_8": S8, "c_tau^(0)": c_tau0, "c_mtau^(0)": c_mt0, "C_2^(0)": C20}
for name in cf:
    d = cf[name] - ser[name]
    ok = cf[name].overlaps(ser[name]) and abs(d) < arb(10) ** -50
    check(f"closed form of {name} agrees with the certified series to 1e-50", ok, f"closed = {cf[name].str(25)}")
assert arb("0.5321408832") < cf["C_2^(0)"] < arb("0.5321408833")

# ------------------------------------------------------------------------------------------------ (B) R_tau(N), R_mtau(N)
ctx.prec = 200
pi = arb.pi(); ln2 = arb(2).log(); gam = arb.const_euler()
G14 = (arb(1) / 4).gamma()
I_Psi = -arb(1) / 2 + arb(3) / 2 * ln2 - pi.log()
# c_tau, c_mtau from their DEFINING series (9.1)-(9.2) (not from the closed forms)
Lam = sum(1 / (arb(k) * ((2 * pi * k).exp() - 1)) for k in range(1, 60)) + arb(0, 1) * arb(10) ** -150
Sig = sum((-1) ** (k + 1) / (arb(k) * (pi * k).sinh()) for k in range(1, 120)) + arb(0, 1) * arb(10) ** -150
c_tau = 8 / pi * (gam + I_Psi + 2 * Lam) - arb(2) / 3
c_mt = arb(2) / 3 + 8 / pi * Sig
kap2 = 2 * arb(1).asinh()
def Phi(x):
    s = x.sin()
    return (1 - s * s) * (1 + s * s).sqrt() / s
def tau_m(N):
    """tau_2, m_2 - tau_2 by Proposition 8.3 (exact single sums, ball arithmetic)."""
    st = arb(0); sa = arb(0)
    h = pi / (2 * N)
    for k in range(1, N):
        x = h * k
        y = 2 * N * x.sin().asinh()
        ph = Phi(x)
        st += ph * y.coth()
        sa += (1 if k % 2 == 1 else -1) * ph / y.sinh()
    tau = 4 * N * st - arb(2) / 3 * (N * N - 1)
    mt = arb(2) / 3 * (N * N - 1) + 4 * N * sa
    return tau, mt
NLIST = [2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 30000, 100000, 300000, 1000000]
prev = None
mono = True; inside = True; lim_ok = True
for N in NLIST:
    tau, mt = tau_m(N)
    Rt = tau - 8 / pi * N * N * arb(N).log() - c_tau * N * N
    Rm = mt - c_mt * N * N
    dt = (Rt - c_tau0) * N * N; dm = (Rm - c_mt0) * N * N
    OUT["R"].append({"N": N, "R_tau": Rt.str(20), "R_mtau": Rm.str(20), "(R_tau-lim)N^2": dt.str(8), "(R_mtau-lim)N^2": dm.str(8)})
    print(f"N={N:6d}  R_tau = {Rt.str(14)}  R_mtau = {Rm.str(14)}  (R_tau-lim)N^2 = {dt.str(8)}  (R_mtau-lim)N^2 = {dm.str(8)}", flush=True)
    inside &= bool(Rt > arb("-2.14") and Rt < arb("3.53") and Rm > arb("-9.06") and Rm < arb("7.72"))
    inside &= bool(Rt > arb("0.85127") and Rt < arb("0.88304") and Rm > arb("-0.53017") and Rm < arb("-0.35089"))   # Observation 9.7
    lim_ok &= bool(Rt < c_tau0 and Rm < c_mt0 and abs(dt) < arb("0.13") and abs(dm) < arb("1.01"))
    if prev is not None:
        mono &= bool(prev[0] < Rt and prev[1] < Rm)
    prev = (Rt, Rm)
    if N == 2:
        check("R_tau(2) = 0.851279..., R_mtau(2) = -0.530169... (tau_2(2) = 5, m_2(2) = 8)",
              arb("0.851279") < Rt < arb("0.851280") and arb("-0.530170") < Rm < arb("-0.530169") and (tau - 5).contains(0) and (tau + mt - 8).contains(0))
check("R_tau, R_mtau inside the windows of Theorem 9.1 and of Observation 9.7 for all N in the list", inside)
check("R_tau, R_mtau increasing along the list (certified comparisons)", mono)
check("R below its limit, |R_tau - limit| N^2 < 0.13 and |R_mtau - limit| N^2 < 1.01 for all N in the list (certified; the sum of the two limits of (R - limit) N^2, -0.0670577 - 1.0025013 = -1.0695590, is the coefficient C_{-2} of the companion note)", lim_ok)
last = OUT["R"][-1]
check(f"at N = {NLIST[-1]}: |R_tau - c_tau^(0)| < 1e-12 and |R_mtau - c_mtau^(0)| < 2e-12", abs(prev[0] - c_tau0) < arb("1e-12") and abs(prev[1] - c_mt0) < arb("2e-12"))
# every N from 2 to 300: strict monotonicity of R_tau and R_mtau in N (certified comparisons of consecutive values), and the windows of Observation 9.7
prev = None; mono_all = True; inside_all = True
for N in range(2, 301):
    tau, mt = tau_m(N)
    Rt = tau - 8 / pi * N * N * arb(N).log() - c_tau * N * N
    Rm = mt - c_mt * N * N
    inside_all &= bool(Rt > arb("0.85127") and Rt < c_tau0 and Rm > arb("-0.53017") and Rm < c_mt0)
    if prev is not None:
        mono_all &= bool(prev[0] < Rt and prev[1] < Rm)
    prev = (Rt, Rm)
check("R_tau(N) < R_tau(N+1) and R_mtau(N) < R_mtau(N+1) for every 2 <= N < 300 (certified), and both below their limits", mono_all and inside_all)

# ------------------------------------------------------------------------------------------------ (C) steps of the proof
import mpmath as mp
mp.mp.dps = 40
def Psi(x):
    x = mp.mpf(x)
    if x == 0: return mp.mpf(0)
    s = mp.sin(x)
    c = 1 / s - 1 / x
    Q = -s * (1 + s ** 2 - s ** 4) / (1 + (1 - s ** 2) * mp.sqrt(1 + s ** 2))
    return c + Q
F = lambda y: mp.coth(y) - 1
kap2f = 2 * mp.asinh(1)
IPs = -mp.mpf(1) / 2 + mp.mpf(3) / 2 * mp.log(2) - mp.log(mp.pi)
# Psi'(0) = -1/3, Psi'(pi/2) = 4/pi^2
check("Psi'(0) = -1/3 and Psi'(pi/2) = 4/pi^2 (numerical differentiation)",
      abs(mp.diff(Psi, mp.mpf("1e-12"), h=mp.mpf("1e-14")) + mp.mpf(1) / 3) < mp.mpf("1e-10") and
      abs(mp.diff(Psi, mp.pi / 2, direction=-1, h=mp.mpf("1e-12")) - 4 / mp.pi ** 2) < mp.mpf("1e-9"))
lim_e1 = 1 / (6 * mp.pi) + mp.pi / 72
errs = []
for N in (10, 100, 1000, 10000):
    h = mp.pi / (2 * N)
    e1 = mp.fsum(Psi(h * k) for k in range(1, N)) - 2 * N / mp.pi * IPs - 1 / mp.pi
    errs.append(abs(N * e1 - lim_e1))
    print(f"  N={N:6d}  N e_1(N) = {mp.nstr(N*e1, 15)}   limit {mp.nstr(lim_e1, 15)}   |N e_1| <= pi^2*2.35/48 = {mp.nstr(mp.pi**2*2.35/48, 6)}")
    assert abs(N * e1) <= mp.pi ** 2 * mp.mpf("2.35") / 48
check("N e_1(N) -> 1/(6 pi) + pi/72 (errors decreasing, < 1e-8 at N = 10^4)", all(errs[i + 1] < errs[i] for i in range(3)) and errs[-1] < mp.mpf("1e-8"), f"errors {[mp.nstr(e, 3) for e in errs]}")
# expansion of arsinh(sin x)
check("(x - arsinh(sin x))/x^3 -> 1/3 with error O(x^2)",
      all(abs((x - mp.asinh(mp.sin(x))) / x ** 3 - mp.mpf(1) / 3) < x ** 2 for x in (mp.mpf("0.3"), mp.mpf("0.1"), mp.mpf("0.01"), mp.mpf("0.001"))))
S5f, S6f, S7f, S8f = (mp.mpf(x.mid().str(40, radius=False)) for x in (S5, S6, S7, S8))
maj_ok = True; ptw_ok = True
sums = {}
for N in (50, 400, 3200, 25600):
    h = mp.pi / (2 * N)
    sa = sb = sap = sbp = mp.mpf(0)
    for k in range(1, N):
        x = h * k
        y = 2 * N * mp.asinh(mp.sin(x))
        ps = Psi(x)
        a = N * ps * F(y); b = N / x * (F(y) - F(mp.pi * k))
        ap = N * ps / mp.sinh(y); bp = N / x * (1 / mp.sinh(y) - 1 / mp.sinh(mp.pi * k))
        if k <= 40:
            A = mp.mpf("1.09") * mp.pi * k / 2 * F(kap2f * k); B = mp.pi ** 2 * k ** 2 / (6 * mp.sinh(kap2f * k) ** 2)
            Ap = mp.mpf("1.09") * mp.pi * k / 2 / mp.sinh(kap2f * k); Bp = mp.pi ** 2 * k ** 2 * mp.cosh(kap2f * k) / (6 * mp.sinh(kap2f * k) ** 2)
            maj_ok &= bool(abs(a) <= A and 0 <= b <= B and abs(ap) <= Ap and 0 <= bp <= Bp and a <= 0)
            maj_ok &= bool(kap2f * k <= y <= mp.pi * k and mp.pi * k - y <= 2 * N * x ** 3 / 3)
        sg = 1 if k % 2 == 1 else -1
        sa += a; sb += b; sap += sg * ap; sbp += sg * bp
        if k > 60 and abs(a) + abs(b) + abs(ap) + abs(bp) < mp.mpf("1e-45"): break
    sums[N] = (sa, sb, sap, sbp)
    print(f"  N={N:6d}  sum a_k = {mp.nstr(sa, 12)} (-> {mp.nstr(-mp.pi/6*S5f, 12)})   sum b_k = {mp.nstr(sb, 12)} (-> {mp.nstr(mp.pi**2/6*S6f, 12)})"
          f"   alt a'_k = {mp.nstr(sap, 12)} (-> {mp.nstr(-mp.pi/6*S7f, 12)})   alt b'_k = {mp.nstr(sbp, 12)} (-> {mp.nstr(mp.pi**2/6*S8f, 12)})")
lims = (-mp.pi / 6 * S5f, mp.pi ** 2 / 6 * S6f, -mp.pi / 6 * S7f, mp.pi ** 2 / 6 * S8f)
Ns = sorted(sums)
conv = all(abs(sums[Ns[i + 1]][j] - lims[j]) < abs(sums[Ns[i]][j] - lims[j]) for i in range(len(Ns) - 1) for j in range(4)) and \
       all(abs(sums[Ns[-1]][j] - lims[j]) < mp.mpf("1e-8") for j in range(4))
check("termwise majorants of the proof (k <= 40, N = 50..25600) and the bounds (9.6) on y_k", maj_ok)
check("the four sums converge to -(pi/6)S_5, (pi^2/6)S_6, -(pi/6)S_7, (pi^2/6)S_8 (errors decreasing, < 1e-8 at N = 25600)", conv)
# pointwise limits for fixed k
for k in (1, 2, 5):
    N = 10 ** 6
    x = mp.pi * k / (2 * N); y = 2 * N * mp.asinh(mp.sin(x))
    a = N * Psi(x) * F(y); b = N / x * (F(y) - F(mp.pi * k))
    ap = N * Psi(x) / mp.sinh(y); bp = N / x * (1 / mp.sinh(y) - 1 / mp.sinh(mp.pi * k))
    la = -mp.pi * k / 6 * F(mp.pi * k); lb = mp.pi ** 2 * k ** 2 / (6 * mp.sinh(mp.pi * k) ** 2)
    lap = -mp.pi * k / (6 * mp.sinh(mp.pi * k)); lbp = mp.pi ** 2 * k ** 2 * mp.cosh(mp.pi * k) / (6 * mp.sinh(mp.pi * k) ** 2)
    ptw_ok &= all(abs(u - v) <= mp.mpf("1e-5") * abs(v) for u, v in ((a, la), (b, lb), (ap, lap), (bp, lbp)))
check("pointwise limits of a_k, b_k, a_k', b_k' (k = 1, 2, 5; N = 10^6; relative 1e-5)", ptw_ok)

OUT["n_fail"] = NFAIL
json.dump(OUT, open(_os.path.join(_R, 'data', 'msc_rigorous_spectral', '19_certified_d2_limits.json'), "w"), indent=1)
print("ALL CERTIFIED CHECKS OK" if NFAIL == 0 else f"{NFAIL} FAILURES")
sys.exit(1 if NFAIL else 0)
