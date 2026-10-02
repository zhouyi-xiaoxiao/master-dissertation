"""Certified (Arb ball arithmetic, python-flint) constants and checks for Part III of note R2.

 (1) constants of the three-dimensional theorem: K(1), end points of the enclosure of eps_b, A_0, kappa_3, B_D, B_e,
     and the resulting enclosure constants; constants of the two-dimensional theorem (re-derived from 06).
 (2) certified values of tau_d(N), m_d(N) (exact spectral sums in ball arithmetic) and certified verification of the
     enclosures of Theorem 9.1 (d = 2) and Theorem 10.1 (d = 3) for the listed N.
 (3) certified enclosures of sigma_0, sigma_1, r_0, r_1 and certified verification of every inequality of Theorem 7.10
     (corner-to-corner, explicit in Xbar, Z_d) and of its N-explicit form (Corollaries 11.1, 11.2).

All comparisons 'a < b' between balls are certified (python-flint returns True only if it holds for all points of the balls).
Output: ../data/09_certified_part3.json
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import os, sys, json, math, itertools
from collections import Counter
import numpy as np
from scipy.optimize import brentq
from flint import arb, acb, ctx
sys.path.insert(0, os.path.dirname(__file__))
from t2lib import corner_spectral_data, group_visible
ctx.prec = 200
HERE = os.path.dirname(__file__)
C06 = json.load(open(_os.path.join(_R, 'data', 'msc_rigorous_spectral', '06_certified_constants.json')))
def from06(name):
    """Ball of script 06, rebuilt from the decimal strings of its midpoint (40 digits) and radius (5 digits).
    Both strings are rounded, so the radius is inflated: 1.001*rad + 1e-38*(|mid|+1) covers both rounding errors."""
    mid = arb(C06[name]["mid"]); rad = arb(C06[name]["rad"])
    return mid + arb(0, 1) * (rad * arb("1.001") + arb("1e-38") * (abs(mid) + 1))
C14 = json.load(open(_os.path.join(_R, 'data', 'msc_rigorous_spectral', '14_certified_epstein.json')))
def from14_upper(name):
    """certified upper bound of a constant of script 14 (decimal string of the upper end, re-inflated)"""
    return arb(C14[name]["upper"]) * (1 + arb("1e-40")) + arb("1e-40")
pi = arb.pi(); ln2 = arb(2).log(); gam = arb.const_euler()
OUT = {"constants": {}, "rows2": [], "rows3": [], "cc": []}
def enc(name, x, note=""):
    OUT["constants"][name] = {"str": x.str(25), "lower": float(x.lower()), "upper": float(x.upper()), "note": note}
    print(f"ENCLOSURE {name:22s} = {x.str(25)}   {note}", flush=True)

# ------------------------------------------------------------------------------------------------ (1) constants
# K(1) = 4 int_0^{pi/2} cos^2(p) sqrt((2+sin^2 p)/(1+sin^2 p)) dp
def fK1(p, analytic):
    s2 = p.sin() ** 2
    return 4 * p.cos() ** 2 * ((2 + s2) / (1 + s2)).sqrt(analytic=analytic)
ctx.prec = 160
K1 = acb.integral(fK1, 0, acb.pi() / 2, rel_tol=arb(2) ** -110, abs_tol=arb(2) ** -110).real
ctx.prec = 200
enc("K(1)", K1, "K(s) = 4 int_0^1 sqrt((1-t^2)(1+s^2+t^2)/(s^2+t^2)) dt at s = 1")
g1_0 = 12 * ln2 - 4 - 2 * pi
g_end = 2 * (K1 - pi)
eb_lo = -g_end; eb_hi = -g1_0 + g_end
enc("g1(0)", g1_0, "12 ln 2 - 4 - 2 pi"); enc("g1(1)=g3(1)", g_end, "2 (K(1) - pi)")
enc("eps_b_lower", eb_lo); enc("eps_b_upper", eb_hi); enc("eps_b_limit", pi + 2 - 6 * ln2, "pi + 2 - 6 ln 2")
I_Psi = -arb(1) / 2 + arb(3) / 2 * ln2 - pi.log()
A0 = -1 - 24 * ln2 / pi + 12 / pi * (gam + I_Psi)
enc("A_0", A0, "-1 - (24 ln 2)/pi + (12/pi)(gamma + I_Psi)")
kap3 = arb(2).sqrt() * arb(2).sqrt().asinh()
enc("kappa_3", kap3, "sqrt(2) asinh(sqrt 2)")
R = 30
B_D = arb(0)
for a in range(0, R + 1):
    for b in range(0, R + 1):
        if a or b:
            r = arb(a * a + b * b).sqrt()
            B_D += (1 if a == 0 else 2) * (1 if b == 0 else 2) / (r * ((2 * kap3 * r).exp() - 1))
tailD = 8 * (-2 * kap3 * (R + 1)).exp() / (1 - (-2 * kap3).exp()) ** 2
B_D = B_D + arb(0, 1) * tailD + tailD * 0   # [B_D - tail, B_D + tail]; only the upper end is used
enc("B_D", B_D, "sum_{k in Z^2\\0} 1/(|k|(e^{2 kappa_3 |k|}-1)); tail beyond |k|_inf > 30 bounded by 8 e^{-2 kappa_3 31}/(1-e^{-2 kappa_3})^2")
B_e = sum((pi / (4 * kap3 * k)).sqrt() * (-2 * kap3 * k).exp() / (1 - (-2 * kap3 * k).exp()) for k in range(1, 61))
tailE = (pi / (4 * kap3)).sqrt() * (-2 * kap3 * 61).exp() / (1 - (-2 * kap3).exp()) ** 2
B_e = B_e + arb(0, 1) * tailE
enc("B_e", B_e, "sum_{k>=1} sqrt(pi/(4 kappa_3 k)) e^{-2 kappa_3 k}/(1-e^{-2 kappa_3 k})")
c3D = 3 * (8 * B_e + 2 * B_D)
enc("3(8B_e+2B_D)", c3D)
M2 = arb("2.35")
S1, S2, S3, S4 = (from06(n) for n in ("S1", "S2", "S3", "S4"))
cEm = arb("1.09") * pi / 2 * S2 + 8 / pi * (-4 * pi).exp() / (1 - (-2 * pi).exp()) ** 2      # N E_c >= -cEm
cEp = pi ** 2 / 6 * S1                                                                      # N E_c <=  cEp
cA = arb("1.09") * pi / 2 * S4 + pi ** 2 / 6 * S3 + 8 / pi * (-2 * pi).exp() / ((1 - (-pi).exp()) * (1 - (-4 * pi).exp()))
enc("c_E^-", cEm, "N E_c >= -c_E^-"); enc("c_E^+", cEp, "N E_c <= c_E^+"); enc("c_A", cA, "N |e_A| <= c_A")
R2_lo = arb(2) / 3 - 4 / (3 * pi) - pi ** 2 * M2 / 12 - 4 * cEm
R2_hi = arb(2) / 3 + pi ** 2 * M2 / 12 + 4 * cEp
enc("R_tau2_lower", R2_lo); enc("R_tau2_upper", R2_hi)
Rm_lo = -arb(2) / 3 - 4 * cA; Rm_hi = -arb(2) / 3 + 4 * cA
enc("R_m2_lower", Rm_lo); enc("R_m2_upper", Rm_hi)
lo3 = A0 + 3 / pi * eb_lo; hi3 = A0 + 3 / pi * eb_hi + c3D
rho_lo = 1 - 2 / pi - pi ** 2 * M2 / 8; rho_hi = 1 + pi ** 2 * M2 / 8
enc("tau3_N2_lower", lo3, "coefficient of N^2 in the lower bound of tau_3 - C_3 N^3"); enc("tau3_N2_upper", hi3)
enc("rho_lower", rho_lo); enc("rho_upper", rho_hi)
c_tau = from06("c_tau_2D"); C3 = from06("C_3"); c_tau3 = from06("c_tau_3D"); c_mt3 = from06("c_m_minus_tau_3D")
Z2 = from14_upper("Z_2"); Z3 = from14_upper("Z_3")        # sharp values (Lemma 7.6, theta splitting)
assert Z2 < from06("Z_2_upper") and Z3 < from06("Z_3_upper")   # consistent with the elementary enclosures of script 06
# the rounded constants quoted in the text must be implied by the enclosures:
assert R2_lo > arb("-2.14") and R2_hi < arb("3.53")
assert Rm_lo > arb("-9.06") and Rm_hi < arb("7.72") and cA < arb("2.096")
assert lo3 > arb("-8.51") and hi3 < arb("-0.53") and rho_lo > arb("-2.54") and rho_hi < arb("3.90")
assert eb_lo > arb("-2.207") and eb_hi < arb("4.172")
assert Z2 < arb("6.02682") and Z3 < arb("16.53232")
# constants of the simplified forms in Corollary 11.1
p2 = pi ** 2
assert arb("-1.7960327") < p2 / 4 * c_tau < arb("-1.7960326")
assert arb("7.1068721") < p2 / 6 * C3 < arb("7.1068722")
assert p2 / 4 * arb("2.14") < arb("5.281") and p2 / 4 * arb("3.53") < arb("8.711") and p2 / 12 < arb("0.8225")
assert arb("5.281") - arb("0.8225") * arb("1.7960326") < arb("3.81") and arb("0.8225") * 2 * pi < arb("5.17")   # b + aB <= 3.81 + 5.17 ln N
# the same statement with the exact a = pi^2/12 (the line above uses the UPPER bound 0.8225 of a in a term with a negative
# coefficient; the margin 0.006 covers this, but the exact form is the rigorous one):  b + a B = [b + a (pi^2/4) c_tau] + a 2 pi ln N
assert p2 / 4 * arb("2.14") + p2 / 12 * (p2 / 4 * c_tau) < arb("3.81") and p2 / 12 * 2 * pi < arb("5.17")
assert arb("3.81") + arb("5.17") < arb("8.8") + arb("5.2") and p2 / 4 * arb("3.53") < arb("8.8")                 # hence |Xbar - B| <= (8.8 + 5.2 ln N)/N^2
# every rounded number quoted in the PROOFS of Theorems 9.1 and 10.1(b) is asserted
assert cEm < arb("0.11009") and cEp < arb("0.23006") and cA < arb("2.0957")                                      # (9.8), Step 4
assert R2_lo > arb("-2.1309") and R2_hi < arb("3.5197") and 4 * cA < arb("8.383")                                # Steps 3, 4
assert lo3 > arb("-8.5082") and hi3 < arb("-0.5328")                                                             # proof of Theorem 10.1(b)
assert rho_lo > arb("-2.536") and rho_hi < arb("3.900")
# d = 3 simplified forms of Corollary 11.1:  a B = (pi^2/12)(pi^2/6)(C_3 N + c_tau3) <= 5.8452 N - 7.331
assert p2 / 12 * p2 / 6 * C3 < arb("5.8452") and p2 / 12 * p2 / 6 * c_tau3 < arb("-7.331")
assert p2 / 6 * arb("17.4") < arb("28.63") and arb("28.63") - arb("7.331") < arb("21.3") and p2 / 6 * arb("19.4") < arb("31.92")
assert p2 / 6 * arb("82.4") < arb("135.55") and p2 / 6 * arb("82.4") + p2 / 12 * p2 / 6 * c_mt3 < arb("137.62") and p2 / 6 * arb("80.4") < arb("132.26")
assert p2 / 4 * arb("9.06") < arb("22.36") and p2 / 4 * arb("9.06") + p2 / 12 * pi * ln2 < arb("24.15") and p2 / 4 * arb("7.72") < arb("19.05")
assert arb("-8.9136330") < p2 / 6 * c_tau3 < arb("-8.9136329") and arb("2.5193561") < p2 / 6 * c_mt3 < arb("2.5193562")
C16 = json.load(open(_os.path.join(_R, 'data', 'msc_rigorous_spectral', '16_certified_d3_remainder.json')))["rounded"]
assert C16["R3_lo"] == "-17.4" and C16["R3_hi"] == "19.4" and C16["3c_alt"] == "81.4"      # constants of Theorem 10.1(c), certified in script 16
print("CERTIFIED: rounded constants of Theorems 9.1 and 10.1 are implied by the enclosures", flush=True)

# ------------------------------------------------------------------------------------------------ (2),(3)
ctx.prec = 256
pi = arb.pi()

def spectral_arb(N, d):
    c = [(pi * k / N).cos() for k in range(N)]
    w1 = [(1 + c[k]) / 2 * (1 if k == 0 else 2) for k in range(N)]
    out = []
    for k in itertools.combinations_with_replacement(range(N), d):
        if not any(k):
            continue
        cnt = Counter(k); mult = math.factorial(d)
        for v in cnt.values(): mult //= math.factorial(v)
        lam = sum((1 - c[j]) for j in k) / d
        w = arb(mult)
        for j in k: w *= w1[j]
        out.append((lam, w, (-1) ** sum(k)))
    return out

def poles(N, d, nz, Lam1, Lam2, eta_bits=150):
    def nG(s):   return -1 / s + sum(w / (l - s) for l, w, _ in nz)
    def nGx(s):  return -1 / s + sum(sg * w / (l - s) for l, w, sg in nz)
    def ndG(s):  return 1 / (s * s) + sum(w / (l - s) ** 2 for l, w, _ in nz)
    lamf, wf, sgf = corner_spectral_data(N, d)
    Lg, Wg, SW = group_visible(lamf, wf, sgf)
    Gf = lambda s: np.sum(Wg / (Lg - s))
    def froot(i):
        gap = Lg[i + 1] - Lg[i]
        e = 1e-9 * gap
        while Gf(Lg[i] + e) > 0: e /= 8
        lo = Lg[i] + e
        e = 1e-9 * gap
        while Gf(Lg[i + 1] - e) < 0: e /= 8
        return brentq(Gf, lo, Lg[i + 1] - e, xtol=1e-300, rtol=1e-15, maxiter=1000)
    res = {}
    brackets = [(arb(0), Lam1), (Lam1, Lam2)]
    for j in (0, 1):
        s = arb(froot(j))
        for _ in range(4):
            s = (s - nG(s) / ndG(s)).mid()
        eta = arb(2) ** (-eta_bits)
        s_lo, s_hi = (s * (1 - eta)).mid(), (s * (1 + eta)).mid()
        ok = (brackets[j][0] < s_lo) and (s_hi < brackets[j][1]) and (nG(s_lo) < 0) and (nG(s_hi) > 0)
        assert ok, f"pole certification failed N={N} d={d} j={j}"
        S = s_lo.union(s_hi)
        res[f"sigma{j}"] = S; res[f"r{j}"] = -nGx(S) / (S * ndG(S))
    return res

def fl(v):
    return None if v is None else float(v.mid())

def cc_bounds(d, X, W, Z, th0, Mp, Mm):
    """Mm <= Lambda_1 (m - tau) <= Mp  (Theorem 7.10: Mm = mu_-, Mp = l_W;  Corollary 11.2: Mm = M_-, Mp = M_+)."""
    """Bounds of Theorem 7.10 / Corollary 11.2 as functions of (a lower/upper pair for) Xbar.  X = (X_lo, X_hi) arb pair."""
    Xl, Xh = X
    b = {}
    if not (Xl > 1):
        return None
    beta = Xl / (Xl - 1)                                   # beta is decreasing in Xbar -> largest at X_lo
    kap_hi = Z / Xl ** 2; kap_lo = W / Xh ** 2
    b["t_lo"] = 2 / (1 + (1 + 4 * beta * kap_hi).sqrt()); b["t_hi"] = 2 / (1 + (1 + 4 * kap_lo).sqrt())
    b["mu_hi"] = Mp / Xl
    b["s0m_lo"] = (1 + Mm / Xh) * b["t_lo"]; b["s0m_hi"] = (1 + Mp / Xl) * b["t_hi"]      # sigma_0 m = (1 + mu) sigma_0 tau
    b["s0m_lo_lin"] = (1 + Mm / Xh) * (1 - beta * kap_hi)
    b["mu_lo"] = Mm / Xh
    b["r0_hi_minus_mu"] = 1 + beta * kap_hi
    b["r0_lo_minus_mu"] = None
    if beta * kap_hi < 1:
        b["r0_lo_factor"] = beta * (1 + beta) * kap_hi         # r0 >= (1+mu)(1 - this)
    else:
        b["r0_lo_factor"] = None
    b["d_hi"] = b["d_lo"] = b["r1_lo"] = b["r1_hi"] = None
    if Xl - W + th0 - 1 > 0:
        d_hi = W / (Xl - W + th0 - 1)
        b["d_hi"] = d_hi
        if d_hi < 1:
            om = 2 * (1 + d_hi) / (1 - d_hi)
            d_lo = W / (Xh - W + om * (Z - W) - 1 / (1 + d_hi))
            b["d_lo"] = d_lo; b["omega"] = om
            y_lo = W - Mp - om * (Z - W) - 1
            y_hi = W - Mm + om * (Z - W) - 1 / (1 + d_hi)
            n_lo = W + (d_lo * y_lo).min(d_hi * y_lo)        # ball containing min / max: rigorous
            n_hi = W + (d_lo * y_hi).max(d_hi * y_hi)
            D_lo = (1 + d_lo) * W + d_lo ** 2 * (1 / (1 + d_hi) + (1 + d_lo) * th0)
            D_hi = (1 + d_hi) * W + d_hi ** 2 * (1 + om ** 2 * (Z - W) / (1 + d_hi))
            b["r1_lo"] = -d_hi * n_hi / D_lo                 # most negative
            if n_lo > 0:
                b["r1_hi"] = -d_lo * n_lo / D_hi
    return b

def run(d, N, check_poles=True):
    nz = spectral_arb(N, d)
    tau = sum(w / l for l, w, s in nz)
    m = sum((1 - s) * w / l for l, w, s in nz)
    K2 = sum(w / (l * l) for l, w, s in nz)
    Lam1 = (1 - (pi / N).cos()) / d
    Lam2 = 2 * Lam1
    W = 2 * d * (pi / (2 * N)).cos() ** 2
    lW = 1 + W.log()
    Z = Z2 if d == 2 else Z3
    th0 = arb(d * (d - 1) // 2) * (pi / (2 * N)).cos() ** 4
    mum = arb(2) / 3 * (N * N - 1) * (pi / (2 * N)).sin() ** 2            # mu_-(N), Lemma 7.2
    Nn = arb(N); u = pi / (2 * N)
    X = Lam1 * tau
    chk = {}
    row = dict(d=d, N=N, tau=tau.str(22), m=m.str(22), X=fl(X))
    # ---- enclosures of tau, m (Theorems 9.1 / 10.1) and of Xbar (Corollary 11.1)
    if d == 2:
        Rt = tau - 8 / pi * Nn ** 2 * Nn.log() - c_tau * Nn ** 2
        Rm = m - tau - 4 * arb(2).log() / pi * Nn ** 2
        chk["R_tau2"] = bool(Rt > arb("-2.14") and Rt < arb("3.53"))
        chk["R_m2"] = bool(Rm > arb("-9.06") and Rm < arb("7.72"))
        row.update(R_tau=fl(Rt), R_m=fl(Rm))
        tau_lo = 8 / pi * Nn ** 2 * Nn.log() + c_tau * Nn ** 2 - arb("2.14"); tau_hi = tau_lo + arb("5.67")
    else:
        Dt = tau - C3 * Nn ** 3
        chk["tau3"] = bool(Dt > arb("-8.51") * Nn ** 2 - arb("2.54") and Dt < arb("-0.53") * Nn ** 2 + arb("3.90"))
        row.update(tau3_minus_C3N3_over_N2=fl(Dt / Nn ** 2), m_minus_tau_over_N2=fl((m - tau) / Nn ** 2))
        R3 = tau - C3 * Nn ** 3 - c_tau3 * Nn ** 2; R3p = m - tau - c_mt3 * Nn ** 2
        chk["tau3_second_order"] = bool(R3 > arb("-17.4") and R3 < arb("19.4") and R3p > arb("-82.4") and R3p < arb("80.4") and m - tau > Nn ** 2 - 1)
        row.update(R_3=fl(R3), R_3prime=fl(R3p))
        tau_lo = C3 * Nn ** 3 + c_tau3 * Nn ** 2 - arb("17.4"); tau_hi = C3 * Nn ** 3 + c_tau3 * Nn ** 2 + arb("19.4")
    lam_lo = (2 / arb(d)) * u ** 2 * (1 - u ** 2 / 3); lam_hi = (2 / arb(d)) * u ** 2
    assert lam_lo < Lam1 and Lam1 < lam_hi
    X_lo = lam_lo * tau_lo; X_hi = lam_hi * tau_hi
    if d == 2:
        mt_lo = 4 * arb(2).log() / pi * Nn ** 2 - arb("9.06"); mt_hi = 4 * arb(2).log() / pi * Nn ** 2 + arb("7.72")
    else:
        mt_lo = c_mt3 * Nn ** 2 - arb("82.4"); mt_hi = c_mt3 * Nn ** 2 + arb("80.4")
    M_lo = mum.max(lam_lo * mt_lo); M_hi = lW.min(lam_hi * mt_hi)                 # M_-, M_+ of Corollary 11.1
    chk["M_enclosure"] = bool(M_lo < Lam1 * (m - tau) and Lam1 * (m - tau) < M_hi)
    row.update(M_lo=fl(M_lo), M_hi=fl(M_hi), Lam1_m_minus_tau=fl(Lam1 * (m - tau)))
    chk["Xbar_enclosure"] = bool(X_lo < X and X < X_hi)
    row.update(X_lo=fl(X_lo), X_hi=fl(X_hi))
    # ---- general-d lemmas of Section 7
    chk["m_minus_tau"] = bool(Lam1 * (m - tau) > mum and mum > 1 - arb(10) ** -40 and Lam1 * (m - tau) < lW)
    # simplified forms of Corollary 11.1 (exact constants)
    if d == 2:
        chk["Xbar_simplified"] = bool(abs(X - (2 * pi * Nn.log() + pi ** 2 / 4 * c_tau)) < (arb("8.8") + arb("5.2") * Nn.log()) / Nn ** 2)
        chk["M_simplified"] = bool(abs(Lam1 * (m - tau) - pi * arb(2).log()) < arb("24.2") / Nn ** 2)
    else:
        Bx = pi ** 2 / 6 * (C3 * Nn + c_tau3)
        chk["Xbar_simplified"] = bool(-arb("5.85") / Nn - arb("21.3") / Nn ** 2 < X - Bx and X - Bx < arb("31.92") / Nn ** 2)
        chk["M_simplified"] = bool(abs(Lam1 * (m - tau) - pi ** 2 / 6 * c_mt3) < arb("137.7") / Nn ** 2)
    chk["K2_le_Z"] = bool(Lam1 ** 2 * K2 < Z)
    chk["K2_ge_W_plus_th0"] = bool(Lam1 ** 2 * K2 > W + th0 - arb(10) ** -40)
    chk["tau_lower"] = bool(tau > 2 * Nn ** d * (1 - Nn ** (-d)) ** 2 - arb(10) ** -40)
    if check_poles:
        res = poles(N, d, nz, Lam1, Lam2)
        s0, s1, r0, r1 = res["sigma0"], res["sigma1"], res["r0"], res["r1"]
        delta = s1 / Lam1 - 1; mu = (m - tau) / tau; t = s0 * tau
        row.update(sigma0_tau=fl(t), sigma0_m=fl(s0 * m), delta=fl(delta), r0=fl(r0), r1=fl(r1), mu=fl(mu))
        chk["signs"] = bool(r0 > 1 and r1 < 0 and s0 * m > 1)            # Theorem 7.9: r_0 > 1, sigma_0 m > 1
        chk["s0m_quant"] = bool(s0 * m > (X + mum) / (X + 1))
        for tag, Xpair, Mp, Mm in (("exactX", (X, X), lW, mum), ("explicitN", (X_lo, X_hi), M_hi, M_lo)):
            b = cc_bounds(d, Xpair, W, Z, th0, Mp, Mm)
            if b is None:
                continue
            c = {}
            c["sigma0_tau"] = bool(b["t_lo"] < t and t < b["t_hi"])
            c["sigma0_m"] = bool(b["s0m_lo_lin"] < s0 * m and b["s0m_lo"] < s0 * m and s0 * m < b["s0m_hi"])
            c["mu_range"] = bool(b["mu_lo"] < mu and mu < b["mu_hi"])
            c["r0_upper"] = bool(r0 < (1 + mu) + (b["r0_hi_minus_mu"] - 1))
            if b["r0_lo_factor"] is not None:
                c["r0_lower"] = bool(r0 > (1 + mu) * (1 - b["r0_lo_factor"]))
                if b["r0_lo_factor"] < 1:
                    c["r0_lower_muminus"] = bool(r0 > (1 + Mm / Xpair[1]) * (1 - b["r0_lo_factor"]))
                    b["r0_lo_N"] = (1 + Mm / Xpair[1]) * (1 - b["r0_lo_factor"])
            b["r0_hi_N"] = 1 + Mp / Xpair[0] + (b["r0_hi_minus_mu"] - 1)
            c["r0_upper_N"] = bool(r0 < b["r0_hi_N"])
            if b["d_hi"] is not None:
                c["delta_upper"] = bool(delta < b["d_hi"])
                c["gap_lower"] = bool(s1 - s0 > Lam1 * (1 - 1 / Xpair[0]))
            if b["d_lo"] is not None:
                c["delta_lower"] = bool(delta > b["d_lo"])
                c["gap_2sided"] = bool(s1 - s0 > Lam1 * (1 + b["d_lo"] - 1 / Xpair[0]) and s1 - s0 < Lam1 * (1 + b["d_hi"] - 1 / (1 + Xpair[1])))
                c["r1_lower"] = bool(r1 > b["r1_lo"])
                if b["r1_hi"] is not None:
                    c["r1_upper"] = bool(r1 < b["r1_hi"])
            chk[tag] = c
            row[tag + "_bounds"] = {k: fl(v) for k, v in b.items() if isinstance(v, arb)}
    row["checks"] = chk
    flat = []
    for k, v in chk.items():
        if isinstance(v, dict):
            flat += [(k + "." + kk, vv) for kk, vv in v.items()]
        else:
            flat.append((k, v))
    bad = [k for k, v in flat if not v]
    row["n_checks"] = len(flat); row["failed"] = bad
    return row

def main():
    allok = True
    plan = [(2, [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16, 20, 25, 35, 50, 70, 100, 150, 200, 300], True),
            (3, [2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25, 30, 40, 50], True),
            (3, [70, 100], False)]
    for d, Ns, cp in plan:
        for N in Ns:
            r = run(d, N, cp)
            OUT["rows2" if d == 2 else "rows3"].append(r)
            allok &= not r["failed"]
            extra = (f"R_tau={r['R_tau']:+.5f} R_m={r['R_m']:+.5f}" if d == 2 else
                     f"(tau-C3N^3)/N^2={r['tau3_minus_C3N3_over_N2']:+.5f} (m-tau)/N^2={r['m_minus_tau_over_N2']:.5f}")
            pol = (f" s0*tau={r['sigma0_tau']:.6f} s0*m={r['sigma0_m']:.6f} delta={r['delta']:.5f} r0={r['r0']:.6f} r1={r['r1']:.6f}" if cp else "")
            print(f"d={d} N={N:4d} X={r['X']:.4f} in [{r['X_lo']:.4f},{r['X_hi']:.4f}] {extra}{pol} "
                  f"checks: {'ALL OK ('+str(r['n_checks'])+')' if not r['failed'] else 'FAILED '+str(r['failed'])}", flush=True)
            json.dump(OUT, open(_os.path.join(_R, 'data', 'msc_rigorous_spectral', '09_certified_part3.json'), "w"), indent=1)
    OUT["all_ok"] = bool(allok)
    json.dump(OUT, open(_os.path.join(_R, 'data', 'msc_rigorous_spectral', '09_certified_part3.json'), "w"), indent=1)
    print("ALL CERTIFIED CHECKS OK:", allok)

if __name__ == "__main__":
    main()
