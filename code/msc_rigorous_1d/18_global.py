"""Theorem 7.7 (all global maximisers for 2/3 <= q <= qbar = 99/100, N >= N0 = 601): certificates.

eps = 1/L^2 <= eps0, eta = 1/L <= eta0, q in the ball [2/3, 4/5], v = q(t-1)/L^2 = c - d eps.
frakS_1(t) = (L^4/q^2)(f(t+1)-f(t)) = G'(v) + eps H(v) + eps^2 R          (Proposition 5.4, j = 1)

Zone I   v in [va, c - delta0]:   G'(V_i) + eps H(V_i) + eps^2 R(V_i) > 0 on a partition (ball arithmetic)
Zone I'  v in [c + delta0, vb]:   ... < 0
Zone II  v in [c - delta0, c + delta0]:  with m := |G''(c)| - (1/2) M3 delta0 - eps0 MH' > 0,
           frakS_1/eps >= m d - |G''(c)|(kappa + q rho) - eps0 Rmax > 0     for d >= d1(q),
           frakS_1/eps < 0   for d <= kappa + q rho - q;
         need d1(q) <= kappa + q rho + q  (overlap with Theorem 6.1(A)) and |G''(c)| q > small.
Zone IV  v >= vb: the first mode dominates:
           sum_{k odd >= 3} (y_k/y_1)^3 exp(-2 vb (beta* y_k^2 - (1+e1) y_1^2)) + high < A_1(zeta_1).
Left region t < t_a (v < va):  f_q(t) <= q max_{k <= Ka} f_1(k),  Ka = ceil(1.5 va L^2) + 1,
           L^2 f_1(k) <= G(v) + eta |calE(v)| + eta^2 (rho_P0 + eta rho_E0),  v in [1.5 va - eps0, 1.5 va + eps0],
         versus  (L^2/q) f_q(t1) = G(v1) + eps H_0(v1) + eps^2 R_0 >= Glow,   v1 in [c - 4 eps0, c + eps0].
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
import core, q1
from core import arb, G, H, interval, ub, R_enclosure, PI

core.set_prec(128)
const = json.load(open(_os.path.join(_R, 'data', 'msc_rigorous_1d', '03_constants.json')))
c = arb(const["c_mid_80_digits"][:40], "1e-36")
G2c, G3c = G(2, c), G(3, c)
kappa = arb(3) / 4 + c / 6 * G3c / G2c
rho = -c / 2 * G3c / G2c
absG2 = abs(G2c)

N0 = int(sys.argv[1]) if len(sys.argv) > 1 else 601
_qn, _qd = map(int, (sys.argv[2] if len(sys.argv) > 2 else "99/100").split("/"))
TOL = sys.argv[3] if len(sys.argv) > 3 else None       # tolerance mode: B_high (and the Zone IV high-mode term) <= TOL is a hypothesis
if TOL is not None:
    core.HIGH_TOL = TOL
L0 = arb(2 * N0 - 1) / 2
eps0 = 1 / (L0 * L0)
eta0 = 1 / L0
eps = interval(arb(0), eps0)
QHI = arb(_qn) / _qd
q = interval(arb(2) / 3, QHI)
qbar = QHI
va = arb("0.15"); vb = arb("0.5"); delta0 = arb("0.01")
k0 = 15
out = dict(N0=N0, qbar=f"{_qn}/{_qd}", va=0.15, vb=0.5, delta0=0.01, k0=k0)


def piece_sign(lo, hi):
    V = interval(lo, hi)
    R, Bnd, parts = R_enclosure(1, V, L0, k0, q=q, qbar=qbar)
    val = G(1, V) + eps * H(1, V, q) + eps * eps * R
    return val, R


# ---- Zone I and I' ----
def zone(a, b_, want, n):
    okz = True
    worst = None
    for i in range(n):
        lo = a + (b_ - a) * i / n
        hi = a + (b_ - a) * (i + 1) / n
        val, R = piece_sign(lo, hi)
        ok = (val > 0) if want > 0 else (val < 0)
        if not ok:
            okz = False
            print("  FAILED piece", lo.str(8), hi.str(8), val.str(8))
        m = arb(val.lower()) if want > 0 else -arb(val.upper())
        worst = m if worst is None or m < worst else worst
    return okz, worst

okI, mI = zone(va, c - delta0, +1, 400)
okI2, mI2 = zone(c + delta0, vb, -1, 400)
print("Zone I  [0.15, c-0.01]: positive:", okI, " min margin", mI.str(6))
print("Zone I' [c+0.01, 0.5] : negative:", okI2, " min margin", mI2.str(6))
out.update(zoneI=bool(okI), zoneI_margin=mI.str(8), zoneIp=bool(okI2), zoneIp_margin=mI2.str(8))

# ---- Zone II ----
V = c + interval(-delta0, delta0)
M3 = ub(G(3, V))
Hp = arb(3) / 4 * G(3, V) + (1 - 3 * q) / 6 * G(3, V) + V / 6 * (1 - 3 * q) * G(4, V)
MH = ub(Hp)
R, Bnd, parts = R_enclosure(1, V, L0, k0, q=q, qbar=qbar)
Rmax = ub(R)
m = absG2 - M3 * delta0 / 2 - eps0 * MH
overlap = bool(m > 0)
right_small = True
right_far = True
d1max = arb(0)
nq = 64
for i in range(nq):
    qi = interval(arb(2) / 3 + (QHI - arb(2) / 3) * i / nq, arb(2) / 3 + (QHI - arb(2) / 3) * (i + 1) / nq)
    d1 = (absG2 * (kappa + qi * rho) + eps0 * Rmax) / m          # zone II is positive for d >= d1(q)
    overlap = overlap and bool(d1 < kappa + qi * rho + qi)       # overlap with the window of Theorem 6.1(A)
    dmax = ub(kappa + qi * rho)
    # right side: d in [0, kappa + q rho - q]:  -|G''| q + d eps0 (M3 d/2 + MH) + eps0 Rmax < 0
    right_small = right_small and bool(-absG2 * qi + dmax * eps0 * (M3 * dmax / 2 + MH) + eps0 * Rmax < 0)
    right_far = right_far and bool(absG2 * (kappa + qi * rho) > eps0 * Rmax)
    if ub(d1) > d1max:
        d1max = ub(d1)
print(f"Zone II: M3={float(M3.mid()):.2f} MH'={float(MH.mid()):.1f} Rmax={float(Rmax.mid()):.1f} m={m.str(8)} max d1={d1max.str(8)} "
      f"overlap (d1(q) < kappa+q rho+q on {nq} q-pieces): {overlap}; right: {right_small and right_far}")
out.update(zoneII=dict(M3=float(M3.upper()), MH=float(MH.upper()), Rmax=float(Rmax.upper()), m=m.str(10), d1max=d1max.str(10),
                       overlap=overlap, right=bool(right_small and right_far)))

# ---- Zone IV ----
y1 = PI / 4
zeta1 = (y1 / L0) ** 2
w1 = 2 * qbar * zeta1                      # w_1 = 2 q sin^2 x_1 <= 2 q zeta_1
e1 = w1 / (1 - w1)                         # B(zeta_1) <= Lam(w_1) <= 1/(1-w_1) = 1 + e1
A1low = 1 - arb(3) / 2 * zeta1             # A_1(zeta_1) >= 1 - (9/6) zeta_1
s = arb(0)
ks = 3
while True:
    y = ks * PI / 4
    term = (y / y1) ** 3 * (-2 * vb * (core.BETA_STAR * y * y - (1 + e1) * y1 * y1)).exp()
    s += term
    if ks > 41:
        break
    ks += 2
# geometric tail beyond ks: ratio <= (1+2/k)^3 exp(-vb beta* pi^2 (k+1)/2) < 1/2
ratio = (1 + arb(2) / ks) ** 3 * (-vb * core.BETA_STAR * PI * PI * (ks + 1) / 2).exp()
s += term * ratio / (1 - ratio)
# high modes: fewer than L, |term|/first <= (pi L/2 / y1)^3 exp(-vb (chi L^2 - 2 y1^2 (1+e1)))  (decreasing in L for L >= L0)
if TOL is None:
    chi = core.chi_rate(qbar)
    high = L0 * (PI * L0 / 2 / y1) ** 3 * (-vb * (chi * L0 * L0 - 2 * y1 * y1 * (1 + e1))).exp()
    okIV = bool(s + high < A1low) and bool(chi * L0 * L0 > 4 / vb)
else:
    high = arb(TOL)        # L (pi L/(2 y1))^3 e^{-vb(chi L^2 - 2 y1^2(1+e1))} <= 4 (pi/2)^3 L^8 e^{-0.15 chi L^2} <= TOL
    okIV = bool(s + high < A1low)
print("Zone IV: sum_{k>=3} + high =", (s + high).str(8), " < A_1(zeta_1) >=", A1low.str(8), ":", okIV)
out.update(zoneIV=dict(sum=(s + high).str(10), ok=okIV))

# ---- left region ----
core.set_prec(200)
w = interval(arb("0.225") - eps0, arb("0.225") + eps0)
rP, rE, rP0, rE0 = q1.remainders(arb(w.lower()), arb(w.upper()), L0)
Gup = G(0, arb(w.upper()))                 # G increasing on (0, c)
ma = Gup + eta0 * ub(q1.calE(0, w)) + eta0 * eta0 * (rP0 + eta0 * rE0)
V1 = c + interval(-4 * eps0, eps0)
R0, Bnd0, _ = R_enclosure(0, V1, L0, k0, q=q, qbar=qbar)
Glow = G(0, V1) + eps * H(0, V1, q) + eps * eps * R0
okL = bool(ma < arb(Glow.lower()))
# class modes are beyond Ka + 2:  1.5 va L^2 + 4 < c L^2 - b L - 1.26 - 10.3/L  at L0 (difference increasing in L)
d16 = json.load(open(_os.path.join(_R, 'data', 'msc_rigorous_1d', '16_q1_theorem.json')))
b = arb(d16["b"]); s2 = arb(d16["s2"]); K1 = arb(str(d16["rows"]["601"]["K1"]))
okC = bool(arb("0.225") * L0 * L0 + 4 < c * L0 * L0 - b * L0 + 1 + s2 - K1 / L0)
print("left region: max_{k<=Ka} L^2 f_1(k) <=", ma.str(10), "  <  (L^2/q) f_q(t1) >=", arb(Glow.lower()).str(10), ":", okL, " class modes beyond Ka:", okC)
out.update(left=dict(ma=ma.str(12), Glow=arb(Glow.lower()).str(12), ok=okL, class_modes_beyond=okC,
                     rho_P0=float(rP0.upper()), rho_E0=float(rE0.upper())))
out["ALL"] = bool(okI and okI2 and overlap and right_small and right_far and okIV and okL and okC)
print("ALL CERTIFICATES:", out["ALL"])
json.dump(out, open(os.path.join(_R, 'data', 'msc_rigorous_1d', f"18_global_N0_{N0}_q{_qn}_{_qd}" + ("_tol" if TOL else "") + ".json"), "w"), indent=1)
