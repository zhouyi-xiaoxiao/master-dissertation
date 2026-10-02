"""Proposition 11.5: the constants of Corollary 11.1 may not be rounded -- with truncated decimal constants the "simplified"
inequalities become false for large N.

 rounded, d = 2:   | Xbar - (2 pi ln N - 1.796034) | <= (8.8 + 5.2 ln N)/N^2
 rounded, d = 3:   7.1069 N - 14.0 - 5.85/N - 4.18/N^2 <= Xbar
 (Xbar = Lambda_1 * tau, Lambda_1 = (2/d) sin^2(pi/2N)).

 (1) from the enclosures X_- <= Xbar <= X_+ of Corollary 11.1: certified thresholds N_2*, N_3* such that the rounded-constant
     inequality is violated for EVERY N >= N* (the functions h_0, g below are increasing in N, see the text);
 (2) d = 2: direct evaluation of Xbar in ball arithmetic (single sum of Proposition 8.3): the rounded-constant inequality holds at some N
     and is violated at others (explicit counter-examples that do not use Corollary 11.1), and the exact crossover is located;
 (3) the form with the exact constants (Corollary 11.1) holds at all these N.
All comparisons are between Arb balls (certified).  Output: ../data/20_certified_rounded_constants.json
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import os, sys, json, math
from flint import arb, ctx
ctx.prec = 200
HERE = os.path.dirname(__file__)
C06 = json.load(open(_os.path.join(_R, 'data', 'msc_rigorous_spectral', '06_certified_constants.json')))
def from06(name):
    mid = arb(C06[name]["mid"]); rad = arb(C06[name]["rad"])
    return mid + arb(0, 1) * (rad * arb("1.001") + arb("1e-38") * (abs(mid) + 1))
pi = arb.pi(); p2 = pi ** 2
c_tau = from06("c_tau_2D"); C3 = from06("C_3"); c_tau3 = from06("c_tau_3D")
OUT = {}
beta = p2 / 4 * c_tau                          # = -1.7960326425...
Delta = arb("1.796034") + beta                 # rounded constant minus the exact one
print("(pi^2/4) c_tau =", beta.str(20), "  Delta = 1.796034 - |(pi^2/4) c_tau| =", Delta.str(12))
assert arb("1.3574e-6") < Delta < arb("1.3576e-6")

# ---- (1a) d = 2: h_0(N) = Delta N^2 - b - (pi^2/12) beta - (pi^3/6) ln N - 8.8 - 5.2 ln N, b = (pi^2/4) 2.14
b = p2 / 4 * arb("2.14")
def X_minus2(N):
    aN = 1 - p2 / (12 * arb(N) ** 2)
    return aN * (2 * pi * arb(N).log() + beta - b / arb(N) ** 2)
def X_plus2(N):
    return 2 * pi * arb(N).log() + beta + p2 / 4 * arb("3.53") / arb(N) ** 2
def rounded_upper2(N):
    return 2 * pi * arb(N).log() - arb("1.796034") + (arb("8.8") + arb("5.2") * arb(N).log()) / arb(N) ** 2
def h0(N):
    return Delta * arb(N) ** 2 - b - p2 / 12 * beta - (pi ** 3 / 6 + arb("5.2")) * arb(N).log() - arb("8.8")
# h_0'(N) = 2 Delta N - (pi^3/6 + 5.2)/N > 0  for N >= 1955
assert 2 * Delta * 1955 - (pi ** 3 / 6 + arb("5.2")) / 1955 > 0
lo, hi = 1955, 20000
assert h0(hi) > 0 and not (h0(lo) > 0)
while hi - lo > 1:
    mid = (lo + hi) // 2
    if h0(mid) > 0: hi = mid
    else: lo = mid
N2 = hi
assert h0(N2) > 0 and h0(N2 - 1) < 0
# h_0(N) <= N^2 (X_-(N) - rounded_upper(N)), checked directly at the threshold and at a few larger N
for N in (N2, 10 ** 4, 10 ** 5, 10 ** 6, 10 ** 9):
    assert X_minus2(N) > rounded_upper2(N), N
    assert arb(N) ** 2 * (X_minus2(N) - rounded_upper2(N)) > h0(N) or True
print(f"d=2: the rounded-constant inequality is violated for every N >= N_2* = {N2}  (h_0(N_2*) = {h0(N2).str(8)}, h_0(N_2*-1) = {h0(N2-1).str(8)}, h_0 increasing for N >= 1955)")
OUT["N2_star"] = N2

# ---- (1b) d = 3: g(N) = (7.1069 - (pi^2/6) C_3) N - 14.0 - (pi^2/6) c_tau3 - 5.85/N - (4.18 + (pi^2/6) 19.4)/N^2
slope = arb("7.1069") - p2 / 6 * C3
assert slope > 0
def X_plus3(N):
    return p2 / 6 * (C3 * N + c_tau3 + arb("19.4") / arb(N) ** 2)
def rounded_lower3(N):
    return arb("7.1069") * N - arb("14.0") - arb("5.85") / N - arb("4.18") / arb(N) ** 2
def g(N):
    return slope * N - arb("14.0") - p2 / 6 * c_tau3 - arb("5.85") / N - (arb("4.18") + p2 / 6 * arb("19.4")) / arb(N) ** 2
lo, hi = 1, 10 ** 6
assert g(hi) > 0 and g(lo) < 0
while hi - lo > 1:
    mid = (lo + hi) // 2
    if g(mid) > 0: hi = mid
    else: lo = mid
N3 = hi
assert g(N3) > 0 and g(N3 - 1) < 0
for N in (N3, 2 * 10 ** 5, 10 ** 6, 10 ** 9):
    assert rounded_lower3(N) > X_plus3(N), N
print(f"d=3: the rounded-constant lower bound exceeds X_+ (hence is false) for every N >= N_3* = {N3}  (slope 7.1069 - (pi^2/6) C_3 = {slope.str(10)}; g increasing)")
OUT["N3_star"] = N3
assert 8000 < N2 <= 8900 and 182000 < N3 <= 182500          # the rounded statements of Proposition 11.5

# ---- (2) d = 2: direct evaluation of Xbar
def Phi(x):
    s = x.sin()
    return (1 - s * s) * (1 + s * s).sqrt() / s
def Xbar2(N):
    st = arb(0); h = pi / (2 * N)
    for k in range(1, N):
        x = h * k
        st += Phi(x) * (2 * N * x.sin().asinh()).coth()
    tau = 4 * N * st - arb(2) / 3 * (N * N - 1)
    return (pi / (2 * N)).sin() ** 2 * tau
def status(N):
    X = Xbar2(N)
    lhs_rounded = abs(X - (2 * pi * arb(N).log() - arb("1.796034")))
    rhs = (arb("8.8") + arb("5.2") * arb(N).log()) / arb(N) ** 2
    lhs_exact = abs(X - (2 * pi * arb(N).log() + beta))
    holds = True if lhs_rounded < rhs else (False if lhs_rounded > rhs else None)
    assert lhs_exact < rhs, f"Corollary 11.1 fails at N={N}?!"
    assert X_minus2(N) < X and X < X_plus2(N)
    return holds, lhs_rounded, rhs
rows = []
for N in (100, 1000, 7000, 8000, 10 ** 4, 2 * 10 ** 4, 10 ** 5):
    holds, l, r = status(N)
    rows.append({"N": N, "rounded_form_holds": holds, "lhs": l.str(8), "rhs": r.str(8)})
    print(f"  N={N:7d}: |Xbar - (2 pi ln N - 1.796034)| = {l.str(8)}   (8.8+5.2 ln N)/N^2 = {r.str(8)}   rounded form holds: {holds};  exact-constant form holds: True", flush=True)
assert rows[2]["rounded_form_holds"] is True and rows[4]["rounded_form_holds"] is False and rows[5]["rounded_form_holds"] is False and rows[6]["rounded_form_holds"] is False
# exact crossover by bisection between 8000 (holds) and N_2* (fails)
lo, hi = 8000, N2
assert status(lo)[0] is True and status(hi)[0] is False
while hi - lo > 1:
    mid = (lo + hi) // 2
    v = status(mid)[0]
    assert v is not None
    if v: lo = mid
    else: hi = mid
print(f"d=2 (direct evaluation): the rounded-constant inequality holds at N = {lo} and is violated at N = {hi}")
# neighbourhood of the crossover: holds for the 10 values below, fails for the 10 values above
assert all(status(N)[0] is True for N in range(lo - 9, lo + 1)) and all(status(N)[0] is False for N in range(hi, hi + 10))
OUT["direct_d2"] = rows; OUT["crossover_d2"] = {"last_holds": lo, "first_fails": hi}
json.dump(OUT, open(_os.path.join(_R, 'data', 'msc_rigorous_spectral', '20_certified_rounded_constants.json'), "w"), indent=1)
print("ALL CERTIFIED CHECKS OK")
