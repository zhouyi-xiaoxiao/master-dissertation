"""Theorem 8.12 (general start, explicit finite-N law), analytic part: for all N >= N0, all starts x0 with
xi_N = (x0 - 1/2)/L in [0, XIMAX], all q in [0, 1/2]  (q = 0: continuous time):

     frakD(v; q) > 0  at  v = c(xi_N) - (kappa + q rho) eps - K eps^2,     eps = 1/L^2,
     frakD(v; q) < 0  at  v = c(xi_N) - (kappa + q rho) eps + K eps^2,

where frakD(v; q) = sum_k sigma_k cos(2 y_k xi_N) 2 y_k (-2 y_k^2) Ast_1(zeta_k) exp(-2 v y_k^2 B(zeta_k; q)) is the scaled
increment (discrete time: (L^4/q^2)(f(t+1) - f(t)) at real t = 1 + v L^2/q; continuous time: G_{N,x0}'(v)), and
     kappa = 1/2 + (c/6) G'''(c)/G''(c),   rho = -(c/2) G'''(c)/G''(c),   G = G_xi, c = c(xi)   (xi = xi_N).

Reduction (proof of Theorem 8.12): with v = c - d eps, d = kappa + q rho - s eps,
   L^4 frakD = s G''(c) + Gamma,
   Gamma = (1/2) G'''(c) d^2 - H'(c) d + eps[-(1/6) G''''(z) d^3 + (1/2) H''(z') d^2] + R,
   H = (1/2) G'' + (v/6)(1-3q) G''',  R = remainder of Proposition 8.11 (j = 1).
Certificate for a ball Xi x Q:  K * inf|G''(c)| > sup|Gamma|  over s in [-K, K], eps in [0, eps0], xi in Xi, q in Q.
Also certified (needed in discrete time): v_- L0 > (1 - xi)/2, which gives t_+ = 1 + v L^2/q > d - 1 for all L >= L0.

The box [0, XIMAX] x [0, 1/2] is covered adaptively: [0, XIMAX] is cut into NSTRIP strips; each strip x [0, 1/2] is bisected
(in xi if 40*width_xi > width_q, else in q) until every leaf box is certified.  One output line per strip (resumable).

Usage: 25_start_theorem.py N0 k0 K NSTRIP first last [XIMAX num/den]     -> data/25_start_N0_<N0>_K<K>.jsonl
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import os, sys, json, time
sys.path.insert(0, os.path.dirname(__file__))
import core, start
from core import arb, interval, ub
from start import Gxi, R_enclosure_start, c_enclosure

core.set_prec(128)


def certify(Xi, Q, L0, k0, K, C=None):
    eps0 = 1 / (L0 * L0)
    eps = interval(arb(0), eps0)
    if C is None:
        C, _ = c_enclosure(Xi)
    G2c, G3c, G4c = Gxi(2, C, Xi), Gxi(3, C, Xi), Gxi(4, C, Xi)
    if not (G2c < 0):
        return False, None, None, C
    kappa = arb(1) / 2 + C / 6 * G3c / G2c
    rho = -C / 2 * G3c / G2c
    base = kappa + Q * rho
    sK = interval(-K, K)
    d = base - sK * eps
    V = C - d * eps
    R, Bnd, parts = R_enclosure_start(1, V, Xi, L0, k0, q=Q, qbar=arb(Q.upper()))
    G4V, G5V = Gxi(4, V, Xi), Gxi(5, V, Xi)
    Hp_c = G3c / 2 + (1 - 3 * Q) / 6 * G3c + C / 6 * (1 - 3 * Q) * G4c
    Hpp = G4V / 2 + (1 - 3 * Q) / 3 * G4V + V / 6 * (1 - 3 * Q) * G5V
    gam = G3c * d * d / 2 - Hp_c * d + eps * (-G4V * d ** 3 / 6 + Hpp * d * d / 2) + R
    need = ub(gam) / arb(abs(G2c).lower())
    ok = bool(K > need)
    # discrete time: the test points lie beyond the trivial zeros (t > d - 1):  v L^2 > q (L (1 - xi) - 2)  <=  v L0 > (1 - xi)/2
    ok2 = bool(arb(V.lower()) * L0 > (1 - arb(Xi.lower())) / 2)
    return ok and ok2, need, Bnd, C


from fractions import Fraction as Fr


def fa(x):
    return arb(x.numerator) / x.denominator


def rec(xa, xb, qa, qb, L0, k0, K, st, depth=0):
    Xi = interval(fa(xa), fa(xb)); Q = interval(fa(qa), fa(qb))
    st["tried"] += 1
    try:
        ok, need, Bnd, C = certify(Xi, Q, L0, k0, K)
    except (AssertionError, RuntimeError):
        ok, need = False, None
    if ok:
        st["leaves"] += 1
        st["need"] = max(st["need"], float(need.upper()))
        st["min_dxi"] = min(st["min_dxi"], float(xb - xa)); st["min_dq"] = min(st["min_dq"], float(qb - qa))
        return True
    if depth > 30:
        st["fail"].append((float(xa), float(xb), float(qa), float(qb)))
        return False
    if (xb - xa) * 40 > (qb - qa):
        xm = (xa + xb) / 2
        return rec(xa, xm, qa, qb, L0, k0, K, st, depth + 1) & rec(xm, xb, qa, qb, L0, k0, K, st, depth + 1)
    qm = (qa + qb) / 2
    return rec(xa, xb, qa, qm, L0, k0, K, st, depth + 1) & rec(xa, xb, qm, qb, L0, k0, K, st, depth + 1)


if __name__ == "__main__":
    N0 = int(sys.argv[1]); k0 = int(sys.argv[2]); Kstr = sys.argv[3]; K = arb(Kstr)
    nstrip = int(sys.argv[4]); first = int(sys.argv[5]); last = int(sys.argv[6])
    xn, xd = map(int, (sys.argv[7] if len(sys.argv) > 7 else "1/2").split("/"))
    L0 = arb(2 * N0 - 1) / 2
    out_name = os.path.join(_R, 'data', 'msc_rigorous_1d', f"25_start_N0_{N0}_K{Kstr.replace('.', 'p')}.jsonl")
    done = set()
    if os.path.exists(out_name):
        for line in open(out_name):
            try:
                done.add(json.loads(line)["strip"])
            except Exception:
                pass
    for i in range(first, last + 1):
        if i in done:
            continue
        t0 = time.time()
        st = dict(tried=0, leaves=0, need=0.0, fail=[], min_dxi=1.0, min_dq=1.0)
        xa, xb = Fr(xn * i, xd * nstrip), Fr(xn * (i + 1), xd * nstrip)
        ok = rec(xa, xb, Fr(0), Fr(1, 2), L0, k0, K, st)
        rec_ = dict(strip=i, nstrip=nstrip, N0=N0, k0=k0, K=Kstr, xi_lo=float(xa), xi_hi=float(xb), certified=bool(ok), boxes_tried=st["tried"],
                    leaves=st["leaves"], max_need=st["need"], min_dxi=st["min_dxi"], min_dq=st["min_dq"], fail=st["fail"][:5],
                    secs=round(time.time() - t0, 1))
        with open(out_name, "a") as fh:
            fh.write(json.dumps(rec_) + "\n")
        print(rec_, flush=True)
