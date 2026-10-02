"""Theorem 8.12, computer-assisted part: for 3 <= N <= NMAX, every start x0 with xi_N = (x0 - 1/2)/L <= 1/2
(i.e. x0 <= (2N+1)/4) and every q in [0, 1/2] (q = 0: continuous time) certify

    F_-(q) := frakD(v_-(q); q) > 0,    F_+(q) := frakD(v_+(q); q) < 0,    v_-+(q) = c - (kappa + q rho)/L^2 -+ K/L^4,

c = c(xi_N), kappa = 1/2 + (c/6) G'''(c)/G''(c), rho = -(c/2) G'''(c)/G''(c)  (G = G_{xi_N}), K = 3.5, where

    frakD(v; q) = sum_{k = 2m-1} sigma_k cos(2 y_k xi_N) 2 y_k (-2 y_k^2) S^3 cos(x_k) exp(-v e_k Lam(q s_k)),
    e_k = 2 L^2 sin^2 x_k,  s_k = 2 sin^2 x_k,  Lam(w) = -ln(1-w)/w,

is the scaled increment: G_{N,x0}'(v) for q = 0, and (L^4/q^2) (Phi(t+1) - Phi(t)) at real t = 1 + v L^2/q for 0 < q <= 1/2
(all eigenvalues 1 - q s_k are positive).  Also certified for 0 < q <= 1/2: the upper test point lies beyond the trivial
zeros, t_+ = 1 + v_+ L^2/q > d - 1 (d = N - x0), i.e. v_+ L^2 > q (d - 2).

Enclosures on a q-ball Q use the mean value form  F(Q) subset F(q_mid) + F'(Q) (Q - q_mid),
    F'(q) = sum_k coef_k exp(-v e_k Lam(w)) (-e_k) [ v'(q) Lam(w) + v s_k Lam'(w) ],  w = q s_k,  v'(q) = -rho/L^2,
with Lam, Lam' enclosed on [w_lo, w_hi] by monotonicity (both are increasing on [0,1)); adaptive bisection in q.
Output: data/26_start_small_N.jsonl (one line per N; resumable).   Usage: 26_start_small_N.py NMIN NMAX [K]
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import os, sys, json, time
sys.path.insert(0, os.path.dirname(__file__))
import core, start
from core import arb, interval
from start import Gxi, c_enclosure

core.set_prec(160)
PI = arb.pi()
NMIN = int(sys.argv[1]) if len(sys.argv) > 1 else 3
NMAX = int(sys.argv[2]) if len(sys.argv) > 2 else 100
KSTR = sys.argv[3] if len(sys.argv) > 3 else "3.5"
KK = arb(KSTR)
OUT = os.path.join(_R, 'data', 'msc_rigorous_1d', f"26_start_small_N_K{KSTR.replace('.', 'p')}.jsonl")
HALF = arb(1) / 2


def modes(N, x0):
    L = arb(2 * N - 1) / 2
    xi = arb(2 * x0 - 1) / (2 * N - 1)
    out = []
    for m in range(1, N):
        k = 2 * m - 1
        y = k * PI / 4
        x = y / L
        sx = x.sin()
        S = sx / x
        sg = 1 if m % 2 == 1 else -1
        out.append((sg * (2 * y * xi).cos() * 2 * y * (-2 * y * y) * S ** 3 * x.cos(), 2 * sx * sx * L * L, 2 * sx * sx))
    return out


def lam_pt(w):
    """(Lam(w), Lam'(w)) at a point (tiny ball) 0 <= w < 1."""
    if w < arb("0.01"):
        l0, l1, _ = core.Lam_derivs(w)
        return l0, l1
    lg = (1 - w).log()
    return -lg / w, (w / (1 - w) + lg) / (w * w)


def lam_hull(w):
    lo = arb(w.lower()); hi = arb(w.upper())
    if lo < 0:
        lo = arb(0)
    a0, a1 = lam_pt(lo)
    b0, b1 = lam_pt(hi)
    return a0.union(b0), a1.union(b1)


def F_and_dF(md, c, kap, rho, eps, sgnK, q, deriv):
    v = c - (kap + q * rho) * eps + sgnK * KK * eps * eps
    s = arb(0)
    if not deriv:
        for ck, e2, s2 in md:
            l0, _ = lam_hull(q * s2)
            s += ck * (-v * e2 * l0).exp()
        return s, v
    dv = -rho * eps
    for ck, e2, s2 in md:
        l0, l1 = lam_hull(q * s2)
        s += ck * (-v * e2 * l0).exp() * (-e2) * (dv * l0 + v * s2 * l1)
    return s, v


def certify_q(md, c, kap, rho, eps, L, dsteps, a, b, st, depth=0):
    """both signs for all q in [a, b]."""
    Q = interval(a, b)
    qm = (arb(a) + arb(b)) / 2
    st["evals"] += 1
    ok = True
    for sgnK, want in ((-1, +1), (+1, -1)):
        Fm, _ = F_and_dF(md, c, kap, rho, eps, sgnK, qm, False)
        dF, vQ = F_and_dF(md, c, kap, rho, eps, sgnK, Q, True)
        val = Fm + dF * (Q - qm)
        if want > 0:
            good = bool(val > 0)
            if not good:
                # alternative: t_- <= d - 1 for all q in Q (then t_- < t_0 trivially); only for q > 0
                good = bool(arb(a) > 0) and bool(vQ * L * L < arb(a) * (dsteps - 2))
        else:
            good = bool(val < 0) and bool(arb(vQ.lower()) * L * L > arb(b) * (dsteps - 2))
        ok = ok and good
        if not ok:
            break
    if ok:
        return True
    if depth > 40:
        st["fail"].append((float(arb(a).mid()), float(arb(b).mid())))
        return False
    mid = (arb(a) + arb(b)) / 2
    return certify_q(md, c, kap, rho, eps, L, dsteps, a, mid, st, depth + 1) and certify_q(md, c, kap, rho, eps, L, dsteps, mid, b, st, depth + 1)


done = set()
if os.path.exists(OUT):
    for line in open(OUT):
        try:
            done.add(json.loads(line)["N"])
        except Exception:
            pass
for N in range(NMIN, NMAX + 1):
    if N in done:
        continue
    t0 = time.time()
    L = arb(2 * N - 1) / 2
    eps = 1 / (L * L)
    okN = True
    evals = 0
    fails = []
    worst = None
    x0max = (2 * N + 1) // 4
    for x0 in range(1, x0max + 1):
        xi = arb(2 * x0 - 1) / (2 * N - 1)
        C, cm = c_enclosure(xi)
        G2, G3 = Gxi(2, C, xi), Gxi(3, C, xi)
        assert G2 < 0
        kap = HALF + C / 6 * G3 / G2
        rho = -C / 2 * G3 / G2
        md = modes(N, x0)
        st = dict(evals=0, fail=[])
        ok = certify_q(md, C, kap, rho, eps, L, N - x0, arb(0), HALF, st)
        evals += st["evals"]
        okN &= ok
        if not ok:
            fails.append((x0, st["fail"][:3]))
    rec = dict(N=N, K=KSTR, n_starts=x0max, certified=bool(okN), evals=evals, fails=fails[:5], secs=round(time.time() - t0, 1))
    with open(OUT, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(rec, flush=True)
