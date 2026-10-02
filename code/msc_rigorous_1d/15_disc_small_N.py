"""Proposition 6.2 (computer-assisted): for 3 <= N <= NMAX and every q in (0, 4/5],

   (A) f(t+1) > f(t)  for the integer t = floor(tau - E)   (if t >= N-1; for t <= N-2 nothing is needed),
   (B) f(t+1) < f(t)  for the integer t = ceil(tau + E),
   tau = (c L^2 - kappa)/q + 1 - rho,  E = Kd/(q L^2),  Kd = 1.78.

Part 1 (0 < q <= 1/2; all eigenvalues positive, Theorem 3.5: the real-variable increment Psi has exactly one
  zero t0 > N-2, positive before, negative after): certify, for q in a ball,
     [ frakD(v_-(q); q) > 0  or  tau - E <= N-2 ]   and   [ frakD(v_+(q); q) < 0 and tau + E > N-2 ],
  v_-+(q) = c - (kappa + q rho)/L^2 -+ Kd/L^4,
  frakD(v; q) = sum_k sigma_k 2 y_k (-2 y_k^2) A_1(zeta_k) exp(-2 v y_k^2 beta_k Lam(2 q zeta_k beta_k)).
  Adaptive bisection in q.
Part 2 (1/2 <= q <= 4/5; integer t): for each integer t >= N-1 and
     q in I_t^- = [q^-_{t+1}, q^-_t] cap [1/2, 4/5],  q^-_t = (c L^2 - kappa - Kd/L^2)/(t - 1 + rho):
        certify  d/dq frakD_q(t) < 0 on I_t^-  and  frakD_q(t) > 0 at the right end point;
     q in I_t^+ = [q^+_t, q^+_{t-1}] cap [1/2, 4/5],  q^+_t = (c L^2 - kappa + Kd/L^2)/(t - 1 + rho):
        certify  d/dq frakD_q(t) < 0 on I_t^+  and  frakD_q(t) < 0 at the left end point.
     (fallback: direct enclosure with bisection).
  frakD_q(t) = sum_k sigma_k 2 y_k (-2 y_k^2) A_1(zeta_k) lambda_k^(t-1),  lambda_k = 1 - 2 q sin^2 x_k.
Output: data/15_disc_small_N.jsonl (one line per N; resumable).
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import os, sys, json, time, math
sys.path.insert(0, os.path.dirname(__file__))
import core
from core import arb, G, interval, PI

core.set_prec(128)
const = json.load(open(_os.path.join(_R, 'data', 'msc_rigorous_1d', '03_constants.json')))
c = arb(const["c_mid_80_digits"][:40], "1e-36")
G2c, G3c = G(2, c), G(3, c)
kappa = arb(3) / 4 + c / 6 * G3c / G2c
rho = -c / 2 * G3c / G2c
KD = arb("1.78")
NMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 40
OUT = _os.path.join(_R, 'data', 'msc_rigorous_1d', '15_disc_small_N.jsonl')
HALF = arb(1) / 2
Q45 = arb(4) / 5


def modes(N):
    """per lattice mode: (coefficient c_k, 2 y^2 beta = L^2 * 2 sin^2 x, 2 sin^2 x)"""
    L = arb(2 * N - 1) / 2
    out = []
    for m in range(1, N):
        k = 2 * m - 1
        y = k * PI / 4
        x = y / L
        sx = x.sin()
        S = sx / x
        A1 = S ** 3 * x.cos() ** 2
        sg = 1 if m % 2 == 1 else -1
        out.append((sg * 2 * y * (-2 * y * y) * A1, 2 * sx * sx * L * L, 2 * sx * sx))
    return out


def Lam_point(w):
    """Lam(w) = -ln(1-w)/w for a point (or tiny ball) w >= 0; 1 <= Lam(w) <= 1/(1-w)."""
    if w.abs_upper() < 1e-30:
        return arb(1).union(arb(1) + arb("2e-30"))
    return -(1 - w).log() / w


def Lam_hull(w):
    lo, hi = arb(w.lower()), arb(w.upper())
    if lo < 0:
        lo = arb(0)
    return Lam_point(lo).union(Lam_point(hi))


def frakD_scaled(md, v, q):
    s = arb(0)
    for ck, e2, s2 in md:
        s += ck * (-v * e2 * Lam_hull(q * s2)).exp()
    return s


def part1(N, md, a, b, depth=0, stats=None):
    """certify the two conditions for all q in [a, b] (0 <= a < b <= 1/2)."""
    L = arb(2 * N - 1) / 2
    q = interval(a, b)
    vm = c - (kappa + q * rho) / L ** 2 - KD / L ** 4
    vp = c - (kappa + q * rho) / L ** 2 + KD / L ** 4
    okL = frakD_scaled(md, vm, q) > 0
    if not okL and a > 0:
        tauE = (c * L * L - kappa - KD / L ** 2) / q + 1 - rho
        okL = tauE < N - 2
    okR = frakD_scaled(md, vp, q) < 0
    if okR:
        if a > 0:
            okR = ((c * L * L - kappa + KD / L ** 2) / q + 1 - rho) > N - 2
        else:
            okR = ((c * L * L - kappa + KD / L ** 2) / arb(b) + 1 - rho) > N - 2     # tau+E decreasing in q
    stats["evals"] += 1
    if okL and okR:
        return True
    if depth > 40:
        stats["fail"].append((float(a), float(b)))
        return False
    mid = (arb(a) + arb(b)) / 2
    return part1(N, md, a, mid, depth + 1, stats) and part1(N, md, mid, b, depth + 1, stats)


def frakD_int(md, t, q, deriv=False):
    s = arb(0)
    for ck, e2, s2 in md:
        lam = 1 - q * s2
        if deriv:
            s += ck * (t - 1) * lam ** (t - 2) * (-s2)
        else:
            s += ck * lam ** (t - 1)
    return s


def sign_on(md, t, lo, hi, want, stats, depth=0):
    """certify sign(frakD_q(t)) == want for all q in [lo, hi] (monotone trick, else bisection)."""
    q = interval(lo, hi)
    stats["evals"] += 1
    dd = frakD_int(md, t, q, deriv=True)
    if dd < 0:
        end = hi if want > 0 else lo          # decreasing in q: min at hi, max at lo
        val = frakD_int(md, t, arb(end))
        if (want > 0 and val > 0) or (want < 0 and val < 0):
            return True
        stats["fail"].append((t, float(arb(lo).mid()), float(arb(hi).mid()), "endpoint"))
        return False
    val = frakD_int(md, t, q)
    if (want > 0 and val > 0) or (want < 0 and val < 0):
        return True
    if depth > 30:
        stats["fail"].append((t, float(arb(lo).mid()), float(arb(hi).mid()), "depth"))
        return False
    mid = (arb(lo) + arb(hi)) / 2
    return sign_on(md, t, lo, mid, want, stats, depth + 1) and sign_on(md, t, mid, hi, want, stats, depth + 1)


def clip(lo, hi):
    """[lo,hi] cap [1/2, 4/5] as point balls (using upper/lower bounds outward), or None"""
    lo2 = lo if lo > HALF else (HALF if lo < HALF else arb(lo.lower()))
    hi2 = hi if hi < Q45 else (Q45 if hi > Q45 else arb(hi.upper()))
    if lo < HALF and not (lo > HALF):
        lo2 = HALF
    if hi2 < lo2:
        return None
    return arb(lo2.lower()), arb(hi2.upper())


def part2(N, md, stats):
    L = arb(2 * N - 1) / 2
    numm = c * L * L - kappa - KD / L ** 2
    nump = c * L * L - kappa + KD / L ** 2
    ok = True
    # t range: q in [1/2, 4/5]  <=>  t - 1 + rho in [num/0.8, num/0.5]
    tlo = int(math.floor(float((numm / Q45 - rho + 1).lower()))) - 2
    thi = int(math.ceil(float((nump / HALF - rho + 1).upper()))) + 2
    nt = 0
    for t in range(max(tlo, 1), thi + 1):
        # (A)  q in [q^-_{t+1}, q^-_t]
        if t >= N - 1:
            iv = clip(numm / (t + rho), numm / (t - 1 + rho))
            if iv is not None:
                nt += 1
                if t == N - 1:
                    # first increment: f(N) - f(N-1) = (q/2)^(N-1) (h_1 - 1) > 0  iff  q < 1 - 1/(2N-3)  (Prop. 3.9(2))
                    thr = 1 - arb(1) / (2 * N - 3)
                    if iv[1] < thr:
                        stats["notes"].append((t, "first increment positive analytically"))
                    elif N == 4 and not (iv[1] > Q45):
                        # N = 4: equality exactly at q = 4/5, where the mode is {5} (Prop. 3.9(4)); strict for q < 4/5
                        stats["notes"].append((t, "N=4: f(4) >= f(3) on [%.6f, 4/5], equality only at q = 4/5" % float(iv[0].mid())))
                    else:
                        ok &= sign_on(md, t, iv[0], iv[1], +1, stats)
                else:
                    ok &= sign_on(md, t, iv[0], iv[1], +1, stats)
        # (B)  q in [q^+_t, q^+_{t-1}]
        if t - 2 + float(rho.mid()) > 0:
            iv = clip(nump / (t - 1 + rho), nump / (t - 2 + rho))
        else:
            iv = clip(nump / (t - 1 + rho), arb(10))
        if iv is not None:
            if t < N - 1:
                stats["fail"].append((t, "B-window below support"))
                ok = False
            else:
                nt += 1
                ok &= sign_on(md, t, iv[0], iv[1], -1, stats)
    stats["nt"] = nt
    return ok


done = set()
if os.path.exists(OUT):
    for line in open(OUT):
        try:
            done.add(json.loads(line)["N"])
        except Exception:
            pass
for N in range(3, NMAX + 1):
    if N in done:
        continue
    t0 = time.time()
    md = modes(N)
    st1 = dict(evals=0, fail=[])
    ok1 = part1(N, md, arb(0), HALF, 0, st1)
    st2 = dict(evals=0, fail=[], notes=[])
    ok2 = part2(N, md, st2)
    rec = dict(N=N, Kd=1.78, part1_ok=bool(ok1), part1_evals=st1["evals"], part1_fail=st1["fail"][:5],
               part2_ok=bool(ok2), part2_windows=st2["nt"], part2_evals=st2["evals"], part2_fail=st2["fail"][:5], part2_notes=st2["notes"],
               secs=round(time.time() - t0, 1))
    with open(OUT, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(rec, flush=True)
