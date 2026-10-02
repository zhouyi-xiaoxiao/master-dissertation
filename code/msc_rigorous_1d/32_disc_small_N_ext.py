"""Proposition 6.2, extension to 4/5 <= q <= QHI (computer-assisted; same method as Part 2 of script 15):
for NMIN <= N <= NMAX and every q in [QLO, QHI] certify, with tau = (c L^2 - kappa)/q + 1 - rho, E = Kd/(q L^2), Kd = 1.78,

   (A) f(t+1) > f(t)  for the integer t = floor(tau - E)   (if t >= N-1; for t <= N-2 nothing is needed),
   (B) f(t+1) < f(t)  for the integer t = ceil(tau + E)    (and t >= N-1).

For each integer t >= N-1 and q in I_t^- = [q^-_{t+1}, q^-_t] cap [QLO, QHI],  q^-_t = (c L^2 - kappa - Kd/L^2)/(t - 1 + rho):
   certify  d/dq frakD_q(t) < 0 on I_t^-  and  frakD_q(t) > 0 at the right end point (else: bisection in q);
for q in I_t^+ = [q^+_t, q^+_{t-1}] cap [QLO, QHI],  q^+_t = (c L^2 - kappa + Kd/L^2)/(t - 1 + rho):
   certify  d/dq frakD_q(t) < 0 on I_t^+  and  frakD_q(t) < 0 at the left end point.
frakD_q(t) = sum_k sigma_k 2 y_k (-2 y_k^2) A_1(zeta_k) lambda_k^(t-1),  lambda_k = 1 - 2 q sin^2 x_k  (integer t; any sign of lambda).

Usage: 32_disc_small_N_ext.py NMIN NMAX QLO(num/den) QHI(num/den)        Output: data/32_disc_small_N_ext.jsonl (appended)
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
NMIN = int(sys.argv[1]); NMAX = int(sys.argv[2])
_a, _b = map(int, sys.argv[3].split("/")); QLO = arb(_a) / _b
_c, _d = map(int, sys.argv[4].split("/")); QHI = arb(_c) / _d
OUT = _os.path.join(_R, 'data', 'msc_rigorous_1d', '32_disc_small_N_ext.jsonl')


def modes(N):
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
        out.append((sg * 2 * y * (-2 * y * y) * A1, 2 * sx * sx))
    return out


def frakD_int(md, t, q, deriv=False):
    s = arb(0)
    for ck, s2 in md:
        lam = 1 - q * s2
        if deriv:
            s += ck * (t - 1) * lam ** (t - 2) * (-s2)
        else:
            s += ck * lam ** (t - 1)
    return s


def sign_on(md, t, lo, hi, want, stats, depth=0):
    q = interval(lo, hi)
    stats["evals"] += 1
    dd = frakD_int(md, t, q, deriv=True)
    if dd < 0:
        end = hi if want > 0 else lo
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
    """[lo,hi] cap [QLO, QHI] as point balls (outward), or None if certainly empty"""
    if hi < QLO or lo > QHI:
        return None
    lo2 = QLO if not (lo > QLO) else lo
    hi2 = QHI if not (hi < QHI) else hi
    return arb(lo2.lower()), arb(hi2.upper())


for N in range(NMIN, NMAX + 1):
    t0 = time.time()
    md = modes(N)
    L = arb(2 * N - 1) / 2
    numm = c * L * L - kappa - KD / L ** 2
    nump = c * L * L - kappa + KD / L ** 2
    st = dict(evals=0, fail=[], notes=[])
    ok = True
    tlo = int(math.floor(float((numm / QHI - rho + 1).lower()))) - 2
    thi = int(math.ceil(float((nump / QLO - rho + 1).upper()))) + 2
    nt = 0
    for t in range(max(tlo, 1), thi + 1):
        if t >= N - 1:
            iv = clip(numm / (t + rho), numm / (t - 1 + rho))
            if iv is not None:
                nt += 1
                if t == N - 1:
                    thr = 1 - arb(1) / (2 * N - 3)      # f(N) > f(N-1) iff q < 1 - 1/(2N-3)  (Prop. 3.9(2))
                    if iv[1] < thr:
                        st["notes"].append((t, "first increment positive analytically"))
                    else:
                        st["fail"].append((t, "window (A) at t = N-1 with q >= 1 - 1/(2N-3)")); ok = False
                else:
                    ok &= sign_on(md, t, iv[0], iv[1], +1, st)
        if t - 2 + float(rho.mid()) > 0:
            iv = clip(nump / (t - 1 + rho), nump / (t - 2 + rho))
        else:
            iv = clip(nump / (t - 1 + rho), arb(10))
        if iv is not None:
            if t < N - 1:
                st["fail"].append((t, "B-window below support")); ok = False
            else:
                nt += 1
                ok &= sign_on(md, t, iv[0], iv[1], -1, st)
    rec = dict(N=N, Kd=1.78, qlo=sys.argv[3], qhi=sys.argv[4], ok=bool(ok), windows=nt, evals=st["evals"], fail=st["fail"][:5], notes=st["notes"],
               secs=round(time.time() - t0, 1))
    with open(OUT, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(rec, flush=True)
