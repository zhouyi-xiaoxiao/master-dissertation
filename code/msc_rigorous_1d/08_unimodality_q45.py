"""Computer-assisted certification (Proposition 3.18): for q = 4/5 (and hence, by Lemma 3.8, for all
q <= 4/5) the first-passage PMF from the reflecting end is unimodal, for 2 <= N <= NMAX; exact modes.

Method
  (1) float64 forward stepping r_t = (Q^t)_{1,N-1}  (f(t+1) = (q/2) r_t).  All quantities are >= 0, every
      entry is a sum of <= 3 products of nonnegative numbers, so (no underflow, round-to-nearest) the
      computed value satisfies  r~_t = r_t (1+theta),  |theta| <= Theta_t := (1+u)^(4t) - 1,  u = 2^-53
      (rounded constant, one multiplication, two additions per term and step).  We use Theta_t = 5 t u.
  (2) sign of f(t+1)-f(t) is certified whenever the float gap exceeds the error bars; the remaining
      (few) indices are resolved in ball arithmetic from the spectral formula, and exact ties by exact
      rational arithmetic.
  (3) tail: for t >= t_end the increment is negative because the m=1 spectral term dominates:
      (rho2/lam1)^(t-1) < c_1 nu_1 / sum_{m>=2} |c_m| nu_m   (ball arithmetic).
Output: data/08_modes_q45.jsonl (one line per N; resumable).
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import os, sys, json, math, time
from fractions import Fraction
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from flint import arb, ctx

ctx.prec = 200
PI = arb.pi()
NMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 300
OUT = _os.path.join(_R, 'data', 'msc_rigorous_1d', '08_modes_q45.jsonl')
U = 2.0 ** -53
q_arb = arb(4) / 5


def spectral_data(N):
    lam, cnu = [], []
    for m in range(1, N):
        th = (2 * m - 1) * PI / (2 * N - 1)
        nu = q_arb * (1 - th.cos())
        lam.append(1 - nu)
        cnu.append(th.sin() * ((N - 1) * th).sin() * nu)
    return lam, cnu


def t_end(N, lam, cnu):
    if N == 2:
        return 2
    rest = arb(0)
    rho2 = arb(0)
    for l, cn in zip(lam[1:], cnu[1:]):
        rest += arb(cn.abs_upper())
        a = arb(l.abs_upper())
        if a > rho2:
            rho2 = a
        elif not (a < rho2):
            rho2 = arb(max(float(a.upper()), float(rho2.upper())) * (1 + 1e-15))
    assert cnu[0] > 0 and lam[0] > rho2
    ratio = arb(cnu[0].lower()) / rest              # need (rho2/lam1)^(t-1) < ratio
    if ratio > 1:
        return 2
    base = rho2 / arb(lam[0].lower())
    T = (ratio.log() / base.log())                  # both logs negative; t-1 > T suffices
    return int(math.ceil(float(T.upper()))) + 2


def increment_ball(N, t, lam, cnu):
    """ball for (f(t+1)-f(t)) * (2N-1)/(2q) = -sum cnu_m lam_m^(t-1)."""
    s = arb(0)
    for l, cn in zip(lam, cnu):
        s -= cn * l ** (t - 1)
    return s


def exact_increment_sign(N, t):
    q = Fraction(4, 5)
    n = N - 1
    v = [Fraction(0)] * n
    v[0] = Fraction(1)
    a, b = 1 - q, q / 2
    f = []
    for s in range(1, t + 2):
        f.append(b * v[n - 1])
        w = [a * x for x in v]
        for j in range(1, n):
            w[j] += b * v[j - 1]
        for j in range(0, n - 1):
            w[j] += b * v[j + 1]
        w[0] += b * v[0]
        v = w
    d = f[t] - f[t - 1]       # f(t+1) - f(t)
    return (d > 0) - (d < 0)


done = set()
if os.path.exists(OUT):
    for line in open(OUT):
        try:
            done.add(json.loads(line)["N"])
        except Exception:
            pass

a, b = 0.2, 0.4
for N in range(2, NMAX + 1):
    if N in done:
        continue
    t0 = time.time()
    n = N - 1
    lam, cnu = spectral_data(N)
    te = t_end(N, lam, cnu)
    steps = te + 2
    v = np.zeros(n)
    v[0] = 1.0
    r = np.empty(steps + 1)
    r[0] = v[n - 1]
    for t in range(1, steps + 1):
        w = a * v
        if n > 1:
            w[1:] += b * v[:-1]
            w[:-1] += b * v[1:]
        w[0] += b * v[0]
        v = w
        r[t] = v[n - 1]
    assert r[n - 1] > 1e-300, "underflow risk"
    # f(t) = b r[t-1];  increment at time t: f(t+1)-f(t) ~ r[t]-r[t-1], t >= 1
    tt = np.arange(0, steps + 1)
    Th = 5.0 * tt * U
    lo = r * (1 - Th)
    hi = r * (1 + Th)
    t_idx = np.arange(1, steps + 1)
    pos = lo[1:] > hi[:-1]          # certified f(t+1) > f(t)
    neg = hi[1:] < lo[:-1]          # certified f(t+1) < f(t)
    zero = (r[1:] == 0) & (r[:-1] == 0)   # both exactly zero (t < N-1): f(t)=f(t+1)=0
    unc = ~(pos | neg | zero)
    signs = np.where(pos, 1, np.where(neg, -1, 0)).astype(int)
    resolved = []
    for t in t_idx[unc]:
        t = int(t)
        ball = increment_ball(N, t, lam, cnu)
        if ball > 0:
            s = 1
        elif ball < 0:
            s = -1
        else:
            s = exact_increment_sign(N, t) if t <= 400 else None
        resolved.append((t, s))
        signs[t - 1] = 2 if s is None else s
    # unimodality on t >= N-1 (f = 0 before): increments at t = N-1, ..., steps
    seq = signs[n - 1:]            # increments for t = n, n+1, ...   (f(n) > 0 is the first nonzero value)
    ok = True
    if (seq == 2).any():
        ok = False
        modes = []
    else:
        # weak unimodality: after the first strictly negative increment there is no positive one
        negs = np.where(seq == -1)[0]
        assert len(negs) > 0
        i_neg = int(negs[0])
        ok = not bool((seq[i_neg:] == 1).any())
        # tail certified by t_end: increments for t >= te are negative (analytic); consistency check
        ok = ok and bool(np.all(seq[te - n:] <= 0))
        # maximisers: f is nondecreasing up to t = n + i_neg, strictly decreasing right after
        modes = [n + i_neg]
        j = i_neg - 1
        while j >= 0 and seq[j] == 0:
            modes.insert(0, n + j)
            j -= 1
    rec = dict(N=N, unimodal=ok, modes=modes, t_end=te, n_uncertain=int(unc.sum()),
               resolved=[(t, s) for t, s in resolved], secs=round(time.time() - t0, 2))
    with open(OUT, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    if N <= 12 or N % 25 == 0:
        print(rec, flush=True)
print("done up to", NMAX)
