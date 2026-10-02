"""Validation (not part of the proof) of Theorem 7.17 (window uniform in q < 1): brute-force first-passage PMF by direct
iteration of the lazy walk (float64), for (d, N, q) with q at (just below) the largest value allowed by sin(pi/N) <= (1-q)/q,
i.e. cases NOT covered by the q_max tables of Theorem 7.14 (two of them with q > 0.99), plus two cases with q <= 1/2.
Global maximiser: the iteration stops at the first t1 >= argmax for which a tail bound shows f(t) < max f for all t > t1:
  f(t) = <Q^{t-1} delta_{x0}, b> = c0 lam0^{t-1} + R(t),  c0 > 0 (Perron), |R(t)| <= lam_*^{t-1-k} ||p_k||_2 ||b||_2 for t-1 >= k,
  lam_* = max(1 - mu, 2 q cos^2 x - 1) (Cauchy interlacing: Q is P with the target row/column removed), ||b||_2 = (q/2d) sqrt(d);
  hence f(t) <= f(t1) + 2 lam_*^{t1-1-k} ||p_k||_2 ||b||_2 for all t >= t1 (k <= t1 - 1).  (Float64; rounding not controlled.)
Window: (t_cf - K^-/(X mu), t_cf + K^+/(X mu) + 2) with K^pm from data/80_tables.json (entry T717 with the largest N0 <= N).
Output: data/79_validate_allq.json (data/79_validate_allq_small.json with argument 'small')."""
import json, math, sys, time, os
import numpy as np
from mpmath import mp, mpf, sin, pi
from r4lib import scalars, mu_real, DATA

mp.dps = 30
TAB = json.load(open(os.path.join(DATA, '80_tables.json')))


def qmax_H3(N):
    """largest q with sin(pi/N) <= (1-q)/q, rounded DOWN to 6 digits (so (H3) holds exactly)."""
    q = mpf(1) / (1 + sin(pi / N))
    qq = math.floor(float(q) * 10 ** 6) / 10 ** 6
    assert sin(pi / N) <= (1 - mpf(qq)) / mpf(qq)
    return qq


def K_for(d, N):
    tab = TAB['T717_d%d' % d]['table']
    ent = [t for t in tab if t[0] <= N]
    assert ent, (d, N)
    N0, km, kp, _ = ent[-1]
    return N0, km, kp


def pmf(d, N, q, tmax):
    """PMF until the tail bound certifies the global maximiser (or tmax); returns f, t_stop, certified flag."""
    p = np.zeros((N,) * d); p[(0,) * d] = 1.0
    tgt = (N - 1,) * d
    a = q / (2 * d)
    mu = mu_real(d, N, q)
    lam = max(1 - mu, 2 * q * math.cos(math.pi / (2 * N)) ** 2 - 1)
    bnorm = a * math.sqrt(d)
    f = np.zeros(tmax + 1)
    moved = np.empty_like(p)
    norms = []                                   # (k, ||p_k||_2)
    stride = max(1, int(0.05 / mu))              # record ||p_k|| every 0.05 units of scaled time
    fmax, tstar = -1.0, 0
    for t in range(1, tmax + 1):
        if (t - 1) % stride == 0:
            norms.append((t - 1, float(np.sqrt(np.sum(p * p)))))
        new = (1 - q) * p
        for ax in range(d):
            moved.fill(0.0)
            src = [slice(None)] * d; dst = [slice(None)] * d; bnd = [slice(None)] * d
            src[ax] = slice(0, N - 1); dst[ax] = slice(1, N); moved[tuple(dst)] += p[tuple(src)]
            bnd[ax] = slice(N - 1, N); moved[tuple(bnd)] += p[tuple(bnd)]          # cancelled move at the upper wall
            src[ax] = slice(1, N); dst[ax] = slice(0, N - 1); moved[tuple(dst)] += p[tuple(src)]
            bnd[ax] = slice(0, 1); moved[tuple(bnd)] += p[tuple(bnd)]              # cancelled move at the lower wall
            new += a * moved
        f[t] = new[tgt]; new[tgt] = 0.0
        p = new
        if f[t] > fmax:
            fmax, tstar = f[t], t
        elif t % stride == 0:
            tail = min(2 * lam ** (t - 1 - k) * nk * bnorm for k, nk in norms if k <= t - 1)
            if f[t] + tail < fmax:
                return f[:t + 1], t, True
    return f, tmax, False


cases = [(3, 12, None), (3, 20, None), (3, 40, None), (3, 15, 0.5), (2, 59, None), (2, 60, None), (2, 100, None),
         (2, 80, 0.3), (2, 200, None), (2, 320, None)]
if len(sys.argv) > 1 and sys.argv[1] == 'small':
    cases = [c for c in cases if c[1] <= 100]
out = []
t0 = time.time()
for d, N, q in cases:
    q = qmax_H3(N) if q is None else q
    s = scalars(d, N); mu = mu_real(d, N, q)
    X, tcf = s['X'], s['tau_cf'] / mu
    N0, km, kp = K_for(d, N)
    lo, hi = tcf - km / (X * mu), tcf + kp / (X * mu) + 2
    f, horizon, cert = pmf(d, N, q, int(math.ceil(4.0 * tcf)))
    ts = int(np.argmax(f))
    rec = dict(d=d, N=N, q=q, X=X, t_cf=tcf, N0=N0, K_minus=km, K_plus=kp, window=[lo, hi], t_star=ts,
               inside=bool(lo < ts < hi), theta=X * (mu * ts - s['tau_cf']), horizon=horizon, tail_certified=cert,
               tau_stop=mu * horizon,
               f_end_over_fmax=float(f[horizon] / f[ts]), mass_to_horizon=float(f.sum()), secs=round(time.time() - t0, 1))
    out.append(rec); print(json.dumps(rec), flush=True)
    json.dump(out, open(os.path.join(DATA, '79_validate_allq%s.json' % ('_small' if len(sys.argv) > 1 else '')), 'w'), indent=1)
print(('ALL INSIDE' if all(r['inside'] for r in out) else 'SOME OUTSIDE') + (', ALL TAILS CERTIFIED' if all(r['tail_certified'] for r in out) else ', SOME TAILS NOT CERTIFIED'), flush=True)
