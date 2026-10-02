"""Brute-force sanity check of Theorems 5.1 and 7.8 for small N (dense eigen-decomposition of the killed generator):
 continuous time:  g'(tau_cf + c_plus/X) < 0 < g'(tau_cf - c_minus/X)  with the certified block constants;
 discrete time (q = 1/2, 1/4):  (Dg)(t_plus) < 0 < (Dg)(t_minus),  t_plus = ceil((tau_cf + c_plus/X)/mu),  t_minus = floor((tau_cf - c_minus/X)/mu),
 and the exact maximiser of the PMF lies in (t_cf - c_minus/(X mu), t_cf + c_plus/(X mu) + 2)."""
import json, math
import numpy as np
from r4lib import scalars, mu_real, DATA

def L1(d, N):
    n = N ** d
    idx = np.arange(n).reshape((N,) * d)
    L = np.zeros((n, n)); rate = 1.0 / (2 * d)
    for ax in range(d):
        a = np.moveaxis(idx, ax, 0)
        src = a[:-1].ravel(); dst = a[1:].ravel()
        L[src, src] += rate; L[dst, dst] += rate; L[src, dst] -= rate; L[dst, src] -= rate
    return L

certC = {d: [json.loads(l) for l in open(DATA + '/52_cert_theoremB_d%d.jsonl' % d)] for d in (2, 3)}
certD = {d: [json.loads(l) for l in open(DATA + '/72_cert_discrete_d%d.jsonl' % d)] for d in (2, 3)}
blk = lambda cert, d, N: [b for b in cert[d] if b['Na'] <= N and (b['Nb'] is None or N <= b['Nb'])][0]
out = []; ok_all = True
for d, N in [(2, 12), (2, 16), (2, 24), (2, 36), (2, 48), (3, 10), (3, 12), (3, 14)]:
    s = scalars(d, N)
    n = N ** d
    mu1 = mu_real(d, N, 1.0)
    La = L1(d, N)[:n - 1, :n - 1] / mu1
    lam, V = np.linalg.eigh(La)
    a = lam * V[0, :] * (V.T @ np.ones(n - 1))
    g1 = lambda t: float(-np.sum(a * lam * np.exp(-lam * t)))
    b = blk(certC, d, N)
    tp = s['tau_cf'] + b['c_plus'] / s['X']; tm = s['tau_cf'] - b['c_minus'] / s['X']
    rec = dict(d=d, N=N, cont=dict(c_minus=b['c_minus'], c_plus=b['c_plus'], gprime_minus=g1(tm), gprime_plus=g1(tp), ok=bool(g1(tm) > 0 > g1(tp))))
    ok_all &= rec['cont']['ok']
    if not (d == 2 and N < 11):
        bD = blk(certD, d, N)
        for q in (0.5, 0.25):
            mu = mu_real(d, N, q)
            G = lambda t: float(-np.sum(a * lam * (1 - mu * lam) ** t))
            gD = lambda t: float(np.sum(a * (1 - mu * lam) ** t))
            t_plus = math.ceil((s['tau_cf'] + bD['c_plus'] / s['X']) / mu); t_minus = math.floor((s['tau_cf'] - bD['c_minus'] / s['X']) / mu)
            ts = np.arange(max(1, t_minus - 5), t_plus + 6)
            tstar = int(ts[np.argmax([gD(t) for t in ts])]) + 1          # maximiser of P(T = t) = mu g(t-1), searched near the window
            lo = (s['tau_cf'] - bD['c_minus'] / s['X']) / mu; hi = (s['tau_cf'] + bD['c_plus'] / s['X']) / mu + 2
            okd = bool(G(t_minus) > 0 > G(t_plus)) and bool(lo < tstar < hi)
            rec['disc_q%s' % q] = dict(t_minus=t_minus, t_plus=t_plus, G_minus=G(t_minus), G_plus=G(t_plus), t_star=tstar, ok=okd)
            ok_all &= okd
    out.append(rec); print(json.dumps(rec), flush=True)
print('ALL OK' if ok_all else 'FAIL')
json.dump(out, open(DATA + '/54_bruteforce_theorem51.json', 'w'), indent=1)
