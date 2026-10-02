"""Validation (not part of the proof) of Theorem 7.14:
 (1) exact discrete modes of the floating-point study for q in {0.8, 0.9} lie in the certified windows (q_max = 0.8 resp. 0.9);
 (2) brute force (dense eigen-decomposition, small N, q = 0.8): the four region claims of Theorem 7.13 hold with the certified constants:
     Dg > 0 on [t_E, t_-], Dg < 0 on [t_+, t_max], g(t) < g(t_-) for t < t_E."""
import json, math
import numpy as np
from r4lib import scalars, exact_modes_D, mu_real, DATA

def load(tag, d):
    return [json.loads(l) for l in open(DATA + '/76_cert_q%s_d%d.jsonl' % (tag, d))]
cert = {('0.8', d): load('0p8', d) for d in (2, 3)}
cert.update({('0.9', d): load('0p9', d) for d in (2, 3)})
blk = lambda key, N: [b for b in cert[key] if b['Na'] <= N and (b['Nb'] is None or N <= b['Nb'])]
out = []; ok_all = True
mD = exact_modes_D()
for (d, N, q), (mode, mfpt) in sorted(mD.items()):
    if d == 1 or q not in (0.8, 0.9) or (d == 2 and N > 1500) or (d == 3 and N > 160):
        continue
    bb = blk(('%g' % q, d), N)
    if not bb or not bb[0]['ok']:
        continue
    b = bb[0]
    s = scalars(d, N); mu = mu_real(d, N, q)
    lo = (s['tau_cf'] - b['c_minus'] / s['X']) / mu; hi = (s['tau_cf'] + b['c_plus'] / s['X']) / mu + 2
    rec = dict(kind='float_study', d=d, N=N, q=q, mode=mode, lo=lo, hi=hi, inside=bool(lo < mode < hi))
    ok_all &= rec['inside']; out.append(rec); print(json.dumps(rec), flush=True)

def L1(d, N):
    n = N ** d
    idx = np.arange(n).reshape((N,) * d)
    L = np.zeros((n, n)); rate = 1.0 / (2 * d)
    for ax in range(d):
        a = np.moveaxis(idx, ax, 0)
        src = a[:-1].ravel(); dst = a[1:].ravel()
        L[src, src] += rate; L[dst, dst] += rate; L[src, dst] -= rate; L[dst, src] -= rate
    return L

TAU_E = {2: 1.2, 3: 1.8}
for d, N, q in [(2, 37, 0.8), (2, 44, 0.8), (3, 13, 0.8), (3, 14, 0.8), (2, 40, 0.5)]:
    b = blk(('0.8', d), N)[0]
    s = scalars(d, N); mu = mu_real(d, N, q)
    n = N ** d
    La = L1(d, N)[:n - 1, :n - 1] * q / mu
    lam, V = np.linalg.eigh(La)
    a = lam * V[0, :] * (V.T @ np.ones(n - 1))
    base = 1 - mu * lam
    tE = math.ceil(TAU_E[d] / mu)
    t_minus = math.floor((s['tau_cf'] - b['c_minus'] / s['X']) / mu); t_plus = math.ceil((s['tau_cf'] + b['c_plus'] / s['X']) / mu)
    tmax = int(12 / mu)
    pw = np.ones_like(base); gs = []
    for t in range(tmax + 2):
        gs.append(float(np.sum(a * pw))); pw = pw * base
    gs = np.array(gs); Dg = np.diff(gs)
    okBC = bool(np.all(Dg[tE:t_minus + 1] > 0)); okA = bool(np.all(Dg[t_plus:tmax] < 0)); okD = bool(np.all(gs[:tE + 1] < gs[t_minus]))
    tstar = int(np.argmax(gs)) + 1
    rec = dict(kind='bruteforce', d=d, N=N, q=q, t_E=tE, t_minus=t_minus, t_plus=t_plus, t_star=tstar, regions_BC=okBC, region_A=okA, region_D=okD,
               inside=bool(t_minus + 2 <= tstar <= t_plus + 1))
    ok_all &= okBC and okA and okD and rec['inside']; out.append(rec); print(json.dumps(rec), flush=True)
print('ALL OK' if ok_all else 'FAIL')
json.dump(out, open(DATA + '/77_validate_q_large.json', 'w'), indent=1)
