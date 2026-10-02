"""COMPUTER-ASSISTED PROOF of Theorem 7.8 (note R3), DISCRETE time, all q in (0, 1/2]: for every block of N, constants c_minus, c_plus with
        G_D(t) < 0 at the integer t in [ (tau_cf + c_plus/X)/mu, ... + 1 )  and  G_D(t) > 0 at the integer t in ( (tau_cf - c_minus/X)/mu - 1, ... ],
   hence  t_cf - c_minus/(X mu) < t*_D < t_cf + c_plus/(X mu) + 2      for all N in the block,
obtained by checking, in Arb ball arithmetic, the sign conditions of Theorem 2.3 at t = tau_cf + c_plus/X, tau_cf - c_minus/X
with the pole data replaced by the enclosures of Sections 3-4 (certlib.py).
Usage: python 72_cert_discrete.py d N0      -> data/72_cert_discrete_d{d}.jsonl (checkpointed, resumable)
"""
import json, sys, os, time, math
from flint import arb
import certlib as C
from certlib import L, U, iv

d = int(sys.argv[1]); N0 = int(sys.argv[2])
OUT = os.path.join(C.DATA, '72_cert_discrete_d%d.jsonl' % d)
KMAX = {2: 24, 3: 48}[d]


def blocks(d, N0):
    bl = []
    n = N0
    single_until = {2: 64, 3: 200}[d]
    while n <= single_until:
        bl.append((n, n)); n += 1
    ratio = {2: 1.08, 3: 1.06}[d]
    top = {2: 10 ** 60, 3: 10 ** 9}[d]
    while n < 10 ** 7:
        m = max(n, int(n * ratio)); bl.append((n, m)); n = m + 1
    while n < top:
        m = int(n ** 1.3) if d == 2 else int(n * 1.5); bl.append((n, m)); n = m + 1
    bl.append((n, None))
    return bl


def certify(E, side):
    return C.find_c(E, side, disc=True)


done = set()
if os.path.exists(OUT):
    for line in open(OUT):
        r = json.loads(line); done.add((r['Na'], r['Nb']))
t0 = time.time()
with open(OUT, 'a') as f:
    for Na, Nb in blocks(d, N0):
        if (Na, Nb) in done:
            continue
        yb = U(1 / arb(C.Xs_lower(d, Na)))
        ya = arb(0) if Nb is None else L(1 / arb(C.Xs_upper(d, Nb)))
        y = iv(ya, yb)
        E = C.enclosures(d, y, Na, Nb, KMAX)
        cp, okp = certify(E, '+'); cm, okm = certify(E, '-')
        rec = dict(d=d, Na=Na, Nb=Nb, y_lo=float(ya), y_hi=float(yb), c_plus=cp, ok_plus=okp, c_minus=cm, ok_minus=okm,
                   e=[float(L(E['e'])), float(U(E['e']))], theta=[float(L(E['theta'])), float(U(E['theta']))],
                   T1=[float(L(E['T1'])), float(U(E['T1']))], xi_hi=float(U(E['xi'])),
                   muX_hi=float(U(E['xi'] * (1 + E['EU'] * y) / d)), t=round(time.time() - t0, 1))
        f.write(json.dumps(rec) + '\n'); f.flush()
        print(json.dumps(rec), flush=True)
print('ALL DONE', flush=True)
