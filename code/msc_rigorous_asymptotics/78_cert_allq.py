"""COMPUTER-ASSISTED PROOF of Theorem 7.17 (note R3): discrete time, every q in (0,1), WITHOUT unimodality and uniformly in q.
For every block of N >= 10 the script checks, in ball arithmetic, the hypotheses (A)-(D) of Theorem 7.13 with the q-dependent
constants of Lemma 7.11 replaced by the q-uniform bounds of Lemma 7.16 (certlib.quni_bounds) and with mu <= (2/d) sin^2 x.
A block marked ok=true proves, for every N in the block and every q in (0,1) with sin(pi/N) <= (1-q)/q,
        t_cf - c_minus/(X mu) < t* < t_cf + c_plus/(X mu) + 2        for every maximiser t* of the PMF.
Same blocks as 52/72/76.  Usage: python 78_cert_allq.py d   -> data/78_cert_allq_d{d}.jsonl (checkpointed, resumable)
"""
import json, sys, os, time
from flint import arb
import certlib as C
from certlib import L, U, iv

d = int(sys.argv[1])
N0 = 10
OUT = os.path.join(C.DATA, '78_cert_allq_d%d.jsonl' % d)
KMAX = {2: 24, 3: 48}[d]
QMAX = 1.0          # mu <= (2 qmax/d) sin^2 x with qmax = 1 covers every q < 1


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
    return C.find_c(E, side, margin=(0.1 if side == '-' else 0.0), disc=True, qmax=QMAX, nounimodal=True, quni=True)


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
        E = C.quni_bounds(C.enclosures(d, y, Na, Nb, KMAX), d, Na, Nb)
        cp, okp = certify(E, '+'); cm, okm = certify(E, '-')
        R = dict(B=False, C=False, D=False)
        if okm and okp:
            try:
                R = C.regions(E, cm, QMAX, quni=True, c_plus=cp)
            except AssertionError:
                pass
        rec = dict(d=d, qmax='all q<1 with sin(pi/N)<=(1-q)/q', Na=Na, Nb=Nb, y_lo=float(ya), y_hi=float(yb),
                   c_plus=cp, ok_A=okp, c_minus=cm, ok_minus=okm,
                   ok_B=R['B'], ok_C=R['C'], ok_D=R['D'], margins=[R.get('B_margin'), R.get('C_margin'), R.get('D_margin')],
                   ok=bool(okp and okm and R['B'] and R['C'] and R['D']), t=round(time.time() - t0, 1))
        f.write(json.dumps(rec) + '\n'); f.flush()
        print(json.dumps(rec), flush=True)
print('ALL DONE', flush=True)
