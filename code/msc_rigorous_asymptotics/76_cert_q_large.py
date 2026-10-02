"""COMPUTER-ASSISTED PROOF of Theorem 7.14 (note R3): discrete time, all q in (0, qmax], qmax < 1, WITHOUT unimodality
(hypotheses of Theorem 7.13 checked block by block).
For every block of N >= max(N0, N_1(qmax)) the script certifies, in ball arithmetic,
  (A)  the upper sign condition (with E_-^lo := 0 and the negative-pole term n_q) at t_+ = ceil((tau_cf + c_plus/X)/mu),
  (B)  the lower sign condition (with n_q) for every tau in [tau_cf - c_minus/X - b_B, tau_cf - c_minus/X],
  (C)  c_E (B_1(t_B) - omega_q) > omega_q''   (positivity of Dg on [t_E, t_B]),
  (D)  the value inequality (no maximiser before t_E),
hence  t_cf - c_minus/(X mu) < t* < t_cf + c_plus/(X mu) + 2  for every maximiser t* of the PMF, every N in the block, every q <= qmax.
Usage: python 76_cert_q_large.py d qmax N0   -> data/76_cert_q{qmax}_d{d}.jsonl
"""
import json, sys, os, time, math
from flint import arb
import certlib as C
from certlib import L, U, iv

d = int(sys.argv[1]); qmax = float(sys.argv[2]); N0 = int(sys.argv[3])
tag = ('%g' % qmax).replace('.', 'p')
OUT = os.path.join(C.DATA, '76_cert_q%s_d%d.jsonl' % (tag, d))
KMAX = {2: 24, 3: 48}[d]
# N_1(q): sin(pi/N) <= (1-q)/q
N1 = 4
while math.sin(math.pi / N1) > (1 - qmax) / qmax * (1 - 1e-12):
    N1 += 1
assert arb(N1) > 0 and (arb.pi() / N1).sin() < arb(str(1 - qmax)) / arb(str(qmax))
N0 = max(N0, N1)


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
    return C.find_c(E, side, margin=(0.1 if side == '-' else 0.0), disc=True, qmax=qmax, nounimodal=True)


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
        R = dict(B=False, C=False, D=False)
        if okm:
            try:
                R = C.regions(E, cm, qmax, c_plus=cp if okp else None)
            except AssertionError:
                pass
        rec = dict(d=d, qmax=qmax, Na=Na, Nb=Nb, y_lo=float(ya), y_hi=float(yb), c_plus=cp, ok_A=okp, c_minus=cm, ok_minus=okm,
                   ok_B=R['B'], ok_C=R['C'], ok_D=R['D'], margins=[R.get('B_margin'), R.get('C_margin'), R.get('D_margin')],
                   ok=bool(okp and okm and R['B'] and R['C'] and R['D']), t=round(time.time() - t0, 1))
        f.write(json.dumps(rec) + '\n'); f.flush()
        print(json.dumps(rec), flush=True)
print('ALL DONE', flush=True)
