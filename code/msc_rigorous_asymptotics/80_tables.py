"""Tables of certified constants from the certificate files (used to fill the tables of note R3).
Rule (proof of Theorem 5.1): K^pm(N0) = max of c_pm(B) over all blocks B that MEET [N0, inf), i.e. blocks with N_b >= N0
(N_b = None for the unbounded block).  A block that contains N0 in its interior is included, so the constants are certified
for every N >= N0.  For each entry the block containing N0 is recorded."""
import json, os, sys
from r4lib import DATA


def load(name):
    return [json.loads(l) for l in open(os.path.join(DATA, name))]


def meets(r, N0):
    return r['Nb'] is None or r['Nb'] >= N0


def containing(rows, N0):
    for r in rows:
        if r['Na'] <= N0 and (r['Nb'] is None or N0 <= r['Nb']):
            return [r['Na'], r['Nb']]
    return None


out = {'rule': 'max over blocks meeting [N0, inf) (Nb >= N0)'}
for name, key, N0s in (('52_cert_theoremB_d2.jsonl', 'T51_d2', [11, 12, 15, 20, 30, 50, 100, 200, 1000, 10 ** 6]),
                       ('52_cert_theoremB_d3.jsonl', 'T51_d3', [10, 15, 20, 30, 50, 100, 200, 1000, 10 ** 6]),
                       ('72_cert_discrete_d2.jsonl', 'T78_d2', [11, 12, 15, 20, 30, 50, 100, 200, 1000]),
                       ('72_cert_discrete_d3.jsonl', 'T78_d3', [10, 15, 20, 30, 50, 100, 200, 1000])):
    rows = load(name)
    allok = lambda sel: all(r['ok_plus'] and r['ok_minus'] for r in sel)
    first = next((r['Na'] for i, r in enumerate(rows) if allok(rows[i:])), None)
    tab = []
    for N0 in N0s:
        sel = [r for r in rows if meets(r, N0)]
        if allok(sel):
            tab.append((N0, max(r['c_minus'] for r in sel), max(r['c_plus'] for r in sel), containing(rows, N0)))
    out[key] = dict(blocks=len(rows), first_ok=first, table=tab)
    print(key, 'blocks', len(rows), 'first N0 with all ok:', first)
    for t in tab: print('   N0=%g  K-=%.3f  K+=%.3f   (block containing N0: %s)' % t)


def qtable(fname, key, extra):
    rows = load(fname)
    first = next((r['Na'] for i, r in enumerate(rows) if all(x['ok'] for x in rows[i:])), None)
    tab = []
    for N0 in sorted(set([first] + extra)):
        if first is None or N0 < first:
            continue
        sel = [r for r in rows if meets(r, N0)]
        assert all(r['ok'] for r in sel)
        tab.append((N0, max(r['c_minus'] for r in sel), max(r['c_plus'] for r in sel), containing(rows, N0)))
    out[key] = dict(blocks=len(rows), N1=rows[0]['Na'], first_ok=first, table=tab,
                    failing=[(r['Na'], r['Nb'], [k for k in ('ok_A', 'ok_minus', 'ok_B', 'ok_C', 'ok_D') if not r[k]]) for r in rows if not r['ok']])
    print(key, 'blocks', len(rows), 'N1', rows[0]['Na'], 'first ok', first)
    for t in tab: print('   N0=%g  K-=%.3f  K+=%.3f   (block containing N0: %s)' % t)


for tag in ('0p8', '0p9', '0p95', '0p99'):
    for d in (2, 3):
        qtable('76_cert_q%s_d%d.jsonl' % (tag, d), 'T714_q%s_d%d' % (tag, d), [20, 30, 50, 100, 200, 1000])
for d in (2, 3):
    if os.path.exists(os.path.join(DATA, '78_cert_allq_d%d.jsonl' % d)):
        qtable('78_cert_allq_d%d.jsonl' % d, 'T717_d%d' % d, [15, 20, 30, 50, 60, 100, 200, 312, 1000])
json.dump(out, open(os.path.join(DATA, '80_tables.json'), 'w'), indent=1)
