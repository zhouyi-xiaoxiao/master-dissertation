"""Numerical observation (no proof): exact discrete modes of the floating-point study (data/msc_modes) for q in {0.8, 0.9, 1.0} versus the windows certified for q <= 1/2
(Theorem 7.8) and for continuous time (Theorem 5.1).  theta := X (mu t* - tau_cf)."""
import json
from r4lib import scalars, exact_modes_D, mu_real, DATA
mD = exact_modes_D()
cert = {d: [json.loads(l) for l in open(DATA + '/72_cert_discrete_d%d.jsonl' % d)] for d in (2, 3)}
out = []
for (d, N, q), (mode, mfpt) in sorted(mD.items()):
    if d == 1 or q <= 0.5 or N < 12 or (d == 2 and N > 1500) or (d == 3 and N > 160):
        continue
    s = scalars(d, N); mu = mu_real(d, N, q)
    r = [b for b in cert[d] if b['Na'] <= N and (b['Nb'] is None or N <= b['Nb'])][0]
    th = float(s['X'] * (mu * mode - s['tau_cf']))
    rec = dict(d=d, N=N, q=q, mode=mode, theta=round(th, 4), inside_q_half_window=bool(-r['c_minus'] < th < r['c_plus'] + 2 * mu * s['X']),
               rel_err_closed_form=round(abs(mode - s['tau_cf'] / mu) / mode, 5))
    out.append(rec); print(json.dumps(rec), flush=True)
json.dump(out, open(DATA + '/74_observe_q_large.json', 'w'), indent=1)
print('inside for all:', all(r['inside_q_half_window'] for r in out), ' n =', len(out))
