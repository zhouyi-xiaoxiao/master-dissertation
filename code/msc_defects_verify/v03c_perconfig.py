"""v03c -- regenerate the main implementation's own placements from its documented seed scheme
(SeedSequence([20261001, N, scheme_id, round(1e4 p), rep]); uniform: rng.choice(arange(1, N^2-1), M) until connected;
smart: generator generate_defects_smart of code/msc_defects/ensemble.py seeded with rng.integers(0, 2^31-1)) and solve them with the solver of this check; compare per placement
with the main implementation's CSV rows.  Output: data/v03c_perconfig.csv"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
import vcore as vc
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _os.path.join(_R, 'code', 'msc_defects'))
from ensemble import generate_defects_smart as gen     # corner-thinned placement law (Section 8 of the article)
N, q = 35, 0.8
rows = []
for scheme, sid in (('uniform', 1), ('smart', 2)):
    for p in (0.02, 0.10, 0.20, 0.30, 0.35):
        tr = pd.read_csv(os.path.join(_R, 'data', 'msc_defects', 'sweep', f'sweep_N35_{scheme}_p{p:.2f}.csv')).set_index('rep')
        for rep in range(40):
            rng = np.random.default_rng(np.random.SeedSequence([20261001, N, sid, int(round(p * 10000)), rep]))
            M = int(round(p * N * N)); att = 0
            while True:
                att += 1
                if scheme == 'uniform':
                    blk = rng.choice(np.arange(1, N * N - 1), M, replace=False)
                    mask = np.zeros(N * N, bool); mask[blk] = True; mask = mask.reshape(N, N)
                else:
                    mask = gen(N, p, int(rng.integers(0, 2 ** 31 - 1)))
                if vc.connected(~mask, (0, 0), (N - 1, N - 1)):
                    break
            r = vc.solve_config(~mask, q)
            t = tr.loc[rep]
            rows.append(dict(scheme=scheme, p=p, rep=rep, att=att, tr_att=int(t.attempts), n=r['n'], tr_n=int(t.n_cluster),
                             mfpt=r['mfpt'], tr_mfpt=t.mfpt, mode=r['mode'], tr_mode=int(t['mode']), sd=r['sd'], tr_sd=t.sd,
                             fmode=r['fmode'], tr_fmode=t.f_mode))
df = pd.DataFrame(rows)
df.to_csv(_os.path.join(_R, 'data', 'msc_defects_verify', 'v03c_perconfig.csv'), index=False)
print('configs', len(df), 'same cluster size', int((df.n == df.tr_n).sum()), 'same attempts', int((df.att == df.tr_att).sum()))
print('max rel diff mfpt', np.max(np.abs(df.mfpt / df.tr_mfpt - 1)), 'max rel diff sd', np.max(np.abs(df.sd / df.tr_sd - 1)))
print('mode identical in', int((df['mode'] == df.tr_mode).sum()), 'of', len(df), '; max abs diff', int(np.max(np.abs(df['mode'] - df.tr_mode))))
print('max rel diff peak height', np.max(np.abs(df.fmode / df.tr_fmode - 1)))
print(df[df['mode'] != df.tr_mode])
