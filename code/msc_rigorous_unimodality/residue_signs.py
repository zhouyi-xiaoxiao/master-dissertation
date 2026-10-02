"""residue_signs.py -- numerical observation (mpmath, 60 digits): number of sign changes S^-(C) of the
residue sequence (grouped by distinct eigenvalue, zero residues dropped) versus the lower bound D-1 of Theorem 3.2."""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mpmath as mp
from verify_lemmas import residues_mp
HERE = os.path.dirname(os.path.abspath(__file__))
cases = [(1, 5, (0,), (4,)), (1, 8, (0,), (7,)), (1, 8, (3,), (7,)),
         (2, 2, (0, 0), (1, 1)), (2, 3, (0, 0), (2, 2)), (2, 4, (0, 0), (3, 3)), (2, 5, (0, 0), (4, 4)), (2, 6, (0, 0), (5, 5)), (2, 7, (0, 0), (6, 6)),
         (3, 2, (0, 0, 0), (1, 1, 1)), (3, 3, (0, 0, 0), (2, 2, 2)), (3, 4, (0, 0, 0), (3, 3, 3)),
         (2, 5, (0, 0), (2, 2)), (2, 7, (0, 0), (3, 3)), (2, 5, (2, 2), (4, 4)), (2, 7, (3, 3), (6, 6)),
         (3, 3, (2, 1, 1), (0, 1, 1)), (2, 8, (5, 5), (2, 3)), (2, 4, (0, 1), (3, 2)), (2, 5, (1, 0), (1, 2))]
out = []
for (d, N, s, a) in cases:
    groups = residues_mp(N, d, s, a)
    tol = mp.mpf(10) ** (-35)
    gr = [(l, c) for (l, c) in groups if abs(c) > tol]
    D = sum(abs(x - y) for x, y in zip(s, a))
    sg = [1 if c > 0 else -1 for (l, c) in gr]
    sc = sum(1 for i in range(len(sg) - 1) if sg[i] != sg[i + 1])
    rec = dict(d=d, N=N, start_0based=list(s), target_0based=list(a), D=D, n_nonzero_residues=len(gr), sign_changes=sc,
               lower_bound=D - 1, pattern="".join("+" if v > 0 else "-" for v in sg))
    out.append(rec); print(rec, flush=True)
json.dump(out, open(_os.path.join(_R, 'data', 'msc_rigorous_unimodality', 'residue_signs.json'), "w"), indent=1)
