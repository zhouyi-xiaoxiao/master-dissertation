"""
table_mode_bounds_small.py -- numerical illustration (float) of Corollary 4.13 for small boxes and d = 1,
complementing scripts/table_mode_bounds.py (which tabulates N >= 8 in d = 2, 3).
Same quantities and the same code (run() of table_mode_bounds.py), continuous time, unit rate.
Not part of any proof.  Output: data/table_mode_bounds_small.json
"""
from __future__ import annotations
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from table_mode_bounds import run  # noqa: E402

DATA = _os.path.join(_R, 'data', 'msc_rigorous_unimodality')

if __name__ == "__main__":
    cases = [("CC", 1, 4), ("CC", 1, 9), ("CC", 1, 15), ("CC", 1, 16),
             ("CC", 2, 3), ("CC", 2, 4), ("CC", 2, 5), ("CC", 2, 7),
             ("C2M", 1, 9), ("C2M", 2, 5), ("C2M", 2, 9),
             ("M2C", 1, 9), ("M2C", 1, 33), ("M2C", 2, 5), ("M2C", 2, 9), ("M2C", 2, 33),
             ("M2C", 3, 3), ("M2C", 3, 5), ("M2C", 3, 9)]
    out = []
    for geo, d, N in cases:
        rec = run(geo, d, N)
        rec["ratio_m1_over_mode"] = rec["m_1"] / rec["mode"]
        out.append(rec)
        print("%s d=%d N=%d: m_u=%.3f mode=%.3f m_1=%.3f m(1/M)=%.3f  m_1/mode=%.4f ok=%s"
              % (geo, d, N, rec["m_u"], rec["mode"], rec["m_1"], rec["m_M"], rec["ratio_m1_over_mode"], rec["ok"]),
              flush=True)
    with open(_os.path.join(_R, 'data', 'msc_rigorous_unimodality', 'table_mode_bounds_small.json'), "w") as fh:
        json.dump(out, fh, indent=1)
