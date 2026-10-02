"""
scan_pairs1d.py -- d = 1, q = 4/5: every start x0 in {1..N'-1} with the target at the end N' (every pair of the
one-dimensional box reduces to this form by reflection), 2 <= N' <= 12.  Exact certificate of Computer-assisted
Theorem 7.4 (cert_thr1d.certify: cone certificate or exact witness), cross-checked against the definition-level
exact sign sequence of verify_thr1d.diffsigns_int on a window of 30 N'^2 + 100 steps.
Output: data/scan_pairs1d_q4-5.json  (used in the remark after Proposition 7.3).
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cert_thr1d import certify          # noqa: E402
from verify_thr1d import diffsigns_int, pattern   # noqa: E402

out = []
bad = 0
for N in range(2, 13):
    for x0 in range(1, N):
        r = certify(N, x0, 4, 5)
        pt = pattern(diffsigns_int(N, x0, 4, 5, 30 * N * N + 100))
        agree = (r["verdict"] is True and pt in ([1, -1], [-1], [1])) or (r["verdict"] is False and len(pt) >= 3)
        bad += not agree
        out.append(dict(N=N, x0=x0, D=N - x0, unimodal=r["verdict"], T_cone=r["T_cone"], witness=r["witness"], window_agrees=agree))
nonuni = [(r["N"], r["x0"], r["D"]) for r in out if r["unimodal"] is False]
summary = dict(q="4/5", cases=len(out), undecided=sum(r["unimodal"] is None for r in out), not_unimodal=nonuni,
               all_not_unimodal_have_x0_ge_2=all(x0 >= 2 for _, x0, _ in nonuni),
               D_values_not_unimodal=sorted(set(D for _, _, D in nonuni)),
               every_x0_ge_2_with_D_in_2_3_4_fails=all(r["unimodal"] is False for r in out if r["x0"] >= 2 and r["D"] in (2, 3, 4)),
               window_disagreements=bad)
json.dump(dict(summary=summary, records=out), open(_os.path.join(_R, 'data', 'msc_rigorous_unimodality', 'scan_pairs1d_q4-5.json'), "w"), indent=1)
print(json.dumps(summary))
print("OK" if bad == 0 and summary["undecided"] == 0 else "PROBLEM")
