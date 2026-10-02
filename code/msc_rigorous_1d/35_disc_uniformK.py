"""Theorem 6.1 with ONE constant K on every q-ball (certificate for the statement exactly as written).

Script 14 determines, on each q-ball [q_i, q_{i+1}], the smallest constant K_i it can certify and reports
max_i K_i.  A certificate at K_i covers the windows (A), (B) of Theorem 6.1 at K_i, which is not literally the
window at the larger stated constant K.  Here the certification function of script 14 (same reduction, same
enclosure of the remainder) is run with the stated constant K itself on every q-ball:

   (A) M(-K) + Xi > 0,   (B) M(K) + Xi < 0,   slope condition,

simultaneously for all eps in [0, eps0], all q in the ball and all d in kappa + q rho + [-(q + K eps0), q + K eps0].
Configurations: the four constants of Theorem 6.1 for q <= 4/5, the three for q <= 9/10, and the constant 1.56 used
in Theorem 7.7 (N >= 601, q <= 1, tolerance mode: B_high replaced by the hypothesis (33), i.e. <= 1e-12).

Output: data/35_disc_uniformK.json; log: logs/35_disc_uniformK.log.
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import os, sys, json, time, importlib.util
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
spec = importlib.util.spec_from_file_location("s14", _os.path.join(_R, 'code', 'msc_rigorous_1d', '14_disc_theorem.py'))
s14 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s14)
import core
from core import arb

CONFIGS = [  # (N0, k0, q_num, q_den, number of q-balls, K, tolerance)
    (41, 11, 4, 5, 2400, "1.68", None),
    (51, 11, 4, 5, 2400, "1.63", None),
    (101, 13, 4, 5, 2400, "1.58", None),
    (601, 15, 4, 5, 2400, "1.56", None),
    (61, 15, 9, 10, 2700, "1.61", None),
    (101, 13, 9, 10, 2700, "1.58", None),
    (601, 15, 9, 10, 2700, "1.56", None),
    (601, 15, 1, 1, 3000, "1.56", "1e-12"),
]

if __name__ == "__main__":
    out = []
    allok = True
    for (N0, k0, qn, qd, nsub, K, tol) in CONFIGS:
        core.HIGH_TOL = tol
        L0 = arb(2 * N0 - 1) / 2
        t0 = time.time()
        fails = []
        for i in range(nsub):
            qlo = arb(qn * i) / (qd * nsub)
            qhi = arb(qn * (i + 1)) / (qd * nsub)
            a, b, s, gam, Bnd = s14.certify(qlo, qhi, L0, k0, arb(K))
            if not (a and b and s):
                fails.append(dict(i=i, A=bool(a), B=bool(b), slope=bool(s)))
        ok = not fails
        allok &= ok
        row = dict(N0=N0, k0=k0, qmax=f"{qn}/{qd}", n_qballs=nsub, K=K, high_tolerance=tol,
                   failing_balls=len(fails), first_failures=fails[:5], certified=ok, seconds=round(time.time() - t0, 1))
        print(row, flush=True)
        out.append(row)
    print("ALL UNIFORM-K CERTIFICATES:", allok)
    json.dump(dict(all_certified=allok, runs=out),
              open(_os.path.join(_R, 'data', 'msc_rigorous_1d', '35_disc_uniformK.json'), "w"), indent=1)
