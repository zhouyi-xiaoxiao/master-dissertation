#!/usr/bin/env python
"""Supplementary Section S2.5 (numerical inversion of a generating function): sizes other than N = 35.

The trapezoidal rule on a circle of FIXED radius r = 0.9, truncated after K = 54 nodes,
        f(t) = r^(-t)/t [ F~(r)/2 + sum_{k=1}^{K} (-1)^k Re F~(r e^{i pi k/t}) ]
is evaluated at the modal time for  d = 2, N = 9;  the chain (d = 1), N = 21;  d = 2, N = 20   (q = 0.8),
against the exact PMF.  F~ and f are obtained from the dense killed matrix Q by direct linear algebra
(code written for this script; no research library is imported).

Output: data/article/s3_methods_inversion_other_sizes.json
Run   : python s3_methods_inversion_other_sizes.py            (a few seconds)
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = _os.path.join(_R, 'data', 'article', 's3_methods_inversion_other_sizes.json')
Q = 0.8


# ----------------------------------------------------------------------------- exact objects
def killed_matrix(N, d, q):
    """dense Q (target = opposite corner removed), start index, absorption vector r"""
    shape = (N,) * d
    n = N ** d
    P = np.zeros((n, n))
    for flat in range(n):
        x = np.unravel_index(flat, shape)
        P[flat, flat] += 1.0 - q
        for ax in range(d):
            for s in (-1, 1):
                y = list(x)
                y[ax] += s
                if 0 <= y[ax] < N:
                    P[flat, np.ravel_multi_index(tuple(y), shape)] += q / (2 * d)
                else:
                    P[flat, flat] += q / (2 * d)                         # cancelled move
    tgt = n - 1                                                           # (N-1, ..., N-1)
    keep = [i for i in range(n) if i != tgt]
    Qm = P[np.ix_(keep, keep)]
    r = P[keep, tgt]
    return Qm, 0, r                                                       # start (0,...,0) has index 0


def exact_pmf(Qm, o, r, tmax):
    f = np.zeros(tmax + 1)
    v = np.zeros(Qm.shape[0])
    v[o] = 1.0
    for t in range(1, tmax + 1):
        f[t] = v @ r
        v = Qm.T @ v
    return f


def fixed_radius_rule(Qm, o, r, t, rad=0.9, K=54):
    """fixed radius, K nodes; F~(z) = z e_o^T (I - z Q)^{-1} r by dense solves"""
    n = Qm.shape[0]
    eye = np.eye(n)

    def Ft(z):
        return z * np.linalg.solve(eye - z * Qm, r.astype(complex))[o]

    s = 0.5 * Ft(rad).real
    for k in range(1, K + 1):
        s += (-1) ** k * Ft(rad * np.exp(1j * np.pi * k / t)).real
    return s / t * math.exp(-t * math.log(rad))


res = {"generated_by": "code/article/s3_methods_inversion_other_sizes.py", "q": Q}

rows = []
for (d, N) in ((2, 9), (1, 21), (2, 20)):
    Qm, o, r = killed_matrix(N, d, Q)
    f = exact_pmf(Qm, o, r, 4000 if (d, N) != (2, 9) else 1500)
    mode = int(np.argmax(f))
    val = fixed_radius_rule(Qm, o, r, mode)
    rows.append({"d": d, "N": N, "mode_exact": mode, "f_mode_exact": float(f[mode]),
                 "fixed_radius_rule_value_at_mode": float(val),
                 "fixed_radius_rule_signed_rel_err_at_mode": float(val / f[mode] - 1.0)})
    print(rows[-1], flush=True)
res["fixed_radius_rule_r0.9_K54_at_modal_time"] = rows

json.dump(res, open(OUT, "w"), indent=1)
print("written", OUT)
