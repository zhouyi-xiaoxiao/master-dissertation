#!/usr/bin/env python
"""High-precision (mpmath, 50 digits) check of the 1D discrete-time mode at very large N, q = 0.8.

f(t) = (2q/(2N-1)) sum_m (-1)^(m+1) cos(th_m/2) sin(th_m) lam_m^(t-1),  th_m = (2m-1) pi/(2N-1),
lam_m = 1 - q(1 - cos th_m).  The integer mode is the first t with f(t+1) - f(t) <= 0.
Checks that the published modes are maximisers of the closed form."""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import json, os
import mpmath as mp
mp.mp.dps = 50
q = mp.mpf(8) / 10


def inc(N, t, M=60):
    s = mp.mpf(0)
    for m in range(1, M + 1):
        th = (2 * m - 1) * mp.pi / (2 * N - 1)
        mu = 2 * mp.sin(th / 2) ** 2
        lam = 1 - q * mu
        s += (-1) ** (m + 1) * mp.cos(th / 2) * mp.sin(th) * (-q * mu) * lam ** (t - 1)
    return s          # proportional to f(t+1) - f(t)


def fval(N, t, M=60):
    s = mp.mpf(0)
    for m in range(1, M + 1):
        th = (2 * m - 1) * mp.pi / (2 * N - 1)
        lam = 1 - q * 2 * mp.sin(th / 2) ** 2
        s += (-1) ** (m + 1) * mp.cos(th / 2) * mp.sin(th) * lam ** (t - 1)
    return s * 2 * q / (2 * N - 1)


out = []
for N, recheck in ((10240, 43679969), (100000, 4166011658), (1000000, 416604915263)):
    r = dict(N=N, recheck=recheck,
             inc_at_recheck_minus_1=mp.nstr(inc(N, recheck - 1), 5), inc_at_recheck=mp.nstr(inc(N, recheck), 5),
             recheck_is_argmax=bool(inc(N, recheck - 1) > 0 and inc(N, recheck) <= 0))
    out.append(r); print(r)
here = os.path.dirname(os.path.abspath(__file__))
json.dump(out, open(_os.path.join(_R, 'data', 'msc_modes_verify', 'bigN_1d_mp.json'), "w"), indent=1)
