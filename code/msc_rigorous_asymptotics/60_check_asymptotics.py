"""Sanity check of Theorems 6.1-6.3 (explicit asymptotic bounds, continuous time) against the exact modes of the floating-point study
(Talbot inversion; d=2: N up to 1.6e7, d=3: N up to 4096) and of Corollary 4.7 (X bounds) against the MFPT of the floating-point study."""
import json, math
import numpy as np
from r4lib import exact_modes_C, mu_real, DATA
C3 = 7.10687
mC = exact_modes_C()
out = []
ok_all = True
for (d, N), (mode, mfpt) in sorted(mC.items()):
    if d == 1 or N < 20:
        continue
    mu = mu_real(d, N)            # q = 1
    tau = mode * mu; X = mfpt * mu
    if d == 2:
        L = 2 * math.pi * math.log(N)
        Xlo, Xhi = L - 0.24, L + 0.60
        lead = math.log(math.log(N)) + math.log(8 * math.pi)
        err = (4.6 + 3 * math.log(8 * math.pi * math.log(N) + 2.4)) / (L - 0.24)
        pref = 4 * N * N / math.pi ** 2
        c1, c2 = 0.405 * N * N * math.log(math.log(N)), 1.60 * N * N * math.log(math.log(N))
        Kp, Km = 7.486, 4.242
    else:
        Xlo, Xhi = C3 * N - 7.63, C3 * N - 5.85
        lead = math.log(N) + math.log(6 * C3)
        err = (9.2 + 5 * math.log(6 * C3 * N)) / (C3 * N - 7.63)
        pref = 6 * N * N / math.pi ** 2
        c1, c2 = 0.607 * N * N * math.log(N), 1.38 * N * N * math.log(N)
        Kp, Km = 3.663, 1.564
    tcf = X * math.log(2 * d * X) / (X + 2 * d - 1)
    theta = X * (tau - tcf)
    chk = dict(X=bool(Xlo <= X <= Xhi), tau=bool(lead - err <= tau < lead), theta=bool(-Km < theta < Kp),
               mode_lo=bool(mode >= pref * (lead - err)), mode_hi=bool(mode <= pref * lead * (1 + 0.83 / N ** 2)),
               crude=bool(c1 <= mode <= c2))
    rec = dict(d=d, N=N, X=X, X_minus_lead=(X - (2 * math.pi * math.log(N) if d == 2 else C3 * N)), tau=tau, lead=lead, R=tau - lead, minus_err=-err,
               theta=theta, ok=all(chk.values()), failed=[k for k, v in chk.items() if not v])
    ok_all &= rec['ok']
    out.append(rec)
    print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)
print('ALL OK' if ok_all else 'SOME FAILED')
json.dump(out, open(DATA + '/60_check_asymptotics.json', 'w'), indent=1)
