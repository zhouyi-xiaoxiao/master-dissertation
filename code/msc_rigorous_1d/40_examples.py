"""Exact checks of three numerical examples quoted in note R1 (start x0 = 1):

  N = 6,  q = 207/250 (0.828): the mode is 11 < tau - E = 11.0097...   (Prop. 6.2, exception to the window)
  N = 9,  q = 887/1000 (0.887): the mode is 26 < tau - E = 26.031...   (Prop. 6.2, exception to the window)
  N = 10, q = 99/100: the mode is 27 (the q = 1 mode), while tau = 29.3... (Remark 7.10, crossover)

f(t) is computed in exact rational arithmetic for t <= H = 30 N^2; the global maximum over all t is certified by the
tail bound f(t) <= (2q/(2N-1)) (N-1) lam_max^(t-1) (from (3), |sin| <= 1), lam_max = max_m |lambda_m| < 1, evaluated in
50-digit arithmetic with a safety factor, which is below max_{t<=H} f(t) for all t > H.
tau = (c L^2 - kappa)/q + 1 - rho and E = 1.78/(q L^2) with c, kappa, rho from data/03_constants.json.
Output: data/40_examples.json; log: logs/40_examples.log.
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import os, json
from fractions import Fraction as Fr
import mpmath as mp

mp.mp.dps = 50
here = os.path.dirname(os.path.abspath(__file__))
const = json.load(open(_os.path.join(_R, 'data', 'msc_rigorous_1d', '03_constants.json')))
c = mp.mpf(const["c_mid_80_digits"])
ball = lambda key: mp.mpf(const[key].strip("[").split(" +/-")[0])
kappa, rho = ball("kappa"), ball("rho")


def pmf(N, q, H):
    n = N - 1
    p = [Fr(0)] * (n + 2)
    p[1] = Fr(1)
    h = q / 2
    out = []
    for _ in range(H):
        new = [Fr(0)] * (n + 2)
        for x in range(1, n + 1):
            if p[x]:
                new[x] += p[x] * ((1 - h) if x == 1 else (1 - q))
                if x > 1:
                    new[x - 1] += p[x] * h
                new[x + 1] += p[x] * h
        out.append(new[N])
        new[N] = Fr(0)
        p = new
    return out


rows, allok = [], True
for N, q, expect_mode in [(6, Fr(207, 250), 11), (9, Fr(887, 1000), 26), (10, Fr(99, 100), 27)]:
    H = 30 * N * N
    f = pmf(N, q, H)
    fmax = max(f)
    modes = [t + 1 for t, v in enumerate(f) if v == fmax]
    qm = mp.mpf(q.numerator) / q.denominator
    lam = [1 - qm * (1 - mp.cos((2 * m - 1) * mp.pi / (2 * N - 1))) for m in range(1, N)]
    lmax = max(abs(x) for x in lam)
    tail = 2 * qm / (2 * N - 1) * (N - 1) * lmax ** H * (1 + mp.mpf(10) ** -30)
    L = mp.mpf(2 * N - 1) / 2
    tau = (c * L ** 2 - kappa) / qm + 1 - rho
    E = mp.mpf("1.78") / (qm * L ** 2)
    ok = modes == [expect_mode] and tail < mp.mpf(fmax.numerator) / fmax.denominator
    allok &= ok
    rows.append(dict(N=N, q=str(q), modes=modes, f_max=float(fmax), tail_bound_beyond_H=mp.nstr(tail, 5),
                     tau=mp.nstr(tau, 8), tau_minus_E=mp.nstr(tau - E, 8), tau_plus_1_plus_E=mp.nstr(tau + 1 + E, 8), ok=ok))
    print(rows[-1], flush=True)
print("ALL OK:", allok)
json.dump(dict(all_ok=allok, rows=rows), open(_os.path.join(_R, 'data', 'msc_rigorous_1d', '40_examples.json'), "w"), indent=1)
