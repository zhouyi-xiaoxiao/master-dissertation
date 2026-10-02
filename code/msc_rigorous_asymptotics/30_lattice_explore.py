"""Float exploration of the lattice-sum lemmas (X_s bounds via Bessel lower bound and monotone Riemann upper bound)."""
import numpy as np, json
from scipy.special import ive
from scipy.integrate import quad
from u3lib import spec, consts, DATA
kap = lambda s: ive(0, s) + ive(1, s)            # e^{-s}(I0+I1)
J3 = quad(lambda s: kap(s) ** 3, 0, 200, limit=400)[0] + (2 / np.pi) ** 1.5 * (2 * 200 ** -0.5 - 0.5 * 200 ** -1.5)
c3 = np.pi ** 2 * J3 / 2
print('J3', J3, 'c3', c3, 'ln(6 c3)', np.log(6 * c3))
# 2D: K2(s0) = int_0^s0 kap^2
s0 = 50.0
K2 = quad(lambda s: kap(s) ** 2, 0, s0, limit=400)[0]
ck = K2 - (2 / np.pi) * np.log(s0) - 1 / (np.pi * s0)       # lower-bound constant
cconst = (np.pi ** 2 / 2) * (ck + (2 / np.pi) * np.log(2 / np.pi) - 2 / np.pi)
print('K2(50)', K2, 'c_kappa_lo', ck, '2D lower const', cconst)
out = []
for d, Ns in ((2, [10, 20, 40, 80, 160, 320, 640, 1280]), (3, [5, 10, 20, 40, 80, 160])):
    for N in Ns:
        mu, w, c = spec(d, N)
        Xs = float(np.sum(w / mu)); sig = float(np.sum(w / mu ** 2))
        x = np.pi / (2 * N)
        if d == 2:
            lo = 2 * np.pi * np.log(N) * (1 - x * x / 3) + cconst * (1 - x * x / 3)
            hi = 2 * np.pi * np.log(N) + np.pi * np.log(2) + 2 + 2 * np.pi ** 2 / 3
        else:
            lo = c3 * N - 3 * np.pi - c3 * np.pi ** 2 / (12 * N)
            hi = c3 * N + 3 * (2 * np.pi * np.log(N) + np.pi * np.log(2) + 2) + np.pi ** 2
        rec = dict(d=d, N=N, Xs=Xs, lo=float(lo), hi=float(hi), sigma=sig, ok=bool(lo <= Xs <= hi), Xs_minus_lo=Xs - float(lo), hi_minus_Xs=float(hi) - Xs)
        out.append(rec); print(rec, flush=True)
# Z_d partial sums
def Z(d, K):
    r = np.arange(-K, K + 1)
    if d == 2:
        a, b = np.meshgrid(r, r, indexing='ij'); n2 = (a * a + b * b).astype(float)
    else:
        a, b, c_ = np.meshgrid(r, r, r, indexing='ij'); n2 = (a * a + b * b + c_ * c_).astype(float)
    n2[n2 == 0] = np.inf
    return float(np.sum(1 / n2 ** 2))
for d in (2, 3):
    for K in (20, 60, 120):
        print('Z', d, K, Z(d, K), flush=True)
json.dump(out, open(DATA + '/30_lattice_explore.json', 'w'), indent=1)
