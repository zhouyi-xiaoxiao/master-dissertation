"""v04 -- effective-medium checks, own code.
(a) lattice Green function value G(0)-G(2,0) (mpmath quadrature) ; (b) one blocked site on an LxL torus:
(1-sigma)L^2 -> pi ; (c) random site dilution on tori: sigma(p), P_inf(p), Theta(p)=P_inf/sigma ;
(d) periodic arrays (one blocked site per AxA cell).   Output: data/v04_homog.json"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import os, sys, json, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spla, scipy.sparse.csgraph as csg
import mpmath as mp
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = _os.path.join(_R, 'data', 'msc_defects_verify', 'v04_homog.json')
res = json.load(open(OUT)) if os.path.exists(OUT) else {}

def save():
    json.dump(res, open(OUT + '.tmp', 'w'), indent=1); os.replace(OUT + '.tmp', OUT)

# (a)  G(0)-G(m,0) = (1/2pi) int_0^{2pi} (1-cos m k)/sqrt((4-2cos k)^2-4) dk   (L G = delta, L = 4 - sum of neighbours)
mp.mp.dps = 30
def _f(m):
    # (1-cos mk)/sqrt((2-2cos k)(6-2cos k)) with 1-cos(mk) = 2 sin^2(mk/2), 2-2cos k = 4 sin^2(k/2); symmetric about pi
    def g(k):
        if k == 0:
            return mp.mpf(0)
        return 2 * mp.sin(m * k / 2) ** 2 / (2 * mp.sin(k / 2) * mp.sqrt(6 - 2 * mp.cos(k)))
    return g
g2 = 2 * mp.quad(_f(2), [0, mp.pi]) / (2 * mp.pi)
g1 = 2 * mp.quad(_f(1), [0, mp.pi]) / (2 * mp.pi)
res['green'] = dict(G0_minus_G20=float(g2), one_minus_2_over_pi=float(1 - 2 / mp.pi), G0_minus_G10=float(g1))
save()


def torus_sigma(open_mask):
    """Conductivity along x (unit bonds between open neighbours, periodic) and largest-cluster fraction.
    Returns sigma (power per lattice site for unit field), P_inf, n_open."""
    L = open_mask.shape[0]
    idx = -np.ones((L, L), np.int64); idx[open_mask] = np.arange(open_mask.sum())
    n = int(open_mask.sum())
    # x-bonds: (i,j)-(i+1,j) ; y-bonds: (i,j)-(i,j+1), periodic.  axis 0 = x.
    a = idx; bx = np.roll(idx, -1, axis=0); by = np.roll(idx, -1, axis=1)
    okx = (a >= 0) & (bx >= 0); oky = (a >= 0) & (by >= 0)
    I = np.r_[a[okx], a[oky]]; J = np.r_[bx[okx], by[oky]]
    dxb = np.r_[np.ones(okx.sum()), np.zeros(oky.sum())]          # x displacement along bond I->J
    A = sp.coo_matrix((np.ones(2 * len(I)), (np.r_[I, J], np.r_[J, I])), shape=(n, n)).tocsr()
    ncomp, lab = csg.connected_components(A, directed=False)
    sizes = np.bincount(lab); big = np.argmax(sizes)
    inb = lab == big
    # restrict to largest component
    keepb = inb[I]
    I = I[keepb]; J = J[keepb]; dxb = dxb[keepb]
    new = -np.ones(n, np.int64); new[inb] = np.arange(inb.sum()); I = new[I]; J = new[J]; m = int(inb.sum())
    Lm = sp.coo_matrix((np.r_[np.ones(len(I)), np.ones(len(I)), -np.ones(len(I)), -np.ones(len(I))],
                        (np.r_[I, J, I, J], np.r_[I, J, J, I])), shape=(m, m)).tocsc()
    # phi = -x + psi ; drop along bond I->J: phi_I - phi_J = dx + psi_I - psi_J.  Kirchhoff: L psi = -div(dx)
    b = np.zeros(m); np.add.at(b, I, -dxb); np.add.at(b, J, dxb)
    psi = np.zeros(m)
    lu = spla.splu(Lm[1:, 1:])
    psi[1:] = lu.solve(b[1:])
    drop = dxb + psi[I] - psi[J]
    sigma = float(np.sum(drop ** 2)) / (L * L)
    return sigma, m / (L * L), n

# (b) single blocked site
res.setdefault('single', {})
for L in (8, 16, 32, 64, 128, 256):
    if str(L) in res['single']:
        continue
    om = np.ones((L, L), bool); om[L // 2, L // 2] = False
    s, _, _ = torus_sigma(om)
    res['single'][str(L)] = (1 - s) * L * L
    print('single', L, res['single'][str(L)], flush=True); save()

# (d) periodic arrays
res.setdefault('periodic', {})
for A_ in (2, 3, 4, 5):
    L = 60
    om = np.ones((L, L), bool); om[::A_, ::A_] = False
    s, pinf, n = torus_sigma(om)
    res['periodic'][str(A_)] = dict(p=1 / A_ ** 2, sigma=s, P_inf=pinf, Theta=pinf / s)
    print('periodic', A_, res['periodic'][str(A_)], flush=True)
save()

# (c) random dilution
res.setdefault('random', {})
PG = [0.01, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.25, 0.30, 0.35]
for L, K in ((160, 10), (80, 30)):
    for p in PG:
        key = f'L{L}_p{p:.2f}'
        if key in res['random']:
            continue
        rng = np.random.default_rng([4242, L, int(round(p * 1e4))])
        sg = []; pi_ = []
        for k in range(K):
            M = int(round(p * L * L))
            mask = np.zeros(L * L, bool); mask[rng.choice(L * L, M, replace=False)] = True
            s, pinf, n = torus_sigma(~mask.reshape(L, L))
            sg.append(s); pi_.append(pinf)
        sg = np.array(sg); pi_ = np.array(pi_); th = pi_ / sg
        res['random'][key] = dict(L=L, p=p, K=K, sigma=float(sg.mean()), sigma_ci=float(1.96 * sg.std(ddof=1) / np.sqrt(K)),
                                  P_inf=float(pi_.mean()), Theta=float(pi_.mean() / sg.mean()),
                                  Theta_mean_of_ratio=float(th.mean()), Theta_ci=float(1.96 * th.std(ddof=1) / np.sqrt(K)),
                                  D_ratio=float(sg.mean() / pi_.mean()))
        print(key, res['random'][key], flush=True); save()
