"""03_homogenization.py -- effective (homogenised) transport coefficients of the blocked-site lattice.

The walk with cancelled moves is the "blind ant": symmetric hop probability q/4 across every open
bond, uniform stationary law on the open cluster.  For a homogenised medium
    MFPT(p)/MFPT(0) = [n(p)/N^2] / [sigma(p)/sigma0]  =  D0 / D_eff(p),
where sigma is the conductivity of the unit-conductance resistor network with the blocked sites
removed and n the number of sites of the accessible cluster.

(a) Dilute limit.  Analytically (lattice Green function, G(0)-G(2,0) = 1-2/pi):
        sigma/sigma0 = 1 - pi p + O(p^2),   D_eff/D0 = 1 - (pi-1) p + O(p^2).
    Verified here with ONE blocked site on an L x L torus:  (1-sigma) L^2 -> pi.
(b) Finite p.  sigma(p), P_inf(p) on L=192 tori with exactly round(p L^2) uniformly placed blocked
    sites, 12 samples per p (seeded).  Output data/homogenization.json.
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import os, sys, json, time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse.csgraph import connected_components

DATA = _os.path.join(_R, 'data', 'msc_defects')


def torus_sigma(open_mask):
    """Return (sigma_xx, sigma_yy, frac_largest_cluster) for unit conductances on open-open bonds
    of a periodic lattice."""
    Lx, Ly = open_mask.shape
    idx = -np.ones(open_mask.size, np.int64)
    flat = np.flatnonzero(open_mask.ravel())
    n = flat.size
    idx[flat] = np.arange(n)
    ii, jj = np.unravel_index(flat, open_mask.shape)
    out = []
    bonds = {}
    for ax, (di, dj) in enumerate(((1, 0), (0, 1))):
        ni, nj = (ii + di) % Lx, (jj + dj) % Ly
        ok = open_mask[ni, nj]
        a = np.arange(n)[ok]
        b = idx[np.ravel_multi_index((ni[ok], nj[ok]), open_mask.shape)]
        bonds[ax] = (a, b)
    a_all = np.concatenate([bonds[0][0], bonds[1][0]])
    b_all = np.concatenate([bonds[0][1], bonds[1][1]])
    W = sp.coo_matrix((np.ones(a_all.size), (a_all, b_all)), shape=(n, n))
    A = (W + W.T).tocsr()
    deg = np.asarray(A.sum(axis=1)).ravel()
    Lap = (sp.diags(deg) - A).tocsr()
    ncomp, lab = connected_components(A, directed=False)
    sizes = np.bincount(lab)
    # pin one node per component
    pins = np.zeros(ncomp, np.int64)
    first = np.full(ncomp, -1)
    order = np.arange(n)
    first[lab[::-1]] = order[::-1]
    free = np.ones(n, bool); free[first] = False
    fi = np.flatnonzero(free)
    Lf = Lap[fi][:, fi].tocsc()
    sig = []
    for ax in (0, 1):
        a, b = bonds[ax]
        rhs = np.zeros(n)
        # current a->b = 1 + psi_a - psi_b ; Kirchhoff: sum_out = 0  ->  Lap psi = -(n_plus - n_minus)
        np.add.at(rhs, a, -1.0)
        np.add.at(rhs, b, +1.0)
        psi = np.zeros(n)
        if fi.size:
            psi[fi] = spla.spsolve(Lf, rhs[fi])
        cur = 1.0 + psi[a] - psi[b]
        sig.append(float(cur.sum() / open_mask.size))
    return sig[0], sig[1], float(sizes.max() / open_mask.size)


def main():
    res = {}
    # (a) single blocked site
    single = []
    for L in (8, 16, 32, 64, 128, 256):
        om = np.ones((L, L), bool); om[L // 2, L // 2] = False
        sx, sy, _ = torus_sigma(om)
        single.append(dict(L=L, one_minus_sigma_times_L2=(1 - sx) * L * L))
        print(single[-1], flush=True)
    res['single_site'] = single
    res['pi'] = float(np.pi)
    # (b) finite density
    L = 192
    pgrid = [0.0, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.225, 0.25, 0.275, 0.30, 0.325,
             0.35, 0.38]
    rows = []
    t0 = time.time()
    for p in pgrid:
        M = int(round(p * L * L))
        vals = []
        for rep in range(1 if p == 0 else 12):
            rng = np.random.default_rng(np.random.SeedSequence([31415, int(round(p * 10000)), rep]))
            om = np.ones(L * L, bool)
            om[rng.choice(L * L, M, replace=False)] = False
            sx, sy, pinf = torus_sigma(om.reshape(L, L))
            vals.append((0.5 * (sx + sy), pinf))
        v = np.array(vals)
        row = dict(p=p, L=L, samples=len(vals), sigma=float(v[:, 0].mean()),
                   sigma_sem=float(v[:, 0].std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0,
                   P_inf=float(v[:, 1].mean()),
                   P_inf_sem=float(v[:, 1].std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0)
        row['D_ratio'] = row['sigma'] / row['P_inf']           # D_eff / D0
        row['time_factor'] = row['P_inf'] / row['sigma']       # predicted MFPT(p)/MFPT(0)
        row['dilute_sigma'] = 1 - np.pi * p
        row['dilute_time_factor'] = 1.0 / (1 - (np.pi - 1) * p) if (np.pi - 1) * p < 1 else None
        rows.append(row)
        print({k: (round(x, 5) if isinstance(x, float) else x) for k, x in row.items()}, f'[{time.time()-t0:.0f}s]', flush=True)
    res['finite_p'] = rows
    with open(_os.path.join(_R, 'data', 'msc_defects', 'homogenization.json'), 'w') as fh:
        json.dump(res, fh, indent=1)

if __name__ == '__main__':
    main()
