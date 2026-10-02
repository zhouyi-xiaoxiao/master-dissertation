"""s7_defects_density_dipole_check.py -- numerical check of Proposition `s7_defects_density:prop-dilute`
and of the power identity used in its proof.

(1) Potential kernel of the planar simple random walk, a(x) = (1/(2 pi)^2) int int (1 - cos(k.x)) / (1 - (cos k1 + cos k2)/2) dk,
    evaluated for x = (1,0) and (2,0) by reducing one integral analytically:
        a(m,0) = (2/pi) int_0^pi (1 - e^{-m phi(k)}) / sinh(phi(k)) dk ,   cosh phi = 2 - cos k
    (half of this is the two-point resistance R(m,0) of the infinite unit-resistor lattice, Cserti 2000),
    and compared with 1 and 4 - 8/pi (Spitzer, Principles of Random Walk).  The Green function used in the
    article is  -a/4 , so that  [Green(0) - Green(2,0)] = a(2,0)/4 = 1 - 2/pi.
(2) Finite check of the statement: unit-conductance square lattice of (2R+1) x (2R+1) sites, potential fixed to -E x_1
    on the two columns x_1 = -R, R (E = 1), free (reflecting) edges at x_2 = -R, R; the central site is removed.
    Reported: potential u at the neighbour +e_1 (claim: -pi/2 as R -> infinity), the reduction of the dissipated power
    W0 - W (claim: pi), and the identity  W0 - W = sum over removed bonds of V0_b V_b  (exact for every R).

Output: data/article/s7_defects_density_dipole_check.json.   Run time: a few seconds.
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import os, json
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.integrate import quad

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = _os.path.join(_R, 'data', 'article')


def kernel(m):
    def integrand(k):
        c = 2.0 - np.cos(k)
        phi = np.arccosh(c)
        if phi < 1e-8:
            return float(m)                       # limit (1 - e^{-m phi})/sinh(phi) -> m
        return (1.0 - np.exp(-m * phi)) / np.sinh(phi)
    val, err = quad(integrand, 0.0, np.pi, epsabs=1e-13, epsrel=1e-13, limit=400)
    return 2.0 * val / np.pi


def finite(R, E=1.0):
    n1 = 2 * R + 1
    idx = np.arange(n1 * n1).reshape(n1, n1)           # idx[i, j], x1 = i - R, x2 = j - R
    x1 = (np.arange(n1) - R)[:, None] * np.ones((1, n1))
    bonds = np.concatenate([np.column_stack([idx[:-1, :].ravel(), idx[1:, :].ravel()]),
                            np.column_stack([idx[:, :-1].ravel(), idx[:, 1:].ravel()])])
    centre = idx[R, R]
    removed = (bonds[:, 0] == centre) | (bonds[:, 1] == centre)
    fixed = np.zeros(n1 * n1, bool); fixed[idx[0, :]] = True; fixed[idx[-1, :]] = True
    phi_fix = (-E * x1).ravel()

    def solve(bd, drop_centre):
        n = n1 * n1
        W = sp.coo_matrix((np.ones(len(bd)), (bd[:, 0], bd[:, 1])), shape=(n, n)); A = (W + W.T).tocsr()
        Lap = (sp.diags(np.asarray(A.sum(1)).ravel()) - A).tocsr()
        free = ~fixed
        if drop_centre:
            free = free.copy(); free[centre] = False
        phi = np.where(fixed, phi_fix, 0.0)               # the isolated centre keeps the value 0
        fi = np.flatnonzero(free); fx = np.flatnonzero(fixed)
        rhs = -Lap[fi][:, fx] @ phi_fix[fx]
        phi[fi] = spla.spsolve(Lap[fi][:, fi].tocsc(), rhs)
        return phi

    phi0 = solve(bonds, False)
    phi = solve(bonds[~removed], True)
    V0 = phi0[bonds[:, 0]] - phi0[bonds[:, 1]]; V = phi[bonds[:, 0]] - phi[bonds[:, 1]]
    W0 = float((V0 ** 2).sum()); W = float((V[~removed] ** 2).sum())
    return dict(R=R, sites=n1 * n1, max_dev_phi0_from_linear=float(np.abs(phi0 - phi_fix).max()),
                u=float(phi[idx[R + 1, R]]), u_over_minus_half_pi=float(phi[idx[R + 1, R]] / (-np.pi / 2 * E)),
                phi_at_e2=float(phi[idx[R, R + 1]]),
                power_reduction=W0 - W, power_reduction_over_pi=(W0 - W) / (np.pi * E ** 2),
                sum_removed_V0V=float((V0[removed] * V[removed]).sum()),
                identity_abs_err=abs((W0 - W) - float((V0[removed] * V[removed]).sum())))


if __name__ == '__main__':
    res = dict(a10=kernel(1), a20=kernel(2), a10_exact=1.0, a20_exact=4 - 8 / np.pi)
    res['green_diff_2'] = res['a20'] / 4; res['one_minus_two_over_pi'] = 1 - 2 / np.pi
    res['finite'] = [finite(R) for R in (10, 20, 40, 80, 160)]
    json.dump(res, open(_os.path.join(_R, 'data', 'article', 's7_defects_density_dipole_check.json'), 'w'), indent=1)
    print('a(1,0) =', res['a10'], ' a(2,0) =', res['a20'], ' 4 - 8/pi =', res['a20_exact'])
    for r in res['finite']:
        print(r)
