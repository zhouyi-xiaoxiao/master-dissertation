"""Sanity check (float) of Lemma 5.2: the exact decomposition
   [ln Lambda - (nu1-nu0) t]/y = T_I + T_II + T_III + T_IV + T_V          at t = tau_cf + c/X,
and of the fact that every exact piece lies in the corresponding ball computed by certlib.Psi (block = {N}).
Also checks the exact correction terms against the balls: ln(1+theta*kappa(t)/W + P_rest e^{nu1 t}/a1)/y etc."""
import json
import numpy as np
from flint import arb
from r4lib import scalars, spec, DATA, kappa_hi, kappa_lo, Prest_hi, Eminus_hi, Eminus_lo
import certlib as C
from certlib import L, U, iv

out = []
for d, N, c in [(2, 12, 3.0), (2, 12, -2.0), (2, 30, 4.0), (2, 200, 1.0), (2, 1000, -1.0), (3, 10, 2.0), (3, 25, -1.0), (3, 60, 1.5), (3, 120, 0.5)]:
    s = scalars(d, N)
    y, W, EU, X = s['y'], s['W'], s['EU'], s['X']
    nu0, nu1, th = s['nu0'], s['nu1'], s['theta']
    e = th / (1 + th) + (1 + th) * s['Snu']
    zeta = 1 / (W * (1 + th) ** 2) + s['S2nu'] / W
    sv = 1 - nu0 / y
    yp = 1 / X
    tcf = s['tau_cf']
    t = tcf + c * yp
    lnLam = np.log(s['a1'] / s['a0'])
    lhs = (lnLam - (nu1 - nu0) * t) / y
    TI = -np.log(1 + (e - 1) * y) / y - np.log(1 + th ** 2 * zeta) / y + np.log(1 / s['b0']) / y - 2 * np.log(1 - sv) / y
    TII = -(np.log(s['m0']) + np.log(1 + EU * y)) / y
    TIII = np.log(np.cos(s['x']) ** 2) / y
    a_ = W + 1 - e
    xi = np.sin(s['x']) ** 2 / y
    q4 = -(2 * d - 1) * EU / (1 + EU * y) - W * a_ / (1 - a_ * y) - sv / y
    TIV = y * tcf * (2 * d * xi + q4)
    TV = -c * (1 + th - nu0) * yp / y
    rhs = TI + TII + TIII + TIV + TV
    # identity (a) of Lemma 3.3 and (d)
    idA = abs(W / th - (s['Xs'] - W - 1 + e)); idD = abs(lnLam - (np.log(W * s['Xs']) - np.log(1 + (e - 1) * y) - np.log(1 + th ** 2 * zeta) + np.log(1 / s['b0']) - 2 * np.log(1 - sv) - np.log(s['m0'])))
    # balls
    yb = iv(float(C.L(1 / arb(C.Xs_upper(d, N)))), float(C.U(1 / arb(C.Xs_lower(d, N)))))
    E = C.enclosures(d, yb, N, N, 24)
    bad = []
    for side in ('+', '-'):
        tot, aux = C.Psi(E, c, side)
        for name, val in (('T1', TI), ('T13', TII), ('T2', TIII), ('T4', TIV), ('T5', TV)):
            lo, hi = float(L(aux[name])), float(U(aux[name]))
            if not (lo - 1e-9 <= val <= hi + 1e-9): bad.append((side, name, lo, val, hi))
        # exact-data correction terms (still upper/lower bounds, but with exact pole data) must be dominated by the balls
        if side == '+':
            T6_exact_bound = (np.log(1 + th * kappa_hi(s) / W + Prest_hi(s, t) * np.exp(nu1 * t) / s['a1']) - np.log(1 + Eminus_lo(s, t))) / y
            if T6_exact_bound > float(U(aux['T6'])) + 1e-9: bad.append(('T6+', T6_exact_bound, float(U(aux['T6']))))
        else:
            T6_exact_bound = (np.log(1 + th * kappa_lo(s, t) / W) - np.log(1 + Eminus_hi(s, t))) / y
            if T6_exact_bound < float(L(aux['T6'])) - 1e-9: bad.append(('T6-', T6_exact_bound, float(L(aux['T6']))))
    rec = dict(d=d, N=N, c=c, lhs=float(lhs), rhs=float(rhs), err=float(abs(lhs - rhs)), id_a=float(idA), id_d=float(idD), bad=bad)
    out.append(rec); print(json.dumps(rec), flush=True)
json.dump(out, open(DATA + '/53_check_decomposition.json', 'w'), indent=1)
