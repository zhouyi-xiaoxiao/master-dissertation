"""Theorem A (semi-explicit window) evaluated with exact (float) scalars, both time scales, against exact modes.

Continuous: maximiser tau* (scaled) must lie in [s_minus, s_plus].
Discrete (activity q): maximiser t* must satisfy ceil(tau_-)+1 <= t* <= ceil(tau_+)+1.
"""
import json, sys
import numpy as np
from scipy.optimize import brentq
from u3lib import scalars, make_E, mu_real, LB, UB, Qlow, exact_modes_C, exact_modes_D, DATA

S1 = {2: 1.3, 3: 1.6}          # scaled early time tau_1


def window(s, scale, q):
    d, N = s['d'], s['N']
    mu = mu_real(d, N, q) if scale == 'D' else 1.0          # continuous: work in scaled time (mu = 1)
    E = make_E(scale, mu)
    W, nu0, nu1 = s['W'], s['nu0'], s['nu1']
    nq = Rr = 0.0
    if scale == 'D' and q > 0.5:
        Rr = s['pia'] * q / 2                               # real-unit bound for the sum of rho_j over negative poles
        nq = Rr * (W / (1 - mu) + (2 * s['C1'] + s['C2'] * s['eps2']) / (2 * (1 - q - mu)))
    negv = Rr / (2 * (1 - q)) if scale == 'D' and q > 0.5 else 0.0     # value-bound correction (scaled g)
    tt = lambda sc: (sc / mu)                               # scaled -> real time
    Phi = lambda tau: LB(s, E, tau, nq) / E(nu1, tau)
    tautil = lambda tau: mu * tau if scale == 'C' else mu * tau / (1 - 2 * mu)
    Psi = lambda tau: UB(s, E, tau, tautil(tau), mu, nq) / E(nu0, tau)
    t1 = tt(S1[d])
    if scale == 'D':
        t1 = float(np.ceil(t1))
    out = dict(H1=bool(Phi(t1) > 0), Phi_t1_rel=float(Phi(t1) / s['a1']))
    if scale == 'C':
        nuhat = nu1 - nu0
    else:
        nuhat = np.log((1 - nu0 * mu) / (1 - nu1 * mu)) / mu      # per scaled time
    sc_c = np.log(s['a1'] / s['a0']) / nuhat                     # two-pole crossing (scaled)
    out['s_c'] = float(sc_c)
    if not out['H1']:
        return out
    # tau_minus: root of Phi between t1 and beyond (concave => unique sign change after t1)
    hi = tt(sc_c * 1.05 + 1)
    if Phi(hi) > 0:
        out['err'] = 'Phi>0 at hi'; return out
    tm = brentq(Phi, t1, hi, xtol=1e-10 * hi, rtol=1e-14)
    # tau_plus: Psi decreasing beyond scaled time 1; find sign change scanning from s_c - 1
    lo = tt(max(1.0, sc_c - 1.0)); hi2 = tt(sc_c + 6)
    if Psi(lo) <= 0 or Psi(hi2) >= 0:
        out['err'] = 'Psi bracket'; out['Psi_lo'] = float(Psi(lo)); out['Psi_hi'] = float(Psi(hi2)); return out
    tp = brentq(Psi, lo, hi2, xtol=1e-10 * hi2, rtol=1e-14)
    out.update(s_minus=float(mu * tm), s_plus=float(mu * tp), tau_minus=float(tm), tau_plus=float(tp))
    # value condition
    lhs = 1 - Qlow(s, E, t1) + nu0 / (1 - nu0) + negv / s['rho0']
    rhs = E(nu0, tm) * (1 - W * E(1.0, tm)) - negv / s['rho0']
    out['H3'] = bool(lhs < rhs); out['value_lhs'] = float(lhs); out['value_rhs'] = float(rhs)
    out['P2rel'] = float((Psi(tp) - (s['a1'] * E(nu1, tp) / E(nu0, tp) - s['a0'])) / s['a0'])
    return out


if __name__ == '__main__':
    mC, mD = exact_modes_C(), exact_modes_D()
    out = []
    NC = {2: [5, 10, 14, 20, 35, 50, 70, 100, 140, 200, 280, 400, 640, 1000, 1280],
          3: [5, 8, 10, 14, 20, 28, 35, 40, 50, 80, 100, 160]}
    for d in (2, 3):
        for N in NC[d]:
            s = scalars(d, N)
            w = window(s, 'C', 1.0)
            X = s['X']
            tcf = X * np.log(2 * d * X) / (X + 2 * d - 1)
            rec = dict(d=d, N=N, scale='C', Xs=s['Xs'], X=X, s_cf=float(tcf), **w)
            if (d, N) in mC and 's_minus' in w:
                ts = mC[(d, N)][0] * mu_real(d, N)
                rec['s_star'] = float(ts)
                rec['inside'] = bool(w['s_minus'] <= ts <= w['s_plus'])
                rec['X(s*-s_cf)'] = float(X * (ts - tcf))
            if 's_minus' in w:
                rec['X(s_minus-s_cf)'] = float(X * (w['s_minus'] - tcf)); rec['X(s_plus-s_cf)'] = float(X * (w['s_plus'] - tcf))
            out.append(rec)
            print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)
    # discrete
    for (d, N, q), (mode, mfpt) in sorted(mD.items()):
        if d == 1 or N < 5 or (d == 2 and N > 1300) or (d == 3 and N > 160):
            continue
        s = scalars(d, N)
        okN = (q <= 0.5) or (q < 1 and N >= np.pi * q / (1 - q))
        if q >= 1:
            continue
        w = window(s, 'D', q)
        rec = dict(d=d, N=N, scale='D', q=q, Xs=s['Xs'], N_cond=bool(okN), mode=mode, **w)
        if 'tau_minus' in w:
            lo_t = int(np.ceil(w['tau_minus'])) + 1; hi_t = int(np.ceil(w['tau_plus'])) + 1
            rec['t_lo'] = lo_t; rec['t_hi'] = hi_t; rec['inside'] = bool(lo_t <= mode <= hi_t)
        out.append(rec)
        print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)
    json.dump(out, open(DATA + '/22_windows.json', 'w'), indent=1)
