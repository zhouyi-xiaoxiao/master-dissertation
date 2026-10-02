"""Validation (not part of the proof): the rigorous enclosures of certlib (Sections 3-4) must contain the exact
(float) values for every tested N, and the certified constants c_+/c_- must enclose X(tau*-tau_cf) from the exact modes."""
import json, sys
import numpy as np
from flint import arb
from r4lib import scalars, exact_modes_C, mu_real, DATA
import certlib as C
from certlib import L, U, iv

def inside(ball, val, name, bad, tol=1e-9):
    lo, hi = float(L(ball)), float(U(ball))
    if not (lo - tol * (1 + abs(lo)) <= val <= hi + tol * (1 + abs(hi))):
        bad.append((name, lo, val, hi))

mC = exact_modes_C()
N0 = {2: 10, 3: 10}
cases = [(2, N) for N in (10, 15, 20, 28, 50, 100, 200, 400, 905, 1810)] + [(3, N) for N in (10, 14, 20, 35, 50, 80, 160)]
out = []
for d, N in cases:
    s = scalars(d, N)
    y = arb(1) / arb(float(s['Xs']))
    bad = []
    # X_s bounds
    xl, xu = float(C.Xs_lower(d, N)), float(C.Xs_upper(d, N))
    if not (xl <= s['Xs'] <= xu): bad.append(('Xs', xl, s['Xs'], xu))
    yb = arb(1) / arb(float(s['Xs'])) ; Nlo, Nhi = C.N_range(d, yb * (1 - 1e-12), yb * (1 + 1e-12), N0[d])
    assert Nlo <= N and (Nhi is None or N <= Nhi)
    E = C.enclosures(d, y, Nlo, Nhi)
    inside(E['nu0'], s['nu0'], 'nu0', bad)
    inside(E['EU'], s['EU'], 'EU', bad)
    th = s['theta']
    e_exact = th / (1 + th) + (1 + th) * s['Snu']
    inside(E['e'], e_exact, 'e', bad)
    inside(E['theta'], th, 'theta', bad)
    inside(E['a1_over_y'], s['a1'] * s['Xs'], 'a1/y', bad)
    inside(E['zeta'], 1 / (s['W'] * (1 + th) ** 2) + s['S2nu'] / s['W'], 'zeta', bad)
    T1_exact = s['S1'] - s['W2'] / 2 - (s['W3'] / 6 if d == 3 else 0)
    inside(E['T1'], T1_exact, 'T1', bad)
    inside(E['betahat_over_y'], s['betahat'] * s['Xs'], 'betahat/y', bad)
    lnm0 = np.log(s['m0'])
    if not (s['nu0'] * float(L(E['EU'])) - 1e-12 <= lnm0 <= s['nu0'] * float(E['lnm0_over_nu0_hi']) + 1e-12): bad.append(('lnm0', lnm0))
    if s['near_nu'][0] - 2 > float(E['th2']) + 1e-12: bad.append(('th2', s['near_nu'][0] - 2, float(E['th2'])))
    if s['near_nu'][0] - 2 < float(E['th2lo']) - 1e-12: bad.append(('th2lo', s['near_nu'][0] - 2, float(E['th2lo'])))
    if s['near_rho'][0] * s['Xs'] > float(E['rho2_over_y']) + 1e-12: bad.append(('rho2',))
    if d == 3:
        if s['near_nu'][1] - 3 > float(E['th3']) + 1e-12: bad.append(('th3',))
        if s['near_nu'][1] - 3 < float(E['th3lo']) - 1e-12: bad.append(('th3lo',))
        if s['near_rho'][1] * s['Xs'] > float(E['rho3_over_y']) + 1e-12: bad.append(('rho3',))
    inside(E['xi'], np.sin(s['x']) ** 2 * s['Xs'], 'xi', bad)
    rec = dict(d=d, N=N, Nlo=float(Nlo), Nhi=(None if Nhi is None else float(Nhi)), Xs=float(s['Xs']), Xs_lo=xl, Xs_hi=xu, e=[float(L(E['e'])), e_exact, float(U(E['e']))],
               theta=[float(L(E['theta'])), th, float(U(E['theta']))], EU=[float(L(E['EU'])), float(s['EU']), float(U(E['EU']))],
               T1=[float(L(E['T1'])), T1_exact, float(U(E['T1']))], th2=[float(E['th2lo']), s['near_nu'][0] - 2, float(E['th2'])], bad=bad)
    # constants c_plus, c_minus at this thin y (N-range from the y-ball)
    cp, okp = C.find_c(E, '+'); cm, okm = C.find_c(E, '-')
    rec.update(c_plus=cp, ok_plus=okp, c_minus=cm, ok_minus=okm)
    if (d, N) in mC:
        ts = mC[(d, N)][0] * mu_real(d, N)
        rec['X(tau*-tau_cf)'] = float(s['X'] * (ts - s['tau_cf']))
        rec['enclosed'] = bool(-cm < rec['X(tau*-tau_cf)'] < cp)
    out.append(rec)
    print(json.dumps({k: ([round(float(z), 4) for z in v] if isinstance(v, list) and k != 'bad' else (round(v, 4) if isinstance(v, float) else v)) for k, v in rec.items()}), flush=True)
json.dump(out, open(DATA + '/51_validate_enclosures.json', 'w'), indent=1, default=str)
