"""v09b -- test of the two-time-scale heuristic t_mode = tau ln[c(1+T/tau)] for target-local perturbations."""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd, scipy.linalg as la
import vcore as vc, v06_structured as v6
HERE = os.path.dirname(os.path.abspath(__file__)); D = _os.path.join(_R, 'data', 'msc_defects_verify')
N, q = 35, 0.8

def spec(mask):
    idx, n = vc.cluster_index(~mask, (0, 0)); a = idx[N - 1, N - 1]
    P = vc.transition(idx, n, q); Q, r, _ = vc.reduce_target(P, a)
    lam, V = la.eigh(Q.toarray()); w = V[0] * (V.T @ r); o = np.argsort(-lam)
    T1 = -1 / np.log(lam[o[0]])
    rec = vc.solve_config(~mask, q)
    # leading correction: effective (c, tau) from the three slowest sub-leading modes, weight-averaged rate
    sub = o[1:8]
    return dict(T1=T1, w1=w[o[0]], taus=[-1 / np.log(lam[k]) for k in sub], rel_w=[w[k] / w[o[0]] for k in sub], mfpt=rec['mfpt'], mode=rec['mode'], peak=rec['peak_sub'])

cl = spec(np.zeros((N, N), bool))
T0, t0 = cl['T1'], cl['peak']
out = dict(clean=cl)
def pred(lmbda, tau, c=None):
    if c is None:                      # calibrate c on the clean mode
        c = np.exp(t0 / tau) / (1 + T0 / tau)
    return tau * np.log(c * (1 + lmbda * T0 / tau)) / (tau * np.log(c * (1 + T0 / tau))), c
cases = {}
m = np.zeros((N, N), bool); m[N - 2, N - 1] = True; cases['one target neighbour'] = m
dd = v6.det()
for k in ('target_box_R3_g1', 'target_box_R5_g1', 'target_box_R8_g1', 'target_box_R5_g3', 'wall_neartarget_gap_centre_g3'):
    cases[k] = dd[k]
rows = []
for name, mask in cases.items():
    s = spec(mask)
    lamT = s['T1'] / T0
    pa, ca = pred(lamT, 620.5089500817162)                 # main implementation: tau = tau_rel, c calibrated
    pb, cb = pred(lamT, 539.5624285475965, 4.1541906451698205)   # first symmetric mode of the clean lattice
    pb2 = pb * (539.5624285475965 * np.log(4.1541906451698205 * (1 + T0 / 539.5624285475965))) / t0
    rows.append(dict(case=name, mfpt_fac=s['mfpt'] / cl['mfpt'], T1_fac=lamT, mode_fac_exact=s['peak'] / t0, elasticity_exact=np.log(s['peak'] / t0) / np.log(s['mfpt'] / cl['mfpt']),
                     pred_tau621=pa, elast_tau621=np.log(pa) / np.log(lamT), pred_tau540=pb, elast_tau540=np.log(pb) / np.log(lamT),
                     tau_sub1=s['taus'][0], relw_sub1=s['rel_w'][0], tau_sub2=s['taus'][1], relw_sub2=s['rel_w'][1]))
df = pd.DataFrame(rows); pd.set_option('display.width', 250)
print(df.round(4).to_string())
df.to_csv(_os.path.join(_R, 'data', 'msc_defects_verify', 'v09b_twotime.csv'), index=False)
print('clean: T1=%.1f peak=%.1f c(tau=621 calibrated)=%.3f ; local elasticity formulas: tau621 -> %.3f, tau540 -> %.3f' % (
    T0, t0, pred(1, 620.5089500817162)[1], 620.509 / t0 * (T0 / 620.509) / (1 + T0 / 620.509), 539.56 / t0 * (T0 / 539.56) / (1 + T0 / 539.56)))
