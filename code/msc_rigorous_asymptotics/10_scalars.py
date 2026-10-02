"""Exact (float) pole/lattice scalars; check of the pole-data lemmas (Lemmas 3.x); tau* vs two-pole tau_c vs closed form."""
import json, sys
import numpy as np
from t3lib import scalars, exact_modes, mu_real, DATA
modes = exact_modes()
cases = [(2, N) for N in (5, 8, 10, 14, 20, 35, 50, 70, 100, 140, 200, 280, 400, 640, 1000, 1500)] + \
        [(3, N) for N in (5, 8, 10, 14, 20, 28, 40, 56, 80, 112, 160)]
out = []
for d, N in cases:
    s = scalars(d, N)
    W, Xs, sig, S1, S2 = s['W'], s['Xs'], s['sigma'], s['S1'], s['S2']
    nu0, nu1, th, Gam, rho0, b0 = s['nu0'], s['nu1'], s['theta'], s['Gam'], s['rho0'], s['b0']
    chk = {}
    # Lemma: nu0 bounds
    chk['nu0_lo'] = nu0 * Xs + nu0 ** 2 * sig <= 1 + 1e-12
    chk['nu0_hi'] = nu0 * Xs + nu0 ** 2 * sig / (1 - nu0) >= 1 - 1e-12
    chk['b0_lo'] = b0 >= 1 / (1 + nu0 ** 2 * sig / (1 - nu0) ** 2) - 1e-13
    chk['b0_hi'] = b0 <= 1 / (1 + nu0 ** 2 * sig) + 1e-13
    # Lemma: theta bounds
    Dth = Xs - W - 1 + S1
    thbar = W / Dth
    thlo = W / (Dth + thbar / (1 + thbar) + S1 * 2 * thbar / (1 - thbar)) if thbar < 1 else float('nan')
    chk['theta'] = (thlo <= th * (1 + 1e-12)) and (th <= thbar * (1 + 1e-12))
    # Lemma: Gamma bounds   1/Gam = Xs - (1-th)/(1+th) + nu1 [S(nu1) + th S_2(nu1)]
    iG = 1 / Gam
    iG_lo = Xs - (1 - th) / (1 + th) + nu1 * (S1 + th * S2)
    iG_hi = Xs - (1 - th) / (1 + th) + nu1 * (S1 / (1 - th) + th * S2 / (1 - th) ** 2)
    chk['Gam'] = (iG_lo <= iG * (1 + 1e-11)) and (iG <= iG_hi * (1 + 1e-11))
    chk['delta2'] = (s['delta2'] >= -1e-14) and (s['delta2'] <= 2 * s['beta2'] + 1e-14) and s['beta2'] >= s['delta2'] / 2 - 1e-14
    chk['m0'] = (s['m0'] >= 1 + nu0 * s['EU'] - 1e-12) and (s['m0'] <= W ** nu0 / (1 - nu0) + 1e-12)
    chk['EU'] = (s['eH'] <= s['EU'] + 1e-12) and (s['EU'] <= d * s['eH'] + 1e-12)
    # two-pole crossing and closed form
    Lam = W * Gam * nu1 * (1 + th * s['k1']) / (rho0 * nu0 * s['m0'])
    tau_c = np.log(Lam) / (nu1 - nu0)
    X = s['X']
    tau_cf = X * np.log(2 * d * X) / (X + 2 * d - 1)
    rec = dict(d=d, N=N, Xs=Xs, X=X, sigma=sig, S1=S1, S2=S2, EU=s['EU'], k1=s['k1'], m0=s['m0'], theta_Xs_over_W=th * Xs / W,
               delta2_Xs2=s['delta2'] * Xs ** 2, beta2_Xs2=s['beta2'] * Xs ** 2, tau_c=tau_c, tau_cf=tau_cf,
               thbar_minus_thlo=thbar - thlo, checks_ok=all(chk.values()), failed=[k for k, v in chk.items() if not v])
    if (d, N) in modes:
        mode, mfpt = modes[(d, N)]
        mur = mu_real(d, N)
        rec['tau_star'] = mode * mur
        rec['X_from_mfpt'] = mfpt * mur
        rec['X_(tau_c-tau_star)'] = X * (tau_c - rec['tau_star'])
        rec['X_(tau_star-tau_cf)'] = X * (rec['tau_star'] - tau_cf)
    out.append(rec)
    print(json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)
json.dump(out, open(DATA + '/10_scalars.json', 'w'), indent=1)
