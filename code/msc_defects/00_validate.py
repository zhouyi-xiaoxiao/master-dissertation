"""00_validate.py -- independent cross-checks of the exact machinery in fptcore.py.

Checks (all written to data/validation.json):
 V1  uniform 2D, N=35, q=0.8, corner->corner: MFPT by (a) sparse solve, (b) spectral sum of the
     absorbing chain, (c) cosine double sum of the reflecting chain (independent formula);
     mode by (a) certified dense spectral argmax, (b) plain time stepping.
 V2  one random blocked-site configuration (N=35, p=0.15): MFPT by solve / spectral / Tetali
     resistance formula; mode by dense spectral vs time stepping with Lanczos certificate.
 V3  Monte Carlo of the literal walker rule (random direction, cancel moves into walls or blocked
     sites) on N=15, p=0.15; compare empirical mean/sd/PMF with the exact values.
Deterministic: all seeds fixed.
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import json, time, sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import fptcore as fc

OUT = _os.path.join(_R, 'data', 'msc_defects', 'validation.json')
res = {}

def cosine_double_sum_mfpt(N, q, s, t):
    # reflecting lazy walk on {0..N-1}^2, eigenvectors cos(pi k (x+1/2)/N)
    k = np.arange(N)
    c = np.cos(np.pi * k / N)
    def vec(x):
        v = np.cos(np.pi * k * (x + 0.5) / N) * np.sqrt(2.0 / N)
        v[0] = 1.0 / np.sqrt(N)
        return v
    sx, sy = vec(s[0]), vec(s[1]); tx, ty = vec(t[0]), vec(t[1])
    mu = 1 - (q / 2) * (2 - c[:, None] - c[None, :])
    num = np.outer(tx, ty) ** 2 - np.outer(sx, sy) * np.outer(tx, ty)
    num[0, 0] = 0.0; den = 1 - mu; den[0, 0] = 1.0
    return float(N * N * np.sum(num / den))

# ---------------- V1
N, q = 35, 0.8
s, t = (0, 0), (N - 1, N - 1)
om = np.ones((N, N), bool)
t0 = time.time()
summ, ch = fc.fpt_summary(om, q, s, t)
tdense = time.time() - t0
mf_cos = cosine_double_sum_mfpt(N, q, s, t)
f_ts, S_ts = fc.pmf_timestep(ch['Q'], ch['r'], ch['s'], 60000)
mode_ts = int(np.argmax(f_ts)) + 1
tet = fc.tetali_parts(ch, q)
res['V1_uniform_N35'] = dict(mfpt_solve=summ['mfpt'], mfpt_spectral=summ['mfpt_spectral'],
                             mfpt_cosine_double_sum=mf_cos, mfpt_tetali=tet['mfpt_tetali'],
                             mode_certified=summ['mode'], mode_timestep=mode_ts,
                             ratio=summ['ratio'], median=summ['median'], sd=summ['sd'],
                             S_mode=summ['S_mode'], S_mfpt=summ['S_mfpt'],
                             t_cert=summ['t_cert'], seconds_dense=tdense,
                             sum_f_timestep_to_60000=float(f_ts.sum()),
                             mean_from_timestep_series=float(np.sum(np.arange(1, 60001) * f_ts) + 60000 * S_ts[-1]))
print('V1', json.dumps(res['V1_uniform_N35'], indent=1))

# ---------------- V2
rng = np.random.default_rng(20261001)
p = 0.15
while True:
    M = int(round(p * N * N))
    cand = np.setdiff1d(np.arange(N * N), [0, N * N - 1])
    blk = rng.choice(cand, M, replace=False)
    om = np.ones(N * N, bool); om[blk] = False; om = om.reshape(N, N)
    if fc.connected(om, s, t):
        break
summ2, ch2 = fc.fpt_summary(om, q, s, t)
tet2 = fc.tetali_parts(ch2, q)
ts2 = fc.pmf_stats_timestep(ch2['Q'], ch2['r'], ch2['s'], summ2['mfpt'])
res['V2_random_N35_p015'] = dict(n_cluster=summ2['n_cluster'], mfpt_solve=summ2['mfpt'],
                                 mfpt_spectral=summ2['mfpt_spectral'], mfpt_tetali=tet2['mfpt_tetali'],
                                 mode_dense=summ2['mode'], mode_timestep=ts2['mode'],
                                 f_mode_dense=summ2['f_mode'], f_mode_timestep=ts2['f_mode'],
                                 t_cert_dense=summ2['t_cert'], t_cert_timestep=ts2['t_cert'],
                                 lam1_dense=summ2['lam1'], lam1_lanczos=ts2['lam1'])
print('V2', json.dumps(res['V2_random_N35_p015'], indent=1))

# ---------------- V3  Monte Carlo of the literal rule
Nm, pm = 15, 0.15
rng = np.random.default_rng(7)
sm, tm = (0, 0), (Nm - 1, Nm - 1)
while True:
    M = int(round(pm * Nm * Nm))
    cand = np.setdiff1d(np.arange(Nm * Nm), [0, Nm * Nm - 1])
    blk = rng.choice(cand, M, replace=False)
    om3 = np.ones(Nm * Nm, bool); om3[blk] = False; om3 = om3.reshape(Nm, Nm)
    if fc.connected(om3, sm, tm):
        break
summ3, ch3 = fc.fpt_summary(om3, q, sm, tm, want_series=True, series_tmax=20000)
W = 200000
x = np.zeros(W, np.int64); y = np.zeros(W, np.int64)
T = np.zeros(W, np.int64); alive = np.ones(W, bool)
dirs = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
step = 0
t0 = time.time()
idx = np.arange(W)
while idx.size:
    step += 1
    m = idx.size
    u = rng.random(m)
    move = u < q
    dsel = rng.integers(0, 4, m)
    nx = x[idx] + np.where(move, dirs[dsel, 0], 0)
    ny = y[idx] + np.where(move, dirs[dsel, 1], 0)
    inside = (nx >= 0) & (nx < Nm) & (ny >= 0) & (ny < Nm)
    ok = inside.copy()
    ok[inside] = om3[nx[inside], ny[inside]]
    x[idx] = np.where(ok, nx, x[idx]); y[idx] = np.where(ok, ny, y[idx])
    hit = (x[idx] == tm[0]) & (y[idx] == tm[1])
    T[idx[hit]] = step
    idx = idx[~hit]
tmc = time.time() - t0
emp_mean, emp_sd = float(T.mean()), float(T.std(ddof=1))
se = emp_sd / np.sqrt(W)
# binned PMF comparison (chi-square on 40 equiprobable-ish bins)
edges = np.unique(np.quantile(T, np.linspace(0, 1, 41)).astype(int))
edges[0] = 0; edges[-1] = max(edges[-1], T.max()) + 1
cumF = np.concatenate([[0.0], np.cumsum(summ3['series_f'])])  # cumF[k] = P(T<=k)
def P_le(k):
    k = min(int(k), len(cumF) - 1)
    return cumF[k]
obs, exp = [], []
for a, b in zip(edges[:-1], edges[1:]):
    obs.append(int(np.sum((T > a) & (T <= b))))
    exp.append(W * (P_le(b) - P_le(a)))
obs = np.array(obs); exp = np.array(exp)
chi2 = float(np.sum((obs - exp) ** 2 / exp)); dof = len(obs) - 1
from scipy.stats import chi2 as chi2d
res['V3_montecarlo_N15_p015'] = dict(walkers=W, exact_mfpt=summ3['mfpt'], exact_sd=summ3['sd'],
                                     mc_mean=emp_mean, mc_se=float(se), z=float((emp_mean - summ3['mfpt']) / se),
                                     mc_sd=emp_sd, chi2=chi2, dof=dof, p_value=float(chi2d.sf(chi2, dof)),
                                     exact_mode=summ3['mode'], seconds=tmc, n_blocked=int((~om3).sum()),
                                     tail_mass_beyond_series=float(1 - cumF[-1]))
print('V3', json.dumps(res['V3_montecarlo_N15_p015'], indent=1))
with open(OUT, 'w') as fh:
    json.dump(res, fh, indent=1)
print('wrote', OUT)
