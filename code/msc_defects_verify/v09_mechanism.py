"""v09 -- mechanism checks on the data of this check: correlations, stratification by target degree, elasticities,
relaxation time / chemical distance, CV-vs-ratio family, two-mode structure of the clean PMF.
Output: data/v09_mechanism.json"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import os, sys, glob, json
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd, scipy.linalg as la, scipy.sparse.linalg as spla, scipy.sparse as sp
from scipy.stats import spearmanr, pearsonr
from collections import deque
import vcore as vc
HERE = os.path.dirname(os.path.abspath(__file__)); D = _os.path.join(_R, 'data', 'msc_defects_verify')
res = {}
ci = lambda x: float(1.96 * np.std(x, ddof=1) / np.sqrt(len(x)))
N, q = 35, 0.8
cl = dict(mfpt=14100.808049917203, mode=2493.0, ratio=2493 / 14100.808049917203, cv=0.9182769845672402)
# --- correlations & stratification
for p in (0.10, 0.20):
    df = pd.read_csv(os.path.join(_R, 'data', 'msc_defects_verify', 'v03_sweep', f'N35_uniform_p{p:.2f}.csv'))
    r = {}
    for col in ('G_aa', 'G_sa', 'blk_t6', 'blk_s6', 'n', 'deg_t'):
        r[col] = dict(mfpt=float(spearmanr(df[col], df.mfpt)[0]), mode=float(spearmanr(df[col], df['mode'])[0]), ratio=float(spearmanr(df[col], df.ratio)[0]))
    b = np.polyfit(np.log(df.mfpt), np.log(df['mode']), 1)
    r['loglog_slope_mode_on_mfpt'] = float(b[0])
    r['identity_max_rel_err'] = float(np.max(np.abs(df.mfpt_res / df.mfpt - 1)))
    for sch in ('uniform', 'cornerthinned'):
        d2 = pd.read_csv(os.path.join(_R, 'data', 'msc_defects_verify', 'v03_sweep', f'N35_{sch}_p{p:.2f}.csv'))
        for k, g in d2.groupby('deg_t'):
            r[f'{sch}_deg{k}'] = dict(K=len(g), mfpt=float(g.mfpt.mean()), mfpt_ci=ci(g.mfpt), mode=float(g['mode'].mean()), mode_ci=ci(g['mode']), ratio=float(g.ratio.mean()), ratio_ci=ci(g.ratio))
    res[f'p{p:.2f}'] = r
allid = []
for fn in glob.glob(_os.path.join(_R, 'data', 'msc_defects_verify', 'v03_sweep', '*.csv')):
    d2 = pd.read_csv(fn); allid.append(np.max(np.abs(d2.mfpt_res / d2.mfpt - 1)))
res['identity_max_rel_err_all_10400'] = float(max(allid))
# --- relaxation time and chemical distance for 100 uniform placements at p = 0.2 (seeds of this check, regenerated)
def chem(om):
    dist = -np.ones(om.shape, int); dist[0, 0] = 0; dq = deque([(0, 0)])
    while dq:
        x, y = dq.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            u, v = x + dx, y + dy
            if 0 <= u < N and 0 <= v < N and om[u, v] and dist[u, v] < 0:
                dist[u, v] = dist[x, y] + 1; dq.append((u, v))
    return dist[N - 1, N - 1]
for p in (0.10, 0.20, 0.30):
    taus = []; cds = []
    for rep in range(100):
        rng = np.random.default_rng(np.random.SeedSequence([987654321, N, 11, int(round(p * 1e4)), rep])); M = int(round(p * N * N))
        while True:
            sites = rng.choice(N * N - 2, M, replace=False) + 1
            m = np.zeros(N * N, bool); m[sites] = True; m = m.reshape(N, N)
            if vc.connected(~m, (0, 0), (N - 1, N - 1)): break
        idx, n = vc.cluster_index(~m, (0, 0))
        Lp = vc.laplacian(idx, n).toarray()
        mu = la.eigvalsh(Lp, subset_by_index=[1, 1])[0]
        taus.append(-1 / np.log(1 - (q / 4) * mu)); cds.append(chem(~m))
    res[f'relax_p{p:.2f}'] = dict(tau_rel_mean=float(np.mean(taus)), tau_factor=float(np.mean(taus) / 620.5089500817162), tau_factor_ci=ci(np.array(taus) / 620.5089500817162),
                                  chem_mean=float(np.mean(cds)), chem_max=int(np.max(cds)), frac_chem_68=float(np.mean(np.array(cds) == 68)))
# --- CV vs ratio family
recheck = [pd.read_csv(f)[['ratio', 'cv']] for f in glob.glob(_os.path.join(_R, 'data', 'msc_defects_verify', 'v03_sweep', '*.csv'))]
recheck += [pd.read_csv(_os.path.join(_R, 'data', 'msc_defects_verify', 'v06_local.csv'))[['ratio', 'cv']], pd.read_csv(_os.path.join(_R, 'data', 'msc_defects_verify', 'v06_det.csv'))[['ratio', 'cv']]]
A = pd.concat(recheck)
co = np.polyfit(A.ratio, A.cv, 2); rr = A.cv - np.polyval(co, A.ratio)
res['family_recheck'] = dict(n=len(A), pearson=float(pearsonr(A.ratio, A.cv)[0]), quad=[float(x) for x in co], rms=float(np.sqrt(np.mean(rr ** 2))), max_abs_resid=float(np.max(np.abs(rr))),
                          frac_resid_gt_0p02=float(np.mean(np.abs(rr) > 0.02)))
TD = _os.path.join(_R, 'data', 'msc_defects')
tr = [pd.read_csv(f)[['ratio', 'cv']] for f in glob.glob(_os.path.join(_R, 'data', 'msc_defects', 'sweep', '*.csv'))] + [pd.read_csv(_os.path.join(_R, 'data', 'msc_defects', 'structured_local.csv'))[['ratio', 'cv']], pd.read_csv(_os.path.join(_R, 'data', 'msc_defects', 'structured_deterministic.csv'))[['ratio', 'cv']]]
T = pd.concat(tr); rt = T.cv - np.polyval([-0.2461, -0.4564, 1.0073], T.ratio)
res['family_main_implementation'] = dict(n=len(T), pearson=float(pearsonr(T.ratio, T.cv)[0]), rms_with_main_implementation_quadratic=float(np.sqrt(np.mean(rt ** 2))), max_abs=float(np.max(np.abs(rt))))
cc = pd.read_csv(_os.path.join(_R, 'data', 'msc_defects_verify', 'v07_clean.csv'))
res['family_clean_resid_main_implementation_quadratic'] = {int(n_): float(c_ - np.polyval([-0.2461, -0.4564, 1.0073], r_)) for n_, r_, c_ in zip(cc.N, cc.ratio, cc.cv)}
res['family_1d_point_resid'] = float(np.sqrt(2 / 3) - np.polyval([-0.2461, -0.4564, 1.0073], 1 / 3))
# --- two-mode structure of the clean PMF
om = np.ones((N, N), bool); idx, n = vc.cluster_index(om, (0, 0)); a = idx[N - 1, N - 1]
P = vc.transition(idx, n, q); Q, r, _ = vc.reduce_target(P, a)
lam, V = la.eigh(Q.toarray()); w = V[0] * (V.T @ r)
order = np.argsort(-lam)
top = [(float(lam[k]), float(w[k]), float(-1 / np.log(lam[k]))) for k in order[:12]]
res['clean_top_modes_lam_w_tau'] = top
sig = [k for k in order if abs(w[k]) > 1e-12 * abs(w[order[0]])]
k1, k2 = sig[0], sig[1]
T1 = -1 / np.log(lam[k1]); tau2 = -1 / np.log(lam[k2])
t = np.arange(1, 20001)
f2 = w[k1] * lam[k1] ** (t - 1) + w[k2] * lam[k2] ** (t - 1)
res['two_mode'] = dict(T1=float(T1), tau2=float(tau2), w1=float(w[k1]), w2=float(w[k2]), mode_two_mode_truncation=int(t[np.argmax(f2)]),
                       # effective tau from the heuristic formula f ~ e^{-t/T}[1 - c e^{-t/tau}], 1/tau = 1/tau2 - 1/T1
                       tau_eff=float(1 / (1 / tau2 - 1 / T1)), c=float(-w[k2] / w[k1]))
# number of modes needed for the truncated PMF to give the exact mode within 1%
for m in (2, 3, 5, 10, 20, 50):
    ks = sig[:m]; fm = (w[ks][None, :] * lam[ks][None, :] ** (t[:, None] - 1)).sum(axis=1)
    res['two_mode'][f'mode_with_{m}_modes'] = int(t[np.argmax(fm)])
# elasticity of the single target neighbour
s = pd.read_csv(_os.path.join(_R, 'data', 'msc_defects_verify', 'v05_sens_N35.csv')); nb = s[(s.i == 33) & (s.j == 34)].iloc[0]
res['target_neighbour'] = dict(chi_mfpt=float(nb.chi_mfpt), chi_mode=float(nb.chi_mode), chi_ratio=float(nb.chi_mode / nb.chi_mfpt), log_ratio=float(np.log1p(nb.chi_mode) / np.log1p(nb.chi_mfpt)))
json.dump(res, open(_os.path.join(_R, 'data', 'msc_defects_verify', 'v09_mechanism.json'), 'w'), indent=1)
print(json.dumps(res, indent=1))
