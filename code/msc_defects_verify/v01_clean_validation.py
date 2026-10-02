"""v01 -- clean-lattice numbers, four independent MFPT routes, two independent mode routes,
Eq. (4.1) algebra, and a Monte Carlo of the literal walker rule.  Output: data/v01_validation.json"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import os, sys, json, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, scipy.sparse as sp, scipy.linalg as la
import vcore as vc
OUT = _os.path.join(_R, 'data', 'msc_defects_verify', 'v01_validation.json')
res = {}
N, q = 35, 0.8
om = np.ones((N, N), bool)
t0 = time.time()
rec = vc.solve_config(om, q, want_pmf=True, extras=True)
pmf = rec.pop('pmf')
res['clean_N35'] = rec
# (b) cosine double sum for the reflecting chain: MFPT = sum_{k != 0} [phi_k(a)^2 - phi_k(s)phi_k(a)] / (1 - lam_k) / pi_a
k = np.arange(N)
alpha = np.where(k == 0, 1.0, 2.0)
tot = 0.0
for k1 in range(N):
    for k2 in range(N):
        if k1 == 0 and k2 == 0:
            continue
        lam = 1 - q + (q / 2) * (np.cos(np.pi * k1 / N) + np.cos(np.pi * k2 / N))
        # eigenfunction h(x)=cos(pi k (x+1/2)/N); norm alpha/N per dimension
        ha = np.cos(np.pi * k1 * (N - 0.5) / N) * np.cos(np.pi * k2 * (N - 0.5) / N)
        hs = np.cos(np.pi * k1 * 0.5 / N) * np.cos(np.pi * k2 * 0.5 / N)
        tot += alpha[k1] * alpha[k2] * (ha * ha - hs * ha) / (1 - lam)
res['clean_N35']['mfpt_cosine_double_sum'] = float(tot)   # (1/N^2)*sum / pi_a with pi_a = 1/N^2
# (c) dense spectral route for mode + mfpt
idx, n = vc.cluster_index(om, (0, 0)); a = idx[N - 1, N - 1]
P = vc.transition(idx, n, q); Q, r, keep = vc.reduce_target(P, a)
lam, V = la.eigh(Q.toarray())
w = V[0, :] * (V.T @ r)
c = V[0, :] * V.sum(axis=0)
res['clean_N35']['mfpt_spectral'] = float(np.sum(c / (1 - lam)))
T = 60000
t = np.arange(1, T + 1)
# f(t) = sum_k w_k lam_k^(t-1): evaluate in blocks
f = np.empty(T)
blk = 4000
for b0 in range(0, T, blk):
    tt = t[b0:b0 + blk] - 1
    f[b0:b0 + blk] = (w[None, :] * lam[None, :] ** tt[:, None]).sum(axis=1)
res['clean_N35']['mode_spectral'] = int(np.argmax(f) + 1)
res['clean_N35']['max_abs_diff_pmf_two_routes'] = float(np.max(np.abs(f[:len(pmf)] - pmf)))
# survival, median
S = lambda tt: float(np.sum(c * lam ** tt))
res['clean_N35']['S_mode'] = S(rec['mode']); res['clean_N35']['S_mfpt'] = S(int(round(rec['mfpt'])))
lo, hi = 1, 200000
while lo < hi:
    mid = (lo + hi) // 2
    if S(mid) <= 0.5: hi = mid
    else: lo = mid + 1
res['clean_N35']['median'] = lo
res['clean_N35']['tau_rel_reflecting'] = float(-1 / np.log(1 - (q / 2) * (1 - np.cos(np.pi / N))))
res['clean_N35']['lam1_Q'] = float(lam[-1]); res['clean_N35']['lam2_Q'] = float(lam[-2]); res['clean_N35']['lam_min_Q'] = float(lam[0])
# brute-force tail certificate: f(t) <= S(t-1); find t where S < fmode
res['clean_N35']['S_at_60000'] = S(60000)
print('clean done', time.time() - t0, flush=True)

# Eq (4.1): T0 - Delta e e^T
Pd = P.toarray()
interior = idx[17, 17]
res['eq41'] = dict(diag_interior=float(Pd[interior, interior]), diag_after_Delta1=float(Pd[interior, interior] - 1.0),
                   colsum_after_Delta=float(Pd[:, interior].sum() - 0.1), note='column sum 1-Delta: killing; Delta=1 -> diagonal -q')
# difference between blocked-site matrix and T0 on a 5x5 example: which entries change
om5 = np.ones((5, 5), bool); i0, n0 = vc.cluster_index(om5, (0, 0)); P0 = vc.transition(i0, n0, q).toarray()
om5b = om5.copy(); om5b[2, 2] = False; i1, n1 = vc.cluster_index(om5b, (0, 0)); P1 = vc.transition(i1, n1, q).toarray()
# embed P1 in the 25-site index space
E = np.zeros((25, 25));
map1 = {int(i1[x, y]): int(i0[x, y]) for x in range(5) for y in range(5) if i1[x, y] >= 0}
for i in range(n1):
    for j in range(n1):
        E[map1[i], map1[j]] = P1[i, j]
D = E - P0
blocked = int(i0[2, 2])
nz = np.argwhere(np.abs(D) > 1e-15)
res['eq41']['blocked_site_changed_entries'] = int(len(nz))
res['eq41']['rank_of_update'] = int(np.linalg.matrix_rank(D))
res['eq41']['changed_diag_of_neighbours'] = sorted({int(i) for i, j in nz if i == j and i != blocked})

# Monte Carlo of the literal rule, N=15, 34 blocked sites, own seed
rng = np.random.default_rng(777)
N2 = 15
while True:
    blk_sites = rng.choice(np.arange(1, N2 * N2 - 1), 34, replace=False)
    m = np.zeros(N2 * N2, bool); m[blk_sites] = True; m = m.reshape(N2, N2)
    if vc.connected(~m, (0, 0), (N2 - 1, N2 - 1)):
        break
ex = vc.solve_config(~m, q, want_pmf=True)
W = 200000
x = np.zeros(W, np.int64); y = np.zeros(W, np.int64); alive = np.arange(W)
tfp = np.zeros(W, np.int64); t = 0
dx = np.array([1, -1, 0, 0]); dy = np.array([0, 0, 1, -1])
while len(alive):
    t += 1
    k = len(alive)
    move = rng.random(k) < q
    d = rng.integers(0, 4, k)
    nx = x[alive] + dx[d] * move; ny = y[alive] + dy[d] * move
    ok = (nx >= 0) & (nx < N2) & (ny >= 0) & (ny < N2)
    nxc = np.clip(nx, 0, N2 - 1); nyc = np.clip(ny, 0, N2 - 1)
    ok &= ~m[nxc, nyc]
    x[alive] = np.where(ok, nx, x[alive]); y[alive] = np.where(ok, ny, y[alive])
    hit = (x[alive] == N2 - 1) & (y[alive] == N2 - 1)
    tfp[alive[hit]] = t
    alive = alive[~hit]
mc_mean = tfp.mean(); mc_se = tfp.std(ddof=1) / np.sqrt(W)
# chi2 on binned PMF using exact cdf from PMF route (extend pmf by stepping survival)
pm = ex.pop('pmf')
# need the cdf far in the tail: compute by direct stepping
idx2, n2 = vc.cluster_index(~m, (0, 0)); a2 = idx2[N2 - 1, N2 - 1]
P2 = vc.transition(idx2, n2, q); Q2, r2, _ = vc.reduce_target(P2, a2)
u = np.zeros(n2 - 1); u[0] = 1.0; Sx = [1.0]
for _ in range(int(tfp.max()) + 5):
    u = Q2 @ u; Sx.append(u.sum())
Sx = np.array(Sx)
edges = np.quantile(tfp, np.linspace(0, 1, 41)).astype(int); edges[0] = 0; edges = np.unique(edges)
obs = np.histogram(tfp, bins=np.r_[edges[:-1] + 0.5, edges[-1] + 0.5])[0]
expc = W * (Sx[edges[:-1]] - Sx[edges[1:]])
# last bin: add the tail mass beyond the max
chi2 = float(np.sum((obs - expc) ** 2 / expc))
res['mc'] = dict(N=N2, blocked=34, walkers=W, exact_mfpt=ex['mfpt'], mc_mean=float(mc_mean), mc_se=float(mc_se),
                 z=float((mc_mean - ex['mfpt']) / mc_se), exact_mode=ex['mode'], chi2=chi2, dof=int(len(obs) - 1),
                 mc_var=float(tfp.var(ddof=1)), exact_sd=ex['sd'], mc_sd=float(tfp.std(ddof=1)))
json.dump(res, open(OUT, 'w'), indent=1)
print(json.dumps(res, indent=1))
