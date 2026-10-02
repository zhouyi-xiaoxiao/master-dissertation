"""V2: separate numerical check of every lemma / intermediate identity used in
the proofs (Supplementary Sections S3 and S4), plus model-level sanity checks (Monte Carlo, pseudo-inverse, resistance,
generating-function limit, float64 behaviour).  Writes results/v2_identities.json."""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import json, os, time
from fractions import Fraction
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import mpmath as mp
import vlib

OUT = _os.path.join(_R, 'data', 'msc_proof_verify', 'v2_identities.json')
R = {}
t0 = time.time()
mp.mp.dps = 40


def save():
    json.dump(R, open(OUT, 'w'), indent=1, default=str)


def build_P(N, q):
    """dense float P from the model rules (not from the Kronecker formula)."""
    n = N * N
    P = np.zeros((n, n))
    for i in range(N):
        for j in range(N):
            x = i * N + j
            P[x, x] += 1 - q
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                a, b = i + di, j + dj
                if 0 <= a < N and 0 <= b < N:
                    P[x, a * N + b] += q / 4
                else:
                    P[x, x] += q / 4
    return P


# ------------------------------------------------------------------ 1. Monte Carlo
rng = np.random.default_rng(20261001)


def mc_mfpt(N, q, start, target, walkers):
    pos = np.tile(np.array(start), (walkers, 1))
    alive = np.ones(walkers, bool)
    t = np.zeros(walkers, np.int64)
    steps = np.array([(1, 0), (-1, 0), (0, 1), (0, -1)])
    while alive.any():
        idx = np.nonzero(alive)[0]
        m = idx.size
        move = rng.random(m) < q
        d = steps[rng.integers(0, 4, m)]
        new = pos[idx] + d * move[:, None]
        ok = ((new >= 0) & (new < N)).all(axis=1)
        new = np.where(ok[:, None], new, pos[idx])
        pos[idx] = new
        t[idx] += 1
        hit = (new[:, 0] == target[0]) & (new[:, 1] == target[1])
        alive[idx[hit]] = False
    return t.mean(), t.std(ddof=1) / np.sqrt(walkers)


mc = []
for (N, q, s, a) in [(6, 0.8, (0, 0), (5, 5)), (5, 0.5, (1, 3), (4, 2)), (7, 1.0, (0, 0), (6, 6))]:
    m, se = mc_mfpt(N, q, s, a, 200000)
    pred = float(vlib.qT_thm3(N, (s[0] + 1, s[1] + 1), (a[0] + 1, a[1] + 1))) / q
    mc.append({'N': N, 'q': q, 'start': s, 'target': a, 'mc_mean': m, 'mc_se': se, 'thm3': pred, 'z': (m - pred) / se})
R['monte_carlo'] = mc
save()
print('MC', mc, flush=True)

# ------------------------------------------------------------------ 2. Lemma 1, Lemma 2
lem = []
for N, q in [(4, 1.0), (5, 0.7), (6, 0.3), (7, 0.9)]:
    P = build_P(N, q)
    n = N * N
    # Lemma 2: Kronecker form and eigenpairs
    T1 = np.zeros((N, N))
    for m in range(N):
        if m > 0:
            T1[m, m - 1] = .5
        if m < N - 1:
            T1[m, m + 1] = .5
    T1[0, 0] = T1[N - 1, N - 1] = .5
    Pk = (1 - q) * np.eye(n) + q / 2 * (np.kron(T1, np.eye(N)) + np.kron(np.eye(N), T1))
    psi = np.array([[np.sqrt((1 if k == 0 else 2) / N) * np.cos(np.pi * k * (2 * m - 1) / (2 * N)) for m in range(1, N + 1)] for k in range(N)])
    eig_res = max(np.abs(T1 @ psi[k] - np.cos(np.pi * k / N) * psi[k]).max() for k in range(N))
    orth = np.abs(psi @ psi.T - np.eye(N)).max()
    # Lemma 1: pinv formula vs direct solve, all targets
    Lp = np.linalg.pinv(np.eye(n) - P)
    # spectral pseudo-inverse (5.2)
    Lsp = np.zeros((n, n))
    for k in range(N):
        for j in range(N):
            if k == 0 and j == 0:
                continue
            v = np.kron(psi[k], psi[j])
            Lsp += np.outer(v, v) / (q * (np.sin(np.pi * k / (2 * N)) ** 2 + np.sin(np.pi * j / (2 * N)) ** 2))
    worst = 0
    for a in range(n):
        keep = [x for x in range(n) if x != a]
        h = np.linalg.solve(np.eye(n - 1) - P[np.ix_(keep, keep)], np.ones(n - 1))
        g = n * (Lp[a, a] - Lp[keep, a])
        worst = max(worst, np.abs(g - h).max() / h.max())
    lem.append({'N': N, 'q': q, 'P_rule_vs_kron': np.abs(P - Pk).max(), 'eig_residual': eig_res, 'orthonormality': orth,
                'pinv_vs_spectral': np.abs(Lp - Lsp).max() / np.abs(Lp).max(), 'lemma1_all_targets_rel': worst,
                'P_symmetric': bool(np.allclose(P, P.T)), 'row_sums_1': bool(np.allclose(P.sum(1), 1))})
R['lemma1_lemma2'] = lem
save()
print('lemma1/2', lem, flush=True)

# ------------------------------------------------------------------ 3. Lemma 3 (6.1),(6.2),(A.13),(A.7/E2)
w61 = w62 = wA13 = wA7 = mp.mpf(0)
n61 = n62 = 0
for M in (1, 2, 3, 4, 7, 12, 25):
    for sig in (mp.mpf('1.0001'), mp.mpf('1.3'), mp.mpf(3), mp.mpf(17)):
        phi = mp.acosh(sig)
        for m in range(0, M + 1):
            lhs = mp.fsum(mp.cos(2 * mp.pi * l * m / M) / (sig - mp.cos(2 * mp.pi * l / M)) for l in range(M))
            rhs = M * mp.cosh((mp.mpf(M) / 2 - m) * phi) / (mp.sinh(phi) * mp.sinh(M * phi / 2))
            w61 = max(w61, abs(lhs - rhs) / abs(rhs)); n61 += 1
out_of_range = None
for N in (2, 3, 5, 8, 13):
    for sig in (mp.mpf('1.0001'), mp.mpf('1.3'), mp.mpf(3)):
        phi = mp.acosh(sig)
        for m in range(0, 2 * N + 1):
            lhs = mp.fsum((1 if j == 0 else 2) * mp.cos(m * mp.pi * j / N) / (sig - mp.cos(mp.pi * j / N)) for j in range(N))
            rhs = 2 * N * mp.cosh((N - m) * phi) / (mp.sinh(phi) * mp.sinh(N * phi)) - (-1) ** m / (sig + 1)
            w62 = max(w62, abs(lhs - rhs) / abs(rhs)); n62 += 1
            # closed form = Giuggioli (E1)
            S = mp.fsum(mp.cos(m * mp.pi * k / N) / (sig - mp.cos(mp.pi * k / N)) for k in range(1, N))
            A13 = N * mp.cosh(abs(N - m) * phi) / (mp.sinh(phi) * mp.sinh(N * phi)) + mp.mpf(1) / 2 * (1 / (1 - sig) + (-1) ** (m + 1) / (1 + sig))
            wA13 = max(wA13, abs(S - A13))
        # outside the range the identity must fail (confirms the stated range is sharp)
        m = 2 * N + 1
        lhs = mp.fsum((1 if j == 0 else 2) * mp.cos(m * mp.pi * j / N) / (sig - mp.cos(mp.pi * j / N)) for j in range(N))
        rhs = 2 * N * mp.cosh((N - m) * phi) / (mp.sinh(phi) * mp.sinh(N * phi)) - (-1) ** m / (sig + 1)
        out_of_range = float(abs(lhs - rhs))
    for m in range(-2 * N, 2 * N + 1):
        F = mp.fsum(mp.cos(m * mp.pi * k / N) / (1 - mp.cos(mp.pi * k / N)) for k in range(1, N))
        A7 = (mp.mpf(N) ** 2 + mp.mpf(1) / 2) / 3 + abs(m) * (mp.mpf(abs(m)) / 2 - N) + mp.mpf((-1) ** (m + 1) - 1) / 4
        wA7 = max(wA7, abs(F - A7))
R['lemma3'] = {'(6.1)_worst_rel': float(w61), 'n61': n61, '(6.2)_worst_rel': float(w62), 'n62': n62,
               '(A.13)_worst_abs': float(wA13), '(A.7)_worst_abs': float(wA7), '(6.2)_at_m=2N+1_abs_violation(sigma=3,N=13)': out_of_range}
save()
print('lemma3', R['lemma3'], flush=True)

# ------------------------------------------------------------------ 4. Lemma 4 (1D), (6.3), (6.4), trig sums
w_tau = 0; w63 = mp.mpf(0); w64 = mp.mpf(0); wc = mp.mpf(0); wco = mp.mpf(0)
for N in range(2, 15):
    # exact 1D chain T from rules: step +-1 w.p. 1/2, cancelled at the walls
    for v in range(1, N + 1):
        A = [[Fraction(0)] * N for _ in range(N)]
        for m in range(1, N + 1):
            for d in (-1, 1):
                mm = m + d
                if 1 <= mm <= N:
                    A[m - 1][mm - 1] += Fraction(1, 2)
                else:
                    A[m - 1][m - 1] += Fraction(1, 2)
        keep = [m for m in range(1, N + 1) if m != v]
        Mx = np.array([[float((1 if a == b else 0) - A[a - 1][b - 1]) for b in keep] for a in keep])
        h = np.linalg.solve(Mx, np.ones(len(keep))) if keep else []
        for u, hv in zip(keep, h):
            w_tau = max(w_tau, abs(hv - vlib.tau1(N, u, v)))
            lhs = mp.fsum(2 * (mp.cos(mp.pi * j * (2 * v - 1) / (2 * N)) ** 2 - mp.cos(mp.pi * j * (2 * u - 1) / (2 * N)) * mp.cos(mp.pi * j * (2 * v - 1) / (2 * N))) / (1 - mp.cos(mp.pi * j / N)) for j in range(1, N))
            w63 = max(w63, abs(lhs - vlib.tau1(N, u, v)))
    w64 = max(w64, abs(mp.fsum(mp.cot(mp.pi * j / (2 * N)) ** 2 for j in range(1, N, 2)) - mp.mpf(N * (N - 1)) / 2))
    wc = max(wc, abs(mp.fsum(mp.cos(mp.pi * k / (2 * N)) ** 2 for k in range(1, N)) - mp.mpf(N - 1) / 2))
    wco = max(wco, abs(mp.fsum(mp.cos(mp.pi * k / (2 * N)) ** 2 for k in range(1, N, 2)) - mp.mpf(N) / 4))
R['lemma4'] = {'tau1_vs_direct_1D_solve_abs': float(w_tau), '(6.3)_abs': float(w63), '(6.4)_abs': float(w64),
               'sum_c2=(N-1)/2_abs': float(wc), 'sum_odd_c2=N/4_abs': float(wco)}
save()
print('lemma4', R['lemma4'], flush=True)

# ------------------------------------------------------------------ 5. (6.6) G_k and H_k
w66 = mp.mpf(0); wH = mp.mpf(0); wB = mp.mpf(0)
for N in (2, 3, 4, 5, 8, 9, 16, 21):
    for k in range(1, N):
        sig = 2 - mp.cos(mp.pi * k / N)
        phi = mp.acosh(sig)
        G = mp.fsum((1 if j == 0 else 2) * mp.cos(mp.pi * j / (2 * N)) ** 2 * (1 - (-1) ** (k + j)) / (sig - mp.cos(mp.pi * j / N)) for j in range(N))
        G66 = N * (1 + sig) * (mp.cosh(N * phi) - (-1) ** k) / (mp.sinh(phi) * mp.sinh(N * phi)) - N
        w66 = max(w66, abs(G - G66) / abs(G))
        B = mp.cosh(N * phi) + mp.cosh((N - 1) * phi) - (-1) ** k * (1 + mp.cosh(phi))
        wB = max(wB, abs(G - N * B / (mp.sinh(phi) * mp.sinh(N * phi))) / abs(G))
        if k % 2:
            H = mp.fsum((1 if j == 0 else 2) * mp.cos(mp.pi * j / (2 * N)) ** 2 / (sig - mp.cos(mp.pi * j / N)) for j in range(0, N, 2))
            Hc = mp.mpf(N) / 2 * (mp.coth(phi / 2) * mp.coth(N * phi / 2) - 1)
            wH = max(wH, abs(H - Hc) / abs(H))
        # phi = 2 asinh(s_k)
        assert abs(phi - 2 * mp.asinh(mp.sin(mp.pi * k / (2 * N)))) < mp.mpf(10) ** -35
R['theorem2_steps'] = {'(6.6)_rel': float(w66), 'G_k=N*B_k/(sinh sinh)_rel': float(wB), 'H_k_rel': float(wH)}
save()
print('thm2 steps', R['theorem2_steps'], flush=True)

# ------------------------------------------------------------------ 6. Corollary 5 (resistance)
res = []
for N in range(2, 21):
    n = N * N
    Lg = np.zeros((n, n))
    for i in range(N):
        for j in range(N):
            x = i * N + j
            for di, dj in ((1, 0), (0, 1)):
                a, b = i + di, j + dj
                if a < N and b < N:
                    y = a * N + b
                    Lg[x, x] += 1; Lg[y, y] += 1; Lg[x, y] -= 1; Lg[y, x] -= 1
    Lp = np.linalg.pinv(Lg)
    Rc = Lp[0, 0] + Lp[n - 1, n - 1] - 2 * Lp[0, n - 1]
    qT = float(vlib.qT_S(N))
    row = {'N': N, 'R_corner': Rc, 'qT/(2N^2)': qT / (2 * N * N), 'rel': abs(Rc - qT / (2 * N * N)) / Rc}
    if N % 2 == 0:  # Essam-Wu (6),(15),(16) with M=N, r=s=1
        ew = 1 + 4 / N * sum(np.cos(p * np.pi / N) ** 2 * np.sqrt(1 + np.sin(p * np.pi / N) ** 2) / np.sin(p * np.pi / N) * np.tanh(N * np.arcsinh(np.sin(p * np.pi / N))) for p in range(1, N // 2 + 1))
        row['EssamWu_rel'] = abs(ew - Rc) / Rc
    res.append(row)
# general pair commute time
N = 6; q = 0.37; P = build_P(N, q); n = N * N
s, a = 7, 28
def hit(P, s, a):
    keep = [x for x in range(P.shape[0]) if x != a]
    h = np.linalg.solve(np.eye(len(keep)) - P[np.ix_(keep, keep)], np.ones(len(keep)))
    return h[keep.index(s)]
Lg = (np.eye(n) - P) * 4 / q
Lp = np.linalg.pinv(Lg)
Rsa = Lp[s, s] + Lp[a, a] - 2 * Lp[s, a]
R['corollary5'] = {'corner': res, 'worst_rel': max(r['rel'] for r in res), 'worst_EW_rel': max(r.get('EssamWu_rel', 0) for r in res),
                   'Wu2004_example4_R4x4=13/7': abs(res[2]['R_corner'] - 13 / 7),
                   'commute_general_pair_rel': abs(hit(P, s, a) + hit(P, a, s) - 4 * n / q * Rsa) / (4 * n / q * Rsa),
                   'I-P=(q/4)L_graph_is_integer_Laplacian': bool(np.allclose(Lg, np.round(Lg)) and np.allclose(Lg.sum(1), 0))}
save()
print('cor5', R['corollary5']['worst_rel'], R['corollary5']['worst_EW_rel'], R['corollary5']['Wu2004_example4_R4x4=13/7'], R['corollary5']['commute_general_pair_rel'], flush=True)

# ------------------------------------------------------------------ 7. generating function: which ratio gives the mean
gf = []
for N, q in [(4, 0.8), (7, 0.5)]:
    P = build_P(N, q); n = N * N; s = 0; a = n - 1
    T = hit(P, s, a)
    for eps in (1e-3, 1e-5, 1e-7):
        z = 1 - eps
        G = np.linalg.inv(np.eye(n) - z * P)
        F = G[s, a] / G[a, a]
        gf.append({'N': N, 'q': q, '1-z': eps, 'T_direct': T, '(1-F)/(1-z)': (1 - F) / (1 - z), '(1-z)/(1-F) [reciprocal ratio]': (1 - z) / (1 - F)})
R['eq_2_12'] = gf
save()
print('gf', gf[-1], flush=True)

# ------------------------------------------------------------------ 8. float64: sparse direct solve vs (S); overflow of (3.5)
def qT_S_float(N):
    k = np.arange(1, N)
    s = np.sin(np.pi * k / (2 * N))
    u = N * np.arcsinh(s)
    g = np.where(k % 2 == 1, 1 / np.tanh(u), np.tanh(u))
    return 4 * N * np.sum((1 - s * s) * np.sqrt(1 + s * s) / s * g)


def qT_form_a_float(N):
    k = np.arange(1, N)
    phi = np.arccosh(2 - np.cos(np.pi * k / N))
    with np.errstate(all='ignore'):
        B = np.cosh(N * phi) + np.cosh((N - 1) * phi) - (-1.0) ** k * (1 + np.cosh(phi))
        return 2 * N * (N - 1) + 4 * N * np.sum(np.cos(np.pi * k / (2 * N)) ** 2 * B / (np.sinh(phi) * np.sinh(N * phi)))


def sparse_T(N, q):
    n = N * N
    idx = np.arange(n).reshape(N, N)
    rows = []; cols = []; vals = []
    diag = np.full((N, N), 1 - q)
    for axis, sl_a, sl_b in ((0, (slice(0, N - 1), slice(None)), (slice(1, N), slice(None))), (1, (slice(None), slice(0, N - 1)), (slice(None), slice(1, N)))):
        a = idx[sl_a].ravel(); b = idx[sl_b].ravel()
        rows += [a, b]; cols += [b, a]; vals += [np.full(a.size, q / 4), np.full(a.size, q / 4)]
    deg = np.full((N, N), 4)
    deg[0, :] -= 1; deg[-1, :] -= 1; deg[:, 0] -= 1; deg[:, -1] -= 1
    diag = 1 - q + q / 4 * (4 - deg)
    P = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)) + sp.diags(diag.ravel())
    A = (sp.eye(n) - P).tocsc()[:n - 1, :n - 1]
    h = spla.spsolve(A, np.ones(n - 1))
    return h[0]


fl = []
for N in (2, 3, 5, 10, 35, 64, 101, 200, 350, 402, 403, 500):
    q = 0.8
    Td = sparse_T(N, q)
    fl.append({'N': N, 'q': q, 'T_sparse_direct': Td, 'T_formS': qT_S_float(N) / q, 'rel_S': abs(Td - qT_S_float(N) / q) / Td,
               'T_form_a_float64': qT_form_a_float(N) / q, 'elapsed': round(time.time() - t0, 1)})
    R['float64'] = fl
    save()
    print(fl[-1], flush=True)
first_nan = next(N for N in range(300, 500) if not np.isfinite(qT_form_a_float(N)))
R['form_a_first_nonfinite_N_float64'] = first_nan
R['form_a_last_finite_rel_vs_S'] = abs(qT_form_a_float(first_nan - 1) - qT_S_float(first_nan - 1)) / qT_S_float(first_nan - 1)
R['seconds'] = round(time.time() - t0, 1)
save()
print('first nan', first_nan, R['form_a_last_finite_rel_vs_S'], 'done', R['seconds'])
