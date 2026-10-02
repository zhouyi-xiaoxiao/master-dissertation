"""V5: extra checks. (i) accuracy of float64 form (S) itself against 40-digit mp;
(ii) Theorem 3 in float64 for every start site at N=35 (corner and interior target)
against a sparse direct solve; (iii) coordinate-swap symmetry of Theorem 3."""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import json, os
import numpy as np, mpmath as mp
import scipy.sparse as sp, scipy.sparse.linalg as spla
import vlib
mp.mp.dps = 40
out = {}
def S_float(N):
    k = np.arange(1, N); s = np.sin(np.pi * k / (2 * N)); u = N * np.arcsinh(s)
    g = np.where(k % 2 == 1, 1 / np.tanh(u), np.tanh(u))
    return 4 * N * np.sum((1 - s * s) * np.sqrt(1 + s * s) / s * g)
out['float64_formS_vs_mp'] = {str(N): float(abs(S_float(N) - vlib.qT_S(N)) / vlib.qT_S(N)) for N in (10, 100, 403, 1000, 6000)}

def thm3_float(N, s, a):
    k = np.arange(1, N)
    phi = np.arccosh(2 - np.cos(np.pi * k / N))
    th = lambda m: np.pi * k * (2 * m - 1) / (2 * N)
    def Psi(u, v):   # overflow-free: cosh((N-m)phi)/sinh(N phi) = (e^{-m phi}+e^{-(2N-m)phi})/(1-e^{-2N phi})
        m1 = abs(u - v); m2 = u + v - 1
        f = lambda m: (np.exp(-m * phi) + np.exp(-(2 * N - m) * phi)) / (1 - np.exp(-2 * N * phi))
        return (f(m1) + f(m2)) / np.sinh(phi)
    return 2 * vlib.tau1(N, s[1], a[1]) + 4 * N * np.sum(np.cos(th(a[0])) ** 2 * Psi(a[1], a[1]) - np.cos(th(s[0])) * np.cos(th(a[0])) * Psi(s[1], a[1]))

def all_starts(N, q, a):
    n = N * N
    deg = np.full((N, N), 4.0); deg[0, :] -= 1; deg[-1, :] -= 1; deg[:, 0] -= 1; deg[:, -1] -= 1
    idx = np.arange(n).reshape(N, N)
    r = []; c = []
    for A, B in ((idx[:-1, :], idx[1:, :]), (idx[:, :-1], idx[:, 1:])):
        r += [A.ravel(), B.ravel()]; c += [B.ravel(), A.ravel()]
    r = np.concatenate(r); c = np.concatenate(c)
    L = (sp.diags(deg.ravel()) - sp.csr_matrix((np.ones(r.size), (r, c)), shape=(n, n))) * (q / 4)   # I - P
    t = (a[0] - 1) * N + (a[1] - 1)
    keep = np.array([x for x in range(n) if x != t])
    h = spla.spsolve(L.tocsc()[keep][:, keep], np.ones(n - 1))
    full = np.zeros(n); full[keep] = h
    return full.reshape(N, N)
N, q = 35, 0.8
res = {}
for a in ((35, 35), (9, 23), (1, 18)):
    H = all_starts(N, q, a)
    worst = 0
    for s1 in range(1, N + 1):
        for s2 in range(1, N + 1):
            if (s1, s2) == a: continue
            worst = max(worst, abs(thm3_float(N, (s1, s2), a) / q - H[s1 - 1, s2 - 1]) / H[s1 - 1, s2 - 1])
    res[str(a)] = worst
out['thm3_float64_all_starts_N35_q0.8_worst_rel'] = res
# coordinate swap symmetry (the formula treats the two coordinates differently)
w = 0
for (N, s, a) in [(7, (2, 5), (6, 3)), (12, (1, 12), (7, 7)), (9, (9, 1), (3, 8))]:
    w = max(w, float(abs(vlib.qT_thm3(N, s, a) - vlib.qT_thm3(N, s[::-1], a[::-1])) / vlib.qT_thm3(N, s, a)))
out['thm3_coordinate_swap_rel'] = w
# 1/q scaling: MFPT_uniform(q_eff) = (q/q_eff) MFPT_uniform(q) exactly; the leading term 2N(N-1)/q_eff is only the k=0 part:
N = 4001
out['fraction_of_MFPT_carried_by_2N(N-1)_term_N4001'] = float(2 * N * (N - 1) / vlib.qT_S(N))
json.dump(out, open(_os.path.join(_R, 'data', 'msc_proof_verify', 'v5_extra.json'), 'w'), indent=1)
print(json.dumps(out, indent=1))
