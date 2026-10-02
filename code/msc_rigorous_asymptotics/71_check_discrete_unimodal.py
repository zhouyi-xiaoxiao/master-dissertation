"""EXACT (integer arithmetic) check of Theorem 7.4 for small cases, q <= 1/2:
 phi(t) = u_{t+1}-u_t (u_t = N^d P^t(x0,a)) is nonnegative and log-concave, phi(t+1) >= (1-mu)phi(t) is checked via integers,
 and the PMF f(t) = P(T=t) has a single sign change of f(t+1)-f(t) (weakly unimodal), over t <= tmax.
q = a/b; one step multiplies integer vectors by the integer matrix  M = (2d b) P  (entries: off-diag a, diag 2db - a*(#moves))."""
import json, itertools, sys
from fractions import Fraction
from r4lib import DATA


def step_matrix(d, N, a, b, kill=None):
    sites = list(itertools.product(range(N), repeat=d))
    idx = {s: i for i, s in enumerate(sites)}
    nb = []
    for s in sites:
        lst = []
        for ax in range(d):
            for dlt in (-1, 1):
                t = list(s); t[ax] += dlt
                if 0 <= t[ax] < N:
                    lst.append(idx[tuple(t)])
        nb.append(lst)
    return sites, idx, nb


def run(d, N, a, b, tmax):
    sites, idx, nb = step_matrix(d, N, a, b)
    n = len(sites); D = 2 * d * b
    start = idx[(0,) * d]; tgt = idx[(N - 1,) * d]
    # unkilled chain: integer vector p (scaled by D^t)
    p = [0] * n; p[start] = 1
    pk = [0] * n; pk[start] = 1                 # killed chain
    u = []; surv = []
    for t in range(tmax + 1):
        u.append(p[tgt])                        # D^t * P^t(x0,a)
        surv.append(sum(pk[i] for i in range(n) if i != tgt))      # D^t * P(T > t)
        newp = [0] * n; newk = [0] * n
        for i in range(n):
            if p[i]:
                stay = D - a * len(nb[i])
                newp[i] += p[i] * stay
                for j in nb[i]:
                    newp[j] += p[i] * a
            if pk[i] and i != tgt:
                stay = D - a * len(nb[i])
                newk[i] += pk[i] * stay
                for j in nb[i]:
                    newk[j] += pk[i] * a
        p, pk = newp, newk
        pk[tgt] = 0
    # phi(t) * D^{t+1} = u[t+1] - D u[t]
    phi = [u[t + 1] - D * u[t] for t in range(tmax)]
    neg = sum(1 for z in phi if z < 0)
    # log-concavity: phi_t^2 >= phi_{t-1} phi_{t+1}  with scalings D^{t+1}: (phi_t D^-(t+1))^2 >= phi_{t-1}D^-t phi_{t+1} D^-(t+2)  <=> phi_t^2 >= phi_{t-1} phi_{t+1}
    lc = sum(1 for t in range(1, tmax - 1) if phi[t] * phi[t] < phi[t - 1] * phi[t + 1])
    # internal zeros
    sup = [t for t in range(tmax) if phi[t] > 0]
    internal_zero = any(phi[t] == 0 for t in range(sup[0], sup[-1] + 1)) if sup else False
    # f(t) D^t = D*surv[t-1] - surv[t]
    f = [None] + [D * surv[t - 1] - surv[t] for t in range(1, tmax + 1)]
    # sign of f(t+1)-f(t): f(t+1) D^{t+1} vs f(t) D^{t+1} = D f[t]
    sg = []
    for t in range(1, tmax):
        dlt = f[t + 1] - D * f[t]
        sg.append((dlt > 0) - (dlt < 0))
    nz = [z for z in sg if z != 0]
    changes = sum(1 for i in range(1, len(nz)) if nz[i] != nz[i - 1])
    ties = sum(1 for z in sg[sg.index(1) if 1 in sg else 0:] if z == 0)
    mode = max(range(1, tmax + 1), key=lambda t: Fraction(f[t], D ** t))
    return dict(d=d, N=N, q='%d/%d' % (a, b), tmax=tmax, phi_negative=neg, phi_logconcave_violations=lc, phi_internal_zero=internal_zero,
                f_sign_changes=changes, ties_after_start=ties, mode=mode)


if __name__ == '__main__':
    out = []
    for d, N, a, b, tmax in [(2, 3, 1, 2, 120), (2, 4, 1, 2, 200), (2, 5, 1, 2, 300), (2, 6, 1, 2, 400), (2, 5, 1, 3, 400), (2, 5, 1, 5, 500),
                             (3, 3, 1, 2, 200), (3, 4, 1, 2, 300), (3, 4, 1, 4, 400), (2, 8, 1, 2, 500)]:
        r = run(d, N, a, b, tmax); out.append(r); print(json.dumps(r), flush=True)
    json.dump(out, open(DATA + '/71_check_discrete_unimodal.json', 'w'), indent=1)
