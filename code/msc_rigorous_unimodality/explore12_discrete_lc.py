"""explore12: exact test of log-concavity of phi(t)=u(t)-u(t-1), u(t)=P_q^t(x0,a) (UNKILLED lazy walk),
for several q; plus the discrete last-exit identity f = E * phi."""
import sys, os, math
from fractions import Fraction
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def unkilled_numerators(N, d, qn, qd, start, target, T):
    """U_t integer with u(t)=U_t/D^t, D=2*d*qd (no gcd reduction), t=0..T"""
    shape = (N,) * d
    D = 2 * d * qd
    nblk = np.zeros(shape, dtype=object)
    for ax in range(d):
        i0 = [slice(None)] * d; i0[ax] = 0; nblk[tuple(i0)] += 1
        i1 = [slice(None)] * d; i1[ax] = N - 1; nblk[tuple(i1)] += 1
    stay = np.empty(shape, dtype=object)
    for idx in np.ndindex(*shape):
        stay[idx] = 2 * d * (qd - qn) + qn * int(nblk[idx])
    R = np.zeros(shape, dtype=object)
    for idx in np.ndindex(*shape):
        R[idx] = 0
    R[tuple(start)] = 1
    sl = []
    for ax in range(d):
        lo = [slice(None)] * d; hi = [slice(None)] * d
        lo[ax] = slice(0, N - 1); hi[ax] = slice(1, N)
        sl.append((tuple(lo), tuple(hi)))
    U = [R[tuple(target)]]
    for t in range(T):
        new = stay * R
        mR = R * qn
        for lo, hi in sl:
            new[hi] += mR[lo]
            new[lo] += mR[hi]
        R = new
        U.append(R[tuple(target)])
    return U, D

def lc_report(N, d, qn, qd, start, target, T):
    U, D = unkilled_numerators(N, d, qn, qd, start, target, T)
    # phi(t) = U_t/D^t - U_{t-1}/D^{t-1} = (U_t - D U_{t-1})/D^t =: Phi_t / D^t
    Phi = [U[0]] + [U[t] - D * U[t - 1] for t in range(1, T + 1)]
    neg = [t for t in range(T + 1) if Phi[t] < 0]
    # log-concavity: Phi_t^2 >= Phi_{t-1} Phi_{t+1}  (the D-powers cancel)
    viol = [t for t in range(1, T) if Phi[t] * Phi[t] < Phi[t - 1] * Phi[t + 1]]
    # internal zeros
    supp = [t for t in range(T + 1) if Phi[t] != 0]
    internal_zero = any(Phi[t] == 0 for t in range(supp[0], supp[-1] + 1)) if supp else False
    return dict(first_neg=neg[:3], n_neg=len(neg), first_viol=viol[:5], n_viol=len(viol), internal_zero=internal_zero)

if __name__ == "__main__":
    for d, Ns in [(1, [3, 4, 5, 8, 9, 12, 20]), (2, [2, 3, 4, 5, 6, 8, 9, 12, 16]), (3, [2, 3, 4, 5, 6])]:
        for N in Ns:
            T = int(6 * d * N * N) + 40
            for (qn, qd) in [(1, 2), (3, 5), (4, 5), (9, 10), (1, 1)]:
                r = lc_report(N, d, qn, qd, (0,) * d, (N - 1,) * d, T)
                print(f"CC d={d} N={N} q={qn}/{qd} T={T}: {r}", flush=True)
