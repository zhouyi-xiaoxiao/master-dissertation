"""Sanity checks (float64, brute force) for the following statements of note R2:

  A. Proposition 5.5 (quasi-stationary averages): sum_x nu(x) r_j(x) = delta_{j0}, sum_x nu(x) m(x) = 1/sigma_0,
     P_nu(T>t) = exp(-sigma_0 t); equality case.                       (random chains + boxes)
  B. Lemma 7.2 lower bound: m - tau >= d (N^2-1)/3 (equality iff d = 1), mu_-(N) >= max(1, (pi^2/6)(1-1/N^2)(1-pi^2/(12N^2))).
  C. Lemma 7.8 (monotone coupling): P_x(T>t) >= P_y(T>t) for x <= y coordinatewise (continuous and discrete time),
     m(.) and r_0(.) coordinatewise non-increasing.
  D. Theorem 7.9: r_0(x_0) = max r_0 > 1, sigma_0 m(x_0) = max > 1, sigma_0 m >= (Xbar + mu_-)/(Xbar + 1).
  E. Theorem 7.10 (ii),(v),(vi) in the sharpened form (mu_- in the lower bounds, y_+ = W - mu_- + ...), spectral sums up to large N.
Output: ../data/15_qsd_coupling_checks.json ; prints ALL OK.
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import os, sys, json, math, itertools
import numpy as np
from scipy.linalg import eigh, expm
from scipy.optimize import brentq
sys.path.insert(0, os.path.dirname(__file__))
from t2lib import build_L, site_index, corner_spectral_data, group_visible
rng = np.random.default_rng(15)
OK = True; OUT = {}
def check(name, cond, info=""):
    global OK
    OK &= bool(cond)
    print(("[OK ] " if cond else "[FAIL] ") + name + ("  " + info if info else ""), flush=True)

def killed(L, a):
    n = len(L); keep = [i for i in range(n) if i != a]
    La = L[np.ix_(keep, keep)]
    mu, U = eigh(La)
    return keep, La, mu, U

# ---------------------------------------------------------------- A. quasi-stationary averages
worst = 0.0
for trial in range(40):
    n = int(rng.integers(4, 12))
    A = np.triu(rng.random((n, n)) * (rng.random((n, n)) < 0.6), 1); A = A + A.T
    L = np.diag(A.sum(1)) - A
    if np.sum(np.linalg.eigvalsh(L) < 1e-10) != 1: continue
    L /= np.max(np.diag(L)); a = int(rng.integers(n))
    keep, La, mu, U = killed(L, a)
    # connectivity of La
    if np.sum(np.abs(np.linalg.matrix_power(np.abs(La) + np.eye(n - 1), n)) < 1e-14) > 0: continue
    psi0 = np.abs(U[:, 0]); nu = psi0 / psi0.sum(); one = np.ones(n - 1)
    m = np.linalg.solve(La, one)
    worst = max(worst, abs(nu @ m - 1 / mu[0]) * mu[0])
    for t in (0.3, 2.0, 9.0):
        worst = max(worst, abs(nu @ (expm(-t * La) @ one) - math.exp(-mu[0] * t)))
    r0 = psi0 * psi0.sum() / (psi0 @ psi0)
    worst = max(worst, abs(nu @ r0 - 1))
    check_ineq = (r0.min() <= 1 + 1e-12 <= r0.max() + 2e-12) and (m.min() <= 1 / mu[0] + 1e-9 <= m.max() + 2e-9)
    if not check_ineq: worst = 1.0
check("A. Prop 5.5 on random chains: nu-averages of r_0, sigma_0 m equal 1; P_nu(T>t)=exp(-sigma_0 t)", worst < 1e-9, f"max err {worst:.1e}")
# equality case: star-like target (every site a neighbour of a with equal weight) -> 1 is an eigenvector of L_a
n = 6; A = np.zeros((n, n)); A[0, 1:] = A[1:, 0] = 0.7
for i in range(1, n):
    for j in range(i + 1, n):
        A[i, j] = A[j, i] = rng.random()
# make the rows of the reduced part have equal off-diagonal sums? not needed: L_a 1 = -L_{.a} = 0.7 * 1
L = np.diag(A.sum(1)) - A
keep, La, mu, U = killed(L, 0)
check("A. equality case: L_ya constant => 1 eigenvector of L_a, r_0 = 1 = sigma_0 m for all starts",
      np.allclose(La @ np.ones(n - 1), 0.7) and abs(mu[0] - 0.7) < 1e-12 and np.allclose(np.linalg.solve(La, np.ones(n - 1)) * mu[0], 1))

# ---------------------------------------------------------------- B. lower bound for m - tau
rows = []
okB = True
for d in (1, 2, 3, 4):
    for N in ([2, 3, 4, 5, 8, 16, 40, 100] if d <= 2 else ([2, 3, 4, 6, 10, 20] if d == 3 else [2, 3, 4, 6])):
        lam, w, sg = corner_spectral_data(N, d)
        nz = lam > 1e-14
        tau = np.sum(w[nz] / lam[nz]); m = np.sum((1 - sg[nz]) * w[nz] / lam[nz])
        lb = d * (N * N - 1) / 3
        L1 = (1 - math.cos(math.pi / N)) / d
        mum = 2 / 3 * (N * N - 1) * math.sin(math.pi / (2 * N)) ** 2
        lb2 = max(1.0, math.pi ** 2 / 6 * (1 - 1 / N ** 2) * (1 - math.pi ** 2 / (12 * N * N))) if N >= 3 else 1.0
        W = 2 * d * math.cos(math.pi / (2 * N)) ** 2
        good = (m - tau >= lb * (1 - 1e-12)) and (abs(m - tau - lb) < 1e-9 * lb if d == 1 else m - tau > lb * (1 + 1e-9)) \
            and mum >= lb2 - 1e-12 and L1 * (m - tau) >= mum * (1 - 1e-12) and (d == 1 and N == 2 or L1 * (m - tau) <= 1 + math.log(W) + 1e-12)
        okB &= good
        rows.append(dict(d=d, N=N, m_minus_tau=m - tau, lower=lb, Lam1_m_minus_tau=L1 * (m - tau), mu_minus=mum, upper=1 + math.log(W)))
check("B. Lemma 7.2: d(N^2-1)/3 <= m - tau (equality iff d=1), mu_- >= max(1, ...), Lam_1(m-tau) in [mu_-, 1+ln W]", okB,
      "; ".join(f"d={r['d']} N={r['N']}: {r['mu_minus']:.4f}<={r['Lam1_m_minus_tau']:.4f}<={r['upper']:.4f}" for r in rows if r['N'] in (2, 20, 100, 6)))
OUT["B"] = rows
mm = [2 / 3 * (N * N - 1) * math.sin(math.pi / (2 * N)) ** 2 for N in range(2, 4000)]
check("B. mu_-(N) increasing in N (2 <= N < 4000), -> pi^2/6", all(mm[i] < mm[i + 1] for i in range(len(mm) - 1)) and abs(mm[-1] - math.pi ** 2 / 6) < 1e-6)

# ---------------------------------------------------------------- C, D. coupling monotonicity; r_0 > 1; sigma_0 m > 1
okC = True; okD = True; rowsD = []
for d, N in ((1, 2), (1, 3), (1, 6), (1, 9), (2, 2), (2, 3), (2, 4), (2, 5), (2, 7), (3, 2), (3, 3), (3, 4), (4, 2), (4, 3)):
    L = build_L(N, d); n = N ** d; a = site_index((N,) * d, N, d); x0 = site_index((1,) * d, N, d)
    keep, La, mu, U = killed(L, a)
    one = np.ones(n - 1)
    pos = {s: i for i, s in enumerate(keep)}
    pts = list(itertools.product(range(1, N + 1), repeat=d))
    pairs = []
    for x in pts:
        for ax in range(d):
            if x[ax] < N:
                y = list(x); y[ax] += 1; pairs.append((site_index(x, N, d), site_index(tuple(y), N, d)))
    def full(v):              # extend by 0 at a
        f = np.zeros(n); f[keep] = v; return f
    for t in (0.2, 1.0, 5.0, 3.0 * N * N):
        S = full(expm(-t * La) @ one)
        okC &= all(S[x] >= S[y] - 1e-13 for x, y in pairs)
    for q in (0.4, 0.8, 1.0):
        Q = np.eye(n - 1) - q * La; v = one.copy()
        for t in range(0, 60):
            S = full(v); okC &= all(S[x] >= S[y] - 1e-13 for x, y in pairs); v = Q @ v
    mvec = full(np.linalg.solve(La, one))
    psi0 = np.abs(U[:, 0]); r0 = full(psi0 * psi0.sum() / (psi0 @ psi0))
    okC &= all(mvec[x] >= mvec[y] - 1e-9 for x, y in pairs) and all(r0[x] >= r0[y] - 1e-12 for x, y in pairs)
    lam, w, sg = corner_spectral_data(N, d); nz = lam > 1e-14
    tau = np.sum(w[nz] / lam[nz]); L1 = (1 - math.cos(math.pi / N)) / d; X = L1 * tau
    mum = 2 / 3 * (N * N - 1) * math.sin(math.pi / (2 * N)) ** 2
    s0 = mu[0]; s0m = s0 * mvec[x0]
    quant = (X + mum) / (X + 1)
    if (d, N) == (1, 2):
        good = abs(r0[x0] - 1) < 1e-12 and abs(s0m - 1) < 1e-12
    else:
        good = r0[x0] > 1 + 1e-9 and abs(r0[x0] - r0.max()) < 1e-12 and s0m > 1 + 1e-9 and abs(mvec[x0] - mvec.max()) < 1e-9 and s0m >= quant - 1e-12
    okD &= good
    rowsD.append(dict(d=d, N=N, r0=r0[x0], sigma0_m=s0m, quant_lower=quant, Xbar=X))
check("C. Lemma 7.8: P_x(T>t) >= P_y(T>t) for x <= y (cont. t and discrete q=0.4,0.8,1); m, r_0 non-increasing (14 boxes, d=1..4)", okC)
check("D. Thm 7.9: r_0(x_0) = max r_0 > 1, sigma_0 m(x_0) = max > 1, sigma_0 m >= (Xbar+mu_-)/(Xbar+1); equality only (d,N)=(1,2)", okD,
      "; ".join(f"d={r['d']} N={r['N']}: r0={r['r0']:.4f} s0m={r['sigma0_m']:.4f}>={r['quant_lower']:.4f}" for r in rowsD))
OUT["D"] = rowsD

# ---------------------------------------------------------------- E. sharpened Theorem 7.8 from spectral sums (large N)
Zd = {2: 6.0268121, 3: 16.5323160}
okE = True; rowsE = []
for d, Ns in ((2, [3, 4, 6, 8, 12, 20, 40, 80, 160, 320, 640]), (3, [3, 4, 6, 8, 12, 20, 30, 48, 64])):
    for N in Ns:
        lam, w, sg = corner_spectral_data(N, d)
        Lg, Wg, SW = group_visible(lam, w, sg)
        nz = lam > 1e-14
        tau = np.sum(w[nz] / lam[nz]); m = np.sum((1 - sg[nz]) * w[nz] / lam[nz]); K2 = np.sum(w[nz] / lam[nz] ** 2)
        L1 = (1 - math.cos(math.pi / N)) / d; X = L1 * tau; W = 2 * d * math.cos(math.pi / (2 * N)) ** 2
        mum = 2 / 3 * (N * N - 1) * math.sin(math.pi / (2 * N)) ** 2; lW = 1 + math.log(W); Z = Zd[d]
        G = lambda s: np.sum(Wg / (Lg - s)); Gx = lambda s: np.sum(SW / (Lg - s)); dG = lambda s: np.sum(Wg / (Lg - s) ** 2)
        def root(i):
            gap = Lg[i + 1] - Lg[i]; e = 1e-9 * gap
            while G(Lg[i] + e) > 0: e /= 8
            lo = Lg[i] + e; e = 1e-9 * gap
            while G(Lg[i + 1] - e) < 0: e /= 8
            return brentq(G, lo, Lg[i + 1] - e, xtol=1e-300, rtol=1e-15, maxiter=1000)
        s0, s1 = root(0), root(1)
        r0 = -Gx(s0) / (s0 * dG(s0)); r1 = -Gx(s1) / (s1 * dG(s1)); mu_ = (m - tau) / tau
        good = (mum / X <= mu_ * (1 + 1e-12) <= lW / X * (1 + 1e-12)) and r0 > 1 and s0 * m > 1 and s0 * m >= (X + mum) / (X + 1) - 1e-12
        if X > 1:
            beta = X / (X - 1)
            good &= s0 * m >= (1 + mum / X) * (1 - beta * Z / X ** 2) - 1e-12 and s0 * m < 1 + lW / X
            good &= r0 <= 1 + mu_ + beta * Z / X ** 2 + 1e-12 and r0 >= (1 + mu_) * (1 - beta * (1 + beta) * Z / X ** 2) - 1e-12
            if beta * (1 + beta) * Z <= X * X:
                good &= r0 >= (1 + mum / X) * (1 - beta * (1 + beta) * Z / X ** 2) - 1e-12
            th0 = d * (d - 1) / 2 * math.cos(math.pi / (2 * N)) ** 4
            if X - W - 1 + th0 > 0:
                dp = W / (X - W - 1 + th0)
                if dp < 1:
                    om = 2 * (1 + dp) / (1 - dp); dm = W / (X - W + om * (Z - W) - 1 / (1 + dp))
                    ym = W - lW - om * (Z - W) - 1; yp = W - mum + om * (Z - W) - 1 / (1 + dp)
                    Nm = W + min(dm * ym, dp * ym); Np = W + max(dm * yp, dp * yp)
                    Dm = (1 + dm) * W + dm ** 2 * (1 / (1 + dp) + (1 + dm) * th0); Dp = (1 + dp) * W + dp ** 2 * (1 + om ** 2 * (Z - W) / (1 + dp))
                    good &= -r1 <= dp * Np / Dm + 1e-12
                    if Nm > 0: good &= -r1 >= dm * Nm / Dp - 1e-12
        okE &= good
        rowsE.append(dict(d=d, N=N, Xbar=X, sigma0_m=s0 * m, lower=(1 + mum / X) * (1 - (X / (X - 1)) * Z / X ** 2) if X > 1 else None, r0=r0, r1=r1))
check("E. Thm 7.10 (ii),(v),(vi) sharpened with mu_- : all inequalities (d=2 up to N=640, d=3 up to N=64)", okE,
      "; ".join(f"d={r['d']} N={r['N']}: {r['lower']:.5f}<=s0m={r['sigma0_m']:.5f}, r0={r['r0']:.5f}" for r in rowsE if r['N'] in (6, 20, 64, 640)))
OUT["E"] = rowsE
OUT["all_ok"] = bool(OK)
json.dump(OUT, open(_os.path.join(_R, 'data', 'msc_rigorous_spectral', '15_qsd_coupling_checks.json'), "w"), indent=1, default=float)
print("ALL OK:", OK)
