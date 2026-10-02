"""Theorems 3.15 / 3.16 (eventual total positivity; EXACT integer arithmetic).

For q = qn/qd and every N in {3, ..., NMAX} we decide whether Q^k is totally positive of order 2 (TP2: all 2x2 minors >= 0)
for every k in {KLO, ..., KHI = 2 KLO - 1}, and we determine exactly whether the first-passage PMF f (start x0 = 1) is
log-concave / unimodal.

  Q = transition matrix on the transient sites 1..n (n = N-1); integer matrix A = 2*qd*Q:
      A_11 = 2 qd - qn,  A_jj = 2 (qd - qn) (j >= 2),  A_{j,j+-1} = qn,       Q^k = A^k / (2 qd)^k.

  TP2 test (Lemma 3.13): for 0 < q < 1 the support of Q^k is exactly the band |x - z| <= k, and then Q^k is TP2 iff all
  CONTIGUOUS minors (rows x, x+1; columns z, z+1 with x+1-k <= z, z+1 <= x+k) are >= 0.  With --full ALL minors that can
  be negative (rows x < y, columns z < w with y-k <= z < w <= x+k) are tested as well.

  K_t(x) = (Q^t)_{x,n},  f(t) = (q/2) K_{t-1}(1),  M_t(x,y) = K_t(x) K_{t-1}(y) - K_{t-1}(x) K_t(y) = [C_2(Q^(t-1)) M_1](x,y),
  f(t+1)^2 - f(t) f(t+2) = (q/2)^3 M_t(1,2).   If Q^k is TP2 for all k >= KLO then M_t >= 0 for all t >= KLO + 1.
  Computed exactly for every N:  sign M_t(1,2), 1 <= t <= KLO;  sign(f(t+1) - f(t)), 1 <= t <= KLO + 2.
  Verdict for N (when TP2 holds for KLO <= k <= KHI, hence for all k >= KLO):
     log-concave  iff  M_t(1,2) >= 0 for all t <= KLO;
     unimodal     iff  the signs of the increments for t <= KLO + 2 contain no '-' followed (later) by '+'
                  (for t >= KLO + 1 the ratios f(t+1)/f(t) are nonincreasing, so no later sign change '- +' can occur).

By the locality lemma (Lemma 3.14), TP2 of Q_N^k for N = NMAX (n >= 4 KHI) implies the same for every N >= NMAX; for such N
also M_t(1,2) = 0 for t <= KLO, so f is log-concave.

Usage: 31_tp2_powers.py [qn qd KLO NMAX] [--full]        defaults: 4 5 10 80
Output: data/31_tp2_powers_q<qn>_<qd>.json
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import os, sys, json, time

args = [a for a in sys.argv[1:] if not a.startswith("--")]
FULL = "--full" in sys.argv
qn = int(args[0]) if len(args) > 0 else 4
qd = int(args[1]) if len(args) > 1 else 5
KLO = int(args[2]) if len(args) > 2 else 10
NMAX = int(args[3]) if len(args) > 3 else 80
KHI = 2 * KLO - 1
assert NMAX - 1 >= 4 * KHI, "need n = NMAX - 1 >= 4*KHI for the locality lemma"
TINC = KLO + 2
t00 = time.time()


def tridiag(n):
    A = [[0] * n for _ in range(n)]
    for i in range(n):
        A[i][i] = 2 * (qd - qn)
        if i > 0:
            A[i][i - 1] = qn
        if i < n - 1:
            A[i][i + 1] = qn
    A[0][0] = 2 * qd - qn
    return A


def times_A(P, A, n, k):
    """P (= A^(k-1), bandwidth k-1) times A, exact integers."""
    R = [[0] * n for _ in range(n)]
    for i in range(n):
        Pi = P[i]; Ri = R[i]
        for j in range(max(0, i - k), min(n, i + k + 1)):
            s = Pi[j] * A[j][j]
            if j > 0:
                s += Pi[j - 1] * A[j - 1][j]
            if j < n - 1:
                s += Pi[j + 1] * A[j + 1][j]
            Ri[j] = s
    return R


def band_ok(P, n, k):
    """support of A^k is exactly the band |x - z| <= k (needed for the contiguous-minor criterion)."""
    for x in range(n):
        Px = P[x]
        for z in range(max(0, x - k), min(n, x + k + 1)):
            if Px[z] <= 0:
                return False
    return True


def contiguous_negative(P, n, k):
    cnt = 0; neg = []
    for x in range(n - 1):
        Px = P[x]; Py = P[x + 1]
        for z in range(max(0, x + 1 - k), min(n - 1, x + k)):
            cnt += 1
            if Px[z] * Py[z + 1] < Px[z + 1] * Py[z]:
                neg.append((x + 1, x + 2, z + 1, z + 2))
    return neg, cnt


def all_negative(P, n, k):
    neg = []; cnt = 0
    for x in range(n):
        Px = P[x]
        for y in range(x + 1, min(n, x + 2 * k)):
            Py = P[y]
            lo = max(0, y - k); hi = min(n - 1, x + k)
            for z in range(lo, hi):
                a1 = Px[z]; a2 = Py[z]
                for w in range(z + 1, hi + 1):
                    cnt += 1
                    if a1 * Py[w] < Px[w] * a2:
                        neg.append((x + 1, y + 1, z + 1, w + 1))
    return neg, cnt


rows = []
tot_contig = tot_full = 0
full_agrees = True
for N in range(3, NMAX + 1):
    n = N - 1
    A = tridiag(n)
    P = [row[:] for row in A]
    Kcol = [[1 if i == n - 1 else 0 for i in range(n)]]
    tp2 = True
    bad_k = []
    neg_all_k = {}
    for k in range(1, max(KHI, TINC) + 1):
        if k > 1:
            P = times_A(P, A, n, k)
        if k <= TINC:
            Kcol.append([P[i][n - 1] for i in range(n)])
        if k >= KLO or N == NMAX:
            neg, cnt = contiguous_negative(P, n, k)
            if N == NMAX:
                neg_all_k[k] = len(neg)
            if k >= KLO:
                tot_contig += cnt
                okb = band_ok(P, n, k)
                if neg or not okb:
                    tp2 = False; bad_k.append(k)
                if FULL:
                    negf, cntf = all_negative(P, n, k)
                    tot_full += cntf
                    if bool(negf) != bool(neg):
                        full_agrees = False
    M12 = []
    for t in range(1, KLO + 1):
        m = Kcol[t][0] * Kcol[t - 1][1] - Kcol[t - 1][0] * Kcol[t][1]
        M12.append((m > 0) - (m < 0))
    inc = []
    for t in range(1, TINC + 1):
        d = Kcol[t][0] - 2 * qd * Kcol[t - 1][0]          # sign of f(t+1) - f(t)
        inc.append((d > 0) - (d < 0))
    seen_minus = False; uni_pattern = True
    for s_ in inc:
        if s_ < 0:
            seen_minus = True
        elif s_ > 0 and seen_minus:
            uni_pattern = False
    lc = tp2 and all(s_ >= 0 for s_ in M12)
    verdict = "log-concave" if lc else (("unimodal, not log-concave" if uni_pattern else "NOT unimodal") if tp2 else
                                        ("NOT unimodal" if not uni_pattern else "undetermined (TP2 fails for k in %s)" % bad_k[:4]))
    rec = dict(N=N, tp2=tp2, verdict=verdict, first_t_with_M12_negative=[t + 1 for t, s_ in enumerate(M12) if s_ < 0][:3],
               increments=("".join("+" if s_ > 0 else ("-" if s_ < 0 else "0") for s_ in inc)) if N <= 12 else None)
    if N == NMAX:
        rec["negative_contiguous_minor_counts_all_k"] = neg_all_k
    rows.append(rec)
    if N <= 14 or N % 40 == 0 or N == NMAX:
        print({k_: v for k_, v in rec.items() if k_ != "negative_contiguous_minor_counts_all_k"}, flush=True)
tp2_fail = [r["N"] for r in rows if not r["tp2"]]
not_lc = [r["N"] for r in rows if r["verdict"] != "log-concave"]
not_uni = [r["N"] for r in rows if r["verdict"] == "NOT unimodal"]
undet = [r["N"] for r in rows if r["verdict"].startswith("undetermined")]
last_bad = max([k for k, v in rows[-1]["negative_contiguous_minor_counts_all_k"].items() if v > 0], default=0)
print(f"q = {qn}/{qd}, k in [{KLO},{KHI}], N in [3,{NMAX}]  ({tot_contig} contiguous minors" + (f", {tot_full} general minors, criteria agree: {full_agrees}" if FULL else "") + f")   [{time.time()-t00:.0f}s]")
print("  TP2 fails for N in", tp2_fail, "; last k with a negative minor at N = NMAX:", last_bad)
print("  not log-concave: N in", not_lc, ";  NOT unimodal: N in", not_uni, ";  undetermined by this method: N in", undet)
json.dump(dict(q=f"{qn}/{qd}", KLO=KLO, KHI=KHI, NMAX=NMAX, full=FULL, full_agrees=full_agrees if FULL else None,
               contiguous_minors_tested=tot_contig, general_minors_tested=tot_full, tp2_fails_for_N=tp2_fail, not_logconcave=not_lc,
               not_unimodal=not_uni, undetermined=undet, last_k_with_negative_minor_at_NMAX=last_bad, rows=rows),
          open(os.path.join(_R, 'data', 'msc_rigorous_1d', f"31_tp2_powers_q{qn}_{qd}.json"), "w"), indent=1)
